from celery import Task
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from tasks.celery_app import celery_app
from ml_core.training.data_collector import DataCollector
from ml_core.training.trainer import ModelTrainer
from ml_core.training.validation import ModelValidator
from config.settings import settings

logger = logging.getLogger(__name__)


class TrainingTask(Task):
    """Base class for training tasks with progress tracking"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Training task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Training task {task_id} completed successfully")
        super().on_success(retval, task_id, args, kwargs)

    def update_state(self, task_id=None, state=None, meta=None):
        """Update task state with progress"""
        super().update_state(task_id=task_id, state=state, meta=meta)


@celery_app.task(base=TrainingTask, bind=True)
def train_model_task(self, model_type: str, data_source: Optional[str] = None,
                     epochs: int = 10) -> Dict[str, Any]:
    """
    Train specified model type
    """
    try:
        logger.info(f"Starting training task for {model_type}")

        # Update task state
        self.update_state(
            state='STARTED',
            meta={'progress': 0, 'model_type': model_type}
        )

        # Load training data
        data_collector = DataCollector()
        training_data = data_collector.load_training_data(limit=1000)

        if not training_data:
            return {
                'status': 'error',
                'error': 'No training data available',
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }

        # Create dataset
        dataset = data_collector.create_dataset(training_data)

        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={'progress': 30, 'dataset_size': len(training_data)}
        )

        # Initialize trainer
        trainer = ModelTrainer()

        # Train model based on type
        if model_type == 'intent_classifier':
            result = trainer.train_intent_classifier(dataset, validation_split=0.2)
        elif model_type == 'embedding_model':
            result = trainer.train_embedding_model(dataset)
        elif model_type == 'ner_model':
            result = trainer.train_ner_model(dataset)
        elif model_type == 'rag_model':
            result = trainer.train_rag_model(dataset)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Update progress
        self.update_state(
            state='VALIDATING',
            meta={'progress': 80, 'model_type': model_type}
        )

        # Validate model
        validator = ModelValidator()

        # Note: Validation would require test data
        # For now, just record the training result

        # Update progress
        self.update_state(
            state='DEPLOYING',
            meta={'progress': 90, 'model_type': model_type}
        )

        # Deploy model
        deploy_result = trainer.deploy_model(model_type)

        # Update progress
        self.update_state(
            state='COMPLETED',
            meta={'progress': 100, 'model_type': model_type}
        )

        return {
            'status': 'success',
            'model_type': model_type,
            'training_result': result,
            'deployment_result': deploy_result,
            'dataset_size': len(training_data),
            'epochs': epochs,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Training task failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(base=TrainingTask, bind=True)
def incremental_training_task(self, model_type: str = 'intent_classifier',
                              new_data_limit: int = 100) -> Dict[str, Any]:
    """
    Perform incremental training with new data
    """
    try:
        logger.info(f"Starting incremental training for {model_type}")

        # Update task state
        self.update_state(
            state='STARTED',
            meta={'progress': 0, 'model_type': model_type}
        )

        # Load recent feedback data
        data_collector = DataCollector()

        # Get feedback data from last week
        from datetime import datetime, timedelta
        import os

        feedback_dir = Path("data/feedback")
        recent_files = []

        if feedback_dir.exists():
            for file in feedback_dir.glob("feedback_*.json"):
                file_time = datetime.fromtimestamp(file.stat().st_mtime)
                if datetime.now() - file_time < timedelta(days=7):
                    recent_files.append(file)

        # Load feedback data
        new_data = []
        for file in recent_files[:10]:  # Limit to 10 most recent files
            try:
                import json
                with open(file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
                    new_data.extend(feedback_data)
            except:
                continue

        if not new_data or len(new_data) < 10:
            return {
                'status': 'skipped',
                'reason': 'Insufficient new data',
                'new_data_count': len(new_data),
                'model_type': model_type,
                'timestamp': datetime.now().isoformat()
            }

        # Limit data size
        new_data = new_data[:new_data_limit]

        # Update progress
        self.update_state(
            state='PROCESSING',
            meta={'progress': 40, 'new_data': len(new_data)}
        )

        # Prepare training data
        training_data = {
            'queries': [item.get('query', '') for item in new_data],
            'intents': [item.get('correct_intent', 'unknown') for item in new_data],
            'responses': [item.get('corrected_response', '') for item in new_data]
        }

        # Perform incremental training
        trainer = ModelTrainer()
        result = trainer.incremental_training(training_data, model_type)

        # Update progress
        self.update_state(
            state='DEPLOYING',
            meta={'progress': 80, 'model_type': model_type}
        )

        # Deploy if training was successful
        if result.get('status') == 'success':
            deploy_result = trainer.deploy_model(model_type, Path(result.get('updated_model_path')))
            result['deployment'] = deploy_result

        # Update progress
        self.update_state(
            state='COMPLETED',
            meta={'progress': 100, 'model_type': model_type}
        )

        return {
            'status': 'success',
            'model_type': model_type,
            'result': result,
            'new_data_used': len(new_data),
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Incremental training failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task
def schedule_weekly_training():
    """
    Schedule weekly model training
    """
    try:
        logger.info("Scheduling weekly training")

        # Schedule different model trainings
        scheduled_tasks = []

        # Intent classifier training (most frequent)
        intent_task = train_model_task.apply_async(
            args=['intent_classifier'],
            kwargs={'epochs': 15},
            queue='training',
            countdown=3600  # Start in 1 hour
        )
        scheduled_tasks.append({'type': 'intent_classifier', 'task_id': intent_task.id})

        # Embedding model training (less frequent)
        embedding_task = train_model_task.apply_async(
            args=['embedding_model'],
            queue='training',
            countdown=7200  # Start in 2 hours
        )
        scheduled_tasks.append({'type': 'embedding_model', 'task_id': embedding_task.id})

        # Incremental training for intent classifier
        incremental_task = incremental_training_task.apply_async(
            args=['intent_classifier'],
            kwargs={'new_data_limit': 200},
            queue='training',
            countdown=1800  # Start in 30 minutes
        )
        scheduled_tasks.append({'type': 'incremental_intent', 'task_id': incremental_task.id})

        return {
            'status': 'scheduled',
            'tasks_scheduled': len(scheduled_tasks),
            'scheduled_tasks': scheduled_tasks,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to schedule weekly training: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task
def validate_models_task():
    """
    Validate all trained models
    """
    try:
        logger.info("Starting model validation")

        validator = ModelValidator()
        results = {}

        # Validate intent classifier
        # Note: This would require test data
        # For now, just check model files exist

        model_types = ['intent_classifier', 'embedding_model', 'ner_model']

        for model_type in model_types:
            model_path = Path(__file__).parent.parent / "ml_core" / "models" / model_type

            if model_path.exists():
                # Check if there are model files
                model_files = list(model_path.rglob("*"))

                results[model_type] = {
                    'exists': True,
                    'files_count': len(model_files),
                    'last_modified': None
                }

                if model_files:
                    # Get most recent file modification time
                    recent_file = max(model_files, key=lambda f: f.stat().st_mtime)
                    results[model_type]['last_modified'] = datetime.fromtimestamp(
                        recent_file.stat().st_mtime
                    ).isoformat()
            else:
                results[model_type] = {
                    'exists': False,
                    'files_count': 0
                }

        return {
            'status': 'completed',
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }