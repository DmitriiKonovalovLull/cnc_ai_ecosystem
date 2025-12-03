import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import joblib
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training on device: {self.device}")

        # Model configurations
        self.model_configs = {
            'intent_classifier': {
                'base_model': settings.INTENT_MODEL,
                'num_labels': 10,  # Adjust based on your intents
                'max_length': 128
            },
            'ner_model': {
                'base_model': settings.NER_MODEL,
                'max_length': 256
            }
        }

    def train_intent_classifier(self, training_data: Dict[str, Any],
                                validation_split: float = 0.2) -> Dict[str, Any]:
        """Train intent classification model"""
        logger.info("Starting intent classifier training")

        try:
            # Prepare data
            queries = training_data.get('queries', [])
            intents = training_data.get('intents', [])

            if not queries or not intents:
                raise ValueError("No training data provided")

            # Create label mapping
            unique_intents = sorted(set(intents))
            label_to_id = {label: i for i, label in enumerate(unique_intents)}
            id_to_label = {i: label for label, i in label_to_id.items()}

            # Convert labels to IDs
            label_ids = [label_to_id[intent] for intent in intents]

            # Create dataset
            dataset = Dataset.from_dict({
                'text': queries,
                'label': label_ids
            })

            # Split dataset
            split_dataset = dataset.train_test_split(test_size=validation_split, seed=42)

            # Load tokenizer and model
            config = self.model_configs['intent_classifier']
            tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
            model = AutoModelForSequenceClassification.from_pretrained(
                config['base_model'],
                num_labels=len(unique_intents)
            ).to(self.device)

            # Tokenize dataset
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=config['max_length']
                )

            tokenized_dataset = split_dataset.map(tokenize_function, batched=True)

            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.models_dir / "intent_classifier" / "checkpoints"),
                num_train_epochs=settings.TRAINING_EPOCHS,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=str(self.models_dir / "intent_classifier" / "logs"),
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                push_to_hub=False,
                report_to="none"
            )

            # Metrics function
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)

                from sklearn.metrics import accuracy_score, precision_recall_fscore_support

                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='weighted'
                )
                accuracy = accuracy_score(labels, predictions)

                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset['train'],
                eval_dataset=tokenized_dataset['test'],
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )

            # Train model
            train_result = trainer.train()

            # Evaluate model
            eval_result = trainer.evaluate()

            # Save model
            model_save_path = self.models_dir / "intent_classifier" / "final_model"
            trainer.save_model(str(model_save_path))
            tokenizer.save_pretrained(str(model_save_path))

            # Save label mappings
            label_info = {
                'label_to_id': label_to_id,
                'id_to_label': id_to_label,
                'num_labels': len(unique_intents)
            }

            with open(model_save_path / "label_mappings.json", 'w', encoding='utf-8') as f:
                json.dump(label_info, f, ensure_ascii=False, indent=2)

            # Train traditional ML model as fallback
            ml_model = self._train_ml_intent_classifier(queries, label_ids, unique_intents)

            # Save ML model
            ml_save_path = self.models_dir / "intent_classifier" / "ml_model"
            ml_save_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(ml_model, ml_save_path / "ml_model.pkl")

            # Prepare results
            results = {
                'model_type': 'transformer',
                'base_model': config['base_model'],
                'training_stats': {
                    'train_loss': train_result.training_loss,
                    'eval_loss': eval_result['eval_loss'],
                    'eval_accuracy': eval_result['eval_accuracy'],
                    'eval_precision': eval_result['eval_precision'],
                    'eval_recall': eval_result['eval_recall'],
                    'eval_f1': eval_result['eval_f1']
                },
                'dataset_info': {
                    'total_samples': len(queries),
                    'train_samples': len(tokenized_dataset['train']),
                    'eval_samples': len(tokenized_dataset['test']),
                    'unique_intents': len(unique_intents),
                    'intent_distribution': {
                        intent: intents.count(intent)
                        for intent in unique_intents
                    }
                },
                'model_path': str(model_save_path),
                'ml_model_path': str(ml_save_path / "ml_model.pkl"),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Intent classifier training completed. Accuracy: {eval_result['eval_accuracy']:.4f}")
            return results

        except Exception as e:
            logger.error(f"Intent classifier training failed: {e}")
            raise

    def _train_ml_intent_classifier(self, queries: List[str], labels: List[int],
                                    unique_intents: List[str]) -> Pipeline:
        """Train traditional ML intent classifier"""
        logger.info("Training ML intent classifier")

        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words=None
            )),
            ('classifier', LinearSVC(
                C=1.0,
                class_weight='balanced',
                max_iter=1000
            ))
        ])

        # Train model
        pipeline.fit(queries, labels)

        return pipeline

    def train_ner_model(self, training_data: Dict[str, Any],
                        validation_split: float = 0.2) -> Dict[str, Any]:
        """Train NER model"""
        logger.info("Starting NER model training")

        try:
            # NER training requires specific format
            # This is a simplified version - in production you'd need proper NER dataset

            logger.warning("Full NER training not implemented - using fine-tuning approach")

            # For now, just save the configuration
            config = self.model_configs['ner_model']

            results = {
                'model_type': 'ner',
                'base_model': config['base_model'],
                'status': 'config_saved',
                'note': 'Full NER training requires annotated dataset',
                'timestamp': datetime.now().isoformat()
            }

            return results

        except Exception as e:
            logger.error(f"NER model training failed: {e}")
            raise

    def train_embedding_model(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train or fine-tune embedding model"""
        logger.info("Starting embedding model training")

        try:
            # This would require sentence-transformers training
            # For now, just use the pre-trained model

            from sentence_transformers import SentenceTransformer, models, losses
            from torch.utils.data import DataLoader

            # Prepare training data
            texts = training_data.get('queries', [])
            if not texts:
                texts = training_data.get('documents', [])

            if not texts:
                raise ValueError("No text data for embedding training")

            # Create simple dataset (in production, you'd need pairs/triplets)
            train_data = []
            for text in texts[:1000]:  # Limit for example
                train_data.append({
                    'anchor': text,
                    'positive': text,  # Same text as positive example
                    'negative': "unrelated text"  # Simple negative
                })

            # Load model
            model = SentenceTransformer(settings.EMBEDDING_MODEL)

            # Create dataloader
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)

            # Define loss
            train_loss = losses.MultipleNegativesRankingLoss(model=model)

            # Training parameters
            num_epochs = 3
            warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

            # Train model
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                show_progress_bar=True
            )

            # Save model
            save_path = self.models_dir / "embedding_model" / "fine_tuned"
            model.save(str(save_path))

            results = {
                'model_type': 'embedding',
                'base_model': settings.EMBEDDING_MODEL,
                'training_stats': {
                    'samples_used': len(train_data),
                    'epochs': num_epochs
                },
                'model_path': str(save_path),
                'timestamp': datetime.now().isoformat()
            }

            logger.info("Embedding model training completed")
            return results

        except Exception as e:
            logger.error(f"Embedding model training failed: {e}")
            raise

    def train_rag_model(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train RAG model components"""
        logger.info("Starting RAG model training")

        try:
            # RAG training involves multiple components
            # 1. Retriever training (embedding model)
            # 2. Generator training (seq2seq model)

            # For now, focus on retriever training
            if 'documents' in training_data and 'queries' in training_data:
                # This would require query-document relevance pairs
                logger.info("RAG training data available")

                # Simplified training - just log the configuration
                results = {
                    'model_type': 'rag',
                    'components': ['retriever', 'generator'],
                    'status': 'configuration_prepared',
                    'note': 'Full RAG training requires relevance-labeled pairs',
                    'data_stats': {
                        'documents': len(training_data.get('documents', [])),
                        'queries': len(training_data.get('queries', [])),
                        'has_relevance_pairs': 'relevance_pairs' in training_data
                    },
                    'timestamp': datetime.now().isoformat()
                }
            else:
                results = {
                    'model_type': 'rag',
                    'status': 'skipped',
                    'reason': 'Insufficient training data',
                    'timestamp': datetime.now().isoformat()
                }

            return results

        except Exception as e:
            logger.error(f"RAG model training failed: {e}")
            raise

    def incremental_training(self, new_data: Dict[str, Any],
                             model_type: str = 'intent_classifier') -> Dict[str, Any]:
        """Perform incremental training with new data"""
        logger.info(f"Starting incremental training for {model_type}")

        try:
            # Load existing model
            model_path = self.models_dir / model_type / "final_model"

            if not model_path.exists():
                logger.warning(f"No existing model found for {model_type}, training from scratch")
                return self._train_from_scratch(new_data, model_type)

            # Prepare data
            if model_type == 'intent_classifier':
                return self._incremental_intent_training(new_data, model_path)
            else:
                logger.warning(f"Incremental training not implemented for {model_type}")
                return {'status': 'not_implemented', 'model_type': model_type}

        except Exception as e:
            logger.error(f"Incremental training failed: {e}")
            raise

    def _incremental_intent_training(self, new_data: Dict[str, Any],
                                     model_path: Path) -> Dict[str, Any]:
        """Incremental training for intent classifier"""
        # Load existing model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))

        # Load label mappings
        with open(model_path / "label_mappings.json", 'r', encoding='utf-8') as f:
            label_info = json.load(f)

        # Prepare new data
        new_queries = new_data.get('queries', [])
        new_intents = new_data.get('intents', [])

        if not new_queries or not new_intents:
            return {'status': 'no_new_data', 'samples': 0}

        # Check for new intents
        existing_labels = set(label_info['id_to_label'].values())
        new_unique_intents = set(new_intents)
        new_labels = new_unique_intents - existing_labels

        if new_labels:
            logger.info(f"Found {len(new_labels)} new intents: {new_labels}")
            # Need to expand model for new labels
            return self._expand_model_for_new_labels(model, tokenizer, label_info,
                                                     new_data, new_labels)

        # Convert new labels to IDs
        label_to_id = label_info['label_to_id']
        new_label_ids = [label_to_id[intent] for intent in new_intents]

        # Create dataset
        dataset = Dataset.from_dict({
            'text': new_queries,
            'label': new_label_ids
        })

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=128
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Training arguments for fine-tuning
        training_args = TrainingArguments(
            output_dir=str(self.models_dir / "intent_classifier" / "incremental_checkpoints"),
            num_train_epochs=2,  # Fewer epochs for fine-tuning
            per_device_train_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=str(self.models_dir / "intent_classifier" / "incremental_logs"),
            logging_steps=10,
            save_strategy="epoch",
            push_to_hub=False,
            report_to="none"
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )

        # Fine-tune
        trainer.train()

        # Save updated model
        updated_path = self.models_dir / "intent_classifier" / "updated_model"
        trainer.save_model(str(updated_path))
        tokenizer.save_pretrained(str(updated_path))

        # Copy label mappings
        import shutil
        shutil.copy2(
            model_path / "label_mappings.json",
            updated_path / "label_mappings.json"
        )

        results = {
            'status': 'success',
            'model_type': 'intent_classifier',
            'training_type': 'incremental',
            'samples_used': len(new_queries),
            'updated_model_path': str(updated_path),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Incremental training completed with {len(new_queries)} samples")
        return results

    def _expand_model_for_new_labels(self, model, tokenizer, label_info,
                                     new_data, new_labels):
        """Expand model to accommodate new labels"""
        # This is complex - requires modifying the classification head
        # For now, retrain from scratch with combined data

        logger.warning(f"New labels detected: {new_labels}. Retraining from scratch recommended.")

        return {
            'status': 'new_labels_detected',
            'new_labels': list(new_labels),
            'recommendation': 'retrain_from_scratch',
            'timestamp': datetime.now().isoformat()
        }

    def _train_from_scratch(self, data: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """Train model from scratch"""
        if model_type == 'intent_classifier':
            return self.train_intent_classifier(data)
        elif model_type == 'ner_model':
            return self.train_ner_model(data)
        elif model_type == 'embedding_model':
            return self.train_embedding_model(data)
        elif model_type == 'rag_model':
            return self.train_rag_model(data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def validate_model(self, model_type: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trained model"""
        logger.info(f"Validating {model_type} model")

        # This would involve loading the model and running evaluation
        # For now, return placeholder

        return {
            'model_type': model_type,
            'validation_status': 'pending',
            'note': 'Model validation not implemented',
            'timestamp': datetime.now().isoformat()
        }

    def deploy_model(self, model_type: str, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """Deploy trained model to inference directory"""
        logger.info(f"Deploying {model_type} model")

        try:
            source_path = model_path or self.models_dir / model_type / "final_model"
            deploy_path = Path(__file__).parent / "inference" / "models" / model_type

            if not source_path.exists():
                raise FileNotFoundError(f"Model not found at {source_path}")

            # Copy model files
            deploy_path.mkdir(parents=True, exist_ok=True)

            import shutil
            if deploy_path.exists():
                shutil.rmtree(deploy_path)

            shutil.copytree(source_path, deploy_path)

            # Update configuration if needed
            if model_type == 'intent_classifier':
                # Update settings to use deployed model
                config_update = {
                    'INTENT_MODEL': str(deploy_path.absolute())
                }
                self._update_model_config(config_update)

            results = {
                'model_type': model_type,
                'status': 'deployed',
                'source_path': str(source_path),
                'deploy_path': str(deploy_path),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Model deployed to {deploy_path}")
            return results

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            raise

    def _update_model_config(self, updates: Dict[str, str]):
        """Update model configuration"""
        # This would update settings or config files
        # For now, just log
        logger.info(f"Model configuration updates: {updates}")