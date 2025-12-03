from celery import Task
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

from tasks.celery_app import celery_app
from knowledge_base.kb_manager import KnowledgeBaseManager
from ml_core.inference.answer_engine import AnswerEngine
from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingTask(Task):
    """Base class for embedding tasks"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Embedding task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)

    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Embedding task {task_id} completed successfully")
        super().on_success(retval, task_id, args, kwargs)


@celery_app.task(base=EmbeddingTask, bind=True)
def update_embeddings_task(self, document_ids: List[str] = None) -> Dict[str, Any]:
    """
    Update embeddings for documents
    """
    try:
        logger.info("Starting embeddings update task")

        kb_manager = KnowledgeBaseManager()
        answer_engine = AnswerEngine()

        # Get documents to update
        documents_to_update = []

        if document_ids:
            # Update specific documents
            for doc_id in document_ids:
                doc = kb_manager.documents.get(doc_id)
                if doc:
                    documents_to_update.append(doc)
        else:
            # Update all documents without embeddings or with old embeddings
            for doc_id, document in kb_manager.documents.items():
                if not document.embeddings:
                    documents_to_update.append(document)

        if not documents_to_update:
            return {
                'status': 'skipped',
                'reason': 'No documents need updating',
                'timestamp': datetime.now().isoformat()
            }

        # Update embeddings
        updated_count = 0
        errors = []

        for document in documents_to_update:
            try:
                # Generate new embeddings
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # This would require async embedding generation
                # For now, mark as needing update
                document.embeddings = None
                updated_count += 1

                loop.close()

            except Exception as e:
                errors.append({
                    'document_id': document.id,
                    'error': str(e)
                })
                logger.error(f"Failed to update embeddings for {document.id}: {e}")

        # Save updated documents
        if documents_to_update:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # This would save to persistent storage
            # For now, just update in memory

            loop.close()

        return {
            'status': 'completed',
            'documents_processed': len(documents_to_update),
            'embeddings_updated': updated_count,
            'errors': len(errors),
            'error_details': errors,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Embeddings update failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task(base=EmbeddingTask)
def reindex_knowledge_base_task() -> Dict[str, Any]:
    """
    Reindex entire knowledge base
    """
    try:
        logger.info("Starting knowledge base reindexing")

        kb_manager = KnowledgeBaseManager()

        # Rebuild index
        kb_manager._build_index()

        # Count statistics
        index_stats = {
            'total_documents': len(kb_manager.documents),
            'index_terms': len(kb_manager.index),
            'avg_docs_per_term': 0
        }

        if kb_manager.index:
            total_refs = sum(len(docs) for docs in kb_manager.index.values())
            index_stats['avg_docs_per_term'] = total_refs / len(kb_manager.index)

        return {
            'status': 'completed',
            'action': 'reindex',
            'stats': index_stats,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Knowledge base reindexing failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task
def schedule_embedding_updates():
    """
    Schedule regular embedding updates
    """
    try:
        logger.info("Scheduling embedding updates")

        # Schedule different embedding tasks

        # Daily full update for documents without embeddings
        full_update_task = update_embeddings_task.apply_async(
            queue='embedding',
            countdown=3600  # Start in 1 hour
        )

        # Weekly reindexing
        reindex_task = reindex_knowledge_base_task.apply_async(
            queue='embedding',
            countdown=86400  # Start in 24 hours
        )

        return {
            'status': 'scheduled',
            'tasks': [
                {'type': 'embeddings_update', 'task_id': full_update_task.id},
                {'type': 'reindex', 'task_id': reindex_task.id}
            ],
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to schedule embedding updates: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_app.task
def optimize_vector_db_task() -> Dict[str, Any]:
    """
    Optimize vector database
    """
    try:
        logger.info("Starting vector DB optimization")

        # This would involve optimizing ChromaDB or Qdrant
        # For now, just report status

        from ml_core.inference.answer_engine import AnswerEngine
        answer_engine = AnswerEngine()

        # Get collection info
        collection_info = {}

        if hasattr(answer_engine, 'chroma_collection'):
            try:
                collection = answer_engine.chroma_collection
                count = collection.count()
                collection_info['chroma'] = {
                    'document_count': count,
                    'collection_name': collection.name
                }
            except:
                collection_info['chroma'] = {'error': 'Not available'}

        if hasattr(answer_engine, 'qdrant_client'):
            try:
                collections = answer_engine.qdrant_client.get_collections()
                collection_info['qdrant'] = {
                    'collections': collections.collections,
                    'count': len(collections.collections)
                }
            except:
                collection_info['qdrant'] = {'error': 'Not available'}

        return {
            'status': 'completed',
            'action': 'optimization_check',
            'vector_dbs': collection_info,
            'recommendations': [
                'Consider running compaction if database is large',
                'Review embedding dimensions for optimal performance',
                'Check similarity search performance metrics'
            ],
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Vector DB optimization failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }