import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
from loguru import logger
import asyncio
from dataclasses import dataclass, asdict
import pandas as pd

from config.settings import settings
from ml_core.inference.answer_engine import AnswerEngine


@dataclass
class KnowledgeDocument:
    """Knowledge document structure"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embeddings: Optional[List[float]] = None
    source: str = "unknown"
    processed_at: str = None
    version: int = 1

    def __post_init__(self):
        if self.processed_at is None:
            self.processed_at = datetime.now().isoformat()

    def generate_hash(self) -> str:
        """Generate content hash for versioning"""
        content_str = self.content + json.dumps(self.metadata, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()


class KnowledgeBaseManager:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(__file__).parent.parent / "data" / "knowledge_base"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize answer engine for embeddings
        self.answer_engine = AnswerEngine()

        # Document storage
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.index: Dict[str, List[str]] = {}  # Term -> document IDs

        # Statistics
        self.stats = {
            'total_documents': 0,
            'by_source': {},
            'by_type': {},
            'last_updated': None
        }

        # Load existing knowledge
        self._load_existing_knowledge()

    def _load_existing_knowledge(self):
        """Load existing knowledge from storage"""
        knowledge_files = list(self.storage_path.glob("knowledge_*.json"))

        for filepath in knowledge_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert dicts to KnowledgeDocument objects
                for doc_data in data:
                    doc = KnowledgeDocument(**doc_data)
                    self.documents[doc.id] = doc

                    # Update statistics
                    self._update_stats(doc)

                logger.info(f"Loaded {len(data)} documents from {filepath}")

            except Exception as e:
                logger.error(f"Error loading knowledge from {filepath}: {e}")

        # Build index
        self._build_index()

        logger.info(f"Knowledge base initialized with {len(self.documents)} documents")

    def _update_stats(self, document: KnowledgeDocument):
        """Update statistics"""
        self.stats['total_documents'] = len(self.documents)

        source = document.source
        doc_type = document.metadata.get('type', 'unknown')

        self.stats['by_source'][source] = self.stats['by_source'].get(source, 0) + 1
        self.stats['by_type'][doc_type] = self.stats['by_type'].get(doc_type, 0) + 1
        self.stats['last_updated'] = datetime.now().isoformat()

    def _build_index(self):
        """Build search index from documents"""
        self.index.clear()

        for doc_id, document in self.documents.items():
            # Simple term extraction (in production, use proper tokenization)
            terms = set(document.content.lower().split()[:100])  # Limit terms

            for term in terms:
                if len(term) > 2:  # Ignore very short terms
                    if term not in self.index:
                        self.index[term] = []
                    self.index[term].append(doc_id)

        logger.info(f"Index built with {len(self.index)} terms")

    async def add_document(self, content: str, metadata: Dict[str, Any],
                           source: str = "unknown") -> str:
        """Add document to knowledge base"""
        try:
            # Generate document ID
            doc_id = f"doc_{hashlib.md5(content.encode()).hexdigest()[:12]}"

            # Check if document already exists
            if doc_id in self.documents:
                # Update existing document
                return await self.update_document(doc_id, content, metadata, source)

            # Create document
            document = KnowledgeDocument(
                id=doc_id,
                content=content,
                metadata=metadata,
                source=source
            )

            # Generate embeddings
            document.embeddings = await self._generate_embeddings(document)

            # Store document
            self.documents[doc_id] = document

            # Update index
            self._add_to_index(document)

            # Update statistics
            self._update_stats(document)

            # Save to persistent storage
            await self._save_document_batch([document])

            logger.info(f"Added document {doc_id} from {source}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise

    async def update_document(self, doc_id: str, content: str,
                              metadata: Dict[str, Any], source: str) -> str:
        """Update existing document"""
        if doc_id not in self.documents:
            return await self.add_document(content, metadata, source)

        try:
            old_document = self.documents[doc_id]

            # Check if content actually changed
            old_hash = old_document.generate_hash()
            new_hash = hashlib.md5(
                content + json.dumps(metadata, sort_keys=True)
            ).hexdigest()

            if old_hash == new_hash:
                logger.info(f"Document {doc_id} unchanged, skipping update")
                return doc_id

            # Update document
            old_document.content = content
            old_document.metadata.update(metadata)
            old_document.source = source
            old_document.processed_at = datetime.now().isoformat()
            old_document.version += 1

            # Update embeddings
            old_document.embeddings = await self._generate_embeddings(old_document)

            # Update index
            self._remove_from_index(old_document)
            self._add_to_index(old_document)

            # Save to storage
            await self._save_document_batch([old_document])

            logger.info(f"Updated document {doc_id} (version {old_document.version})")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            raise

    async def _generate_embeddings(self, document: KnowledgeDocument) -> List[float]:
        """Generate embeddings for document"""
        try:
            # Prepare text for embedding
            text_for_embedding = f"""
            {document.metadata.get('title', '')}
            {document.content[:1000]}
            """

            # Generate embedding using answer engine
            embedding = self.answer_engine.embedding_model.encode(text_for_embedding)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return []

    def _add_to_index(self, document: KnowledgeDocument):
        """Add document to search index"""
        terms = set(document.content.lower().split()[:100])

        for term in terms:
            if len(term) > 2:
                if term not in self.index:
                    self.index[term] = []

                # Add document ID if not already in list
                if document.id not in self.index[term]:
                    self.index[term].append(document.id)

    def _remove_from_index(self, document: KnowledgeDocument):
        """Remove document from search index"""
        # Rebuild index for simplicity
        # In production, you'd want a more efficient approach
        self._build_index()

    async def _save_document_batch(self, documents: List[KnowledgeDocument]):
        """Save documents to persistent storage"""
        if not documents:
            return

        # Convert to dicts
        doc_dicts = [asdict(doc) for doc in documents]

        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"knowledge_batch_{timestamp}.json"
        filepath = self.storage_path / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(doc_dicts, f, ensure_ascii=False, indent=2)

            logger.debug(f"Saved {len(documents)} documents to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save documents: {e}")

    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents using multiple methods"""
        results = []

        # Method 1: Vector search
        vector_results = await self._vector_search(query, limit)
        results.extend(vector_results)

        # Method 2: Keyword search
        keyword_results = self._keyword_search(query, limit)
        results.extend(keyword_results)

        # Deduplicate and rank results
        unique_results = self._deduplicate_results(results)

        return unique_results[:limit]

    async def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using vector embeddings"""
        try:
            # Generate query embedding
            query_embedding = self.answer_engine.embedding_model.encode(query).tolist()

            # Calculate similarities
            similarities = []
            for doc_id, document in self.documents.items():
                if document.embeddings:
                    similarity = self._cosine_similarity(query_embedding, document.embeddings)
                    similarities.append({
                        'doc_id': doc_id,
                        'similarity': similarity,
                        'document': document,
                        'method': 'vector'
                    })

            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # Format results
            results = []
            for item in similarities[:limit]:
                doc = item['document']
                results.append({
                    'id': doc.id,
                    'content': doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                    'metadata': doc.metadata,
                    'source': doc.source,
                    'similarity': item['similarity'],
                    'method': 'vector',
                    'version': doc.version
                })

            return results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _keyword_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Search using keywords"""
        try:
            # Extract query terms
            query_terms = set(query.lower().split())

            # Score documents
            scores = {}
            for term in query_terms:
                if term in self.index:
                    for doc_id in self.index[term]:
                        scores[doc_id] = scores.get(doc_id, 0) + 1

            # Sort by score
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Format results
            results = []
            for doc_id, score in sorted_docs[:limit]:
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    results.append({
                        'id': doc.id,
                        'content': doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                        'metadata': doc.metadata,
                        'source': doc.source,
                        'score': score / len(query_terms),  # Normalized score
                        'method': 'keyword',
                        'version': doc.version
                    })

            return results

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(b * b for b in vec2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.error(f"Cosine similarity calculation failed: {e}")
            return 0.0

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate and merge search results"""
        seen_ids = set()
        deduplicated = []

        for result in results:
            doc_id = result['id']

            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                deduplicated.append(result)
            else:
                # Merge with existing result
                for existing in deduplicated:
                    if existing['id'] == doc_id:
                        # Take the higher score/similarity
                        existing_score = existing.get('similarity', existing.get('score', 0))
                        new_score = result.get('similarity', result.get('score', 0))

                        if new_score > existing_score:
                            existing.update(result)
                        break

        # Sort by combined score
        deduplicated.sort(
            key=lambda x: x.get('similarity', x.get('score', 0)),
            reverse=True
        )

        return deduplicated

    async def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """Get document by ID"""
        return self.documents.get(doc_id)

    async def delete_document(self, doc_id: str) -> bool:
        """Delete document from knowledge base"""
        if doc_id not in self.documents:
            return False

        try:
            # Remove document
            del self.documents[doc_id]

            # Rebuild index
            self._build_index()

            # Update statistics
            self.stats['total_documents'] = len(self.documents)

            logger.info(f"Deleted document {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    async def export_knowledge(self, format: str = 'json') -> str:
        """Export knowledge base"""
        try:
            if format == 'json':
                # Export as JSON
                export_data = [asdict(doc) for doc in self.documents.values()]
                return json.dumps(export_data, ensure_ascii=False, indent=2)

            elif format == 'csv':
                # Export as CSV
                data = []
                for doc in self.documents.values():
                    row = {
                        'id': doc.id,
                        'source': doc.source,
                        'type': doc.metadata.get('type', ''),
                        'title': doc.metadata.get('title', ''),
                        'content_preview': doc.content[:200],
                        'processed_at': doc.processed_at,
                        'version': doc.version
                    }
                    data.append(row)

                df = pd.DataFrame(data)
                return df.to_csv(index=False, encoding='utf-8')

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

    async def import_knowledge(self, data: str, format: str = 'json') -> Dict[str, Any]:
        """Import knowledge base"""
        try:
            imported_count = 0
            updated_count = 0

            if format == 'json':
                import_data = json.loads(data)

                for item in import_data:
                    try:
                        # Create document
                        doc = KnowledgeDocument(**item)

                        # Check if exists
                        if doc.id in self.documents:
                            # Update existing
                            self.documents[doc.id] = doc
                            updated_count += 1
                        else:
                            # Add new
                            self.documents[doc.id] = doc
                            imported_count += 1

                    except Exception as e:
                        logger.error(f"Error importing item: {e}")
                        continue

            # Rebuild index
            self._build_index()

            # Update statistics
            self.stats['total_documents'] = len(self.documents)

            results = {
                'imported': imported_count,
                'updated': updated_count,
                'total': len(self.documents),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Knowledge import completed: {results}")
            return results

        except Exception as e:
            logger.error(f"Import failed: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            **self.stats,
            'index_size': len(self.index),
            'document_types': list(self.stats['by_type'].keys()),
            'sources': list(self.stats['by_source'].keys())
        }

    async def cleanup(self, max_age_days: int = 30):
        """Cleanup old documents"""
        try:
            cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            removed_count = 0

            doc_ids_to_remove = []

            for doc_id, document in self.documents.items():
                try:
                    doc_date = datetime.fromisoformat(document.processed_at.replace('Z', '+00:00'))
                    if doc_date.timestamp() < cutoff_date:
                        doc_ids_to_remove.append(doc_id)
                except:
                    # If date parsing fails, keep the document
                    continue

            # Remove old documents
            for doc_id in doc_ids_to_remove:
                await self.delete_document(doc_id)
                removed_count += 1

            logger.info(f"Cleanup removed {removed_count} old documents")
            return {'removed': removed_count}

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {'removed': 0, 'error': str(e)}