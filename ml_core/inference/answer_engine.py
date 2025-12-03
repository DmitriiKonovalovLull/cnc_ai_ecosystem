import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
from loguru import logger

from config.settings import settings
from .intent_recognizer import IntentRecognizer
from .entity_extractor import EntityExtractor


class AnswerEngine:
    def __init__(self):
        # Initialize components
        self.intent_recognizer = IntentRecognizer()
        self.entity_extractor = EntityExtractor()

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        # Initialize vector databases
        self._init_vector_dbs()

        # Initialize LLM
        self._init_llm()

        # Cache for frequent queries
        self.query_cache = {}
        self.cache_size = 100

        # Feedback collection
        self.feedback_buffer = []

        logger.info("Answer engine initialized")

    def _init_vector_dbs(self):
        """Initialize vector databases"""
        try:
            # ChromaDB for general knowledge
            self.chroma_client = chromadb.PersistentClient(
                path=str(Path(__file__).parent.parent.parent / "data" / "vector_db" / "chroma"),
                settings=Settings(anonymized_telemetry=False)
            )

            # Try to get existing collection or create new
            try:
                self.chroma_collection = self.chroma_client.get_collection("cnc_knowledge_base")
                logger.info("Loaded existing ChromaDB collection")
            except:
                self.chroma_collection = self.chroma_client.create_collection(
                    name="cnc_knowledge_base",
                    metadata={"description": "CNC Knowledge Base"}
                )
                logger.info("Created new ChromaDB collection")

            # Qdrant for production (optional)
            if settings.QDRANT_URL:
                self.qdrant_client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY
                )

                # Create collection if doesn't exist
                try:
                    self.qdrant_client.get_collection("cnc_knowledge")
                except:
                    self.qdrant_client.create_collection(
                        collection_name="cnc_knowledge",
                        vectors_config=models.VectorParams(
                            size=self.embedding_model.get_sentence_embedding_dimension(),
                            distance=models.Distance.COSINE
                        )
                    )
                    logger.info("Created Qdrant collection")

            logger.success("Vector databases initialized")

        except Exception as e:
            logger.error(f"Failed to initialize vector DBs: {e}")
            raise

    def _init_llm(self):
        """Initialize LLM"""
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.llm_available = True
            logger.info("OpenAI LLM initialized")
        else:
            self.llm_available = False
            logger.warning("OpenAI API key not found, using template responses")

    def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process user query and generate answer"""
        start_time = datetime.now()

        # Check cache first
        cache_key = self._generate_cache_key(query, context)
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            cached_result['cached'] = True
            logger.info(f"Cache hit for query: {query[:50]}...")
            return cached_result

        try:
            # Step 1: Intent recognition
            intent_result = self.intent_recognizer.recognize(query)

            # Step 2: Entity extraction
            entities = self.entity_extractor.extract(query)
            entity_groups = self.entity_extractor.group_entities_by_type(entities)
            parameters = self.entity_extractor.extract_parameters(entities)

            # Step 3: Retrieve relevant context from vector DB
            retrieved_context = self._retrieve_context(query, intent_result, entities)

            # Step 4: Generate answer
            answer = self._generate_answer(
                query=query,
                intent=intent_result,
                entities=entities,
                parameters=parameters,
                retrieved_context=retrieved_context,
                user_context=context
            )

            # Step 5: Prepare response
            response = {
                'query': query,
                'answer': answer['text'],
                'intent': intent_result,
                'entities': [
                    {
                        'text': e.text,
                        'type': e.type,
                        'confidence': e.confidence
                    }
                    for e in entities[:10]  # Limit entities in response
                ],
                'parameters': parameters,
                'sources': retrieved_context.get('sources', []),
                'confidence': answer['confidence'],
                'suggested_actions': self._get_suggested_actions(intent_result, entities),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'timestamp': datetime.now().isoformat(),
                'cached': False
            }

            # Cache the result
            self._cache_result(cache_key, response)

            logger.info(f"Query processed in {response['processing_time']:.2f}s")

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return self._generate_error_response(query, e)

    def _generate_cache_key(self, query: str, context: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        import hashlib

        key_data = query + (json.dumps(context, sort_keys=True) if context else "")
        return hashlib.md5(key_data.encode()).hexdigest()

    def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache query result"""
        if len(self.query_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.query_cache))
            del self.query_cache[oldest_key]

        self.query_cache[cache_key] = result

    def _retrieve_context(self, query: str, intent: Dict[str, Any],
                          entities: List[Any]) -> Dict[str, Any]:
        """Retrieve relevant context from knowledge base"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search in ChromaDB
            chroma_results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=["documents", "metadatas", "distances"]
            )

            # Prepare sources
            sources = []
            relevant_docs = []

            if chroma_results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                        chroma_results['documents'][0],
                        chroma_results['metadatas'][0],
                        chroma_results['distances'][0]
                )):
                    if distance < 0.5:  # Relevance threshold
                        source_info = {
                            'content': doc[:500] + "..." if len(doc) > 500 else doc,
                            'metadata': metadata,
                            'relevance': 1 - distance,
                            'source': 'chroma_db'
                        }
                        sources.append(source_info)
                        relevant_docs.append(doc)

            # Search in Qdrant if available
            if hasattr(self, 'qdrant_client'):
                try:
                    qdrant_results = self.qdrant_client.search(
                        collection_name="cnc_knowledge",
                        query_vector=query_embedding,
                        limit=3
                    )

                    for result in qdrant_results:
                        if result.score > 0.7:  # Qdrant uses similarity score
                            sources.append({
                                'content': result.payload.get('content', '')[:500],
                                'metadata': result.payload.get('metadata', {}),
                                'relevance': result.score,
                                'source': 'qdrant'
                            })
                except Exception as e:
                    logger.debug(f"Qdrant search failed: {e}")

            return {
                'sources': sources,
                'documents': relevant_docs,
                'source_count': len(sources)
            }

        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return {'sources': [], 'documents': [], 'source_count': 0, 'error': str(e)}

    def _generate_answer(self, query: str, intent: Dict[str, Any],
                         entities: List[Any], parameters: Dict[str, Any],
                         retrieved_context: Dict[str, Any],
                         user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate answer using LLM or templates"""

        # Prepare context for generation
        context_parts = []

        # Add retrieved documents
        if retrieved_context.get('documents'):
            context_parts.append("Релевантная информация из базы знаний:")
            for i, doc in enumerate(retrieved_context['documents'][:3], 1):
                context_parts.append(f"{i}. {doc[:300]}...")

        # Add entities
        if entities:
            entity_summary = []
            for entity_type, ents in self.entity_extractor.group_entities_by_type(entities).items():
                if ents:
                    entity_texts = [e.text for e in ents[:3]]
                    entity_summary.append(f"{entity_type}: {', '.join(entity_texts)}")

            if entity_summary:
                context_parts.append("Извлеченные сущности: " + "; ".join(entity_summary))

        # Add parameters
        if parameters:
            param_summary = []
            for param_name, param_data in parameters.items():
                param_summary.append(f"{param_name}: {param_data['value']} {param_data.get('unit', '')}")

            context_parts.append("Параметры: " + ", ".join(param_summary))

        context_text = "\n\n".join(context_parts) if context_parts else "Контекст отсутствует"

        # Generate answer using LLM if available
        if self.llm_available and settings.OPENAI_API_KEY:
            try:
                return self._generate_with_openai(query, context_text, intent)
            except Exception as e:
                logger.error(f"OpenAI generation failed: {e}")
                # Fallback to templates
                return self._generate_with_templates(query, intent, entities, context_text)
        else:
            # Use template-based generation
            return self._generate_with_templates(query, intent, entities, context_text)

    def _generate_with_openai(self, query: str, context: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate answer using OpenAI"""
        try:
            # Prepare system message based on intent
            system_message = self._get_system_message(intent)

            # Prepare user message
            user_message = f"""Вопрос пользователя: {query}

Контекст из базы знаний:
{context}

Пожалуйста, предоставьте:
1. Прямой ответ на вопрос
2. Обоснование на основе контекста
3. Практические рекомендации если применимо
4. Указание на источники информации

Отвечай на русском языке, будь точным и технически корректным."""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            answer_text = response.choices[0].message.content

            return {
                'text': answer_text,
                'confidence': 0.8,
                'method': 'openai',
                'model': settings.LLM_MODEL
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _generate_with_templates(self, query: str, intent: Dict[str, Any],
                                 entities: List[Any], context: str) -> Dict[str, Any]:
        """Generate answer using templates"""
        intent_type = intent.get('intent', 'general_info')

        # Get template based on intent
        template = self._get_answer_template(intent_type, entities)

        # Fill template with entities and context
        answer_text = template['template']

        # Replace placeholders
        if entities:
            # Add entity mentions
            entity_mentions = []
            for entity in entities[:5]:
                entity_mentions.append(f"{entity.type}: {entity.text}")

            if entity_mentions:
                answer_text = answer_text.replace("{entities}", "\n".join(entity_mentions))

        # Add context snippet
        if context and len(context) > 50:
            context_snippet = context[:200] + "..." if len(context) > 200 else context
            answer_text = answer_text.replace("{context}", f"\n\nНа основе информации: {context_snippet}")

        # Clean up unused placeholders
        answer_text = answer_text.replace("{entities}", "").replace("{context}", "")

        return {
            'text': answer_text.strip(),
            'confidence': intent.get('confidence', 0.5) * 0.8,  # Template confidence is lower
            'method': 'template',
            'template_type': template['type']
        }

    def _get_system_message(self, intent: Dict[str, Any]) -> str:
        """Get system message for OpenAI based on intent"""
        intent_type = intent.get('intent', 'general_info')

        messages = {
            'gost_search': """Ты - экспертный ассистент по ГОСТам и стандартам в области машиностроения и CNC обработки.
            Твоя задача - предоставлять точную информацию о стандартах, их применении и требованиях.
            Будь точным, цитируй конкретные пункты стандартов если они есть в контексте.
            Если информации недостаточно, укажи это и предложи где найти больше информации.""",

            'parameter_calculation': """Ты - инженер-технолог с экспертизой в расчетах режимов резания.
            Твои ответы должны быть технически точными, содержать формулы и практические рекомендации.
            Всегда указывай единицы измерения и условия применения расчетов.
            Если данных недостаточно для точного расчета, укажи какие дополнительные данные нужны.""",

            'tool_selection': """Ты - специалист по режущему инструменту для CNC станков.
            Рекомендуй инструменты на основе материала заготовки, типа операции и требований к обработке.
            Учитывай стойкость инструмента, стоимость и доступность.
            Рекомендуй конкретные модели инструментов если они есть в базе знаний.""",

            'general_info': """Ты - экспертный ассистент по CNC обработке и машиностроению.
            Отвечай точно и информативно, используй технические термины правильно.
            Если информация в контексте противоречива, укажи на это.
            Предоставляй практические советы и предупреждения о возможных проблемах."""
        }

        return messages.get(intent_type, messages['general_info'])

    def _get_answer_template(self, intent_type: str, entities: List[Any]) -> Dict[str, Any]:
        """Get answer template for intent type"""
        templates = {
            'gost_search': {
                'template': """На основе вашего запроса о ГОСТах и стандартах:

{entities}

{context}

Основные положения:
1. [Ключевое требование 1]
2. [Ключевое требование 2] 
3. [Рекомендации по применению]

Для получения полного текста стандарта рекомендую обратиться к официальным источникам.""",
                'type': 'gost_info'
            },

            'parameter_calculation': {
                'template': """Для расчета параметров обработки:

{entities}

{context}

Рекомендуемые параметры:
- Скорость резания: [расчет]
- Подача: [расчет] 
- Глубина резания: [рекомендация]

Формулы и обоснования:
[Техническое обоснование]""",
                'type': 'calculation'
            },

            'tool_selection': {
                'template': """Подбор инструмента для указанных условий:

{entities}

{context}

Рекомендуемый инструмент:
- Тип: [рекомендация]
- Материал: [обоснование]
- Геометрия: [параметры]

Рекомендации по применению:
[Практические советы]""",
                'type': 'tool_recommendation'
            }
        }

        return templates.get(intent_type, {
            'template': """Анализ вашего запроса:

{entities}

{context}

Основная информация:
[Обобщенный ответ на вопрос]

Дополнительные рекомендации:
[Практические советы]""",
            'type': 'general'
        })

    def _get_suggested_actions(self, intent: Dict[str, Any], entities: List[Any]) -> List[Dict[str, str]]:
        """Get suggested follow-up actions"""
        actions = []

        intent_type = intent.get('intent')

        # General actions
        actions.append({
            'action': 'clarify',
            'text': 'Уточнить параметры',
            'reason': 'Для более точного ответа'
        })

        # Intent-specific actions
        if intent_type == 'gost_search':
            if any(e.type == 'GOST' for e in entities):
                actions.append({
                    'action': 'download',
                    'text': 'Скачать полный текст ГОСТа',
                    'reason': 'Для детального изучения'
                })

        elif intent_type == 'parameter_calculation':
            actions.append({
                'action': 'calculate',
                'text': 'Рассчитать полную таблицу режимов',
                'reason': 'Для всех материалов и инструментов'
            })

        return actions

    def _generate_error_response(self, query: str, error: Exception) -> Dict[str, Any]:
        """Generate error response"""
        return {
            'query': query,
            'answer': f'Извините, произошла ошибка при обработке запроса: {str(error)}',
            'intent': {'intent': 'error', 'confidence': 0.0},
            'entities': [],
            'parameters': {},
            'sources': [],
            'confidence': 0.0,
            'suggested_actions': [],
            'processing_time': 0,
            'timestamp': datetime.now().isoformat(),
            'cached': False,
            'error': True
        }

    def collect_feedback(self, query: str, response: Dict[str, Any],
                         user_feedback: Dict[str, Any]):
        """Collect feedback for improvement"""
        feedback_entry = {
            'query': query,
            'response': response,
            'user_feedback': user_feedback,
            'timestamp': datetime.now().isoformat()
        }

        self.feedback_buffer.append(feedback_entry)

        # Save to file if buffer is full
        if len(self.feedback_buffer) >= 10:
            self._save_feedback_batch()

    def _save_feedback_batch(self):
        """Save feedback batch to file"""
        if not self.feedback_buffer:
            return

        feedback_dir = Path(__file__).parent.parent / 'training' / 'feedback'
        feedback_dir.mkdir(parents=True, exist_ok=True)

        filename = feedback_dir / f"answer_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_buffer, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(self.feedback_buffer)} feedback entries to {filename}")
            self.feedback_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

    def add_to_knowledge_base(self, document: Dict[str, Any]):
        """Add document to knowledge base"""
        try:
            # Prepare document for indexing
            doc_text = self._prepare_document_text(document)
            doc_id = document.get('id') or f"doc_{hash(doc_text) % 10 ** 10}"

            # Generate embedding
            embedding = self.embedding_model.encode(doc_text).tolist()

            # Add to ChromaDB
            self.chroma_collection.add(
                documents=[doc_text],
                embeddings=[embedding],
                metadatas=[{
                    'source': document.get('source', 'unknown'),
                    'type': document.get('type', 'article'),
                    'title': document.get('title', ''),
                    'gost_codes': document.get('gost_codes', []),
                    'timestamp': datetime.now().isoformat()
                }],
                ids=[doc_id]
            )

            # Add to Qdrant if available
            if hasattr(self, 'qdrant_client'):
                self.qdrant_client.upsert(
                    collection_name="cnc_knowledge",
                    points=[
                        models.PointStruct(
                            id=hash(doc_id) % (2 ** 63 - 1),
                            vector=embedding,
                            payload={
                                'content': doc_text[:1000],
                                'metadata': document.get('metadata', {}),
                                'source': document.get('source', 'unknown')
                            }
                        )
                    ]
                )

            logger.info(f"Added document to knowledge base: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add document to KB: {e}")
            return False

    def _prepare_document_text(self, document: Dict[str, Any]) -> str:
        """Prepare document text for embedding"""
        parts = []

        # Add title
        if 'title' in document:
            parts.append(f"Title: {document['title']}")

        # Add content
        if 'content' in document:
            content = document['content'][:2000]  # Limit content length
            parts.append(f"Content: {content}")

        # Add GOST codes
        if 'gost_codes' in document and document['gost_codes']:
            parts.append(f"GOST Codes: {', '.join(document['gost_codes'][:5])}")

        # Add keywords/tags
        if 'keywords' in document and document['keywords']:
            parts.append(f"Keywords: {', '.join(document['keywords'][:10])}")

        return "\n\n".join(parts)

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar content in knowledge base"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()

            # Search in ChromaDB
            results = self.chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted_results = []
            if results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                )):
                    formatted_results.append({
                        'rank': i + 1,
                        'content': doc[:500] + "..." if len(doc) > 500 else doc,
                        'metadata': metadata,
                        'similarity': 1 - distance,
                        'source': 'chroma_db'
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []