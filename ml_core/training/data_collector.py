import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataPoint:
    """Single training data point"""
    query: str
    intent: str
    entities: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None
    source: str = "user"
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class DataCollector:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data" / "training"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Data storage
        self.training_data = []
        self.feedback_data = []
        self.batch_size = 100

        # Statistics
        self.stats = {
            'total_points': 0,
            'by_intent': {},
            'by_source': {},
            'last_updated': None
        }

    async def collect_from_feedback(self, feedback_file: Path):
        """Collect training data from feedback files"""
        if not feedback_file.exists():
            logger.warning(f"Feedback file not found: {feedback_file}")
            return

        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)

            for feedback in feedback_data:
                if isinstance(feedback, dict):
                    data_point = TrainingDataPoint(
                        query=feedback.get('query', ''),
                        intent=feedback.get('correct_intent', 'unknown'),
                        entities=feedback.get('entities', []),
                        response=feedback.get('response', ''),
                        feedback={'type': 'correction', 'data': feedback},
                        source='feedback_file',
                        timestamp=feedback.get('timestamp')
                    )

                    self.training_data.append(data_point)

            logger.info(f"Collected {len(feedback_data)} data points from feedback")

        except Exception as e:
            logger.error(f"Error reading feedback file: {e}")

    async def collect_from_api(self, api_data: List[Dict[str, Any]]):
        """Collect training data from API interactions"""
        for item in api_data:
            try:
                data_point = TrainingDataPoint(
                    query=item.get('query', ''),
                    intent=item.get('intent', {}).get('intent', 'unknown'),
                    entities=item.get('entities', []),
                    response=item.get('answer', ''),
                    context=item.get('context', {}),
                    source='api_interaction',
                    timestamp=item.get('timestamp')
                )

                self.training_data.append(data_point)

            except Exception as e:
                logger.error(f"Error processing API data: {e}")
                continue

        logger.info(f"Collected {len(api_data)} data points from API")

    async def collect_from_operators(self, operator_data: List[Dict[str, Any]]):
        """Collect training data from operator corrections"""
        for correction in operator_data:
            try:
                data_point = TrainingDataPoint(
                    query=correction.get('original_query', ''),
                    intent=correction.get('corrected_intent', 'unknown'),
                    entities=correction.get('corrected_entities', []),
                    response=correction.get('corrected_response', ''),
                    feedback={
                        'type': 'operator_correction',
                        'operator_id': correction.get('operator_id'),
                        'confidence': correction.get('confidence', 1.0)
                    },
                    source='operator',
                    timestamp=correction.get('timestamp')
                )

                self.training_data.append(data_point)

            except Exception as e:
                logger.error(f"Error processing operator data: {e}")
                continue

        logger.info(f"Collected {len(operator_data)} data points from operators")

    async def collect_from_scraping(self, scraped_data: List[Dict[str, Any]]):
        """Collect training data from scraped content"""
        for item in scraped_data:
            try:
                # Extract potential Q&A pairs from content
                qa_pairs = self._extract_qa_pairs(item.get('content', ''))

                for qa in qa_pairs:
                    data_point = TrainingDataPoint(
                        query=qa.get('question', ''),
                        intent=self._infer_intent_from_text(qa.get('question', '')),
                        entities=self._extract_entities_from_text(qa.get('question', '')),
                        response=qa.get('answer', ''),
                        source='scraped_content',
                        timestamp=item.get('processed_at')
                    )

                    self.training_data.append(data_point)

            except Exception as e:
                logger.error(f"Error processing scraped data: {e}")
                continue

        logger.info(f"Collected {len(scraped_data)} data points from scraping")

    def _extract_qa_pairs(self, content: str) -> List[Dict[str, str]]:
        """Extract potential Q&A pairs from text"""
        qa_pairs = []

        # Simple pattern matching for Q&A
        lines = content.split('\n')

        i = 0
        while i < len(lines) - 1:
            line = lines[i].strip()
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""

            # Check if line looks like a question
            if self._looks_like_question(line) and len(next_line) > 20:
                qa_pairs.append({
                    'question': line,
                    'answer': next_line[:500]  # Limit answer length
                })
                i += 2
            else:
                i += 1

        return qa_pairs

    def _looks_like_question(self, text: str) -> bool:
        """Check if text looks like a question"""
        question_indicators = [
            '?', 'как', 'что', 'почему', 'где', 'когда', 'кто', 'чем',
            'how', 'what', 'why', 'where', 'when', 'who'
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in question_indicators)

    def _infer_intent_from_text(self, text: str) -> str:
        """Infer intent from text"""
        text_lower = text.lower()

        intent_keywords = {
            'gost_search': ['гост', 'стандарт', 'iso', 'din'],
            'parameter_calculation': ['рассчитать', 'параметр', 'скорость', 'подача'],
            'tool_selection': ['инструмент', 'фреза', 'сверло', 'пластина'],
            'material_info': ['материал', 'сталь', 'алюминий', 'титан'],
            'troubleshooting': ['проблема', 'ошибка', 'не работает', 'вибрация']
        }

        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return intent

        return 'general_info'

    def _extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using simple patterns"""
        entities = []

        # GOST patterns
        gost_patterns = [
            r'ГОСТ\s+[\d\-\.]+',
            r'ОСТ\s+[\d\-\.]+',
            r'ISO\s+[\d\-\.]+'
        ]

        for pattern in gost_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match,
                    'type': 'GOST',
                    'start': text.find(match),
                    'end': text.find(match) + len(match)
                })

        # Material patterns
        material_patterns = [
            r'сталь\s+[0-9Хх]+',
            r'алюминий\s+[А-Я0-9]+'
        ]

        for pattern in material_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match,
                    'type': 'MATERIAL',
                    'start': text.find(match),
                    'end': text.find(match) + len(match)
                })

        return entities

    def save_batch(self, batch_name: Optional[str] = None):
        """Save collected data to file"""
        if not self.training_data:
            logger.info("No data to save")
            return

        # Generate filename
        if batch_name:
            filename = f"training_data_{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        else:
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.data_dir / filename

        # Convert dataclasses to dicts
        data_dicts = [asdict(point) for point in self.training_data]

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_dicts, f, ensure_ascii=False, indent=2)

            # Update statistics
            self._update_stats(data_dicts)

            logger.info(f"Saved {len(data_dicts)} data points to {filepath}")

            # Clear in-memory data
            self.training_data.clear()

        except Exception as e:
            logger.error(f"Error saving training data: {e}")

    def _update_stats(self, data_points: List[Dict[str, Any]]):
        """Update collection statistics"""
        self.stats['total_points'] += len(data_points)
        self.stats['last_updated'] = datetime.now().isoformat()

        for point in data_points:
            # Count by intent
            intent = point.get('intent', 'unknown')
            self.stats['by_intent'][intent] = self.stats['by_intent'].get(intent, 0) + 1

            # Count by source
            source = point.get('source', 'unknown')
            self.stats['by_source'][source] = self.stats['by_source'].get(source, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return self.stats

    def load_training_data(self, limit: Optional[int] = None) -> List[TrainingDataPoint]:
        """Load training data from files"""
        all_data = []

        # Find all training data files
        data_files = list(self.data_dir.glob("training_data_*.json"))

        for filepath in data_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert dicts back to dataclasses
                for item in data:
                    data_point = TrainingDataPoint(
                        query=item.get('query', ''),
                        intent=item.get('intent', 'unknown'),
                        entities=item.get('entities', []),
                        context=item.get('context'),
                        response=item.get('response'),
                        feedback=item.get('feedback'),
                        source=item.get('source', 'unknown'),
                        timestamp=item.get('timestamp')
                    )
                    all_data.append(data_point)

            except Exception as e:
                logger.error(f"Error loading training data from {filepath}: {e}")
                continue

        # Apply limit if specified
        if limit and len(all_data) > limit:
            all_data = all_data[:limit]

        logger.info(f"Loaded {len(all_data)} training data points")
        return all_data

    def create_dataset(self, data_points: List[TrainingDataPoint]) -> Dict[str, Any]:
        """Create training dataset from data points"""
        dataset = {
            'queries': [],
            'intents': [],
            'entities': [],
            'responses': [],
            'metadata': []
        }

        for point in data_points:
            dataset['queries'].append(point.query)
            dataset['intents'].append(point.intent)
            dataset['entities'].append(point.entities)
            dataset['responses'].append(point.response or '')
            dataset['metadata'].append({
                'source': point.source,
                'timestamp': point.timestamp,
                'has_feedback': point.feedback is not None
            })

        # Add statistics
        dataset['stats'] = {
            'total_samples': len(data_points),
            'intent_distribution': {},
            'source_distribution': {}
        }

        # Calculate distributions
        intents = dataset['intents']
        sources = [m['source'] for m in dataset['metadata']]

        for intent in set(intents):
            dataset['stats']['intent_distribution'][intent] = intents.count(intent)

        for source in set(sources):
            dataset['stats']['source_distribution'][source] = sources.count(source)

        return dataset

    async def process_feedback_loop(self, feedback_queue: asyncio.Queue):
        """Process feedback from queue"""
        logger.info("Starting feedback loop processor")

        while True:
            try:
                feedback = await feedback_queue.get()

                if feedback is None:  # Sentinel value to stop
                    break

                # Process feedback
                await self._process_feedback(feedback)

                # Save periodically
                if len(self.training_data) >= self.batch_size:
                    self.save_batch("feedback_batch")

                feedback_queue.task_done()

            except Exception as e:
                logger.error(f"Error processing feedback: {e}")

    async def _process_feedback(self, feedback: Dict[str, Any]):
        """Process single feedback item"""
        try:
            feedback_type = feedback.get('type', 'general')

            if feedback_type == 'intent_correction':
                data_point = TrainingDataPoint(
                    query=feedback.get('query', ''),
                    intent=feedback.get('correct_intent', 'unknown'),
                    entities=feedback.get('entities', []),
                    feedback=feedback,
                    source='feedback_correction',
                    timestamp=feedback.get('timestamp')
                )
                self.training_data.append(data_point)

            elif feedback_type == 'response_quality':
                # Log quality feedback for analysis
                self.feedback_data.append(feedback)

            elif feedback_type == 'operator_correction':
                data_point = TrainingDataPoint(
                    query=feedback.get('original_query', ''),
                    intent=feedback.get('corrected_intent', 'unknown'),
                    entities=feedback.get('corrected_entities', []),
                    response=feedback.get('corrected_response', ''),
                    feedback=feedback,
                    source='operator_correction',
                    timestamp=feedback.get('timestamp')
                )
                self.training_data.append(data_point)

            logger.debug(f"Processed feedback of type: {feedback_type}")

        except Exception as e:
            logger.error(f"Error processing feedback item: {e}")