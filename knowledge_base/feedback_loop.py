import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import hashlib
from loguru import logger
import pandas as pd


@dataclass
class FeedbackItem:
    """Feedback item from operator"""
    id: str
    query: str
    original_response: str
    corrected_response: str
    operator_id: str
    confidence: float = 1.0
    timestamp: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

        # Generate ID if not provided
        if not self.id:
            content_str = f"{self.query}{self.corrected_response}{self.operator_id}"
            self.id = f"feedback_{hashlib.md5(content_str.encode()).hexdigest()[:12]}"


class FeedbackLoop:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(__file__).parent.parent / "data" / "feedback"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Feedback storage
        self.feedback_queue: asyncio.Queue = asyncio.Queue()
        self.feedback_items: Dict[str, FeedbackItem] = {}
        self.processed_feedback: List[FeedbackItem] = []

        # Statistics
        self.stats = {
            'total_received': 0,
            'total_processed': 0,
            'by_operator': {},
            'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
            'last_processed': None
        }

        # Load existing feedback
        self._load_existing_feedback()

    def _load_existing_feedback(self):
        """Load existing feedback from storage"""
        feedback_files = list(self.storage_path.glob("feedback_*.json"))

        for filepath in feedback_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for item_data in data:
                    try:
                        item = FeedbackItem(**item_data)
                        self.feedback_items[item.id] = item

                        if item.processed:
                            self.processed_feedback.append(item)

                    except Exception as e:
                        logger.error(f"Error loading feedback item: {e}")
                        continue

                logger.info(f"Loaded {len(data)} feedback items from {filepath}")

            except Exception as e:
                logger.error(f"Error loading feedback from {filepath}: {e}")

        # Update statistics
        self._update_stats()
        logger.info(f"Feedback loop initialized with {len(self.feedback_items)} items")

    def _update_stats(self):
        """Update feedback statistics"""
        self.stats['total_received'] = len(self.feedback_items)
        self.stats['total_processed'] = len(self.processed_feedback)

        # Count by operator
        operator_counts = {}
        confidence_counts = {'high': 0, 'medium': 0, 'low': 0}

        for item in self.feedback_items.values():
            operator_counts[item.operator_id] = operator_counts.get(item.operator_id, 0) + 1

            if item.confidence >= 0.8:
                confidence_counts['high'] += 1
            elif item.confidence >= 0.5:
                confidence_counts['medium'] += 1
            else:
                confidence_counts['low'] += 1

        self.stats['by_operator'] = operator_counts
        self.stats['by_confidence'] = confidence_counts

        if self.processed_feedback:
            self.stats['last_processed'] = max(
                item.timestamp for item in self.processed_feedback
            )

    async def submit_feedback(self, query: str, original_response: str,
                              corrected_response: str, operator_id: str,
                              confidence: float = 1.0,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit feedback from operator"""
        try:
            # Create feedback item
            item = FeedbackItem(
                query=query,
                original_response=original_response,
                corrected_response=corrected_response,
                operator_id=operator_id,
                confidence=confidence,
                metadata=metadata or {}
            )

            # Add to storage
            self.feedback_items[item.id] = item

            # Add to processing queue
            await self.feedback_queue.put(item)

            # Update statistics
            self._update_stats()

            # Save to persistent storage
            await self._save_feedback_batch([item])

            logger.info(f"Feedback submitted: {item.id} from {operator_id}")
            return item.id

        except Exception as e:
            logger.error(f"Failed to submit feedback: {e}")
            raise

    async def _save_feedback_batch(self, items: List[FeedbackItem]):
        """Save feedback items to persistent storage"""
        if not items:
            return

        # Convert to dicts
        item_dicts = [asdict(item) for item in items]

        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"feedback_batch_{timestamp}.json"
        filepath = self.storage_path / filename

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(item_dicts, f, ensure_ascii=False, indent=2)

            logger.debug(f"Saved {len(items)} feedback items to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")

    async def process_feedback(self, processor_callback):
        """Process feedback items from queue"""
        logger.info("Starting feedback processor")

        while True:
            try:
                # Get feedback from queue
                item = await self.feedback_queue.get()

                if item is None:  # Sentinel value to stop
                    break

                # Process feedback
                await self._process_feedback_item(item, processor_callback)

                # Mark as processed
                item.processed = True
                self.processed_feedback.append(item)

                # Update statistics
                self._update_stats()

                # Save processed state
                await self._save_feedback_batch([item])

                self.feedback_queue.task_done()

                logger.debug(f"Processed feedback: {item.id}")

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Error processing feedback: {e}")
                # Put item back in queue for retry
                await asyncio.sleep(1)
                await self.feedback_queue.put(item)

    async def _process_feedback_item(self, item: FeedbackItem, processor_callback):
        """Process single feedback item"""
        try:
            # Extract learning data from feedback
            learning_data = {
                'query': item.query,
                'original_response': item.original_response,
                'corrected_response': item.corrected_response,
                'operator_id': item.operator_id,
                'confidence': item.confidence,
                'metadata': item.metadata,
                'type': 'correction'
            }

            # Call processor callback
            if callable(processor_callback):
                await processor_callback(learning_data)

            # Additional processing can be added here:
            # 1. Update vector embeddings
            # 2. Train models incrementally
            # 3. Update knowledge base

            logger.debug(f"Feedback {item.id} processed successfully")

        except Exception as e:
            logger.error(f"Failed to process feedback item {item.id}: {e}")
            raise

    async def get_feedback_stats(self, time_period: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get feedback statistics for time period"""
        items_to_analyze = self.feedback_items.values()

        if time_period:
            cutoff_time = datetime.now() - time_period
            items_to_analyze = [
                item for item in items_to_analyze
                if datetime.fromisoformat(item.timestamp.replace('Z', '+00:00')) > cutoff_time
            ]

        # Calculate statistics
        stats = {
            'total': len(items_to_analyze),
            'processed': sum(1 for item in items_to_analyze if item.processed),
            'pending': sum(1 for item in items_to_analyze if not item.processed),
            'average_confidence': 0.0,
            'operator_activity': {},
            'time_period': str(time_period) if time_period else 'all_time'
        }

        if items_to_analyze:
            stats['average_confidence'] = sum(
                item.confidence for item in items_to_analyze
            ) / len(items_to_analyze)

            # Operator activity
            operator_counts = {}
            for item in items_to_analyze:
                operator_counts[item.operator_id] = operator_counts.get(item.operator_id, 0) + 1

            stats['operator_activity'] = operator_counts

        return stats

    async def get_training_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get processed feedback as training data"""
        training_data = []

        for item in self.processed_feedback:
            if item.confidence >= 0.7:  # Only high-confidence corrections
                data = {
                    'query': item.query,
                    'correct_response': item.corrected_response,
                    'source': 'operator_feedback',
                    'operator_id': item.operator_id,
                    'confidence': item.confidence,
                    'timestamp': item.timestamp,
                    'metadata': item.metadata
                }
                training_data.append(data)

        # Apply limit if specified
        if limit and len(training_data) > limit:
            training_data = training_data[:limit]

        return training_data

    async def export_feedback(self, format: str = 'json') -> str:
        """Export feedback data"""
        try:
            if format == 'json':
                # Export as JSON
                export_data = [asdict(item) for item in self.feedback_items.values()]
                return json.dumps(export_data, ensure_ascii=False, indent=2)

            elif format == 'csv':
                # Export as CSV
                data = []
                for item in self.feedback_items.values():
                    row = {
                        'id': item.id,
                        'query': item.query[:100],
                        'operator_id': item.operator_id,
                        'confidence': item.confidence,
                        'processed': item.processed,
                        'timestamp': item.timestamp
                    }
                    data.append(row)

                df = pd.DataFrame(data)
                return df.to_csv(index=False, encoding='utf-8')

            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise

    async def cleanup_old_feedback(self, max_age_days: int = 90):
        """Cleanup old feedback items"""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            removed_count = 0

            items_to_remove = []

            for item_id, item in self.feedback_items.items():
                try:
                    item_date = datetime.fromisoformat(item.timestamp.replace('Z', '+00:00'))
                    if item_date < cutoff_date:
                        items_to_remove.append(item_id)
                except:
                    # If date parsing fails, keep the item
                    continue

            # Remove old items
            for item_id in items_to_remove:
                if item_id in self.feedback_items:
                    del self.feedback_items[item_id]
                    removed_count += 1

            # Update processed list
            self.processed_feedback = [
                item for item in self.processed_feedback
                if item.id not in items_to_remove
            ]

            # Update statistics
            self._update_stats()

            logger.info(f"Cleanup removed {removed_count} old feedback items")
            return {'removed': removed_count}

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {'removed': 0, 'error': str(e)}

    def get_queue_status(self) -> Dict[str, Any]:
        """Get feedback queue status"""
        return {
            'queue_size': self.feedback_queue.qsize(),
            'items_in_memory': len(self.feedback_items),
            'items_processed': len(self.processed_feedback),
            'stats': self.stats
        }

    async def stop_processing(self):
        """Stop feedback processing"""
        # Put sentinel value in queue
        await self.feedback_queue.put(None)
        logger.info("Feedback processing stopped")