import re
import spacy
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import pipeline
import torch

from config.settings import settings


@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    text: str
    type: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EntityExtractor:
    def __init__(self):
        self.nlp = None
        self.ner_pipeline = None
        self.device = 0 if torch.cuda.is_available() else -1

        # Entity patterns for CNC domain
        self.entity_patterns = {
            'GOST': [
                r'ГОСТ\s+[\d\-\.]+',
                r'ГОСТ\s+[ИСО\d\-\.]+',
                r'ОСТ\s+[\d\-\.]+',
                r'СТ\s+[А-Я\d\-\.]+',
                r'ISO\s+[\d\-\.]+',
                r'DIN\s+[\d\-\.]+',
                r'ANSI\s+[A-Z]+\s+[\d\-\.]+',
                r'JIS\s+[A-Z]+\s+[\d\-\.]+'
            ],
            'MATERIAL': [
                r'сталь\s+(?:[0-9]+|[0-9Хх]+)',
                r'алюминий\s+[А-Я0-9]+',
                r'титан\s+[А-Я0-9]+',
                r'чугун\s+[А-Я0-9]+',
                r'сплав\s+[А-Я0-9]+',
                r'Steel\s+[A-Z0-9]+',
                r'Aluminum\s+[A-Z0-9]+',
                r'Titanium\s+[A-Z0-9]+',
                r'Cast\s+Iron\s+[A-Z0-9]+'
            ],
            'TOOL': [
                r'фреза\s+(?:[А-Яа-яA-Z0-9\-\s]+)',
                r'сверло\s+(?:[А-Яа-яA-Z0-9\-\s]+)',
                r'резец\s+(?:[А-Яа-яA-Z0-9\-\s]+)',
                r'пластина\s+[A-Z0-9\-]+',
                r'insert\s+[A-Z0-9\-]+',
                r'end\s+mill\s+[A-Z0-9\-]+',
                r'drill\s+[A-Z0-9\-]+',
                r'tap\s+[A-Z0-9\-]+'
            ],
            'PARAMETER': [
                r'скорость\s+резания\s*[:=]?\s*[\d\.]+\s*(?:м/мин|m/min)',
                r'подача\s*[:=]?\s*[\d\.]+\s*(?:мм/об|mm/rev)',
                r'глубина\s+резания\s*[:=]?\s*[\d\.]+\s*(?:мм|mm)',
                r'стойкость\s*[:=]?\s*[\d\.]+\s*(?:мин|min)',
                r'шероховатость\s*[:=]?\s*Ra\s*[\d\.]+',
                r'допуск\s*[:=]?\s*[\d\.]+\s*(?:мм|mm)',
                r'частота\s+вращения\s*[:=]?\s*[\d\.]+\s*(?:об/мин|rpm)',
                r'мощность\s*[:=]?\s*[\d\.]+\s*(?:кВт|kW)'
            ],
            'MACHINE': [
                r'станок\s+(?:[А-Яа-яA-Z0-9\-\s]+)',
                r'ЧПУ\s+(?:[А-Яа-яA-Z0-9\-\s]+)',
                r'CNC\s+(?:[A-Z0-9\-\s]+)',
                r'токарный\s+станок',
                r'фрезерный\s+станок',
                r'сверлильный\s+станок',
                r'шлифовальный\s+станок'
            ],
            'OPERATION': [
                r'токарная\s+обработка',
                r'фрезерование',
                r'сверление',
                r'растачивание',
                r'нарезание\s+резьбы',
                r'шлифование',
                r'зенкерование',
                r'развертывание'
            ]
        }

        self._load_models()

    def _load_models(self):
        """Load NLP models"""
        try:
            # Load spaCy for Russian
            try:
                self.nlp = spacy.load("ru_core_news_lg")
            except:
                # Fallback to medium model
                self.nlp = spacy.load("ru_core_news_md")

            # Load transformer NER model
            self.ner_pipeline = pipeline(
                "ner",
                model=settings.NER_MODEL,
                device=self.device,
                aggregation_strategy="simple"
            )

            print("Entity extractor models loaded successfully")

        except Exception as e:
            print(f"Error loading models: {e}")
            # Models will be None, use pattern matching only

    def extract(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from text using multiple methods"""
        all_entities = []

        # Method 1: Pattern matching
        pattern_entities = self._extract_with_patterns(text)
        all_entities.extend(pattern_entities)

        # Method 2: spaCy NER
        if self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            all_entities.extend(spacy_entities)

        # Method 3: Transformer NER
        if self.ner_pipeline:
            transformer_entities = self._extract_with_transformer(text)
            all_entities.extend(transformer_entities)

        # Remove duplicates and overlaps
        unique_entities = self._deduplicate_entities(all_entities)

        return unique_entities

    def _extract_with_patterns(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using regex patterns"""
        entities = []

        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = ExtractedEntity(
                        text=match.group(0),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,  # High confidence for pattern matches
                        metadata={'method': 'pattern_matching', 'pattern': pattern}
                    )
                    entities.append(entity)

        return entities

    def _extract_with_spacy(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using spaCy"""
        entities = []

        doc = self.nlp(text)

        for ent in doc.ents:
            # Map spaCy entity types to our types
            mapped_type = self._map_spacy_entity_type(ent.label_)

            if mapped_type:
                entity = ExtractedEntity(
                    text=ent.text,
                    type=mapped_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.7,
                    metadata={
                        'method': 'spacy',
                        'original_label': ent.label_,
                        'spacy_label': ent.label_
                    }
                )
                entities.append(entity)

        return entities

    def _extract_with_transformer(self, text: str) -> List[ExtractedEntity]:
        """Extract entities using transformer model"""
        entities = []

        try:
            ner_results = self.ner_pipeline(text)

            for result in ner_results:
                mapped_type = self._map_ner_entity_type(result['entity_group'])

                if mapped_type:
                    entity = ExtractedEntity(
                        text=result['word'],
                        type=mapped_type,
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        metadata={
                            'method': 'transformer',
                            'original_label': result['entity_group'],
                            'score': result['score']
                        }
                    )
                    entities.append(entity)

        except Exception as e:
            print(f"Transformer NER failed: {e}")

        return entities

    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to our entity types"""
        mapping = {
            'ORG': 'ORGANIZATION',
            'PER': 'PERSON',
            'LOC': 'LOCATION',
            'MISC': 'MISC',
            'PRODUCT': 'TOOL',
            'EVENT': 'OPERATION'
        }

        return mapping.get(spacy_label)

    def _map_ner_entity_type(self, ner_label: str) -> Optional[str]:
        """Map NER model labels to our entity types"""
        mapping = {
            'ORG': 'ORGANIZATION',
            'PER': 'PERSON',
            'LOC': 'LOCATION',
            'MISC': 'MISC',
            'PRODUCT': 'TOOL',
            'WORK_OF_ART': 'GOST'  # Sometimes GOSTs are classified as works of art
        }

        return mapping.get(ner_label)

    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate and overlapping entities"""
        if not entities:
            return []

        # Sort by start position
        entities.sort(key=lambda x: x.start)

        unique_entities = []
        current_end = -1

        for entity in entities:
            # Check for overlap
            if entity.start >= current_end:
                # No overlap, add entity
                unique_entities.append(entity)
                current_end = entity.end
            else:
                # Overlap detected, choose entity with higher confidence
                last_entity = unique_entities[-1]

                if entity.confidence > last_entity.confidence:
                    # Replace with higher confidence entity
                    unique_entities[-1] = entity
                    current_end = entity.end
                elif entity.confidence == last_entity.confidence:
                    # Same confidence, choose longer entity
                    if (entity.end - entity.start) > (last_entity.end - last_entity.start):
                        unique_entities[-1] = entity
                        current_end = entity.end

        return unique_entities

    def group_entities_by_type(self, entities: List[ExtractedEntity]) -> Dict[str, List[ExtractedEntity]]:
        """Group entities by their type"""
        grouped = {}

        for entity in entities:
            if entity.type not in grouped:
                grouped[entity.type] = []
            grouped[entity.type].append(entity)

        return grouped

    def extract_parameters(self, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        """Extract and structure parameters from entities"""
        parameters = {}

        for entity in entities:
            if entity.type == 'PARAMETER':
                # Parse parameter value and unit
                parsed = self._parse_parameter(entity.text)
                if parsed:
                    param_name = parsed['name']
                    parameters[param_name] = {
                        'value': parsed['value'],
                        'unit': parsed['unit'],
                        'source': entity.text,
                        'confidence': entity.confidence
                    }

        return parameters

    def _parse_parameter(self, param_text: str) -> Optional[Dict[str, Any]]:
        """Parse parameter text into name, value, and unit"""
        # Common parameter patterns
        patterns = {
            'cutting_speed': [
                r'скорость\s+резания\s*[:=]?\s*([\d\.]+)\s*(м/мин|m/min)',
                r'cutting\s+speed\s*[:=]?\s*([\d\.]+)\s*(m/min)'
            ],
            'feed_rate': [
                r'подача\s*[:=]?\s*([\d\.]+)\s*(мм/об|mm/rev)',
                r'feed\s+rate\s*[:=]?\s*([\d\.]+)\s*(mm/rev)'
            ],
            'depth_of_cut': [
                r'глубина\s+резания\s*[:=]?\s*([\d\.]+)\s*(мм|mm)',
                r'depth\s+of\s+cut\s*[:=]?\s*([\d\.]+)\s*(mm)'
            ],
            'surface_roughness': [
                r'шероховатость\s*[:=]?\s*Ra\s*([\d\.]+)',
                r'surface\s+roughness\s*[:=]?\s*Ra\s*([\d\.]+)'
            ],
            'tolerance': [
                r'допуск\s*[:=]?\s*([\d\.]+)\s*(мм|mm)',
                r'tolerance\s*[:=]?\s*([\d\.]+)\s*(mm)'
            ]
        }

        for param_name, param_patterns in patterns.items():
            for pattern in param_patterns:
                match = re.search(pattern, param_text, re.IGNORECASE)
                if match:
                    return {
                        'name': param_name,
                        'value': float(match.group(1)),
                        'unit': match.group(2)
                    }

        return None

    def validate_entities(self, entities: List[ExtractedEntity], context: str = None) -> List[ExtractedEntity]:
        """Validate extracted entities with context"""
        validated_entities = []

        for entity in entities:
            # Check entity validity
            is_valid = self._validate_entity(entity, context)

            if is_valid:
                validated_entities.append(entity)
            else:
                # Mark as low confidence but keep
                entity.confidence *= 0.5
                validated_entities.append(entity)

        return validated_entities

    def _validate_entity(self, entity: ExtractedEntity, context: str = None) -> bool:
        """Validate a single entity"""
        # Basic validation rules
        if len(entity.text.strip()) < 2:
            return False

        # Check for common false positives
        false_positives = [
            'и т.д.',
            'и др.',
            'например',
            'около',
            'примерно'
        ]

        if entity.text.lower() in false_positives:
            return False

        # Type-specific validation
        if entity.type == 'GOST':
            # GOST codes should have numbers
            if not any(char.isdigit() for char in entity.text):
                return False

        elif entity.type == 'MATERIAL':
            # Materials should not be single letters
            if len(entity.text.strip()) < 3:
                return False

        return True

    def save_feedback(self, text: str, entities: List[ExtractedEntity],
                      corrections: List[ExtractedEntity] = None):
        """Save feedback for model improvement"""
        feedback_path = Path(__file__).parent.parent / 'training' / 'feedback' / 'entity_feedback.csv'
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd
        from datetime import datetime

        feedback_data = {
            'text': text,
            'extracted_entities': str([e.text for e in entities]),
            'extracted_types': str([e.type for e in entities]),
            'corrections': str([e.text for e in corrections]) if corrections else '',
            'correction_types': str([e.type for e in corrections]) if corrections else '',
            'timestamp': datetime.now().isoformat()
        }

        # Append to feedback file
        try:
            if feedback_path.exists():
                df = pd.read_csv(feedback_path)
                df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
            else:
                df = pd.DataFrame([feedback_data])

            df.to_csv(feedback_path, index=False, encoding='utf-8')

        except Exception as e:
            print(f"Error saving feedback: {e}")