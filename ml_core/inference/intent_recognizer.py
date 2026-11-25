import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import json
from pathlib import Path

from config.settings import settings
from ml_core.label_mapping import resolve_intent_label


class IntentRecognizer:
    def __init__(self, model_path: str = None):
        models_root = Path(__file__).parent.parent / 'models'
        trained_model_path = models_root / 'intent_classifier' / 'final_model'
        self.model_path = str(model_path or (trained_model_path if trained_model_path.exists() else settings.INTENT_MODEL))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Intent categories for CNC domain
        self.intent_categories = {
            'gost_search': 'Поиск ГОСТ/стандартов',
            'parameter_calculation': 'Расчет параметров обработки',
            'tool_selection': 'Подбор инструмента',
            'material_info': 'Информация о материалах',
            'troubleshooting': 'Решение проблем',
            'programming': 'Программирование станков',
            'machine_setup': 'Настройка станка',
            'quality_control': 'Контроль качества',
            'safety': 'Техника безопасности',
            'general_info': 'Общая информация'
        }

        # Keywords for each intent
        self.intent_keywords = {
            'gost_search': ['гост', 'ост', 'стандарт', 'iso', 'din', 'норматив'],
            'parameter_calculation': ['рассчитать', 'параметр', 'скорость', 'подача', 'расчет'],
            'tool_selection': ['инструмент', 'фреза', 'сверло', 'пластина', 'подобрать'],
            'material_info': ['материал', 'сталь', 'алюминий', 'титан', 'сплав'],
            'troubleshooting': ['проблема', 'ошибка', 'не работает', 'ломка', 'вибрация'],
            'programming': ['g-код', 'программа', 'код', 'постпроцессор', 'cam'],
            'machine_setup': ['настройка', 'установка', 'калибровка', 'юстировка'],
            'quality_control': ['качество', 'допуск', 'шероховатость', 'точность'],
            'safety': ['безопасность', 'опасность', 'травма', 'защита'],
            'general_info': ['что такое', 'как работает', 'объясните', 'информация']
        }

        self.model = None
        self.tokenizer = None
        self.ml_pipeline = None
        self.id_to_label = {}
        self.ml_id_to_label = {}
        self._load_models()

    def _load_models(self):
        """Load ML models"""
        model_dir = Path(self.model_path)
        mapping_path = model_dir / 'label_mappings.json'
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Transformer model unavailable: {e}")
            self.model = None
            self.tokenizer = None

        if mapping_path.is_file():
            try:
                self.id_to_label = self._read_label_mapping(mapping_path)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
                print(f"Transformer label mapping unavailable: {e}")
                self.model = None
                self.tokenizer = None
        elif self.model is not None:
            self.id_to_label = {
                str(class_id): label
                for class_id, label in getattr(self.model.config, 'id2label', {}).items()
                if label in self.intent_categories
            }

        if self.model is not None and not self.id_to_label:
            # A base model's default LABEL_0 mapping is not a trustworthy intent map.
            self.model = None
            self.tokenizer = None

        ml_model_dir = (
            model_dir.parent / 'ml_model' if model_dir.is_dir()
            else Path(__file__).parent.parent / 'models' / 'intent_classifier' / 'ml_model'
        )
        ml_model_path = ml_model_dir / 'ml_model.pkl'
        ml_mapping_path = ml_model_dir / 'label_mappings.json'
        try:
            if ml_model_path.is_file() and ml_mapping_path.is_file():
                self.ml_pipeline = joblib.load(ml_model_path)
                self.ml_id_to_label = self._read_label_mapping(ml_mapping_path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"Traditional ML model unavailable: {e}")
            self.ml_pipeline = None

    @staticmethod
    def _read_label_mapping(path: Path) -> dict[str, str]:
        with path.open(encoding='utf-8') as mapping_file:
            payload = json.load(mapping_file)
        mapping = payload.get('id_to_label')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('label mapping must contain a non-empty id_to_label object')
        return {str(class_id): str(label) for class_id, label in mapping.items()}

    def recognize(self, text: str) -> Dict[str, Any]:
        """Recognize intent from text"""
        text_lower = text.lower().strip()

        # Try deep learning model first
        if self.model and self.tokenizer:
            dl_result = self._predict_with_dl(text_lower)
            if dl_result['confidence'] > 0.7:
                return dl_result

        # Fallback to traditional ML
        ml_result = self._predict_with_ml(text_lower)
        if ml_result['confidence'] > 0.6:
            return ml_result

        # Fallback to rule-based
        return self._predict_with_rules(text_lower)

    def _predict_with_dl(self, text: str) -> Dict[str, Any]:
        """Predict using deep learning model"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()

            intent_id = resolve_intent_label(predicted_class, self.id_to_label)
            probabilities_by_intent = {
                resolve_intent_label(class_id, self.id_to_label): float(probability)
                for class_id, probability in enumerate(probabilities[0].cpu().numpy())
            }

            return {
                'intent': intent_id,
                'intent_name': self.intent_categories.get(intent_id, intent_id),
                'confidence': float(confidence),
                'method': 'deep_learning',
                'all_probabilities': probabilities_by_intent
            }

        except Exception as e:
            print(f"DL prediction failed: {e}")
            # Fallback
            return self._predict_with_ml(text)

    def _predict_with_ml(self, text: str) -> Dict[str, Any]:
        """Predict using traditional ML"""
        try:
            if self.ml_pipeline is None:
                raise RuntimeError('No trained traditional ML pipeline is available')
            predicted_class = int(self.ml_pipeline.predict([text])[0])
            decision_scores = np.asarray(self.ml_pipeline.decision_function([text]), dtype=float)
            classes = np.asarray(self.ml_pipeline.named_steps['classifier'].classes_, dtype=int)
            if decision_scores.ndim == 1:
                if len(classes) != 2:
                    raise ValueError('Unexpected decision score shape for the trained classifier')
                decision_scores = np.array([-decision_scores[0], decision_scores[0]])
            else:
                decision_scores = decision_scores[0]
            if len(decision_scores) != len(classes):
                raise ValueError('Classifier scores do not match the trained class list')

            shifted_scores = decision_scores - np.max(decision_scores)
            scores = np.exp(shifted_scores)
            probabilities = scores / np.sum(scores)
            intent_id = resolve_intent_label(predicted_class, self.ml_id_to_label)
            confidence = float(probabilities[np.where(classes == predicted_class)[0][0]])
            probabilities_by_intent = {
                resolve_intent_label(class_id, self.ml_id_to_label): float(probability)
                for class_id, probability in zip(classes, probabilities)
            }

            return {
                'intent': intent_id,
                'intent_name': self.intent_categories.get(intent_id, intent_id),
                'confidence': float(confidence),
                'method': 'traditional_ml',
                'all_probabilities': probabilities_by_intent
            }

        except Exception as e:
            print(f"ML prediction failed: {e}")
            # Fallback to rules
            return self._predict_with_rules(text)

    def _predict_with_rules(self, text: str) -> Dict[str, Any]:
        """Predict using rule-based approach"""
        scores = {}

        # Calculate keyword scores
        for intent_id, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    # Weight by keyword importance
                    score += 1

            # Normalize score
            scores[intent_id] = score / len(keywords) if keywords else 0

        # Find best match
        best_intent = max(scores.items(), key=lambda x: x[1])

        # Calculate confidence (normalize between 0.3 and 0.7 for rule-based)
        confidence = 0.3 + (best_intent[1] * 0.4)

        return {
            'intent': best_intent[0],
            'intent_name': self.intent_categories.get(best_intent[0], 'Неизвестно'),
            'confidence': float(confidence),
            'method': 'rule_based',
            'keyword_scores': scores
        }

    def extract_subintent(self, text: str, main_intent: str) -> Dict[str, Any]:
        """Extract sub-intent or specific aspect"""
        subintents = {
            'gost_search': {
                'download': ['скачать', 'загрузить', 'файл'],
                'info': ['информация', 'описание', 'что такое'],
                'application': ['применение', 'использование', 'где'],
                'parameters': ['параметры', 'размеры', 'требования']
            },
            'parameter_calculation': {
                'cutting_speed': ['скорость резания', 'vc', 'v_c'],
                'feed_rate': ['подача', 'f', 'feed'],
                'depth_of_cut': ['глубина резания', 'ap', 'a_p'],
                'power': ['мощность', 'энергия', 'сила']
            }
            # Add more subintents as needed
        }

        subintent_info = subintents.get(main_intent, {})
        detected_subintents = []

        for subintent, keywords in subintent_info.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    detected_subintents.append(subintent)
                    break

        return {
            'main_intent': main_intent,
            'subintents': detected_subintents,
            'has_subintent': len(detected_subintents) > 0
        }

    def get_response_template(self, intent: str) -> Dict[str, Any]:
        """Get response template for intent"""
        templates = {
            'gost_search': {
                'greeting': 'Поищу информацию о ГОСТах и стандартах...',
                'structure': 'Сначала найду актуальные стандарты, затем предоставлю детали.'
            },
            'parameter_calculation': {
                'greeting': 'Помогу рассчитать параметры обработки...',
                'structure': 'Нужно узнать материал, инструмент и желаемые результаты.'
            },
            'tool_selection': {
                'greeting': 'Помогу подобрать подходящий инструмент...',
                'structure': 'Уточните материал заготовки, тип операции и требования к обработке.'
            }
        }

        return templates.get(intent, {
            'greeting': 'Анализирую ваш запрос...',
            'structure': 'Ищу наиболее релевантную информацию.'
        })

    def save_feedback(self, text: str, predicted_intent: str, correct_intent: str):
        """Save feedback for model improvement"""
        feedback_path = Path(__file__).parent.parent / 'training' / 'feedback' / 'intent_feedback.csv'
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        import pandas as pd
        from datetime import datetime

        feedback_data = {
            'text': text,
            'predicted_intent': predicted_intent,
            'correct_intent': correct_intent,
            'timestamp': datetime.now().isoformat(),
            'needs_training': predicted_intent != correct_intent
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
