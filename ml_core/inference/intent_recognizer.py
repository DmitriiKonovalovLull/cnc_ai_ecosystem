import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
from pathlib import Path

from config.settings import settings


class IntentRecognizer:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.INTENT_MODEL
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

        self._load_models()

    def _load_models(self):
        """Load ML models"""
        try:
            # Load transformer model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(self.intent_categories)
            ).to(self.device)

            # Load traditional ML model as fallback
            self.tfidf_vectorizer = self._load_or_create('tfidf_vectorizer.pkl')
            self.svm_classifier = self._load_or_create('svm_classifier.pkl')

            print(f"Intent recognizer loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading models: {e}")
            # Fallback to rule-based
            self.model = None
            self.tokenizer = None

    def _load_or_create(self, filename: str):
        """Load model or create default"""
        model_path = Path(__file__).parent / 'models' / 'intent_classifier' / filename

        if model_path.exists():
            return joblib.load(model_path)
        else:
            # Create default model
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.svm import LinearSVC

            if 'vectorizer' in filename:
                return TfidfVectorizer(max_features=1000)
            else:
                return LinearSVC()

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

            intent_id = list(self.intent_categories.keys())[predicted_class]

            return {
                'intent': intent_id,
                'intent_name': self.intent_categories[intent_id],
                'confidence': float(confidence),
                'method': 'deep_learning',
                'all_probabilities': {
                    intent: float(prob)
                    for intent, prob in zip(self.intent_categories.keys(), probabilities[0].cpu().numpy())
                }
            }

        except Exception as e:
            print(f"DL prediction failed: {e}")
            # Fallback
            return self._predict_with_ml(text)

    def _predict_with_ml(self, text: str) -> Dict[str, Any]:
        """Predict using traditional ML"""
        try:
            # Transform text
            X = self.tfidf_vectorizer.transform([text])

            # Predict
            predicted_class = self.svm_classifier.predict(X)[0]
            decision_scores = self.svm_classifier.decision_function(X)[0]

            # Convert to probability-like scores
            probabilities = np.exp(decision_scores) / np.sum(np.exp(decision_scores))
            confidence = probabilities[predicted_class]

            intent_id = list(self.intent_categories.keys())[predicted_class]

            return {
                'intent': intent_id,
                'intent_name': self.intent_categories[intent_id],
                'confidence': float(confidence),
                'method': 'traditional_ml',
                'all_probabilities': {
                    intent: float(prob)
                    for intent, prob in zip(self.intent_categories.keys(), probabilities)
                }
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