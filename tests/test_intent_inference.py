"""Regression tests for inference contracts with local model doubles."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import numpy as np
import torch


def load_recognizer_class():
    # No network, model downloads, application settings, or database required.
    transformers = types.ModuleType('transformers')
    transformers.AutoTokenizer = Mock()
    transformers.AutoModelForSequenceClassification = Mock()
    settings = types.ModuleType('config.settings')
    settings.settings = types.SimpleNamespace(INTENT_MODEL='unused/base-model')
    path = Path(__file__).parents[1] / 'ml_core/inference/intent_recognizer.py'
    spec = importlib.util.spec_from_file_location('intent_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'transformers': transformers, 'config.settings': settings}):
        spec.loader.exec_module(module)
    return module


class IntentInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_recognizer_class()

    def test_binary_svm_uses_saved_class_mapping(self):
        recognizer = self.module.IntentRecognizer.__new__(self.module.IntentRecognizer)
        recognizer.intent_categories = {'alpha': 'Alpha', 'zeta': 'Zeta'}
        recognizer.ml_id_to_label = {'0': 'zeta', '1': 'alpha'}
        pipeline = Mock()
        pipeline.predict.return_value = np.array([1])
        pipeline.decision_function.return_value = np.array([1000.0])
        pipeline.named_steps = {'classifier': types.SimpleNamespace(classes_=np.array([0, 1]))}
        recognizer.ml_pipeline = pipeline
        result = recognizer._predict_with_ml('test input')
        self.assertEqual(result['intent'], 'alpha')
        self.assertEqual(result['method'], 'traditional_ml')
        self.assertTrue(np.isfinite(result['confidence']))
        self.assertAlmostEqual(sum(result['all_probabilities'].values()), 1.0)

    def test_transformer_result_uses_saved_mapping_instead_of_ui_order(self):
        recognizer = self.module.IntentRecognizer.__new__(self.module.IntentRecognizer)
        recognizer.device = 'cpu'
        recognizer.intent_categories = {'zeta': 'Zeta', 'alpha': 'Alpha'}
        recognizer.id_to_label = {'0': 'alpha', '1': 'zeta'}
        recognizer.tokenizer = Mock(return_value=types.SimpleNamespace(to=lambda device: {}))
        recognizer.model = Mock(return_value=types.SimpleNamespace(logits=torch.tensor([[10.0, 0.0]])))
        result = recognizer._predict_with_dl('test input')
        self.assertEqual(result['intent'], 'alpha')
        self.assertGreater(result['all_probabilities']['alpha'], 0.99)

    def test_multiclass_scores_follow_classifier_classes(self):
        recognizer = self.module.IntentRecognizer.__new__(self.module.IntentRecognizer)
        recognizer.intent_categories = {}
        recognizer.ml_id_to_label = {'1': 'alpha', '4': 'beta', '7': 'gamma'}
        pipeline = Mock()
        pipeline.predict.return_value = np.array([4])
        pipeline.decision_function.return_value = np.array([[0.0, 10.0, -2.0]])
        pipeline.named_steps = {'classifier': types.SimpleNamespace(classes_=np.array([1, 4, 7]))}
        recognizer.ml_pipeline = pipeline
        result = recognizer._predict_with_ml('test input')
        self.assertEqual(result['intent'], 'beta')
        self.assertGreater(result['all_probabilities']['beta'], 0.9)

    def test_trained_pipeline_loads_when_transformer_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'final_model'
            model.mkdir()
            fallback = root / 'ml_model'
            fallback.mkdir()
            (fallback / 'ml_model.pkl').write_bytes(b'local test sentinel')
            (fallback / 'label_mappings.json').write_text(
                json.dumps({'id_to_label': {'0': 'gost_search'}}), encoding='utf-8'
            )
            with patch.object(self.module.AutoTokenizer, 'from_pretrained', side_effect=OSError('offline')), patch.object(
                self.module.joblib, 'load', return_value='loaded pipeline'
            ) as loader:
                recognizer = self.module.IntentRecognizer(str(model))
            self.assertEqual(recognizer.ml_pipeline, 'loaded pipeline')
            self.assertEqual(recognizer.ml_id_to_label, {'0': 'gost_search'})
            loader.assert_called_once_with(fallback / 'ml_model.pkl')


if __name__ == '__main__':
    unittest.main()
