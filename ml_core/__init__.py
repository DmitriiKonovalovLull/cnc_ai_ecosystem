"""ML components for CNC AI Ecosystem.

Keep heavyweight model dependencies lazy so data contracts and utilities can be
used by lightweight tools and tests without importing PyTorch/Transformers.
"""

from importlib import import_module

_EXPORTS = {
    "IntentRecognizer": ".inference.intent_recognizer",
    "EntityExtractor": ".inference.entity_extractor",
    "AnswerEngine": ".inference.answer_engine",
    "DataCollector": ".training.data_collector",
    "ModelTrainer": ".training.trainer",
    "ModelValidator": ".training.validation",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
