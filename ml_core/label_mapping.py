"""Helpers for keeping classifier output IDs aligned with intent labels."""

from collections import Counter
from math import ceil
from typing import Iterable, Mapping


def build_label_mappings(labels: Iterable[str]) -> tuple[dict[str, int], dict[str, str]]:
    """Return deterministic label-to-ID and JSON-safe ID-to-label mappings."""
    normalized = [str(label).strip() for label in labels]
    if not normalized or any(not label for label in normalized):
        raise ValueError("Intent labels must be non-empty strings")

    unique_labels = sorted(set(normalized))
    label_to_id = {label: index for index, label in enumerate(unique_labels)}
    id_to_label = {str(index): label for label, index in label_to_id.items()}
    return label_to_id, id_to_label


def validate_intent_training_data(
    queries: Iterable[str], intents: Iterable[str], validation_split: float
) -> tuple[list[str], list[str]]:
    """Validate aligned data and ensure a stratified split can retain every class."""
    query_list = list(queries)
    intent_list = [str(intent).strip() for intent in intents]

    if not query_list or not intent_list:
        raise ValueError("No training data provided")
    if len(query_list) != len(intent_list):
        raise ValueError("queries and intents must have the same number of items")
    if any(not isinstance(query, str) or not query.strip() for query in query_list):
        raise ValueError("Queries must be non-empty strings")
    if any(not intent for intent in intent_list):
        raise ValueError("Intent labels must be non-empty strings")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")

    counts = Counter(intent_list)
    if len(counts) < 2:
        raise ValueError("At least two intent classes are required")
    if min(counts.values()) < 2:
        raise ValueError("Each intent needs at least two samples for a stratified split")

    validation_size = ceil(len(query_list) * validation_split)
    train_size = len(query_list) - validation_size
    if validation_size < len(counts) or train_size < len(counts):
        raise ValueError("The split must leave at least one sample per intent in both sets")

    return query_list, intent_list


def resolve_intent_label(class_id: object, id_to_label: Mapping[str, str]) -> str:
    """Resolve model class IDs without depending on dictionary insertion order."""
    try:
        key = str(int(class_id))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid classifier class ID: {class_id!r}") from exc

    try:
        return id_to_label[key]
    except KeyError as exc:
        raise ValueError(f"No intent label is configured for class ID {key}") from exc
