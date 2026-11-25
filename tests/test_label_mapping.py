import unittest

from ml_core.label_mapping import (
    build_label_mappings,
    resolve_intent_label,
    validate_intent_training_data,
)


class LabelMappingTests(unittest.TestCase):
    def test_mapping_is_deterministic_and_json_safe(self):
        label_to_id, id_to_label = build_label_mappings(["zeta", "alpha", "zeta"])

        self.assertEqual(label_to_id, {"alpha": 0, "zeta": 1})
        self.assertEqual(id_to_label, {"0": "alpha", "1": "zeta"})
        self.assertEqual(resolve_intent_label(1, id_to_label), "zeta")

    def test_unknown_class_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "No intent label"):
            resolve_intent_label(4, {"0": "alpha"})

    def test_empty_or_misaligned_training_data_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            validate_intent_training_data(["a", "b"], ["x"], 0.2)

    def test_singleton_classes_cannot_be_stratified(self):
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            validate_intent_training_data(
                ["a", "b", "c"], ["alpha", "alpha", "beta"], 0.34
            )

    def test_validation_split_must_fit_all_classes(self):
        with self.assertRaisesRegex(ValueError, "at least one sample per intent"):
            validate_intent_training_data(
                ["a", "b", "c", "d"], ["alpha", "alpha", "beta", "beta"], 0.25
            )

    def test_valid_data_is_returned_with_trimmed_labels(self):
        queries, intents = validate_intent_training_data(
            ["one", "two", "three", "four", "five", "six"],
            [" alpha ", "alpha", "alpha", "beta", "beta", "beta"],
            0.34,
        )

        self.assertEqual(len(queries), len(intents))
        self.assertEqual(intents[0], "alpha")


if __name__ == "__main__":
    unittest.main()
