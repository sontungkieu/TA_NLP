from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def load_solution_module():
    module_path = Path(__file__).resolve().parents[1] / "solution" / "pos_tagger.py"
    spec = importlib.util.spec_from_file_location("practice_2_1_solution", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOLUTION = load_solution_module()


class PosTaggerTests(unittest.TestCase):
    def test_word_shape_maps_character_classes(self) -> None:
        self.assertEqual(SOLUTION.word_shape("McFly-2026"), "XxXxx-dddd")

    def test_extract_token_features_uses_context(self) -> None:
        sentence = ["Time", "flies", "fast"]
        features = SOLUTION.extract_token_features(sentence, 1, prev_tag="NOUN")

        self.assertEqual(features["word.lower"], "flies")
        self.assertEqual(features["prev.word.lower"], "time")
        self.assertEqual(features["next.word.lower"], "fast")
        self.assertEqual(features["prev_tag"], "NOUN")

    def test_build_examples_matches_number_of_tokens(self) -> None:
        tagged_sentences = [
            [("Time", "NOUN"), ("flies", "VERB")],
            [("Fruit", "NOUN"), ("flies", "NOUN"), ("daily", "ADV")],
        ]

        feature_rows, labels = SOLUTION.build_examples(tagged_sentences)
        self.assertEqual(len(feature_rows), 5)
        self.assertEqual(len(labels), 5)

    def test_compute_metrics_matches_manual_example(self) -> None:
        gold_tags = ["NOUN", "VERB", "NOUN"]
        pred_tags = ["NOUN", "NOUN", "NOUN"]

        metrics = SOLUTION.compute_metrics(gold_tags, pred_tags)

        self.assertAlmostEqual(metrics["per_tag"]["NOUN"]["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["per_tag"]["NOUN"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["per_tag"]["VERB"]["recall"], 0.0)
        self.assertAlmostEqual(metrics["macro_f1"], (0.8 + 0.0) / 2)

    def test_confusion_matrix_counts_gold_predicted_pairs(self) -> None:
        gold_tags = ["NOUN", "VERB", "NOUN", "ADV"]
        pred_tags = ["NOUN", "NOUN", "ADV", "ADV"]
        labels = ["ADV", "NOUN", "VERB"]

        confusion = SOLUTION.build_confusion_matrix(gold_tags, pred_tags, labels)

        self.assertEqual(confusion, [[1, 0, 0], [1, 1, 0], [0, 1, 0]])

    def test_train_and_evaluate_split_returns_consistent_shapes(self) -> None:
        train_sentences = [
            [("Time", "NOUN"), ("flies", "VERB"), ("fast", "ADV")],
            [("Fruit", "NOUN"), ("flies", "NOUN"), ("daily", "ADV")],
            [("Birds", "NOUN"), ("sing", "VERB"), ("loudly", "ADV")],
            [("The", "DET"), ("planes", "NOUN"), ("land", "VERB")],
        ]
        test_sentences = [
            [("Time", "NOUN"), ("sing", "VERB"), ("daily", "ADV")],
            [("Fruit", "NOUN"), ("flies", "NOUN"), ("fast", "ADV")],
        ]

        results = SOLUTION.train_and_evaluate_split(train_sentences, test_sentences, compare_baseline=True)

        self.assertEqual(len(results["predicted_sentences"]), len(test_sentences))
        self.assertEqual(len(results["labels"]), len(results["confusion_matrix"]))
        self.assertIn("baseline", results)
        self.assertGreaterEqual(results["metrics"]["macro_f1"], 0.0)
        self.assertLessEqual(results["metrics"]["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
