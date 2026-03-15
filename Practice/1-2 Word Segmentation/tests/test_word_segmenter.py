from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SOLUTION_PATH = REPO_ROOT / "Practice/1-2 Word Segmentation/solution/word_segmenter.py"


def load_solution_module():
    spec = importlib.util.spec_from_file_location("solution_word_segmenter", SOLUTION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load solution module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WordSegmenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_solution_module()

    def test_load_dictionary_normalizes_unicode_and_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dict_path = Path(tmp_dir) / "dict.txt"
            dict_path.write_text("a\u0300\nà\nthành phố\n", encoding="utf-8")

            lexicon, max_word_len = self.module.load_dictionary(dict_path)

        self.assertEqual(lexicon, {"à", "thành phố"})
        self.assertEqual(max_word_len, 2)

    def test_segment_text_prefers_longest_match(self) -> None:
        lexicon = {"công an", "thành phố", "nghi phạm", "huế"}
        segmented = self.module.segment_text("Công an thành phố Huế bắt nghi phạm", lexicon, 2)

        self.assertEqual(segmented, ["Công_an", "thành_phố", "Huế", "bắt", "nghi_phạm"])

    def test_segment_text_handles_attached_punctuation(self) -> None:
        lexicon = {"bộ y tế", "điều trị", "đậu mùa khỉ", "thế nào"}
        segmented = self.module.segment_text(
            "Bộ Y tế điều trị bệnh đậu mùa khỉ thế nào?",
            lexicon,
            3,
        )

        self.assertEqual(segmented[-2:], ["thế_nào", "?"])
        self.assertIn("Bộ_Y_tế", segmented)
        self.assertIn("đậu_mùa_khỉ", segmented)

    def test_evaluate_uses_word_boundaries(self) -> None:
        precision, recall, f1 = self.module.evaluate(
            ["học_sinh Việt Nam"],
            ["học_sinh Việt_Nam"],
        )

        self.assertAlmostEqual(precision, 1 / 3)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.4)

    def test_cli_segment_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dict_path = tmp_path / "dict.txt"
            input_path = tmp_path / "input.txt"
            output_path = tmp_path / "pred.txt"

            dict_path.write_text("thành phố\ncông an\n", encoding="utf-8")
            input_path.write_text("Công an thành phố Huế\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    "python3",
                    str(SOLUTION_PATH),
                    "--mode",
                    "segment",
                    "--dict",
                    str(dict_path),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

            self.assertEqual(completed.stdout, "")
            self.assertEqual(output_path.read_text(encoding="utf-8").strip(), "Công_an thành_phố Huế")

    def test_cli_evaluate_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dict_path = tmp_path / "dict.txt"
            input_path = tmp_path / "input.txt"
            gold_path = tmp_path / "gold.txt"
            pred_path = tmp_path / "pred.txt"

            dict_path.write_text("công an\nthành phố\n", encoding="utf-8")
            input_path.write_text("Công an thành phố Huế\n", encoding="utf-8")
            gold_path.write_text("Công_an thành_phố Huế\n", encoding="utf-8")

            completed = subprocess.run(
                [
                    "python3",
                    str(SOLUTION_PATH),
                    "--mode",
                    "evaluate",
                    "--dict",
                    str(dict_path),
                    "--input",
                    str(input_path),
                    "--gold",
                    str(gold_path),
                    "--output",
                    str(pred_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            )

            self.assertIn("Precision: 1.0000", completed.stdout)
            self.assertIn("Recall: 1.0000", completed.stdout)
            self.assertIn("F1: 1.0000", completed.stdout)
            self.assertEqual(pred_path.read_text(encoding="utf-8").strip(), "Công_an thành_phố Huế")


if __name__ == "__main__":
    unittest.main()
