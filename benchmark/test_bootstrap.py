"""Regression checks for order-independent confirmatory bootstrap sampling."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap_ci import bootstrap_class, prediction_rows, quantile  # noqa: E402


class BootstrapTest(unittest.TestCase):
    def test_variant_and_class_seeds_are_stable_and_independent(self) -> None:
        values = [1.0] * 75 + [0.0] * 25
        original_real = bootstrap_class(values, 20_260_813, "original", 0, 500)
        self.assertEqual(original_real, bootstrap_class(values, 20_260_813, "original", 0, 500))
        self.assertNotEqual(original_real, bootstrap_class(values, 20_260_813, "screenshot", 0, 500))
        self.assertNotEqual(original_real, bootstrap_class(values, 20_260_813, "original", 1, 500))
        self.assertLessEqual(quantile(original_real, 0.025), 0.75)
        self.assertGreaterEqual(quantile(original_real, 0.975), 0.75)

    def test_impossible_logit_probability_pair_cannot_enter_bootstrap(self) -> None:
        manifest = {
            "fixture": {"id": "fixture", "label": 1, "source": "generator", "groupId": "prompt"},
        }
        row = {
            **manifest["fixture"],
            "variant": "original",
            "logit": 999.0,
            "rawProbability": 0.99,
        }
        with tempfile.TemporaryDirectory() as temporary:
            predictions = Path(temporary) / "predictions.jsonl"
            predictions.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "stale or malformed"):
                prediction_rows(predictions, manifest)


if __name__ == "__main__":
    unittest.main()
