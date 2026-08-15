from __future__ import annotations

import gzip
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m3.contracts import (  # noqa: E402
    canonical_source_url,
    deterministic_gzip,
    exact_partition,
    priority,
    assert_unique_evidence_rows,
    assert_pixel_facts,
    source_group,
)


class M3PreparationContractTests(unittest.TestCase):
    def test_frozen_counts_encode_fresh_selector_and_regression_boundary(self) -> None:
        import json

        recipe = json.loads((ROOT / "benchmark/m3/recipe.json").read_text())
        self.assertEqual(recipe["metSource"]["holdoutTarget"], 600)
        self.assertEqual(recipe["metSource"]["developmentTarget"], 300)
        self.assertEqual(recipe["metSource"]["trainingTarget"], 2400)
        self.assertEqual(recipe["syntheticDevelopmentSource"]["target"], 300)
        self.assertEqual(recipe["expectedTotalCount"], 108378)
        self.assertEqual(recipe["expectedTrainingFeatureViews"], 133512)
        self.assertEqual(recipe["expectedValidationCount"], 600)
        self.assertEqual(recipe["expectedValidationFeatureViews"], 2400)
        self.assertEqual(recipe["expectedRegressionCount"], 900)
        self.assertEqual(recipe["regressionManifests"][0]["role"], "consumed-m2-post-selection-regression")
        self.assertEqual(recipe["validationGates"]["minimumRealRecallBySource"], {"met-open-access": 0.93})
        self.assertEqual(recipe["baseTraining"]["trainingEvaluationPerceptualReview"]["items"], 100)
        self.assertIn(recipe["metSource"]["commit"], recipe["metSource"]["csvUrl"])
        self.assertIn(
            recipe["metSource"]["huggingFaceMetadata"]["revision"],
            recipe["metSource"]["huggingFaceMetadata"]["downloadUrl"],
        )

    def test_holdout_development_and_training_namespaces_are_distinct(self) -> None:
        import json

        recipe = json.loads((ROOT / "benchmark/m3/recipe.json").read_text())
        source = recipe["metSource"]
        identifier = "met-open-access:e901de:123"
        values = {
            priority(source["holdoutPriorityNamespace"], identifier),
            priority(source["developmentPriorityNamespace"], identifier),
            priority(source["trainingPriorityNamespace"], identifier),
        }
        self.assertEqual(len(values), 3)

    def test_partition_selection_is_exact_score_blind_and_disjoint(self) -> None:
        candidates = [{"id": f"item-{index}"} for index in range(12)]
        excluded = {"item-0"}
        holdout = exact_partition(candidates, namespace="holdout:", target=4, excluded_ids=excluded)
        development = exact_partition(candidates, namespace="development:", target=3, excluded_ids=excluded)
        training = exact_partition(candidates, namespace="training:", target=4, excluded_ids=excluded)
        ids = [str(row["id"]) for rows in (holdout, development, training) for row in rows]
        self.assertEqual(len(ids), 11)
        self.assertEqual(len(set(ids)), 11)
        self.assertNotIn("item-0", ids)
        self.assertTrue(all("selectionPriority" in row for row in holdout + development + training))

    def test_met_image_url_and_group_are_canonical(self) -> None:
        value = "https://images.metmuseum.org/CRDImages/dp/web-large/example.jpg"
        self.assertEqual(canonical_source_url(value, allowed_host="images.metmuseum.org"), value)
        self.assertEqual(source_group("met:", value), source_group("met:", value))
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            canonical_source_url("https://example.com/image.jpg", allowed_host="images.metmuseum.org")
        with self.assertRaisesRegex(ValueError, "not canonical"):
            canonical_source_url(value + "?mutable=1", allowed_host="images.metmuseum.org")

    def test_deterministic_gzip_has_canonical_header(self) -> None:
        raw = b"prooflens-m3\n" * 20
        first = deterministic_gzip(raw)
        self.assertEqual(first, deterministic_gzip(raw))
        self.assertEqual(first[9], 0xFF)
        self.assertEqual(gzip.decompress(first), raw)

    def test_source_failure_evidence_does_not_embed_environment_specific_errors(self) -> None:
        source = (ROOT / "benchmark/m3/prepare.py").read_text()
        self.assertNotIn('"detail": str(result)', source)
        self.assertEqual(source.count('"failureCategory": "source-unavailable-or-invalid"'), 2)

    def test_historical_row_dhash_may_come_from_locked_perceptual_evidence(self) -> None:
        row = {
            "id": "historical-validation:1",
            "imageSha256": "a" * 64,
            "rowIndex": 0,
        }
        assert_unique_evidence_rows(
            [row],
            label="historical validation",
            fallback_dhashes={row["id"]: "0123456789abcdef"},
        )
        with self.assertRaisesRegex(ValueError, "invalid dHash"):
            assert_unique_evidence_rows([row], label="historical validation")
        for invalid in ("", "not-a-hash"):
            malformed = {**row, "perceptualDhash64": invalid}
            with self.assertRaisesRegex(ValueError, "invalid dHash"):
                assert_unique_evidence_rows(
                    [malformed],
                    label="historical validation",
                    fallback_dhashes={row["id"]: "0123456789abcdef"},
                )

    def test_legacy_dimensions_may_be_absent_but_never_partial_or_wrong(self) -> None:
        row = {"perceptualDhash64": "0123456789abcdef"}
        assert_pixel_facts(
            row,
            width=640,
            height=480,
            perceptual_dhash64="0123456789abcdef",
            label="legacy row",
        )
        with self.assertRaisesRegex(ValueError, "incomplete dimension"):
            assert_pixel_facts(
                {**row, "width": 640},
                width=640,
                height=480,
                perceptual_dhash64="0123456789abcdef",
                label="legacy row",
            )
        with self.assertRaisesRegex(ValueError, "dimensions changed"):
            assert_pixel_facts(
                {**row, "width": 1, "height": 480},
                width=640,
                height=480,
                perceptual_dhash64="0123456789abcdef",
                label="legacy row",
            )


if __name__ == "__main__":
    unittest.main()
