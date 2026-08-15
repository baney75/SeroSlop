from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m3.publication_contract import (  # noqa: E402
    PUBLICATION_ROWS,
    TRANSACTIONAL_ROWS,
    UPSTREAM_MODEL_SHA256,
    build_publication_lock,
    expected_cache_paths,
    expected_training_arguments,
    parse_canonical_publication_lock,
    validate_model_comparison,
    validate_publication_lock,
    require_variant_gates,
)


class M3PublicationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = {
            "candidateHashes": {
                "training-summary.json": "1" * 64,
                "calibration.json": "2" * 64,
                "candidate-grid.json": "3" * 64,
                "model.onnx": "4" * 64,
            },
            "modelBytes": 87_442_080,
            "freshRunId": "fresh-run-m3",
        }
        self.source_commit = "a" * 40
        self.source_tree = "b" * 40
        self.candidate_evidence_json = {
            "training-summary.json": "{}\n",
            "calibration.json": "{}\n",
            "candidate-grid.json": "[]\n",
            "model-comparison.json": "{}\n",
            "fixture-manifest.json": "{}\n",
        }
        self.classifier_patch = {
            "schemaVersion": 1,
            "baseSha256": UPSTREAM_MODEL_SHA256,
            "candidateSha256": "4" * 64,
            "candidateBytes": 87_442_080,
            "replacements": [],
        }

    def test_frozen_command_and_cache_surface(self) -> None:
        arguments = expected_training_arguments()
        self.assertEqual(arguments[-2:], ["--output-dir", "benchmark/candidates/prooflens-cf384-m3"])
        self.assertIn("--regression-manifest", arguments)
        self.assertIn("--reextract-cached-features", arguments)
        paths = expected_cache_paths()
        self.assertEqual(len(paths), 57)
        self.assertEqual(paths[0], "benchmark/candidates/prooflens-cf384-m3/features/train-00000.npz")
        self.assertEqual(paths[-2:], [
            "benchmark/candidates/prooflens-cf384-m3/features/validation-00000.npz",
            "benchmark/candidates/prooflens-cf384-m3/features/regression-00000.npz",
        ])

    def test_publication_lock_is_exact_and_mutation_sensitive(self) -> None:
        lock = build_publication_lock(
            packet=self.packet,
            comparison_sha256="5" * 64,
            source_commit=self.source_commit,
            source_tree=self.source_tree,
            public_document_hashes={"README.md": "a" * 64, "MODEL_CARD.md": "b" * 64, "BENCHMARK.md": "c" * 64},
            fixture_manifest_sha256="d" * 64,
            candidate_evidence_json=self.candidate_evidence_json,
            classifier_patch=self.classifier_patch,
        )
        self.assertEqual(lock["candidateModelBytes"], 87_442_080)
        self.assertEqual(lock["publicationRows"], [
            {"path": path, "status": status} for path, status in PUBLICATION_ROWS
        ])
        self.assertEqual(len(TRANSACTIONAL_ROWS), 12)
        validate_publication_lock(
            lock,
            packet=self.packet,
            comparison_sha256="5" * 64,
            source_commit=self.source_commit,
            source_tree=self.source_tree,
            public_document_hashes={"README.md": "a" * 64, "MODEL_CARD.md": "b" * 64, "BENCHMARK.md": "c" * 64},
            fixture_manifest_sha256="d" * 64,
            candidate_evidence_json=self.candidate_evidence_json,
            classifier_patch=self.classifier_patch,
        )
        for field in (
            "sourceCommit",
            "sourceTree",
            "trainerSha256",
            "recipeSha256",
            "selectionSummarySha256",
            "modelComparisonSha256",
            "freshRunId",
            "finalizerSha256",
            "publicationContractSha256",
            "fixtureSelectorSha256",
            "documentationRendererSha256",
            "publicDocumentHashes",
            "fixtureManifestSha256",
            "candidateEvidenceJson",
            "classifierPatch",
            "candidateModelBytes",
        ):
            mutated = deepcopy(lock)
            mutated[field] = "changed"
            with self.assertRaisesRegex(ValueError, "publication lock"):
                validate_publication_lock(
                    mutated,
                    packet=self.packet,
                    comparison_sha256="5" * 64,
                    source_commit=self.source_commit,
                    source_tree=self.source_tree,
                    public_document_hashes={"README.md": "a" * 64, "MODEL_CARD.md": "b" * 64, "BENCHMARK.md": "c" * 64},
                    fixture_manifest_sha256="d" * 64,
                    candidate_evidence_json=self.candidate_evidence_json,
                    classifier_patch=self.classifier_patch,
                )

    def test_classifier_only_comparison_rejects_schema_confusion(self) -> None:
        comparison = {
            "schemaVersion": 1,
            "base": {"path": "weights/prooflens-cf384.onnx", "sha256": UPSTREAM_MODEL_SHA256},
            "candidate": {
                "path": "benchmark/candidates/prooflens-cf384-m3/model.onnx",
                "sha256": "4" * 64,
                "bytes": 87_442_080,
            },
            "changedInitializers": [
                {
                    "name": "classifier.weight",
                    "dimensions": [1, 384],
                    "beforeSha256": "6" * 64,
                    "afterSha256": "7" * 64,
                },
                {
                    "name": "classifier.bias",
                    "dimensions": [1],
                    "beforeSha256": "8" * 64,
                    "afterSha256": "9" * 64,
                },
            ],
            "unchangedInitializerCount": 198,
            "graphNodesSha256": "a" * 64,
            "graphInputsSha256": "b" * 64,
            "graphOutputsSha256": "c" * 64,
            "opsetsSha256": "d" * 64,
        }
        validate_model_comparison(
            comparison,
            candidate_sha256="4" * 64,
            candidate_bytes=87_442_080,
        )
        for field in ("base", "candidate"):
            malformed = deepcopy(comparison)
            malformed[field] = []
            with self.assertRaisesRegex(ValueError, "comparison"):
                validate_model_comparison(
                    malformed,
                    candidate_sha256="4" * 64,
                    candidate_bytes=87_442_080,
                )
        malformed = deepcopy(comparison)
        malformed["changedInitializers"] = ["classifier.weight", "classifier.bias"]
        with self.assertRaisesRegex(ValueError, "comparison"):
            validate_model_comparison(
                malformed,
                candidate_sha256="4" * 64,
                candidate_bytes=87_442_080,
            )

    def test_publication_lock_json_is_canonical_and_duplicate_free(self) -> None:
        value = {"schemaVersion": 1, "h3HoldoutScored": False}
        canonical = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        self.assertEqual(parse_canonical_publication_lock(canonical), value)
        duplicate = canonical.replace(
            '  "h3HoldoutScored": false',
            '  "h3HoldoutScored": true,\n  "h3HoldoutScored": false',
        )
        with self.assertRaisesRegex(ValueError, "canonical JSON"):
            parse_canonical_publication_lock(duplicate)

    def test_regression_family_set_is_exact(self) -> None:
        gates = {
            "minimumBalancedAccuracyPerVariant": 0.5,
            "minimumRealRecallPerVariant": 0.5,
            "minimumSyntheticRecallPerVariant": 0.5,
            "minimumSyntheticRecallPerFamily": 0.5,
            "minimumRealRecallBySource": {"open-images": 0.5, "stockimages-cc0": 0.5},
        }
        row = {
            "balancedAccuracy": 1,
            "realRecall": 1,
            "syntheticRecall": 1,
            "syntheticRecallBySource": {"GLM-Image": 1, "HunyuanImage-3.0": 1},
            "realRecallBySource": {"open-images": 1, "stockimages-cc0": 1},
        }
        variants = {name: dict(row) for name in ("original", "screenshot", "social-q75", "social-heavy")}
        require_variant_gates(
            variants,
            gates,
            label="regression",
            expected_synthetic_sources={"GLM-Image", "HunyuanImage-3.0"},
            expected_real_sources={"open-images", "stockimages-cc0"},
        )
        variants["original"] = {**row, "syntheticRecallBySource": {"wrong-family": 1}}
        with self.assertRaisesRegex(ValueError, "synthetic-family"):
            require_variant_gates(
                variants,
                gates,
                label="regression",
                expected_synthetic_sources={"GLM-Image", "HunyuanImage-3.0"},
                expected_real_sources={"open-images", "stockimages-cc0"},
            )


if __name__ == "__main__":
    unittest.main()
