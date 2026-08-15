from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "benchmark/modern/train_rehead.py"


def selected_functions(*names: str) -> dict[str, object]:
    module = ast.parse(TRAINER.read_text())
    selected = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    namespace: dict[str, object] = {
        "np": np,
        "json": json,
        "Path": Path,
        "sha256": sha256,
        "Item": object,
        "ThreadPoolExecutor": ThreadPoolExecutor,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRAINER), "exec"), namespace)
    return namespace


@contextmanager
def changed_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class M2TrainerContractTests(unittest.TestCase):
    def test_fresh_extraction_verifies_manifest_bytes_first(self) -> None:
        module = ast.parse(TRAINER.read_text())
        function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "extract_or_load_sharded"
        )
        fresh_branch = next(
            node for node in ast.walk(function)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "loaded is None"
        )
        self.assertGreaterEqual(len(fresh_branch.body), 2)
        self.assertEqual(ast.unparse(fresh_branch.body[0]), "verify_item_files(shard_items)")
        self.assertIn("extract_features", ast.unparse(fresh_branch.body[1]))

        functions = selected_functions("digest", "verify_item_files")
        verify_item_files = functions["verify_item_files"]
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "pixel.bin"
            path.write_bytes(b"wrong")
            item = SimpleNamespace(id="fixture", path=path, image_sha256=sha256(b"expected").hexdigest())
            with self.assertRaisesRegex(ValueError, "Image integrity mismatch"):
                verify_item_files([item])

    def test_source_balanced_loss_assigns_equal_real_source_mass(self) -> None:
        function = selected_functions("source_balanced_weights")["source_balanced_weights"]
        labels = np.asarray([0, 0, 0, 1, 1], dtype=np.float32)
        sources = np.asarray(["open", "docci", "stock", "old-synth", "new-synth"])
        weights = function(labels, sources)
        np.testing.assert_allclose(weights[:3], np.asarray([1 / 6] * 3), atol=1e-7)
        np.testing.assert_allclose(weights[3:], np.asarray([1 / 4] * 2), atol=1e-7)

    def test_stock_gate_is_inclusive_at_point_ninety_three_for_every_view(self) -> None:
        function = selected_functions("passes_validation_gates")["passes_validation_gates"]
        gates = {
            "minimumBalancedAccuracyPerVariant": 0.85,
            "minimumRealRecallPerVariant": 0.85,
            "minimumSyntheticRecallPerVariant": 0.75,
            "minimumSyntheticRecallPerFamily": 0.6,
            "minimumRealRecallBySource": {"stockimages-cc0": 0.93},
        }
        row = {
            "balancedAccuracy": 0.95,
            "realRecall": 0.95,
            "syntheticRecall": 0.95,
            "syntheticRecallBySource": {"family": 0.9},
            "realRecallBySource": {"stockimages-cc0": 0.93},
        }
        values = {variant: dict(row) for variant in ("original", "screenshot", "social-q75", "social-heavy")}
        self.assertTrue(function(values, gates))
        values["social-q75"] = {**row, "realRecallBySource": {"stockimages-cc0": 0.929999}}
        self.assertFalse(function(values, gates))

    def test_generic_schema_two_and_legacy_schema_one_packets_both_validate(self) -> None:
        function = selected_functions("digest", "validate_large_training_packet")["validate_large_training_packet"]
        with tempfile.TemporaryDirectory() as name, changed_directory(Path(name)):
            root = Path.cwd()
            train = root / "train.jsonl"
            train.write_text("{}\n{}\n")
            evaluation = root / "evaluation.jsonl"
            evaluation.write_text(json.dumps({"id": "eval", "imageSha256": "e" * 64}) + "\n")
            review = root / "review.json"
            review.write_text(json.dumps({"items": []}))
            training_review = root / "training-review.json"
            training_review.write_text(json.dumps({"items": []}))
            items = [
                SimpleNamespace(id="real", image_sha256="a" * 64, label=0, source="real-source"),
                SimpleNamespace(id="fake", image_sha256="b" * 64, label=1, source="fake-source"),
            ]
            recipe = {
                "expectedTotalCount": 2,
                "expectedSourceCounts": {"fake-source": 1, "real-source": 1},
                "expectedClassCounts": {"real": 1, "synthetic": 1},
                "evaluationManifests": [
                    {"path": str(evaluation), "dataRoot": "data", "role": "development-validation"}
                ],
                "additionalTrainingExclusionManifests": [],
                "perceptualOverlapReview": str(review),
                "trainingPerceptualOverlapReview": str(training_review),
                "perceptualDuplicateHammingThreshold": 8,
            }
            recipe_path = root / "recipe.json"
            recipe_path.write_text(json.dumps(recipe))
            summary = {
                "schemaVersion": 2,
                "recipeSha256": sha256(recipe_path.read_bytes()).hexdigest(),
                "manifestSha256": sha256(train.read_bytes()).hexdigest(),
                "counts": {"total": 2},
                "sourceCounts": {"fake-source": 1, "real-source": 1},
                "classCounts": {"real": 1, "synthetic": 1},
                "evaluationExclusions": [
                    {
                        "path": str(evaluation),
                        "sha256": sha256(evaluation.read_bytes()).hexdigest(),
                        "rows": 1,
                        "dataRoot": "data",
                        "role": "development-validation",
                    }
                ],
                "perceptualOverlapReview": {
                    "path": str(review), "sha256": sha256(review.read_bytes()).hexdigest(),
                    "reviewedPairCount": 0, "hammingThreshold": 8,
                },
                "trainingPerceptualOverlapReview": {
                    "path": str(training_review), "sha256": sha256(training_review.read_bytes()).hexdigest(),
                    "reviewedPairCount": 0, "hammingThreshold": 8,
                },
                "overlapWithEvaluation": {
                    "ids": 0, "imageHashes": 0,
                    "unreviewedPerceptualDhashPairsAtOrBelowThreshold": 0,
                    "reviewedVisuallyDistinctDhashPairsAtOrBelowThreshold": 0,
                },
            }
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary))
            function(recipe_path, recipe, summary_path, train, items)

            legacy_recipe = {
                **{key: value for key, value in recipe.items() if key not in {"expectedSourceCounts", "expectedClassCounts"}},
                "diffusionDb": {"targetCount": 1},
                "openImages": {"targetCount": 1},
                "expectedModernTrainingCount": 0,
            }
            legacy_items = [
                SimpleNamespace(id="real", image_sha256="a" * 64, label=0, source="open-images-train"),
                SimpleNamespace(id="fake", image_sha256="b" * 64, label=1, source="diffusiondb-stable-diffusion"),
            ]
            legacy_recipe_path = root / "legacy-recipe.json"
            legacy_recipe_path.write_text(json.dumps(legacy_recipe))
            legacy_summary = {
                **summary,
                "schemaVersion": 1,
                "recipeSha256": sha256(legacy_recipe_path.read_bytes()).hexdigest(),
                "sourceCounts": {"diffusiondb-stable-diffusion": 1, "open-images-train": 1},
            }
            legacy_summary_path = root / "legacy-summary.json"
            legacy_summary_path.write_text(json.dumps(legacy_summary))
            function(legacy_recipe_path, legacy_recipe, legacy_summary_path, train, legacy_items)


if __name__ == "__main__":
    unittest.main()
