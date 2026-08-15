from __future__ import annotations

import ast
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "benchmark/modern/train_rehead.py"


def selected_functions(*names: str) -> dict[str, object]:
    module = ast.parse(TRAINER.read_text())
    selected = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, object] = {
        "np": np,
        "json": json,
        "math": math,
        "os": os,
        "Path": Path,
        "sha256": sha256,
        "Item": object,
        "REPOSITORY_ROOT": ROOT,
        "VARIANTS": ("original", "screenshot", "social-q75", "social-heavy"),
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRAINER), "exec"), namespace)
    return namespace


FUNCTIONS = selected_functions(
    "digest",
    "lexical_absolute",
    "require_repository_path",
    "require_partition_contract",
    "variant_metrics",
    "passes_validation_gates",
    "evaluate_frozen_regression",
)
evaluate_frozen_regression = FUNCTIONS["evaluate_frozen_regression"]
require_partition_contract = FUNCTIONS["require_partition_contract"]


class M3TrainerContractTests(unittest.TestCase):
    def test_cli_exposes_explicit_regression_packet(self) -> None:
        module = ast.parse(TRAINER.read_text())
        constants = {
            node.value
            for node in ast.walk(module)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertIn("--regression-data-root", constants)
        self.assertIn("--regression-manifest", constants)

    def test_recipe_binds_selector_and_regression_as_distinct_roles(self) -> None:
        recipe = json.loads((ROOT / "benchmark/m3/recipe.json").read_text())
        self.assertEqual(len(recipe["evaluationManifests"]), 1)
        self.assertEqual(len(recipe["regressionManifests"]), 1)
        self.assertNotEqual(
            recipe["evaluationManifests"][0]["path"],
            recipe["regressionManifests"][0]["path"],
        )
        self.assertEqual(recipe["regressionManifests"][0]["role"], "consumed-m2-post-selection-regression")

    def test_partition_contract_rejects_wrong_regression_path_before_training(self) -> None:
        recipe = json.loads((ROOT / "benchmark/m3/recipe.json").read_text())
        item = SimpleNamespace(id="one", path=Path("one.jpg"), image_sha256="0" * 64, label=0, source="open-images")
        with self.assertRaisesRegex(ValueError, "regression manifest path"):
            require_partition_contract(
                recipe,
                selector_manifest=ROOT / recipe["evaluationManifests"][0]["path"],
                selector_data_root=ROOT / recipe["evaluationManifests"][0]["dataRoot"],
                selector_items=[item] * int(recipe["expectedValidationCount"]),
                regression_manifest=ROOT / "benchmark/evidence/m2/not-validation.jsonl",
                regression_data_root=ROOT / recipe["regressionValidation"]["dataRoot"],
                regression_items=[item] * int(recipe["expectedRegressionCount"]),
            )

    def test_frozen_regression_gate_is_terminal_and_uses_selector_threshold(self) -> None:
        features = np.asarray([[1.0], [0.0]] * 4, dtype=np.float32)
        labels = np.asarray([1, 0] * 4, dtype=np.float32)
        variants = np.asarray([index for index in range(4) for _ in range(2)], dtype=np.int64)
        sources = np.asarray(["synthetic-a", "real-a"] * 4)
        gates = {
            "minimumBalancedAccuracyPerVariant": 1.0,
            "minimumRealRecallPerVariant": 1.0,
            "minimumSyntheticRecallPerVariant": 1.0,
            "minimumSyntheticRecallPerFamily": 1.0,
            "minimumRealRecallBySource": {"real-a": 1.0},
        }
        values = evaluate_frozen_regression(
            np.asarray([10.0], dtype=np.float32),
            -5.0,
            0.0,
            (features, labels, variants, sources),
            gates,
        )
        self.assertEqual(values["original"]["balancedAccuracy"], 1.0)
        with self.assertRaisesRegex(RuntimeError, "post-selection regression gates"):
            evaluate_frozen_regression(
                np.asarray([10.0], dtype=np.float32),
                -5.0,
                6.0,
                (features, labels, variants, sources),
                gates,
            )

    def test_candidate_grid_call_has_no_regression_input(self) -> None:
        module = ast.parse(TRAINER.read_text())
        main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "fit_candidate"
        ]
        self.assertEqual(len(calls), 1)
        rendered = ast.unparse(calls[0])
        self.assertNotIn("regression", rendered)
        self.assertIn("validation", rendered)

    def test_regression_cache_and_evidence_partition_are_named(self) -> None:
        source = TRAINER.read_text()
        self.assertIn('"regressionManifestSha256"', source)
        self.assertIn('"selectionInfluenced": False', source)
        self.assertIn('feature_configuration_hashes["regression"] = feature_configuration_hash(', source)


if __name__ == "__main__":
    unittest.main()
