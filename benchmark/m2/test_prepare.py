from __future__ import annotations

import ast
import gzip
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m2.contracts import deterministic_gzip, priority  # noqa: E402


PREPARE = ROOT / "benchmark/m2/prepare.py"
VERIFY = ROOT / "benchmark/m2/verify.py"
RECOVERY_PREPARE = ROOT / "benchmark/recovery_v3/prepare.py"


def selected_recovery_functions(*names: str) -> dict[str, object]:
    module = ast.parse(RECOVERY_PREPARE.read_text())
    selected = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace: dict[str, object] = {
        "ROOT": ROOT,
        "Path": Path,
        "md5": md5,
        "sha256": sha256,
        "os": os,
        "time": time,
        "urllib": sys.modules["urllib"],
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(RECOVERY_PREPARE), "exec"), namespace)
    return namespace


class M2PreparationContractTests(unittest.TestCase):
    def test_recipe_counts_and_partitions_are_frozen(self) -> None:
        recipe = json.loads((ROOT / "benchmark/m2/recipe.json").read_text())
        stock = recipe["stockImagesSource"]
        self.assertEqual(stock["candidateRows"], 3999)
        self.assertEqual(stock["expectedEligibleRows"], 2699)
        self.assertEqual(stock["developmentTarget"], 300)
        self.assertEqual(stock["trainingTarget"], 2378)
        self.assertEqual(stock["expectedCrossCandidateRejects"], 21)
        self.assertEqual(300 + 2378 + 21, 2699)
        self.assertEqual(recipe["expectedTotalCount"], 105978)
        self.assertEqual(recipe["expectedTrainingFeatureViews"], 123912)
        self.assertEqual(recipe["validationGates"]["minimumRealRecallBySource"], {"stockimages-cc0": 0.93})

    def test_selection_namespaces_are_distinct_and_deterministic(self) -> None:
        recipe = json.loads((ROOT / "benchmark/m2/recipe.json").read_text())
        stock = recipe["stockImagesSource"]
        identifier = "stockimages-cc0:fixed:test-row"
        development = priority(stock["developmentPriorityNamespace"], identifier)
        training = priority(stock["trainingPriorityNamespace"], identifier)
        self.assertEqual(development, priority(stock["developmentPriorityNamespace"], identifier))
        self.assertNotEqual(development, training)

    def test_deterministic_gzip_canonicalizes_header_and_round_trips(self) -> None:
        value = b"prooflens-m2\n" * 10
        first = deterministic_gzip(value)
        second = deterministic_gzip(value)
        self.assertEqual(first, second)
        self.assertEqual(first[9], 0xFF)
        self.assertEqual(gzip.decompress(first), value)

    def test_consumed_manifests_are_all_frozen_exclusions(self) -> None:
        recipe = json.loads((ROOT / "benchmark/m2/recipe.json").read_text())
        paths = {row["path"] for row in recipe["consumedEvaluationExclusions"]}
        self.assertEqual(
            paths,
            {
                "benchmark/manifests/test.jsonl",
                "benchmark/manifests/web-negative.jsonl",
                "benchmark/manifests/test-v2.jsonl",
                "benchmark/manifests/web-negative-v2.jsonl",
            },
        )

    def test_offline_missing_source_fails_closed(self) -> None:
        functions = selected_recovery_functions(
            "digest", "reject_symlink_components", "validate_data_root", "safe_output_path", "download"
        )
        download = functions["download"]
        with tempfile.TemporaryDirectory(dir=ROOT / "benchmark/data") as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "unavailable in offline verification"):
                download(
                    "https://example.invalid/pinned.parquet",
                    root / "pinned.parquet",
                    1,
                    "0" * 64,
                    allow_download=False,
                    allowed_root=root,
                )

    def test_cli_routes_download_and_offline_modes_explicitly(self) -> None:
        module = ast.parse(PREPARE.read_text())
        main = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        derive = next(
            node for node in ast.walk(main)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "derive_packet"
        )
        arguments = [ast.unparse(node) for node in ast.walk(main) if isinstance(node, ast.Constant)]
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in derive.keywords}
        self.assertIn("'--offline'", arguments)
        self.assertEqual(keywords["allow_download"], "not args.offline")
        self.assertEqual(keywords["materialize_pixels"], "args.materialize")

        derive_packet = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == "derive_packet"
        )
        load_stock = next(
            node for node in ast.walk(derive_packet)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "load_stock_candidates"
        )
        self.assertEqual(
            {keyword.arg: ast.unparse(keyword.value) for keyword in load_stock.keywords}["allow_download"],
            "allow_download",
        )

        verify_module = ast.parse(VERIFY.read_text())
        verify_call = next(
            node for node in ast.walk(verify_module)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "derive_packet"
        )
        self.assertEqual(
            {keyword.arg: ast.unparse(keyword.value) for keyword in verify_call.keywords}["allow_download"],
            "False",
        )


if __name__ == "__main__":
    unittest.main()
