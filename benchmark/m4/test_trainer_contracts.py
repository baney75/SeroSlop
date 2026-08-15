from __future__ import annotations

import ast
import base64
import json
from pathlib import Path
import unittest

import numpy as np

from benchmark.m4.train_adapter import AdapterCandidate, candidate_seal, validate_candidate


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "benchmark/m4/train_adapter.py"
SOURCE = TRAINER.read_text()
TREE = ast.parse(SOURCE)
RECIPE = json.loads((ROOT / "benchmark/m4/recipe.json").read_text())


def function(name: str) -> ast.FunctionDef:
    return next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == name)


class M4TrainerContractTests(unittest.TestCase):
    @staticmethod
    def candidate(
        candidate_id: str = "wd-0.003-anchor-0.01",
        weight_decay: float = 0.003,
        anchor_coefficient: float = 0.01,
    ) -> AdapterCandidate:
        return AdapterCandidate(
            candidate_id=candidate_id,
            weight_decay=weight_decay,
            anchor_coefficient=anchor_coefficient,
            mean=np.zeros(384, dtype=np.float32),
            std=np.ones(384, dtype=np.float32),
            input_weight=np.zeros((64, 384), dtype=np.float32),
            input_bias=np.zeros(64, dtype=np.float32),
            output_weight=np.zeros((384, 64), dtype=np.float32),
            output_bias=np.zeros(384, dtype=np.float32),
        )

    def test_candidate_seal_embeds_canonical_float32_tensor_bytes(self) -> None:
        candidate = self.candidate()
        seal = candidate_seal(candidate)
        self.assertEqual(list(seal["tensorFloat32Base64"]), sorted(candidate.arrays()))
        for name, value in sorted(candidate.arrays().items()):
            self.assertEqual(
                base64.b64decode(seal["tensorFloat32Base64"][name], validate=True),
                np.ascontiguousarray(value).tobytes(),
            )

    def test_candidate_identity_is_exactly_bound_to_the_frozen_pair(self) -> None:
        validate_candidate(self.candidate())
        for candidate in (
            self.candidate(candidate_id="../../outside"),
            self.candidate(candidate_id="/tmp/outside"),
            self.candidate(candidate_id="wd-0.003-anchor-0.03"),
            self.candidate(weight_decay=0.004),
        ):
            with self.subTest(candidate_id=candidate.candidate_id, weight_decay=candidate.weight_decay):
                with self.assertRaisesRegex(ValueError, "identity"):
                    validate_candidate(candidate)

    def test_optimizer_cannot_accept_selector_or_regression_arrays(self) -> None:
        arguments = [argument.arg for argument in function("fit_candidate").args.args]
        self.assertEqual(
            arguments,
            [
                "train_features", "train_labels", "train_sources", "upstream_weight",
                "upstream_bias",
            ],
        )
        keyword_only = [argument.arg for argument in function("fit_candidate").args.kwonlyargs]
        self.assertEqual(keyword_only, ["weight_decay", "anchor_coefficient", "recipe"])
        names = {node.id for node in ast.walk(function("fit_candidate")) if isinstance(node, ast.Name)}
        self.assertNotIn("selector", names)
        self.assertNotIn("regression", names)

    def test_all_candidate_tensors_are_sealed_before_selector_and_winner_before_regression(self) -> None:
        fit = SOURCE.index("candidates = fit_candidate_grid(")
        tensor_seal = SOURCE.index('seal_path = args.output_dir / "candidate-tensor-seal.json"')
        selector = SOURCE.index("grid, selected, selected_row = select_candidate(")
        winner = SOURCE.index('selection_lock_path = args.output_dir / "selection-lock.json"')
        regression = SOURCE.index("regression_specs = [")
        self.assertLess(fit, tensor_seal)
        self.assertLess(tensor_seal, selector)
        self.assertLess(selector, winner)
        self.assertLess(winner, regression)
        self.assertIn('"selectionLock": selection_lock', SOURCE)

    def test_finite_grid_and_adapter_shape_match_recipe(self) -> None:
        self.assertEqual(RECIPE["training"]["weightDecays"], [0.003, 0.01, 0.03])
        self.assertEqual(RECIPE["training"]["anchorCoefficients"], [0.01, 0.03, 0.1, 0.3])
        self.assertEqual(
            RECIPE["training"]["candidateCount"],
            len(RECIPE["training"]["weightDecays"]) * len(RECIPE["training"]["anchorCoefficients"]),
        )
        self.assertEqual(RECIPE["adapter"]["trainableParameters"], 49_600)
        self.assertTrue(RECIPE["adapter"]["classifierFrozen"])
        self.assertEqual(RECIPE["adapter"]["width"], 64)

    def test_source_balanced_bce_anchor_and_deterministic_epoch_order_are_executable(self) -> None:
        body = ast.unparse(function("fit_candidate"))
        self.assertIn("source_balanced_weights", body)
        self.assertIn("binary_cross_entropy_with_logits", body)
        self.assertIn("protectedAnchorSources", body)
        self.assertIn("anchor_coefficient", body)
        self.assertIn("manual_seed(int(recipe['seed']) + epoch)", body)
        self.assertIn("torch.randperm", body)
        self.assertIn("adapter_out.weight.zero_()", body)
        self.assertIn("adapter_out.bias.zero_()", body)
        self.assertIn("torch.sum(bce * batch_weights) * batch_count", body)
        self.assertNotIn("/ torch.sum(batch_weights)", body)
        self.assertIn("/ protected_count", body)

    def test_selector_labels_cannot_enter_candidate_fitting(self) -> None:
        fit_names = {node.id for node in ast.walk(function("fit_candidate_grid")) if isinstance(node, ast.Name)}
        self.assertNotIn("selector_labels", fit_names)
        self.assertNotIn("selector", fit_names)
        call = next(
            node for node in ast.walk(function("main"))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "fit_candidate_grid"
        )
        positional = [ast.unparse(value) for value in call.args]
        self.assertEqual(positional[:3], ["train[0]", "train[1]", "train[3]"])
        self.assertFalse(any("selector" in value or "regression" in value for value in positional))

    def test_selection_uses_complete_binary64_partitions_and_terminal_order(self) -> None:
        body = ast.unparse(function("evaluate_selector_candidate"))
        self.assertIn("complete_decision_thresholds", body)
        self.assertIn("thresholdPartitions", body)
        self.assertEqual(
            RECIPE["selectionPolicy"]["regressionOrder"],
            ["m3-selector-regression", "m2-development-regression"],
        )
        self.assertIn("terminal", RECIPE["selectionPolicy"]["regressionFailure"])
        self.assertIn("begin_once(state_path", SOURCE)
        self.assertIn("outcome is unknown after interruption", SOURCE)

    def test_h3_is_not_a_trainer_argument_or_score_input(self) -> None:
        parser_body = ast.unparse(function("parse_args"))
        self.assertNotIn("h3", parser_body.lower())
        self.assertNotIn("benchmark/data/h3-met-holdout-v1", SOURCE)
        self.assertIn('"h3HoldoutScored": False', SOURCE)
        self.assertIn('"h3PixelsRead": False', SOURCE)

    def test_manifest_paths_are_checked_before_feature_session_creation(self) -> None:
        main = SOURCE[SOURCE.index("def main()") :]
        self.assertLess(main.index("safe_manifest_items(args.train_manifest"), main.index("create_session(args)"))
        safe_body = ast.unparse(function("safe_manifest_items"))
        self.assertIn("is_absolute", safe_body)
        self.assertIn("'..' in Path(relative).parts", safe_body)
        self.assertIn("is_symlink", safe_body)

    def test_completed_regressions_are_recomputed_and_chain_bound(self) -> None:
        body = ast.unparse(function("validate_completed_regression_state"))
        self.assertIn("decode_regression_logits", body)
        self.assertIn("variant_metrics", body)
        self.assertIn("passes_gates", body)
        self.assertIn("previousRegressionStateSha256", body)
        self.assertIn("type(state.get('passed')) is not bool", body)
        self.assertIn("'featureShardEvidence'", body)

    def test_grid_and_candidate_files_are_write_once(self) -> None:
        self.assertIn('write_exact_or_compare(args.output_dir / "candidate-grid.json", grid_packet)', SOURCE)
        self.assertIn("M4 sealed candidate changed on resume", SOURCE)

    def test_onnx_graph_contract_is_exact_and_classifier_stays_frozen(self) -> None:
        contract = RECIPE["onnxContract"]
        self.assertEqual(contract["opset"], 18)
        self.assertEqual(contract["featureTensor"], "/Gather_output_0")
        self.assertEqual(contract["classifierNodeName"], "/classifier/Gemm")
        self.assertEqual(
            contract["addedNodeNames"],
            [
                "m4_sub_mean", "m4_div_std", "m4_adapter_in", "m4_relu",
                "m4_adapter_out", "m4_scale_residual", "m4_add_residual",
            ],
        )
        export_body = ast.unparse(function("export_adapter_model"))
        self.assertIn("classifier.weight", export_body)
        self.assertIn("classifier.bias", export_body)
        self.assertNotIn("CopyFrom", export_body)
        self.assertIn("classifier.input[0] = 'm4.adapted'", export_body)


if __name__ == "__main__":
    unittest.main()
