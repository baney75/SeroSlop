from __future__ import annotations

import copy
from base64 import b64encode
from collections import namedtuple
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import pickle
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from benchmark.m5.contracts import (
    VARIANTS,
    branch_candidate_ids,
    choose_selector_threshold,
    complete_thresholds,
    load_recipe,
    metrics_at_threshold,
    read_jsonl,
    regression_gates_pass,
    regression_metrics,
    source_balanced_weights,
    validate_environment_receipt,
    validate_provisioning_receipt,
    validate_regression_state,
    validate_failure_receipt,
    validate_manifest_rows,
    validate_recipe,
    validate_selection_lock,
    canonical_json,
)
from benchmark.m5.train_gpu import (
    PaidTimeBudget,
    accumulation_window_samples,
    atomic_torch_save,
    ensure_run_marker,
    export_onnx,
    load_or_recover_branch_history,
    pack_float32,
    parser as training_parser,
    write_json,
)
from benchmark.m5.large_synthetic import DhashIndex, canonical_gzip, generator_is_excluded
from benchmark.m5.evaluate_large_synthetic import pack_float32 as pack_large_float32, validate_evaluation_receipt, wilson
from benchmark.m5.finalize import FINAL_ROWS, model_signature, replace_section


ROOT = Path(__file__).resolve().parents[2]


def selector_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(300):
        rows.append({"id": f"real-{index}", "label": 0, "source": "british-library-plates"})
    sources = ("rapidata-dalle-3", "rapidata-flux", "rapidata-midjourney", "rapidata-stable-diffusion")
    for index in range(300):
        rows.append({"id": f"synthetic-{index}", "label": 1, "source": sources[index % 4]})
    return rows


class M5ContractsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = load_recipe(ROOT / "benchmark/m5/recipe.json")

    def test_recipe_and_candidate_grid(self) -> None:
        self.assertEqual(
            branch_candidate_ids(self.recipe),
            [
                "last4-epoch-4", "last4-epoch-6", "last4-epoch-8",
                "full-epoch-4", "full-epoch-6", "full-epoch-8",
            ],
        )

    def test_recipe_mutations_reject(self) -> None:
        mutations = []
        for mutate in (
            lambda value: value["training"].__setitem__("requiredGpuProduct", "NVIDIA A100"),
            lambda value: value["selection"]["gates"]["original"].__setitem__("minimumRealRecall", 0.99),
            lambda value: value["h3Boundary"].__setitem__("pixelsMayBeRead", True),
            lambda value: value["deliverable"].__setitem__("maximumBytes", 900_000_000),
            lambda value: value["sourceEvidence"]["trainingManifest"].__setitem__("items", 1),
            lambda value: value["largeSyntheticEvaluation"].__setitem__("minimumMeanBatchRecallExclusive", 0.95 + 1e-6),
            lambda value: value["training"].__setitem__("maximumPaidWallClockSeconds", 28_801),
            lambda value: value["training"].__setitem__("providerAutoStopRequired", True),
        ):
            candidate = copy.deepcopy(self.recipe)
            mutate(candidate)
            mutations.append(candidate)
        for candidate in mutations:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    validate_recipe(candidate)

    def test_global_source_balanced_mass(self) -> None:
        rows = [
            {"label": 0, "source": "real-a"}, {"label": 0, "source": "real-a"},
            {"label": 0, "source": "real-b"},
            {"label": 1, "source": "fake-a"}, {"label": 1, "source": "fake-b"},
            {"label": 1, "source": "fake-b"},
        ]
        weights = source_balanced_weights(rows)
        self.assertEqual(weights, [0.75, 0.75, 1.5, 1.5, 0.75, 0.75])
        self.assertAlmostEqual(sum(weights), len(rows))

    def test_thresholds_cover_float_boundaries(self) -> None:
        values = [1.0, math.nextafter(1.0, math.inf), -float.fromhex("0x1.fffffffffffffp+1023")]
        thresholds = complete_thresholds(values)
        self.assertIn(1.0, thresholds)
        self.assertIn(math.nextafter(1.0, math.inf), thresholds)
        self.assertLess(thresholds[0], values[-1])

    def test_zero_false_positive_selector_can_pass(self) -> None:
        rows = selector_rows()
        logits = [-4.0] * 300 + [4.0] * 300
        selected = choose_selector_threshold({variant: logits for variant in VARIANTS}, rows, self.recipe["selection"]["gates"])
        self.assertIsNotNone(selected)
        assert selected is not None
        threshold, metrics, _ = selected
        self.assertGreater(threshold, -4.0)
        self.assertTrue(all(value.false_positives == 0 for value in metrics.values()))

    def test_one_false_positive_fails_every_threshold_that_keeps_all_synthetic(self) -> None:
        rows = selector_rows()
        logits = [-4.0] * 300 + [4.0] * 300
        logits[0] = 5.0
        selected = choose_selector_threshold({variant: logits for variant in VARIANTS}, rows, self.recipe["selection"]["gates"])
        self.assertIsNone(selected)

    def test_manifest_duplicate_and_escape_reject(self) -> None:
        rows = [
            {"id": "a", "imageSha256": "0" * 64, "path": "train/a.jpg", "label": 0, "source": "real", "rowIndex": 0},
            {"id": "b", "imageSha256": "1" * 64, "path": "train/b.jpg", "label": 1, "source": "fake", "rowIndex": 1},
        ]
        validate_manifest_rows(rows, expected_items=2, expected_class_counts={"real": 1, "synthetic": 1})
        broken = copy.deepcopy(rows)
        broken[1]["path"] = "../h3/secret.jpg"
        with self.assertRaises(ValueError):
            validate_manifest_rows(broken, expected_items=2, expected_class_counts={"real": 1, "synthetic": 1})

    def test_environment_requires_runpod_l40s(self) -> None:
        receipt = {
            "provider": self.recipe["training"]["provider"],
            "gpuProduct": "NVIDIA L40S",
            "gpuMemoryBytes": 48_000_000_000,
            "cudaAvailable": True,
            "cudaVersion": "12.8",
            "driverVersion": "570",
            "torchVersion": "2.8.0+cu128",
            "transformersVersion": "5.4.0",
            "pythonVersion": "3.11.11",
            "runpodPodIdSha256": "a" * 64,
            "provisioningReceiptSha256": "b" * 64,
            "containerImage": self.recipe["training"]["containerImage"],
            "requirementsSha256": self.recipe["training"]["requirementsSha256"],
            "providerEvidenceBoundary": "operator-recorded-not-cryptographic-attestation",
        }
        validate_environment_receipt(receipt, self.recipe)
        receipt["gpuProduct"] = "NVIDIA RTX 4090"
        with self.assertRaises(ValueError):
            validate_environment_receipt(receipt, self.recipe)

    def _provisioning(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "status": "runpod-provisioned",
            "provider": "RunPod",
            "cloudType": "SECURE",
            "gpuProduct": "NVIDIA L40S",
            "containerImage": self.recipe["training"]["containerImage"],
            "podIdSha256": "a" * 64,
            "createdAtUnix": 1_000_000,
            "maximumRuntimeSeconds": 28_800,
            "workloadStopAtUnix": 1_028_800,
            "providerAutoStopAvailable": False,
            "operatorStopRequired": True,
            "stopControl": "trainer-deadline-plus-authenticated-operator-stop",
            "controlPlaneObservationSha256": "c" * 64,
            "evidenceBoundary": "operator-recorded-control-plane-observation-not-cryptographic-attestation",
        }

    def test_provisioning_and_paid_deadline_are_executable(self) -> None:
        receipt = self._provisioning()
        validate_provisioning_receipt(receipt, self.recipe)
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / "paid-time.json"
            budget = PaidTimeBudget(receipt, self.recipe, state, clock=lambda: 1_000_100.0)
            budget.check("test")
            self.assertTrue(state.is_file())
            expired = PaidTimeBudget.__new__(PaidTimeBudget)
            expired.created = 1_000_000
            expired.workload_stop = 1_028_800
            expired.deadline = 1_028_500
            expired.maximum = 28_800
            expired.state_path = state
            expired.clock = lambda: 1_028_500.0
            expired.last_persisted = 0.0
            with self.assertRaises(TimeoutError):
                expired.check("after-deadline")

    def test_terminal_regression_state_recomputes_every_metric(self) -> None:
        selection = {
            "selectedCandidateId": "last4-epoch-4",
            "selectedModel": {"sha256": "d" * 64},
            "rawThreshold": 0.0,
            "selectorMetrics": {"bound": True},
        }
        results = []
        for regression in self.recipe["terminalRegressions"]:
            rows = read_jsonl(ROOT / regression["manifest"])
            values = [-10.0 if row["label"] == 0 else 10.0 for row in rows]
            metrics = {variant: regression_metrics(values, rows, 0.0) for variant in VARIANTS}
            self.assertTrue(regression_gates_pass(metrics, regression["gates"]))
            results.append({
                "name": regression["name"],
                "manifestSha256": regression["sha256"],
                "items": len(rows),
                "metrics": metrics,
                "logits": {variant: pack_float32(values) for variant in VARIANTS},
                "gates": regression["gates"],
                "passed": True,
            })
        state = {
            "schemaVersion": 1,
            "status": "regression-pass",
            "lockCommit": "a" * 40,
            "selectionLockSha256": "b" * 64,
            "selectedCandidateId": selection["selectedCandidateId"],
            "selectedModelSha256": selection["selectedModel"]["sha256"],
            "rawThreshold": 0.0,
            "selectorOnnxReplay": {
                "selectorManifestSha256": self.recipe["sourceEvidence"]["selectorManifest"]["sha256"],
                "items": 600,
                "maximumAbsoluteLogitDeltaByVariant": {variant: 0.0 for variant in VARIANTS},
                "parityTolerance": self.recipe["initialModel"]["maximumPytorchOnnxParityError"],
                "lockedThreshold": 0.0,
                "replayedBestThreshold": 0.0,
                "metricsAtLockedThreshold": selection["selectorMetrics"],
                "passed": True,
            },
            "results": results,
            "selectionInfluencedByRegression": False,
            "h3PixelsRead": False,
        }
        validate_regression_state(
            state, self.recipe, selection, lock_commit="a" * 40, selection_lock_sha256="b" * 64,
        )
        broken = copy.deepcopy(state)
        broken["results"][0]["passed"] = False
        with self.assertRaises(ValueError):
            validate_regression_state(
                broken, self.recipe, selection, lock_commit="a" * 40, selection_lock_sha256="b" * 64,
            )

    def test_finalizer_surface_and_marker_replacement_are_exact(self) -> None:
        self.assertEqual(len(FINAL_ROWS), 13)
        self.assertEqual(len({row[0] for row in FINAL_ROWS}), 13)
        self.assertEqual(replace_section("a<start>old<end>b", "<start>", "<end>", "new"), "anewb")
        with self.assertRaises(ValueError):
            replace_section("<start>x<start>y<end>", "<start>", "<end>", "new")

    def test_exported_onnx_uses_finalizer_batch_interface(self) -> None:
        import torch

        tiny_output = namedtuple("TinyOutput", ("logits",))

        class TinyClassifier(torch.nn.Module):
            def forward(self, *, pixel_values: object) -> object:
                return tiny_output(pixel_values.mean(dim=(1, 2, 3)).unsqueeze(1))

        candidates = ROOT / "benchmark/candidates"
        candidates.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=candidates) as directory:
            destination = Path(directory) / "tiny.onnx"
            export_onnx(TinyClassifier(), destination, self.recipe)
            signature = model_signature(destination)
        self.assertEqual(signature["inputs"], [("pixel_values", ["batch", 3, 384, 384])])
        self.assertEqual(signature["outputs"], [("logits", ["batch", 1])])

    def test_uneven_final_accumulation_window_uses_actual_items(self) -> None:
        self.assertEqual(accumulation_window_samples(112_562, 64, 2, 1), 128)
        self.assertEqual(accumulation_window_samples(112_562, 64, 2, 1_758), 128)
        self.assertEqual(accumulation_window_samples(112_562, 64, 2, 1_759), 50)
        self.assertEqual(1 / accumulation_window_samples(112_562, 64, 2, 1_759), 1 / 50)

    def test_full_training_requires_bound_preflight(self) -> None:
        environment = {
            "provider": self.recipe["training"]["provider"],
            "gpuProduct": "NVIDIA L40S",
            "gpuMemoryBytes": 48_000_000_000,
            "cudaAvailable": True,
            "cudaVersion": "12.8",
            "driverVersion": "570",
            "torchVersion": "2.8.0+cu128",
            "transformersVersion": "5.4.0",
            "pythonVersion": "3.11.11",
            "runpodPodIdSha256": "a" * 64,
            "provisioningReceiptSha256": "b" * 64,
            "containerImage": self.recipe["training"]["containerImage"],
            "requirementsSha256": self.recipe["training"]["requirementsSha256"],
            "providerEvidenceBoundary": "operator-recorded-not-cryptographic-attestation",
        }
        with tempfile.TemporaryDirectory(dir=ROOT / "benchmark") as directory:
            output = Path(directory)
            (output / "runpod-provisioning-receipt.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "preflight"):
                ensure_run_marker(output, self.recipe, "a" * 40, environment, self._provisioning(), "b" * 64)
            receipt = {
                "schemaVersion": 1,
                "status": "preflight-pass",
                "protocolCommit": "a" * 40,
                "recipeSha256": sha256((ROOT / "benchmark/m5/recipe.json").read_bytes()).hexdigest(),
                "environment": environment,
                "provisioningReceiptSha256": "b" * 64,
                "selectorRead": False,
                "terminalRegressionsRead": False,
                "h3PixelsRead": False,
            }
            write_json(output / "preflight" / "preflight-receipt.json", receipt)
            ensure_run_marker(output, self.recipe, "a" * 40, environment, self._provisioning(), "b" * 64)
            self.assertTrue((output / "run-marker.json").is_file())
            receipt["selectorRead"] = True
            write_json(output / "preflight" / "preflight-receipt.json", receipt)
            with self.assertRaisesRegex(ValueError, "preflight"):
                ensure_run_marker(output, self.recipe, "a" * 40, environment, self._provisioning(), "b" * 64)

    def test_unsealed_complete_epoch_is_recovered_and_sealed(self) -> None:
        fake_torch = SimpleNamespace(
            save=lambda value, path: Path(path).write_bytes(pickle.dumps(value)),
            load=lambda path, **_kwargs: pickle.loads(Path(path).read_bytes()),
        )
        candidates_root = ROOT / "benchmark/candidates"
        candidates_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=candidates_root) as directory, mock.patch.dict(sys.modules, {"torch": fake_torch}):
            output = Path(directory)
            payload = {
                "model": {"weight": [1.0]},
                "optimizer": {},
                "scheduler": {},
                "branch": "last4",
                "epoch": 1,
                "globalStep": 2,
                "epochReceipts": [{"epoch": 1}],
            }
            atomic_torch_save(output / "resume/last4-epoch-1.pt", payload)
            restored, receipts = load_or_recover_branch_history(output, {"name": "last4", "candidateEpochs": [1]})
            self.assertEqual(restored, payload)
            self.assertEqual(receipts, [{"epoch": 1}])
            self.assertTrue((output / "checkpoints/last4-epoch-1.pt").is_file())
            self.assertTrue((output / "seals/last4-epoch-1.json").is_file())
            (output / "checkpoints/last4-epoch-1.pt").unlink()
            with self.assertRaisesRegex(ValueError, "candidate"):
                load_or_recover_branch_history(output, {"name": "last4", "candidateEpochs": [1]})

    def test_large_panel_dhash_and_generator_exclusions(self) -> None:
        payload = b"row\n" * 100
        self.assertEqual(gzip.decompress(canonical_gzip(payload)), payload)
        self.assertEqual(canonical_gzip(gzip.decompress(canonical_gzip(payload))), canonical_gzip(payload))
        index = DhashIndex()
        index.add("0000000000000000")
        self.assertTrue(index.has_near("00000000000000ff"))
        self.assertFalse(index.has_near("00000000000001ff"))
        excluded = self.recipe["largeSyntheticEvaluation"]["source"]["excludedGeneratorFamilies"]
        self.assertTrue(generator_is_excluded("FLUX.1-dev", excluded))
        self.assertFalse(generator_is_excluded("Kandinsky-3", excluded))

    def test_large_panel_strict_mean_and_median_gate_recomputes_logits(self) -> None:
        panel_rows = [
            {"batchIndex": index // 100, "generator": "generator-a"}
            for index in range(100_000)
        ]
        selection = {"selectedModel": {"path": "model.onnx", "bytes": 1, "sha256": "a" * 64}, "rawThreshold": 0.0}
        values = [1.0] * 100_000
        packet = pack_large_float32(values)
        batch_results = [{"batchIndex": index, "items": 100, "correct": 100, "recall": 1.0} for index in range(1_000)]
        lower, upper = wilson(100_000, 100_000)
        receipt = {
            "schemaVersion": 1,
            "status": "large-synthetic-pass",
            "acceptanceEligible": True,
            "sourceLockCommit": "b" * 40,
            "selectionLockCommit": "c" * 40,
            "selectionLockSha256": "d" * 64,
            "sourceLockSha256": "e" * 64,
            "manifestSha256": "f" * 64,
            "batchAssignmentSha256": "1" * 64,
            "model": selection["selectedModel"],
            "rawThreshold": 0.0,
            "items": 100_000,
            "batchSize": 100,
            "batches": 1_000,
            "correct": 100_000,
            "overallRecall": 1.0,
            "meanBatchRecall": 1.0,
            "medianBatchRecall": 1.0,
            "minimumBatchRecall": 1.0,
            "wilson95": {"lower": lower, "upper": upper},
            "batchResults": batch_results,
            "generatorResults": {"generator-a": {"items": 100_000, "correct": 100_000, "recall": 1.0}},
            "minimumGeneratorRecall": 1.0,
            "logits": packet,
            "selectionInfluence": False,
            "regressionStateSha256": "2" * 64,
            "h3PixelsRead": False,
        }
        validate_evaluation_receipt(
            receipt, self.recipe, selection_lock=selection, source_lock_commit="b" * 40, panel_rows=panel_rows,
            verify_artifact_bindings=False,
        )
        broken = copy.deepcopy(receipt)
        broken["meanBatchRecall"] = 0.95
        broken["medianBatchRecall"] = 0.95
        with self.assertRaises(ValueError):
            validate_evaluation_receipt(
                broken, self.recipe, selection_lock=selection, source_lock_commit="b" * 40, panel_rows=panel_rows,
                verify_artifact_bindings=False,
            )

    def test_preflight_mode_is_explicit_and_selector_free_by_contract(self) -> None:
        arguments = training_parser().parse_args(["--protocol-commit", "a" * 40, "--preflight-only"])
        self.assertTrue(arguments.preflight_only)
        self.assertFalse(hasattr(arguments, "h3_manifest"))

    def test_selection_lock_is_recomputed_from_embedded_logits(self) -> None:
        rows = selector_rows()
        logits = [-4.0] * 300 + [4.0] * 300
        payload = b"".join(struct.pack("<f", value) for value in logits)
        packet = {
            "dtype": "float32-little-endian",
            "count": 600,
            "sha256": sha256(payload).hexdigest(),
            "base64": b64encode(payload).decode("ascii"),
        }
        candidates = []
        for order, candidate_id in enumerate(branch_candidate_ids(self.recipe)):
            selected = choose_selector_threshold({variant: logits for variant in VARIANTS}, rows, self.recipe["selection"]["gates"])
            assert selected is not None
            threshold, metrics, key = selected
            epoch = int(candidate_id.rpartition("-epoch-")[2])
            ranking = (*key[:-1], -order, -epoch, key[-1])
            model = {
                "path": f"benchmark/candidates/prooflens-cf384-m5/models/{candidate_id}.onnx",
                "bytes": 87_000_000,
                "sha256": f"{order + 1:064x}",
                "parityMaximumAbsoluteError": 0.0,
                "parityProvider": "CUDAExecutionProvider",
            }
            candidates.append({
                "candidateId": candidate_id,
                "checkpoint": {"path": f"x/{candidate_id}.pt", "bytes": 1, "sha256": "f" * 64},
                "model": model,
                "selectorLogits": {variant: packet for variant in VARIANTS},
                "accepted": True,
                "rawThreshold": threshold,
                "metrics": {
                    variant: {
                        "balancedAccuracy": value.balanced_accuracy,
                        "realRecall": value.real_recall,
                        "syntheticRecall": value.synthetic_recall,
                        "syntheticRecallBySource": value.synthetic_recall_by_source,
                        "falsePositives": value.false_positives,
                    }
                    for variant, value in metrics.items()
                },
                "selectionKey": list(ranking),
            })
        recipe_sha = sha256((ROOT / "benchmark/m5/recipe.json").read_bytes()).hexdigest()
        grid = {
            "schemaVersion": 1,
            "recipeSha256": recipe_sha,
            "protocolCommit": "a" * 40,
            "candidates": candidates,
            "selectorManifestSha256": self.recipe["sourceEvidence"]["selectorManifest"]["sha256"],
            "h3PixelsRead": False,
        }
        summary = {"status": "selector-pass"}
        winner = candidates[0]
        threshold = winner["rawThreshold"]
        lock = {
            "schemaVersion": 1,
            "status": "m5-selected-pre-regression",
            "acceptanceEligible": False,
            "recipeSha256": recipe_sha,
            "protocolCommit": "a" * 40,
            "trainingSummary": summary,
            "trainingSummarySha256": sha256(canonical_json(summary)).hexdigest(),
            "candidateGrid": grid,
            "candidateGridSha256": sha256(canonical_json(grid)).hexdigest(),
            "selectedCandidateId": winner["candidateId"],
            "selectedModel": winner["model"],
            "rawThreshold": threshold,
            "calibration": {"slope": 1.0, "intercept": math.log(0.65 / 0.35) - threshold, "displayThreshold": 0.65},
            "selectorMetrics": winner["metrics"],
            "selectionKey": winner["selectionKey"],
            "selectionInfluencedByRegression": False,
            "terminalRegressionsRead": False,
            "h3PixelsRead": False,
        }
        validate_selection_lock(lock, self.recipe, rows)
        broken = copy.deepcopy(lock)
        broken["candidateGrid"]["candidates"][0]["selectorLogits"]["original"]["base64"] = "AAAA"
        broken["candidateGridSha256"] = sha256(canonical_json(broken["candidateGrid"])).hexdigest()
        with self.assertRaises(ValueError):
            validate_selection_lock(broken, self.recipe, rows)

    def test_failure_receipt_recomputes_all_rejected_candidates(self) -> None:
        rows = selector_rows()
        logits = [0.0] * len(rows)
        payload = b"".join(struct.pack("<f", value) for value in logits)
        packet = {
            "dtype": "float32-little-endian",
            "count": len(rows),
            "sha256": sha256(payload).hexdigest(),
            "base64": b64encode(payload).decode("ascii"),
        }
        candidates = [{
            "candidateId": candidate_id,
            "checkpoint": {"path": f"x/{candidate_id}.pt", "bytes": 1, "sha256": "f" * 64},
            "model": {
                "path": f"benchmark/candidates/prooflens-cf384-m5/models/{candidate_id}.onnx",
                "bytes": 87_000_000,
                "sha256": f"{index + 1:064x}",
                "parityMaximumAbsoluteError": 0.0,
                "parityProvider": "CUDAExecutionProvider",
            },
            "selectorLogits": {variant: packet for variant in VARIANTS},
            "accepted": False,
        } for index, candidate_id in enumerate(branch_candidate_ids(self.recipe))]
        recipe_sha = sha256((ROOT / "benchmark/m5/recipe.json").read_bytes()).hexdigest()
        grid = {
            "schemaVersion": 1,
            "recipeSha256": recipe_sha,
            "protocolCommit": "a" * 40,
            "candidates": candidates,
            "selectorManifestSha256": self.recipe["sourceEvidence"]["selectorManifest"]["sha256"],
            "h3PixelsRead": False,
        }
        summary = {
            "status": "selector-fail", "protocolCommit": "a" * 40, "selectedCandidateId": None,
            "h3PixelsRead": False, "terminalRegressionsRead": False,
        }
        receipt = {
            "schemaVersion": 1,
            "status": "failed-m5-selector",
            "acceptanceEligible": False,
            "recipeSha256": recipe_sha,
            "protocolCommit": "a" * 40,
            "trainingSummary": summary,
            "trainingSummarySha256": sha256(canonical_json(summary)).hexdigest(),
            "candidateGrid": grid,
            "candidateGridSha256": sha256(canonical_json(grid)).hexdigest(),
            "h3PixelsRead": False,
            "terminalRegressionsRead": False,
            "reason": "No frozen candidate passed.",
        }
        validate_failure_receipt(receipt, self.recipe, rows)
        broken = copy.deepcopy(receipt)
        broken["candidateGrid"]["candidates"][0]["accepted"] = True
        broken["candidateGridSha256"] = sha256(canonical_json(broken["candidateGrid"])).hexdigest()
        with self.assertRaises(ValueError):
            validate_failure_receipt(broken, self.recipe, rows)


if __name__ == "__main__":
    unittest.main()
