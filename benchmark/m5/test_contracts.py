from __future__ import annotations

import copy
from base64 import b64encode
from collections import namedtuple
import gzip
from hashlib import sha256
import io
import inspect
import json
import math
import os
from pathlib import Path
import pickle
import runpy
import struct
import subprocess
import sys
import tarfile
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
    validate_run_authorization,
    validate_runpod_environment_authorization,
    validate_runtime_recovery_authorization,
    validate_regression_state,
    validate_failure_receipt,
    validate_manifest_rows,
    validate_numeric_audit_authorization,
    validate_parity_recovery_authorization,
    validate_cublas_recovery_authorization,
    validate_recipe,
    validate_initial_parity_diagnostic,
    validate_selection_lock,
    canonical_json,
    ort_cuda_providers,
    parse_json_bytes,
)
from benchmark.m5.train_gpu import (
    M5_BASE_SOURCE_COMMIT,
    M5_BASE_SOURCE_TREE,
    M5_FAILED_SOURCE_COMMIT,
    M5_FAILED_SOURCE_TREE,
    M5_CI_RECOVERY_COMMIT,
    M5_CI_RECOVERY_TREE,
    M5_ORIGINAL_PROTOCOL_COMMIT,
    M5_ORIGINAL_PROTOCOL_TREE,
    M5_P2_COMMIT,
    M5_P2_TREE,
    M5_P4_COMMIT,
    M5_P4_TREE,
    M5_A4_COMMIT,
    M5_A4_TREE,
    M5_R5_ROWS,
    M5_RUNTIME_AUTHORIZATION_COMMIT,
    M5_RUNTIME_AUTHORIZATION_TREE,
    M5_RUNTIME_RECOVERY_COMMIT,
    M5_RUNTIME_RECOVERY_TREE,
    M5_RUNPOD_ENV_AUTHORIZATION_COMMIT,
    M5_RUNPOD_ENV_AUTHORIZATION_TREE,
    M5_RUNPOD_ENV_RECOVERY_COMMIT,
    M5_RUNPOD_ENV_RECOVERY_TREE,
    M5_PROTOCOL_RECOVERY_PATHS,
    M5_SOURCE_CI_RECOVERY_ROWS,
    M5_SOURCE_RECOVERY_ROWS,
    M5_RUNTIME_RECOVERY_ROWS,
    M5_RUNPOD_ENV_RECOVERY_ROWS,
    M5_NUMERIC_AUDIT_RECOVERY_ROWS,
    PaidTimeBudget,
    accumulation_window_samples,
    atomic_torch_save,
    ensure_run_marker,
    evaluate_candidates,
    export_onnx,
    load_or_recover_branch_history,
    pack_float32,
    parser as training_parser,
    predict_onnx_variant,
    require_cuda_determinism_environment,
    resolve_authorized_protocol_commit,
    validate_source_recovery_history,
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


def training_summary_fixture(
    recipe: dict[str, object],
    *,
    candidate_grid_sha256: str,
    status: str,
    selected_candidate_id: str | None,
) -> dict[str, object]:
    training = recipe["training"]
    source_evidence = recipe["sourceEvidence"]
    upstream = recipe["upstream"]
    environment = {
        "provider": training["provider"],
        "gpuProduct": "NVIDIA L40S",
        "gpuMemoryBytes": 48_000_000_000,
        "cudaAvailable": True,
        "cudaVersion": "12.8",
        "driverVersion": "570",
        "torchVersion": "2.8.0+cu128",
        "transformersVersion": "5.4.0",
        "pythonVersion": "3.11.11",
        "launchNodeVersion": "v24.18.1",
        "launchNodeSha256": "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a",
        "runpodPodIdSha256": "a" * 64,
        "provisioningReceiptSha256": "b" * 64,
        "containerImage": training["containerImage"],
        "requirementsSha256": training["requirementsSha256"],
        "providerEvidenceBoundary": "operator-recorded-not-cryptographic-attestation",
        "providerIdentityEvidence": training["providerIdentityEvidence"],
        "providerSignedAttestation": False,
        "runtimeConsistencyEvidence": training["runtimeConsistencyEvidence"],
        "cublasWorkspaceConfig": training["deterministicCudaRuntime"]["cublasWorkspaceConfig"],
        "sourceCommit": "a" * 40,
        "sourceTree": "c" * 40,
        "authorizationCommit": "d" * 40,
        "authorizationReceiptSha256": "e" * 64,
        "authorizationPublicCi": {
            "conclusion": "success", "event": "push", "headSha": "d" * 40, "runId": 456,
            "status": "completed", "url": "https://github.com/baney75/prooflens/actions/runs/456",
            "workflowPath": ".github/workflows/quality.yml",
        },
    }
    epoch_receipts = []
    global_step = 0
    for branch in training["branches"]:
        for epoch in range(1, int(training["epochs"]) + 1):
            global_step += 1
            epoch_receipts.append({
                "branch": branch["name"],
                "epoch": epoch,
                "globalStep": global_step,
                "seconds": 1.0,
                "images": source_evidence["trainingManifest"]["items"],
                "meanWeightedBce": 0.1,
                "meanMaskedTeacherMse": 0.1,
                "learningRates": {"classifier": 0.0001},
            })
    return {
        "schemaVersion": 1,
        "status": status,
        "recipeSha256": sha256((ROOT / "benchmark/m5/recipe.json").read_bytes()).hexdigest(),
        "protocolCommit": "a" * 40,
        "environment": environment,
        "upstreamSourceSha256": {
            upstream[key]["path"]: upstream[key]["sha256"]
            for key in ("config", "preprocessor", "pytorchWeights")
        },
        "initialPytorchOnnxParityMaximumAbsoluteError": 0.0,
        "trainingManifestSha256": source_evidence["trainingManifest"]["compressedSha256"],
        "selectorManifestSha256": source_evidence["selectorManifest"]["sha256"],
        "trainingItems": source_evidence["trainingManifest"]["items"],
        "selectorItems": source_evidence["selectorManifest"]["items"],
        "epochReceipts": epoch_receipts,
        "candidateGrid": {
            "path": f"{recipe['output']['candidateRoot']}/candidate-grid.json",
            "sha256": candidate_grid_sha256,
        },
        "selectedCandidateId": selected_candidate_id,
        "h3PixelsRead": False,
        "terminalRegressionsRead": False,
    }


class M5ContractsTest(unittest.TestCase):

    def test_initial_parity_diagnostic_is_score_blind_and_tf32_bound(self) -> None:
        diagnostic_path = ROOT / self.recipe["sourceEvidence"]["initialParityDiagnostic"]["path"]
        diagnostic = parse_json_bytes(diagnostic_path.read_bytes(), label="initial parity diagnostic")
        validate_initial_parity_diagnostic(diagnostic, self.recipe)

    def test_onnx_scoring_disables_tf32_and_uses_exported_candidates(self) -> None:
        fake_ort = SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.assertEqual(
            ort_cuda_providers(fake_ort),
            [("CUDAExecutionProvider", {"use_tf32": "0"})],
        )
        with self.assertRaisesRegex(ValueError, "CUDAExecutionProvider"):
            ort_cuda_providers(SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]))
        evaluation_source = inspect.getsource(evaluate_candidates)
        prediction_source = inspect.getsource(predict_onnx_variant)
        self.assertIn("export_onnx", evaluation_source)
        self.assertIn("ort.InferenceSession", evaluation_source)
        self.assertIn("predict_onnx_variant", evaluation_source)
        self.assertIn("session.run", prediction_source)
        self.assertNotIn("autocast", prediction_source)

    def test_run_authorization_binds_exact_p3_source(self) -> None:
        paths = [
            {"path": "benchmark/m5/contracts.py", "sha256": "a" * 64},
            {"path": "scripts/m5-run-authorization.mjs", "sha256": "b" * 64},
        ]
        receipt = {
            "schemaVersion": 1, "status": "m5-source-recovery-authorized",
            "protocolCommit": "1c4ac973785f937fa9023018863941e6d89d8693",
            "protocolTree": "a56caae4291e275029076417fb2111be76b07a41",
            "sourceCommit": "c" * 40, "sourceTree": "d" * 40,
            "sourcePathMap": paths, "authorizationPath": "benchmark/evidence/m5/run-authorization.json",
            "sourcePublicCi": {
                "conclusion": "success", "event": "push", "headSha": "c" * 40,
                "runId": 123, "status": "completed",
                "url": "https://github.com/baney75/prooflens/actions/runs/123",
                "workflowPath": ".github/workflows/quality.yml",
            },
            "scoreBlind": True, "h3PixelsRead": False,
        }
        validate_run_authorization(receipt, protocol_commit=receipt["protocolCommit"], protocol_tree=receipt["protocolTree"],
                                    source_commit=receipt["sourceCommit"], source_tree=receipt["sourceTree"], source_path_map=paths)
        for key, value in (("sourceCommit", "e" * 40), ("sourceTree", "e" * 40)):
            broken = dict(receipt); broken[key] = value
            with self.assertRaises(ValueError):
                validate_run_authorization(broken, protocol_commit=receipt["protocolCommit"], protocol_tree=receipt["protocolTree"],
                                            source_commit=receipt["sourceCommit"], source_tree=receipt["sourceTree"], source_path_map=paths)
        broken = copy.deepcopy(receipt)
        broken["sourcePublicCi"]["conclusion"] = "failure"
        with self.assertRaises(ValueError):
            validate_run_authorization(broken, protocol_commit=receipt["protocolCommit"], protocol_tree=receipt["protocolTree"],
                                       source_commit=receipt["sourceCommit"], source_tree=receipt["sourceTree"], source_path_map=paths)

    def test_runtime_recovery_authorization_binds_prior_p4_and_new_source(self) -> None:
        paths = [{"path": "benchmark/m5/train_gpu.py", "sha256": "a" * 64}]
        receipt = {
            "schemaVersion": 2, "status": "m5-runtime-recovery-authorized",
            "protocolCommit": M5_P2_COMMIT, "protocolTree": M5_P2_TREE,
            "priorAuthorizationCommit": M5_P4_COMMIT, "priorAuthorizationTree": M5_P4_TREE,
            "priorAuthorizationPath": "benchmark/evidence/m5/run-authorization.json",
            "priorAuthorizationSha256": "d1fcdc2fab96873d3860abfeb71d1edd74b14bfb080b1feadd837ce8d4e011d3",
            "sourceCommit": "c" * 40, "sourceTree": "d" * 40, "sourcePathMap": paths,
            "sourcePublicCi": {
                "conclusion": "success", "event": "push", "headSha": "c" * 40,
                "runId": 456, "status": "completed",
                "url": "https://github.com/baney75/prooflens/actions/runs/456",
                "workflowPath": ".github/workflows/quality.yml",
            },
            "authorizationPath": "benchmark/evidence/m5/runtime-recovery-authorization.json",
            "scoreBlind": True, "h3PixelsRead": False,
        }
        arguments = {
            "protocol_commit": M5_P2_COMMIT, "protocol_tree": M5_P2_TREE,
            "prior_authorization_commit": M5_P4_COMMIT, "prior_authorization_tree": M5_P4_TREE,
            "prior_authorization_sha256": receipt["priorAuthorizationSha256"],
            "source_commit": receipt["sourceCommit"], "source_tree": receipt["sourceTree"],
            "source_path_map": paths,
        }
        validate_runtime_recovery_authorization(receipt, **arguments)
        for key in ("priorAuthorizationCommit", "priorAuthorizationSha256", "sourceCommit", "authorizationPath"):
            broken = copy.deepcopy(receipt)
            broken[key] = "e" * (64 if key.endswith("Sha256") else 40)
            with self.assertRaises(ValueError):
                validate_runtime_recovery_authorization(broken, **arguments)

    def test_runpod_environment_authorization_binds_prior_runtime_authorization(self) -> None:
        paths = [{"path": "benchmark/m5/train_gpu.py", "sha256": "a" * 64}]
        receipt = {
            "schemaVersion": 3, "status": "m5-runpod-environment-recovery-authorized",
            "protocolCommit": M5_P2_COMMIT, "protocolTree": M5_P2_TREE,
            "priorAuthorizationCommit": M5_RUNTIME_AUTHORIZATION_COMMIT,
            "priorAuthorizationTree": M5_RUNTIME_AUTHORIZATION_TREE,
            "priorAuthorizationPath": "benchmark/evidence/m5/runtime-recovery-authorization.json",
            "priorAuthorizationSha256": "eeee532d699705faef1e35d49748c8264387880e0e573d2b8c61e412944c9ce9",
            "sourceCommit": "c" * 40, "sourceTree": "d" * 40, "sourcePathMap": paths,
            "sourcePublicCi": {
                "conclusion": "success", "event": "push", "headSha": "c" * 40,
                "runId": 789, "status": "completed",
                "url": "https://github.com/baney75/prooflens/actions/runs/789",
                "workflowPath": ".github/workflows/quality.yml",
            },
            "environmentBoundary": "validated-single-runpod-pod-id-from-pid1-environ-no-other-record-forwarded",
            "authorizationPath": "benchmark/evidence/m5/runpod-environment-authorization.json",
            "scoreBlind": True, "h3PixelsRead": False,
        }
        arguments = {
            "protocol_commit": M5_P2_COMMIT, "protocol_tree": M5_P2_TREE,
            "prior_authorization_commit": M5_RUNTIME_AUTHORIZATION_COMMIT,
            "prior_authorization_tree": M5_RUNTIME_AUTHORIZATION_TREE,
            "prior_authorization_sha256": receipt["priorAuthorizationSha256"],
            "source_commit": receipt["sourceCommit"], "source_tree": receipt["sourceTree"],
            "source_path_map": paths,
        }
        validate_runpod_environment_authorization(receipt, **arguments)
        for key in ("priorAuthorizationCommit", "priorAuthorizationSha256", "sourceCommit", "authorizationPath", "environmentBoundary"):
            broken = copy.deepcopy(receipt)
            broken[key] = "e" * (64 if key.endswith("Sha256") else 40)
            with self.assertRaises(ValueError):
                validate_runpod_environment_authorization(broken, **arguments)

    def test_numeric_audit_authorization_binds_prior_runpod_environment_authorization(self) -> None:
        paths = [{"path": "benchmark/m5/train_gpu.py", "sha256": "a" * 64}]
        receipt = {
            "schemaVersion": 4, "status": "m5-numeric-audit-recovery-authorized",
            "protocolCommit": M5_P2_COMMIT, "protocolTree": M5_P2_TREE,
            "priorAuthorizationCommit": M5_RUNPOD_ENV_AUTHORIZATION_COMMIT,
            "priorAuthorizationTree": M5_RUNPOD_ENV_AUTHORIZATION_TREE,
            "priorAuthorizationPath": "benchmark/evidence/m5/runpod-environment-authorization.json",
            "priorAuthorizationSha256": "031f8b85dca9362d7afe06bcd30dc400f9f76a82a314012259b4300b237a8662",
            "sourceCommit": "c" * 40, "sourceTree": "d" * 40, "sourcePathMap": paths,
            "sourcePublicCi": {
                "conclusion": "success", "event": "push", "headSha": "c" * 40,
                "runId": 987, "status": "completed",
                "url": "https://github.com/baney75/prooflens/actions/runs/987",
                "workflowPath": ".github/workflows/quality.yml",
            },
            "numericBoundary": "source-balanced-weights-unchanged-math-fsum-audit-only",
            "authorizationPath": "benchmark/evidence/m5/numeric-audit-authorization.json",
            "scoreBlind": True, "h3PixelsRead": False,
        }
        arguments = {
            "protocol_commit": M5_P2_COMMIT, "protocol_tree": M5_P2_TREE,
            "prior_authorization_commit": M5_RUNPOD_ENV_AUTHORIZATION_COMMIT,
            "prior_authorization_tree": M5_RUNPOD_ENV_AUTHORIZATION_TREE,
            "prior_authorization_sha256": receipt["priorAuthorizationSha256"],
            "source_commit": receipt["sourceCommit"], "source_tree": receipt["sourceTree"],
            "source_path_map": paths,
        }
        validate_numeric_audit_authorization(receipt, **arguments)
        for key in ("priorAuthorizationCommit", "priorAuthorizationSha256", "sourceCommit", "authorizationPath", "numericBoundary"):
            broken = copy.deepcopy(receipt)
            broken[key] = "e" * (64 if key.endswith("Sha256") else 40)
            with self.assertRaises(ValueError):
                validate_numeric_audit_authorization(broken, **arguments)

    def test_parity_recovery_authorization_binds_a4_and_r5(self) -> None:
        paths = [{"path": "benchmark/m5/train_gpu.py", "sha256": "a" * 64}]
        receipt = {
            "schemaVersion": 5, "status": "m5-parity-recovery-authorized",
            "protocolCommit": M5_P2_COMMIT, "protocolTree": M5_P2_TREE,
            "priorAuthorizationCommit": "f3d86077cf5e7a124d09b593d69e9a1769d7e295",
            "priorAuthorizationTree": "d93819ff013943ac48c6dafc659effc9cfbf3e95",
            "priorAuthorizationPath": "benchmark/evidence/m5/numeric-audit-authorization.json",
            "priorAuthorizationSha256": "8286dc24babe83a16fdf898fa5e70b6202a1da8c46ae2aeda8cf557134db0f03",
            "sourceCommit": "c" * 40, "sourceTree": "d" * 40, "sourcePathMap": paths,
            "sourcePublicCi": {
                "conclusion": "success", "event": "push", "headSha": "c" * 40,
                "runId": 988, "status": "completed",
                "url": "https://github.com/baney75/prooflens/actions/runs/988",
                "workflowPath": ".github/workflows/quality.yml",
            },
            "parityBoundary": "packaged-m2-reference-preserved-real-input-parity-and-onnx-scoring",
            "diagnosticSha256": "c9c673efa0b1a6e4ea79b195ec16c71ae8ac91f962390a49c4e570b6d8de5c11",
            "authorizationPath": "benchmark/evidence/m5/parity-recovery-authorization.json",
            "scoreBlind": True, "h3PixelsRead": False,
        }
        arguments = {
            "protocol_commit": M5_P2_COMMIT, "protocol_tree": M5_P2_TREE,
            "prior_authorization_commit": receipt["priorAuthorizationCommit"],
            "prior_authorization_tree": receipt["priorAuthorizationTree"],
            "prior_authorization_sha256": receipt["priorAuthorizationSha256"],
            "source_commit": receipt["sourceCommit"], "source_tree": receipt["sourceTree"],
            "source_path_map": paths,
        }
        validate_parity_recovery_authorization(receipt, **arguments)
        for key in (
            "priorAuthorizationCommit", "priorAuthorizationSha256", "sourceCommit",
            "authorizationPath", "parityBoundary", "diagnosticSha256", "scoreBlind",
        ):
            broken = copy.deepcopy(receipt)
            broken[key] = False if key == "scoreBlind" else "e" * (64 if key.endswith("Sha256") else 40)
            with self.assertRaises(ValueError):
                validate_parity_recovery_authorization(broken, **arguments)

    def setUp(self) -> None:
        self.recipe = load_recipe(ROOT / "benchmark/m5/recipe.json")

    def test_runpod_node_runtime_lock_and_archive_paths(self) -> None:
        bootstrap = runpy.run_path(ROOT / "scripts/m5_node_bootstrap.py")
        lock = bootstrap["runtime_lock"]()
        self.assertEqual(lock["version"], "v24.18.1")
        self.assertEqual(lock["npmVersion"], "11.16.0")
        self.assertEqual(lock["archive"]["bytes"], 31_525_884)
        self.assertEqual(lock["archive"]["sha256"], "d6c664df3f3f61458e8c277585571328522d705166723a7c7823a9253a4d15a0")
        self.assertEqual(lock["nodeSha256"], "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a")
        safe = tarfile.TarInfo("node-v24.18.1-linux-x64/bin/npm")
        safe.type = tarfile.SYMTYPE
        safe.linkname = "../lib/node_modules/npm/bin/npm-cli.js"
        bootstrap["validate_member"](safe)
        unsafe = tarfile.TarInfo("node-v24.18.1-linux-x64/bin/escape")
        unsafe.type = tarfile.SYMTYPE
        unsafe.linkname = "../../outside"
        with self.assertRaises(ValueError):
            bootstrap["validate_member"](unsafe)

    def test_runpod_bootstrap_forwards_only_exact_pid1_pod_id(self) -> None:
        bootstrap = runpy.run_path(ROOT / "scripts/m5-preexec-bootstrap.py")
        parse = bootstrap["parse_runpod_pod_id"]
        pod_id = "pod_123-abc"
        secret = "never-forward-this-api-key"
        payload = f"RUNPOD_API_KEY={secret}\0RUNPOD_POD_ID={pod_id}\0AWS_SECRET_ACCESS_KEY=neighbor\0".encode()
        self.assertEqual(parse(payload), pod_id)
        cases = [
            b"RUNPOD_API_KEY=secret\0",
            b"RUNPOD_POD_ID=one\0RUNPOD_POD_ID=two\0",
            b"RUNPOD_POD_ID=\0",
            b"RUNPOD_POD_ID=has space\0",
            b"RUNPOD_POD_ID=bad\xff\0",
            b"RUNPOD_POD_ID=unterminated",
            b"X" * (1024 * 1024 + 1),
        ]
        for broken in cases:
            with self.subTest(broken=broken[:40]):
                with self.assertRaises(Exception) as raised:
                    parse(broken)
                self.assertNotIn(secret, str(raised.exception))
                self.assertNotIn(pod_id, str(raised.exception))

        function_globals = bootstrap["runpod_pod_id_from_init"].__globals__
        original_path = function_globals["Path"]

        class FakeProcPath:
            def open(self, mode: str) -> io.BytesIO:
                self_mode = mode
                if self_mode != "rb":
                    raise AssertionError(self_mode)
                return io.BytesIO(payload)

        try:
            function_globals["Path"] = lambda _value: FakeProcPath()
            with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": pod_id}, clear=False):
                self.assertEqual(bootstrap["runpod_pod_id_from_init"](), pod_id)
            with mock.patch.dict(os.environ, {"RUNPOD_POD_ID": "different"}, clear=False):
                with self.assertRaisesRegex(ValueError, "differ"):
                    bootstrap["runpod_pod_id_from_init"]()
        finally:
            function_globals["Path"] = original_path

        original_reader = function_globals["runpod_pod_id_from_init"]
        try:
            function_globals["runpod_pod_id_from_init"] = lambda: pod_id
            environment = bootstrap["runpod_environment"]()
        finally:
            function_globals["runpod_pod_id_from_init"] = original_reader
        self.assertEqual(environment["RUNPOD_POD_ID"], pod_id)
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
        self.assertNotIn("RUNPOD_API_KEY", environment)
        self.assertFalse(any(key.startswith("AWS_") for key in environment))
        source = (ROOT / "scripts/m5-preexec-bootstrap.py").read_text()
        self.assertLess(source.index('environment["CUBLAS_WORKSPACE_CONFIG"]'), source.index('os.execve("/bin/bash"'))
        self.assertLess(source.index('environment["CUBLAS_WORKSPACE_CONFIG"]'), source.index('"scripts/m5-runpod-launch.sh"'))
        try:
            function_globals["runpod_pod_id_from_init"] = lambda: pod_id
            with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": "caller-override"}, clear=False):
                environment = bootstrap["runpod_environment"]()
        finally:
            function_globals["runpod_pod_id_from_init"] = original_reader
        self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_cublas_recovery_authorization_is_score_blind_and_exact(self) -> None:
        paths = [{"path": "scripts/m5-preexec-bootstrap.py", "sha256": "a" * 64}]
        receipt = {
            "schemaVersion": 6, "status": "m5-cublas-recovery-authorized",
            "protocolCommit": M5_P2_COMMIT, "protocolTree": M5_P2_TREE,
            "priorAuthorizationCommit": "c" * 40, "priorAuthorizationTree": "d" * 40,
            "priorAuthorizationPath": "benchmark/evidence/m5/parity-recovery-authorization.json",
            "priorAuthorizationSha256": "e" * 64, "sourceCommit": "f" * 40, "sourceTree": "a" * 40,
            "sourcePathMap": paths,
            "runtimeBoundary": "trusted-runpod-execution-child-environment-before-torch-import",
            "cublasWorkspaceConfig": ":4096:8",
            "sourcePublicCi": {"conclusion": "success", "event": "push", "headSha": "f" * 40,
                               "runId": 1234, "status": "completed",
                               "url": "https://github.com/baney75/prooflens/actions/runs/1234",
                               "workflowPath": ".github/workflows/quality.yml"},
            "authorizationPath": "benchmark/evidence/m5/cublas-recovery-authorization.json",
            "scoreBlind": True, "h3PixelsRead": False,
        }
        kwargs = {"protocol_commit": M5_P2_COMMIT, "protocol_tree": M5_P2_TREE,
                  "prior_authorization_commit": receipt["priorAuthorizationCommit"],
                  "prior_authorization_tree": receipt["priorAuthorizationTree"],
                  "prior_authorization_sha256": receipt["priorAuthorizationSha256"],
                  "source_commit": receipt["sourceCommit"], "source_tree": receipt["sourceTree"],
                  "source_path_map": paths}
        validate_cublas_recovery_authorization(receipt, **kwargs)
        broken = copy.deepcopy(receipt)
        broken["cublasWorkspaceConfig"] = ":16:8"
        with self.assertRaises(ValueError):
            validate_cublas_recovery_authorization(broken, **kwargs)

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
            lambda value: value["training"].__setitem__("maximumPaidWallClockSeconds", 86_401),
            lambda value: value["training"].__setitem__("maximumPaidWallClockSeconds", 28_800),
            lambda value: value["training"].__setitem__("providerSignedAttestation", True),
            lambda value: value["selection"]["falsePositiveConfidence"].__setitem__("poolAcrossVariants", True),
            lambda value: value["selection"]["falsePositiveConfidence"].__setitem__("trialsPerVariant", 1_200),
            lambda value: value["largeSyntheticEvaluation"]["scoreBlindnessEvidence"].__setitem__("privatePriorScoringAbsenceProven", True),
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

    def test_source_balanced_audit_uses_stable_sum_without_changing_weights(self) -> None:
        rows = read_jsonl(ROOT / "benchmark/evidence/m4/train-manifest.jsonl.gz")
        with mock.patch("benchmark.m5.contracts.math.fsum", wraps=math.fsum) as stable_sum:
            weights = source_balanced_weights(rows)
        self.assertGreaterEqual(stable_sum.call_count, 4)
        indexes = [
            index for index, row in enumerate(rows)
            if row["label"] == 0 and row["source"] == "open-images-train"
        ]
        self.assertEqual(len(rows), 112_562)
        self.assertEqual(len(indexes), 50_000)
        expected_weight = 112_562 / (2.0 * 6 * 50_000)
        self.assertTrue(all(weights[index] == expected_weight for index in indexes))
        expected_mass = 112_562 / (2.0 * 6)
        sequential = 0.0
        for index in indexes:
            sequential += weights[index]
        self.assertGreater(abs(sequential - expected_mass), 1e-8)
        self.assertLessEqual(abs(math.fsum(weights[index] for index in indexes) - expected_mass), 1e-8)

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
        self.assertTrue(all(value.false_positive_trials == 300 for value in metrics.values()))
        self.assertTrue(all(value.false_positive_rate == 0.0 for value in metrics.values()))
        self.assertTrue(all(value.false_positive_wilson95 == {"lower": 0.0, "upper": 0.012642971224546027} for value in metrics.values()))

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
            "launchNodeVersion": "v24.18.1",
            "launchNodeSha256": "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a",
            "runpodPodIdSha256": "a" * 64,
            "provisioningReceiptSha256": "b" * 64,
            "containerImage": self.recipe["training"]["containerImage"],
            "requirementsSha256": self.recipe["training"]["requirementsSha256"],
            "providerEvidenceBoundary": "operator-recorded-not-cryptographic-attestation",
            "providerIdentityEvidence": "operator-attested-control-plane-observation",
            "providerSignedAttestation": False,
            "runtimeConsistencyEvidence": "RUNPOD_POD_ID hash and locally observed GPU match the operator-authored receipt",
            "cublasWorkspaceConfig": ":4096:8",
            "sourceCommit": "a" * 40,
            "sourceTree": "c" * 40,
            "authorizationCommit": "d" * 40,
            "authorizationReceiptSha256": "e" * 64,
            "authorizationPublicCi": {
                "conclusion": "success", "event": "push", "headSha": "d" * 40, "runId": 456,
                "status": "completed", "url": "https://github.com/baney75/prooflens/actions/runs/456",
                "workflowPath": ".github/workflows/quality.yml",
            },
        }
        validate_environment_receipt(receipt, self.recipe)
        with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, clear=False):
            require_cuda_determinism_environment(self.recipe)
        with mock.patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}, clear=False):
            with self.assertRaises(ValueError):
                require_cuda_determinism_environment(self.recipe)
        source = (ROOT / "benchmark/m5/train_gpu.py").read_text(encoding="utf-8")
        self.assertLess(source.index("require_cuda_determinism_environment(recipe)"), source.index("    import torch\n", source.index("def execute")))
        for key, value in (("launchNodeVersion", "v24.18.0"), ("launchNodeSha256", "0" * 64)):
            broken = dict(receipt)
            broken[key] = value
            with self.assertRaises(ValueError):
                validate_environment_receipt(broken, self.recipe)
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
            "maximumRuntimeSeconds": 86_400,
            "workloadStopAtUnix": 1_086_400,
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
            expired.workload_stop = 1_086_400
            expired.deadline = 1_086_100
            expired.maximum = 86_400
            expired.state_path = state
            expired.clock = lambda: 1_086_100.0
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
            export_onnx(TinyClassifier(), destination, self.recipe, providers=["CPUExecutionProvider"])
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
            "launchNodeVersion": "v24.18.1",
            "launchNodeSha256": "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a",
            "runpodPodIdSha256": "a" * 64,
            "provisioningReceiptSha256": "b" * 64,
            "containerImage": self.recipe["training"]["containerImage"],
            "requirementsSha256": self.recipe["training"]["requirementsSha256"],
            "providerEvidenceBoundary": "operator-recorded-not-cryptographic-attestation",
            "providerIdentityEvidence": "operator-attested-control-plane-observation",
            "providerSignedAttestation": False,
            "runtimeConsistencyEvidence": "RUNPOD_POD_ID hash and locally observed GPU match the operator-authored receipt",
            "cublasWorkspaceConfig": ":4096:8",
            "sourceCommit": "a" * 40,
            "sourceTree": "c" * 40,
            "authorizationCommit": "d" * 40,
            "authorizationReceiptSha256": "e" * 64,
            "authorizationPublicCi": {
                "conclusion": "success", "event": "push", "headSha": "d" * 40, "runId": 456,
                "status": "completed", "url": "https://github.com/baney75/prooflens/actions/runs/456",
                "workflowPath": ".github/workflows/quality.yml",
            },
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
            "scoreBlindness": self.recipe["largeSyntheticEvaluation"]["scoreBlindnessEvidence"],
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
        arguments = training_parser().parse_args(["--preflight-only"])
        self.assertTrue(arguments.preflight_only)
        self.assertFalse(hasattr(arguments, "protocol_commit"))
        self.assertFalse(hasattr(arguments, "h3_manifest"))

    def test_protocol_commit_is_git_derived_and_exact(self) -> None:
        head = "c" * 40

        def fake_run(command: list[str], *, cwd: Path = ROOT) -> str:
            del cwd
            if command == ["git", "rev-parse", "HEAD"]:
                return head
            if command == ["git", "rev-list", "--parents", "-n", "1", head]:
                return f"{head} {M5_ORIGINAL_PROTOCOL_COMMIT}"
            if command == ["git", "rev-parse", f"{M5_ORIGINAL_PROTOCOL_COMMIT}^{{tree}}"]:
                return M5_ORIGINAL_PROTOCOL_TREE
            if command == ["git", "rev-list", "--parents", "-n", "1", M5_ORIGINAL_PROTOCOL_COMMIT]:
                return f"{M5_ORIGINAL_PROTOCOL_COMMIT} {M5_BASE_SOURCE_COMMIT}"
            if command == ["git", "rev-parse", f"{M5_BASE_SOURCE_COMMIT}^{{tree}}"]:
                return M5_BASE_SOURCE_TREE
            if command == ["git", "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", head]:
                return "\n".join(f"M\t{path}" for path in sorted(M5_PROTOCOL_RECOVERY_PATHS))
            raise AssertionError(command)

        with mock.patch("benchmark.m5.train_gpu.run", side_effect=fake_run), \
             mock.patch("benchmark.m5.train_gpu.assert_worktree_exact"):
            self.assertEqual(resolve_authorized_protocol_commit(), head)

        def wrong_parent(command: list[str], *, cwd: Path = ROOT) -> str:
            if command == ["git", "rev-list", "--parents", "-n", "1", head]:
                return f"{head} {'0' * 40}"
            return fake_run(command, cwd=cwd)

        with mock.patch("benchmark.m5.train_gpu.run", side_effect=wrong_parent), \
             mock.patch("benchmark.m5.train_gpu.assert_worktree_exact"):
            with self.assertRaisesRegex(ValueError, "append-only"):
                resolve_authorized_protocol_commit()

        with mock.patch("benchmark.m5.train_gpu.run", side_effect=fake_run), \
             mock.patch("benchmark.m5.train_gpu.assert_worktree_exact", side_effect=ValueError("exact-worktree untracked surface")):
            with self.assertRaisesRegex(ValueError, "untracked surface"):
                resolve_authorized_protocol_commit()

    def test_source_recovery_history_is_append_only_and_exact(self) -> None:
        source = "c" * 40

        def rows(payload: dict[str, str]) -> str:
            return "\n".join(f"{status}\t{path}" for path, status in sorted(payload.items()))

        def fake_run(command: list[str], *, cwd: Path = ROOT) -> str:
            del cwd
            if command == ["git", "rev-list", "--parents", "-n", "1", source]:
                return f"{source} {M5_A4_COMMIT}"
            if command == ["git", "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", source]:
                return rows(M5_R5_ROWS)
            if command == ["git", "rev-parse", f"{M5_A4_COMMIT}^{{tree}}"]:
                return M5_A4_TREE
            if command == ["git", "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", M5_A4_COMMIT]:
                return "A\tbenchmark/evidence/m5/numeric-audit-authorization.json"
            raise AssertionError(command)

        with mock.patch("benchmark.m5.train_gpu.run", side_effect=fake_run):
            validate_source_recovery_history(source)

        def skipped_prior_authorization(command: list[str], *, cwd: Path = ROOT) -> str:
            if command == ["git", "rev-list", "--parents", "-n", "1", source]:
                return f"{source} {M5_P2_COMMIT}"
            return fake_run(command, cwd=cwd)

        with mock.patch("benchmark.m5.train_gpu.run", side_effect=skipped_prior_authorization):
            with self.assertRaisesRegex(ValueError, "A4 -> R5"):
                validate_source_recovery_history(source)

        def wrong_a4_tree(command: list[str], *, cwd: Path = ROOT) -> str:
            if command == ["git", "rev-parse", f"{M5_A4_COMMIT}^{{tree}}"]:
                return "0" * 40
            return fake_run(command, cwd=cwd)

        with mock.patch("benchmark.m5.train_gpu.run", side_effect=wrong_a4_tree):
            with self.assertRaisesRegex(ValueError, "A4 tree"):
                validate_source_recovery_history(source)

    def test_python_history_map_matches_literal_failed_p3_commit(self) -> None:
        output = subprocess.check_output([
            "/usr/bin/git", "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r",
            M5_FAILED_SOURCE_COMMIT,
        ], cwd=ROOT, text=True)
        actual = {}
        for line in output.splitlines():
            status, path = line.split("\t", maxsplit=1)
            actual[path] = status
        self.assertEqual(len(actual), 26)
        self.assertEqual(actual.get("scripts/m5-preexec-bootstrap.py"), "A")
        self.assertEqual(actual, M5_SOURCE_RECOVERY_ROWS)

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
                "parityProviderOptions": {"use_tf32": "0"},
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
                        "falsePositiveTrials": value.false_positive_trials,
                        "falsePositiveRate": value.false_positive_rate,
                        "falsePositiveWilson95": value.false_positive_wilson95,
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
        winner = candidates[0]
        grid_sha256 = sha256(canonical_json(grid)).hexdigest()
        summary = training_summary_fixture(
            self.recipe,
            candidate_grid_sha256=grid_sha256,
            status="selector-pass",
            selected_candidate_id=winner["candidateId"],
        )
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
            "candidateGridSha256": grid_sha256,
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
        for key, replacement in (
            ("providerIdentityEvidence", "provider-signed"),
            ("providerSignedAttestation", True),
            ("runtimeConsistencyEvidence", "unbound"),
            ("cublasWorkspaceConfig", ":16:8"),
        ):
            broken_environment = copy.deepcopy(lock)
            broken_environment["trainingSummary"]["environment"][key] = replacement
            broken_environment["trainingSummarySha256"] = sha256(
                canonical_json(broken_environment["trainingSummary"])
            ).hexdigest()
            with self.subTest(provider_field=key), self.assertRaises(ValueError):
                validate_selection_lock(broken_environment, self.recipe, rows)
        broken = copy.deepcopy(lock)
        broken["candidateGrid"]["candidates"][0]["selectorLogits"]["original"]["base64"] = "AAAA"
        broken["candidateGridSha256"] = sha256(canonical_json(broken["candidateGrid"])).hexdigest()
        broken["trainingSummary"]["candidateGrid"]["sha256"] = broken["candidateGridSha256"]
        broken["trainingSummarySha256"] = sha256(canonical_json(broken["trainingSummary"])).hexdigest()
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
                "parityProviderOptions": {"use_tf32": "0"},
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
        grid_sha256 = sha256(canonical_json(grid)).hexdigest()
        summary = training_summary_fixture(
            self.recipe,
            candidate_grid_sha256=grid_sha256,
            status="selector-fail",
            selected_candidate_id=None,
        )
        receipt = {
            "schemaVersion": 1,
            "status": "failed-m5-selector",
            "acceptanceEligible": False,
            "recipeSha256": recipe_sha,
            "protocolCommit": "a" * 40,
            "trainingSummary": summary,
            "trainingSummarySha256": sha256(canonical_json(summary)).hexdigest(),
            "candidateGrid": grid,
            "candidateGridSha256": grid_sha256,
            "h3PixelsRead": False,
            "terminalRegressionsRead": False,
            "reason": "No predeclared candidate and exhaustive raw threshold passed every fresh-selector gate.",
        }
        # Failure receipts are published as canonical JSON, which sorts object
        # keys.  Replay must follow the frozen VARIANTS tuple rather than the
        # serialized mapping order (the old tuple(selectorLogits) check rejected
        # this valid canonical round-trip).
        canonical_receipt = parse_json_bytes(canonical_json(receipt), label="canonical failure receipt")
        validate_failure_receipt(canonical_receipt, self.recipe, rows)
        for mutation in ("missing", "extra"):
            broken_keys = copy.deepcopy(canonical_receipt)
            selector_logits = broken_keys["candidateGrid"]["candidates"][0]["selectorLogits"]
            if mutation == "missing":
                del selector_logits[VARIANTS[-1]]
            else:
                selector_logits["unexpected"] = selector_logits[VARIANTS[0]]
            broken_keys["candidateGridSha256"] = sha256(canonical_json(broken_keys["candidateGrid"])).hexdigest()
            broken_keys["trainingSummary"]["candidateGrid"]["sha256"] = broken_keys["candidateGridSha256"]
            broken_keys["trainingSummarySha256"] = sha256(canonical_json(broken_keys["trainingSummary"])).hexdigest()
            with self.subTest(selector_logit_keys=mutation), self.assertRaises(ValueError):
                validate_failure_receipt(broken_keys, self.recipe, rows)
        broken = copy.deepcopy(receipt)
        broken["candidateGrid"]["candidates"][0]["accepted"] = True
        broken["candidateGridSha256"] = sha256(canonical_json(broken["candidateGrid"])).hexdigest()
        broken["trainingSummary"]["candidateGrid"]["sha256"] = broken["candidateGridSha256"]
        broken["trainingSummarySha256"] = sha256(canonical_json(broken["trainingSummary"])).hexdigest()
        with self.assertRaises(ValueError):
            validate_failure_receipt(broken, self.recipe, rows)
        broken_reason = copy.deepcopy(canonical_receipt)
        broken_reason["reason"] = "A different or contradictory failure claim."
        with self.assertRaises(ValueError):
            validate_failure_receipt(broken_reason, self.recipe, rows)


if __name__ == "__main__":
    unittest.main()
