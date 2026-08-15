"""Regression tests for cache, prediction, and sealed-output integrity."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import numpy as np
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation_contract import (  # noqa: E402
    require_anonymous_public_file,
    require_canonical_output_directory,
    require_canonical_repository_path,
    require_public_pre_score_freeze,
    require_successful_public_quality_run,
)
from evaluate import sigmoid  # noqa: E402
from feature_cache_contract import expected_view_metadata, validate_feature_arrays  # noqa: E402
from fresh_feature_run import (  # noqa: E402
    cache_belongs_to_fresh_feature_run,
    complete_fresh_feature_run,
    marker_sha256,
    open_or_create_fresh_feature_run,
)
import finalize_training_evidence as finalizer  # noqa: E402
from finalize_training_evidence import build_model_lock, publish_with_rollback  # noqa: E402
from prediction_contract import require_logit_probability_consistency, sigmoid_scalar  # noqa: E402


@dataclass(frozen=True)
class Item:
    label: int
    source: str


class IntegrityContractsTest(unittest.TestCase):
    @staticmethod
    def _git(repository: Path, *arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repository, text=True
        ).strip()

    def _build_recovery_freeze_chain(
        self,
        root: Path,
        *,
        receipt_immutable_mode: str = "exact",
        receipt_allowed_paths: list[str] | None = None,
        declared_failed_repair_paths: list[str] | None = None,
        declared_repair_paths: list[str] | None = None,
    ) -> dict[str, object]:
        repository = root / "repository"
        remote = root / "remote.git"
        repository.mkdir()
        subprocess.run(["git", "init", "--bare", remote], check=True, capture_output=True)
        subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "ProofLens test"], cwd=repository, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True)
        (repository / ".github/workflows").mkdir(parents=True)
        (repository / ".github/workflows/quality.yml").write_text("name: quality\n")
        (repository / "a2.txt").write_text("a2 pending\n")
        (repository / "a3.txt").write_text("a3 pending\n")
        (repository / "a4.txt").write_text("a4 pending\n")
        (repository / "receipt-seed.json").write_text("{}\n")
        subprocess.run(
            ["git", "add", ".github", "a2.txt", "a3.txt", "a4.txt", "receipt-seed.json"],
            cwd=repository,
            check=True,
        )
        subprocess.run(["git", "commit", "-m", "Legacy source"], cwd=repository,
                       check=True, capture_output=True)
        legacy_source = self._git(repository, "rev-parse", "HEAD")
        legacy_path = Path("benchmark/evidence/evaluation/pre-score-freeze.json")
        legacy_receipt = repository / legacy_path
        legacy_receipt.parent.mkdir(parents=True)
        legacy_receipt.write_text(json.dumps({"schemaVersion": 2, "sourceCommit": legacy_source}) + "\n")
        legacy_hash = sha256(legacy_receipt.read_bytes()).hexdigest()
        subprocess.run(["git", "add", str(legacy_path)], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Failed legacy freeze"], cwd=repository,
                       check=True, capture_output=True)
        legacy_freeze = self._git(repository, "rev-parse", "HEAD")
        legacy_freeze_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        (repository / "a2.txt").write_text("failed recovery\n")
        subprocess.run(["git", "add", "a2.txt"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Failed recovery source"], cwd=repository,
                       check=True, capture_output=True)
        failed_recovery_source = self._git(repository, "rev-parse", "HEAD")
        failed_recovery_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        (repository / "a3.txt").write_text("green second recovery\n")
        subprocess.run(["git", "add", "a3.txt"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Second recovery source"], cwd=repository,
                       check=True, capture_output=True)
        second_recovery_source = self._git(repository, "rev-parse", "HEAD")
        second_recovery_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        second_path = Path("benchmark/evidence/evaluation/pre-score-freeze-v2.json")
        second_receipt = repository / second_path
        second_receipt.write_text(json.dumps({"schemaVersion": 3, "sourceCommit": second_recovery_source}) + "\n")
        second_hash = sha256(second_receipt.read_bytes()).hexdigest()
        subprocess.run(["git", "add", str(second_path)], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Second freeze"], cwd=repository,
                       check=True, capture_output=True)
        second_freeze = self._git(repository, "rev-parse", "HEAD")
        second_freeze_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        failed_path = Path("failed.txt")
        (repository / failed_path).write_text("failed numeric evidence\n")
        subprocess.run(["git", "add", str(failed_path)], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Failed evidence"], cwd=repository,
                       check=True, capture_output=True)
        failed_evaluation = self._git(repository, "rev-parse", "HEAD")
        failed_evaluation_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        selection_path = Path("selection.txt")
        (repository / selection_path).write_text("score-blind replacement\n")
        subprocess.run(["git", "add", str(selection_path)], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Replacement selection"], cwd=repository,
                       check=True, capture_output=True)
        replacement_selection = self._git(repository, "rev-parse", "HEAD")
        replacement_selection_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        (repository / "a4.txt").write_text("binary64 numeric recovery\n")
        subprocess.run(["git", "add", "a4.txt"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Numeric recovery source"], cwd=repository,
                       check=True, capture_output=True)
        recovery_source = self._git(repository, "rev-parse", "HEAD")
        recovery_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        workflow_hash = sha256((repository / ".github/workflows/quality.yml").read_bytes()).hexdigest()
        source_hash = sha256((repository / "a4.txt").read_bytes()).hexdigest()
        expected_allowed_paths = ["benchmark/evidence/evaluation/confirmatory-v2/**", "README.md"]
        failed_repair_paths = ["a2.txt"] if declared_failed_repair_paths is None else declared_failed_repair_paths
        second_repair_paths = ["a3.txt"]
        repair_paths = ["a4.txt"] if declared_repair_paths is None else declared_repair_paths
        if receipt_immutable_mode == "exact":
            receipt_immutable_files = {"a4.txt": source_hash}
        elif receipt_immutable_mode == "missing":
            receipt_immutable_files = {}
        elif receipt_immutable_mode == "extra":
            receipt_immutable_files = {"a4.txt": source_hash, "extra.txt": "0" * 64}
        else:  # pragma: no cover - test helper misuse
            raise ValueError(f"Unknown immutable test mode: {receipt_immutable_mode}")
        freeze_path = Path("benchmark/evidence/evaluation/pre-score-freeze-v3.json")
        freeze_file = repository / freeze_path
        freeze_file.write_text(json.dumps({
            "schemaVersion": 4,
            "generation": 3,
            "mode": "public replacement-v2 pre-score source freeze before any replacement confirmatory or web-negative inference",
            "receiptPath": str(freeze_path),
            "repository": "https://github.com/baney75/prooflens",
            "branch": "main",
            "sourceCommit": recovery_source,
            "sourceTree": recovery_tree,
            "remoteObservedHead": recovery_source,
            "publicCommitUrl": f"https://github.com/baney75/prooflens/commit/{recovery_source}",
            "sourcePublicCiProof": {
                "workflowName": "quality", "workflowPath": ".github/workflows/quality.yml",
                "event": "push", "headSha": recovery_source, "status": "completed",
                "conclusion": "success", "runId": 900, "runUrl": "https://example.invalid/900",
                "workflowFileSha256": workflow_hash,
            },
            "allowedPostScorePaths": (
                expected_allowed_paths if receipt_allowed_paths is None else receipt_allowed_paths
            ),
            "immutableFilesSha256": receipt_immutable_files,
            "replacementProtocol": {"fixture": "replacement-v2"},
            "lineage": {
                "reason": "test-resource-exhaustion",
                "legacySourceCommit": legacy_source,
                "legacySourceTree": self._git(repository, "rev-parse", f"{legacy_source}^{{tree}}"),
                "legacySourceActionsRunId": 100,
                "legacySourceActionsConclusion": "success",
                "legacyFreezeCommit": legacy_freeze,
                "legacyFreezeTree": legacy_freeze_tree,
                "legacyFreezeReceiptPath": str(legacy_path),
                "legacyFreezeReceiptSha256": legacy_hash,
                "legacyFreezeActionsRunId": 101,
                "legacyFreezeActionsConclusion": "failure",
                "failedRecoverySourceCommit": failed_recovery_source,
                "failedRecoverySourceTree": failed_recovery_tree,
                "failedRecoveryActionsRunId": 12345,
                "failedRecoveryActionsRunUrl": "https://example.invalid/actions/runs/12345",
                "failedRecoveryActionsConclusion": "failure",
                "failedRecoveryReason": "test-missing-dependencies",
                "failedRecoveryRepairPaths": failed_repair_paths,
                "secondRecoverySourceCommit": second_recovery_source,
                "secondRecoverySourceTree": second_recovery_tree,
                "secondRecoveryActionsRunId": 102,
                "secondRecoveryActionsConclusion": "success",
                "secondRecoveryRepairPaths": second_repair_paths,
                "secondFreezeCommit": second_freeze,
                "secondFreezeTree": second_freeze_tree,
                "secondFreezeReceiptPath": str(second_path),
                "secondFreezeReceiptSha256": second_hash,
                "secondFreezeActionsRunId": 103,
                "secondFreezeActionsConclusion": "success",
                "failedEvaluationCommit": failed_evaluation,
                "failedEvaluationTree": failed_evaluation_tree,
                "failedEvaluationActionsRunId": 104,
                "failedEvaluationActionsConclusion": "failure",
                "failedEvaluationReason": "test-float32-sigmoid",
                "failedEvaluationPathsSha256": {
                    str(failed_path): sha256((repository / failed_path).read_bytes()).hexdigest(),
                },
                "replacementSelectionCommit": replacement_selection,
                "replacementSelectionTree": replacement_selection_tree,
                "replacementSelectionActionsRunId": 105,
                "replacementSelectionActionsConclusion": "failure",
                "replacementSelectionReason": "test-selection-ci-policy",
                "replacementSelectionPathsSha256": {
                    str(selection_path): sha256((repository / selection_path).read_bytes()).hexdigest(),
                },
                "numericRecoverySourceCommit": recovery_source,
                "numericRecoverySourceTree": recovery_tree,
                "numericRecoveryRepairPaths": repair_paths,
                "repositoryEvidenceLimitation": "Repository history cannot prove unrecorded pixel access.",
            },
        }, indent=2) + "\n")
        subprocess.run(["git", "add", str(freeze_path)], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Recovery freeze"], cwd=repository,
                       check=True, capture_output=True)
        freeze_commit = self._git(repository, "rev-parse", "HEAD")
        subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repository, check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repository,
                       check=True, capture_output=True)
        contract = {
            "freezePath": str(freeze_path),
            "secondFreezePath": str(second_path),
            "legacyFreezePath": str(legacy_path),
            "legacySourceCommit": legacy_source,
            "legacySourceTree": self._git(repository, "rev-parse", f"{legacy_source}^{{tree}}"),
            "legacySourceActionsRunId": 100,
            "legacyFreezeCommit": legacy_freeze,
            "legacyFreezeTree": legacy_freeze_tree,
            "legacyFreezeSha256": legacy_hash,
            "legacyFreezeActionsRunId": 101,
            "failedRecoverySourceCommit": failed_recovery_source,
            "failedRecoverySourceTree": failed_recovery_tree,
            "failedRecoveryActionsRunId": 12345,
            "failedRecoveryActionsRunUrl": "https://example.invalid/actions/runs/12345",
            "failedRecoveryReason": "test-missing-dependencies",
            "failedRecoveryRepairPaths": failed_repair_paths,
            "secondRecoverySourceCommit": second_recovery_source,
            "secondRecoverySourceTree": second_recovery_tree,
            "secondRecoveryActionsRunId": 102,
            "secondRecoveryRepairPaths": second_repair_paths,
            "secondFreezeCommit": second_freeze,
            "secondFreezeTree": second_freeze_tree,
            "secondFreezeSha256": second_hash,
            "secondFreezeActionsRunId": 103,
            "failedEvaluationCommit": failed_evaluation,
            "failedEvaluationTree": failed_evaluation_tree,
            "failedEvaluationActionsRunId": 104,
            "failedEvaluationReason": "test-float32-sigmoid",
            "failedEvaluationPaths": [str(failed_path)],
            "replacementSelectionCommit": replacement_selection,
            "replacementSelectionTree": replacement_selection_tree,
            "replacementSelectionActionsRunId": 105,
            "replacementSelectionReason": "test-selection-ci-policy",
            "replacementSelectionPaths": [str(selection_path)],
            "reason": "test-resource-exhaustion",
            "repairPaths": repair_paths,
            "replacementProtocol": {"fixture": "replacement-v2"},
            "replacementFilesSha256": {
                str(selection_path): sha256((repository / selection_path).read_bytes()).hexdigest(),
            },
            "immutableFiles": ["a4.txt"],
            "allowedPostScorePaths": expected_allowed_paths,
        }
        return {
            "repository": repository,
            "remote": remote,
            "freezeFile": freeze_file,
            "freezeBytes": freeze_file.read_bytes(),
            "freezeCommit": freeze_commit,
            "recoverySource": recovery_source,
            "failedRecoverySource": failed_recovery_source,
            "legacyFreeze": legacy_freeze,
            "contract": contract,
        }

    def test_cache_metadata_matches_exact_view_expansion_order(self) -> None:
        labels, variants, sources = expected_view_metadata(
            [Item(1, "single"), Item(0, "augmented")],
            training=True,
            single_view_sources=frozenset({"single"}),
        )
        self.assertEqual(labels, [1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(variants, [0, 0, 1, 2, 3])
        self.assertEqual(sources, ["single", "augmented", "augmented", "augmented", "augmented"])

    def test_cache_rejects_feature_dtypes_the_extractor_cannot_produce(self) -> None:
        arguments = {
            "labels": np.asarray([1.0], dtype=np.float32),
            "variants": np.asarray([0], dtype=np.int64),
            "sources": np.asarray(["source"]),
            "expected_labels": [1.0],
            "expected_variants": [0],
            "expected_sources": ["source"],
        }
        for dtype in (np.int64, np.complex128, np.float64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "shape or dtype"):
                validate_feature_arrays(np.zeros((1, 384), dtype=dtype), **arguments)
        validate_feature_arrays(np.zeros((1, 384), dtype=np.float32), **arguments)

    def test_fresh_feature_run_resumes_only_the_exact_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "fresh-feature-run.json"
            context = {"manifestSha256": "a" * 64, "pipelineVersion": 8}
            created = open_or_create_fresh_feature_run(
                marker,
                context,
                run_id="1" * 32,
            )
            resumed = open_or_create_fresh_feature_run(marker, context)
            self.assertEqual(resumed, created)
            completed = complete_fresh_feature_run(marker, context, created["runId"])
            self.assertEqual(completed["state"], "complete")
            self.assertEqual(marker_sha256(completed), sha256(marker.read_bytes()).hexdigest())

    def test_fresh_feature_run_rejects_stale_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "fresh-feature-run.json"
            open_or_create_fresh_feature_run(
                marker,
                {"manifestSha256": "a" * 64},
                run_id="2" * 32,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                open_or_create_fresh_feature_run(marker, {"manifestSha256": "b" * 64})

    def test_fresh_feature_cache_requires_the_exact_run_id(self) -> None:
        run_id = "3" * 32
        self.assertTrue(cache_belongs_to_fresh_feature_run(run_id, run_id))
        self.assertFalse(cache_belongs_to_fresh_feature_run(run_id, "4" * 32))
        self.assertFalse(cache_belongs_to_fresh_feature_run(None, run_id))

    def test_impossible_logit_probability_pair_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sigmoid"):
            require_logit_probability_consistency(999.0, 0.01)
        probability = sigmoid_scalar(-3.25)
        require_logit_probability_consistency(-3.25, probability)

    def test_float32_logits_use_binary64_sigmoid_serialization(self) -> None:
        logits = np.asarray([-8.523, -4.073137283325195, -0.9029163122177123, 0, 1, 8], dtype=np.float32)
        probabilities = sigmoid(logits)
        self.assertEqual(probabilities.dtype, np.float64)
        for logit, probability in zip(logits, probabilities, strict=True):
            expected = sigmoid_scalar(float(logit))
            self.assertLessEqual(abs(float(probability) - expected), 2e-12)
            require_logit_probability_consistency(float(logit), float(probability), decision_threshold=0.2884515632035137)
        old = np.empty_like(logits, dtype=np.float64)
        positive = logits >= 0
        old[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_value = np.exp(logits[~positive])
        old[~positive] = exp_value / (1.0 + exp_value)
        self.assertGreater(abs(float(old[1]) - sigmoid_scalar(float(logits[1]))), 2e-12)

    def test_probability_tolerance_cannot_cross_the_decision_boundary(self) -> None:
        threshold = 0.65
        logit = math.log(threshold / (1 - threshold))
        with self.assertRaisesRegex(ValueError, "decision boundary"):
            require_logit_probability_consistency(
                logit,
                threshold - 1e-13,
                decision_threshold=threshold,
            )

    def test_protocol_output_cannot_be_redirected_to_a_disposable_directory(self) -> None:
        repository_root = Path.cwd().resolve()
        canonical = repository_root / "benchmark/evidence/evaluation/confirmatory-v2"
        self.assertEqual(
            require_canonical_output_directory("confirmatory-v2", canonical, repository_root=repository_root),
            canonical,
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Canonical"):
                require_canonical_output_directory(
                    "confirmatory-v2", Path(temporary), repository_root=repository_root
                )

    def test_protocol_output_rejects_symlinked_canonical_component(self) -> None:
        with tempfile.TemporaryDirectory() as repository, tempfile.TemporaryDirectory() as disposable:
            repository_root = Path(repository)
            evaluation_root = repository_root / "benchmark/evidence/evaluation"
            evaluation_root.mkdir(parents=True)
            (evaluation_root / "confirmatory-v2").symlink_to(disposable, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                require_canonical_output_directory(
                    "confirmatory-v2",
                    evaluation_root / "confirmatory-v2",
                    repository_root=repository_root,
                )

    def test_canonical_model_path_rejects_a_symlink_component(self) -> None:
        with tempfile.TemporaryDirectory() as repository, tempfile.TemporaryDirectory() as disposable:
            repository_root = Path(repository)
            (repository_root / "weights").symlink_to(disposable, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                require_canonical_repository_path(
                    repository_root / "weights/prooflens-cf384.onnx",
                    Path("weights/prooflens-cf384.onnx"),
                    repository_root=repository_root,
                    label="model",
                )

    def test_nonfinite_prediction_values_are_rejected(self) -> None:
        for logit, probability in ((math.inf, 1.0), (0.0, math.nan), (0.0, 1.1)):
            with self.assertRaises(ValueError):
                require_logit_probability_consistency(logit, probability)

    def test_anonymous_public_file_rejects_an_authenticated_only_head(self) -> None:
        expected_head = "a" * 40
        with self.assertRaisesRegex(ValueError, "Anonymous public main"):
            require_anonymous_public_file(
                expected_head=expected_head,
                file_commit=expected_head,
                path=Path("model-lock.json"),
                expected_bytes=b"public\n",
                head_resolver=lambda: "b" * 40,
                byte_fetcher=lambda _url: b"public\n",
            )

    def test_sealed_inference_requires_immutable_public_freeze_only_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chain = self._build_recovery_freeze_chain(Path(temporary))
            repository = chain["repository"]
            remote = chain["remote"]
            freeze_path = chain["freezeFile"]
            freeze_commit = chain["freezeCommit"]
            committed_freeze = chain["freezeBytes"]
            observed = require_public_pre_score_freeze(
                repository_root=repository,
                canonical_origin_urls=frozenset({str(remote)}),
                anonymous_head_resolver=lambda: str(freeze_commit),
                anonymous_byte_fetcher=lambda _url: committed_freeze,
                recovery_contract=chain["contract"],
            )
            self.assertEqual(observed["freezeCommit"], freeze_commit)

            freeze_path.write_text(freeze_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed after"):
                require_public_pre_score_freeze(
                    repository_root=repository,
                    canonical_origin_urls=frozenset({str(remote)}),
                    recovery_contract=chain["contract"],
                )
            freeze_path.write_bytes(committed_freeze)
            (repository / "source.txt").write_text("changed after freeze\n")
            subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "Unpushed source descendant"],
                           cwd=repository, check=True, capture_output=True)
            with self.assertRaisesRegex(ValueError, "Local HEAD"):
                require_public_pre_score_freeze(
                    repository_root=repository,
                    canonical_origin_urls=frozenset({str(remote)}),
                    recovery_contract=chain["contract"],
                )
            with self.assertRaisesRegex(ValueError, "Public origin/main"):
                require_public_pre_score_freeze(
                    repository_root=repository,
                    allow_public_descendant=True,
                    canonical_origin_urls=frozenset({str(remote)}),
                    anonymous_head_resolver=lambda: str(freeze_commit),
                    anonymous_byte_fetcher=lambda _url: committed_freeze,
                    recovery_contract=chain["contract"],
                )

    def test_legacy_freeze_alone_cannot_authorize_sealed_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chain = self._build_recovery_freeze_chain(Path(temporary))
            repository = chain["repository"]
            freeze_file = chain["freezeFile"]
            freeze_bytes = chain["freezeBytes"]
            freeze_file.unlink()
            with self.assertRaises(FileNotFoundError):
                require_public_pre_score_freeze(
                    repository_root=repository,
                    recovery_contract=chain["contract"],
                )
            freeze_file.write_bytes(freeze_bytes)

    def test_replacement_inference_requires_green_exact_head_public_ci(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chain = self._build_recovery_freeze_chain(Path(temporary))
            common = {
                "repository_root": chain["repository"],
                "canonical_origin_urls": frozenset({str(chain["remote"])}),
                "anonymous_head_resolver": lambda: str(chain["freezeCommit"]),
                "anonymous_byte_fetcher": lambda _url: chain["freezeBytes"],
                "recovery_contract": chain["contract"],
                "require_freeze_ci_success": True,
            }
            with self.assertRaisesRegex(ValueError, "completed successful"):
                require_public_pre_score_freeze(
                    **common,
                    actions_runs_resolver=lambda _head: [{
                        "id": 1, "html_url": "https://example.invalid/1",
                        "head_sha": chain["freezeCommit"], "name": "quality",
                        "path": ".github/workflows/quality.yml", "event": "push",
                        "status": "completed", "conclusion": "failure",
                    }],
                )
            with self.assertRaisesRegex(ValueError, "source public CI proof does not match"):
                require_public_pre_score_freeze(
                    **common,
                    actions_runs_resolver=lambda head: [{
                        "id": 901 if head == chain["recoverySource"] else 2,
                        "html_url": (
                            "https://example.invalid/901"
                            if head == chain["recoverySource"]
                            else "https://example.invalid/2"
                        ),
                        "head_sha": head, "name": "quality",
                        "path": ".github/workflows/quality.yml", "event": "push",
                        "status": "completed", "conclusion": "success",
                    }],
                )
            proof = require_public_pre_score_freeze(
                **common,
                actions_runs_resolver=lambda head: [{
                    "id": 900 if head == chain["recoverySource"] else 2,
                    "html_url": (
                        "https://example.invalid/900"
                        if head == chain["recoverySource"]
                        else "https://example.invalid/2"
                    ),
                    "head_sha": head, "name": "quality",
                    "path": ".github/workflows/quality.yml", "event": "push",
                    "status": "completed", "conclusion": "success",
                }],
            )
            self.assertEqual(proof["sourcePublicCiRevalidation"]["runId"], 900)
            self.assertEqual(proof["freezePublicCiProof"]["runId"], 2)

    def test_public_quality_run_rejects_wrong_workflow_event_or_head(self) -> None:
        head = "a" * 40
        invalid_rows = [
            {"id": 1, "html_url": "x", "head_sha": "b" * 40, "name": "quality",
             "path": ".github/workflows/quality.yml", "event": "push", "status": "completed", "conclusion": "success"},
            {"id": 2, "html_url": "x", "head_sha": head, "name": "quality",
             "path": ".github/workflows/other.yml", "event": "push", "status": "completed", "conclusion": "success"},
            {"id": 3, "html_url": "x", "head_sha": head, "name": "quality",
             "path": ".github/workflows/quality.yml", "event": "workflow_dispatch", "status": "completed", "conclusion": "success"},
        ]
        with self.assertRaisesRegex(ValueError, "completed successful"):
            require_successful_public_quality_run(head, runs_resolver=lambda _head: invalid_rows)

    def test_replacement_parity_selection_is_pre_score_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "parity.json"
            subprocess.run(
                [
                    sys.executable,
                    "benchmark/select_parity_ids.py",
                    "--manifest", "benchmark/manifests/test-v2.jsonl",
                    "--output", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                sha256(output.read_bytes()).hexdigest(),
                "0f0e72ac4bd91549af10a76c494138b6cf0c22328d904134b67be82d79badf99",
            )
            manifest = {
                row["id"]: row
                for row in map(json.loads, Path("benchmark/manifests/test-v2.jsonl").read_text().splitlines())
            }
            identifiers = json.loads(output.read_text())
            self.assertEqual(len(identifiers), 60)
            self.assertEqual(len(set(identifiers)), 60)
            self.assertTrue(all(
                int(manifest[identifier]["label"]) == 0
                and manifest[identifier]["source"] == "stockimages-cc0"
                for identifier in identifiers[:30]
            ))
            self.assertTrue(all(
                int(manifest[identifier]["label"]) == 1
                and manifest[identifier]["source"] == "coxy7-infinity"
                for identifier in identifiers[30:]
            ))

    def test_sealed_inference_rejects_underbound_recovery_receipts(self) -> None:
        cases = (
            ("missing", None, "immutable-file list"),
            ("extra", None, "immutable-file list"),
            ("exact", ["README.md", "benchmark/evidence/evaluation/web-negative-v2/**"], "path policy"),
            ("exact", ["benchmark/evidence/evaluation/confirmatory-v2/**", "README.md", "MODEL_CARD.md"], "path policy"),
        )
        for index, (immutable_mode, allowed_paths, message) in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temporary:
                chain = self._build_recovery_freeze_chain(
                    Path(temporary),
                    receipt_immutable_mode=immutable_mode,
                    receipt_allowed_paths=allowed_paths,
                )
                with self.assertRaisesRegex(ValueError, message):
                    require_public_pre_score_freeze(
                        repository_root=chain["repository"],
                        canonical_origin_urls=frozenset({str(chain["remote"])}),
                        anonymous_head_resolver=lambda: str(chain["freezeCommit"]),
                        anonymous_byte_fetcher=lambda _url: chain["freezeBytes"],
                        recovery_contract=chain["contract"],
                    )

    def test_second_recovery_requires_both_exact_repair_surfaces(self) -> None:
        cases = (
            ({"declared_failed_repair_paths": ["frozen.txt"]}, "Failed first recovery source"),
            ({"declared_repair_paths": ["frozen.txt"]}, "Numeric recovery source changed"),
        )
        for arguments, message in cases:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary:
                chain = self._build_recovery_freeze_chain(Path(temporary), **arguments)
                with self.assertRaisesRegex(ValueError, message):
                    require_public_pre_score_freeze(
                        repository_root=chain["repository"],
                        canonical_origin_urls=frozenset({str(chain["remote"])}),
                        anonymous_head_resolver=lambda: str(chain["freezeCommit"]),
                        anonymous_byte_fetcher=lambda _url: chain["freezeBytes"],
                        recovery_contract=chain["contract"],
                    )

    def test_alternate_freeze_receipt_cannot_authorize_a_descendant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chain = self._build_recovery_freeze_chain(Path(temporary))
            repository = chain["repository"]
            alternate = repository / "benchmark/evidence/evaluation/pre-score-freeze-v4.json"
            subprocess.run(
                ["git", "mv", "receipt-seed.json", str(alternate.relative_to(repository))],
                cwd=repository,
                check=True,
            )
            subprocess.run(["git", "commit", "-m", "Forbidden second freeze"], cwd=repository,
                           check=True, capture_output=True)
            alternate.unlink()
            subprocess.run(["git", "add", "-u"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "Hide forbidden second freeze"], cwd=repository,
                           check=True, capture_output=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=repository,
                           check=True, capture_output=True)
            descendant = self._git(repository, "rev-parse", "HEAD")
            self.assertFalse(alternate.exists())
            with self.assertRaisesRegex(ValueError, "alternate pre-score freeze receipt"):
                require_public_pre_score_freeze(
                    repository_root=repository,
                    allow_public_descendant=True,
                    canonical_origin_urls=frozenset({str(chain["remote"])}),
                    anonymous_head_resolver=lambda: descendant,
                    anonymous_byte_fetcher=lambda _url: chain["freezeBytes"],
                    recovery_contract=chain["contract"],
                )

    def test_forbidden_source_rename_cannot_hide_in_an_allowed_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chain = self._build_recovery_freeze_chain(Path(temporary))
            repository = chain["repository"]
            allowed = "benchmark/evidence/evaluation/confirmatory-v2/renamed-a4.txt"
            (repository / "benchmark/evidence/evaluation/confirmatory-v2").mkdir()
            subprocess.run(["git", "mv", "a4.txt", allowed], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "Hide frozen source in evidence"], cwd=repository,
                           check=True, capture_output=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=repository,
                           check=True, capture_output=True)
            descendant = self._git(repository, "rev-parse", "HEAD")
            with self.assertRaisesRegex(ValueError, "Immutable pre-score file|frozen source path"):
                require_public_pre_score_freeze(
                    repository_root=repository,
                    allow_public_descendant=True,
                    canonical_origin_urls=frozenset({str(chain["remote"])}),
                    anonymous_head_resolver=lambda: descendant,
                    anonymous_byte_fetcher=lambda _url: chain["freezeBytes"],
                    recovery_contract=chain["contract"],
                )

    def test_evaluator_rejects_missing_recovery_freeze_before_input_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            benchmark = root / "benchmark"
            benchmark.mkdir()
            shutil.copy2(Path(__file__).parent / "evaluate.py", benchmark / "evaluate.py")
            shutil.copy2(
                Path(__file__).parent / "evaluation_contract.py",
                benchmark / "evaluation_contract.py",
            )
            dependency_marker = "INFERENCE_DEPENDENCY_IMPORTED_BEFORE_FREEZE"
            (benchmark / "onnxruntime.py").write_text(
                f'raise RuntimeError("{dependency_marker}")\n'
            )
            pillow = benchmark / "PIL"
            pillow.mkdir()
            (pillow / "__init__.py").write_text(
                f'raise RuntimeError("{dependency_marker}")\n'
            )
            model = root / "weights/prooflens-cf384.onnx"
            calibration = root / "benchmark/evidence/large/calibration.json"
            manifest = root / "benchmark/manifests/test-v2.jsonl"
            data_root = root / "benchmark/data/replacement-v2"
            output = root / "benchmark/evidence/evaluation/confirmatory-v2"
            result = subprocess.run(
                [
                    sys.executable,
                    str(benchmark / "evaluate.py"),
                    "--model", str(model),
                    "--expected-model-sha256", "941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c",
                    "--data-root", str(data_root),
                    "--manifest", str(manifest),
                    "--expected-manifest-sha256", "773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8",
                    "--output-dir", str(output),
                    "--protocol", "confirmatory-v2",
                    "--execution-provider", "cpu",
                    "--calibration", str(calibration),
                    "--expected-calibration-sha256", "607ec2d8a4428f97cd51ae020f3168bf451201a19b117372033d7becd5a5559c",
                ],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pre-score-freeze-v3.json", result.stderr)
            self.assertNotIn(dependency_marker, result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)
            self.assertNotIn("IsADirectoryError", result.stderr)

    def test_sealed_inference_rejects_noncanonical_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chain = self._build_recovery_freeze_chain(Path(temporary))
            with self.assertRaisesRegex(ValueError, "canonical"):
                require_public_pre_score_freeze(
                    repository_root=chain["repository"],
                    recovery_contract=chain["contract"],
                )

    def test_multi_file_publication_restores_prior_bytes_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_target = root / "published" / "first.bin"
            first_target.parent.mkdir()
            first_target.write_bytes(b"old")
            first_stage = root / "first.stage"
            first_stage.write_bytes(b"new")
            missing_stage = root / "missing.stage"
            second_target = root / "published" / "second.bin"
            with self.assertRaises(FileNotFoundError):
                publish_with_rollback(
                    [(first_stage, first_target), (missing_stage, second_target)],
                    root / "backups",
                )
            self.assertEqual(first_target.read_bytes(), b"old")
            self.assertFalse(second_target.exists())

    def test_publication_rolls_back_when_replacement_raises_after_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "published" / "model.bin"
            target.parent.mkdir()
            target.write_bytes(b"old")
            stage = root / "model.stage"
            stage.write_bytes(b"new")
            original_replace = finalizer.replace_from_stage

            def replace_then_interrupt(source: Path, destination: Path) -> None:
                original_replace(source, destination)
                if source == stage:
                    raise KeyboardInterrupt("injected post-replace interruption")

            finalizer.replace_from_stage = replace_then_interrupt
            try:
                with self.assertRaises(KeyboardInterrupt):
                    publish_with_rollback([(stage, target)], root / "backups")
            finally:
                finalizer.replace_from_stage = original_replace
            self.assertEqual(target.read_bytes(), b"old")

    def test_publication_rejects_duplicate_destination_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "published.bin"
            target.write_bytes(b"old")
            first = root / "first.stage"
            second = root / "second.stage"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            with self.assertRaisesRegex(ValueError, "Duplicate publication target"):
                publish_with_rollback([(first, target), (second, target)], root / "backups")
            self.assertEqual(target.read_bytes(), b"old")

    def test_finalizer_rejects_nonfinite_and_boolean_gate_metrics(self) -> None:
        gates = {
            "minimumBalancedAccuracyPerVariant": 0.85,
            "minimumRealRecallPerVariant": 0.85,
            "minimumSyntheticRecallPerVariant": 0.75,
            "minimumSyntheticRecallPerFamily": 0.6,
            "minimumRealRecallBySource": {"stockimages-cc0": 0.93},
        }
        valid_metrics = {
            variant: {
                "balancedAccuracy": 0.95,
                "realRecall": 0.95,
                "syntheticRecall": 0.95,
                "syntheticRecallBySource": {"GLM-Image": 0.9},
                "realRecallBySource": {"stockimages-cc0": 0.95},
            }
            for variant in finalizer.VARIANTS
        }
        finalizer.require_variant_gates(valid_metrics, gates, label="fixture")
        for invalid in (float("nan"), float("inf"), float("-inf"), True):
            malformed = json.loads(json.dumps(valid_metrics))
            malformed["original"]["balancedAccuracy"] = invalid
            with self.assertRaisesRegex(ValueError, "minimumBalancedAccuracyPerVariant"):
                finalizer.require_variant_gates(malformed, gates, label="fixture")
            self.assertFalse(finalizer.finite_number(invalid))

    def test_m2_publication_profile_is_closed_to_reviewed_paths_and_hashes(self) -> None:
        profile = finalizer.PROFILES["m2"]
        self.assertEqual(profile.identity, "prooflens-cf384-m2-head-v1")
        self.assertEqual(profile.candidate_dir, Path("benchmark/candidates/prooflens-cf384-m2"))
        self.assertEqual(profile.recipe, Path("benchmark/m2/recipe.json"))
        self.assertEqual(profile.selection_summary, Path("benchmark/evidence/m2/selection-summary.json"))
        self.assertEqual(profile.evidence_dir, Path("benchmark/evidence/m2"))
        self.assertEqual(
            profile.expected_model_sha256,
            "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47",
        )
        self.assertEqual(
            profile.expected_model_comparison_sha256,
            "7e037912f28a69ac7ea9620471f1410b7b1ab445b7bb30ce9d7bdbe0c24f96ac",
        )

    def test_finalizer_rejects_non_classifier_model_comparison(self) -> None:
        comparison = {
            "schemaVersion": 1,
            "base": {
                "path": "benchmark/candidates/upstream-cf384.onnx",
                "sha256": finalizer.UPSTREAM_SHA256,
            },
            "candidate": {
                "path": "benchmark/candidates/prooflens-cf384-m2/model.onnx",
                "sha256": finalizer.PROFILES["m2"].expected_model_sha256,
                "bytes": 87_442_080,
            },
            "changedInitializers": [
                {"name": "classifier.bias", "beforeSha256": "1" * 64,
                 "afterSha256": "2" * 64, "dimensions": [1]},
                {"name": "encoder.weight", "beforeSha256": "3" * 64,
                 "afterSha256": "4" * 64, "dimensions": [384, 384]},
            ],
            "unchangedInitializerCount": 198,
            "graphNodesSha256": "5" * 64,
            "graphInputsSha256": "6" * 64,
            "graphOutputsSha256": "7" * 64,
            "opsetsSha256": "8" * 64,
        }
        with self.assertRaisesRegex(ValueError, "Classifier-only model comparison"):
            finalizer.validate_model_comparison(
                comparison,
                profile=finalizer.PROFILES["m2"],
                candidate_sha256=finalizer.PROFILES["m2"].expected_model_sha256,
                candidate_bytes=87_442_080,
            )

    def test_final_model_lock_binds_large_recipe_and_calibration(self) -> None:
        template = {
            "artifact": "weights/prooflens-cf384.onnx",
            "format": "ONNX FP32",
            "input": {"name": "pixel_values"},
            "output": {"name": "logits"},
            "upstream": {"artifactSha256": "a" * 64},
        }
        lock = build_model_lock(
            template,
            candidate_sha256="b" * 64,
            candidate_bytes=123,
            calibration={
                "slope": 1,
                "intercept": 0.25,
                "displayThreshold": 0.65,
                "validationThresholdLogit": 0.3,
            },
            recipe_sha256="c" * 64,
            selection_summary_sha256="d" * 64,
            train_manifest_sha256="e" * 64,
            training_summary_sha256="f" * 64,
            calibration_sha256="1" * 64,
            candidate_grid_sha256="2" * 64,
        )
        self.assertEqual(lock["schemaVersion"], 2)
        self.assertEqual(lock["sha256"], "b" * 64)
        self.assertEqual(lock["trainingEvidence"]["recipeSha256"], "c" * 64)
        self.assertEqual(lock["trainingEvidence"]["selectionSummarySha256"], "d" * 64)
        self.assertEqual(lock["calibration"]["displayThreshold"], 0.65)

    def test_final_model_lock_binds_m2_profile_without_cross_writing_m1(self) -> None:
        template = {
            "artifact": "weights/prooflens-cf384.onnx",
            "format": "ONNX FP32",
            "input": {"name": "pixel_values"},
            "output": {"name": "logits"},
            "upstream": {"artifactSha256": "a" * 64},
        }
        lock = build_model_lock(
            template,
            candidate_sha256="b" * 64,
            candidate_bytes=123,
            calibration={
                "slope": 1,
                "intercept": 0.25,
                "displayThreshold": 0.65,
                "validationThresholdLogit": 0.3,
            },
            recipe_sha256="c" * 64,
            selection_summary_sha256="d" * 64,
            train_manifest_sha256="e" * 64,
            training_summary_sha256="f" * 64,
            calibration_sha256="1" * 64,
            candidate_grid_sha256="2" * 64,
            training_recipe_identity="prooflens-cf384-m2-head-v1",
            recipe_path="benchmark/m2/recipe.json",
            selection_summary_path="benchmark/evidence/m2/selection-summary.json",
        )
        self.assertEqual(
            lock["trainingRecipe"],
            f"prooflens-cf384-m2-head-v1:{'c' * 64}:{'d' * 64}",
        )
        self.assertEqual(lock["trainingEvidence"]["recipe"], "benchmark/m2/recipe.json")
        self.assertEqual(
            lock["trainingEvidence"]["selectionSummary"],
            "benchmark/evidence/m2/selection-summary.json",
        )


if __name__ == "__main__":
    unittest.main()
