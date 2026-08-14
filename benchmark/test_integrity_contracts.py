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
    require_public_pre_score_freeze,
)
from feature_cache_contract import expected_view_metadata, validate_feature_arrays  # noqa: E402
from fresh_feature_run import (  # noqa: E402
    cache_belongs_to_fresh_feature_run,
    complete_fresh_feature_run,
    marker_sha256,
    open_or_create_fresh_feature_run,
)
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
        source_file = repository / "source.txt"
        source_file.write_text("legacy source\n")
        second_source_file = repository / "second-source.txt"
        second_source_file.write_text("second recovery pending\n")
        (repository / "frozen.txt").write_text("frozen behavior\n")
        (repository / "receipt-seed.json").write_text("{}\n")
        subprocess.run(
            ["git", "add", "source.txt", "second-source.txt", "frozen.txt", "receipt-seed.json"],
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
        source_file.write_text("streaming verifier repair\n")
        subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Recovery source"], cwd=repository,
                       check=True, capture_output=True)
        failed_recovery_source = self._git(repository, "rev-parse", "HEAD")
        failed_recovery_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        second_source_file.write_text("dependency-loading repair\n")
        subprocess.run(["git", "add", "second-source.txt"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-m", "Second recovery source"], cwd=repository,
                       check=True, capture_output=True)
        recovery_source = self._git(repository, "rev-parse", "HEAD")
        recovery_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
        source_hash = sha256(second_source_file.read_bytes()).hexdigest()
        expected_allowed_paths = ["benchmark/evidence/evaluation/**", "README.md"]
        failed_repair_paths = ["source.txt"] if declared_failed_repair_paths is None else declared_failed_repair_paths
        repair_paths = ["second-source.txt"] if declared_repair_paths is None else declared_repair_paths
        if receipt_immutable_mode == "exact":
            receipt_immutable_files = {"second-source.txt": source_hash}
        elif receipt_immutable_mode == "missing":
            receipt_immutable_files = {}
        elif receipt_immutable_mode == "extra":
            receipt_immutable_files = {"second-source.txt": source_hash, "extra.txt": "0" * 64}
        else:  # pragma: no cover - test helper misuse
            raise ValueError(f"Unknown immutable test mode: {receipt_immutable_mode}")
        freeze_path = Path("benchmark/evidence/evaluation/pre-score-freeze-v2.json")
        freeze_file = repository / freeze_path
        freeze_file.write_text(json.dumps({
            "schemaVersion": 3,
            "generation": 2,
            "mode": "public second-recovery pre-score source freeze before any confirmatory or web-negative inference",
            "receiptPath": str(freeze_path),
            "sourceCommit": recovery_source,
            "sourceTree": recovery_tree,
            "allowedPostScorePaths": (
                expected_allowed_paths if receipt_allowed_paths is None else receipt_allowed_paths
            ),
            "immutableFilesSha256": receipt_immutable_files,
            "recovery": {
                "reason": "test-resource-exhaustion",
                "legacySourceCommit": legacy_source,
                "legacyFreezeCommit": legacy_freeze,
                "legacyReceiptPath": str(legacy_path),
                "legacyReceiptSha256": legacy_hash,
                "failedRecoverySourceCommit": failed_recovery_source,
                "failedRecoverySourceTree": failed_recovery_tree,
                "failedRecoveryActionsRunId": 12345,
                "failedRecoveryActionsRunUrl": "https://example.invalid/actions/runs/12345",
                "failedRecoveryReason": "test-missing-dependencies",
                "failedRecoveryRepairPaths": failed_repair_paths,
                "repairPaths": repair_paths,
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
            "legacyFreezePath": str(legacy_path),
            "legacySourceCommit": legacy_source,
            "legacyFreezeCommit": legacy_freeze,
            "legacyFreezeSha256": legacy_hash,
            "failedRecoverySourceCommit": failed_recovery_source,
            "failedRecoverySourceTree": failed_recovery_tree,
            "failedRecoveryActionsRunId": 12345,
            "failedRecoveryActionsRunUrl": "https://example.invalid/actions/runs/12345",
            "failedRecoveryReason": "test-missing-dependencies",
            "failedRecoveryRepairPaths": failed_repair_paths,
            "reason": "test-resource-exhaustion",
            "repairPaths": repair_paths,
            "immutableFiles": ["second-source.txt"],
            "allowedPostScorePaths": expected_allowed_paths,
        }
        return {
            "repository": repository,
            "remote": remote,
            "freezeFile": freeze_file,
            "freezeBytes": freeze_file.read_bytes(),
            "freezeCommit": freeze_commit,
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

    def test_protocol_output_cannot_be_redirected_to_a_disposable_directory(self) -> None:
        repository_root = Path.cwd().resolve()
        canonical = repository_root / "benchmark/evidence/evaluation/confirmatory"
        self.assertEqual(
            require_canonical_output_directory("confirmatory", canonical, repository_root=repository_root),
            canonical,
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "must be written"):
                require_canonical_output_directory(
                    "confirmatory", Path(temporary), repository_root=repository_root
                )

    def test_protocol_output_rejects_symlinked_canonical_component(self) -> None:
        with tempfile.TemporaryDirectory() as repository, tempfile.TemporaryDirectory() as disposable:
            repository_root = Path(repository)
            evaluation_root = repository_root / "benchmark/evidence/evaluation"
            evaluation_root.mkdir(parents=True)
            (evaluation_root / "confirmatory").symlink_to(disposable, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                require_canonical_output_directory(
                    "confirmatory",
                    evaluation_root / "confirmatory",
                    repository_root=repository_root,
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

    def test_sealed_inference_rejects_underbound_recovery_receipts(self) -> None:
        cases = (
            ("missing", None, "immutable-file list"),
            ("extra", None, "immutable-file list"),
            ("exact", ["README.md", "benchmark/evidence/evaluation/**"], "path policy"),
            ("exact", ["benchmark/evidence/evaluation/**", "README.md", "MODEL_CARD.md"], "path policy"),
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
            ({"declared_repair_paths": ["frozen.txt"]}, "Recovery source changed"),
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
            alternate = repository / "benchmark/evidence/evaluation/pre-score-freeze-v3.json"
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
            allowed = "benchmark/evidence/evaluation/renamed-frozen.txt"
            subprocess.run(["git", "mv", "frozen.txt", allowed], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "Hide frozen source in evidence"], cwd=repository,
                           check=True, capture_output=True)
            subprocess.run(["git", "push", "origin", "main"], cwd=repository,
                           check=True, capture_output=True)
            descendant = self._git(repository, "rev-parse", "HEAD")
            with self.assertRaisesRegex(ValueError, "frozen source path"):
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
            sentinel = root / "sentinel-input"
            sentinel.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(benchmark / "evaluate.py"),
                    "--model", str(sentinel),
                    "--expected-model-sha256", "0" * 64,
                    "--data-root", str(sentinel),
                    "--manifest", str(sentinel),
                    "--expected-manifest-sha256", "28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae",
                    "--output-dir", str(benchmark / "evidence/evaluation/confirmatory"),
                    "--protocol", "confirmatory",
                    "--execution-provider", "cpu",
                    "--calibration", str(sentinel),
                    "--expected-calibration-sha256", "0" * 64,
                ],
                cwd=root,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pre-score-freeze-v2.json", result.stderr)
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


if __name__ == "__main__":
    unittest.main()
