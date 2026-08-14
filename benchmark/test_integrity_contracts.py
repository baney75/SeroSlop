"""Regression tests for cache, prediction, and sealed-output integrity."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import numpy as np
from pathlib import Path
import json
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
            root = Path(temporary)
            repository = root / "repository"
            remote = root / "remote.git"
            repository.mkdir()
            subprocess.run(["git", "init", "--bare", remote], check=True, capture_output=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True,
                           capture_output=True)
            subprocess.run(["git", "config", "user.name", "ProofLens test"], cwd=repository,
                           check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                           cwd=repository, check=True)
            (repository / "source.txt").write_text("frozen\n")
            subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "Freeze source"], cwd=repository,
                           check=True, capture_output=True)
            source_commit = self._git(repository, "rev-parse", "HEAD")
            source_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
            source_hash = sha256((repository / "source.txt").read_bytes()).hexdigest()
            freeze_path = repository / "benchmark/evidence/evaluation/pre-score-freeze.json"
            freeze_path.parent.mkdir(parents=True)
            freeze_path.write_text(json.dumps({
                "sourceCommit": source_commit,
                "sourceTree": source_tree,
                "allowedPostScorePaths": ["benchmark/evidence/evaluation/**"],
                "immutableFilesSha256": {"source.txt": source_hash},
            }, indent=2) + "\n")
            subprocess.run(["git", "add", str(freeze_path.relative_to(repository))],
                           cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "Anchor pre-score freeze"], cwd=repository,
                           check=True, capture_output=True)
            freeze_commit = self._git(repository, "rev-parse", "HEAD")
            subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repository,
                           check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repository,
                           check=True, capture_output=True)
            committed_freeze = freeze_path.read_bytes()
            observed = require_public_pre_score_freeze(
                repository_root=repository,
                canonical_origin_urls=frozenset({str(remote)}),
                anonymous_head_resolver=lambda: freeze_commit,
                anonymous_byte_fetcher=lambda _url: committed_freeze,
            )
            self.assertEqual(observed["freezeCommit"], freeze_commit)

            freeze_path.write_text(freeze_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed after"):
                require_public_pre_score_freeze(
                    repository_root=repository,
                    canonical_origin_urls=frozenset({str(remote)}),
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
                )
            with self.assertRaisesRegex(ValueError, "Public origin/main"):
                require_public_pre_score_freeze(
                    repository_root=repository,
                    allow_public_descendant=True,
                    canonical_origin_urls=frozenset({str(remote)}),
                    anonymous_head_resolver=lambda: freeze_commit,
                    anonymous_byte_fetcher=lambda _url: committed_freeze,
                )

    def test_sealed_inference_rejects_noncanonical_origin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "repository"
            remote = root / "private.git"
            repository.mkdir()
            subprocess.run(["git", "init", "--bare", remote], check=True, capture_output=True)
            subprocess.run(["git", "init", "-b", "main"], cwd=repository, check=True,
                           capture_output=True)
            subprocess.run(["git", "config", "user.name", "ProofLens test"], cwd=repository,
                           check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"],
                           cwd=repository, check=True)
            (repository / "source.txt").write_text("frozen\n")
            subprocess.run(["git", "add", "source.txt"], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "Freeze source"], cwd=repository,
                           check=True, capture_output=True)
            source_commit = self._git(repository, "rev-parse", "HEAD")
            source_tree = self._git(repository, "rev-parse", "HEAD^{tree}")
            freeze_path = repository / "benchmark/evidence/evaluation/pre-score-freeze.json"
            freeze_path.parent.mkdir(parents=True)
            freeze_path.write_text(json.dumps({
                "sourceCommit": source_commit,
                "sourceTree": source_tree,
            }) + "\n")
            subprocess.run(["git", "add", str(freeze_path.relative_to(repository))],
                           cwd=repository, check=True)
            subprocess.run(["git", "commit", "-m", "Anchor pre-score freeze"], cwd=repository,
                           check=True, capture_output=True)
            subprocess.run(["git", "remote", "add", "origin", str(remote)], cwd=repository,
                           check=True)
            subprocess.run(["git", "push", "-u", "origin", "main"], cwd=repository,
                           check=True, capture_output=True)
            with self.assertRaisesRegex(ValueError, "canonical"):
                require_public_pre_score_freeze(
                    repository_root=repository,
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
