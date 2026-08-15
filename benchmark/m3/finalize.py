"""Lock and transactionally publish the frozen M3 classifier-head candidate."""

from __future__ import annotations

import argparse
import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.finalize_training_evidence import (  # noqa: E402
    build_model_lock,
    publish_with_rollback,
    replace_from_stage,
    verified_copy,
    weights_readme,
)
from benchmark.m3.publication_contract import (  # noqa: E402
    CANDIDATE_DIR,
    PUBLICATION_LOCK_PATH,
    PUBLICATION_ROWS,
    TRANSACTIONAL_ROWS,
    RECIPE_PATH,
    SELECTION_SUMMARY_PATH,
    UPSTREAM_MODEL_PATH,
    UPSTREAM_MODEL_SHA256,
    build_publication_lock,
    digest,
    parse_canonical_publication_lock,
    validate_model_comparison,
    validate_publication_lock,
    validate_training_packet,
)
from benchmark.m3.select_model_state_fixtures import build_manifest as build_fixture_manifest  # noqa: E402


EVIDENCE_DIR = ROOT / "benchmark/evidence/m3"
SHIPPED_MODEL = ROOT / "weights/prooflens-cf384.onnx"
MODEL_LOCK = ROOT / "model-lock.json"
WEIGHTS_README = ROOT / "weights/README.md"


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def require_clean_repository() -> None:
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("M3 finalization requires a completely clean repository")


def require_stage(stage: str) -> None:
    script = {
        "source": "scripts/check-m3-source-stage.mjs",
        "pinned": "scripts/check-m3-publication-lock.mjs",
    }[stage]
    subprocess.run(["node", script], cwd=ROOT, check=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def generate_comparison(destination: Path, packet: dict[str, Any]) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            "benchmark/compare_models.py",
            "--base", "weights/prooflens-cf384.onnx",
            "--expected-base-sha256", UPSTREAM_MODEL_SHA256,
            "--candidate", "benchmark/candidates/prooflens-cf384-m3/model.onnx",
            "--expected-candidate-sha256", packet["modelSha256"],
            "--output", str(destination),
        ],
        cwd=ROOT,
        check=True,
    )
    if not destination.is_file():
        raise ValueError("M3 classifier-only comparison was not produced")
    comparison = json.loads(destination.read_text())
    validate_model_comparison(
        comparison,
        candidate_sha256=packet["modelSha256"],
        candidate_bytes=packet["modelBytes"],
    )
    return comparison


def stage_public_documents(temporary_root: Path, packet: dict[str, Any]) -> dict[str, Path]:
    output = temporary_root / "public-docs"
    subprocess.run(
        [
            "node",
            "scripts/render-m3-public-docs.mjs",
            "--summary", str(CANDIDATE_DIR / "validation-summary.json"),
            "--model", str(CANDIDATE_DIR / "model.onnx"),
            "--output-dir", str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    stages = {
        "README.md": output / "README.md",
        "MODEL_CARD.md": output / "MODEL_CARD.md",
        "BENCHMARK.md": output / "BENCHMARK.md",
    }
    if any(not path.is_file() for path in stages.values()):
        raise ValueError("M3 documentation renderer did not produce the exact public surface")
    return stages


def stage_fixture_manifest(temporary_root: Path) -> Path:
    destination = temporary_root / "fixture-manifest.json"
    write_json(destination, build_fixture_manifest())
    return destination


def build_candidate_evidence_json(comparison_stage: Path, fixture_stage: Path) -> dict[str, str]:
    paths = {
        "training-summary.json": CANDIDATE_DIR / "validation-summary.json",
        "calibration.json": CANDIDATE_DIR / "calibration.json",
        "candidate-grid.json": CANDIDATE_DIR / "candidate-grid.json",
        "model-comparison.json": comparison_stage,
        "fixture-manifest.json": fixture_stage,
    }
    return {name: path.read_text() for name, path in paths.items()}


def build_classifier_patch(packet: dict[str, Any]) -> dict[str, Any]:
    import onnx  # Local training/finalization dependency; not required by static CI.

    base_bytes = UPSTREAM_MODEL_PATH.read_bytes()
    candidate_path = CANDIDATE_DIR / "model.onnx"
    candidate_bytes = candidate_path.read_bytes()
    if len(base_bytes) != len(candidate_bytes):
        raise ValueError("M3 candidate serialization length differs from the frozen M2 model")
    base = onnx.load_model_from_string(base_bytes)
    candidate = onnx.load_model_from_string(candidate_bytes)
    base_initializers = {value.name: value for value in base.graph.initializer}
    candidate_initializers = {value.name: value for value in candidate.graph.initializer}
    replacements: list[dict[str, Any]] = []
    reconstructed = bytearray(base_bytes)
    for name in ("classifier.bias", "classifier.weight"):
        before_value = base_initializers.get(name)
        after_value = candidate_initializers.get(name)
        if before_value is None or after_value is None:
            raise ValueError(f"M3 classifier initializer is missing: {name}")
        before = bytes(before_value.raw_data)
        after = bytes(after_value.raw_data)
        if (
            not before
            or before == after
            or len(before) != len(after)
            or list(before_value.dims) != list(after_value.dims)
        ):
            raise ValueError(f"M3 classifier raw tensor representation changed: {name}")
        offset = base_bytes.find(before)
        if offset < 0 or base_bytes.find(before, offset + 1) >= 0:
            raise ValueError(f"M3 classifier raw tensor is not unique in the base model: {name}")
        reconstructed[offset:offset + len(before)] = after
        replacements.append({
            "name": name,
            "dimensions": list(after_value.dims),
            "offset": offset,
            "bytes": len(after),
            "beforeSha256": sha256(before).hexdigest(),
            "afterSha256": sha256(after).hexdigest(),
            "afterBase64": base64.b64encode(after).decode("ascii"),
        })
    if bytes(reconstructed) != candidate_bytes:
        raise ValueError("M3 candidate differs from M2 outside the two classifier raw tensors")
    if packet["modelSha256"] == UPSTREAM_MODEL_SHA256 or sha256(candidate_bytes).hexdigest() != packet["modelSha256"]:
        raise ValueError("M3 reconstructed candidate hash differs from the validated training packet")
    return {
        "schemaVersion": 1,
        "baseSha256": UPSTREAM_MODEL_SHA256,
        "candidateSha256": packet["modelSha256"],
        "candidateBytes": len(candidate_bytes),
        "replacements": replacements,
    }


def build_staged_publication(
    temporary_root: Path,
    *,
    packet: dict[str, Any],
    comparison_stage: Path,
    document_stages: dict[str, Path],
    fixture_stage: Path,
    lock: dict[str, Any],
    lock_commit: str,
) -> tuple[list[tuple[Path, Path]], dict[str, Any]]:
    sources = {
        "training-summary.json": CANDIDATE_DIR / "validation-summary.json",
        "calibration.json": CANDIDATE_DIR / "calibration.json",
        "candidate-grid.json": CANDIDATE_DIR / "candidate-grid.json",
    }
    model_stage = temporary_root / "prooflens-cf384.onnx"
    verified_copy(CANDIDATE_DIR / "model.onnx", model_stage)
    evidence_stages: dict[str, Path] = {}
    for name, source in sources.items():
        stage = temporary_root / name
        verified_copy(source, stage)
        evidence_stages[name] = stage

    template = json.loads(MODEL_LOCK.read_text())
    calibration = packet["calibration"]
    summary = packet["summary"]
    lock_sha256 = digest(PUBLICATION_LOCK_PATH)
    model_lock = build_model_lock(
        template,
        candidate_sha256=packet["modelSha256"],
        candidate_bytes=packet["modelBytes"],
        calibration=calibration,
        recipe_sha256=digest(RECIPE_PATH),
        selection_summary_sha256=digest(SELECTION_SUMMARY_PATH),
        train_manifest_sha256=summary["trainManifestSha256"],
        training_summary_sha256=digest(sources["training-summary.json"]),
        calibration_sha256=digest(sources["calibration.json"]),
        candidate_grid_sha256=digest(sources["candidate-grid.json"]),
        training_recipe_identity="prooflens-cf384-m3-cultural-heritage-head-v1",
        recipe_path="benchmark/m3/recipe.json",
        selection_summary_path="benchmark/evidence/m3/selection-summary.json",
    )
    model_lock["trainingEvidence"].update({
        "validationManifestSha256": summary["validationManifestSha256"],
        "regressionManifestSha256": summary["regressionManifestSha256"],
        "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
        "freshFeatureRunId": packet["freshRunId"],
        "publicationLockSha256": lock_sha256,
    })
    model_lock_stage = temporary_root / "model-lock.json"
    write_json(model_lock_stage, model_lock)
    weights_readme_stage = temporary_root / "weights-README.md"
    weights_readme_stage.write_text(
        weights_readme(
            candidate_sha256=packet["modelSha256"],
            candidate_bytes=packet["modelBytes"],
        )
    )

    receipt = {
        "schemaVersion": 1,
        "profile": "m3",
        "sourceCommit": lock["sourceCommit"],
        "sourceTree": lock["sourceTree"],
        "lockCommit": lock_commit,
        "publicationLockSha256": lock_sha256,
        "candidateDirectory": "benchmark/candidates/prooflens-cf384-m3",
        "upstreamSha256": UPSTREAM_MODEL_SHA256,
        "shippedModel": {
            "path": "weights/prooflens-cf384.onnx",
            "sha256": packet["modelSha256"],
            "bytes": packet["modelBytes"],
        },
        "sourceEvidenceSha256": {
            name: digest(source) for name, source in sources.items()
        },
        "publishedEvidenceSha256": {
            **{name: digest(stage) for name, stage in evidence_stages.items()},
            "model-comparison.json": digest(comparison_stage),
        },
        "publishedRepositorySha256": {
            "weights/prooflens-cf384.onnx": packet["modelSha256"],
            "model-lock.json": digest(model_lock_stage),
            "weights/README.md": digest(weights_readme_stage),
            **{name: digest(stage) for name, stage in document_stages.items()},
            "tests/fixtures/model-states/fixture-manifest.json": digest(fixture_stage),
        },
        "selectorGatesPassed": True,
        "regressionGatesPassed": True,
        "selectionInfluencedByRegression": False,
        "h3HoldoutScored": False,
        "transactionalRows": [
            {"path": path, "status": status} for path, status in TRANSACTIONAL_ROWS
        ],
        "requiredFinalCommitRows": [
            {"path": path, "status": status} for path, status in PUBLICATION_ROWS
        ],
    }
    receipt_stage = temporary_root / "finalization-receipt.json"
    write_json(receipt_stage, receipt)
    staged = [
        *[(stage, ROOT / name) for name, stage in document_stages.items()],
        (model_stage, SHIPPED_MODEL),
        (model_lock_stage, MODEL_LOCK),
        (weights_readme_stage, WEIGHTS_README),
        *[(stage, EVIDENCE_DIR / name) for name, stage in evidence_stages.items()],
        (comparison_stage, EVIDENCE_DIR / "model-comparison.json"),
        (fixture_stage, ROOT / "tests/fixtures/model-states/fixture-manifest.json"),
        (receipt_stage, EVIDENCE_DIR / "finalization-receipt.json"),
    ]
    expected_destinations = {ROOT / path for path, _ in PUBLICATION_ROWS}
    if {destination for _, destination in staged} != expected_destinations:
        raise ValueError("M3 finalizer staging surface differs from the final-commit contract")
    return staged, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write-lock", action="store_true")
    modes.add_argument("--publish", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require_clean_repository()
    require_stage("source" if args.write_lock else "pinned")
    packet = validate_training_packet()

    with tempfile.TemporaryDirectory(prefix=".prooflens-m3-finalize-", dir=ROOT) as temporary_name:
        temporary_root = Path(temporary_name)
        comparison_stage = temporary_root / "model-comparison.json"
        generate_comparison(comparison_stage, packet)
        document_stages = stage_public_documents(temporary_root, packet)
        fixture_stage = stage_fixture_manifest(temporary_root)
        public_document_hashes = {name: digest(stage) for name, stage in document_stages.items()}
        fixture_manifest_sha256 = digest(fixture_stage)
        candidate_evidence_json = build_candidate_evidence_json(comparison_stage, fixture_stage)
        classifier_patch = build_classifier_patch(packet)
        source_commit = git("rev-parse", "HEAD" if args.write_lock else "HEAD^")
        source_tree = git("rev-parse", f"{source_commit}^{{tree}}")
        expected_lock = build_publication_lock(
            packet=packet,
            comparison_sha256=digest(comparison_stage),
            source_commit=source_commit,
            source_tree=source_tree,
            public_document_hashes=public_document_hashes,
            fixture_manifest_sha256=fixture_manifest_sha256,
            candidate_evidence_json=candidate_evidence_json,
            classifier_patch=classifier_patch,
        )

        if args.write_lock:
            if PUBLICATION_LOCK_PATH.exists():
                raise ValueError("M3 publication lock already exists")
            lock_stage = temporary_root / "publication-lock.json"
            write_json(lock_stage, expected_lock)
            if not args.check_only:
                PUBLICATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
                replace_from_stage(lock_stage, PUBLICATION_LOCK_PATH)
            print(json.dumps({**expected_lock, "checkOnly": args.check_only}, indent=2))
            return

        lock = parse_canonical_publication_lock(PUBLICATION_LOCK_PATH.read_text())
        validate_publication_lock(
            lock,
            packet=packet,
            comparison_sha256=digest(comparison_stage),
            source_commit=source_commit,
            source_tree=source_tree,
            public_document_hashes=public_document_hashes,
            fixture_manifest_sha256=fixture_manifest_sha256,
            candidate_evidence_json=candidate_evidence_json,
            classifier_patch=classifier_patch,
        )
        for name in (
            "training-summary.json",
            "calibration.json",
            "candidate-grid.json",
            "model-comparison.json",
            "finalization-receipt.json",
        ):
            if (EVIDENCE_DIR / name).exists():
                raise ValueError(f"M3 publication target already exists: {name}")
        staged, receipt = build_staged_publication(
            temporary_root,
            packet=packet,
            comparison_stage=comparison_stage,
            document_stages=document_stages,
            fixture_stage=fixture_stage,
            lock=lock,
            lock_commit=git("rev-parse", "HEAD"),
        )
        if not args.check_only:
            publish_with_rollback(staged, temporary_root / "backups")
        print(json.dumps({**receipt, "checkOnly": args.check_only}, indent=2))


if __name__ == "__main__":
    main()
