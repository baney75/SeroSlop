"""Lock, publish, or terminally record the frozen M4 development attempt."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.finalize_training_evidence import (  # noqa: E402
    publish_with_rollback,
    replace_from_stage,
    verified_copy,
)
from benchmark.m4.contracts import canonical_json, load_frozen_protocol  # noqa: E402
from benchmark.m4.publication_contract import (  # noqa: E402
    BASE_COMMIT,
    CANDIDATE_DIR,
    LOCKS_PATH,
    PROFILE,
    PUBLICATION_LOCK_PATH,
    PUBLICATION_ROWS,
    RECIPE_PATH,
    SELECTION_SUMMARY_PATH,
    TRANSACTIONAL_ROWS,
    UPSTREAM_MODEL_BYTES,
    UPSTREAM_MODEL_SHA256,
    build_publication_lock,
    compare_adapter_models,
    digest,
    parse_canonical_json_bytes,
    parse_canonical_publication_lock,
    validate_publication_lock,
    validate_failed_training_packet,
    validate_training_packet,
)
EVIDENCE_DIR = ROOT / "benchmark/evidence/m4"
SHIPPED_MODEL = ROOT / "weights/prooflens-cf384.onnx"
MODEL_LOCK = ROOT / "model-lock.json"
WEIGHTS_README = ROOT / "weights/README.md"
FAILURE_DIAGNOSTIC = EVIDENCE_DIR / "failed-selector-diagnostic-1.json"
FAILURE_RECEIPT = EVIDENCE_DIR / "failed-training-attempt-1.json"


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def require_clean_repository() -> None:
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("M4 finalization requires a completely clean tracked repository")


def require_stage(stage: str) -> None:
    scripts = {
        "source": "scripts/check-m4-source-stage.mjs",
        "pinned": "scripts/check-m4-publication-lock.mjs",
    }
    if stage not in scripts:
        raise ValueError(f"Unknown M4 finalizer stage: {stage}")
    subprocess.run(["node", scripts[stage]], cwd=ROOT, check=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value, pretty=True))


def stage_comparison(destination: Path, packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    comparison, patch = compare_adapter_models(SHIPPED_MODEL, CANDIDATE_DIR / "model.onnx")
    if comparison["candidate"]["sha256"] != packet["modelSha256"] or comparison["candidate"]["bytes"] != packet["modelBytes"]:
        raise ValueError("M4 model comparison does not bind the candidate packet")
    write_json(destination, comparison)
    return comparison, patch


def stage_public_documents(temporary_root: Path, packet: dict[str, Any]) -> dict[str, Path]:
    output = temporary_root / "documents"
    subprocess.run([
        "node", "scripts/render-m4-public-docs.mjs",
        "--summary", str(CANDIDATE_DIR / "validation-summary.json"),
        "--model", str(CANDIDATE_DIR / "model.onnx"),
        "--output-dir", str(output),
    ], cwd=ROOT, check=True)
    stages = {name: output / name for name in ("README.md", "MODEL_CARD.md", "BENCHMARK.md")}
    if not all(path.is_file() for path in stages.values()):
        raise ValueError("M4 documentation renderer did not produce the exact public surface")
    return stages


def stage_fixture_manifest(temporary_root: Path) -> Path:
    from benchmark.m4.select_model_state_fixtures import build_manifest as build_fixture_manifest

    destination = temporary_root / "fixture-manifest.json"
    write_json(destination, build_fixture_manifest())
    return destination


def candidate_evidence_json(comparison: Path, fixture: Path) -> dict[str, str]:
    sources = {
        "training-summary.json": CANDIDATE_DIR / "validation-summary.json",
        "calibration.json": CANDIDATE_DIR / "calibration.json",
        "candidate-grid.json": CANDIDATE_DIR / "candidate-grid.json",
        "selection-lock.json": CANDIDATE_DIR / "selection-lock.json",
        "candidate-tensor-seal.json": CANDIDATE_DIR / "candidate-tensor-seal.json",
        "fresh-feature-run.json": CANDIDATE_DIR / "fresh-feature-run.json",
        "model-comparison.json": comparison,
        "fixture-manifest.json": fixture,
    }
    output: dict[str, str] = {}
    for name, path in sources.items():
        value = path.read_bytes()
        value.decode("utf-8", errors="strict")
        output[name] = value.decode("utf-8")
    return output


def build_model_lock(packet: dict[str, Any], publication_lock_sha256: str) -> dict[str, Any]:
    template = json.loads(MODEL_LOCK.read_text())
    calibration = packet["calibration"]
    summary = packet["summary"]
    return {
        "schemaVersion": 2,
        "artifact": template["artifact"],
        "bytes": packet["modelBytes"],
        "sha256": packet["modelSha256"],
        "format": template["format"],
        "input": template["input"],
        "output": template["output"],
        "upstream": template["upstream"],
        "trainingRecipe": f"prooflens-m4-residual-adapter-v1:{digest(RECIPE_PATH)}:{digest(SELECTION_SUMMARY_PATH)}",
        "trainingEvidence": {
            "recipe": "benchmark/m4/recipe.json",
            "recipeSha256": digest(RECIPE_PATH),
            "sourceLocks": "benchmark/m4/source-locks.json",
            "sourceLocksSha256": digest(LOCKS_PATH),
            "selectionSummary": "benchmark/evidence/m4/selection-summary.json",
            "selectionSummarySha256": digest(SELECTION_SUMMARY_PATH),
            "trainManifestSha256": summary["trainManifestSha256"],
            "selectorManifestSha256": summary["selectorManifestSha256"],
            "m3RegressionManifestSha256": summary["m3RegressionManifestSha256"],
            "m2RegressionManifestSha256": summary["m2RegressionManifestSha256"],
            "trainingSummarySha256": packet["candidateHashes"]["summary"],
            "calibrationSha256": packet["candidateHashes"]["calibration"],
            "candidateGridSha256": packet["candidateHashes"]["grid"],
            "selectionLockSha256": packet["candidateHashes"]["selectionLock"],
            "candidateTensorSealSha256": packet["candidateHashes"]["tensorSeal"],
            "freshFeatureRunId": summary["freshFeatureRunId"],
            "publicationLockSha256": publication_lock_sha256,
            "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
            "architecture": "frozen M2 backbone/classifier plus 384-to-64-to-384 residual feature adapter",
        },
        "calibration": {
            "slope": calibration["slope"],
            "intercept": calibration["intercept"],
            "displayThreshold": calibration["displayThreshold"],
            "validationThresholdLogit": calibration["rawThreshold"],
        },
    }


def weights_readme(*, model_sha256: str, model_bytes: int) -> str:
    return f"""# Packaged detector

`prooflens-cf384.onnx` is the exact FP32 artifact described by the repository root [model-lock.json](../model-lock.json).

```text
bytes    {model_bytes:,}
sha256   {model_sha256}
input    pixel_values [N,3,384,384] float32
output   logits [N,1] float32
```

M4 preserves the M2 backbone and classifier tensors byte-for-byte and inserts one 384→64→384 residual feature adapter. The extension loads this one packaged model locally; it does not download a model or call a detector service after installation.
"""


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
    lock_sha256 = digest(PUBLICATION_LOCK_PATH)
    model_lock_stage = temporary_root / "model-lock.json"
    write_json(model_lock_stage, build_model_lock(packet, lock_sha256))
    weights_readme_stage = temporary_root / "weights-README.md"
    weights_readme_stage.write_text(weights_readme(
        model_sha256=packet["modelSha256"], model_bytes=packet["modelBytes"],
    ))
    receipt = {
        "schemaVersion": 1,
        "profile": PROFILE,
        "sourceCommit": lock["sourceCommit"],
        "sourceTree": lock["sourceTree"],
        "lockCommit": lock_commit,
        "publicationLockSha256": lock_sha256,
        "candidateDirectory": "benchmark/candidates/prooflens-cf384-m4",
        "upstreamSha256": UPSTREAM_MODEL_SHA256,
        "shippedModel": {
            "path": "weights/prooflens-cf384.onnx",
            "sha256": packet["modelSha256"],
            "bytes": packet["modelBytes"],
        },
        "sourceEvidenceSha256": {name: digest(source) for name, source in sources.items()},
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
        "m3RegressionGatesPassed": True,
        "m2RegressionGatesPassed": True,
        "selectionInfluencedByRegression": False,
        "h3HoldoutScored": False,
        "h3PixelsRead": False,
        "transactionalRows": [{"path": path, "status": status} for path, status in TRANSACTIONAL_ROWS],
        "requiredFinalCommitRows": [{"path": path, "status": status} for path, status in PUBLICATION_ROWS],
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
    expected = {ROOT / path for path, _status in PUBLICATION_ROWS}
    if {destination for _stage, destination in staged} != expected or staged[-1][1] != EVIDENCE_DIR / "finalization-receipt.json":
        raise ValueError("M4 finalizer staging surface or receipt order changed")
    return staged, receipt


def cache_inventory(root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if not root.exists():
        return output
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"M4 candidate cache contains a symlink: {path}")
        if path.is_file():
            output.append({
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
    return output


def build_failure_packet() -> tuple[dict[str, Any], dict[str, Any]]:
    from benchmark.m4 import train_adapter

    recipe, _locks = load_frozen_protocol(RECIPE_PATH, LOCKS_PATH)
    summary_path = CANDIDATE_DIR / "validation-summary.json"
    grid_path = CANDIDATE_DIR / "candidate-grid.json"
    seal_path = CANDIDATE_DIR / "candidate-tensor-seal.json"
    for path in (summary_path, grid_path, seal_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    summary = parse_canonical_json_bytes(summary_path.read_bytes(), label="M4 failed summary")
    grid = parse_canonical_json_bytes(grid_path.read_bytes(), label="M4 failed grid")
    seal = parse_canonical_json_bytes(seal_path.read_bytes(), label="M4 failed tensor seal")
    status = summary.get("status")
    allowed = {
        "failed-selector": "failed-selector",
        "failed-m3-selector-regression": "failed-m3-selector-regression",
        "failed-m2-development-regression": "failed-m2-development-regression",
    }
    if status not in allowed or summary.get("commandArguments") != train_arguments() or summary.get("h3HoldoutScored") is not False or summary.get("h3PixelsRead") is not False:
        raise ValueError("M4 failure summary is not a terminal frozen outcome")
    for path in CANDIDATE_DIR.glob("*-state.json"):
        state = parse_canonical_json_bytes(path.read_bytes(), label=f"M4 {path.name}")
        if state.get("status") == "started":
            raise ValueError(f"M4 regression outcome is unknown: {path.name}")
    selection_lock_path = CANDIDATE_DIR / "selection-lock.json"
    selection_lock = None
    if selection_lock_path.exists():
        selection_lock = parse_canonical_json_bytes(selection_lock_path.read_bytes(), label="M4 failed selection lock")
    if status == "failed-selector" and selection_lock is not None:
        raise ValueError("M4 selector failure cannot have a winner lock")
    if status != "failed-selector" and (selection_lock is None or summary.get("selectionLock") != selection_lock):
        raise ValueError("M4 regression failure lacks the pre-regression winner lock")
    regressions = summary.get("regressions", [])
    if not isinstance(regressions, list):
        raise ValueError("M4 failure regression list changed")
    names = [row.get("name") for row in regressions]
    order = recipe["selectionPolicy"]["regressionOrder"]
    if names != order[: len(names)] or (status == "failed-selector" and names) or (status != "failed-selector" and not names):
        raise ValueError("M4 failure regression order changed")
    if status == "failed-m3-selector-regression" and names != [order[0]]:
        raise ValueError("M4 M3 regression failure sequence changed")
    if status == "failed-m2-development-regression" and names != order:
        raise ValueError("M4 M2 regression failure sequence changed")
    if status != "failed-selector" and (regressions[-1].get("passed") is not False or type(regressions[-1].get("passed")) is not bool):
        raise ValueError("M4 terminal regression failure is not explicit")
    completed_count = {
        "failed-selector": 0,
        "failed-m3-selector-regression": 1,
        "failed-m2-development-regression": 2,
    }[status]
    regression_specs = [
        (
            "m3-selector-regression",
            ROOT / recipe["regressions"][0]["manifest"],
            ROOT / recipe["regressions"][0]["dataRoot"],
            recipe["regressions"][0]["gates"],
        ),
        (
            "m2-development-regression",
            ROOT / recipe["regressions"][1]["manifest"],
            ROOT / recipe["regressions"][1]["dataRoot"],
            recipe["regressions"][1]["gates"],
        ),
    ]
    expected_states = {
        CANDIDATE_DIR / f"{name}-state.json"
        for name, _manifest, _data_root, _gates in regression_specs[:completed_count]
    }
    actual_states = set(CANDIDATE_DIR.glob("*-state.json"))
    if actual_states != expected_states:
        raise ValueError("M4 terminal regression state-file set changed")
    previous_state_sha256: str | None = None
    for index, (name, manifest, data_root, gates) in enumerate(regression_specs[:completed_count]):
        state_path = CANDIDATE_DIR / f"{name}-state.json"
        state = parse_canonical_json_bytes(state_path.read_bytes(), label=f"M4 {name} state")
        items = train_adapter.safe_manifest_items(manifest, data_root)
        result, state_evidence = train_adapter.validate_completed_regression_state(
            state,
            name=name,
            selection_lock_sha256=digest(selection_lock_path),
            manifest=manifest,
            items=items,
            threshold=float(summary["selectedCandidate"]["rawThreshold"]),
            gates=gates,
            previous_state_sha256=previous_state_sha256,
        )
        if result != regressions[index] or state_evidence != summary["featureShardEvidence"][58 + index : 59 + index]:
            raise ValueError(f"M4 failure summary disagrees with regression state: {name}")
        if index < completed_count - 1 and result["passed"] is not True:
            raise ValueError("M4 failure continued after an earlier regression failed")
        previous_state_sha256 = digest(state_path)
    validate_failed_training_packet(summary, grid, seal, selection_lock)
    artifacts = {
        "validation-summary.json": digest(summary_path),
        "candidate-grid.json": digest(grid_path),
        "candidate-tensor-seal.json": digest(seal_path),
        "fresh-feature-run.json": digest(CANDIDATE_DIR / "fresh-feature-run.json"),
    }
    if selection_lock_path.exists():
        artifacts["selection-lock.json"] = digest(selection_lock_path)
    source_commit = git("rev-parse", "HEAD")
    source_tree = git("rev-parse", "HEAD^{tree}")
    diagnostic = {
        "schemaVersion": 1,
        "profile": PROFILE,
        "sourceCommit": source_commit,
        "sourceTree": source_tree,
        "baseCommit": BASE_COMMIT,
        "recipeSha256": digest(RECIPE_PATH),
        "sourceLocksSha256": digest(LOCKS_PATH),
        "trainerSha256": digest(ROOT / "benchmark/m4/train_adapter.py"),
        "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
        "commandArguments": train_arguments(),
        "candidateDirectory": "benchmark/candidates/prooflens-cf384-m4",
        "terminalOutcome": allowed[status],
        "attemptStatus": status,
        "candidateArtifactSha256": artifacts,
        "selectionLockSha256": digest(selection_lock_path) if selection_lock_path.exists() else None,
        "selectionLock": selection_lock,
        "candidateGrid": grid,
        "candidateTensorSeal": seal,
        "freshFeatureMarkerJson": (CANDIDATE_DIR / "fresh-feature-run.json").read_text(),
        "trainingSummary": summary,
        "selectedCandidate": summary.get("selectedCandidate"),
        "completedRegressions": regressions,
        "notRunRegressions": order[len(regressions):],
        "freshFeatureRunId": summary.get("freshFeatureRunId"),
        "featureShardEvidence": summary.get("featureShardEvidence"),
        "publishedModel": False,
        "publicationLockCreated": False,
        "successfulM4PublicationEvidencePresent": False,
        "h3HoldoutScored": False,
        "h3PixelsRead": False,
        "terminality": {
            "reselectionPermitted": False,
            "thresholdChangePermitted": False,
            "gateChangePermitted": False,
            "retrySameSelectorPermitted": False,
        },
    }
    diagnostic_bytes = canonical_json(diagnostic, pretty=True)
    inventory = cache_inventory(CANDIDATE_DIR)
    receipt = {
        "schemaVersion": 1,
        "profile": PROFILE,
        "sourceCommit": source_commit,
        "sourceTree": source_tree,
        "diagnosticPath": "benchmark/evidence/m4/failed-selector-diagnostic-1.json",
        "diagnosticSha256": sha256(diagnostic_bytes).hexdigest(),
        "terminalOutcome": diagnostic["terminalOutcome"],
        "candidateCacheSnapshot": {
            "fileCount": len(inventory),
            "bytes": sum(row["bytes"] for row in inventory),
            "inventory": inventory,
        },
        "successfulM4PublicationEvidencePresent": False,
        "h3HoldoutScored": False,
        "h3PixelsRead": False,
        "requiredFailureCommitRows": [
            {"path": "benchmark/evidence/m4/failed-selector-diagnostic-1.json", "status": "A"},
            {"path": "benchmark/evidence/m4/failed-training-attempt-1.json", "status": "A"},
        ],
    }
    return diagnostic, receipt


def train_arguments() -> list[str]:
    from benchmark.m4.train_adapter import expected_arguments

    return expected_arguments()


def publish_failure(*, check_only: bool) -> None:
    require_stage("source")
    for path in (PUBLICATION_LOCK_PATH, FAILURE_DIAGNOSTIC, FAILURE_RECEIPT):
        if path.exists():
            raise ValueError(f"M4 terminal output already exists: {path}")
    for name in ("calibration.json", "candidate-grid.json", "finalization-receipt.json", "model-comparison.json", "training-summary.json"):
        if (EVIDENCE_DIR / name).exists():
            raise ValueError(f"M4 successful publication evidence already exists: {name}")
    diagnostic, receipt = build_failure_packet()
    with tempfile.TemporaryDirectory(prefix=".prooflens-m4-failure-", dir=ROOT) as temporary_name:
        temporary = Path(temporary_name)
        diagnostic_stage = temporary / FAILURE_DIAGNOSTIC.name
        receipt_stage = temporary / FAILURE_RECEIPT.name
        write_json(diagnostic_stage, diagnostic)
        if digest(diagnostic_stage) != receipt["diagnosticSha256"]:
            raise ValueError("M4 failure diagnostic digest changed during staging")
        write_json(receipt_stage, receipt)
        if not check_only:
            publish_with_rollback([
                (diagnostic_stage, FAILURE_DIAGNOSTIC),
                (receipt_stage, FAILURE_RECEIPT),
            ], temporary / "backups")
    print(json.dumps({**receipt, "checkOnly": check_only}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--write-lock", action="store_true")
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--publish-failure", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    require_clean_repository()
    if args.publish_failure:
        publish_failure(check_only=args.check_only)
        return
    require_stage("source" if args.write_lock else "pinned")
    packet = validate_training_packet()
    with tempfile.TemporaryDirectory(prefix=".prooflens-m4-finalize-", dir=ROOT) as temporary_name:
        temporary_root = Path(temporary_name)
        comparison_stage = temporary_root / "model-comparison.json"
        _comparison, adapter_patch = stage_comparison(comparison_stage, packet)
        document_stages = stage_public_documents(temporary_root, packet)
        fixture_stage = stage_fixture_manifest(temporary_root)
        public_document_hashes = {name: digest(stage) for name, stage in document_stages.items()}
        fixture_sha256 = digest(fixture_stage)
        evidence_json = candidate_evidence_json(comparison_stage, fixture_stage)
        source_commit = git("rev-parse", "HEAD" if args.write_lock else "HEAD^")
        source_tree = git("rev-parse", f"{source_commit}^{{tree}}")
        expected_lock = build_publication_lock(
            source_commit=source_commit,
            source_tree=source_tree,
            packet=packet,
            comparison_sha256=digest(comparison_stage),
            adapter_patch=adapter_patch,
            candidate_evidence_json=evidence_json,
            public_document_hashes=public_document_hashes,
            fixture_manifest_sha256=fixture_sha256,
        )
        if args.write_lock:
            if PUBLICATION_LOCK_PATH.exists():
                raise ValueError("M4 publication lock already exists")
            lock_stage = temporary_root / "publication-lock.json"
            write_json(lock_stage, expected_lock)
            if not args.check_only:
                PUBLICATION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
                replace_from_stage(lock_stage, PUBLICATION_LOCK_PATH)
            print(json.dumps({**expected_lock, "checkOnly": args.check_only}, indent=2))
            return
        lock = parse_canonical_publication_lock(PUBLICATION_LOCK_PATH.read_bytes())
        validate_publication_lock(
            lock,
            source_commit=source_commit,
            source_tree=source_tree,
            packet=packet,
            comparison_sha256=digest(comparison_stage),
            adapter_patch=adapter_patch,
            candidate_evidence_json=evidence_json,
            public_document_hashes=public_document_hashes,
            fixture_manifest_sha256=fixture_sha256,
        )
        for name in ("training-summary.json", "calibration.json", "candidate-grid.json", "model-comparison.json", "finalization-receipt.json"):
            if (EVIDENCE_DIR / name).exists():
                raise ValueError(f"M4 publication target already exists: {name}")
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
