"""Transactionally publish a validated large-head candidate and evidence packet."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verified_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(destination) != digest(source):
        raise ValueError(f"Copy verification failed: {destination}")


def replace_from_stage(stage: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        verified_copy(stage, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_with_rollback(staged: list[tuple[Path, Path]], backup_root: Path) -> None:
    """Publish staged bytes; restore every prior target if any replacement fails."""
    backups: dict[Path, Path | None] = {}
    for index, (_, destination) in enumerate(staged):
        if destination in backups:
            raise ValueError(f"Duplicate publication target: {destination}")
        if destination.exists():
            backup = backup_root / f"{index:02d}-{destination.name}"
            verified_copy(destination, backup)
            backups[destination] = backup
        else:
            backups[destination] = None

    published: list[Path] = []
    try:
        for stage, destination in staged:
            replace_from_stage(stage, destination)
            published.append(destination)
            if digest(destination) != digest(stage):
                raise ValueError(f"Published bytes do not match staged bytes: {destination}")
    except BaseException as publication_error:
        rollback_errors: list[str] = []
        for destination in reversed(published):
            backup = backups[destination]
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    replace_from_stage(backup, destination)
            except BaseException as rollback_error:  # pragma: no cover - emergency path
                rollback_errors.append(f"{destination}: {rollback_error}")
        if rollback_errors:
            raise RuntimeError(
                "Publication failed and rollback was incomplete: " + "; ".join(rollback_errors)
            ) from publication_error
        raise


def build_model_lock(
    template: dict[str, object],
    *,
    candidate_sha256: str,
    candidate_bytes: int,
    calibration: dict[str, object],
    recipe_sha256: str,
    selection_summary_sha256: str,
    train_manifest_sha256: str,
    training_summary_sha256: str,
    calibration_sha256: str,
    candidate_grid_sha256: str,
) -> dict[str, object]:
    return {
        "schemaVersion": 2,
        "artifact": template["artifact"],
        "bytes": candidate_bytes,
        "sha256": candidate_sha256,
        "format": template["format"],
        "input": template["input"],
        "output": template["output"],
        "upstream": template["upstream"],
        "trainingRecipe": (
            f"prooflens-cf384-large-head-v1:{recipe_sha256}:{selection_summary_sha256}"
        ),
        "trainingEvidence": {
            "recipe": "benchmark/large/recipe.json",
            "recipeSha256": recipe_sha256,
            "selectionSummary": "benchmark/evidence/large/selection-summary.json",
            "selectionSummarySha256": selection_summary_sha256,
            "trainManifestSha256": train_manifest_sha256,
            "trainingSummarySha256": training_summary_sha256,
            "calibrationSha256": calibration_sha256,
            "candidateGridSha256": candidate_grid_sha256,
        },
        "calibration": {
            "slope": calibration["slope"],
            "intercept": calibration["intercept"],
            "displayThreshold": calibration["displayThreshold"],
            "validationThresholdLogit": calibration["validationThresholdLogit"],
        },
    }


def weights_readme(*, candidate_sha256: str, candidate_bytes: int) -> str:
    return f"""# Packaged detector

`prooflens-cf384.onnx` is the exact FP32 artifact described by the repository root [model-lock.json](../model-lock.json).

```text
bytes    {candidate_bytes:,}
sha256   {candidate_sha256}
input    pixel_values [N,3,384,384] float32
output   logits [N,1] float32
license  MIT
```

The build fails if the byte count or SHA-256 changes.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=Path("benchmark/candidates/prooflens-cf384-large"))
    parser.add_argument("--upstream", type=Path, default=Path("benchmark/candidates/upstream-cf384.onnx"))
    parser.add_argument(
        "--expected-upstream-sha256",
        default="a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1",
    )
    parser.add_argument("--shipped-model", type=Path, default=Path("weights/prooflens-cf384.onnx"))
    parser.add_argument("--model-lock", type=Path, default=Path("model-lock.json"))
    parser.add_argument("--weights-readme", type=Path, default=Path("weights/README.md"))
    parser.add_argument("--recipe", type=Path, default=Path("benchmark/large/recipe.json"))
    parser.add_argument(
        "--selection-summary",
        type=Path,
        default=Path("benchmark/evidence/large/selection-summary.json"),
    )
    parser.add_argument("--evidence-dir", type=Path, default=Path("benchmark/evidence/large"))
    args = parser.parse_args()

    sources = {
        "training-summary.json": args.candidate_dir / "validation-summary.json",
        "calibration.json": args.candidate_dir / "calibration.json",
        "candidate-grid.json": args.candidate_dir / "candidate-grid.json",
    }
    candidate_model = args.candidate_dir / "model.onnx"
    for path in [
        args.upstream,
        candidate_model,
        args.model_lock,
        args.recipe,
        args.selection_summary,
        *sources.values(),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if digest(args.upstream) != args.expected_upstream_sha256:
        raise ValueError("Pinned upstream model SHA-256 changed")

    summary = json.loads(sources["training-summary.json"].read_text())
    calibration = json.loads(sources["calibration.json"].read_text())
    grid = json.loads(sources["candidate-grid.json"].read_text())
    candidate_sha256 = digest(candidate_model)
    recipe_sha256 = digest(args.recipe)
    selection_summary_sha256 = digest(args.selection_summary)
    selection_summary = json.loads(args.selection_summary.read_text())
    if (
        summary.get("validationGatesPassed") is not True
        or summary.get("model", {}).get("sha256") != candidate_sha256
        or summary.get("model", {}).get("bytes") != candidate_model.stat().st_size
        or calibration.get("modelSha256") != candidate_sha256
        or summary.get("candidateCount") != 25
        or len(grid) != 25
        or summary.get("recipeSha256") != recipe_sha256
        or summary.get("selectionSummarySha256") != selection_summary_sha256
        or summary.get("trainManifestSha256") != selection_summary.get("manifestSha256")
    ):
        raise ValueError("Candidate output packet is internally inconsistent")
    trainer = Path("benchmark/modern/train_rehead.py")
    if summary.get("trainerSha256") != digest(trainer):
        raise ValueError("Training summary does not bind the current trainer")

    with tempfile.TemporaryDirectory(prefix=".prooflens-finalize-", dir=Path.cwd()) as temporary_name:
        temporary_root = Path(temporary_name)
        comparison_stage = temporary_root / "model-comparison.json"

        # The classifier-only structural proof is a precondition, not a post-publication check.
        subprocess.run(
            [
                sys.executable,
                "benchmark/compare_models.py",
                "--base", str(args.upstream),
                "--expected-base-sha256", args.expected_upstream_sha256,
                "--candidate", str(candidate_model),
                "--expected-candidate-sha256", candidate_sha256,
                "--output", str(comparison_stage),
            ],
            check=True,
        )
        if not comparison_stage.is_file():
            raise ValueError("Classifier-only comparison did not produce evidence")

        model_stage = temporary_root / "prooflens-cf384.onnx"
        verified_copy(candidate_model, model_stage)
        evidence_stages: dict[str, Path] = {}
        for name, source in sources.items():
            stage = temporary_root / name
            verified_copy(source, stage)
            evidence_stages[name] = stage

        model_lock_template = json.loads(args.model_lock.read_text())
        model_lock = build_model_lock(
            model_lock_template,
            candidate_sha256=candidate_sha256,
            candidate_bytes=candidate_model.stat().st_size,
            calibration=calibration,
            recipe_sha256=recipe_sha256,
            selection_summary_sha256=selection_summary_sha256,
            train_manifest_sha256=summary["trainManifestSha256"],
            training_summary_sha256=digest(sources["training-summary.json"]),
            calibration_sha256=digest(sources["calibration.json"]),
            candidate_grid_sha256=digest(sources["candidate-grid.json"]),
        )
        model_lock_stage = temporary_root / "model-lock.json"
        model_lock_stage.write_text(json.dumps(model_lock, indent=2) + "\n")
        weights_readme_stage = temporary_root / "weights-README.md"
        weights_readme_stage.write_text(
            weights_readme(
                candidate_sha256=candidate_sha256,
                candidate_bytes=candidate_model.stat().st_size,
            )
        )

        receipt = {
            "schemaVersion": 2,
            "candidateDirectory": str(args.candidate_dir),
            "upstreamSha256": args.expected_upstream_sha256,
            "shippedModel": {
                "path": str(args.shipped_model),
                "sha256": candidate_sha256,
                "bytes": candidate_model.stat().st_size,
            },
            "sourceEvidenceSha256": {name: digest(source) for name, source in sources.items()},
            "publishedEvidenceSha256": {
                **{name: digest(stage) for name, stage in evidence_stages.items()},
                "model-comparison.json": digest(comparison_stage),
            },
            "publishedRepositorySha256": {
                str(args.shipped_model): candidate_sha256,
                str(args.model_lock): digest(model_lock_stage),
                str(args.weights_readme): digest(weights_readme_stage),
            },
        }
        receipt_stage = temporary_root / "finalization-receipt.json"
        receipt_stage.write_text(json.dumps(receipt, indent=2) + "\n")

        # The receipt is deliberately last: its presence commits the complete packet.
        staged = [
            (model_stage, args.shipped_model),
            (model_lock_stage, args.model_lock),
            (weights_readme_stage, args.weights_readme),
            *[(stage, args.evidence_dir / name) for name, stage in evidence_stages.items()],
            (comparison_stage, args.evidence_dir / "model-comparison.json"),
            (receipt_stage, args.evidence_dir / "finalization-receipt.json"),
        ]
        publish_with_rollback(staged, temporary_root / "backups")

    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
