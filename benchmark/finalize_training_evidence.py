"""Transactionally publish a validated classifier-head candidate and evidence packet."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


UPSTREAM_SHA256 = "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1"
HEX64 = frozenset("0123456789abcdef")
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")


@dataclass(frozen=True)
class PublicationProfile:
    name: str
    identity: str
    candidate_dir: Path
    upstream: Path
    shipped_model: Path
    model_lock: Path
    weights_readme: Path
    recipe: Path
    selection_summary: Path
    evidence_dir: Path
    pipeline_version: int
    train_images: int
    train_views: int
    validation_images: int
    validation_views: int
    selection_key_length: int
    receipt_schema_version: int
    validation_data_root: str
    validation_manifest: str
    expected_model_sha256: str | None = None
    expected_training_summary_sha256: str | None = None
    expected_calibration_sha256: str | None = None
    expected_candidate_grid_sha256: str | None = None
    expected_model_comparison_sha256: str | None = None
    expected_fresh_run_id: str | None = None
    expected_valid_candidates: int | None = None


PROFILES = {
    "m1": PublicationProfile(
        name="m1",
        identity="prooflens-cf384-large-head-v1",
        candidate_dir=Path("benchmark/candidates/prooflens-cf384-large"),
        upstream=Path("benchmark/candidates/upstream-cf384.onnx"),
        shipped_model=Path("weights/prooflens-cf384.onnx"),
        model_lock=Path("model-lock.json"),
        weights_readme=Path("weights/README.md"),
        recipe=Path("benchmark/large/recipe.json"),
        selection_summary=Path("benchmark/evidence/large/selection-summary.json"),
        evidence_dir=Path("benchmark/evidence/large"),
        pipeline_version=8,
        train_images=103_600,
        train_views=114_400,
        validation_images=600,
        validation_views=2_400,
        selection_key_length=4,
        receipt_schema_version=2,
        validation_data_root="benchmark/data/modern-head",
        validation_manifest="benchmark/manifests/validation.jsonl",
    ),
    "m2": PublicationProfile(
        name="m2",
        identity="prooflens-cf384-m2-head-v1",
        candidate_dir=Path("benchmark/candidates/prooflens-cf384-m2"),
        upstream=Path("benchmark/candidates/upstream-cf384.onnx"),
        shipped_model=Path("weights/prooflens-cf384.onnx"),
        model_lock=Path("model-lock.json"),
        weights_readme=Path("weights/README.md"),
        recipe=Path("benchmark/m2/recipe.json"),
        selection_summary=Path("benchmark/evidence/m2/selection-summary.json"),
        evidence_dir=Path("benchmark/evidence/m2"),
        pipeline_version=9,
        train_images=105_978,
        train_views=123_912,
        validation_images=900,
        validation_views=3_600,
        selection_key_length=5,
        receipt_schema_version=3,
        validation_data_root="benchmark/data/m2-head",
        validation_manifest="benchmark/evidence/m2/validation-manifest.jsonl",
        expected_model_sha256="a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47",
        expected_training_summary_sha256="c3d49719e50b1fbf5fdc9ba5b8c1df57712910af0f0284a3c3acdf6bad931c04",
        expected_calibration_sha256="06d2452a8db9de26d42285cdc9dad0d233d397a6015583604c64480aec560e2c",
        expected_candidate_grid_sha256="7ac1028543607a94af88a95af585bdd973205849aa20bbf841f656598b4afe1c",
        expected_model_comparison_sha256="7e037912f28a69ac7ea9620471f1410b7b1ab445b7bb30ce9d7bdbe0c24f96ac",
        expected_fresh_run_id="add5d5306942c5c729c97556bd61cabd",
        expected_valid_candidates=24,
    ),
}


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
    """Publish staged bytes; restore every possibly touched target on an exception."""
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
            # Record the target before replacement. An asynchronous BaseException can
            # occur after os.replace mutates the destination but before the call returns.
            published.append(destination)
            replace_from_stage(stage, destination)
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
    training_recipe_identity: str = "prooflens-cf384-large-head-v1",
    recipe_path: str = "benchmark/large/recipe.json",
    selection_summary_path: str = "benchmark/evidence/large/selection-summary.json",
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
        "trainingRecipe": f"{training_recipe_identity}:{recipe_sha256}:{selection_summary_sha256}",
        "trainingEvidence": {
            "recipe": recipe_path,
            "recipeSha256": recipe_sha256,
            "selectionSummary": selection_summary_path,
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


def is_hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def expected_training_arguments(profile: PublicationProfile) -> list[str]:
    return [
        "--model", str(profile.upstream),
        "--expected-model-sha256", UPSTREAM_SHA256,
        "--data-root", "benchmark/data/large-head" if profile.name == "m1" else "benchmark/data/m2-head",
        "--train-manifest", (
            "benchmark/data/large-head/train-manifest.jsonl" if profile.name == "m1"
            else "benchmark/data/m2-head/train-manifest.jsonl"
        ),
        "--validation-data-root", profile.validation_data_root,
        "--validation-manifest", profile.validation_manifest,
        "--recipe", str(profile.recipe),
        "--selection-summary", (
            "benchmark/data/large-head/selection-summary.json" if profile.name == "m1"
            else "benchmark/data/m2-head/selection-summary.json"
        ),
        "--single-view-source", "diffusiondb-stable-diffusion",
        "--single-view-source", "open-images-train",
        "--execution-provider", "cpu",
        "--batch-size", "24",
        "--feature-shard-images", "2000",
        "--reextract-cached-features",
        "--output-dir", str(profile.candidate_dir),
    ]


def require_variant_gates(
    variants: object,
    gates: dict[str, object],
    *,
    label: str,
) -> None:
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise ValueError(f"{label} variant set changed")
    for variant in VARIANTS:
        metrics = variants[variant]
        if not isinstance(metrics, dict):
            raise ValueError(f"{label} {variant} metrics are malformed")
        requirements = (
            ("balancedAccuracy", "minimumBalancedAccuracyPerVariant"),
            ("realRecall", "minimumRealRecallPerVariant"),
            ("syntheticRecall", "minimumSyntheticRecallPerVariant"),
        )
        for metric, gate in requirements:
            value = metrics.get(metric)
            minimum = gates.get(gate)
            if not finite_number(value) or not finite_number(minimum) or value < minimum:
                raise ValueError(f"{label} {variant} failed {gate}")
        family_minimum = gates.get("minimumSyntheticRecallPerFamily")
        family_metrics = metrics.get("syntheticRecallBySource")
        if not isinstance(family_metrics, dict) or not family_metrics:
            raise ValueError(f"{label} {variant} lacks synthetic family metrics")
        if not finite_number(family_minimum) or any(
            not finite_number(value) or value < family_minimum for value in family_metrics.values()
        ):
            raise ValueError(f"{label} {variant} failed the synthetic-family gate")
        source_minimum = gates.get("minimumRealRecallBySource", {})
        source_metrics = metrics.get("realRecallBySource")
        if source_minimum:
            if not isinstance(source_metrics, dict):
                raise ValueError(f"{label} {variant} lacks real-source metrics")
            for source, minimum in source_minimum.items():
                value = source_metrics.get(source)
                if not finite_number(value) or not finite_number(minimum) or value < minimum:
                    raise ValueError(f"{label} {variant} failed the {source} real-recall gate")


def validate_candidate_packet(
    profile: PublicationProfile,
    *,
    summary: dict[str, object],
    calibration: dict[str, object],
    grid: list[dict[str, object]],
    recipe: dict[str, object],
    selection_summary: dict[str, object],
    candidate_model: Path,
    recipe_sha256: str,
    selection_summary_sha256: str,
) -> None:
    candidate_sha256 = digest(candidate_model)
    trainer = Path("benchmark/modern/train_rehead.py")
    if summary.get("schemaVersion") != 1 or summary.get("pipelineVersion") != profile.pipeline_version:
        raise ValueError("Candidate training schema or pipeline version changed")
    if summary.get("trainerSha256") != digest(trainer):
        raise ValueError("Training summary does not bind the current trainer")
    if summary.get("commandArguments") != expected_training_arguments(profile):
        raise ValueError("Training command does not match the closed publication profile")
    exact_outputs = (
        (candidate_sha256, profile.expected_model_sha256, "candidate model"),
        (digest(profile.candidate_dir / "validation-summary.json"), profile.expected_training_summary_sha256,
         "training summary"),
        (digest(profile.candidate_dir / "calibration.json"), profile.expected_calibration_sha256, "calibration"),
        (digest(profile.candidate_dir / "candidate-grid.json"), profile.expected_candidate_grid_sha256,
         "candidate grid"),
    )
    for actual, expected, label_name in exact_outputs:
        if expected is not None and actual != expected:
            raise ValueError(f"The reviewed M2 {label_name} bytes changed")
    if summary.get("recipeSha256") != recipe_sha256 or summary.get("selectionSummarySha256") != selection_summary_sha256:
        raise ValueError("Candidate recipe or selection binding changed")
    if summary.get("trainManifestSha256") != selection_summary.get("manifestSha256"):
        raise ValueError("Candidate train manifest is not bound to the selection summary")
    if (
        summary.get("trainImages") != profile.train_images
        or summary.get("trainFeatureViews") != profile.train_views
        or summary.get("validationImages") != profile.validation_images
        or summary.get("validationFeatureViews") != profile.validation_views
        or summary.get("uniqueTrainingImagesCovered") != profile.train_images
        or summary.get("uniqueTrainingFeatureViewsCovered") != profile.train_views
    ):
        raise ValueError("Candidate image or feature-view coverage changed")
    if (
        summary.get("sourceBalancedSampling") is not False
        or summary.get("sourceBalancedLoss") is not True
        or summary.get("trainingEpochs") != 12
        or summary.get("trainingBatchSize") != 2_048
        or summary.get("featureBatchSize") != 24
        or summary.get("featureShardImages") != 2_000
        or summary.get("cachedFeatureSourcePixelsReverified") is not True
        or summary.get("cachedFeatureArraysValidated") is not True
        or summary.get("cachedFeatureValuesReextracted") is not True
    ):
        raise ValueError("Candidate training procedure changed")
    fresh_run = summary.get("freshFeatureRun")
    shard_rows = summary.get("featureShardEvidence")
    if not isinstance(fresh_run, dict) or fresh_run.get("state") != "complete" or not isinstance(shard_rows, list):
        raise ValueError("Fresh-feature evidence is incomplete")
    run_id = fresh_run.get("runId")
    expected_shards = (profile.train_images + 1_999) // 2_000 + 1
    if len(shard_rows) != expected_shards or not isinstance(run_id, str) or len(run_id) != 32:
        raise ValueError("Fresh-feature shard count or run ID changed")
    if profile.expected_fresh_run_id is not None and run_id != profile.expected_fresh_run_id:
        raise ValueError("The reviewed M2 fresh-feature run ID changed")
    feature_hashes = summary.get("featureConfigurationHashes")
    expected_fresh_context = {
        "pipelineVersion": profile.pipeline_version,
        "featureExtractorContract": "cf384-static-batch24-preprocess-v1",
        "upstreamModelSha256": UPSTREAM_SHA256,
        "trainManifestSha256": summary.get("trainManifestSha256"),
        "validationManifestSha256": summary.get("validationManifestSha256"),
        "selectionSummarySha256": selection_summary_sha256,
        "featureBatchSize": 24,
        "featureShardImages": 2_000,
        "singleViewTrainingSources": ["diffusiondb-stable-diffusion", "open-images-train"],
        "featureConfigurationHashes": feature_hashes,
    }
    marker = profile.candidate_dir / "fresh-feature-run.json"
    if (
        not marker.is_file()
        or digest(marker) != summary.get("freshFeatureRunMarkerSha256")
        or json.loads(marker.read_text()) != fresh_run
        or fresh_run.get("context") != expected_fresh_context
        or not isinstance(feature_hashes, dict)
        or not all(is_hex64(feature_hashes.get(name)) for name in ("training", "validation"))
        or feature_hashes.get("training") == feature_hashes.get("validation")
    ):
        raise ValueError("Fresh-feature run marker or extraction context changed")
    if sum(row.get("items", 0) for row in shard_rows) != profile.train_images + profile.validation_images:
        raise ValueError("Fresh-feature shards do not cover every source image")
    if sum(row.get("views", 0) for row in shard_rows) != profile.train_views + profile.validation_views:
        raise ValueError("Fresh-feature shards do not cover every feature view")
    expected_shard_names = [
        *(str(profile.candidate_dir / "features" / f"train-{index:05d}.npz")
          for index in range((profile.train_images + 1_999) // 2_000)),
        str(profile.candidate_dir / "features" / "validation-00000.npz"),
    ]
    if [row.get("cache") for row in shard_rows] != expected_shard_names:
        raise ValueError("Fresh-feature shard paths changed")
    cache_hashes: set[str] = set()
    item_id_hashes: set[str] = set()
    for row in shard_rows:
        is_training = "/train-" in row.get("cache", "")
        if (
            row.get("freshFeatureRunId") != run_id
            or row.get("freshlyExtractedThisRun") is not True
            or row.get("freshlyExtractedThisProcess") is not True
            or row.get("replacedCacheSha256") is not None
            or not is_hex64(row.get("cacheSha256"))
            or not is_hex64(row.get("itemIdsSha256"))
            or row.get("featureConfigurationSha256") != feature_hashes[
                "training" if is_training else "validation"
            ]
            or not all(is_hex64(row.get("arraySha256", {}).get(name)) for name in ("features", "labels", "variants", "sources"))
        ):
            raise ValueError("A feature shard is not bound to the complete fresh run")
        cache_hashes.add(row["cacheSha256"])
        item_id_hashes.add(row["itemIdsSha256"])
    if len(cache_hashes) != len(shard_rows) or len(item_id_hashes) != len(shard_rows):
        raise ValueError("Fresh-feature shard or item-ID digests are duplicated")
    environment = summary.get("environment", {})
    if (
        environment.get("executionProvider") != "cpu"
        or environment.get("providers") != ["CPUExecutionProvider"]
        or environment.get("torchDeterministicAlgorithms") is not True
        or environment.get("cuda") is not None
        or environment.get("gpu") is not None
    ):
        raise ValueError("Candidate was not produced by the frozen deterministic CPU profile")
    if summary.get("validationGates") != recipe.get("validationGates") or summary.get("validationGatesPassed") is not True:
        raise ValueError("Candidate validation gates changed or failed")
    if not isinstance(grid, list) or len(grid) != 25 or summary.get("candidateCount") != 25:
        raise ValueError("Candidate grid must contain all 25 predeclared pairs")
    expected_parameters = {
        (weight_decay, upstream_blend_alpha)
        for weight_decay in (0.1, 0.03, 0.01, 0.003, 0.001)
        for upstream_blend_alpha in (0.4, 0.55, 0.7, 0.85, 1)
    }
    parameter_rows = [
        (row.get("parameters", {}).get("weightDecay"), row.get("parameters", {}).get("upstreamBlendAlpha"))
        for row in grid
    ]
    if len(set(parameter_rows)) != 25 or set(parameter_rows) != expected_parameters:
        raise ValueError("Candidate grid parameter coverage changed")
    valid = [row for row in grid if isinstance(row.get("selectionKey"), list)]
    if (
        len(valid) != summary.get("validCandidateCount")
        or not valid
        or (profile.expected_valid_candidates is not None and len(valid) != profile.expected_valid_candidates)
    ):
        raise ValueError("Candidate valid-grid count changed")
    rejected = [row for row in grid if not isinstance(row.get("selectionKey"), list)]
    if any(
        row.get("status") != "rejected"
        or not isinstance(row.get("reason"), str)
        or not row["reason"]
        for row in rejected
    ):
        raise ValueError("A rejected grid candidate lacks a reason")
    if profile.name == "m2" and (
        len(rejected) != 1
        or rejected[0].get("parameters") != {"weightDecay": 0.1, "upstreamBlendAlpha": 0.4}
        or rejected[0].get("reason") != "No threshold satisfies the frozen validation gates"
    ):
        raise ValueError("The reviewed M2 rejected grid candidate changed")
    for row in valid:
        key = row["selectionKey"]
        if (
            len(key) != profile.selection_key_length
            or not all(finite_number(value) for value in key)
            or not finite_number(row.get("thresholdLogit"))
        ):
            raise ValueError("Candidate selection key is malformed")
        require_variant_gates(row.get("variants"), recipe["validationGates"], label="grid candidate")
    selected = next((row for row in valid if row.get("parameters") == summary.get("selectedParameters")), None)
    deterministic_best = max(valid, key=lambda row: tuple(row["selectionKey"]))
    if (
        selected is None
        or selected is not deterministic_best
        or selected.get("selectionKey") != summary.get("selectionKey")
        or selected.get("thresholdLogit") != summary.get("thresholdLogit")
        or selected.get("variants") != summary.get("variants")
    ):
        raise ValueError("Selected candidate is not the deterministic grid maximum")
    require_variant_gates(summary.get("variants"), recipe["validationGates"], label="selected candidate")
    model = summary.get("model", {})
    if (
        model.get("sha256") != candidate_sha256
        or model.get("bytes") != candidate_model.stat().st_size
        or not finite_number(model.get("maxAbsParityError"))
        or model["maxAbsParityError"] > 2e-4
    ):
        raise ValueError("Candidate model hash, size, or export parity changed")
    if (
        calibration.get("schemaVersion") != 1
        or calibration.get("modelSha256") != candidate_sha256
        or calibration.get("trainManifestSha256") != summary.get("trainManifestSha256")
        or calibration.get("validationManifestSha256") != summary.get("validationManifestSha256")
        or calibration.get("selectionSummarySha256") != selection_summary_sha256
        or calibration.get("slope") != 1
        or calibration.get("displayThreshold") != 0.65
        or calibration.get("validationThresholdLogit") != summary.get("thresholdLogit")
        or not all(finite_number(calibration.get(name)) for name in (
            "slope", "intercept", "displayThreshold", "validationThresholdLogit", "rawProbabilityThreshold"
        ))
        or not 0 < calibration.get("rawProbabilityThreshold") < 1
    ):
        raise ValueError("Candidate calibration is not bound to the selected model and fixed display threshold")


def validate_model_comparison(
    comparison: dict[str, object],
    *,
    profile: PublicationProfile,
    candidate_sha256: str,
    candidate_bytes: int,
) -> None:
    changed = comparison.get("changedInitializers")
    if (
        comparison.get("schemaVersion") != 1
        or comparison.get("base", {}).get("path") != str(profile.upstream)
        or comparison.get("base", {}).get("sha256") != UPSTREAM_SHA256
        or comparison.get("candidate", {}).get("path") != str(profile.candidate_dir / "model.onnx")
        or comparison.get("candidate", {}).get("sha256") != candidate_sha256
        or comparison.get("candidate", {}).get("bytes") != candidate_bytes
        or not isinstance(changed, list)
        or {row.get("name") for row in changed} != {"classifier.weight", "classifier.bias"}
        or comparison.get("unchangedInitializerCount", 0) <= 0
        or not all(is_hex64(comparison.get(name)) for name in (
            "graphNodesSha256", "graphInputsSha256", "graphOutputsSha256", "opsetsSha256"
        ))
    ):
        raise ValueError("Classifier-only model comparison is malformed")
    expected_dimensions = {"classifier.weight": [1, 384], "classifier.bias": [1]}
    for row in changed:
        if (
            row.get("dimensions") != expected_dimensions[row["name"]]
            or not is_hex64(row.get("beforeSha256"))
            or not is_hex64(row.get("afterSha256"))
            or row.get("beforeSha256") == row.get("afterSha256")
        ):
            raise ValueError("Classifier-only initializer evidence is malformed")


def require_clean_tracked_tree() -> None:
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"], text=True
    ).strip()
    if status:
        raise ValueError("Training finalization requires a clean tracked worktree")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="m1")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    require_clean_tracked_tree()

    sources = {
        "training-summary.json": profile.candidate_dir / "validation-summary.json",
        "calibration.json": profile.candidate_dir / "calibration.json",
        "candidate-grid.json": profile.candidate_dir / "candidate-grid.json",
    }
    candidate_model = profile.candidate_dir / "model.onnx"
    for path in [
        profile.upstream,
        candidate_model,
        profile.model_lock,
        profile.recipe,
        profile.selection_summary,
        *sources.values(),
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if digest(profile.upstream) != UPSTREAM_SHA256:
        raise ValueError("Pinned upstream model SHA-256 changed")

    summary = json.loads(sources["training-summary.json"].read_text())
    calibration = json.loads(sources["calibration.json"].read_text())
    grid = json.loads(sources["candidate-grid.json"].read_text())
    candidate_sha256 = digest(candidate_model)
    recipe_sha256 = digest(profile.recipe)
    selection_summary_sha256 = digest(profile.selection_summary)
    recipe = json.loads(profile.recipe.read_text())
    selection_summary = json.loads(profile.selection_summary.read_text())
    validate_candidate_packet(
        profile,
        summary=summary,
        calibration=calibration,
        grid=grid,
        recipe=recipe,
        selection_summary=selection_summary,
        candidate_model=candidate_model,
        recipe_sha256=recipe_sha256,
        selection_summary_sha256=selection_summary_sha256,
    )

    with tempfile.TemporaryDirectory(prefix=".prooflens-finalize-", dir=Path.cwd()) as temporary_name:
        temporary_root = Path(temporary_name)
        comparison_stage = temporary_root / "model-comparison.json"

        # The classifier-only structural proof is a precondition, not a post-publication check.
        subprocess.run(
            [
                sys.executable,
                "benchmark/compare_models.py",
                "--base", str(profile.upstream),
                "--expected-base-sha256", UPSTREAM_SHA256,
                "--candidate", str(candidate_model),
                "--expected-candidate-sha256", candidate_sha256,
                "--output", str(comparison_stage),
            ],
            check=True,
        )
        if not comparison_stage.is_file():
            raise ValueError("Classifier-only comparison did not produce evidence")
        comparison = json.loads(comparison_stage.read_text())
        validate_model_comparison(
            comparison,
            profile=profile,
            candidate_sha256=candidate_sha256,
            candidate_bytes=candidate_model.stat().st_size,
        )
        if (
            profile.expected_model_comparison_sha256 is not None
            and digest(comparison_stage) != profile.expected_model_comparison_sha256
        ):
            raise ValueError("The reviewed M2 classifier-only comparison bytes changed")

        model_stage = temporary_root / "prooflens-cf384.onnx"
        verified_copy(candidate_model, model_stage)
        evidence_stages: dict[str, Path] = {}
        for name, source in sources.items():
            stage = temporary_root / name
            verified_copy(source, stage)
            evidence_stages[name] = stage

        model_lock_template = json.loads(profile.model_lock.read_text())
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
            training_recipe_identity=profile.identity,
            recipe_path=str(profile.recipe),
            selection_summary_path=str(profile.selection_summary),
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
            "schemaVersion": profile.receipt_schema_version,
            "profile": profile.name,
            "candidateDirectory": str(profile.candidate_dir),
            "upstreamSha256": UPSTREAM_SHA256,
            "shippedModel": {
                "path": str(profile.shipped_model),
                "sha256": candidate_sha256,
                "bytes": candidate_model.stat().st_size,
            },
            "sourceEvidenceSha256": {name: digest(source) for name, source in sources.items()},
            "publishedEvidenceSha256": {
                **{name: digest(stage) for name, stage in evidence_stages.items()},
                "model-comparison.json": digest(comparison_stage),
            },
            "publishedRepositorySha256": {
                str(profile.shipped_model): candidate_sha256,
                str(profile.model_lock): digest(model_lock_stage),
                str(profile.weights_readme): digest(weights_readme_stage),
            },
        }
        receipt_stage = temporary_root / "finalization-receipt.json"
        receipt_stage.write_text(json.dumps(receipt, indent=2) + "\n")

        # The receipt is deliberately last: its presence commits the complete packet.
        staged = [
            (model_stage, profile.shipped_model),
            (model_lock_stage, profile.model_lock),
            (weights_readme_stage, profile.weights_readme),
            *[(stage, profile.evidence_dir / name) for name, stage in evidence_stages.items()],
            (comparison_stage, profile.evidence_dir / "model-comparison.json"),
            (receipt_stage, profile.evidence_dir / "finalization-receipt.json"),
        ]
        if not args.check_only:
            publish_with_rollback(staged, temporary_root / "backups")

    print(json.dumps({**receipt, "checkOnly": args.check_only}, indent=2))


if __name__ == "__main__":
    main()
