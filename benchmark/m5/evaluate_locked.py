#!/usr/bin/env python3
"""Run the two frozen M5 terminal regressions after the public selection lock."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.m5.contracts import (
    VARIANTS,
    canonical_json,
    choose_selector_threshold,
    digest_file,
    load_recipe,
    metrics_at_threshold,
    parse_json_bytes,
    read_jsonl,
    regression_gates_pass,
    regression_metrics,
    unpack_float32,
    validate_manifest_rows,
    validate_selection_lock,
)
from benchmark.m5.train_gpu import (
    Item,
    ImageDataset,
    ROOT,
    collate,
    load_items,
    pack_float32,
    require_canonical_path,
    verify_all_items,
)


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()


def require_lock_head(lock_commit: str, expected_lock_path: str) -> str:
    if len(lock_commit) != 40 or git("rev-parse", "HEAD") != lock_commit:
        raise ValueError("M5 terminal regression requires the exact public selection-lock commit")
    if git("status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("M5 terminal regression requires a clean tracked worktree")
    parents = git("rev-list", "--parents", "-n", "1", lock_commit).split()[1:]
    if len(parents) != 1:
        raise ValueError("M5 selection lock must have exactly one protocol parent")
    rows = [line for line in git(
        "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", lock_commit,
    ).splitlines() if line]
    if rows != [f"A\t{expected_lock_path}"]:
        raise ValueError("M5 selection-lock commit changed more than the canonical lock")
    return parents[0]


def selector_metrics_dict(logits: Mapping[str, Sequence[float]], rows: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    return {
        variant: {
            "balancedAccuracy": value.balanced_accuracy,
            "realRecall": value.real_recall,
            "syntheticRecall": value.synthetic_recall,
            "syntheticRecallBySource": value.synthetic_recall_by_source,
            "falsePositives": value.false_positives,
        }
        for variant in VARIANTS
        for value in [metrics_at_threshold(logits[variant], rows, threshold)]
    }


def replay_locked_selector(
    session: Any,
    items: Sequence[Item],
    rows: Sequence[dict[str, Any]],
    lock: Mapping[str, Any],
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    selected = next(
        candidate for candidate in lock["candidateGrid"]["candidates"]
        if candidate["candidateId"] == lock["selectedCandidateId"]
    )
    actual = {variant: predict(session, items, variant, recipe["training"]["workers"]) for variant in VARIANTS}
    tolerance = float(recipe["initialModel"]["maximumPytorchOnnxParityError"])
    maximum_deltas: dict[str, float] = {}
    for variant in VARIANTS:
        recorded = unpack_float32(selected["selectorLogits"][variant], expected_count=len(rows))
        maximum = max(abs(left - right) for left, right in zip(actual[variant], recorded, strict=True))
        if maximum > tolerance:
            raise ValueError(f"M5 locked ONNX selector replay changed: {variant} {maximum}")
        maximum_deltas[variant] = maximum
    locked_metrics = selector_metrics_dict(actual, rows, float(lock["rawThreshold"]))
    if locked_metrics != lock["selectorMetrics"]:
        raise ValueError("M5 locked ONNX does not reproduce selector decisions at the frozen threshold")
    replay_selection = choose_selector_threshold(actual, rows, recipe["selection"]["gates"])
    if replay_selection is None:
        raise ValueError("M5 locked ONNX fails the fresh-selector gates")
    replay_threshold, replay_metrics, _key = replay_selection
    if abs(float(replay_threshold) - float(lock["rawThreshold"])) > tolerance:
        raise ValueError("M5 locked ONNX selector threshold moved beyond the parity tolerance")
    if selector_metrics_dict(actual, rows, replay_threshold) != {
        variant: {
            "balancedAccuracy": value.balanced_accuracy,
            "realRecall": value.real_recall,
            "syntheticRecall": value.synthetic_recall,
            "syntheticRecallBySource": value.synthetic_recall_by_source,
            "falsePositives": value.false_positives,
        }
        for variant, value in replay_metrics.items()
    }:
        raise ValueError("M5 selector replay metrics are inconsistent")
    return {
        "selectorManifestSha256": recipe["sourceEvidence"]["selectorManifest"]["sha256"],
        "items": len(items),
        "maximumAbsoluteLogitDeltaByVariant": maximum_deltas,
        "parityTolerance": tolerance,
        "lockedThreshold": lock["rawThreshold"],
        "replayedBestThreshold": replay_threshold,
        "metricsAtLockedThreshold": locked_metrics,
        "passed": True,
    }


def predict(session: Any, items: Sequence[Item], variant: str, workers: int) -> list[float]:
    import torch
    from torch.utils.data import DataLoader

    dataset = ImageDataset(items, branch="regression", epoch=0, variant=variant, training=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate)
    values: list[float] = []
    for pixels, _labels, _weights, _anchors, _indexes in loader:
        output = session.run(["logits"], {"pixel_values": pixels.numpy()})[0]
        values.extend(float(value) for value in np.asarray(output, dtype=np.float32).reshape(-1))
    return values


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(canonical_json(value))
    os.replace(temporary, path)


def run_regression(
    regression: Mapping[str, Any],
    root_argument: str,
    manifest_argument: str,
    session: Any,
    threshold: float,
    workers: int,
) -> dict[str, Any]:
    data_root = require_canonical_path(root_argument, regression["dataRoot"], label=f"{regression['name']} data root")
    manifest_path = require_canonical_path(manifest_argument, regression["manifest"], label=f"{regression['name']} manifest")
    if digest_file(manifest_path) != regression["sha256"]:
        raise ValueError(f"M5 {regression['name']} manifest bytes changed")
    rows = read_jsonl(manifest_path)
    class_counts = Counter("real" if row["label"] == 0 else "synthetic" for row in rows)
    validate_manifest_rows(rows, expected_items=regression["items"], expected_class_counts=dict(class_counts))
    items = load_items(rows, data_root, [1.0] * len(rows))
    verify_all_items(items, workers=workers)
    logits = {variant: predict(session, items, variant, workers) for variant in VARIANTS}
    metrics = {variant: regression_metrics(logits[variant], rows, threshold) for variant in VARIANTS}
    return {
        "name": regression["name"],
        "manifestSha256": regression["sha256"],
        "items": len(rows),
        "metrics": metrics,
        "logits": {variant: pack_float32(logits[variant]) for variant in VARIANTS},
        "gates": regression["gates"],
        "passed": regression_gates_pass(metrics, regression["gates"]),
    }


def execute(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    recipe = load_recipe(ROOT / "benchmark/m5/recipe.json")
    lock_path = require_canonical_path(args.selection_lock, recipe["output"]["selectionLock"], label="selection lock")
    protocol_commit = require_lock_head(args.lock_commit, recipe["output"]["selectionLock"])
    lock = parse_json_bytes(lock_path.read_bytes(), label="selection lock")
    if lock.get("protocolCommit") != protocol_commit:
        raise ValueError("M5 selection lock is not bound to its exact protocol parent")
    selector_rows = read_jsonl(ROOT / recipe["sourceEvidence"]["selectorManifest"]["path"])
    validate_selection_lock(lock, recipe, selector_rows)
    model_path = ROOT / lock["selectedModel"]["path"]
    expected_root = (ROOT / recipe["output"]["candidateRoot"]).resolve(strict=True)
    resolved_model = model_path.resolve(strict=True)
    if expected_root not in resolved_model.parents:
        raise ValueError("M5 locked model escapes candidate root")
    if model_path.stat().st_size != lock["selectedModel"]["bytes"] or digest_file(model_path) != lock["selectedModel"]["sha256"]:
        raise ValueError("M5 locked model bytes changed")
    output_dir = require_canonical_path(args.output_dir, recipe["output"]["candidateRoot"], label="candidate output")
    state_path = output_dir / "regression-state.json"
    if state_path.exists():
        raise ValueError("M5 terminal regression state already exists; replay is forbidden")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ValueError("M5 terminal regression requires CUDAExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    selector_manifest = ROOT / recipe["sourceEvidence"]["selectorManifest"]["path"]
    if digest_file(selector_manifest) != recipe["sourceEvidence"]["selectorManifest"]["sha256"]:
        raise ValueError("M5 selector manifest bytes changed before ONNX replay")
    validate_manifest_rows(
        selector_rows,
        expected_items=recipe["sourceEvidence"]["selectorManifest"]["items"],
        expected_class_counts=recipe["sourceEvidence"]["selectorManifest"]["classCounts"],
        expected_source_counts=recipe["sourceEvidence"]["selectorManifest"]["sourceCounts"],
    )
    selector_root = require_canonical_path(
        recipe["sourceEvidence"]["dataRoot"], recipe["sourceEvidence"]["dataRoot"], label="selector data root",
    )
    selector_items = load_items(selector_rows, selector_root, [1.0] * len(selector_rows))
    verify_all_items(selector_items, workers=recipe["training"]["workers"])
    selector_replay = replay_locked_selector(session, selector_items, selector_rows, lock, recipe)
    results: list[dict[str, Any]] = []
    arguments = [
        (args.m3_data_root, args.m3_manifest),
        (args.m2_data_root, args.m2_manifest),
    ]
    for index, (regression, paths) in enumerate(zip(recipe["terminalRegressions"], arguments, strict=True)):
        result = run_regression(regression, paths[0], paths[1], session, float(lock["rawThreshold"]), workers=recipe["training"]["workers"])
        results.append(result)
        state = {
            "schemaVersion": 1,
            "status": "regression-pass" if result["passed"] and index == 1 else ("in-progress" if result["passed"] else "regression-fail"),
            "lockCommit": args.lock_commit,
            "selectionLockSha256": digest_file(lock_path),
            "selectedCandidateId": lock["selectedCandidateId"],
            "selectedModelSha256": lock["selectedModel"]["sha256"],
            "rawThreshold": lock["rawThreshold"],
            "selectorOnnxReplay": selector_replay,
            "results": results,
            "selectionInfluencedByRegression": False,
            "h3PixelsRead": False,
        }
        write_json(state_path, state)
        if not result["passed"]:
            print(json.dumps({"event": "terminal-regression-fail", "name": regression["name"], "h3PixelsRead": False}), flush=True)
            return 2
    print(json.dumps({"event": "terminal-regressions-pass", "selectedCandidateId": lock["selectedCandidateId"], "h3PixelsRead": False}), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--lock-commit", required=True)
    value.add_argument("--selection-lock", default="benchmark/evidence/m5/selection-lock.json")
    value.add_argument("--m3-data-root", default="benchmark/data/m3-head")
    value.add_argument("--m3-manifest", default="benchmark/evidence/m3/validation-manifest.jsonl")
    value.add_argument("--m2-data-root", default="benchmark/data/m2-head")
    value.add_argument("--m2-manifest", default="benchmark/evidence/m2/validation-manifest.jsonl")
    value.add_argument("--output-dir", default="benchmark/candidates/prooflens-cf384-m5")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(execute(parser().parse_args()))
    except Exception as error:
        print(f"M5 locked regression failed: {error}", file=sys.stderr)
        raise
