#!/usr/bin/env python3
"""Score the public, fixed 100,000-image SeroSlop M5 synthetic panel once."""

from __future__ import annotations

import argparse
from base64 import b64encode
from base64 import b64decode
from collections import Counter, defaultdict
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.m5.contracts import (
    canonical_json,
    digest_file,
    load_recipe,
    parse_json_bytes,
    read_jsonl,
    validate_regression_state,
    validate_selection_lock,
)
from benchmark.m5.large_synthetic import DATA_ROOT, verify_public_packet
from benchmark.m5.train_gpu import Item, preprocess_image


ROOT = REPOSITORY_ROOT
RECIPE_PATH = ROOT / "benchmark/m5/recipe.json"


def run(command: Sequence[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def commit_rows(commit: str) -> list[tuple[str, str]]:
    output = run(["git", "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
    rows = []
    for line in output.splitlines():
        status, path = line.split("\t", maxsplit=1)
        rows.append((path, status))
    return rows


def require_source_lock_head(recipe: Mapping[str, Any], requested: str) -> tuple[str, str]:
    head = run(["git", "rev-parse", "HEAD"])
    if head != requested or run(["git", "status", "--porcelain=v1", "--untracked-files=no"]):
        raise ValueError("M5 large-synthetic evaluation requires the exact clean public source-lock commit")
    parents = run(["git", "rev-list", "--parents", "-n", "1", head]).split()[1:]
    expected = sorted((path, "A") for path in (
        recipe["largeSyntheticEvaluation"]["manifest"],
        recipe["largeSyntheticEvaluation"]["batchAssignment"],
        recipe["largeSyntheticEvaluation"]["sourceLock"],
        recipe["largeSyntheticEvaluation"]["attribution"],
    ))
    if len(parents) != 1 or sorted(commit_rows(head)) != expected:
        raise ValueError("M5 large-synthetic source lock has the wrong commit shape")
    lock_commit = parents[0]
    lock_parents = run(["git", "rev-list", "--parents", "-n", "1", lock_commit]).split()[1:]
    if len(lock_parents) != 1 or commit_rows(lock_commit) != [(recipe["output"]["selectionLock"], "A")]:
        raise ValueError("M5 large-synthetic source lock is not the direct child of the one-file selection lock")
    return lock_commit, lock_parents[0]


def pack_float32(values: Sequence[float]) -> dict[str, Any]:
    payload = np.asarray(values, dtype="<f4").tobytes(order="C")
    return {"dtype": "float32-little-endian", "count": len(values), "sha256": sha256(payload).hexdigest(), "base64": b64encode(payload).decode("ascii")}


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise ValueError("M5 large-synthetic evaluation evidence already exists; replay is forbidden")
    with temporary.open("wb") as handle:
        handle.write(canonical_json(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_rows(recipe: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = ROOT / recipe["largeSyntheticEvaluation"]["manifest"]
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def score_batch(session: Any, rows: Sequence[dict[str, Any]]) -> list[float]:
    arrays: list[np.ndarray] = []
    for row in rows:
        path = DATA_ROOT / row["path"]
        item = Item(
            id=row["id"], path=path, image_sha256=row["imageSha256"], label=1,
            source="omni-fake-set", row_index=row["rowIndex"], weight=1.0, anchor=False,
        )
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        arrays.append(preprocess_image(image, item, "original", training=False, branch="large-synthetic", epoch=0))
    pixels = np.stack(arrays).astype(np.float32)
    logits = session.run(["logits"], {"pixel_values": pixels})[0].reshape(-1)
    return [float(value) for value in logits]


def validate_evaluation_receipt(
    receipt: Mapping[str, Any],
    recipe: Mapping[str, Any],
    *,
    selection_lock: Mapping[str, Any],
    source_lock_commit: str,
    panel_rows: Sequence[Mapping[str, Any]],
    verify_artifact_bindings: bool = True,
    regression_state_path: Path | None = None,
) -> None:
    required = {
        "schemaVersion", "status", "acceptanceEligible", "sourceLockCommit", "selectionLockCommit",
        "selectionLockSha256", "sourceLockSha256", "manifestSha256", "batchAssignmentSha256",
        "model", "rawThreshold", "items", "batchSize", "batches", "correct", "overallRecall",
        "meanBatchRecall", "medianBatchRecall", "minimumBatchRecall", "wilson95", "batchResults",
        "generatorResults", "minimumGeneratorRecall", "logits", "selectionInfluence", "regressionStateSha256",
        "h3PixelsRead",
    }
    if set(receipt) != required:
        raise ValueError("M5 large-synthetic evaluation receipt schema changed")
    large = recipe["largeSyntheticEvaluation"]
    selection_path = ROOT / recipe["output"]["selectionLock"]
    source_lock_path = ROOT / large["sourceLock"]
    manifest_path = ROOT / large["manifest"]
    batches_path = ROOT / large["batchAssignment"]
    bound_regression_path = regression_state_path or (ROOT / recipe["output"]["candidateRoot"] / "regression-state.json")
    passed = receipt["meanBatchRecall"] > large["minimumMeanBatchRecallExclusive"] and receipt["medianBatchRecall"] > large["minimumMedianBatchRecallExclusive"]
    if (
        receipt["schemaVersion"] != 1 or receipt["sourceLockCommit"] != source_lock_commit or
        (verify_artifact_bindings and receipt["selectionLockSha256"] != digest_file(selection_path)) or
        (verify_artifact_bindings and receipt["sourceLockSha256"] != digest_file(source_lock_path)) or
        (verify_artifact_bindings and receipt["manifestSha256"] != digest_file(manifest_path)) or
        (verify_artifact_bindings and receipt["batchAssignmentSha256"] != digest_file(batches_path)) or
        (verify_artifact_bindings and receipt["regressionStateSha256"] != digest_file(bound_regression_path)) or
        receipt["model"] != selection_lock["selectedModel"] or receipt["rawThreshold"] != selection_lock["rawThreshold"] or
        receipt["items"] != 100_000 or receipt["batchSize"] != 100 or receipt["batches"] != 1_000 or
        receipt["selectionInfluence"] is not False or receipt["h3PixelsRead"] is not False or
        receipt["status"] != ("large-synthetic-pass" if passed else "large-synthetic-fail") or
        receipt["acceptanceEligible"] is not passed
    ):
        raise ValueError("M5 large-synthetic evaluation boundary changed")
    batch_results = receipt["batchResults"]
    if len(batch_results) != 1_000 or [row.get("batchIndex") for row in batch_results] != list(range(1_000)):
        raise ValueError("M5 large-synthetic batch coverage changed")
    recalls = []
    correct = 0
    for row in batch_results:
        if set(row) != {"batchIndex", "items", "correct", "recall"} or row["items"] != 100 or row["recall"] != row["correct"] / 100:
            raise ValueError("M5 large-synthetic batch metric changed")
        correct += row["correct"]
        recalls.append(row["recall"])
    if (
        receipt["correct"] != correct or receipt["overallRecall"] != correct / 100_000 or
        receipt["meanBatchRecall"] != statistics.fmean(recalls) or
        receipt["medianBatchRecall"] != statistics.median(recalls) or
        receipt["minimumBatchRecall"] != min(recalls)
    ):
        raise ValueError("M5 large-synthetic aggregate metric changed")
    lower, upper = wilson(correct, 100_000)
    if receipt["wilson95"] != {"lower": lower, "upper": upper}:
        raise ValueError("M5 large-synthetic Wilson interval changed")
    logits = receipt["logits"]
    if set(logits) != {"dtype", "count", "sha256", "base64"} or logits["count"] != 100_000:
        raise ValueError("M5 large-synthetic logit packet changed")
    try:
        payload = b64decode(str(logits["base64"]), validate=True)
    except Exception as error:
        raise ValueError("M5 large-synthetic logit packet base64 changed") from error
    if logits["dtype"] != "float32-little-endian" or len(payload) != 400_000 or sha256(payload).hexdigest() != logits["sha256"]:
        raise ValueError("M5 large-synthetic logit bytes changed")
    values = np.frombuffer(payload, dtype="<f4")
    if not np.isfinite(values).all():
        raise ValueError("M5 large-synthetic logits contain non-finite values")
    recomputed_batch = []
    for batch_index in range(1_000):
        batch_values = values[batch_index * 100:(batch_index + 1) * 100]
        recomputed_batch.append(int(np.count_nonzero(batch_values >= float(receipt["rawThreshold"]))))
    if [row["correct"] for row in batch_results] != recomputed_batch:
        raise ValueError("M5 large-synthetic batch decisions do not match logits")
    ordered_rows = [row for batch_index in range(1_000) for row in panel_rows if row["batchIndex"] == batch_index]
    if len(ordered_rows) != 100_000:
        raise ValueError("M5 large-synthetic panel ordering changed")
    generator_totals: Counter[str] = Counter()
    generator_correct: Counter[str] = Counter()
    for row, value in zip(ordered_rows, values, strict=True):
        generator_totals[str(row["generator"])] += 1
        generator_correct[str(row["generator"])] += int(value >= float(receipt["rawThreshold"]))
    expected_generator_results = {
        generator: {"items": total, "correct": generator_correct[generator], "recall": generator_correct[generator] / total}
        for generator, total in sorted(generator_totals.items())
    }
    if receipt["generatorResults"] != expected_generator_results or receipt["minimumGeneratorRecall"] != min(value["recall"] for value in expected_generator_results.values()):
        raise ValueError("M5 large-synthetic generator metrics changed")


def execute(args: argparse.Namespace) -> int:
    import onnxruntime as ort

    recipe = load_recipe(RECIPE_PATH)
    lock_commit, protocol_commit = require_source_lock_head(recipe, args.source_lock_commit)
    public = verify_public_packet(recipe, verify_pixels=True)
    selection_path = ROOT / recipe["output"]["selectionLock"]
    selection = parse_json_bytes(selection_path.read_bytes(), label="selection lock")
    selector_rows = read_jsonl(ROOT / recipe["sourceEvidence"]["selectorManifest"]["path"])
    validate_selection_lock(selection, recipe, selector_rows)
    if selection["protocolCommit"] != protocol_commit:
        raise ValueError("M5 large-synthetic evaluation selection ancestry changed")
    regression_path = ROOT / recipe["output"]["candidateRoot"] / "regression-state.json"
    regression = parse_json_bytes(regression_path.read_bytes(), label="terminal regression state")
    validate_regression_state(
        regression,
        recipe,
        selection,
        lock_commit=lock_commit,
        selection_lock_sha256=digest_file(selection_path),
    )
    source_lock = parse_json_bytes(
        (ROOT / recipe["largeSyntheticEvaluation"]["sourceLock"]).read_bytes(),
        label="large-synthetic source lock",
    )
    if source_lock.get("regressionStateSha256") != digest_file(regression_path):
        raise ValueError("M5 large-synthetic source lock is not bound to the validated regressions")
    model_path = ROOT / selection["selectedModel"]["path"]
    if model_path.stat().st_size != selection["selectedModel"]["bytes"] or digest_file(model_path) != selection["selectedModel"]["sha256"]:
        raise ValueError("M5 large-synthetic selected model bytes changed")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ValueError("M5 100,000-image evaluation requires CUDAExecutionProvider")
    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    rows = load_rows(recipe)
    threshold = float(selection["rawThreshold"])
    logits: list[float] = []
    batch_results = []
    generator_totals: Counter[str] = Counter()
    generator_correct: Counter[str] = Counter()
    for batch_index in range(1_000):
        batch_rows = [row for row in rows if row["batchIndex"] == batch_index]
        batch_logits = score_batch(session, batch_rows)
        logits.extend(batch_logits)
        batch_correct = sum(value >= threshold for value in batch_logits)
        batch_results.append({"batchIndex": batch_index, "items": 100, "correct": batch_correct, "recall": batch_correct / 100})
        for row, value in zip(batch_rows, batch_logits, strict=True):
            generator_totals[row["generator"]] += 1
            generator_correct[row["generator"]] += int(value >= threshold)
        if batch_index % 10 == 0:
            print(json.dumps({"event": "large-synthetic-progress", "batches": batch_index + 1, "items": len(logits)}), flush=True)
    correct = sum(row["correct"] for row in batch_results)
    recalls = [row["recall"] for row in batch_results]
    lower, upper = wilson(correct, len(rows))
    generator_results = {
        generator: {"items": total, "correct": generator_correct[generator], "recall": generator_correct[generator] / total}
        for generator, total in sorted(generator_totals.items())
    }
    mean = statistics.fmean(recalls)
    median = statistics.median(recalls)
    passed = mean > recipe["largeSyntheticEvaluation"]["minimumMeanBatchRecallExclusive"] and median > recipe["largeSyntheticEvaluation"]["minimumMedianBatchRecallExclusive"]
    receipt = {
        "schemaVersion": 1,
        "status": "large-synthetic-pass" if passed else "large-synthetic-fail",
        "acceptanceEligible": passed,
        "sourceLockCommit": args.source_lock_commit,
        "selectionLockCommit": lock_commit,
        "selectionLockSha256": digest_file(selection_path),
        "sourceLockSha256": public["sourceLockSha256"],
        "manifestSha256": digest_file(ROOT / recipe["largeSyntheticEvaluation"]["manifest"]),
        "batchAssignmentSha256": digest_file(ROOT / recipe["largeSyntheticEvaluation"]["batchAssignment"]),
        "model": selection["selectedModel"],
        "rawThreshold": selection["rawThreshold"],
        "items": len(rows),
        "batchSize": 100,
        "batches": 1_000,
        "correct": correct,
        "overallRecall": correct / len(rows),
        "meanBatchRecall": mean,
        "medianBatchRecall": median,
        "minimumBatchRecall": min(recalls),
        "wilson95": {"lower": lower, "upper": upper},
        "batchResults": batch_results,
        "generatorResults": generator_results,
        "minimumGeneratorRecall": min(value["recall"] for value in generator_results.values()),
        "logits": pack_float32(logits),
        "selectionInfluence": False,
        "regressionStateSha256": digest_file(regression_path),
        "h3PixelsRead": False,
    }
    validate_evaluation_receipt(
        receipt, recipe, selection_lock=selection, source_lock_commit=args.source_lock_commit, panel_rows=rows,
    )
    write_json(ROOT / recipe["largeSyntheticEvaluation"]["evaluationReceipt"], receipt)
    print(json.dumps({
        "event": receipt["status"], "items": len(rows), "meanBatchRecall": mean,
        "medianBatchRecall": median, "overallRecall": receipt["overallRecall"], "h3PixelsRead": False,
    }, sort_keys=True))
    return 0 if passed else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-lock-commit", required=True)
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(execute(parser().parse_args()))
    except Exception as error:
        print(f"M5 large-synthetic evaluation failed: {error}", file=sys.stderr)
        raise
