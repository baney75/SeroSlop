#!/usr/bin/env python3
"""Publish the exact selected SeroSlop M5 model and its validated evidence packet."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.finalize_training_evidence import publish_with_rollback, verified_copy  # noqa: E402
from benchmark.m5.contracts import (  # noqa: E402
    canonical_json,
    digest_file,
    load_recipe,
    parse_json_bytes,
    read_jsonl,
    validate_regression_state,
    validate_selection_lock,
)
from benchmark.m5.evaluate_large_synthetic import load_rows, validate_evaluation_receipt  # noqa: E402
from benchmark.m5.large_synthetic import verify_public_packet  # noqa: E402
from benchmark.m5.train_gpu import Item, preprocess_image  # noqa: E402


RECIPE_PATH = ROOT / "benchmark/m5/recipe.json"
EVIDENCE_DIR = ROOT / "benchmark/evidence/m5"
REGRESSION_PATH = ROOT / "benchmark/candidates/prooflens-cf384-m5/regression-state.json"
FINAL_ROWS = (
    ("BENCHMARK.md", "M"),
    ("MODEL_CARD.md", "M"),
    ("README.md", "M"),
    ("benchmark/evidence/m5/calibration.json", "A"),
    ("benchmark/evidence/m5/large-synthetic-evaluation.json", "A"),
    ("benchmark/evidence/m5/finalization-receipt.json", "A"),
    ("benchmark/evidence/m5/model-comparison.json", "A"),
    ("benchmark/evidence/m5/regression-summary.json", "A"),
    ("benchmark/evidence/m5/training-summary.json", "A"),
    ("model-lock.json", "M"),
    ("tests/fixtures/model-states/fixture-manifest.json", "M"),
    ("weights/README.md", "M"),
    ("weights/prooflens-cf384.onnx", "M"),
)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def commit_rows(commit: str) -> list[tuple[str, str]]:
    output = git("diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit)
    return [(path, status) for status, path in (line.split("\t", maxsplit=1) for line in output.splitlines() if line)]


def require_source_lock_head(recipe: Mapping[str, Any]) -> tuple[str, str]:
    head = git("rev-parse", "HEAD")
    parents = git("rev-list", "--parents", "-n", "1", head).split()[1:]
    expected = sorted((path, "A") for path in (
        recipe["largeSyntheticEvaluation"]["manifest"],
        recipe["largeSyntheticEvaluation"]["batchAssignment"],
        recipe["largeSyntheticEvaluation"]["sourceLock"],
        recipe["largeSyntheticEvaluation"]["attribution"],
    ))
    if len(parents) != 1 or sorted(commit_rows(head)) != expected:
        raise ValueError("M5 finalizer requires the exact public 100K source-lock commit")
    lock_commit = parents[0]
    lock_parents = git("rev-list", "--parents", "-n", "1", lock_commit).split()[1:]
    if len(lock_parents) != 1 or commit_rows(lock_commit) != [(recipe["output"]["selectionLock"], "A")]:
        raise ValueError("M5 finalizer source lock is not the direct child of the one-file selection lock")
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    expected_status = f"?? {recipe['largeSyntheticEvaluation']['evaluationReceipt']}"
    if status != expected_status:
        raise ValueError("M5 finalizer requires only the completed 100K evaluation receipt to be untracked")
    return head, lock_commit


def write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value))


def model_signature(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest_file(path),
        "irVersion": int(model.ir_version),
        "opsets": sorted({(item.domain, int(item.version)) for item in model.opset_import}),
        "inputs": [(item.name, [dimension.dim_value or dimension.dim_param for dimension in item.type.tensor_type.shape.dim]) for item in model.graph.input],
        "outputs": [(item.name, [dimension.dim_value or dimension.dim_param for dimension in item.type.tensor_type.shape.dim]) for item in model.graph.output],
        "nodes": len(model.graph.node),
        "initializers": len(model.graph.initializer),
    }


def build_model_comparison(before: Path, after: Path) -> dict[str, Any]:
    import onnx

    before_model = onnx.load(before, load_external_data=False)
    after_model = onnx.load(after, load_external_data=False)
    before_initializers = {item.name: item.SerializeToString(deterministic=True) for item in before_model.graph.initializer}
    after_initializers = {item.name: item.SerializeToString(deterministic=True) for item in after_model.graph.initializer}
    changed = sorted(
        name for name in set(before_initializers) | set(after_initializers)
        if before_initializers.get(name) != after_initializers.get(name)
    )
    result = {
        "schemaVersion": 1,
        "profile": "m5-runpod-vit-finetune",
        "before": model_signature(before),
        "after": model_signature(after),
        "changedInitializerNames": changed,
        "changedInitializerCount": len(changed),
        "graphExportMayDiffer": True,
        "selectionBinding": "The selected ONNX bytes are independently replayed over all 600 selector images before terminal regressions.",
    }
    if result["after"]["inputs"] != [("pixel_values", ["batch", 3, 384, 384])]:
        raise ValueError("M5 selected model input contract changed")
    if result["after"]["outputs"] != [("logits", ["batch", 1])]:
        raise ValueError("M5 selected model output contract changed")
    return result


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def build_fixture_manifest(model_path: Path, lock: Mapping[str, Any], summary_sha256: str, calibration_sha256: str) -> dict[str, Any]:
    import onnxruntime as ort
    from PIL import Image, ImageOps

    old = parse_json_bytes((ROOT / "tests/fixtures/model-states/fixture-manifest.json").read_bytes(), label="fixture manifest")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    items = []
    for row in old["items"]:
        asset = ROOT / "tests/fixtures/model-states" / row["asset"]
        with Image.open(asset) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        item = Item(
            id=str(row["id"]), path=asset, image_sha256=str(row["assetSha256"]),
            label=int(row["label"]), source=str(row["source"]), row_index=0, weight=1.0, anchor=False,
        )
        pixels = preprocess_image(image, item, "original", training=False, branch="fixture", epoch=0)[None, ...]
        logit = float(session.run(["logits"], {"pixel_values": pixels.astype(np.float32)})[0].reshape(-1)[0])
        raw = sigmoid(logit)
        display = sigmoid(logit + float(lock["calibration"]["intercept"]))
        updated = dict(row)
        updated["selectionFeatureDisplayScore"] = display
        updated["referenceLogit"] = logit
        updated["referenceRawProbability"] = raw
        updated["referenceDisplayScore"] = display
        items.append(updated)
    by_role = {row["role"]: row for row in items}
    if by_role["likely-ai"]["referenceDisplayScore"] < 0.80 or by_role["below-threshold"]["referenceDisplayScore"] > 0.45:
        raise ValueError("M5 selected model does not preserve the two high-margin browser fixtures")
    return {
        "schemaVersion": 3,
        "selection": "M5 fixed high-margin browser QA fixtures; not acceptance evidence",
        "modelSha256": lock["selectedModel"]["sha256"],
        "calibrationSha256": calibration_sha256,
        "trainingSummarySha256": summary_sha256,
        "minimumLikelyAiScore": 0.80,
        "maximumBelowThresholdScore": 0.45,
        "items": items,
    }


def replace_section(text: str, start: str, end: str, replacement: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError("M5 public-document section markers changed")
    first = text.index(start)
    last = text.index(end) + len(end)
    return text[:first] + replacement.rstrip() + text[last:]


def render_documents(lock: Mapping[str, Any], regression: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, str]:
    metrics = lock["selectorMetrics"]
    metric_text = ", ".join(f"{name} {value['balancedAccuracy'] * 100:.2f}%" for name, value in metrics.items())
    common = (
        f"The selected local model is `{lock['selectedModel']['sha256']}` "
        f"({lock['selectedModel']['bytes']:,} bytes). It was selected only on the frozen 600-image M4 selector. "
        f"Selector balanced accuracy was {metric_text}, with zero observed false positives among 300 real selector images in every view. "
        f"Both frozen terminal regressions passed. On the separate 100,000-image synthetic panel, the mean batch recall was "
        f"{evaluation['meanBatchRecall'] * 100:.3f}% and the median was {evaluation['medianBatchRecall'] * 100:.3f}%. "
        "These are fixed development and synthetic-recall results, not a universal accuracy claim or the untouched H3 result."
    )
    replacements = {
        "README.md": "<!-- SEROSLOP_CURRENT_M5_START -->\n## Current M5 local model\n\n" + common + "\n<!-- SEROSLOP_CURRENT_M5_END -->",
        "MODEL_CARD.md": "<!-- SEROSLOP_CURRENT_M5_START -->\n## Current M5 training and evidence\n\n" + common + "\n<!-- SEROSLOP_CURRENT_M5_END -->",
        "BENCHMARK.md": "<!-- SEROSLOP_CURRENT_M5_START -->\n## Current M5 frozen evaluation boundary\n\n" + common + "\n<!-- SEROSLOP_CURRENT_M5_END -->",
    }
    output: dict[str, str] = {}
    for name, replacement in replacements.items():
        source = (ROOT / name).read_text(encoding="utf-8")
        rendered = replace_section(source, "<!-- PROOFLENS_CURRENT_M2_START -->", "<!-- PROOFLENS_CURRENT_M2_END -->", replacement)
        if name == "README.md":
            rendered = rendered.replace("current M2 model, evidence, package, and lineage checks", "current M5 model, evidence, package, and lineage checks")
            rendered = rendered.replace("on the M2 publication", "on the M5 publication")
        elif name == "MODEL_CARD.md":
            rendered = rendered.replace("Development validation contains", "Historical M2 development validation contains")
            rendered = rendered.replace("It is the only split used for M2 candidate and threshold selection.", "It was used only for historical M2 candidate and threshold selection.")
        else:
            rendered = rendered.replace("M2 responds only to", "Historical M2 responded only to")
            rendered = rendered.replace("## Current M2 training and development splits", "## Historical M2 training and development splits")
            rendered = rendered.replace("M2 head training, finalization, and development validation are complete.", "Historical M2 head training, finalization, and development validation are complete.")
        output[name] = rendered
    if regression["status"] != "regression-pass":
        raise ValueError("M5 documents cannot render without passed terminal regressions")
    return output


def build_model_lock(
    lock: Mapping[str, Any],
    *,
    source_lock_commit: str,
    source_lock_sha256: str,
    regression_sha256: str,
    evaluation_sha256: str,
    summary_sha256: str,
    calibration_sha256: str,
) -> dict[str, Any]:
    template = json.loads((ROOT / "model-lock.json").read_text(encoding="utf-8"))
    calibration = lock["calibration"]
    return {
        "schemaVersion": 2,
        "artifact": "weights/prooflens-cf384.onnx",
        "bytes": lock["selectedModel"]["bytes"],
        "sha256": lock["selectedModel"]["sha256"],
        "format": template["format"],
        "input": template["input"],
        "output": template["output"],
        "upstream": template["upstream"],
        "trainingRecipe": f"seroslop-m5-runpod-vit-finetune:{digest_file(RECIPE_PATH)}:{lock['recipeSha256']}",
        "trainingEvidence": {
            "recipe": "benchmark/m5/recipe.json",
            "recipeSha256": digest_file(RECIPE_PATH),
            "selectionLock": "benchmark/evidence/m5/selection-lock.json",
            "selectionLockSha256": digest_file(ROOT / "benchmark/evidence/m5/selection-lock.json"),
            "largeSyntheticSourceLockCommit": source_lock_commit,
            "largeSyntheticSourceLockSha256": source_lock_sha256,
            "trainingSummarySha256": summary_sha256,
            "regressionSummarySha256": regression_sha256,
            "largeSyntheticEvaluationSha256": evaluation_sha256,
            "calibrationSha256": calibration_sha256,
            "architecture": "RunPod-fine-tuned Community Forensics ViT-S/16 with a 384-to-1 classifier",
        },
        "calibration": {
            "slope": calibration["slope"],
            "intercept": calibration["intercept"],
            "displayThreshold": calibration["displayThreshold"],
        },
    }


def execute(_args: argparse.Namespace) -> int:
    recipe = load_recipe(RECIPE_PATH)
    source_lock_commit, lock_commit = require_source_lock_head(recipe)
    selection_path = ROOT / recipe["output"]["selectionLock"]
    selection = parse_json_bytes(selection_path.read_bytes(), label="selection lock")
    selector_rows = read_jsonl(ROOT / recipe["sourceEvidence"]["selectorManifest"]["path"])
    validate_selection_lock(selection, recipe, selector_rows)
    protocol_commit = git("rev-parse", f"{lock_commit}^")
    if selection["protocolCommit"] != protocol_commit:
        raise ValueError("M5 finalizer selection lock ancestry changed")
    regression = parse_json_bytes(REGRESSION_PATH.read_bytes(), label="terminal regression state")
    validate_regression_state(
        regression, recipe, selection, lock_commit=lock_commit,
        selection_lock_sha256=digest_file(selection_path),
    )
    public = verify_public_packet(recipe, verify_pixels=False)
    source_lock_path = ROOT / recipe["largeSyntheticEvaluation"]["sourceLock"]
    source_lock = parse_json_bytes(source_lock_path.read_bytes(), label="large-synthetic source lock")
    if source_lock["regressionStateSha256"] != digest_file(REGRESSION_PATH):
        raise ValueError("M5 finalizer source lock is not bound to the terminal regressions")
    evaluation_path = ROOT / recipe["largeSyntheticEvaluation"]["evaluationReceipt"]
    evaluation = parse_json_bytes(evaluation_path.read_bytes(), label="large-synthetic evaluation")
    validate_evaluation_receipt(
        evaluation, recipe, selection_lock=selection, source_lock_commit=source_lock_commit,
        panel_rows=load_rows(recipe),
    )
    if evaluation["status"] != "large-synthetic-pass" or evaluation["acceptanceEligible"] is not True:
        raise ValueError("M5 finalizer requires the strict 100K mean and median gates to pass")
    selected_model = ROOT / selection["selectedModel"]["path"]
    if selected_model.stat().st_size != selection["selectedModel"]["bytes"] or digest_file(selected_model) != selection["selectedModel"]["sha256"]:
        raise ValueError("M5 selected model bytes changed before finalization")

    with tempfile.TemporaryDirectory(prefix="seroslop-m5-finalize-") as temporary_directory:
        temporary = Path(temporary_directory)
        model_stage = temporary / "prooflens-cf384.onnx"
        verified_copy(selected_model, model_stage)
        summary_stage = temporary / "training-summary.json"
        write_canonical(summary_stage, selection["trainingSummary"])
        calibration_stage = temporary / "calibration.json"
        write_canonical(calibration_stage, selection["calibration"])
        regression_stage = temporary / "regression-summary.json"
        write_canonical(regression_stage, regression)
        evaluation_stage = temporary / "large-synthetic-evaluation.json"
        verified_copy(evaluation_path, evaluation_stage)
        comparison_stage = temporary / "model-comparison.json"
        write_canonical(comparison_stage, build_model_comparison(ROOT / "weights/prooflens-cf384.onnx", selected_model))
        fixture_stage = temporary / "fixture-manifest.json"
        write_canonical(fixture_stage, build_fixture_manifest(
            selected_model, selection, digest_file(summary_stage), digest_file(calibration_stage),
        ))
        documents = render_documents(selection, regression, evaluation)
        document_stages: dict[str, Path] = {}
        for name, text in documents.items():
            stage = temporary / name
            stage.write_text(text, encoding="utf-8")
            document_stages[name] = stage
        model_lock_stage = temporary / "model-lock.json"
        write_canonical(model_lock_stage, build_model_lock(
            selection,
            source_lock_commit=source_lock_commit,
            source_lock_sha256=public["sourceLockSha256"],
            regression_sha256=digest_file(regression_stage),
            evaluation_sha256=digest_file(evaluation_stage),
            summary_sha256=digest_file(summary_stage),
            calibration_sha256=digest_file(calibration_stage),
        ))
        weights_readme_stage = temporary / "weights-README.md"
        weights_readme_stage.write_text(
            "# Packaged detector\n\n"
            "`prooflens-cf384.onnx` is the exact fully local FP32 artifact described by the repository root `model-lock.json`.\n\n"
            f"- bytes: {selection['selectedModel']['bytes']:,}\n"
            f"- sha256: `{selection['selectedModel']['sha256']}`\n"
            "- input: `pixel_values [N,3,384,384] float32`\n"
            "- output: `logits [N,1] float32`\n",
            encoding="utf-8",
        )
        staged = [
            (document_stages["BENCHMARK.md"], ROOT / "BENCHMARK.md"),
            (document_stages["MODEL_CARD.md"], ROOT / "MODEL_CARD.md"),
            (document_stages["README.md"], ROOT / "README.md"),
            (calibration_stage, EVIDENCE_DIR / "calibration.json"),
            (evaluation_stage, EVIDENCE_DIR / "large-synthetic-evaluation.json"),
            (comparison_stage, EVIDENCE_DIR / "model-comparison.json"),
            (regression_stage, EVIDENCE_DIR / "regression-summary.json"),
            (summary_stage, EVIDENCE_DIR / "training-summary.json"),
            (model_lock_stage, ROOT / "model-lock.json"),
            (fixture_stage, ROOT / "tests/fixtures/model-states/fixture-manifest.json"),
            (weights_readme_stage, ROOT / "weights/README.md"),
            (model_stage, ROOT / "weights/prooflens-cf384.onnx"),
        ]
        published_without_receipt = {destination.relative_to(ROOT).as_posix(): digest_file(stage) for stage, destination in staged}
        receipt = {
            "schemaVersion": 1,
            "status": "m5-finalized",
            "acceptanceEligible": False,
            "protocolCommit": protocol_commit,
            "selectionLockCommit": lock_commit,
            "selectionLockSha256": digest_file(selection_path),
            "largeSyntheticSourceLockCommit": source_lock_commit,
            "largeSyntheticSourceLockSha256": public["sourceLockSha256"],
            "selectedCandidateId": selection["selectedCandidateId"],
            "selectedModelSha256": selection["selectedModel"]["sha256"],
            "trainingSummarySha256": digest_file(summary_stage),
            "regressionSummarySha256": digest_file(regression_stage),
            "largeSyntheticEvaluationSha256": digest_file(evaluation_stage),
            "calibrationSha256": digest_file(calibration_stage),
            "modelComparisonSha256": digest_file(comparison_stage),
            "shippedModel": {"path": "weights/prooflens-cf384.onnx", "bytes": selection["selectedModel"]["bytes"], "sha256": selection["selectedModel"]["sha256"]},
            "publishedSha256": dict(sorted(published_without_receipt.items())),
            "publicationRows": [list(row) for row in FINAL_ROWS],
            "h3HoldoutScored": False,
        }
        receipt_stage = temporary / "finalization-receipt.json"
        write_canonical(receipt_stage, receipt)
        staged.insert(5, (receipt_stage, EVIDENCE_DIR / "finalization-receipt.json"))
        expected = {ROOT / path for path, _status in FINAL_ROWS}
        if {destination for _stage, destination in staged} != expected:
            raise ValueError("M5 finalizer publication surface changed")
        receipt_item = staged.pop(5)
        staged.append(receipt_item)
        if staged[-1][1] != EVIDENCE_DIR / "finalization-receipt.json":
            raise ValueError("M5 finalization receipt is not published last")
        publish_with_rollback(staged, temporary / "backups")

    print(json.dumps({
        "event": "m5-finalized",
        "modelSha256": selection["selectedModel"]["sha256"],
        "meanBatchRecall": evaluation["meanBatchRecall"],
        "medianBatchRecall": evaluation["medianBatchRecall"],
        "h3HoldoutScored": False,
    }, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    try:
        raise SystemExit(execute(parser().parse_args()))
    except Exception as error:
        print(f"M5 finalization failed: {error}", file=sys.stderr)
        raise
