"""Record the failed M3 selector attempt without reinterpreting it as a model result.

This diagnostic runs only after the frozen M3 attempt has terminated. It binds
the ignored feature-cache bytes, validates their manifest-derived metadata, and
refits the public 25-head grid solely to prove whether any frozen selector
threshold exists. It never reads H3 pixels, publishes a model, changes a gate,
or uses the consumed M2 regression packet for candidate selection.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.modern import train_rehead as trainer
from benchmark.thresholds import complete_decision_thresholds


ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "2e6de2187d1cff5aea48e57ad9a30f15541fc4df"
SOURCE_TREE = "a6ab771f27b1efd108caa0b08128118fd7465334"
BASE_COMMIT = "0adbd55d8cdc25ad3d20e773a315ec57d14c7973"
UPSTREAM_MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47"
RUN_ID = "447053b4b6f924488653d237e2372230"
CACHE_ROOT = ROOT / "benchmark/candidates/prooflens-cf384-m3"
FEATURE_ROOT = CACHE_ROOT / "features"
RECIPE_PATH = ROOT / "benchmark/m3/recipe.json"
TRAIN_MANIFEST_GZIP = ROOT / "benchmark/evidence/m3/train-manifest.jsonl.gz"
LOCAL_TRAIN_MANIFEST = ROOT / "benchmark/data/m3-head/train-manifest.jsonl"
SELECTOR_MANIFEST = ROOT / "benchmark/evidence/m3/validation-manifest.jsonl"
REGRESSION_MANIFEST = ROOT / "benchmark/evidence/m2/validation-manifest.jsonl"
RECEIPT_PATH = ROOT / "benchmark/evidence/m3/failed-training-attempt-1.json"
DIAGNOSTIC_PATH = ROOT / "benchmark/evidence/m3/failed-selector-diagnostic-1.json"
SINGLE_VIEW_SOURCES = frozenset(("diffusiondb-stable-diffusion", "open-images-train"))
DECAYS = (0.10, 0.03, 0.01, 0.003, 0.001)
ALPHAS = (0.40, 0.55, 0.70, 0.85, 1.0)
CANDIDATE_OUTPUTS = (
    "benchmark/candidates/prooflens-cf384-m3/model.onnx",
    "benchmark/candidates/prooflens-cf384-m3/calibration.json",
    "benchmark/candidates/prooflens-cf384-m3/candidate-grid.json",
    "benchmark/candidates/prooflens-cf384-m3/validation-summary.json",
)
PUBLICATION_OUTPUTS = (
    "benchmark/evidence/m3/publication-lock.json",
    "benchmark/evidence/m3/calibration.json",
    "benchmark/evidence/m3/candidate-grid.json",
    "benchmark/evidence/m3/finalization-receipt.json",
    "benchmark/evidence/m3/model-comparison.json",
    "benchmark/evidence/m3/training-summary.json",
)
COMMAND_ARGUMENTS = (
    "--model", "weights/prooflens-cf384.onnx",
    "--expected-model-sha256", UPSTREAM_MODEL_SHA256,
    "--data-root", "benchmark/data/m3-head",
    "--train-manifest", "benchmark/data/m3-head/train-manifest.jsonl",
    "--validation-data-root", "benchmark/data/m3-head",
    "--validation-manifest", "benchmark/evidence/m3/validation-manifest.jsonl",
    "--regression-data-root", "benchmark/data/m2-head",
    "--regression-manifest", "benchmark/evidence/m2/validation-manifest.jsonl",
    "--recipe", "benchmark/m3/recipe.json",
    "--selection-summary", "benchmark/evidence/m3/selection-summary.json",
    "--single-view-source", "diffusiondb-stable-diffusion",
    "--single-view-source", "open-images-train",
    "--execution-provider", "cpu",
    "--batch-size", "24",
    "--feature-shard-images", "2000",
    "--reextract-cached-features",
    "--output-dir", "benchmark/candidates/prooflens-cf384-m3",
)


@dataclass(frozen=True)
class ManifestItem:
    id: str
    path: Path
    image_sha256: str
    label: int
    source: str


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git(arguments: list[str], *, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=ROOT,
        text=not binary,
    ).strip() if not binary else subprocess.check_output(["git", *arguments], cwd=ROOT)


def parse_manifest_bytes(value: bytes, *, data_root: Path) -> list[ManifestItem]:
    rows = [json.loads(line) for line in value.decode("utf-8").splitlines() if line]
    items = [
        ManifestItem(
            id=str(row["id"]),
            path=data_root / str(row["path"]),
            image_sha256=str(row["imageSha256"]),
            label=int(row["label"]),
            source=str(row["source"]),
        )
        for row in rows
    ]
    if not items or len({item.id for item in items}) != len(items):
        raise ValueError("Diagnostic manifest is empty or contains duplicate IDs")
    if len({item.image_sha256 for item in items}) != len(items):
        raise ValueError("Diagnostic manifest contains duplicate image bytes")
    return items


def expected_cache_names() -> tuple[str, ...]:
    return (
        "feature-model.onnx",
        *(f"features/train-{index:05d}.npz" for index in range(55)),
        "features/validation-00000.npz",
        "features/regression-00000.npz",
        "fresh-feature-run.json",
    )


def inventory_cache() -> list[dict[str, Any]]:
    if CACHE_ROOT.is_symlink() or not CACHE_ROOT.is_dir():
        raise ValueError("M3 diagnostic cache root is missing or symlinked")
    actual: list[str] = []
    for path in sorted(CACHE_ROOT.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"M3 diagnostic cache contains a symlink: {path}")
        if path.is_file():
            actual.append(path.relative_to(CACHE_ROOT).as_posix())
    expected = list(expected_cache_names())
    if actual != sorted(expected):
        raise ValueError("M3 diagnostic cache inventory changed")
    return [
        {
            "path": f"benchmark/candidates/prooflens-cf384-m3/{relative}",
            "bytes": (CACHE_ROOT / relative).stat().st_size,
            "sha256": digest_file(CACHE_ROOT / relative),
        }
        for relative in actual
    ]


def read_items() -> tuple[list[ManifestItem], list[ManifestItem], list[ManifestItem], bytes]:
    compressed = TRAIN_MANIFEST_GZIP.read_bytes()
    train_bytes = gzip.decompress(compressed)
    if LOCAL_TRAIN_MANIFEST.exists() and LOCAL_TRAIN_MANIFEST.read_bytes() != train_bytes:
        raise ValueError("Local M3 train manifest differs from the committed expanded manifest")
    return (
        parse_manifest_bytes(train_bytes, data_root=ROOT / "benchmark/data/m3-head"),
        parse_manifest_bytes(SELECTOR_MANIFEST.read_bytes(), data_root=ROOT / "benchmark/data/m3-head"),
        parse_manifest_bytes(REGRESSION_MANIFEST.read_bytes(), data_root=ROOT / "benchmark/data/m2-head"),
        train_bytes,
    )


def load_partition(
    cache_name: str,
    items: list[ManifestItem],
    *,
    manifest_hash: str,
    configuration_hash: str,
    training: bool,
    concatenate: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for shard_index, offset in enumerate(range(0, len(items), 2_000)):
        shard_items = items[offset : offset + 2_000]
        path = FEATURE_ROOT / f"{cache_name}-{shard_index:05d}.npz"
        with np.load(path, allow_pickle=False) as data:
            required = {
                "features", "labels", "variants", "sources", "manifest_hash", "model_hash",
                "item_ids_hash", "feature_configuration_hash", "fresh_feature_run_id", "training",
                "pipeline_version",
            }
            if set(data.files) != required:
                raise ValueError(f"Unexpected feature-cache fields: {path}")
            item_ids_hash = sha256("\n".join(item.id for item in shard_items).encode()).hexdigest()
            scalars = {
                "manifest_hash": manifest_hash,
                "model_hash": UPSTREAM_MODEL_SHA256,
                "item_ids_hash": item_ids_hash,
                "feature_configuration_hash": configuration_hash,
                "fresh_feature_run_id": RUN_ID,
            }
            if any(str(data[name].item()) != value for name, value in scalars.items()):
                raise ValueError(f"Feature-cache scalar binding changed: {path}")
            if bool(data["training"].item()) != training or int(data["pipeline_version"].item()) != 9:
                raise ValueError(f"Feature-cache execution metadata changed: {path}")
            result = tuple(
                np.asarray(data[name]).copy()
                for name in ("features", "labels", "variants", "sources")
            )
        checked = trainer.validate_feature_result(
            result,  # type: ignore[arg-type]
            shard_items,  # type: ignore[arg-type]
            training=training,
            single_view_sources=SINGLE_VIEW_SOURCES if training else frozenset(),
        )
        if concatenate:
            outputs.append(checked)
    if not concatenate:
        return None
    return tuple(np.concatenate([part[index] for part in outputs]) for index in range(4))  # type: ignore[return-value]


def gate_margins(values: dict[str, object], gates: dict[str, object]) -> tuple[list[float], dict[str, float]]:
    margins: list[float] = []
    minima = {
        "balancedAccuracy": 1.0,
        "realRecall": 1.0,
        "syntheticRecall": 1.0,
        "syntheticRecallBySource": 1.0,
        "metRecall": 1.0,
    }
    for row_value in values.values():
        row = dict(row_value)
        family = min(float(value) for value in dict(row["syntheticRecallBySource"]).values())
        met = float(dict(row["realRecallBySource"])["met-open-access"])
        minima["balancedAccuracy"] = min(minima["balancedAccuracy"], float(row["balancedAccuracy"]))
        minima["realRecall"] = min(minima["realRecall"], float(row["realRecall"]))
        minima["syntheticRecall"] = min(minima["syntheticRecall"], float(row["syntheticRecall"]))
        minima["syntheticRecallBySource"] = min(minima["syntheticRecallBySource"], family)
        minima["metRecall"] = min(minima["metRecall"], met)
        margins.extend((
            float(row["balancedAccuracy"]) - float(gates["minimumBalancedAccuracyPerVariant"]),
            float(row["realRecall"]) - float(gates["minimumRealRecallPerVariant"]),
            float(row["syntheticRecall"]) - float(gates["minimumSyntheticRecallPerVariant"]),
            family - float(gates["minimumSyntheticRecallPerFamily"]),
            met - float(dict(gates["minimumRealRecallBySource"])["met-open-access"]),
        ))
    return margins, minima


def diagnose_candidate(
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    selector: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    upstream_weight: np.ndarray,
    upstream_bias: float,
    gates: dict[str, object],
    decay: float,
    alpha: float,
) -> dict[str, Any]:
    weight, bias, _, _, _ = trainer.fit_candidate(
        train[0], train[1], train[3], selector[0], selector[1], selector[2], selector[3],
        upstream_weight, upstream_bias, decay, alpha, torch.device("cpu"), None,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logits = selector[0] @ weight + bias
    if not np.isfinite(logits).all():
        raise FloatingPointError("Diagnostic candidate produced non-finite selector logits")
    logit_bytes = np.asarray(logits, dtype="<f4").tobytes()
    decision_logits = np.frombuffer(logit_bytes, dtype="<f4").astype(np.float64)
    candidates = complete_decision_thresholds(decision_logits.tolist())
    feasible_count = 0
    closest: tuple[tuple[float, ...], float, dict[str, float]] | None = None
    real_met: list[tuple[float, dict[str, object]]] = []
    synthetic: list[tuple[float, dict[str, object]]] = []
    for threshold in candidates:
        values = trainer.variant_metrics(decision_logits, selector[1], selector[2], selector[3], threshold)
        margins, minima = gate_margins(values, gates)
        if min(margins) >= 0:
            feasible_count += 1
        key = (
            min(margins),
            minima["balancedAccuracy"],
            minima["realRecall"],
            minima["syntheticRecall"],
            minima["metRecall"],
            -abs(threshold),
        )
        if closest is None or key > closest[0]:
            closest = (key, float(threshold), minima)
        rows = [dict(row) for row in values.values()]
        if all(
            float(row["realRecall"]) >= float(gates["minimumRealRecallPerVariant"])
            and float(dict(row["realRecallBySource"])["met-open-access"])
            >= float(dict(gates["minimumRealRecallBySource"])["met-open-access"])
            for row in rows
        ):
            real_met.append((float(threshold), values))
        if all(
            float(row["syntheticRecall"]) >= float(gates["minimumSyntheticRecallPerVariant"])
            and min(float(value) for value in dict(row["syntheticRecallBySource"]).values())
            >= float(gates["minimumSyntheticRecallPerFamily"])
            for row in rows
        ):
            synthetic.append((float(threshold), values))
    if closest is None or not real_met or not synthetic:
        raise AssertionError("M3 selector diagnostic did not expose both threshold sides")
    lower = min(threshold for threshold, _ in real_met)
    upper = max(threshold for threshold, _ in synthetic)
    best_synthetic_when_real_met = max(
        min(float(dict(row)["syntheticRecall"]) for row in values.values())
        for _, values in real_met
    )
    best_met_when_synthetic = max(
        min(float(dict(dict(row)["realRecallBySource"])["met-open-access"]) for row in values.values())
        for _, values in synthetic
    )
    return {
        "parameters": {"weightDecay": decay, "upstreamBlendAlpha": alpha},
        "selectorLogits": {
            "encoding": "base64-float32-little-endian",
            "count": int(logits.shape[0]),
            "bytes": len(logit_bytes),
            "sha256": digest_bytes(logit_bytes),
            "base64": base64.b64encode(logit_bytes).decode("ascii"),
        },
        "selectorDecisionThresholds": len(candidates),
        "feasibleThresholds": feasible_count,
        "thresholdConflict": {
            "minimumThresholdForRealAndMetGates": lower,
            "maximumThresholdForSyntheticAndFamilyGates": upper,
            "infeasibleGapLogit": lower - upper,
            "bestWorstVariantSyntheticRecallWhileRealAndMetPass": best_synthetic_when_real_met,
            "bestWorstVariantMetRecallWhileSyntheticAndFamilyPass": best_met_when_synthetic,
        },
        "closestCompromise": {
            "thresholdLogit": closest[1],
            "minimumGateMargin": closest[0][0],
            "worstVariant": closest[2],
        },
    }


def build_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    if git(["rev-parse", f"{SOURCE_COMMIT}^{{tree}}"]).__str__() != SOURCE_TREE:
        raise ValueError("Frozen M3 source tree changed")
    if git(["rev-parse", f"{SOURCE_COMMIT}^"]).__str__() != BASE_COMMIT:
        raise ValueError("Frozen M3 source ancestry changed")
    source_trainer = git(["show", f"{SOURCE_COMMIT}:benchmark/modern/train_rehead.py"], binary=True)
    if not isinstance(source_trainer, bytes) or source_trainer != (ROOT / "benchmark/modern/train_rehead.py").read_bytes():
        raise ValueError("Current trainer differs from the frozen M3 source")
    if digest_file(ROOT / "weights/prooflens-cf384.onnx") != UPSTREAM_MODEL_SHA256:
        raise ValueError("Shipped M2 model changed before M3 failure recording")
    marker_path = CACHE_ROOT / "fresh-feature-run.json"
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    recipe = json.loads(RECIPE_PATH.read_text())
    if marker.get("runId") != RUN_ID or marker.get("state") != "extracting":
        raise ValueError("M3 feature marker no longer records the incomplete attempt")
    train_items, selector_items, regression_items, train_bytes = read_items()
    context = dict(marker["context"])
    expected_hashes = {
        "train": digest_bytes(train_bytes),
        "selector": digest_file(SELECTOR_MANIFEST),
        "regression": digest_file(REGRESSION_MANIFEST),
    }
    if (
        context.get("trainManifestSha256") != expected_hashes["train"]
        or context.get("validationManifestSha256") != expected_hashes["selector"]
        or context.get("regressionManifestSha256") != expected_hashes["regression"]
    ):
        raise ValueError("M3 feature marker manifest hashes changed")
    train = load_partition(
        "train", train_items,
        manifest_hash=expected_hashes["train"],
        configuration_hash=str(dict(context["featureConfigurationHashes"])["training"]),
        training=True,
        concatenate=True,
    )
    selector = load_partition(
        "validation", selector_items,
        manifest_hash=expected_hashes["selector"],
        configuration_hash=str(dict(context["featureConfigurationHashes"])["validation"]),
        training=False,
        concatenate=True,
    )
    load_partition(
        "regression", regression_items,
        manifest_hash=expected_hashes["regression"],
        configuration_hash=str(dict(context["featureConfigurationHashes"])["regression"]),
        training=False,
        concatenate=False,
    )
    if train is None or selector is None:
        raise AssertionError("M3 diagnostic feature arrays were not loaded")
    if train[0].shape != (133_512, 384) or selector[0].shape != (2_400, 384):
        raise ValueError("M3 diagnostic feature-view counts changed")
    model = onnx.load(ROOT / "weights/prooflens-cf384.onnx")
    initializers = {value.name: value for value in model.graph.initializer}
    upstream_weight = numpy_helper.to_array(initializers["classifier.weight"]).astype(np.float32).reshape(-1)
    upstream_bias = float(numpy_helper.to_array(initializers["classifier.bias"]).reshape(-1)[0])
    torch.manual_seed(trainer.SEED)
    np.random.seed(trainer.SEED)
    torch.use_deterministic_algorithms(True)
    gates = dict(recipe["validationGates"])
    candidates = [
        diagnose_candidate(train, selector, upstream_weight, upstream_bias, gates, decay, alpha)
        for decay in DECAYS
        for alpha in ALPHAS
    ]
    if any(candidate["feasibleThresholds"] != 0 for candidate in candidates):
        raise RuntimeError("A frozen M3 candidate unexpectedly became selector-feasible")
    closest = max(
        candidates,
        key=lambda candidate: (
            float(dict(candidate["closestCompromise"])["minimumGateMargin"]),
            float(dict(dict(candidate["closestCompromise"])["worstVariant"])["balancedAccuracy"]),
        ),
    )
    inventory = inventory_cache()
    for path in (*CANDIDATE_OUTPUTS, *PUBLICATION_OUTPUTS):
        if (ROOT / path).exists():
            raise ValueError(f"M3 failure record cannot coexist with an output: {path}")
    diagnostic = {
        "schemaVersion": 1,
        "profile": "m3",
        "attemptId": "m3-failed-training-attempt-1",
        "role": "post-failure-selector-replay",
        "source": {
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
        },
        "inputBindings": {
            "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
            "trainerSha256": digest_bytes(source_trainer),
            "diagnosticGeneratorSha256": digest_file(Path(__file__)),
            "recipeSha256": digest_file(RECIPE_PATH),
            "selectionSummarySha256": str(context["selectionSummarySha256"]),
            "trainManifestSha256": expected_hashes["train"],
            "selectorManifestSha256": expected_hashes["selector"],
            "runId": RUN_ID,
        },
        "method": {
            "fit": "Frozen M3 linear-head fit replayed from the exact cached training and selector features after the original attempt terminated",
            "candidateOrder": "five weight decays in frozen order, then five upstream blend alphas in frozen order",
            "thresholdEnumeration": "minimum serialized float32 selector logit plus binary64 nextafter(logit,+infinity) for every unique finite selector logit; comparisons are binary64 so every stored-logit partition is tested",
            "selectorViewOrder": list(trainer.VARIANTS),
            "logitEncoding": "base64-float32-little-endian",
            "regressionUsedForSelection": False,
            "h3AcceptedAsInput": False,
        },
        "frozenGates": gates,
        "candidates": candidates,
        "aggregate": {
            "candidateCount": len(candidates),
            "feasibleCandidateCount": 0,
            "allFrozenSelectorCandidatesInfeasible": True,
            "minimumThresholdConflictLogit": min(
                float(dict(candidate["thresholdConflict"])["infeasibleGapLogit"])
                for candidate in candidates
            ),
            "maximumThresholdConflictLogit": max(
                float(dict(candidate["thresholdConflict"])["infeasibleGapLogit"])
                for candidate in candidates
            ),
            "bestWorstVariantSyntheticRecallWhileRealAndMetPass": max(
                float(dict(candidate["thresholdConflict"])["bestWorstVariantSyntheticRecallWhileRealAndMetPass"])
                for candidate in candidates
            ),
            "bestWorstVariantMetRecallWhileSyntheticAndFamilyPass": max(
                float(dict(candidate["thresholdConflict"])["bestWorstVariantMetRecallWhileSyntheticAndFamilyPass"])
                for candidate in candidates
            ),
            "closestCandidateParameters": dict(closest["parameters"]),
        },
    }
    diagnostic_bytes = canonical_bytes(diagnostic)
    receipt = {
        "schemaVersion": 1,
        "profile": "m3",
        "attemptId": "m3-failed-training-attempt-1",
        "stage": "m3-failed",
        "source": {
            "commit": SOURCE_COMMIT,
            "tree": SOURCE_TREE,
            "parent": BASE_COMMIT,
        },
        "inputBindings": {
            "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
            "trainerSha256": digest_bytes(source_trainer),
            "diagnosticGeneratorSha256": digest_file(Path(__file__)),
            "recipeSha256": digest_file(RECIPE_PATH),
            "selectionSummarySha256": str(context["selectionSummarySha256"]),
            "selectorManifestSha256": expected_hashes["selector"],
            "regressionManifestSha256": expected_hashes["regression"],
            "commandArguments": list(COMMAND_ARGUMENTS),
        },
        "operatorObservation": {
            "evidenceClass": "operator-observed",
            "exitCode": 1,
            "terminalExceptionType": "RuntimeError",
            "terminalMessage": "No trained candidate",
            "durableStdoutCaptured": False,
            "limitation": "The original console stream was not preserved; the post-failure diagnostic is separate evidence.",
        },
        "cacheSnapshot": {
            "evidenceClass": "captured-local-artifact-inventory",
            "root": "benchmark/candidates/prooflens-cf384-m3",
            "runId": RUN_ID,
            "marker": {
                "path": "benchmark/candidates/prooflens-cf384-m3/fresh-feature-run.json",
                "sha256": digest_bytes(marker_bytes),
                "bytes": len(marker_bytes),
                "state": marker["state"],
                "payload": marker,
            },
            "fileCount": len(inventory),
            "totalBytes": sum(int(item["bytes"]) for item in inventory),
            "inventory": inventory,
            "persistenceRequired": False,
        },
        "diagnostic": {
            "path": "benchmark/evidence/m3/failed-selector-diagnostic-1.json",
            "sha256": digest_bytes(diagnostic_bytes),
            "method": "post-failure deterministic refit from captured cached features",
            "candidateCount": len(candidates),
            "feasibleCandidateCount": 0,
            "allFrozenSelectorCandidatesInfeasible": True,
            "originalConsoleReplacement": False,
        },
        "absence": {
            "candidateSelectionCompleted": False,
            "candidateOutputsAbsent": list(CANDIDATE_OUTPUTS),
            "publicationOutputsAbsent": list(PUBLICATION_OUTPUTS),
            "successfulCandidateOutputsPresent": False,
            "publicationLockPresent": False,
            "successfulM3PublicationEvidencePresent": False,
            "h3AcceptedAsInput": False,
            "trackedH3ScoreArtifactsPresent": False,
        },
        "h3Observation": {
            "evidenceClass": "operator-observed",
            "h3PixelsReadOrScored": False,
            "machineVerifiableBoundary": "The canonical command and diagnostic accept no H3 input, and no tracked H3 score artifact exists.",
        },
        "shippedModel": {
            "path": "weights/prooflens-cf384.onnx",
            "sha256": UPSTREAM_MODEL_SHA256,
            "bytes": (ROOT / "weights/prooflens-cf384.onnx").stat().st_size,
            "retained": True,
        },
        "decision": {
            "status": "terminal-frozen-selector-failure",
            "selectedCandidate": None,
            "thresholdChanged": False,
            "gatesChanged": False,
            "modelPublished": False,
            "nextAttemptMayReuseThisSelectorForSelection": False,
        },
        "limitations": [
            "The original process stdout was not retained as a durable file; its exit details are operator-observed rather than machine-verifiable.",
            "The post-failure diagnostic replays the frozen fit and threshold search from exact cached feature bytes; it is not the original process log.",
            "The ignored feature caches may be removed after their hashes and the diagnostic result are committed.",
            "This failed development attempt is not acceptance evidence and says nothing about a future H3 result.",
        ],
    }
    return diagnostic, receipt


def canonical_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonical_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [canonical_json_value(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(canonical_json_value(payload), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=RECEIPT_PATH)
    parser.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC_PATH)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    diagnostic, receipt = build_evidence()
    diagnostic_bytes = canonical_bytes(diagnostic)
    receipt_bytes = canonical_bytes(receipt)
    diagnostic_path = args.diagnostic.resolve()
    receipt_path = args.receipt.resolve()
    if args.verify:
        if not diagnostic_path.is_file() or diagnostic_path.read_bytes() != diagnostic_bytes:
            raise ValueError("Committed M3 failure diagnostic differs from deterministic replay")
        if not receipt_path.is_file() or receipt_path.read_bytes() != receipt_bytes:
            raise ValueError("Committed M3 failure receipt differs from deterministic replay")
    else:
        write_atomic(diagnostic_path, diagnostic_bytes)
        write_atomic(receipt_path, receipt_bytes)
    print(json.dumps({
        "policy": "pass",
        "receipt": str(receipt_path.relative_to(ROOT)),
        "receiptSha256": digest_bytes(receipt_bytes),
        "diagnostic": str(diagnostic_path.relative_to(ROOT)),
        "diagnosticSha256": digest_bytes(diagnostic_bytes),
        "candidates": diagnostic["aggregate"]["candidateCount"],
        "feasibleCandidates": diagnostic["aggregate"]["feasibleCandidateCount"],
        "h3AcceptedAsInput": diagnostic["method"]["h3AcceptedAsInput"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
