"""Train the frozen finite M4 residual-adapter grid on fresh CPU features.

Candidate tensors are fit from training arrays only and sealed before the fresh
selector is evaluated. The selected candidate and threshold are then sealed
before the consumed M3 and M2 regressions are opened in fixed order. H3 is not
an accepted argument or input.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m4.contracts import (  # noqa: E402
    ADAPTER_ANCHOR_COEFFICIENTS,
    ADAPTER_WEIGHT_DECAYS,
    MODELS,
    VARIANTS,
    canonical_json,
    load_frozen_protocol,
)
from benchmark.thresholds import complete_decision_thresholds  # noqa: E402
from benchmark.modern import train_rehead as feature_pipeline  # noqa: E402


PIPELINE_VERSION = 1
SEED = 20260815
UPSTREAM_MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47"
RECIPE_PATH = ROOT / "benchmark/m4/recipe.json"
LOCKS_PATH = ROOT / "benchmark/m4/source-locks.json"
SELECTOR_SYNTHETIC_SOURCES = {
    "rapidata-dalle-3", "rapidata-flux", "rapidata-midjourney", "rapidata-stable-diffusion",
}
SELECTOR_REAL_SOURCES = {"british-library-plates"}
REGRESSION_SOURCE_SETS = {
    "m3-selector-regression": ({"flux-1-dev-development"}, {"met-open-access"}),
    "m2-development-regression": ({"GLM-Image", "HunyuanImage-3.0"}, {"open-images", "stockimages-cc0"}),
}


@dataclass
class AdapterCandidate:
    candidate_id: str
    weight_decay: float
    anchor_coefficient: float
    mean: np.ndarray
    std: np.ndarray
    input_weight: np.ndarray
    input_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "m4.feature_mean": self.mean,
            "m4.feature_std": self.std,
            "m4.adapter_in.weight": self.input_weight,
            "m4.adapter_in.bias": self.input_bias,
            "m4.adapter_out.weight": self.output_weight,
            "m4.adapter_out.bias": self.output_bias,
        }


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def array_digest(value: np.ndarray) -> str:
    return feature_pipeline.array_digest(np.asarray(value))


def candidate_seal(candidate: AdapterCandidate) -> dict[str, Any]:
    arrays = dict(sorted(candidate.arrays().items()))
    return {
        "candidateId": candidate.candidate_id,
        "weightDecay": candidate.weight_decay,
        "anchorCoefficient": candidate.anchor_coefficient,
        "trainableParameters": 49_600,
        "tensorSha256": {name: array_digest(value) for name, value in arrays.items()},
        "tensorShapes": {name: list(value.shape) for name, value in arrays.items()},
        "tensorDtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "tensorFloat32Base64": {
            name: base64.b64encode(np.ascontiguousarray(value, dtype=np.float32).tobytes()).decode("ascii")
            for name, value in arrays.items()
        },
    }


def save_candidate(path: Path, candidate: AdapterCandidate) -> None:
    if path.exists():
        if candidate_seal(load_candidate(path)) != candidate_seal(candidate):
            raise ValueError(f"M4 sealed candidate changed on resume: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            candidate_id=np.asarray(candidate.candidate_id),
            weight_decay=np.asarray(candidate.weight_decay, dtype=np.float64),
            anchor_coefficient=np.asarray(candidate.anchor_coefficient, dtype=np.float64),
            feature_mean=candidate.mean,
            feature_std=candidate.std,
            adapter_in_weight=candidate.input_weight,
            adapter_in_bias=candidate.input_bias,
            adapter_out_weight=candidate.output_weight,
            adapter_out_bias=candidate.output_bias,
        )
    os.replace(temporary, path)


def load_candidate(path: Path) -> AdapterCandidate:
    with np.load(path, allow_pickle=False) as data:
        candidate = AdapterCandidate(
            candidate_id=str(data["candidate_id"].item()),
            weight_decay=float(data["weight_decay"].item()),
            anchor_coefficient=float(data["anchor_coefficient"].item()),
            mean=np.asarray(data["feature_mean"], dtype=np.float32).copy(),
            std=np.asarray(data["feature_std"], dtype=np.float32).copy(),
            input_weight=np.asarray(data["adapter_in_weight"], dtype=np.float32).copy(),
            input_bias=np.asarray(data["adapter_in_bias"], dtype=np.float32).copy(),
            output_weight=np.asarray(data["adapter_out_weight"], dtype=np.float32).copy(),
            output_bias=np.asarray(data["adapter_out_bias"], dtype=np.float32).copy(),
        )
    validate_candidate(candidate)
    return candidate


def validate_candidate(candidate: AdapterCandidate) -> None:
    expected_id = {
        (float(weight_decay), float(anchor)): f"wd-{weight_decay:.3f}-anchor-{anchor:.2f}"
        for weight_decay in ADAPTER_WEIGHT_DECAYS
        for anchor in ADAPTER_ANCHOR_COEFFICIENTS
    }.get((candidate.weight_decay, candidate.anchor_coefficient))
    if expected_id is None or candidate.candidate_id != expected_id:
        raise ValueError("M4 candidate identity does not match the frozen hyperparameter grid")
    expected = {
        "m4.feature_mean": (384,),
        "m4.feature_std": (384,),
        "m4.adapter_in.weight": (64, 384),
        "m4.adapter_in.bias": (64,),
        "m4.adapter_out.weight": (384, 64),
        "m4.adapter_out.bias": (384,),
    }
    for name, array in candidate.arrays().items():
        if array.dtype != np.float32 or array.shape != expected[name] or not np.isfinite(array).all():
            raise ValueError(f"M4 candidate tensor is invalid: {name}")
    if np.any(candidate.std < np.float32(1e-5)):
        raise ValueError("M4 feature standard deviation is below the frozen clip")


def fit_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_sources: np.ndarray,
    upstream_weight: np.ndarray,
    upstream_bias: float,
    *,
    weight_decay: float,
    anchor_coefficient: float,
    recipe: dict[str, Any],
) -> AdapterCandidate:
    """Fit one candidate using training arrays only; selector arrays are absent by design."""
    import torch
    from torch import nn

    features = np.asarray(train_features, dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.float32)
    sources = np.asarray(train_sources)
    mean = features.mean(axis=0).astype(np.float32)
    std = features.std(axis=0).clip(min=1e-5).astype(np.float32)
    normalized = torch.from_numpy((features - mean) / std)
    raw = torch.from_numpy(features)
    target = torch.from_numpy(labels).unsqueeze(1)
    source_weights = torch.from_numpy(feature_pipeline.source_balanced_weights(labels, sources))
    upstream_w = torch.from_numpy(np.asarray(upstream_weight, dtype=np.float32)).unsqueeze(1)
    upstream_b = torch.tensor(float(upstream_bias), dtype=torch.float32)
    protected = torch.from_numpy(np.isin(sources, recipe["adapter"]["protectedAnchorSources"]))
    batch_count = math.ceil(features.shape[0] / int(recipe["training"]["batchSize"]))
    protected_count = int(protected.sum().item())
    if protected_count == 0:
        raise ValueError("M4 anchor source set has no protected feature views")

    torch.manual_seed(int(recipe["seed"]))
    adapter_in = nn.Linear(384, 64)
    adapter_out = nn.Linear(64, 384)
    with torch.no_grad():
        nn.init.kaiming_uniform_(adapter_in.weight)
        adapter_in.bias.zero_()
        adapter_out.weight.zero_()
        adapter_out.bias.zero_()
    optimizer = torch.optim.AdamW(
        [*adapter_in.parameters(), *adapter_out.parameters()],
        lr=float(recipe["training"]["learningRate"]),
        weight_decay=float(weight_decay),
    )
    for epoch in range(int(recipe["training"]["epochs"])):
        generator = torch.Generator(device="cpu").manual_seed(int(recipe["seed"]) + epoch)
        order = torch.randperm(features.shape[0], generator=generator)
        for offset in range(0, order.numel(), int(recipe["training"]["batchSize"])):
            indices = order[offset : offset + int(recipe["training"]["batchSize"])]
            residual = adapter_out(torch.relu(adapter_in(normalized[indices])))
            adapted = raw[indices] + torch.from_numpy(std) * residual
            logits = adapted @ upstream_w + upstream_b
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits,
                target[indices],
                reduction="none",
            ).squeeze(1)
            batch_weights = source_weights[indices]
            loss = torch.sum(bce * batch_weights) * batch_count
            anchor_indices = protected[indices]
            if bool(anchor_indices.any()):
                upstream_logits = raw[indices][anchor_indices] @ upstream_w + upstream_b
                anchor_logits = logits[anchor_indices]
                anchor_loss = torch.sum((anchor_logits - upstream_logits) ** 2) / protected_count
                loss = loss + float(anchor_coefficient) * anchor_loss * batch_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    candidate = AdapterCandidate(
        candidate_id=f"wd-{weight_decay:.3f}-anchor-{anchor_coefficient:.2f}",
        weight_decay=float(weight_decay),
        anchor_coefficient=float(anchor_coefficient),
        mean=mean,
        std=std,
        input_weight=adapter_in.weight.detach().cpu().numpy().astype(np.float32),
        input_bias=adapter_in.bias.detach().cpu().numpy().astype(np.float32),
        output_weight=adapter_out.weight.detach().cpu().numpy().astype(np.float32),
        output_bias=adapter_out.bias.detach().cpu().numpy().astype(np.float32),
    )
    validate_candidate(candidate)
    return candidate


def fit_candidate_grid(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_sources: np.ndarray,
    upstream_weight: np.ndarray,
    upstream_bias: float,
    recipe: dict[str, Any],
    candidate_directory: Path,
) -> list[AdapterCandidate]:
    candidates: list[AdapterCandidate] = []
    for weight_decay in ADAPTER_WEIGHT_DECAYS:
        for anchor in ADAPTER_ANCHOR_COEFFICIENTS:
            candidate = fit_candidate(
                train_features,
                train_labels,
                train_sources,
                upstream_weight,
                upstream_bias,
                weight_decay=weight_decay,
                anchor_coefficient=anchor,
                recipe=recipe,
            )
            save_candidate(candidate_directory / f"{candidate.candidate_id}.npz", candidate)
            candidates.append(candidate)
    if len(candidates) != recipe["training"]["candidateCount"]:
        raise AssertionError("M4 candidate grid count changed")
    return candidates


def candidate_logits(
    candidate: AdapterCandidate,
    features: np.ndarray,
    upstream_weight: np.ndarray,
    upstream_bias: float,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    normalized = (values - candidate.mean) / candidate.std
    hidden = np.maximum(
        normalized @ candidate.input_weight.T + candidate.input_bias,
        np.float32(0),
    )
    residual = hidden @ candidate.output_weight.T + candidate.output_bias
    adapted = values + candidate.std * residual
    with np.errstate(over="ignore", invalid="ignore"):
        logits = adapted @ np.asarray(upstream_weight, dtype=np.float32) + np.float32(upstream_bias)
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    if not np.isfinite(logits).all():
        raise FloatingPointError("M4 candidate produced non-finite logits")
    return logits


def variant_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    variants: np.ndarray,
    sources: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    return feature_pipeline.variant_metrics(logits, labels, variants, sources, threshold)


def require_source_sets(
    values: dict[str, Any],
    *,
    synthetic_sources: set[str],
    real_sources: set[str],
    label: str,
) -> None:
    if set(values) != set(VARIANTS):
        raise ValueError(f"{label} variant set changed")
    for variant in VARIANTS:
        row = values[variant]
        if set(row["syntheticRecallBySource"]) != synthetic_sources:
            raise ValueError(f"{label} synthetic source set changed for {variant}")
        if set(row["realRecallBySource"]) != real_sources:
            raise ValueError(f"{label} real source set changed for {variant}")


def passes_gates(values: dict[str, Any], gates: dict[str, Any]) -> bool:
    return feature_pipeline.passes_validation_gates(values, gates)


def threshold_key(values: dict[str, Any], gates: dict[str, Any]) -> tuple[float, ...]:
    rows = [values[name] for name in VARIANTS]
    family = [float(value) for row in rows for value in row["syntheticRecallBySource"].values()]
    required_real = [
        float(row["realRecallBySource"][source])
        for row in rows
        for source in gates.get("minimumRealRecallBySource", {})
    ]
    return (
        min(float(row["balancedAccuracy"]) for row in rows),
        min(float(row["realRecall"]) for row in rows),
        min(float(row["syntheticRecall"]) for row in rows),
        min(family),
        min(required_real) if required_real else 1.0,
    )


def evaluate_selector_candidate(
    candidate: AdapterCandidate,
    selector: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    upstream_weight: np.ndarray,
    upstream_bias: float,
    gates: dict[str, Any],
) -> dict[str, Any]:
    features, labels, variants, sources = selector
    logits = candidate_logits(candidate, features, upstream_weight, upstream_bias)
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    thresholds = complete_decision_thresholds(float(value) for value in logits)
    for threshold in thresholds:
        values = variant_metrics(logits, labels, variants, sources, threshold)
        require_source_sets(
            values,
            synthetic_sources=SELECTOR_SYNTHETIC_SOURCES,
            real_sources=SELECTOR_REAL_SOURCES,
            label="M4 selector",
        )
        if not passes_gates(values, gates):
            continue
        key = threshold_key(values, gates)
        if best is None or key > best[0]:
            best = (key, float(threshold), values)
    encoded = base64.b64encode(np.ascontiguousarray(logits).tobytes()).decode("ascii")
    result: dict[str, Any] = {
        **candidate_seal(candidate),
        "selectorLogitsFloat32Base64": encoded,
        "selectorLogitsSha256": array_digest(logits),
        "selectorLogitCount": int(logits.size),
        "thresholdPartitions": len(thresholds),
        "valid": best is not None,
    }
    if best is not None:
        result.update({
            "rawThreshold": best[1],
            "selectorMetrics": best[2],
            "selectorKey": list(best[0]),
            "candidateSelectionKey": [*best[0], -candidate.weight_decay, -candidate.anchor_coefficient],
        })
    return result


def select_candidate(
    candidates: list[AdapterCandidate],
    selector: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    upstream_weight: np.ndarray,
    upstream_bias: float,
    gates: dict[str, Any],
) -> tuple[list[dict[str, Any]], AdapterCandidate | None, dict[str, Any] | None]:
    rows = [
        evaluate_selector_candidate(candidate, selector, upstream_weight, upstream_bias, gates)
        for candidate in candidates
    ]
    valid = [(row, candidate) for row, candidate in zip(rows, candidates, strict=True) if row["valid"]]
    if not valid:
        return rows, None, None
    selected_row, selected = max(
        valid,
        key=lambda pair: (tuple(pair[0]["candidateSelectionKey"]), pair[0]["candidateId"]),
    )
    return rows, selected, selected_row


def write_exact_or_compare(path: Path, payload: dict[str, Any]) -> str:
    value = canonical_json(payload, pretty=True)
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"M4 sealed state changed on resume: {path.name}")
    else:
        atomic_write(path, value)
    return digest(path)


def write_bytes_exact_or_compare(path: Path, value: bytes) -> str:
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"M4 sealed bytes changed on resume: {path.name}")
    else:
        atomic_write(path, value)
    return digest(path)


def begin_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("status") == "started":
            raise RuntimeError(f"M4 regression outcome is unknown after interruption: {path.name}")
        raise RuntimeError(f"M4 regression has already been scored: {path.name}")
    atomic_write(path, canonical_json({**payload, "status": "started"}, pretty=True))


def complete_once(path: Path, payload: dict[str, Any]) -> None:
    state = json.loads(path.read_text())
    if state.get("status") != "started":
        raise RuntimeError(f"M4 regression state is not open: {path.name}")
    atomic_write(path, canonical_json({**payload, "status": "complete"}, pretty=True))


def safe_manifest_items(manifest: Path, data_root: Path) -> list[Any]:
    root = Path(os.path.abspath(data_root))
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"M4 manifest path escapes its declared data root: {row.get('id')}")
        candidate = root / relative
        current = root
        if root.is_symlink():
            raise ValueError("M4 manifest data root is a symlink")
        for part in Path(relative).parts:
            current /= part
            if current.is_symlink():
                raise ValueError(f"M4 manifest path traverses a symlink: {row.get('id')}")
        try:
            Path(os.path.abspath(candidate)).relative_to(root)
        except ValueError as error:
            raise ValueError(f"M4 manifest path escapes its declared data root: {row.get('id')}") from error
    return feature_pipeline.load_manifest(manifest, data_root)


def validate_selection_summary(path: Path, recipe: dict[str, Any]) -> dict[str, Any]:
    raw = path.read_bytes()
    summary = json.loads(raw)
    if canonical_json(summary, pretty=True) != raw:
        raise ValueError("M4 selection summary is not canonical JSON")
    required = {
        "schemaVersion", "recipeSha256", "sourceLocksSha256", "scoreBlind", "modelOutputsRead",
        "h3PixelsRead", "h3ManifestSha256", "selectionOrder", "training", "freshSelector",
        "partitionGroups", "overlap", "publicArtifacts",
    }
    if set(summary) != required:
        raise ValueError("M4 selection summary schema changed")
    expected_training_sources = dict(recipe["baseTraining"]["sourceCounts"])
    expected_training_sources.update(recipe["expectedTraining"]["newSourceCounts"])
    if (
        summary["schemaVersion"] != 1
        or summary["recipeSha256"] != digest(RECIPE_PATH)
        or summary["sourceLocksSha256"] != digest(LOCKS_PATH)
        or summary["scoreBlind"] is not True
        or summary["modelOutputsRead"] is not False
        or summary["h3PixelsRead"] is not False
        or summary["h3ManifestSha256"] != recipe["h3Exclusion"]["sha256"]
        or summary["training"]["items"] != recipe["expectedTraining"]["images"]
        or summary["training"]["featureViews"] != recipe["expectedTraining"]["featureViews"]
        or summary["training"]["classCounts"] != recipe["expectedTraining"]["classCounts"]
        or summary["training"]["sourceCounts"] != dict(sorted(expected_training_sources.items()))
        or summary["training"]["basePrefixItems"] != recipe["baseTraining"]["items"]
        or summary["training"]["baseExpandedSha256"] != recipe["baseTraining"]["expandedSha256"]
        or summary["freshSelector"]["items"] != recipe["freshSelector"]["items"]
        or summary["freshSelector"]["featureViews"] != recipe["freshSelector"]["featureViews"]
        or summary["freshSelector"]["sourceCounts"] != recipe["freshSelector"]["sourceCounts"]
        or summary["freshSelector"]["classCounts"] != recipe["freshSelector"]["classCounts"]
        or summary["overlap"]["admittedCrossPoolMatches"] != 0
        or summary["overlap"]["reviewExceptions"] != 0
    ):
        raise ValueError("M4 selection summary bindings or composition changed")
    artifacts = summary["publicArtifacts"]
    if set(artifacts) != {
        "attribution.json", "british-source-index.json", "perceptual-review.json",
        "rapidata-source-index.json", "rejects.jsonl", "train-manifest.jsonl",
        "validation-manifest.jsonl",
    }:
        raise ValueError("M4 selection summary public artifact set changed")
    return summary


def decode_regression_logits(row: dict[str, Any]) -> np.ndarray:
    encoded = row.get("logitsFloat32Base64")
    if not isinstance(encoded, str):
        raise ValueError("M4 regression logits are missing")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("M4 regression logits are not canonical base64") from error
    if base64.b64encode(raw).decode("ascii") != encoded or len(raw) != int(row.get("logitCount", -1)) * 4:
        raise ValueError("M4 regression logit bytes changed")
    logits = np.frombuffer(raw, dtype=np.float32).copy()
    if not np.isfinite(logits).all() or array_digest(logits) != row.get("logitsSha256"):
        raise ValueError("M4 regression logit digest changed")
    return logits


def validate_completed_regression_state(
    state: dict[str, Any],
    *,
    name: str,
    selection_lock_sha256: str,
    manifest: Path,
    items: list[Any],
    threshold: float,
    gates: dict[str, Any],
    previous_state_sha256: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = {
        "schemaVersion", "name", "selectionLockSha256", "manifestSha256",
        "previousRegressionStateSha256", "startedBeforeFeatureOrScoreRead", "metrics", "passed",
        "logitsSha256", "logitsFloat32Base64", "logitCount", "featureShardEvidence", "status",
    }
    if set(state) != required or state.get("status") != "complete" or state.get("schemaVersion") != 1:
        raise ValueError(f"M4 completed regression state schema changed: {name}")
    if (
        state.get("name") != name
        or state.get("selectionLockSha256") != selection_lock_sha256
        or state.get("manifestSha256") != digest(manifest)
        or state.get("previousRegressionStateSha256") != previous_state_sha256
        or state.get("startedBeforeFeatureOrScoreRead") is not True
        or type(state.get("passed")) is not bool
        or not isinstance(state.get("featureShardEvidence"), list)
        or not state["featureShardEvidence"]
    ):
        raise ValueError(f"M4 completed regression state binding changed: {name}")
    logits = decode_regression_logits(state)
    expected_labels, expected_variants, expected_sources = feature_pipeline.expected_view_metadata(
        items, training=False, single_view_sources=frozenset(),
    )
    if logits.size != expected_labels.size:
        raise ValueError(f"M4 completed regression logit count changed: {name}")
    metrics = variant_metrics(logits, expected_labels, expected_variants, expected_sources, threshold)
    expected_synthetic, expected_real = REGRESSION_SOURCE_SETS[name]
    require_source_sets(metrics, synthetic_sources=expected_synthetic, real_sources=expected_real, label=name)
    passed = passes_gates(metrics, gates)
    if state["metrics"] != metrics or state["passed"] is not passed:
        raise ValueError(f"M4 completed regression metrics do not recompute: {name}")
    result = {
        key: state[key]
        for key in ("name", "metrics", "passed", "logitsSha256", "logitsFloat32Base64", "logitCount")
    }
    return result, state["featureShardEvidence"]


def evaluate_regression(
    name: str,
    candidate: AdapterCandidate,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    upstream_weight: np.ndarray,
    upstream_bias: float,
    threshold: float,
    gates: dict[str, Any],
) -> dict[str, Any]:
    features, labels, variants, sources = arrays
    logits = candidate_logits(candidate, features, upstream_weight, upstream_bias)
    values = variant_metrics(logits, labels, variants, sources, threshold)
    expected_synthetic, expected_real = REGRESSION_SOURCE_SETS[name]
    require_source_sets(
        values,
        synthetic_sources=expected_synthetic,
        real_sources=expected_real,
        label=name,
    )
    return {
        "name": name,
        "metrics": values,
        "passed": passes_gates(values, gates),
        "logitsSha256": array_digest(logits),
        "logitsFloat32Base64": base64.b64encode(np.ascontiguousarray(logits).tobytes()).decode("ascii"),
        "logitCount": int(logits.size),
    }


def classifier_tensors(model_path: Path) -> tuple[np.ndarray, float]:
    import onnx
    from onnx import numpy_helper

    model = onnx.load(model_path)
    initializers = {value.name: value for value in model.graph.initializer}
    weight = numpy_helper.to_array(initializers["classifier.weight"]).astype(np.float32).reshape(-1)
    bias = float(numpy_helper.to_array(initializers["classifier.bias"]).reshape(-1)[0])
    return weight, bias


def export_adapter_model(source: Path, destination: Path, candidate: AdapterCandidate, recipe: dict[str, Any]) -> None:
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(source)
    contract = recipe["onnxContract"]
    classifier = next((node for node in model.graph.node if node.name == contract["classifierNodeName"]), None)
    if classifier is None or classifier.op_type != "Gemm" or list(classifier.input[1:]) != ["classifier.weight", "classifier.bias"]:
        raise ValueError("Frozen M2 classifier node changed")
    if classifier.input[0] != contract["featureTensor"] or classifier.output[0] != contract["outputTensor"]:
        raise ValueError("Frozen M2 classifier interface changed")
    arrays = candidate.arrays()
    model.graph.initializer.extend(numpy_helper.from_array(value, name) for name, value in arrays.items())
    feature = contract["featureTensor"]
    nodes = [
        helper.make_node("Sub", [feature, "m4.feature_mean"], ["m4.centered"], name="m4_sub_mean"),
        helper.make_node("Div", ["m4.centered", "m4.feature_std"], ["m4.normalized"], name="m4_div_std"),
        helper.make_node(
            "Gemm", ["m4.normalized", "m4.adapter_in.weight", "m4.adapter_in.bias"],
            ["m4.hidden_pre"], name="m4_adapter_in", alpha=1.0, beta=1.0, transB=1,
        ),
        helper.make_node("Relu", ["m4.hidden_pre"], ["m4.hidden"], name="m4_relu"),
        helper.make_node(
            "Gemm", ["m4.hidden", "m4.adapter_out.weight", "m4.adapter_out.bias"],
            ["m4.residual_normalized"], name="m4_adapter_out", alpha=1.0, beta=1.0, transB=1,
        ),
        helper.make_node(
            "Mul", ["m4.residual_normalized", "m4.feature_std"], ["m4.residual"],
            name="m4_scale_residual",
        ),
        helper.make_node("Add", [feature, "m4.residual"], ["m4.adapted"], name="m4_add_residual"),
    ]
    classifier.input[0] = "m4.adapted"
    classifier_index = list(model.graph.node).index(classifier)
    for offset, node in enumerate(nodes):
        model.graph.node.insert(classifier_index + offset, node)
    onnx.checker.check_model(model)
    write_bytes_exact_or_compare(destination, model.SerializeToString(deterministic=True))


def zero_candidate(train_features: np.ndarray, recipe: dict[str, Any]) -> AdapterCandidate:
    mean = np.asarray(train_features, dtype=np.float32).mean(axis=0).astype(np.float32)
    std = np.asarray(train_features, dtype=np.float32).std(axis=0).clip(min=1e-5).astype(np.float32)
    import torch
    from torch import nn

    torch.manual_seed(int(recipe["seed"]))
    adapter_in = nn.Linear(384, 64)
    with torch.no_grad():
        nn.init.kaiming_uniform_(adapter_in.weight)
        adapter_in.bias.zero_()
    return AdapterCandidate(
        candidate_id="zero-adapter-reference",
        weight_decay=0.0,
        anchor_coefficient=0.0,
        mean=mean,
        std=std,
        input_weight=adapter_in.weight.detach().numpy().astype(np.float32),
        input_bias=np.zeros(64, dtype=np.float32),
        output_weight=np.zeros((384, 64), dtype=np.float32),
        output_bias=np.zeros(384, dtype=np.float32),
    )


def feature_level_zero_parity(
    candidate: AdapterCandidate,
    features: np.ndarray,
    upstream_weight: np.ndarray,
    upstream_bias: float,
) -> float:
    with np.errstate(all="ignore"):
        original = np.asarray(features, dtype=np.float32) @ upstream_weight + np.float32(upstream_bias)
    adapted = candidate_logits(candidate, features, upstream_weight, upstream_bias)
    return float(np.max(np.abs(original - adapted)))


def item_counts(items: list[Any]) -> tuple[dict[str, int], dict[str, int]]:
    sources: dict[str, int] = {}
    classes = {"real": 0, "synthetic": 0}
    for item in items:
        sources[item.source] = sources.get(item.source, 0) + 1
        classes["synthetic" if int(item.label) == 1 else "real"] += 1
    return dict(sorted(sources.items())), classes


def image_level_onnx_parity(
    upstream_model: Path,
    zero_model: Path,
    candidate_model: Path,
    selector_items: list[Any],
    selector_features: np.ndarray,
    candidate: AdapterCandidate,
    upstream_weight: np.ndarray,
    upstream_bias: float,
    *,
    batch_size: int,
) -> tuple[float, float]:
    import onnxruntime as ort

    sample_items = selector_items[:6]
    views = [feature_pipeline.preprocess_views(item, False) for item in sample_items]
    tensor = np.stack([pixels for item_views in views for pixels in item_views])
    upstream = ort.InferenceSession(str(upstream_model), providers=["CPUExecutionProvider"])
    zero = ort.InferenceSession(str(zero_model), providers=["CPUExecutionProvider"])
    selected = ort.InferenceSession(str(candidate_model), providers=["CPUExecutionProvider"])
    upstream_logits = np.asarray(upstream.run(["logits"], {"pixel_values": tensor})[0]).reshape(-1)
    zero_logits = np.asarray(zero.run(["logits"], {"pixel_values": tensor})[0]).reshape(-1)
    selected_logits = np.asarray(selected.run(["logits"], {"pixel_values": tensor})[0]).reshape(-1)
    expected = candidate_logits(
        candidate,
        selector_features[: tensor.shape[0]],
        upstream_weight,
        upstream_bias,
    )
    return (
        float(np.max(np.abs(upstream_logits - zero_logits))),
        float(np.max(np.abs(expected - selected_logits))),
    )


def expected_arguments() -> list[str]:
    return [
        "--model", "weights/prooflens-cf384.onnx",
        "--data-root", "benchmark/data/m4-head",
        "--train-manifest", "benchmark/data/m4-head/train-manifest.jsonl",
        "--selector-manifest", "benchmark/evidence/m4/validation-manifest.jsonl",
        "--m3-regression-data-root", "benchmark/data/m3-head",
        "--m3-regression-manifest", "benchmark/evidence/m3/validation-manifest.jsonl",
        "--m2-regression-data-root", "benchmark/data/m2-head",
        "--m2-regression-manifest", "benchmark/evidence/m2/validation-manifest.jsonl",
        "--selection-summary", "benchmark/evidence/m4/selection-summary.json",
        "--single-view-source", "diffusiondb-stable-diffusion",
        "--single-view-source", "open-images-train",
        "--execution-provider", "cpu",
        "--batch-size", "24",
        "--feature-shard-images", "2000",
        "--reextract-cached-features",
        "--output-dir", "benchmark/candidates/prooflens-cf384-m4",
    ]


def require_exact_path(path: Path, relative: str, *, label: str) -> Path:
    expected = Path(os.path.abspath(ROOT / relative))
    actual = Path(os.path.abspath(path))
    if actual != expected:
        raise ValueError(f"M4 {label} path changed")
    current = ROOT
    for part in expected.relative_to(ROOT).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"M4 {label} traverses a symlink")
    return expected


def validate_inputs(args: argparse.Namespace, recipe: dict[str, Any]) -> dict[str, str]:
    paths = {
        "model": require_exact_path(args.model, recipe["upstreamModel"]["path"], label="model"),
        "train": require_exact_path(args.train_manifest, "benchmark/data/m4-head/train-manifest.jsonl", label="training manifest"),
        "selector": require_exact_path(args.selector_manifest, recipe["freshSelector"]["manifest"], label="selector manifest"),
        "summary": require_exact_path(args.selection_summary, "benchmark/evidence/m4/selection-summary.json", label="selection summary"),
        "m3": require_exact_path(args.m3_regression_manifest, recipe["regressions"][0]["manifest"], label="M3 regression"),
        "m2": require_exact_path(args.m2_regression_manifest, recipe["regressions"][1]["manifest"], label="M2 regression"),
    }
    require_exact_path(args.data_root, recipe["output"]["dataRoot"], label="data root")
    require_exact_path(args.m3_regression_data_root, recipe["regressions"][0]["dataRoot"], label="M3 regression root")
    require_exact_path(args.m2_regression_data_root, recipe["regressions"][1]["dataRoot"], label="M2 regression root")
    require_exact_path(args.output_dir, recipe["output"]["candidateRoot"], label="candidate root")
    if args.model.stat().st_size != recipe["upstreamModel"]["bytes"] or digest(args.model) != UPSTREAM_MODEL_SHA256:
        raise ValueError("M4 upstream M2 model changed")
    summary = validate_selection_summary(paths["summary"], recipe)
    public_train = summary["publicArtifacts"]["train-manifest.jsonl"]
    if digest(paths["train"]) != public_train["expandedSha256"]:
        raise ValueError("M4 local training manifest does not match public source evidence")
    if digest(paths["selector"]) != summary["publicArtifacts"]["validation-manifest.jsonl"]["sha256"]:
        raise ValueError("M4 selector manifest does not match public source evidence")
    for index, name in enumerate(("m3", "m2")):
        if digest(paths[name]) != recipe["regressions"][index]["sha256"]:
            raise ValueError(f"M4 {name.upper()} regression manifest changed")
    if sorted(args.single_view_source) != sorted(recipe["expectedTraining"]["singleViewSources"]):
        raise ValueError("M4 single-view sources changed")
    if args.execution_provider != "cpu" or args.batch_size != 24 or args.feature_shard_images != 2000:
        raise ValueError("M4 feature-extraction command changed")
    if not args.reextract_cached_features or sys.argv[1:] != expected_arguments():
        raise ValueError("M4 canonical training arguments changed")
    return {name: digest(path) for name, path in paths.items()}


def extract(
    session: Any,
    items: list[Any],
    manifest: Path,
    output_dir: Path,
    name: str,
    model_hash: str,
    batch_size: int,
    training: bool,
    shard_images: int,
    single_view_sources: frozenset[str],
    run_id: str,
    evidence: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return feature_pipeline.extract_or_load_sharded(
        session,
        items,
        manifest,
        output_dir / "features",
        name,
        model_hash,
        batch_size,
        training,
        shard_images,
        single_view_sources,
        True,
        run_id,
        evidence,
    )


def create_session(args: argparse.Namespace) -> tuple[Any, np.ndarray, float, tuple[str, ...]]:
    import onnxruntime as ort

    feature_model = args.output_dir / "feature-model.onnx"
    weight, bias = feature_pipeline.make_feature_model(args.model, feature_model, args.batch_size)
    session = ort.InferenceSession(str(feature_model), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"M4 requires CPU-only feature extraction: {session.get_providers()}")
    return session, weight, bias, tuple(session.get_providers())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--m3-regression-data-root", type=Path, required=True)
    parser.add_argument("--m3-regression-manifest", type=Path, required=True)
    parser.add_argument("--m2-regression-data-root", type=Path, required=True)
    parser.add_argument("--m2-regression-manifest", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--single-view-source", action="append", default=[])
    parser.add_argument("--execution-provider", choices=("cpu",), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--feature-shard-images", type=int, required=True)
    parser.add_argument("--reextract-cached-features", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe, _ = load_frozen_protocol(RECIPE_PATH, LOCKS_PATH)
    input_hashes = validate_inputs(args, recipe)
    import onnxruntime as ort
    import torch
    from PIL import __version__ as pillow_version

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_items = safe_manifest_items(args.train_manifest, args.data_root)
    selector_items = safe_manifest_items(args.selector_manifest, args.data_root)
    if len(train_items) != recipe["expectedTraining"]["images"] or len(selector_items) != recipe["freshSelector"]["items"]:
        raise ValueError("M4 train or selector item count changed")
    if {item.id for item in train_items} & {item.id for item in selector_items}:
        raise ValueError("M4 selector IDs entered training")
    if {item.image_sha256 for item in train_items} & {item.image_sha256 for item in selector_items}:
        raise ValueError("M4 selector bytes entered training")
    session, upstream_weight, upstream_bias, providers = create_session(args)
    feature_hashes = {
        "training": feature_pipeline.feature_configuration_hash(
            training=True,
            single_view_sources=frozenset(args.single_view_source),
            providers=providers,
            feature_batch_size=args.batch_size,
        ),
        "evaluation": feature_pipeline.feature_configuration_hash(
            training=False,
            single_view_sources=frozenset(),
            providers=providers,
            feature_batch_size=args.batch_size,
        ),
    }
    context = {
        "pipelineVersion": PIPELINE_VERSION,
        "trainerSha256": digest(Path(__file__)),
        "recipeSha256": digest(RECIPE_PATH),
        "modelSha256": input_hashes["model"],
        "trainManifestSha256": input_hashes["train"],
        "selectorManifestSha256": input_hashes["selector"],
        "m3RegressionManifestSha256": input_hashes["m3"],
        "m2RegressionManifestSha256": input_hashes["m2"],
        "selectionSummarySha256": input_hashes["summary"],
        "featureConfigurationHashes": feature_hashes,
        "featureBatchSize": args.batch_size,
        "featureShardImages": args.feature_shard_images,
        "singleViewSources": sorted(args.single_view_source),
    }
    marker_path = args.output_dir / "fresh-feature-run.json"
    marker = feature_pipeline.open_or_create_fresh_feature_run(marker_path, context)
    run_id = str(marker["runId"])
    evidence: list[dict[str, Any]] = []
    train = extract(
        session, train_items, args.train_manifest, args.output_dir, "train", UPSTREAM_MODEL_SHA256,
        args.batch_size, True, args.feature_shard_images, frozenset(args.single_view_source), run_id, evidence,
    )
    selector = extract(
        session, selector_items, args.selector_manifest, args.output_dir, "selector", UPSTREAM_MODEL_SHA256,
        args.batch_size, False, args.feature_shard_images, frozenset(), run_id, evidence,
    )
    if train[0].shape[0] != recipe["expectedTraining"]["featureViews"] or selector[0].shape[0] != recipe["freshSelector"]["featureViews"]:
        raise ValueError("M4 feature-view count changed")

    zero = zero_candidate(train[0], recipe)
    zero_error = feature_level_zero_parity(zero, selector[0], upstream_weight, upstream_bias)
    if zero_error != 0.0:
        raise RuntimeError(f"M4 zero adapter is not exactly M2-equivalent at feature level: {zero_error}")
    candidates = fit_candidate_grid(
        train[0], train[1], train[3], upstream_weight, upstream_bias, recipe,
        args.output_dir / "candidates",
    )
    seals = [candidate_seal(candidate) for candidate in candidates]
    seal_path = args.output_dir / "candidate-tensor-seal.json"
    seal_hash = write_exact_or_compare(seal_path, {
        "schemaVersion": 1,
        "createdBeforeSelectorEvaluation": True,
        "candidateCount": len(seals),
        "candidates": seals,
    })
    grid, selected, selected_row = select_candidate(
        candidates, selector, upstream_weight, upstream_bias, recipe["validationGates"],
    )
    grid_packet = {
        "schemaVersion": 1,
        "candidateTensorSealSha256": seal_hash,
        "selectorManifestSha256": input_hashes["selector"],
        "candidateCount": len(grid),
        "validCandidateCount": sum(bool(row["valid"]) for row in grid),
        "candidates": grid,
    }
    write_exact_or_compare(args.output_dir / "candidate-grid.json", grid_packet)
    common_summary = {
        "schemaVersion": 1,
        "pipelineVersion": PIPELINE_VERSION,
        "status": "failed-selector" if selected is None else "selected",
        "seed": SEED,
        "commandArguments": sys.argv[1:],
        "trainerSha256": digest(Path(__file__)),
        "recipeSha256": digest(RECIPE_PATH),
        "sourceLocksSha256": digest(LOCKS_PATH),
        "selectionSummarySha256": input_hashes["summary"],
        "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
        "trainManifestSha256": input_hashes["train"],
        "selectorManifestSha256": input_hashes["selector"],
        "m3RegressionManifestSha256": input_hashes["m3"],
        "m2RegressionManifestSha256": input_hashes["m2"],
        "trainImages": len(train_items),
        "trainFeatureViews": int(train[0].shape[0]),
        "trainSourceCounts": item_counts(train_items)[0],
        "trainClassCounts": item_counts(train_items)[1],
        "selectorImages": len(selector_items),
        "selectorFeatureViews": int(selector[0].shape[0]),
        "selectorSourceCounts": item_counts(selector_items)[0],
        "selectorClassCounts": item_counts(selector_items)[1],
        "candidateTensorSealSha256": seal_hash,
        "candidateGridSha256": digest(args.output_dir / "candidate-grid.json"),
        "zeroAdapterFeatureParityMaximumAbsoluteError": zero_error,
        "featureConfigurationHashes": feature_hashes,
        "featureShardEvidence": evidence,
        "freshFeatureRunId": run_id,
        "freshFeatureMarkerSha256": feature_pipeline.marker_sha256(marker),
        "sourceBalancedLoss": True,
        "anchorLossProtectedSources": recipe["adapter"]["protectedAnchorSources"],
        "candidateCount": len(grid),
        "validCandidateCount": grid_packet["validCandidateCount"],
        "regressionOrder": recipe["selectionPolicy"]["regressionOrder"],
        "h3HoldoutScored": False,
        "h3PixelsRead": False,
        "selectionInfluencedByRegression": False,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "onnxRuntime": ort.__version__,
            "pillow": pillow_version,
            "platform": platform.platform(),
            "providers": list(providers),
        },
    }
    if selected is None or selected_row is None:
        atomic_write(args.output_dir / "validation-summary.json", canonical_json(common_summary, pretty=True))
        raise RuntimeError("No M4 candidate satisfies the frozen fresh-selector gates")

    selection_lock = {
        "schemaVersion": 1,
        "candidateTensorSealSha256": seal_hash,
        "candidateGridSha256": digest(args.output_dir / "candidate-grid.json"),
        "selectedCandidateId": selected.candidate_id,
        "selectedTensorSha256": candidate_seal(selected)["tensorSha256"],
        "rawThreshold": selected_row["rawThreshold"],
        "selectorMetrics": selected_row["selectorMetrics"],
        "candidateSelectionKey": selected_row["candidateSelectionKey"],
        "selectorManifestSha256": input_hashes["selector"],
        "createdBeforeRegressionEvaluation": True,
        "selectionInfluencedByRegression": False,
        "h3HoldoutScored": False,
    }
    selection_lock_path = args.output_dir / "selection-lock.json"
    selection_lock_hash = write_exact_or_compare(selection_lock_path, selection_lock)

    regression_results: list[dict[str, Any]] = []
    previous_regression_state_sha256: str | None = None
    regression_specs = [
        (
            "m3-selector-regression", args.m3_regression_manifest, args.m3_regression_data_root,
            recipe["regressions"][0], "regression-m3",
        ),
        (
            "m2-development-regression", args.m2_regression_manifest, args.m2_regression_data_root,
            recipe["regressions"][1], "regression-m2",
        ),
    ]
    for name, manifest, data_root, spec, cache_name in regression_specs:
        state_path = args.output_dir / f"{name}-state.json"
        items = safe_manifest_items(manifest, data_root)
        if state_path.exists():
            prior_bytes = state_path.read_bytes()
            prior = json.loads(prior_bytes)
            if canonical_json(prior, pretty=True) != prior_bytes:
                raise RuntimeError(f"M4 completed regression state is not canonical: {state_path.name}")
            if prior.get("status") == "started":
                raise RuntimeError(f"M4 regression outcome is unknown after interruption: {state_path.name}")
            result, prior_evidence = validate_completed_regression_state(
                prior,
                name=name,
                selection_lock_sha256=selection_lock_hash,
                manifest=manifest,
                items=items,
                threshold=float(selected_row["rawThreshold"]),
                gates=spec["gates"],
                previous_state_sha256=previous_regression_state_sha256,
            )
            evidence.extend(prior_evidence)
            regression_results.append(result)
            previous_regression_state_sha256 = digest(state_path)
            if not result["passed"]:
                failed = {
                    **common_summary,
                    "status": f"failed-{name}",
                    "selectionLockSha256": selection_lock_hash,
                    "selectionLock": selection_lock,
                    "selectedCandidate": selected_row,
                    "regressions": regression_results,
                    "featureShardEvidence": evidence,
                }
                atomic_write(args.output_dir / "validation-summary.json", canonical_json(failed, pretty=True))
                raise RuntimeError(f"Selected M4 candidate previously failed terminal regression: {name}")
            continue
        begin_once(state_path, {
            "schemaVersion": 1,
            "name": name,
            "selectionLockSha256": selection_lock_hash,
            "manifestSha256": digest(manifest),
            "previousRegressionStateSha256": previous_regression_state_sha256,
            "startedBeforeFeatureOrScoreRead": True,
        })
        evidence_start = len(evidence)
        arrays = extract(
            session, items, manifest, args.output_dir, cache_name, UPSTREAM_MODEL_SHA256,
            args.batch_size, False, args.feature_shard_images, frozenset(), run_id, evidence,
        )
        if len(items) != spec["items"] or arrays[0].shape[0] != spec["featureViews"]:
            raise ValueError(f"M4 {name} item/view count changed")
        result = evaluate_regression(
            name, selected, arrays, upstream_weight, upstream_bias,
            float(selected_row["rawThreshold"]), spec["gates"],
        )
        complete_once(state_path, {
            "schemaVersion": 1,
            "name": name,
            "selectionLockSha256": selection_lock_hash,
            "manifestSha256": digest(manifest),
            "previousRegressionStateSha256": previous_regression_state_sha256,
            "startedBeforeFeatureOrScoreRead": True,
            **result,
            "featureShardEvidence": evidence[evidence_start:],
        })
        regression_results.append(result)
        previous_regression_state_sha256 = digest(state_path)
        if not result["passed"]:
            failed = {
                **common_summary,
                "status": f"failed-{name}",
                "selectionLockSha256": selection_lock_hash,
                "selectionLock": selection_lock,
                "selectedCandidate": selected_row,
                "regressions": regression_results,
                "featureShardEvidence": evidence,
            }
            atomic_write(args.output_dir / "validation-summary.json", canonical_json(failed, pretty=True))
            raise RuntimeError(f"Selected M4 candidate failed terminal regression: {name}")

    complete_marker = feature_pipeline.complete_fresh_feature_run(marker_path, context, run_id)
    selected_model = args.output_dir / "model.onnx"
    export_adapter_model(args.model, selected_model, selected, recipe)
    zero_model = args.output_dir / "zero-adapter.onnx"
    export_adapter_model(args.model, zero_model, zero, recipe)
    zero_image_error, selected_image_error = image_level_onnx_parity(
        args.model,
        zero_model,
        selected_model,
        selector_items,
        selector[0],
        selected,
        upstream_weight,
        upstream_bias,
        batch_size=args.batch_size,
    )
    if zero_image_error > recipe["runtimeParity"]["zeroAdapterMaximumAbsoluteLogitError"]:
        raise RuntimeError(f"M4 zero-adapter ONNX parity failed: {zero_image_error}")
    if selected_image_error > recipe["runtimeParity"]["exportedCandidateMaximumAbsoluteLogitError"]:
        raise RuntimeError(f"M4 exported candidate ONNX parity failed: {selected_image_error}")
    model_hash = digest(selected_model)
    raw_threshold = float(selected_row["rawThreshold"])
    display_intercept = math.log(0.65 / 0.35) - raw_threshold
    calibration = {
        "schemaVersion": 1,
        "mode": "threshold-alignment-not-probability-calibration",
        "slope": 1.0,
        "intercept": display_intercept,
        "rawThreshold": raw_threshold,
        "displayThreshold": 0.65,
        "rawProbabilityAtThreshold": sigmoid(raw_threshold),
        "displayProbabilityAtRawThreshold": sigmoid(raw_threshold + display_intercept),
        "modelSha256": model_hash,
        "selectionLockSha256": selection_lock_hash,
        "selectorManifestSha256": input_hashes["selector"],
    }
    atomic_write(args.output_dir / "calibration.json", canonical_json(calibration, pretty=True))
    summary = {
        **common_summary,
        "status": "accepted-development-candidate",
        "selectionLockSha256": selection_lock_hash,
        "selectionLock": selection_lock,
        "selectedCandidate": selected_row,
        "regressions": regression_results,
        "featureShardEvidence": evidence,
        "freshFeatureMarkerSha256": feature_pipeline.marker_sha256(complete_marker),
        "freshFeatureRunComplete": complete_marker["state"] == "complete",
        "modelSha256": model_hash,
        "modelBytes": selected_model.stat().st_size,
        "zeroAdapterImageParityMaximumAbsoluteError": zero_image_error,
        "exportedCandidateImageParityMaximumAbsoluteError": selected_image_error,
        "calibrationSha256": digest(args.output_dir / "calibration.json"),
        "h3HoldoutScored": False,
        "h3PixelsRead": False,
        "selectionInfluencedByRegression": False,
    }
    atomic_write(args.output_dir / "validation-summary.json", canonical_json(summary, pretty=True))
    print(json.dumps({
        "status": summary["status"],
        "selectedCandidate": selected.candidate_id,
        "modelSha256": model_hash,
        "h3HoldoutScored": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
