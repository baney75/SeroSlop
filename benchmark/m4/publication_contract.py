"""Pure evidence and publication contract for the frozen M4 adapter protocol."""

from __future__ import annotations

from base64 import b64decode, b64encode
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m4.contracts import (  # noqa: E402
    ADAPTER_ANCHOR_COEFFICIENTS,
    ADAPTER_WEIGHT_DECAYS,
    VARIANTS,
    canonical_json,
    load_frozen_protocol,
)
from benchmark.fresh_feature_run import marker_bytes as fresh_marker_bytes  # noqa: E402
from benchmark.thresholds import complete_decision_thresholds  # noqa: E402


PROFILE = "m4"
BASE_COMMIT = "439b2481dc88a887f8317be669096495760fbeb1"
BASE_TREE = "440931a595c87ca3d293f5a6f980c75169ddb899"
UPSTREAM_MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47"
UPSTREAM_MODEL_BYTES = 87_442_080
RECIPE_PATH = ROOT / "benchmark/m4/recipe.json"
LOCKS_PATH = ROOT / "benchmark/m4/source-locks.json"
SELECTION_SUMMARY_PATH = ROOT / "benchmark/evidence/m4/selection-summary.json"
CANDIDATE_DIR = ROOT / "benchmark/candidates/prooflens-cf384-m4"
PUBLICATION_LOCK_PATH = ROOT / "benchmark/evidence/m4/publication-lock.json"
PUBLICATION_ROWS = (
    ("BENCHMARK.md", "M"),
    ("MODEL_CARD.md", "M"),
    ("README.md", "M"),
    ("benchmark/evidence/m4/calibration.json", "A"),
    ("benchmark/evidence/m4/candidate-grid.json", "A"),
    ("benchmark/evidence/m4/finalization-receipt.json", "A"),
    ("benchmark/evidence/m4/model-comparison.json", "A"),
    ("benchmark/evidence/m4/training-summary.json", "A"),
    ("model-lock.json", "M"),
    ("tests/fixtures/model-states/fixture-manifest.json", "M"),
    ("weights/README.md", "M"),
    ("weights/prooflens-cf384.onnx", "M"),
)
TRANSACTIONAL_ROWS = PUBLICATION_ROWS
ADDED_INITIALIZERS = (
    "m4.feature_mean",
    "m4.feature_std",
    "m4.adapter_in.weight",
    "m4.adapter_in.bias",
    "m4.adapter_out.weight",
    "m4.adapter_out.bias",
)
ADDED_INITIALIZER_SHAPES = {
    "m4.feature_mean": [384],
    "m4.feature_std": [384],
    "m4.adapter_in.weight": [64, 384],
    "m4.adapter_in.bias": [64],
    "m4.adapter_out.weight": [384, 64],
    "m4.adapter_out.bias": [384],
}
ADDED_NODES = (
    ("m4_sub_mean", "Sub", ["/Gather_output_0", "m4.feature_mean"], ["m4.centered"]),
    ("m4_div_std", "Div", ["m4.centered", "m4.feature_std"], ["m4.normalized"]),
    ("m4_adapter_in", "Gemm", ["m4.normalized", "m4.adapter_in.weight", "m4.adapter_in.bias"], ["m4.hidden_pre"]),
    ("m4_relu", "Relu", ["m4.hidden_pre"], ["m4.hidden"]),
    ("m4_adapter_out", "Gemm", ["m4.hidden", "m4.adapter_out.weight", "m4.adapter_out.bias"], ["m4.residual_normalized"]),
    ("m4_scale_residual", "Mul", ["m4.residual_normalized", "m4.feature_std"], ["m4.residual"]),
    ("m4_add_residual", "Add", ["/Gather_output_0", "m4.residual"], ["m4.adapted"]),
)
SELECTOR_SYNTHETIC_SOURCES = {
    "rapidata-dalle-3", "rapidata-flux", "rapidata-midjourney", "rapidata-stable-diffusion",
}
SELECTOR_REAL_SOURCES = {"british-library-plates"}
GRID_TENSOR_KEYS = (
    "candidateId", "weightDecay", "anchorCoefficient", "trainableParameters",
    "tensorSha256", "tensorShapes", "tensorDtypes", "tensorFloat32Base64",
)
EXPECTED_CANDIDATE_PAIRS = {
    f"wd-{weight_decay:.3f}-anchor-{anchor:.2f}": (float(weight_decay), float(anchor))
    for weight_decay in ADAPTER_WEIGHT_DECAYS
    for anchor in ADAPTER_ANCHOR_COEFFICIENTS
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def candidate_artifact_path(candidate_id: object) -> Path:
    """Resolve one frozen candidate without accepting traversal or symlink indirection."""
    if not isinstance(candidate_id, str) or candidate_id not in EXPECTED_CANDIDATE_PAIRS:
        raise ValueError("M4 candidate ID is outside the frozen hyperparameter grid")
    directory = CANDIDATE_DIR / "candidates"
    try:
        relative_directory = directory.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("M4 candidate directory escaped the repository") from error
    current = ROOT
    for part in relative_directory.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("M4 candidate directory contains a symlink")
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    resolved_root = ROOT.resolve(strict=True)
    resolved_directory = directory.resolve(strict=True)
    try:
        resolved_directory.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("M4 candidate directory resolved outside the repository") from error
    path = directory / f"{candidate_id}.npz"
    if path.parent != directory or path.is_symlink() or not path.is_file():
        raise ValueError(f"M4 candidate artifact is not a direct regular file: {candidate_id}")
    if path.resolve(strict=True).parent != resolved_directory:
        raise ValueError(f"M4 candidate artifact resolved outside its directory: {candidate_id}")
    return path


def require_grid_tensor_bindings(
    grid: dict[str, Any], tensor_seal: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidates = tensor_seal.get("candidates")
    rows = grid.get("candidates")
    if not isinstance(candidates, list) or not isinstance(rows, list):
        raise ValueError("M4 candidate tensor records are missing")
    sealed = {row.get("candidateId"): row for row in candidates if isinstance(row, dict)}
    if set(sealed) != set(EXPECTED_CANDIDATE_PAIRS) or len(candidates) != len(sealed):
        raise ValueError("M4 candidate tensor seal IDs changed")
    observed: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("M4 candidate grid row changed")
        candidate_id = row.get("candidateId")
        expected = sealed.get(candidate_id)
        actual = {key: row.get(key) for key in GRID_TENSOR_KEYS}
        if candidate_id in observed or expected is None or actual != expected:
            raise ValueError(f"M4 candidate grid tensor binding changed: {candidate_id}")
        observed.add(candidate_id)
    if observed != set(sealed):
        raise ValueError("M4 candidate grid does not cover the frozen tensor seal")
    return sealed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key in M4 publication packet: {key}")
        value[key] = item
    return value


def parse_canonical_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = value.decode("utf-8", errors="strict")
        parsed = json.loads(text, object_pairs_hook=_unique_object, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict) or canonical_json(parsed, pretty=True) != value:
        raise ValueError(f"{label} bytes are not canonical")
    return parsed


def parse_canonical_publication_lock(value: bytes | str) -> dict[str, Any]:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return parse_canonical_json_bytes(raw, label="M4 publication lock")


def load_canonical(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return parse_canonical_json_bytes(path.read_bytes(), label=label)


def load_fresh_feature_marker(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    try:
        parsed = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("M4 fresh feature marker is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict) or fresh_marker_bytes(parsed) != raw:
        raise ValueError("M4 fresh feature marker bytes are not canonical")
    return parsed


def expected_training_arguments() -> list[str]:
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


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    fingerprint = sha256()
    fingerprint.update(array.dtype.str.encode())
    fingerprint.update(json.dumps(array.shape, separators=(",", ":")).encode())
    fingerprint.update(array.tobytes())
    return fingerprint.hexdigest()


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value)) if value >= 0 else math.exp(value) / (1.0 + math.exp(value))


def variant_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    variants: np.ndarray,
    sources: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for index, name in enumerate(VARIANTS):
        selected = variants == index
        y = labels[selected]
        z = logits[selected]
        source_values = sources[selected]
        real = y == 0
        synthetic = y == 1
        real_recall = float((z[real] < threshold).mean())
        synthetic_recall = float((z[synthetic] >= threshold).mean())
        output[name] = {
            "balancedAccuracy": (real_recall + synthetic_recall) / 2,
            "realRecall": real_recall,
            "syntheticRecall": synthetic_recall,
            "syntheticRecallBySource": {
                source: float((z[(source_values == source) & synthetic] >= threshold).mean())
                for source in sorted(set(source_values[synthetic].tolist()))
            },
            "realRecallBySource": {
                source: float((z[(source_values == source) & real] < threshold).mean())
                for source in sorted(set(source_values[real].tolist()))
            },
        }
    return output


def passes_gates(values: dict[str, Any], gates: dict[str, Any]) -> bool:
    rows = list(values.values())
    family = [float(recall) for row in rows for recall in row["syntheticRecallBySource"].values()]
    return (
        all(
            float(row["balancedAccuracy"]) >= float(gates["minimumBalancedAccuracyPerVariant"])
            and float(row["realRecall"]) >= float(gates["minimumRealRecallPerVariant"])
            and float(row["syntheticRecall"]) >= float(gates["minimumSyntheticRecallPerVariant"])
            for row in rows
        )
        and min(family) >= float(gates["minimumSyntheticRecallPerFamily"])
        and all(
            source in row["realRecallBySource"]
            and float(row["realRecallBySource"][source]) >= float(minimum)
            for row in rows
            for source, minimum in gates.get("minimumRealRecallBySource", {}).items()
        )
    )


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


def load_manifest_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != len({row.get("id") for row in rows}):
        raise ValueError(f"M4 manifest contains duplicate IDs: {path}")
    labels: list[float] = []
    variants: list[int] = []
    sources: list[str] = []
    for row in rows:
        if row.get("label") not in (0, 1) or not isinstance(row.get("source"), str):
            raise ValueError(f"M4 manifest row is malformed: {path}")
        for index, _variant in enumerate(VARIANTS):
            labels.append(float(row["label"]))
            variants.append(index)
            sources.append(row["source"])
    return np.asarray(labels, dtype=np.float32), np.asarray(variants, dtype=np.int64), np.asarray(sources), len(rows)


def decode_logits(row: dict[str, Any], key: str, count_key: str, digest_key: str) -> np.ndarray:
    encoded = row.get(key)
    if not isinstance(encoded, str):
        raise ValueError(f"M4 {key} is missing")
    try:
        raw = b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError(f"M4 {key} is not canonical base64") from error
    if b64encode(raw).decode("ascii") != encoded or len(raw) != int(row.get(count_key, -1)) * 4:
        raise ValueError(f"M4 {key} length changed")
    values = np.frombuffer(raw, dtype=np.float32).copy()
    if not np.isfinite(values).all() or array_digest(values) != row.get(digest_key):
        raise ValueError(f"M4 {key} digest or values changed")
    return values


def require_source_sets(metrics: dict[str, Any], synthetic: set[str], real: set[str], label: str) -> None:
    if set(metrics) != set(VARIANTS):
        raise ValueError(f"{label} variant set changed")
    for variant in VARIANTS:
        if set(metrics[variant]["syntheticRecallBySource"]) != synthetic:
            raise ValueError(f"{label} synthetic source set changed for {variant}")
        if set(metrics[variant]["realRecallBySource"]) != real:
            raise ValueError(f"{label} real source set changed for {variant}")


def recompute_candidate_row(
    row: dict[str, Any],
    labels: np.ndarray,
    variants: np.ndarray,
    sources: np.ndarray,
    gates: dict[str, Any],
) -> dict[str, Any]:
    logits = decode_logits(row, "selectorLogitsFloat32Base64", "selectorLogitCount", "selectorLogitsSha256")
    if logits.shape != labels.shape:
        raise ValueError("M4 selector logit count changed")
    best: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    thresholds = complete_decision_thresholds(float(value) for value in logits)
    for threshold in thresholds:
        metrics = variant_metrics(logits, labels, variants, sources, threshold)
        require_source_sets(metrics, SELECTOR_SYNTHETIC_SOURCES, SELECTOR_REAL_SOURCES, "M4 selector")
        if not passes_gates(metrics, gates):
            continue
        key = threshold_key(metrics, gates)
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    expected: dict[str, Any] = {
        **{key: row[key] for key in (
            "candidateId", "weightDecay", "anchorCoefficient", "trainableParameters",
            "tensorSha256", "tensorShapes", "tensorDtypes", "tensorFloat32Base64",
            "selectorLogitsFloat32Base64", "selectorLogitsSha256", "selectorLogitCount",
        )},
        "thresholdPartitions": len(thresholds),
        "valid": best is not None,
    }
    if best is not None:
        expected.update({
            "rawThreshold": best[1],
            "selectorMetrics": best[2],
            "selectorKey": list(best[0]),
            "candidateSelectionKey": [*best[0], -float(row["weightDecay"]), -float(row["anchorCoefficient"])],
        })
    if expected != row:
        raise ValueError(f"M4 selector evidence does not recompute: {row.get('candidateId')}")
    return expected


def recompute_regression_row(
    row: dict[str, Any],
    spec: dict[str, Any],
    expected_synthetic: set[str],
    expected_real: set[str],
    threshold: float,
) -> dict[str, Any]:
    manifest = ROOT / spec["manifest"]
    labels, variants, sources, items = load_manifest_metadata(manifest)
    if items != spec["items"] or labels.size != spec["featureViews"]:
        raise ValueError(f"M4 regression manifest count changed: {spec['name']}")
    logits = decode_logits(row, "logitsFloat32Base64", "logitCount", "logitsSha256")
    if logits.shape != labels.shape:
        raise ValueError(f"M4 regression logit count changed: {spec['name']}")
    metrics = variant_metrics(logits, labels, variants, sources, threshold)
    require_source_sets(metrics, expected_synthetic, expected_real, spec["name"])
    expected = {
        "name": spec["name"],
        "metrics": metrics,
        "passed": passes_gates(metrics, spec["gates"]),
        "logitsSha256": array_digest(logits),
        "logitsFloat32Base64": b64encode(np.ascontiguousarray(logits).tobytes()).decode("ascii"),
        "logitCount": int(logits.size),
    }
    if row != expected:
        raise ValueError(f"M4 regression evidence does not recompute: {spec['name']}")
    return expected


def validate_fresh_feature_evidence(
    summary: dict[str, Any],
    recipe: dict[str, Any],
    *,
    marker_state: str,
    completed_regressions: int,
) -> dict[str, Any]:
    """Bind every reported shard to its manifest, source pixels, arrays, and run marker."""
    from benchmark.m4 import train_adapter

    feature_pipeline = train_adapter.feature_pipeline
    feature_hashes = {
        "training": feature_pipeline.feature_configuration_hash(
            training=True,
            single_view_sources=frozenset(recipe["expectedTraining"]["singleViewSources"]),
            providers=("CPUExecutionProvider",),
            feature_batch_size=24,
        ),
        "evaluation": feature_pipeline.feature_configuration_hash(
            training=False,
            single_view_sources=frozenset(),
            providers=("CPUExecutionProvider",),
            feature_batch_size=24,
        ),
    }
    if summary.get("featureConfigurationHashes") != feature_hashes:
        raise ValueError("M4 fresh feature-configuration hashes changed")
    context = {
        "pipelineVersion": 1,
        "trainerSha256": digest(ROOT / "benchmark/m4/train_adapter.py"),
        "recipeSha256": digest(RECIPE_PATH),
        "modelSha256": UPSTREAM_MODEL_SHA256,
        "trainManifestSha256": summary["trainManifestSha256"],
        "selectorManifestSha256": summary["selectorManifestSha256"],
        "m3RegressionManifestSha256": summary["m3RegressionManifestSha256"],
        "m2RegressionManifestSha256": summary["m2RegressionManifestSha256"],
        "selectionSummarySha256": digest(SELECTION_SUMMARY_PATH),
        "featureConfigurationHashes": feature_hashes,
        "featureBatchSize": 24,
        "featureShardImages": 2_000,
        "singleViewSources": sorted(recipe["expectedTraining"]["singleViewSources"]),
    }
    marker_path = CANDIDATE_DIR / "fresh-feature-run.json"
    marker = load_fresh_feature_marker(marker_path)
    expected_marker = {
        "schemaVersion": 1,
        "runId": summary.get("freshFeatureRunId"),
        "state": marker_state,
        "context": context,
    }
    if marker != expected_marker:
        changed = sorted(key for key in expected_marker["context"] if marker.get("context", {}).get(key) != expected_marker["context"][key])
        raise ValueError(f"M4 fresh feature-run marker context changed: {changed}")
    if summary.get("freshFeatureMarkerSha256") != feature_pipeline.marker_sha256(marker):
        raise ValueError("M4 fresh feature-run marker digest changed")

    partitions: list[tuple[str, Path, Path, bool, frozenset[str]]] = [
        (
            "train",
            ROOT / "benchmark/data/m4-head/train-manifest.jsonl",
            ROOT / "benchmark/data/m4-head",
            True,
            frozenset(recipe["expectedTraining"]["singleViewSources"]),
        ),
        (
            "selector",
            ROOT / recipe["freshSelector"]["manifest"],
            ROOT / recipe["freshSelector"]["dataRoot"],
            False,
            frozenset(),
        ),
    ]
    for index in range(completed_regressions):
        spec = recipe["regressions"][index]
        partitions.append((
            "regression-m3" if index == 0 else "regression-m2",
            ROOT / spec["manifest"],
            ROOT / spec["dataRoot"],
            False,
            frozenset(),
        ))

    evidence = summary.get("featureShardEvidence")
    if not isinstance(evidence, list):
        raise ValueError("M4 feature-shard evidence is missing")
    recomputed: list[dict[str, Any]] = []
    for name, manifest, data_root, training, single_view_sources in partitions:
        items = train_adapter.safe_manifest_items(manifest, data_root)
        manifest_hash = digest(manifest)
        configuration_hash = feature_hashes["training" if training else "evaluation"]
        for shard_index, offset in enumerate(range(0, len(items), 2_000)):
            shard_items = items[offset : offset + 2_000]
            feature_pipeline.verify_item_files(shard_items)
            item_ids_hash = sha256("\n".join(item.id for item in shard_items).encode()).hexdigest()
            relative_cache = Path(
                f"benchmark/candidates/prooflens-cf384-m4/features/{name}-{shard_index:05d}.npz"
            )
            cache = ROOT / relative_cache
            if not cache.is_file() or cache.is_symlink():
                raise FileNotFoundError(cache)
            with np.load(cache, allow_pickle=False) as data:
                required = {
                    "features", "labels", "variants", "sources", "manifest_hash", "model_hash",
                    "item_ids_hash", "feature_configuration_hash", "fresh_feature_run_id", "training",
                    "pipeline_version",
                }
                if set(data.files) != required:
                    raise ValueError(f"M4 feature cache schema changed: {relative_cache}")
                if (
                    str(data["manifest_hash"].item()) != manifest_hash
                    or str(data["model_hash"].item()) != UPSTREAM_MODEL_SHA256
                    or str(data["item_ids_hash"].item()) != item_ids_hash
                    or str(data["feature_configuration_hash"].item()) != configuration_hash
                    or str(data["fresh_feature_run_id"].item()) != marker["runId"]
                    or bool(data["training"].item()) is not training
                    or int(data["pipeline_version"].item()) != feature_pipeline.PIPELINE_VERSION
                ):
                    raise ValueError(f"M4 feature cache binding changed: {relative_cache}")
                arrays = tuple(
                    np.asarray(data[key]).copy()
                    for key in ("features", "labels", "variants", "sources")
                )
            arrays = feature_pipeline.validate_feature_result(
                arrays,
                shard_items,
                training=training,
                single_view_sources=single_view_sources,
            )
            evidence_index = len(recomputed)
            if evidence_index >= len(evidence) or not isinstance(evidence[evidence_index], dict):
                raise ValueError(f"M4 feature cache evidence is missing: {relative_cache}")
            recorded = evidence[evidence_index]
            if type(recorded.get("freshlyExtractedThisProcess")) is not bool:
                raise ValueError(f"M4 feature cache process flag changed: {relative_cache}")
            expected = {
                "cache": relative_cache.as_posix(),
                "cacheSha256": digest(cache),
                "replacedCacheSha256": None,
                "freshFeatureRunId": marker["runId"],
                "freshlyExtractedThisRun": True,
                "freshlyExtractedThisProcess": recorded["freshlyExtractedThisProcess"],
                "items": len(shard_items),
                "views": int(arrays[0].shape[0]),
                "itemIdsSha256": item_ids_hash,
                "featureConfigurationSha256": configuration_hash,
                "arraySha256": {
                    "features": feature_pipeline.array_digest(arrays[0]),
                    "labels": feature_pipeline.array_digest(arrays[1]),
                    "variants": feature_pipeline.array_digest(arrays[2]),
                    "sources": feature_pipeline.array_digest(arrays[3]),
                },
            }
            recomputed.append(expected)
    if evidence != recomputed:
        raise ValueError("M4 fresh feature-shard evidence changed")
    return marker


def validate_regression(
    row: dict[str, Any],
    spec: dict[str, Any],
    expected_synthetic: set[str],
    expected_real: set[str],
    threshold: float,
) -> None:
    expected = recompute_regression_row(row, spec, expected_synthetic, expected_real, threshold)
    if not expected["passed"]:
        raise ValueError(f"M4 accepted packet failed terminal regression: {spec['name']}")


def validate_training_packet() -> dict[str, Any]:
    from benchmark.m4 import train_adapter

    recipe, _locks = load_frozen_protocol(RECIPE_PATH, LOCKS_PATH)
    sources = {
        "model": CANDIDATE_DIR / "model.onnx",
        "summary": CANDIDATE_DIR / "validation-summary.json",
        "calibration": CANDIDATE_DIR / "calibration.json",
        "grid": CANDIDATE_DIR / "candidate-grid.json",
        "selectionLock": CANDIDATE_DIR / "selection-lock.json",
        "tensorSeal": CANDIDATE_DIR / "candidate-tensor-seal.json",
        "freshMarker": CANDIDATE_DIR / "fresh-feature-run.json",
    }
    for path in sources.values():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    summary = load_canonical(sources["summary"], label="M4 training summary")
    calibration = load_canonical(sources["calibration"], label="M4 calibration")
    grid = load_canonical(sources["grid"], label="M4 candidate grid")
    selection_lock = load_canonical(sources["selectionLock"], label="M4 selection lock")
    tensor_seal = load_canonical(sources["tensorSeal"], label="M4 tensor seal")
    fresh_marker = load_fresh_feature_marker(sources["freshMarker"])
    selection_summary = load_canonical(SELECTION_SUMMARY_PATH, label="M4 source selection summary")

    expected_hashes = {
        "trainerSha256": digest(ROOT / "benchmark/m4/train_adapter.py"),
        "recipeSha256": digest(RECIPE_PATH),
        "sourceLocksSha256": digest(LOCKS_PATH),
        "selectionSummarySha256": digest(SELECTION_SUMMARY_PATH),
        "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
        "trainManifestSha256": selection_summary["publicArtifacts"]["train-manifest.jsonl"]["expandedSha256"],
        "selectorManifestSha256": digest(ROOT / recipe["freshSelector"]["manifest"]),
        "m3RegressionManifestSha256": recipe["regressions"][0]["sha256"],
        "m2RegressionManifestSha256": recipe["regressions"][1]["sha256"],
        "candidateTensorSealSha256": digest(sources["tensorSeal"]),
        "candidateGridSha256": digest(sources["grid"]),
        "selectionLockSha256": digest(sources["selectionLock"]),
        "calibrationSha256": digest(sources["calibration"]),
    }
    for key, expected in expected_hashes.items():
        if summary.get(key) != expected:
            raise ValueError(f"M4 training summary binding changed: {key}")
    if (
        summary.get("status") != "accepted-development-candidate"
        or summary.get("commandArguments") != expected_training_arguments()
        or summary.get("seed") != recipe["seed"]
        or summary.get("pipelineVersion") != 1
        or summary.get("h3HoldoutScored") is not False
        or summary.get("h3PixelsRead") is not False
        or summary.get("selectionInfluencedByRegression") is not False
        or summary.get("sourceBalancedLoss") is not True
        or summary.get("anchorLossProtectedSources") != recipe["adapter"]["protectedAnchorSources"]
        or summary.get("regressionOrder") != recipe["selectionPolicy"]["regressionOrder"]
    ):
        raise ValueError("M4 training policy evidence changed")
    expected_training_sources = dict(recipe["baseTraining"]["sourceCounts"])
    for name, count in recipe["expectedTraining"]["newSourceCounts"].items():
        expected_training_sources[name] = count
    if (
        summary.get("trainImages") != recipe["expectedTraining"]["images"]
        or summary.get("trainFeatureViews") != recipe["expectedTraining"]["featureViews"]
        or summary.get("trainClassCounts") != recipe["expectedTraining"]["classCounts"]
        or summary.get("trainSourceCounts") != dict(sorted(expected_training_sources.items()))
        or summary.get("selectorImages") != recipe["freshSelector"]["items"]
        or summary.get("selectorFeatureViews") != recipe["freshSelector"]["featureViews"]
        or summary.get("selectorClassCounts") != recipe["freshSelector"]["classCounts"]
        or summary.get("selectorSourceCounts") != recipe["freshSelector"]["sourceCounts"]
    ):
        raise ValueError("M4 item, view, class, or source counts changed")
    if validate_fresh_feature_evidence(
        summary,
        recipe,
        marker_state="complete",
        completed_regressions=2,
    ) != fresh_marker:
        raise ValueError("M4 successful fresh feature marker changed")
    if (
        summary.get("freshFeatureRunComplete") is not True
        or fresh_marker.get("state") != "complete"
        or summary.get("freshFeatureRunId") != fresh_marker.get("runId")
        or summary.get("freshFeatureMarkerSha256") != train_adapter.feature_pipeline.marker_sha256(fresh_marker)
        or not isinstance(summary.get("featureShardEvidence"), list)
        or len(summary["featureShardEvidence"]) != 60
    ):
        raise ValueError("M4 fresh feature-run evidence changed")
    cache_names: set[str] = set()
    for row in summary["featureShardEvidence"]:
        if (
            not isinstance(row, dict)
            or row.get("freshFeatureRunId") != summary["freshFeatureRunId"]
            or row.get("freshlyExtractedThisRun") is not True
            or row.get("replacedCacheSha256") is not None
            or not hex64(row.get("cacheSha256"))
            or not isinstance(row.get("arraySha256"), dict)
            or set(row["arraySha256"]) != {"features", "labels", "variants", "sources"}
            or not all(hex64(value) for value in row["arraySha256"].values())
        ):
            raise ValueError("M4 feature-shard evidence changed")
        cache = str(row.get("cache"))
        if cache in cache_names:
            raise ValueError("M4 feature-shard cache path repeated")
        cache_names.add(cache)
    environment = summary.get("environment")
    if not isinstance(environment, dict) or environment.get("providers") != ["CPUExecutionProvider"]:
        raise ValueError("M4 CPU-only environment evidence changed")

    if (
        tensor_seal.get("schemaVersion") != 1
        or tensor_seal.get("createdBeforeSelectorEvaluation") is not True
        or tensor_seal.get("candidateCount") != recipe["training"]["candidateCount"]
        or not isinstance(tensor_seal.get("candidates"), list)
    ):
        raise ValueError("M4 candidate tensor seal changed")
    sealed_by_id = {row.get("candidateId"): row for row in tensor_seal["candidates"]}
    if set(sealed_by_id) != set(EXPECTED_CANDIDATE_PAIRS) or len(tensor_seal["candidates"]) != len(sealed_by_id):
        raise ValueError("M4 candidate tensor seal IDs changed")
    expected_pairs = {(decay, anchor) for decay in ADAPTER_WEIGHT_DECAYS for anchor in ADAPTER_ANCHOR_COEFFICIENTS}
    observed_pairs: set[tuple[float, float]] = set()
    for candidate_id, seal in sealed_by_id.items():
        candidate_path = candidate_artifact_path(candidate_id)
        candidate = train_adapter.load_candidate(candidate_path)
        actual = train_adapter.candidate_seal(candidate)
        if actual != seal or digest(candidate_path) == "":
            raise ValueError(f"M4 sealed candidate changed: {candidate_id}")
        observed_pairs.add((candidate.weight_decay, candidate.anchor_coefficient))
    if observed_pairs != expected_pairs:
        raise ValueError("M4 candidate hyperparameter grid changed")

    selector_path = ROOT / recipe["freshSelector"]["manifest"]
    labels, variants, sources_array, item_count = load_manifest_metadata(selector_path)
    if item_count != recipe["freshSelector"]["items"]:
        raise ValueError("M4 selector item count changed")
    if (
        grid.get("schemaVersion") != 1
        or grid.get("candidateTensorSealSha256") != digest(sources["tensorSeal"])
        or grid.get("selectorManifestSha256") != digest(selector_path)
        or grid.get("candidateCount") != recipe["training"]["candidateCount"]
        or not isinstance(grid.get("candidates"), list)
        or len(grid["candidates"]) != recipe["training"]["candidateCount"]
    ):
        raise ValueError("M4 candidate grid header changed")
    require_grid_tensor_bindings(grid, tensor_seal)
    recomputed = [
        recompute_candidate_row(row, labels, variants, sources_array, recipe["validationGates"])
        for row in grid["candidates"]
    ]
    valid_rows = [row for row in recomputed if row["valid"]]
    if grid.get("validCandidateCount") != len(valid_rows) or not valid_rows:
        raise ValueError("M4 candidate grid valid count changed")
    selected = max(valid_rows, key=lambda row: (tuple(row["candidateSelectionKey"]), row["candidateId"]))
    if summary.get("selectedCandidate") != selected:
        raise ValueError("M4 selected candidate is not the deterministic selector winner")
    expected_lock = {
        "schemaVersion": 1,
        "candidateTensorSealSha256": digest(sources["tensorSeal"]),
        "candidateGridSha256": digest(sources["grid"]),
        "selectedCandidateId": selected["candidateId"],
        "selectedTensorSha256": selected["tensorSha256"],
        "rawThreshold": selected["rawThreshold"],
        "selectorMetrics": selected["selectorMetrics"],
        "candidateSelectionKey": selected["candidateSelectionKey"],
        "selectorManifestSha256": digest(selector_path),
        "createdBeforeRegressionEvaluation": True,
        "selectionInfluencedByRegression": False,
        "h3HoldoutScored": False,
    }
    if selection_lock != expected_lock or summary.get("selectionLock") != expected_lock:
        raise ValueError("M4 pre-regression selection lock changed")

    regressions = summary.get("regressions")
    if not isinstance(regressions, list) or [row.get("name") for row in regressions] != recipe["selectionPolicy"]["regressionOrder"]:
        raise ValueError("M4 terminal regression order changed")
    validate_regression(
        regressions[0], recipe["regressions"][0], {"flux-1-dev-development"}, {"met-open-access"},
        float(selected["rawThreshold"]),
    )
    validate_regression(
        regressions[1], recipe["regressions"][1], {"GLM-Image", "HunyuanImage-3.0"},
        {"open-images", "stockimages-cc0"}, float(selected["rawThreshold"]),
    )

    model_hash = digest(sources["model"])
    if summary.get("modelSha256") != model_hash or summary.get("modelBytes") != sources["model"].stat().st_size:
        raise ValueError("M4 candidate model binding changed")
    if (
        not finite(summary.get("zeroAdapterFeatureParityMaximumAbsoluteError"))
        or float(summary["zeroAdapterFeatureParityMaximumAbsoluteError"]) != 0.0
        or not finite(summary.get("zeroAdapterImageParityMaximumAbsoluteError"))
        or float(summary["zeroAdapterImageParityMaximumAbsoluteError"]) > recipe["runtimeParity"]["zeroAdapterMaximumAbsoluteLogitError"]
        or not finite(summary.get("exportedCandidateImageParityMaximumAbsoluteError"))
        or float(summary["exportedCandidateImageParityMaximumAbsoluteError"]) > recipe["runtimeParity"]["exportedCandidateMaximumAbsoluteLogitError"]
    ):
        raise ValueError("M4 runtime parity evidence changed")
    raw_threshold = float(selected["rawThreshold"])
    intercept = math.log(0.65 / 0.35) - raw_threshold
    expected_calibration = {
        "schemaVersion": 1,
        "mode": "threshold-alignment-not-probability-calibration",
        "slope": 1.0,
        "intercept": intercept,
        "rawThreshold": raw_threshold,
        "displayThreshold": 0.65,
        "rawProbabilityAtThreshold": sigmoid(raw_threshold),
        "displayProbabilityAtRawThreshold": sigmoid(raw_threshold + intercept),
        "modelSha256": model_hash,
        "selectionLockSha256": digest(sources["selectionLock"]),
        "selectorManifestSha256": digest(selector_path),
    }
    if calibration != expected_calibration:
        raise ValueError("M4 calibration changed")
    return {
        "recipe": recipe,
        "summary": summary,
        "calibration": calibration,
        "grid": grid,
        "selectionLock": selection_lock,
        "tensorSeal": tensor_seal,
        "modelSha256": model_hash,
        "modelBytes": sources["model"].stat().st_size,
        "candidateHashes": {name: digest(path) for name, path in sources.items()},
    }


def validate_failed_training_packet(
    summary: dict[str, Any],
    grid: dict[str, Any],
    tensor_seal: dict[str, Any],
    selection_lock: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recompute every completed stage of a terminal M4 development failure."""
    from benchmark.m4 import train_adapter

    recipe, _locks = load_frozen_protocol(RECIPE_PATH, LOCKS_PATH)
    selection_summary = load_canonical(SELECTION_SUMMARY_PATH, label="M4 source selection summary")
    expected_common_keys = {
        "schemaVersion", "pipelineVersion", "status", "seed", "commandArguments", "trainerSha256",
        "recipeSha256", "sourceLocksSha256", "selectionSummarySha256", "upstreamModelSha256",
        "trainManifestSha256", "selectorManifestSha256", "m3RegressionManifestSha256",
        "m2RegressionManifestSha256", "trainImages", "trainFeatureViews", "trainSourceCounts",
        "trainClassCounts", "selectorImages", "selectorFeatureViews", "selectorSourceCounts",
        "selectorClassCounts", "candidateTensorSealSha256", "candidateGridSha256",
        "zeroAdapterFeatureParityMaximumAbsoluteError", "featureConfigurationHashes",
        "featureShardEvidence", "freshFeatureRunId", "freshFeatureMarkerSha256", "sourceBalancedLoss",
        "anchorLossProtectedSources", "candidateCount", "validCandidateCount", "regressionOrder",
        "h3HoldoutScored", "h3PixelsRead", "selectionInfluencedByRegression", "environment",
    }
    status = summary.get("status")
    statuses = {
        "failed-selector": 0,
        "failed-m3-selector-regression": 1,
        "failed-m2-development-regression": 2,
    }
    if status not in statuses:
        raise ValueError("M4 failure status is not terminal")
    expected_keys = expected_common_keys | (
        set() if status == "failed-selector" else {
            "selectionLockSha256", "selectionLock", "selectedCandidate", "regressions",
        }
    )
    if set(summary) != expected_keys:
        raise ValueError("M4 failed summary schema changed")
    expected_hashes = {
        "trainerSha256": digest(ROOT / "benchmark/m4/train_adapter.py"),
        "recipeSha256": digest(RECIPE_PATH),
        "sourceLocksSha256": digest(LOCKS_PATH),
        "selectionSummarySha256": digest(SELECTION_SUMMARY_PATH),
        "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
        "trainManifestSha256": selection_summary["publicArtifacts"]["train-manifest.jsonl"]["expandedSha256"],
        "selectorManifestSha256": digest(ROOT / recipe["freshSelector"]["manifest"]),
        "m3RegressionManifestSha256": recipe["regressions"][0]["sha256"],
        "m2RegressionManifestSha256": recipe["regressions"][1]["sha256"],
        "candidateTensorSealSha256": digest(CANDIDATE_DIR / "candidate-tensor-seal.json"),
        "candidateGridSha256": digest(CANDIDATE_DIR / "candidate-grid.json"),
    }
    if any(summary.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("M4 failed summary input binding changed")
    expected_sources = dict(recipe["baseTraining"]["sourceCounts"])
    expected_sources.update(recipe["expectedTraining"]["newSourceCounts"])
    if (
        summary.get("schemaVersion") != 1
        or summary.get("pipelineVersion") != 1
        or summary.get("seed") != recipe["seed"]
        or summary.get("commandArguments") != expected_training_arguments()
        or summary.get("trainImages") != recipe["expectedTraining"]["images"]
        or summary.get("trainFeatureViews") != recipe["expectedTraining"]["featureViews"]
        or summary.get("trainSourceCounts") != dict(sorted(expected_sources.items()))
        or summary.get("trainClassCounts") != recipe["expectedTraining"]["classCounts"]
        or summary.get("selectorImages") != recipe["freshSelector"]["items"]
        or summary.get("selectorFeatureViews") != recipe["freshSelector"]["featureViews"]
        or summary.get("selectorSourceCounts") != recipe["freshSelector"]["sourceCounts"]
        or summary.get("selectorClassCounts") != recipe["freshSelector"]["classCounts"]
        or summary.get("sourceBalancedLoss") is not True
        or summary.get("anchorLossProtectedSources") != recipe["adapter"]["protectedAnchorSources"]
        or summary.get("regressionOrder") != recipe["selectionPolicy"]["regressionOrder"]
        or summary.get("h3HoldoutScored") is not False
        or summary.get("h3PixelsRead") is not False
        or summary.get("selectionInfluencedByRegression") is not False
        or summary.get("zeroAdapterFeatureParityMaximumAbsoluteError") != 0.0
    ):
        raise ValueError("M4 failed summary policy or composition changed")
    environment = summary.get("environment")
    if not isinstance(environment, dict) or environment.get("providers") != ["CPUExecutionProvider"]:
        raise ValueError("M4 failed summary CPU environment changed")
    feature_hashes = summary.get("featureConfigurationHashes")
    if (
        not isinstance(feature_hashes, dict)
        or set(feature_hashes) != {"training", "evaluation"}
        or not all(hex64(value) for value in feature_hashes.values())
        or feature_hashes["training"] == feature_hashes["evaluation"]
    ):
        raise ValueError("M4 failed feature configuration changed")
    marker_path = CANDIDATE_DIR / "fresh-feature-run.json"
    marker = validate_fresh_feature_evidence(
        summary,
        recipe,
        marker_state="extracting",
        completed_regressions=statuses[status],
    )
    if (
        marker.get("state") != "extracting"
        or summary.get("freshFeatureRunId") != marker.get("runId")
        or summary.get("freshFeatureMarkerSha256") != train_adapter.feature_pipeline.marker_sha256(marker)
    ):
        raise ValueError("M4 failed fresh-run marker changed")
    completed = statuses[status]
    evidence = summary.get("featureShardEvidence")
    expected_shards = 58 + completed
    if not isinstance(evidence, list) or len(evidence) != expected_shards:
        raise ValueError("M4 failed feature-shard count changed")
    caches: set[str] = set()
    for row in evidence:
        if (
            not isinstance(row, dict)
            or row.get("freshFeatureRunId") != summary["freshFeatureRunId"]
            or row.get("freshlyExtractedThisRun") is not True
            or row.get("replacedCacheSha256") is not None
            or not hex64(row.get("cacheSha256"))
            or not isinstance(row.get("arraySha256"), dict)
            or set(row["arraySha256"]) != {"features", "labels", "variants", "sources"}
            or not all(hex64(value) for value in row["arraySha256"].values())
            or not isinstance(row.get("cache"), str)
            or row["cache"] in caches
        ):
            raise ValueError("M4 failed feature-shard evidence changed")
        caches.add(row["cache"])

    if (
        tensor_seal.get("schemaVersion") != 1
        or tensor_seal.get("createdBeforeSelectorEvaluation") is not True
        or tensor_seal.get("candidateCount") != recipe["training"]["candidateCount"]
        or not isinstance(tensor_seal.get("candidates"), list)
    ):
        raise ValueError("M4 failed candidate tensor seal changed")
    sealed = {row.get("candidateId"): row for row in tensor_seal["candidates"]}
    if set(sealed) != set(EXPECTED_CANDIDATE_PAIRS) or len(tensor_seal["candidates"]) != len(sealed):
        raise ValueError("M4 failed candidate tensor IDs changed")
    observed_pairs: set[tuple[float, float]] = set()
    for candidate_id, expected in sealed.items():
        candidate_path = candidate_artifact_path(candidate_id)
        candidate = train_adapter.load_candidate(candidate_path)
        if train_adapter.candidate_seal(candidate) != expected:
            raise ValueError(f"M4 failed candidate tensors changed: {candidate_id}")
        observed_pairs.add((candidate.weight_decay, candidate.anchor_coefficient))
    if observed_pairs != {
        (decay, anchor) for decay in ADAPTER_WEIGHT_DECAYS for anchor in ADAPTER_ANCHOR_COEFFICIENTS
    }:
        raise ValueError("M4 failed candidate grid changed")

    selector_path = ROOT / recipe["freshSelector"]["manifest"]
    labels, variants, sources_array, selector_items = load_manifest_metadata(selector_path)
    if selector_items != recipe["freshSelector"]["items"]:
        raise ValueError("M4 failed selector manifest count changed")
    if (
        grid.get("schemaVersion") != 1
        or grid.get("candidateTensorSealSha256") != summary["candidateTensorSealSha256"]
        or grid.get("selectorManifestSha256") != summary["selectorManifestSha256"]
        or grid.get("candidateCount") != recipe["training"]["candidateCount"]
        or not isinstance(grid.get("candidates"), list)
        or len(grid["candidates"]) != recipe["training"]["candidateCount"]
    ):
        raise ValueError("M4 failed selector-grid header changed")
    require_grid_tensor_bindings(grid, tensor_seal)
    recomputed = [
        recompute_candidate_row(row, labels, variants, sources_array, recipe["validationGates"])
        for row in grid["candidates"]
    ]
    valid = [row for row in recomputed if row["valid"]]
    if grid.get("validCandidateCount") != len(valid) or summary.get("validCandidateCount") != len(valid):
        raise ValueError("M4 failed valid-candidate count changed")
    if summary.get("candidateCount") != recipe["training"]["candidateCount"]:
        raise ValueError("M4 failed candidate count changed")
    if status == "failed-selector":
        if valid or selection_lock is not None:
            raise ValueError("M4 selector failure contains a feasible winner")
        return {"summary": summary, "grid": grid, "tensorSeal": tensor_seal, "selectionLock": None}

    if not valid or selection_lock is None:
        raise ValueError("M4 regression failure lacks a selector winner")
    selected = max(valid, key=lambda row: (tuple(row["candidateSelectionKey"]), row["candidateId"]))
    expected_lock = {
        "schemaVersion": 1,
        "candidateTensorSealSha256": summary["candidateTensorSealSha256"],
        "candidateGridSha256": summary["candidateGridSha256"],
        "selectedCandidateId": selected["candidateId"],
        "selectedTensorSha256": selected["tensorSha256"],
        "rawThreshold": selected["rawThreshold"],
        "selectorMetrics": selected["selectorMetrics"],
        "candidateSelectionKey": selected["candidateSelectionKey"],
        "selectorManifestSha256": summary["selectorManifestSha256"],
        "createdBeforeRegressionEvaluation": True,
        "selectionInfluencedByRegression": False,
        "h3HoldoutScored": False,
    }
    if (
        selection_lock != expected_lock
        or summary.get("selectionLock") != expected_lock
        or summary.get("selectionLockSha256") != digest(CANDIDATE_DIR / "selection-lock.json")
        or summary.get("selectedCandidate") != selected
    ):
        raise ValueError("M4 failed pre-regression selection lock changed")
    regressions = summary.get("regressions")
    if not isinstance(regressions, list) or len(regressions) != completed:
        raise ValueError("M4 failed completed-regression count changed")
    specs = (
        (recipe["regressions"][0], {"flux-1-dev-development"}, {"met-open-access"}),
        (recipe["regressions"][1], {"GLM-Image", "HunyuanImage-3.0"}, {"open-images", "stockimages-cc0"}),
    )
    for index, row in enumerate(regressions):
        expected = recompute_regression_row(
            row, specs[index][0], specs[index][1], specs[index][2], float(selected["rawThreshold"]),
        )
        if index < len(regressions) - 1 and not expected["passed"]:
            raise ValueError("M4 failure continued after an earlier regression failed")
    if regressions[-1]["passed"] is not False:
        raise ValueError("M4 terminal regression failure is not recomputed false")
    return {
        "summary": summary,
        "grid": grid,
        "tensorSeal": tensor_seal,
        "selectionLock": selection_lock,
    }


def _proto_digest(value: Any) -> str:
    return digest_bytes(value.SerializeToString(deterministic=True))


def _attribute_map(node: Any) -> dict[str, object]:
    import onnx

    output: dict[str, object] = {}
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.FLOAT:
            output[attribute.name] = float(attribute.f)
        elif attribute.type == onnx.AttributeProto.INT:
            output[attribute.name] = int(attribute.i)
        else:
            raise ValueError(f"M4 node has an unexpected attribute type: {node.name}/{attribute.name}")
    return output


def compare_adapter_models(base_path: Path, candidate_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import onnx

    base_bytes = base_path.read_bytes()
    candidate_bytes = candidate_path.read_bytes()
    if digest_bytes(base_bytes) != UPSTREAM_MODEL_SHA256 or len(base_bytes) != UPSTREAM_MODEL_BYTES:
        raise ValueError("M4 base model bytes changed")
    base = onnx.load_model_from_string(base_bytes)
    candidate = onnx.load_model_from_string(candidate_bytes)
    onnx.checker.check_model(candidate)
    if (
        [_proto_digest(value) for value in base.graph.input] != [_proto_digest(value) for value in candidate.graph.input]
        or [_proto_digest(value) for value in base.graph.output] != [_proto_digest(value) for value in candidate.graph.output]
        or [_proto_digest(value) for value in base.opset_import] != [_proto_digest(value) for value in candidate.opset_import]
    ):
        raise ValueError("M4 model interface or opset changed")
    base_initializers = {value.name: value for value in base.graph.initializer}
    candidate_initializers = {value.name: value for value in candidate.graph.initializer}
    if len(base_initializers) != len(base.graph.initializer) or len(candidate_initializers) != len(candidate.graph.initializer):
        raise ValueError("M4 model contains duplicate initializer names")
    if set(candidate_initializers) != set(base_initializers) | set(ADDED_INITIALIZERS):
        raise ValueError("M4 initializer set changed")
    for name, value in base_initializers.items():
        if _proto_digest(value) != _proto_digest(candidate_initializers[name]):
            raise ValueError(f"M4 changed a frozen base initializer: {name}")
    added_initializer_rows = []
    for name in ADDED_INITIALIZERS:
        value = candidate_initializers[name]
        raw = bytes(value.raw_data)
        if list(value.dims) != ADDED_INITIALIZER_SHAPES[name] or value.data_type != onnx.TensorProto.FLOAT or not raw:
            raise ValueError(f"M4 added initializer contract changed: {name}")
        added_initializer_rows.append({
            "name": name,
            "dimensions": list(value.dims),
            "dataType": "FLOAT",
            "tensorProtoSha256": _proto_digest(value),
            "tensorProtoBase64": b64encode(value.SerializeToString(deterministic=True)).decode("ascii"),
            "rawDataSha256": digest_bytes(raw),
            "rawDataBase64": b64encode(raw).decode("ascii"),
        })
    base_nodes = list(base.graph.node)
    candidate_nodes = list(candidate.graph.node)
    base_classifier_index = next((index for index, node in enumerate(base_nodes) if node.name == "/classifier/Gemm"), None)
    candidate_classifier_index = next((index for index, node in enumerate(candidate_nodes) if node.name == "/classifier/Gemm"), None)
    if base_classifier_index is None or candidate_classifier_index is None or candidate_classifier_index != base_classifier_index + len(ADDED_NODES):
        raise ValueError("M4 classifier placement changed")
    if len(candidate_nodes) != len(base_nodes) + len(ADDED_NODES):
        raise ValueError("M4 graph node count changed")
    for index, base_node in enumerate(base_nodes):
        candidate_index = index if index < base_classifier_index else index + len(ADDED_NODES)
        candidate_node = candidate_nodes[candidate_index]
        expected = onnx.NodeProto()
        expected.CopyFrom(base_node)
        if base_node.name == "/classifier/Gemm":
            if list(base_node.input) != ["/Gather_output_0", "classifier.weight", "classifier.bias"]:
                raise ValueError("M4 base classifier input changed")
            expected.input[0] = "m4.adapted"
        if _proto_digest(expected) != _proto_digest(candidate_node):
            raise ValueError(f"M4 changed a frozen graph node: {base_node.name}")
    added_node_rows = []
    for offset, expected in enumerate(ADDED_NODES):
        node = candidate_nodes[base_classifier_index + offset]
        name, op_type, inputs, outputs = expected
        expected_attributes = {"alpha": 1.0, "beta": 1.0, "transB": 1} if op_type == "Gemm" else {}
        if (
            node.name != name or node.op_type != op_type or list(node.input) != inputs
            or list(node.output) != outputs or _attribute_map(node) != expected_attributes
        ):
            raise ValueError(f"M4 added node contract changed: {name}")
        added_node_rows.append({
            "name": name,
            "opType": op_type,
            "inputs": inputs,
            "outputs": outputs,
            "attributes": expected_attributes,
            "nodeProtoSha256": _proto_digest(node),
            "nodeProtoBase64": b64encode(node.SerializeToString(deterministic=True)).decode("ascii"),
        })

    reconstructed = onnx.ModelProto()
    reconstructed.CopyFrom(base)
    for name in ADDED_INITIALIZERS:
        value = reconstructed.graph.initializer.add()
        value.CopyFrom(candidate_initializers[name])
    classifier = next(node for node in reconstructed.graph.node if node.name == "/classifier/Gemm")
    insertion_index = list(reconstructed.graph.node).index(classifier)
    for offset in range(len(ADDED_NODES)):
        value = reconstructed.graph.node.add()
        value.CopyFrom(candidate_nodes[base_classifier_index + offset])
        reconstructed.graph.node.insert(insertion_index + offset, value)
        del reconstructed.graph.node[-1]
    classifier = next(node for node in reconstructed.graph.node if node.name == "/classifier/Gemm")
    classifier.input[0] = "m4.adapted"
    reconstructed_bytes = reconstructed.SerializeToString(deterministic=True)
    if reconstructed_bytes != candidate_bytes:
        raise ValueError("M4 adapter patch does not byte-reconstruct the candidate model")
    candidate_hash = digest_bytes(candidate_bytes)
    candidate_classifier = next(node for node in candidate.graph.node if node.name == "/classifier/Gemm")
    candidate_classifier_bytes = candidate_classifier.SerializeToString(deterministic=True)
    comparison = {
        "schemaVersion": 1,
        "profile": PROFILE,
        "base": {"path": "weights/prooflens-cf384.onnx", "sha256": UPSTREAM_MODEL_SHA256, "bytes": len(base_bytes)},
        "candidate": {
            "path": "benchmark/candidates/prooflens-cf384-m4/model.onnx",
            "sha256": candidate_hash,
            "bytes": len(candidate_bytes),
        },
        "unchangedBaseInitializerCount": len(base_initializers),
        "unchangedBaseNodeCount": len(base_nodes),
        "addedInitializers": added_initializer_rows,
        "addedNodes": added_node_rows,
        "classifierInputBefore": "/Gather_output_0",
        "classifierInputAfter": "m4.adapted",
        "classifierNodeProtoSha256": digest_bytes(candidate_classifier_bytes),
        "classifierNodeProtoBase64": b64encode(candidate_classifier_bytes).decode("ascii"),
        "reconstructedCandidateSha256": candidate_hash,
        "reconstructedCandidateBytes": len(reconstructed_bytes),
        "backboneAndClassifierInitializersByteIdentical": True,
    }
    patch = {
        "schemaVersion": 1,
        "baseSha256": UPSTREAM_MODEL_SHA256,
        "candidateSha256": candidate_hash,
        "candidateBytes": len(candidate_bytes),
        "featureTensor": "/Gather_output_0",
        "classifierNodeName": "/classifier/Gemm",
        "classifierInputBefore": "/Gather_output_0",
        "classifierInputAfter": "m4.adapted",
        "addedInitializers": added_initializer_rows,
        "addedNodes": added_node_rows,
        "classifierNodeProtoSha256": digest_bytes(candidate_classifier_bytes),
        "classifierNodeProtoBase64": b64encode(candidate_classifier_bytes).decode("ascii"),
        "reconstructedCandidateSha256": candidate_hash,
    }
    return comparison, patch


def build_publication_lock(
    *,
    source_commit: str,
    source_tree: str,
    packet: dict[str, Any],
    comparison_sha256: str,
    adapter_patch: dict[str, Any],
    candidate_evidence_json: dict[str, str],
    public_document_hashes: dict[str, str],
    fixture_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "profile": PROFILE,
        "sourceCommit": source_commit,
        "sourceTree": source_tree,
        "baseCommit": BASE_COMMIT,
        "baseTree": BASE_TREE,
        "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
        "upstreamModelBytes": UPSTREAM_MODEL_BYTES,
        "upstreamModelLockSha256": digest(ROOT / "model-lock.json"),
        "trainerSha256": digest(ROOT / "benchmark/m4/train_adapter.py"),
        "recipeSha256": digest(RECIPE_PATH),
        "sourceLocksSha256": digest(LOCKS_PATH),
        "selectionSummarySha256": digest(SELECTION_SUMMARY_PATH),
        "candidateHashes": packet["candidateHashes"],
        "candidateModelBytes": packet["modelBytes"],
        "selectionLock": packet["selectionLock"],
        "modelComparisonSha256": comparison_sha256,
        "adapterPatch": adapter_patch,
        "finalizerSha256": digest(ROOT / "benchmark/m4/finalize.py"),
        "publicationContractSha256": digest(Path(__file__)),
        "fixtureSelectorSha256": digest(ROOT / "benchmark/m4/select_model_state_fixtures.py"),
        "documentationRendererSha256": digest(ROOT / "scripts/render-m4-public-docs.mjs"),
        "publicDocumentHashes": public_document_hashes,
        "fixtureManifestSha256": fixture_manifest_sha256,
        "candidateEvidenceJson": candidate_evidence_json,
        "publicationRows": [{"path": path, "status": status} for path, status in PUBLICATION_ROWS],
        "selectionInfluencedByRegression": False,
        "h3HoldoutScored": False,
        "h3PixelsRead": False,
    }


def validate_publication_lock(lock: dict[str, Any], **arguments: Any) -> None:
    expected = build_publication_lock(**arguments)
    if lock != expected:
        raise ValueError("M4 publication lock does not exactly match the frozen candidate packet")
