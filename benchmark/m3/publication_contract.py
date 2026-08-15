"""Pure validation contract for the frozen M3 candidate and publication lock."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_DIR = ROOT / "benchmark/candidates/prooflens-cf384-m3"
RECIPE_PATH = ROOT / "benchmark/m3/recipe.json"
SELECTION_SUMMARY_PATH = ROOT / "benchmark/evidence/m3/selection-summary.json"
TRAIN_MANIFEST_PATH = ROOT / "benchmark/data/m3-head/train-manifest.jsonl"
VALIDATION_MANIFEST_PATH = ROOT / "benchmark/evidence/m3/validation-manifest.jsonl"
REGRESSION_MANIFEST_PATH = ROOT / "benchmark/evidence/m2/validation-manifest.jsonl"
UPSTREAM_MODEL_PATH = ROOT / "weights/prooflens-cf384.onnx"
UPSTREAM_MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47"
PUBLICATION_LOCK_PATH = ROOT / "benchmark/evidence/m3/publication-lock.json"
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")
SEED = 20260813
PUBLICATION_ROWS = (
    ("BENCHMARK.md", "M"),
    ("MODEL_CARD.md", "M"),
    ("README.md", "M"),
    ("benchmark/evidence/m3/calibration.json", "A"),
    ("benchmark/evidence/m3/candidate-grid.json", "A"),
    ("benchmark/evidence/m3/finalization-receipt.json", "A"),
    ("benchmark/evidence/m3/model-comparison.json", "A"),
    ("benchmark/evidence/m3/training-summary.json", "A"),
    ("model-lock.json", "M"),
    ("tests/fixtures/model-states/fixture-manifest.json", "M"),
    ("weights/README.md", "M"),
    ("weights/prooflens-cf384.onnx", "M"),
)
TRANSACTIONAL_ROWS = PUBLICATION_ROWS


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def is_hex64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"M3 publication lock contains a duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_canonical_publication_lock(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            value,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"M3 publication lock contains non-finite JSON: {constant}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError("M3 publication lock is not canonical JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("M3 publication lock must be a JSON object")
    canonical = json.dumps(parsed, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if value != canonical:
        raise ValueError("M3 publication lock bytes are not canonical JSON")
    return parsed


def expected_training_arguments() -> list[str]:
    return [
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
    ]


def require_variant_gates(
    variants: object,
    gates: dict[str, Any],
    *,
    label: str,
    expected_synthetic_sources: set[str],
    expected_real_sources: set[str],
) -> None:
    if not isinstance(variants, dict) or set(variants) != set(VARIANTS):
        raise ValueError(f"{label} variant set changed")
    for variant in VARIANTS:
        metrics = variants[variant]
        if not isinstance(metrics, dict):
            raise ValueError(f"{label} {variant} metrics are malformed")
        for metric, gate in (
            ("balancedAccuracy", "minimumBalancedAccuracyPerVariant"),
            ("realRecall", "minimumRealRecallPerVariant"),
            ("syntheticRecall", "minimumSyntheticRecallPerVariant"),
        ):
            value = metrics.get(metric)
            minimum = gates.get(gate)
            if not finite_number(value) or not finite_number(minimum) or value < minimum:
                raise ValueError(f"{label} {variant} failed {gate}")
        family_metrics = metrics.get("syntheticRecallBySource")
        family_minimum = gates.get("minimumSyntheticRecallPerFamily")
        if (
            not isinstance(family_metrics, dict)
            or not family_metrics
            or set(family_metrics) != expected_synthetic_sources
            or not finite_number(family_minimum)
            or any(not finite_number(value) or value < family_minimum for value in family_metrics.values())
        ):
            raise ValueError(f"{label} {variant} failed the synthetic-family gate")
        real_metrics = metrics.get("realRecallBySource")
        required_real = gates.get("minimumRealRecallBySource", {})
        if not isinstance(real_metrics, dict) or set(real_metrics) != expected_real_sources:
            raise ValueError(f"{label} {variant} lacks real-source metrics")
        for source, minimum in required_real.items():
            value = real_metrics.get(source)
            if not finite_number(value) or not finite_number(minimum) or value < minimum:
                raise ValueError(f"{label} {variant} failed the {source} real-recall gate")


def expected_cache_paths() -> list[str]:
    return [
        *(f"benchmark/candidates/prooflens-cf384-m3/features/train-{index:05d}.npz" for index in range(55)),
        "benchmark/candidates/prooflens-cf384-m3/features/validation-00000.npz",
        "benchmark/candidates/prooflens-cf384-m3/features/regression-00000.npz",
    ]


def validate_training_packet() -> dict[str, Any]:
    sources = {
        "training-summary.json": CANDIDATE_DIR / "validation-summary.json",
        "calibration.json": CANDIDATE_DIR / "calibration.json",
        "candidate-grid.json": CANDIDATE_DIR / "candidate-grid.json",
        "model.onnx": CANDIDATE_DIR / "model.onnx",
    }
    for path in [
        *sources.values(), RECIPE_PATH, SELECTION_SUMMARY_PATH, TRAIN_MANIFEST_PATH,
        VALIDATION_MANIFEST_PATH, REGRESSION_MANIFEST_PATH, UPSTREAM_MODEL_PATH,
        CANDIDATE_DIR / "fresh-feature-run.json",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if digest(UPSTREAM_MODEL_PATH) != UPSTREAM_MODEL_SHA256:
        raise ValueError("M3 upstream M2 model changed")
    summary = json.loads(sources["training-summary.json"].read_text())
    calibration = json.loads(sources["calibration.json"].read_text())
    grid = json.loads(sources["candidate-grid.json"].read_text())
    recipe = json.loads(RECIPE_PATH.read_text())
    selection = json.loads(SELECTION_SUMMARY_PATH.read_text())
    model_hash = digest(sources["model.onnx"])
    trainer_path = ROOT / "benchmark/modern/train_rehead.py"
    if (
        summary.get("schemaVersion") != 2
        or summary.get("pipelineVersion") != 9
        or summary.get("seed") != SEED
        or summary.get("upstreamModelSha256") != UPSTREAM_MODEL_SHA256
    ):
        raise ValueError("M3 training schema or pipeline version changed")
    if summary.get("trainerSha256") != digest(trainer_path):
        raise ValueError("M3 summary does not bind the current trainer")
    if summary.get("commandArguments") != expected_training_arguments():
        raise ValueError("M3 training command differs from the frozen CPU command")
    if (
        summary.get("recipeSha256") != digest(RECIPE_PATH)
        or summary.get("selectionSummarySha256") != digest(SELECTION_SUMMARY_PATH)
        or summary.get("trainManifestSha256") != selection.get("manifestSha256")
        or summary.get("trainManifestSha256") != digest(TRAIN_MANIFEST_PATH)
        or summary.get("validationManifestSha256") != digest(VALIDATION_MANIFEST_PATH)
        or summary.get("regressionManifestSha256") != digest(REGRESSION_MANIFEST_PATH)
    ):
        raise ValueError("M3 training inputs are not bound to the source packet")
    if (
        summary.get("trainImages") != 108_378
        or summary.get("trainFeatureViews") != 133_512
        or summary.get("validationImages") != 600
        or summary.get("validationFeatureViews") != 2_400
        or summary.get("regressionImages") != 900
        or summary.get("regressionFeatureViews") != 3_600
        or summary.get("uniqueTrainingImagesCovered") != 108_378
        or summary.get("uniqueTrainingFeatureViewsCovered") != 133_512
    ):
        raise ValueError("M3 image or feature-view coverage changed")
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
        or summary.get("cachedFeatureDtypes") != {
            "features": "float32",
            "labels": "float32",
            "variants": "int64",
            "sources": "unicode",
        }
        or summary.get("singleViewTrainingSources") != ["diffusiondb-stable-diffusion", "open-images-train"]
        or summary.get("trainingSourceCounts") != recipe.get("expectedSourceCounts")
    ):
        raise ValueError("M3 training procedure changed")

    feature_hashes = summary.get("featureConfigurationHashes")
    fresh_run = summary.get("freshFeatureRun")
    marker_path = CANDIDATE_DIR / "fresh-feature-run.json"
    shard_rows = summary.get("featureShardEvidence")
    if (
        not isinstance(feature_hashes, dict)
        or set(feature_hashes) != {"training", "validation", "regression"}
        or not all(is_hex64(value) for value in feature_hashes.values())
        or feature_hashes["training"] == feature_hashes["validation"]
        or feature_hashes["validation"] != feature_hashes["regression"]
        or not isinstance(fresh_run, dict)
        or fresh_run.get("state") != "complete"
        or not isinstance(shard_rows, list)
        or len(shard_rows) != 57
    ):
        raise ValueError("M3 fresh-feature configuration changed")
    expected_context = {
        "pipelineVersion": 9,
        "featureExtractorContract": "cf384-static-batch24-preprocess-v1",
        "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
        "trainManifestSha256": summary["trainManifestSha256"],
        "validationManifestSha256": summary["validationManifestSha256"],
        "regressionManifestSha256": summary["regressionManifestSha256"],
        "regressionDataRoot": "benchmark/data/m2-head",
        "regressionFeatureViews": 3_600,
        "selectionSummarySha256": summary["selectionSummarySha256"],
        "featureBatchSize": 24,
        "featureShardImages": 2_000,
        "singleViewTrainingSources": ["diffusiondb-stable-diffusion", "open-images-train"],
        "featureConfigurationHashes": feature_hashes,
    }
    if (
        fresh_run.get("context") != expected_context
        or digest(marker_path) != summary.get("freshFeatureRunMarkerSha256")
        or json.loads(marker_path.read_text()) != fresh_run
    ):
        raise ValueError("M3 fresh-run marker changed")
    if [row.get("cache") for row in shard_rows] != expected_cache_paths():
        raise ValueError("M3 feature-shard paths changed")
    if sum(row.get("items", 0) for row in shard_rows) != 109_878 or sum(row.get("views", 0) for row in shard_rows) != 139_512:
        raise ValueError("M3 feature shards do not cover all three partitions")
    run_id = fresh_run.get("runId")
    cache_hashes: set[str] = set()
    item_hashes: set[str] = set()
    for row in shard_rows:
        cache = str(row.get("cache", ""))
        partition = "training" if "/train-" in cache else ("regression" if "/regression-" in cache else "validation")
        if (
            row.get("freshFeatureRunId") != run_id
            or row.get("freshlyExtractedThisRun") is not True
            or row.get("replacedCacheSha256") is not None
            or row.get("featureConfigurationSha256") != feature_hashes[partition]
            or not is_hex64(row.get("cacheSha256"))
            or not is_hex64(row.get("itemIdsSha256"))
            or row.get("freshlyExtractedThisProcess") is not True
            or not isinstance(row.get("arraySha256"), dict)
            or not all(is_hex64(row["arraySha256"].get(name)) for name in ("features", "labels", "variants", "sources"))
        ):
            raise ValueError("An M3 feature shard is not bound to the fresh run")
        cache_hashes.add(row["cacheSha256"])
        item_hashes.add(row["itemIdsSha256"])
    if len(cache_hashes) != len(shard_rows) or len(item_hashes) != len(shard_rows):
        raise ValueError("M3 feature-shard digests are duplicated")

    environment = summary.get("environment", {})
    if (
        environment.get("executionProvider") != "cpu"
        or environment.get("providers") != ["CPUExecutionProvider"]
        or environment.get("torchDeterministicAlgorithms") is not True
        or environment.get("cuda") is not None
        or environment.get("gpu") is not None
    ):
        raise ValueError("M3 was not produced by the frozen deterministic CPU profile")
    selector = summary.get("selector")
    regression = summary.get("regression")
    if (
        not isinstance(selector, dict)
        or selector.get("manifestSha256") != digest(VALIDATION_MANIFEST_PATH)
        or selector.get("role") != "fresh-m3-selection-validation"
        or selector.get("images") != 600
        or selector.get("featureViews") != 2_400
        or selector.get("gates") != recipe["validationGates"]
        or selector.get("gatesPassed") is not True
        or selector.get("thresholdLogit") != summary.get("thresholdLogit")
        or selector.get("variants") != summary.get("variants")
    ):
        raise ValueError("M3 selector evidence changed")
    if (
        not isinstance(regression, dict)
        or regression.get("manifestSha256") != digest(REGRESSION_MANIFEST_PATH)
        or regression.get("dataRoot") != "benchmark/data/m2-head"
        or regression.get("role") != "consumed-m2-post-selection-regression"
        or regression.get("images") != 900
        or regression.get("featureViews") != 3_600
        or regression.get("gates") != recipe["regressionGates"]
        or regression.get("gatesPassed") is not True
        or regression.get("thresholdLogitFromSelector") != summary.get("thresholdLogit")
        or regression.get("selectionInfluenced") is not False
    ):
        raise ValueError("M3 post-selection regression evidence changed")
    require_variant_gates(
        selector.get("variants"),
        recipe["validationGates"],
        label="M3 selector",
        expected_synthetic_sources={"flux-1-dev-development"},
        expected_real_sources={"met-open-access"},
    )
    require_variant_gates(
        regression.get("variants"),
        recipe["regressionGates"],
        label="M2 regression",
        expected_synthetic_sources={"GLM-Image", "HunyuanImage-3.0"},
        expected_real_sources={"open-images", "stockimages-cc0"},
    )

    if (
        not isinstance(grid, list)
        or len(grid) != 25
        or not all(isinstance(row, dict) for row in grid)
        or summary.get("candidateCount") != 25
    ):
        raise ValueError("M3 candidate grid must contain all 25 predeclared pairs")
    expected_parameters = {
        (decay, alpha)
        for decay in (0.1, 0.03, 0.01, 0.003, 0.001)
        for alpha in (0.4, 0.55, 0.7, 0.85, 1.0)
    }
    if {
        (row.get("parameters", {}).get("weightDecay"), row.get("parameters", {}).get("upstreamBlendAlpha"))
        for row in grid
    } != expected_parameters:
        raise ValueError("M3 candidate-grid parameter coverage changed")
    if any("regression" in row for row in grid):
        raise ValueError("M3 regression evidence leaked into candidate selection")
    valid = [row for row in grid if isinstance(row.get("selectionKey"), list)]
    rejected = [row for row in grid if row not in valid]
    if len(valid) != summary.get("validCandidateCount") or not valid:
        raise ValueError("M3 valid-candidate count changed")
    if any(row.get("status") != "rejected" or not row.get("reason") for row in rejected):
        raise ValueError("An M3 rejected candidate lacks a reason")
    for row in valid:
        if len(row["selectionKey"]) != 5 or not all(finite_number(value) for value in row["selectionKey"]):
            raise ValueError("M3 candidate selection key changed")
        require_variant_gates(
            row.get("variants"),
            recipe["validationGates"],
            label="M3 grid candidate",
            expected_synthetic_sources={"flux-1-dev-development"},
            expected_real_sources={"met-open-access"},
        )
    selected = next((row for row in valid if row.get("parameters") == summary.get("selectedParameters")), None)
    deterministic_best = max(valid, key=lambda row: tuple(row["selectionKey"]))
    if (
        selected is None
        or selected is not deterministic_best
        or selected.get("selectionKey") != summary.get("selectionKey")
        or selected.get("thresholdLogit") != summary.get("thresholdLogit")
        or selected.get("variants") != summary.get("variants")
    ):
        raise ValueError("M3 selected candidate is not the deterministic selector maximum")

    model = summary.get("model")
    if not isinstance(model, dict):
        raise ValueError("M3 model evidence is malformed")
    parity = model.get("maxAbsParityErrorByPartition") if isinstance(model, dict) else None
    if (
        model.get("sha256") != model_hash
        or model.get("bytes") != sources["model.onnx"].stat().st_size
        or not finite_number(model.get("maxAbsParityError"))
        or model["maxAbsParityError"] > 2e-4
        or not isinstance(parity, dict)
        or set(parity) != {"selector", "regression"}
        or any(not finite_number(value) or value > 2e-4 for value in parity.values())
    ):
        raise ValueError("M3 model hash, size, or export parity changed")
    if (
        calibration.get("schemaVersion") != 1
        or calibration.get("modelSha256") != model_hash
        or calibration.get("trainManifestSha256") != summary["trainManifestSha256"]
        or calibration.get("validationManifestSha256") != summary["validationManifestSha256"]
        or calibration.get("regressionManifestSha256") != summary["regressionManifestSha256"]
        or calibration.get("selectionSummarySha256") != summary["selectionSummarySha256"]
        or calibration.get("slope") != 1
        or calibration.get("displayThreshold") != 0.65
        or calibration.get("validationThresholdLogit") != summary.get("thresholdLogit")
        or not 0 < calibration.get("rawProbabilityThreshold", -1) < 1
    ):
        raise ValueError("M3 calibration is not bound to the selector-derived threshold")
    return {
        "summary": summary,
        "calibration": calibration,
        "grid": grid,
        "modelSha256": model_hash,
        "modelBytes": sources["model.onnx"].stat().st_size,
        "candidateHashes": {name: digest(path) for name, path in sources.items()},
        "freshRunId": run_id,
    }


def validate_model_comparison(comparison: dict[str, Any], *, candidate_sha256: str, candidate_bytes: int) -> None:
    changed = comparison.get("changedInitializers")
    base = comparison.get("base")
    candidate = comparison.get("candidate")
    if (
        comparison.get("schemaVersion") != 1
        or not isinstance(base, dict)
        or not isinstance(candidate, dict)
        or base.get("path") != "weights/prooflens-cf384.onnx"
        or base.get("sha256") != UPSTREAM_MODEL_SHA256
        or candidate.get("path") != "benchmark/candidates/prooflens-cf384-m3/model.onnx"
        or candidate.get("sha256") != candidate_sha256
        or candidate.get("bytes") != candidate_bytes
        or not isinstance(changed, list)
        or not all(isinstance(row, dict) for row in changed)
        or {row.get("name") for row in changed} != {"classifier.weight", "classifier.bias"}
        or comparison.get("unchangedInitializerCount", 0) <= 0
        or not all(is_hex64(comparison.get(name)) for name in (
            "graphNodesSha256", "graphInputsSha256", "graphOutputsSha256", "opsetsSha256"
        ))
    ):
        raise ValueError("M3 classifier-only model comparison is malformed")
    dimensions = {"classifier.weight": [1, 384], "classifier.bias": [1]}
    for row in changed:
        if (
            row.get("dimensions") != dimensions[row["name"]]
            or not is_hex64(row.get("beforeSha256"))
            or not is_hex64(row.get("afterSha256"))
            or row.get("beforeSha256") == row.get("afterSha256")
        ):
            raise ValueError("M3 classifier initializer evidence is malformed")


def build_publication_lock(
    *,
    packet: dict[str, Any],
    comparison_sha256: str,
    source_commit: str,
    source_tree: str,
    public_document_hashes: dict[str, str],
    fixture_manifest_sha256: str,
    candidate_evidence_json: dict[str, str],
    classifier_patch: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "profile": "m3",
        "sourceCommit": source_commit,
        "sourceTree": source_tree,
        "upstreamModelSha256": UPSTREAM_MODEL_SHA256,
        "trainerSha256": digest(ROOT / "benchmark/modern/train_rehead.py"),
        "recipeSha256": digest(RECIPE_PATH),
        "selectionSummarySha256": digest(SELECTION_SUMMARY_PATH),
        "candidateHashes": packet["candidateHashes"],
        "candidateModelBytes": packet["modelBytes"],
        "modelComparisonSha256": comparison_sha256,
        "freshRunId": packet["freshRunId"],
        "finalizerSha256": digest(ROOT / "benchmark/m3/finalize.py"),
        "publicationContractSha256": digest(Path(__file__)),
        "fixtureSelectorSha256": digest(ROOT / "benchmark/m3/select_model_state_fixtures.py"),
        "documentationRendererSha256": digest(ROOT / "scripts/render-m3-public-docs.mjs"),
        "publicDocumentHashes": public_document_hashes,
        "fixtureManifestSha256": fixture_manifest_sha256,
        "candidateEvidenceJson": candidate_evidence_json,
        "classifierPatch": classifier_patch,
        "publicationRows": [{"path": path, "status": status} for path, status in PUBLICATION_ROWS],
        "selectionInfluencedByRegression": False,
        "h3HoldoutScored": False,
    }


def validate_publication_lock(
    lock: dict[str, Any],
    *,
    packet: dict[str, Any],
    comparison_sha256: str,
    source_commit: str,
    source_tree: str,
    public_document_hashes: dict[str, str],
    fixture_manifest_sha256: str,
    candidate_evidence_json: dict[str, str],
    classifier_patch: dict[str, Any],
) -> None:
    expected = build_publication_lock(
        packet=packet,
        comparison_sha256=comparison_sha256,
        source_commit=source_commit,
        source_tree=source_tree,
        public_document_hashes=public_document_hashes,
        fixture_manifest_sha256=fixture_manifest_sha256,
        candidate_evidence_json=candidate_evidence_json,
        classifier_patch=classifier_patch,
    )
    if lock != expected:
        raise ValueError("M3 publication lock does not match the reviewed candidate and source commit")
