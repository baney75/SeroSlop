"""Dependency-light contracts for the RunPod-only M5 fine-tuning protocol."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from base64 import b64decode
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any, Iterable, Mapping, Sequence


VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")

# Every M5 ONNX score is pinned to CUDA with TF32 disabled.  Keeping this as
# data (rather than relying on ORT's defaults) makes parity reproducible across
# the train, replay, and large-panel processes.
ORT_CUDA_PROVIDER = ("CUDAExecutionProvider", {"use_tf32": "0"})


def ort_cuda_providers(ort: Any) -> list[tuple[str, dict[str, str]]]:
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise ValueError("M5 requires CUDAExecutionProvider")
    return [(ORT_CUDA_PROVIDER[0], dict(ORT_CUDA_PROVIDER[1]))]
ROOT = Path(__file__).resolve().parents[2]
SELECTOR_SOURCES = (
    "british-library-plates",
    "rapidata-dalle-3",
    "rapidata-flux",
    "rapidata-midjourney",
    "rapidata-stable-diffusion",
)
SYNTHETIC_SELECTOR_SOURCES = SELECTOR_SOURCES[1:]
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"M5 JSON contains a duplicate key: {key}")
        result[key] = value
    return result


def parse_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = value.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"M5 {label} contains non-finite JSON: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"M5 {label} is not strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"M5 {label} must be a JSON object")
    return parsed


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = parse_json_bytes(path.read_bytes(), label="recipe")
    validate_recipe(recipe)
    return recipe


def validate_initial_parity_diagnostic(value: Mapping[str, Any], recipe: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("M5 initial parity diagnostic must be an object")
    required = {"comparisons", "h3PixelsRead", "items", "modelSha256", "pixelsSha256", "providers", "schemaVersion", "selectorRead", "status", "terminalRegressionsRead"}
    if set(value) != required or value["schemaVersion"] != 1 or value["status"] != "m5-initial-parity-diagnostic" or value["items"] != 16:
        raise ValueError("M5 initial parity diagnostic schema changed")
    if value["modelSha256"] != recipe["initialModel"]["sha256"] or value["h3PixelsRead"] or value["selectorRead"] or value["terminalRegressionsRead"]:
        raise ValueError("M5 initial parity diagnostic boundary changed")
    if not isinstance(value.get("comparisons"), Mapping) or not isinstance(value.get("providers"), Mapping):
        raise ValueError("M5 initial parity diagnostic nested objects changed")
    comparisons = value["comparisons"]
    for name in ("pytorchCpuVsOrtCudaDefault", "pytorchCpuVsOrtCudaTf32Disabled"):
        if not isinstance(comparisons.get(name), Mapping) or not isinstance(comparisons[name].get("maximumAbsoluteError"), (int, float)):
            raise ValueError("M5 initial parity diagnostic comparison types changed")
    if comparisons["pytorchCpuVsOrtCudaDefault"]["maximumAbsoluteError"] <= 0.02:
        raise ValueError("M5 initial parity diagnostic does not record the TF32/default failure")
    if comparisons["pytorchCpuVsOrtCudaTf32Disabled"]["maximumAbsoluteError"] > recipe["initialModel"]["maximumPytorchOnnxParityError"]:
        raise ValueError("M5 TF32-disabled parity diagnostic exceeds the frozen tolerance")
    try:
        options = value["providers"]["cudaTf32Disabled"]["CUDAExecutionProvider"]
        default_options = value["providers"]["cudaDefault"]["CUDAExecutionProvider"]
    except (KeyError, TypeError):
        raise ValueError("M5 parity diagnostic provider nesting changed") from None
    if not isinstance(options, Mapping) or not isinstance(default_options, Mapping):
        raise ValueError("M5 parity diagnostic provider types changed")
    if options.get("use_tf32") != "0" or default_options.get("use_tf32") != "1":
        raise ValueError("M5 parity diagnostic provider evidence changed")


def _require_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"M5 {label} schema changed")


def validate_recipe(recipe: dict[str, Any]) -> None:
    _require_keys(
        recipe,
        {
            "schemaVersion", "name", "seed", "purpose", "baseSource", "deliverable", "upstream",
            "initialModel", "preprocessing", "sourceEvidence", "training", "selection",
            "terminalRegressions", "regressionPolicy", "largeSyntheticEvaluation", "h3Boundary", "output",
        },
        "recipe",
    )
    if recipe["schemaVersion"] != 1 or recipe["name"] != "prooflens-m5-runpod-vit-finetune":
        raise ValueError("M5 recipe identity changed")
    if recipe["seed"] != 20260815:
        raise ValueError("M5 seed changed")
    if recipe["baseSource"] != {
        "commit": "5ab375fad2a744620b6ec75f09e6153c8a409049",
        "tree": "fc0afc8a746f3f41c29bbd8713f309856d2bdc53",
    }:
        raise ValueError("M5 source commit changed")
    deliverable = recipe["deliverable"]
    if (
        deliverable["format"] != "ONNX FP32"
        or deliverable["maximumBytes"] != 90_000_000
        or deliverable["browserExecution"] != ["wasm", "webgpu"]
        or deliverable["networkAfterInstall"] is not False
    ):
        raise ValueError("M5 local delivery boundary changed")
    upstream = recipe["upstream"]
    if (
        upstream["repository"] != "buildborderless/CommunityForensics-DeepfakeDet-ViT"
        or upstream["revision"] != "ac6ee457bea904a373065754107451793b56db00"
        or upstream["pytorchWeights"] != {
            "path": "model.safetensors",
            "bytes": 87_270_764,
            "sha256": "275ba982236ddd6afddf7131f8133e89f537574b964cf8fa5825b4956d741692",
        }
        or upstream["onnxReference"] != {
            "path": "onnx/model.onnx",
            "bytes": 87_442_080,
            "sha256": "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1",
        }
    ):
        raise ValueError("M5 pinned upstream changed")
    initial = recipe["initialModel"]
    if initial["sha256"] != "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47" or initial["bytes"] != 87_442_080:
        raise ValueError("M5 initial model changed")
    preprocessing = recipe["preprocessing"]
    if preprocessing != {
        "exifOrientation": True,
        "resizeShortestEdge": 440,
        "centerCrop": 384,
        "resampling": "bicubic",
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    }:
        raise ValueError("M5 browser preprocessing changed")
    source = recipe["sourceEvidence"]
    diagnostic = source["initialParityDiagnostic"]
    if diagnostic != {
        "path": "benchmark/evidence/m5/initial-parity-diagnostic.json",
        "sha256": "c9c673efa0b1a6e4ea79b195ec16c71ae8ac91f962390a49c4e570b6d8de5c11",
    } or digest_file(ROOT / diagnostic["path"]) != diagnostic["sha256"]:
        raise ValueError("M5 initial parity diagnostic binding changed")
    validate_initial_parity_diagnostic(parse_json_bytes((ROOT / diagnostic["path"]).read_bytes(), label="initial parity diagnostic"), recipe)
    training_manifest = source["trainingManifest"]
    if training_manifest != {
        "trackedPath": "benchmark/evidence/m4/train-manifest.jsonl.gz",
        "compressedSha256": "e5dfc79869541ae5c6703b60de250930f1fb8247790f55b67bb1805f5ac73a93",
        "expandedSha256": "33ef93361be0da1b7942e5b3c15368e1cf1d87351fbaea214dab8c72e90cf43e",
        "items": 112_562,
        "classCounts": {"real": 59_578, "synthetic": 52_984},
    }:
        raise ValueError("M5 training manifest binding changed")
    selector = source["selectorManifest"]
    if (
        selector["path"] != "benchmark/evidence/m4/validation-manifest.jsonl"
        or selector["sha256"] != "643eb365a603309b94b112403ef4250b565b9863d2ec61a5cc48aa80d5f85caa"
        or selector["items"] != 600
        or selector["sourceCounts"] != {
            "british-library-plates": 300,
            "rapidata-dalle-3": 75,
            "rapidata-flux": 75,
            "rapidata-midjourney": 75,
            "rapidata-stable-diffusion": 75,
        }
    ):
        raise ValueError("M5 fresh selector binding changed")
    training = recipe["training"]
    _require_keys(training, {
        "provider", "onnxRuntimeProviderPolicy", "deterministicCudaRuntime", "providerIdentityEvidence", "providerSignedAttestation", "runtimeConsistencyEvidence",
        "requiredGpuProduct", "containerImage", "requirementsPath", "requirementsSha256",
        "minimumGpuMemoryBytes", "provisioningReceiptPath", "maximumPaidWallClockSeconds",
        "deadlineSafetySeconds", "providerAutoStopAvailable", "providerAutoStopRequired", "stopControl",
        "cudaRequired", "mixedPrecision", "attentionImplementation", "epochs", "perGpuBatchSize",
        "gradientAccumulationSteps", "effectiveBatchSize", "workers", "gradientClipNorm", "warmupRatio",
        "scheduler", "optimizer", "sourceBalancedLoss", "teacherAnchor", "viewSchedule", "resumability",
        "branches", "candidateCount",
    }, "training")
    if (
        training["provider"] != "RunPod Secure Cloud (operator-recorded control-plane receipt)"
        or training["onnxRuntimeProviderPolicy"] != {"provider": "CUDAExecutionProvider", "useTf32": False}
        or training["deterministicCudaRuntime"] != {"cublasWorkspaceConfig": ":4096:8", "boundary": "trusted-runpod-execution-child-environment-before-torch-import"}
        or training["providerIdentityEvidence"] != "operator-attested-control-plane-observation"
        or training["providerSignedAttestation"] is not False
        or training["runtimeConsistencyEvidence"] != "RUNPOD_POD_ID hash and locally observed GPU match the operator-authored receipt"
        or training["requiredGpuProduct"] != "NVIDIA L40S"
        or training["containerImage"] != "pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385"
        or training["requirementsPath"] != "benchmark/m5/runpod-requirements.txt"
        or training["requirementsSha256"] != "ec87953539172609d20e1a969b8acdbf34e98a3cc8a71a6df08212c30cd41f11"
        or digest_file(ROOT / training["requirementsPath"]) != training["requirementsSha256"]
        or training["provisioningReceiptPath"] != "benchmark/candidates/prooflens-cf384-m5/runpod-provisioning-receipt.json"
        or training["maximumPaidWallClockSeconds"] != 86_400
        or training["deadlineSafetySeconds"] != 300
        or training["providerAutoStopAvailable"] is not False
        or training["providerAutoStopRequired"] is not False
        or training["stopControl"] != "trainer-deadline-plus-authenticated-operator-stop"
        or training["cudaRequired"] is not True
        or training["mixedPrecision"] != "bfloat16"
        or training["attentionImplementation"] != "eager"
        or training["epochs"] != 8
        or training["perGpuBatchSize"] != 64
        or training["gradientAccumulationSteps"] != 2
        or training["effectiveBatchSize"] != 128
        or training["candidateCount"] != 6
    ):
        raise ValueError("M5 RunPod training boundary changed")
    branches = training["branches"]
    if [branch["name"] for branch in branches] != ["last4", "full"]:
        raise ValueError("M5 branch order changed")
    if [branch["candidateEpochs"] for branch in branches] != [[4, 6, 8], [4, 6, 8]]:
        raise ValueError("M5 candidate checkpoints changed")
    if branches[0]["trainableEncoderBlocks"] != [8, 9, 10, 11] or branches[1]["trainableEncoderBlocks"] != list(range(12)):
        raise ValueError("M5 trainable backbone surface changed")
    selection = recipe["selection"]
    if selection["displayThreshold"] != 0.65:
        raise ValueError("M5 display threshold changed")
    gates = selection["gates"]
    if tuple(gates) != VARIANTS:
        raise ValueError("M5 selection variants changed")
    for variant in VARIANTS:
        if gates[variant]["minimumRealRecall"] != 1.0:
            raise ValueError("M5 zero-observed-false-positive gate changed")
    if gates["original"]["minimumBalancedAccuracy"] != 0.97 or gates["original"]["minimumSyntheticRecall"] != 0.94:
        raise ValueError("M5 original accuracy target changed")
    if selection["falsePositiveConfidence"] != {
        "method": "Wilson score interval for false-positive proportions",
        "confidenceLevel": 0.95,
        "sampleUnit": "base-real-image",
        "trialsPerVariant": 300,
        "poolAcrossVariants": False,
        "sharedBaseImagesAcrossVariants": True,
    }:
        raise ValueError("M5 selector false-positive confidence contract changed")
    regressions = recipe["terminalRegressions"]
    if [row["name"] for row in regressions] != ["m3-selector-regression", "m2-development-regression"]:
        raise ValueError("M5 terminal regression order changed")
    large_evaluation = recipe["largeSyntheticEvaluation"]
    if large_evaluation != {
        "role": "post-training-fresh-synthetic-recall-only",
        "sourceStatus": "must-be-fixed-and-public-before-first-score",
        "minimumItems": 100_000,
        "batchSize": 100,
        "minimumBatches": 1_000,
        "metric": "For each deterministic batch, synthetic recall is correctly flagged images divided by 100 at the locked model and raw threshold.",
        "minimumMeanBatchRecallExclusive": 0.95,
        "minimumMedianBatchRecallExclusive": 0.95,
        "source": {
            "repository": "JamalLee/Omni-Fake-SET",
            "revision": "724e97f5fc9f4b89f59631a8d4e6331712b7d441",
            "configuration": "image",
            "splits": ["train", "validation"],
            "sourceReportedLicense": "CC-BY-4.0",
            "expectedParquetShards": 71,
            "expectedParquetBytes": 49_751_776_056,
            "eligibleLabel": "full_synthetic",
            "excludedGeneratorFamilies": ["DALL-E", "FLUX.1-dev", "Midjourney", "Stable Diffusion"],
            "selectionNamespace": "seroslop:m5:synthetic-eval:v1",
        },
        "manifest": "benchmark/evidence/m5/large-synthetic/manifest.jsonl.gz",
        "batchAssignment": "benchmark/evidence/m5/large-synthetic/batches.json",
        "sourceLock": "benchmark/evidence/m5/large-synthetic/source-lock.json",
        "attribution": "benchmark/evidence/m5/large-synthetic/attribution.json",
        "evaluationReceipt": "benchmark/evidence/m5/large-synthetic-evaluation.json",
        "trainingOverlapAllowed": False,
        "selectorOverlapAllowed": False,
        "regressionOverlapAllowed": False,
        "selectionInfluence": False,
        "scoreBlindnessEvidence": {
            "repositoryScoreArtifactsPresentAtSourceLock": False,
            "publicSourceLockPrecedesEvaluationReceipt": True,
            "firstInferenceAfterLock": "operator-attested",
            "privatePriorScoringAbsenceProven": False,
            "trainingExclusionClaim": "not-used-in-seroslop-m2-through-m5-gradients-or-selection",
        },
        "failure": "If either aggregate is not strictly above 0.95, the model is not accepted. Any retraining must use new training data and a newly fixed 100,000-image panel. Scored evaluation rows may never enter training or selection. The public source lock precedes the repository evaluation receipt; absence of private prior scoring remains operator-attested, not cryptographically proven.",
    }:
        raise ValueError("M5 100,000-image evaluation boundary changed")
    h3 = recipe["h3Boundary"]
    if (
        h3["sha256"] != "50574778ab0d58f839f1dccc3c99da5f6dca98150186f13aeca8d9ba052e9547"
        or h3["pixelsMayBeRead"] is not False
        or h3["acceptedArguments"] is not False
        or h3["scoreArtifactsBeforeFinalLock"] is not False
    ):
        raise ValueError("M5 H3 boundary changed")


def require_safe_relative_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"M5 {label} path is empty")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"M5 {label} path escapes its data root")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    handle = gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("rt", encoding="utf-8")
    with handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"M5 manifest row {line_number} is invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"M5 manifest row {line_number} is not an object")
            rows.append(row)
    return rows


def validate_manifest_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_items: int,
    expected_class_counts: Mapping[str, int],
    expected_source_counts: Mapping[str, int] | None = None,
) -> None:
    if len(rows) != expected_items:
        raise ValueError("M5 manifest count changed")
    ids: set[str] = set()
    hashes: set[str] = set()
    paths: set[str] = set()
    row_indexes: set[int] = set()
    class_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for row in rows:
        identifier = row.get("id")
        image_hash = row.get("imageSha256")
        source = row.get("source")
        label = row.get("label")
        row_index = row.get("rowIndex")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("M5 manifest ID is missing or duplicated")
        if not isinstance(image_hash, str) or not HEX64.fullmatch(image_hash) or image_hash in hashes:
            raise ValueError("M5 manifest image hash is missing or duplicated")
        path = require_safe_relative_path(row.get("path"), label="manifest")
        if path in paths:
            raise ValueError("M5 manifest path is duplicated")
        if not isinstance(source, str) or not source:
            raise ValueError("M5 manifest source is invalid")
        if label not in (0, 1):
            raise ValueError("M5 manifest label is invalid")
        if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index in row_indexes:
            raise ValueError("M5 manifest rowIndex is invalid or duplicated")
        ids.add(identifier)
        hashes.add(image_hash)
        paths.add(path)
        row_indexes.add(row_index)
        class_counts["real" if label == 0 else "synthetic"] += 1
        source_counts[source] += 1
    if row_indexes != set(range(expected_items)):
        raise ValueError("M5 manifest rowIndex is not a complete permutation")
    if dict(class_counts) != dict(expected_class_counts):
        raise ValueError("M5 manifest class counts changed")
    if expected_source_counts is not None and dict(source_counts) != dict(expected_source_counts):
        raise ValueError("M5 manifest source counts changed")


def source_balanced_weights(rows: Sequence[dict[str, Any]]) -> list[float]:
    pair_counts = Counter((int(row["label"]), str(row["source"])) for row in rows)
    sources_by_class = {
        label: sorted(source for (pair_label, source) in pair_counts if pair_label == label)
        for label in (0, 1)
    }
    if not all(sources_by_class.values()):
        raise ValueError("M5 source-balanced loss requires both classes")
    total = len(rows)
    result: list[float] = []
    for row in rows:
        label = int(row["label"])
        source = str(row["source"])
        result.append(total / (2.0 * len(sources_by_class[label]) * pair_counts[(label, source)]))
    for label in (0, 1):
        mass = math.fsum(weight for row, weight in zip(rows, result, strict=True) if int(row["label"]) == label)
        if not math.isclose(mass, total / 2.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("M5 source-balanced class mass is wrong")
        for source in sources_by_class[label]:
            source_mass = math.fsum(
                weight for row, weight in zip(rows, result, strict=True)
                if int(row["label"]) == label and str(row["source"]) == source
            )
            expected = total / (2.0 * len(sources_by_class[label]))
            if not math.isclose(source_mass, expected, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("M5 source-balanced source mass is wrong")
    return result


def complete_thresholds(logits: Iterable[float]) -> list[float]:
    values = sorted(set(float(value) for value in logits))
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("M5 threshold logits must be finite and nonempty")
    below = math.nextafter(values[0], -math.inf)
    thresholds = [below]
    for value in values:
        thresholds.append(value)
        above = math.nextafter(value, math.inf)
        if math.isfinite(above):
            thresholds.append(above)
    return sorted(set(thresholds))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if isinstance(successes, bool) or isinstance(total, bool) or not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("M5 Wilson counts must be integers")
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("M5 Wilson counts are out of range")
    z = 1.9599639845400534
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


@dataclass(frozen=True)
class VariantMetrics:
    balanced_accuracy: float
    real_recall: float
    synthetic_recall: float
    synthetic_recall_by_source: dict[str, float]
    false_positives: int
    false_positive_trials: int
    false_positive_rate: float
    false_positive_wilson95: dict[str, float]


def metrics_at_threshold(
    logits: Sequence[float],
    rows: Sequence[dict[str, Any]],
    threshold: float,
) -> VariantMetrics:
    if len(logits) != len(rows) or not logits:
        raise ValueError("M5 prediction coverage changed")
    totals = Counter(int(row["label"]) for row in rows)
    correct = Counter()
    source_total = Counter()
    source_correct = Counter()
    false_positives = 0
    for logit, row in zip(logits, rows, strict=True):
        if not math.isfinite(float(logit)):
            raise ValueError("M5 prediction logit is not finite")
        label = int(row["label"])
        prediction = 1 if float(logit) >= threshold else 0
        correct[label] += int(prediction == label)
        if label == 0 and prediction == 1:
            false_positives += 1
        if label == 1:
            source = str(row["source"])
            source_total[source] += 1
            source_correct[source] += int(prediction == 1)
    if totals[0] == 0 or totals[1] == 0:
        raise ValueError("M5 selector needs both classes")
    real = correct[0] / totals[0]
    synthetic = correct[1] / totals[1]
    lower, upper = wilson_interval(false_positives, totals[0])
    return VariantMetrics(
        balanced_accuracy=(real + synthetic) / 2.0,
        real_recall=real,
        synthetic_recall=synthetic,
        synthetic_recall_by_source={source: source_correct[source] / total for source, total in sorted(source_total.items())},
        false_positives=false_positives,
        false_positive_trials=totals[0],
        false_positive_rate=false_positives / totals[0],
        false_positive_wilson95={"lower": lower, "upper": upper},
    )


def selector_metrics_pass(metrics: Mapping[str, VariantMetrics], gates: Mapping[str, Mapping[str, float]]) -> bool:
    if tuple(metrics) != VARIANTS or tuple(gates) != VARIANTS:
        raise ValueError("M5 selector variant coverage changed")
    for variant in VARIANTS:
        value = metrics[variant]
        gate = gates[variant]
        if value.false_positives != 0 or value.real_recall != 1.0:
            return False
        if value.balanced_accuracy + 1e-15 < gate["minimumBalancedAccuracy"]:
            return False
        if value.synthetic_recall + 1e-15 < gate["minimumSyntheticRecall"]:
            return False
        if set(value.synthetic_recall_by_source) != set(SYNTHETIC_SELECTOR_SOURCES):
            raise ValueError("M5 selector synthetic source coverage changed")
        if min(value.synthetic_recall_by_source.values()) + 1e-15 < gate["minimumSyntheticRecallBySource"]:
            return False
    return True


def choose_selector_threshold(
    logits_by_variant: Mapping[str, Sequence[float]],
    rows: Sequence[dict[str, Any]],
    gates: Mapping[str, Mapping[str, float]],
) -> tuple[float, dict[str, VariantMetrics], tuple[float, ...]] | None:
    if tuple(logits_by_variant) != VARIANTS:
        raise ValueError("M5 selector prediction variants changed")
    thresholds = complete_thresholds(value for values in logits_by_variant.values() for value in values)
    best: tuple[float, dict[str, VariantMetrics], tuple[float, ...]] | None = None
    for threshold in thresholds:
        metrics = {
            variant: metrics_at_threshold(logits_by_variant[variant], rows, threshold)
            for variant in VARIANTS
        }
        if not selector_metrics_pass(metrics, gates):
            continue
        key = (
            min(value.balanced_accuracy for value in metrics.values()),
            metrics["original"].balanced_accuracy,
            min(value.synthetic_recall for value in metrics.values()),
            min(source for value in metrics.values() for source in value.synthetic_recall_by_source.values()),
            threshold,
        )
        if best is None or key > best[2]:
            best = (threshold, metrics, key)
    return best


def branch_candidate_ids(recipe: Mapping[str, Any]) -> list[str]:
    return [
        f"{branch['name']}-epoch-{epoch}"
        for branch in recipe["training"]["branches"]
        for epoch in branch["candidateEpochs"]
    ]


def validate_environment_receipt(receipt: Mapping[str, Any], recipe: Mapping[str, Any]) -> None:
    required = {
        "provider", "gpuProduct", "gpuMemoryBytes", "cudaAvailable", "cudaVersion", "driverVersion",
        "torchVersion", "transformersVersion", "pythonVersion", "runpodPodIdSha256",
        "launchNodeVersion", "launchNodeSha256",
        "provisioningReceiptSha256", "containerImage", "requirementsSha256", "providerEvidenceBoundary",
        "providerIdentityEvidence", "providerSignedAttestation", "runtimeConsistencyEvidence",
        "cublasWorkspaceConfig",
        "sourceCommit", "sourceTree", "authorizationCommit", "authorizationReceiptSha256", "authorizationPublicCi",
    }
    _require_keys(receipt, required, "environment receipt")
    if receipt["provider"] != recipe["training"]["provider"]:
        raise ValueError("M5 training provider changed")
    if receipt["gpuProduct"] != recipe["training"]["requiredGpuProduct"]:
        raise ValueError("M5 GPU product changed")
    if (
        receipt["containerImage"] != recipe["training"]["containerImage"]
        or receipt["requirementsSha256"] != recipe["training"]["requirementsSha256"]
    ):
        raise ValueError("M5 RunPod image or dependency lock changed")
    if receipt["cudaAvailable"] is not True or int(receipt["gpuMemoryBytes"]) < recipe["training"]["minimumGpuMemoryBytes"]:
        raise ValueError("M5 CUDA capacity is insufficient")
    if receipt["transformersVersion"] != "5.4.0" or receipt["torchVersion"].split("+")[0] != "2.8.0":
        raise ValueError("M5 training runtime changed")
    if (
        receipt["launchNodeVersion"] != "v24.18.1"
        or receipt["launchNodeSha256"] != "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a"
    ):
        raise ValueError("M5 pre-import Node runtime changed")
    if not HEX64.fullmatch(str(receipt["runpodPodIdSha256"])):
        raise ValueError("M5 RunPod identity receipt is invalid")
    if not HEX64.fullmatch(str(receipt["provisioningReceiptSha256"])):
        raise ValueError("M5 provisioning receipt binding is invalid")
    if not HEX40.fullmatch(str(receipt["sourceCommit"])) or not HEX40.fullmatch(str(receipt["sourceTree"])):
        raise ValueError("M5 source authorization binding is invalid")
    if not HEX40.fullmatch(str(receipt["authorizationCommit"])) or not HEX64.fullmatch(str(receipt["authorizationReceiptSha256"])):
        raise ValueError("M5 authorization receipt binding is invalid")
    authorization_ci = receipt["authorizationPublicCi"]
    _require_keys(authorization_ci, {"conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"}, "M5 authorization public CI")
    if (
        authorization_ci["conclusion"] != "success"
        or authorization_ci["event"] != "push"
        or authorization_ci["headSha"] != receipt["authorizationCommit"]
        or isinstance(authorization_ci["runId"], bool)
        or not isinstance(authorization_ci["runId"], int)
        or authorization_ci["runId"] <= 0
        or authorization_ci["status"] != "completed"
        or authorization_ci["url"] != f"https://github.com/baney75/prooflens/actions/runs/{authorization_ci['runId']}"
        or authorization_ci["workflowPath"] != ".github/workflows/quality.yml"
    ):
        raise ValueError("M5 authorization public CI binding changed")
    if receipt["providerEvidenceBoundary"] != "operator-recorded-not-cryptographic-attestation":
        raise ValueError("M5 provider evidence boundary changed")
    if (
        receipt["providerIdentityEvidence"] != recipe["training"]["providerIdentityEvidence"]
        or receipt["providerSignedAttestation"] is not False
        or receipt["runtimeConsistencyEvidence"] != recipe["training"]["runtimeConsistencyEvidence"]
        or receipt["cublasWorkspaceConfig"] != recipe["training"]["deterministicCudaRuntime"]["cublasWorkspaceConfig"]
    ):
        raise ValueError("M5 provider identity or deterministic CUDA claim exceeds its evidence")


def validate_run_authorization(
    receipt: Mapping[str, Any],
    *,
    protocol_commit: str,
    protocol_tree: str,
    source_commit: str,
    source_tree: str,
    source_path_map: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the canonical P4 receipt binding the exact P3 source bytes."""
    _require_keys(receipt, {
        "schemaVersion", "status", "protocolCommit", "protocolTree", "sourceCommit", "sourceTree",
        "sourcePathMap", "sourcePublicCi", "authorizationPath", "scoreBlind", "h3PixelsRead",
    }, "M5 run authorization")
    if (
        receipt["schemaVersion"] != 1
        or receipt["status"] != "m5-source-recovery-authorized"
        or receipt["protocolCommit"] != protocol_commit
        or receipt["protocolTree"] != protocol_tree
        or receipt["sourceCommit"] != source_commit
        or receipt["sourceTree"] != source_tree
        or receipt["authorizationPath"] != "benchmark/evidence/m5/run-authorization.json"
        or receipt["scoreBlind"] is not True
        or receipt["h3PixelsRead"] is not False
        or list(receipt["sourcePathMap"]) != list(source_path_map)
    ):
        raise ValueError("M5 run authorization binding changed")
    if not HEX40.fullmatch(str(source_commit)) or not HEX40.fullmatch(str(source_tree)):
        raise ValueError("M5 run authorization source digest is invalid")
    for row in receipt["sourcePathMap"]:
        _require_keys(row, {"path", "sha256"}, "M5 run authorization source path")
        if not isinstance(row["path"], str) or not HEX64.fullmatch(str(row["sha256"])):
            raise ValueError("M5 run authorization source path digest is invalid")
    source_ci = receipt["sourcePublicCi"]
    _require_keys(source_ci, {"conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"}, "M5 source public CI")
    if (
        source_ci["conclusion"] != "success"
        or source_ci["event"] != "push"
        or source_ci["headSha"] != source_commit
        or isinstance(source_ci["runId"], bool)
        or not isinstance(source_ci["runId"], int)
        or source_ci["runId"] <= 0
        or source_ci["status"] != "completed"
        or source_ci["url"] != f"https://github.com/baney75/prooflens/actions/runs/{source_ci['runId']}"
        or source_ci["workflowPath"] != ".github/workflows/quality.yml"
    ):
        raise ValueError("M5 source public CI binding changed")


def validate_runtime_recovery_authorization(
    receipt: Mapping[str, Any],
    *,
    protocol_commit: str,
    protocol_tree: str,
    prior_authorization_commit: str,
    prior_authorization_tree: str,
    prior_authorization_sha256: str,
    source_commit: str,
    source_tree: str,
    source_path_map: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the second receipt authorizing the append-only runtime repair."""
    _require_keys(receipt, {
        "schemaVersion", "status", "protocolCommit", "protocolTree",
        "priorAuthorizationCommit", "priorAuthorizationTree", "priorAuthorizationPath",
        "priorAuthorizationSha256", "sourceCommit", "sourceTree", "sourcePathMap",
        "sourcePublicCi", "authorizationPath", "scoreBlind", "h3PixelsRead",
    }, "M5 runtime recovery authorization")
    if (
        receipt["schemaVersion"] != 2
        or receipt["status"] != "m5-runtime-recovery-authorized"
        or receipt["protocolCommit"] != protocol_commit
        or receipt["protocolTree"] != protocol_tree
        or receipt["priorAuthorizationCommit"] != prior_authorization_commit
        or receipt["priorAuthorizationTree"] != prior_authorization_tree
        or receipt["priorAuthorizationPath"] != "benchmark/evidence/m5/run-authorization.json"
        or receipt["priorAuthorizationSha256"] != prior_authorization_sha256
        or receipt["sourceCommit"] != source_commit
        or receipt["sourceTree"] != source_tree
        or receipt["authorizationPath"] != "benchmark/evidence/m5/runtime-recovery-authorization.json"
        or receipt["scoreBlind"] is not True
        or receipt["h3PixelsRead"] is not False
        or list(receipt["sourcePathMap"]) != list(source_path_map)
    ):
        raise ValueError("M5 runtime recovery authorization binding changed")
    for value in (
        protocol_commit, protocol_tree, prior_authorization_commit,
        prior_authorization_tree, source_commit, source_tree,
    ):
        if not HEX40.fullmatch(str(value)):
            raise ValueError("M5 runtime recovery authorization Git identity is invalid")
    if not HEX64.fullmatch(str(prior_authorization_sha256)):
        raise ValueError("M5 prior authorization digest is invalid")
    for row in receipt["sourcePathMap"]:
        _require_keys(row, {"path", "sha256"}, "M5 runtime recovery source path")
        if not isinstance(row["path"], str) or not HEX64.fullmatch(str(row["sha256"])):
            raise ValueError("M5 runtime recovery source path digest is invalid")
    source_ci = receipt["sourcePublicCi"]
    _require_keys(source_ci, {"conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"}, "M5 runtime recovery public CI")
    if (
        source_ci["conclusion"] != "success"
        or source_ci["event"] != "push"
        or source_ci["headSha"] != source_commit
        or isinstance(source_ci["runId"], bool)
        or not isinstance(source_ci["runId"], int)
        or source_ci["runId"] <= 0
        or source_ci["status"] != "completed"
        or source_ci["url"] != f"https://github.com/baney75/prooflens/actions/runs/{source_ci['runId']}"
        or source_ci["workflowPath"] != ".github/workflows/quality.yml"
    ):
        raise ValueError("M5 runtime recovery public CI binding changed")


def validate_runpod_environment_authorization(
    receipt: Mapping[str, Any],
    *,
    protocol_commit: str,
    protocol_tree: str,
    prior_authorization_commit: str,
    prior_authorization_tree: str,
    prior_authorization_sha256: str,
    source_commit: str,
    source_tree: str,
    source_path_map: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the receipt authorizing PID-1 RunPod Pod-ID propagation."""
    _require_keys(receipt, {
        "schemaVersion", "status", "protocolCommit", "protocolTree",
        "priorAuthorizationCommit", "priorAuthorizationTree", "priorAuthorizationPath",
        "priorAuthorizationSha256", "sourceCommit", "sourceTree", "sourcePathMap", "environmentBoundary",
        "sourcePublicCi", "authorizationPath", "scoreBlind", "h3PixelsRead",
    }, "M5 RunPod environment authorization")
    if (
        receipt["schemaVersion"] != 3
        or receipt["status"] != "m5-runpod-environment-recovery-authorized"
        or receipt["protocolCommit"] != protocol_commit
        or receipt["protocolTree"] != protocol_tree
        or receipt["priorAuthorizationCommit"] != prior_authorization_commit
        or receipt["priorAuthorizationTree"] != prior_authorization_tree
        or receipt["priorAuthorizationPath"] != "benchmark/evidence/m5/runtime-recovery-authorization.json"
        or receipt["priorAuthorizationSha256"] != prior_authorization_sha256
        or receipt["sourceCommit"] != source_commit
        or receipt["sourceTree"] != source_tree
        or receipt["authorizationPath"] != "benchmark/evidence/m5/runpod-environment-authorization.json"
        or receipt["environmentBoundary"] != "validated-single-runpod-pod-id-from-pid1-environ-no-other-record-forwarded"
        or receipt["scoreBlind"] is not True
        or receipt["h3PixelsRead"] is not False
        or list(receipt["sourcePathMap"]) != list(source_path_map)
    ):
        raise ValueError("M5 RunPod environment authorization binding changed")
    for value in (
        protocol_commit, protocol_tree, prior_authorization_commit,
        prior_authorization_tree, source_commit, source_tree,
    ):
        if not HEX40.fullmatch(str(value)):
            raise ValueError("M5 RunPod environment authorization Git identity is invalid")
    if not HEX64.fullmatch(str(prior_authorization_sha256)):
        raise ValueError("M5 prior runtime authorization digest is invalid")
    for row in receipt["sourcePathMap"]:
        _require_keys(row, {"path", "sha256"}, "M5 RunPod environment source path")
        if not isinstance(row["path"], str) or not HEX64.fullmatch(str(row["sha256"])):
            raise ValueError("M5 RunPod environment source path digest is invalid")
    source_ci = receipt["sourcePublicCi"]
    _require_keys(source_ci, {"conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"}, "M5 RunPod environment public CI")
    if (
        source_ci["conclusion"] != "success"
        or source_ci["event"] != "push"
        or source_ci["headSha"] != source_commit
        or isinstance(source_ci["runId"], bool)
        or not isinstance(source_ci["runId"], int)
        or source_ci["runId"] <= 0
        or source_ci["status"] != "completed"
        or source_ci["url"] != f"https://github.com/baney75/prooflens/actions/runs/{source_ci['runId']}"
        or source_ci["workflowPath"] != ".github/workflows/quality.yml"
    ):
        raise ValueError("M5 RunPod environment public CI binding changed")


def validate_numeric_audit_authorization(
    receipt: Mapping[str, Any],
    *,
    protocol_commit: str,
    protocol_tree: str,
    prior_authorization_commit: str,
    prior_authorization_tree: str,
    prior_authorization_sha256: str,
    source_commit: str,
    source_tree: str,
    source_path_map: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the receipt authorizing the audit-only stable-sum recovery."""
    _require_keys(receipt, {
        "schemaVersion", "status", "protocolCommit", "protocolTree",
        "priorAuthorizationCommit", "priorAuthorizationTree", "priorAuthorizationPath",
        "priorAuthorizationSha256", "sourceCommit", "sourceTree", "sourcePathMap", "numericBoundary",
        "sourcePublicCi", "authorizationPath", "scoreBlind", "h3PixelsRead",
    }, "M5 numeric audit authorization")
    if (
        receipt["schemaVersion"] != 4
        or receipt["status"] != "m5-numeric-audit-recovery-authorized"
        or receipt["protocolCommit"] != protocol_commit
        or receipt["protocolTree"] != protocol_tree
        or receipt["priorAuthorizationCommit"] != prior_authorization_commit
        or receipt["priorAuthorizationTree"] != prior_authorization_tree
        or receipt["priorAuthorizationPath"] != "benchmark/evidence/m5/runpod-environment-authorization.json"
        or receipt["priorAuthorizationSha256"] != prior_authorization_sha256
        or receipt["sourceCommit"] != source_commit
        or receipt["sourceTree"] != source_tree
        or receipt["authorizationPath"] != "benchmark/evidence/m5/numeric-audit-authorization.json"
        or receipt["numericBoundary"] != "source-balanced-weights-unchanged-math-fsum-audit-only"
        or receipt["scoreBlind"] is not True
        or receipt["h3PixelsRead"] is not False
        or list(receipt["sourcePathMap"]) != list(source_path_map)
    ):
        raise ValueError("M5 numeric audit authorization binding changed")
    for value in (
        protocol_commit, protocol_tree, prior_authorization_commit,
        prior_authorization_tree, source_commit, source_tree,
    ):
        if not HEX40.fullmatch(str(value)):
            raise ValueError("M5 numeric audit authorization Git identity is invalid")
    if not HEX64.fullmatch(str(prior_authorization_sha256)):
        raise ValueError("M5 prior RunPod environment authorization digest is invalid")
    for row in receipt["sourcePathMap"]:
        _require_keys(row, {"path", "sha256"}, "M5 numeric audit source path")
        if not isinstance(row["path"], str) or not HEX64.fullmatch(str(row["sha256"])):
            raise ValueError("M5 numeric audit source path digest is invalid")
    source_ci = receipt["sourcePublicCi"]
    _require_keys(source_ci, {"conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"}, "M5 numeric audit public CI")
    if (
        source_ci["conclusion"] != "success"
        or source_ci["event"] != "push"
        or source_ci["headSha"] != source_commit
        or isinstance(source_ci["runId"], bool)
        or not isinstance(source_ci["runId"], int)
        or source_ci["runId"] <= 0
        or source_ci["status"] != "completed"
        or source_ci["url"] != f"https://github.com/baney75/prooflens/actions/runs/{source_ci['runId']}"
        or source_ci["workflowPath"] != ".github/workflows/quality.yml"
    ):
        raise ValueError("M5 numeric audit public CI binding changed")


def validate_parity_recovery_authorization(
    receipt: Mapping[str, Any],
    *,
    protocol_commit: str,
    protocol_tree: str,
    prior_authorization_commit: str,
    prior_authorization_tree: str,
    prior_authorization_sha256: str,
    source_commit: str,
    source_tree: str,
    source_path_map: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the receipt authorizing the TF32-off, ONNX-scored recovery."""
    _require_keys(receipt, {
        "schemaVersion", "status", "protocolCommit", "protocolTree",
        "priorAuthorizationCommit", "priorAuthorizationTree", "priorAuthorizationPath",
        "priorAuthorizationSha256", "sourceCommit", "sourceTree", "sourcePathMap",
        "parityBoundary", "diagnosticSha256", "sourcePublicCi", "authorizationPath",
        "scoreBlind", "h3PixelsRead",
    }, "M5 parity recovery authorization")
    if (
        receipt["schemaVersion"] != 5
        or receipt["status"] != "m5-parity-recovery-authorized"
        or receipt["protocolCommit"] != protocol_commit
        or receipt["protocolTree"] != protocol_tree
        or receipt["priorAuthorizationCommit"] != prior_authorization_commit
        or receipt["priorAuthorizationTree"] != prior_authorization_tree
        or receipt["priorAuthorizationPath"] != "benchmark/evidence/m5/numeric-audit-authorization.json"
        or receipt["priorAuthorizationSha256"] != prior_authorization_sha256
        or receipt["sourceCommit"] != source_commit
        or receipt["sourceTree"] != source_tree
        or list(receipt["sourcePathMap"]) != list(source_path_map)
        or receipt["parityBoundary"] != "packaged-m2-reference-preserved-real-input-parity-and-onnx-scoring"
        or receipt["diagnosticSha256"] != "c9c673efa0b1a6e4ea79b195ec16c71ae8ac91f962390a49c4e570b6d8de5c11"
        or receipt["authorizationPath"] != "benchmark/evidence/m5/parity-recovery-authorization.json"
        or receipt["scoreBlind"] is not True
        or receipt["h3PixelsRead"] is not False
    ):
        raise ValueError("M5 parity recovery authorization binding changed")
    for value in (
        protocol_commit, protocol_tree, prior_authorization_commit,
        prior_authorization_tree, source_commit, source_tree,
    ):
        if not HEX40.fullmatch(str(value)):
            raise ValueError("M5 parity recovery authorization Git identity is invalid")
    if not HEX64.fullmatch(str(prior_authorization_sha256)):
        raise ValueError("M5 prior numeric authorization digest is invalid")
    for row in receipt["sourcePathMap"]:
        _require_keys(row, {"path", "sha256"}, "M5 parity recovery source path")
        if not isinstance(row["path"], str) or not HEX64.fullmatch(str(row["sha256"])):
            raise ValueError("M5 parity recovery source path digest is invalid")
    source_ci = receipt["sourcePublicCi"]
    _require_keys(source_ci, {"conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"}, "M5 parity recovery public CI")
    if (
        source_ci["conclusion"] != "success"
        or source_ci["event"] != "push"
        or source_ci["headSha"] != source_commit
        or isinstance(source_ci["runId"], bool)
        or not isinstance(source_ci["runId"], int)
        or source_ci["runId"] <= 0
        or source_ci["status"] != "completed"
        or source_ci["url"] != f"https://github.com/baney75/prooflens/actions/runs/{source_ci['runId']}"
        or source_ci["workflowPath"] != ".github/workflows/quality.yml"
    ):
        raise ValueError("M5 parity recovery public CI binding changed")


def validate_cublas_recovery_authorization(
    receipt: Mapping[str, Any], *, protocol_commit: str, protocol_tree: str,
    prior_authorization_commit: str, prior_authorization_tree: str,
    prior_authorization_sha256: str, source_commit: str, source_tree: str,
    source_path_map: Sequence[Mapping[str, Any]],
) -> None:
    """Validate the score-blind deterministic CUDA workspace recovery receipt."""
    _require_keys(receipt, {
        "schemaVersion", "status", "protocolCommit", "protocolTree",
        "priorAuthorizationCommit", "priorAuthorizationTree", "priorAuthorizationPath",
        "priorAuthorizationSha256", "sourceCommit", "sourceTree", "sourcePathMap",
        "runtimeBoundary", "cublasWorkspaceConfig", "sourcePublicCi", "authorizationPath",
        "scoreBlind", "h3PixelsRead",
    }, "M5 cuBLAS recovery authorization")
    if (
        receipt["schemaVersion"] != 6 or receipt["status"] != "m5-cublas-recovery-authorized"
        or receipt["protocolCommit"] != protocol_commit or receipt["protocolTree"] != protocol_tree
        or receipt["priorAuthorizationCommit"] != prior_authorization_commit
        or receipt["priorAuthorizationTree"] != prior_authorization_tree
        or receipt["priorAuthorizationPath"] != "benchmark/evidence/m5/parity-recovery-authorization.json"
        or receipt["priorAuthorizationSha256"] != prior_authorization_sha256
        or receipt["sourceCommit"] != source_commit or receipt["sourceTree"] != source_tree
        or list(receipt["sourcePathMap"]) != list(source_path_map)
        or receipt["runtimeBoundary"] != "trusted-runpod-execution-child-environment-before-torch-import"
        or receipt["cublasWorkspaceConfig"] != ":4096:8"
        or receipt["authorizationPath"] != "benchmark/evidence/m5/cublas-recovery-authorization.json"
        or receipt["scoreBlind"] is not True or receipt["h3PixelsRead"] is not False
    ):
        raise ValueError("M5 cuBLAS recovery authorization binding changed")
    for value in (protocol_commit, protocol_tree, prior_authorization_commit, prior_authorization_tree, source_commit, source_tree):
        if not HEX40.fullmatch(str(value)):
            raise ValueError("M5 cuBLAS recovery authorization Git identity is invalid")
    if not HEX64.fullmatch(str(prior_authorization_sha256)):
        raise ValueError("M5 prior parity authorization digest is invalid")
    for row in receipt["sourcePathMap"]:
        _require_keys(row, {"path", "sha256"}, "M5 cuBLAS recovery source path")
        if not isinstance(row["path"], str) or not HEX64.fullmatch(str(row["sha256"])):
            raise ValueError("M5 cuBLAS recovery source path digest is invalid")
    source_ci = receipt["sourcePublicCi"]
    _require_keys(source_ci, {"conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"}, "M5 cuBLAS recovery public CI")
    if (source_ci["conclusion"] != "success" or source_ci["event"] != "push" or source_ci["headSha"] != source_commit
        or isinstance(source_ci["runId"], bool) or not isinstance(source_ci["runId"], int) or source_ci["runId"] <= 0
        or source_ci["status"] != "completed" or source_ci["url"] != f"https://github.com/baney75/prooflens/actions/runs/{source_ci['runId']}"
        or source_ci["workflowPath"] != ".github/workflows/quality.yml"):
        raise ValueError("M5 cuBLAS recovery public CI binding changed")


def validate_provisioning_receipt(receipt: Mapping[str, Any], recipe: Mapping[str, Any]) -> None:
    _require_keys(
        receipt,
        {
            "schemaVersion", "status", "provider", "cloudType", "gpuProduct", "containerImage", "podIdSha256",
            "createdAtUnix", "maximumRuntimeSeconds", "workloadStopAtUnix", "providerAutoStopAvailable",
            "operatorStopRequired", "stopControl",
            "controlPlaneObservationSha256", "evidenceBoundary",
        },
        "RunPod provisioning receipt",
    )
    training = recipe["training"]
    if (
        receipt["schemaVersion"] != 1
        or receipt["status"] != "runpod-provisioned"
        or receipt["provider"] != "RunPod"
        or receipt["cloudType"] != "SECURE"
        or receipt["gpuProduct"] != training["requiredGpuProduct"]
        or receipt["containerImage"] != training["containerImage"]
        or receipt["providerAutoStopAvailable"] is not False
        or receipt["operatorStopRequired"] is not True
        or receipt["stopControl"] != training["stopControl"]
        or receipt["maximumRuntimeSeconds"] != training["maximumPaidWallClockSeconds"]
        or receipt["evidenceBoundary"] != "operator-recorded-control-plane-observation-not-cryptographic-attestation"
    ):
        raise ValueError("M5 RunPod provisioning boundary changed")
    if not HEX64.fullmatch(str(receipt["podIdSha256"])) or not HEX64.fullmatch(str(receipt["controlPlaneObservationSha256"])):
        raise ValueError("M5 RunPod provisioning digests are invalid")
    created = receipt["createdAtUnix"]
    workload_stop = receipt["workloadStopAtUnix"]
    if (
        isinstance(created, bool)
        or not isinstance(created, int)
        or isinstance(workload_stop, bool)
        or not isinstance(workload_stop, int)
        or workload_stop != created + training["maximumPaidWallClockSeconds"]
    ):
        raise ValueError("M5 RunPod paid deadline changed")


def unpack_float32(packet: Mapping[str, Any], *, expected_count: int) -> list[float]:
    _require_keys(packet, {"dtype", "count", "sha256", "base64"}, "logit packet")
    if packet["dtype"] != "float32-little-endian" or packet["count"] != expected_count:
        raise ValueError("M5 logit packet shape changed")
    try:
        payload = b64decode(str(packet["base64"]), validate=True)
    except Exception as error:
        raise ValueError("M5 logit packet base64 is invalid") from error
    if len(payload) != expected_count * 4 or digest_bytes(payload) != packet["sha256"]:
        raise ValueError("M5 logit packet bytes changed")
    values = [value[0] for value in struct.iter_unpack("<f", payload)]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("M5 logit packet contains non-finite values")
    return values


def regression_metrics(
    logits: Sequence[float],
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    totals: Counter[int] = Counter()
    correct: Counter[int] = Counter()
    source_totals: Counter[tuple[int, str]] = Counter()
    source_correct: Counter[tuple[int, str]] = Counter()
    for logit, row in zip(logits, rows, strict=True):
        if not math.isfinite(float(logit)):
            raise ValueError("M5 regression logit is non-finite")
        label = int(row["label"])
        source = str(row["source"])
        prediction = int(float(logit) >= threshold)
        totals[label] += 1
        correct[label] += int(prediction == label)
        source_totals[(label, source)] += 1
        source_correct[(label, source)] += int(prediction == label)
    if not totals[0] or not totals[1]:
        raise ValueError("M5 regression requires both classes")
    real = correct[0] / totals[0]
    synthetic = correct[1] / totals[1]
    return {
        "balancedAccuracy": (real + synthetic) / 2.0,
        "realRecall": real,
        "syntheticRecall": synthetic,
        "realRecallBySource": {
            source: source_correct[(0, source)] / count
            for (label, source), count in sorted(source_totals.items()) if label == 0
        },
        "syntheticRecallBySource": {
            source: source_correct[(1, source)] / count
            for (label, source), count in sorted(source_totals.items()) if label == 1
        },
    }


def regression_gates_pass(metrics: Mapping[str, Mapping[str, Any]], gates: Mapping[str, Any]) -> bool:
    if tuple(metrics) != VARIANTS:
        raise ValueError("M5 regression variant coverage changed")
    required_real_sources = set(gates["minimumRealRecallBySource"])
    for variant in VARIANTS:
        value = metrics[variant]
        if set(value) != {
            "balancedAccuracy", "realRecall", "syntheticRecall",
            "realRecallBySource", "syntheticRecallBySource",
        }:
            raise ValueError("M5 regression metric schema changed")
        if set(value["realRecallBySource"]) != required_real_sources:
            raise ValueError("M5 regression real-source coverage changed")
        if not value["syntheticRecallBySource"]:
            raise ValueError("M5 regression synthetic-source coverage changed")
        if value["balancedAccuracy"] + 1e-15 < gates["minimumBalancedAccuracyPerVariant"]:
            return False
        if value["realRecall"] + 1e-15 < gates["minimumRealRecallPerVariant"]:
            return False
        if value["syntheticRecall"] + 1e-15 < gates["minimumSyntheticRecallPerVariant"]:
            return False
        if min(value["syntheticRecallBySource"].values()) + 1e-15 < gates["minimumSyntheticRecallPerFamily"]:
            return False
        for source, minimum in gates["minimumRealRecallBySource"].items():
            if value["realRecallBySource"][source] + 1e-15 < minimum:
                return False
    return True


def validate_regression_state(
    state: Mapping[str, Any],
    recipe: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    *,
    lock_commit: str,
    selection_lock_sha256: str | None = None,
) -> None:
    _require_keys(
        state,
        {
            "schemaVersion", "status", "lockCommit", "selectionLockSha256",
            "selectedCandidateId", "selectedModelSha256", "rawThreshold",
            "selectorOnnxReplay", "results", "selectionInfluencedByRegression", "h3PixelsRead",
        },
        "terminal regression state",
    )
    expected_lock_sha = selection_lock_sha256 or digest_file(ROOT / recipe["output"]["selectionLock"])
    if (
        state["schemaVersion"] != 1
        or state["status"] != "regression-pass"
        or state["lockCommit"] != lock_commit
        or state["selectionLockSha256"] != expected_lock_sha
        or state["selectedCandidateId"] != selection_lock["selectedCandidateId"]
        or state["selectedModelSha256"] != selection_lock["selectedModel"]["sha256"]
        or float(state["rawThreshold"]) != float(selection_lock["rawThreshold"])
        or state["selectionInfluencedByRegression"] is not False
        or state["h3PixelsRead"] is not False
    ):
        raise ValueError("M5 terminal regression binding changed")
    replay = state["selectorOnnxReplay"]
    _require_keys(
        replay,
        {
            "selectorManifestSha256", "items", "maximumAbsoluteLogitDeltaByVariant",
            "parityTolerance", "lockedThreshold", "replayedBestThreshold",
            "metricsAtLockedThreshold", "passed",
        },
        "selector ONNX replay",
    )
    tolerance = float(recipe["initialModel"]["maximumPytorchOnnxParityError"])
    deltas = replay["maximumAbsoluteLogitDeltaByVariant"]
    if (
        replay["selectorManifestSha256"] != recipe["sourceEvidence"]["selectorManifest"]["sha256"]
        or replay["items"] != recipe["sourceEvidence"]["selectorManifest"]["items"]
        or tuple(deltas) != VARIANTS
        or any(not math.isfinite(float(value)) or float(value) < 0 or float(value) > tolerance for value in deltas.values())
        or float(replay["parityTolerance"]) != tolerance
        or float(replay["lockedThreshold"]) != float(selection_lock["rawThreshold"])
        or abs(float(replay["replayedBestThreshold"]) - float(selection_lock["rawThreshold"])) > tolerance
        or replay["metricsAtLockedThreshold"] != selection_lock["selectorMetrics"]
        or replay["passed"] is not True
    ):
        raise ValueError("M5 selector ONNX replay binding changed")
    results = state["results"]
    regressions = recipe["terminalRegressions"]
    if not isinstance(results, list) or len(results) != len(regressions):
        raise ValueError("M5 terminal regression coverage changed")
    for result, regression in zip(results, regressions, strict=True):
        _require_keys(result, {"name", "manifestSha256", "items", "metrics", "logits", "gates", "passed"}, "terminal regression result")
        manifest_path = ROOT / regression["manifest"]
        if digest_file(manifest_path) != regression["sha256"]:
            raise ValueError(f"M5 frozen regression manifest changed: {regression['name']}")
        rows = read_jsonl(manifest_path)
        if (
            result["name"] != regression["name"]
            or result["manifestSha256"] != regression["sha256"]
            or result["items"] != regression["items"]
            or len(rows) != regression["items"]
            or result["gates"] != regression["gates"]
        ):
            raise ValueError(f"M5 terminal regression receipt changed: {regression['name']}")
        if tuple(result["logits"]) != VARIANTS:
            raise ValueError(f"M5 terminal regression logit variants changed: {regression['name']}")
        recomputed = {
            variant: regression_metrics(
                unpack_float32(result["logits"][variant], expected_count=len(rows)),
                rows,
                float(selection_lock["rawThreshold"]),
            )
            for variant in VARIANTS
        }
        passed = regression_gates_pass(recomputed, regression["gates"])
        if result["metrics"] != recomputed or result["passed"] is not passed or not passed:
            raise ValueError(f"M5 terminal regression metrics or gates failed: {regression['name']}")


def _metrics_dict(metrics: Mapping[str, VariantMetrics]) -> dict[str, Any]:
    return {
        variant: {
            "balancedAccuracy": value.balanced_accuracy,
            "realRecall": value.real_recall,
            "syntheticRecall": value.synthetic_recall,
            "syntheticRecallBySource": value.synthetic_recall_by_source,
            "falsePositives": value.false_positives,
            "falsePositiveTrials": value.false_positive_trials,
            "falsePositiveRate": value.false_positive_rate,
            "falsePositiveWilson95": value.false_positive_wilson95,
        }
        for variant, value in metrics.items()
    }


def validate_training_summary(
    summary: Mapping[str, Any],
    recipe: Mapping[str, Any],
    *,
    protocol_commit: str,
    candidate_grid_sha256: str,
) -> None:
    _require_keys(
        summary,
        {
            "schemaVersion", "status", "recipeSha256", "protocolCommit", "environment",
            "upstreamSourceSha256", "initialPytorchOnnxParityMaximumAbsoluteError",
            "trainingManifestSha256", "selectorManifestSha256", "trainingItems", "selectorItems",
            "epochReceipts", "candidateGrid", "selectedCandidateId", "h3PixelsRead", "terminalRegressionsRead",
        },
        "training summary",
    )
    recipe_sha256 = digest_bytes((ROOT / "benchmark/m5/recipe.json").read_bytes())
    expected_upstream = {
        recipe["upstream"][key]["path"]: recipe["upstream"][key]["sha256"]
        for key in ("config", "preprocessor", "pytorchWeights")
    }
    expected_grid_path = f"{recipe['output']['candidateRoot']}/candidate-grid.json"
    if (
        summary["schemaVersion"] != 1
        or summary["status"] not in {"selector-pass", "selector-fail"}
        or summary["recipeSha256"] != recipe_sha256
        or summary["protocolCommit"] != protocol_commit
        or summary["upstreamSourceSha256"] != expected_upstream
        or not math.isfinite(float(summary["initialPytorchOnnxParityMaximumAbsoluteError"]))
        or float(summary["initialPytorchOnnxParityMaximumAbsoluteError"]) < 0
        or float(summary["initialPytorchOnnxParityMaximumAbsoluteError"]) > recipe["initialModel"]["maximumPytorchOnnxParityError"]
        or summary["trainingManifestSha256"] != recipe["sourceEvidence"]["trainingManifest"]["compressedSha256"]
        or summary["selectorManifestSha256"] != recipe["sourceEvidence"]["selectorManifest"]["sha256"]
        or summary["trainingItems"] != recipe["sourceEvidence"]["trainingManifest"]["items"]
        or summary["selectorItems"] != recipe["sourceEvidence"]["selectorManifest"]["items"]
        or summary["candidateGrid"] != {"path": expected_grid_path, "sha256": candidate_grid_sha256}
        or (summary["selectedCandidateId"] is None) is not (summary["status"] == "selector-fail")
        or summary["h3PixelsRead"] is not False
        or summary["terminalRegressionsRead"] is not False
    ):
        raise ValueError("M5 training summary binding changed")
    validate_environment_receipt(summary["environment"], recipe)
    environment = summary["environment"]
    if environment["sourceCommit"] != protocol_commit:
        raise ValueError("M5 training summary authorization binding changed")
    receipts = summary["epochReceipts"]
    expected_epochs = [
        (branch["name"], epoch)
        for branch in recipe["training"]["branches"]
        for epoch in range(1, int(recipe["training"]["epochs"]) + 1)
    ]
    if not isinstance(receipts, list) or len(receipts) != len(expected_epochs):
        raise ValueError("M5 training epoch receipt coverage changed")
    last_step_by_branch: dict[str, int] = {}
    for receipt, (branch, epoch) in zip(receipts, expected_epochs, strict=True):
        _require_keys(
            receipt,
            {
                "branch", "epoch", "globalStep", "seconds", "images", "meanWeightedBce",
                "meanMaskedTeacherMse", "learningRates",
            },
            "training epoch receipt",
        )
        numeric = (receipt["seconds"], receipt["meanWeightedBce"], receipt["meanMaskedTeacherMse"])
        if (
            receipt["branch"] != branch
            or receipt["epoch"] != epoch
            or not isinstance(receipt["globalStep"], int)
            or receipt["globalStep"] <= last_step_by_branch.get(branch, -1)
            or receipt["images"] != recipe["sourceEvidence"]["trainingManifest"]["items"]
            or any(not math.isfinite(float(value)) or float(value) < 0 for value in numeric)
            or not isinstance(receipt["learningRates"], dict)
            or not receipt["learningRates"]
            or any(not math.isfinite(float(value)) or float(value) < 0 for value in receipt["learningRates"].values())
        ):
            raise ValueError("M5 training epoch receipt changed")
        last_step_by_branch[branch] = receipt["globalStep"]


def validate_selection_lock(
    lock: Mapping[str, Any],
    recipe: Mapping[str, Any],
    selector_rows: Sequence[dict[str, Any]],
) -> None:
    _require_keys(
        lock,
        {
            "schemaVersion", "status", "acceptanceEligible", "recipeSha256", "protocolCommit",
            "trainingSummary", "trainingSummarySha256", "candidateGrid", "candidateGridSha256",
            "selectedCandidateId", "selectedModel", "rawThreshold", "calibration", "selectorMetrics",
            "selectionKey", "selectionInfluencedByRegression", "terminalRegressionsRead", "h3PixelsRead",
        },
        "selection lock",
    )
    if (
        lock["schemaVersion"] != 1
        or lock["status"] != "m5-selected-pre-regression"
        or lock["acceptanceEligible"] is not False
        or lock["selectionInfluencedByRegression"] is not False
        or lock["terminalRegressionsRead"] is not False
        or lock["h3PixelsRead"] is not False
    ):
        raise ValueError("M5 selection-lock boundary changed")
    if not HEX40.fullmatch(str(lock["protocolCommit"])):
        raise ValueError("M5 protocol commit is invalid")
    if lock["recipeSha256"] != digest_bytes((ROOT / "benchmark/m5/recipe.json").read_bytes()):
        raise ValueError("M5 selection lock recipe binding changed")
    if lock["trainingSummarySha256"] != digest_bytes(canonical_json(lock["trainingSummary"])):
        raise ValueError("M5 embedded training summary changed")
    grid = lock["candidateGrid"]
    if lock["candidateGridSha256"] != digest_bytes(canonical_json(grid)):
        raise ValueError("M5 embedded candidate grid changed")
    validate_training_summary(
        lock["trainingSummary"],
        recipe,
        protocol_commit=str(lock["protocolCommit"]),
        candidate_grid_sha256=str(lock["candidateGridSha256"]),
    )
    _require_keys(
        grid,
        {"schemaVersion", "recipeSha256", "protocolCommit", "candidates", "selectorManifestSha256", "h3PixelsRead"},
        "candidate grid",
    )
    if (
        grid["schemaVersion"] != 1
        or grid["recipeSha256"] != lock["recipeSha256"]
        or grid["protocolCommit"] != lock["protocolCommit"]
        or grid["selectorManifestSha256"] != recipe["sourceEvidence"]["selectorManifest"]["sha256"]
        or grid["h3PixelsRead"] is not False
    ):
        raise ValueError("M5 candidate-grid binding changed")
    candidates = grid["candidates"]
    expected_ids = branch_candidate_ids(recipe)
    if not isinstance(candidates, list) or [candidate.get("candidateId") for candidate in candidates] != expected_ids:
        raise ValueError("M5 candidate grid coverage changed")
    winner: tuple[tuple[float, ...], Mapping[str, Any]] | None = None
    for branch_order, candidate in enumerate(candidates):
        common_keys = {"candidateId", "checkpoint", "model", "selectorLogits", "accepted"}
        accepted_keys = common_keys | {"rawThreshold", "metrics", "selectionKey"}
        _require_keys(candidate, accepted_keys if candidate.get("accepted") is True else common_keys, "candidate")
        model = candidate["model"]
        _require_keys(model, {"path", "bytes", "sha256", "parityMaximumAbsoluteError", "parityProvider", "parityProviderOptions"}, "candidate model")
        if (
            not isinstance(model["bytes"], int)
            or model["bytes"] <= 0
            or model["bytes"] > recipe["deliverable"]["maximumBytes"]
            or not HEX64.fullmatch(str(model["sha256"]))
            or model["parityProvider"] != "CUDAExecutionProvider"
            or model["parityProviderOptions"] != {"use_tf32": "0"}
            or float(model["parityMaximumAbsoluteError"]) > recipe["initialModel"]["maximumPytorchOnnxParityError"]
        ):
            raise ValueError("M5 candidate model receipt is invalid")
        logits = {
            variant: unpack_float32(candidate["selectorLogits"][variant], expected_count=len(selector_rows))
            for variant in VARIANTS
        }
        if tuple(candidate["selectorLogits"]) != VARIANTS:
            raise ValueError("M5 candidate logit variants changed")
        selected = choose_selector_threshold(logits, selector_rows, recipe["selection"]["gates"])
        if (selected is not None) is not candidate["accepted"]:
            raise ValueError("M5 candidate acceptance changed")
        if selected is None:
            continue
        threshold, metrics, key = selected
        candidate_id = str(candidate["candidateId"])
        _branch_name, _separator, epoch_text = candidate_id.rpartition("-epoch-")
        ranking = (*key[:-1], -branch_order, -int(epoch_text), key[-1])
        if float(candidate["rawThreshold"]) != threshold:
            raise ValueError("M5 candidate raw threshold changed")
        if candidate["metrics"] != _metrics_dict(metrics) or list(ranking) != candidate["selectionKey"]:
            raise ValueError("M5 candidate metrics or ranking changed")
        if winner is None or ranking > winner[0]:
            winner = (ranking, candidate)
    if winner is None:
        raise ValueError("M5 selection lock has no accepted candidate")
    selected = winner[1]
    if lock["selectedCandidateId"] != selected["candidateId"] or lock["selectedModel"] != selected["model"]:
        raise ValueError("M5 selected candidate changed")
    if lock["trainingSummary"]["selectedCandidateId"] != selected["candidateId"]:
        raise ValueError("M5 training summary selected candidate changed")
    if float(lock["rawThreshold"]) != float(selected["rawThreshold"]):
        raise ValueError("M5 locked threshold changed")
    if lock["selectorMetrics"] != selected["metrics"] or lock["selectionKey"] != selected["selectionKey"]:
        raise ValueError("M5 locked selector evidence changed")
    calibration = lock["calibration"]
    if set(calibration) != {"slope", "intercept", "displayThreshold"} or calibration["slope"] != 1.0 or calibration["displayThreshold"] != 0.65:
        raise ValueError("M5 calibration contract changed")
    expected_intercept = math.log(0.65 / 0.35) - float(lock["rawThreshold"])
    if not math.isclose(float(calibration["intercept"]), expected_intercept, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("M5 calibration does not map the raw threshold to 0.65")


def validate_failure_receipt(
    receipt: Mapping[str, Any],
    recipe: Mapping[str, Any],
    selector_rows: Sequence[dict[str, Any]],
) -> None:
    _require_keys(
        receipt,
        {
            "schemaVersion", "status", "acceptanceEligible", "recipeSha256", "protocolCommit",
            "trainingSummary", "trainingSummarySha256", "candidateGrid", "candidateGridSha256",
            "h3PixelsRead", "terminalRegressionsRead", "reason",
        },
        "failure receipt",
    )
    if (
        receipt["schemaVersion"] != 1
        or receipt["status"] != "failed-m5-selector"
        or receipt["acceptanceEligible"] is not False
        or receipt["h3PixelsRead"] is not False
        or receipt["terminalRegressionsRead"] is not False
        or not HEX40.fullmatch(str(receipt["protocolCommit"]))
    ):
        raise ValueError("M5 failure boundary changed")
    recipe_sha = digest_bytes((ROOT / "benchmark/m5/recipe.json").read_bytes())
    if receipt["recipeSha256"] != recipe_sha:
        raise ValueError("M5 failure recipe binding changed")
    if receipt["trainingSummarySha256"] != digest_bytes(canonical_json(receipt["trainingSummary"])):
        raise ValueError("M5 failure training summary changed")
    if receipt["candidateGridSha256"] != digest_bytes(canonical_json(receipt["candidateGrid"])):
        raise ValueError("M5 failure candidate grid changed")
    validate_training_summary(
        receipt["trainingSummary"],
        recipe,
        protocol_commit=str(receipt["protocolCommit"]),
        candidate_grid_sha256=str(receipt["candidateGridSha256"]),
    )
    summary = receipt["trainingSummary"]
    if (
        summary.get("status") != "selector-fail"
        or summary.get("protocolCommit") != receipt["protocolCommit"]
        or summary.get("selectedCandidateId") is not None
        or summary.get("h3PixelsRead") is not False
        or summary.get("terminalRegressionsRead") is not False
    ):
        raise ValueError("M5 failure summary boundary changed")
    grid = receipt["candidateGrid"]
    _require_keys(
        grid,
        {"schemaVersion", "recipeSha256", "protocolCommit", "candidates", "selectorManifestSha256", "h3PixelsRead"},
        "failure candidate grid",
    )
    if (
        grid["schemaVersion"] != 1
        or grid["recipeSha256"] != recipe_sha
        or grid["protocolCommit"] != receipt["protocolCommit"]
        or grid["selectorManifestSha256"] != recipe["sourceEvidence"]["selectorManifest"]["sha256"]
        or grid["h3PixelsRead"] is not False
    ):
        raise ValueError("M5 failure candidate-grid binding changed")
    candidates = grid["candidates"]
    if [candidate.get("candidateId") for candidate in candidates] != branch_candidate_ids(recipe):
        raise ValueError("M5 failure candidate coverage changed")
    for candidate in candidates:
        _require_keys(candidate, {"candidateId", "checkpoint", "model", "selectorLogits", "accepted"}, "failed candidate")
        if candidate["accepted"] is not False or tuple(candidate["selectorLogits"]) != VARIANTS:
            raise ValueError("M5 failure contains an accepted or malformed candidate")
        logits = {
            variant: unpack_float32(candidate["selectorLogits"][variant], expected_count=len(selector_rows))
            for variant in VARIANTS
        }
        if choose_selector_threshold(logits, selector_rows, recipe["selection"]["gates"]) is not None:
            raise ValueError("M5 failure candidate actually passes the selector")
