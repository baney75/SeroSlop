"""Train a modern, transformation-robust linear head on corrected CF ViT features.

The transformer backbone remains frozen. Training reads only the public training
manifest; model and threshold selection read only the generator-family-held-out
validation manifest. The sealed test manifest is deliberately not accepted as
an argument by this program.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, __version__ as PILLOW_VERSION
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feature_cache_contract import expected_view_metadata, validate_feature_arrays
from fresh_feature_run import (
    cache_belongs_to_fresh_feature_run,
    complete_fresh_feature_run,
    marker_sha256,
    open_or_create_fresh_feature_run,
)
from thresholds import complete_decision_thresholds


SEED = 20260813
PIPELINE_VERSION = 9
INPUT_SIZE = 384
RESIZE_SHORT_EDGE = 440
MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")
FEATURE_OUTPUT = "/Gather_output_0"
DISPLAY_THRESHOLD = 0.65
TRAINING_EPOCHS = 12
TRAINING_BATCH_SIZE = 2_048
LEGACY_FEATURE_CACHE_TRAINER_SHA256 = "00f3bdea0ae58166d2b1708eeb4582562631d18b6a60f43c2273de1bd68e1377"
FEATURE_EXTRACTOR_CONTRACT = "cf384-static-batch24-preprocess-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Item:
    id: str
    path: Path
    image_sha256: str
    label: int
    source: str


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_manifest(path: Path, data_root: Path) -> list[Item]:
    items: list[Item] = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        items.append(
            Item(
                id=str(row["id"]),
                path=data_root / str(row["path"]),
                image_sha256=str(row["imageSha256"]),
                label=int(row["label"]),
                source=str(row["source"]),
            )
        )
    if not items:
        raise ValueError(f"Manifest is empty: {path}")
    if len({item.id for item in items}) != len(items):
        raise ValueError(f"Manifest contains duplicate IDs: {path}")
    if len({item.image_sha256 for item in items}) != len(items):
        raise ValueError(f"Manifest contains duplicate image bytes: {path}")
    return items


def lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def require_repository_path(requested: Path, expected_relative: str, *, label: str) -> Path:
    expected = lexical_absolute(REPOSITORY_ROOT / expected_relative)
    actual = lexical_absolute(requested)
    if actual != expected:
        raise ValueError(f"{label} path does not match the frozen recipe")
    try:
        relative = expected.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    current = REPOSITORY_ROOT
    if current.is_symlink():
        raise ValueError(f"{label} repository root is symlinked")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} traverses a symlink")
    return expected


def require_partition_contract(
    recipe: dict[str, object],
    *,
    selector_manifest: Path,
    selector_data_root: Path,
    selector_items: list[Item],
    regression_manifest: Path,
    regression_data_root: Path,
    regression_items: list[Item],
) -> dict[str, object]:
    selector_configs = list(recipe.get("evaluationManifests", []))
    regression_configs = list(recipe.get("regressionManifests", []))
    if len(selector_configs) != 1 or len(regression_configs) != 1:
        raise ValueError("M3 requires exactly one selector and one regression manifest")
    selector = dict(selector_configs[0])
    regression = dict(regression_configs[0])
    selector_path = require_repository_path(
        selector_manifest,
        str(selector["path"]),
        label="selector manifest",
    )
    require_repository_path(
        selector_data_root,
        str(selector["dataRoot"]),
        label="selector data root",
    )
    regression_path = require_repository_path(
        regression_manifest,
        str(regression["path"]),
        label="regression manifest",
    )
    require_repository_path(
        regression_data_root,
        str(regression["dataRoot"]),
        label="regression data root",
    )
    selector_hash = digest(selector_path)
    regression_hash = digest(regression_path)
    if selector_hash != selector.get("sha256"):
        raise ValueError("Selector manifest hash does not match the frozen recipe")
    if regression_hash != regression.get("sha256"):
        raise ValueError("Regression manifest hash does not match the frozen recipe")
    if len(selector_items) != int(selector["items"]) or len(selector_items) != int(recipe["expectedValidationCount"]):
        raise ValueError("Selector manifest count does not match the frozen recipe")
    if len(regression_items) != int(regression["items"]) or len(regression_items) != int(recipe["expectedRegressionCount"]):
        raise ValueError("Regression manifest count does not match the frozen recipe")
    if int(selector["featureViews"]) != int(recipe["expectedValidationFeatureViews"]):
        raise ValueError("Selector feature-view contract changed")
    if int(regression["featureViews"]) != int(recipe["expectedRegressionFeatureViews"]):
        raise ValueError("Regression feature-view contract changed")
    detailed = dict(recipe.get("regressionValidation", {}))
    expected_detailed = {
        "manifest": regression["path"],
        "sha256": regression["sha256"],
        "items": regression["items"],
        "featureViews": regression["featureViews"],
        "dataRoot": regression["dataRoot"],
        "role": regression["role"],
    }
    if detailed != expected_detailed:
        raise ValueError("Detailed regression contract disagrees with the frozen manifest entry")
    return {
        "selectorManifestSha256": selector_hash,
        "selectorRole": selector["role"],
        "regressionManifestSha256": regression_hash,
        "regressionRole": regression["role"],
    }


def validate_large_training_packet(
    recipe_path: Path,
    recipe: dict[str, object],
    selection_summary_path: Path,
    train_manifest: Path,
    train_items: list[Item],
) -> tuple[dict[str, object], str]:
    summary = json.loads(selection_summary_path.read_text())
    summary_hash = digest(selection_summary_path)
    if summary.get("schemaVersion") not in {1, 2, 3}:
        raise ValueError("Unsupported large-corpus selection summary")
    if summary.get("recipeSha256") != digest(recipe_path):
        raise ValueError("Selection summary does not target the training recipe")
    if summary.get("manifestSha256") != digest(train_manifest):
        raise ValueError("Selection summary does not bind the training manifest")
    expected_total = int(recipe["expectedTotalCount"])
    if len(train_items) != expected_total or summary.get("counts", {}).get("total") != expected_total:
        raise ValueError("Large-corpus total count does not match recipe and summary")
    source_counts = {
        source: sum(item.source == source for item in train_items)
        for source in sorted({item.source for item in train_items})
    }
    if source_counts != summary.get("sourceCounts"):
        raise ValueError("Training manifest source counts do not match the selection summary")
    expected_sources = recipe.get("expectedSourceCounts")
    if expected_sources is not None:
        if source_counts != expected_sources:
            raise ValueError("Training manifest source counts do not match the generic recipe")
    else:
        if source_counts.get("diffusiondb-stable-diffusion") != int(recipe["diffusionDb"]["targetCount"]):
            raise ValueError("DiffusionDB training count does not match the recipe")
        if source_counts.get("open-images-train") != int(recipe["openImages"]["targetCount"]):
            raise ValueError("Open Images training count does not match the recipe")
        if sum(count for source, count in source_counts.items() if source not in {
            "diffusiondb-stable-diffusion", "open-images-train"
        }) != int(recipe["expectedModernTrainingCount"]):
            raise ValueError("Modern training source counts do not match the recipe")
    class_counts = {
        "real": sum(item.label == 0 for item in train_items),
        "synthetic": sum(item.label == 1 for item in train_items),
    }
    if class_counts != summary.get("classCounts"):
        raise ValueError("Training manifest class counts do not match the selection summary")
    if recipe.get("expectedClassCounts") is not None and class_counts != recipe["expectedClassCounts"]:
        raise ValueError("Training manifest class counts do not match the generic recipe")
    train_ids = {item.id for item in train_items}
    train_hashes = {item.image_sha256 for item in train_items}
    exclusions: list[dict[str, object]] = []
    exclusion_manifests = [
        *recipe["evaluationManifests"],
        *recipe.get("additionalTrainingExclusionManifests", []),
    ]
    for manifest in exclusion_manifests:
        path = Path(str(manifest["path"]))
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if train_ids.intersection(str(row["id"]) for row in rows):
            raise ValueError("Training IDs overlap a frozen evaluation manifest")
        if train_hashes.intersection(str(row["imageSha256"]) for row in rows):
            raise ValueError("Training image bytes overlap a frozen evaluation manifest")
        exclusions.append(
            {
                "path": str(manifest["path"]),
                "sha256": digest(path),
                "rows": len(rows),
                "dataRoot": str(manifest["dataRoot"]),
                "role": str(manifest["role"])
                if manifest.get("role") is not None
                else (
                    "validation-or-confirmatory-test"
                    if manifest in recipe["evaluationManifests"]
                    else "web-negative-training-exclusion"
                ),
            }
        )
    if exclusions != summary.get("evaluationExclusions"):
        raise ValueError("Selection summary does not bind the frozen evaluation manifests")
    review_path = Path(str(recipe["perceptualOverlapReview"]))
    review = json.loads(review_path.read_text())
    expected_review_evidence = {
        "path": str(recipe["perceptualOverlapReview"]),
        "sha256": digest(review_path),
        "reviewedPairCount": len(review.get("items", [])),
        "hammingThreshold": int(recipe["perceptualDuplicateHammingThreshold"]),
    }
    if summary.get("perceptualOverlapReview") != expected_review_evidence:
        raise ValueError("Selection summary does not bind the perceptual-overlap review")
    training_review_path = Path(str(recipe["trainingPerceptualOverlapReview"]))
    training_review = json.loads(training_review_path.read_text())
    expected_training_review_evidence = {
        "path": str(recipe["trainingPerceptualOverlapReview"]),
        "sha256": digest(training_review_path),
        "reviewedPairCount": len(training_review.get("items", [])),
        "hammingThreshold": int(recipe["perceptualDuplicateHammingThreshold"]),
    }
    if summary.get("trainingPerceptualOverlapReview") != expected_training_review_evidence:
        raise ValueError("Selection summary does not bind the training/evaluation perceptual review")
    overlap = summary.get("overlapWithEvaluation", {})
    if (
        overlap.get("ids") != 0
        or overlap.get("imageHashes") != 0
        or overlap.get("unreviewedPerceptualDhashPairsAtOrBelowThreshold") != 0
        or overlap.get("reviewedVisuallyDistinctDhashPairsAtOrBelowThreshold")
        != len(training_review.get("items", []))
    ):
        raise ValueError("Selection summary reports training/evaluation overlap")
    return summary, summary_hash


def verify_item_files(items: list[Item]) -> None:
    def verify(item: Item) -> None:
        if digest(item.path) != item.image_sha256:
            raise ValueError(f"Image integrity mismatch while reusing cached features: {item.id}")

    with ThreadPoolExecutor(max_workers=min(8, len(items))) as executor:
        list(executor.map(verify, items))


def seeded_random(item: Item, purpose: str) -> random.Random:
    value = int(sha256(f"{SEED}:{item.id}:{purpose}".encode()).hexdigest()[:16], 16)
    return random.Random(value)


def feature_configuration_hash(
    *,
    training: bool,
    single_view_sources: frozenset[str],
    providers: tuple[str, ...],
    feature_batch_size: int,
) -> str:
    """Bind a feature cache to every input-affecting extraction choice."""
    configuration = {
        "pipelineVersion": PIPELINE_VERSION,
        "featureExtractorContract": FEATURE_EXTRACTOR_CONTRACT,
        "inputSize": INPUT_SIZE,
        "resizeShortEdge": RESIZE_SHORT_EDGE,
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "variants": list(VARIANTS),
        "training": training,
        "singleViewSources": sorted(single_view_sources),
        "onnxRuntime": ort.__version__,
        "pillow": PILLOW_VERSION,
        "providers": list(providers),
        "featureBatchSize": feature_batch_size,
    }
    encoded = json.dumps(configuration, separators=(",", ":"), sort_keys=True).encode()
    return sha256(encoded).hexdigest()


def legacy_feature_configuration_hash(
    *,
    training: bool,
    single_view_sources: frozenset[str],
    providers: tuple[str, ...],
    feature_batch_size: int,
) -> str:
    """Recognize the exact pre-threshold-fix cache after source-pixel revalidation."""
    configuration = {
        "pipelineVersion": PIPELINE_VERSION,
        "trainerSha256": LEGACY_FEATURE_CACHE_TRAINER_SHA256,
        "commandArguments": sys.argv[1:],
        "inputSize": INPUT_SIZE,
        "resizeShortEdge": RESIZE_SHORT_EDGE,
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "variants": list(VARIANTS),
        "training": training,
        "singleViewSources": sorted(single_view_sources),
        "onnxRuntime": ort.__version__,
        "pillow": PILLOW_VERSION,
        "providers": list(providers),
        "featureBatchSize": feature_batch_size,
    }
    encoded = json.dumps(configuration, separators=(",", ":"), sort_keys=True).encode()
    return sha256(encoded).hexdigest()


def resize_long_edge(image: Image.Image, maximum: int) -> Image.Image:
    if max(image.size) <= maximum:
        return image
    scale = maximum / max(image.size)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def transform(image: Image.Image, item: Item, variant: str, training: bool) -> Image.Image:
    rng = seeded_random(item, variant)
    if variant == "original":
        transformed = image
    elif variant == "screenshot":
        frame = Image.new("RGB", (1170, 1400), (238, 241, 244))
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle(
            (46, 46, 1124, 1354),
            radius=24,
            fill=(255, 255, 255),
            outline=(217, 222, 229),
            width=2,
        )
        draw.ellipse((74, 69, 122, 117), fill=(72, 132, 220))
        draw.rounded_rectangle((138, 84, 318, 99), radius=8, fill=(200, 206, 214))
        left, top, width, height = 47, 140, 1076, 1110
        draw.rectangle((left, top, left + width - 1, top + height - 1), fill=(17, 21, 26))
        scale = min(width / image.width, height / image.height)
        if training:
            scale *= rng.uniform(0.94, 1.0)
        rendered = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        frame.paste(rendered, (left + (width - rendered.width) // 2, top + (height - rendered.height) // 2))
        for x in (78, 124, 170):
            draw.ellipse((x, 1291, x + 22, 1313), outline=(136, 145, 155), width=3)
        transformed = frame
    elif variant == "social-q75":
        edge = rng.randint(800, 1280) if training else 1080
        quality = rng.randint(60, 88) if training else 75
        transformed = resize_long_edge(image, edge)
        if training and rng.random() < 0.5:
            transformed = transformed.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.65)))
        transformed = jpeg_roundtrip(transformed, quality)
    elif variant == "social-heavy":
        first_edge = rng.randint(600, 900) if training else 720
        second_edge = rng.randint(480, 720) if training else 640
        first_quality = rng.randint(38, 62) if training else 50
        second_quality = rng.randint(28, 48) if training else 38
        transformed = jpeg_roundtrip(resize_long_edge(image, first_edge), first_quality)
        transformed = jpeg_roundtrip(resize_long_edge(transformed, second_edge), second_quality)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    if training:
        appearance = seeded_random(item, f"appearance:{variant}")
        if appearance.random() < 0.5:
            transformed = ImageOps.mirror(transformed)
        transformed = ImageEnhance.Color(transformed).enhance(appearance.uniform(0.90, 1.10))
        transformed = ImageEnhance.Contrast(transformed).enhance(appearance.uniform(0.92, 1.08))
    return transformed


def preprocess_image(image: Image.Image, item: Item, variant: str, training: bool) -> np.ndarray:
    image = transform(image, item, variant, training)
    scale = RESIZE_SHORT_EDGE / min(image.size)
    resized = image.resize(
        (max(INPUT_SIZE, round(image.width * scale)), max(INPUT_SIZE, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - INPUT_SIZE) // 2
    top = (resized.height - INPUT_SIZE) // 2
    pixels = np.asarray(resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE)), dtype=np.float32) / 255.0
    return np.transpose((pixels - MEAN) / STD, (2, 0, 1)).astype(np.float32)


def preprocess_views(item: Item, training: bool, variants: tuple[str, ...] = VARIANTS) -> list[np.ndarray]:
    if digest(item.path) != item.image_sha256:
        raise ValueError(f"Image integrity mismatch: {item.id}")
    with Image.open(item.path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    return [preprocess_image(image.copy(), item, variant, training) for variant in variants]


def make_feature_model(source: Path, destination: Path, batch_size: int) -> tuple[np.ndarray, float]:
    model = onnx.load(source)
    if not any(output.name == FEATURE_OUTPUT for output in model.graph.output):
        model.graph.output.append(helper.make_tensor_value_info(FEATURE_OUTPUT, TensorProto.FLOAT, ["batch_size", 384]))
    # A fixed feature-extraction batch lets ONNX Runtime fully optimize this
    # frozen graph on CPU. The final exported classifier remains dynamic.
    for value in [*model.graph.input, *model.graph.output]:
        shape = value.type.tensor_type.shape
        if shape.dim:
            shape.dim[0].ClearField("dim_param")
            shape.dim[0].dim_value = batch_size
    initializers = {value.name: value for value in model.graph.initializer}
    upstream_weight = numpy_helper.to_array(initializers["classifier.weight"]).astype(np.float32).reshape(-1)
    upstream_bias = float(numpy_helper.to_array(initializers["classifier.bias"]).reshape(-1)[0])
    onnx.save(model, destination)
    return upstream_weight, upstream_bias


def extract_features(
    session: ort.InferenceSession,
    items: list[Item],
    batch_size: int,
    training: bool,
    single_view_sources: frozenset[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    variants: list[int] = []
    sources: list[str] = []
    view_names = [
        ("original",) if training and item.source in single_view_sources else VARIANTS
        for item in items
    ]
    expanded_count = sum(len(names) for names in view_names)
    completed_views = np.cumsum([0, *(len(names) for names in view_names)])
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(8, batch_size)) as executor:
        offset = 0
        while offset < len(items):
            end = offset
            batch_views = 0
            while end < len(items) and batch_views + len(view_names[end]) <= batch_size:
                batch_views += len(view_names[end])
                end += 1
            if end == offset:
                end += 1
            batch_items = items[offset:end]
            batch_view_names = view_names[offset : offset + len(batch_items)]
            views = list(
                executor.map(
                    lambda pair: preprocess_views(pair[0], training, pair[1]),
                    zip(batch_items, batch_view_names, strict=True),
                )
            )
            tensor = np.stack([pixels for item_views in views for pixels in item_views])
            true_batch_size = tensor.shape[0]
            if true_batch_size < batch_size:
                padding = np.zeros((batch_size - true_batch_size, *tensor.shape[1:]), dtype=np.float32)
                tensor = np.concatenate((tensor, padding))
            output = np.asarray(session.run([FEATURE_OUTPUT], {"pixel_values": tensor})[0], dtype=np.float32)[
                :true_batch_size
            ]
            features.append(output)
            labels.extend(item.label for item, item_views in zip(batch_items, views, strict=True) for _ in item_views)
            variants.extend(
                VARIANTS.index(name)
                for names in batch_view_names
                for name in names
            )
            sources.extend(item.source for item, item_views in zip(batch_items, views, strict=True) for _ in item_views)
            completed = int(completed_views[end])
            if completed % (batch_size * 10) == 0 or completed == expanded_count:
                elapsed = time.perf_counter() - started
                print(f"features {completed}/{expanded_count} ({completed / max(elapsed, 1e-6):.1f}/s)", flush=True)
            offset = end
    return (
        np.concatenate(features),
        np.asarray(labels, dtype=np.float32),
        np.asarray(variants, dtype=np.int64),
        np.asarray(sources),
    )


def save_feature_shard(
    path: Path,
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    manifest_hash: str,
    model_hash: str,
    item_ids_hash: str,
    feature_configuration_hash: str,
    training: bool,
    fresh_feature_run_id: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=result[0],
            labels=result[1],
            variants=result[2],
            sources=result[3],
            manifest_hash=np.asarray(manifest_hash),
            model_hash=np.asarray(model_hash),
            item_ids_hash=np.asarray(item_ids_hash),
            feature_configuration_hash=np.asarray(feature_configuration_hash),
            fresh_feature_run_id=np.asarray(fresh_feature_run_id or ""),
            training=np.asarray(training),
            pipeline_version=np.asarray(PIPELINE_VERSION),
        )
    partial.replace(path)


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    fingerprint = sha256()
    fingerprint.update(array.dtype.str.encode())
    fingerprint.update(json.dumps(array.shape, separators=(",", ":")).encode())
    fingerprint.update(array.tobytes())
    return fingerprint.hexdigest()


def validate_feature_result(
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    items: list[Item],
    *,
    training: bool,
    single_view_sources: frozenset[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expected_labels, expected_variants, expected_sources = expected_view_metadata(
        items,
        training=training,
        single_view_sources=single_view_sources,
    )
    features, labels, variants, sources = (np.asarray(value) for value in result)
    validate_feature_arrays(
        features,
        labels,
        variants,
        sources,
        expected_labels=expected_labels,
        expected_variants=expected_variants,
        expected_sources=expected_sources,
    )
    return features, labels, variants, sources


def extract_or_load_sharded(
    session: ort.InferenceSession,
    items: list[Item],
    manifest: Path,
    cache_directory: Path,
    cache_name: str,
    model_hash: str,
    batch_size: int,
    training: bool,
    shard_images: int,
    single_view_sources: frozenset[str],
    reextract_cached_features: bool,
    fresh_feature_run_id: str | None,
    shard_evidence: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest_hash = digest(manifest)
    configuration_hash = feature_configuration_hash(
        training=training,
        single_view_sources=single_view_sources,
        providers=tuple(session.get_providers()),
        feature_batch_size=batch_size,
    )
    legacy_configuration_hash = legacy_feature_configuration_hash(
        training=training,
        single_view_sources=single_view_sources,
        providers=tuple(session.get_providers()),
        feature_batch_size=batch_size,
    )
    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for shard_index, offset in enumerate(range(0, len(items), shard_images)):
        shard_items = items[offset : offset + shard_images]
        item_ids_hash = sha256("\n".join(item.id for item in shard_items).encode()).hexdigest()
        cache = cache_directory / f"{cache_name}-{shard_index:05d}.npz"
        loaded: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
        freshly_extracted_this_process = False
        freshly_extracted_this_run = False
        replaced_cache_sha256: str | None = None
        if cache.exists():
            with np.load(cache, allow_pickle=False) as data:
                required = {
                    "features",
                    "labels",
                    "variants",
                    "sources",
                    "manifest_hash",
                    "model_hash",
                    "item_ids_hash",
                    "feature_configuration_hash",
                    "fresh_feature_run_id",
                    "training",
                    "pipeline_version",
                }
                cached_configuration_hash = str(data["feature_configuration_hash"].item()) if "feature_configuration_hash" in data.files else ""
                cached_fresh_feature_run_id = str(data["fresh_feature_run_id"].item()) if "fresh_feature_run_id" in data.files else ""
                belongs_to_fresh_run = cache_belongs_to_fresh_feature_run(
                    fresh_feature_run_id,
                    cached_fresh_feature_run_id,
                )
                may_load = not reextract_cached_features or belongs_to_fresh_run
                if reextract_cached_features and not belongs_to_fresh_run:
                    replaced_cache_sha256 = digest(cache)
                if may_load and required <= set(data.files) and (
                    str(data["manifest_hash"].item()) == manifest_hash
                    and str(data["model_hash"].item()) == model_hash
                    and str(data["item_ids_hash"].item()) == item_ids_hash
                    and cached_configuration_hash in {configuration_hash, legacy_configuration_hash}
                    and bool(data["training"].item()) == training
                    and int(data["pipeline_version"].item()) == PIPELINE_VERSION
                ):
                    verify_item_files(shard_items)
                    candidate = tuple(
                        np.asarray(data[name]).copy()
                        for name in ("features", "labels", "variants", "sources")
                    )  # type: ignore[assignment]
                    try:
                        loaded = validate_feature_result(
                            candidate,
                            shard_items,
                            training=training,
                            single_view_sources=single_view_sources,
                        )
                    except ValueError as error:
                        print(f"rejected {cache}: {error}", flush=True)
                        loaded = None
                    if loaded is not None:
                        if cached_configuration_hash == legacy_configuration_hash:
                            save_feature_shard(
                                cache,
                                loaded,
                                manifest_hash=manifest_hash,
                                model_hash=model_hash,
                                item_ids_hash=item_ids_hash,
                                feature_configuration_hash=configuration_hash,
                                training=training,
                                fresh_feature_run_id=cached_fresh_feature_run_id or None,
                            )
                            print(f"verified and migrated {cache}", flush=True)
                        else:
                            print(f"loaded {cache}", flush=True)
                        freshly_extracted_this_run = belongs_to_fresh_run
        if loaded is None:
            verify_item_files(shard_items)
            loaded = extract_features(session, shard_items, batch_size, training, single_view_sources)
            loaded = validate_feature_result(
                loaded,
                shard_items,
                training=training,
                single_view_sources=single_view_sources,
            )
            save_feature_shard(
                cache,
                loaded,
                manifest_hash=manifest_hash,
                model_hash=model_hash,
                item_ids_hash=item_ids_hash,
                feature_configuration_hash=configuration_hash,
                training=training,
                fresh_feature_run_id=fresh_feature_run_id,
            )
            freshly_extracted_this_process = True
            freshly_extracted_this_run = fresh_feature_run_id is not None
            print(f"{'refreshed' if replaced_cache_sha256 else 'saved'} {cache}", flush=True)
        shard_evidence.append(
            {
                "cache": str(cache),
                "cacheSha256": digest(cache),
                "replacedCacheSha256": replaced_cache_sha256,
                "freshFeatureRunId": fresh_feature_run_id,
                "freshlyExtractedThisRun": freshly_extracted_this_run,
                "freshlyExtractedThisProcess": freshly_extracted_this_process,
                "items": len(shard_items),
                "views": int(loaded[0].shape[0]),
                "itemIdsSha256": item_ids_hash,
                "featureConfigurationSha256": configuration_hash,
                "arraySha256": {
                    "features": array_digest(loaded[0]),
                    "labels": array_digest(loaded[1]),
                    "variants": array_digest(loaded[2]),
                    "sources": array_digest(loaded[3]),
                },
            }
        )
        outputs.append(loaded)
    return tuple(np.concatenate([output[index] for output in outputs]) for index in range(4))  # type: ignore[return-value]


def source_balanced_weights(labels: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Give each class half the loss and each source an equal share of its class."""
    weights = np.zeros(labels.shape[0], dtype=np.float32)
    for label in (0, 1):
        label_sources = sorted(set(sources[labels == label].tolist()))
        if not label_sources:
            raise ValueError(f"Training data has no rows for label {label}")
        for source in label_sources:
            selected = (labels == label) & (sources == source)
            count = int(selected.sum())
            if count == 0:
                raise AssertionError("source selection became empty")
            weights[selected] = 0.5 / (len(label_sources) * count)
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-6):
        raise AssertionError("source-balanced weights do not sum to one")
    return weights


def variant_metrics(logits: np.ndarray, labels: np.ndarray, variants: np.ndarray, sources: np.ndarray, threshold: float) -> dict[str, object]:
    output: dict[str, object] = {}
    for index, name in enumerate(VARIANTS):
        selected = variants == index
        y = labels[selected]
        z = logits[selected]
        s = sources[selected]
        real = y == 0
        synthetic = y == 1
        real_recall = float((z[real] < threshold).mean())
        synthetic_recall = float((z[synthetic] >= threshold).mean())
        source_recall = {
            source: float((z[(s == source) & synthetic] >= threshold).mean())
            for source in sorted(set(s[synthetic].tolist()))
        }
        real_source_recall = {
            source: float((z[(s == source) & real] < threshold).mean())
            for source in sorted(set(s[real].tolist()))
        }
        output[name] = {
            "balancedAccuracy": (real_recall + synthetic_recall) / 2,
            "realRecall": real_recall,
            "syntheticRecall": synthetic_recall,
            "syntheticRecallBySource": source_recall,
            "realRecallBySource": real_source_recall,
        }
    return output


def passes_validation_gates(values: dict[str, object], gates: dict[str, object] | None) -> bool:
    if gates is None:
        return True
    rows = list(values.values())
    family_recalls = [
        float(recall)
        for row in rows
        for recall in dict(row["syntheticRecallBySource"]).values()
    ]
    required_real_sources = dict(gates.get("minimumRealRecallBySource", {}))
    real_source_gate_passes = all(
        source in dict(row["realRecallBySource"])
        and float(dict(row["realRecallBySource"])[source]) >= float(minimum)
        for row in rows
        for source, minimum in required_real_sources.items()
    )
    return all(
        float(row["balancedAccuracy"]) >= float(gates["minimumBalancedAccuracyPerVariant"])
        and float(row["realRecall"]) >= float(gates["minimumRealRecallPerVariant"])
        and float(row["syntheticRecall"]) >= float(gates["minimumSyntheticRecallPerVariant"])
        for row in rows
    ) and min(family_recalls) >= float(gates["minimumSyntheticRecallPerFamily"]) and real_source_gate_passes


def evaluate_frozen_regression(
    weight: np.ndarray,
    bias: float,
    threshold: float,
    regression: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    gates: dict[str, object],
) -> dict[str, object]:
    """Evaluate one already-selected head without feeding regression results back into selection."""

    features, labels, variants, sources = regression
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logits = features @ weight + bias
    if not np.isfinite(logits).all():
        raise RuntimeError("Selected classifier produced non-finite post-selection regression logits")
    values = variant_metrics(logits, labels, variants, sources, threshold)
    if not passes_validation_gates(values, gates):
        raise RuntimeError("Selected candidate failed frozen post-selection regression gates")
    return values


def choose_threshold(
    logits: np.ndarray,
    labels: np.ndarray,
    variants: np.ndarray,
    sources: np.ndarray,
    gates: dict[str, object] | None,
) -> tuple[float, dict[str, object], tuple[float, ...]]:
    candidates = complete_decision_thresholds(logits.tolist())
    best: tuple[tuple[float, ...], float, dict[str, object]] | None = None
    for threshold in candidates:
        values = variant_metrics(logits, labels, variants, sources, float(threshold))
        if not passes_validation_gates(values, gates):
            continue
        rows = list(values.values())
        family_recalls = [
            float(recall)
            for row in rows
            for recall in dict(row["syntheticRecallBySource"]).values()
        ]
        base_key = (
            min(float(row["balancedAccuracy"]) for row in rows),
            sum(float(row["balancedAccuracy"]) for row in rows) / len(rows),
            min(float(row["realRecall"]) for row in rows),
        )
        required_real_sources = dict((gates or {}).get("minimumRealRecallBySource", {}))
        required_real_recalls = [
            float(dict(row["realRecallBySource"])[source])
            for row in rows
            for source in required_real_sources
        ]
        key = (
            *base_key,
            *([min(required_real_recalls)] if required_real_recalls else []),
            min(family_recalls),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), values)
    if best is None:
        raise RuntimeError("No threshold satisfies the frozen validation gates")
    if not math.isfinite(best[1]):
        raise RuntimeError("Selected validation threshold is not finite")
    return best[1], best[2], best[0]


def fit_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    train_sources: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_variants: np.ndarray,
    validation_sources: np.ndarray,
    upstream_weight: np.ndarray,
    upstream_bias: float,
    decay: float,
    alpha: float,
    device: torch.device,
    validation_gates: dict[str, object] | None,
) -> tuple[np.ndarray, float, float, dict[str, object], tuple[float, ...]]:
    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).clip(min=1e-5).astype(np.float32)
    x = torch.from_numpy((train_features - mean) / std).to(device)
    y = torch.from_numpy(train_labels).to(device).unsqueeze(1)
    example_weights = torch.from_numpy(source_balanced_weights(train_labels, train_sources)).to(device)
    head = nn.Linear(train_features.shape[1], 1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(upstream_weight * std).to(device).unsqueeze(0))
        head.bias.copy_(torch.tensor([upstream_bias + float(np.dot(upstream_weight, mean))], device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.012, weight_decay=decay)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(TRAINING_EPOCHS):
        order = torch.randperm(train_features.shape[0], generator=generator)
        for offset in range(0, order.numel(), TRAINING_BATCH_SIZE):
            indices = order[offset : offset + TRAINING_BATCH_SIZE].to(device)
            losses = nn.functional.binary_cross_entropy_with_logits(
                head(x[indices]), y[indices], reduction="none"
            ).squeeze(1)
            batch_weights = example_weights[indices]
            loss = torch.sum(losses * batch_weights) / torch.sum(batch_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    learned_normalized_weight = head.weight.detach().float().cpu().numpy()[0]
    learned_normalized_bias = float(head.bias.detach().float().cpu().item())
    learned_weight = learned_normalized_weight / std
    learned_bias = learned_normalized_bias - float(np.dot(learned_normalized_weight, mean / std))
    weight = upstream_weight * (1 - alpha) + learned_weight * alpha
    bias = upstream_bias * (1 - alpha) + learned_bias * alpha
    if not np.isfinite(weight).all() or not math.isfinite(bias):
        raise FloatingPointError("Candidate classifier parameters are non-finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        logits = validation_features @ weight + bias
    if not np.isfinite(logits).all():
        raise FloatingPointError("Candidate validation logits are non-finite")
    threshold, values, key = choose_threshold(
        logits,
        validation_labels,
        validation_variants,
        validation_sources,
        validation_gates,
    )
    return weight, bias, threshold, values, key


def replace_classifier(source: Path, destination: Path, weight: np.ndarray, bias: float) -> None:
    model = onnx.load(source)
    replacements = {
        "classifier.weight": numpy_helper.from_array(weight.astype(np.float32).reshape(1, -1), "classifier.weight"),
        "classifier.bias": numpy_helper.from_array(np.asarray([bias], dtype=np.float32), "classifier.bias"),
    }
    for index, value in enumerate(model.graph.initializer):
        if value.name in replacements:
            model.graph.initializer[index].CopyFrom(replacements[value.name])
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def exported_parity_error(
    session: ort.InferenceSession,
    items: list[Item],
    features: np.ndarray,
    weight: np.ndarray,
    bias: float,
    batch_size: int,
) -> float:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        expected = features @ weight + bias
    if not np.isfinite(expected).all():
        raise RuntimeError("Selected classifier produced non-finite reference logits")
    actual: list[np.ndarray] = []
    images_per_batch = max(1, batch_size // len(VARIANTS))
    with ThreadPoolExecutor(max_workers=min(8, images_per_batch)) as executor:
        for offset in range(0, len(items), images_per_batch):
            batch_items = items[offset : offset + images_per_batch]
            views = list(executor.map(lambda item: preprocess_views(item, False), batch_items))
            tensor = np.stack([pixels for item_views in views for pixels in item_views])
            actual.append(np.asarray(session.run(["logits"], {"pixel_values": tensor})[0]).reshape(-1))
    actual_values = np.concatenate(actual)
    if not np.isfinite(actual_values).all():
        raise RuntimeError("Exported classifier produced non-finite ONNX logits")
    error = float(np.max(np.abs(expected - actual_values)))
    if not math.isfinite(error) or error > 2e-4:
        raise RuntimeError(f"Export parity error too high: {error}")
    return error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--validation-data-root", type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--regression-data-root", type=Path)
    parser.add_argument("--regression-manifest", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--selection-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-shard-images", type=int, default=2_000)
    parser.add_argument("--single-view-source", action="append", default=[])
    parser.add_argument("--execution-provider", choices=("cuda", "coreml", "cpu"), default="cuda")
    parser.add_argument(
        "--reextract-cached-features",
        action="store_true",
        help=(
            "Reject caches from earlier runs, then recompute from source pixels. "
            "Interrupted invocations resume only shards carrying this exact run marker."
        ),
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    model_hash = digest(args.model)
    if model_hash != args.expected_model_sha256:
        raise ValueError(f"Unexpected model SHA-256: {model_hash}")
    if args.feature_shard_images < 1:
        raise ValueError("feature-shard-images must be positive")
    if args.batch_size < len(VARIANTS):
        raise ValueError(f"batch-size must be at least {len(VARIANTS)}")
    train_manifest = args.train_manifest or (args.data_root / "train-manifest.jsonl")
    validation_data_root = args.validation_data_root or args.data_root
    validation_manifest = args.validation_manifest or (validation_data_root / "validation-manifest.jsonl")
    recipe_config = json.loads(args.recipe.read_text()) if args.recipe else None
    if recipe_config is not None and recipe_config.get("regressionManifests"):
        if args.regression_manifest is None or args.regression_data_root is None:
            raise ValueError("Recipe-backed M3 training requires explicit regression manifest and data root")
        selector_configs = list(recipe_config.get("evaluationManifests", []))
        regression_configs = list(recipe_config.get("regressionManifests", []))
        if len(selector_configs) != 1 or len(regression_configs) != 1:
            raise ValueError("M3 requires exactly one selector and one regression manifest")
        selector_config = dict(selector_configs[0])
        regression_config = dict(regression_configs[0])
        require_repository_path(validation_manifest, str(selector_config["path"]), label="selector manifest")
        require_repository_path(validation_data_root, str(selector_config["dataRoot"]), label="selector data root")
        require_repository_path(args.regression_manifest, str(regression_config["path"]), label="regression manifest")
        require_repository_path(args.regression_data_root, str(regression_config["dataRoot"]), label="regression data root")
    train_items = load_manifest(train_manifest, args.data_root)
    validation_items = load_manifest(validation_manifest, validation_data_root)
    single_view_sources = frozenset(str(source) for source in args.single_view_source)
    missing_single_view_sources = single_view_sources - {item.source for item in train_items}
    if missing_single_view_sources:
        raise ValueError(f"single-view sources are absent from training data: {sorted(missing_single_view_sources)}")
    validation_gates = recipe_config.get("validationGates") if recipe_config else None
    selection_summary_hash: str | None = None
    regression_items: list[Item] | None = None
    regression_manifest: Path | None = None
    regression_data_root: Path | None = None
    regression_contract: dict[str, object] | None = None
    if recipe_config is not None:
        if int(recipe_config["expectedTotalCount"]) != len(train_items):
            raise ValueError("Training manifest count does not match recipe")
        if (
            recipe_config.get("expectedValidationCount") is not None
            and int(recipe_config["expectedValidationCount"]) != len(validation_items)
        ):
            raise ValueError("Validation manifest count does not match recipe")
        if frozenset(recipe_config["singleViewTrainingSources"]) != single_view_sources:
            raise ValueError("single-view-source arguments do not match recipe")
        selection_summary_path = args.selection_summary or (args.data_root / "selection-summary.json")
        _, selection_summary_hash = validate_large_training_packet(
            args.recipe,
            recipe_config,
            selection_summary_path,
            train_manifest,
            train_items,
        )
        if recipe_config.get("regressionManifests"):
            if args.regression_manifest is None or args.regression_data_root is None:
                raise ValueError("Recipe-backed M3 training requires explicit regression manifest and data root")
            regression_manifest = args.regression_manifest
            regression_data_root = args.regression_data_root
            regression_items = load_manifest(regression_manifest, regression_data_root)
            regression_contract = require_partition_contract(
                recipe_config,
                selector_manifest=validation_manifest,
                selector_data_root=validation_data_root,
                selector_items=validation_items,
                regression_manifest=regression_manifest,
                regression_data_root=regression_data_root,
                regression_items=regression_items,
            )
        elif args.regression_manifest is not None or args.regression_data_root is not None:
            raise ValueError("Regression arguments are not declared by this recipe")
    elif args.regression_manifest is not None or args.regression_data_root is not None:
        raise ValueError("Regression arguments require a recipe-backed contract")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_model = args.output_dir / "feature-model.onnx"
    upstream_weight, upstream_bias = make_feature_model(args.model, feature_model, args.batch_size)
    providers = {
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }[args.execution_provider]
    session = ort.InferenceSession(str(feature_model), providers=providers)
    if args.execution_provider == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDA provider unavailable: {session.get_providers()}")
    if args.execution_provider == "coreml" and "CoreMLExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CoreML provider unavailable: {session.get_providers()}")
    provider_tuple = tuple(session.get_providers())
    feature_configuration_hashes = {
        "training": feature_configuration_hash(
            training=True,
            single_view_sources=single_view_sources,
            providers=provider_tuple,
            feature_batch_size=args.batch_size,
        ),
        "validation": feature_configuration_hash(
            training=False,
            single_view_sources=frozenset(),
            providers=provider_tuple,
            feature_batch_size=args.batch_size,
        ),
    }
    if regression_items is not None:
        feature_configuration_hashes["regression"] = feature_configuration_hash(
            training=False,
            single_view_sources=frozenset(),
            providers=provider_tuple,
            feature_batch_size=args.batch_size,
        )
    fresh_feature_marker = args.output_dir / "fresh-feature-run.json"
    fresh_feature_context = {
        "pipelineVersion": PIPELINE_VERSION,
        "featureExtractorContract": FEATURE_EXTRACTOR_CONTRACT,
        "upstreamModelSha256": model_hash,
        "trainManifestSha256": digest(train_manifest),
        "validationManifestSha256": digest(validation_manifest),
        "regressionManifestSha256": digest(regression_manifest) if regression_manifest else None,
        "regressionDataRoot": str(regression_data_root) if regression_data_root else None,
        "regressionFeatureViews": int(recipe_config["expectedRegressionFeatureViews"])
        if recipe_config is not None and regression_items is not None
        else None,
        "selectionSummarySha256": selection_summary_hash,
        "featureBatchSize": args.batch_size,
        "featureShardImages": args.feature_shard_images,
        "singleViewTrainingSources": sorted(single_view_sources),
        "featureConfigurationHashes": feature_configuration_hashes,
    }
    fresh_feature_run = (
        open_or_create_fresh_feature_run(fresh_feature_marker, fresh_feature_context)
        if args.reextract_cached_features
        else None
    )
    fresh_feature_run_id = str(fresh_feature_run["runId"]) if fresh_feature_run else None
    feature_shard_evidence: list[dict[str, object]] = []
    train = extract_or_load_sharded(
        session,
        train_items,
        train_manifest,
        args.output_dir / "features",
        "train",
        model_hash,
        args.batch_size,
        True,
        args.feature_shard_images,
        single_view_sources,
        args.reextract_cached_features,
        fresh_feature_run_id,
        feature_shard_evidence,
    )
    validation = extract_or_load_sharded(
        session,
        validation_items,
        validation_manifest,
        args.output_dir / "features",
        "validation",
        model_hash,
        args.batch_size,
        False,
        args.feature_shard_images,
        frozenset(),
        args.reextract_cached_features,
        fresh_feature_run_id,
        feature_shard_evidence,
    )
    regression: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    if regression_items is not None and regression_manifest is not None:
        regression = extract_or_load_sharded(
            session,
            regression_items,
            regression_manifest,
            args.output_dir / "features",
            "regression",
            model_hash,
            args.batch_size,
            False,
            args.feature_shard_images,
            frozenset(),
            args.reextract_cached_features,
            fresh_feature_run_id,
            feature_shard_evidence,
        )
    if recipe_config is not None and int(recipe_config["expectedTrainingFeatureViews"]) != int(train[0].shape[0]):
        raise ValueError("Extracted training feature-view count does not match recipe")
    if (
        recipe_config is not None
        and recipe_config.get("expectedValidationFeatureViews") is not None
        and int(recipe_config["expectedValidationFeatureViews"]) != int(validation[0].shape[0])
    ):
        raise ValueError("Extracted validation feature-view count does not match recipe")
    if (
        recipe_config is not None
        and regression is not None
        and int(recipe_config["expectedRegressionFeatureViews"]) != int(regression[0].shape[0])
    ):
        raise ValueError("Extracted regression feature-view count does not match recipe")
    device = torch.device("cuda" if args.execution_provider == "cuda" else "cpu")
    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, ...], np.ndarray, float, float, dict[str, object], dict[str, float]] | None = None
    for decay in (0.10, 0.03, 0.01, 0.003, 0.001):
        for alpha in (0.40, 0.55, 0.70, 0.85, 1.0):
            parameters = {"weightDecay": decay, "upstreamBlendAlpha": alpha}
            try:
                weight, bias, threshold, values, key = fit_candidate(
                    train[0], train[1], train[3], validation[0], validation[1], validation[2], validation[3],
                    upstream_weight, upstream_bias, decay, alpha, device, validation_gates,
                )
            except (FloatingPointError, RuntimeError) as error:
                record = {"parameters": parameters, "status": "rejected", "reason": str(error)}
                candidates.append(record)
                print(json.dumps(record, separators=(",", ":")), flush=True)
                continue
            record = {"parameters": parameters, "selectionKey": key, "thresholdLogit": threshold, "variants": values}
            candidates.append(record)
            print(json.dumps(record, separators=(",", ":")), flush=True)
            if best is None or key > best[0]:
                best = (key, weight, bias, threshold, values, parameters)
    if best is None:
        raise RuntimeError("No trained candidate")
    _, weight, bias, threshold, values, parameters = best
    if recipe_config is not None:
        if not passes_validation_gates(values, recipe_config["validationGates"]):
            raise RuntimeError("Selected candidate failed frozen validation gates")
    regression_values: dict[str, object] | None = None
    if regression is not None:
        if recipe_config is None or recipe_config.get("regressionGates") is None:
            raise ValueError("Regression features exist without frozen regression gates")
        regression_values = evaluate_frozen_regression(
            weight,
            bias,
            threshold,
            regression,
            dict(recipe_config["regressionGates"]),
        )
    model_path = args.output_dir / "model.onnx"
    replace_classifier(args.model, model_path, weight, bias)
    exported_hash = digest(model_path)
    parity_session = ort.InferenceSession(str(model_path), providers=providers)
    export_parity = {
        "selector": exported_parity_error(
            parity_session,
            validation_items,
            validation[0],
            weight,
            bias,
            args.batch_size,
        )
    }
    if regression is not None and regression_items is not None:
        export_parity["regression"] = exported_parity_error(
            parity_session,
            regression_items,
            regression[0],
            weight,
            bias,
            args.batch_size,
        )
    max_error = max(export_parity.values())
    calibration_intercept = math.log(DISPLAY_THRESHOLD / (1 - DISPLAY_THRESHOLD)) - threshold
    calibration = {
        "schemaVersion": 1,
        "method": "Validation-selected logit alignment to fixed 65/100 display threshold; not probability calibration",
        "slope": 1,
        "intercept": calibration_intercept,
        "validationThresholdLogit": threshold,
        "rawProbabilityThreshold": 1 / (1 + math.exp(-threshold)),
        "displayThreshold": DISPLAY_THRESHOLD,
        "modelSha256": exported_hash,
        "trainManifestSha256": digest(train_manifest),
        "validationManifestSha256": digest(validation_manifest),
        "regressionManifestSha256": digest(regression_manifest) if regression_manifest else None,
        "selectionSummarySha256": selection_summary_hash,
    }
    source_counts = {
        source: sum(item.source == source for item in train_items)
        for source in sorted({item.source for item in train_items})
    }
    if fresh_feature_run is not None and fresh_feature_run_id is not None:
        fresh_feature_run = complete_fresh_feature_run(
            fresh_feature_marker,
            fresh_feature_context,
            fresh_feature_run_id,
        )
    summary = {
        "schemaVersion": 2 if regression is not None else 1,
        "seed": SEED,
        "pipelineVersion": PIPELINE_VERSION,
        "trainerSha256": digest(Path(__file__)),
        "commandArguments": sys.argv[1:],
        "recipeSha256": digest(args.recipe) if args.recipe else None,
        "selectionSummarySha256": selection_summary_hash,
        "upstreamModelSha256": model_hash,
        "trainManifestSha256": digest(train_manifest),
        "validationManifestSha256": digest(validation_manifest),
        "regressionManifestSha256": digest(regression_manifest) if regression_manifest else None,
        "trainImages": len(train_items),
        "validationImages": len(validation_items),
        "viewsPerImage": len(VARIANTS),
        "trainFeatureViews": int(train[0].shape[0]),
        "validationFeatureViews": int(validation[0].shape[0]),
        "regressionImages": len(regression_items) if regression_items is not None else None,
        "regressionFeatureViews": int(regression[0].shape[0]) if regression is not None else None,
        "sourceBalancedSampling": False,
        "sourceBalancedLoss": True,
        "trainingEpochs": TRAINING_EPOCHS,
        "trainingBatchSize": TRAINING_BATCH_SIZE,
        "uniqueTrainingImagesCovered": len(train_items),
        "uniqueTrainingFeatureViewsCovered": int(train[0].shape[0]),
        "trainingSourceCounts": source_counts,
        "featureShardImages": args.feature_shard_images,
        "featureBatchSize": args.batch_size,
        "cachedFeatureSourcePixelsReverified": True,
        "cachedFeatureArraysValidated": True,
        "cachedFeatureValuesReextracted": args.reextract_cached_features,
        "freshFeatureRun": fresh_feature_run,
        "freshFeatureRunMarkerSha256": marker_sha256(fresh_feature_run) if fresh_feature_run else None,
        "cachedFeatureDtypes": {
            "features": "float32",
            "labels": "float32",
            "variants": "int64",
            "sources": "unicode",
        },
        "singleViewTrainingSources": sorted(single_view_sources),
        "featureConfigurationHashes": feature_configuration_hashes,
        "featureShardEvidence": feature_shard_evidence,
        "selectedParameters": parameters,
        "validationGates": recipe_config.get("validationGates") if recipe_config else None,
        "validationGatesPassed": True,
        "selectionKey": best[0],
        "thresholdLogit": threshold,
        "variants": values,
        "selector": {
            "manifestSha256": digest(validation_manifest),
            "role": regression_contract["selectorRole"] if regression_contract else "validation-selection",
            "images": len(validation_items),
            "featureViews": int(validation[0].shape[0]),
            "gates": recipe_config.get("validationGates") if recipe_config else None,
            "gatesPassed": True,
            "thresholdLogit": threshold,
            "variants": values,
        },
        "regression": {
            "manifestSha256": digest(regression_manifest),
            "dataRoot": str(regression_data_root),
            "role": regression_contract["regressionRole"],
            "images": len(regression_items),
            "featureViews": int(regression[0].shape[0]),
            "gates": recipe_config.get("regressionGates") if recipe_config else None,
            "gatesPassed": True,
            "thresholdLogitFromSelector": threshold,
            "variants": regression_values,
            "selectionInfluenced": False,
        }
        if regression is not None
        and regression_items is not None
        and regression_manifest is not None
        and regression_data_root is not None
        and regression_contract is not None
        else None,
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": exported_hash,
            "maxAbsParityError": max_error,
            "maxAbsParityErrorByPartition": export_parity,
        },
        "candidateCount": len(candidates),
        "validCandidateCount": sum("selectionKey" in candidate for candidate in candidates),
        "environment": {
            "numpy": np.__version__,
            "onnxRuntime": ort.__version__,
            "pillow": PILLOW_VERSION,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if args.execution_provider == "cuda" else None,
            "executionProvider": args.execution_provider,
            "providers": parity_session.get_providers(),
            "torchDeterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
            "torchThreads": torch.get_num_threads(),
        },
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (args.output_dir / "validation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "candidate-grid.json").write_text(json.dumps(candidates, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
