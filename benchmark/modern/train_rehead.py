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
from pathlib import Path
import random
import time

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
import torch
from torch import nn


SEED = 20260813
PIPELINE_VERSION = 1
INPUT_SIZE = 384
RESIZE_SHORT_EDGE = 440
MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")
FEATURE_OUTPUT = "/Gather_output_0"
DISPLAY_THRESHOLD = 0.65


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
    return items


def seeded_random(item: Item, purpose: str) -> random.Random:
    value = int(sha256(f"{SEED}:{item.id}:{purpose}".encode()).hexdigest()[:16], 16)
    return random.Random(value)


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


def preprocess_views(item: Item, training: bool) -> list[np.ndarray]:
    if digest(item.path) != item.image_sha256:
        raise ValueError(f"Image integrity mismatch: {item.id}")
    with Image.open(item.path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    return [preprocess_image(image.copy(), item, variant, training) for variant in VARIANTS]


def make_feature_model(source: Path, destination: Path) -> tuple[np.ndarray, float]:
    model = onnx.load(source)
    if not any(output.name == FEATURE_OUTPUT for output in model.graph.output):
        model.graph.output.append(helper.make_tensor_value_info(FEATURE_OUTPUT, TensorProto.FLOAT, ["batch_size", 384]))
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    variants: list[int] = []
    sources: list[str] = []
    expanded_count = len(items) * len(VARIANTS)
    started = time.perf_counter()
    images_per_batch = max(1, batch_size // len(VARIANTS))
    with ThreadPoolExecutor(max_workers=min(8, images_per_batch)) as executor:
        for offset in range(0, len(items), images_per_batch):
            batch_items = items[offset : offset + images_per_batch]
            views = list(executor.map(lambda item: preprocess_views(item, training), batch_items))
            tensor = np.stack([pixels for item_views in views for pixels in item_views])
            output = np.asarray(session.run([FEATURE_OUTPUT], {"pixel_values": tensor})[0], dtype=np.float32)
            features.append(output)
            labels.extend(item.label for item in batch_items for _ in VARIANTS)
            variants.extend(index for _ in batch_items for index in range(len(VARIANTS)))
            sources.extend(item.source for item in batch_items for _ in VARIANTS)
            completed = min(offset + images_per_batch, len(items)) * len(VARIANTS)
            if completed % (batch_size * 10) == 0 or completed == expanded_count:
                elapsed = time.perf_counter() - started
                print(f"features {completed}/{expanded_count} ({completed / max(elapsed, 1e-6):.1f}/s)", flush=True)
    return (
        np.concatenate(features),
        np.asarray(labels, dtype=np.float32),
        np.asarray(variants, dtype=np.int64),
        np.asarray(sources),
    )


def extract_or_load(
    session: ort.InferenceSession,
    items: list[Item],
    manifest: Path,
    cache: Path,
    model_hash: str,
    batch_size: int,
    training: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manifest_hash = digest(manifest)
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if (
            str(data["manifest_hash"].item()) == manifest_hash
            and str(data["model_hash"].item()) == model_hash
            and int(data["pipeline_version"].item()) == PIPELINE_VERSION
        ):
            print(f"loaded {cache}", flush=True)
            return data["features"], data["labels"], data["variants"], data["sources"]
    result = extract_features(session, items, batch_size, training)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        features=result[0],
        labels=result[1],
        variants=result[2],
        sources=result[3],
        manifest_hash=np.asarray(manifest_hash),
        model_hash=np.asarray(model_hash),
        pipeline_version=np.asarray(PIPELINE_VERSION),
    )
    return result


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
        output[name] = {
            "balancedAccuracy": (real_recall + synthetic_recall) / 2,
            "realRecall": real_recall,
            "syntheticRecall": synthetic_recall,
            "syntheticRecallBySource": source_recall,
        }
    return output


def choose_threshold(logits: np.ndarray, labels: np.ndarray, variants: np.ndarray, sources: np.ndarray) -> tuple[float, dict[str, object], tuple[float, ...]]:
    candidates = np.unique(np.quantile(logits, np.linspace(0.01, 0.99, 513)))
    candidates = np.concatenate(([float(logits.min()) - 1e-6], candidates, [float(logits.max()) + 1e-6]))
    best: tuple[tuple[float, ...], float, dict[str, object]] | None = None
    for threshold in candidates:
        values = variant_metrics(logits, labels, variants, sources, float(threshold))
        rows = list(values.values())
        family_recalls = [
            float(recall)
            for row in rows
            for recall in dict(row["syntheticRecallBySource"]).values()
        ]
        key = (
            min(float(row["balancedAccuracy"]) for row in rows),
            sum(float(row["balancedAccuracy"]) for row in rows) / len(rows),
            min(float(row["realRecall"]) for row in rows),
            min(family_recalls),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), values)
    if best is None:
        raise RuntimeError("No threshold candidates")
    return best[1], best[2], best[0]


def fit_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    validation_variants: np.ndarray,
    validation_sources: np.ndarray,
    upstream_weight: np.ndarray,
    upstream_bias: float,
    decay: float,
    alpha: float,
    device: torch.device,
) -> tuple[np.ndarray, float, float, dict[str, object], tuple[float, ...]]:
    mean = train_features.mean(axis=0).astype(np.float32)
    std = train_features.std(axis=0).clip(min=1e-5).astype(np.float32)
    x = torch.from_numpy((train_features - mean) / std).to(device)
    y = torch.from_numpy(train_labels).to(device).unsqueeze(1)
    x_validation = torch.from_numpy((validation_features - mean) / std).to(device)
    head = nn.Linear(train_features.shape[1], 1).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(upstream_weight * std).to(device).unsqueeze(0))
        head.bias.copy_(torch.tensor([upstream_bias + float(np.dot(upstream_weight, mean))], device=device))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.012, weight_decay=decay)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    real_indices = torch.from_numpy(np.flatnonzero(train_labels == 0))
    synthetic_indices = torch.from_numpy(np.flatnonzero(train_labels == 1))
    half = min(768, real_indices.numel(), synthetic_indices.numel())
    for _ in range(900):
        indices = torch.cat(
            (
                real_indices[torch.randint(0, real_indices.numel(), (half,), generator=generator)],
                synthetic_indices[torch.randint(0, synthetic_indices.numel(), (half,), generator=generator)],
            )
        ).to(device)
        loss = nn.functional.binary_cross_entropy_with_logits(head(x[indices]), y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    learned_normalized_weight = head.weight.detach().float().cpu().numpy()[0]
    learned_normalized_bias = float(head.bias.detach().float().cpu().item())
    learned_weight = learned_normalized_weight / std
    learned_bias = learned_normalized_bias - float(np.dot(learned_normalized_weight, mean / std))
    weight = upstream_weight * (1 - alpha) + learned_weight * alpha
    bias = upstream_bias * (1 - alpha) + learned_bias * alpha
    logits = validation_features @ weight + bias
    threshold, values, key = choose_threshold(logits, validation_labels, validation_variants, validation_sources)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model_hash = digest(args.model)
    if model_hash != args.expected_model_sha256:
        raise ValueError(f"Unexpected model SHA-256: {model_hash}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest = args.data_root / "train-manifest.jsonl"
    validation_manifest = args.data_root / "validation-manifest.jsonl"
    feature_model = args.output_dir / "feature-model.onnx"
    upstream_weight, upstream_bias = make_feature_model(args.model, feature_model)
    session = ort.InferenceSession(str(feature_model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"CUDA provider unavailable: {session.get_providers()}")
    train = extract_or_load(
        session,
        load_manifest(train_manifest, args.data_root),
        train_manifest,
        args.output_dir / "train-features.npz",
        model_hash,
        args.batch_size,
        True,
    )
    validation = extract_or_load(
        session,
        load_manifest(validation_manifest, args.data_root),
        validation_manifest,
        args.output_dir / "validation-features.npz",
        model_hash,
        args.batch_size,
        False,
    )
    device = torch.device("cuda")
    candidates: list[dict[str, object]] = []
    best: tuple[tuple[float, ...], np.ndarray, float, float, dict[str, object], dict[str, float]] | None = None
    for decay in (0.10, 0.03, 0.01, 0.003, 0.001):
        for alpha in (0.40, 0.55, 0.70, 0.85, 1.0):
            weight, bias, threshold, values, key = fit_candidate(
                train[0], train[1], validation[0], validation[1], validation[2], validation[3],
                upstream_weight, upstream_bias, decay, alpha, device,
            )
            parameters = {"weightDecay": decay, "upstreamBlendAlpha": alpha}
            record = {"parameters": parameters, "selectionKey": key, "thresholdLogit": threshold, "variants": values}
            candidates.append(record)
            print(json.dumps(record, separators=(",", ":")), flush=True)
            if best is None or key > best[0]:
                best = (key, weight, bias, threshold, values, parameters)
    if best is None:
        raise RuntimeError("No trained candidate")
    _, weight, bias, threshold, values, parameters = best
    model_path = args.output_dir / "model.onnx"
    replace_classifier(args.model, model_path, weight, bias)
    exported_hash = digest(model_path)
    parity_session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    expected = validation[0] @ weight + bias
    actual: list[np.ndarray] = []
    validation_items = load_manifest(validation_manifest, args.data_root)
    images_per_batch = max(1, args.batch_size // len(VARIANTS))
    with ThreadPoolExecutor(max_workers=min(8, images_per_batch)) as executor:
        for offset in range(0, len(validation_items), images_per_batch):
            batch_items = validation_items[offset : offset + images_per_batch]
            views = list(executor.map(lambda item: preprocess_views(item, False), batch_items))
            tensor = np.stack([pixels for item_views in views for pixels in item_views])
            actual.append(np.asarray(parity_session.run(["logits"], {"pixel_values": tensor})[0]).reshape(-1))
    max_error = float(np.max(np.abs(expected - np.concatenate(actual))))
    if max_error > 2e-4:
        raise RuntimeError(f"Export parity error too high: {max_error}")
    calibration_intercept = math.log(DISPLAY_THRESHOLD / (1 - DISPLAY_THRESHOLD)) - threshold
    calibration = {
        "schemaVersion": 1,
        "method": "Held-out-generator logit alignment to fixed 65% display threshold",
        "slope": 1,
        "intercept": calibration_intercept,
        "validationThresholdLogit": threshold,
        "rawProbabilityThreshold": 1 / (1 + math.exp(-threshold)),
        "displayThreshold": DISPLAY_THRESHOLD,
        "modelSha256": exported_hash,
    }
    summary = {
        "schemaVersion": 1,
        "seed": SEED,
        "upstreamModelSha256": model_hash,
        "trainManifestSha256": digest(train_manifest),
        "validationManifestSha256": digest(validation_manifest),
        "trainImages": len(load_manifest(train_manifest, args.data_root)),
        "validationImages": len(load_manifest(validation_manifest, args.data_root)),
        "viewsPerImage": len(VARIANTS),
        "selectedParameters": parameters,
        "selectionKey": best[0],
        "thresholdLogit": threshold,
        "variants": values,
        "model": {"path": str(model_path), "bytes": model_path.stat().st_size, "sha256": exported_hash, "maxAbsParityError": max_error},
        "candidateCount": len(candidates),
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (args.output_dir / "validation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "candidate-grid.json").write_text(json.dumps(candidates, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
