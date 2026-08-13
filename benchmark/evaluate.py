"""Evaluate a Community Forensics ONNX detector on a frozen image manifest.

This research harness is intentionally independent from the extension runtime.
It preserves the detector's corrected 384 px preprocessing contract, records
per-image predictions, and can freeze a one-parameter calibration against the
bounty's fixed 0.65 display threshold. Dataset pixels stay outside the repo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


SEED = 20260813
DISPLAY_THRESHOLD = 0.65
INPUT_SIZE = 384
RESIZE_SHORT_EDGE = 440
MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")


@dataclass(frozen=True)
class Item:
    id: str
    path: Path
    image_sha256: str
    label: int
    source: str


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def seeded_random(item: Item, variant: str) -> random.Random:
    value = int(sha256(f"{SEED}:{item.id}:{variant}".encode()).hexdigest()[:16], 16)
    return random.Random(value)


def resize_long_edge(image: Image.Image, maximum: int) -> Image.Image:
    width, height = image.size
    if max(width, height) <= maximum:
        return image
    scale = maximum / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def transform(image: Image.Image, item: Item, variant: str) -> Image.Image:
    rng = seeded_random(item, variant)
    if variant == "original":
        return image
    if variant == "screenshot":
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
        media_left, media_top, media_width, media_height = 47, 140, 1076, 1110
        draw.rectangle(
            (media_left, media_top, media_left + media_width - 1, media_top + media_height - 1),
            fill=(17, 21, 26),
        )
        scale = min(media_width / image.width, media_height / image.height)
        rendered = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        frame.paste(
            rendered,
            (
                media_left + (media_width - rendered.width) // 2,
                media_top + (media_height - rendered.height) // 2,
            ),
        )
        for x in (78, 124, 170):
            draw.ellipse((x, 1291, x + 22, 1313), outline=(136, 145, 155), width=3)
        return frame
    if variant == "social-q75":
        transformed = resize_long_edge(image, 1080)
        return jpeg_roundtrip(transformed, 75)
    if variant == "social-heavy":
        transformed = jpeg_roundtrip(resize_long_edge(image, 720), 50)
        return jpeg_roundtrip(resize_long_edge(transformed, 640), 38)
    raise ValueError(f"Unknown variant: {variant}")


def preprocess(item: Item, variant: str) -> np.ndarray:
    if file_digest(item.path) != item.image_sha256:
        raise ValueError(f"Image integrity mismatch: {item.id}")
    with Image.open(item.path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image = transform(image, item, variant)
    width, height = image.size
    scale = RESIZE_SHORT_EDGE / min(width, height)
    resized = image.resize(
        (max(INPUT_SIZE, round(width * scale)), max(INPUT_SIZE, round(height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - INPUT_SIZE) // 2
    top = (resized.height - INPUT_SIZE) // 2
    cropped = resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
    pixels = np.asarray(cropped, dtype=np.float32) / 255.0
    normalized = (pixels - MEAN) / STD
    return np.transpose(normalized, (2, 0, 1)).astype(np.float32)


def sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def calibrated_probability(raw_probability: np.ndarray, intercept: float) -> np.ndarray:
    clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    logits = np.log(clipped / (1 - clipped)) + intercept
    return sigmoid(logits)


def metrics(records: list[dict[str, object]], raw_threshold: float) -> dict[str, object]:
    real = [record for record in records if int(record["label"]) == 0]
    synthetic = [record for record in records if int(record["label"]) == 1]
    if not real or not synthetic:
        raise ValueError("Evaluation requires both real and synthetic images")
    real_recall = sum(float(record["rawProbability"]) < raw_threshold for record in real) / len(real)
    synthetic_recall = sum(float(record["rawProbability"]) >= raw_threshold for record in synthetic) / len(synthetic)
    sources = sorted({str(record["source"]) for record in synthetic})
    by_source = {}
    for source in sources:
        group = [record for record in synthetic if str(record["source"]) == source]
        by_source[source] = sum(float(record["rawProbability"]) >= raw_threshold for record in group) / len(group)
    return {
        "balancedAccuracy": (real_recall + synthetic_recall) / 2,
        "realRecall": real_recall,
        "syntheticRecall": synthetic_recall,
        "syntheticRecallBySource": by_source,
        "count": len(records),
    }


def choose_threshold(records_by_variant: dict[str, list[dict[str, object]]]) -> tuple[float, dict[str, object]]:
    probabilities = sorted(
        {float(record["rawProbability"]) for records in records_by_variant.values() for record in records}
    )
    candidates = [0.0, *probabilities, 1.0]
    best: tuple[tuple[float, float, float, float], float, dict[str, object]] | None = None
    for threshold in candidates:
        by_variant = {variant: metrics(records, threshold) for variant, records in records_by_variant.items()}
        values = list(by_variant.values())
        source_recalls = [
            float(recall)
            for value in values
            for recall in dict(value["syntheticRecallBySource"]).values()
        ]
        passes = all(
            float(value["balancedAccuracy"]) >= 0.75
            and float(value["realRecall"]) >= 0.85
            and float(value["syntheticRecall"]) >= 0.70
            for value in values
        ) and min(source_recalls) >= 0.60
        if not passes:
            continue
        accuracies = [float(value["balancedAccuracy"]) for value in values]
        key = (
            min(accuracies),
            sum(accuracies) / len(accuracies),
            min(float(value["realRecall"]) for value in values),
            min(source_recalls),
        )
        if best is None or key > best[0]:
            best = (key, threshold, by_variant)
    if best is None:
        raise RuntimeError("No threshold satisfies the frozen validation gates")
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--execution-provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--fit-calibration", action="store_true")
    args = parser.parse_args()

    model_hash = file_digest(args.model)
    if model_hash != args.expected_model_sha256:
        raise ValueError(f"Unexpected model SHA-256: {model_hash}")
    manifest_hash = file_digest(args.manifest)
    items = load_manifest(args.manifest, args.data_root)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.execution_provider == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    if session.get_inputs()[0].name != "pixel_values" or session.get_outputs()[0].name != "logits":
        raise ValueError("Unexpected ONNX graph interface")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records_by_variant: dict[str, list[dict[str, object]]] = {}
    started = time.perf_counter()
    for variant in args.variants:
        records: list[dict[str, object]] = []
        for offset in range(0, len(items), args.batch_size):
            batch_items = items[offset : offset + args.batch_size]
            tensor = np.stack([preprocess(item, variant) for item in batch_items])
            logits = np.asarray(session.run(["logits"], {"pixel_values": tensor})[0]).reshape(-1)
            probabilities = sigmoid(logits)
            for item, logit, probability in zip(batch_items, logits, probabilities, strict=True):
                records.append(
                    {
                        "id": item.id,
                        "label": item.label,
                        "source": item.source,
                        "variant": variant,
                        "logit": float(logit),
                        "rawProbability": float(probability),
                    }
                )
            completed = min(offset + args.batch_size, len(items))
            if completed % (args.batch_size * 10) == 0 or completed == len(items):
                elapsed = time.perf_counter() - started
                print(f"{variant}: {completed}/{len(items)} ({completed / max(elapsed, 1e-6):.1f} images/s)", flush=True)
        records_by_variant[variant] = records
        predictions = args.output_dir / f"{args.name}-{variant}-predictions.jsonl"
        predictions.write_text("\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n")

    if args.fit_calibration:
        raw_threshold, validation_metrics = choose_threshold(records_by_variant)
        intercept = math.log(DISPLAY_THRESHOLD / (1 - DISPLAY_THRESHOLD)) - math.log(raw_threshold / (1 - raw_threshold))
        calibration = {
            "schemaVersion": 1,
            "method": "Frozen generator-family-held-out validation threshold alignment",
            "slope": 1,
            "intercept": intercept,
            "rawProbabilityThreshold": raw_threshold,
            "displayThreshold": DISPLAY_THRESHOLD,
            "modelSha256": model_hash,
            "manifestSha256": manifest_hash,
            "validation": validation_metrics,
        }
        calibration_path = args.output_dir / f"{args.name}-calibration.json"
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
    elif args.calibration:
        calibration = json.loads(args.calibration.read_text())
        if calibration.get("modelSha256") != model_hash:
            raise ValueError("Calibration targets a different model")
        if "rawProbabilityThreshold" in calibration:
            raw_threshold = float(calibration["rawProbabilityThreshold"])
        elif "validationThresholdLogit" in calibration:
            raw_threshold = 1 / (1 + math.exp(-float(calibration["validationThresholdLogit"])))
        else:
            raise ValueError("Calibration lacks a raw probability or logit threshold")
    else:
        calibration = None
        raw_threshold = DISPLAY_THRESHOLD

    report = {
        "schemaVersion": 1,
        "name": args.name,
        "model": {"path": str(args.model), "sha256": model_hash, "bytes": args.model.stat().st_size},
        "dataset": {"manifest": str(args.manifest), "sha256": manifest_hash, "items": len(items)},
        "runtime": {"providers": session.get_providers(), "batchSize": args.batch_size},
        "threshold": {"display": DISPLAY_THRESHOLD, "raw": raw_threshold, "calibration": calibration},
        "variants": {variant: metrics(records, raw_threshold) for variant, records in records_by_variant.items()},
        "elapsedSeconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / f"{args.name}-summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
