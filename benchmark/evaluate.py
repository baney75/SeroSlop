"""Evaluate a Community Forensics ONNX detector on a frozen image manifest.

This research harness is intentionally independent from the extension runtime.
It preserves the detector's corrected 384 px preprocessing contract and records
predictions only after every frozen input has passed a fail-closed preflight.
Dataset pixels stay outside the repo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, __version__ as PILLOW_VERSION

from evaluation_contract import require_canonical_output_directory, require_public_pre_score_freeze


SEED = 20260813
DISPLAY_THRESHOLD = 0.65
INPUT_SIZE = 384
RESIZE_SHORT_EDGE = 440
MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")
PROTOCOLS = {
    "validation": {
        "name": "prooflens-validation",
        "manifestSha256": "41be10ef876ecef0635744ed29677a1888a7759cc8060dc7a392f76f83ab263b",
        "items": 600,
        "labels": {0: 300, 1: 300},
        "sources": {"GLM-Image": 150, "HunyuanImage-3.0": 150, "open-images": 300},
        "negativeOnly": False,
    },
    "confirmatory": {
        "name": "prooflens-confirmatory-test",
        "manifestSha256": "28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae",
        "items": 600,
        "labels": {0: 300, 1: 300},
        "sources": {"kling_v2_1": 300, "library-of-congress-fsa-owi-color": 300},
        "negativeOnly": False,
    },
    "web-negative": {
        "name": "prooflens-web-negative",
        "manifestSha256": "ad8b3f30a37feb3b6b046683db2d4071e236e6878612c7d8733869699d7f7824",
        "items": 319,
        "labels": {0: 319},
        "sources": {"chartography-expert-created": 19, "library-of-congress-fsa-owi-color": 300},
        "negativeOnly": True,
    },
}


@dataclass(frozen=True)
class Item:
    id: str
    path: Path
    image_sha256: str
    label: int
    source: str
    group_id: str


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
        image_path = data_root / str(row["path"])
        try:
            image_path.resolve().relative_to(data_root.resolve())
        except ValueError as error:
            raise ValueError(f"Manifest path escapes its data root: {row['id']}") from error
        items.append(
            Item(
                id=str(row["id"]),
                path=image_path,
                image_sha256=str(row["imageSha256"]),
                label=int(row["label"]),
                source=str(row["source"]),
                group_id=str(row.get("groupId", row["id"])),
            )
        )
    if not items:
        raise ValueError(f"Manifest is empty: {path}")
    if len({item.id for item in items}) != len(items):
        raise ValueError(f"Manifest contains duplicate IDs: {path}")
    if len({item.image_sha256 for item in items}) != len(items):
        raise ValueError(f"Manifest contains duplicate image bytes: {path}")
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


def negative_metrics(records: list[dict[str, object]], raw_threshold: float) -> dict[str, object]:
    if not records or any(int(record["label"]) != 0 for record in records):
        raise ValueError("Negative-only evaluation requires only real-image labels")
    by_source: dict[str, object] = {}
    for source in sorted({str(record["source"]) for record in records}):
        group = [record for record in records if str(record["source"]) == source]
        false_positive_rate = sum(
            float(record["rawProbability"]) >= raw_threshold for record in group
        ) / len(group)
        by_source[source] = {"count": len(group), "falsePositiveRate": false_positive_rate}
    false_positive_rate = sum(
        float(record["rawProbability"]) >= raw_threshold for record in records
    ) / len(records)
    return {
        "count": len(records),
        "realRecall": 1 - false_positive_rate,
        "falsePositiveRate": false_positive_rate,
        "bySource": by_source,
    }


def counts(values: list[object]) -> dict[object, int]:
    return {value: values.count(value) for value in sorted(set(values))}


def encoded_json(value: object) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def encoded_json_lines(rows: list[dict[str, object]]) -> bytes:
    return ("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--execution-provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="recompute the frozen protocol and require byte-identical existing outputs",
    )
    args = parser.parse_args()

    protocol = PROTOCOLS[args.protocol]
    name = str(protocol["name"])
    repository_root = Path(__file__).resolve().parents[1]
    args.output_dir = require_canonical_output_directory(
        args.protocol,
        args.output_dir,
        repository_root=repository_root,
    )
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.expected_manifest_sha256 != protocol["manifestSha256"]:
        raise ValueError(f"{args.protocol} requires its predeclared manifest SHA-256")
    model_hash = file_digest(args.model)
    if model_hash != args.expected_model_sha256:
        raise ValueError(f"Unexpected model SHA-256: {model_hash}")
    manifest_hash = file_digest(args.manifest)
    if manifest_hash != args.expected_manifest_sha256:
        raise ValueError(f"Unexpected manifest SHA-256: {manifest_hash}")
    calibration_hash = file_digest(args.calibration)
    if calibration_hash != args.expected_calibration_sha256:
        raise ValueError(f"Unexpected calibration SHA-256: {calibration_hash}")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("modelSha256") != model_hash:
        raise ValueError("Calibration targets a different model")
    if calibration.get("validationManifestSha256") != PROTOCOLS["validation"]["manifestSha256"]:
        raise ValueError("Calibration is not bound to the frozen validation manifest")
    if calibration.get("slope") != 1 or calibration.get("displayThreshold") != DISPLAY_THRESHOLD:
        raise ValueError("Calibration contract changed")
    raw_threshold = float(calibration.get("rawProbabilityThreshold", math.nan))
    if not math.isfinite(raw_threshold) or not 0 < raw_threshold < 1:
        raise ValueError("Calibration raw threshold must be finite and strictly between zero and one")
    if not math.isfinite(float(calibration.get("intercept", math.nan))):
        raise ValueError("Calibration intercept is not finite")
    if args.protocol in {"confirmatory", "web-negative"}:
        require_public_pre_score_freeze(
            repository_root=repository_root,
            allow_public_descendant=args.verify_existing,
        )

    items = load_manifest(args.manifest, args.data_root)
    if len(items) != protocol["items"] or counts([item.label for item in items]) != protocol["labels"]:
        raise ValueError(f"{args.protocol} manifest class allocation changed")
    if counts([item.source for item in items]) != protocol["sources"]:
        raise ValueError(f"{args.protocol} manifest source allocation changed")
    for item in items:
        if not item.path.is_file() or file_digest(item.path) != item.image_sha256:
            raise ValueError(f"Image integrity mismatch during preflight: {item.id}")

    targets = [args.output_dir / f"{name}-{variant}-predictions.jsonl" for variant in VARIANTS]
    targets.extend([
        args.output_dir / f"{name}-summary.json",
        args.output_dir / f"{name}-complete.json",
    ])
    existing = [target.exists() for target in targets]
    if args.verify_existing:
        if not all(existing):
            raise FileNotFoundError("Verification mode requires every canonical output to exist")
    elif any(existing):
        raise FileExistsError("A canonical output already exists; refusing to overwrite frozen evidence")

    available_providers = set(ort.get_available_providers())
    if args.execution_provider == "cuda" and "CUDAExecutionProvider" not in available_providers:
        raise RuntimeError(f"CUDA provider unavailable: {sorted(available_providers)}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.execution_provider == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(args.model), providers=providers)
    if session.get_inputs()[0].name != "pixel_values" or session.get_outputs()[0].name != "logits":
        raise ValueError("Unexpected ONNX graph interface")

    records_by_variant: dict[str, list[dict[str, object]]] = {}
    for variant in VARIANTS:
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
                        "groupId": item.group_id,
                        "variant": variant,
                        "logit": float(logit),
                        "rawProbability": float(probability),
                    }
                )
            completed = min(offset + args.batch_size, len(items))
            if completed % (args.batch_size * 10) == 0 or completed == len(items):
                print(f"{variant}: {completed}/{len(items)}", flush=True)
        records_by_variant[variant] = records

    report = {
        "schemaVersion": 2,
        "name": name,
        "protocol": args.protocol,
        "commitMarkerPublication": True,
        "allVariantsRequired": True,
        "model": {"path": str(args.model), "sha256": model_hash, "bytes": args.model.stat().st_size},
        "dataset": {"manifest": str(args.manifest), "sha256": manifest_hash, "items": len(items)},
        "runtime": {
            "executionProvider": args.execution_provider,
            "providers": session.get_providers(),
            "batchSize": args.batch_size,
            "numpy": np.__version__,
            "onnxRuntime": ort.__version__,
            "pillow": PILLOW_VERSION,
        },
        "threshold": {
            "display": DISPLAY_THRESHOLD,
            "raw": raw_threshold,
            "calibrationSha256": calibration_hash,
            "calibration": calibration,
        },
        "variants": {
            variant: negative_metrics(records, raw_threshold) if protocol["negativeOnly"] else metrics(records, raw_threshold)
            for variant, records in records_by_variant.items()
        },
    }
    output_bytes = {
        args.output_dir / f"{name}-{variant}-predictions.jsonl": encoded_json_lines(records)
        for variant, records in records_by_variant.items()
    }
    output_bytes[args.output_dir / f"{name}-summary.json"] = encoded_json(report)
    completion_path = args.output_dir / f"{name}-complete.json"
    output_bytes[completion_path] = encoded_json({
        "schemaVersion": 1,
        "protocol": args.protocol,
        "modelSha256": model_hash,
        "manifestSha256": manifest_hash,
        "calibrationSha256": calibration_hash,
        "files": {
            target.name: sha256(value).hexdigest()
            for target, value in output_bytes.items()
        },
    })
    args.output_dir = require_canonical_output_directory(
        args.protocol,
        args.output_dir,
        repository_root=repository_root,
    )
    if args.verify_existing:
        for target, expected in output_bytes.items():
            if target.read_bytes() != expected:
                raise ValueError(f"Recomputed evidence differs from {target}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=args.output_dir) as temporary:
            temporary_root = Path(temporary)
            staged = []
            for target, value in output_bytes.items():
                stage_path = temporary_root / target.name
                stage_path.write_bytes(value)
                staged.append((stage_path, target))
            for stage_path, target in [row for row in staged if row[1] != completion_path]:
                os.replace(stage_path, target)
            completion_stage = next(stage_path for stage_path, target in staged if target == completion_path)
            os.replace(completion_stage, completion_path)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
