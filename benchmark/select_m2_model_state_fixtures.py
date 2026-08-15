"""Freeze deterministic high-margin browser fixtures for the shipped M2 model."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil

import numpy as np


MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47"
CALIBRATION_SHA256 = "06d2452a8db9de26d42285cdc9dad0d233d397a6015583604c64480aec560e2c"
TRAINING_SUMMARY_SHA256 = "c3d49719e50b1fbf5fdc9ba5b8c1df57712910af0f0284a3c3acdf6bad931c04"
VALIDATION_MANIFEST_SHA256 = "a63953148040e1a4223f16fa04ebf4b85c4022da65531ead0b25ce46434eab93"
VALIDATION_FEATURE_SHARD_SHA256 = "7ea076370fddfcb39c29aabeef682aeebc40687198e37b2705da5c0dab9cccf1"
FRESH_FEATURE_RUN_ID = "add5d5306942c5c729c97556bd61cabd"
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")
SELECTION = (
    "M2 development-validation QA fixtures: deterministic maximum-margin synthetic "
    "and Open Images real items from the fresh-feature packet"
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    fingerprint = sha256()
    fingerprint.update(array.dtype.str.encode())
    fingerprint.update(json.dumps(array.shape, separators=(",", ":")).encode())
    fingerprint.update(array.tobytes())
    return fingerprint.hexdigest()


def json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("weights/prooflens-cf384.onnx"))
    parser.add_argument(
        "--calibration", type=Path, default=Path("benchmark/evidence/m2/calibration.json")
    )
    parser.add_argument(
        "--training-summary", type=Path, default=Path("benchmark/evidence/m2/training-summary.json")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmark/evidence/m2/validation-manifest.jsonl")
    )
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data/m2-head"))
    parser.add_argument(
        "--feature-shard",
        type=Path,
        default=Path("benchmark/candidates/prooflens-cf384-m2/features/validation-00000.npz"),
    )
    parser.add_argument(
        "--open-images-attribution",
        type=Path,
        default=Path("benchmark/manifests/open-images-attribution.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures/model-states"))
    args = parser.parse_args()

    locked = {
        args.model: MODEL_SHA256,
        args.calibration: CALIBRATION_SHA256,
        args.training_summary: TRAINING_SUMMARY_SHA256,
        args.manifest: VALIDATION_MANIFEST_SHA256,
        args.feature_shard: VALIDATION_FEATURE_SHARD_SHA256,
    }
    for path, expected in locked.items():
        observed = digest(path)
        if observed != expected:
            raise ValueError(f"Unexpected SHA-256 for {path}: {observed}")

    calibration = json.loads(args.calibration.read_text())
    training_summary = json.loads(args.training_summary.read_text())
    manifest = json_lines(args.manifest)
    if len(manifest) != 900 or [int(row["rowIndex"]) for row in manifest] != list(range(900)):
        raise ValueError("M2 validation manifest order or count changed")
    if calibration.get("modelSha256") != MODEL_SHA256 or calibration.get(
        "validationManifestSha256"
    ) != VALIDATION_MANIFEST_SHA256:
        raise ValueError("M2 calibration is not bound to the model and validation manifest")
    validation_shards = [
        row
        for row in training_summary.get("featureShardEvidence", [])
        if row.get("cache") == str(args.feature_shard)
    ]
    if len(validation_shards) != 1:
        raise ValueError("M2 training summary does not bind one validation feature shard")
    shard_evidence = validation_shards[0]
    if (
        shard_evidence.get("cacheSha256") != VALIDATION_FEATURE_SHARD_SHA256
        or shard_evidence.get("freshFeatureRunId") != FRESH_FEATURE_RUN_ID
        or shard_evidence.get("items") != 900
        or shard_evidence.get("views") != 3600
    ):
        raise ValueError("M2 validation feature-shard evidence changed")

    with np.load(args.feature_shard, allow_pickle=False) as cache:
        features = np.asarray(cache["features"])
        labels = np.asarray(cache["labels"])
        variants = np.asarray(cache["variants"])
        sources = np.asarray(cache["sources"])
        if (
            features.shape != (3600, 384)
            or features.dtype != np.float32
            or labels.dtype != np.float32
            or variants.dtype != np.int64
            or sources.dtype.kind != "U"
            or not np.isfinite(features).all()
        ):
            raise ValueError("M2 validation feature arrays changed")
        for name, value in {
            "features": features,
            "labels": labels,
            "variants": variants,
            "sources": sources,
        }.items():
            if array_digest(value) != shard_evidence["arraySha256"][name]:
                raise ValueError(f"M2 validation {name} array digest changed")
        if (
            str(cache["manifest_hash"].item()) != VALIDATION_MANIFEST_SHA256
            or str(cache["fresh_feature_run_id"].item()) != FRESH_FEATURE_RUN_ID
            or bool(cache["training"].item())
        ):
            raise ValueError("M2 validation feature metadata changed")

    expected_labels = np.repeat(np.asarray([int(row["label"]) for row in manifest], dtype=np.float32), 4)
    expected_variants = np.tile(np.arange(4, dtype=np.int64), 900)
    expected_sources = np.repeat(np.asarray([str(row["source"]) for row in manifest]), 4)
    if not (
        np.array_equal(labels, expected_labels)
        and np.array_equal(variants, expected_variants)
        and np.array_equal(sources, expected_sources)
    ):
        raise ValueError("M2 validation feature rows do not match the manifest")
    item_ids_sha256 = sha256("\n".join(str(row["id"]) for row in manifest).encode()).hexdigest()
    if item_ids_sha256 != shard_evidence.get("itemIdsSha256"):
        raise ValueError("M2 validation item order changed")

    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    from modern.train_rehead import load_manifest, preprocess_views

    model = onnx.load(args.model)
    initializers = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    weight = initializers["classifier.weight"].astype(np.float32).reshape(-1)
    bias = float(initializers["classifier.bias"].reshape(-1)[0])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        selection_logits = features @ weight + bias
    if not np.isfinite(selection_logits).all():
        raise ValueError("M2 validation selection logits are non-finite")
    original_indices = np.flatnonzero(variants == 0)
    scored = [
        (sigmoid(float(selection_logits[feature_index]) + float(calibration["intercept"])), row_index)
        for row_index, feature_index in enumerate(original_indices)
    ]
    likely = sorted(
        (pair for pair in scored if int(manifest[pair[1]]["label"]) == 1 and pair[0] >= 0.80),
        key=lambda pair: (-pair[0], str(manifest[pair[1]]["id"])),
    )
    below = sorted(
        (
            pair
            for pair in scored
            if int(manifest[pair[1]]["label"]) == 0
            and manifest[pair[1]]["source"] == "open-images"
            and pair[0] <= 0.45
        ),
        key=lambda pair: (pair[0], str(manifest[pair[1]]["id"])),
    )
    if not likely or not below:
        raise RuntimeError("M2 validation lacks the predeclared browser-fixture score margins")
    selected_rows = [("likely-ai", likely[0]), ("below-threshold", below[0])]

    items = load_manifest(args.manifest, args.data_root)
    selected_items = [items[row_index] for _, (_, row_index) in selected_rows]
    tensors = np.stack([preprocess_views(item, False, ("original",))[0] for item in selected_items])
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"M2 fixture inference provider changed: {session.get_providers()}")
    reference_logits = np.asarray(
        session.run(["logits"], {"pixel_values": tensors})[0], dtype=np.float32
    ).reshape(-1)
    if not np.isfinite(reference_logits).all():
        raise ValueError("M2 fixture reference logits are non-finite")

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen UI-state fixtures: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    attribution_by_id = {
        str(row["imageId"]): row for row in json.loads(args.open_images_attribution.read_text())
    }
    frozen_items = []
    for (role, (selection_score, row_index)), item, reference_logit in zip(
        selected_rows, selected_items, reference_logits, strict=True
    ):
        row = manifest[row_index]
        source = args.data_root / str(row["path"])
        if digest(source) != row["imageSha256"]:
            raise ValueError(f"Fixture source bytes do not match the manifest: {row['id']}")
        suffix = source.suffix.lower() or ".img"
        destination = args.output_dir / f"{role}{suffix}"
        shutil.copyfile(source, destination)
        if int(row["label"]) == 0:
            image_id = str(row["id"]).split(":")[-1]
            provenance = attribution_by_id.get(image_id)
            if not provenance or provenance.get("license") != "https://creativecommons.org/licenses/by/2.0/":
                raise ValueError(f"Open Images fixture attribution is missing: {row['id']}")
        else:
            provenance = {
                "dataset": row["dataset"],
                "datasetRevision": row["datasetRevision"],
                "license": "Apache-2.0",
                "source": row["source"],
            }
        logit = float(reference_logit)
        frozen_items.append(
            {
                "role": role,
                "id": row["id"],
                "source": row["source"],
                "label": row["label"],
                "groupId": row.get("groupId"),
                "asset": destination.name,
                "assetSha256": digest(destination),
                "selectionFeatureDisplayScore": selection_score,
                "referenceLogit": logit,
                "referenceRawProbability": sigmoid(logit),
                "referenceDisplayScore": sigmoid(logit + float(calibration["intercept"])),
                "provenance": provenance,
            }
        )

    evidence = {
        "schemaVersion": 2,
        "selection": SELECTION,
        "selectorSha256": digest(Path(__file__).resolve()),
        "modelSha256": MODEL_SHA256,
        "calibrationSha256": CALIBRATION_SHA256,
        "trainingSummarySha256": TRAINING_SUMMARY_SHA256,
        "validationManifestSha256": VALIDATION_MANIFEST_SHA256,
        "validationFeatureShardSha256": VALIDATION_FEATURE_SHARD_SHA256,
        "freshFeatureRunId": FRESH_FEATURE_RUN_ID,
        "minimumLikelyAiScore": 0.80,
        "maximumBelowThresholdScore": 0.45,
        "items": frozen_items,
    }
    output = args.output_dir / "fixture-manifest.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
