"""Freeze two high-margin UI fixtures from validation-only predictions."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("benchmark/evidence/evaluation/validation/prooflens-validation-original-predictions.jsonl"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("benchmark/manifests/validation.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data/modern-head"))
    parser.add_argument(
        "--completion",
        type=Path,
        default=Path("benchmark/evidence/evaluation/validation/prooflens-validation-complete.json"),
    )
    parser.add_argument(
        "--open-images-attribution",
        type=Path,
        default=Path("benchmark/manifests/open-images-attribution.json"),
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("benchmark/evidence/large/calibration.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tests/fixtures/model-states"))
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    completion = json.loads(args.completion.read_text())
    manifest_sha256 = digest(args.manifest)
    predictions_sha256 = digest(args.predictions)
    if manifest_sha256 != "41be10ef876ecef0635744ed29677a1888a7759cc8060dc7a392f76f83ab263b":
        raise ValueError("UI-state fixtures require the frozen validation manifest")
    if (
        completion.get("protocol") != "validation"
        or completion.get("modelSha256") != calibration.get("modelSha256")
        or completion.get("manifestSha256") != manifest_sha256
        or completion.get("calibrationSha256") != digest(args.calibration)
        or completion.get("files", {}).get(args.predictions.name) != predictions_sha256
    ):
        raise ValueError("Validation predictions are not completion-marker-bound")
    rows = {str(row["id"]): row for row in json_lines(args.manifest)}
    predictions = json_lines(args.predictions)
    if len(predictions) != len(rows) or len({str(row["id"]) for row in predictions}) != len(rows):
        raise ValueError("Original predictions do not cover the validation manifest exactly once")
    attribution_by_id = {
        str(row["imageId"]): row
        for row in json.loads(args.open_images_attribution.read_text())
    }
    scored: list[tuple[dict[str, object], dict[str, object], float]] = []
    for prediction in predictions:
        item = rows.get(str(prediction["id"]))
        if item is None or int(item["label"]) != int(prediction["label"]):
            raise ValueError(f"Prediction does not match validation manifest: {prediction['id']}")
        score = sigmoid(float(prediction["logit"]) + float(calibration["intercept"]))
        scored.append((item, prediction, score))

    likely_candidates = sorted(
        (row for row in scored if int(row[0]["label"]) == 1 and row[2] >= 0.80),
        key=lambda row: (-row[2], str(row[0]["id"])),
    )
    below_candidates = sorted(
        (row for row in scored if int(row[0]["label"]) == 0 and row[2] <= 0.45),
        key=lambda row: (row[2], str(row[0]["id"])),
    )
    if not likely_candidates or not below_candidates:
        raise RuntimeError("Validation predictions lack the predeclared QA-fixture score margins")

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen UI-state fixtures: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_items = []
    for role, selected in (("likely-ai", likely_candidates[0]), ("below-threshold", below_candidates[0])):
        item, prediction, score = selected
        source = args.data_root / str(item["path"])
        if digest(source) != item["imageSha256"]:
            raise ValueError(f"Fixture source bytes do not match the manifest: {item['id']}")
        suffix = source.suffix.lower() or ".img"
        destination = args.output_dir / f"{role}{suffix}"
        shutil.copyfile(source, destination)
        if int(item["label"]) == 0:
            image_id = str(item["id"]).split(":")[-1]
            provenance = attribution_by_id.get(image_id)
            if not provenance or provenance.get("license") != "https://creativecommons.org/licenses/by/2.0/":
                raise ValueError(f"Open Images fixture attribution is missing: {item['id']}")
        else:
            provenance = {
                "dataset": item["dataset"],
                "datasetRevision": item["datasetRevision"],
                "license": "Apache-2.0",
                "source": item["source"],
            }
        frozen_items.append({
            "role": role,
            "id": item["id"],
            "source": item["source"],
            "label": item["label"],
            "groupId": item.get("groupId"),
            "asset": destination.name,
            "assetSha256": digest(destination),
            "referenceLogit": prediction["logit"],
            "referenceRawProbability": prediction["rawProbability"],
            "referenceDisplayScore": score,
            "provenance": provenance,
        })

    evidence = {
        "schemaVersion": 1,
        "selection": "Pre-confirmatory validation QA fixtures: deterministic maximum-margin synthetic and real items at fixed weights/calibration",
        "modelSha256": calibration["modelSha256"],
        "calibrationSha256": digest(args.calibration),
        "validationManifestSha256": manifest_sha256,
        "validationCompletionSha256": digest(args.completion),
        "originalPredictionsSha256": predictions_sha256,
        "minimumLikelyAiScore": 0.80,
        "maximumBelowThresholdScore": 0.45,
        "items": frozen_items,
    }
    output = args.output_dir / "fixture-manifest.json"
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
