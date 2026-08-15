"""Re-infer the two fixed QA assets under M3 and bind their visible states."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
MODERN = ROOT / "benchmark/modern"
if str(MODERN) not in sys.path:
    sys.path.insert(0, str(MODERN))

from train_rehead import Item, preprocess_views  # noqa: E402


MODEL = ROOT / "benchmark/candidates/prooflens-cf384-m3/model.onnx"
CALIBRATION = ROOT / "benchmark/candidates/prooflens-cf384-m3/calibration.json"
SUMMARY = ROOT / "benchmark/candidates/prooflens-cf384-m3/validation-summary.json"
MANIFEST = ROOT / "tests/fixtures/model-states/fixture-manifest.json"
ASSET_ROOT = MANIFEST.parent


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def build_manifest() -> dict[str, object]:
    for path in (MODEL, CALIBRATION, SUMMARY, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    previous = json.loads(MANIFEST.read_text())
    if previous.get("schemaVersion") not in {2, 3} or not isinstance(previous.get("items"), list):
        raise ValueError("The existing model-state fixture manifest is not a reviewed two-state packet")
    calibration = json.loads(CALIBRATION.read_text())
    summary = json.loads(SUMMARY.read_text())
    if calibration.get("modelSha256") != digest(MODEL) or summary.get("model", {}).get("sha256") != digest(MODEL):
        raise ValueError("M3 model, calibration, and training summary do not match")
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError(f"Fixture inference did not use CPU only: {session.get_providers()}")
    output: list[dict[str, object]] = []
    for row in previous["items"]:
        asset = ASSET_ROOT / str(row["asset"])
        if not asset.is_file() or digest(asset) != row["assetSha256"]:
            raise ValueError(f"Model-state asset changed: {row.get('asset')}")
        item = Item(
            id=str(row["id"]),
            path=asset,
            image_sha256=str(row["assetSha256"]),
            label=int(row["label"]),
            source=str(row["source"]),
        )
        pixels = np.stack(preprocess_views(item, training=False, variants=("original",)))
        logit = float(session.run(["logits"], {"pixel_values": pixels})[0].reshape(-1)[0])
        raw_probability = sigmoid(logit)
        display_score = sigmoid(logit + float(calibration["intercept"]))
        role = str(row["role"])
        if role == "likely-ai" and display_score < 0.8:
            raise ValueError("The fixed likely-AI fixture no longer has a stable high-score margin")
        if role == "below-threshold" and display_score > 0.45:
            raise ValueError("The fixed below-threshold fixture no longer has a stable low-score margin")
        output.append({
            key: row[key]
            for key in ("role", "id", "source", "label", "groupId", "asset", "assetSha256", "provenance")
        } | {
            "referenceLogit": logit,
            "referenceRawProbability": raw_probability,
            "referenceDisplayScore": display_score,
        })
    return {
        "schemaVersion": 3,
        "selection": "The two fixed M2 development-only QA assets re-inferred under the frozen M3 candidate; no holdout or acceptance result was used",
        "selectorSha256": digest(Path(__file__)),
        "modelSha256": digest(MODEL),
        "calibrationSha256": digest(CALIBRATION),
        "trainingSummarySha256": digest(SUMMARY),
        "inferenceProvider": "CPUExecutionProvider",
        "assetsUnchangedFromM2": True,
        "minimumLikelyAiScore": 0.8,
        "maximumBelowThresholdScore": 0.45,
        "items": output,
    }


def main() -> None:
    manifest = build_manifest()
    print(json.dumps({
        "modelSha256": manifest["modelSha256"],
        "likelyAiScore": manifest["items"][0]["referenceDisplayScore"],
        "belowThresholdScore": manifest["items"][1]["referenceDisplayScore"],
        "written": False,
        "policy": "pass",
    }, indent=2))


if __name__ == "__main__":
    main()
