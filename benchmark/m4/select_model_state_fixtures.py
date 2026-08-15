"""Re-infer the two fixed development-only QA assets under the M4 candidate."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MODERN = ROOT / "benchmark/modern"


MODEL = ROOT / "benchmark/candidates/prooflens-cf384-m4/model.onnx"
CALIBRATION = ROOT / "benchmark/candidates/prooflens-cf384-m4/calibration.json"
SUMMARY = ROOT / "benchmark/candidates/prooflens-cf384-m4/validation-summary.json"
MANIFEST = ROOT / "tests/fixtures/model-states/fixture-manifest.json"
ASSET_ROOT = MANIFEST.parent
EXPECTED_FIXTURES = (
    (
        "likely-ai",
        "qwen-image-bench:d2493deb153b020cf169c7e3f57d15e4dd697038:HunyuanImage-3.0:000054_08081038.png",
        "likely-ai.png",
    ),
    ("below-threshold", "open-images:v7-validation-cvdf:validation:6f34d599f8a37eb0", "below-threshold.jpg"),
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def validate_previous_manifest(previous: dict[str, object]) -> list[dict[str, object]]:
    if previous.get("schemaVersion") not in {2, 3, 4} or not isinstance(previous.get("items"), list):
        raise ValueError("The existing model-state fixture manifest is not a reviewed two-state packet")
    items = previous["items"]
    if len(items) != len(EXPECTED_FIXTURES):
        raise ValueError("The M4 fixture packet must contain exactly two reviewed items")
    for row, expected in zip(items, EXPECTED_FIXTURES, strict=True):
        if not isinstance(row, dict):
            raise ValueError("The M4 fixture item is not an object")
        role, identifier, asset_name = expected
        asset_relative = Path(str(row.get("asset", "")))
        if (
            row.get("role") != role
            or row.get("id") != identifier
            or asset_relative != Path(asset_name)
            or asset_relative.is_absolute()
            or len(asset_relative.parts) != 1
            or ".." in asset_relative.parts
        ):
            raise ValueError("The M4 fixture identity or direct-child asset path changed")
    return items


def build_manifest() -> dict[str, object]:
    for path in (MODEL, CALIBRATION, SUMMARY, MANIFEST):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    previous = json.loads(MANIFEST.read_text())
    items = validate_previous_manifest(previous)
    import numpy as np
    import onnxruntime as ort

    if str(MODERN) not in sys.path:
        sys.path.insert(0, str(MODERN))
    from train_rehead import Item, preprocess_views
    calibration = json.loads(CALIBRATION.read_text())
    summary = json.loads(SUMMARY.read_text())
    model_hash = digest(MODEL)
    if calibration.get("modelSha256") != model_hash or summary.get("modelSha256") != model_hash:
        raise ValueError("M4 model, calibration, and training summary do not match")
    if summary.get("h3HoldoutScored") is not False or summary.get("selectionInfluencedByRegression") is not False:
        raise ValueError("M4 fixture selection boundary changed")
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise ValueError(f"Fixture inference did not use CPU only: {session.get_providers()}")
    output: list[dict[str, object]] = []
    for row in items:
        asset = ASSET_ROOT / str(row["asset"])
        if asset.parent.resolve() != ASSET_ROOT.resolve():
            raise ValueError("The M4 fixture asset escapes its reviewed directory")
        if not asset.is_file() or asset.is_symlink() or digest(asset) != row["assetSha256"]:
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
    if [row["role"] for row in output] != ["likely-ai", "below-threshold"]:
        raise ValueError("M4 fixture role order changed")
    return {
        "schemaVersion": 4,
        "selection": "The two fixed M2 development-only QA assets re-inferred under the frozen M4 residual-adapter candidate; no holdout or acceptance result was used",
        "selectorSha256": digest(Path(__file__)),
        "modelSha256": model_hash,
        "calibrationSha256": digest(CALIBRATION),
        "trainingSummarySha256": digest(SUMMARY),
        "inferenceProvider": "CPUExecutionProvider",
        "assetsUnchangedFromM2": True,
        "adapterModel": True,
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
