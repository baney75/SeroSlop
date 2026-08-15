"""Materialize the pinned browser-parity subset from reconstructed test pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


EXPECTED_TEST_MANIFEST_SHA256 = "773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8"
EXPECTED_PARITY_IDS_SHA256 = "0f0e72ac4bd91549af10a76c494138b6cf0c22328d904134b67be82d79badf99"
SEED = 20260813


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def selection_priority(identifier: str) -> str:
    return hashlib.sha256(f"{SEED}:browser-parity:{identifier}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data/replacement-v2"))
    parser.add_argument("--test-manifest", type=Path, default=Path("benchmark/manifests/test-v2.jsonl"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("benchmark/evidence/evaluation/confirmatory-v2/prooflens-confirmatory-v2-original-predictions.jsonl"),
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=Path("benchmark/evidence/evaluation/confirmatory-v2/bootstrap.json"),
    )
    parser.add_argument("--ids", type=Path, default=Path("benchmark/manifests/parity-ids-v2.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/data/browser-parity-v2"))
    args = parser.parse_args()

    if digest(args.test_manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError("Parity preparation requires the frozen replacement-v2 test manifest")
    bootstrap = json.loads(args.bootstrap.read_text())
    if (
        bootstrap.get("schemaVersion") != 3
        or bootstrap.get("manifestSha256") != EXPECTED_TEST_MANIFEST_SHA256
        or digest(args.predictions)
        != bootstrap.get("variants", {}).get("original", {}).get("predictionsSha256")
    ):
        raise ValueError("Parity preparation requires the bootstrap-bound original predictions")

    item_rows = json_lines(args.test_manifest)
    prediction_rows = json_lines(args.predictions)
    item_ids = [str(row["id"]) for row in item_rows]
    prediction_ids = [str(row.get("id")) for row in prediction_rows]
    if (
        len(item_rows) != 600
        or len(set(item_ids)) != 600
        or len(prediction_rows) != 600
        or len(set(prediction_ids)) != 600
        or set(prediction_ids) != set(item_ids)
    ):
        raise ValueError("Parity preparation requires exact one-to-one replacement-v2 predictions")
    items = {str(row["id"]): row for row in item_rows}
    predictions = {str(row["id"]): row for row in prediction_rows}
    for identifier, prediction in predictions.items():
        item = items[identifier]
        if (
            prediction.get("variant") != "original"
            or int(prediction.get("label", -1)) != int(item["label"])
            or str(prediction.get("source")) != str(item["source"])
        ):
            raise ValueError(f"Parity prediction is stale or malformed: {identifier}")
    if digest(args.ids) != EXPECTED_PARITY_IDS_SHA256:
        raise ValueError("Parity IDs differ from the pre-score replacement-v2 selection")
    selected_ids = json.loads(args.ids.read_text())
    if not isinstance(selected_ids, list) or len(selected_ids) != 60 or len(set(selected_ids)) != 60:
        raise ValueError("Parity IDs must contain exactly 60 unique items")
    expected_ids: list[str] = []
    for label in (0, 1):
        candidates = sorted(
            (row for row in items.values() if int(row["label"]) == label),
            key=lambda row: (selection_priority(str(row["id"])), str(row["id"])),
        )
        expected_ids.extend(str(row["id"]) for row in candidates[:30])
    if selected_ids != expected_ids:
        raise ValueError("Parity IDs are not the deterministic 30-per-class selection")

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, identifier in enumerate(selected_ids):
        item = items.get(str(identifier))
        prediction = predictions.get(str(identifier))
        if item is None or prediction is None:
            raise ValueError(f"Parity item is absent from frozen evidence: {identifier}")
        source = args.data_root / str(item["path"])
        if digest(source) != item["imageSha256"]:
            raise ValueError(f"Parity image integrity mismatch: {identifier}")
        suffix = source.suffix.lower() or ".img"
        relative = Path("images") / f"{index:02d}-{item['imageSha256'][:16]}{suffix}"
        shutil.copyfile(source, args.output_dir / relative)
        rows.append({
            "id": identifier,
            "label": item["label"],
            "source": item["source"],
            "path": relative.as_posix(),
            "imageSha256": item["imageSha256"],
            "referenceRawProbability": prediction["rawProbability"],
        })

    real_count = sum(int(row["label"]) == 0 for row in rows)
    source_counts = {
        source: sum(row["source"] == source for row in rows)
        for source in sorted({str(row["source"]) for row in rows if int(row["label"]) == 1})
    }
    if real_count != 30 or source_counts != {"coxy7-infinity": 30}:
        raise ValueError(f"Unexpected parity balance: real={real_count}, synthetic={source_counts}")
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "items": len(rows), "manifestSha256": digest(manifest)}))


if __name__ == "__main__":
    main()
