"""Materialize the pinned browser-parity subset from reconstructed test pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


EXPECTED_TEST_MANIFEST_SHA256 = "28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae"
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
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data"))
    parser.add_argument("--test-manifest", type=Path, default=Path("benchmark/manifests/test.jsonl"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-original-predictions.jsonl"),
    )
    parser.add_argument(
        "--bootstrap",
        type=Path,
        default=Path("benchmark/evidence/evaluation/confirmatory/bootstrap.json"),
    )
    parser.add_argument("--ids", type=Path, default=Path("benchmark/manifests/parity-ids.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/data/browser-parity"))
    args = parser.parse_args()

    if digest(args.test_manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError("Parity preparation requires the frozen test manifest")
    bootstrap = json.loads(args.bootstrap.read_text())
    if digest(args.predictions) != bootstrap.get("variants", {}).get("original", {}).get("predictionsSha256"):
        raise ValueError("Parity preparation requires the bootstrap-bound original predictions")

    items = {str(row["id"]): row for row in json_lines(args.test_manifest)}
    predictions = {str(row["id"]): row for row in json_lines(args.predictions)}
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
    if real_count != 30 or source_counts != {"kling_v2_1": 30}:
        raise ValueError(f"Unexpected parity balance: real={real_count}, synthetic={source_counts}")
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "items": len(rows), "manifestSha256": digest(manifest)}))


if __name__ == "__main__":
    main()
