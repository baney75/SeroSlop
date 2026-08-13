"""Materialize the pinned browser-parity subset from reconstructed test pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


EXPECTED_TEST_MANIFEST_SHA256 = "cd4d09fbb59d695ebc0cc4dc96f0dd17caea9e0e8865d7658c6100ef723e977f"
EXPECTED_PREDICTIONS_SHA256 = "47a237f2ee70a128b3be2b88641ad3f2bed8bdb6622443c0bd31511914f2a2e2"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data/modern-head"))
    parser.add_argument("--test-manifest", type=Path, default=Path("benchmark/manifests/test.jsonl"))
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("benchmark/predictions/cf384-rehead-sealed-test-original-predictions.jsonl"),
    )
    parser.add_argument("--ids", type=Path, default=Path("benchmark/manifests/parity-ids.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/data/browser-parity"))
    args = parser.parse_args()

    if digest(args.test_manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError("Parity preparation requires the frozen test manifest")
    if digest(args.predictions) != EXPECTED_PREDICTIONS_SHA256:
        raise ValueError("Parity preparation requires the frozen original predictions")

    items = {str(row["id"]): row for row in json_lines(args.test_manifest)}
    predictions = {str(row["id"]): row for row in json_lines(args.predictions)}
    selected_ids = json.loads(args.ids.read_text())
    if not isinstance(selected_ids, list) or len(selected_ids) != 60 or len(set(selected_ids)) != 60:
        raise ValueError("Parity IDs must contain exactly 60 unique items")

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
    if real_count != 30 or set(source_counts.values()) != {5} or len(source_counts) != 6:
        raise ValueError(f"Unexpected parity balance: real={real_count}, synthetic={source_counts}")
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_dir), "items": len(rows), "manifestSha256": digest(manifest)}))


if __name__ == "__main__":
    main()
