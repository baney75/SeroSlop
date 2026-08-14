"""Publish deterministic, pixel-free large-corpus evidence from a completed materialization."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile


ARTIFACTS = {
    "selection-plan.json": "selection-plan.json.gz",
    "train-manifest.jsonl": "train-manifest.jsonl.gz",
    "open-images-attribution.jsonl": "open-images-attribution.jsonl.gz",
    "rejects.jsonl": "rejects.jsonl.gz",
    "evaluation-perceptual-hashes.json": "evaluation-perceptual-hashes.json.gz",
}


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def atomic_write(destination: Path, value: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data/large-head"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/evidence/large"))
    parser.add_argument("--recipe", type=Path, default=Path("benchmark/large/recipe.json"))
    args = parser.parse_args()

    summary_path = args.data_root / "selection-summary.json"
    summary = json.loads(summary_path.read_text())
    plan_path = args.data_root / "selection-plan.json"
    if (
        summary.get("schemaVersion") != 1
        or summary.get("recipeSha256") != digest_bytes(args.recipe.read_bytes())
        or summary.get("planSha256") != digest_bytes(plan_path.read_bytes())
    ):
        raise ValueError("Selection summary does not bind the current recipe and materialization plan")
    expected_hashes = {
        "selection-plan.json": summary["planSha256"],
        "train-manifest.jsonl": summary["manifestSha256"],
        "open-images-attribution.jsonl": summary["attributionSha256"],
        "rejects.jsonl": summary["rejectsSha256"],
        "evaluation-perceptual-hashes.json": summary["evaluationPerceptualHashesSha256"],
    }
    published = {}
    for source_name, destination_name in ARTIFACTS.items():
        source = args.data_root / source_name
        value = source.read_bytes()
        if digest_bytes(value) != expected_hashes[source_name]:
            raise ValueError(f"Selection summary does not bind {source}")
        compressed = gzip.compress(value, compresslevel=9, mtime=0)
        destination = args.output_dir / destination_name
        atomic_write(destination, compressed)
        if gzip.decompress(destination.read_bytes()) != value:
            raise ValueError(f"Published gzip did not round-trip: {destination}")
        published[destination_name] = {
            "expandedSha256": expected_hashes[source_name],
            "compressedSha256": digest_bytes(compressed),
            "compressedBytes": len(compressed),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_summary = args.output_dir / ".selection-summary.json.partial"
    shutil.copyfile(summary_path, temporary_summary)
    if temporary_summary.read_bytes() != summary_path.read_bytes():
        raise ValueError("Selection summary copy verification failed")
    os.replace(temporary_summary, args.output_dir / "selection-summary.json")
    print(json.dumps({"schemaVersion": 1, "artifacts": published}, indent=2))


if __name__ == "__main__":
    main()
