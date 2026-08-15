"""Select a deterministic, class-balanced browser-parity subset."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


SEED = 20260813
EXPECTED_MANIFEST_SHA256 = "773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8"
EXPECTED_SOURCES = {"stockimages-cc0": 300, "coxy7-infinity": 300}


def priority(identifier: str) -> str:
    return sha256(f"{SEED}:browser-parity:{identifier}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("benchmark/manifests/test-v2.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/manifests/parity-ids-v2.json"))
    parser.add_argument("--per-class", type=int, default=30)
    args = parser.parse_args()

    manifest_bytes = args.manifest.read_bytes()
    if sha256(manifest_bytes).hexdigest() != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Parity selection requires the frozen replacement-v2 manifest")
    rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line]
    if len(rows) != 600:
        raise ValueError("Replacement-v2 manifest must contain exactly 600 rows")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Confirmatory manifest contains duplicate IDs")
    source_counts = {
        source: sum(str(row["source"]) == source for row in rows)
        for source in sorted({str(row["source"]) for row in rows})
    }
    label_counts = {label: sum(int(row["label"]) == label for row in rows) for label in (0, 1)}
    if source_counts != EXPECTED_SOURCES or label_counts != {0: 300, 1: 300}:
        raise ValueError("Replacement-v2 manifest allocation changed")
    selected: list[str] = []
    for label in (0, 1):
        candidates = [row for row in rows if int(row["label"]) == label]
        candidates.sort(key=lambda row: (priority(str(row["id"])), str(row["id"])))
        if len(candidates) < args.per_class:
            raise ValueError(f"Class {label} has fewer than {args.per_class} candidates")
        selected.extend(str(row["id"]) for row in candidates[: args.per_class])
    if len(selected) != args.per_class * 2 or len(set(selected)) != len(selected):
        raise AssertionError("Parity selection is not balanced and unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "items": len(selected), "perClass": args.per_class}))


if __name__ == "__main__":
    main()
