"""Select a deterministic, class-balanced browser-parity subset."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path


SEED = 20260813


def priority(identifier: str) -> str:
    return sha256(f"{SEED}:browser-parity:{identifier}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("benchmark/manifests/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("benchmark/manifests/parity-ids.json"))
    parser.add_argument("--per-class", type=int, default=30)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Confirmatory manifest contains duplicate IDs")
    selected: list[str] = []
    for label in (0, 1):
        candidates = [row for row in rows if int(row["label"]) == label]
        candidates.sort(key=lambda row: (priority(str(row["id"])), str(row["id"])))
        if len(candidates) < args.per_class:
            raise ValueError(f"Class {label} has fewer than {args.per_class} candidates")
        selected.extend(str(row["id"]) for row in candidates[: args.per_class])
    if len(selected) != args.per_class * 2 or len(set(selected)) != len(selected):
        raise AssertionError("Parity selection is not balanced and unique")
    args.output.write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "items": len(selected), "perClass": args.per_class}))


if __name__ == "__main__":
    main()
