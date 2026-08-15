"""Independently rederive and verify the public M2 selection packet."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m2.prepare import RECIPE_PATH, derive_packet  # noqa: E402
from benchmark.recovery_v3.prepare import reject_symlink_components, validate_data_root  # noqa: E402


COMPRESSED = {"train-manifest.jsonl", "stock-selection.json", "rejects.jsonl"}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_public_packet(recipe: dict[str, object], packet: dict[str, bytes]) -> None:
    evidence_root = ROOT / str(recipe["output"]["evidenceRoot"])
    for name, expected in packet.items():
        path = evidence_root / (f"{name}.gz" if name in COMPRESSED else name)
        actual = gzip.decompress(path.read_bytes()) if name in COMPRESSED else path.read_bytes()
        if actual != expected:
            raise ValueError(f"Published M2 evidence does not match isolated derivation: {path}")


def verify_pixels(recipe: dict[str, object], packet: dict[str, bytes]) -> None:
    data_root = validate_data_root(ROOT / str(recipe["output"]["dataRoot"]))
    for manifest_name in ("train-manifest.jsonl", "validation-manifest.jsonl"):
        rows = [json.loads(line) for line in packet[manifest_name].splitlines() if line]
        for index, row in enumerate(rows, start=1):
            path = reject_symlink_components(
                data_root,
                data_root / str(row["path"]),
                label=f"M2 {manifest_name} pixel",
            )
            if not path.is_file() or digest(path) != row["imageSha256"]:
                raise ValueError(f"M2 pixel integrity mismatch: {row['id']}")
            if index % 10_000 == 0:
                print(f"verified {manifest_name}: {index}/{len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-pixels", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(RECIPE_PATH.read_text())
    packet = derive_packet(recipe, materialize_pixels=False, allow_download=False)
    verify_public_packet(recipe, packet)
    if args.verify_pixels:
        verify_pixels(recipe, packet)
    summary = json.loads(packet["selection-summary.json"])
    print(
        json.dumps(
            {
                "policy": "pass",
                "trainingImages": summary["counts"]["total"],
                "trainingViews": summary["counts"]["trainingFeatureViews"],
                "validationImages": summary["counts"]["validation"],
                "stockTraining": summary["sourceCounts"]["stockimages-cc0"],
                "consumedRowsUsed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
