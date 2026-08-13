"""Compute deterministic stratified bootstrap CIs from frozen predictions."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--raw-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--replicates", type=int, default=20_000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output: dict[str, object] = {
        "schemaVersion": 1,
        "method": "Stratified nonparametric bootstrap over real and synthetic images",
        "confidenceLevel": 0.95,
        "seed": args.seed,
        "replicates": args.replicates,
        "rawProbabilityThreshold": args.raw_threshold,
        "variants": {},
    }
    for path in args.predictions:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        variants = {str(row["variant"]) for row in rows}
        if len(variants) != 1:
            raise ValueError(f"Prediction file does not contain exactly one variant: {path}")
        variant = variants.pop()
        real = np.asarray(
            [float(row["rawProbability"]) < args.raw_threshold for row in rows if int(row["label"]) == 0],
            dtype=np.float64,
        )
        synthetic = np.asarray(
            [float(row["rawProbability"]) >= args.raw_threshold for row in rows if int(row["label"]) == 1],
            dtype=np.float64,
        )
        if not len(real) or not len(synthetic):
            raise ValueError(f"Both classes are required: {path}")
        real_samples = rng.integers(0, len(real), size=(args.replicates, len(real)))
        synthetic_samples = rng.integers(0, len(synthetic), size=(args.replicates, len(synthetic)))
        balanced = (real[real_samples].mean(axis=1) + synthetic[synthetic_samples].mean(axis=1)) / 2
        output["variants"][variant] = {
            "balancedAccuracy": float((real.mean() + synthetic.mean()) / 2),
            "lower95": float(np.quantile(balanced, 0.025)),
            "upper95": float(np.quantile(balanced, 0.975)),
            "realCount": len(real),
            "syntheticCount": len(synthetic),
            "predictionsSha256": digest(path),
        }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
