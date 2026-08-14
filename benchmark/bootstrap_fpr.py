"""Compute Wilson-score false-positive intervals for a negative challenge."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile

from prediction_contract import require_logit_probability_consistency


VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def interval(successes: int, count: int) -> tuple[float, float]:
    if count < 1:
        raise ValueError("Wilson interval requires observations")
    z = 1.959963984540054
    probability = successes / count
    denominator = 1 + z * z / count
    center = (probability + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(probability * (1 - probability) / count + z * z / (4 * count * count)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--raw-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    if digest(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("Unexpected web-negative manifest SHA-256")
    if not math.isfinite(args.raw_threshold) or not 0 < args.raw_threshold < 1:
        raise ValueError("raw-threshold must be finite and strictly between zero and one")
    manifest_rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    manifest_by_id = {str(row["id"]): row for row in manifest_rows}
    if len(manifest_rows) != 319 or len(manifest_by_id) != 319 or any(int(row["label"]) != 0 for row in manifest_rows):
        raise ValueError("The frozen web-negative manifest must contain 319 unique real-image IDs")
    resolved_paths = [path.resolve() for path in args.predictions]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("Duplicate prediction files are not allowed")

    output: dict[str, object] = {
        "schemaVersion": 2,
        "method": "Wilson score interval for false-positive proportions",
        "confidenceLevel": 0.95,
        "manifestSha256": args.expected_manifest_sha256,
        "rawProbabilityThreshold": args.raw_threshold,
        "variants": {},
    }
    seen_variants: set[str] = set()
    for path in args.predictions:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        variants = {str(row["variant"]) for row in rows}
        if len(variants) != 1:
            raise ValueError(f"Prediction file does not contain one variant: {path}")
        variant = variants.pop()
        ids = [str(row.get("id")) for row in rows]
        if variant not in VARIANTS or variant in seen_variants:
            raise ValueError(f"Duplicate or unknown prediction variant: {variant}")
        if len(rows) != 319 or len(set(ids)) != 319 or set(ids) != set(manifest_by_id):
            raise ValueError(f"Predictions do not cover the frozen web-negative manifest: {path}")
        for row in rows:
            item = manifest_by_id[str(row["id"])]
            probability = float(row.get("rawProbability", math.nan))
            logit = float(row.get("logit", math.nan))
            if (
                int(row.get("label", -1)) != 0
                or str(row.get("source")) != str(item["source"])
                or str(row.get("groupId")) != str(item.get("groupId", item["id"]))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
                or not math.isfinite(logit)
            ):
                raise ValueError(f"Prediction row is stale or malformed: {row.get('id')}")
            try:
                require_logit_probability_consistency(logit, probability)
            except ValueError as error:
                raise ValueError(f"Prediction row is stale or malformed: {row.get('id')}") from error
        seen_variants.add(variant)
        values = [float(row["rawProbability"]) >= args.raw_threshold for row in rows]
        successes = sum(values)
        lower, upper = interval(successes, len(values))
        sources: dict[str, object] = {}
        for source in sorted({str(row["source"]) for row in rows}):
            selected = [
                float(row["rawProbability"]) >= args.raw_threshold
                for row in rows
                if str(row["source"]) == source
            ]
            source_successes = sum(selected)
            source_lower, source_upper = interval(source_successes, len(selected))
            sources[source] = {
                "count": len(selected),
                "falsePositiveRate": source_successes / len(selected),
                "lower95": source_lower,
                "upper95": source_upper,
            }
        output["variants"][variant] = {
            "count": len(values),
            "falsePositiveRate": successes / len(values),
            "lower95": lower,
            "upper95": upper,
            "bySource": sources,
            "predictionsSha256": digest(path),
        }
    if seen_variants != set(VARIANTS):
        raise ValueError("All four predeclared variants are required")
    encoded = (json.dumps(output, indent=2) + "\n").encode()
    if args.verify_existing:
        if not args.output.is_file() or args.output.read_bytes() != encoded:
            raise ValueError("Recomputed Wilson evidence differs from the existing output")
    else:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite Wilson evidence: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=f".{args.output.name}.", delete=False) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        os.replace(temporary, args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
