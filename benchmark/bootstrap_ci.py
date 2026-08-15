"""Compute a deterministic, fail-closed class-stratified bootstrap interval."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import tempfile

from prediction_contract import require_logit_probability_consistency


VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def prediction_rows(
    path: Path,
    manifest_by_id: dict[str, dict[str, object]],
    threshold: float = 0.5,
) -> tuple[str, list[dict[str, object]]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    variants = {str(row.get("variant")) for row in rows}
    if len(variants) != 1:
        raise ValueError(f"Prediction file does not contain exactly one variant: {path}")
    variant = variants.pop()
    if variant not in VARIANTS:
        raise ValueError(f"Unknown prediction variant: {variant}")
    ids = [str(row.get("id")) for row in rows]
    if len(rows) != len(manifest_by_id) or len(set(ids)) != len(ids) or set(ids) != set(manifest_by_id):
        raise ValueError(f"Predictions do not cover the frozen manifest exactly once: {path}")
    for row in rows:
        item = manifest_by_id[str(row["id"])]
        probability = float(row.get("rawProbability", math.nan))
        logit = float(row.get("logit", math.nan))
        if (
            int(row.get("label", -1)) != int(item["label"])
            or str(row.get("source")) != str(item["source"])
            or str(row.get("groupId")) != str(item.get("groupId", item["id"]))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
            or not math.isfinite(logit)
        ):
            raise ValueError(f"Prediction row is stale or malformed: {row.get('id')}")
        try:
            require_logit_probability_consistency(logit, probability, decision_threshold=threshold)
        except ValueError as error:
            raise ValueError(f"Prediction row is stale or malformed: {row.get('id')}") from error
    return variant, rows


def clustered_correctness(rows: list[dict[str, object]], label: int, threshold: float) -> list[float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        if int(row["label"]) != label:
            continue
        group_id = str(row.get("groupId", row["id"]))
        probability = float(row["rawProbability"])
        correct = probability < threshold if label == 0 else probability >= threshold
        groups.setdefault(group_id, []).append(float(correct))
    if not groups:
        raise ValueError(f"Class {label} has no prediction groups")
    if any(len(values) != 1 for values in groups.values()):
        raise ValueError("The frozen confirmatory bootstrap requires one image per cluster")
    return [groups[key][0] for key in sorted(groups)]


def bootstrap_class(values: list[float], seed: int, variant: str, label: int, replicates: int) -> list[float]:
    successes = int(sum(values))
    count = len(values)
    probability = successes / count
    derived_seed = int.from_bytes(
        sha256(f"prooflens-bootstrap-v1:{seed}:{variant}:{label}".encode()).digest()[:16], "big"
    )
    rng = random.Random(derived_seed)
    return [rng.binomialvariate(count, probability) / count for _ in range(replicates)]


def calculate(
    prediction_paths: list[Path], manifest: Path, expected_manifest_sha256: str,
    threshold: float, seed: int, replicates: int,
) -> dict[str, object]:
    if digest(manifest) != expected_manifest_sha256:
        raise ValueError("Unexpected bootstrap manifest SHA-256")
    manifest_rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    manifest_by_id = {str(row["id"]): row for row in manifest_rows}
    if len(manifest_rows) != 600 or len(manifest_by_id) != 600:
        raise ValueError("The confirmatory manifest must contain 600 unique IDs")
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("raw-threshold must be finite and strictly between zero and one")
    if seed != 20_260_813 or replicates != 20_000:
        raise ValueError("Bootstrap seed/replicate contract changed")
    resolved_paths = [path.resolve() for path in prediction_paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("Duplicate prediction files are not allowed")

    parsed: dict[str, tuple[Path, list[dict[str, object]]]] = {}
    for path in prediction_paths:
        variant, rows = prediction_rows(path, manifest_by_id, threshold)
        if variant in parsed:
            raise ValueError(f"Duplicate prediction variant: {variant}")
        parsed[variant] = (path, rows)
    if set(parsed) != set(VARIANTS):
        raise ValueError("All four predeclared variants are required")

    output: dict[str, object] = {
        "schemaVersion": 3,
        "method": "Class-stratified one-image-cluster bootstrap with variant-derived deterministic RNG",
        "confidenceLevel": 0.95,
        "seed": seed,
        "replicates": replicates,
        "manifestSha256": expected_manifest_sha256,
        "rawProbabilityThreshold": threshold,
        "variantSeedDerivation": "first 128 bits of SHA-256(prooflens-bootstrap-v1:seed:variant:label)",
        "variants": {},
    }
    for variant in VARIANTS:
        path, rows = parsed[variant]
        real = clustered_correctness(rows, 0, threshold)
        synthetic = clustered_correctness(rows, 1, threshold)
        real_replicates = bootstrap_class(real, seed, variant, 0, replicates)
        synthetic_replicates = bootstrap_class(synthetic, seed, variant, 1, replicates)
        balanced = [
            (real_value + synthetic_value) / 2
            for real_value, synthetic_value in zip(real_replicates, synthetic_replicates, strict=True)
        ]
        output["variants"][variant] = {
            "balancedAccuracy": (sum(real) / len(real) + sum(synthetic) / len(synthetic)) / 2,
            "lower95": quantile(balanced, 0.025),
            "upper95": quantile(balanced, 0.975),
            "realRecall": sum(real) / len(real),
            "realRecallLower95": quantile(real_replicates, 0.025),
            "realRecallUpper95": quantile(real_replicates, 0.975),
            "syntheticRecall": sum(synthetic) / len(synthetic),
            "syntheticRecallLower95": quantile(synthetic_replicates, 0.025),
            "syntheticRecallUpper95": quantile(synthetic_replicates, 0.975),
            "realCount": len(real),
            "realClusters": len(real),
            "realClusterSize": 1,
            "syntheticCount": len(synthetic),
            "syntheticClusters": len(synthetic),
            "syntheticClusterSize": 1,
            "predictionsSha256": digest(path),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--raw-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_813)
    parser.add_argument("--replicates", type=int, default=20_000)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()

    output = calculate(
        args.predictions, args.manifest, args.expected_manifest_sha256,
        args.raw_threshold, args.seed, args.replicates,
    )
    encoded = (json.dumps(output, indent=2) + "\n").encode()
    if args.verify_existing:
        if not args.output.is_file() or args.output.read_bytes() != encoded:
            raise ValueError("Recomputed bootstrap evidence differs from the existing output")
    else:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite bootstrap evidence: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=f".{args.output.name}.", delete=False) as handle:
            handle.write(encoded)
            temporary = Path(handle.name)
        os.replace(temporary, args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
