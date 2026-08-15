"""Deterministically prepare the M2 StockImages hard-negative development packet."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m2.contracts import deterministic_gzip, priority
from benchmark.recovery_v3.prepare import (
    PerceptualIndex,
    admit,
    bytes_digest,
    historical_evidence,
    load_stock_candidates,
    public_candidate,
    qualify,
    reject_symlink_components,
    safe_output_path,
    validate_data_root,
    write_data_atomic,
)


RECIPE_PATH = ROOT / "benchmark/m2/recipe.json"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_json(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
        for row in rows
    )


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_base_training(recipe: dict[str, Any]) -> tuple[list[dict[str, Any]], bytes]:
    config = recipe["baseTraining"]
    path = ROOT / str(config["manifest"])
    compressed = path.read_bytes()
    if bytes_digest(compressed) != config["compressedSha256"]:
        raise ValueError("M1 training manifest compressed bytes changed")
    expanded = gzip.decompress(compressed)
    if bytes_digest(expanded) != config["expandedSha256"]:
        raise ValueError("M1 training manifest expanded bytes changed")
    rows = [json.loads(line) for line in expanded.splitlines() if line]
    if len(rows) != int(config["items"]):
        raise ValueError("M1 training manifest row count changed")
    summary_path = ROOT / str(config["selectionSummary"])
    if digest(summary_path) != config["selectionSummarySha256"]:
        raise ValueError("M1 training selection summary changed")
    return rows, expanded


def load_base_validation(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    config = recipe["baseValidation"]
    path = ROOT / str(config["manifest"])
    if digest(path) != config["sha256"]:
        raise ValueError("Base validation manifest changed")
    rows = load_jsonl(path)
    if len(rows) != int(config["items"]):
        raise ValueError("Base validation manifest row count changed")
    return rows


def add_consumed_exclusions(
    recipe: dict[str, Any],
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for config in recipe["consumedEvaluationExclusions"]:
        path = ROOT / str(config["path"])
        if digest(path) != config["sha256"]:
            raise ValueError(f"Consumed evaluation manifest changed: {path}")
        rows = load_jsonl(path)
        for row in rows:
            identifier = str(row["id"])
            image_hash = str(row["imageSha256"])
            group = str(row.get("sourceGroupId") or row.get("groupId") or identifier)
            dhash = str(row.get("perceptualDhash64", ""))
            identifier_is_new = identifier not in ids
            ids.add(identifier)
            hashes.add(image_hash)
            groups.add(group)
            if dhash and identifier_is_new:
                perceptual.add(dhash, identifier)
        evidence.append(
            {
                "path": str(config["path"]),
                "sha256": str(config["sha256"]),
                "rows": len(rows),
                "dataRoot": str(config["dataRoot"]),
                "role": str(config["role"]),
            }
        )
    return evidence


def selected_row(candidate: dict[str, Any], *, split: str, relative_path: str) -> dict[str, Any]:
    row = public_candidate(candidate)
    row["path"] = relative_path
    row["split"] = split
    return row


def stock_filename(candidate: dict[str, Any]) -> str:
    return (
        f"{int(candidate['globalOffset']):05d}-"
        f"{str(candidate['imageSha256'])[:16]}{candidate['extension']}"
    )


def select_stock_partitions(
    recipe: dict[str, Any],
    candidates: list[dict[str, Any]],
    source_rejects: list[dict[str, object]],
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
    *,
    output_root: Path,
    materialize_pixels: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, object]], int, int]:
    config = recipe["stockImagesSource"]
    threshold = int(recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"])
    base_rejects: list[dict[str, object]] = list(source_rejects)
    eligible: list[dict[str, Any]] = []
    for original in candidates:
        issue = qualify(original, ids, hashes, groups, perceptual, threshold)
        if issue is None:
            eligible.append(original)
        else:
            base_rejects.append({"candidateId": original["id"], "phase": "eligibility", **issue})
    if len(eligible) != int(config["expectedEligibleRows"]):
        raise ValueError(
            f"Expected {config['expectedEligibleRows']} eligible StockImages rows, received {len(eligible)}"
        )

    development: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    cross_rejected_ids: set[str] = set()
    for original in sorted(
        eligible,
        key=lambda row: (priority(str(config["developmentPriorityNamespace"]), str(row["id"])), str(row["id"])),
    ):
        candidate = dict(original)
        candidate["selectionPriority"] = priority(
            str(config["developmentPriorityNamespace"]), str(candidate["id"])
        )
        issue = qualify(candidate, ids, hashes, groups, perceptual, threshold)
        if issue is not None:
            base_rejects.append({"candidateId": candidate["id"], "phase": "development", **issue})
            cross_rejected_ids.add(str(candidate["id"]))
            continue
        filename = stock_filename(candidate)
        relative = f"validation/real/stockimages-cc0/{filename}"
        if materialize_pixels:
            write_data_atomic(output_root, relative, candidate["imageBytes"])
        admit(candidate, ids, hashes, groups, perceptual)
        selected_ids.add(str(candidate["id"]))
        development.append(selected_row(candidate, split="validation", relative_path=relative))
        if len(development) == int(config["developmentTarget"]):
            break
    if len(development) != int(config["developmentTarget"]):
        raise ValueError("The frozen StockImages development target is unavailable")

    training: list[dict[str, Any]] = []
    for original in sorted(
        (
            row for row in eligible
            if str(row["id"]) not in selected_ids and str(row["id"]) not in cross_rejected_ids
        ),
        key=lambda row: (priority(str(config["trainingPriorityNamespace"]), str(row["id"])), str(row["id"])),
    ):
        candidate = dict(original)
        candidate["selectionPriority"] = priority(
            str(config["trainingPriorityNamespace"]), str(candidate["id"])
        )
        issue = qualify(candidate, ids, hashes, groups, perceptual, threshold)
        if issue is not None:
            base_rejects.append({"candidateId": candidate["id"], "phase": "training", **issue})
            cross_rejected_ids.add(str(candidate["id"]))
            continue
        filename = stock_filename(candidate)
        relative = f"train/real/stockimages-cc0/{filename}"
        if materialize_pixels:
            write_data_atomic(output_root, relative, candidate["imageBytes"])
        admit(candidate, ids, hashes, groups, perceptual)
        training.append(selected_row(candidate, split="train", relative_path=relative))
    if len(training) != int(config["trainingTarget"]):
        raise ValueError(
            f"Expected {config['trainingTarget']} StockImages training rows, received {len(training)}"
        )
    if len(cross_rejected_ids) != int(config["expectedCrossCandidateRejects"]):
        raise ValueError(
            f"Expected {config['expectedCrossCandidateRejects']} new cross-candidate rejects, "
            f"received {len(cross_rejected_ids)}"
        )
    if len(training) + len(development) + len(cross_rejected_ids) != len(eligible):
        raise ValueError("The frozen StockImages eligible pool was not allocated or rejected exactly")
    return training, development, base_rejects, len(eligible), len(cross_rejected_ids)


def hardlink_verified(
    source_root: Path,
    destination_root: Path,
    relative: str,
    expected_sha256: str,
) -> None:
    source_root = validate_data_root(source_root)
    destination_root = validate_data_root(destination_root)
    source = reject_symlink_components(source_root, source_root / relative, label="source pixel")
    if not source.is_file() or digest(source) != expected_sha256:
        raise ValueError(f"Source pixel integrity changed: {source}")
    destination = safe_output_path(destination_root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = safe_output_path(destination_root, relative)
    if destination.exists():
        if not destination.is_file() or digest(destination) != expected_sha256:
            raise ValueError(f"Existing M2 pixel integrity changed: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.link")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        os.link(source, temporary)
        if digest(temporary) != expected_sha256:
            raise ValueError(f"Hard-linked M2 pixel failed verification: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_base_pixels(
    recipe: dict[str, Any],
    training_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    output_root: Path,
) -> None:
    training_root = ROOT / str(recipe["baseTraining"]["dataRoot"])
    validation_root = ROOT / str(recipe["baseValidation"]["dataRoot"])
    for index, row in enumerate(training_rows, start=1):
        hardlink_verified(training_root, output_root, str(row["path"]), str(row["imageSha256"]))
        if index % 5000 == 0:
            print(f"linked base training pixels: {index}/{len(training_rows)}", flush=True)
    for row in validation_rows:
        hardlink_verified(validation_root, output_root, str(row["path"]), str(row["imageSha256"]))


def reindex(rows: Iterable[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = dict(original)
        row["rowIndex"] = index
        row["split"] = split
        output.append(row)
    return output


def derive_packet(
    recipe: dict[str, Any],
    *,
    materialize_pixels: bool,
    allow_download: bool,
) -> dict[str, bytes]:
    output_root = validate_data_root(ROOT / str(recipe["output"]["dataRoot"]))
    source_root = validate_data_root(ROOT / str(recipe["stockImagesSource"]["sourceDataRoot"]))
    if materialize_pixels:
        output_root.mkdir(parents=True, exist_ok=True)
        output_root = validate_data_root(output_root)

    base_training, _ = load_base_training(recipe)
    base_validation = load_base_validation(recipe)
    ids, hashes, groups, perceptual, historical_counts = historical_evidence(recipe)
    consumed_evidence = add_consumed_exclusions(recipe, ids, hashes, groups, perceptual)
    stock_candidates, source_rejects = load_stock_candidates(
        recipe,
        source_root,
        allow_download=allow_download,
    )
    stock_training, stock_development, rejects, eligible_count, cross_reject_count = select_stock_partitions(
        recipe,
        stock_candidates,
        source_rejects,
        ids,
        hashes,
        groups,
        perceptual,
        output_root=output_root,
        materialize_pixels=materialize_pixels,
    )
    if materialize_pixels:
        materialize_base_pixels(recipe, base_training, base_validation, output_root)

    train_rows = reindex([*base_training, *sorted(stock_training, key=lambda row: str(row["id"]))], split="train")
    validation_rows = reindex(
        [*base_validation, *sorted(stock_development, key=lambda row: str(row["id"]))],
        split="validation",
    )
    train_bytes = jsonl_bytes(train_rows)
    validation_bytes = jsonl_bytes(validation_rows)
    rejects_bytes = jsonl_bytes(sorted(rejects, key=lambda row: (str(row.get("candidateId", "")), str(row.get("phase", "")))))

    source_counts = dict(sorted(Counter(str(row["source"]) for row in train_rows).items()))
    class_counts = {
        "real": sum(int(row["label"]) == 0 for row in train_rows),
        "synthetic": sum(int(row["label"]) == 1 for row in train_rows),
    }
    if source_counts != recipe["expectedSourceCounts"] or class_counts != recipe["expectedClassCounts"]:
        raise ValueError("M2 source or class counts changed")

    evaluation_exclusions = [
        {
            "path": "benchmark/evidence/m2/validation-manifest.jsonl",
            "sha256": bytes_digest(validation_bytes),
            "rows": len(validation_rows),
            "dataRoot": "benchmark/data/m2-head",
            "role": str(recipe["evaluationManifests"][0]["role"]),
        },
        *[
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "rows": item["rows"],
                "dataRoot": item["dataRoot"],
                "role": item["role"],
            }
            for item in consumed_evidence
        ],
    ]
    old_review = ROOT / str(recipe["perceptualOverlapReview"])
    training_review = ROOT / str(recipe["trainingPerceptualOverlapReview"])
    old_review_rows = json.loads(old_review.read_text()).get("items", [])
    training_review_rows = json.loads(training_review.read_text()).get("items", [])

    selection = {
        "schemaVersion": 1,
        "name": recipe["name"],
        "scoreIndependent": True,
        "consumedV2RowsUsedForGradients": False,
        "consumedV2RowsUsedForDevelopmentMetrics": False,
        "recipeSha256": digest(RECIPE_PATH),
        "historicalPerceptualIndexSha256": recipe["historicalExclusions"]["historicalPerceptualIndexSha256"],
        "historicalCounts": historical_counts,
        "sourceShardLocks": recipe["stockImagesSource"]["shards"],
        "candidateRows": int(recipe["stockImagesSource"]["candidateRows"]),
        "eligibleRows": eligible_count,
        "trainingRows": len(stock_training),
        "developmentRows": len(stock_development),
        "crossCandidateRejectedRows": cross_reject_count,
        "trainingPriorityNamespace": recipe["stockImagesSource"]["trainingPriorityNamespace"],
        "developmentPriorityNamespace": recipe["stockImagesSource"]["developmentPriorityNamespace"],
        "trainingIds": sorted(str(row["id"]) for row in stock_training),
        "developmentIds": sorted(str(row["id"]) for row in stock_development),
        "rejectedRows": len(rejects),
        "overlap": {
            "id": 0,
            "imageSha256": 0,
            "sourceGroupId": 0,
            "perceptualDhash64AtOrBelow8": 0,
        },
    }
    selection_bytes = canonical_json(selection, pretty=True)
    review_bytes = canonical_json(
        {
            "schemaVersion": 1,
            "algorithm": recipe["overlapPolicy"]["perceptualDhash64"]["algorithm"],
            "maximumHammingDistance": recipe["perceptualDuplicateHammingThreshold"],
            "policy": "Every new StockImages near match is rejected; M2 admits no new perceptual exceptions.",
            "retainedPairs": [],
        },
        pretty=True,
    )
    attribution_bytes = canonical_json(
        {
            "schemaVersion": 1,
            "dataset": recipe["stockImagesSource"]["dataset"],
            "revision": recipe["stockImagesSource"]["revision"],
            "sourceReportedLicense": recipe["stockImagesSource"]["sourceReportedLicense"],
            "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/",
            "sourceMaterialUrl": "https://huggingface.co/datasets/KoalaAI/StockImages-CC0/tree/206f3575579f1187548c6f47042ae9174c0a51fc",
            "notice": "The fixed dataset card reports CC0-1.0. This uploader-provided statement is not independent verification of contributor ownership or other third-party rights. Source pixels remain outside Git.",
            "uses": {"training": len(stock_training), "development": len(stock_development)},
        },
        pretty=True,
    )
    summary = {
        "schemaVersion": 2,
        "name": recipe["name"],
        "recipeSha256": digest(RECIPE_PATH),
        "baseTrainingManifestSha256": recipe["baseTraining"]["expandedSha256"],
        "manifestSha256": bytes_digest(train_bytes),
        "validationManifestSha256": bytes_digest(validation_bytes),
        "stockSelectionSha256": bytes_digest(selection_bytes),
        "rejectsSha256": bytes_digest(rejects_bytes),
        "attributionSha256": bytes_digest(attribution_bytes),
        "newPerceptualReviewSha256": bytes_digest(review_bytes),
        "counts": {
            "total": len(train_rows),
            "trainingFeatureViews": int(recipe["expectedTrainingFeatureViews"]),
            "validation": len(validation_rows),
            "validationFeatureViews": int(recipe["expectedValidationFeatureViews"]),
        },
        "sourceCounts": source_counts,
        "classCounts": class_counts,
        "evaluationExclusions": evaluation_exclusions,
        "perceptualOverlapReview": {
            "path": str(recipe["perceptualOverlapReview"]),
            "sha256": digest(old_review),
            "reviewedPairCount": len(old_review_rows),
            "hammingThreshold": int(recipe["perceptualDuplicateHammingThreshold"]),
        },
        "trainingPerceptualOverlapReview": {
            "path": str(recipe["trainingPerceptualOverlapReview"]),
            "sha256": digest(training_review),
            "reviewedPairCount": len(training_review_rows),
            "hammingThreshold": int(recipe["perceptualDuplicateHammingThreshold"]),
        },
        "overlapWithEvaluation": {
            "ids": 0,
            "imageHashes": 0,
            "perceptualHammingThreshold": int(recipe["perceptualDuplicateHammingThreshold"]),
            "reviewedVisuallyDistinctDhashPairsAtOrBelowThreshold": len(training_review_rows),
            "unreviewedPerceptualDhashPairsAtOrBelowThreshold": 0,
        },
    }
    summary_bytes = canonical_json(summary, pretty=True)
    return {
        "train-manifest.jsonl": train_bytes,
        "validation-manifest.jsonl": validation_bytes,
        "selection-summary.json": summary_bytes,
        "stock-selection.json": selection_bytes,
        "rejects.jsonl": rejects_bytes,
        "attribution.json": attribution_bytes,
        "perceptual-review.json": review_bytes,
    }


def write_local(recipe: dict[str, Any], packet: dict[str, bytes]) -> None:
    root = validate_data_root(ROOT / str(recipe["output"]["dataRoot"]))
    for name, value in packet.items():
        atomic_write(safe_output_path(root, name), value)


def publish(recipe: dict[str, Any], packet: dict[str, bytes]) -> None:
    root = ROOT / str(recipe["output"]["evidenceRoot"])
    root.mkdir(parents=True, exist_ok=True)
    compressed = {"train-manifest.jsonl", "stock-selection.json", "rejects.jsonl"}
    for name, value in packet.items():
        destination = root / (f"{name}.gz" if name in compressed else name)
        atomic_write(destination, deterministic_gzip(value) if name in compressed else value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true", help="Hard-link base pixels and extract selected StockImages bytes")
    parser.add_argument("--write-local", action="store_true", help="Refresh local manifests and receipts without relinking pixels")
    parser.add_argument("--publish", action="store_true", help="Publish deterministic pixel-free evidence under benchmark/evidence/m2")
    parser.add_argument("--offline", action="store_true", help="Require both pinned StockImages shards to already be present")
    args = parser.parse_args()
    recipe = json.loads(RECIPE_PATH.read_text())
    packet = derive_packet(
        recipe,
        materialize_pixels=args.materialize,
        allow_download=not args.offline,
    )
    if args.materialize or args.write_local:
        write_local(recipe, packet)
    if args.publish:
        publish(recipe, packet)
    summary = json.loads(packet["selection-summary.json"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
