"""Independently rederive and verify the public M4 score-blind source packet."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
from typing import Any


ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

from benchmark.m4.contracts import load_frozen_protocol
from benchmark.m4.prepare import (
    COMPRESSED_ARTIFACTS,
    LOCKS_PATH,
    PUBLIC_EVIDENCE_ROOT,
    RECIPE_PATH,
    ROOT,
    assign_paths,
    compare_public_packet,
    deterministic_gzip,
    derive_packet,
    digest,
    digest_bytes,
    jsonl_bytes,
    load_base_training,
    load_jsonl_bytes,
    packet_hashes,
    public_candidate,
    reindex,
    seed_exclusions,
    select_british_phase,
    select_rapidata_phase,
    verify_materialized_rows,
)
from benchmark.m4.contracts import ALLOWED_BRITISH_DECADES, MODELS, canonical_json


HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")
DHASH64 = __import__("re").compile(r"^[0-9a-f]{16}$")


def _read_public_artifacts(evidence_root: Path) -> tuple[dict[str, bytes], dict[str, bytes]]:
    expected = {
        "attribution.json", "british-source-index.json.gz", "perceptual-review.json",
        "rapidata-source-index.json.gz", "rejects.jsonl.gz", "selection-summary.json",
        "train-manifest.jsonl.gz", "validation-manifest.jsonl",
    }
    actual = {path.name for path in evidence_root.iterdir() if path.is_file()} if evidence_root.is_dir() else set()
    if actual != expected:
        raise ValueError("M4 public source-packet file set changed")
    raw = {name: (evidence_root / name).read_bytes() for name in expected}
    expanded: dict[str, bytes] = {}
    for name, value in raw.items():
        if name.endswith(".gz"):
            if len(value) < 10 or value[3] != 0 or int.from_bytes(value[4:8], "little") != 0 or value[9] != 0xFF:
                raise ValueError(f"M4 public gzip header changed: {name}")
            decoded = gzip.decompress(value)
            if deterministic_gzip(decoded) != value:
                raise ValueError(f"M4 public gzip bytes are not canonical: {name}")
            expanded[name[:-3]] = decoded
        else:
            expanded[name] = value
    return raw, expanded


def _canonical_object(value: bytes, *, label: str, pretty: bool = False) -> dict[str, Any]:
    try:
        text = value.decode("utf-8", errors="strict")
        parsed = json.loads(text, object_pairs_hook=lambda pairs: _unique_object(pairs, label=label))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict JSON") from error
    if not isinstance(parsed, dict) or canonical_json(parsed, pretty=pretty) != value:
        raise ValueError(f"{label} is not canonical JSON")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]], *, label: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"{label} contains a duplicate JSON key: {key}")
        output[key] = value
    return output


def _require_jsonl(value: bytes, *, label: str) -> list[dict[str, Any]]:
    rows = load_jsonl_bytes(value)
    if jsonl_bytes(rows) != value:
        raise ValueError(f"{label} is not canonical JSONL")
    return rows


def _require_candidate_common(row: dict[str, Any], *, label: int, source: str) -> None:
    if (
        row.get("label") != label
        or row.get("source") != source
        or not isinstance(row.get("id"), str)
        or not HEX64.fullmatch(str(row.get("imageSha256", "")))
        or not DHASH64.fullmatch(str(row.get("perceptualDhash64", "")))
        or not isinstance(row.get("width"), int)
        or not isinstance(row.get("height"), int)
        or row["width"] <= 0
        or row["height"] <= 0
    ):
        raise ValueError(f"Malformed M4 public source candidate: {row.get('id')}")


def _validate_british_source_eligibility(
    british: list[dict[str, Any]], eligibility: object, recipe: dict[str, Any],
    lock_by_path: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(eligibility, dict) or set(eligibility) != {
        "sourceRows", "eligibleCandidateRows", "rejectedSourceRows", "rejectedDateCounts",
        "sourceRowCounts", "rejectedItems",
    }:
        raise ValueError("M4 British source eligibility schema changed")
    expected_rejected = recipe["britishLibrary"]["expectedRejectedDates"]
    row_counts = eligibility["sourceRowCounts"]
    rejected_items = eligibility["rejectedItems"]
    if (
        eligibility["sourceRows"] != recipe["britishLibrary"]["expectedSourceRows"]
        or eligibility["eligibleCandidateRows"] != len(british)
        or eligibility["rejectedSourceRows"] != recipe["britishLibrary"]["expectedRejectedSourceRows"]
        or eligibility["rejectedDateCounts"] != expected_rejected
        or not isinstance(row_counts, dict)
        or set(row_counts) != set(lock_by_path)
        or digest_bytes(canonical_json(row_counts))
        != recipe["britishLibrary"]["sourceRowCountsCanonicalSha256"]
        or not isinstance(rejected_items, list)
        or len(rejected_items) != eligibility["rejectedSourceRows"]
    ):
        raise ValueError("M4 British source eligibility boundary changed")
    eligible_positions = {(str(row["sourceShard"]), int(row["sourceRow"])) for row in british}
    if len(eligible_positions) != len(british):
        raise ValueError("M4 British eligible-source positions are duplicated")
    rejected_positions: set[tuple[str, int]] = set()
    rejected_counts: Counter[str] = Counter()
    for row in rejected_items:
        if not isinstance(row, dict) or set(row) != {
            "sourceShard", "sourceShardSha256", "sourceRow", "rawDate", "reason",
        }:
            raise ValueError("M4 British rejected-source schema changed")
        shard = str(row["sourceShard"])
        source_row = row["sourceRow"]
        raw_date = str(row["rawDate"])
        lock = lock_by_path.get(shard)
        position = (shard, source_row) if isinstance(source_row, int) and not isinstance(source_row, bool) else None
        if (
            lock is None
            or position is None
            or source_row < 0
            or source_row >= row_counts[shard]
            or row["sourceShardSha256"] != lock["sha256"]
            or row["reason"] != "date-not-in-frozen-strata"
            or raw_date not in expected_rejected
            or position in rejected_positions
            or position in eligible_positions
        ):
            raise ValueError("M4 British rejected-source provenance changed")
        rejected_positions.add(position)
        rejected_counts[raw_date] += 1
    expected_positions = {
        (shard, source_row)
        for shard, count in row_counts.items()
        if isinstance(count, int) and not isinstance(count, bool) and count > 0
        for source_row in range(count)
    }
    if (
        dict(sorted(rejected_counts.items())) != expected_rejected
        or len(expected_positions) != eligibility["sourceRows"]
        or eligible_positions | rejected_positions != expected_positions
    ):
        raise ValueError("M4 British source rows are not exhaustively accounted")


def _validate_public_indexes(
    british_packet: dict[str, Any], rapidata_packet: dict[str, Any], recipe: dict[str, Any], locks: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if set(british_packet) != {
        "schemaVersion", "dataset", "revision", "sourceLocksSha256", "sourceEligibility", "items",
    }:
        raise ValueError("M4 British source-index schema changed")
    if set(rapidata_packet) != {"schemaVersion", "dataset", "revision", "sourceLocksSha256", "items"}:
        raise ValueError("M4 Rapidata source-index schema changed")
    expected_lock_hash = recipe["sourceLocksSha256"]
    for packet, name in ((british_packet, "britishLibrary"), (rapidata_packet, "rapidata")):
        lock = locks[name]
        if (
            packet["schemaVersion"] != 1
            or packet["dataset"] != lock["dataset"]
            or packet["revision"] != lock["revision"]
            or packet["sourceLocksSha256"] != expected_lock_hash
            or not isinstance(packet["items"], list)
        ):
            raise ValueError(f"M4 {name} source-index identity changed")

    british_keys = {
        "id", "dataset", "datasetRevision", "sourceSplit", "sourceShard", "sourceShardSha256",
        "sourceRow", "bookId", "decade", "fnameSha256", "imageSha256", "perceptualDhash64",
        "sourceGroupId", "label", "source", "sourceReportedLicense", "width", "height", "extension",
    }
    british = british_packet["items"]
    lock_by_path = {str(row["path"]): row for row in locks["britishLibrary"]["files"]}
    if len(british) != recipe["britishLibrary"]["expectedCandidateRows"]:
        raise ValueError("M4 British source-index item count changed")
    for row in british:
        if set(row) != british_keys:
            raise ValueError("M4 British candidate schema changed")
        _require_candidate_common(row, label=0, source="british-library-plates")
        shard = str(row["sourceShard"])
        source_row = row["sourceRow"]
        book = str(row["bookId"])
        lock = lock_by_path.get(shard)
        if (
            lock is None
            or not isinstance(source_row, int)
            or isinstance(source_row, bool)
            or source_row < 0
            or row["sourceShardSha256"] != lock["sha256"]
            or row["dataset"] != locks["britishLibrary"]["dataset"]
            or row["datasetRevision"] != locks["britishLibrary"]["revision"]
            or row["sourceSplit"] != "plates/train"
            or row["decade"] not in ALLOWED_BRITISH_DECADES
            or row["extension"] not in {".jpg", ".png", ".webp"}
            or row["id"] != f"british-library:{locks['britishLibrary']['revision']}:{shard}:{source_row}"
            or row["sourceGroupId"] != f"british-library:{locks['britishLibrary']['revision']}:book:{book}"
            or not HEX64.fullmatch(str(row["fnameSha256"]))
        ):
            raise ValueError(f"M4 British candidate provenance changed: {row.get('id')}")
    if len({str(row["id"]) for row in british}) != len(british) or len({str(row["imageSha256"]) for row in british}) != len(british):
        raise ValueError("M4 British source index contains duplicate IDs or image bytes")
    distinct = {
        decade: len({str(row["bookId"]) for row in british if row["decade"] == decade})
        for decade in ALLOWED_BRITISH_DECADES
    }
    if distinct != recipe["britishLibrary"]["expectedDistinctBooksByDecade"]:
        raise ValueError("M4 British source-index book capacity changed")
    if len({str(row["bookId"]) for row in british}) != recipe["britishLibrary"]["expectedDistinctBooks"]:
        raise ValueError("M4 British source-index global book capacity changed")
    _validate_british_source_eligibility(
        british, british_packet["sourceEligibility"], recipe, lock_by_path,
    )

    rapidata_keys = {
        "id", "dataset", "datasetRevision", "sourceSplit", "firstSourceShard",
        "firstSourceShardSha256", "firstSourceRow", "imagePath", "promptSha256", "model",
        "imageSha256", "perceptualDhash64", "sourceGroupId", "label", "source",
        "sourceReportedLicense", "width", "height", "extension",
    }
    rapidata = rapidata_packet["items"]
    rapidata_lock_by_path = {str(row["path"]): row for row in locks["rapidata"]["files"]}
    if len(rapidata) != recipe["rapidata"]["expectedUniquePaths"]:
        raise ValueError("M4 Rapidata source-index item count changed")
    groups: dict[str, dict[str, Any]] = {}
    for row in rapidata:
        if set(row) != rapidata_keys:
            raise ValueError("M4 Rapidata candidate schema changed")
        model = str(row["model"])
        if model not in MODELS:
            raise ValueError("M4 Rapidata model family changed")
        _require_candidate_common(row, label=1, source=recipe["rapidata"]["models"][model])
        shard = str(row["firstSourceShard"])
        source_row = row["firstSourceRow"]
        image_path = str(row["imagePath"])
        prompt_hash = str(row["promptSha256"])
        group = str(row["sourceGroupId"])
        lock = rapidata_lock_by_path.get(shard)
        if (
            lock is None
            or not isinstance(source_row, int)
            or isinstance(source_row, bool)
            or source_row < 0
            or row["firstSourceShardSha256"] != lock["sha256"]
            or row["dataset"] != locks["rapidata"]["dataset"]
            or row["datasetRevision"] != locks["rapidata"]["revision"]
            or row["sourceSplit"] != "default/train"
            or row["extension"] not in {".jpg", ".png", ".webp"}
            or not image_path
            or not HEX64.fullmatch(prompt_hash)
            or row["id"] != f"rapidata:{locks['rapidata']['revision']}:{image_path}"
            or group != f"rapidata:{locks['rapidata']['revision']}:prompt:{prompt_hash}"
        ):
            raise ValueError(f"M4 Rapidata candidate provenance changed: {row.get('id')}")
        entry = groups.setdefault(group, {"promptSha256": prompt_hash, "models": defaultdict(list)})
        if entry["promptSha256"] != prompt_hash:
            raise ValueError("M4 Rapidata group changed prompt hash")
        entry["models"][model].append(image_path)
    if len({str(row["id"]) for row in rapidata}) != len(rapidata) or len({str(row["imageSha256"]) for row in rapidata}) != len(rapidata):
        raise ValueError("M4 Rapidata source index contains duplicate IDs or image bytes")
    unique_by_model = Counter(str(row["model"]) for row in rapidata)
    one_each = sum(all(len(row["models"].get(model, [])) >= 1 for model in MODELS) for row in groups.values())
    four_each = sum(all(len(row["models"].get(model, [])) == 4 for model in MODELS) for row in groups.values())
    if (
        len(groups) != recipe["rapidata"]["expectedPromptGroups"]
        or dict(unique_by_model) != recipe["rapidata"]["expectedUniquePathsByModel"]
        or one_each != recipe["rapidata"]["expectedOnePerFamilyGroups"]
        or four_each != recipe["rapidata"]["expectedFourPerFamilyGroups"]
    ):
        raise ValueError("M4 Rapidata public source capacity changed")
    return british, rapidata, groups


def verify_public_only(recipe: dict[str, Any], locks: dict[str, Any], evidence_root: Path = PUBLIC_EVIDENCE_ROOT) -> dict[str, Any]:
    raw, expanded = _read_public_artifacts(evidence_root)
    summary = _canonical_object(expanded["selection-summary.json"], label="M4 selection summary", pretty=True)
    british_packet = _canonical_object(expanded["british-source-index.json"], label="M4 British source index")
    rapidata_packet = _canonical_object(expanded["rapidata-source-index.json"], label="M4 Rapidata source index")
    attribution = _canonical_object(expanded["attribution.json"], label="M4 attribution", pretty=True)
    review = _canonical_object(expanded["perceptual-review.json"], label="M4 perceptual review", pretty=True)
    rejects = _require_jsonl(expanded["rejects.jsonl"], label="M4 rejects")
    published_train = _require_jsonl(expanded["train-manifest.jsonl"], label="M4 training manifest")
    published_selector = _require_jsonl(expanded["validation-manifest.jsonl"], label="M4 selector manifest")

    artifacts_without_summary = {
        "attribution.json": expanded["attribution.json"],
        "british-source-index.json": expanded["british-source-index.json"],
        "perceptual-review.json": expanded["perceptual-review.json"],
        "rapidata-source-index.json": expanded["rapidata-source-index.json"],
        "rejects.jsonl": expanded["rejects.jsonl"],
        "train-manifest.jsonl": expanded["train-manifest.jsonl"],
        "validation-manifest.jsonl": expanded["validation-manifest.jsonl"],
    }
    if summary.get("publicArtifacts") != packet_hashes(artifacts_without_summary):
        raise ValueError("M4 selection summary does not bind every public artifact")
    for name in COMPRESSED_ARTIFACTS:
        if digest_bytes(raw[f"{name}.gz"]) != summary["publicArtifacts"][name]["compressedSha256"]:
            raise ValueError(f"M4 compressed public artifact binding changed: {name}")

    british, rapidata, rapidata_groups = _validate_public_indexes(
        british_packet, rapidata_packet, recipe, locks,
    )
    base_rows, base_expanded = load_base_training(recipe)
    state, exclusion_counts = seed_exclusions(recipe, base_rows)
    replay_rejects: list[dict[str, Any]] = []
    british_selector, british_considered = select_british_phase(
        [dict(row) for row in british], phase="validation-real",
        quotas=recipe["britishLibrary"]["selectorDecadeQuotas"],
        namespace=recipe["britishLibrary"]["selectorPriorityNamespace"], excluded_books=set(),
        state=state, rejects=replay_rejects,
    )
    rapidata_selector, rapidata_considered = select_rapidata_phase(
        rapidata_groups, [dict(row) for row in rapidata], phase="validation-synthetic",
        target_groups=recipe["rapidata"]["selectorPromptGroups"],
        namespace=recipe["rapidata"]["selectorPriorityNamespace"],
        image_namespace=recipe["rapidata"]["imagePriorityNamespace"], excluded_groups=set(),
        state=state, rejects=replay_rejects,
    )
    british_training, _ = select_british_phase(
        [dict(row) for row in british], phase="train-real",
        quotas=recipe["britishLibrary"]["trainingDecadeQuotas"],
        namespace=recipe["britishLibrary"]["trainingPriorityNamespace"],
        excluded_books=british_considered, state=state, rejects=replay_rejects,
    )
    rapidata_training, rapidata_training_considered = select_rapidata_phase(
        rapidata_groups, [dict(row) for row in rapidata], phase="train-synthetic",
        target_groups=recipe["rapidata"]["trainingPromptGroups"],
        namespace=recipe["rapidata"]["trainingPriorityNamespace"],
        image_namespace=recipe["rapidata"]["imagePriorityNamespace"], excluded_groups=rapidata_considered,
        state=state, rejects=replay_rejects,
    )
    assign_paths(british_selector, rapidata_selector, british_training, rapidata_training)
    new_training = [*british_training, *rapidata_training]
    expected_train = reindex([*base_rows, *new_training], split="train")
    expected_selector = reindex(sorted(
        [*british_selector, *rapidata_selector],
        key=lambda row: (int(row["label"]), str(row["source"]), str(row["id"])),
    ), split="validation")
    expected_train_bytes = base_expanded + jsonl_bytes(expected_train[len(base_rows):])
    expected_selector_bytes = jsonl_bytes(expected_selector)
    if expected_train_bytes != expanded["train-manifest.jsonl"] or expected_train != published_train:
        raise ValueError("M4 public training selection did not replay exactly")
    if expected_selector_bytes != expanded["validation-manifest.jsonl"] or expected_selector != published_selector:
        raise ValueError("M4 public selector selection did not replay exactly")
    if replay_rejects != rejects:
        raise ValueError("M4 public rejection evidence did not replay exactly")

    source_counts = Counter(str(row["source"]) for row in expected_train)
    class_counts = {
        "real": sum(int(row["label"]) == 0 for row in expected_train),
        "synthetic": sum(int(row["label"]) == 1 for row in expected_train),
    }
    expected_summary = {
        "schemaVersion": 1,
        "recipeSha256": digest(RECIPE_PATH),
        "sourceLocksSha256": digest(LOCKS_PATH),
        "scoreBlind": True,
        "modelOutputsRead": False,
        "h3PixelsRead": False,
        "h3ManifestSha256": recipe["h3Exclusion"]["sha256"],
        "selectionOrder": ["british-selector", "rapidata-selector", "british-training", "rapidata-training"],
        "sourceEligibility": {"britishLibrary": british_packet["sourceEligibility"]},
        "training": {
            "items": len(expected_train), "featureViews": recipe["expectedTraining"]["featureViews"],
            "classCounts": class_counts, "sourceCounts": dict(sorted(source_counts.items())),
            "basePrefixItems": len(base_rows), "baseExpandedSha256": digest_bytes(base_expanded),
        },
        "freshSelector": {
            "items": len(expected_selector), "featureViews": recipe["freshSelector"]["featureViews"],
            "sourceCounts": dict(sorted(Counter(str(row["source"]) for row in expected_selector).items())),
            "classCounts": {
                "real": sum(int(row["label"]) == 0 for row in expected_selector),
                "synthetic": sum(int(row["label"]) == 1 for row in expected_selector),
            },
        },
        "partitionGroups": {
            "britishSelectorBooks": len({row["bookId"] for row in british_selector}),
            "britishTrainingBooks": len({row["bookId"] for row in british_training}),
            "rapidataSelectorPrompts": len({row["promptSha256"] for row in rapidata_selector}),
            "rapidataTrainingPrompts": len({row["promptSha256"] for row in rapidata_training}),
            "rapidataPreOverlapReserveGroups": recipe["rapidata"]["expectedCompleteGroupAllocation"]["reserve"],
            "rapidataUnassignedCompleteGroups": recipe["rapidata"]["expectedFourPerFamilyGroups"]
            - len(rapidata_considered | rapidata_training_considered),
        },
        "overlap": {
            "threshold": state.threshold, "admittedCrossPoolMatches": 0,
            "exclusionCounts": exclusion_counts, "rejectCount": len(replay_rejects), "reviewExceptions": 0,
        },
        "publicArtifacts": packet_hashes(artifacts_without_summary),
    }
    if summary != expected_summary:
        raise ValueError("M4 public selection summary did not replay exactly")
    expected_attribution = {
        "schemaVersion": 1,
        "britishLibrary": {
            "dataset": recipe["britishLibrary"]["source"], "revision": recipe["britishLibrary"]["revision"],
            "sourceReportedLicense": recipe["britishLibrary"]["sourceReportedLicense"],
            "notice": "The plates config is an algorithmic page-layout category. Selected bytes remain under ignored benchmark/data; the public repository distributes pixel-free provenance only. Source labels do not independently clear depicted or third-party rights.",
        },
        "rapidata": {
            "dataset": recipe["rapidata"]["source"], "revision": recipe["rapidata"]["revision"],
            "sourceReportedLicense": recipe["rapidata"]["sourceReportedLicense"],
            "sourceReportedProvenance": recipe["rapidata"]["sourceReportedProvenance"],
            "developmentOnly": True, "neverAcceptanceEvidence": True,
            "notice": "Publisher-authored family labels do not identify exact generator revisions or seeds and are not independent rights clearance. Selected bytes remain local-only.",
        },
    }
    expected_review = {
        "schemaVersion": 1,
        "algorithm": recipe["overlapPolicy"]["perceptualDhash64"]["algorithm"],
        "maximumHammingDistance": recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"],
        "policy": "No cross-pool dHash exceptions; every match at or below the threshold is rejected.",
        "items": [],
    }
    if attribution != expected_attribution or review != expected_review:
        raise ValueError("M4 public attribution or perceptual-review evidence changed")
    return expected_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "benchmark/data/m4-source")
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmark/data/m4-head")
    parser.add_argument("--skip-pixel-output-check", action="store_true")
    parser.add_argument("--public-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe, locks = load_frozen_protocol(RECIPE_PATH, LOCKS_PATH)
    if args.public_only:
        summary = verify_public_only(recipe, locks)
        print(json.dumps({
            "isolatedPublicRederivation": "pass",
            "trainingItems": summary["training"]["items"],
            "selectorItems": summary["freshSelector"]["items"],
            "h3PixelsRead": summary["h3PixelsRead"],
            "policy": "pass",
        }, sort_keys=True))
        return
    packet = derive_packet(
        recipe,
        locks,
        source_root=args.source_root,
        output_root=args.output_root,
        allow_download=False,
        materialize_pixels=False,
    )
    compare_public_packet(packet)
    if not args.skip_pixel_output_check:
        for name in ("train-manifest.jsonl", "validation-manifest.jsonl", "selection-summary.json"):
            if (args.output_root / name).read_bytes() != packet[name]:
                raise ValueError(f"M4 local materialization manifest changed: {name}")
        verify_materialized_rows(load_jsonl_bytes(packet["train-manifest.jsonl"]), args.output_root)
        verify_materialized_rows(load_jsonl_bytes(packet["validation-manifest.jsonl"]), args.output_root)
    summary = json.loads(packet["selection-summary.json"])
    print(json.dumps({
        "isolatedRederivation": "pass",
        "trainingItems": summary["training"]["items"],
        "selectorItems": summary["freshSelector"]["items"],
        "h3PixelsRead": summary["h3PixelsRead"],
        "policy": "pass",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
