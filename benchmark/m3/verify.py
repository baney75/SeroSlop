"""Independently rederive and verify the M3 source-selection packet."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m3.contracts import (  # noqa: E402
    assert_pixel_facts,
    assert_unique_evidence_rows,
    deterministic_gzip,
)
from benchmark.m3.prepare import RECIPE_PATH, derive_packet, evaluation_dhashes  # noqa: E402
from benchmark.recovery_v3.prepare import (  # noqa: E402
    PerceptualIndex,
    historical_evidence,
    image_facts,
    reject_symlink_components,
    safe_output_path,
    validate_data_root,
)


COMPRESSED = {"train-manifest.jsonl", "rejects.jsonl", "flux-source-index.json"}
def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def bytes_digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def load_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.splitlines() if line]


def published_path(recipe: dict[str, Any], name: str) -> Path:
    if name == "met-development-probe-v1.jsonl":
        return ROOT / str(recipe["metSource"]["developmentProbe"]["manifestPath"])
    evidence_root = ROOT / str(recipe["output"]["evidenceRoot"])
    return evidence_root / (f"{name}.gz" if name in COMPRESSED else name)


def verify_public_packet(recipe: dict[str, Any], packet: dict[str, bytes]) -> None:
    expected_names = {
        "attribution.json",
        "flux-source-index.json",
        "h3-met-holdout-manifest.jsonl",
        "met-development-probe-v1.jsonl",
        "perceptual-review.json",
        "rejects.jsonl",
        "selection-summary.json",
        "train-manifest.jsonl",
        "training-evaluation-perceptual-review.json",
        "validation-manifest.jsonl",
    }
    if set(packet) != expected_names:
        raise ValueError("M3 derived packet surface changed")
    for name, expanded in packet.items():
        path = published_path(recipe, name)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked M3 evidence: {path}")
        actual = path.read_bytes()
        expected = deterministic_gzip(expanded) if name in COMPRESSED else expanded
        if actual != expected:
            raise ValueError(f"Published M3 evidence differs from offline rederivation: {path}")
        if name in COMPRESSED:
            if actual[:10] != expected[:10] or actual[9] != 0xFF or gzip.decompress(actual) != expanded:
                raise ValueError(f"M3 gzip is not canonical: {path}")


def row_group(row: dict[str, Any]) -> str:
    return str(row.get("sourceGroupId") or row.get("groupId") or row["id"])


def admit_prior(
    rows: Iterable[dict[str, Any]],
    *,
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    index: PerceptualIndex,
    fallback_dhashes: dict[str, str],
) -> None:
    for row in rows:
        identifier = str(row["id"])
        image_hash = str(row["imageSha256"])
        perceptual_hash = str(row.get("perceptualDhash64") or fallback_dhashes.get(identifier, ""))
        if not perceptual_hash:
            raise ValueError(f"Frozen prior row has no dHash: {identifier}")
        if identifier in ids:
            if image_hash not in hashes:
                raise ValueError(f"Frozen prior ID changed image bytes: {identifier}")
            continue
        ids.add(identifier)
        hashes.add(image_hash)
        groups.add(row_group(row))
        index.add(perceptual_hash, identifier)


def assert_and_admit_new(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    index: PerceptualIndex,
    threshold: int,
) -> None:
    for row in rows:
        identifier = str(row["id"])
        image_hash = str(row["imageSha256"])
        group = row_group(row)
        perceptual_hash = str(row["perceptualDhash64"])
        if identifier in ids or image_hash in hashes or group in groups:
            raise ValueError(f"{label} overlaps a frozen ID, byte hash, or source group: {identifier}")
        matches = index.matches(perceptual_hash, threshold)
        if matches:
            closest = min(matches, key=lambda row: (int(row["distance"]), str(row["matchingId"])))
            raise ValueError(
                f"{label} has a dHash match at distance {closest['distance']}: "
                f"{identifier} -> {closest['matchingId']}"
            )
        ids.add(identifier)
        hashes.add(image_hash)
        groups.add(group)
        index.add(perceptual_hash, identifier)


def verify_partition_contract(recipe: dict[str, Any], packet: dict[str, bytes]) -> None:
    train_rows = load_jsonl_bytes(packet["train-manifest.jsonl"])
    selector_rows = load_jsonl_bytes(packet["validation-manifest.jsonl"])
    holdout_rows = load_jsonl_bytes(packet["h3-met-holdout-manifest.jsonl"])
    probe_rows = load_jsonl_bytes(packet["met-development-probe-v1.jsonl"])
    base_count = int(recipe["baseTraining"]["items"])
    base_rows = train_rows[:base_count]
    new_train_rows = train_rows[base_count:]
    regression_rows = load_jsonl_bytes((ROOT / str(recipe["regressionValidation"]["manifest"])).read_bytes())

    fallback_dhashes = evaluation_dhashes(recipe)
    for label, rows, fallbacks in (
        ("M3 training", train_rows, None),
        ("M3 selector", selector_rows, None),
        ("H3 Met reserve", holdout_rows, None),
        ("consumed Met probe", probe_rows, None),
        ("M2 regression", regression_rows, fallback_dhashes),
    ):
        assert_unique_evidence_rows(rows, label=label, fallback_dhashes=fallbacks)

    expected_base = gzip.decompress((ROOT / str(recipe["baseTraining"]["manifest"])).read_bytes())
    if packet["train-manifest.jsonl"][: len(expected_base)] != expected_base:
        raise ValueError("M3 training manifest does not preserve the exact M2 byte prefix")
    if len(train_rows) != int(recipe["expectedTotalCount"]) or len(new_train_rows) != 2_400:
        raise ValueError("M3 training counts changed")
    if len(selector_rows) != 600 or len(holdout_rows) != 600 or len(probe_rows) != 100:
        raise ValueError("M3 selector, holdout, or probe count changed")
    if {str(row["source"]) for row in new_train_rows} != {"met-open-access"}:
        raise ValueError("M3 appended training rows are not exactly Met")
    selector_sources = {str(row["source"]): sum(str(item["source"]) == str(row["source"]) for item in selector_rows) for row in selector_rows}
    if selector_sources != {"flux-1-dev-development": 300, "met-open-access": 300}:
        raise ValueError("M3 selector source balance changed")
    if {str(row["source"]) for row in holdout_rows} != {"met-open-access"}:
        raise ValueError("H3 reserved rows are not exactly Met")
    for label, rows in (
        ("new M3 training", new_train_rows),
        ("M3 selector", selector_rows),
        ("H3 Met reserve", holdout_rows),
        ("consumed Met probe", probe_rows),
    ):
        if any(
            "width" not in row or "height" not in row
            or int(row["width"]) <= 0 or int(row["height"]) <= 0
            for row in rows
        ):
            raise ValueError(f"{label} lacks pixel dimensions")

    ids, hashes, groups, index, _ = historical_evidence(recipe)
    admit_prior(
        [*base_rows, *regression_rows],
        ids=ids,
        hashes=hashes,
        groups=groups,
        index=index,
        fallback_dhashes=fallback_dhashes,
    )
    for config in recipe["consumedEvaluationExclusions"]:
        rows = load_jsonl_bytes((ROOT / str(config["path"])).read_bytes())
        admit_prior(
            rows,
            ids=ids,
            hashes=hashes,
            groups=groups,
            index=index,
            fallback_dhashes=fallback_dhashes,
        )
    # The already-scored probe is itself an exclusion and may have historical
    # near matches; only later partitions must be disjoint from it.
    admit_prior(
        probe_rows,
        ids=ids,
        hashes=hashes,
        groups=groups,
        index=index,
        fallback_dhashes=fallback_dhashes,
    )
    threshold = int(recipe["perceptualDuplicateHammingThreshold"])
    assert_and_admit_new(
        holdout_rows,
        label="H3 Met reserve",
        ids=ids,
        hashes=hashes,
        groups=groups,
        index=index,
        threshold=threshold,
    )
    assert_and_admit_new(
        selector_rows,
        label="M3 selector",
        ids=ids,
        hashes=hashes,
        groups=groups,
        index=index,
        threshold=threshold,
    )
    assert_and_admit_new(
        new_train_rows,
        label="new M3 training",
        ids=ids,
        hashes=hashes,
        groups=groups,
        index=index,
        threshold=threshold,
    )

    summary = json.loads(packet["selection-summary.json"])
    if summary.get("scoresReadDuringSelection") is not False:
        raise ValueError("M3 selection does not prove score-blind operation")
    if summary.get("consumedDevelopmentRowsUsedForSelection") is not False:
        raise ValueError("Consumed development rows influenced M3 selection")
    if summary.get("newM3OverlapWithEvaluation", {}).get("perceptualDhashPairsAtOrBelowThreshold") != 0:
        raise ValueError("M3 summary reports a retained new perceptual overlap")
    carried = recipe["baseTraining"]["trainingEvaluationPerceptualReview"]
    if summary.get("carriedForwardM2PerceptualReview") != {
        "path": carried["path"],
        "sha256": carried["sha256"],
        "reviewedPairCount": carried["items"],
        "policy": "The unchanged M2 prefix carries forward these previously reviewed visually distinct pairs; none comes from the new M3 rows.",
    }:
        raise ValueError("M3 summary does not distinguish inherited dHash reviews")


def verify_manifest_pixels(
    rows: list[dict[str, Any]],
    *,
    root: Path,
    label: str,
) -> None:
    root = validate_data_root(root)
    for index, row in enumerate(rows, start=1):
        relative = str(row["path"])
        path = safe_output_path(root, relative)
        path = reject_symlink_components(root, path, label=f"{label} pixel")
        if not path.is_file() or digest(path) != row["imageSha256"]:
            raise ValueError(f"{label} pixel integrity changed: {row['id']}")
        value = path.read_bytes()
        width, height, perceptual_hash, _ = image_facts(value)
        assert_pixel_facts(
            row,
            width=width,
            height=height,
            perceptual_dhash64=perceptual_hash,
            label=f"{label} {row['id']}",
        )
        if index % 10_000 == 0:
            print(f"verified {label}: {index}/{len(rows)}", flush=True)


def verify_pixels(recipe: dict[str, Any], packet: dict[str, bytes]) -> None:
    verify_manifest_pixels(
        load_jsonl_bytes(packet["train-manifest.jsonl"]),
        root=ROOT / str(recipe["output"]["dataRoot"]),
        label="M3 training",
    )
    verify_manifest_pixels(
        load_jsonl_bytes(packet["validation-manifest.jsonl"]),
        root=ROOT / str(recipe["output"]["dataRoot"]),
        label="M3 selector",
    )
    verify_manifest_pixels(
        load_jsonl_bytes(packet["h3-met-holdout-manifest.jsonl"]),
        root=ROOT / str(recipe["output"]["holdoutDataRoot"]),
        label="H3 Met reserve",
    )
    probe = recipe["metSource"]["developmentProbe"]
    verify_manifest_pixels(
        load_jsonl_bytes(packet["met-development-probe-v1.jsonl"]),
        root=ROOT / str(probe["imageRoot"]),
        label="consumed Met probe",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-pixels", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(RECIPE_PATH.read_text())
    packet = derive_packet(recipe, materialize_pixels=False, allow_download=False)
    verify_public_packet(recipe, packet)
    verify_partition_contract(recipe, packet)
    if args.verify_pixels:
        verify_pixels(recipe, packet)
    summary = json.loads(packet["selection-summary.json"])
    print(
        json.dumps(
            {
                "policy": "pass",
                "trainingImages": summary["counts"]["total"],
                "trainingViews": summary["counts"]["trainingFeatureViews"],
                "selectorImages": summary["counts"]["validation"],
                "regressionImages": summary["counts"]["regression"],
                "reservedH3MetImages": summary["counts"]["h3MetHoldout"],
                "newPerceptualOverlapExceptions": 0,
                "scoresRead": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
