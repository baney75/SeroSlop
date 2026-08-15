"""Independently rederive and verify the replacement-holdout packet."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


sys.path.insert(0, str(Path(__file__).parent))

from build_historical_index import (  # noqa: E402
    build_index,
    canonical_json as canonical_historical_json,
    deterministic_gzip,
)
from prepare import (  # noqa: E402
    ROOT,
    PerceptualIndex,
    bytes_digest,
    derive_packet,
    digest,
    historical_evidence,
    image_facts,
    list_digest,
    load_jsonl,
    safe_output_path,
    validate_data_root,
)


RECIPE_PATH = Path(__file__).with_name("recipe.json")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def manifest_rows(path: Path, expected_items: int, expected_split: str) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    require(len(rows) == expected_items, f"Unexpected row count: {path}")
    require([int(row["rowIndex"]) for row in rows] == list(range(expected_items)), f"Non-canonical row indexes: {path}")
    require({str(row["split"]) for row in rows} == {expected_split}, f"Unexpected split: {path}")
    return rows


def main() -> None:
    recipe = json.loads(RECIPE_PATH.read_text())
    output = recipe["output"]
    data_root = validate_data_root(ROOT / output["dataRoot"])
    selection_path = ROOT / output["selectionEvidence"]
    review_path = ROOT / output["perceptualReview"]
    attribution_path = ROOT / output["attribution"]
    selection = json.loads(selection_path.read_text())
    review = json.loads(review_path.read_text())
    require(selection["recipeSha256"] == digest(RECIPE_PATH), "Selection evidence is not recipe-bound")
    require(
        selection["sourcePerceptualReviewSha256"] == digest(ROOT / recipe["reviewedPerceptualPairsPath"]),
        "Selection evidence is not source-review-bound",
    )
    require(
        selection["historicalPerceptualIndexSha256"]
        == recipe["historicalExclusions"]["historicalPerceptualIndexSha256"],
        "Selection evidence is not historical-index-bound",
    )
    require(
        selection["artifacts"]["perceptualReviewSha256"] == digest(review_path)
        and selection["artifacts"]["attributionSha256"] == digest(attribution_path),
        "Selection side artifacts changed",
    )

    confirmatory_path = ROOT / output["confirmatoryManifest"]
    web_path = ROOT / output["webNegativeManifest"]
    confirmatory = manifest_rows(confirmatory_path, 600, "confirmatory-v2")
    web = manifest_rows(web_path, 319, "web-negative-v2")
    require(selection["confirmatory"]["manifestSha256"] == digest(confirmatory_path), "Confirmation manifest changed")
    require(selection["webNegative"]["manifestSha256"] == digest(web_path), "Web-negative manifest changed")
    require(Counter(int(row["label"]) for row in confirmatory) == Counter({0: 300, 1: 300}), "Confirmation labels changed")
    require(Counter(int(row["label"]) for row in web) == Counter({0: 319}), "Web-negative labels changed")
    confirm_sources = Counter(str(row["source"]) for row in confirmatory)
    web_sources = Counter(str(row["source"]) for row in web)
    require(
        confirm_sources == Counter({"coxy7-infinity": 300, "stockimages-cc0": 300}),
        "Confirmation source allocation changed",
    )
    require(web_sources == Counter({"stockimages-cc0": 319}), "Web-negative source allocation changed")
    require(dict(sorted(confirm_sources.items())) == selection["confirmatory"]["sources"], "Confirmation sources changed")
    require(dict(sorted(web_sources.items())) == selection["webNegative"]["sources"], "Web-negative sources changed")

    all_rows = confirmatory + web
    ids = [str(row["id"]) for row in all_rows]
    hashes = [str(row["imageSha256"]) for row in all_rows]
    groups = [str(row["sourceGroupId"]) for row in all_rows]
    require(len(set(ids)) == len(ids), "Replacement IDs are not unique")
    require(len(set(hashes)) == len(hashes), "Replacement image bytes are not unique")
    require(len(set(groups)) == len(groups), "Replacement source groups are not unique")
    for row in all_rows:
        path = safe_output_path(data_root, str(row["path"]))
        require(path.is_file() and not path.is_symlink(), f"Unsafe or missing pixel path: {row['id']}")
        value = path.read_bytes()
        width, height, dhash, _ = image_facts(value)
        require(bytes_digest(value) == row["imageSha256"], f"Pixel hash changed: {row['id']}")
        require((width, height, dhash) == (row["width"], row["height"], row["perceptualDhash64"]), f"Pixel facts changed: {row['id']}")

    historical_ids, historical_hashes, historical_groups, historical_index, historical_counts = historical_evidence(recipe)
    require(selection["historicalCounts"] == historical_counts, "Historical exclusion counts changed")
    require(not set(ids) & historical_ids, "Replacement ID overlaps historical data")
    require(not set(hashes) & historical_hashes, "Replacement bytes overlap historical data")
    require(not set(groups) & historical_groups, "Replacement source group overlaps historical data")
    threshold = int(recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"])
    observed_pairs = []
    sequential = historical_index
    for row in all_rows:
        matches = sequential.matches(str(row["perceptualDhash64"]), threshold)
        observed_pairs.extend((row["id"], match) for match in matches)
        sequential.add(str(row["perceptualDhash64"]), str(row["id"]))
    require(observed_pairs == [], "Selected replacement rows contain an unreviewed perceptual overlap")
    require(review.get("schemaVersion") == 2 and review.get("retainedPairs") == [], "Perceptual review must contain no exceptions")
    require(review["rejectedCandidateCount"] == len(selection["rejectedCandidates"]), "Rejected-candidate count changed")
    require(selection["overlap"]["perceptualDhash64AtOrBelow8"] == 0, "Overlap summary changed")
    require(
        selection["confirmatory"]["selectedIdsSha256"] == list_digest(sorted(str(row["id"]) for row in confirmatory))
        and selection["confirmatory"]["selectedImageHashesSha256"]
        == list_digest(sorted(str(row["imageSha256"]) for row in confirmatory)),
        "Confirmation selection digests changed",
    )
    infinity = sorted((row for row in confirmatory if int(row["label"]) == 1), key=lambda row: int(row["globalOffset"]))
    require(
        selection["confirmatory"]["infinityOffsetsSha256"]
        == list_digest(str(row["globalOffset"]) for row in infinity),
        "Infinity offset selection changed",
    )
    require(
        selection["webNegative"]["selectedIdsSha256"] == list_digest(sorted(str(row["id"]) for row in web))
        and selection["webNegative"]["selectedImageHashesSha256"]
        == list_digest(sorted(str(row["imageSha256"]) for row in web)),
        "Web-negative selection digests changed",
    )
    require(
        selection["webNegative"]["sharesDatasetWithConfirmatoryReal"] is True
        and selection["webNegative"]["sharedIds"] == 0
        and selection["webNegative"]["sharedImageSha256"] == 0,
        "Shared source-dataset limitation is not disclosed",
    )

    # Recompute all 1,200 legacy dHashes from their exact SHA-locked cached bytes,
    # merge them with the already-verified 104,819 rows, and byte-compare the
    # resulting canonical packet with the tracked historical index.
    historical_packet = build_index(
        recipe,
        data_root / "source" / "legacy-evaluation",
        workers=1,
        allow_download=False,
    )
    expected_historical = canonical_historical_json(historical_packet)
    historical_path = ROOT / recipe["historicalExclusions"]["historicalPerceptualIndexPath"]
    require(
        historical_path.read_bytes() == deterministic_gzip(expected_historical),
        "Historical exclusion index does not byte-rederive from its pinned source bytes",
    )

    # Rerun the complete score-blind selector in an isolated ignored directory.
    # This independently binds Coxy/StockImages rows, priorities, rejections,
    # selected bytes, and every tracked packet artifact to the pinned parquets.
    with tempfile.TemporaryDirectory(prefix=".replacement-v2-verify-", dir=data_root.parent) as temporary:
        isolated_root = validate_data_root(Path(temporary))
        recomputed = derive_packet(
            recipe,
            source_root=data_root,
            output_root=isolated_root,
            allow_download=False,
        )
        artifact_paths = {
            "confirmatoryManifest": confirmatory_path,
            "webNegativeManifest": web_path,
            "selectionEvidence": selection_path,
            "perceptualReview": review_path,
            "attribution": attribution_path,
        }
        for name, path in artifact_paths.items():
            require(recomputed[name] == path.read_bytes(), f"Isolated selection changed tracked artifact: {name}")
        for row in all_rows:
            isolated_pixel = safe_output_path(isolated_root, str(row["path"]))
            require(
                isolated_pixel.is_file() and bytes_digest(isolated_pixel.read_bytes()) == row["imageSha256"],
                f"Isolated selector changed selected pixels: {row['id']}",
            )

    print(
        json.dumps(
            {
                "confirmatory": {"items": len(confirmatory), "sources": dict(sorted(confirm_sources.items()))},
                "webNegative": {
                    "items": len(web),
                    "sources": dict(sorted(web_sources.items())),
                    "sharedDatasetWithConfirmatoryReal": True,
                    "sharedIds": 0,
                    "sharedImageSha256": 0,
                },
                "historical": historical_counts,
                "rejectedCandidates": len(selection["rejectedCandidates"]),
                "reviewedVisuallyDistinctPairs": 0,
                "isolatedRederivation": "pass",
                "policy": "pass",
            }
        )
    )


if __name__ == "__main__":
    main()
