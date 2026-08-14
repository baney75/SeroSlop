"""Prepare the source-pinned 103,600-image ProofLens head-training corpus.

Pixels stay outside Git. The committed recipe pins source revisions and metadata
digests; this program records every selected source ID, image digest, license,
and attribution field in machine-readable manifests.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import heapq
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable
import urllib.error
import urllib.request
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLAN_NAME = "selection-plan.json"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(data)
    partial.replace(path)


def selection_priority(seed: int, revision: str, kind: str, stable_id: str) -> str:
    return sha256(f"{seed}:{revision}:{kind}:{stable_id}".encode()).hexdigest()


def download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    attempts: int = 6,
) -> Path:
    if destination.exists():
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError(f"Unexpected cached size for {destination}")
        if expected_sha256 is not None and digest(destination) != expected_sha256:
            raise ValueError(f"Unexpected cached SHA-256 for {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ProofLens benchmark/1"})
            with urllib.request.urlopen(request, timeout=180) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise ValueError(f"Unexpected downloaded size for {destination}")
            if expected_sha256 is not None and digest(partial) != expected_sha256:
                raise ValueError(f"Unexpected downloaded SHA-256 for {destination}")
            partial.replace(destination)
            return destination
        except (OSError, urllib.error.URLError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def load_recipe(path: Path) -> dict[str, Any]:
    recipe = json.loads(path.read_text())
    if recipe.get("schemaVersion") != 1:
        raise ValueError("Unsupported large-corpus recipe")
    if int(recipe["expectedTotalCount"]) != (
        int(recipe["diffusionDb"]["targetCount"])
        + int(recipe["openImages"]["targetCount"])
        + int(recipe["expectedModernTrainingCount"])
    ):
        raise ValueError("Large-corpus recipe counts do not add up")
    if int(recipe["diffusionDb"]["reserveCount"]) < 1:
        raise ValueError("DiffusionDB reserveCount must be positive")
    if int(recipe["openImages"]["reserveCount"]) < 1:
        raise ValueError("Open Images reserveCount must be positive")
    if not recipe.get("evaluationManifests"):
        raise ValueError("At least one frozen evaluation manifest is required")
    resources = recipe.get("resourceEnvelope") or {}
    if int(resources.get("minimumFreeBytesBeforeMaterialization", 0)) < 1:
        raise ValueError("A positive materialization free-space preflight is required")
    if int(resources.get("maximumOpenImageBytes", 0)) < 1:
        raise ValueError("A positive Open Images response limit is required")
    if not 1 <= int(resources.get("maximumArchiveWorkers", 0)) <= 8:
        raise ValueError("maximumArchiveWorkers must be between 1 and 8")
    if not 1 <= int(resources.get("maximumOpenImageWorkers", 0)) <= 32:
        raise ValueError("maximumOpenImageWorkers must be between 1 and 32")
    return recipe


def parse_link_next(value: str | None) -> str | None:
    if not value:
        return None
    for item in value.split(","):
        pieces = item.strip().split(";")
        if len(pieces) >= 2 and pieces[1].strip() == 'rel="next"':
            return pieces[0].strip()[1:-1]
    return None


def diffusion_archive_locks(dataset: str, revision: str, part_ids: set[int]) -> list[dict[str, object]]:
    url: str | None = (
        f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/images"
        "?recursive=false&expand=false&limit=1000"
    )
    locks: list[dict[str, object]] = []
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": "ProofLens benchmark/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            rows = json.loads(response.read())
            url = parse_link_next(response.headers.get("Link"))
        for row in rows:
            name = Path(str(row.get("path", ""))).name
            if not name.startswith("part-") or not name.endswith(".zip"):
                continue
            part_id = int(name[5:11])
            if part_id not in part_ids:
                continue
            lfs = row.get("lfs") or {}
            file_hash = str(lfs.get("oid", ""))
            if len(file_hash) != 64:
                raise ValueError(f"Missing LFS SHA-256 for {name}")
            locks.append(
                {
                    "partId": part_id,
                    "path": str(row["path"]),
                    "sha256": file_hash,
                    "bytes": int(lfs.get("size", row["size"])),
                }
            )
    if {int(row["partId"]) for row in locks} != part_ids:
        missing = sorted(part_ids - {int(row["partId"]) for row in locks})
        raise ValueError(f"Missing DiffusionDB archive locks: {missing}")
    return sorted(locks, key=lambda row: int(row["partId"]))


def select_diffusiondb_candidates(
    recipe: dict[str, Any], metadata: Path
) -> tuple[list[int], list[dict[str, object]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("pyarrow is required for the DiffusionDB catalog") from error

    config = recipe["diffusionDb"]
    seed = int(recipe["seed"])
    revision = str(config["revision"])
    part_ids = sorted(
        range(1, 2001),
        key=lambda part_id: selection_priority(seed, revision, "part", str(part_id)),
    )[: int(config["candidatePartCount"])]
    selected_parts = set(part_ids)
    table = parquet.read_table(
        metadata,
        columns=["image_name", "part_id", "width", "height", "image_nsfw", "prompt_nsfw"],
    ).to_pydict()
    candidates: list[dict[str, object]] = []
    for index, image_name in enumerate(table["image_name"]):
        part_id = int(table["part_id"][index])
        image_nsfw = table["image_nsfw"][index]
        prompt_nsfw = table["prompt_nsfw"][index]
        if part_id not in selected_parts:
            continue
        if min(int(table["width"][index]), int(table["height"][index])) < int(recipe["minimumShortEdge"]):
            continue
        if image_nsfw is None or float(image_nsfw) > float(config["maximumImageNsfw"]):
            continue
        if prompt_nsfw is None or float(prompt_nsfw) > float(config["maximumPromptNsfw"]):
            continue
        candidates.append(
            {
                "id": str(image_name),
                "partId": part_id,
                "priority": selection_priority(seed, revision, "image", str(image_name)),
                "width": int(table["width"][index]),
                "height": int(table["height"][index]),
            }
        )
    candidates.sort(key=lambda row: (str(row["priority"]), str(row["id"])))
    target = int(config["targetCount"])
    required = target + int(config["reserveCount"])
    if len(candidates) < required:
        raise ValueError(
            f"DiffusionDB produced only {len(candidates)} eligible candidates; need {required} "
            "including the frozen reserve"
        )
    return part_ids, candidates


def select_diffusiondb(recipe: dict[str, Any], metadata: Path) -> dict[str, object]:
    config = recipe["diffusionDb"]
    part_ids, candidates = select_diffusiondb_candidates(recipe, metadata)
    locks = diffusion_archive_locks(str(config["dataset"]), str(config["revision"]), set(part_ids))
    return {"partIds": part_ids, "archives": locks, "candidates": candidates}


def negative_priority(value: str) -> int:
    return -int(value, 16)


def select_open_images(recipe: dict[str, Any], metadata: Path) -> list[dict[str, object]]:
    config = recipe["openImages"]
    seed = int(recipe["seed"])
    revision = str(config["revision"])
    target = int(config["targetCount"]) + int(config["reserveCount"])
    heap: list[tuple[int, str, dict[str, object]]] = []
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image_id = row.get("ImageID", "")
            license_url = row.get("License", "")
            if not image_id or "creativecommons.org/licenses/by/2.0" not in license_url:
                continue
            priority = selection_priority(seed, revision, "image", image_id)
            candidate = {
                "id": image_id,
                "priority": priority,
                "license": license_url,
                "author": row.get("Author", ""),
                "authorProfileUrl": row.get("AuthorProfileURL", ""),
                "title": row.get("Title", ""),
                "originalUrl": row.get("OriginalURL", ""),
                "landingUrl": row.get("OriginalLandingURL", ""),
                "rotation": row.get("Rotation", ""),
            }
            item = (negative_priority(priority), image_id, candidate)
            if len(heap) < target:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if len(heap) < target:
        raise ValueError(f"Open Images produced only {len(heap)} eligible candidates; need {target}")
    return sorted((item[2] for item in heap), key=lambda row: (str(row["priority"]), str(row["id"])))


def build_plan(recipe_path: Path, recipe: dict[str, Any], data_root: Path) -> dict[str, object]:
    cache = data_root / "cache"
    diffusion_config = recipe["diffusionDb"]
    diffusion_metadata = download(
        str(diffusion_config["metadataUrl"]),
        cache / "diffusiondb-metadata.parquet",
        expected_sha256=str(diffusion_config["metadataSha256"]),
    )
    open_config = recipe["openImages"]
    open_metadata = download(
        str(open_config["metadataUrl"]),
        cache / "open-images-train.csv",
        expected_sha256=str(open_config["metadataSha256"]),
    )
    plan = {
        "schemaVersion": 1,
        "recipeSha256": digest(recipe_path),
        "diffusionDb": select_diffusiondb(recipe, diffusion_metadata),
        "openImages": {"candidates": select_open_images(recipe, open_metadata)},
    }
    write_atomic(data_root / PLAN_NAME, json_bytes(plan, pretty=True))
    print(
        f"planned {len(plan['diffusionDb']['candidates'])} DiffusionDB and "
        f"{len(plan['openImages']['candidates'])} Open Images candidates",
        flush=True,
    )
    return plan


def validate_plan(
    recipe_path: Path,
    recipe: dict[str, Any],
    plan: dict[str, Any],
    data_root: Path | None = None,
) -> None:
    if plan.get("schemaVersion") != 1:
        raise ValueError("Unsupported selection-plan schema")
    if plan.get("recipeSha256") != digest(recipe_path):
        raise ValueError("Selection plan targets a different recipe")

    diffusion = plan.get("diffusionDb")
    open_images = plan.get("openImages")
    if not isinstance(diffusion, dict) or not isinstance(open_images, dict):
        raise ValueError("Selection plan is missing dataset sections")
    revision = str(recipe["diffusionDb"]["revision"])
    expected_parts = sorted(
        range(1, 2001),
        key=lambda part_id: selection_priority(int(recipe["seed"]), revision, "part", str(part_id)),
    )[: int(recipe["diffusionDb"]["candidatePartCount"])]
    part_ids = [int(value) for value in diffusion.get("partIds", [])]
    if part_ids != expected_parts:
        raise ValueError("Selection-plan DiffusionDB parts do not match the deterministic recipe")
    archives = list(diffusion.get("archives", []))
    archive_parts = [int(row.get("partId", -1)) for row in archives]
    if archive_parts != sorted(expected_parts) or len(set(archive_parts)) != len(archive_parts):
        raise ValueError("Selection-plan archive locks are missing, duplicated, or unsorted")
    for archive in archives:
        if len(str(archive.get("sha256", ""))) != 64 or int(archive.get("bytes", 0)) < 1:
            raise ValueError("Selection-plan archive lock is incomplete")

    candidates = list(diffusion.get("candidates", []))
    required_diffusion = int(recipe["diffusionDb"]["targetCount"]) + int(
        recipe["diffusionDb"]["reserveCount"]
    )
    if len(candidates) < required_diffusion:
        raise ValueError("Selection-plan DiffusionDB reserve is incomplete")
    candidate_ids = [str(row.get("id", "")) for row in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Selection-plan DiffusionDB candidates contain duplicate IDs")
    if any(int(row.get("partId", -1)) not in set(expected_parts) for row in candidates):
        raise ValueError("Selection-plan candidate points outside the locked archives")
    expected_candidate_order = sorted(
        candidates, key=lambda row: (str(row.get("priority", "")), str(row.get("id", "")))
    )
    if candidates != expected_candidate_order:
        raise ValueError("Selection-plan DiffusionDB candidates are not deterministically ordered")
    for row in candidates:
        if str(row.get("priority")) != selection_priority(
            int(recipe["seed"]), revision, "image", str(row.get("id", ""))
        ):
            raise ValueError("Selection-plan DiffusionDB priority is invalid")

    open_candidates = list(open_images.get("candidates", []))
    expected_open_count = int(recipe["openImages"]["targetCount"]) + int(
        recipe["openImages"]["reserveCount"]
    )
    if len(open_candidates) != expected_open_count:
        raise ValueError("Selection-plan Open Images reserve count is invalid")
    open_ids = [str(row.get("id", "")) for row in open_candidates]
    if len(set(open_ids)) != len(open_ids):
        raise ValueError("Selection-plan Open Images candidates contain duplicate IDs")
    expected_open_order = sorted(
        open_candidates, key=lambda row: (str(row.get("priority", "")), str(row.get("id", "")))
    )
    if open_candidates != expected_open_order:
        raise ValueError("Selection-plan Open Images candidates are not deterministically ordered")
    open_revision = str(recipe["openImages"]["revision"])
    for row in open_candidates:
        if str(row.get("priority")) != selection_priority(
            int(recipe["seed"]), open_revision, "image", str(row.get("id", ""))
        ):
            raise ValueError("Selection-plan Open Images priority is invalid")

    if data_root is not None:
        diffusion_metadata = data_root / "cache/diffusiondb-metadata.parquet"
        open_metadata = data_root / "cache/open-images-train.csv"
        if digest(diffusion_metadata) != recipe["diffusionDb"]["metadataSha256"]:
            raise ValueError("Cached DiffusionDB metadata no longer matches the recipe")
        if digest(open_metadata) != recipe["openImages"]["metadataSha256"]:
            raise ValueError("Cached Open Images metadata no longer matches the recipe")
        recomputed_parts, recomputed_diffusion = select_diffusiondb_candidates(recipe, diffusion_metadata)
        recomputed_open = select_open_images(recipe, open_metadata)
        if part_ids != recomputed_parts or candidates != recomputed_diffusion:
            raise ValueError("Selection-plan DiffusionDB candidates do not match pinned metadata")
        if open_candidates != recomputed_open:
            raise ValueError("Selection-plan Open Images candidates do not match pinned metadata")


def preflight_resources(recipe: dict[str, Any], plan: dict[str, Any], data_root: Path) -> dict[str, object]:
    resources = recipe["resourceEnvelope"]
    archive_bytes = sum(int(row["bytes"]) for row in plan["diffusionDb"]["archives"])
    data_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(data_root).free
    required_free = int(resources["minimumFreeBytesBeforeMaterialization"])
    report = {
        "schemaVersion": 1,
        "diffusionDbArchiveBytes": archive_bytes,
        "diffusionDbArchiveCount": len(plan["diffusionDb"]["archives"]),
        "diffusionDbCandidateCount": len(plan["diffusionDb"]["candidates"]),
        "openImagesCandidateCount": len(plan["openImages"]["candidates"]),
        "maximumArchiveWorkers": int(resources["maximumArchiveWorkers"]),
        "maximumOpenImageWorkers": int(resources["maximumOpenImageWorkers"]),
        "maximumOpenImageBytes": int(resources["maximumOpenImageBytes"]),
        "requiredFreeBytes": required_free,
        "availableFreeBytes": free_bytes,
    }
    write_atomic(data_root / "resource-report.json", json_bytes(report, pretty=True))
    if free_bytes < required_free:
        raise OSError(
            f"Large-corpus materialization requires at least {required_free} free bytes; "
            f"only {free_bytes} are available"
        )
    print(json.dumps(report, indent=2), flush=True)
    return report


def hamming_neighbors_16(value: int) -> Iterable[int]:
    yield value
    for first in range(16):
        yield value ^ (1 << first)
    for first in range(16):
        for second in range(first + 1, 16):
            yield value ^ (1 << first) ^ (1 << second)


def build_perceptual_index(
    rows: Iterable[dict[str, str]],
) -> tuple[list[dict[int, list[tuple[str, int]]]], list[dict[str, str]]]:
    indices: list[dict[int, list[tuple[str, int]]]] = [{}, {}, {}, {}]
    frozen = list(rows)
    for row in frozen:
        value = int(row["perceptualDhash64"], 16)
        for block in range(4):
            key = (value >> (block * 16)) & 0xFFFF
            indices[block].setdefault(key, []).append((row["id"], value))
    return indices, frozen


def perceptual_matches(
    perceptual_hash: str,
    index: tuple[list[dict[int, list[tuple[str, int]]]], list[dict[str, str]]],
    threshold: int,
) -> list[dict[str, object]]:
    value = int(perceptual_hash, 16)
    candidates: dict[str, int] = {}
    for block, buckets in enumerate(index[0]):
        block_value = (value >> (block * 16)) & 0xFFFF
        for neighbor in hamming_neighbors_16(block_value):
            for item_id, candidate_value in buckets.get(neighbor, []):
                candidates[item_id] = candidate_value
    return [
        {"id": item_id, "hammingDistance": (value ^ candidate_value).bit_count()}
        for item_id, candidate_value in sorted(candidates.items())
        if (value ^ candidate_value).bit_count() <= threshold
    ]


def load_evaluation_exclusions(
    recipe: dict[str, Any],
) -> tuple[
    set[str],
    set[str],
    tuple[list[dict[int, list[tuple[str, int]]]], list[dict[str, str]]],
    list[dict[str, object]],
    dict[str, object],
]:
    blocked_ids: set[str] = set()
    blocked_hashes: set[str] = set()
    evidence: list[dict[str, object]] = []
    perceptual_rows: list[dict[str, str]] = []
    threshold = int(recipe["perceptualDuplicateHammingThreshold"])
    review_path = REPOSITORY_ROOT / str(recipe["perceptualOverlapReview"])
    review = json.loads(review_path.read_text())
    expected_manifest_hashes = {
        str(manifest["path"]): digest(REPOSITORY_ROOT / str(manifest["path"]))
        for manifest in recipe["evaluationManifests"]
    }
    if (
        review.get("schemaVersion") != 1
        or int(review.get("hammingThreshold", -1)) != threshold
        or review.get("evaluationManifestSha256ByPath") != expected_manifest_hashes
    ):
        raise ValueError("Perceptual-overlap review is not bound to the frozen evaluation packet")
    reviewed_pairs: dict[tuple[str, str], dict[str, object]] = {}
    for item in review.get("items", []):
        ids = tuple(sorted(str(value) for value in item.get("ids", [])))
        if len(ids) != 2 or ids in reviewed_pairs or item.get("decision") != "visually-distinct":
            raise ValueError("Perceptual-overlap review contains an invalid or duplicate decision")
        reviewed_pairs[ids] = item
    observed_reviewed_pairs: set[tuple[str, str]] = set()
    for manifest in recipe["evaluationManifests"]:
        relative = str(manifest["path"])
        data_root = REPOSITORY_ROOT / str(manifest["dataRoot"])
        path = REPOSITORY_ROOT / relative
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        row_ids = [str(row["id"]) for row in rows]
        row_hashes = [str(row["imageSha256"]) for row in rows]
        if len(set(row_ids)) != len(row_ids) or len(set(row_hashes)) != len(row_hashes):
            raise ValueError(f"Frozen evaluation manifest contains duplicates: {relative}")
        if blocked_ids.intersection(row_ids) or blocked_hashes.intersection(row_hashes):
            raise ValueError("Frozen evaluation manifests overlap one another")
        blocked_ids.update(row_ids)
        blocked_hashes.update(row_hashes)
        manifest_perceptual: list[dict[str, str]] = []
        for row in rows:
            image_path = data_root / str(row["path"])
            data = image_path.read_bytes()
            if sha256(data).hexdigest() != str(row["imageSha256"]):
                raise ValueError(f"Frozen evaluation image integrity mismatch: {row['id']}")
            _, _, perceptual_hash = validate_image(data, 1)
            manifest_perceptual.append(
                {
                    "id": str(row["id"]),
                    "imageSha256": str(row["imageSha256"]),
                    "perceptualDhash64": perceptual_hash,
                    "manifest": relative,
                }
            )
        if perceptual_rows:
            existing_index = build_perceptual_index(perceptual_rows)
            overlaps = [
                {"id": row["id"], "matches": perceptual_matches(row["perceptualDhash64"], existing_index, threshold)}
                for row in manifest_perceptual
            ]
            overlaps = [row for row in overlaps if row["matches"]]
            if overlaps:
                existing_by_id = {row["id"]: row for row in perceptual_rows}
                for overlap in overlaps:
                    current = next(row for row in manifest_perceptual if row["id"] == overlap["id"])
                    for match in overlap["matches"]:
                        previous = existing_by_id[str(match["id"])]
                        pair = tuple(sorted((str(current["id"]), str(previous["id"]))))
                        decision = reviewed_pairs.get(pair)
                        exact_hashes = {
                            str(current["id"]): str(current["imageSha256"]),
                            str(previous["id"]): str(previous["imageSha256"]),
                        }
                        exact_dhashes = {
                            str(current["id"]): str(current["perceptualDhash64"]),
                            str(previous["id"]): str(previous["perceptualDhash64"]),
                        }
                        if (
                            decision is None
                            or decision.get("imageSha256ById") != exact_hashes
                            or decision.get("perceptualDhash64ById") != exact_dhashes
                            or int(decision.get("hammingDistance", -1)) != int(match["hammingDistance"])
                            or not str(decision.get("rationale", "")).strip()
                            or not str(decision.get("reviewer", "")).strip()
                            or not str(decision.get("reviewedAt", "")).strip()
                        ):
                            raise ValueError(
                                f"Frozen evaluation manifests have an unreviewed perceptual match: {pair}"
                            )
                        observed_reviewed_pairs.add(pair)
        perceptual_rows.extend(manifest_perceptual)
        evidence.append(
            {
                "path": str(relative),
                "sha256": digest(path),
                "rows": len(rows),
                "dataRoot": str(manifest["dataRoot"]),
                "role": "validation-or-confirmatory-test",
            }
        )
    for manifest in recipe.get("additionalTrainingExclusionManifests", []):
        relative = str(manifest["path"])
        data_root = REPOSITORY_ROOT / str(manifest["dataRoot"])
        path = REPOSITORY_ROOT / relative
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        row_ids = [str(row["id"]) for row in rows]
        row_hashes = [str(row["imageSha256"]) for row in rows]
        if (
            len(set(row_ids)) != len(row_ids)
            or len(set(row_hashes)) != len(row_hashes)
            or blocked_ids.intersection(row_ids)
            or blocked_hashes.intersection(row_hashes)
        ):
            raise ValueError(f"Additional training exclusion is duplicated or overlaps: {relative}")
        for row in rows:
            image_path = data_root / str(row["path"])
            data = image_path.read_bytes()
            if sha256(data).hexdigest() != str(row["imageSha256"]):
                raise ValueError(f"Additional training-exclusion image integrity mismatch: {row['id']}")
            _, _, perceptual_hash = validate_image(data, 1)
            perceptual_rows.append(
                {
                    "id": str(row["id"]),
                    "imageSha256": str(row["imageSha256"]),
                    "perceptualDhash64": perceptual_hash,
                    "manifest": relative,
                }
            )
        blocked_ids.update(row_ids)
        blocked_hashes.update(row_hashes)
        evidence.append(
            {
                "path": relative,
                "sha256": digest(path),
                "rows": len(rows),
                "dataRoot": str(manifest["dataRoot"]),
                "role": "web-negative-training-exclusion",
            }
        )
    if observed_reviewed_pairs != set(reviewed_pairs):
        raise ValueError("Perceptual-overlap review contains a stale or missing pair")
    review_evidence = {
        "path": str(recipe["perceptualOverlapReview"]),
        "sha256": digest(review_path),
        "reviewedPairCount": len(observed_reviewed_pairs),
        "hammingThreshold": threshold,
    }
    return blocked_ids, blocked_hashes, build_perceptual_index(perceptual_rows), evidence, review_evidence


def validate_training_perceptual_review(
    recipe: dict[str, Any],
    modern_rows: list[dict[str, object]],
    evaluation_index: tuple[list[dict[int, list[tuple[str, int]]]], list[dict[str, str]]],
) -> tuple[dict[str, object], set[tuple[str, str]]]:
    review_path = REPOSITORY_ROOT / str(recipe["trainingPerceptualOverlapReview"])
    review = json.loads(review_path.read_text())
    threshold = int(recipe["perceptualDuplicateHammingThreshold"])
    training_manifest_path = REPOSITORY_ROOT / str(recipe["modernTrainingManifest"])
    exclusion_manifests = [
        *recipe["evaluationManifests"],
        *recipe.get("additionalTrainingExclusionManifests", []),
    ]
    exclusion_hashes = {
        str(manifest["path"]): digest(REPOSITORY_ROOT / str(manifest["path"]))
        for manifest in exclusion_manifests
    }
    if (
        review.get("schemaVersion") != 1
        or int(review.get("hammingThreshold", -1)) != threshold
        or review.get("trainingManifest") != str(recipe["modernTrainingManifest"])
        or review.get("trainingManifestSha256") != digest(training_manifest_path)
        or review.get("evaluationExclusionSha256ByPath") != exclusion_hashes
    ):
        raise ValueError("Training/evaluation perceptual review is not bound to the frozen packet")

    training_by_id = {
        str(row["id"]): row
        for row in (
            json.loads(line) for line in training_manifest_path.read_text().splitlines() if line
        )
    }
    evaluation_by_id: dict[str, dict[str, str]] = {}
    dhash_by_evaluation_id = {
        str(row["id"]): str(row["perceptualDhash64"])
        for row in evaluation_index[1]
    }
    for manifest in exclusion_manifests:
        relative = str(manifest["path"])
        path = REPOSITORY_ROOT / relative
        for row in (json.loads(line) for line in path.read_text().splitlines() if line):
            evaluation_by_id[str(row["id"])] = {
                "evaluationId": str(row["id"]),
                "evaluationImageSha256": str(row["imageSha256"]),
                "evaluationPerceptualDhash64": dhash_by_evaluation_id[str(row["id"])],
                "evaluationManifest": relative,
                "evaluationDataRoot": str(manifest["dataRoot"]),
                "evaluationPath": str(row["path"]),
            }

    expected: dict[tuple[str, str], dict[str, object]] = {}
    for row in modern_rows:
        training_id = str(row["id"])
        original = training_by_id[training_id]
        for match in perceptual_matches(str(row["perceptualDhash64"]), evaluation_index, threshold):
            evaluation_id = str(match["id"])
            key = (training_id, evaluation_id)
            expected[key] = {
                "trainingId": training_id,
                "trainingImageSha256": str(original["imageSha256"]),
                "trainingPerceptualDhash64": str(row["perceptualDhash64"]),
                "trainingPath": str(original["path"]),
                **evaluation_by_id[evaluation_id],
                "hammingDistance": int(match["hammingDistance"]),
            }

    reviewed = require_exact_visually_distinct_pairs(review.get("items", []), expected)
    evidence = {
        "path": str(recipe["trainingPerceptualOverlapReview"]),
        "sha256": digest(review_path),
        "reviewedPairCount": len(reviewed),
        "hammingThreshold": threshold,
    }
    return evidence, set(expected)


def require_exact_visually_distinct_pairs(
    items: object,
    expected: dict[tuple[str, str], dict[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    if not isinstance(items, list):
        raise ValueError("Training/evaluation perceptual review has no item list")
    reviewed: dict[tuple[str, str], dict[str, object]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Training/evaluation perceptual review contains a non-object item")
        key = (str(item.get("trainingId", "")), str(item.get("evaluationId", "")))
        if key in reviewed:
            raise ValueError("Training/evaluation perceptual review contains a duplicate pair")
        reviewed[key] = item
    if set(reviewed) != set(expected):
        raise ValueError("Training/evaluation perceptual review is stale or incomplete")
    for key, exact in expected.items():
        item = reviewed[key]
        if (
            any(item.get(field) != value for field, value in exact.items())
            or item.get("decision") != "visually-distinct"
            or not str(item.get("rationale", "")).strip()
            or not str(item.get("reviewer", "")).strip()
            or not str(item.get("reviewedAt", "")).strip()
        ):
            raise ValueError(f"Training/evaluation perceptual candidate is not approved exactly: {key}")
    return reviewed


def validate_image(data: bytes, minimum_short_edge: int) -> tuple[int, int, str]:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError("Pillow is required to materialize the large corpus") from error
    with Image.open(BytesIO(data)) as opened:
        opened.verify()
    with Image.open(BytesIO(data)) as opened:
        oriented = ImageOps.exif_transpose(opened)
        size = oriented.size
        grayscale = oriented.convert("RGB").resize((9, 8), Image.Resampling.LANCZOS).convert("L")
        pixels = list(grayscale.getdata())
    if min(size) < minimum_short_edge:
        raise ValueError(f"image short edge {min(size)} is below {minimum_short_edge}")
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return int(size[0]), int(size[1]), f"{bits:016x}"


def fetch_bounded_bytes(url: str, maximum_bytes: int, attempts: int = 6) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ProofLens benchmark/1"})
            with urllib.request.urlopen(request, timeout=120) as source:
                content_length = source.headers.get("Content-Length")
                if content_length is not None and int(content_length) > maximum_bytes:
                    raise ValueError(f"image response exceeds {maximum_bytes} bytes")
                data = source.read(maximum_bytes + 1)
            if len(data) > maximum_bytes:
                raise ValueError(f"image response exceeds {maximum_bytes} bytes")
            return data
        except ValueError:
            raise
        except (OSError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def materialize_diffusiondb(
    recipe: dict[str, Any],
    plan: dict[str, Any],
    data_root: Path,
    rejects: list[dict[str, object]],
    archive_workers: int,
) -> list[dict[str, object]]:
    config = recipe["diffusionDb"]
    minimum = int(recipe["minimumShortEdge"])
    candidates_by_part: dict[int, list[dict[str, object]]] = {}
    for candidate in plan["diffusionDb"]["candidates"]:
        candidates_by_part.setdefault(int(candidate["partId"]), []).append(candidate)
    def fetch_archive(archive: dict[str, object]) -> tuple[int, Path]:
        path = download(
            f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['revision']}/{archive['path']}",
            data_root / "cache" / "diffusiondb" / Path(str(archive["path"])).name,
            expected_sha256=str(archive["sha256"]),
            expected_size=int(archive["bytes"]),
        )
        print(f"Downloaded DiffusionDB archive {archive['partId']}", flush=True)
        return int(archive["partId"]), path

    valid: list[dict[str, object]] = []
    archives = list(plan["diffusionDb"]["archives"])
    for offset in range(0, len(archives), archive_workers):
        batch = archives[offset : offset + archive_workers]
        with ThreadPoolExecutor(max_workers=archive_workers) as executor:
            archive_paths = dict(executor.map(fetch_archive, batch))
        for archive in batch:
            part_id = int(archive["partId"])
            archive_path = archive_paths[part_id]
            try:
                with zipfile.ZipFile(archive_path) as source:
                    members = {Path(name).name: name for name in source.namelist()}
                    for candidate in candidates_by_part.get(part_id, []):
                        image_name = str(candidate["id"])
                        try:
                            data = source.read(members[image_name])
                            width, height, perceptual_hash = validate_image(data, minimum)
                            image_hash = sha256(data).hexdigest()
                            relative = Path("train/synthetic/diffusiondb") / image_name
                            destination = data_root / relative
                            if not destination.exists():
                                write_atomic(destination, data)
                            elif digest(destination) != image_hash:
                                raise ValueError("cached image hash mismatch")
                            valid.append(
                                {
                                    "id": f"diffusiondb:{config['revision']}:{image_name}",
                                    "dataset": config["dataset"],
                                    "datasetRevision": config["revision"],
                                    "split": "train",
                                    "rowIndex": len(valid),
                                    "path": relative.as_posix(),
                                    "imageSha256": image_hash,
                                    "perceptualDhash64": perceptual_hash,
                                    "label": 1,
                                    "source": "diffusiondb-stable-diffusion",
                                    "sourceId": image_name,
                                    "sourcePart": part_id,
                                    "license": config["license"],
                                    "width": width,
                                    "height": height,
                                    "selectionPriority": candidate["priority"],
                                }
                            )
                        except (KeyError, OSError, ValueError, zipfile.BadZipFile) as error:
                            rejects.append(
                                {"dataset": config["dataset"], "id": image_name, "reason": str(error)}
                            )
            except (OSError, zipfile.BadZipFile) as error:
                for candidate in candidates_by_part.get(part_id, []):
                    rejects.append(
                        {"dataset": config["dataset"], "id": str(candidate["id"]), "reason": str(error)}
                    )
            print(f"DiffusionDB archive {part_id}: {len(valid)} valid candidates", flush=True)
    valid.sort(key=lambda row: (str(row["selectionPriority"]), str(row["id"])))
    return valid


def fetch_open_image(
    candidate: dict[str, object], recipe: dict[str, Any], data_root: Path
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    config = recipe["openImages"]
    image_id = str(candidate["id"])
    relative = Path("train/real/open-images-train") / f"{image_id}.jpg"
    destination = data_root / relative
    maximum_bytes = int(recipe["resourceEnvelope"]["maximumOpenImageBytes"])
    try:
        if destination.exists():
            if destination.stat().st_size > maximum_bytes:
                raise ValueError(f"cached image exceeds {maximum_bytes} bytes")
            data = destination.read_bytes()
        else:
            url = f"{config['imageBaseUrl']}/{image_id}.jpg"
            data = fetch_bounded_bytes(url, maximum_bytes)
        width, height, perceptual_hash = validate_image(data, int(recipe["minimumShortEdge"]))
        image_hash = sha256(data).hexdigest()
        if not destination.exists():
            write_atomic(destination, data)
        elif digest(destination) != image_hash:
            raise ValueError("cached image hash mismatch")
        return (
            {
                "id": f"open-images:{config['revision']}:train:{image_id}",
                "dataset": config["dataset"],
                "datasetRevision": config["revision"],
                "split": "train",
                "rowIndex": 0,
                "path": relative.as_posix(),
                "imageSha256": image_hash,
                "perceptualDhash64": perceptual_hash,
                "label": 0,
                "source": "open-images-train",
                "sourceId": image_id,
                "license": config["license"],
                "width": width,
                "height": height,
                "selectionPriority": candidate["priority"],
                "attribution": {key: value for key, value in candidate.items() if key not in {"priority"}},
            },
            None,
        )
    except (OSError, urllib.error.URLError, ValueError) as error:
        return None, {"dataset": config["dataset"], "id": image_id, "reason": str(error)}


def materialize_open_images(
    recipe: dict[str, Any], plan: dict[str, Any], data_root: Path, workers: int, rejects: list[dict[str, object]]
) -> list[dict[str, object]]:
    candidates = list(plan["openImages"]["candidates"])
    target = int(recipe["openImages"]["targetCount"])
    valid: list[dict[str, object]] = []
    batch_size = 1_000
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(candidates), batch_size):
            results = executor.map(
                lambda candidate: fetch_open_image(candidate, recipe, data_root),
                candidates[offset : offset + batch_size],
            )
            for item, rejection in results:
                if item is not None:
                    valid.append(item)
                if rejection is not None:
                    rejects.append(rejection)
            print(f"Open Images: {len(valid)}/{target} valid candidates", flush=True)
    valid.sort(key=lambda row: (str(row["selectionPriority"]), str(row["id"])))
    return valid


def load_modern_training(recipe: dict[str, Any], modern_root: Path, data_root: Path) -> list[dict[str, object]]:
    manifest_path = REPOSITORY_ROOT / str(recipe["modernTrainingManifest"])
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    if len(rows) != int(recipe["expectedModernTrainingCount"]):
        raise ValueError("Unexpected modern training manifest count")
    output: list[dict[str, object]] = []
    for row in rows:
        source = modern_root / str(row["path"])
        if not source.exists() or digest(source) != row["imageSha256"]:
            raise ValueError(f"Missing or corrupt modern training image: {row['id']}")
        _, _, perceptual_hash = validate_image(source.read_bytes(), 1)
        relative = Path("train/modern") / str(row["path"])
        destination = data_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        if digest(destination) != row["imageSha256"]:
            raise ValueError(f"Modern image copy mismatch: {row['id']}")
        output.append({**row, "path": relative.as_posix(), "perceptualDhash64": perceptual_hash})
    return output


def select_unique(
    rows: Iterable[dict[str, object]],
    target: int,
    used_ids: set[str],
    used_hashes: set[str],
    blocked_ids: set[str],
    blocked_hashes: set[str],
    blocked_perceptual: tuple[list[dict[int, list[tuple[str, int]]]], list[dict[str, str]]],
    perceptual_threshold: int,
    rejects: list[dict[str, object]],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for row in rows:
        item_id = str(row["id"])
        image_hash = str(row["imageSha256"])
        if item_id in blocked_ids or image_hash in blocked_hashes:
            rejects.append(
                {
                    "dataset": row["dataset"],
                    "id": item_id,
                    "reason": "matches a frozen evaluation ID or image SHA-256",
                }
            )
            continue
        if item_id in used_ids or image_hash in used_hashes:
            rejects.append(
                {
                    "dataset": row["dataset"],
                    "id": item_id,
                    "reason": "duplicate training ID or image SHA-256",
                }
            )
            continue
        matches = perceptual_matches(str(row["perceptualDhash64"]), blocked_perceptual, perceptual_threshold)
        if matches:
            rejects.append(
                {
                    "dataset": row["dataset"],
                    "id": item_id,
                    "reason": "perceptual match to a frozen evaluation image",
                    "matches": matches,
                    "hammingThreshold": perceptual_threshold,
                }
            )
            continue
        used_ids.add(item_id)
        used_hashes.add(image_hash)
        selected.append(row)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError(f"Only {len(selected)} unique valid rows available; need {target}")
    return selected


def materialize(
    recipe_path: Path,
    recipe: dict[str, Any],
    data_root: Path,
    modern_root: Path,
    workers: int,
) -> None:
    plan_path = data_root / PLAN_NAME
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else build_plan(recipe_path, recipe, data_root)
    validate_plan(recipe_path, recipe, plan, data_root)
    resource_report = preflight_resources(recipe, plan, data_root)
    rejects: list[dict[str, object]] = []
    (
        blocked_ids,
        blocked_hashes,
        blocked_perceptual,
        evaluation_evidence,
        perceptual_review_evidence,
    ) = load_evaluation_exclusions(recipe)
    perceptual_threshold = int(recipe["perceptualDuplicateHammingThreshold"])
    modern = load_modern_training(recipe, modern_root, data_root)
    used_ids = {str(row["id"]) for row in modern}
    used_hashes = {str(row["imageSha256"]) for row in modern}
    if len(used_ids) != len(modern) or len(used_hashes) != len(modern):
        raise ValueError("Duplicate IDs or image bytes in modern training manifest")
    if used_ids.intersection(blocked_ids) or used_hashes.intersection(blocked_hashes):
        raise ValueError("Modern training manifest overlaps a frozen evaluation manifest")
    training_review_evidence, reviewed_training_pairs = validate_training_perceptual_review(
        recipe, modern, blocked_perceptual
    )
    archive_workers = min(int(recipe["resourceEnvelope"]["maximumArchiveWorkers"]), workers)
    diffusion_candidates = materialize_diffusiondb(recipe, plan, data_root, rejects, archive_workers)
    diffusion = select_unique(
        diffusion_candidates,
        int(recipe["diffusionDb"]["targetCount"]),
        used_ids,
        used_hashes,
        blocked_ids,
        blocked_hashes,
        blocked_perceptual,
        perceptual_threshold,
        rejects,
    )
    open_workers = min(int(recipe["resourceEnvelope"]["maximumOpenImageWorkers"]), workers)
    open_candidates = materialize_open_images(recipe, plan, data_root, open_workers, rejects)
    open_images = select_unique(
        open_candidates,
        int(recipe["openImages"]["targetCount"]),
        used_ids,
        used_hashes,
        blocked_ids,
        blocked_hashes,
        blocked_perceptual,
        perceptual_threshold,
        rejects,
    )
    combined = sorted([*modern, *diffusion, *open_images], key=lambda row: str(row["id"]))
    if len(combined) != int(recipe["expectedTotalCount"]):
        raise ValueError("Materialized corpus count does not match recipe")
    for index, row in enumerate(combined):
        row["rowIndex"] = index
    manifest = data_root / "train-manifest.jsonl"
    write_atomic(manifest, b"".join(json_bytes(row) for row in combined))
    unreviewed_perceptual_overlap = []
    for row in combined:
        matches = perceptual_matches(
            str(row["perceptualDhash64"]), blocked_perceptual, perceptual_threshold
        )
        unreviewed = [
            match
            for match in matches
            if (str(row["id"]), str(match["id"])) not in reviewed_training_pairs
        ]
        if unreviewed:
            unreviewed_perceptual_overlap.append({"id": row["id"], "matches": unreviewed})
    if unreviewed_perceptual_overlap:
        raise ValueError(
            f"Final training manifest has unreviewed evaluation matches: {unreviewed_perceptual_overlap[:5]}"
        )
    evaluation_perceptual_path = data_root / "evaluation-perceptual-hashes.json"
    write_atomic(
        evaluation_perceptual_path,
        json_bytes(
            {
                "schemaVersion": 1,
                "algorithm": "EXIF-oriented RGB dHash 64-bit, LANCZOS 9x8 grayscale",
                "hammingThreshold": perceptual_threshold,
                "items": sorted(blocked_perceptual[1], key=lambda row: row["id"]),
            },
            pretty=True,
        ),
    )
    attribution = [
        {
            **row["attribution"],
            "manifestId": row["id"],
            "selectedImageSha256": row["imageSha256"],
        }
        for row in open_images
    ]
    write_atomic(data_root / "open-images-attribution.jsonl", b"".join(json_bytes(row) for row in attribution))
    write_atomic(data_root / "rejects.jsonl", b"".join(json_bytes(row) for row in rejects))
    summary = {
        "schemaVersion": 1,
        "recipeSha256": digest(recipe_path),
        "planSha256": digest(plan_path),
        "manifestSha256": digest(manifest),
        "attributionSha256": digest(data_root / "open-images-attribution.jsonl"),
        "rejectsSha256": digest(data_root / "rejects.jsonl"),
        "evaluationPerceptualHashesSha256": digest(evaluation_perceptual_path),
        "counts": {
            "modern": len(modern),
            "diffusionDb": len(diffusion),
            "openImages": len(open_images),
            "total": len(combined),
            "rejected": len(rejects),
        },
        "classCounts": {
            "real": sum(int(row["label"]) == 0 for row in combined),
            "synthetic": sum(int(row["label"]) == 1 for row in combined),
        },
        "sourceCounts": {
            source: sum(str(row["source"]) == source for row in combined)
            for source in sorted({str(row["source"]) for row in combined})
        },
        "sourceHashes": {
            "diffusionDbMetadata": recipe["diffusionDb"]["metadataSha256"],
            "openImagesMetadata": recipe["openImages"]["metadataSha256"],
        },
        "evaluationExclusions": evaluation_evidence,
        "perceptualOverlapReview": perceptual_review_evidence,
        "trainingPerceptualOverlapReview": training_review_evidence,
        "evaluationExcludedIds": len(blocked_ids),
        "evaluationExcludedImageHashes": len(blocked_hashes),
        "overlapWithEvaluation": {
            "ids": 0,
            "imageHashes": 0,
            "unreviewedPerceptualDhashPairsAtOrBelowThreshold": 0,
            "reviewedVisuallyDistinctDhashPairsAtOrBelowThreshold": len(reviewed_training_pairs),
            "perceptualHammingThreshold": perceptual_threshold,
        },
        "resourceEnvelope": {
            key: value for key, value in resource_report.items() if key != "availableFreeBytes"
        },
        "diffusionDbArchiveLocks": plan["diffusionDb"]["archives"],
    }
    write_atomic(data_root / "selection-summary.json", json_bytes(summary, pretty=True))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, default=Path("benchmark/large/recipe.json"))
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data/large-head"))
    parser.add_argument("--modern-data-root", type=Path, default=Path("benchmark/data/modern-head"))
    parser.add_argument("--phase", choices=("plan", "materialize", "all"), default="all")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--replan", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 64:
        raise ValueError("workers must be between 1 and 64")
    recipe = load_recipe(args.recipe)
    if args.phase in {"plan", "all"}:
        plan_path = args.data_root / PLAN_NAME
        if plan_path.exists() and not args.replan:
            plan = json.loads(plan_path.read_text())
            validate_plan(args.recipe, recipe, plan, args.data_root)
            print(f"reused {plan_path} ({digest(plan_path)})", flush=True)
        else:
            plan = build_plan(args.recipe, recipe, args.data_root)
            validate_plan(args.recipe, recipe, plan, args.data_root)
    if args.phase in {"materialize", "all"}:
        materialize(args.recipe, recipe, args.data_root, args.modern_data_root, args.workers)


if __name__ == "__main__":
    main()
