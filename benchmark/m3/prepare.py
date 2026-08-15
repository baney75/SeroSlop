"""Prepare the M3 cultural-heritage hard-negative development packet.

The script is deliberately score-blind.  It fixes the untouched Met holdout
first, then a fresh Met/FLUX selector, and only then Met training rows.  Pixels
remain under ignored ``benchmark/data`` roots.  Public artifacts contain only
provenance, hashes, manifests, and overlap evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Any, Iterable
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m3.contracts import (  # noqa: E402
    canonical_source_url,
    deterministic_gzip,
    priority,
    source_group,
)
from benchmark.recovery_v3.prepare import (  # noqa: E402
    PerceptualIndex,
    admit,
    bytes_digest,
    historical_evidence,
    image_facts,
    qualify,
    reject_symlink_components,
    safe_output_path,
    validate_data_root,
)


RECIPE_PATH = ROOT / "benchmark/m3/recipe.json"
USER_AGENT = "ProofLens/1.0 (https://github.com/baney75/prooflens)"
COMPRESSED_PACKET_NAMES = {
    "train-manifest.jsonl",
    "rejects.jsonl",
    "flux-source-index.json",
}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def load_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.splitlines() if line]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_bytes(path.read_bytes())


def checked_path(relative: str, *, label: str) -> Path:
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f"Missing {label}: {relative}")
    return path


def load_base_training(recipe: dict[str, Any]) -> tuple[list[dict[str, Any]], bytes]:
    config = recipe["baseTraining"]
    path = checked_path(str(config["manifest"]), label="M2 training manifest")
    compressed = path.read_bytes()
    if bytes_digest(compressed) != config["compressedSha256"]:
        raise ValueError("M2 training manifest compressed bytes changed")
    expanded = gzip.decompress(compressed)
    if bytes_digest(expanded) != config["expandedSha256"]:
        raise ValueError("M2 training manifest expanded bytes changed")
    rows = load_jsonl_bytes(expanded)
    if len(rows) != int(config["items"]):
        raise ValueError("M2 training manifest row count changed")
    return rows, expanded


def load_m2_regression(recipe: dict[str, Any]) -> list[dict[str, Any]]:
    config = recipe["regressionValidation"]
    path = checked_path(str(config["manifest"]), label="M2 regression manifest")
    if digest(path) != config["sha256"]:
        raise ValueError("M2 regression manifest changed")
    rows = load_jsonl(path)
    if len(rows) != int(config["items"]):
        raise ValueError("M2 regression count changed")
    return rows


def load_carried_forward_perceptual_review(recipe: dict[str, Any]) -> tuple[bytes, int]:
    config = recipe["baseTraining"]["trainingEvaluationPerceptualReview"]
    path = checked_path(str(config["path"]), label="M2 training/evaluation perceptual review")
    value = path.read_bytes()
    if bytes_digest(value) != config["sha256"]:
        raise ValueError("M2 training/evaluation perceptual review changed")
    packet = json.loads(value)
    count = len(packet.get("items", []))
    if count != int(config["items"]):
        raise ValueError("M2 training/evaluation perceptual review count changed")
    return value, count


def verify_upstream_lineage(recipe: dict[str, Any]) -> None:
    config = recipe["upstreamModel"]
    checks = (
        (config["path"], config["sha256"], config["bytes"], "M2 ONNX"),
        (config["modelLock"], config["modelLockSha256"], None, "M2 model lock"),
        (config["selectionSummary"], config["selectionSummarySha256"], None, "M2 selection summary"),
        (config["trainingSummary"], config["trainingSummarySha256"], None, "M2 training summary"),
        (config["finalizationReceipt"], config["finalizationReceiptSha256"], None, "M2 finalization receipt"),
    )
    for relative, expected_hash, expected_bytes, label in checks:
        path = checked_path(str(relative), label=label)
        if digest(path) != expected_hash or (
            expected_bytes is not None and path.stat().st_size != int(expected_bytes)
        ):
            raise ValueError(f"{label} changed")


def evaluation_dhashes(recipe: dict[str, Any]) -> dict[str, str]:
    exclusions = recipe["historicalExclusions"]
    path = checked_path(
        "benchmark/evidence/large/evaluation-perceptual-hashes.json.gz",
        label="evaluation perceptual evidence",
    )
    compressed = path.read_bytes()
    if bytes_digest(compressed) != exclusions["evaluationPerceptualCompressedSha256"]:
        raise ValueError("Evaluation perceptual evidence compressed bytes changed")
    expanded = gzip.decompress(compressed)
    if bytes_digest(expanded) != exclusions["evaluationPerceptualExpandedSha256"]:
        raise ValueError("Evaluation perceptual evidence expanded bytes changed")
    packet = json.loads(expanded)
    return {str(row["id"]): str(row["perceptualDhash64"]) for row in packet["items"]}


def add_row_to_index(
    row: dict[str, Any],
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
    dhash_by_id: dict[str, str],
) -> None:
    identifier = str(row["id"])
    image_hash = str(row["imageSha256"])
    group = str(row.get("sourceGroupId") or row.get("groupId") or identifier)
    value = str(row.get("perceptualDhash64") or dhash_by_id.get(identifier, ""))
    if identifier in ids:
        if image_hash not in hashes:
            raise ValueError(f"Frozen ID changed image bytes: {identifier}")
        return
    if not value:
        raise ValueError(f"Frozen row has no perceptual hash: {identifier}")
    candidate = {
        "id": identifier,
        "imageSha256": image_hash,
        "sourceGroupId": group,
        "perceptualDhash64": value,
    }
    admit(candidate, ids, hashes, groups, perceptual)


def seed_exclusions(
    recipe: dict[str, Any],
    base_training: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[str], PerceptualIndex, dict[str, int]]:
    ids, hashes, groups, perceptual, historical_counts = historical_evidence(recipe)
    dhash_by_id = evaluation_dhashes(recipe)
    for row in base_training:
        add_row_to_index(row, ids, hashes, groups, perceptual, dhash_by_id)
    for row in regression_rows:
        add_row_to_index(row, ids, hashes, groups, perceptual, dhash_by_id)
    for config in recipe["consumedEvaluationExclusions"]:
        path = checked_path(str(config["path"]), label="consumed evaluation manifest")
        if digest(path) != config["sha256"]:
            raise ValueError(f"Consumed evaluation manifest changed: {path}")
        for row in load_jsonl(path):
            add_row_to_index(row, ids, hashes, groups, perceptual, dhash_by_id)
    return ids, hashes, groups, perceptual, historical_counts


def csv_candidates(recipe: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    config = recipe["metSource"]
    path = checked_path(str(config["csvPath"]), label="Met source CSV")
    if path.stat().st_size != int(config["csvBytes"]) or digest(path) != config["csvSha256"]:
        raise ValueError("Met source CSV changed")
    allowed_departments = set(config["allowedDepartments"])
    official_rows: dict[int, dict[str, str]] = {}
    reasons: Counter[str] = Counter()
    seen_ids: set[int] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Object ID", "Is Public Domain", "Object End Date", "Department"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Met source CSV schema changed")
        for row in reader:
            try:
                object_id = int(row["Object ID"])
                object_end_date = int(row["Object End Date"])
            except (TypeError, ValueError):
                reasons["invalid-id-or-date"] += 1
                continue
            if object_id in seen_ids:
                raise ValueError(f"Met source CSV has duplicate object ID: {object_id}")
            seen_ids.add(object_id)
            if row["Is Public Domain"].strip().lower() != "true":
                reasons["not-public-domain"] += 1
                continue
            if object_end_date > int(config["maximumObjectEndDate"]):
                reasons["too-recent"] += 1
                continue
            department = row["Department"].strip()
            if department not in allowed_departments:
                reasons["department-not-allowlisted"] += 1
                continue
            canonical_row = {key: str(value) for key, value in row.items()}
            official_rows[object_id] = {
                "objectID": str(object_id),
                "objectEndDate": str(object_end_date),
                "department": department,
                "csvRecordSha256": bytes_digest(canonical_json(canonical_row)),
            }

    metadata = config["huggingFaceMetadata"]
    metadata_path = checked_path(str(metadata["path"]), label="fixed Met image metadata")
    if (
        metadata_path.stat().st_size != int(metadata["bytes"])
        or digest(metadata_path) != metadata["sha256"]
    ):
        raise ValueError("Fixed Met image metadata changed")
    candidates: list[dict[str, str]] = []
    seen_metadata_ids: set[int] = set()
    with gzip.open(metadata_path, "rt", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "Object ID", "Is Public Domain", "Object End Date", "Department",
            "Object Name", "Title", "Artist Display Name", "Object Date", "Medium",
            "Credit Line", "Link Resource", "Metadata Date", "Classification",
            "primaryImageSmall",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Fixed Met image metadata schema changed")
        for row in reader:
            try:
                object_id = int(row["Object ID"])
            except (TypeError, ValueError):
                reasons["image-metadata-invalid-id"] += 1
                continue
            if object_id in seen_metadata_ids:
                raise ValueError(f"Fixed Met image metadata has duplicate object ID: {object_id}")
            seen_metadata_ids.add(object_id)
            official = official_rows.get(object_id)
            if official is None:
                reasons["not-in-official-policy-pool"] += 1
                continue
            if (
                row["Is Public Domain"].strip().lower() != "true"
                or row["Object End Date"].strip() != official["objectEndDate"]
                or row["Department"].strip() != official["department"]
            ):
                reasons["metadata-disagrees-with-official-csv"] += 1
                continue
            image_url = row["primaryImageSmall"].strip()
            if not image_url:
                reasons["missing-primary-image-small"] += 1
                continue
            try:
                image_url = canonical_source_url(image_url, allowed_host=str(config["imageHost"]))
            except ValueError:
                reasons["invalid-primary-image-small"] += 1
                continue
            canonical_metadata_row = {key: str(value) for key, value in row.items()}
            candidates.append(
                {
                    **official,
                    "primaryImageSmall": image_url,
                    "objectURL": row["Link Resource"].strip().replace("http://", "https://", 1),
                    "title": row["Title"].strip(),
                    "artistDisplayName": row["Artist Display Name"].strip(),
                    "objectName": row["Object Name"].strip(),
                    "classification": row["Classification"].strip(),
                    "medium": row["Medium"].strip(),
                    "objectDate": row["Object Date"].strip(),
                    "creditLine": row["Credit Line"].strip(),
                    "metadataDate": row["Metadata Date"].strip(),
                    "metadataRecordSha256": bytes_digest(canonical_json(canonical_metadata_row)),
                }
            )
    if len(candidates) < sum(
        int(config[key]) for key in ("holdoutTarget", "developmentTarget", "trainingTarget")
    ):
        raise ValueError("Met CSV does not contain enough score-blind candidates")
    return candidates, dict(sorted(reasons.items()))


def fetch_bytes(
    url: str,
    destination: Path,
    *,
    allow_download: bool,
    allowed_root: Path,
    minimum_delay: float = 0.0,
) -> tuple[bytes, str, str]:
    root = validate_data_root(allowed_root)
    try:
        relative = str(destination.absolute().relative_to(root))
    except ValueError as error:
        raise ValueError(f"Source cache escapes benchmark/data: {destination}") from error
    destination = safe_output_path(root, relative)
    if destination.is_file():
        value = destination.read_bytes()
        if not value:
            raise ValueError(f"Cached source is empty: {destination}")
        return value, url, "cache"
    if not allow_download:
        raise ValueError(f"Pinned source is unavailable in offline verification: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(7):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=20) as response:
                value = response.read()
                final_url = response.geturl()
            if not value:
                raise ValueError(f"Downloaded source is empty: {url}")
            atomic_write(destination, value)
            elapsed = time.monotonic() - started
            if minimum_delay > elapsed:
                time.sleep(minimum_delay - elapsed)
            return value, final_url, "download"
        except (OSError, ValueError, urllib.error.HTTPError, urllib.error.URLError) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == 6:
                break
            retry_after = 0.0
            if isinstance(error, urllib.error.HTTPError):
                try:
                    retry_after = float(error.headers.get("Retry-After", "0"))
                except ValueError:
                    retry_after = 0.0
            delay = max(retry_after, min(2**attempt, 30)) + random.Random(f"{url}:{attempt}").random()
            time.sleep(delay)
    raise RuntimeError(f"Source download failed after retries: {url}") from last_error


def ensure_locked_source_inputs(recipe: dict[str, Any], *, allow_download: bool) -> None:
    config = recipe["metSource"]
    sources = (
        (
            str(config["csvUrl"]),
            ROOT / str(config["csvPath"]),
            int(config["csvBytes"]),
            str(config["csvSha256"]),
            "Met source CSV",
        ),
        (
            str(config["huggingFaceMetadata"]["downloadUrl"]),
            ROOT / str(config["huggingFaceMetadata"]["path"]),
            int(config["huggingFaceMetadata"]["bytes"]),
            str(config["huggingFaceMetadata"]["sha256"]),
            "fixed Met image metadata",
        ),
    )
    for url, destination, expected_bytes, expected_hash, label in sources:
        value, _, _ = fetch_bytes(
            url,
            destination,
            allow_download=allow_download,
            allowed_root=destination.parent,
        )
        if len(value) != expected_bytes or bytes_digest(value) != expected_hash:
            raise ValueError(f"{label} bytes changed")


def met_candidate(
    recipe: dict[str, Any],
    csv_row: dict[str, str],
    *,
    allow_download: bool,
) -> dict[str, Any]:
    config = recipe["metSource"]
    object_id = int(csv_row["objectID"])
    image_url = canonical_source_url(
        str(csv_row["primaryImageSmall"]),
        allowed_host=str(config["imageHost"]),
    )
    image_root = validate_data_root(ROOT / str(config["imageCacheRoot"]))
    image_path = safe_output_path(image_root, f"{object_id}.image")
    image_bytes, final_image_url, image_transport = fetch_bytes(
        image_url,
        image_path,
        allow_download=allow_download,
        allowed_root=image_root,
    )
    final_image_url = canonical_source_url(final_image_url, allowed_host=str(config["imageHost"]))
    if final_image_url != image_url:
        raise ValueError(f"Met image URL redirected away from its frozen identity: {object_id}")
    width, height, perceptual_hash, extension = image_facts(image_bytes)
    identifier = f"met-open-access:{config['commit']}:{object_id}"
    return {
        "id": identifier,
        "label": 0,
        "source": "met-open-access",
        "dataset": config["dataset"],
        "datasetRevision": config["commit"],
        "sourceReportedLicense": config["sourceReportedLicense"],
        "imageMetadataDataset": config["huggingFaceMetadata"]["dataset"],
        "imageMetadataRevision": config["huggingFaceMetadata"]["revision"],
        "imageMetadataSourceReportedLicense": config["huggingFaceMetadata"]["sourceReportedLicense"],
        "objectID": object_id,
        "objectURL": str(csv_row["objectURL"]),
        "title": str(csv_row["title"]),
        "artistDisplayName": str(csv_row["artistDisplayName"]),
        "objectName": str(csv_row["objectName"]),
        "classification": str(csv_row["classification"]),
        "department": str(csv_row["department"]),
        "medium": str(csv_row["medium"]),
        "objectDate": str(csv_row["objectDate"]),
        "objectEndDate": int(csv_row["objectEndDate"]),
        "creditLine": str(csv_row["creditLine"]),
        "metadataDate": str(csv_row["metadataDate"]),
        "primaryImageSmall": image_url,
        "csvRecordSha256": csv_row["csvRecordSha256"],
        "metadataRecordSha256": csv_row["metadataRecordSha256"],
        "imageBytes": len(image_bytes),
        "imageSha256": bytes_digest(image_bytes),
        "perceptualDhash64": perceptual_hash,
        "sourceGroupId": source_group("met-primary-image:", image_url),
        "width": width,
        "height": height,
        "extension": extension,
        "pixelBytes": image_bytes,
        "imageTransport": image_transport,
    }


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in {"pixelBytes", "extension", "detailTransport", "imageTransport"}
    }


def probe_rows(
    recipe: dict[str, Any],
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
) -> list[dict[str, Any]]:
    config = recipe["metSource"]
    probe = config["developmentProbe"]
    report_path = checked_path(str(probe["reportPath"]), label="Met development report")
    if digest(report_path) != probe["reportSha256"]:
        raise ValueError("Met development report changed")
    report = json.loads(report_path.read_text())
    selected_ids = list(report.get("selectedObjectIds", []))
    selected_hashes = list(report.get("selectedImageSha256", []))
    if (
        report.get("developmentOnly") is not True
        or report.get("neverAcceptanceEvidence") is not True
        or len(selected_ids) != int(probe["items"])
        or len(selected_hashes) != int(probe["items"])
    ):
        raise ValueError("Met development report contract changed")
    rows: list[dict[str, Any]] = []
    for rank, (object_id, expected_image_hash) in enumerate(zip(selected_ids, selected_hashes, strict=True)):
        detail_path = checked_path(
            f"{probe['detailRoot']}/{int(object_id)}.json",
            label="Met development detail",
        )
        detail_bytes = detail_path.read_bytes()
        detail = json.loads(detail_bytes)
        image_path = checked_path(
            f"{probe['imageRoot']}/{int(object_id)}.jpg",
            label="Met development image",
        )
        image_bytes = image_path.read_bytes()
        image_hash = bytes_digest(image_bytes)
        if image_hash != expected_image_hash or int(detail.get("objectID", -1)) != int(object_id):
            raise ValueError(f"Met development probe bytes changed: {object_id}")
        image_url = canonical_source_url(
            str(detail.get("primaryImageSmall", "")),
            allowed_host=str(config["imageHost"]),
        )
        width, height, perceptual_hash, _ = image_facts(image_bytes)
        candidate = {
            "id": f"met-open-access:{config['commit']}:{int(object_id)}",
            "label": 0,
            "source": "met-open-access",
            "split": "consumed-development",
            "path": f"{int(object_id)}.jpg",
            "dataset": config["dataset"],
            "datasetRevision": config["commit"],
            "objectID": int(object_id),
            "imageSha256": image_hash,
            "perceptualDhash64": perceptual_hash,
            "sourceGroupId": source_group("met-primary-image:", image_url),
            "primaryImageSmall": image_url,
            "detailBytes": len(detail_bytes),
            "detailRawSha256": bytes_digest(detail_bytes),
            "detailCanonicalSha256": bytes_digest(canonical_json(detail)),
            "width": width,
            "height": height,
            "probeRank": rank,
            "developmentOnly": True,
            "neverAcceptanceEvidence": True,
        }
        # This probe is already consumed.  Its job here is exclusion, so a
        # historical near match is retained as another owner rather than
        # reviewed away or made eligible for a later partition.
        if str(candidate["id"]) not in ids:
            admit(candidate, ids, hashes, groups, perceptual)
        rows.append(candidate)
    return rows


def fetch_hf_tree(recipe: dict[str, Any], *, allow_download: bool) -> list[dict[str, Any]]:
    config = recipe["syntheticDevelopmentSource"]
    evidence_path = ROOT / str(config["sourceIndexPath"])
    if evidence_path.is_file():
        compressed = evidence_path.read_bytes()
        expected = str(config["sourceIndexSha256"])
        if expected != "PENDING_MATERIALIZATION" and bytes_digest(compressed) != expected:
            raise ValueError("FLUX source index changed")
        packet = json.loads(gzip.decompress(compressed))
        rows = list(packet["items"])
    else:
        if not allow_download:
            raise ValueError("FLUX source index is unavailable in offline verification")
        url: str | None = (
            "https://huggingface.co/api/datasets/"
            f"{config['dataset']}/tree/{config['revision']}?recursive=true&expand=true&limit=100"
        )
        tree: list[dict[str, Any]] = []
        while url is not None:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                tree.extend(json.load(response))
                link = response.headers.get("Link", "")
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
        rows = []
        for row in tree:
            path = str(row.get("path", ""))
            lfs = row.get("lfs") or {}
            if not path.startswith("images/") or not path.endswith(".png"):
                continue
            rows.append(
                {
                    "path": path,
                    "bytes": int(row["size"]),
                    "lfsSha256": str(lfs["oid"]),
                    "gitOid": str(row["oid"]),
                    "xetHash": str(row.get("xetHash", "")),
                }
            )
        rows.sort(key=lambda row: row["path"])
    if len(rows) != int(config["candidateRows"]):
        raise ValueError("FLUX source index row count changed")
    if len({str(row["path"]) for row in rows}) != len(rows):
        raise ValueError("FLUX source index contains duplicate paths")
    return rows


def flux_candidate(
    recipe: dict[str, Any],
    source_row: dict[str, Any],
    *,
    allow_download: bool,
) -> dict[str, Any]:
    config = recipe["syntheticDevelopmentSource"]
    relative = str(source_row["path"])
    url = f"{config['repository']}/resolve/{config['revision']}/{relative}?download=true"
    cache_root = validate_data_root(ROOT / str(config["cacheRoot"]))
    cache_path = safe_output_path(cache_root, relative)
    value, _, transport = fetch_bytes(
        url,
        cache_path,
        allow_download=allow_download,
        allowed_root=cache_root,
    )
    if len(value) != int(source_row["bytes"]) or bytes_digest(value) != source_row["lfsSha256"]:
        raise ValueError(f"FLUX source bytes changed: {relative}")
    width, height, perceptual_hash, extension = image_facts(value)
    identifier = f"flux-development:{config['revision']}:{relative}"
    return {
        "id": identifier,
        "label": 1,
        "source": "flux-1-dev-development",
        "dataset": config["dataset"],
        "datasetRevision": config["revision"],
        "sourceReportedLicense": config["sourceReportedLicense"],
        "sourceReportedGenerator": config["sourceReportedGenerator"],
        "sourcePath": relative,
        "sourceBytes": int(source_row["bytes"]),
        "sourceLfsSha256": str(source_row["lfsSha256"]),
        "sourceGitOid": str(source_row["gitOid"]),
        "sourceXetHash": str(source_row["xetHash"]),
        "imageSha256": bytes_digest(value),
        "perceptualDhash64": perceptual_hash,
        "sourceGroupId": f"flux-development:{config['revision']}:{relative}",
        "width": width,
        "height": height,
        "extension": extension,
        "pixelBytes": value,
        "imageTransport": transport,
    }


def link_or_write(source: Path, destination_root: Path, relative: str, expected_hash: str) -> None:
    root = validate_data_root(destination_root)
    destination = safe_output_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = safe_output_path(root, relative)
    source = reject_symlink_components(
        validate_data_root(source.parent),
        source,
        label="source pixel",
    )
    if not source.is_file() or digest(source) != expected_hash:
        raise ValueError(f"Source pixel integrity changed: {source}")
    if destination.exists():
        if destination.is_symlink() or digest(destination) != expected_hash:
            raise ValueError(f"Destination pixel integrity changed: {destination}")
        return
    os.link(source, destination)


def materialize_base_pixels(recipe: dict[str, Any], base_rows: list[dict[str, Any]], output_root: Path) -> None:
    source_root = validate_data_root(ROOT / str(recipe["baseTraining"]["dataRoot"]))
    for row in base_rows:
        relative = str(row["path"])
        source = reject_symlink_components(source_root, source_root / relative, label="M2 source pixel")
        link_or_write(source, output_root, relative, str(row["imageSha256"]))


def write_selected_pixel(candidate: dict[str, Any], root: Path, relative: str) -> None:
    root = validate_data_root(root)
    destination = safe_output_path(root, relative)
    if destination.is_file():
        if digest(destination) != candidate["imageSha256"]:
            raise ValueError(f"Selected destination changed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = safe_output_path(root, relative)
    atomic_write(destination, bytes(candidate["pixelBytes"]))


def selected_row(
    candidate: dict[str, Any],
    *,
    split: str,
    relative: str,
    selection_namespace: str,
) -> dict[str, Any]:
    row = public_candidate(candidate)
    row["split"] = split
    row["path"] = relative
    row["selectionPriority"] = priority(selection_namespace, str(row["id"]))
    return row


def choose_met_partition(
    recipe: dict[str, Any],
    csv_rows: list[dict[str, str]],
    *,
    phase: str,
    namespace: str,
    target: int,
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
    rejected_object_ids: set[int],
    rejects: list[dict[str, object]],
    allow_download: bool,
    materialize_pixels: bool,
) -> list[dict[str, Any]]:
    threshold = int(recipe["perceptualDuplicateHammingThreshold"])
    selected: list[dict[str, Any]] = []
    ordered = [
        row for row in sorted(
        csv_rows,
        key=lambda row: (priority(namespace, row["objectID"]), int(row["objectID"])),
        )
        if int(row["objectID"]) not in rejected_object_ids
        and f"met-open-access:{recipe['metSource']['commit']}:{int(row['objectID'])}" not in ids
    ]
    batch_size = 64
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(met_candidate, recipe, row, allow_download=allow_download)
                for row in batch
            ]
            results: list[dict[str, Any] | Exception] = []
            for future in futures:
                try:
                    results.append(future.result())
                except (OSError, ValueError, RuntimeError) as error:
                    results.append(error)
        for csv_row, result in zip(batch, results, strict=True):
            object_id = int(csv_row["objectID"])
            identifier = f"met-open-access:{recipe['metSource']['commit']}:{object_id}"
            if isinstance(result, Exception):
                rejected_object_ids.add(object_id)
                rejects.append({
                    "candidateId": identifier,
                    "phase": phase,
                    "reason": "source-policy-or-download",
                    "failureCategory": "source-unavailable-or-invalid",
                })
                continue
            candidate = result
            issue = qualify(candidate, ids, hashes, groups, perceptual, threshold)
            if issue is not None:
                rejected_object_ids.add(object_id)
                rejects.append({"candidateId": identifier, "phase": phase, **issue})
                continue
            extension = str(candidate["extension"])
            relative = (
                f"real/met-open-access/{object_id}-{str(candidate['imageSha256'])[:16]}{extension}"
                if phase == "h3-holdout"
                else f"{phase}/real/met-open-access/{object_id}-{str(candidate['imageSha256'])[:16]}{extension}"
            )
            if materialize_pixels:
                output_root = ROOT / (
                    str(recipe["output"]["holdoutDataRoot"])
                    if phase == "h3-holdout"
                    else str(recipe["output"]["dataRoot"])
                )
                write_selected_pixel(candidate, output_root, relative)
            admit(candidate, ids, hashes, groups, perceptual)
            selected.append(
                selected_row(
                    candidate,
                    split="confirmatory-reserved" if phase == "h3-holdout" else phase,
                    relative=relative,
                    selection_namespace=namespace,
                )
            )
            if len(selected) == target:
                return selected
            if len(selected) % 100 == 0:
                print(f"selected Met {phase}: {len(selected)}/{target}", flush=True)
    raise ValueError(f"Frozen Met partition target is unavailable: {phase} {target}")


def choose_flux_development(
    recipe: dict[str, Any],
    source_rows: list[dict[str, Any]],
    *,
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
    rejects: list[dict[str, object]],
    allow_download: bool,
    materialize_pixels: bool,
) -> list[dict[str, Any]]:
    config = recipe["syntheticDevelopmentSource"]
    namespace = str(config["priorityNamespace"])
    threshold = int(recipe["perceptualDuplicateHammingThreshold"])
    selected: list[dict[str, Any]] = []
    ordered = sorted(
        source_rows,
        key=lambda row: (priority(namespace, str(row["path"])), str(row["path"])),
    )
    batch_size = 48
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(flux_candidate, recipe, row, allow_download=allow_download)
                for row in batch
            ]
            results: list[dict[str, Any] | Exception] = []
            for future in futures:
                try:
                    results.append(future.result())
                except (OSError, ValueError, RuntimeError) as error:
                    results.append(error)
        for source_row, result in zip(batch, results, strict=True):
            identifier = f"flux-development:{config['revision']}:{source_row['path']}"
            if isinstance(result, Exception):
                rejects.append({
                    "candidateId": identifier,
                    "phase": "validation-synthetic",
                    "reason": "source-policy-or-download",
                    "failureCategory": "source-unavailable-or-invalid",
                })
                continue
            candidate = result
            issue = qualify(candidate, ids, hashes, groups, perceptual, threshold)
            if issue is not None:
                rejects.append({"candidateId": identifier, "phase": "validation-synthetic", **issue})
                continue
            relative = (
                "validation/synthetic/flux-1-dev/"
                f"{Path(str(source_row['path'])).stem}-{str(candidate['imageSha256'])[:16]}{candidate['extension']}"
            )
            if materialize_pixels:
                write_selected_pixel(candidate, ROOT / str(recipe["output"]["dataRoot"]), relative)
            admit(candidate, ids, hashes, groups, perceptual)
            selected.append(
                selected_row(
                    candidate,
                    split="validation",
                    relative=relative,
                    selection_namespace=namespace,
                )
            )
            if len(selected) == int(config["target"]):
                return selected
    raise ValueError("Frozen FLUX development target is unavailable")


def reindex(rows: Iterable[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = dict(original)
        row["rowIndex"] = index
        row["split"] = split
        output.append(row)
    return output


def exclusion_evidence(
    recipe: dict[str, Any],
    packet: dict[str, bytes],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for config in [*recipe["evaluationManifests"], *recipe["additionalTrainingExclusionManifests"]]:
        path = str(config["path"])
        if path == "benchmark/evidence/m3/validation-manifest.jsonl":
            value = packet["validation-manifest.jsonl"]
        elif path == "benchmark/manifests/met-development-probe-v1.jsonl":
            value = packet["met-development-probe-v1.jsonl"]
        elif path == "benchmark/evidence/m3/h3-met-holdout-manifest.jsonl":
            value = packet["h3-met-holdout-manifest.jsonl"]
        else:
            value = checked_path(path, label="training exclusion manifest").read_bytes()
        output.append(
            {
                "path": path,
                "sha256": bytes_digest(value),
                "rows": len(load_jsonl_bytes(value)),
                "dataRoot": str(config["dataRoot"]),
                "role": str(config["role"]),
            }
        )
    return output


def derive_packet(
    recipe: dict[str, Any],
    *,
    materialize_pixels: bool,
    allow_download: bool,
) -> dict[str, bytes]:
    verify_upstream_lineage(recipe)
    ensure_locked_source_inputs(recipe, allow_download=allow_download)
    base_rows, base_expanded = load_base_training(recipe)
    regression_rows = load_m2_regression(recipe)
    carried_review_bytes, carried_review_count = load_carried_forward_perceptual_review(recipe)
    ids, hashes, groups, perceptual, historical_counts = seed_exclusions(
        recipe,
        base_rows,
        regression_rows,
    )
    counts_before_probe = {
        "ids": len(ids),
        "imageSha256": len(hashes),
        "sourceGroupIds": len(groups),
        "perceptualDhash64": len(perceptual),
    }
    consumed_probe_rows = probe_rows(recipe, ids, hashes, groups, perceptual)
    csv_rows, csv_reject_counts = csv_candidates(recipe)
    rejects: list[dict[str, object]] = []
    rejected_met_ids: set[int] = set()
    met_config = recipe["metSource"]

    holdout_rows = choose_met_partition(
        recipe,
        csv_rows,
        phase="h3-holdout",
        namespace=str(met_config["holdoutPriorityNamespace"]),
        target=int(met_config["holdoutTarget"]),
        ids=ids,
        hashes=hashes,
        groups=groups,
        perceptual=perceptual,
        rejected_object_ids=rejected_met_ids,
        rejects=rejects,
        allow_download=allow_download,
        materialize_pixels=materialize_pixels,
    )
    met_development_rows = choose_met_partition(
        recipe,
        csv_rows,
        phase="validation",
        namespace=str(met_config["developmentPriorityNamespace"]),
        target=int(met_config["developmentTarget"]),
        ids=ids,
        hashes=hashes,
        groups=groups,
        perceptual=perceptual,
        rejected_object_ids=rejected_met_ids,
        rejects=rejects,
        allow_download=allow_download,
        materialize_pixels=materialize_pixels,
    )
    flux_source_rows = fetch_hf_tree(recipe, allow_download=allow_download)
    flux_development_rows = choose_flux_development(
        recipe,
        flux_source_rows,
        ids=ids,
        hashes=hashes,
        groups=groups,
        perceptual=perceptual,
        rejects=rejects,
        allow_download=allow_download,
        materialize_pixels=materialize_pixels,
    )
    met_training_rows = choose_met_partition(
        recipe,
        csv_rows,
        phase="train",
        namespace=str(met_config["trainingPriorityNamespace"]),
        target=int(met_config["trainingTarget"]),
        ids=ids,
        hashes=hashes,
        groups=groups,
        perceptual=perceptual,
        rejected_object_ids=rejected_met_ids,
        rejects=rejects,
        allow_download=allow_download,
        materialize_pixels=materialize_pixels,
    )

    if materialize_pixels:
        materialize_base_pixels(recipe, base_rows, ROOT / str(recipe["output"]["dataRoot"]))

    training_rows = reindex([*base_rows, *met_training_rows], split="train")
    validation_rows = reindex([*met_development_rows, *flux_development_rows], split="validation")
    holdout_rows = reindex(holdout_rows, split="confirmatory-reserved")
    train_bytes = base_expanded + jsonl_bytes(training_rows[len(base_rows):])
    if load_jsonl_bytes(train_bytes) != training_rows:
        raise AssertionError("M3 manifest did not preserve the exact M2 prefix")
    packet: dict[str, bytes] = {
        "train-manifest.jsonl": train_bytes,
        "validation-manifest.jsonl": jsonl_bytes(validation_rows),
        "h3-met-holdout-manifest.jsonl": jsonl_bytes(holdout_rows),
        "met-development-probe-v1.jsonl": jsonl_bytes(consumed_probe_rows),
        "flux-source-index.json": canonical_json(
            {
                "schemaVersion": 1,
                "dataset": recipe["syntheticDevelopmentSource"]["dataset"],
                "revision": recipe["syntheticDevelopmentSource"]["revision"],
                "items": flux_source_rows,
            },
            pretty=True,
        ),
        "rejects.jsonl": jsonl_bytes(rejects),
        "attribution.json": canonical_json(
            {
                "schemaVersion": 1,
                "met": {
                    "dataset": met_config["dataset"],
                    "repository": met_config["repository"],
                    "commit": met_config["commit"],
                    "sourceReportedLicense": met_config["sourceReportedLicense"],
                    "notice": "Selected public-domain cultural-heritage image bytes stay under ignored benchmark/data. This repository publishes pixel-free provenance and overlap evidence only.",
                },
                "fluxDevelopment": {
                    "dataset": recipe["syntheticDevelopmentSource"]["dataset"],
                    "repository": recipe["syntheticDevelopmentSource"]["repository"],
                    "revision": recipe["syntheticDevelopmentSource"]["revision"],
                    "sourceReportedLicense": recipe["syntheticDevelopmentSource"]["sourceReportedLicense"],
                    "sourceReportedGenerator": recipe["syntheticDevelopmentSource"]["sourceReportedGenerator"],
                    "developmentOnly": True,
                    "neverAcceptanceEvidence": True,
                },
            },
            pretty=True,
        ),
        "perceptual-review.json": canonical_json(
            {
                "schemaVersion": 1,
                "algorithm": recipe["overlapPolicy"]["perceptualDhash64"]["algorithm"],
                "maximumHammingDistance": recipe["perceptualDuplicateHammingThreshold"],
                "policy": "reject every cross-pool match; no visual exceptions",
                "items": [],
            },
            pretty=True,
        ),
        "training-evaluation-perceptual-review.json": carried_review_bytes,
    }
    exclusions = exclusion_evidence(recipe, packet)
    source_counts = dict(Counter(str(row["source"]) for row in training_rows))
    class_counts = {
        "real": sum(int(row["label"]) == 0 for row in training_rows),
        "synthetic": sum(int(row["label"]) == 1 for row in training_rows),
    }
    if source_counts != recipe["expectedSourceCounts"] or class_counts != recipe["expectedClassCounts"]:
        raise ValueError("M3 source or class counts changed")
    if len(training_rows) != int(recipe["expectedTotalCount"]):
        raise ValueError("M3 training count changed")
    if len(validation_rows) != int(recipe["expectedValidationCount"]):
        raise ValueError("M3 validation count changed")
    summary = {
        "schemaVersion": 3,
        "recipeSha256": digest(RECIPE_PATH),
        "manifestSha256": bytes_digest(packet["train-manifest.jsonl"]),
        "validationManifestSha256": bytes_digest(packet["validation-manifest.jsonl"]),
        "holdoutManifestSha256": bytes_digest(packet["h3-met-holdout-manifest.jsonl"]),
        "developmentProbeManifestSha256": bytes_digest(packet["met-development-probe-v1.jsonl"]),
        "fluxSourceIndexExpandedSha256": bytes_digest(packet["flux-source-index.json"]),
        "publicArtifacts": {
            "trainManifest": {
                "path": "benchmark/evidence/m3/train-manifest.jsonl.gz",
                "expandedSha256": bytes_digest(packet["train-manifest.jsonl"]),
                "compressedSha256": bytes_digest(deterministic_gzip(packet["train-manifest.jsonl"])),
            },
            "validationManifest": {
                "path": "benchmark/evidence/m3/validation-manifest.jsonl",
                "sha256": bytes_digest(packet["validation-manifest.jsonl"]),
            },
            "h3MetHoldoutManifest": {
                "path": "benchmark/evidence/m3/h3-met-holdout-manifest.jsonl",
                "sha256": bytes_digest(packet["h3-met-holdout-manifest.jsonl"]),
            },
            "developmentProbeManifest": {
                "path": "benchmark/manifests/met-development-probe-v1.jsonl",
                "sha256": bytes_digest(packet["met-development-probe-v1.jsonl"]),
            },
            "fluxSourceIndex": {
                "path": "benchmark/evidence/m3/flux-source-index.json.gz",
                "expandedSha256": bytes_digest(packet["flux-source-index.json"]),
                "compressedSha256": bytes_digest(deterministic_gzip(packet["flux-source-index.json"])),
            },
            "rejects": {
                "path": "benchmark/evidence/m3/rejects.jsonl.gz",
                "expandedSha256": bytes_digest(packet["rejects.jsonl"]),
                "compressedSha256": bytes_digest(deterministic_gzip(packet["rejects.jsonl"])),
            },
            "attribution": {
                "path": "benchmark/evidence/m3/attribution.json",
                "sha256": bytes_digest(packet["attribution.json"]),
            },
            "perceptualReview": {
                "path": "benchmark/evidence/m3/perceptual-review.json",
                "sha256": bytes_digest(packet["perceptual-review.json"]),
            },
            "trainingEvaluationPerceptualReview": {
                "path": "benchmark/evidence/m3/training-evaluation-perceptual-review.json",
                "sha256": bytes_digest(packet["training-evaluation-perceptual-review.json"]),
            },
        },
        "upstreamModel": recipe["upstreamModel"],
        "baseTraining": recipe["baseTraining"],
        "regressionValidation": recipe["regressionValidation"],
        "sourceLocks": {
            "metCsv": {
                "path": met_config["csvPath"],
                "url": met_config["csvUrl"],
                "bytes": met_config["csvBytes"],
                "sha256": met_config["csvSha256"],
                "commit": met_config["commit"],
            },
            "metImageMetadata": {
                "dataset": met_config["huggingFaceMetadata"]["dataset"],
                "revision": met_config["huggingFaceMetadata"]["revision"],
                "repositoryPath": met_config["huggingFaceMetadata"]["repositoryPath"],
                "downloadUrl": met_config["huggingFaceMetadata"]["downloadUrl"],
                "path": met_config["huggingFaceMetadata"]["path"],
                "bytes": met_config["huggingFaceMetadata"]["bytes"],
                "sha256": met_config["huggingFaceMetadata"]["sha256"],
                "sourceReportedLicense": met_config["huggingFaceMetadata"]["sourceReportedLicense"],
            },
            "fluxSourceIndex": {
                "dataset": recipe["syntheticDevelopmentSource"]["dataset"],
                "revision": recipe["syntheticDevelopmentSource"]["revision"],
                "rows": len(flux_source_rows),
                "compressedSha256": recipe["syntheticDevelopmentSource"]["sourceIndexSha256"],
                "expandedSha256": bytes_digest(packet["flux-source-index.json"]),
            },
        },
        "counts": {
            "total": len(training_rows),
            "trainingFeatureViews": int(recipe["expectedTrainingFeatureViews"]),
            "validation": len(validation_rows),
            "validationFeatureViews": int(recipe["expectedValidationFeatureViews"]),
            "regression": len(regression_rows),
            "regressionFeatureViews": int(recipe["expectedRegressionFeatureViews"]),
            "h3MetHoldout": len(holdout_rows),
            "consumedMetProbe": len(consumed_probe_rows),
        },
        "sourceCounts": source_counts,
        "classCounts": class_counts,
        "selectorSourceCounts": dict(Counter(str(row["source"]) for row in validation_rows)),
        "selectorClassCounts": {
            "real": sum(int(row["label"]) == 0 for row in validation_rows),
            "synthetic": sum(int(row["label"]) == 1 for row in validation_rows),
        },
        "selectionOrder": ["h3-met-holdout", "m3-met-development", "m3-flux-development", "m3-met-training"],
        "evaluationExclusions": exclusions,
        "perceptualOverlapReview": {
            "path": recipe["perceptualOverlapReview"],
            "sha256": bytes_digest(packet["perceptual-review.json"]),
            "reviewedPairCount": 0,
            "hammingThreshold": recipe["perceptualDuplicateHammingThreshold"],
        },
        "trainingPerceptualOverlapReview": {
            "path": recipe["trainingPerceptualOverlapReview"],
            "sha256": bytes_digest(packet["training-evaluation-perceptual-review.json"]),
            "reviewedPairCount": carried_review_count,
            "hammingThreshold": recipe["perceptualDuplicateHammingThreshold"],
        },
        "overlapWithEvaluation": {
            "ids": 0,
            "imageHashes": 0,
            "sourceGroupIds": 0,
            "unreviewedPerceptualDhashPairsAtOrBelowThreshold": 0,
            "reviewedVisuallyDistinctDhashPairsAtOrBelowThreshold": carried_review_count,
        },
        "newM3OverlapWithEvaluation": {
            "ids": 0,
            "imageHashes": 0,
            "sourceGroupIds": 0,
            "perceptualDhashPairsAtOrBelowThreshold": 0,
            "policy": "Every new Met or FLUX cross-pool match at or below the frozen threshold was rejected; no exceptions were retained.",
        },
        "carriedForwardM2PerceptualReview": {
            "path": recipe["baseTraining"]["trainingEvaluationPerceptualReview"]["path"],
            "sha256": recipe["baseTraining"]["trainingEvaluationPerceptualReview"]["sha256"],
            "reviewedPairCount": carried_review_count,
            "policy": "The unchanged M2 prefix carries forward these previously reviewed visually distinct pairs; none comes from the new M3 rows.",
        },
        "historicalCounts": historical_counts,
        "exclusionCountsBeforeProbe": counts_before_probe,
        "metCsvPolicyRejectCounts": csv_reject_counts,
        "materializationRejectCounts": dict(Counter(str(row["reason"]) for row in rejects)),
        "holdoutReservedBeforeTrainingSelection": True,
        "consumedDevelopmentRowsUsedForSelection": False,
        "scoresReadDuringSelection": False,
    }
    packet["selection-summary.json"] = canonical_json(summary, pretty=True)
    return packet


def write_local(recipe: dict[str, Any], packet: dict[str, bytes]) -> None:
    data_root = validate_data_root(ROOT / str(recipe["output"]["dataRoot"]))
    for name in (
        "train-manifest.jsonl",
        "validation-manifest.jsonl",
        "selection-summary.json",
        "rejects.jsonl",
        "attribution.json",
        "perceptual-review.json",
        "training-evaluation-perceptual-review.json",
    ):
        atomic_write(safe_output_path(data_root, name), packet[name])
    holdout_root = validate_data_root(ROOT / str(recipe["output"]["holdoutDataRoot"]))
    atomic_write(
        safe_output_path(holdout_root, "manifest.jsonl"),
        packet["h3-met-holdout-manifest.jsonl"],
    )


def publish(recipe: dict[str, Any], packet: dict[str, bytes]) -> None:
    evidence_root = ROOT / str(recipe["output"]["evidenceRoot"])
    evidence_root.mkdir(parents=True, exist_ok=True)
    for name, value in packet.items():
        if name == "met-development-probe-v1.jsonl":
            destination = ROOT / str(recipe["metSource"]["developmentProbe"]["manifestPath"])
        else:
            destination = evidence_root / (
                f"{name}.gz" if name in COMPRESSED_PACKET_NAMES else name
            )
        atomic_write(destination, deterministic_gzip(value) if name in COMPRESSED_PACKET_NAMES else value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialize", action="store_true")
    parser.add_argument("--write-local", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--offline", action="store_true")
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
    print(packet["selection-summary.json"].decode(), flush=True)


if __name__ == "__main__":
    main()
