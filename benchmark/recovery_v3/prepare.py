"""Materialize the score-blind replacement confirmation and web-negative sets.

Selection consumes only pinned public dataset bytes and the complete historical
exclusion index.  It never imports ProofLens inference code or reads model
outputs.  Pixels stay under the ignored benchmark data directory; the tracked
packet contains only manifests, provenance, and overlap evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import md5, sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Iterable
import urllib.error
import urllib.request

import pyarrow.parquet as parquet
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[2]
RECIPE_PATH = Path(__file__).with_name("recipe.json")
DHASH_PATTERN = re.compile(r"^[0-9a-f]{16}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = sha256() if algorithm == "sha256" else md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def bytes_digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def canonical_json(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def write_atomic(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def reject_symlink_components(root: Path, path: Path, *, label: str) -> Path:
    """Require ``path`` to remain below ``root`` without traversing a symlink."""

    lexical_root = root.absolute()
    lexical = path.absolute()
    try:
        relative = lexical.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its allowed root: {path}") from error
    current = lexical_root
    if current.is_symlink():
        raise ValueError(f"Symlinked {label} component: {current}")
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(f"Symlinked {label} component: {current}")
    resolved_root = lexical_root.resolve(strict=False)
    resolved = lexical.resolve(strict=False)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Resolved {label} escapes its allowed root: {path}")
    return lexical


def validate_data_root(path: Path) -> Path:
    """Require a lexical and resolved non-symlink path below benchmark/data."""

    allowed = (ROOT / "benchmark" / "data").absolute()
    lexical = path.absolute()
    try:
        lexical.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Data root escapes benchmark/data: {path}") from error
    return reject_symlink_components(allowed, lexical, label="data-root")


def safe_output_path(data_root: Path, relative: str) -> Path:
    safe_root = validate_data_root(data_root)
    candidate = (safe_root / relative).absolute()
    try:
        candidate.relative_to(safe_root)
    except ValueError as error:
        raise ValueError(f"Output path escapes data root: {relative}") from error
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe output path: {relative}")
    return reject_symlink_components(safe_root, candidate, label="data path")


def write_data_atomic(data_root: Path, relative: str, value: bytes) -> None:
    """Write a selected byte only after pre- and post-mkdir containment checks."""

    target = safe_output_path(data_root, relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = safe_output_path(data_root, relative)
    write_atomic(target, value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def priority(namespace: str, value: str | int) -> str:
    return sha256(f"{namespace}{value}".encode()).hexdigest()


def image_facts(value: bytes) -> tuple[int, int, str, str]:
    with Image.open(BytesIO(value)) as opened:
        opened.verify()
    with Image.open(BytesIO(value)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        resized = image.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
        pixels = list(resized.getdata())
        source_format = (opened.format or "JPEG").lower()
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    extension = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(source_format, ".jpg")
    return width, height, f"{bits:016x}", extension


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def hamming_neighbors_16(value: int) -> Iterable[int]:
    yield value
    for first in range(16):
        yield value ^ (1 << first)
    for first in range(16):
        for second in range(first + 1, 16):
            yield value ^ (1 << first) ^ (1 << second)


class PerceptualIndex:
    """Threshold-complete 4x16-bit dHash index for Hamming distance <= 8."""

    def __init__(self, rows: Iterable[tuple[str, str]] = ()) -> None:
        self._buckets: list[dict[int, list[tuple[str, int]]]] = [{}, {}, {}, {}]
        self._values: dict[str, int] = {}
        for perceptual_hash, identifier in rows:
            self.add(perceptual_hash, identifier)

    def add(self, perceptual_hash: str, identifier: str) -> None:
        if not DHASH_PATTERN.fullmatch(perceptual_hash) or identifier in self._values:
            raise ValueError(f"Invalid or duplicate perceptual owner: {identifier}")
        value = int(perceptual_hash, 16)
        self._values[identifier] = value
        for block in range(4):
            key = (value >> (block * 16)) & 0xFFFF
            self._buckets[block].setdefault(key, []).append((identifier, value))

    def matches(self, perceptual_hash: str, threshold: int) -> list[dict[str, object]]:
        if threshold > 8:
            raise ValueError("The 4x16-bit index is proven complete only through Hamming distance 8")
        value = int(perceptual_hash, 16)
        candidates: dict[str, int] = {}
        for block, buckets in enumerate(self._buckets):
            block_value = (value >> (block * 16)) & 0xFFFF
            for neighbor in hamming_neighbors_16(block_value):
                for identifier, candidate_value in buckets.get(neighbor, []):
                    candidates[identifier] = candidate_value
        return [
            {
                "matchingId": identifier,
                "matchingDhash64": f"{candidate_value:016x}",
                "distance": (value ^ candidate_value).bit_count(),
            }
            for identifier, candidate_value in sorted(candidates.items())
            if (value ^ candidate_value).bit_count() <= threshold
        ]

    def __len__(self) -> int:
        return len(self._values)


def download(
    url: str,
    output: Path,
    expected_bytes: int,
    expected_sha256: str,
    *,
    allow_download: bool,
    allowed_root: Path,
) -> None:
    safe_root = validate_data_root(allowed_root)
    try:
        relative = str(output.absolute().relative_to(safe_root))
    except ValueError as error:
        raise ValueError(f"Pinned source escapes its allowed root: {output}") from error
    output = safe_output_path(safe_root, relative)
    if output.is_file() and output.stat().st_size == expected_bytes and digest(output) == expected_sha256:
        return
    if not allow_download:
        raise ValueError(f"Pinned source is unavailable in offline verification: {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = safe_output_path(safe_root, relative)
    partial_relative = str(output.with_suffix(output.suffix + ".partial").relative_to(safe_root))
    partial = safe_output_path(safe_root, partial_relative)
    for attempt in range(6):
        existing = partial.stat().st_size if partial.exists() else 0
        if existing > expected_bytes:
            partial.unlink()
            existing = 0
        request = urllib.request.Request(
            url,
            headers={
                "Range": f"bytes={existing}-" if existing else "bytes=0-",
                "User-Agent": "ProofLens/1.0 (https://github.com/baney75/prooflens)",
            },
        )
        try:
            partial = safe_output_path(safe_root, partial_relative)
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("ab" if existing else "wb") as handle:
                while chunk := response.read(8 * 1024 * 1024):
                    handle.write(chunk)
            if partial.stat().st_size == expected_bytes:
                break
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 20))
    if partial.stat().st_size != expected_bytes or digest(partial) != expected_sha256:
        raise ValueError(f"Pinned download failed size/hash verification: {output.name}")
    output = safe_output_path(safe_root, relative)
    partial = safe_output_path(safe_root, partial_relative)
    os.replace(partial, output)


def historical_evidence(
    recipe: dict[str, Any],
) -> tuple[set[str], set[str], set[str], PerceptualIndex, dict[str, int]]:
    exclusions = recipe["historicalExclusions"]
    path = ROOT / exclusions["historicalPerceptualIndexPath"]
    compressed = path.read_bytes()
    if bytes_digest(compressed) != exclusions["historicalPerceptualIndexSha256"]:
        raise ValueError("Historical perceptual exclusion index changed")
    expanded = gzip.decompress(compressed)
    if bytes_digest(expanded) != exclusions["historicalPerceptualExpandedSha256"]:
        raise ValueError("Historical perceptual exclusion expansion changed")
    packet = json.loads(expanded)
    expected_inputs = {
        "largeTrainManifestSha256": exclusions["largeTrainManifestSha256"],
        "evaluationPerceptualCompressedSha256": exclusions["evaluationPerceptualCompressedSha256"],
        "evaluationPerceptualExpandedSha256": exclusions["evaluationPerceptualExpandedSha256"],
        "legacyExclusionsSha256": exclusions["legacyExclusionsSha256"],
        "docciMetadataSha256": exclusions["docciMetadata"]["sha256"],
    }
    if (
        packet.get("schemaVersion") != 1
        or packet.get("algorithm") != recipe["overlapPolicy"]["perceptualDhash64"]["algorithm"]
        or packet.get("maximumHammingDistance")
        != recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"]
        or any(packet.get("inputs", {}).get(key) != value for key, value in expected_inputs.items())
    ):
        raise ValueError("Historical perceptual exclusion contract changed")
    rows = packet.get("items", [])
    if len(rows) != int(exclusions["expectedUniqueItems"]):
        raise ValueError("Historical perceptual exclusion count changed")
    ids: set[str] = set()
    hashes: set[str] = set()
    groups: set[str] = set()
    index = PerceptualIndex()
    for row in rows:
        identifier = str(row.get("id", ""))
        image_hash = str(row.get("imageSha256", ""))
        group = str(row.get("sourceGroupId", ""))
        value = str(row.get("perceptualDhash64", ""))
        if not identifier or not group or not SHA256_PATTERN.fullmatch(image_hash) or not DHASH_PATTERN.fullmatch(value):
            raise ValueError("Historical perceptual exclusion row is malformed")
        if identifier in ids or image_hash in hashes:
            raise ValueError("Historical perceptual exclusion IDs/bytes are not unique")
        ids.add(identifier)
        hashes.add(image_hash)
        groups.add(group)
        index.add(value, identifier)
    counts = {
        "ids": len(ids),
        "imageSha256": len(hashes),
        "sourceGroupIds": len(groups),
        "perceptualDhash64": len(index),
    }
    if packet.get("counts", {}).get("items") != len(ids):
        raise ValueError("Historical perceptual exclusion summary changed")
    return ids, hashes, groups, index, counts


def qualify(
    candidate: dict[str, Any],
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
    threshold: int,
) -> dict[str, object] | None:
    if candidate["id"] in ids:
        return {"reason": "id", "matchingId": candidate["id"]}
    if candidate["imageSha256"] in hashes:
        return {"reason": "imageSha256", "imageSha256": candidate["imageSha256"]}
    if candidate["sourceGroupId"] in groups:
        return {"reason": "sourceGroupId", "sourceGroupId": candidate["sourceGroupId"]}
    matches = perceptual.matches(str(candidate["perceptualDhash64"]), threshold)
    if matches:
        return {"reason": "perceptualDhash64", **min(matches, key=lambda row: (int(row["distance"]), str(row["matchingId"])))}
    return None


def admit(
    candidate: dict[str, Any],
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
) -> None:
    ids.add(str(candidate["id"]))
    hashes.add(str(candidate["imageSha256"]))
    groups.add(str(candidate["sourceGroupId"]))
    perceptual.add(str(candidate["perceptualDhash64"]), str(candidate["id"]))


def public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key not in {"imageBytes", "extension", "protocol"}}


def materialize_coxy(
    recipe: dict[str, Any],
    source_root: Path,
    output_root: Path,
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
    rejects: list[dict[str, object]],
    *,
    allow_download: bool,
) -> list[dict[str, Any]]:
    config = recipe["confirmatory"]["synthetic"]
    source_directory = source_root / "source" / "coxy7"
    revision = str(config["revision"])
    shard_tables: dict[int, Any] = {}
    for shard in config["shards"]:
        destination = source_directory / Path(shard["path"]).name
        url = f"https://huggingface.co/datasets/{config['dataset']}/resolve/{revision}/{shard['path']}?download=true"
        download(
            url,
            destination,
            int(shard["bytes"]),
            str(shard["sha256"]),
            allow_download=allow_download,
            allowed_root=source_root,
        )
        table = parquet.read_table(
            destination,
            columns=["generator", "uid", "image", "original_prompt", "format", "height", "width"],
        )
        if table.num_rows != int(shard["rows"]):
            raise ValueError(f"Pinned Coxy7 shard row count changed: {shard['path']}")
        shard_tables[int(shard["globalRowStart"])] = table

    offsets = sorted(
        range(int(config["globalOffsetStartInclusive"]), int(config["globalOffsetEndExclusive"])),
        key=lambda value: priority(str(config["priorityNamespace"]), value),
    )[: int(config["candidateReserve"])]
    selected: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    threshold = int(recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"])
    for offset in offsets:
        shard_start = max(value for value in shard_tables if value <= offset)
        table = shard_tables[shard_start]
        local = offset - shard_start
        generator = str(table["generator"][local].as_py())
        uid = str(table["uid"][local].as_py())
        if generator != config["generator"] or uid in seen_uids:
            raise ValueError(f"Pinned Coxy7 Infinity interval changed at offset {offset}")
        seen_uids.add(uid)
        image_value = table["image"][local].as_py()
        image_bytes = image_value.get("bytes") if isinstance(image_value, dict) else None
        if not isinstance(image_bytes, bytes):
            raise ValueError(f"Coxy7 row {offset} does not contain embedded image bytes")
        width, height, dhash, extension = image_facts(image_bytes)
        identifier = f"coxy7:{revision}:Infinity:{uid}"
        group_id = f"coxy7:{revision}:prompt:{uid}"
        candidate: dict[str, Any] = {
            "id": identifier,
            "dataset": config["dataset"],
            "datasetRevision": revision,
            "sourceSplit": "fake",
            "globalOffset": offset,
            "selectionPriority": priority(str(config["priorityNamespace"]), offset),
            "sourceGroupId": group_id,
            "groupId": group_id,
            "promptSha256": bytes_digest(str(table["original_prompt"][local].as_py()).strip().encode()),
            "imageSha256": bytes_digest(image_bytes),
            "perceptualDhash64": dhash,
            "label": 1,
            "source": "coxy7-infinity",
            "sourceReportedLicense": config["sourceReportedLicense"],
            "width": width,
            "height": height,
        }
        issue = qualify(candidate, ids, hashes, groups, perceptual, threshold)
        if issue is not None:
            rejects.append({"candidateId": identifier, "protocol": "confirmatory-v2", **issue})
            continue
        filename = f"{offset:05d}-{uid}{extension}"
        candidate["path"] = f"confirmatory/synthetic/infinity/{filename}"
        write_data_atomic(output_root, candidate["path"], image_bytes)
        admit(candidate, ids, hashes, groups, perceptual)
        selected.append(candidate)
        if len(selected) == int(config["target"]):
            break
    if len(selected) != int(config["target"]):
        raise ValueError(f"Only {len(selected)} Coxy7 Infinity rows survived frozen overlap checks")
    return selected


def load_stock_candidates(
    recipe: dict[str, Any], source_root: Path, *, allow_download: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, object]]]:
    config = recipe["stockImagesSource"]
    source_directory = source_root / "source" / "stockimages-cc0"
    candidates: list[dict[str, Any]] = []
    source_rejects: list[dict[str, object]] = []
    global_offset = 0
    for shard_index, shard in enumerate(config["shards"]):
        filename = Path(shard["path"]).name
        destination = source_directory / filename
        url = f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['revision']}/{shard['path']}?download=true"
        download(
            url,
            destination,
            int(shard["bytes"]),
            str(shard["sha256"]),
            allow_download=allow_download,
            allowed_root=source_root,
        )
        file = parquet.ParquetFile(destination)
        if file.metadata.num_rows != int(shard["rows"]):
            raise ValueError(f"Pinned StockImages shard row count changed: {filename}")
        shard_row = 0
        for batch in file.iter_batches(batch_size=32, columns=["image", "tags"]):
            for row in batch.to_pylist():
                image_value = row.get("image")
                value = image_value.get("bytes") if isinstance(image_value, dict) else None
                if not isinstance(value, bytes):
                    raise ValueError(f"StockImages row has no embedded bytes: {filename}:{shard_row}")
                identifier = f"stockimages-cc0:{config['revision']}:{filename}:{shard_row}"
                try:
                    width, height, dhash, extension = image_facts(value)
                except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
                    source_rejects.append(
                        {
                            "candidateId": identifier,
                            "protocol": "source-screen",
                            "reason": "image-decode",
                            "sourceShard": shard["path"],
                            "sourceRow": shard_row,
                            "imageSha256": bytes_digest(value),
                        }
                    )
                    shard_row += 1
                    global_offset += 1
                    continue
                group = f"stockimages-cc0:{config['revision']}:row:{filename}:{shard_row}"
                candidates.append(
                    {
                        "id": identifier,
                        "dataset": config["dataset"],
                        "datasetRevision": config["revision"],
                        "sourceSplit": "train",
                        "sourceShard": shard["path"],
                        "sourceShardSha256": shard["sha256"],
                        "sourceRow": shard_row,
                        "globalOffset": global_offset,
                        "sourceGroupId": group,
                        "groupId": group,
                        "tagsSha256": bytes_digest(str(row.get("tags", "")).strip().encode()),
                        "imageSha256": bytes_digest(value),
                        "perceptualDhash64": dhash,
                        "label": 0,
                        "source": "stockimages-cc0",
                        "sourceReportedLicense": config["sourceReportedLicense"],
                        "width": width,
                        "height": height,
                        "imageBytes": value,
                        "extension": extension,
                    }
                )
                shard_row += 1
                global_offset += 1
    if len(candidates) + len(source_rejects) != int(config["candidateRows"]):
        raise ValueError("Pinned StockImages candidate count changed")
    return candidates, source_rejects


def select_stock(
    candidates: list[dict[str, Any]],
    recipe: dict[str, Any],
    output_root: Path,
    ids: set[str],
    hashes: set[str],
    groups: set[str],
    perceptual: PerceptualIndex,
    rejects: list[dict[str, object]],
    *,
    protocol: str,
    target: int,
    namespace: str,
    directory: str,
) -> list[dict[str, Any]]:
    threshold = int(recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"])
    ordered = sorted(candidates, key=lambda row: priority(namespace, str(row["id"])))
    selected: list[dict[str, Any]] = []
    for original in ordered:
        candidate = dict(original)
        candidate["selectionPriority"] = priority(namespace, str(candidate["id"]))
        issue = qualify(candidate, ids, hashes, groups, perceptual, threshold)
        if issue is not None:
            rejects.append({"candidateId": candidate["id"], "protocol": protocol, **issue})
            continue
        filename = f"{int(candidate['globalOffset']):05d}-{str(candidate['imageSha256'])[:16]}{candidate['extension']}"
        candidate["path"] = f"{directory}/{filename}"
        write_data_atomic(output_root, candidate["path"], candidate["imageBytes"])
        admit(candidate, ids, hashes, groups, perceptual)
        selected.append(candidate)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise ValueError(f"Only {len(selected)} StockImages rows survived {protocol} overlap checks")
    return selected


def manifest_bytes(rows: Iterable[dict[str, Any]], *, split: str) -> bytes:
    output = []
    for index, original in enumerate(sorted(rows, key=lambda row: (int(row["label"]), str(row["source"]), str(row["id"])))):
        row = public_candidate(original)
        row["rowIndex"] = index
        row["split"] = split
        output.append(json.dumps(row, separators=(",", ":"), sort_keys=True))
    return ("\n".join(output) + "\n").encode()


def list_digest(values: Iterable[str]) -> str:
    return bytes_digest(("\n".join(values) + "\n").encode())


def derive_packet(
    recipe: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    allow_download: bool,
) -> dict[str, bytes]:
    source_root = validate_data_root(source_root)
    output_root = validate_data_root(output_root)
    source_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = validate_data_root(source_root)
    output_root = validate_data_root(output_root)
    ids, hashes, groups, perceptual, historical_counts = historical_evidence(recipe)
    rejects: list[dict[str, object]] = []
    review_source = json.loads((ROOT / recipe["reviewedPerceptualPairsPath"]).read_text())
    if review_source.get("pairs") != []:
        raise ValueError("Replacement v2 admits no perceptual overlap exceptions")

    synthetic = materialize_coxy(
        recipe,
        source_root,
        output_root,
        ids,
        hashes,
        groups,
        perceptual,
        rejects,
        allow_download=allow_download,
    )
    stock_candidates, source_rejects = load_stock_candidates(recipe, source_root, allow_download=allow_download)
    rejects.extend(source_rejects)
    real_config = recipe["confirmatory"]["real"]
    real = select_stock(
        stock_candidates,
        recipe,
        output_root,
        ids,
        hashes,
        groups,
        perceptual,
        rejects,
        protocol="confirmatory-v2",
        target=int(real_config["target"]),
        namespace=str(real_config["priorityNamespace"]),
        directory="confirmatory/real/stockimages-cc0",
    )
    web_config = recipe["webNegative"]
    web = select_stock(
        stock_candidates,
        recipe,
        output_root,
        ids,
        hashes,
        groups,
        perceptual,
        rejects,
        protocol="web-negative-v2",
        target=int(web_config["items"]),
        namespace=str(web_config["priorityNamespace"]),
        directory="web-negative/real/stockimages-cc0",
    )

    confirmatory_rows = real + synthetic
    confirmatory = manifest_bytes(confirmatory_rows, split="confirmatory-v2")
    web_negative = manifest_bytes(web, split="web-negative-v2")
    source_counts = Counter(str(row["source"]) for row in confirmatory_rows)
    web_counts = Counter(str(row["source"]) for row in web)
    review_bytes = canonical_json(
        {
            "schemaVersion": 2,
            "algorithm": recipe["overlapPolicy"]["perceptualDhash64"]["algorithm"],
            "maximumHammingDistance": recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"],
            "policy": "Every candidate at or below the frozen dHash threshold is rejected; replacement v2 admits no overlap exceptions.",
            "retainedPairs": [],
            "rejectedCandidateCount": len(rejects),
        },
        pretty=True,
    )
    attribution_bytes = canonical_json(
        {
            "schemaVersion": 2,
            "datasets": [
                {
                    "name": recipe["confirmatory"]["synthetic"]["dataset"],
                    "revision": recipe["confirmatory"]["synthetic"]["revision"],
                    "sourceReportedLicense": recipe["confirmatory"]["synthetic"]["sourceReportedLicense"],
                    "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
                    "homepage": "https://huggingface.co/datasets/Coxy7/AIGI-Detection-Quality-Paradox",
                    "sourceMaterialUrl": "https://huggingface.co/datasets/Coxy7/AIGI-Detection-Quality-Paradox/tree/9244882a51dbe33e658fe514488692155d20e5dd",
                    "creators": [
                        "Yao Xiao",
                        "Binbin Yang",
                        "Weiyan Chen",
                        "Jiahao Chen",
                        "Zijie Cao",
                        "Ziyi Dong",
                        "Xiangyang Ji",
                        "Liang Lin",
                        "Wei Ke",
                        "Pengxu Wei",
                    ],
                    "citation": "Yao Xiao, Binbin Yang, Weiyan Chen, Jiahao Chen, Zijie Cao, Ziyi Dong, Xiangyang Ji, Liang Lin, Wei Ke, and Pengxu Wei, Are High-Quality AI-Generated Images More Difficult for Models to Detect?, ICML 2025",
                    "notice": "300 unmodified Infinity image bytes are retained only in ignored benchmark/data/replacement-v2. This public repository distributes provenance manifests, not source pixels. ProofLens performed deterministic selection and generated the manifests; no endorsement is implied.",
                    "use": "300 Infinity-generated confirmation images; source pixels are local-only",
                },
                {
                    "name": recipe["stockImagesSource"]["dataset"],
                    "revision": recipe["stockImagesSource"]["revision"],
                    "sourceReportedLicense": recipe["stockImagesSource"]["sourceReportedLicense"],
                    "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "homepage": "https://huggingface.co/datasets/KoalaAI/StockImages-CC0",
                    "sourceMaterialUrl": "https://huggingface.co/datasets/KoalaAI/StockImages-CC0/tree/206f3575579f1187548c6f47042ae9174c0a51fc",
                    "citation": "KoalaAI, CC0 Stock Images Dataset",
                    "notice": "The pinned dataset card reports CC0-1.0. That uploader-provided statement is not an independent verification of contributor ownership or other third-party rights. Source pixels remain only in ignored benchmark/data/replacement-v2.",
                    "use": "300 confirmation and 319 separate web-negative stock photographs; exact rows and bytes are disjoint, source dataset is shared and disclosed",
                },
            ],
        },
        pretty=True,
    )
    selection = {
        "schemaVersion": 2,
        "name": recipe["name"],
        "scoreBlindSelection": True,
        "observedV1ResultsUsed": False,
        "recipeSha256": digest(RECIPE_PATH),
        "sourcePerceptualReviewSha256": digest(ROOT / recipe["reviewedPerceptualPairsPath"]),
        "historicalPerceptualIndexSha256": recipe["historicalExclusions"]["historicalPerceptualIndexSha256"],
        "dataRoot": recipe["output"]["dataRoot"],
        "artifacts": {
            "perceptualReviewSha256": bytes_digest(review_bytes),
            "attributionSha256": bytes_digest(attribution_bytes),
        },
        "historicalCounts": historical_counts,
        "confirmatory": {
            "manifest": recipe["output"]["confirmatoryManifest"],
            "manifestSha256": bytes_digest(confirmatory),
            "items": len(confirmatory_rows),
            "labels": {"real": len(real), "synthetic": len(synthetic)},
            "sources": dict(sorted(source_counts.items())),
            "selectedIdsSha256": list_digest(sorted(str(row["id"]) for row in confirmatory_rows)),
            "selectedImageHashesSha256": list_digest(sorted(str(row["imageSha256"]) for row in confirmatory_rows)),
            "infinityOffsetsSha256": list_digest(
                str(row["globalOffset"]) for row in sorted(synthetic, key=lambda row: int(row["globalOffset"]))
            ),
        },
        "webNegative": {
            "manifest": recipe["output"]["webNegativeManifest"],
            "manifestSha256": bytes_digest(web_negative),
            "items": len(web),
            "sources": dict(sorted(web_counts.items())),
            "selectedIdsSha256": list_digest(sorted(str(row["id"]) for row in web)),
            "selectedImageHashesSha256": list_digest(sorted(str(row["imageSha256"]) for row in web)),
            "sharesDatasetWithConfirmatoryReal": True,
            "sharedIds": 0,
            "sharedImageSha256": 0,
        },
        "overlap": {
            "id": 0,
            "imageSha256": 0,
            "sourceGroupId": 0,
            "perceptualDhash64AtOrBelow8": 0,
            "reviewedVisuallyDistinctPairs": 0,
            "rejectedCandidates": len(rejects),
        },
        "rejectedCandidates": rejects,
    }
    return {
        "confirmatoryManifest": confirmatory,
        "webNegativeManifest": web_negative,
        "selectionEvidence": canonical_json(selection, pretty=True),
        "perceptualReview": review_bytes,
        "attribution": attribution_bytes,
    }


def write_packet(recipe: dict[str, Any], packet: dict[str, bytes]) -> None:
    output = recipe["output"]
    for key, value in packet.items():
        write_atomic(ROOT / output[key], value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Require all pinned source shards to already be present")
    args = parser.parse_args()
    recipe = json.loads(RECIPE_PATH.read_text())
    data_root = validate_data_root(ROOT / recipe["output"]["dataRoot"])
    packet = derive_packet(
        recipe,
        source_root=data_root,
        output_root=data_root,
        allow_download=not args.offline,
    )
    write_packet(recipe, packet)
    selection = json.loads(packet["selectionEvidence"])
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
