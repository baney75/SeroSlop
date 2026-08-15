"""Materialize the score-blind M4 training and selector source packet.

The script never imports model inference code and accepts no H3 data root. It
selects the fresh selector before any new training row, then emits only
pixel-free public evidence. Source and selected pixels stay below ignored
``benchmark/data`` roots.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Iterable
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
RECIPE_PATH = ROOT / "benchmark/m4/recipe.json"
LOCKS_PATH = ROOT / "benchmark/m4/source-locks.json"
PUBLIC_EVIDENCE_ROOT = ROOT / "benchmark/evidence/m4"
USER_AGENT = "ProofLens-M4/1.0 (https://github.com/baney75/prooflens)"
COMPRESSED_ARTIFACTS = {
    "british-source-index.json",
    "rapidata-source-index.json",
    "rejects.jsonl",
    "train-manifest.jsonl",
}
DHASH_PATTERN = re.compile(r"^[0-9a-f]{16}$")


if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from benchmark.m4.contracts import (  # noqa: E402
    ALLOWED_BRITISH_DECADES,
    MODELS,
    british_book_id,
    canonical_json,
    classify_british_date,
    collect_rapidata_groups,
    digest_bytes,
    load_frozen_protocol,
    priority,
)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(row, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        for row in rows
    )


def load_jsonl_bytes(value: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in value.splitlines() if line]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return load_jsonl_bytes(path.read_bytes())


def deterministic_gzip(value: bytes) -> bytes:
    compressed = bytearray(gzip.compress(value, compresslevel=9, mtime=0))
    compressed[9] = 0xFF
    return bytes(compressed)


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reject_symlink_components(root: Path, path: Path, *, label: str) -> Path:
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
    allowed = (ROOT / "benchmark/data").absolute()
    lexical = path.absolute()
    try:
        lexical.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"M4 data root escapes benchmark/data: {path}") from error
    return reject_symlink_components(allowed, lexical, label="M4 data root")


def safe_output_path(root: Path, relative: str) -> Path:
    safe_root = validate_data_root(root)
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe M4 data path: {relative}")
    return reject_symlink_components(safe_root, safe_root / relative_path, label="M4 data path")


def write_data_atomic(root: Path, relative: str, value: bytes, expected_sha256: str) -> None:
    target = safe_output_path(root, relative)
    if target.is_file() and digest(target) == expected_sha256:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target = safe_output_path(root, relative)
    atomic_write(target, value)
    if digest(target) != expected_sha256:
        raise ValueError(f"Materialized M4 pixel digest changed: {relative}")


def image_facts(value: bytes) -> tuple[int, int, str, str]:
    from PIL import Image, ImageOps

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
    extension = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp"}.get(source_format)
    if extension is None:
        raise ValueError(f"Unsupported M4 source image format: {source_format}")
    return width, height, f"{bits:016x}", extension


def hamming_neighbors_16(value: int) -> Iterable[int]:
    yield value
    for first in range(16):
        yield value ^ (1 << first)
    for first in range(16):
        for second in range(first + 1, 16):
            yield value ^ (1 << first) ^ (1 << second)


class PerceptualIndex:
    """Threshold-complete four-block index for 64-bit dHash distance <= 8."""

    def __init__(self) -> None:
        self._buckets: list[dict[int, list[tuple[str, int]]]] = [{}, {}, {}, {}]
        self._values: dict[str, int] = {}

    def add(self, perceptual_hash: str, identifier: str) -> None:
        if not DHASH_PATTERN.fullmatch(perceptual_hash) or identifier in self._values:
            raise ValueError(f"Invalid or duplicate M4 perceptual owner: {identifier}")
        value = int(perceptual_hash, 16)
        self._values[identifier] = value
        for block in range(4):
            key = (value >> (block * 16)) & 0xFFFF
            self._buckets[block].setdefault(key, []).append((identifier, value))

    def matches(self, perceptual_hash: str, threshold: int) -> list[dict[str, Any]]:
        if threshold > 8 or not DHASH_PATTERN.fullmatch(perceptual_hash):
            raise ValueError("M4 perceptual lookup is valid only for 64-bit dHash threshold <= 8")
        value = int(perceptual_hash, 16)
        candidates: dict[str, int] = {}
        for block, buckets in enumerate(self._buckets):
            block_value = (value >> (block * 16)) & 0xFFFF
            for neighbor in hamming_neighbors_16(block_value):
                for identifier, candidate in buckets.get(neighbor, []):
                    candidates[identifier] = candidate
        return [
            {
                "matchingId": identifier,
                "matchingDhash64": f"{candidate:016x}",
                "distance": (value ^ candidate).bit_count(),
            }
            for identifier, candidate in sorted(candidates.items())
            if (value ^ candidate).bit_count() <= threshold
        ]

    def __len__(self) -> int:
        return len(self._values)


class OverlapState:
    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.ids: set[str] = set()
        self.hashes: set[str] = set()
        self.groups: set[str] = set()
        self.perceptual = PerceptualIndex()
        self.by_id: dict[str, str] = {}

    def add_frozen(self, row: dict[str, Any]) -> None:
        identifier = str(row["id"])
        image_hash = str(row["imageSha256"])
        group = str(row.get("sourceGroupId") or row.get("groupId") or identifier)
        perceptual_hash = str(row.get("perceptualDhash64") or "")
        if identifier in self.ids:
            if self.by_id[identifier] != image_hash:
                raise ValueError(f"Frozen M4 exclusion ID changed bytes: {identifier}")
            return
        if not DHASH_PATTERN.fullmatch(perceptual_hash):
            raise ValueError(f"Frozen M4 exclusion lacks dHash: {identifier}")
        self.ids.add(identifier)
        self.hashes.add(image_hash)
        self.groups.add(group)
        self.by_id[identifier] = image_hash
        self.perceptual.add(perceptual_hash, identifier)

    def issue(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        identifier = str(candidate["id"])
        image_hash = str(candidate["imageSha256"])
        group = str(candidate["sourceGroupId"])
        if identifier in self.ids:
            return {"reason": "id", "matchingId": identifier}
        if image_hash in self.hashes:
            return {"reason": "imageSha256", "imageSha256": image_hash}
        if group in self.groups:
            return {"reason": "sourceGroupId", "sourceGroupId": group}
        matches = self.perceptual.matches(str(candidate["perceptualDhash64"]), self.threshold)
        if matches:
            return {
                "reason": "perceptualDhash64",
                **min(matches, key=lambda row: (int(row["distance"]), str(row["matchingId"]))),
            }
        return None

    def admit_group(self, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return {"reason": "emptyGroup"}
        ids = [str(row["id"]) for row in candidates]
        hashes = [str(row["imageSha256"]) for row in candidates]
        groups = {str(row["sourceGroupId"]) for row in candidates}
        if len(set(ids)) != len(ids):
            return {"reason": "duplicateIdWithinGroup"}
        if len(set(hashes)) != len(hashes):
            return {"reason": "duplicateImageSha256WithinGroup"}
        if len(groups) != 1:
            return {"reason": "mixedSourceGroup"}
        for candidate in candidates:
            issue = self.issue(candidate)
            if issue is not None:
                return issue
        for candidate in candidates:
            identifier = str(candidate["id"])
            self.ids.add(identifier)
            self.hashes.add(str(candidate["imageSha256"]))
            self.groups.add(str(candidate["sourceGroupId"]))
            self.by_id[identifier] = str(candidate["imageSha256"])
            self.perceptual.add(str(candidate["perceptualDhash64"]), identifier)
        return None


def checked_file(relative: str, expected_sha256: str, *, label: str) -> Path:
    path = ROOT / relative
    if not path.is_file() or digest(path) != expected_sha256:
        raise ValueError(f"Frozen {label} changed: {relative}")
    return path


def load_base_training(recipe: dict[str, Any]) -> tuple[list[dict[str, Any]], bytes]:
    config = recipe["baseTraining"]
    path = checked_file(str(config["manifest"]), str(config["compressedSha256"]), label="M3 training manifest")
    expanded = gzip.decompress(path.read_bytes())
    if digest_bytes(expanded) != config["expandedSha256"]:
        raise ValueError("Frozen M3 training manifest expanded bytes changed")
    rows = load_jsonl_bytes(expanded)
    if len(rows) != config["items"]:
        raise ValueError("Frozen M3 training item count changed")
    if Counter(str(row["source"]) for row in rows) != Counter(config["sourceCounts"]):
        raise ValueError("Frozen M3 training source counts changed")
    return rows, expanded


def seed_exclusions(recipe: dict[str, Any], base_rows: list[dict[str, Any]]) -> tuple[OverlapState, dict[str, int]]:
    config = recipe["exclusionInputs"]["historicalPerceptualIndex"]
    path = checked_file(str(config["path"]), str(config["sha256"]), label="historical exclusion index")
    expanded = gzip.decompress(path.read_bytes())
    if digest_bytes(expanded) != config["expandedSha256"]:
        raise ValueError("Historical exclusion index expanded bytes changed")
    packet = json.loads(expanded)
    if len(packet.get("items", [])) != config["items"]:
        raise ValueError("Historical exclusion count changed")
    state = OverlapState(int(recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"]))
    for row in packet["items"]:
        state.add_frozen(row)
    for row in base_rows:
        state.add_frozen(row)
    exclusion_rows = 0
    for manifest in recipe["exclusionInputs"]["manifests"]:
        path = checked_file(str(manifest["path"]), str(manifest["sha256"]), label=str(manifest["role"]))
        rows = load_jsonl(path)
        exclusion_rows += len(rows)
        for row in rows:
            state.add_frozen(row)
    review = recipe["exclusionInputs"]["trainingEvaluationPerceptualReview"]
    review_path = checked_file(str(review["path"]), str(review["sha256"]), label="training/evaluation review")
    review_packet = json.loads(review_path.read_bytes())
    if len(review_packet.get("items", [])) != review["items"]:
        raise ValueError("Training/evaluation perceptual review changed")
    return state, {
        "historicalItems": len(packet["items"]),
        "baseTrainingItems": len(base_rows),
        "explicitExclusionRows": exclusion_rows,
        "uniqueIds": len(state.ids),
        "uniqueImageSha256": len(state.hashes),
        "uniqueSourceGroups": len(state.groups),
        "uniquePerceptualOwners": len(state.perceptual),
    }


def canonical_inventory(rows: Iterable[dict[str, Any]]) -> bytes:
    normalized = [
        {
            "path": str(row["path"]),
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
            "gitOid": str(row["gitOid"]),
        }
        for row in rows
    ]
    return canonical_json(sorted(normalized, key=lambda row: row["path"]))


def fetch_inventory(endpoint: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    url: str | None = endpoint
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            page = json.load(response)
            link = response.headers.get("Link", "")
        for row in page:
            if row.get("type") != "file" or not str(row.get("path", "")).endswith(".parquet"):
                continue
            lfs = row.get("lfs") or {}
            rows.append({
                "path": row["path"],
                "bytes": row["size"],
                "sha256": lfs.get("oid"),
                "gitOid": row.get("oid"),
            })
        match = re.search(r'<([^>]+)>; rel="next"', link)
        url = match.group(1) if match else None
    return rows


def verify_remote_inventories(locks: dict[str, Any]) -> None:
    for name in ("britishLibrary", "rapidata"):
        config = locks[name]
        rows = fetch_inventory(str(config["repositoryInventoryEndpoint"]))
        if digest_bytes(canonical_inventory(rows)) != config["repositoryInventoryCanonicalSha256"]:
            raise ValueError(f"Pinned {name} repository inventory changed")
        if name == "britishLibrary":
            ordered = sorted(
                rows,
                key=lambda row: (
                    priority("prooflens:m4:british-library-shard-v1:", str(row["path"])),
                    str(row["path"]),
                ),
            )[:32]
            if ordered != config["files"]:
                raise ValueError("British Library locked shards are not the frozen lowest-priority 32")
        elif sorted(rows, key=lambda row: row["path"]) != config["files"]:
            raise ValueError("Rapidata locked file inventory changed")


def download_locked(url: str, target: Path, row: dict[str, Any], *, allow_download: bool, source_root: Path) -> None:
    expected_bytes = int(row["bytes"])
    expected_hash = str(row["sha256"])
    relative = str(target.absolute().relative_to(validate_data_root(source_root)))
    target = safe_output_path(source_root, relative)
    if target.is_file() and target.stat().st_size == expected_bytes and digest(target) == expected_hash:
        return
    if not allow_download:
        raise ValueError(f"Pinned M4 source is unavailable offline: {row['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = safe_output_path(source_root, relative + ".partial")
    for attempt in range(8):
        existing = partial.stat().st_size if partial.is_file() else 0
        if existing > expected_bytes:
            partial.unlink()
            existing = 0
        headers = {"User-Agent": USER_AGENT}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                append = existing > 0 and getattr(response, "status", None) == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(8 * 1024 * 1024):
                        handle.write(chunk)
            if partial.stat().st_size == expected_bytes:
                break
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            if attempt == 7:
                raise
            time.sleep(min(2**attempt, 30))
    if partial.stat().st_size != expected_bytes or digest(partial) != expected_hash:
        raise ValueError(f"Pinned M4 source bytes changed: {row['path']}")
    os.replace(partial, target)


def ensure_sources(locks: dict[str, Any], source_root: Path, *, allow_download: bool) -> dict[str, list[Path]]:
    source_root = validate_data_root(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    source_root = validate_data_root(source_root)
    if allow_download:
        verify_remote_inventories(locks)
    output: dict[str, list[Path]] = {"britishLibrary": [], "rapidata": []}
    for name, directory in (("britishLibrary", "british-library"), ("rapidata", "rapidata")):
        config = locks[name]
        for row in config["files"]:
            target = safe_output_path(source_root, f"{directory}/{row['path']}")
            encoded = urllib.parse.quote(str(row["path"]), safe="/")
            url = f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['revision']}/{encoded}?download=true"
            download_locked(url, target, row, allow_download=allow_download, source_root=source_root)
            output[name].append(target)
    return output


def embedded_bytes(value: object, *, label: str) -> bytes:
    result = value.get("bytes") if isinstance(value, dict) else None
    if not isinstance(result, bytes):
        raise ValueError(f"{label} has no embedded image bytes")
    return result


def scan_british(
    files: list[Path], locks: dict[str, Any], recipe: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import pyarrow.parquet as parquet

    by_path = {str(row["path"]): row for row in locks["britishLibrary"]["files"]}
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_row_counts: dict[str, int] = {}
    for path in files:
        source_path = f"plates/{path.name}"
        lock = by_path[source_path]
        source_row = 0
        file = parquet.ParquetFile(path)
        for batch in file.iter_batches(batch_size=16, columns=["image", "date", "fname", "image_type"]):
            for row in batch.to_pylist():
                row_index = source_row
                source_row += 1
                decade, raw_date = classify_british_date(row.get("date"))
                if decade is None:
                    rejected.append({
                        "sourceShard": source_path,
                        "sourceShardSha256": lock["sha256"],
                        "sourceRow": row_index,
                        "rawDate": raw_date,
                        "reason": "date-not-in-frozen-strata",
                    })
                    continue
                value = embedded_bytes(row.get("image"), label=f"British Library {source_path}:{row_index}")
                width, height, dhash, extension = image_facts(value)
                book = british_book_id(row.get("fname"))
                identifier = f"british-library:{locks['britishLibrary']['revision']}:{source_path}:{row_index}"
                group = f"british-library:{locks['britishLibrary']['revision']}:book:{book}"
                candidates.append({
                    "id": identifier,
                    "dataset": locks["britishLibrary"]["dataset"],
                    "datasetRevision": locks["britishLibrary"]["revision"],
                    "sourceSplit": "plates/train",
                    "sourceShard": source_path,
                    "sourceShardSha256": lock["sha256"],
                    "sourceRow": row_index,
                    "bookId": book,
                    "decade": decade,
                    "fnameSha256": digest_bytes(str(row["fname"]).encode("utf-8")),
                    "imageSha256": digest_bytes(value),
                    "perceptualDhash64": dhash,
                    "sourceGroupId": group,
                    "label": 0,
                    "source": "british-library-plates",
                    "sourceReportedLicense": recipe["britishLibrary"]["sourceReportedLicense"],
                    "width": width,
                    "height": height,
                    "extension": extension,
                })
        source_row_counts[source_path] = source_row
    counts = Counter(str(row["decade"]) for row in candidates)
    distinct = Counter()
    for decade in ALLOWED_BRITISH_DECADES:
        distinct[decade] = len({row["bookId"] for row in candidates if row["decade"] == decade})
    if len(candidates) != recipe["britishLibrary"]["expectedCandidateRows"]:
        raise ValueError("British Library pinned-shard candidate count changed")
    if dict(distinct) != recipe["britishLibrary"]["expectedDistinctBooksByDecade"]:
        raise ValueError("British Library distinct-book counts changed")
    if len({str(row["bookId"]) for row in candidates}) != recipe["britishLibrary"]["expectedDistinctBooks"]:
        raise ValueError("British Library global distinct-book count changed")
    if sum(counts.values()) != len(candidates):
        raise AssertionError("British Library decade accounting changed")
    rejected_counts = dict(sorted(Counter(str(row["rawDate"]) for row in rejected).items()))
    eligibility = {
        "sourceRows": sum(source_row_counts.values()),
        "eligibleCandidateRows": len(candidates),
        "rejectedSourceRows": len(rejected),
        "rejectedDateCounts": rejected_counts,
        "sourceRowCounts": dict(sorted(source_row_counts.items())),
        "rejectedItems": sorted(
            rejected, key=lambda row: (str(row["sourceShard"]), int(row["sourceRow"])),
        ),
    }
    if (
        eligibility["sourceRows"] != recipe["britishLibrary"]["expectedSourceRows"]
        or eligibility["eligibleCandidateRows"] != recipe["britishLibrary"]["expectedCandidateRows"]
        or eligibility["rejectedSourceRows"] != recipe["britishLibrary"]["expectedRejectedSourceRows"]
        or eligibility["rejectedDateCounts"] != recipe["britishLibrary"]["expectedRejectedDates"]
        or digest_bytes(canonical_json(eligibility["sourceRowCounts"]))
        != recipe["britishLibrary"]["sourceRowCountsCanonicalSha256"]
    ):
        raise ValueError("British Library pinned-shard source eligibility changed")
    return candidates, eligibility


def scan_rapidata(files: list[Path], locks: dict[str, Any], recipe: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    import pyarrow.parquet as parquet

    vote_rows: list[dict[str, Any]] = []
    images: dict[str, dict[str, Any]] = {}
    source_rows = 0
    by_path = {str(row["path"]): row for row in locks["rapidata"]["files"]}
    for path in files:
        source_path = f"data/{path.name}"
        lock = by_path[source_path]
        file = parquet.ParquetFile(path)
        local_row = 0
        columns = ["prompt", "image1", "image2", "model1", "model2", "image1_path", "image2_path"]
        for batch in file.iter_batches(batch_size=16, columns=columns):
            for row in batch.to_pylist():
                mapped = {
                    "prompt": row.get("prompt"),
                    "model1": row.get("model1"),
                    "model2": row.get("model2"),
                    "image1Path": row.get("image1_path"),
                    "image2Path": row.get("image2_path"),
                }
                vote_rows.append(mapped)
                prompt_hash = digest_bytes(str(row.get("prompt", "")).encode("utf-8"))
                for side in (1, 2):
                    pathname = row.get(f"image{side}_path")
                    model = row.get(f"model{side}")
                    value = embedded_bytes(row.get(f"image{side}"), label=f"Rapidata {source_path}:{local_row}:{side}")
                    image_hash = digest_bytes(value)
                    prior = images.get(str(pathname))
                    if prior is not None:
                        if prior["imageSha256"] != image_hash or prior["model"] != model or prior["promptSha256"] != prompt_hash:
                            raise ValueError(f"Rapidata repeated path changed ownership or bytes: {pathname}")
                        continue
                    width, height, dhash, extension = image_facts(value)
                    identifier = f"rapidata:{locks['rapidata']['revision']}:{pathname}"
                    group = f"rapidata:{locks['rapidata']['revision']}:prompt:{prompt_hash}"
                    images[str(pathname)] = {
                        "id": identifier,
                        "dataset": locks["rapidata"]["dataset"],
                        "datasetRevision": locks["rapidata"]["revision"],
                        "sourceSplit": "default/train",
                        "firstSourceShard": source_path,
                        "firstSourceShardSha256": lock["sha256"],
                        "firstSourceRow": local_row,
                        "imagePath": pathname,
                        "promptSha256": prompt_hash,
                        "model": model,
                        "imageSha256": image_hash,
                        "perceptualDhash64": dhash,
                        "sourceGroupId": group,
                        "label": 1,
                        "source": recipe["rapidata"]["models"][model],
                        "sourceReportedLicense": recipe["rapidata"]["sourceReportedLicense"],
                        "width": width,
                        "height": height,
                        "extension": extension,
                    }
                local_row += 1
                source_rows += 1
    groups = collect_rapidata_groups(vote_rows)
    unique_by_model = Counter(str(row["model"]) for row in images.values())
    one_each = sum(all(len(row["models"].get(model, [])) >= 1 for model in MODELS) for row in groups.values())
    four_each = sum(all(len(row["models"].get(model, [])) == 4 for model in MODELS) for row in groups.values())
    config = recipe["rapidata"]
    if (
        source_rows != config["expectedVoteRows"]
        or len(groups) != config["expectedPromptGroups"]
        or len(images) != config["expectedUniquePaths"]
        or dict(unique_by_model) != config["expectedUniquePathsByModel"]
        or one_each != config["expectedOnePerFamilyGroups"]
        or four_each != config["expectedFourPerFamilyGroups"]
    ):
        raise ValueError("Rapidata pinned source capacity changed")
    return list(images.values()), groups


def _reject(rejects: list[dict[str, Any]], *, phase: str, candidate: str, issue: dict[str, Any]) -> None:
    rejects.append({"phase": phase, "candidateId": candidate, **issue})


def select_british_phase(
    candidates: list[dict[str, Any]],
    *,
    phase: str,
    quotas: dict[str, int],
    namespace: str,
    excluded_books: set[str],
    state: OverlapState,
    rejects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    books: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        books[str(candidate["decade"])][str(candidate["bookId"])].append(candidate)
    selected: list[dict[str, Any]] = []
    considered: set[str] = set()
    for decade in ALLOWED_BRITISH_DECADES:
        ordered_books = sorted(
            (book for book in books[decade] if book not in excluded_books),
            key=lambda book: (priority(namespace, book), book),
        )
        decade_selected = 0
        for book in ordered_books:
            considered.add(book)
            ordered_rows = sorted(
                books[decade][book],
                key=lambda row: (
                    priority(namespace, f"{book}:{row['sourceShard']}:{row['sourceRow']}"),
                    str(row["sourceShard"]), int(row["sourceRow"]),
                ),
            )
            accepted: dict[str, Any] | None = None
            for original in ordered_rows:
                candidate = dict(original)
                candidate["selectionPriority"] = priority(namespace, str(candidate["id"]))
                issue = state.admit_group([candidate])
                if issue is None:
                    accepted = candidate
                    break
                _reject(rejects, phase=phase, candidate=str(candidate["id"]), issue=issue)
            if accepted is None:
                continue
            selected.append(accepted)
            decade_selected += 1
            if decade_selected == int(quotas[decade]):
                break
        if decade_selected != int(quotas[decade]):
            raise ValueError(f"British Library {phase} capacity failed for decade {decade}")
    return selected, considered


def _rapidata_group_candidates(
    group: str,
    groups: dict[str, dict[str, Any]],
    images: dict[str, dict[str, Any]],
    *,
    selector: bool,
    image_namespace: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for model in MODELS:
        paths = groups[group]["models"].get(model, [])
        if len(paths) != 4:
            raise ValueError("Rapidata selected prompt group is not complete")
        ordered = sorted(paths, key=lambda path: (priority(image_namespace, f"{group}:{model}:{path}"), path))
        chosen = ordered[:1] if selector else ordered
        selected.extend(dict(images[path]) for path in chosen)
    return selected


def select_rapidata_phase(
    groups: dict[str, dict[str, Any]],
    image_rows: list[dict[str, Any]],
    *,
    phase: str,
    target_groups: int,
    namespace: str,
    image_namespace: str,
    excluded_groups: set[str],
    state: OverlapState,
    rejects: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    images = {str(row["imagePath"]): row for row in image_rows}
    complete = [
        group for group, row in groups.items()
        if all(len(row["models"].get(model, [])) == 4 for model in MODELS)
        and group not in excluded_groups
    ]
    ordered = sorted(complete, key=lambda group: (priority(namespace, group), group))
    selected: list[dict[str, Any]] = []
    considered: set[str] = set()
    selector = phase == "validation-synthetic"
    selected_groups = 0
    for group in ordered:
        considered.add(group)
        candidates = _rapidata_group_candidates(
            group, groups, images, selector=selector, image_namespace=image_namespace,
        )
        for candidate in candidates:
            candidate["selectionPriority"] = priority(namespace, str(candidate["id"]))
        if selector:
            issue = state.admit_group(candidates)
            if issue is not None:
                _reject(rejects, phase=phase, candidate=group, issue=issue)
                continue
            accepted = candidates
        else:
            accepted = []
            accepted_models: set[str] = set()
            for candidate in candidates:
                issue = state.issue(candidate)
                if issue is not None:
                    _reject(rejects, phase=phase, candidate=str(candidate["id"]), issue=issue)
                    continue
                accepted.append(candidate)
                accepted_models.add(str(candidate["model"]))
            missing = sorted(set(MODELS) - accepted_models)
            if missing:
                _reject(
                    rejects,
                    phase=phase,
                    candidate=group,
                    issue={"reason": "missingCleanFamily", "missingFamilies": missing},
                )
                continue
            issue = state.admit_group(accepted)
            if issue is not None:
                _reject(rejects, phase=phase, candidate=group, issue=issue)
                continue
        selected.extend(accepted)
        selected_groups += 1
        if selected_groups == target_groups:
            return selected, considered
    raise ValueError(f"Rapidata {phase} prompt-group capacity is insufficient")


def public_candidate(original: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in original.items() if key not in {"extension"}}


def source_index_candidate(original: dict[str, Any]) -> dict[str, Any]:
    """Retain the source format needed for a pixel-free output-path replay."""
    return dict(original)


def reindex(rows: Iterable[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = public_candidate(original)
        row["rowIndex"] = index
        row["split"] = split
        output.append(row)
    return output


def assign_paths(
    british_selector: list[dict[str, Any]],
    rapidata_selector: list[dict[str, Any]],
    british_training: list[dict[str, Any]],
    rapidata_training: list[dict[str, Any]],
) -> None:
    for phase, rows in (("validation", british_selector), ("train", british_training)):
        for row in rows:
            row["path"] = (
                f"{phase}/real/british-library-plates/{row['bookId']}-"
                f"{str(row['imageSha256'])[:16]}{row['extension']}"
            )
    for phase, rows in (("validation", rapidata_selector), ("train", rapidata_training)):
        for row in rows:
            stem = Path(str(row["imagePath"])).stem
            row["path"] = (
                f"{phase}/synthetic/{row['source']}/{stem}-"
                f"{str(row['imageSha256'])[:16]}{row['extension']}"
            )


def materialize_base_pixels(base_rows: list[dict[str, Any]], source_root: Path, output_root: Path) -> None:
    source_root = validate_data_root(source_root)
    output_root = validate_data_root(output_root)
    for index, row in enumerate(base_rows):
        relative = str(row["path"])
        source = safe_output_path(source_root, relative)
        if not source.is_file() or digest(source) != row["imageSha256"]:
            raise ValueError(f"M3 base pixel changed: {row['id']}")
        destination = safe_output_path(output_root, relative)
        if destination.is_file() and digest(destination) == row["imageSha256"]:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination = safe_output_path(output_root, relative)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
        if digest(destination) != row["imageSha256"]:
            raise ValueError(f"M4 base hardlink/copy changed: {row['id']}")
        if index and index % 10_000 == 0:
            print(f"materialized M4 base pixels: {index}/{len(base_rows)}", flush=True)


def extract_british_pixels(files: list[Path], rows: list[dict[str, Any]], output_root: Path) -> None:
    import pyarrow.parquet as parquet

    targets = {
        (str(row["sourceShard"]), int(row["sourceRow"])): row
        for row in rows
    }
    written: set[tuple[str, int]] = set()
    for path in files:
        source_path = f"plates/{path.name}"
        wanted = {index for shard, index in targets if shard == source_path}
        if not wanted:
            continue
        index = 0
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=16, columns=["image"]):
            for item in batch.to_pylist():
                if index in wanted:
                    row = targets[(source_path, index)]
                    value = embedded_bytes(item.get("image"), label=f"British selected {source_path}:{index}")
                    write_data_atomic(output_root, str(row["path"]), value, str(row["imageSha256"]))
                    written.add((source_path, index))
                index += 1
    if written != set(targets):
        raise ValueError("Not every selected British Library pixel was materialized")


def extract_rapidata_pixels(files: list[Path], rows: list[dict[str, Any]], output_root: Path) -> None:
    import pyarrow.parquet as parquet

    targets = {str(row["imagePath"]): row for row in rows}
    written: set[str] = set()
    for path in files:
        columns = ["image1", "image2", "image1_path", "image2_path"]
        for batch in parquet.ParquetFile(path).iter_batches(batch_size=16, columns=columns):
            for item in batch.to_pylist():
                for side in (1, 2):
                    pathname = str(item.get(f"image{side}_path"))
                    if pathname not in targets or pathname in written:
                        continue
                    row = targets[pathname]
                    value = embedded_bytes(item.get(f"image{side}"), label=f"Rapidata selected {pathname}")
                    write_data_atomic(output_root, str(row["path"]), value, str(row["imageSha256"]))
                    written.add(pathname)
    if written != set(targets):
        raise ValueError("Not every selected Rapidata pixel was materialized")


def verify_materialized_rows(rows: Iterable[dict[str, Any]], output_root: Path) -> None:
    for row in rows:
        path = safe_output_path(output_root, str(row["path"]))
        if not path.is_file() or digest(path) != row["imageSha256"]:
            raise ValueError(f"M4 materialized pixel changed: {row['id']}")


def packet_hashes(packet: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, value in packet.items():
        if name == "selection-summary.json":
            continue
        if name in COMPRESSED_ARTIFACTS:
            compressed = deterministic_gzip(value)
            output[name] = {
                "path": f"benchmark/evidence/m4/{name}.gz",
                "expandedSha256": digest_bytes(value),
                "compressedSha256": digest_bytes(compressed),
                "bytes": len(compressed),
            }
        else:
            output[name] = {
                "path": f"benchmark/evidence/m4/{name}",
                "sha256": digest_bytes(value),
                "bytes": len(value),
            }
    return dict(sorted(output.items()))


def derive_packet(
    recipe: dict[str, Any],
    locks: dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    allow_download: bool,
    materialize_pixels: bool,
) -> dict[str, bytes]:
    source_root = validate_data_root(source_root)
    output_root = validate_data_root(output_root)
    files = ensure_sources(locks, source_root, allow_download=allow_download)
    base_rows, base_expanded = load_base_training(recipe)
    state, exclusion_counts = seed_exclusions(recipe, base_rows)
    british_candidates, british_eligibility = scan_british(files["britishLibrary"], locks, recipe)
    rapidata_images, rapidata_groups = scan_rapidata(files["rapidata"], locks, recipe)
    rejects: list[dict[str, Any]] = []

    british_selector, british_considered = select_british_phase(
        british_candidates,
        phase="validation-real",
        quotas=recipe["britishLibrary"]["selectorDecadeQuotas"],
        namespace=recipe["britishLibrary"]["selectorPriorityNamespace"],
        excluded_books=set(),
        state=state,
        rejects=rejects,
    )
    rapidata_selector, rapidata_considered = select_rapidata_phase(
        rapidata_groups,
        rapidata_images,
        phase="validation-synthetic",
        target_groups=recipe["rapidata"]["selectorPromptGroups"],
        namespace=recipe["rapidata"]["selectorPriorityNamespace"],
        image_namespace=recipe["rapidata"]["imagePriorityNamespace"],
        excluded_groups=set(),
        state=state,
        rejects=rejects,
    )
    british_training, _ = select_british_phase(
        british_candidates,
        phase="train-real",
        quotas=recipe["britishLibrary"]["trainingDecadeQuotas"],
        namespace=recipe["britishLibrary"]["trainingPriorityNamespace"],
        excluded_books=british_considered,
        state=state,
        rejects=rejects,
    )
    rapidata_training, rapidata_training_considered = select_rapidata_phase(
        rapidata_groups,
        rapidata_images,
        phase="train-synthetic",
        target_groups=recipe["rapidata"]["trainingPromptGroups"],
        namespace=recipe["rapidata"]["trainingPriorityNamespace"],
        image_namespace=recipe["rapidata"]["imagePriorityNamespace"],
        excluded_groups=rapidata_considered,
        state=state,
        rejects=rejects,
    )
    rapidata_training_by_family = Counter(str(row["model"]) for row in rapidata_training)
    rapidata_training_image_rejects = sum(
        row["phase"] == "train-synthetic" and row["reason"] == "perceptualDhash64"
        for row in rejects
    )
    rapidata_training_missing_family_groups = sum(
        row["phase"] == "train-synthetic" and row["reason"] == "missingCleanFamily"
        for row in rejects
    )
    rapidata_policy = recipe["rapidata"]["trainingGroupPolicy"]
    if (
        dict(rapidata_training_by_family) != recipe["rapidata"]["trainingImagesByFamily"]
        or len(rapidata_training) != rapidata_policy["expectedSelectedImages"]
        or rapidata_training_image_rejects != rapidata_policy["expectedImageLevelPerceptualRejects"]
        or rapidata_training_missing_family_groups != rapidata_policy["expectedRejectedGroupsMissingFamily"]
    ):
        raise ValueError("Rapidata overlap-clean training allocation changed")
    assign_paths(british_selector, rapidata_selector, british_training, rapidata_training)

    if materialize_pixels:
        output_root.mkdir(parents=True, exist_ok=True)
        materialize_base_pixels(base_rows, ROOT / recipe["baseTraining"]["dataRoot"], output_root)
        extract_british_pixels(
            files["britishLibrary"], [*british_selector, *british_training], output_root,
        )
        extract_rapidata_pixels(
            files["rapidata"], [*rapidata_selector, *rapidata_training], output_root,
        )

    new_training = [*british_training, *rapidata_training]
    training_rows = reindex([*base_rows, *new_training], split="train")
    validation_rows = reindex(
        sorted(
            [*british_selector, *rapidata_selector],
            key=lambda row: (int(row["label"]), str(row["source"]), str(row["id"])),
        ),
        split="validation",
    )
    all_rows = [*training_rows, *validation_rows]
    if len({str(row["id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("M4 selected manifests contain duplicate IDs")
    if len({str(row["imageSha256"]) for row in all_rows}) != len(all_rows):
        raise ValueError("M4 selected manifests contain duplicate image bytes")
    if len({str(row["path"]) for row in all_rows}) != len(all_rows):
        raise ValueError("M4 selected manifests contain duplicate output paths")
    if materialize_pixels:
        verify_materialized_rows(training_rows, output_root)
        verify_materialized_rows(validation_rows, output_root)

    source_counts = Counter(str(row["source"]) for row in training_rows)
    expected_sources = Counter(recipe["baseTraining"]["sourceCounts"])
    expected_sources.update(recipe["expectedTraining"]["newSourceCounts"])
    class_counts = {
        "real": sum(int(row["label"]) == 0 for row in training_rows),
        "synthetic": sum(int(row["label"]) == 1 for row in training_rows),
    }
    if (
        len(training_rows) != recipe["expectedTraining"]["images"]
        or source_counts != expected_sources
        or class_counts != recipe["expectedTraining"]["classCounts"]
        or len(validation_rows) != recipe["freshSelector"]["items"]
        or Counter(str(row["source"]) for row in validation_rows) != Counter(recipe["freshSelector"]["sourceCounts"])
    ):
        raise ValueError("M4 selected source composition changed")

    training_bytes = base_expanded + jsonl_bytes(training_rows[len(base_rows):])
    if load_jsonl_bytes(training_bytes) != training_rows:
        raise AssertionError("M4 training manifest did not preserve the exact M3 prefix")
    packet: dict[str, bytes] = {
        "attribution.json": canonical_json({
            "schemaVersion": 1,
            "britishLibrary": {
                "dataset": recipe["britishLibrary"]["source"],
                "revision": recipe["britishLibrary"]["revision"],
                "sourceReportedLicense": recipe["britishLibrary"]["sourceReportedLicense"],
                "notice": "The plates config is an algorithmic page-layout category. Selected bytes remain under ignored benchmark/data; the public repository distributes pixel-free provenance only. Source labels do not independently clear depicted or third-party rights.",
            },
            "rapidata": {
                "dataset": recipe["rapidata"]["source"],
                "revision": recipe["rapidata"]["revision"],
                "sourceReportedLicense": recipe["rapidata"]["sourceReportedLicense"],
                "sourceReportedProvenance": recipe["rapidata"]["sourceReportedProvenance"],
                "developmentOnly": True,
                "neverAcceptanceEvidence": True,
                "notice": "Publisher-authored family labels do not identify exact generator revisions or seeds and are not independent rights clearance. Selected bytes remain local-only.",
            },
        }, pretty=True),
        "british-source-index.json": canonical_json({
            "schemaVersion": 1,
            "dataset": locks["britishLibrary"]["dataset"],
            "revision": locks["britishLibrary"]["revision"],
            "sourceLocksSha256": recipe["sourceLocksSha256"],
            "sourceEligibility": british_eligibility,
            "items": [source_index_candidate(row) for row in sorted(british_candidates, key=lambda row: row["id"])],
        }),
        "rapidata-source-index.json": canonical_json({
            "schemaVersion": 1,
            "dataset": locks["rapidata"]["dataset"],
            "revision": locks["rapidata"]["revision"],
            "sourceLocksSha256": recipe["sourceLocksSha256"],
            "items": [source_index_candidate(row) for row in sorted(rapidata_images, key=lambda row: row["id"])],
        }),
        "perceptual-review.json": canonical_json({
            "schemaVersion": 1,
            "algorithm": recipe["overlapPolicy"]["perceptualDhash64"]["algorithm"],
            "maximumHammingDistance": recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"],
            "policy": "No cross-pool dHash exceptions; every match at or below the threshold is rejected.",
            "items": [],
        }, pretty=True),
        "rejects.jsonl": jsonl_bytes(rejects),
        "train-manifest.jsonl": training_bytes,
        "validation-manifest.jsonl": jsonl_bytes(validation_rows),
    }
    artifact_hashes = packet_hashes(packet)
    packet["selection-summary.json"] = canonical_json({
        "schemaVersion": 1,
        "recipeSha256": digest(RECIPE_PATH),
        "sourceLocksSha256": digest(LOCKS_PATH),
        "scoreBlind": True,
        "modelOutputsRead": False,
        "h3PixelsRead": False,
        "h3ManifestSha256": recipe["h3Exclusion"]["sha256"],
        "selectionOrder": ["british-selector", "rapidata-selector", "british-training", "rapidata-training"],
        "sourceEligibility": {"britishLibrary": british_eligibility},
        "training": {
            "items": len(training_rows),
            "featureViews": recipe["expectedTraining"]["featureViews"],
            "classCounts": class_counts,
            "sourceCounts": dict(sorted(source_counts.items())),
            "basePrefixItems": len(base_rows),
            "baseExpandedSha256": digest_bytes(base_expanded),
        },
        "freshSelector": {
            "items": len(validation_rows),
            "featureViews": recipe["freshSelector"]["featureViews"],
            "sourceCounts": dict(sorted(Counter(str(row["source"]) for row in validation_rows).items())),
            "classCounts": {
                "real": sum(int(row["label"]) == 0 for row in validation_rows),
                "synthetic": sum(int(row["label"]) == 1 for row in validation_rows),
            },
        },
        "partitionGroups": {
            "britishSelectorBooks": len({row["bookId"] for row in british_selector}),
            "britishTrainingBooks": len({row["bookId"] for row in british_training}),
            "rapidataSelectorPrompts": len({row["promptSha256"] for row in rapidata_selector}),
            "rapidataTrainingPrompts": len({row["promptSha256"] for row in rapidata_training}),
            "rapidataTrainingImages": len(rapidata_training),
            "rapidataTrainingImagesByFamily": dict(sorted(rapidata_training_by_family.items())),
            "rapidataTrainingImageLevelPerceptualRejects": rapidata_training_image_rejects,
            "rapidataTrainingGroupsRejectedMissingFamily": rapidata_training_missing_family_groups,
            "rapidataPreOverlapReserveGroups": recipe["rapidata"]["expectedCompleteGroupAllocation"]["reserve"],
            "rapidataUnassignedCompleteGroups": (
                recipe["rapidata"]["expectedFourPerFamilyGroups"]
                - len(rapidata_considered | rapidata_training_considered)
            ),
        },
        "overlap": {
            "threshold": state.threshold,
            "admittedCrossPoolMatches": 0,
            "exclusionCounts": exclusion_counts,
            "rejectCount": len(rejects),
            "reviewExceptions": 0,
        },
        "publicArtifacts": artifact_hashes,
    }, pretty=True)
    return packet


def publish_packet(packet: dict[str, bytes], evidence_root: Path = PUBLIC_EVIDENCE_ROOT) -> None:
    evidence_root.mkdir(parents=True, exist_ok=True)
    for name, value in packet.items():
        target = evidence_root / (f"{name}.gz" if name in COMPRESSED_ARTIFACTS else name)
        atomic_write(target, deterministic_gzip(value) if name in COMPRESSED_ARTIFACTS else value)


def publish_local_manifests(packet: dict[str, bytes], output_root: Path) -> None:
    output_root = validate_data_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("train-manifest.jsonl", "validation-manifest.jsonl", "selection-summary.json"):
        target = safe_output_path(output_root, name)
        atomic_write(target, packet[name])


def compare_public_packet(packet: dict[str, bytes], evidence_root: Path = PUBLIC_EVIDENCE_ROOT) -> None:
    expected_paths = {
        f"{name}.gz" if name in COMPRESSED_ARTIFACTS else name
        for name in packet
    }
    actual_paths = {path.name for path in evidence_root.iterdir() if path.is_file()} if evidence_root.is_dir() else set()
    if actual_paths != expected_paths:
        raise ValueError("M4 public source-packet file set changed")
    for name, value in packet.items():
        path = evidence_root / (f"{name}.gz" if name in COMPRESSED_ARTIFACTS else name)
        expected = deterministic_gzip(value) if name in COMPRESSED_ARTIFACTS else value
        if path.read_bytes() != expected:
            raise ValueError(f"M4 public source evidence changed: {path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "benchmark/data/m4-source")
    parser.add_argument("--output-root", type=Path, default=ROOT / "benchmark/data/m4-head")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--verify-public", action="store_true")
    parser.add_argument("--no-materialize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe, locks = load_frozen_protocol(RECIPE_PATH, LOCKS_PATH)
    packet = derive_packet(
        recipe,
        locks,
        source_root=args.source_root,
        output_root=args.output_root,
        allow_download=args.allow_download,
        materialize_pixels=not args.no_materialize,
    )
    if args.publish:
        publish_packet(packet)
    if not args.no_materialize:
        publish_local_manifests(packet, args.output_root)
    if args.verify_public:
        compare_public_packet(packet)
        verify_materialized_rows(load_jsonl_bytes(packet["train-manifest.jsonl"]), args.output_root)
        verify_materialized_rows(load_jsonl_bytes(packet["validation-manifest.jsonl"]), args.output_root)
    summary = json.loads(packet["selection-summary.json"])
    print(json.dumps({
        "trainingItems": summary["training"]["items"],
        "selectorItems": summary["freshSelector"]["items"],
        "rejects": summary["overlap"]["rejectCount"],
        "h3PixelsRead": summary["h3PixelsRead"],
        "policy": "pass",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
