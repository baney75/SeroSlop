#!/usr/bin/env python3
"""Materialize and verify the score-blind 100,000-image SeroSlop M5 panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


def _require_runpod_launcher() -> None:
    if __name__ != "__main__" or "--verify-public" in sys.argv:
        return
    expected = Path("/workspace/.seroslop/runtime/node-v24.18.1-linux-x64/bin/node")
    parent = Path(f"/proc/{os.getppid()}/exe")
    try:
        actual = parent.resolve(strict=True)
        digest = sha256()
        with actual.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        launcher = Path(__file__).resolve().parents[2] / "scripts/m5-python-launch.mjs"
        launcher_digest = sha256(launcher.read_bytes()).hexdigest()
        command = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    except OSError as error:
        raise RuntimeError("M5 production Python could not verify its pinned Node parent") from error
    if (
        sys.platform != "linux" or actual != expected or not expected.is_file() or expected.is_symlink()
        or digest.hexdigest() != "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a"
        or launcher_digest != "e8b3b37a79d9a71be1f2e4ff9b584d52da164eed8937a5608872895ab867834a"
        or len(command) < 4 or command[:4] != [str(expected).encode("utf-8"), b"scripts/m5-python-launch.mjs", b"lock-large-synthetic", b"--"]
        or os.environ.get("SEROSLOP_M5_LAUNCH_NODE_VERSION") != "v24.18.1"
        or os.environ.get("SEROSLOP_M5_LAUNCH_NODE_SHA256") != "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a"
    ):
        raise RuntimeError("M5 production Python must start through the pinned RunPod launcher")


_require_runpod_launcher()

from PIL import Image, ImageOps

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.m5.contracts import (
    canonical_json,
    digest_file,
    load_recipe,
    parse_json_bytes,
    read_jsonl,
    validate_regression_state,
    validate_selection_lock,
)
from benchmark.m5.train_gpu import assert_worktree_exact, git_text, validate_authorization_commit


ROOT = REPOSITORY_ROOT
RECIPE_PATH = ROOT / "benchmark/m5/recipe.json"
DATA_ROOT = ROOT / "benchmark/data/m5-large-synthetic"
HISTORICAL_INDEX = ROOT / "benchmark/manifests/historical-perceptual-exclusions-v1.json.gz"
EXCLUSION_MANIFESTS = (
    ROOT / "benchmark/evidence/m4/train-manifest.jsonl.gz",
    ROOT / "benchmark/evidence/m4/validation-manifest.jsonl",
    ROOT / "benchmark/evidence/m3/validation-manifest.jsonl",
    ROOT / "benchmark/evidence/m2/validation-manifest.jsonl",
    ROOT / "benchmark/evidence/m3/h3-met-holdout-manifest.jsonl",
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def run(command: Sequence[str]) -> str:
    if not command or command[0] != "git":
        raise ValueError("M5 large-synthetic materializer run helper only accepts Git commands")
    return git_text(command[1:])


def canonical_gzip(payload: bytes) -> bytes:
    compressed = bytearray(gzip.compress(payload, compresslevel=9, mtime=0))
    compressed[9] = 255
    return bytes(compressed)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != ROOT.parent):
        raise ValueError(f"M5 large-synthetic output traverses a symlink: {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise ValueError(f"M5 large-synthetic partial output exists: {temporary.name}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def dhash64(image: Image.Image) -> str:
    grayscale = ImageOps.exif_transpose(image).convert("RGB").convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return f"{value:016x}"


@lru_cache(maxsize=65_536)
def nearby_blocks(value: int) -> tuple[int, ...]:
    values = {value}
    for first in range(16):
        values.add(value ^ (1 << first))
        for second in range(first + 1, 16):
            values.add(value ^ (1 << first) ^ (1 << second))
    return tuple(sorted(values))


class DhashIndex:
    def __init__(self, threshold: int = 8) -> None:
        self.threshold = threshold
        self.values: set[int] = set()
        self.blocks: list[dict[int, set[int]]] = [defaultdict(set) for _ in range(4)]

    def add(self, value: str) -> None:
        integer = int(value, 16)
        if integer in self.values:
            return
        self.values.add(integer)
        for block in range(4):
            self.blocks[block][(integer >> (block * 16)) & 0xFFFF].add(integer)

    def has_near(self, value: str) -> bool:
        integer = int(value, 16)
        candidates: set[int] = set()
        for block in range(4):
            part = (integer >> (block * 16)) & 0xFFFF
            for neighbor in nearby_blocks(part):
                candidates.update(self.blocks[block].get(neighbor, ()))
        return any((integer ^ other).bit_count() <= self.threshold for other in candidates)


def exclusion_evidence() -> tuple[set[str], set[str], DhashIndex, dict[str, str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    index = DhashIndex()
    bindings: dict[str, str] = {}
    historical = json.load(gzip.open(HISTORICAL_INDEX, "rt", encoding="utf-8"))
    bindings[HISTORICAL_INDEX.relative_to(ROOT).as_posix()] = digest_file(HISTORICAL_INDEX)
    for row in historical["items"]:
        ids.add(str(row["id"]))
        hashes.add(str(row["imageSha256"]))
        index.add(str(row["perceptualDhash64"]))
    historical_ids = set(ids)
    historical_hashes = set(hashes)
    for path in EXCLUSION_MANIFESTS:
        bindings[path.relative_to(ROOT).as_posix()] = digest_file(path)
        for row in read_jsonl(path):
            identifier = str(row["id"])
            image_hash = str(row["imageSha256"])
            ids.add(identifier)
            hashes.add(image_hash)
            perceptual = row.get("perceptualDhash64")
            if perceptual is not None:
                index.add(str(perceptual))
            elif identifier not in historical_ids and image_hash not in historical_hashes:
                raise ValueError(f"M5 exclusion row lacks a bound perceptual hash: {identifier}")
    return ids, hashes, index, bindings


def normalize_generator(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def generator_is_excluded(generator: str, excluded: Sequence[str]) -> bool:
    normalized = normalize_generator(generator)
    return any(normalize_generator(value) in normalized or normalized in normalize_generator(value) for value in excluded)


def source_locks(recipe: Mapping[str, Any], cache_dir: Path, *, allow_download: bool) -> list[dict[str, Any]]:
    from huggingface_hub import HfApi, hf_hub_download

    source = recipe["largeSyntheticEvaluation"]["source"]
    entries = list(HfApi().list_repo_tree(
        source["repository"], repo_type="dataset", revision=source["revision"], recursive=True, expand=True,
    ))
    files = sorted(
        (entry for entry in entries if entry.path.startswith("data/Image/") and entry.path.endswith(".parquet")),
        key=lambda entry: entry.path,
    )
    if len(files) != source["expectedParquetShards"] or sum(int(entry.size) for entry in files) != source["expectedParquetBytes"]:
        raise ValueError("M5 Omni-Fake source shard inventory changed")
    locks: list[dict[str, Any]] = []
    for entry in files:
        if entry.lfs is None or not HEX64.fullmatch(str(entry.lfs.sha256)):
            raise ValueError(f"M5 Omni-Fake shard lacks an LFS SHA-256: {entry.path}")
        if allow_download:
            local = Path(hf_hub_download(
                source["repository"], entry.path, repo_type="dataset", revision=source["revision"], cache_dir=cache_dir,
            ))
            if local.stat().st_size != entry.size or digest_file(local) != entry.lfs.sha256:
                raise ValueError(f"M5 Omni-Fake shard bytes changed: {entry.path}")
            local_path: str | None = str(local)
        else:
            local_path = None
        locks.append({"path": entry.path, "bytes": int(entry.size), "sha256": str(entry.lfs.sha256), "localPath": local_path})
    return locks


def image_facts(payload: bytes) -> tuple[str, str, int, int]:
    image_hash = sha256(payload).hexdigest()
    with Image.open(BytesIO(payload)) as opened:
        opened.load()
        width, height = opened.size
        perceptual = dhash64(opened)
    if width <= 0 or height <= 0:
        raise ValueError("M5 Omni-Fake image dimensions are invalid")
    return image_hash, perceptual, width, height


def _cell(table: Any, column: str, index: int) -> Any:
    return table.column(column)[index].as_py()


def scan_candidates(recipe: Mapping[str, Any], locks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import pyarrow.parquet as parquet

    source = recipe["largeSyntheticEvaluation"]["source"]
    candidates: list[dict[str, Any]] = []
    allowed_splits = set(source["splits"])
    for lock in locks:
        if not any(f"/{split}-" in lock["path"] for split in allowed_splits):
            continue
        local_path = lock.get("localPath")
        if not local_path:
            raise ValueError("M5 Omni-Fake scan requires downloaded source locks")
        table = parquet.read_table(local_path, columns=["image", "label", "generator", "filename", "split"])
        for source_row in range(table.num_rows):
            label = str(_cell(table, "label", source_row))
            generator = str(_cell(table, "generator", source_row))
            split = str(_cell(table, "split", source_row))
            filename = str(_cell(table, "filename", source_row))
            if label != source["eligibleLabel"] or split not in allowed_splits or generator_is_excluded(generator, source["excludedGeneratorFamilies"]):
                continue
            image_value = _cell(table, "image", source_row)
            payload = image_value.get("bytes") if isinstance(image_value, dict) else None
            if not isinstance(payload, bytes) or not payload:
                raise ValueError(f"M5 Omni-Fake selected source image has no bytes: {lock['path']}:{source_row}")
            image_hash, perceptual, width, height = image_facts(payload)
            identifier = f"omni-fake-set:{source['revision']}:{lock['sha256']}:{source_row}"
            priority = sha256(
                f"{source['selectionNamespace']}:{source['revision']}:{generator}:{filename}:{lock['sha256']}:{source_row}".encode()
            ).hexdigest()
            candidates.append({
                "id": identifier,
                "generator": generator,
                "filename": filename,
                "sourceSplit": split,
                "sourceShard": lock["path"],
                "sourceShardSha256": lock["sha256"],
                "sourceRow": source_row,
                "imageSha256": image_hash,
                "perceptualDhash64": perceptual,
                "width": width,
                "height": height,
                "selectionPriority": priority,
            })
    return candidates


def select_panel(
    candidates: Sequence[dict[str, Any]],
    recipe: Mapping[str, Any],
    excluded_ids: set[str],
    excluded_hashes: set[str],
    perceptual: DhashIndex,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    target = int(recipe["largeSyntheticEvaluation"]["minimumItems"])
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[str(row["generator"])].append(row)
    for values in groups.values():
        values.sort(key=lambda row: (row["selectionPriority"], row["id"]))
    order = sorted(groups, key=lambda value: sha256(
        f"{recipe['largeSyntheticEvaluation']['source']['selectionNamespace']}:generator:{value}".encode()
    ).hexdigest())
    positions = {value: 0 for value in order}
    selected: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    while len(selected) < target:
        progressed = False
        for generator in order:
            position = positions[generator]
            values = groups[generator]
            while position < len(values):
                row = values[position]
                position += 1
                positions[generator] = position
                if row["id"] in excluded_ids:
                    rejects["id-overlap"] += 1
                    continue
                if row["imageSha256"] in excluded_hashes:
                    rejects["byte-overlap"] += 1
                    continue
                if perceptual.has_near(row["perceptualDhash64"]):
                    rejects["dhash-overlap"] += 1
                    continue
                excluded_ids.add(row["id"])
                excluded_hashes.add(row["imageSha256"])
                perceptual.add(row["perceptualDhash64"])
                selected.append(dict(row))
                progressed = True
                break
            if len(selected) == target:
                break
        if not progressed:
            raise ValueError(f"M5 Omni-Fake panel has only {len(selected)} overlap-clean synthetic rows")
    for index, row in enumerate(selected):
        row["rowIndex"] = index
        row["batchIndex"] = index % int(recipe["largeSyntheticEvaluation"]["minimumBatches"])
        row["batchPosition"] = index // int(recipe["largeSyntheticEvaluation"]["minimumBatches"])
        row["label"] = 1
        row["source"] = "omni-fake-set"
        row["datasetRevision"] = recipe["largeSyntheticEvaluation"]["source"]["revision"]
        row["sourceReportedLicense"] = recipe["largeSyntheticEvaluation"]["source"]["sourceReportedLicense"]
        row["path"] = f"images/{index:06d}-{row['imageSha256'][:16]}.img"
    return selected, rejects


def extract_selected(rows: Sequence[dict[str, Any]], locks: Sequence[Mapping[str, Any]], data_root: Path) -> None:
    import pyarrow.parquet as parquet

    wanted: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        wanted[row["sourceShard"]][int(row["sourceRow"])] = row
    for lock in locks:
        selected = wanted.get(str(lock["path"]))
        if not selected:
            continue
        table = parquet.read_table(str(lock["localPath"]), columns=["image"])
        for source_row, row in selected.items():
            image_value = _cell(table, "image", source_row)
            payload = image_value.get("bytes") if isinstance(image_value, dict) else None
            if not isinstance(payload, bytes) or sha256(payload).hexdigest() != row["imageSha256"]:
                raise ValueError(f"M5 Omni-Fake selected image changed during extraction: {row['id']}")
            atomic_write(data_root / row["path"], payload)


def manifest_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) for row in rows)


def build_batches(rows: Sequence[Mapping[str, Any]], recipe: Mapping[str, Any]) -> dict[str, Any]:
    batch_count = int(recipe["largeSyntheticEvaluation"]["minimumBatches"])
    result = []
    for batch_index in range(batch_count):
        batch_rows = [row for row in rows if row["batchIndex"] == batch_index]
        if len(batch_rows) != recipe["largeSyntheticEvaluation"]["batchSize"]:
            raise ValueError("M5 large-synthetic batch size changed")
        result.append({
            "batchIndex": batch_index,
            "rowIndices": [row["rowIndex"] for row in batch_rows],
            "generatorCounts": dict(sorted(Counter(row["generator"] for row in batch_rows).items())),
        })
    return {"schemaVersion": 1, "items": len(rows), "batchSize": 100, "batches": result}


def verify_public_packet(recipe: Mapping[str, Any], *, verify_pixels: bool = False, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    large = recipe["largeSyntheticEvaluation"]
    manifest_path = ROOT / large["manifest"]
    batches_path = ROOT / large["batchAssignment"]
    lock_path = ROOT / large["sourceLock"]
    attribution_path = ROOT / large["attribution"]
    lock = parse_json_bytes(lock_path.read_bytes(), label="large-synthetic source lock")
    if canonical_json(lock) != lock_path.read_bytes():
        raise ValueError("M5 large-synthetic source lock is not canonical JSON")
    if set(lock) != {
        "schemaVersion", "status", "lockCommit", "protocolCommit", "recipeSha256", "source", "sourceShards",
        "artifacts", "candidateCounts", "generatorCounts", "rejectionCounts", "exclusionEvidenceSha256ByPath",
        "overlap", "selection", "regressionStateSha256", "scoreBlindness", "selectionInfluence", "h3PixelsRead",
    }:
        raise ValueError("M5 large-synthetic source-lock schema changed")
    raw = gzip.decompress(manifest_path.read_bytes())
    if canonical_gzip(raw) != manifest_path.read_bytes():
        raise ValueError("M5 large-synthetic manifest gzip is not canonical")
    rows = [parse_json_bytes(line + b"\n", label=f"large-synthetic manifest row {index}") for index, line in enumerate(raw.splitlines())]
    if len(rows) != large["minimumItems"] or [row.get("rowIndex") for row in rows] != list(range(len(rows))):
        raise ValueError("M5 large-synthetic manifest coverage changed")
    ids: set[str] = set()
    hashes: set[str] = set()
    generators: Counter[str] = Counter()
    for row in rows:
        required = {
            "id", "generator", "filename", "sourceSplit", "sourceShard", "sourceShardSha256", "sourceRow",
            "imageSha256", "perceptualDhash64", "width", "height", "selectionPriority", "rowIndex", "batchIndex",
            "batchPosition", "label", "source", "datasetRevision", "sourceReportedLicense", "path",
        }
        if set(row) != required or row["label"] != 1 or row["source"] != "omni-fake-set":
            raise ValueError("M5 large-synthetic manifest row schema changed")
        if row["id"] in ids or row["imageSha256"] in hashes or not HEX64.fullmatch(row["imageSha256"]):
            raise ValueError("M5 large-synthetic manifest identity changed")
        if generator_is_excluded(row["generator"], large["source"]["excludedGeneratorFamilies"]):
            raise ValueError("M5 large-synthetic panel contains an excluded generator family")
        ids.add(row["id"])
        hashes.add(row["imageSha256"])
        generators[row["generator"]] += 1
        if verify_pixels:
            pixel = data_root / row["path"]
            if digest_file(pixel) != row["imageSha256"]:
                raise ValueError(f"M5 large-synthetic pixel changed: {row['id']}")
            with Image.open(pixel) as opened:
                opened.load()
                if opened.size != (row["width"], row["height"]) or dhash64(opened) != row["perceptualDhash64"]:
                    raise ValueError(f"M5 large-synthetic pixel facts changed: {row['id']}")
    excluded_ids, excluded_hashes, perceptual, exclusion_bindings = exclusion_evidence()
    for row in rows:
        if row["id"] in excluded_ids or row["imageSha256"] in excluded_hashes or perceptual.has_near(row["perceptualDhash64"]):
            raise ValueError("M5 large-synthetic panel overlaps training, selector, regression, H3 metadata, or history")
        excluded_ids.add(row["id"])
        excluded_hashes.add(row["imageSha256"])
        perceptual.add(row["perceptualDhash64"])
    batches = parse_json_bytes(batches_path.read_bytes(), label="large-synthetic batches")
    if canonical_json(batches) != batches_path.read_bytes():
        raise ValueError("M5 large-synthetic batches are not canonical JSON")
    if batches != build_batches(rows, recipe):
        raise ValueError("M5 large-synthetic batch assignment changed")
    attribution = parse_json_bytes(attribution_path.read_bytes(), label="large-synthetic attribution")
    if canonical_json(attribution) != attribution_path.read_bytes():
        raise ValueError("M5 large-synthetic attribution is not canonical JSON")
    if attribution != {
        "schemaVersion": 1,
        "repository": large["source"]["repository"],
        "revision": large["source"]["revision"],
        "sourceReportedLicense": "CC-BY-4.0",
        "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
        "sourceUrl": f"https://huggingface.co/datasets/{large['source']['repository']}/tree/{large['source']['revision']}",
        "notice": "SeroSlop distributes only pixel-free provenance and evaluation evidence. Source image bytes remain local and are unmodified.",
    }:
        raise ValueError("M5 large-synthetic attribution changed")
    expected_artifacts = {
        "manifest": {"path": large["manifest"], "bytes": manifest_path.stat().st_size, "sha256": digest_file(manifest_path), "expandedSha256": sha256(raw).hexdigest(), "items": len(rows)},
        "batches": {"path": large["batchAssignment"], "sha256": digest_file(batches_path), "batches": len(batches["batches"])},
        "attribution": {"path": large["attribution"], "sha256": digest_file(attribution_path)},
    }
    if (
        lock["schemaVersion"] != 1 or lock["status"] != "m5-large-synthetic-source-locked" or
        lock["recipeSha256"] != digest_file(RECIPE_PATH) or lock["source"] != large["source"] or
        lock["artifacts"] != expected_artifacts or lock["generatorCounts"] != dict(sorted(generators.items())) or
        not isinstance(lock["candidateCounts"].get("decodedEligibleBeforeOverlap"), int) or
        lock["candidateCounts"].get("decodedEligibleBeforeOverlap", 0) < 100_000 or
        lock["candidateCounts"].get("selected") != 100_000 or
        any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in lock["rejectionCounts"].values()) or
        lock["exclusionEvidenceSha256ByPath"] != exclusion_bindings or
        lock["overlap"] != {"id": 0, "byte": 0, "dhashAtOrBelow8": 0, "unreviewed": 0} or
        lock["selection"] != {"items": 100_000, "batches": 1_000, "batchSize": 100, "namespace": large["source"]["selectionNamespace"], "generatorStratified": True} or
        not HEX64.fullmatch(str(lock["regressionStateSha256"])) or
        lock["scoreBlindness"] != large["scoreBlindnessEvidence"] or
        lock["selectionInfluence"] is not False or lock["h3PixelsRead"] is not False
    ):
        raise ValueError("M5 large-synthetic source lock changed")
    shards = lock["sourceShards"]
    if len(shards) != large["source"]["expectedParquetShards"] or sum(row["bytes"] for row in shards) != large["source"]["expectedParquetBytes"]:
        raise ValueError("M5 large-synthetic shard lock changed")
    if any(set(row) != {"path", "bytes", "sha256"} or not HEX64.fullmatch(row["sha256"]) for row in shards):
        raise ValueError("M5 large-synthetic shard receipt changed")
    if [row["path"] for row in shards] != sorted({row["path"] for row in shards}) or any(not row["path"].startswith("data/Image/") for row in shards):
        raise ValueError("M5 large-synthetic shard ordering changed")
    return {"items": len(rows), "batches": len(batches["batches"]), "generators": len(generators), "sourceLockSha256": digest_file(lock_path)}


def prepare(args: argparse.Namespace) -> int:
    recipe = load_recipe(RECIPE_PATH)
    head = run(["git", "rev-parse", "HEAD"])
    if head != args.lock_commit:
        raise ValueError("M5 large-synthetic materialization requires the exact clean public selection-lock commit")
    assert_worktree_exact()
    lock_path = ROOT / recipe["output"]["selectionLock"]
    selection_lock = parse_json_bytes(lock_path.read_bytes(), label="selection lock")
    selector_rows = read_jsonl(ROOT / recipe["sourceEvidence"]["selectorManifest"]["path"])
    validate_selection_lock(selection_lock, recipe, selector_rows)
    authorization_commit = run(["git", "rev-parse", f"{args.lock_commit}^"])
    protocol_commit, _source_tree, _authorization_sha256 = validate_authorization_commit(authorization_commit)
    if selection_lock["protocolCommit"] != protocol_commit:
        raise ValueError("M5 large-synthetic selection lock ancestry changed")
    regression_path = ROOT / recipe["output"]["candidateRoot"] / "regression-state.json"
    regression = parse_json_bytes(regression_path.read_bytes(), label="terminal regression state")
    validate_regression_state(
        regression,
        recipe,
        selection_lock,
        lock_commit=args.lock_commit,
        selection_lock_sha256=digest_file(lock_path),
    )
    regression_sha256 = digest_file(regression_path)
    locks = source_locks(recipe, DATA_ROOT / "hf-cache", allow_download=args.allow_download)
    candidates = scan_candidates(recipe, locks)
    excluded_ids, excluded_hashes, perceptual, exclusion_bindings = exclusion_evidence()
    rows, rejects = select_panel(candidates, recipe, excluded_ids, excluded_hashes, perceptual)
    extract_selected(rows, locks, DATA_ROOT)
    large = recipe["largeSyntheticEvaluation"]
    raw = manifest_bytes(rows)
    atomic_write(ROOT / large["manifest"], canonical_gzip(raw))
    batches = build_batches(rows, recipe)
    atomic_write(ROOT / large["batchAssignment"], canonical_json(batches))
    attribution = {
        "schemaVersion": 1,
        "repository": large["source"]["repository"],
        "revision": large["source"]["revision"],
        "sourceReportedLicense": "CC-BY-4.0",
        "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
        "sourceUrl": f"https://huggingface.co/datasets/{large['source']['repository']}/tree/{large['source']['revision']}",
        "notice": "SeroSlop distributes only pixel-free provenance and evaluation evidence. Source image bytes remain local and are unmodified.",
    }
    atomic_write(ROOT / large["attribution"], canonical_json(attribution))
    manifest_path = ROOT / large["manifest"]
    batches_path = ROOT / large["batchAssignment"]
    attribution_path = ROOT / large["attribution"]
    lock = {
        "schemaVersion": 1,
        "status": "m5-large-synthetic-source-locked",
        "lockCommit": args.lock_commit,
        "protocolCommit": protocol_commit,
        "recipeSha256": digest_file(RECIPE_PATH),
        "source": large["source"],
        "sourceShards": [{key: row[key] for key in ("path", "bytes", "sha256")} for row in locks],
        "artifacts": {
            "manifest": {"path": large["manifest"], "bytes": manifest_path.stat().st_size, "sha256": digest_file(manifest_path), "expandedSha256": sha256(raw).hexdigest(), "items": len(rows)},
            "batches": {"path": large["batchAssignment"], "sha256": digest_file(batches_path), "batches": len(batches["batches"])},
            "attribution": {"path": large["attribution"], "sha256": digest_file(attribution_path)},
        },
        "candidateCounts": {"decodedEligibleBeforeOverlap": len(candidates), "selected": len(rows)},
        "generatorCounts": dict(sorted(Counter(row["generator"] for row in rows).items())),
        "rejectionCounts": dict(sorted(rejects.items())),
        "exclusionEvidenceSha256ByPath": exclusion_bindings,
        "overlap": {"id": 0, "byte": 0, "dhashAtOrBelow8": 0, "unreviewed": 0},
        "selection": {"items": 100_000, "batches": 1_000, "batchSize": 100, "namespace": large["source"]["selectionNamespace"], "generatorStratified": True},
        "regressionStateSha256": regression_sha256,
        "scoreBlindness": large["scoreBlindnessEvidence"],
        "selectionInfluence": False,
        "h3PixelsRead": False,
    }
    atomic_write(ROOT / large["sourceLock"], canonical_json(lock))
    result = verify_public_packet(recipe, verify_pixels=True)
    print(json.dumps({"event": "m5-large-synthetic-source-locked", **result, "h3PixelsRead": False}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--lock-commit")
    value.add_argument("--allow-download", action="store_true")
    value.add_argument("--verify-public", action="store_true")
    value.add_argument("--verify-pixels", action="store_true")
    return value


def main(arguments: argparse.Namespace) -> int:
    if arguments.verify_public:
        result = verify_public_packet(load_recipe(RECIPE_PATH), verify_pixels=arguments.verify_pixels)
        print(json.dumps({"event": "m5-large-synthetic-verified", **result}, sort_keys=True))
        return 0
    if not arguments.lock_commit or not arguments.allow_download:
        raise ValueError("M5 large-synthetic materialization requires --lock-commit and --allow-download")
    return prepare(arguments)


if __name__ == "__main__":
    try:
        raise SystemExit(main(parser().parse_args()))
    except Exception as error:
        print(f"M5 large-synthetic pipeline failed: {error}", file=sys.stderr)
        raise
