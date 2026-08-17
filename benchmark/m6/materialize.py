"""Resumable metadata materializer for the pinned Omni-Fake image shards.

This stage verifies every Parquet shard before reading it, decodes only the
public fresh-source images, and emits canonical per-shard metadata fragments.
It never reads H3 or any M2-M5 pixel path and never scores a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, BinaryIO, Callable

from benchmark.m6.contracts import ROOT, canonical_json, parse_json_bytes
from benchmark.m6.prepare import (
    OOD_DATASET,
    OOD_REVISION,
    SET_DATASET,
    SET_REVISION,
    canonical_fresh_row,
    canonical_gzip,
)


SOURCE_SHARDS_PATH = ROOT / "benchmark/m6/source-shards.json"
SOURCE_SHARDS_SHA256 = "a86c7209e76248edddd61537f397379194a7aaa908405e0cede7c8f5a3d7fbfe"
DECODED_RGB_NAMESPACE = b"seroslop-m6-decoded-rgb-v1\0"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_SOURCES = {
    "omniFakeSet": {
        "dataset": SET_DATASET,
        "revision": SET_REVISION,
        "partitions": {"train": (19, 10447263801), "validation": (47, 37645535846)},
    },
    "omniFakeOOD": {
        "dataset": OOD_DATASET,
        "revision": OOD_REVISION,
        "partitions": {"test": (19, 16521804450)},
    },
}
SET_REAL_PREFIXES = {"COCO", "FFHQ", "OpenForensics_real", "OpenImages", "WIDER", "celebA", "flickr30k-images"}
SHARD_RECEIPT_FIELDS = {
    "decodedRgbDefinition", "dhashDefinition", "expandedSha256", "fragmentBytes",
    "fragmentSha256", "h3PixelsRead", "labels", "pixelsRead", "rows",
    "schemaVersion", "sourceShard", "status",
}
DOWNLOAD_RECEIPT_FIELDS = {
    "bytes", "dataset", "h3PixelsRead", "localRelativePath", "pixelsRead",
    "revision", "schemaVersion", "sha256", "sourceKey", "sourceShard",
    "status", "tokenUsed",
}
DOWNLOAD_SUMMARY_FIELDS = {
    "artifacts", "h3PixelsRead", "pixelsRead", "schemaVersion",
    "sourceShardsSha256", "status", "totalBytes", "totalShards",
}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree_durable(path: Path, parent: Path, *, label: str) -> None:
    try:
        if path.exists() or path.is_symlink():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        if path.exists() or path.is_symlink():
            raise OSError(f"{label} still exists")
        _fsync_directory(parent)
    except Exception as exc:
        raise RuntimeError(f"{label} publication state unknown after rollback failure") from exc


def validate_shard_receipt(receipt: dict[str, Any], shard: dict[str, Any]) -> None:
    if not isinstance(receipt, dict) or set(receipt) != SHARD_RECEIPT_FIELDS:
        raise ValueError("M6 shard receipt schema changed")
    if (
        type(receipt["schemaVersion"]) is not int
        or receipt["schemaVersion"] != 1
        or receipt["status"] != "m6-shard-materialized"
        or receipt["sourceShard"] != shard
        or receipt["pixelsRead"] is not True
        or receipt["h3PixelsRead"] is not False
        or receipt["decodedRgbDefinition"] != "sha256(namespace || width-u32be || height-u32be || EXIF-transposed RGB bytes)"
        or receipt["dhashDefinition"] != "EXIF-transposed RGB; LANCZOS 9x8; grayscale; left>right row-major MSB-first"
        or type(receipt["rows"]) is not int
        or receipt["rows"] <= 0
        or type(receipt["fragmentBytes"]) is not int
        or receipt["fragmentBytes"] <= 0
        or not isinstance(receipt["labels"], dict)
        or any(label not in {"real", "full_synthetic", "tampered"} or type(count) is not int or count < 0 for label, count in receipt["labels"].items())
        or sum(receipt["labels"].values()) != receipt["rows"]
        or not isinstance(receipt["fragmentSha256"], str)
        or not HEX64.fullmatch(receipt["fragmentSha256"])
        or not isinstance(receipt["expandedSha256"], str)
        or not HEX64.fullmatch(receipt["expandedSha256"])
    ):
        raise ValueError("M6 shard receipt boundary changed")


def load_source_shards(path: Path = SOURCE_SHARDS_PATH) -> dict[str, Any]:
    raw = path.read_bytes()
    if path == SOURCE_SHARDS_PATH and sha256(raw).hexdigest() != SOURCE_SHARDS_SHA256:
        raise ValueError("M6 source-shard inventory bytes changed")
    value = parse_json_bytes(raw, label="source-shard inventory")
    validate_source_shards(value)
    return value


def validate_source_shards(value: dict[str, Any]) -> None:
    if set(value) != {"h3PixelsRead", "pixelsRead", "schemaVersion", "sources", "status", "totalBytes", "totalShards"}:
        raise ValueError("M6 source-shard inventory schema changed")
    if (
        type(value["schemaVersion"]) is not int
        or value["schemaVersion"] != 1
        or value["status"] != "m6-source-shards-pinned"
        or value["pixelsRead"] is not False
        or value["h3PixelsRead"] is not False
        or set(value["sources"]) != set(EXPECTED_SOURCES)
    ):
        raise ValueError("M6 source-shard inventory boundary changed")
    total_shards = 0
    total_bytes = 0
    seen: set[tuple[str, str]] = set()
    for key, expected in EXPECTED_SOURCES.items():
        source = value["sources"][key]
        if set(source) != {"dataset", "revision", "shards"}:
            raise ValueError("M6 source-shard source schema changed")
        if source["dataset"] != expected["dataset"] or source["revision"] != expected["revision"]:
            raise ValueError("M6 source-shard source pin changed")
        if not isinstance(source["shards"], list):
            raise ValueError("M6 source-shard list missing")
        counts = Counter()
        sizes = Counter()
        prior_path = ""
        for shard in source["shards"]:
            if not isinstance(shard, dict) or set(shard) != {"lfsSha256", "partition", "path", "size"}:
                raise ValueError("M6 source-shard row schema changed")
            relative = PurePosixPath(shard["path"])
            if (
                relative.is_absolute()
                or any(piece in ("", ".", "..") for piece in relative.parts)
                or not shard["path"].startswith("data/Image/")
                or not shard["path"].endswith(".parquet")
                or shard["path"] <= prior_path
            ):
                raise ValueError("M6 source-shard path/order changed")
            prior_path = shard["path"]
            if shard["partition"] not in expected["partitions"] or not relative.name.startswith(shard["partition"] + "-"):
                raise ValueError("M6 source-shard partition changed")
            if type(shard["size"]) is not int or shard["size"] <= 0:
                raise ValueError("M6 source-shard size invalid")
            if not isinstance(shard["lfsSha256"], str) or not HEX64.fullmatch(shard["lfsSha256"]):
                raise ValueError("M6 source-shard LFS SHA-256 invalid")
            identity = (key, shard["path"])
            if identity in seen:
                raise ValueError("duplicate M6 source shard")
            seen.add(identity)
            counts[shard["partition"]] += 1
            sizes[shard["partition"]] += shard["size"]
            total_shards += 1
            total_bytes += shard["size"]
        for partition, (count, size) in expected["partitions"].items():
            if counts[partition] != count or sizes[partition] != size:
                raise ValueError("M6 source-shard partition census changed")
    if type(value["totalShards"]) is not int or value["totalShards"] != total_shards or total_shards != 85:
        raise ValueError("M6 source-shard total count changed")
    if type(value["totalBytes"]) is not int or value["totalBytes"] != total_bytes or total_bytes != 64614604097:
        raise ValueError("M6 source-shard total bytes changed")


def image_facts(value: bytes) -> tuple[str, str, str, int, int]:
    from PIL import Image, ImageOps

    if not isinstance(value, bytes) or not value:
        raise ValueError("M6 source image bytes are missing")
    with Image.open(BytesIO(value)) as opened:
        opened.verify()
    with Image.open(BytesIO(value)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("M6 decoded image dimensions invalid")
        pixels = image.tobytes()
        resized = image.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
        gray = list(resized.getdata())
    decoded_preimage = (
        DECODED_RGB_NAMESPACE
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + pixels
    )
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(gray[y * 9 + x] > gray[y * 9 + x + 1])
    return (
        sha256(value).hexdigest(),
        sha256(decoded_preimage).hexdigest(),
        f"{bits:016x}",
        width,
        height,
    )


def source_group(dataset: str, partition: str, label: str, generator: Any, filename: str) -> str:
    if label == "real":
        pieces = filename.split("/")
        if dataset == SET_DATASET and len(pieces) >= 4 and pieces[:2] == [partition, "real"]:
            prefix = pieces[2]
            if prefix not in SET_REAL_PREFIXES: raise ValueError("M6 SET real source prefix changed")
            if isinstance(generator, str) and generator and generator != prefix:
                raise ValueError("M6 real source prefix disagrees with filename")
            return prefix
        if dataset == OOD_DATASET and pieces and pieces[0] == "real":
            if isinstance(generator, str) and generator not in {"", "OOD-real"}:
                raise ValueError("M6 OOD real source prefix changed")
            return "OOD-real"
        raise ValueError("M6 real source prefix unavailable")
    if isinstance(generator, str) and generator:
        return generator
    pieces = filename.split("/")
    raise ValueError("M6 source group is unavailable")


def _decode_record(task: tuple[str, str, str, str, int, dict[str, Any]]) -> dict[str, str]:
    dataset, revision, partition, shard_path, ordinal, raw = task
    if not isinstance(raw, dict) or set(raw) != {"image", "label", "generator", "filename", "split"}:
        raise ValueError("M6 Parquet row schema changed")
    if raw["split"] != partition or raw["label"] not in {"real", "full_synthetic", "tampered"}:
        raise ValueError("M6 Parquet split/label changed")
    filename = raw["filename"]
    if not isinstance(filename, str) or not filename:
        raise ValueError("M6 Parquet filename missing")
    relative = PurePosixPath(filename)
    if relative.is_absolute() or any(piece in ("", ".", "..") for piece in relative.parts):
        raise ValueError("M6 Parquet filename unsafe")
    image = raw["image"]
    if not isinstance(image, dict) or set(image) != {"bytes", "path"} or not isinstance(image["bytes"], bytes):
        raise ValueError("M6 Parquet image payload changed")
    encoded, decoded, dhash, _, _ = image_facts(image["bytes"])
    row = {
        "dataset": dataset,
        "decodedRgbSha256": decoded,
        "dhash64": dhash,
        "encodedBytesSha256": encoded,
        "filename": filename,
        "label": raw["label"],
        "partition": partition,
        "revision": revision,
        "rowId": f"{dataset}:{revision}:{shard_path}:{ordinal}",
        "sourceGroup": source_group(dataset, partition, raw["label"], raw["generator"], filename),
    }
    key = "set_train" if dataset == SET_DATASET and partition == "train" else (
        "set_validation" if dataset == SET_DATASET else "ood_test"
    )
    return canonical_fresh_row(row, key)


def _digest_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_shard_path(cache_root: Path, source_key: str, shard: dict[str, Any]) -> Path:
    local_key = "omni-fake-set" if source_key == "omniFakeSet" else "omni-fake-ood"
    return cache_root / local_key / PurePosixPath(shard["path"])


def _open_verified_file(path: Path, *, expected_size: int, expected_sha256: str, label: str) -> BinaryIO:
    """Open, hash, and rewind one regular file without reopening its pathname."""
    if path.resolve(strict=True) != path:
        raise ValueError(f"{label} path is not physical")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != expected_size:
            raise ValueError(f"{label} type/size changed")
        handle = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = -1
        digest = sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            handle.close()
            raise ValueError(f"{label} SHA-256 changed")
        handle.seek(0)
        return handle
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _open_local_shard(cache_root: Path, source_key: str, shard: dict[str, Any]) -> BinaryIO:
    return _open_verified_file(
        _local_shard_path(cache_root, source_key, shard),
        expected_size=shard["size"],
        expected_sha256=shard["lfsSha256"],
        label=f"M6 local shard {shard['path']}",
    )


def _validate_inventory_member(
    source_key: str, source: dict[str, Any], shard: dict[str, Any], inventory: dict[str, Any],
) -> None:
    expected = EXPECTED_SOURCES.get(source_key)
    if expected is None or not isinstance(source, dict) or not {"dataset", "revision"} <= set(source):
        raise ValueError("M6 download source changed")
    if source["dataset"] != expected["dataset"] or source["revision"] != expected["revision"]:
        raise ValueError("M6 download source pin changed")
    if not isinstance(shard, dict) or set(shard) != {"lfsSha256", "partition", "path", "size"}:
        raise ValueError("M6 download shard schema changed")
    relative = PurePosixPath(shard["path"])
    if (
        relative.is_absolute()
        or any(piece in ("", ".", "..") for piece in relative.parts)
        or not shard["path"].startswith("data/Image/")
        or not shard["path"].endswith(".parquet")
        or shard["partition"] not in expected["partitions"]
        or not relative.name.startswith(shard["partition"] + "-")
        or type(shard["size"]) is not int
        or shard["size"] <= 0
        or not isinstance(shard["lfsSha256"], str)
        or not HEX64.fullmatch(shard["lfsSha256"])
    ):
        raise ValueError("M6 download shard pin changed")
    pinned_source = inventory.get("sources", {}).get(source_key)
    if (
        not isinstance(pinned_source, dict)
        or pinned_source.get("dataset") != source["dataset"]
        or pinned_source.get("revision") != source["revision"]
        or shard not in pinned_source.get("shards", [])
    ):
        raise ValueError("M6 shard is not a member of the pinned inventory")


def _validate_download_input(source_key: str, source: dict[str, Any], shard: dict[str, Any]) -> None:
    _validate_inventory_member(source_key, source, shard, load_source_shards())


def _download_receipt(source_key: str, source: dict[str, Any], shard: dict[str, Any]) -> dict[str, Any]:
    local_key = "omni-fake-set" if source_key == "omniFakeSet" else "omni-fake-ood"
    return {
        "bytes": shard["size"], "dataset": source["dataset"], "h3PixelsRead": False,
        "localRelativePath": f"{local_key}/{shard['path']}", "pixelsRead": False,
        "revision": source["revision"], "schemaVersion": 1,
        "sha256": shard["lfsSha256"], "sourceKey": source_key,
        "sourceShard": shard, "status": "m6-source-shard-downloaded",
        "tokenUsed": False,
    }


def _validate_download_receipt(
    receipt: dict[str, Any], source_key: str, source: dict[str, Any], shard: dict[str, Any]
) -> None:
    if not isinstance(receipt, dict) or set(receipt) != DOWNLOAD_RECEIPT_FIELDS:
        raise ValueError("M6 download receipt schema changed")
    if receipt != _download_receipt(source_key, source, shard):
        raise ValueError("M6 download receipt boundary changed")


def _download_receipt_dir(cache_root: Path, source_key: str, shard: dict[str, Any]) -> Path:
    return cache_root / "download-receipts" / source_key / PurePosixPath(shard["path"]).name


def _write_atomic_receipt_directory(final: Path, payload: bytes) -> None:
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve(strict=True) != parent or final.exists() or final.is_symlink():
        raise FileExistsError("M6 receipt destination is unsafe or already exists")
    temporary = final.with_name(final.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("stale M6 receipt partial exists")
    temporary.mkdir(mode=0o700)
    installed = False
    try:
        with (temporary / "receipt.json").open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        os.replace(temporary, final)
        installed = True
        _fsync_directory(parent)
    except Exception:
        if installed:
            _remove_tree_durable(final, parent, label="M6 receipt")
        else:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_exact_canonical_object(
    raw: bytes, expected: dict[str, Any], *, fields: set[str], label: str,
) -> None:
    parsed = parse_json_bytes(raw, label=label)
    if set(parsed) != fields or raw != canonical_json(parsed) or parsed != expected:
        raise ValueError(f"M6 {label} boundary changed")


def _publish_summary(
    path: Path, expected: dict[str, Any], *, fields: set[str], label: str,
    failure_hook: Callable[[str], None] | None = None,
) -> None:
    """Idempotently publish one canonical stage marker with durable rollback."""
    parent = path.parent
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists() or partial.is_symlink():
        if (
            partial.is_symlink() or not partial.is_file()
            or partial.resolve(strict=True) != partial
        ):
            raise ValueError(f"M6 {label} partial is unsafe")
        partial.unlink()
        _fsync_directory(parent)
    payload = canonical_json(expected)
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink() or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise ValueError(f"M6 {label} path is unsafe")
        _validate_exact_canonical_object(
            path.read_bytes(), expected, fields=fields, label=label,
        )
        return
    installed = False
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if failure_hook:
            failure_hook("summary-written")
            failure_hook("before-summary-rename")
        os.replace(partial, path)
        installed = True
        if failure_hook:
            failure_hook("after-summary-rename")
        _fsync_directory(parent)
        if failure_hook:
            failure_hook("summary-parent-fsynced")
    except Exception:
        try:
            if installed:
                path.unlink()
            elif partial.exists() or partial.is_symlink():
                partial.unlink()
            if failure_hook:
                failure_hook("summary-rollback-fsync")
            _fsync_directory(parent)
        except Exception as cleanup_error:
            raise RuntimeError(
                f"M6 {label} publication state unknown after rollback failure"
            ) from cleanup_error
        raise


def _require_download_receipt(
    cache_root: Path, source_key: str, source: dict[str, Any], shard: dict[str, Any]
) -> dict[str, Any]:
    receipt_dir = _download_receipt_dir(cache_root, source_key, shard)
    if (
        not receipt_dir.is_dir() or receipt_dir.is_symlink()
        or receipt_dir.resolve(strict=True) != receipt_dir
        or set(os.listdir(receipt_dir)) != {"receipt.json"}
    ):
        raise ValueError("M6 shard download receipt is missing or unsafe")
    raw = (receipt_dir / "receipt.json").read_bytes()
    receipt = parse_json_bytes(raw, label="M6 shard download receipt")
    _validate_download_receipt(receipt, source_key, source, shard)
    if raw != canonical_json(receipt):
        raise ValueError("M6 shard download receipt is not canonical")
    with _open_local_shard(cache_root, source_key, shard):
        pass
    return receipt


def _download_shard_from_inventory(
    *, cache_root: Path, source_key: str, source: dict[str, Any], shard: dict[str, Any],
    inventory: dict[str, Any], downloader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    _validate_inventory_member(source_key, source, shard, inventory)
    destination = _local_shard_path(cache_root, source_key, shard)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.resolve(strict=True) != destination.parent:
        raise ValueError("M6 shard download parent is not physical")
    if destination.is_symlink():
        raise ValueError("M6 shard cache entry is a symlink")
    if not destination.exists():
        fetch = downloader
        if fetch is None:
            from huggingface_hub import hf_hub_download
            fetch = hf_hub_download
        scratch = cache_root / ".hf-downloads" / source_key
        scratch.mkdir(parents=True, exist_ok=True)
        cached = fetch(
            repo_id=source["dataset"], filename=shard["path"],
            repo_type="dataset", revision=source["revision"], token=False,
            local_dir=str(scratch),
        )
        source_path = Path(cached)
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError("stale M6 shard download partial exists")
        installed = False
        try:
            with _open_verified_file(
                source_path, expected_size=shard["size"],
                expected_sha256=shard["lfsSha256"], label="M6 downloaded shard",
            ) as source_handle, temporary.open("xb") as target:
                for block in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, destination)
            installed = True
            _fsync_directory(destination.parent)
        except Exception:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            if installed:
                try:
                    destination.unlink()
                    _fsync_directory(destination.parent)
                except Exception as exc:
                    raise RuntimeError("M6 shard download publication state unknown") from exc
            raise
    with _open_local_shard(cache_root, source_key, shard):
        pass
    receipt = _download_receipt(source_key, source, shard)
    receipt_dir = _download_receipt_dir(cache_root, source_key, shard)
    if receipt_dir.exists() or receipt_dir.is_symlink():
        _require_download_receipt(cache_root, source_key, source, shard)
    else:
        _write_atomic_receipt_directory(receipt_dir, canonical_json(receipt))
    return receipt


def download_shard(
    *, cache_root: Path, source_key: str, source: dict[str, Any], shard: dict[str, Any],
    downloader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Fetch one exact member of the repository-pinned 85-shard inventory."""
    return _download_shard_from_inventory(
        cache_root=cache_root, source_key=source_key, source=source, shard=shard,
        inventory=load_source_shards(), downloader=downloader,
    )


def _download_shard_fixture(
    *, cache_root: Path, source_key: str, source: dict[str, Any], shard: dict[str, Any],
    downloader: Callable[..., str],
) -> dict[str, Any]:
    """Private small-fixture seam; production entry points cannot inject inventory."""
    inventory = {"sources": {source_key: {**source, "shards": [shard]}}}
    return _download_shard_from_inventory(
        cache_root=cache_root, source_key=source_key, source=source, shard=shard,
        inventory=inventory, downloader=downloader,
    )


def download_all_shards(
    cache_root: Path, *, downloader: Callable[..., str] | None = None,
    summary_failure_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Acquire exactly the 85 pinned shards and publish canonical receipts."""
    inventory = load_source_shards()
    if cache_root.exists():
        cache_root = cache_root.resolve(strict=True)
        if cache_root.is_symlink() or not cache_root.is_dir():
            raise ValueError("M6 shard cache root is unsafe")
    else:
        parent = cache_root.parent.resolve(strict=True)
        cache_root = parent / cache_root.name
        cache_root.mkdir(mode=0o700)
        _fsync_directory(parent)
    summary_path = cache_root / "download-summary.json"
    artifacts: list[dict[str, Any]] = []
    for source_key in sorted(inventory["sources"]):
        source = inventory["sources"][source_key]
        for shard in source["shards"]:
            receipt = _download_shard_from_inventory(
                cache_root=cache_root, source_key=source_key, source=source,
                shard=shard, inventory=inventory, downloader=downloader,
            )
            artifacts.append({
                "path": shard["path"],
                "receiptSha256": sha256(canonical_json(receipt)).hexdigest(),
                "sourceKey": source_key,
            })
    summary = {
        "artifacts": artifacts, "h3PixelsRead": False, "pixelsRead": False,
        "schemaVersion": 1, "sourceShardsSha256": SOURCE_SHARDS_SHA256,
        "status": "m6-source-shards-downloaded", "totalBytes": inventory["totalBytes"],
        "totalShards": inventory["totalShards"],
    }
    _publish_summary(
        summary_path, summary, fields=DOWNLOAD_SUMMARY_FIELDS,
        label="download summary", failure_hook=summary_failure_hook,
    )
    return summary


def _require_download_summary(cache_root: Path) -> dict[str, Any]:
    inventory = load_source_shards()
    path = cache_root / "download-summary.json"
    if not path.is_file() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("M6 download summary is missing or unsafe")
    raw = path.read_bytes()
    summary = parse_json_bytes(raw, label="M6 download summary")
    expected_artifacts = []
    for source_key in sorted(inventory["sources"]):
        source = inventory["sources"][source_key]
        for shard in source["shards"]:
            receipt = _require_download_receipt(cache_root, source_key, source, shard)
            expected_artifacts.append({
                "path": shard["path"],
                "receiptSha256": sha256(canonical_json(receipt)).hexdigest(),
                "sourceKey": source_key,
            })
    if (
        not isinstance(summary, dict) or set(summary) != DOWNLOAD_SUMMARY_FIELDS
        or raw != canonical_json(summary)
        or summary != {
            "artifacts": expected_artifacts,
            "h3PixelsRead": False,
            "pixelsRead": False,
            "schemaVersion": 1,
            "sourceShardsSha256": SOURCE_SHARDS_SHA256,
            "status": "m6-source-shards-downloaded",
            "totalBytes": inventory["totalBytes"],
            "totalShards": inventory["totalShards"],
        }
    ):
        raise ValueError("M6 download summary boundary changed")
    return summary


def _derive_shard_materialization(
    *, cache_root: Path, source_key: str, source: dict[str, Any],
    shard: dict[str, Any], workers: int,
) -> tuple[bytes, bytes, dict[str, Any]]:
    from pyarrow import parquet as pq

    if type(workers) is not int or workers < 1 or workers > 32:
        raise ValueError("M6 materializer worker count invalid")
    rows: list[dict[str, str]] = []
    with _open_local_shard(cache_root, source_key, shard) as verified_handle:
        parquet = pq.ParquetFile(verified_handle)
        columns = ["image", "label", "generator", "filename", "split"]
        offset = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for batch in parquet.iter_batches(batch_size=128, columns=columns):
                tasks = [
                    (
                        source["dataset"], source["revision"], shard["partition"],
                        shard["path"], offset + index, row,
                    )
                    for index, row in enumerate(batch.to_pylist())
                ]
                rows.extend(executor.map(_decode_record, tasks))
                offset += len(tasks)
    expanded = b"".join(canonical_json(row) for row in rows)
    compressed = canonical_gzip(expanded)
    receipt = {
        "decodedRgbDefinition": "sha256(namespace || width-u32be || height-u32be || EXIF-transposed RGB bytes)",
        "dhashDefinition": "EXIF-transposed RGB; LANCZOS 9x8; grayscale; left>right row-major MSB-first",
        "expandedSha256": sha256(expanded).hexdigest(),
        "fragmentBytes": len(compressed),
        "fragmentSha256": sha256(compressed).hexdigest(),
        "h3PixelsRead": False,
        "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
        "pixelsRead": True,
        "rows": len(rows),
        "schemaVersion": 1,
        "sourceShard": shard,
        "status": "m6-shard-materialized",
    }
    validate_shard_receipt(receipt, shard)
    return expanded, compressed, receipt


def _validate_materialized_directory(
    final: Path, *, expanded: bytes, compressed: bytes, receipt: dict[str, Any],
) -> None:
    if (
        not final.is_dir() or final.is_symlink() or final.resolve(strict=True) != final
        or set(os.listdir(final)) != {"fragment.jsonl.gz", "receipt.json"}
    ):
        raise ValueError("M6 materializer shard directory is incomplete or unsafe")
    fragment = final / "fragment.jsonl.gz"
    receipt_path = final / "receipt.json"
    if fragment.is_symlink() or receipt_path.is_symlink():
        raise ValueError("M6 materializer shard files are unsafe")
    fragment_raw = fragment.read_bytes()
    receipt_raw = receipt_path.read_bytes()
    parsed = parse_json_bytes(receipt_raw, label="materializer shard receipt")
    validate_shard_receipt(parsed, receipt["sourceShard"])
    try:
        actual_expanded = gzip.decompress(fragment_raw)
    except (OSError, EOFError) as exc:
        raise ValueError("M6 materializer fragment gzip invalid") from exc
    if (
        receipt_raw != canonical_json(parsed)
        or parsed != receipt
        or fragment_raw != compressed
        or actual_expanded != expanded
        or canonical_gzip(actual_expanded) != fragment_raw
        or sha256(fragment_raw).hexdigest() != parsed["fragmentSha256"]
        or sha256(actual_expanded).hexdigest() != parsed["expandedSha256"]
        or len(fragment_raw) != parsed["fragmentBytes"]
    ):
        raise ValueError("M6 materializer resumed shard is not source-derived")


def _materialize_shard_from_inventory(
    *, cache_root: Path, output_root: Path, source_key: str,
    source: dict[str, Any], shard: dict[str, Any], workers: int,
    inventory: dict[str, Any], failure_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Derive and atomically publish one source-bound shard directory."""
    _validate_inventory_member(source_key, source, shard, inventory)
    _require_download_receipt(cache_root, source_key, source, shard)
    fragment_parent = output_root / "fragments" / source_key
    fragment_parent.mkdir(parents=True, exist_ok=True)
    if fragment_parent.resolve(strict=True) != fragment_parent:
        raise ValueError("M6 materializer fragment parent is not physical")
    name = PurePosixPath(shard["path"]).name
    final = fragment_parent / name
    temporary = final.with_name(final.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        _remove_tree_durable(temporary, fragment_parent, label="M6 materializer partial")

    expanded, compressed, receipt = _derive_shard_materialization(
        cache_root=cache_root, source_key=source_key, source=source,
        shard=shard, workers=workers,
    )
    if final.exists() or final.is_symlink():
        _validate_materialized_directory(
            final, expanded=expanded, compressed=compressed, receipt=receipt,
        )
        return receipt

    temporary.mkdir(mode=0o700)
    installed = False
    try:
        for stage, path, payload in (
            ("fragment-written", temporary / "fragment.jsonl.gz", compressed),
            ("receipt-written", temporary / "receipt.json", canonical_json(receipt)),
        ):
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if failure_hook:
                failure_hook(stage)
        _fsync_directory(temporary)
        if failure_hook:
            failure_hook("temporary-directory-fsynced")
            failure_hook("before-rename")
        os.replace(temporary, final)
        installed = True
        if failure_hook:
            failure_hook("after-rename")
        _fsync_directory(fragment_parent)
        if failure_hook:
            failure_hook("parent-directory-fsynced")
    except Exception:
        if installed:
            try:
                if final.is_dir() and not final.is_symlink():
                    shutil.rmtree(final)
                else:
                    final.unlink()
                if failure_hook:
                    failure_hook("rollback-fsync")
                _fsync_directory(fragment_parent)
            except Exception as cleanup_error:
                raise RuntimeError(
                    "M6 materializer shard publication state unknown after rollback failure"
                ) from cleanup_error
        else:
            _remove_tree_durable(temporary, fragment_parent, label="M6 materializer partial")
        raise
    return receipt


def materialize_shard(
    *, cache_root: Path, output_root: Path, source_key: str,
    source: dict[str, Any], shard: dict[str, Any], workers: int,
    failure_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Materialize one exact member of the pinned 85-shard inventory."""
    return _materialize_shard_from_inventory(
        cache_root=cache_root, output_root=output_root, source_key=source_key,
        source=source, shard=shard, workers=workers,
        inventory=load_source_shards(), failure_hook=failure_hook,
    )


def _materialize_shard_fixture(
    *, cache_root: Path, output_root: Path, source_key: str,
    source: dict[str, Any], shard: dict[str, Any], workers: int,
    failure_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Private small-fixture seam; production entry points cannot inject inventory."""
    inventory = {"sources": {source_key: {**source, "shards": [shard]}}}
    return _materialize_shard_from_inventory(
        cache_root=cache_root, output_root=output_root, source_key=source_key,
        source=source, shard=shard, workers=workers, inventory=inventory,
        failure_hook=failure_hook,
    )


def materialize_all(
    cache_root: Path, output_root: Path, *, workers: int = 8,
    summary_failure_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    inventory = load_source_shards()
    cache_root = cache_root.resolve(strict=True)
    if cache_root.is_symlink():
        raise ValueError("M6 source cache root must be physical")
    _require_download_summary(cache_root)
    output_parent = output_root.parent.resolve(strict=True)
    output_root = output_parent / output_root.name
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        raise ValueError("M6 materialization output root is unsafe")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "materialization-summary.json"
    artifacts = []
    for source_key in sorted(inventory["sources"]):
        source = inventory["sources"][source_key]
        for shard in source["shards"]:
            receipt = _materialize_shard_from_inventory(
                cache_root=cache_root,
                output_root=output_root,
                source_key=source_key,
                source=source,
                shard=shard,
                workers=workers,
                inventory=inventory,
            )
            artifacts.append({
                "fragmentSha256": receipt["fragmentSha256"],
                "path": shard["path"],
                "receiptSha256": sha256(canonical_json(receipt)).hexdigest(),
                "rows": receipt["rows"],
                "sourceKey": source_key,
            })
    labels = Counter()
    rows = 0
    for artifact, receipt in zip(artifacts, (
        parse_json_bytes(
            (
                output_root / "fragments" / artifact["sourceKey"]
                / PurePosixPath(artifact["path"]).name / "receipt.json"
            ).read_bytes(),
            label="materializer shard receipt",
        )
        for artifact in artifacts
    ), strict=True):
        rows += receipt["rows"]
        labels.update(receipt["labels"])
    if rows != 523040 or labels != Counter({"real": 157853, "full_synthetic": 209019, "tampered": 156168}):
        raise ValueError("M6 materialized row/label census changed")
    summary = {
        "decodedRgbNamespaceHex": DECODED_RGB_NAMESPACE.hex(),
        "artifacts": artifacts,
        "fragments": len(artifacts),
        "h3PixelsRead": False,
        "labels": dict(sorted(labels.items())),
        "pixelsRead": True,
        "rows": rows,
        "schemaVersion": 1,
        "sourceShardsSha256": SOURCE_SHARDS_SHA256,
        "status": "m6-fresh-metadata-materialized",
    }
    _publish_summary(
        summary_path, summary,
        fields={
            "decodedRgbNamespaceHex", "artifacts", "fragments", "h3PixelsRead",
            "labels", "pixelsRead", "rows", "schemaVersion",
            "sourceShardsSha256", "status",
        },
        label="materialization summary", failure_hook=summary_failure_hook,
    )
    return summary


def load_materialized_parts(
    output_root: Path, *, cache_root: Path | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Re-verify a complete materialization cache and return the three frozen parts."""
    inventory = load_source_shards()
    output_root = output_root.resolve(strict=True)
    if output_root.is_symlink():
        raise ValueError("M6 materialization root must be physical")
    summary_path = output_root / "materialization-summary.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise ValueError("M6 materialization summary missing")
    summary_raw = summary_path.read_bytes()
    summary = parse_json_bytes(summary_raw, label="materialization summary")
    expected_summary_keys = {
        "artifacts", "decodedRgbNamespaceHex", "fragments", "h3PixelsRead", "labels",
        "pixelsRead", "rows", "schemaVersion", "sourceShardsSha256", "status",
    }
    if (
        set(summary) != expected_summary_keys
        or summary_raw != canonical_json(summary)
        or summary["schemaVersion"] != 1
        or summary["status"] != "m6-fresh-metadata-materialized"
        or summary["h3PixelsRead"] is not False
        or summary["pixelsRead"] is not True
        or summary["sourceShardsSha256"] != SOURCE_SHARDS_SHA256
        or summary["decodedRgbNamespaceHex"] != DECODED_RGB_NAMESPACE.hex()
        or summary["fragments"] != 85
        or summary["rows"] != 523040
        or summary["labels"] != {"full_synthetic": 209019, "real": 157853, "tampered": 156168}
        or not isinstance(summary["artifacts"], list)
        or len(summary["artifacts"]) != 85
    ):
        raise ValueError("M6 materialization summary boundary changed")
    expected_artifacts = []
    for source_key in sorted(inventory["sources"]):
        for shard in inventory["sources"][source_key]["shards"]:
            expected_artifacts.append((source_key, shard))
    parts: dict[str, list[dict[str, str]]] = {"set_train": [], "set_validation": [], "ood_test": []}
    seen_artifacts: set[tuple[str, str]] = set()
    for artifact, (source_key, shard) in zip(summary["artifacts"], expected_artifacts, strict=True):
        if not isinstance(artifact, dict) or set(artifact) != {"fragmentSha256", "path", "receiptSha256", "rows", "sourceKey"}:
            raise ValueError("M6 materialization artifact schema changed")
        if artifact["sourceKey"] != source_key or artifact["path"] != shard["path"]:
            raise ValueError("M6 materialization artifact order changed")
        identity = (source_key, shard["path"])
        if identity in seen_artifacts:
            raise ValueError("duplicate M6 materialization artifact")
        seen_artifacts.add(identity)
        name = PurePosixPath(shard["path"]).name
        shard_dir = output_root / "fragments" / source_key / name
        if (
            not shard_dir.is_dir() or shard_dir.is_symlink()
            or shard_dir.resolve(strict=True) != shard_dir
            or set(os.listdir(shard_dir)) != {"fragment.jsonl.gz", "receipt.json"}
        ):
            raise ValueError("M6 materialization shard directory is unsafe")
        fragment = shard_dir / "fragment.jsonl.gz"
        receipt_path = shard_dir / "receipt.json"
        if (
            not fragment.is_file() or fragment.is_symlink()
            or not receipt_path.is_file() or receipt_path.is_symlink()
            or fragment.resolve(strict=True) != fragment
            or receipt_path.resolve(strict=True) != receipt_path
        ):
            raise ValueError("M6 materialization artifact path is unsafe")
        receipt_raw = receipt_path.read_bytes()
        receipt = parse_json_bytes(receipt_raw, label="materializer shard receipt")
        validate_shard_receipt(receipt, shard)
        if (
            receipt_raw != canonical_json(receipt)
            or sha256(receipt_raw).hexdigest() != artifact["receiptSha256"]
            or receipt.get("sourceShard") != shard
            or receipt.get("fragmentSha256") != artifact["fragmentSha256"]
            or receipt.get("rows") != artifact["rows"]
            or fragment.stat().st_size != receipt["fragmentBytes"]
            or _digest_file(fragment) != artifact["fragmentSha256"]
        ):
            raise ValueError("M6 materialization artifact receipt changed")
        try:
            expanded = gzip.decompress(fragment.read_bytes())
        except (OSError, EOFError) as exc:
            raise ValueError("M6 materialization fragment gzip invalid") from exc
        if sha256(expanded).hexdigest() != receipt.get("expandedSha256") or not expanded.endswith(b"\n"):
            raise ValueError("M6 materialization fragment expansion changed")
        if canonical_gzip(expanded) != fragment.read_bytes():
            raise ValueError("M6 materialization fragment is not canonical gzip")
        part = "set_train" if source_key == "omniFakeSet" and shard["partition"] == "train" else (
            "set_validation" if source_key == "omniFakeSet" else "ood_test"
        )
        rows = []
        for index, line in enumerate(expanded.splitlines()):
            row = parse_json_bytes(line, label=f"materialized {name} row {index}")
            if canonical_json(row) != line + b"\n":
                raise ValueError("M6 materialization row is not canonical")
            rows.append(canonical_fresh_row(row, part))
        if len(rows) != receipt["rows"] or Counter(row["label"] for row in rows) != Counter(receipt["labels"]):
            raise ValueError("M6 materialization fragment census changed")
        if cache_root is not None:
            cache_root = cache_root.resolve(strict=True)
            _require_download_receipt(
                cache_root, source_key, inventory["sources"][source_key], shard,
            )
            expected_expanded, expected_compressed, expected_receipt = _derive_shard_materialization(
                cache_root=cache_root, source_key=source_key,
                source=inventory["sources"][source_key], shard=shard, workers=1,
            )
            if (
                expected_expanded != expanded or expected_compressed != fragment.read_bytes()
                or expected_receipt != receipt
            ):
                raise ValueError("M6 materialization source preimage changed")
        parts[part].extend(rows)
    from benchmark.m6.prepare import census_all
    census_all(parts)
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("download", "materialize"), required=True)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    if args.phase == "download":
        if args.output_root is not None:
            raise SystemExit("download does not accept --output-root")
        result = download_all_shards(args.cache_root)
    else:
        if args.output_root is None:
            raise SystemExit("materialize requires --output-root")
        result = materialize_all(args.cache_root, args.output_root, workers=args.workers)
    print(canonical_json(result).decode(), end="")


if __name__ == "__main__":
    main()
