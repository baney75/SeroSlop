"""P6 frontier data gate.

This is a score-blind, metadata-first boundary.  It can validate an injected
member inventory and (when a caller has already acquired one exact member)
verify its bytes and decoded RGB facts.  It never downloads a source, reads a
model score, or claims that upstream rights have been cleared.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import stat
from typing import Any, Iterable, Mapping

from benchmark.m6.materialize import image_facts
from benchmark.m6.prepare import DHashIndex

ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = ROOT / "benchmark/m6/p6-frontier-inventory.json"
HEX64 = set("0123456789abcdef")

SOURCE_COMMIT = "d4bc2fb9299534e821f05914d51cab0d41e7a030"
SOURCE_LOCK_COMMIT = "20154cb1bceda64aa74bbf1020f0a8ae8b2a513f"
SOURCE_LOCK_TREE = "8469c2609db99096695bf9d9f4ee88b54cc09d0b"

# Exact P5 pins.  Container digests are authoritative until a member-level
# inventory is supplied; absent member facts are deliberately not guessed.
SOURCES: dict[str, dict[str, Any]] = {
    "aigenimages2026-train": {
        "dataset": "pthan12/AIGenImages2026", "publisher": "pthan12",
        "revision": "073e1924d9d0d85ac97a53b07947b6ac95ce241c", "partition": "train",
        "role": "balanced-training", "label": "synthetic", "rows": 4879,
        "containerPath": "aigenimages2026.tar.gz", "containerBytes": 11138511098,
        "containerSha256": "67c6042712f783aebfdb29f8a8903dfc94fc7ac54fee5c154eaf6b880d0ec498",
    },
    "aigenimages2026-test": {
        "dataset": "pthan12/AIGenImages2026", "publisher": "pthan12",
        "revision": "073e1924d9d0d85ac97a53b07947b6ac95ce241c", "partition": "val",
        "role": "synthetic-acceptance", "label": "synthetic", "rows": 559,
        "containerPath": "aigenimages2026.tar.gz", "containerBytes": 11138511098,
        "containerSha256": "67c6042712f783aebfdb29f8a8903dfc94fc7ac54fee5c154eaf6b880d0ec498",
    },
    "taste": {
        "dataset": "purvanshi/TASTE", "publisher": "purvanshi",
        "revision": "731a7f588d433214c6d864d2e9f47978d91aed6b", "partition": "train",
        "role": "synthetic-acceptance", "label": "synthetic", "rows": 644,
        "assetsRows": 644, "assetsParquetBytes": 26772,
        "assetsParquetSha256": "326e9300bac89f5ed884de7a9a59dccfc7d5aa203f6d2844f604000dc4e32bf1",
        "modelCounts": {"GPT Image 1.5": 161, "Nano Banana 2": 161, "FLUX.2 [max]": 161, "Seedream 5.0 Lite": 161},
        "memberCount": 644, "memberBytes": 1598239561,
        "inventorySha256": "9819630a92174c134a99332f78d619003a5c303442d93fb22ca2ba7e5382d729",
    },
    "nano-banana": {
        "dataset": "bitmind/nano-banana", "publisher": "bitmind",
        "revision": "9ea8da32a5be03f4946e6cb10c2d2f8e90f0a0a4", "partition": "train",
        "role": "synthetic-acceptance", "label": "synthetic", "rows": 9457,
        "publisherImageCount": 9457, "memberCount": 31, "memberBytes": 14853195253,
        "inventorySha256": "0ae6aa06cb0be3d58eccb0188173a6dc9cd9f81d91950f108796349a07a9fcf4",
    },
    "x-aigd": {
        "dataset": "Coxy7/X-AIGD", "publisher": "Coxy7",
        "revision": "92180f32030507ab54a40d6f1b88f39d6cec8178", "partition": "labeled_test",
        "role": "synthetic-acceptance", "label": "synthetic", "rows": 2419,
        "containerPath": "data/labeled_test-00000-of-00001.parquet", "containerBytes": 3488049189,
        "containerSha256": "f86630ae51ef1103de204c879ad74d70bacaeca258489f2c32102851344a5c75",
    },
}
SOURCE_CARDS = {
    "aigenimages2026-train": ("53d6bbd13bdeb16c6da0f4d7a780fca4b411422e26debf61450ec758847a6ff0", "benchmark/m6/source-cards/aigenimages2026.README.md"),
    "aigenimages2026-test": ("53d6bbd13bdeb16c6da0f4d7a780fca4b411422e26debf61450ec758847a6ff0", "benchmark/m6/source-cards/aigenimages2026.README.md"),
    "x-aigd": ("e0121b9772a42b5953b68d26e99c4b0d6be29b7995156e13494a67b6dde99484", "benchmark/m6/source-cards/x-aigd.README.md"),
    "taste": ("a602cf5bf11e67ab3e89f4fe3853326ae2978a8312f8d1633b1f2a9d3d1c562b", "benchmark/m6/source-cards/taste.README.md"),
    "nano-banana": ("d7f5f97cea4776c0a0c1ed97ba5139afda4f94dc42d5aec5dff2dd2c459f875c", "benchmark/m6/source-cards/nano-banana.README.md"),
}
for _key, (_sha, _path) in SOURCE_CARDS.items():
    SOURCES[_key].update({"cardSha256": _sha, "cardPath": _path, "licenseClaim": "cc-by-4.0" if _key.startswith(("aigen", "x-aigd")) else "mit", "labelEvidenceScope": "publisherAssertionOnly", "publisherAssertionOnly": True, "independentOriginProofClaimed": False})


def validate_source_cards() -> None:
    for source_key, (expected, relative) in SOURCE_CARDS.items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"publisher card bytes missing or changed: {source_key}")

OMNI_SOURCES = {
    "omni-set-train-real": ("JamalLee/Omni-Fake-SET", "724e97f5fc9f4b89f59631a8d4e6331712b7d441", "train", "real", 48662),
    "omni-set-train-synthetic": ("JamalLee/Omni-Fake-SET", "724e97f5fc9f4b89f59631a8d4e6331712b7d441", "train", "synthetic", 43783),
    "omni-set-selector-real": ("JamalLee/Omni-Fake-SET", "724e97f5fc9f4b89f59631a8d4e6331712b7d441", "validation", "real", 2000),
    "omni-set-selector-synthetic": ("JamalLee/Omni-Fake-SET", "724e97f5fc9f4b89f59631a8d4e6331712b7d441", "validation", "synthetic", 2000),
    "omni-set-validation-real": ("JamalLee/Omni-Fake-SET", "724e97f5fc9f4b89f59631a8d4e6331712b7d441", "validation", "real", 5000),
    "omni-ood-test-real": ("JamalLee/Omni-Fake-OOD", "9ed7e38bbdb4aeb2eb553896a5890680a9ffcf17", "test", "real", 5000),
    "omni-ood-test-synthetic": ("JamalLee/Omni-Fake-OOD", "9ed7e38bbdb4aeb2eb553896a5890680a9ffcf17", "test", "synthetic", 28693),
    "omni-set-validation-synthetic": ("JamalLee/Omni-Fake-SET", "724e97f5fc9f4b89f59631a8d4e6331712b7d441", "validation", "synthetic", 58228),
}

ALLOCATIONS = {
    "train": {"real": 48662, "synthetic": 48662, "aigenimages2026-train": 4879},
    "selector": {"real": 2000, "synthetic": 2000},
    "synthetic-acceptance": {"setValidation": 58228, "ood": 28693, "aigenimages2026-test": 559, "taste": 644, "x-aigd": 2419, "nano-banana": 9457},
    "real-acceptance": {"setValidationReal": 5000, "oodReddit": 5000},
}
SOURCE_ROLES = {key: "train" if key in {"omni-set-train-real", "omni-set-train-synthetic", "aigenimages2026-train"} else "selector" if key.startswith("omni-set-selector") else "synthetic-acceptance" if key in {"omni-set-validation-synthetic", "omni-ood-test-synthetic", "aigenimages2026-test", "taste", "x-aigd", "nano-banana"} else "real-acceptance" for key in set(OMNI_SOURCES) | set(SOURCES)}
QUOTA_CORRECTION = {"status": "publisher-metadata-missing-member-quarantined", "aigenimages2026TrainPublisherRows": 4880, "aigenimages2026TrainAdmittedRows": 4879, "aigenimages2026TrainMissingMember": "image_midjourneyv7_300.png", "omniSetTrainSynthetic": 43783, "syntheticTrainTotal": 48662, "rightsClaimed": False}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def inventory_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    normalized = []
    for row in rows:
        if set(row) != {"bytes", "path", "sha256"}:
            raise ValueError("inventory member key set changed")
        if type(row["bytes"]) is not int or row["bytes"] < 0 or not isinstance(row["path"], str):
            raise ValueError("invalid inventory member")
        digest = row["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in HEX64 for c in digest):
            raise ValueError("invalid member sha256")
        normalized.append({"bytes": row["bytes"], "path": row["path"], "sha256": digest})
    normalized.sort(key=lambda r: r["path"].encode("utf-8"))
    if len({r["path"] for r in normalized}) != len(normalized):
        raise ValueError("duplicate inventory member")
    return hashlib.sha256(b"".join(canonical_json(r) for r in normalized)).hexdigest()


def validate_repo_tree(source_key: str, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    if source_key not in SOURCES:
        raise ValueError("unknown source key")
    rows = list(rows); spec = SOURCES[source_key]
    expected = spec.get("memberCount")
    if expected is not None and len(rows) != expected:
        raise ValueError("unexpected inventory member count")
    if expected is not None and sum(r["bytes"] for r in rows) != spec["memberBytes"]:
        raise ValueError("unexpected inventory byte total")
    if expected is not None and inventory_digest(rows) != spec["inventorySha256"]:
        raise ValueError("inventory digest mismatch")
    return {"sourceKey": source_key, "memberCount": len(rows), "memberBytes": sum(r["bytes"] for r in rows), "inventorySha256": inventory_digest(rows)}


def validate_taste_assets(rows: Iterable[Mapping[str, Any]], tree_paths: Iterable[str]) -> dict[str, int]:
    rows = list(rows); paths = set(tree_paths)
    if len(rows) != 644 or {r.get("asset_id") for r in rows} != set(range(1, 645)) or len({r.get("image_path") for r in rows}) != 644 or len({r.get("image_url") for r in rows}) != 644 or {r.get("image_path") for r in rows} != paths:
        raise ValueError("TASTE assets paths/count mismatch")
    counts = {name: 0 for name in SOURCES["taste"]["modelCounts"]}
    for row in rows:
        if set(row) != {"asset_id", "model", "image_url", "track", "image_path"} or row["model"] not in counts or type(row["asset_id"]) is not int or not isinstance(row["image_url"], str) or not row["image_url"].startswith("https://") or not isinstance(row["track"], str) or row["track"] not in {"descriptions", "aesthetics"}:
            raise ValueError("invalid TASTE assets row")
        counts[row["model"]] += 1
    if counts != SOURCES["taste"]["modelCounts"]:
        raise ValueError("TASTE model counts mismatch")
    tracks = {track: sum(row["track"] == track for row in rows) for track in {row["track"] for row in rows}}
    if tracks != {"descriptions": 320, "aesthetics": 324}:
        raise ValueError("TASTE track census mismatch")
    return counts


def validate_taste_member_binding(rows: Iterable[Mapping[str, Any]], tree_paths: Iterable[str]) -> int:
    """Require every TASTE metadata row to bind one exact image member."""
    paths = set(tree_paths); rows = list(rows)
    if len(rows) != len(paths) or any(set(row) != {"asset_id", "model", "image_url", "track", "image_path", "memberPath"} or row["image_path"] != row["memberPath"] for row in rows):
        raise ValueError("TASTE row/member binding changed")
    validate_taste_assets(({key: row[key] for key in ("asset_id", "model", "image_url", "track", "image_path")} for row in rows), paths)
    return len(rows)


def validate_nano_metadata(shard_rows: Iterable[Any], row_census: int) -> int:
    rows = list(shard_rows)
    if len(rows) != 31:
        raise ValueError("nano-banana requires 31 parquet shards")
    paths = [r if isinstance(r, str) else r.get("path") for r in rows]
    if len(set(paths)) != 31 or any(not isinstance(p, str) or not p.startswith("data/") or not p.endswith(".parquet") for p in paths):
        raise ValueError("invalid nano-banana shard paths")
    if row_census != SOURCES["nano-banana"]["publisherImageCount"]:
        raise ValueError("nano-banana row census mismatch")
    return row_census


def validate_aigen_tar_members(tar_path: Path, *, partition: str, expected_rows: int) -> list[str]:
    """Validate injected AIGen archive member names without extracting pixels."""
    if partition not in {"train", "val"} or type(expected_rows) is not int or expected_rows <= 0:
        raise ValueError("AIGen partition/count invalid")
    members = []
    root = PurePosixPath("mnt/cephfs/home/common/datasets/online_sid_dataset/AIGenImages/AIGenImages2026")
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or any(piece in ("", ".", "..") for piece in name.parts) or any(ord(c) < 32 for c in member.name):
                raise ValueError("unsafe AIGen TAR member")
            if not name.is_relative_to(root):
                raise ValueError("AIGen member outside pinned root")
            relative = name.relative_to(root)
            if member.isdir():
                continue
            if not member.isfile() or bool(member.sparse):
                raise ValueError("unsafe AIGen TAR member type/mode")
            if not relative.parts or relative.parts[0] != partition or len(relative.parts) < 3 or relative.parts[1] != "1_fake":
                continue
            if PurePosixPath(member.name).suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            members.append(member.name)
    if len(members) != expected_rows or len(set(members)) != len(members):
        raise ValueError("AIGen partition census mismatch")
    return sorted(members, key=lambda value: value.encode("utf-8"))


def validate_aigen_metadata_csv(rows: Iterable[Mapping[str, Any]], image_members: Iterable[str], *, partition: str) -> int:
    """Bind publisher metadata rows to the exact image member set."""
    members = set(image_members); rows = list(rows)
    if partition not in {"train", "val"}:
        raise ValueError("AIGen metadata census mismatch")
    required = {"image_id", "filename", "caption", "caption_id", "split"}
    expected_rows = {"train": 4880, "val": 559}[partition]
    if len(rows) != expected_rows:
        raise ValueError("AIGen publisher metadata census mismatch")
    basenames = {PurePosixPath(member).name for member in members}
    if len(basenames) != len(members) or any(set(row) != required or row["split"] != partition for row in rows):
        raise ValueError("AIGen publisher metadata binding changed")
    metadata_names = {PurePosixPath(row["filename"]).name for row in rows}
    if len(metadata_names) != len(rows):
        raise ValueError("duplicate AIGen metadata member")
    missing = sorted(metadata_names - basenames)
    if partition == "train" and missing != ["image_midjourneyv7_300.png"]:
        raise ValueError("unexpected AIGen missing member")
    if partition == "val" and (missing or metadata_names != basenames):
        raise ValueError("AIGen validation metadata/member set mismatch")
    if partition == "train" and metadata_names - basenames != {"image_midjourneyv7_300.png"}:
        raise ValueError("AIGen train metadata/member set mismatch")
    if partition == "train" and basenames - metadata_names:
        raise ValueError("AIGen train metadata/member set mismatch")
    return {"publisherRows": len(rows), "admittedRows": len(metadata_names & basenames), "quarantined": missing}


def validate_parquet_rows(rows: Iterable[Mapping[str, Any]], *, source_key: str, expected_rows: int) -> int:
    """Strict injected row-shape/census seam for X-AIGD and nano-banana."""
    if source_key not in {"x-aigd", "nano-banana"}:
        raise ValueError("Parquet source is not pinned")
    rows = list(rows)
    if len(rows) != expected_rows:
        raise ValueError("Parquet row census mismatch")
    x_fields = {"image", "generator", "uid", "labels", "original_prompt", "positive_prompt", "negative_prompt", "guidance_scale", "num_inference_steps", "scheduler", "seed", "width", "height", "image_format", "jpeg_quality", "chroma_subsampling"}
    nano_fields = {"id", "image", "format", "mode", "width", "height", "uploadtime"}
    required = x_fields if source_key == "x-aigd" else nano_fields
    if any(set(row) != required for row in rows):
        raise ValueError("Parquet row schema changed")
    if source_key == "x-aigd":
        groups = {str(row["generator"]) for row in rows}
        expected_groups = {"SD_3.5L-raw": 200, "SD_1.5_realistic_vision-raw": 199, "FLUX.1_dev-raw": 198, "Infinity-raw": 198, "FLUX.1_dev-iPhonePhoto-raw": 194, "SD_3-raw": 194, "SD_3.5L-iPhonePhoto-raw": 194, "PA_alpha-raw": 193, "SDXL_realism_engine-raw": 190, "HYDiT_1.2-raw": 181, "Lumina_Next-raw": 173, "SD_1.4-raw": 161, "SDXL_1.0-raw": 144}
        if len({row["uid"] for row in rows}) != 1290 or len({(row["generator"], row["uid"]) for row in rows}) != len(rows) or groups != set(expected_groups) or {key: sum(row["generator"] == key for row in rows) for key in expected_groups} != expected_groups or any(not isinstance(row["generator"], str) or not row["generator"] or not isinstance(row["image"], Mapping) or set(row["image"]) != {"bytes", "path"} for row in rows):
            raise ValueError("X-AIGD image/uid schema changed")
    else:
        if {row["id"] for row in rows} != set(range(9457)) or any(row["format"] != "PNG" or row["mode"] != "RGB" or row["width"] <= 0 or row["height"] <= 0 or not isinstance(row["image"], Mapping) or set(row["image"]) != {"bytes", "path"} for row in rows):
            raise ValueError("nano image/id schema changed")
    return len(rows)


def _safe_member_path(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise ValueError("member path is not a regular physical file")


def verify_member(path: Path, expected: Mapping[str, Any], *, publisher: str, label: str, read_pixels: bool = True, member_path: str | None = None) -> dict[str, Any]:
    """Verify one already-acquired member; ambiguous facts raise and caller quarantines."""
    if expected.get("publisher") != publisher or expected.get("label") != label:
        raise ValueError("publisher/label binding changed")
    _safe_member_path(path)
    # Stream the acquisition hash.  A container must never be passed to the
    # image decoder; callers set ``read_pixels`` only for an actual image row.
    digest_ctx = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest_ctx.update(block)
    digest = digest_ctx.hexdigest()
    if type(expected.get("bytes")) is not int or path.stat().st_size != expected["bytes"]:
        raise ValueError("member byte count mismatch")
    if expected.get("sha256") != digest:
        raise ValueError("member SHA-256 mismatch")
    result = {"path": path.as_posix(), "bytes": expected["bytes"], "sha256": digest, "publisher": publisher, "label": label}
    if member_path is not None:
        relative = PurePosixPath(member_path)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
            raise ValueError("member path is unsafe")
        result["memberPath"] = member_path
    if read_pixels:
        # Actual image members are bounded before decoding; archive/parquet
        # containers are metadata-only and therefore cannot reach this branch.
        if expected.get("kind") not in {"image", None}:
            raise ValueError("container cannot be decoded as an image")
        if expected["bytes"] > 512 * 1024 * 1024:
            raise ValueError("image member exceeds decode safety bound")
        payload = path.read_bytes()
        encoded, decoded, dhash, width, height = image_facts(payload)
        if encoded != digest:
            raise ValueError("encoded image hash changed")
        result.update({"decodedRgbSha256": decoded, "dhash64": dhash, "width": width, "height": height})
    return result


def verify_downloaded_member(path: Path, source_key: str, *, expected_bytes: int, expected_sha256: str, member_path: str) -> dict[str, Any]:
    """Offline verifier for one exact source member (no HTTP/client side effects)."""
    source = SOURCES.get(source_key)
    if source is None:
        raise ValueError("unknown source key")
    return verify_member(path, {"bytes": expected_bytes, "sha256": expected_sha256, "publisher": source["publisher"], "label": source["label"], "kind": "container"}, publisher=source["publisher"], label=source["label"], read_pixels=False, member_path=member_path)


def materialize_image_member(path: Path, source_key: str, *, expected_bytes: int, expected_sha256: str, member_path: str) -> dict[str, Any]:
    """Decode one already verified image member and bind EXIF-transposed RGB facts."""
    source = SOURCES.get(source_key)
    if source is None:
        raise ValueError("unknown source key")
    return verify_member(path, {"bytes": expected_bytes, "sha256": expected_sha256, "publisher": source["publisher"], "label": source["label"], "kind": "image"}, publisher=source["publisher"], label=source["label"], read_pixels=True, member_path=member_path)


def quarantine(path: Path, quarantine_root: Path, *, reason: str) -> Path:
    """Emit a durable reject ledger without mutating the acquired cache."""
    quarantine_root.mkdir(parents=True, exist_ok=True)
    target = quarantine_root / (path.name + ".reject.json")
    if target.exists() or target.is_symlink():
        raise FileExistsError("quarantine destination already exists")
    atomic_write_receipt(target, {"path": path.as_posix(), "reason": reason, "status": "quarantined", "sourcePreserved": True})
    return target


def atomic_write_receipt(final: Path, receipt: Any) -> None:
    """Publish canonical receipt exactly once, with rollback on failure."""
    final.parent.mkdir(parents=True, exist_ok=True)
    if not final.parent.resolve(strict=True).is_dir() or final.parent.is_symlink():
        raise ValueError("receipt parent is not a physical directory")
    if final.exists() or final.is_symlink():
        if final.is_symlink() or not final.is_file():
            raise FileExistsError("receipt destination already exists")
        existing = final.read_bytes()
        if existing == canonical_json(receipt):
            return
        raise FileExistsError("receipt destination already exists with different bytes")
    partial = final.with_name(final.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError("stale receipt partial exists")
    payload = canonical_json(receipt); partial.write_bytes(payload)
    with partial.open("rb") as handle: os.fsync(handle.fileno())
    installed = False
    try:
        os.replace(partial, final)
        installed = True
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        if installed:
            try:
                final.unlink(missing_ok=True)
                descriptor = os.open(final.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except Exception:
                raise RuntimeError("receipt publication state unknown")
        partial.unlink(missing_ok=True)
        raise


def canonical_identity(row: Mapping[str, Any]) -> str:
    fields = ("dataset", "revision", "partition", "sourceGroup", "rowId")
    if any(not isinstance(row.get(k), str) or not row[k] for k in fields):
        raise ValueError("identity fields missing")
    return hashlib.sha256(canonical_json({k: row[k] for k in fields})).hexdigest()


def verification_receipt_digest(row: Mapping[str, Any], materialization_receipt: Mapping[str, Any]) -> str:
    facts = {key: materialization_receipt.get(key) for key in ("schemaVersion", "status", "sourceKey", "dataset", "revision", "partition", "memberPath", "containerSha256", "inventorySha256", "containerPath", "memberContainerPath", "memberContainerSha256", "cardSha256", "licenseClaim", "publisherAssertionOnly", "independentOriginProofClaimed", "encodedBytesSha256", "decodedRgbSha256", "dhash64", "width", "height", "labelEvidenceScope")}
    return hashlib.sha256(canonical_json(facts)).hexdigest()


def validate_admission(rows: Iterable[Mapping[str, Any]], historical: Iterable[Mapping[str, Any]] = (), verified_receipts: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build a canonical ledger; identity/hash/dHash collisions are rejected."""
    admitted = []; quarantined = []; seen = set(); history = list(historical)
    historical_ids = {canonical_identity(r) for r in history}
    encoded = {r.get("encodedBytesSha256") for r in history}; decoded = {r.get("decodedRgbSha256") for r in history if r.get("decodedRgbSha256")}; dhash_index = DHashIndex()
    for index, old in enumerate(sorted((int(r["dhash64"], 16) for r in history if isinstance(r.get("dhash64"), str) and len(r["dhash64"]) == 16))): dhash_index.add(old, index)
    for row in rows:
        try:
            required = {"sourceKey", "publisher", "dataset", "revision", "partition", "sourceGroup", "rowId", "memberPath", "label", "encodedBytesSha256", "decodedRgbSha256", "dhash64"}
            receipt_required = verified_receipts is not None
            allowed_keys = (required | {"verificationReceiptSha256"}) if receipt_required else required
            if set(row) not in (allowed_keys, allowed_keys | {"role"}):
                raise ValueError("admission member schema changed")
            source = SOURCES.get(row.get("sourceKey"))
            omni = OMNI_SOURCES.get(row.get("sourceKey"))
            if source is None and omni is None:
                raise ValueError("source or publisher label not pinned")
            if source is not None:
                expected_triplet = (source["dataset"], source["revision"], source["partition"]); expected_label = source["label"]; expected_publisher = source["publisher"]
            else:
                expected_triplet = omni[:3]; expected_label = omni[3]; expected_publisher = "JamalLee"
            if row.get("publisher") != expected_publisher or row.get("label") != expected_label:
                raise ValueError("source or publisher label not pinned")
            if (row.get("dataset"), row.get("revision"), row.get("partition")) != expected_triplet:
                raise ValueError("member source pin mismatch")
            relative = PurePosixPath(row["memberPath"])
            if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
                raise ValueError("member path is unsafe")
            for name in ("encodedBytesSha256", "decodedRgbSha256"):
                if not isinstance(row[name], str) or len(row[name]) != 64 or any(c not in HEX64 for c in row[name]):
                    raise ValueError("member image SHA-256 schema invalid")
            if not isinstance(row["dhash64"], str) or len(row["dhash64"]) != 16 or any(c not in HEX64 for c in row["dhash64"]):
                raise ValueError("member dHash schema invalid")
            if receipt_required:
                receipt = verified_receipts.get(canonical_identity(row)) if verified_receipts else None
                receipt_keys = {"schemaVersion", "status", "sourceKey", "dataset", "revision", "partition", "memberPath", "containerSha256", "inventorySha256", "containerPath", "memberContainerPath", "memberContainerSha256", "cardSha256", "licenseClaim", "publisherAssertionOnly", "independentOriginProofClaimed", "encodedBytesSha256", "decodedRgbSha256", "dhash64", "width", "height", "labelEvidenceScope", "receiptSha256"}
                if not isinstance(receipt, Mapping) or set(receipt) != receipt_keys or receipt.get("schemaVersion") != 1 or receipt.get("status") != "m6-materialized-member-verified" or row.get("verificationReceiptSha256") != receipt.get("receiptSha256"):
                    raise ValueError("member lacks strict materialization receipt")
                expected_scope = {"taste": "per-item-model", "x-aigd": "publisher-dataset-split:labeled_test", "nano-banana": "publisher-dataset-split:train"}.get(row["sourceKey"], "folder+publisher-metadata" if row["sourceKey"].startswith("aigenimages2026") else "materialized-label")
                expected_inventory = SOURCES[row["sourceKey"]].get("inventorySha256") if source is not None else None
                expected_card = SOURCES[row["sourceKey"]].get("cardSha256") if source is not None else None
                if not isinstance(receipt.get("width"), int) or not isinstance(receipt.get("height"), int) or receipt["width"] <= 0 or receipt["height"] <= 0 or receipt.get("publisherAssertionOnly") is not True or receipt.get("independentOriginProofClaimed") is not False or (expected_card is not None and receipt.get("cardSha256") != expected_card) or receipt.get("labelEvidenceScope") != expected_scope or (expected_inventory is not None and receipt.get("inventorySha256") != expected_inventory) or any(receipt.get(key) != row.get(key) for key in ("sourceKey", "dataset", "revision", "partition", "memberPath", "encodedBytesSha256", "decodedRgbSha256", "dhash64")) or (source is not None and source.get("containerSha256") and receipt.get("containerSha256") != source["containerSha256"]) or receipt.get("receiptSha256") != verification_receipt_digest(row, receipt):
                    raise ValueError("member lacks verified materialization receipt")
            identity = canonical_identity(row)
            dhash = int(row["dhash64"], 16) if isinstance(row.get("dhash64"), str) and len(row["dhash64"]) == 16 else None
            if identity in seen or identity in historical_ids or row.get("encodedBytesSha256") in encoded or (row.get("decodedRgbSha256") and row["decodedRgbSha256"] in decoded) or (dhash is not None and dhash_index.matches(dhash, 8)):
                raise ValueError("overlap or duplicate")
            seen.add(identity)
            if row["encodedBytesSha256"] in encoded or row["decodedRgbSha256"] in decoded or dhash_index.matches(dhash, 8):
                raise ValueError("fresh-to-fresh overlap")
            encoded.add(row["encodedBytesSha256"]); decoded.add(row["decodedRgbSha256"]); dhash_index.add(dhash, len(dhash_index))
            admitted.append(dict(row))
        except (KeyError, ValueError) as exc:
            quarantined.append({"row": dict(row), "reason": str(exc)})
    admitted.sort(key=lambda r: canonical_identity(r).encode())
    ledger_hash = hashlib.sha256(canonical_json(admitted)).hexdigest()
    overlap = {"canonicalIdentity": len(historical_ids), "encodedBytesSha256": len(encoded), "decodedRgbSha256": len(decoded), "dhashHammingLe8": len(dhash_index), "quarantined": len(quarantined)}
    overlap["receiptSha256"] = hashlib.sha256(canonical_json(overlap)).hexdigest()
    status = "p6-admission-ledger" if verified_receipts is not None else "p6-admission-provisional"
    return {"status": status, "rows": admitted, "quarantined": quarantined, "rowCount": len(admitted), "quarantineCount": len(quarantined), "ledgerSha256": ledger_hash, "overlapReceipt": overlap}


def admit_candidates(candidates: Iterable[Mapping[str, Any]], historical: Iterable[Mapping[str, Any]], verified_receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Exhaustively admit an injected candidate census before allocation."""
    validate_source_cards()
    candidates = list(candidates)
    ledger = validate_admission(candidates, historical, verified_receipts)
    if ledger["rowCount"] + ledger["quarantineCount"] != len(candidates):
        raise ValueError("candidate census does not reconcile")
    ordered = sorted(candidates, key=lambda row: canonical_identity(row).encode())
    return {**ledger, "candidateRows": len(candidates), "candidateSha256": hashlib.sha256(canonical_json(ordered)).hexdigest()}


def allocate_from_admitted(admitted: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Score-blind deterministic allocator; shortages are terminal."""
    pools: dict[str, list[Mapping[str, Any]]] = {}
    for row in admitted:
        key = row.get("sourceKey")
        role = SOURCE_ROLES.get(key)
        if role is None: raise ValueError("candidate source is not pinned")
        pools.setdefault(key, []).append(row)
    for key in pools: pools[key] = sorted(pools[key], key=lambda row: canonical_identity(row).encode())
    quotas = {"omni-set-train-real": 48662, "omni-set-train-synthetic": 43783, "aigenimages2026-train": 4879, "omni-set-selector-real": 2000, "omni-set-selector-synthetic": 2000, "omni-set-validation-synthetic": 58228, "omni-ood-test-synthetic": 28693, "aigenimages2026-test": 559, "taste": 644, "x-aigd": 2419, "nano-banana": 9457, "omni-set-validation-real": 5000, "omni-ood-test-real": 5000}
    chosen = []
    for source, quota in quotas.items():
        if len(pools.get(source, [])) < quota: raise ValueError(f"allocation shortage: {source}")
        chosen.extend(dict(row, role=SOURCE_ROLES[source]) for row in pools[source][:quota])
    if len({canonical_identity(row) for row in chosen}) != len(chosen): raise ValueError("allocation leakage")
    return {"status": "p6-allocation-verified", "rows": chosen, "rowCount": len(chosen), "allocationSha256": hashlib.sha256(canonical_json(chosen)).hexdigest()}


def write_admission_bundle(output: Path, rows: Iterable[Mapping[str, Any]], historical: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Atomically publish canonical ledger, reject ledger, and overlap receipt."""
    raise RuntimeError("authoritative admission publication is disabled until byte-rederiving materializer integration")


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_p6_source_lock(*, output: Path, candidates: Iterable[Mapping[str, Any]], historical: Iterable[Mapping[str, Any]], artifacts: Mapping[str, str], verified_receipts: Mapping[str, Mapping[str, Any]] | None = None, source_commit: str | None = None, source_tree: str | None = None, failure_injection: str | None = None) -> dict[str, Any]:
    """Build an exact injected-row source-lock bundle; never acquires pixels.

    The function is intentionally unusable as a production shortcut: missing
    frontier rows, history, or artifact receipts produce a metadata-only error.
    """
    raise RuntimeError("P6 operational source-lock publication is disabled until byte-rederiving materializer integration")
    candidates = list(candidates); historical = list(historical)
    if not candidates or not historical:
        raise ValueError("P6 source lock requires exhaustive candidates and authoritative history")
    if verified_receipts is None:
        raise ValueError("P6 source lock requires verified materialization receipts")
    candidate_ledger = admit_candidates(candidates, historical, verified_receipts)
    allocation = allocate_from_admitted(candidate_ledger["rows"])
    rows = allocation["rows"]
    required_artifacts = {"inventory", "history", "materialization", "admission", "overlap", "allocation"}
    if set(artifacts) != required_artifacts:
        raise ValueError("source lock artifact set incomplete")
    lock = source_lock_receipt(source_commit=source_commit, source_tree=source_tree, artifacts=artifacts, admission_sha256=candidate_ledger["ledgerSha256"], allocation_sha256=allocation["allocationSha256"], overlap_sha256=candidate_ledger["overlapReceipt"]["receiptSha256"], history_sha256=artifacts["history"], materialization_sha256=artifacts["materialization"])
    manifests = []; files: dict[str, bytes] = {}
    def round_robin(subset: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        groups = {key: sorted((row for row in subset if row["sourceKey"] == key), key=lambda row: canonical_identity(row).encode()) for key in {row["sourceKey"] for row in subset}}
        output_rows = []
        for index in range(max((len(group) for group in groups.values()), default=0)):
            for key in sorted(groups, key=lambda value: value.encode("utf-8")):
                if index < len(groups[key]): output_rows.append(groups[key][index])
        return output_rows
    for role, filename in (("train", "train-manifest.json"), ("selector", "selector-manifest.json"), ("synthetic-acceptance", "synthetic-acceptance-manifest.json"), ("real-acceptance", "real-acceptance-manifest.json")):
        subset = round_robin([row for row in rows if row.get("role") == role])
        payload = canonical_json(subset); files[filename] = payload
        manifests.append({"path": filename, "rows": len(subset), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    evaluation = round_robin([row for row in rows if row.get("role") == "synthetic-acceptance"])
    if len(evaluation) != 100000:
        raise ValueError("synthetic acceptance must contain exactly 100000 rows")
    batch_receipts = []
    for index in range(0, len(evaluation), 100):
        payload = canonical_json(evaluation[index:index + 100]); filename = f"batches/{index // 100:04d}.json"; files[filename] = payload
        batch_receipts.append({"path": filename, "batch": index // 100, "rows": 100, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    files["admission-ledger.json"] = canonical_json({"status": candidate_ledger["status"], "rows": candidate_ledger["rows"], "ledgerSha256": candidate_ledger["ledgerSha256"]})
    files["reject-ledger.json"] = canonical_json({"status": "p6-reject-ledger", "rows": candidate_ledger["quarantined"], "rowCount": candidate_ledger["quarantineCount"]})
    files["overlap-receipt.json"] = canonical_json(candidate_ledger["overlapReceipt"])
    files["allocation-receipt.json"] = canonical_json(allocation)
    artifact_index = [{"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} for path, payload in sorted(files.items())]
    candidate_sorted = sorted(candidates, key=lambda row: canonical_identity(row).encode())
    candidate_hash = hashlib.sha256(canonical_json(candidate_sorted)).hexdigest()
    packet = {"schemaVersion": 1, "status": "p6-source-lock-verified", "sourceLock": lock, "candidateCensus": {"rows": len(candidates), "sha256": candidate_hash}, "admission": {"rows": candidate_ledger["rowCount"], "sha256": candidate_ledger["ledgerSha256"], "rejects": candidate_ledger["quarantined"]}, "allocation": allocation, "manifests": manifests, "syntheticAcceptanceBatches": batch_receipts, "artifacts": artifact_index, "sourceMaterializationPixelsRead": True, "sourceLockPixelsRead": False, "h3PixelsRead": False, "commercialRightsClearanceClaimed": False}
    files["source-lock.json"] = canonical_json(packet)
    if output.exists() or output.is_symlink():
        raise FileExistsError("source-lock output already exists")
    partial = output.with_name(output.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError("stale source-lock partial exists")
    partial.mkdir(parents=True)
    installed = False
    try:
        for relative, payload in files.items():
            target = partial / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_receipt(target, json.loads(payload.decode("utf-8")))
        _fsync_dir(partial)
        if failure_injection == "before-rename":
            raise RuntimeError("injected source-lock failure before rename")
        os.replace(partial, output); installed = True; _fsync_dir(output.parent)
        if failure_injection == "after-rename":
            raise RuntimeError("injected source-lock failure after rename")
        return packet
    except Exception:
        if installed:
            try:
                shutil.rmtree(output)
                _fsync_dir(output.parent)
            except Exception as cleanup_error:
                raise RuntimeError("source-lock publication state unknown after rollback failure") from cleanup_error
        else:
            shutil.rmtree(partial, ignore_errors=True)
        raise


def build_p6_source_lock_fixture(output: Path, *, artifacts: Mapping[str, str]) -> dict[str, Any]:
    """Write/reopen a tiny non-production packet; fixed quotas remain explicit."""
    if set(artifacts) != {"inventory", "history", "materialization", "admission", "overlap", "allocation"}:
        raise ValueError("fixture artifact set incomplete")
    count_receipt = {"status": "p6-allocation-count-fixture", "counts": ALLOCATIONS, "rows": 211324, "selectionInfluence": False}
    validate_allocation_count_receipt(count_receipt)
    rows = [{"sourceKey": key, "role": role, "rowId": f"fixture-{role}-{key}"} for role, keys in (("train", ("omni-set-train-real", "omni-set-train-synthetic", "aigenimages2026-train")), ("selector", ("omni-set-selector-real", "omni-set-selector-synthetic")), ("synthetic-acceptance", ("omni-set-validation-synthetic",)), ("real-acceptance", ("omni-set-validation-real",))) for key in keys]
    files = {"train-manifest.json": canonical_json([r for r in rows if r["role"] == "train"]), "selector-manifest.json": canonical_json([r for r in rows if r["role"] == "selector"]), "synthetic-acceptance-manifest.json": canonical_json([r for r in rows if r["role"] == "synthetic-acceptance"]), "real-acceptance-manifest.json": canonical_json([r for r in rows if r["role"] == "real-acceptance"]), "admission-ledger.json": canonical_json({"status": "fixture", "rows": rows}), "reject-ledger.json": canonical_json({"status": "fixture", "rows": []}), "overlap-receipt.json": canonical_json({"status": "fixture", "rows": 0}), "allocation-receipt.json": canonical_json(count_receipt), "batches/0000.json": canonical_json(rows[:1])}
    index = [{"path": path, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} for path, payload in sorted(files.items())]
    packet = {"schemaVersion": 1, "status": "p6-source-lock-fixture", "artifacts": index, "syntheticAcceptanceBatches": [{"path": "batches/0000.json", "batch": 0, "rows": 1, "bytes": len(files["batches/0000.json"]), "sha256": hashlib.sha256(files["batches/0000.json"]).hexdigest()}], "allocation": count_receipt, "h3PixelsRead": False, "pixelsRead": False, "commercialRightsClearanceClaimed": False}
    files["source-lock.json"] = canonical_json(packet)
    if output.exists() or output.is_symlink(): raise FileExistsError("source-lock fixture exists")
    partial = output.with_name(output.name + ".partial"); partial.mkdir(parents=True)
    try:
        for relative, payload in files.items():
            target = partial / relative; target.parent.mkdir(parents=True, exist_ok=True); atomic_write_receipt(target, json.loads(payload.decode("utf-8")))
        _fsync_dir(partial); os.replace(partial, output); _fsync_dir(output.parent)
        return reopen_p6_source_lock(output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True); shutil.rmtree(output, ignore_errors=True); raise


def reopen_p6_source_lock(output: Path) -> dict[str, Any]:
    """Re-open and re-hash every published source-lock artifact."""
    root = output.resolve(strict=True)
    if not root.is_dir() or output.is_symlink():
        raise ValueError("source-lock bundle is not a physical directory")
    lock_path = root / "source-lock.json"
    packet = json.loads(lock_path.read_text("utf-8"))
    status = packet.get("status")
    if lock_path.read_bytes() != canonical_json(packet) or status != "p6-source-lock-fixture":
        raise ValueError("source-lock receipt is not canonical or verified")
    items = packet.get("artifacts", [])
    paths = [item.get("path") for item in items]
    if len(paths) != len(set(paths)) or any(not isinstance(path, str) or PurePosixPath(path).is_absolute() or any(part in ("", ".", "..") for part in PurePosixPath(path).parts) for path in paths):
        raise ValueError("source-lock artifact paths are unsafe or duplicated")
    expected_prefixes = {"train-manifest.json", "selector-manifest.json", "synthetic-acceptance-manifest.json", "real-acceptance-manifest.json", "admission-ledger.json", "reject-ledger.json", "overlap-receipt.json", "allocation-receipt.json"}
    batch_count = 1
    if not expected_prefixes.issubset(set(paths)) or len([path for path in paths if path.startswith("batches/")]) != batch_count:
        raise ValueError("source-lock artifact inventory incomplete")
    for item in items:
        path = root / item["path"]
        if path.resolve(strict=True) != path or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("source-lock artifact bytes changed")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual_files != set(paths) | {"source-lock.json"}:
        raise ValueError("source-lock contains missing or extra files")
    for candidate in root.rglob("*"):
        if candidate.is_symlink() or not stat.S_ISREG(os.lstat(candidate).st_mode) and not stat.S_ISDIR(os.lstat(candidate).st_mode):
            raise ValueError("source-lock contains symlink or unsafe file")
    if False:  # authoritative verification is disabled pending byte re-derivation
        expected_manifest_rows = {"train-manifest.json": 97324, "selector-manifest.json": 4000, "synthetic-acceptance-manifest.json": 100000, "real-acceptance-manifest.json": 10000}
        manifest_map = {item["path"]: item for item in items if item["path"].endswith("-manifest.json")}
        if {path: manifest_map.get(path, {}).get("rows") for path in expected_manifest_rows} != expected_manifest_rows:
            raise ValueError("source-lock cohort manifest counts changed")
        batches = [item for item in items if item["path"].startswith("batches/")]
        if any(item.get("rows") != 100 for item in batches) or sorted(item.get("batch") for item in batches) != list(range(1000)):
            raise ValueError("source-lock batch enumeration changed")
        manifests = []
        for filename in expected_manifest_rows:
            raw = (root / filename).read_bytes(); parsed = json.loads(raw.decode("utf-8"))
            if raw != canonical_json(parsed) or not isinstance(parsed, list):
                raise ValueError("source-lock manifest is not canonical")
            manifests.extend(parsed)
        validate_allocations(packet["allocation"]["counts"], manifests)
        synthetic = json.loads((root / "synthetic-acceptance-manifest.json").read_text("utf-8"))
        reconstructed = []
        for batch in sorted(batches, key=lambda item: item["batch"]):
            batch_rows = json.loads((root / batch["path"]).read_text("utf-8"))
            if len(batch_rows) != 100 or (root / batch["path"]).read_bytes() != canonical_json(batch_rows):
                raise ValueError("source-lock batch bytes changed")
            reconstructed.extend(batch_rows)
        if reconstructed != synthetic:
            raise ValueError("source-lock batches do not concatenate to synthetic manifest")
    return packet


def validate_allocations(allocation: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    if dict(allocation) != ALLOCATIONS:
        raise ValueError("P6 allocation contract changed")
    rows = list(rows); ids = [canonical_identity(r) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("allocation leakage or duplicate identity")
    allowed = set(OMNI_SOURCES) | set(SOURCES)
    if any(r.get("sourceKey") not in allowed for r in rows):
        raise ValueError("allocation source is not pinned")
    counts: dict[str, int] = {}
    for row in rows:
        source = row["sourceKey"]
        role = row.get("role")
        if role not in {"train", "selector", "synthetic-acceptance", "real-acceptance"}:
            raise ValueError("allocation role missing")
        key = f"{role}:{source}"
        counts[key] = counts.get(key, 0) + 1
        if source in SOURCES and row.get("label") != SOURCES[source]["label"]:
            raise ValueError("frontier publisher label mismatch")
    wanted = {
        "train:omni-set-train-real": 48662, "train:omni-set-train-synthetic": 43783,
        "train:aigenimages2026-train": 4879, "selector:omni-set-selector-real": 2000,
        "selector:omni-set-selector-synthetic": 2000, "synthetic-acceptance:omni-set-validation-synthetic": 58228,
        "synthetic-acceptance:omni-ood-test-synthetic": 28693, "synthetic-acceptance:aigenimages2026-test": 559,
        "synthetic-acceptance:taste": 644, "synthetic-acceptance:x-aigd": 2419,
        "synthetic-acceptance:nano-banana": 9457, "real-acceptance:omni-set-validation-real": 5000,
        "real-acceptance:omni-ood-test-real": 5000,
    }
    if counts != wanted:
        raise ValueError("allocation membership/counts are not exact")
    return {"status": "p6-allocation-verified", "counts": ALLOCATIONS, "rowCount": len(rows), "allocationSha256": hashlib.sha256(canonical_json(ALLOCATIONS)).hexdigest()}


def validate_allocation_count_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a score-blind count census without pretending rows were admitted."""
    expected = {"train": {"real": 48662, "synthetic": 48662, "aigenimages2026-train": 4879}, "selector": {"real": 2000, "synthetic": 2000}, "synthetic-acceptance": {"setValidation": 58228, "ood": 28693, "aigenimages2026-test": 559, "taste": 644, "x-aigd": 2419, "nano-banana": 9457}, "real-acceptance": {"setValidationReal": 5000, "oodReddit": 5000}}
    if set(receipt) != {"status", "counts", "rows", "selectionInfluence"} or receipt["status"] != "p6-allocation-count-fixture" or receipt["counts"] != expected or receipt["rows"] != 211324 or receipt["selectionInfluence"] is not False:
        raise ValueError("allocation count receipt is not exact")
    return {"status": "p6-allocation-counts-verified", "counts": expected, "rows": 211324, "selectionInfluence": False}


def source_lock_receipt(*, source_commit: str | None = None, source_tree: str | None = None, artifacts: Mapping[str, str], admission_sha256: str, allocation_sha256: str, overlap_sha256: str | None = None, history_sha256: str | None = None, materialization_sha256: str | None = None, p5_commit: str = "c878c2dc7ecbb49edb1cac4395aa20649471a330") -> dict[str, Any]:
    if source_commit is not None or source_tree is not None:
        if source_commit != SOURCE_LOCK_COMMIT or source_tree != SOURCE_LOCK_TREE:
            raise ValueError("source lock prior authorization parent/tree mismatch")
    if not artifacts or any(not isinstance(k, str) or not isinstance(v, str) or len(v) != 64 or any(c not in HEX64 for c in v) for k, v in artifacts.items()):
        raise ValueError("source lock artifact hashes incomplete")
    for value in (admission_sha256, allocation_sha256, overlap_sha256, history_sha256, materialization_sha256):
        if value is None:
            continue
        if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value):
            raise ValueError("source lock receipt digest invalid")
    if p5_commit != "c878c2dc7ecbb49edb1cac4395aa20649471a330":
        raise ValueError("P5 public commit mismatch")
    return {"schemaVersion": 1, "status": "p6-source-lock", "p5Commit": p5_commit, "actualP6SourceCommit": None, "actualP6SourceTree": None, "priorAuthorizationCommit": SOURCE_LOCK_COMMIT, "priorAuthorizationTree": SOURCE_LOCK_TREE, "artifacts": dict(sorted(artifacts.items())), "admissionLedgerSha256": admission_sha256, "allocationSha256": allocation_sha256, "overlapSha256": overlap_sha256, "historySha256": history_sha256, "materializationSha256": materialization_sha256, "h3PixelsRead": False, "commercialRightsClearanceClaimed": False}


def load_inventory() -> dict[str, Any]:
    raw = JSON_PATH.read_bytes(); value = json.loads(raw.decode("utf-8"))
    if raw != canonical_json(value):
        raise ValueError("inventory JSON is not canonical UTF-8 JSON plus LF")
    validate_inventory_document(value); return value


def validate_inventory_document(value: Mapping[str, Any]) -> None:
    expected = {"schemaVersion", "status", "sourceCommit", "claims", "quotaCorrection", "sources"}
    if set(value) != expected or value["schemaVersion"] != 1 or value["status"] != "metadata-only; no acquisition or materialization" or value["sourceCommit"] != SOURCE_COMMIT:
        raise ValueError("P6 inventory document key/value mismatch")
    if value["claims"] != {"publisherMetadataIsGroundTruthBoundary": True, "commercialRightsClearanceClaimed": False, "h3PixelsRead": False}:
        raise ValueError("claim boundary mismatch")
    if value["quotaCorrection"] != QUOTA_CORRECTION:
        raise ValueError("quota correction changed")
    if set(value["sources"]) != set(SOURCES):
        raise ValueError("source keys changed")
    for key, spec in SOURCES.items():
        if value["sources"][key] != spec:
            raise ValueError(f"{key} metadata mismatch")


validate_taste_assets_metadata = validate_taste_assets
validate_nano_shards = validate_nano_metadata
