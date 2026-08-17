"""P7 operational, byte-rederiving frontier materializer and source lock.

P7 is deliberately source-stage only: it re-opens verified local bytes and
emits receipts; it never trains, scores, or grants rights.
"""
from __future__ import annotations

import argparse, csv, hashlib, io, json, os, shutil, stat, tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from .materialize import image_facts, load_materialized_parts
from .historical import build_history_bundle
from .p6_frontier_inventory import (SOURCES, validate_source_cards, validate_taste_member_binding,
    inventory_digest, validate_aigen_tar_members, validate_aigen_metadata_csv,
    validate_parquet_rows, validate_nano_metadata)

ROOT = Path(__file__).resolve().parents[2]
P7_PARENT = "e6412f90736d39985332ef009a360614865306d2"
P7_PARENT_TREE = "b99608de8093fcdd70d7f3520ebfbce3344dfcb5"
PYTHON_PILLOW = "11.3.0"
FRONTIER_SOURCES = ("aigenimages2026-train", "aigenimages2026-test", "x-aigd", "taste", "nano-banana")
P7_STATUS = "p7-phase1-taste-unverified"
ALLOCATIONS_CORRECTED = {"train": {"real": 48662, "synthetic": {"omni": 43783, "aigen": 4879}}, "selector": {"real": 2000, "synthetic": 2000}, "syntheticPanel": {"set": 58228, "ood": 28693, "frontier": 13079}, "realPanel": {"set": 5000, "ood": 5000}}
TASTE_RECEIPT_KEYS = frozenset({
    "asset_id", "card", "container", "dataset", "decodedRgbSha256", "dhash64",
    "encodedBytes", "encodedBytesSha256", "h3PixelsRead", "height",
    "labelEvidenceScope", "member", "model", "partition", "pillow", "pixelsRead",
    "publisher", "receiptSha256", "revision", "sourceKey", "sourceScope", "track", "width",
})
TASTE_SUMMARY_KEYS = frozenset({
    "expandedSha256", "h3PixelsRead", "inventorySha256", "memberBytes", "pixelsRead",
    "rows", "schemaVersion", "source", "status",
})

def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()

def _hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

def _strict_json_value(raw: bytes, *, label: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {label} key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=unique,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite {token}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label} JSON: {exc}") from exc
    if raw != canonical_json(value):
        raise ValueError(f"{label} is not canonical")
    return value

def verify_member(row: Mapping[str, Any], *, source_scope: str) -> dict[str, Any]:
    """Validate one re-derived row and return a canonical receipt."""
    required = {"card", "container", "member", "encodedBytesSha256", "decodedRgbSha256", "dhash64", "width", "height", "pillow"}
    if not required.issubset(row) or not all(_hex(row[k]) for k in ("encodedBytesSha256", "decodedRgbSha256")):
        raise ValueError("P7 member receipt schema/byte hash mismatch")
    if not isinstance(row["dhash64"], str) or len(row["dhash64"]) != 16 or any(c not in "0123456789abcdef" for c in row["dhash64"]):
        raise ValueError("P7 member dHash mismatch")
    try:
        from PIL import __version__ as pillow_version
    except Exception as exc: raise ValueError("Pillow unavailable") from exc
    if type(row["width"]) is not int or type(row["height"]) is not int or row["width"] <= 0 or row["height"] <= 0 or row["pillow"] != PYTHON_PILLOW or pillow_version != PYTHON_PILLOW:
        raise ValueError("P7 member dimensions/Pillow mismatch")
    if not isinstance(source_scope, str) or not source_scope:
        raise ValueError("P7 source scope missing")
    return {**dict(row), "sourceScope": source_scope, "pixelsRead": True, "h3PixelsRead": False}

def verify_receipts(receipts: Iterable[Mapping[str, Any]], *, expected: int | None = None) -> dict[str, Any]:
    rows = [dict(r) for r in receipts]
    if expected is not None and len(rows) != expected: raise ValueError("P7 receipt shortage")
    keys = [(r.get("card"), r.get("container"), r.get("member")) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError("P7 duplicate member receipt")
    payload = b"".join(canonical_json(r) for r in sorted(rows, key=lambda x: (x["card"], x["container"], x["member"])))
    return {"schemaVersion": 1, "status": "p7-member-receipts-verified", "rows": len(rows), "expandedSha256": hashlib.sha256(payload).hexdigest(), "pixelsRead": True, "h3PixelsRead": False}

def atomic_write_bundle(output: Path, payloads: Mapping[str, bytes], *, failure_hook=None) -> None:
    if output.exists() or output.is_symlink(): raise FileExistsError("P7 output already exists")
    partial = output.with_name(output.name + ".partial")
    if partial.exists() or partial.is_symlink(): raise FileExistsError("stale P7 partial exists")
    partial.mkdir(parents=True); installed = False
    try:
        for name, data in payloads.items():
            target = partial / name
            with target.open("xb") as fh: fh.write(data); fh.flush(); os.fsync(fh.fileno())
        directory_fd = os.open(partial, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
        os.replace(partial, output)
        installed = True
        if failure_hook: failure_hook("renamed")
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try: os.fsync(parent_fd)
        finally: os.close(parent_fd)
    except Exception:
        if installed and output.exists():
            try:
                import shutil; shutil.rmtree(output)
                fd=os.open(output.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)
            except Exception as rollback: raise RuntimeError("P7 output rollback unknown state") from rollback
        if partial.exists():
            import shutil; shutil.rmtree(partial)
        raise

def verify_inputs(*, cache_root: Path, materialized_root: Path, output: Path) -> dict[str, Any]:
    """Phase-1 operational packet: reopen Omni, history, and immutable P6 cards."""
    parts = load_materialized_parts(materialized_root, cache_root=cache_root)
    history = build_history_bundle()
    validate_source_cards()
    packet = {"schemaVersion": 1, "status": "p7-inputs-verified", "parentCommit": P7_PARENT, "parentTree": P7_PARENT_TREE,
              "parts": {k: len(v) for k, v in parts.items()}, "history": history["receipt"],
              "frontierSources": list(FRONTIER_SOURCES), "h3PixelsRead": False, "publisherAssertionOnly": True}
    atomic_write_bundle(output, {"input-verification.json": canonical_json(packet)})
    reopened = json.loads((output / "input-verification.json").read_bytes())
    if reopened != packet: raise ValueError("P7 input packet reopen mismatch")
    return packet

def _materialize_taste_rows(*, root: Path, output: Path, rows: list[Mapping[str, Any]], expected_count: int, enforce_production: bool) -> dict[str, Any]:
    if output.exists() or output.is_symlink(): raise FileExistsError("P7 output already exists")
    if enforce_production and expected_count != 644: raise ValueError("production TASTE requires 644 rows")
    if len(rows) != expected_count: raise ValueError("TASTE row count mismatch")
    if len({r.get("asset_id") for r in rows}) != len(rows): raise ValueError("duplicate TASTE asset id")
    parquet = root / "assets.parquet"; container_hash = hashlib.sha256(parquet.read_bytes()).hexdigest(); receipts = []
    for row in sorted([dict(r) for r in rows], key=lambda r: r["asset_id"]):
        if not isinstance(row.get("memberPath"), str) or not row["memberPath"].startswith("images/") or ".." in Path(row["memberPath"]).parts or Path(row["memberPath"]).is_absolute(): raise ValueError("TASTE member path unsafe")
        member = root / row["memberPath"]
        physical_root = root.resolve(strict=True)
        if member.is_symlink() or not member.is_file() or not member.resolve(strict=True).is_relative_to(physical_root): raise ValueError("TASTE member path unsafe")
        raw = member.read_bytes(); _, decoded, dhash, width, height = image_facts(raw)
        rec = verify_member({"card": SOURCES["taste"]["cardSha256"], "container": container_hash, "member": row["memberPath"], "encodedBytes": len(raw), "encodedBytesSha256": hashlib.sha256(raw).hexdigest(), "decodedRgbSha256": decoded, "dhash64": dhash, "width": width, "height": height, "pillow": PYTHON_PILLOW, "asset_id": row["asset_id"], "model": row["model"], "track": row["track"], "dataset": SOURCES["taste"]["dataset"], "revision": SOURCES["taste"]["revision"], "partition": SOURCES["taste"]["partition"], "publisher": SOURCES["taste"]["publisher"], "sourceKey": "taste", "labelEvidenceScope": SOURCES["taste"]["labelEvidenceScope"]}, source_scope="taste")
        rec["receiptSha256"] = hashlib.sha256(canonical_json(rec)).hexdigest(); receipts.append(rec)
    inv = [{"path": r["member"], "bytes": r["encodedBytes"], "sha256": r["encodedBytesSha256"]} for r in receipts]
    actual_inventory = inventory_digest(inv); member_bytes = sum(row["bytes"] for row in inv)
    if enforce_production and (actual_inventory != SOURCES["taste"]["inventorySha256"] or len(inv) != 644 or member_bytes != SOURCES["taste"]["memberBytes"]): raise ValueError("TASTE inventory changed")
    summary = verify_receipts(receipts, expected=expected_count); summary.update({"source": "taste", "inventorySha256": actual_inventory, "memberBytes": member_bytes}); atomic_write_bundle(output, {"taste-receipts.json": canonical_json(receipts), "taste-summary.json": canonical_json(summary)}); return summary

def reopen_taste(output: Path, *, enforce_production: bool = True) -> dict[str, Any]:
    if output.is_symlink() or not output.is_dir() or output.parent.resolve(strict=True) != output.resolve(strict=True).parent: raise ValueError("TASTE output unsafe")
    names = {p.name for p in output.iterdir()}; expected = {"taste-receipts.json", "taste-summary.json"}
    if names != expected: raise ValueError("TASTE output files changed")
    for name in expected:
        p=output/name
        if p.is_symlink() or not p.is_file(): raise ValueError("TASTE artifact file unsafe")
    receipts = _strict_json_value((output/"taste-receipts.json").read_bytes(), label="TASTE receipts")
    summary = _strict_json_value((output/"taste-summary.json").read_bytes(), label="TASTE summary")
    if not isinstance(receipts, list) or not isinstance(summary, dict) or set(summary) != TASTE_SUMMARY_KEYS: raise ValueError("TASTE output schema changed")
    if summary.get("rows") != len(receipts) or type(summary.get("memberBytes")) is not int or summary["memberBytes"] <= 0: raise ValueError("TASTE receipt count changed")
    if summary.get("schemaVersion") != 1 or summary.get("status") != "p7-member-receipts-verified" or summary.get("source") != "taste" or summary.get("pixelsRead") is not True or summary.get("h3PixelsRead") is not False or not _hex(summary.get("expandedSha256")) or not _hex(summary.get("inventorySha256")): raise ValueError("TASTE summary binding changed")
    if len({r.get("asset_id") for r in receipts if isinstance(r, dict)}) != len(receipts) or len({r.get("member") for r in receipts if isinstance(r, dict)}) != len(receipts): raise ValueError("TASTE identities changed")
    for rec in receipts:
        if not isinstance(rec, dict) or set(rec) != TASTE_RECEIPT_KEYS: raise ValueError("TASTE receipt schema changed")
        if type(rec["asset_id"]) is not int or rec["asset_id"] <= 0 or type(rec["encodedBytes"]) is not int or rec["encodedBytes"] <= 0 or type(rec["width"]) is not int or rec["width"] <= 0 or type(rec["height"]) is not int or rec["height"] <= 0: raise ValueError("TASTE receipt numeric fields changed")
        if rec["card"] != SOURCES["taste"]["cardSha256"] or not _hex(rec["container"]) or (enforce_production and rec["container"] != SOURCES["taste"]["assetsParquetSha256"]) or rec["dataset"] != SOURCES["taste"]["dataset"] or rec["revision"] != SOURCES["taste"]["revision"] or rec["partition"] != SOURCES["taste"]["partition"] or rec["publisher"] != SOURCES["taste"]["publisher"] or rec["sourceKey"] != "taste" or rec["sourceScope"] != "taste" or rec["labelEvidenceScope"] != SOURCES["taste"]["labelEvidenceScope"] or rec["pillow"] != PYTHON_PILLOW or rec["pixelsRead"] is not True or rec["h3PixelsRead"] is not False: raise ValueError("TASTE receipt immutable binding changed")
        if rec["model"] not in SOURCES["taste"]["modelCounts"] or rec["track"] not in {"descriptions", "aesthetics"} or not isinstance(rec["member"], str) or not rec["member"].startswith("images/") or not _hex(rec["encodedBytesSha256"]) or not _hex(rec["decodedRgbSha256"]) or not isinstance(rec["dhash64"], str) or len(rec["dhash64"]) != 16 or any(char not in "0123456789abcdef" for char in rec["dhash64"]): raise ValueError("TASTE receipt value changed")
        copy=dict(rec); claimed = copy.pop("receiptSha256", None)
        if not _hex(claimed) or claimed != hashlib.sha256(canonical_json(copy)).hexdigest(): raise ValueError("TASTE receipt digest changed")
    inventory = [{"path": row["member"], "bytes": row["encodedBytes"], "sha256": row["encodedBytesSha256"]} for row in receipts]
    actual_inventory = inventory_digest(inventory); member_bytes = sum(row["bytes"] for row in inventory)
    expected_summary = verify_receipts(receipts, expected=len(receipts)); expected_summary.update({"source": "taste", "inventorySha256": actual_inventory, "memberBytes": member_bytes})
    if summary != expected_summary: raise ValueError("TASTE summary digest changed")
    if enforce_production and (len(receipts) != SOURCES["taste"]["memberCount"] or member_bytes != SOURCES["taste"]["memberBytes"] or actual_inventory != SOURCES["taste"]["inventorySha256"] or {row["asset_id"] for row in receipts} != set(range(1, 645))): raise ValueError("TASTE production inventory changed")
    return summary

def materialize_taste(*, cache_root: Path, output: Path, rows: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if output.exists() or output.is_symlink(): raise FileExistsError("P7 output already exists")
    validate_source_cards(); root = cache_root.resolve(strict=True); parquet = root / "assets.parquet"
    if parquet.is_symlink() or not parquet.is_file() or parquet.resolve(strict=True) != parquet: raise ValueError("TASTE parquet missing/unsafe")
    if parquet.stat().st_size != SOURCES["taste"]["assetsParquetBytes"] or hashlib.sha256(parquet.read_bytes()).hexdigest() != SOURCES["taste"]["assetsParquetSha256"]: raise ValueError("TASTE assets parquet bytes changed")
    if rows is None:
        import pyarrow.parquet as pq
        rows = pq.read_table(parquet).to_pylist()
    rows = [dict(r) for r in rows]; bound = [{**r, "memberPath": r.get("memberPath", r.get("image_path"))} for r in rows]
    validate_taste_member_binding(bound, [r["memberPath"] for r in bound]); result = _materialize_taste_rows(root=root, output=output, rows=bound, expected_count=644, enforce_production=True)
    try: reopen_taste(output, enforce_production=True)
    except Exception:
        import shutil
        try:
            shutil.rmtree(output); fd = os.open(output.parent, os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except Exception as rollback: raise RuntimeError("P7 output rollback unknown state") from rollback
        raise
    return result

# P8 frontier adapters.  Their public entry points accept only a cache and an
# output.  Test data enters through private helpers below: production never
# accepts caller-provided rows, metadata, or image bytes.
P8_MAX_IMAGE_BYTES = 512 * 1024 * 1024
X_AIGD_GENERATOR_COUNTS = {"SD_3.5L-raw":200,"SD_1.5_realistic_vision-raw":199,"FLUX.1_dev-raw":198,"Infinity-raw":198,"SD_3-raw":194,"SD_3.5L-iPhonePhoto-raw":194,"PA_alpha-raw":193,"SDXL_realism_engine-raw":190,"HYDiT_1.2-raw":181,"Lumina_Next-raw":173,"SD_1.4-raw":161,"SDXL_1.0-raw":144,"FLUX.1_dev-iPhonePhoto-raw":194}
P8_RECEIPT_COMMON = frozenset({"sourceKey","dataset","revision","partition","publisher","card","container","member","labelEvidenceScope","encodedBytes","encodedBytesSha256","decodedRgbSha256","decodedFormat","dhash64","width","height","pillow","pixelsRead","h3PixelsRead","receiptSha256"})
P8_SUMMARY_KEYS = frozenset({"schemaVersion","status","source","rows","expandedSha256","inventorySha256","memberBytes","pixelsRead","h3PixelsRead"})
def _safe_member(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("frontier member path unsafe")
    candidate = root / relative
    base = root.resolve(strict=True)
    if candidate.is_symlink() or not candidate.is_file() or not candidate.resolve(strict=True).is_relative_to(base):
        raise ValueError("frontier member path unsafe")
    return candidate

def _physical_file(root: Path, relative: str) -> Path:
    candidate = _safe_member(root, relative)
    if candidate.resolve(strict=True) != candidate.absolute():
        raise ValueError("frontier container must be physical")
    return candidate

def _sha256_stream(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def _source_file(root: Path, source_key: str) -> Path:
    spec=SOURCES[source_key]; path=_physical_file(root, spec["containerPath"])
    if path.stat().st_size != spec["containerBytes"] or _sha256_stream(path) != spec["containerSha256"]:
        raise ValueError(f"{source_key} pinned container changed")
    return path

def _receipt(source_key: str, *, container: str, member: str, raw: bytes, identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw)>P8_MAX_IMAGE_BYTES: raise ValueError("frontier encoded image is unsafe")
    spec=SOURCES[source_key]; _, decoded, dhash, width, height=image_facts(raw)
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image: decoded_format=image.format
    if not isinstance(decoded_format,str): raise ValueError("frontier decoded format missing")
    record={"sourceKey":source_key,"dataset":spec["dataset"],"revision":spec["revision"],"partition":spec["partition"],"publisher":spec["publisher"],"card":spec["cardSha256"],"container":container,"member":member,"labelEvidenceScope":spec["labelEvidenceScope"],"encodedBytes":len(raw),"encodedBytesSha256":hashlib.sha256(raw).hexdigest(),"decodedRgbSha256":decoded,"decodedFormat":decoded_format,"dhash64":dhash,"width":width,"height":height,"pillow":PYTHON_PILLOW,"pixelsRead":True,"h3PixelsRead":False,**dict(identity)}
    record["receiptSha256"]=hashlib.sha256(canonical_json(record)).hexdigest()
    return record

def _x_member(generator: str, uid: str) -> str:
    if not isinstance(generator,str) or not generator or not isinstance(uid,str) or not uid:
        raise ValueError("X generator/uid type changed")
    return "row/" + hashlib.sha256(canonical_json({"generator":generator,"uid":uid})).hexdigest()

def _fsync_parent(path: Path) -> None:
    fd=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

def _frontier_bundle(source_key: str, receipts: list[Mapping[str, Any]], *, output: Path, expected: int | None, production: bool, extras: Mapping[str, Any] | None = None, failure_hook=None) -> dict[str, Any]:
    spec = SOURCES[source_key]
    if output.exists() or output.is_symlink(): raise FileExistsError("P8 output already exists")
    receipts=[dict(r) for r in receipts]
    receipts.sort(key=lambda r: r["member"].encode("utf-8"))
    if expected is not None and len(receipts) != expected: raise ValueError("frontier row count mismatch")
    if len({r["member"] for r in receipts}) != len(receipts): raise ValueError("frontier duplicate member")
    inv = [{"path": r["member"], "bytes": r["encodedBytes"], "sha256": r["encodedBytesSha256"]} for r in receipts]
    digest = inventory_digest(inv); member_bytes=sum(r["bytes"] for r in inv)
    if production and (len(receipts) != spec.get("memberCount", spec.get("rows")) or
                       (spec.get("memberBytes") is not None and member_bytes != spec["memberBytes"]) or
                       (spec.get("inventorySha256") is not None and digest != spec["inventorySha256"])):
        raise ValueError("frontier production inventory changed")
    extra=dict(extras or {})
    if source_key.startswith("aigenimages2026"):
        extra.setdefault("quarantine",[]); extra.setdefault("quarantineCount",0)
        extra["quarantineSha256"]=hashlib.sha256(canonical_json(extra["quarantine"])).hexdigest()
    summary=verify_receipts(receipts, expected=expected); summary.update({"source": source_key, "inventorySha256": digest, "memberBytes": member_bytes, **extra})
    payloads={f"{source_key}-receipts.json": canonical_json(receipts), f"{source_key}-summary.json": canonical_json(summary)}
    if source_key.startswith("aigenimages2026"): payloads[f"{source_key}-quarantine.json"]=canonical_json(extra["quarantine"])
    atomic_write_bundle(output, payloads)
    try:
        if failure_hook: failure_hook("after-publish")
        reopened=_reopen_frontier(output, source_key, production=production)
        if reopened != summary: raise ValueError("frontier reopen result changed")
    except Exception:
        try: shutil.rmtree(output); _fsync_parent(output)
        except Exception as rollback: raise RuntimeError("P8 output rollback unknown state") from rollback
        raise
    return summary

def _reopen_frontier(output: Path, source_key: str, *, production: bool = False) -> dict[str, Any]:
    if output.is_symlink() or not output.is_dir(): raise ValueError("frontier output unsafe")
    names={p.name for p in output.iterdir()}; prefix=f"{source_key}-"; expected={prefix+"receipts.json",prefix+"summary.json"}
    if source_key.startswith("aigenimages2026"): expected.add(prefix+"quarantine.json")
    if names != expected: raise ValueError("frontier output file set changed")
    receipts=_strict_json_value((output/(prefix+"receipts.json")).read_bytes(),label="frontier receipts")
    summary=_strict_json_value((output/(prefix+"summary.json")).read_bytes(),label="frontier summary")
    if any((output/name).is_symlink() or not (output/name).is_file() for name in expected): raise ValueError("frontier artifact unsafe")
    if not isinstance(receipts,list) or not isinstance(summary,dict): raise ValueError("frontier summary changed")
    allowed=P8_SUMMARY_KEYS | ({"containerInventory","containerInventorySha256","containerCount","containerBytes"} if source_key=="nano-banana" else set()) | ({"quarantine","quarantineCount","quarantineSha256"} if source_key.startswith("aigenimages2026") else set())
    if set(summary)!=allowed: raise ValueError("frontier summary schema changed")
    identity={"aigenimages2026-train":{"image_id","filename","quarantined"},"aigenimages2026-test":{"image_id","filename","quarantined"},"x-aigd":{"generator","uid","nativePath","declaredWidth","declaredHeight","declaredFormat"},"nano-banana":{"id","shard","row_index","nativePath","declaredWidth","declaredHeight","declaredFormat","declaredMode"}}[source_key]
    spec=SOURCES[source_key]
    identities=set()
    for rec in receipts:
        if not isinstance(rec,dict) or set(rec) != P8_RECEIPT_COMMON | identity or not _hex(rec.get("receiptSha256")): raise ValueError("frontier receipt schema changed")
        if rec["sourceKey"] != source_key or rec["dataset"] != spec["dataset"] or rec["revision"] != spec["revision"] or rec["partition"] != spec["partition"] or rec["publisher"] != spec["publisher"] or rec["card"] != spec["cardSha256"] or rec["labelEvidenceScope"] != spec["labelEvidenceScope"] or rec["pixelsRead"] is not True or rec["h3PixelsRead"] is not False or not _hex(rec["container"]) or not _hex(rec["encodedBytesSha256"]) or not _hex(rec["decodedRgbSha256"]) or type(rec["encodedBytes"]) is not int or rec["encodedBytes"] <= 0 or type(rec["width"]) is not int or rec["width"] <= 0 or type(rec["height"]) is not int or rec["height"] <= 0 or rec["pillow"] != PYTHON_PILLOW: raise ValueError("frontier immutable binding changed")
        if production and source_key in {"aigenimages2026-train","aigenimages2026-test","x-aigd"} and rec["container"] != spec["containerSha256"]: raise ValueError("frontier production container binding changed")
        body=dict(rec); claimed=body.pop("receiptSha256");
        if claimed != hashlib.sha256(canonical_json(body)).hexdigest(): raise ValueError("frontier receipt selfhash changed")
        if not isinstance(rec["member"],str) or rec["member"].startswith("/") or ".." in PurePosixPath(rec["member"]).parts or not isinstance(rec["dhash64"],str) or len(rec["dhash64"]) != 16 or any(c not in "0123456789abcdef" for c in rec["dhash64"]): raise ValueError("frontier member/dhash grammar changed")
        key=(rec.get("image_id"),rec.get("filename")) if source_key.startswith("aigen") else (rec.get("generator"),rec.get("uid")) if source_key=="x-aigd" else (rec.get("id"),rec.get("shard"),rec.get("row_index"))
        if key in identities: raise ValueError("frontier duplicate source identity")
        identities.add(key)
        if source_key=="x-aigd" and (not isinstance(rec["generator"],str) or not rec["generator"] or not isinstance(rec["uid"],str) or not rec["uid"] or rec["member"] != _x_member(rec["generator"],rec["uid"])): raise ValueError("X identity/member binding changed")
        if source_key=="x-aigd" and (not isinstance(rec["nativePath"],str) or not rec["nativePath"] or rec["nativePath"].startswith("/") or any(part in (".","..") for part in PurePosixPath(rec["nativePath"]).parts) or any(ord(c)<32 for c in rec["nativePath"]) or type(rec["declaredWidth"]) is not int or type(rec["declaredHeight"]) is not int or rec["declaredWidth"] != rec["width"] or rec["declaredHeight"] != rec["height"] or not isinstance(rec["declaredFormat"],str) or not rec["declaredFormat"] or rec["decodedFormat"] != "PNG"): raise ValueError("X declared image facts changed")
        if source_key=="nano-banana" and (type(rec["id"]) is not int or type(rec["row_index"]) is not int or not isinstance(rec["shard"],str) or rec["member"] != f"{rec['shard']}#{rec['row_index']}"): raise ValueError("Nano identity/member binding changed")
        if source_key=="nano-banana" and (not isinstance(rec["nativePath"],str) or not rec["nativePath"] or rec["nativePath"].startswith("/") or any(part in (".","..") for part in PurePosixPath(rec["nativePath"]).parts) or any(ord(c)<32 for c in rec["nativePath"]) or rec["declaredWidth"] != rec["width"] or rec["declaredHeight"] != rec["height"] or rec["declaredFormat"] != rec["decodedFormat"] or rec["declaredMode"] != "RGB"): raise ValueError("Nano declared image facts changed")
        if source_key.startswith("aigenimages2026"):
            suffix=PurePosixPath(rec["filename"]).suffix.lower() if isinstance(rec.get("filename"),str) else ""
            expected_format={".png":"PNG",".jpg":"JPEG",".jpeg":"JPEG",".webp":"WEBP"}.get(suffix)
            if not isinstance(rec["image_id"],str) or not isinstance(rec["filename"],str) or PurePosixPath(rec["filename"]).name != rec["filename"] or rec["quarantined"] is not False or not expected_format or not isinstance(rec["decodedFormat"],str) or not rec["decodedFormat"] or rec["decodedFormat"] != expected_format or rec["member"] != f"mnt/cephfs/home/common/datasets/online_sid_dataset/AIGenImages/AIGenImages2026/{spec['partition']}/1_fake/{rec['filename']}": raise ValueError("AIGen identity/member binding changed")
    inventory=[{"path":r["member"],"bytes":r["encodedBytes"],"sha256":r["encodedBytesSha256"]} for r in receipts]
    recomputed=verify_receipts(receipts,expected=len(receipts)); recomputed.update({"source":source_key,"inventorySha256":inventory_digest(inventory),"memberBytes":sum(r["bytes"] for r in inventory)})
    if source_key=="nano-banana":
        containers={r["shard"]:{"path":r["shard"],"bytes":0,"sha256":r["container"]} for r in receipts}
        # Byte sizes are bound when published; a zero/forged inventory is rejected.
        listed=summary["containerInventory"]
        if not isinstance(listed,list) or len(listed)!=summary["containerCount"] or any(set(x)!={"path","bytes","sha256"} or type(x["bytes"]) is not int or x["bytes"]<=0 for x in listed): raise ValueError("Nano container inventory changed")
        by_path={x["path"]:x for x in listed}
        if inventory_digest(listed)!=summary["containerInventorySha256"] or sum(x["bytes"] for x in listed)!=summary["containerBytes"] or set(by_path)!={r["shard"] for r in receipts} or any(by_path[r["shard"]]["sha256"]!=r["container"] for r in receipts): raise ValueError("Nano container binding changed")
        recomputed.update({"containerInventory":listed,"containerInventorySha256":inventory_digest(listed),"containerCount":len(listed),"containerBytes":sum(x["bytes"] for x in listed)})
    if source_key.startswith("aigenimages2026"):
        quarantine=_strict_json_value((output/(prefix+"quarantine.json")).read_bytes(),label="AIGen quarantine")
        if not isinstance(quarantine,list) or summary["quarantine"]!=quarantine or summary["quarantineCount"]!=len(quarantine) or summary["quarantineSha256"]!=hashlib.sha256(canonical_json(quarantine)).hexdigest(): raise ValueError("AIGen quarantine changed")
        if source_key.endswith("test") and quarantine: raise ValueError("AIGen validation quarantine changed")
        if source_key.endswith("train") and production and (len(quarantine)!=1 or set(quarantine[0])!={"schemaVersion","sourceKey","container","filename","image_id","reason","publisherRows","admittedRows","pixelsRead","quarantineSha256"} or quarantine[0]["filename"]!="image_midjourneyv7_300.png" or quarantine[0]["reason"]!="publisher-metadata-missing-member" or quarantine[0]["publisherRows"]!=4880 or quarantine[0]["admittedRows"]!=4879 or quarantine[0]["pixelsRead"] is not False or quarantine[0]["sourceKey"]!=source_key or quarantine[0]["container"]!=SOURCES[source_key]["containerSha256"] or not isinstance(quarantine[0]["image_id"],str) or quarantine[0]["quarantineSha256"]!=hashlib.sha256(canonical_json({k:v for k,v in quarantine[0].items() if k!="quarantineSha256"})).hexdigest()): raise ValueError("AIGen train quarantine binding changed")
        recomputed.update({"quarantine":quarantine,"quarantineCount":len(quarantine),"quarantineSha256":hashlib.sha256(canonical_json(quarantine)).hexdigest()})
    if summary != recomputed: raise ValueError("frontier summary recomputation changed")
    if production and len(receipts)!=SOURCES[source_key]["rows"]: raise ValueError("frontier production census changed")
    if source_key=="nano-banana" and production and (summary["containerCount"]!=31 or summary["containerBytes"]!=SOURCES[source_key]["memberBytes"] or summary["containerInventorySha256"]!=SOURCES[source_key]["inventorySha256"]): raise ValueError("Nano production container inventory changed")
    return summary

def _aigen_fixture(*, root: Path, output: Path, partition: str, metadata_rows: list[Mapping[str, Any]], image_rows: Mapping[str, bytes]) -> dict[str, Any]:
    """Private test seam; production cannot pass rows or bytes."""
    source_key="aigenimages2026-train" if partition=="train" else "aigenimages2026-test"
    admitted={r["filename"]:r for r in metadata_rows}
    receipts=[]
    for member,raw in sorted(image_rows.items()):
        if member not in admitted: raise ValueError("fixture AIGen metadata/member mismatch")
        receipts.append(_receipt(source_key,container="f"*64,member=f"mnt/cephfs/home/common/datasets/online_sid_dataset/AIGenImages/AIGenImages2026/{partition}/1_fake/{member}",raw=raw,identity={"image_id":str(admitted[member]["image_id"]),"filename":member,"quarantined":False}))
    return _frontier_bundle(source_key,receipts,output=output,expected=len(receipts),production=False)

def _scan_aigen_tar(tar_path: Path, partition: str, *, expected_rows: int, production: bool = False) -> tuple[list[dict[str,Any]],list[str]]:
    if partition not in {"train","val"}: raise ValueError("AIGen partition invalid")
    wanted_root=PurePosixPath("mnt/cephfs/home/common/datasets/online_sid_dataset/AIGenImages/AIGenImages2026")
    candidates=[]; members=[]; seen_names=set(); csv_count=0
    with tarfile.open(tar_path,"r|*") as arc:
      for info in arc:
        name=PurePosixPath(info.name)
        if info.name in seen_names or any(ord(c)<32 for c in info.name) or name.is_absolute() or any(x in ("", ".", "..") for x in name.parts) or not name.is_relative_to(wanted_root): raise ValueError("unsafe AIGen TAR member")
        seen_names.add(info.name)
        if info.isdir(): continue
        if not info.isfile() or info.issym() or info.islnk() or info.size>P8_MAX_IMAGE_BYTES: raise ValueError("unsafe AIGen TAR type")
        rel=name.relative_to(wanted_root)
        if name.suffix.lower()==".csv":
          csv_count+=1
          payload=arc.extractfile(info).read(P8_MAX_IMAGE_BYTES+1)
          if len(payload)>P8_MAX_IMAGE_BYTES: raise ValueError("AIGen metadata too large")
          reader=csv.DictReader(io.TextIOWrapper(io.BytesIO(payload),encoding="utf-8",newline="")); rows=list(reader)
          if reader.fieldnames==["image_id","filename","caption","caption_id","split"] and rows and all(r.get("split")==partition for r in rows): candidates.append(rows)
        elif len(rel.parts)>=3 and rel.parts[0]==partition and rel.parts[1]=="1_fake" and name.suffix.lower() in {".png",".jpg",".jpeg",".webp"}: members.append(info.name)
    if csv_count!=3 or len(candidates)!=1 or len(members)!=expected_rows: raise ValueError("AIGen synthetic metadata discovery changed")
    metadata=candidates[0]
    if production: validate_aigen_metadata_csv(metadata,members,partition=partition)
    return metadata,sorted(members,key=lambda item:item.encode())

def materialize_aigen(*, cache_root: Path, output: Path, partition: str) -> dict[str, Any]:
    source_key = "aigenimages2026-train" if partition == "train" else "aigenimages2026-test" if partition == "val" else ""
    if not source_key: raise ValueError("AIGen partition invalid")
    root=cache_root.resolve(strict=True); tar_path=_source_file(root,source_key)
    metadata,members=_scan_aigen_tar(tar_path,partition,expected_rows=SOURCES[source_key]["rows"],production=True)
    by_name={PurePosixPath(r["filename"]).name:r for r in metadata}; receipts=[]
    with tarfile.open(tar_path,"r|*") as arc:
      for info in arc:
        if info.name not in set(members): continue
        raw=arc.extractfile(info).read(P8_MAX_IMAGE_BYTES+1)
        if len(raw)>P8_MAX_IMAGE_BYTES: raise ValueError("AIGen image too large")
        meta=by_name[PurePosixPath(info.name).name]
        receipts.append(_receipt(source_key,container=SOURCES[source_key]["containerSha256"],member=info.name,raw=raw,identity={"image_id":meta["image_id"],"filename":meta["filename"],"quarantined":False}))
    quarantine=[]
    if partition=="train":
        body={"schemaVersion":1,"sourceKey":source_key,"container":SOURCES[source_key]["containerSha256"],"filename":"image_midjourneyv7_300.png","image_id":by_name["image_midjourneyv7_300.png"]["image_id"],"reason":"publisher-metadata-missing-member","publisherRows":4880,"admittedRows":4879,"pixelsRead":False}
        body["quarantineSha256"]=hashlib.sha256(canonical_json(body)).hexdigest(); quarantine=[body]
    result=_frontier_bundle(source_key,receipts,output=output,expected=SOURCES[source_key]["rows"],production=False,extras={"quarantine":quarantine,"quarantineCount":len(quarantine)})
    _reopen_frontier(output,source_key,production=True)
    return result

def _iter_parquet_rows(path: Path, source_key: str):
    try:
      import pyarrow.parquet as pq
    except Exception as exc: raise RuntimeError("pyarrow is required for local frontier parquet") from exc
    expected={"x-aigd":[("image","struct<bytes: binary, path: string>"),("generator","string"),("uid","string"),("labels","list<element: struct<label: string, points: list<element: list<element: double>>>>"),("original_prompt","string"),("positive_prompt","string"),("negative_prompt","string"),("guidance_scale","double"),("num_inference_steps","int64"),("scheduler","string"),("seed","int64"),("width","int64"),("height","int64"),("image_format","string"),("jpeg_quality","int64"),("chroma_subsampling","string")],"nano-banana":[("id","int64"),("image","struct<bytes: binary, path: string>"),("format","string"),("mode","string"),("width","int32"),("height","int32"),("uploadtime","string")]}[source_key]
    parquet=pq.ParquetFile(path)
    schema=parquet.schema_arrow
    if [(field.name,str(field.type)) for field in schema] != expected or any(field.nullable is not True for field in schema): raise ValueError("frontier Arrow schema/order/nullability changed")
    for batch in parquet.iter_batches(batch_size=64):
      for row in batch.to_pylist(): yield dict(row)

def _x_fixture(*, output: Path, rows: list[Mapping[str,Any]]) -> dict[str,Any]:
    required={"image","generator","uid","labels","original_prompt","positive_prompt","negative_prompt","guidance_scale","num_inference_steps","scheduler","seed","width","height","image_format","jpeg_quality","chroma_subsampling"}
    if not rows or any(set(r)!=required or not isinstance(r["image"],Mapping) or set(r["image"])!={"bytes","path"} for r in rows) or len({(r["generator"],r["uid"]) for r in rows})!=len(rows): raise ValueError("X fixture schema/identity changed")
    return _frontier_bundle("x-aigd",[_receipt("x-aigd",container="f"*64,member=_x_member(r["generator"],r["uid"]),raw=bytes(r["image"]["bytes"]),identity={"generator":r["generator"],"uid":r["uid"],"nativePath":r["image"]["path"],"declaredWidth":r["width"],"declaredHeight":r["height"],"declaredFormat":r["image_format"]}) for r in rows],output=output,expected=len(rows),production=False)

def materialize_x_aigd(*, cache_root: Path, output: Path) -> dict[str,Any]:
    root=cache_root.resolve(strict=True); path=_source_file(root,"x-aigd"); receipts=[]; uids=set(); pairs=set(); groups={}; count=0
    required={"image","generator","uid","labels","original_prompt","positive_prompt","negative_prompt","guidance_scale","num_inference_steps","scheduler","seed","width","height","image_format","jpeg_quality","chroma_subsampling"}
    for r in _iter_parquet_rows(path,"x-aigd"):
        if set(r)!=required or not isinstance(r["image"],Mapping) or set(r["image"])!={"bytes","path"} or not isinstance(r["image"]["bytes"],bytes) or not isinstance(r["image"]["path"],str) or type(r["uid"]) is not str or type(r["width"]) is not int or type(r["height"]) is not int: raise ValueError("X native row schema changed")
        pair=(r["generator"],r["uid"])
        if pair in pairs: raise ValueError("X duplicate generator/uid")
        pairs.add(pair); uids.add(r["uid"]); groups[r["generator"]]=groups.get(r["generator"],0)+1; count+=1
        receipts.append(_receipt("x-aigd",container=SOURCES["x-aigd"]["containerSha256"],member=_x_member(r["generator"],r["uid"]),raw=r["image"]["bytes"],identity={"generator":r["generator"],"uid":r["uid"],"nativePath":r["image"]["path"],"declaredWidth":r["width"],"declaredHeight":r["height"],"declaredFormat":r["image_format"]}))
    if count!=2419 or len(uids)!=1290 or groups!=X_AIGD_GENERATOR_COUNTS: raise ValueError("X pinned census changed")
    result=_frontier_bundle("x-aigd",receipts,output=output,expected=SOURCES["x-aigd"]["rows"],production=False)
    _reopen_frontier(output,"x-aigd",production=True)
    return result

def _nano_fixture(*, output: Path, rows: list[Mapping[str,Any]], shards: list[str]) -> dict[str,Any]:
    if not shards or len(set(shards))!=len(shards): raise ValueError("Nano fixture shards changed")
    required={"id","image","format","mode","width","height","uploadtime"}
    if not rows or any(set(r)!=required or not isinstance(r["image"],Mapping) or set(r["image"])!={"bytes","path"} or r["format"]!="PNG" or r["mode"]!="RGB" for r in rows) or len({r["id"] for r in rows})!=len(rows): raise ValueError("Nano fixture schema/id changed")
    inventory=[{"path":shard,"bytes":1,"sha256":"f"*64} for shard in shards]; local={shard:0 for shard in shards}; receipts=[]
    for i,r in enumerate(rows):
      shard=shards[i%len(shards)]; index=local[shard]; local[shard]+=1
      receipts.append(_receipt("nano-banana",container="f"*64,member=f"{shard}#{index}",raw=bytes(r["image"]["bytes"]),identity={"id":r["id"],"shard":shard,"row_index":index,"nativePath":r["image"]["path"],"declaredWidth":r["width"],"declaredHeight":r["height"],"declaredFormat":r["format"],"declaredMode":r["mode"]}))
    return _frontier_bundle("nano-banana",receipts,output=output,expected=len(rows),production=False,extras={"containerInventory":inventory,"containerInventorySha256":inventory_digest(inventory),"containerCount":len(inventory),"containerBytes":len(inventory)})

def materialize_nano(*, cache_root: Path, output: Path) -> dict[str,Any]:
    root=cache_root.resolve(strict=True); data=root/"data"
    if data.is_symlink() or not data.is_dir(): raise ValueError("Nano data directory unsafe")
    shards=sorted((p.relative_to(root).as_posix() for p in data.glob("*.parquet")),key=lambda x:x.encode())
    if len(shards)!=31 or len(set(shards))!=31: raise ValueError("Nano requires exactly 31 lexical parquet shards")
    inv=[]; receipts=[]; all_ids=set()
    for shard in shards:
      path=_physical_file(root,shard); inv.append({"path":shard,"bytes":path.stat().st_size,"sha256":_sha256_stream(path)})
      required={"id","image","format","mode","width","height","uploadtime"}; shard_count=0
      for i,r in enumerate(_iter_parquet_rows(path,"nano-banana")):
        if set(r)!=required or r["id"] in all_ids or r["format"]!="PNG" or r["mode"]!="RGB" or not isinstance(r["image"],Mapping) or set(r["image"])!={"bytes","path"} or not isinstance(r["image"]["bytes"],bytes) or not isinstance(r["image"]["path"],str) or type(r["width"]) is not int or type(r["height"]) is not int: raise ValueError("Nano id/image schema changed")
        all_ids.add(r["id"]); receipts.append(_receipt("nano-banana",container=inv[-1]["sha256"],member=f"{shard}#{i}",raw=bytes(r["image"]["bytes"]),identity={"id":r["id"],"shard":shard,"row_index":i,"nativePath":r["image"]["path"],"declaredWidth":r["width"],"declaredHeight":r["height"],"declaredFormat":r["format"],"declaredMode":r["mode"]}))
        shard_count+=1
    if len(receipts)!=9457 or all_ids!=set(range(9457)) or sum(x["bytes"] for x in inv)!=SOURCES["nano-banana"]["memberBytes"] or inventory_digest(inv)!=SOURCES["nano-banana"]["inventorySha256"]: raise ValueError("Nano pinned inventory/census changed")
    result=_frontier_bundle("nano-banana",receipts,output=output,expected=9457,production=False,extras={"containerInventory":inv,"containerInventorySha256":inventory_digest(inv),"containerCount":len(inv),"containerBytes":sum(x["bytes"] for x in inv)})
    _reopen_frontier(output,"nano-banana",production=True)
    return result
def build_source_lock(*, source_commit: str | None = None, source_tree: str | None = None, receipts: Mapping[str, Any] | None = None, materialized_root: Path | None = None) -> dict[str, Any]:
    raise RuntimeError("P7 source-lock disabled until all frontier adapters and overlap/allocation gates exist")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", required=True, choices=("verify-inputs", "materialize-frontier", "source-lock")); parser.add_argument("--source", choices=("taste", "aigenimages2026-train", "aigenimages2026-test", "x-aigd", "nano-banana")); parser.add_argument("--cache-root", type=Path); parser.add_argument("--materialized-root", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--source-commit"); parser.add_argument("--source-tree")
    args = parser.parse_args(argv)
    if args.phase == "verify-inputs":
        if args.cache_root is None or args.materialized_root is None or args.output is None: raise SystemExit("cache, materialized root, and output required")
        print(json.dumps(verify_inputs(cache_root=args.cache_root, materialized_root=args.materialized_root, output=args.output), sort_keys=True)); return 0
    if args.phase == "materialize-frontier" and args.source == "taste":
        if args.cache_root is None or args.output is None: raise SystemExit("TASTE cache and output required")
        print(json.dumps(materialize_taste(cache_root=args.cache_root, output=args.output), sort_keys=True)); return 0
    if args.phase == "materialize-frontier" and args.source in {"aigenimages2026-train", "aigenimages2026-test", "x-aigd", "nano-banana"}:
        if args.cache_root is None or args.output is None: raise SystemExit("frontier cache and output required")
        if args.source.startswith("aigenimages2026"):
            result=materialize_aigen(cache_root=args.cache_root,output=args.output,partition="train" if args.source.endswith("train") else "val")
        elif args.source == "x-aigd": result=materialize_x_aigd(cache_root=args.cache_root,output=args.output)
        else: result=materialize_nano(cache_root=args.cache_root,output=args.output)
        print(json.dumps(result,sort_keys=True)); return 0
    if args.phase == "materialize-frontier":
        if args.materialized_root is None: raise SystemExit("materialized root required")
        parts = load_materialized_parts(args.materialized_root, cache_root=args.cache_root)
        raise SystemExit("P7 frontier adapters are fail-closed pending exact local container/member adapters; Omni re-opened but no frontier receipt was published")
    if args.output is None: raise SystemExit("output required")
    if args.materialized_root is None: raise SystemExit("P7 source-lock requires a verified materialized root and complete frontier receipts")
    raise SystemExit("P7 source-lock disabled until four frontier adapters emit complete receipts; no source-lock claim is published")
    atomic_write_bundle(args.output, {"source-lock.json": canonical_json(lock)})
    return 0

if __name__ == "__main__": raise SystemExit(main())
