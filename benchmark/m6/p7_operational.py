"""P7 operational, byte-rederiving frontier materializer and source lock.

P7 is deliberately source-stage only: it re-opens verified local bytes and
emits receipts; it never trains, scores, or grants rights.
"""
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path
from typing import Any, Iterable, Mapping

from .materialize import image_facts, load_materialized_parts
from .historical import build_history_bundle
from .p6_frontier_inventory import SOURCES, validate_source_cards, validate_taste_member_binding, inventory_digest

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
def build_source_lock(*, source_commit: str | None = None, source_tree: str | None = None, receipts: Mapping[str, Any] | None = None, materialized_root: Path | None = None) -> dict[str, Any]:
    raise RuntimeError("P7 source-lock disabled until all frontier adapters and overlap/allocation gates exist")

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--phase", required=True, choices=("verify-inputs", "materialize-frontier", "source-lock")); parser.add_argument("--source", choices=("taste",)); parser.add_argument("--cache-root", type=Path); parser.add_argument("--materialized-root", type=Path); parser.add_argument("--output", type=Path); parser.add_argument("--source-commit"); parser.add_argument("--source-tree")
    args = parser.parse_args(argv)
    if args.phase == "verify-inputs":
        if args.cache_root is None or args.materialized_root is None or args.output is None: raise SystemExit("cache, materialized root, and output required")
        print(json.dumps(verify_inputs(cache_root=args.cache_root, materialized_root=args.materialized_root, output=args.output), sort_keys=True)); return 0
    if args.phase == "materialize-frontier" and args.source == "taste":
        if args.cache_root is None or args.output is None: raise SystemExit("TASTE cache and output required")
        print(json.dumps(materialize_taste(cache_root=args.cache_root, output=args.output), sort_keys=True)); return 0
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
