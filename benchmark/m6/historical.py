"""Authoritative metadata-only M6 history normalizer.

This module re-opens the fixed committed H3/M2-M5 artifacts, verifies their
raw bytes, and emits the exact comparator rows consumed by the M6 source lock.
It never opens a historical image path.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any

from benchmark.m6.contracts import ROOT, canonical_json, parse_json_bytes
from benchmark.m6.prepare import (
    canonical_gzip,
    canonical_history_row,
    selection_rank,
    validate_history_bundle,
)


HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
HISTORY_SPECS = {
    "h3": (
        ("benchmark/evidence/m3/h3-met-holdout-manifest.jsonl", 989390, "50574778ab0d58f839f1dccc3c99da5f6dca98150186f13aeca8d9ba052e9547", 600, "jsonl"),
    ),
    "m2": (
        ("benchmark/evidence/m2/train-manifest.jsonl.gz", 17371060, "fc84b9646d9ee91ff5dd40e7a79b38226fa4e83418dfd5fce974aef7daf533d8", 105978, "jsonl-gzip"),
        ("benchmark/evidence/m2/validation-manifest.jsonl", 569680, "a63953148040e1a4223f16fa04ebf4b85c4022da65531ead0b25ce46434eab93", 900, "jsonl"),
    ),
    "m3": (
        ("benchmark/evidence/m3/train-manifest.jsonl.gz", 18247686, "87bcd95c92582760ec444acdd467be4b824251a5c4763260790717bbcbc2ced3", 108378, "jsonl-gzip"),
        ("benchmark/evidence/m3/validation-manifest.jsonl", 831945, "86b7b9119fc7a118ee8b0a85e1a6b7dce6635154b3eff49cc1197b9d40fca2ff", 600, "jsonl"),
    ),
    "m4": (
        ("benchmark/evidence/m4/train-manifest.jsonl.gz", 18906386, "e5dfc79869541ae5c6703b60de250930f1fb8247790f55b67bb1805f5ac73a93", 112562, "jsonl-gzip"),
        ("benchmark/evidence/m4/validation-manifest.jsonl", 666194, "643eb365a603309b94b112403ef4250b565b9863d2ec61a5cc48aa80d5f85caa", 600, "jsonl"),
    ),
    "m5": (
        ("benchmark/evidence/m4/train-manifest.jsonl.gz", 18906386, "e5dfc79869541ae5c6703b60de250930f1fb8247790f55b67bb1805f5ac73a93", 112562, "jsonl-gzip"),
        ("benchmark/evidence/m4/validation-manifest.jsonl", 666194, "643eb365a603309b94b112403ef4250b565b9863d2ec61a5cc48aa80d5f85caa", 600, "jsonl"),
        ("benchmark/evidence/m5/failed-training-attempt-1.json", 95589, "c7577db83ecf7ba3e988a1923edb396d349bbb31d9d3e01746db07e4fa3fb0bf", 0, "m5-failure"),
    ),
}


def _read_fixed(path_text: str, expected_bytes: int, expected_sha256: str) -> bytes:
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or any(piece in ("", ".", "..") for piece in relative.parts):
        raise ValueError("historical source path is unsafe")
    path = ROOT.joinpath(*relative.parts)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"historical source is not a regular file: {path_text}")
    raw = path.read_bytes()
    if len(raw) != expected_bytes or sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"historical source bytes changed: {path_text}")
    return raw


def _jsonl_rows(raw: bytes, *, compressed: bool, label: str) -> list[dict[str, Any]]:
    try:
        expanded = gzip.decompress(raw) if compressed else raw
    except (OSError, EOFError) as exc:
        raise ValueError(f"historical {label} gzip is invalid") from exc
    if not expanded.endswith(b"\n"):
        raise ValueError(f"historical {label} must end with LF")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(expanded.splitlines()):
        if not line:
            raise ValueError(f"historical {label} contains an empty row")
        rows.append(parse_json_bytes(line, label=f"{label} row {index}"))
    return rows


def _source_text(row: dict[str, Any], key: str, *, path: bool = False) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"historical source {key} invalid")
    if path:
        relative = PurePosixPath(value)
        if relative.is_absolute() or any(piece in ("", ".", "..") for piece in relative.parts):
            raise ValueError("historical source image path invalid")
    return value


def normalize_source_row(row: dict[str, Any], cohort: str) -> dict[str, Any]:
    image_sha = _source_text(row, "imageSha256")
    dhash = row.get("perceptualDhash64")
    if not HEX64.fullmatch(image_sha):
        raise ValueError("historical source image SHA-256 invalid")
    if dhash is not None and (not isinstance(dhash, str) or not HEX16.fullmatch(dhash)):
        raise ValueError("historical source dHash64 invalid")
    return canonical_history_row({
        "cohort": cohort,
        "dataset": _source_text(row, "dataset"),
        "decodedRgbSha256": None,
        "dhash64": dhash,
        "encodedBytesSha256": image_sha,
        "filename": _source_text(row, "path", path=True),
        "revision": _source_text(row, "datasetRevision"),
        "rowId": _source_text(row, "id"),
        "sourceGroup": _source_text(row, "source"),
    })


def _validate_m5_failure(raw: bytes) -> None:
    value = parse_json_bytes(raw, label="M5 terminal selector failure")
    if (
        value.get("schemaVersion") != 1
        or value.get("status") != "failed-m5-selector"
        or value.get("acceptanceEligible") is not False
        or value.get("h3PixelsRead") is not False
        or value.get("terminalRegressionsRead") is not False
    ):
        raise ValueError("M5 terminal failure boundary changed")
    summary = value.get("trainingSummary")
    if not isinstance(summary, dict) or (
        summary.get("trainingItems") != 112562
        or summary.get("selectorItems") != 600
        or summary.get("trainingManifestSha256") != HISTORY_SPECS["m4"][0][2]
        or summary.get("selectorManifestSha256") != HISTORY_SPECS["m4"][1][2]
        or summary.get("h3PixelsRead") is not False
    ):
        raise ValueError("M5 historical manifest binding changed")


def build_history_bundle() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for cohort in ("h3", "m2", "m3", "m4", "m5"):
        cohort_rows = 0
        for path, expected_bytes, expected_sha, expected_rows, kind in HISTORY_SPECS[cohort]:
            raw = _read_fixed(path, expected_bytes, expected_sha)
            if kind == "m5-failure":
                _validate_m5_failure(raw)
                source_rows: list[dict[str, Any]] = []
            else:
                source_rows = _jsonl_rows(raw, compressed=(kind == "jsonl-gzip"), label=path)
                if len(source_rows) != expected_rows:
                    raise ValueError(f"historical source row count changed: {path}")
                rows.extend(normalize_source_row(row, cohort) for row in source_rows)
            cohort_rows += len(source_rows)
            manifests.append({
                "bytes": expected_bytes,
                "cohort": cohort,
                "path": path,
                "rows": expected_rows,
                "sha256": expected_sha,
            })
        counts[cohort] = cohort_rows
    rows.sort(key=lambda row: (row["cohort"], selection_rank(row)))
    expanded = b"".join(canonical_json(row) for row in rows)
    receipt = {
        "cohortRowCounts": counts,
        "h3PixelsRead": False,
        "manifests": manifests,
        "normalizedExpandedSha256": sha256(expanded).hexdigest(),
        "normalizedRows": len(rows),
        "schemaVersion": 1,
        "status": "m6-historical-metadata-locked",
    }
    bundle = {"receipt": receipt, "rows": rows}
    validate_history_bundle(bundle, production=True)
    return bundle


def write_history_bundle(output: Path, *, failure_hook: Any = None) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("M6 historical output must be absent")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("stale M6 historical partial exists")
    bundle = build_history_bundle()
    expanded = b"".join(canonical_json(row) for row in bundle["rows"])
    payloads = {
        "historical-comparators.jsonl.gz": canonical_gzip(expanded),
        "historical-metadata-receipt.json": canonical_json(bundle["receipt"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(mode=0o700)
    installed = False
    try:
        for name, payload in payloads.items():
            with (temporary / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.replace(temporary, output)
        installed = True
        if failure_hook: failure_hook("renamed")
        parent = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        if failure_hook: failure_hook("parent-fsynced")
    except Exception:
        if installed:
            try:
                shutil.rmtree(output)
                if output.exists() or output.is_symlink():
                    raise OSError("installed M6 historical output could not be removed")
                if failure_hook:
                    failure_hook("rollback-fsync")
                parent = os.open(output.parent, os.O_RDONLY)
                try:
                    os.fsync(parent)
                finally:
                    os.close(parent)
            except Exception as rollback_error:
                raise RuntimeError("M6 historical publication state unknown after rollback failure") from rollback_error
        else:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "h3PixelsRead": False,
        "normalizedRows": len(bundle["rows"]),
        "status": "m6-historical-metadata-locked",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(canonical_json(write_history_bundle(args.output)).decode(), end="")


if __name__ == "__main__":
    main()
