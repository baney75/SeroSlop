"""Dependency-light deterministic primitives for the M3 source contract."""

from __future__ import annotations

import gzip
from hashlib import sha256
from urllib.parse import urlsplit, urlunsplit


FORBIDDEN_RESULT_FIELDS = frozenset({"displayScore", "logit", "prediction", "rawProbability"})
FORBIDDEN_MATERIALIZATION_FIELDS = frozenset({"pixelBytes", "imageTransport", "detailTransport"})


def priority(namespace: str, value: str | int) -> str:
    return sha256(f"{namespace}{value}".encode()).hexdigest()


def deterministic_gzip(value: bytes) -> bytes:
    output = bytearray(gzip.compress(value, compresslevel=9, mtime=0))
    if len(output) < 10:
        raise ValueError("gzip output is truncated")
    output[9] = 0xFF
    return bytes(output)


def canonical_source_url(value: str, *, allowed_host: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname != allowed_host or not parsed.path:
        raise ValueError(f"Unapproved source URL: {value}")
    if parsed.username or parsed.password or parsed.port or parsed.query or parsed.fragment:
        raise ValueError(f"Source URL is not canonical: {value}")
    return urlunsplit(("https", allowed_host, parsed.path, "", ""))


def source_group(namespace: str, canonical_url: str) -> str:
    return f"{namespace}{sha256(canonical_url.encode()).hexdigest()}"


def assert_unique_evidence_rows(
    rows: list[dict[str, object]],
    *,
    label: str,
    fallback_dhashes: dict[str, str] | None = None,
) -> None:
    ids = [str(row["id"]) for row in rows]
    hashes = [str(row["imageSha256"]) for row in rows]
    if len(set(ids)) != len(rows) or len(set(hashes)) != len(rows):
        raise ValueError(f"{label} contains duplicate IDs or image bytes")
    for index, row in enumerate(rows):
        if int(row.get("rowIndex", index)) != index:
            raise ValueError(f"{label} row indices changed")
        if FORBIDDEN_RESULT_FIELDS.intersection(row):
            raise ValueError(f"{label} contains model-result fields")
        if FORBIDDEN_MATERIALIZATION_FIELDS.intersection(row):
            raise ValueError(f"{label} contains non-public materialization fields")
        value = str(
            row["perceptualDhash64"]
            if "perceptualDhash64" in row
            else (fallback_dhashes or {}).get(str(row["id"]), "")
        )
        if len(value) != 16:
            raise ValueError(f"{label} contains an invalid dHash")
        int(value, 16)


def assert_pixel_facts(
    row: dict[str, object],
    *,
    width: int,
    height: int,
    perceptual_dhash64: str,
    label: str,
) -> None:
    has_width = "width" in row
    has_height = "height" in row
    if has_width != has_height:
        raise ValueError(f"{label} has an incomplete dimension contract")
    if has_width and (int(row["width"]) != width or int(row["height"]) != height):
        raise ValueError(f"{label} pixel dimensions changed")
    if str(row["perceptualDhash64"]) != perceptual_dhash64:
        raise ValueError(f"{label} perceptual hash changed")


def exact_partition(
    candidates: list[dict[str, object]],
    *,
    namespace: str,
    target: int,
    excluded_ids: set[str],
) -> list[dict[str, object]]:
    """Choose a score-blind exact partition from already-qualified fixtures."""

    selected: list[dict[str, object]] = []
    for original in sorted(
        candidates,
        key=lambda row: (priority(namespace, str(row["id"])), str(row["id"])),
    ):
        identifier = str(original["id"])
        if identifier in excluded_ids:
            continue
        row = dict(original)
        row["selectionPriority"] = priority(namespace, identifier)
        selected.append(row)
        excluded_ids.add(identifier)
        if len(selected) == target:
            return selected
    raise ValueError(f"Frozen partition target is unavailable: {target}")
