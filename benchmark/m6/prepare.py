"""Deterministic, metadata-only M6 P/S source preparation.

P validates the frozen source census.  The source-lock builder consumes image
facts produced by a later materializer; it never opens pixels, scores a model,
or reads H3 payloads.  Production quotas are not caller-configurable.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.m6.contracts import GENERATORS, canonical_json, load_recipe, parse_json_bytes


SET_DATASET = "JamalLee/Omni-Fake-SET"
SET_REVISION = "724e97f5fc9f4b89f59631a8d4e6331712b7d441"
OOD_DATASET = "JamalLee/Omni-Fake-OOD"
OOD_REVISION = "9ed7e38bbdb4aeb2eb553896a5890680a9ffcf17"
REAL_SOURCE_PREFIXES = (
    "COCO", "FFHQ", "OpenForensics_real", "OpenImages", "WIDER",
    "celebA", "flickr30k-images",
)
FRESH_FIELDS = {
    "dataset", "revision", "partition", "rowId", "filename", "sourceGroup",
    "label", "encodedBytesSha256", "decodedRgbSha256", "dhash64",
}
HISTORY_FIELDS = {
    "cohort", "dataset", "revision", "rowId", "filename", "sourceGroup",
    "encodedBytesSha256", "decodedRgbSha256", "dhash64",
}
PARTITIONS = {
    "set_train": (SET_DATASET, SET_REVISION, "train"),
    "set_validation": (SET_DATASET, SET_REVISION, "validation"),
    "ood_test": (OOD_DATASET, OOD_REVISION, "test"),
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
SOURCE_CLEAN_NAMESPACE = b"m6-source-clean-v1\0"
M6_PROTOCOL_COMMIT = "3b29ea2f9e1ad46e4cd78f47c9ccf5fe3a99877e"
REQUIRED_HISTORY_COHORTS = ("h3", "m2", "m3", "m4", "m5")
HISTORY_RECEIPT_FIELDS = {
    "schemaVersion", "status", "h3PixelsRead", "cohortRowCounts", "manifests",
    "normalizedRows", "normalizedExpandedSha256",
}


def census(rows: list[dict[str, Any]], expected: dict[str, int] | None = None) -> dict[str, Any]:
    """Validate an injected SET-validation census without fetching pixels."""
    recipe = load_recipe()
    expected = expected or recipe["sources"]["omniFakeSet"]["imageSplits"]["validation"]
    if not isinstance(rows, list):
        raise ValueError("census rows must be a list")
    if any(not isinstance(row, dict) or "rowId" not in row or "label" not in row for row in rows):
        raise ValueError("schema missing rowId/label")
    labels = {label: sum(row.get("label") == label for row in rows)
              for label in ("real", "full_synthetic", "tampered")}
    wanted = {key: expected[key] for key in labels}
    if labels != wanted:
        raise ValueError("split label census changed")
    return {"rows": len(rows), "labels": labels}


def _validate_text(value: Any, *, field: str, path: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"source-lock {field} invalid")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"source-lock {field} contains control characters")
    if path:
        pieces = value.split("/")
        if value.startswith("/") or "\\" in value or any(piece in ("", ".", "..") for piece in pieces):
            raise ValueError(f"source-lock {field} path invalid")
        if len(PurePosixPath(value).parts) != len(pieces):
            raise ValueError(f"source-lock {field} path normalization changed")
    return value


def canonical_fresh_row(row: dict[str, Any], part: str) -> dict[str, str]:
    if not isinstance(row, dict) or set(row) != FRESH_FIELDS:
        raise ValueError("source-lock fresh row schema changed")
    dataset, revision, partition = PARTITIONS[part]
    if row["dataset"] != dataset or row["revision"] != revision or row["partition"] != partition:
        raise ValueError("source-lock split dataset/revision/partition mismatch")
    out = {key: _validate_text(row[key], field=key, path=(key == "filename")) for key in FRESH_FIELDS}
    if out["label"] not in {"real", "full_synthetic", "tampered"}:
        raise ValueError("source-lock label invalid")
    if not HEX64.fullmatch(out["encodedBytesSha256"]) or not HEX64.fullmatch(out["decodedRgbSha256"]):
        raise ValueError("source-lock SHA-256 evidence invalid")
    if not HEX16.fullmatch(out["dhash64"]):
        raise ValueError("source-lock dHash64 evidence invalid")
    return out


def canonical_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """Validate normalized historical metadata; decoded RGB is unavailable."""
    if not isinstance(row, dict) or set(row) != HISTORY_FIELDS:
        raise ValueError("source-lock historical row schema changed")
    out = dict(row)
    for key in ("cohort", "dataset", "revision", "rowId", "sourceGroup"):
        out[key] = _validate_text(out[key], field=f"historical {key}")
    out["filename"] = _validate_text(out["filename"], field="historical filename", path=True)
    if not isinstance(out["encodedBytesSha256"], str) or not HEX64.fullmatch(out["encodedBytesSha256"]):
        raise ValueError("historical encoded SHA-256 invalid")
    if out["decodedRgbSha256"] is not None:
        raise ValueError("historical decoded RGB SHA must be explicitly unavailable")
    if not isinstance(out["dhash64"], str) or not HEX16.fullmatch(out["dhash64"]):
        raise ValueError("historical dHash64 invalid")
    return out


def validate_history_bundle(bundle: dict[str, Any], *, production: bool) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    if not isinstance(bundle, dict) or set(bundle) != {"receipt", "rows"}:
        raise ValueError("historical metadata bundle must contain exact receipt and rows")
    if not isinstance(bundle["rows"], list) or (production and not bundle["rows"]):
        raise ValueError("historical metadata rows are required")
    rows = [canonical_history_row(row) for row in bundle["rows"]]
    rows.sort(key=lambda row: (row["cohort"], selection_rank(row)))
    expanded = _jsonl(rows)
    receipt = bundle["receipt"]
    if not isinstance(receipt, dict) or set(receipt) != HISTORY_RECEIPT_FIELDS:
        raise ValueError("historical metadata receipt schema changed")
    expected_status = "m6-historical-metadata-locked" if production else "m6-historical-metadata-fixture"
    if (
        type(receipt["schemaVersion"]) is not int
        or receipt["schemaVersion"] != 1
        or receipt["status"] != expected_status
        or receipt["h3PixelsRead"] is not False
    ):
        raise ValueError("historical metadata receipt boundary changed")
    cohort_names = REQUIRED_HISTORY_COHORTS if production else tuple(sorted({row["cohort"] for row in rows}))
    if production and any(row["cohort"] not in REQUIRED_HISTORY_COHORTS for row in rows):
        raise ValueError("unknown production historical cohort")
    counts = {cohort: sum(row["cohort"] == cohort for row in rows) for cohort in cohort_names}
    if production:
        if not isinstance(receipt["cohortRowCounts"], dict) or set(receipt["cohortRowCounts"]) != set(REQUIRED_HISTORY_COHORTS):
            raise ValueError("required historical cohorts changed")
        if any(type(receipt["cohortRowCounts"][cohort]) is not int or receipt["cohortRowCounts"][cohort] < 0 for cohort in REQUIRED_HISTORY_COHORTS):
            raise ValueError("historical cohort count invalid")
        if any(receipt["cohortRowCounts"][cohort] == 0 for cohort in REQUIRED_HISTORY_COHORTS):
            raise ValueError("required historical cohort is empty")
        if counts != receipt["cohortRowCounts"]:
            raise ValueError("historical cohort counts changed")
    elif counts != receipt["cohortRowCounts"]:
        raise ValueError("fixture historical cohort counts changed")
    if (
        type(receipt["normalizedRows"]) is not int
        or receipt["normalizedRows"] != len(rows)
        or not isinstance(receipt["normalizedExpandedSha256"], str)
        or receipt["normalizedExpandedSha256"] != sha256(expanded).hexdigest()
    ):
        raise ValueError("historical normalized rows/hash changed")
    if not isinstance(receipt["manifests"], list) or (production and not receipt["manifests"]):
        raise ValueError("historical manifest inventory missing")
    manifest_keys = {"cohort", "path", "bytes", "sha256", "rows"}
    seen_paths: set[str] = set()
    manifest_rows = {cohort: 0 for cohort in receipt["cohortRowCounts"]}
    for manifest in receipt["manifests"]:
        if not isinstance(manifest, dict) or set(manifest) != manifest_keys:
            raise ValueError("historical manifest receipt schema changed")
        if manifest["cohort"] not in receipt["cohortRowCounts"]:
            raise ValueError("historical manifest cohort changed")
        _validate_text(manifest["path"], field="historical manifest path", path=True)
        if manifest["path"] in seen_paths:
            raise ValueError("duplicate historical manifest path")
        seen_paths.add(manifest["path"])
        if type(manifest["bytes"]) is not int or manifest["bytes"] < 0 or type(manifest["rows"]) is not int or manifest["rows"] < 0:
            raise ValueError("historical manifest size/count invalid")
        if manifest["rows"] > 0 and manifest["bytes"] == 0:
            raise ValueError("nonempty historical manifest cannot have zero bytes")
        if not isinstance(manifest["sha256"], str) or not HEX64.fullmatch(manifest["sha256"]):
            raise ValueError("historical manifest SHA-256 invalid")
        manifest_rows[manifest["cohort"]] += manifest["rows"]
    if manifest_rows != receipt["cohortRowCounts"] or sum(manifest_rows.values()) != receipt["normalizedRows"]:
        raise ValueError("historical manifest inventory does not reconcile")
    return rows, receipt, expanded


def fixture_history_bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Construct a strict test-only history receipt from normalized rows."""
    normalized = [canonical_history_row(row) for row in rows]
    normalized.sort(key=lambda row: (row["cohort"], selection_rank(row)))
    expanded = _jsonl(normalized)
    counts = {cohort: sum(row["cohort"] == cohort for row in normalized)
              for cohort in sorted({row["cohort"] for row in normalized})}
    manifests = []
    for cohort in sorted(counts):
        cohort_payload = _jsonl([row for row in normalized if row["cohort"] == cohort])
        manifests.append({
            "bytes": len(cohort_payload),
            "cohort": cohort,
            "path": f"fixture/{cohort}.jsonl",
            "rows": counts[cohort],
            "sha256": sha256(cohort_payload).hexdigest(),
        })
    return {
        "receipt": {
            "cohortRowCounts": counts,
            "h3PixelsRead": False,
            "manifests": manifests,
            "normalizedExpandedSha256": sha256(expanded).hexdigest(),
            "normalizedRows": len(normalized),
            "schemaVersion": 1,
            "status": "m6-historical-metadata-fixture",
        },
        "rows": normalized,
    }


def census_all(parts: dict[str, list[dict[str, Any]]], expected: dict[str, Any] | None = None) -> dict[str, int]:
    """Validate the full pinned SET train/validation and OOD test census."""
    if set(parts) != set(PARTITIONS):
        raise ValueError("required split census missing")
    if expected is None:
        recipe = load_recipe()
        set_splits = recipe["sources"]["omniFakeSet"]["imageSplits"]
        ood_split = recipe["sources"]["omniFakeOOD"]["imageSplits"]["test"]
        expected = {
            "set_train": set_splits["train"],
            "set_validation": set_splits["validation"],
            "ood_test": ood_split,
        }
    if set(expected) != set(PARTITIONS):
        raise ValueError("frozen census expectations incomplete")
    output: dict[str, int] = {}
    for part in PARTITIONS:
        rows = [canonical_fresh_row(row, part) for row in parts[part]]
        row_ids = [row["rowId"] for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError(f"duplicate row identity within {part}")
        counts = {label: sum(row["label"] == label for row in rows)
                  for label in ("real", "full_synthetic", "tampered")}
        wanted = {label: expected[part][label] for label in counts}
        if counts != wanted or len(rows) != expected[part]["rows"]:
            raise ValueError(f"label/row census changed: {part}")
        output[part] = len(rows)
    return output


def hamming_neighbors(value: int, bits: int) -> Iterable[int]:
    yield value
    for first in range(bits):
        yield value ^ (1 << first)
    for first in range(bits):
        for second in range(first + 1, bits):
            yield value ^ (1 << first) ^ (1 << second)


class DHashIndex:
    """Threshold-complete 22/21/21-bit multi-index for dHash distance <= 8.

    With three blocks, every pair at total distance at most eight has at least
    one block at distance at most two. Wider blocks reduce candidate bucket
    occupancy at the frozen corpus size while exact popcount remains decisive.
    """

    BLOCKS = ((0, 22), (22, 21), (43, 21))
    PROBES_PER_QUERY = sum(1 + bits + (bits * (bits - 1)) // 2 for _, bits in BLOCKS)

    def __init__(self) -> None:
        self._buckets: list[dict[int, list[tuple[int, int]]]] = [
            {} for _ in self.BLOCKS
        ]
        self._values: list[int] = []

    def add(self, value: int) -> int:
        if not isinstance(value, int) or value < 0 or value >= 1 << 64:
            raise ValueError("dHash index value invalid")
        owner = len(self._values)
        self._values.append(value)
        for block, (shift, bits) in enumerate(self.BLOCKS):
            key = (value >> shift) & ((1 << bits) - 1)
            self._buckets[block].setdefault(key, []).append((owner, value))
        return owner

    def matches(self, value: int, distance: int = 8) -> list[tuple[int, int]]:
        if not isinstance(distance, int) or distance < 0 or distance > 8:
            raise ValueError("dHash index threshold must be between 0 and 8")
        candidates: dict[int, int] = {}
        for block, (shift, bits) in enumerate(self.BLOCKS):
            buckets = self._buckets[block]
            block_value = (value >> shift) & ((1 << bits) - 1)
            for neighbor in hamming_neighbors(block_value, bits):
                for owner, candidate in buckets.get(neighbor, []):
                    candidates[owner] = candidate
        return sorted(
            (owner, (value ^ candidate).bit_count())
            for owner, candidate in candidates.items()
            if (value ^ candidate).bit_count() <= distance
        )

    def __len__(self) -> int:
        return len(self._values)


def stable_identity(row: dict[str, Any]) -> tuple[str, ...]:
    """Partition is intentionally excluded so source-row reuse cannot hide."""
    return tuple(str(row[key]) for key in ("dataset", "revision", "rowId", "filename", "sourceGroup"))


def identity_digest(row: dict[str, Any]) -> str:
    return sha256(canonical_json(list(stable_identity(row)))).hexdigest()


def selection_rank(row: dict[str, Any]) -> str:
    return sha256(SOURCE_CLEAN_NAMESPACE + canonical_json(row)).hexdigest()


def clean_source_rows(
    parts: dict[str, list[dict[str, Any]]],
    historical: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, Any]]]:
    """Apply history-first exact overlap filtering and return a complete audit."""
    if set(parts) != set(PARTITIONS):
        raise ValueError("source-lock partitions changed")
    owners: list[dict[str, Any]] = []
    identity_owners: dict[tuple[str, ...], int] = {}
    encoded_owners: dict[str, int] = {}
    decoded_owners: dict[str, int] = {}
    perceptual = DHashIndex()
    rejects: list[dict[str, Any]] = []

    def admit(row: dict[str, Any], cohort: str, artifact: str, row_index: int) -> bool:
        identity = stable_identity(row)
        comparator: int | None = identity_owners.get(identity)
        layer = "canonical-identity" if comparator is not None else None
        if comparator is None:
            comparator = encoded_owners.get(str(row["encodedBytesSha256"]))
            layer = "encoded-bytes-sha256" if comparator is not None else None
        decoded = row.get("decodedRgbSha256")
        if comparator is None and decoded is not None:
            comparator = decoded_owners.get(str(decoded))
            layer = "decoded-rgb-sha256" if comparator is not None else None
        distance: int | None = None
        if comparator is None:
            matches = perceptual.matches(int(str(row["dhash64"]), 16), 8)
            if matches:
                comparator, distance = min(
                    matches,
                    key=lambda match: (match[1], owners[match[0]]["identitySha256"]),
                )
                layer = "dhash-hamming<=8"
        if comparator is not None:
            owner = owners[comparator]
            rejects.append({
                "candidateArtifact": artifact,
                "candidateCohort": cohort,
                "candidateIdentitySha256": identity_digest(row),
                "candidateRowIndex": row_index,
                "comparatorArtifact": owner["artifact"],
                "comparatorCohort": owner["cohort"],
                "comparatorIdentitySha256": owner["identitySha256"],
                "comparatorRowIndex": owner["rowIndex"],
                "dhashDistance": distance,
                "layer": layer,
            })
            return False
        owner_index = len(owners)
        owner = {
            "artifact": artifact,
            "cohort": cohort,
            "identitySha256": identity_digest(row),
            "rowIndex": row_index,
            "row": row,
        }
        owners.append(owner)
        identity_owners[identity] = owner_index
        encoded_owners[str(row["encodedBytesSha256"])] = owner_index
        if decoded is not None:
            decoded_owners[str(decoded)] = owner_index
        index_owner = perceptual.add(int(str(row["dhash64"]), 16))
        if index_owner != owner_index:
            raise AssertionError("dHash owner index changed")
        return True

    normalized_history = [canonical_history_row(row) for row in (historical or [])]
    for row_index, row in enumerate(sorted(normalized_history, key=lambda item: (item["cohort"], selection_rank(item)))):
        admit(row, str(row["cohort"]), "historical-comparators.jsonl.gz", row_index)

    clean: dict[str, list[dict[str, str]]] = {part: [] for part in PARTITIONS}
    artifact_names = {
        "set_train": "fresh-set-train.jsonl.gz",
        "set_validation": "fresh-set-validation.jsonl.gz",
        "ood_test": "fresh-ood-test.jsonl.gz",
    }
    for part in ("set_train", "set_validation", "ood_test"):
        rows = [canonical_fresh_row(row, part) for row in parts[part]]
        for row_index, row in enumerate(sorted(rows, key=selection_rank)):
            if admit(row, part, artifact_names[part], row_index):
                clean[part].append(row)
    rejects.sort(key=lambda row: (row["candidateCohort"], row["candidateIdentitySha256"], row["layer"]))
    return clean, rejects


def round_robin(rows: list[dict[str, Any]], quota: int) -> list[dict[str, Any]]:
    if not isinstance(quota, int) or quota < 0:
        raise ValueError("source-lock round-robin quota invalid")
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda item: (item["sourceGroup"], selection_rank(item))):
        buckets.setdefault(str(row["sourceGroup"]), []).append(row)
    output: list[dict[str, Any]] = []
    while len(output) < quota:
        progressed = False
        for source in sorted(buckets):
            if buckets[source]:
                output.append(buckets[source].pop(0))
                progressed = True
            if len(output) == quota:
                break
        if not progressed:
            break
    if len(output) != quota:
        raise ValueError("source-lock round-robin quota unavailable")
    return output


def _manifest_rows(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [{**row, "role": role, "roleIndex": index} for index, row in enumerate(rows)]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) for row in rows)


def canonical_gzip(payload: bytes) -> bytes:
    """Create a fixed-header gzip stream instead of using platform fast paths."""
    output = BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as handle:
        handle.write(payload)
    return output.getvalue()


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def source_lock(
    parts: dict[str, list[dict[str, Any]]],
    output: Path,
    *,
    history_bundle: dict[str, Any],
    protocol_commit: str,
) -> dict[str, Any]:
    """Build the production metadata source lock with frozen quotas only."""
    if protocol_commit != M6_PROTOCOL_COMMIT:
        raise ValueError("M6 protocol commit changed")
    recipe = load_recipe()
    expected = {
        "set_train": recipe["sources"]["omniFakeSet"]["imageSplits"]["train"],
        "set_validation": recipe["sources"]["omniFakeSet"]["imageSplits"]["validation"],
        "ood_test": recipe["sources"]["omniFakeOOD"]["imageSplits"]["test"],
    }
    census_all(parts, expected)
    history_rows, history_receipt, history_expanded = validate_history_bundle(history_bundle, production=True)
    return _build_source_lock(
        parts, output, historical=history_rows, history_receipt=history_receipt,
        history_expanded=history_expanded, protocol_commit=protocol_commit,
        minimum_train=40_000,
        selector_each=2_000, set_per_batch=70, ood_per_batch=30,
        batch_count=1_000, production=True,
    )


def _source_lock_fixture(
    parts: dict[str, list[dict[str, Any]]],
    output: Path,
    *,
    historical: list[dict[str, Any]] | None = None,
    minimum_train: int = 1,
    selector_each: int = 1,
    set_per_batch: int = 1,
    ood_per_batch: int = 1,
    batch_count: int = 1,
    fail_after_artifacts: int | None = None,
    fail_stage: str | None = None,
) -> dict[str, Any]:
    """Test-only builder with explicit miniature quotas and failure injection."""
    bundle = fixture_history_bundle(historical or [])
    history_rows, history_receipt, history_expanded = validate_history_bundle(bundle, production=False)
    return _build_source_lock(
        parts, output, historical=history_rows, history_receipt=history_receipt,
        history_expanded=history_expanded, protocol_commit=M6_PROTOCOL_COMMIT,
        minimum_train=minimum_train,
        selector_each=selector_each, set_per_batch=set_per_batch,
        ood_per_batch=ood_per_batch, batch_count=batch_count,
        production=False, fail_after_artifacts=fail_after_artifacts,
        fail_stage=fail_stage,
    )


def _build_source_lock(
    parts: dict[str, list[dict[str, Any]]],
    output: Path,
    *,
    historical: list[dict[str, Any]],
    history_receipt: dict[str, Any],
    history_expanded: bytes,
    protocol_commit: str,
    minimum_train: int,
    selector_each: int,
    set_per_batch: int,
    ood_per_batch: int,
    batch_count: int,
    production: bool,
    fail_after_artifacts: int | None = None,
    fail_stage: str | None = None,
) -> dict[str, Any]:
    if any(not isinstance(value, int) or value <= 0 for value in (
        minimum_train, selector_each, set_per_batch, ood_per_batch, batch_count,
    )):
        raise ValueError("source-lock quota invalid")
    if fail_stage not in (None, "before-rename", "rename", "after-rename", "rollback-fsync"):
        raise ValueError("source-lock failure stage invalid")
    clean, rejects = clean_source_rows(parts, historical)
    train_real = [row for row in clean["set_train"] if row["label"] == "real"]
    train_synthetic = [row for row in clean["set_train"] if row["label"] == "full_synthetic"]
    per_class = min(len(train_real), len(train_synthetic))
    if per_class < minimum_train:
        raise ValueError("source-lock training minimum unavailable")
    train = round_robin(train_real, per_class) + round_robin(train_synthetic, per_class)

    validation_real = [row for row in clean["set_validation"] if row["label"] == "real"]
    validation_synthetic = [row for row in clean["set_validation"] if row["label"] == "full_synthetic"]
    selector_real = round_robin(validation_real, selector_each)
    selector_synthetic = round_robin(validation_synthetic, selector_each)
    selector = selector_real + selector_synthetic
    if production:
        if set(row["sourceGroup"] for row in selector_synthetic) != set(GENERATORS):
            raise ValueError("production selector generator coverage changed")
        if set(row["sourceGroup"] for row in selector_real) != set(REAL_SOURCE_PREFIXES):
            raise ValueError("production selector real-prefix coverage changed")

    selector_ids = {stable_identity(row) for row in selector}
    set_pool = [row for row in validation_synthetic if stable_identity(row) not in selector_ids]
    set_evaluation = round_robin(set_pool, set_per_batch * batch_count)
    ood_evaluation = round_robin(
        [row for row in clean["ood_test"] if row["label"] == "full_synthetic"],
        ood_per_batch * batch_count,
    )
    evaluation: list[dict[str, Any]] = []
    batch_receipts: list[dict[str, Any]] = []
    set_cursor = 0
    ood_cursor = 0
    for batch in range(batch_count):
        batch_rows = (
            set_evaluation[set_cursor:set_cursor + set_per_batch]
            + ood_evaluation[ood_cursor:ood_cursor + ood_per_batch]
        )
        set_cursor += set_per_batch
        ood_cursor += ood_per_batch
        first = len(evaluation)
        evaluation.extend(batch_rows)
        batch_receipts.append({
            "batch": batch,
            "firstEvalIndex": first,
            "items": len(batch_rows),
            "lastEvalIndex": first + len(batch_rows) - 1,
            "ood": ood_per_batch,
            "setValidation": set_per_batch,
        })

    train_rows = _manifest_rows(train, "train")
    selector_rows = _manifest_rows(selector, "selector")
    evaluation_rows = [
        {**row, "batch": index // (set_per_batch + ood_per_batch),
         "evalIndex": index, "role": "evaluation", "roleIndex": index}
        for index, row in enumerate(evaluation)
    ]
    identities = {
        "train": {stable_identity(row) for row in train_rows},
        "selector": {stable_identity(row) for row in selector_rows},
        "evaluation": {stable_identity(row) for row in evaluation_rows},
    }
    role_disjoint = not (
        identities["train"] & identities["selector"]
        or identities["train"] & identities["evaluation"]
        or identities["selector"] & identities["evaluation"]
    )
    if not role_disjoint:
        raise ValueError("source-lock role identities overlap")

    train_expanded = _jsonl(train_rows)
    selector_payload = _jsonl(selector_rows)
    evaluation_expanded = _jsonl(evaluation_rows)
    overlap_expanded = _jsonl(rejects)
    fresh_expanded = {
        "fresh-set-train.jsonl.gz": _jsonl(sorted(
            (canonical_fresh_row(row, "set_train") for row in parts["set_train"]),
            key=selection_rank,
        )),
        "fresh-set-validation.jsonl.gz": _jsonl(sorted(
            (canonical_fresh_row(row, "set_validation") for row in parts["set_validation"]),
            key=selection_rank,
        )),
        "fresh-ood-test.jsonl.gz": _jsonl(sorted(
            (canonical_fresh_row(row, "ood_test") for row in parts["ood_test"]),
            key=selection_rank,
        )),
    }
    evaluation_manifest_sha = sha256(evaluation_expanded).hexdigest()
    batch_payload = canonical_json({
        "batchSize": set_per_batch + ood_per_batch,
        "batches": batch_receipts,
        "evaluationManifestExpandedSha256": evaluation_manifest_sha,
        "items": len(evaluation_rows),
        "oodPerBatch": ood_per_batch,
        "schemaVersion": 1,
        "setValidationPerBatch": set_per_batch,
    })
    payloads = {
        "evaluation-batches.json": batch_payload,
        "evaluation-manifest.jsonl.gz": canonical_gzip(evaluation_expanded),
        **{name: canonical_gzip(payload) for name, payload in fresh_expanded.items()},
        "historical-comparators.jsonl.gz": canonical_gzip(history_expanded),
        "historical-metadata-receipt.json": canonical_json(history_receipt),
        "overlap-rejects.jsonl.gz": canonical_gzip(overlap_expanded),
        "selector-manifest.jsonl": selector_payload,
        "train-manifest.jsonl.gz": canonical_gzip(train_expanded),
    }

    if output.exists() or output.is_symlink():
        raise FileExistsError("M6 source-lock output must be absent")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("stale M6 source-lock partial exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir(mode=0o700)
    installed = False
    try:
        written = 0
        for name in sorted(payloads):
            _write_fsynced(temporary / name, payloads[name])
            written += 1
            if fail_after_artifacts is not None and written >= fail_after_artifacts:
                raise RuntimeError("injected source-lock write failure")
        artifacts: dict[str, Any] = {}
        for name in sorted(payloads):
            raw = payloads[name]
            expanded = gzip.decompress(raw) if name.endswith(".gz") else raw
            artifacts[name] = {
                "bytes": len(raw),
                "expandedBytes": len(expanded),
                "expandedSha256": sha256(expanded).hexdigest(),
                "sha256": sha256(raw).hexdigest(),
            }
        reject_counts: dict[str, int] = {}
        reject_cohorts: dict[str, int] = {}
        for row in rejects:
            reject_counts[row["layer"]] = reject_counts.get(row["layer"], 0) + 1
            key = f'{row["candidateCohort"]}->{row["comparatorCohort"]}'
            reject_cohorts[key] = reject_cohorts.get(key, 0) + 1
        summary = {
            "artifacts": artifacts,
            "coverage": {
                "selectorRealPrefixes": sorted({row["sourceGroup"] for row in selector_real}),
                "selectorSyntheticGroups": sorted({row["sourceGroup"] for row in selector_synthetic}),
            },
            "h3PixelsRead": False,
            "historyReceiptSha256": sha256(canonical_json(history_receipt)).hexdigest(),
            "production": production,
            "protocolCommit": protocol_commit,
            "quotas": {
                "batchCount": batch_count,
                "minimumTrainPerClass": minimum_train,
                "oodPerBatch": ood_per_batch,
                "selectorPerClass": selector_each,
                "setValidationPerBatch": set_per_batch,
            },
            "rejectCountsByCohort": reject_cohorts,
            "rejectCountsByLayer": reject_counts,
            "recipeSha256": sha256((Path(__file__).with_name("recipe.json")).read_bytes()).hexdigest(),
            "roleDisjoint": True,
            "rows": {
                "evaluation": len(evaluation_rows),
                "overlapRejects": len(rejects),
                "selector": len(selector_rows),
                "train": len(train_rows),
            },
            "schemaVersion": 1,
            "status": "m6-source-lock-prepared" if production else "m6-source-lock-fixture",
        }
        _write_fsynced(temporary / "source-lock-summary.json", canonical_json(summary))
        _fsync_directory(temporary)
        if fail_stage == "before-rename":
            raise RuntimeError("injected source-lock failure before rename")
        if fail_stage == "rename":
            raise OSError("injected source-lock rename failure")
        os.replace(temporary, output)
        installed = True
        if fail_stage in ("after-rename", "rollback-fsync"):
            raise RuntimeError("injected source-lock failure after rename")
        _fsync_directory(output.parent)
    except Exception:
        if installed:
            shutil.rmtree(output, ignore_errors=True)
            try:
                if output.exists() or output.is_symlink():
                    raise OSError("installed source-lock directory could not be removed")
                if fail_stage == "rollback-fsync":
                    raise OSError("injected source-lock rollback fsync failure")
                _fsync_directory(output.parent)
            except Exception as cleanup_error:
                raise RuntimeError("M6 source-lock publication state unknown after rollback failure") from cleanup_error
        else:
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    status = "m6-source-lock-prepared" if production else "m6-source-lock-fixture"
    return {
        "evaluationItems": len(evaluation_rows),
        "h3PixelsRead": False,
        "overlapRejects": len(rejects),
        "pixelsRead": False,
        "selectorItems": len(selector_rows),
        "production": production,
        "status": status,
        "trainItems": len(train_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("census", "source-lock"), required=True)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "census":
        load_recipe()
        print(json.dumps({
            "liveCensus": False,
            "pixelsRead": False,
            "stage": "P",
            "status": "metadata-contract-valid",
        }, sort_keys=True))
        return
    if args.rows is None or args.output is None:
        raise SystemExit("source-lock requires metadata JSON and output directory")
    payload = parse_json_bytes(args.rows.read_bytes(), label="source-lock input")
    if set(payload) != {"historyBundle", "parts", "protocolCommit"}:
        raise SystemExit("source-lock input must contain exact historyBundle, parts, and protocolCommit keys")
    print(json.dumps(source_lock(
        payload["parts"], args.output,
        history_bundle=payload["historyBundle"],
        protocol_commit=payload["protocolCommit"],
    ), sort_keys=True))


if __name__ == "__main__":
    main()
