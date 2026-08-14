"""Run-bound marker for resumable, from-pixel feature extraction."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
from typing import Any


RUN_ID = re.compile(r"^[a-f0-9]{32}$")


def cache_belongs_to_fresh_feature_run(
    fresh_feature_run_id: str | None,
    cached_fresh_feature_run_id: str,
) -> bool:
    return (
        fresh_feature_run_id is not None
        and RUN_ID.fullmatch(fresh_feature_run_id) is not None
        and cached_fresh_feature_run_id == fresh_feature_run_id
    )


def marker_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode()


def marker_sha256(payload: dict[str, Any]) -> str:
    return sha256(marker_bytes(payload)).hexdigest()


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(marker_bytes(payload))
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _validate(payload: dict[str, Any], context: dict[str, Any]) -> None:
    if (
        payload.get("schemaVersion") != 1
        or not RUN_ID.fullmatch(str(payload.get("runId", "")))
        or payload.get("state") not in {"extracting", "complete"}
        or payload.get("context") != context
    ):
        raise ValueError("Fresh-feature marker does not match this extraction contract")


def open_or_create_fresh_feature_run(
    path: Path,
    context: dict[str, Any],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create one extraction run or resume only shards carrying its exact ID."""
    if path.exists():
        payload = json.loads(path.read_text())
        _validate(payload, context)
        return payload
    identifier = run_id or secrets.token_hex(16)
    if not RUN_ID.fullmatch(identifier):
        raise ValueError("Fresh-feature run ID must be 32 lowercase hexadecimal characters")
    payload: dict[str, Any] = {
        "schemaVersion": 1,
        "runId": identifier,
        "state": "extracting",
        "context": context,
    }
    _write_atomic(path, payload)
    return payload


def complete_fresh_feature_run(
    path: Path,
    context: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    _validate(payload, context)
    if payload["runId"] != run_id:
        raise ValueError("Fresh-feature run ID changed before completion")
    payload["state"] = "complete"
    _write_atomic(path, payload)
    return payload
