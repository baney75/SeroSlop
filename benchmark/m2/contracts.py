"""Dependency-light deterministic primitives for the M2 preparation contract."""

from __future__ import annotations

import gzip
from hashlib import sha256


def priority(namespace: str, value: str | int) -> str:
    return sha256(f"{namespace}{value}".encode()).hexdigest()


def deterministic_gzip(value: bytes) -> bytes:
    output = bytearray(gzip.compress(value, compresslevel=9, mtime=0))
    if len(output) < 10:
        raise ValueError("gzip output is truncated")
    output[9] = 0xFF
    return bytes(output)
