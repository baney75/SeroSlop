"""Deterministic decision boundaries for a >= synthetic / < real classifier."""

from __future__ import annotations

from collections.abc import Iterable
import math


def complete_decision_thresholds(values: Iterable[float]) -> list[float]:
    """Return one threshold for every distinct classification partition."""
    unique = sorted(set(float(value) for value in values))
    if not unique or not all(math.isfinite(value) for value in unique):
        raise ValueError("Decision scores must be nonempty and finite")
    return [unique[0], *(math.nextafter(value, math.inf) for value in unique)]
