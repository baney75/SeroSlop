"""Pure helpers for binding cached features to their manifest-derived views."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np


VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")


class FeatureItem(Protocol):
    label: int
    source: str


def expected_view_metadata(
    items: Iterable[FeatureItem],
    *,
    training: bool,
    single_view_sources: frozenset[str],
) -> tuple[list[float], list[int], list[str]]:
    """Expand manifest labels, variant indexes, and sources in extraction order."""
    labels: list[float] = []
    variants: list[int] = []
    sources: list[str] = []
    for item in items:
        view_names = ("original",) if training and item.source in single_view_sources else VARIANTS
        labels.extend(float(item.label) for _ in view_names)
        variants.extend(VARIANTS.index(name) for name in view_names)
        sources.extend(item.source for _ in view_names)
    return labels, variants, sources


def validate_feature_arrays(
    features: np.ndarray,
    labels: np.ndarray,
    variants: np.ndarray,
    sources: np.ndarray,
    *,
    expected_labels: list[float],
    expected_variants: list[int],
    expected_sources: list[str],
    feature_width: int = 384,
) -> None:
    expected_count = len(expected_labels)
    if features.dtype != np.dtype(np.float32) or features.shape != (expected_count, feature_width):
        raise ValueError("Feature cache shape or dtype does not match the frozen ViT feature contract")
    if labels.dtype != np.dtype(np.float32) or labels.shape != (expected_count,):
        raise ValueError("Feature cache labels have an unexpected shape or dtype")
    if variants.dtype != np.dtype(np.int64) or variants.shape != (expected_count,):
        raise ValueError("Feature cache variants have an unexpected shape or dtype")
    if sources.dtype.kind != "U" or sources.shape != (expected_count,):
        raise ValueError("Feature cache sources have an unexpected shape or dtype")
    if not np.isfinite(features).all():
        raise ValueError("Feature cache contains non-finite values")
    if not np.array_equal(labels, np.asarray(expected_labels, dtype=np.float32)):
        raise ValueError("Feature cache labels do not match the manifest-derived view expansion")
    if not np.array_equal(variants, np.asarray(expected_variants, dtype=np.int64)):
        raise ValueError("Feature cache variants do not match the manifest-derived view expansion")
    if not np.array_equal(sources, np.asarray(expected_sources)):
        raise ValueError("Feature cache sources do not match the manifest-derived view expansion")
