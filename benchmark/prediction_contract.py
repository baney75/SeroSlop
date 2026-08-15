"""Shared integrity checks for recorded model logits and probabilities."""

from __future__ import annotations

import math


PROBABILITY_TOLERANCE = 2e-12


def sigmoid_scalar(logit: float) -> float:
    if not math.isfinite(logit):
        raise ValueError("Prediction logit must be finite")
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def require_logit_probability_consistency(
    logit: float,
    probability: float,
    *,
    tolerance: float = PROBABILITY_TOLERANCE,
    decision_threshold: float | None = None,
) -> None:
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Prediction probability must be finite and between zero and one")
    expected = sigmoid_scalar(logit)
    if abs(expected - probability) > tolerance:
        raise ValueError("Prediction probability does not equal sigmoid(logit)")
    if decision_threshold is not None:
        if not math.isfinite(decision_threshold) or not 0 < decision_threshold < 1:
            raise ValueError("Prediction decision threshold must be finite and strictly between zero and one")
        if (expected >= decision_threshold) != (probability >= decision_threshold):
            raise ValueError("Prediction probability crosses the binary64 decision boundary")
