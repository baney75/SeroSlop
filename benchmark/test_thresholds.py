"""Regression checks for exhaustive validation-threshold selection."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from thresholds import complete_decision_thresholds  # noqa: E402


def linear_quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


class CompleteDecisionThresholdsTest(unittest.TestCase):
    def test_covers_every_partition_that_quantile_sampling_can_skip(self) -> None:
        values = [float(value) for value in range(2_000)]
        legacy = [
            linear_quantile(values, 0.01 + index * (0.98 / 512))
            for index in range(513)
        ]
        skipped = next(
            index
            for index in range(len(values) - 1)
            if not any(values[index] < threshold < values[index + 1] for threshold in legacy)
        )
        labels = [0 if index <= skipped else 1 for index in range(len(values))]

        def balanced_accuracy(threshold: float) -> float:
            real = [value < threshold for value, label in zip(values, labels, strict=True) if label == 0]
            synthetic = [value >= threshold for value, label in zip(values, labels, strict=True) if label == 1]
            return (sum(real) / len(real) + sum(synthetic) / len(synthetic)) / 2

        complete = complete_decision_thresholds(values)
        self.assertEqual(len(complete), len(values) + 1)
        self.assertEqual(max(map(balanced_accuracy, complete)), 1.0)
        self.assertLess(max(map(balanced_accuracy, legacy)), 1.0)

    def test_rejects_empty_or_nonfinite_scores(self) -> None:
        with self.assertRaises(ValueError):
            complete_decision_thresholds([])
        with self.assertRaises(ValueError):
            complete_decision_thresholds([0.0, math.nan])

    def test_separates_adjacent_representable_scores(self) -> None:
        right = math.nextafter(1.0, math.inf)
        candidates = complete_decision_thresholds([1.0, right])
        partitions = [tuple(value >= threshold for value in (1.0, right)) for threshold in candidates]
        self.assertEqual(partitions, [(True, True), (False, True), (False, False)])

    def test_handles_extreme_finite_scores_without_arithmetic_overflow(self) -> None:
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        candidates = complete_decision_thresholds([-maximum, maximum])
        self.assertEqual(candidates[0], -maximum)
        self.assertTrue(-maximum < candidates[1] <= maximum)
        self.assertTrue(math.isinf(candidates[-1]))


if __name__ == "__main__":
    unittest.main()
