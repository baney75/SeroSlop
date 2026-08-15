"""Dependency-free checks for the tracked M3 failed-attempt packet."""

from __future__ import annotations

import base64
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC = ROOT / "benchmark/evidence/m3/failed-selector-diagnostic-1.json"
RECEIPT = ROOT / "benchmark/evidence/m3/failed-training-attempt-1.json"


def canonical(path: Path) -> tuple[bytes, dict[str, object]]:
    value = path.read_bytes()
    decoded = json.loads(value.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    expected = (json.dumps(decoded, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if value != expected:
        raise ValueError(f"Noncanonical M3 failure JSON: {path}")
    return value, decoded


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


class M3FailureDiagnosticTests(unittest.TestCase):
    def test_candidate_logits_are_complete_and_finite(self) -> None:
        _, diagnostic = canonical(DIAGNOSTIC)
        candidates = list(diagnostic["candidates"])  # type: ignore[arg-type]
        expected_parameters = [
            {"weightDecay": decay, "upstreamBlendAlpha": alpha}
            for decay in (0.1, 0.03, 0.01, 0.003, 0.001)
            for alpha in (0.4, 0.55, 0.7, 0.85, 1)
        ]
        self.assertEqual(len(candidates), 25)
        self.assertEqual([dict(row)["parameters"] for row in candidates], expected_parameters)
        for row_value in candidates:
            row = dict(row_value)
            logits = dict(row["selectorLogits"])
            raw = base64.b64decode(str(logits["base64"]), validate=True)
            self.assertEqual(len(raw), 9_600)
            self.assertEqual(logits["count"], 2_400)
            self.assertEqual(logits["bytes"], 9_600)
            self.assertEqual(logits["sha256"], sha256(raw).hexdigest())
            values = struct.unpack("<2400f", raw)
            self.assertTrue(all(math.isfinite(value) for value in values))
            self.assertEqual(row["feasibleThresholds"], 0)

    def test_receipt_binds_diagnostic_and_inventory(self) -> None:
        diagnostic_bytes, _ = canonical(DIAGNOSTIC)
        _, receipt = canonical(RECEIPT)
        diagnostic = dict(receipt["diagnostic"])
        self.assertEqual(diagnostic["sha256"], sha256(diagnostic_bytes).hexdigest())
        snapshot = dict(receipt["cacheSnapshot"])
        inventory = list(snapshot["inventory"])
        self.assertEqual(len(inventory), 59)
        self.assertEqual(snapshot["fileCount"], 59)
        self.assertEqual(
            snapshot["totalBytes"],
            sum(int(dict(item)["bytes"]) for item in inventory),
        )
        self.assertEqual(dict(snapshot["marker"])["state"], "extracting")
        self.assertFalse(dict(receipt["absence"])["candidateSelectionCompleted"])
        self.assertFalse(dict(receipt["h3Observation"])["h3PixelsReadOrScored"])
        self.assertTrue(dict(receipt["shippedModel"])["retained"])


if __name__ == "__main__":
    unittest.main()
