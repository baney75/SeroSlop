import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark import bounty_proxy_m2 as proxy


class BountyProxyTests(unittest.TestCase):
    @staticmethod
    def taste_fixture():
        rows = []
        asset_id = 1
        for group in proxy.TASTE_GROUPS:
            for offset in range(161):
                rows.append({
                    "asset_id": asset_id,
                    "image_path": f"images/{asset_id:07d}.png",
                    "image_url": f"https://example.invalid/{asset_id}",
                    "model": group,
                    "track": "descriptions" if offset < 80 else "aesthetics",
                })
                asset_id += 1
        return rows

    def test_selection_is_exact_deterministic_and_leaves_eleven_per_group(self):
        rows = self.taste_fixture()
        selected, omitted = proxy.select_taste(rows)
        self.assertEqual(len(selected), 600)
        self.assertEqual(len({row["assetId"] for row in selected}), 600)
        self.assertEqual({group: sum(row["sourceGroup"] == group for row in selected) for group in proxy.TASTE_GROUPS}, {group: 150 for group in proxy.TASTE_GROUPS})
        self.assertEqual({group: len(ids) for group, ids in omitted.items()}, {group: 11 for group in proxy.TASTE_GROUPS})
        self.assertEqual(proxy.select_taste(list(reversed(rows))), (selected, omitted))

    def test_selection_rejects_unknown_or_short_group(self):
        rows = self.taste_fixture()
        rows[0]["model"] = "invented"
        with self.assertRaises(ValueError):
            proxy.select_taste(rows)
        with self.assertRaises(ValueError):
            proxy.select_taste(self.taste_fixture()[:-1])

    def test_inclusive_display_boundary_and_balanced_accuracy(self):
        rows = [
            {"label": 0, "displayScore": proxy.DISPLAY_THRESHOLD - 1e-12, "sourceGroup": "Met Open Access"},
            {"label": 0, "displayScore": proxy.DISPLAY_THRESHOLD, "sourceGroup": "Met Open Access"},
            {"label": 1, "displayScore": proxy.DISPLAY_THRESHOLD, "sourceGroup": proxy.TASTE_GROUPS[0]},
            {"label": 1, "displayScore": proxy.DISPLAY_THRESHOLD - 1e-12, "sourceGroup": proxy.TASTE_GROUPS[0]},
        ]
        result = proxy.metrics(rows)
        self.assertEqual(result["confusion"], {"fn": 1, "fp": 1, "tn": 1, "tp": 1})
        self.assertEqual(result["balancedAccuracy"], 0.5)
        self.assertEqual(proxy.metrics([
            {"label": 0, "displayScore": 0.0},
            {"label": 1, "displayScore": proxy.DISPLAY_THRESHOLD},
        ])["balancedAccuracy"], 1.0)

    def test_metrics_reject_nonfinite_and_missing_class(self):
        with self.assertRaises(ValueError):
            proxy.metrics([{"label": 0, "displayScore": float("nan")}, {"label": 1, "displayScore": 1.0}])
        with self.assertRaises(ValueError):
            proxy.metrics([{"label": 1, "displayScore": 1.0}])

    def test_paths_reject_escape_backslash_control_and_wrong_prefix(self):
        for value in ("../x", "/x", "a\\b", "a/./b", "a/../b", "a\x00b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                proxy._safe_relative(value)
        with self.assertRaises(ValueError):
            proxy._safe_relative("other/1.png", prefix="images/")
        self.assertEqual(proxy._safe_relative("images/1.png", prefix="images/"), "images/1.png")

    def test_canonical_parser_rejects_duplicates_and_nonfinite(self):
        with self.assertRaises(ValueError):
            proxy.parse_json_bytes(b'{"a":1,"a":2}\n')
        with self.assertRaises(ValueError):
            proxy.parse_json_bytes(b'{"a":NaN}\n')
        value = {"b": 1, "a": "ok"}
        self.assertEqual(proxy.canonical_json(proxy.parse_json_bytes(proxy.canonical_json(value))), proxy.canonical_json(value))

    def test_actual_freeze_is_metadata_only_exact_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            with patch.object(proxy, "_physical_file", side_effect=AssertionError("freeze opened a selected image")):
                lock = proxy.freeze_manifest(evidence)
            rows, reopened, manifest, raw_lock = proxy.reopen_frozen(evidence)
            self.assertEqual(lock, reopened)
            self.assertEqual(len(rows), 1200)
            self.assertEqual(sum(row["label"] == 0 for row in rows), 600)
            self.assertEqual(sum(row["label"] == 1 for row in rows), 600)
            self.assertTrue(reopened["pixelsReadAtFreeze"] is False)
            self.assertTrue(reopened["inferenceRun"] is False)
            self.assertTrue(reopened["bountyAcceptanceClaimed"] is False)
            self.assertEqual(len(manifest.splitlines()), 1200)
            self.assertEqual(proxy.canonical_json(json.loads(raw_lock)), raw_lock)
            for group, omitted in reopened["selection"]["tasteOmittedAssetIds"].items():
                self.assertIn(group, proxy.TASTE_GROUPS)
                self.assertEqual(len(omitted), 11)
            with self.assertRaises(FileExistsError):
                proxy.freeze_manifest(evidence)

    def test_atomic_directory_rolls_back_after_rename_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)

            def fail(stage):
                if stage == "after-rename":
                    raise RuntimeError("injected")

            with self.assertRaises(RuntimeError):
                proxy._publish_directory(parent, "packet", {"one.json": b"{}\n"}, failure_hook=fail)
            self.assertFalse((parent / "packet").exists())
            self.assertFalse((parent / ".packet.partial").exists())


if __name__ == "__main__":
    unittest.main()
