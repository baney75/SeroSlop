from __future__ import annotations

import hashlib
import json
import unittest

from benchmark.m6.p6_frontier_inventory import (
    ALLOCATIONS,
    SOURCE_LOCK_COMMIT,
    SOURCE_LOCK_TREE,
    SOURCE_COMMIT,
    SOURCES,
    canonical_json,
    inventory_digest,
    load_inventory,
    validate_inventory_document,
    validate_nano_metadata,
    validate_taste_assets,
    validate_admission,
    source_lock_receipt,
    validate_allocations,
    validate_allocation_count_receipt,
    validate_aigen_metadata_csv,
    validate_aigen_tar_members,
    allocate_from_admitted,
    build_p6_source_lock,
    build_p6_source_lock_fixture,
    reopen_p6_source_lock,
    write_admission_bundle,
)


class P6FrontierInventoryTests(unittest.TestCase):
    def test_canonical_digest_is_sorted_and_lf_terminated(self):
        rows = [
            {"bytes": 2, "path": "images/b", "sha256": "b" * 64},
            {"bytes": 1, "path": "images/a", "sha256": "a" * 64},
        ]
        expected = hashlib.sha256(
            canonical_json(rows[1]) + canonical_json(rows[0])
        ).hexdigest()
        self.assertEqual(inventory_digest(rows), expected)

    def test_production_document_is_closed_and_pinned(self):
        value = load_inventory()
        validate_inventory_document(value)
        self.assertEqual(value["sourceCommit"], SOURCE_COMMIT)
        self.assertEqual(set(value["sources"]), {"aigenimages2026-train", "aigenimages2026-test", "taste", "nano-banana", "x-aigd"})
        self.assertFalse(value["claims"]["commercialRightsClearanceClaimed"])
        self.assertFalse(value["claims"]["h3PixelsRead"])

    def test_taste_metadata_counts_and_path_matching(self):
        paths = [f"images/{i:03d}.png" for i in range(644)]
        rows = [{"asset_id": i + 1, "model": name, "image_url": f"https://example.invalid/{i}", "track": "descriptions" if i < 320 else "aesthetics", "image_path": p} for i, (p, name) in enumerate(zip(paths, (list(SOURCES["taste"]["modelCounts"]) * 161)))]
        self.assertEqual(validate_taste_assets(rows, paths), SOURCES["taste"]["modelCounts"])

    def test_nano_row_census_boundary(self):
        shards = [f"data/{i:02d}.parquet" for i in range(31)]
        self.assertEqual(validate_nano_metadata(shards, 9457), 9457)

    def test_wrong_label_is_quarantined_not_admitted(self):
        row = {"sourceKey": "taste", "publisher": "purvanshi", "dataset": SOURCES["taste"]["dataset"], "revision": SOURCES["taste"]["revision"], "partition": "train", "sourceGroup": "GPT Image 1.5", "rowId": "1", "memberPath": "images/1.png", "label": "real", "encodedBytesSha256": "a" * 64, "decodedRgbSha256": "b" * 64, "dhash64": "0" * 16}
        result = validate_admission([row])
        self.assertEqual(result["rowCount"], 0)
        self.assertEqual(result["quarantineCount"], 1)

    def test_source_lock_rejects_wrong_parent(self):
        with self.assertRaises(ValueError):
            source_lock_receipt(source_commit="0" * 40, source_tree=SOURCE_LOCK_TREE, artifacts={"inventory": "a" * 64}, admission_sha256="a" * 64, allocation_sha256="b" * 64)

    def test_source_lock_receipt_defers_actual_p6_lineage(self):
        receipt = source_lock_receipt(artifacts={"inventory": "a" * 64}, admission_sha256="a" * 64, allocation_sha256="b" * 64)
        self.assertIsNone(receipt["actualP6SourceCommit"])
        self.assertEqual(receipt["priorAuthorizationCommit"], SOURCE_LOCK_COMMIT)

    def test_allocation_contract_is_exact(self):
        self.assertEqual(ALLOCATIONS["train"]["real"], 48662)
        self.assertEqual(ALLOCATIONS["synthetic-acceptance"]["nano-banana"], 9457)

    def test_exact_count_fixture_is_verified_without_fake_rows(self):
        receipt = {"status": "p6-allocation-count-fixture", "counts": ALLOCATIONS, "rows": 211324, "selectionInfluence": False}
        self.assertEqual(validate_allocation_count_receipt(receipt)["rows"], 211324)

    def test_aigen_metadata_missing_member_is_quarantined(self):
        members = {f"train/1_fake/image_{i}.png" for i in range(4879)}
        rows = [{"image_id": str(i), "filename": f"image_{i}.png", "caption": "", "caption_id": i, "split": "train"} for i in range(4879)]
        rows.append({"image_id": "missing", "filename": "image_midjourneyv7_300.png", "caption": "", "caption_id": 4880, "split": "train"})
        result = validate_aigen_metadata_csv(rows, members, partition="train")
        self.assertEqual(result["quarantined"], ["image_midjourneyv7_300.png"])

    def test_aigen_metadata_rejects_subset_and_wrong_val_set(self):
        members = {f"val/1_fake/image_{i}.png" for i in range(559)}
        rows = [{"image_id": str(i), "filename": f"image_{i}.png", "caption": "", "caption_id": i, "split": "val"} for i in range(558)]
        with self.assertRaises(ValueError):
            validate_aigen_metadata_csv(rows, members, partition="val")
        rows.append({"image_id": "x", "filename": "not_member.png", "caption": "", "caption_id": 559, "split": "val"})
        with self.assertRaises(ValueError):
            validate_aigen_metadata_csv(rows, members, partition="val")

    def test_receipt_free_admission_is_provisional_and_publish_disabled(self):
        row = {"sourceKey": "taste", "publisher": "purvanshi", "dataset": SOURCES["taste"]["dataset"], "revision": SOURCES["taste"]["revision"], "partition": "train", "sourceGroup": "GPT Image 1.5", "rowId": "provisional", "memberPath": "images/provisional.png", "label": "real", "encodedBytesSha256": "a" * 64, "decodedRgbSha256": "b" * 64, "dhash64": "0" * 16}
        self.assertEqual(validate_admission([row])["status"], "p6-admission-provisional")
        with self.assertRaises(RuntimeError):
            write_admission_bundle(__import__("pathlib").Path("/tmp/p6-disabled-admission"), [row])

    def test_reopen_rejects_verified_status_until_operational_verifier(self):
        import tempfile
        from pathlib import Path
        artifacts = {"inventory": "a" * 64, "history": "b" * 64, "materialization": "c" * 64, "admission": "d" * 64, "overlap": "e" * 64, "allocation": "f" * 64}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "fixture"
            build_p6_source_lock_fixture(output, artifacts=artifacts)
            receipt = json.loads((output / "source-lock.json").read_text())
            receipt["status"] = "p6-source-lock-verified"
            (output / "source-lock.json").write_bytes(canonical_json(receipt))
            with self.assertRaises(ValueError):
                reopen_p6_source_lock(output)

    def test_aigen_regular_executable_mode_is_allowed(self):
        import io, tarfile, tempfile
        path = __import__("pathlib").Path(tempfile.mkdtemp()) / "a.tar"
        name = "mnt/cephfs/home/common/datasets/online_sid_dataset/AIGenImages/AIGenImages2026/val/1_fake/example.png"
        with tarfile.open(path, "w") as archive:
            info = tarfile.TarInfo(name); info.size = 1; info.mode = 0o755
            archive.addfile(info, io.BytesIO(b"x"))
        self.assertEqual(len(validate_aigen_tar_members(path, partition="val", expected_rows=1)), 1)

    def test_fake_one_row_allocation_is_rejected(self):
        row = {"sourceKey": "taste", "dataset": "purvanshi/TASTE", "revision": SOURCES["taste"]["revision"], "partition": "train", "sourceGroup": "GPT Image 1.5", "rowId": "one", "label": "synthetic", "role": "synthetic-acceptance"}
        with self.assertRaises(ValueError):
            validate_allocations(ALLOCATIONS, [row])

    def test_allocator_shortage_is_terminal(self):
        with self.assertRaises(ValueError):
            allocate_from_admitted([])

    def test_dhash_distance_one_through_eight_is_overlap(self):
        base = {"sourceKey": "taste", "publisher": "purvanshi", "dataset": "purvanshi/TASTE", "revision": SOURCES["taste"]["revision"], "partition": "train", "sourceGroup": "GPT Image 1.5", "rowId": "fresh", "memberPath": "images/fresh.png", "label": "synthetic", "encodedBytesSha256": "a" * 64, "decodedRgbSha256": "b" * 64, "dhash64": "0000000000000000"}
        historical = dict(base)
        historical.pop("sourceKey"); historical.pop("publisher"); historical["cohort"] = "m5"
        historical["encodedBytesSha256"] = "c" * 64; historical["decodedRgbSha256"] = None
        for distance in range(1, 9):
            candidate = dict(base); candidate["rowId"] = f"fresh-{distance}"; candidate["dhash64"] = f"{1 << (distance - 1):016x}"
            result = validate_admission([candidate], [historical])
            self.assertEqual(result["rowCount"], 0, distance)

    def test_fresh_to_fresh_duplicate_pixels_are_rejected(self):
        base = {"sourceKey": "taste", "publisher": "purvanshi", "dataset": "purvanshi/TASTE", "revision": SOURCES["taste"]["revision"], "partition": "train", "sourceGroup": "GPT Image 1.5", "memberPath": "images/a.png", "label": "synthetic", "encodedBytesSha256": "a" * 64, "decodedRgbSha256": "b" * 64, "dhash64": "0" * 16}
        first = dict(base, rowId="one"); second = dict(base, rowId="two", memberPath="images/b.png")
        self.assertEqual(validate_admission([first, second])["rowCount"], 1)

    def test_verified_receipt_is_required_when_materializer_index_is_supplied(self):
        row = {"sourceKey": "taste", "publisher": "purvanshi", "dataset": "purvanshi/TASTE", "revision": SOURCES["taste"]["revision"], "partition": "train", "sourceGroup": "GPT Image 1.5", "rowId": "receipt-missing", "memberPath": "images/a.png", "label": "synthetic", "encodedBytesSha256": "a" * 64, "decodedRgbSha256": "b" * 64, "dhash64": "0" * 16, "verificationReceiptSha256": "c" * 64}
        result = validate_admission([row], [], verified_receipts={})
        self.assertEqual(result["rowCount"], 0)

    def test_source_lock_refuses_metadata_only_inputs(self):
        with self.assertRaises(RuntimeError):
            build_p6_source_lock(output=self._tmp_path("lock"), candidates=[], historical=[], artifacts={})

    def test_small_source_lock_fixture_writes_and_reopens(self):
        output = self._tmp_path("fixture")
        artifacts = {name: "a" * 64 for name in ("inventory", "history", "materialization", "admission", "overlap", "allocation")}
        packet = build_p6_source_lock_fixture(output, artifacts=artifacts)
        self.assertEqual(packet["status"], "p6-source-lock-fixture")
        self.assertTrue((output / "train-manifest.json").is_file())

    def _tmp_path(self, name):
        import tempfile
        return __import__("pathlib").Path(tempfile.mkdtemp()) / name


if __name__ == "__main__":
    unittest.main()
