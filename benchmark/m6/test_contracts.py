import json
import unittest
from pathlib import Path
from benchmark.m6.contracts import load_recipe, parse_json_bytes, validate_manifest_row, validate_preflight, overlap_reject, validate_census_evidence, CENSUS_PATH
from benchmark.m6.preflight import project


class M6ContractTests(unittest.TestCase):
    def test_recipe(self):
        recipe = load_recipe()
        self.assertEqual(recipe["evaluation"]["items"], 100000)
        self.assertEqual([b["name"] for b in recipe["training"]["branches"]], ["last6-consistency", "last6-margin"])
        self.assertEqual(sum(len(b["snapshots"]) for b in recipe["training"]["branches"]), 4)

    def test_census_exact_and_mutations_reject(self):
        census = parse_json_bytes(CENSUS_PATH.read_bytes(), label="census")
        validate_census_evidence(census)
        for mutation in ({**census, "extra": True}, {**census, "status": "S"}):
            with self.assertRaises(ValueError): validate_census_evidence(mutation)
        broken = json.loads(json.dumps(census)); broken["sources"]["omniFakeOOD"]["generatorCounts"]["GPT4o"] += 1
        with self.assertRaises(ValueError): validate_census_evidence(broken)

    def test_duplicate_and_strict_utf8(self):
        with self.assertRaises(ValueError): parse_json_bytes(b'{"a":1,"a":2}', label="x")
        with self.assertRaises(ValueError): parse_json_bytes(b"\xff", label="x")

    def test_tampered_and_digest_rejected(self):
        row = {"dataset":"x","revision":"0"*40,"partition":"test","rowId":"1","filename":"x","sourceGroup":"x","label":"tampered","imageSha256":"0"*64,"dhash64":"0"*16}
        with self.assertRaises(ValueError): validate_manifest_row(row)

    def test_preflight_aborts(self):
        with self.assertRaises(ValueError): validate_preflight({"pairedItemsPerSecond": 1})
        validate_preflight({"pairedItemsPerSecond": 100,"peakGpuMemoryBytes":1,"freeRamBytes":30_000_000_000,"freeDiskBytes":300_000_000_000,"projectedPeakDiskBytes":100_000_000_000,"hourlyUsd":3,"projectedGpuUsd":20,"allInUsd":25,"projectedWallSeconds":40000})

    def test_old_wall_bypass_is_red(self):
        with self.assertRaises(ValueError):
            project({"measuredBatchUniqueItems": 64, "sourceLockedUniqueItems": 100000, "oneBatchSeconds": 150000, "hourlyUsd": 1, "selectorSeconds": 0, "regressionSeconds": 0, "evalSeconds": 0})

    def test_manifest_types_and_overlap(self):
        row = {"dataset":"JamalLee/Omni-Fake-SET","revision":"0"*40,"partition":"train","rowId":"1","filename":"ok/a.jpg","sourceGroup":"src","label":"real","imageSha256":"0"*64,"dhash64":"0"*16}
        validate_manifest_row(row)
        with self.assertRaises(ValueError): validate_manifest_row({**row, "filename": 1})
        self.assertEqual(overlap_reject([row], [row]), [])


if __name__ == "__main__": unittest.main()
