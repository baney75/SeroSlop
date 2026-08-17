import copy
import base64
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import tempfile
import unittest
from unittest import mock

from benchmark.m6.contracts import (
    CENSUS_PATH,
    canonical_json,
    load_recipe,
    parse_json_bytes,
    validate_census_evidence,
    validate_manifest_row,
    validate_preflight,
)
from benchmark.m6.prepare import (
    DHashIndex,
    OOD_DATASET,
    OOD_REVISION,
    SET_DATASET,
    SET_REVISION,
    _source_lock_fixture,
    canonical_gzip,
    clean_source_rows,
    fixture_history_bundle,
    round_robin,
    source_lock,
    stable_identity,
    validate_history_bundle,
)
from benchmark.m6.preflight import project
from benchmark.m6.historical import build_history_bundle, write_history_bundle
from benchmark.m6.materialize import (
    SOURCE_SHARDS_SHA256,
    _derive_shard_materialization,
    _download_shard_fixture,
    _local_shard_path,
    _materialize_shard_fixture,
    _open_local_shard,
    _publish_summary,
    download_shard,
    image_facts,
    load_source_shards,
    materialize_shard,
    source_group,
    validate_source_shards,
)


def _far_hashes(count: int) -> list[str]:
    output: list[int] = []
    nonce = 0
    while len(output) < count:
        candidate = int.from_bytes(sha256(f"m6-fixture-{nonce}".encode()).digest()[:8], "big")
        nonce += 1
        if all((candidate ^ prior).bit_count() > 8 for prior in output):
            output.append(candidate)
    return [f"{value:016x}" for value in output]


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAFElEQVR4nGP4z8DAAMaMXCJyEAAAMTMD7ybEdyAAAAAASUVORK5CYII="
)


def _write_parquet(path: Path, *, filename: str = "train/real/COCO/shared.png") -> None:
    import pyarrow as pa
    from pyarrow import parquet as pq

    table = pa.Table.from_pylist([{
        "filename": filename,
        "generator": "COCO",
        "image": {"bytes": _PNG, "path": filename},
        "label": "real",
        "split": "train",
    }])
    pq.write_table(table, path)


def _fixture_download(
    root: Path, *, shard_name: str = "train-00000-of-00019.parquet",
    filename: str = "train/real/COCO/shared.png",
) -> tuple[Path, dict, dict]:
    source_file = root / (shard_name + ".source")
    _write_parquet(source_file, filename=filename)
    raw = source_file.read_bytes()
    source = {"dataset": SET_DATASET, "revision": SET_REVISION}
    shard = {
        "lfsSha256": sha256(raw).hexdigest(),
        "partition": "train",
        "path": f"data/Image/{shard_name}",
        "size": len(raw),
    }
    cache_root = root / "cache"
    _download_shard_fixture(
        cache_root=cache_root, source_key="omniFakeSet", source=source,
        shard=shard, downloader=lambda **_: str(source_file),
    )
    return cache_root, source, shard


def _fresh_row(
    index: int,
    label: str,
    partition: str,
    source_group: str,
    dhash64: str,
    *,
    dataset: str = SET_DATASET,
    revision: str = SET_REVISION,
    row_id: str | None = None,
    filename: str | None = None,
    encoded: str | None = None,
    decoded: str | None = None,
) -> dict:
    return {
        "dataset": dataset,
        "decodedRgbSha256": decoded or sha256(f"decoded-{index}".encode()).hexdigest(),
        "dhash64": dhash64,
        "encodedBytesSha256": encoded or sha256(f"encoded-{index}".encode()).hexdigest(),
        "filename": filename or f"images/{index}.png",
        "label": label,
        "partition": partition,
        "revision": revision,
        "rowId": row_id or f"row-{index}",
        "sourceGroup": source_group,
    }


def _history_from(row: dict, cohort: str = "m4") -> dict:
    return {
        "cohort": cohort,
        "dataset": row["dataset"],
        "decodedRgbSha256": None,
        "dhash64": row["dhash64"],
        "encodedBytesSha256": row["encodedBytesSha256"],
        "filename": row["filename"],
        "revision": row["revision"],
        "rowId": row["rowId"],
        "sourceGroup": row["sourceGroup"],
    }


def _fixture_parts() -> dict[str, list[dict]]:
    hashes = iter(_far_hashes(40))
    train = [
        _fresh_row(1, "real", "train", "COCO", next(hashes)),
        _fresh_row(2, "real", "train", "FFHQ", next(hashes)),
        _fresh_row(3, "full_synthetic", "train", "g1", next(hashes)),
        _fresh_row(4, "full_synthetic", "train", "g2", next(hashes)),
    ]
    validation = [
        _fresh_row(10, "real", "validation", "COCO", next(hashes)),
        _fresh_row(11, "real", "validation", "FFHQ", next(hashes)),
    ]
    validation.extend(
        _fresh_row(20 + index, "full_synthetic", "validation", f"g{index % 2 + 1}", next(hashes))
        for index in range(6)
    )
    ood = [
        _fresh_row(
            40 + index, "full_synthetic", "test", f"ood{index + 1}", next(hashes),
            dataset=OOD_DATASET, revision=OOD_REVISION,
        )
        for index in range(2)
    ]
    return {"set_train": train, "set_validation": validation, "ood_test": ood}


class M6ContractTests(unittest.TestCase):
    def test_recipe(self):
        recipe = load_recipe()
        self.assertEqual(recipe["evaluation"]["items"], 100_000)
        self.assertEqual(
            [branch["name"] for branch in recipe["training"]["branches"]],
            ["last6-consistency", "last6-margin"],
        )
        self.assertEqual(sum(len(branch["snapshots"]) for branch in recipe["training"]["branches"]), 4)

    def test_census_exact_and_mutations_reject(self):
        census = parse_json_bytes(CENSUS_PATH.read_bytes(), label="census")
        validate_census_evidence(census)
        for mutation in ({**census, "extra": True}, {**census, "status": "S"}):
            with self.assertRaises(ValueError):
                validate_census_evidence(mutation)
        broken = copy.deepcopy(census)
        broken["sources"]["omniFakeOOD"]["generatorCounts"]["GPT4o"] += 1
        with self.assertRaises(ValueError):
            validate_census_evidence(broken)

    def test_round_robin_is_source_balanced_and_order_independent(self):
        rows = [
            {"sourceGroup": "b", "encodedBytesSha256": "f" * 64, "rowId": "b"},
            {"sourceGroup": "a", "encodedBytesSha256": "1" * 64, "rowId": "a"},
            {"sourceGroup": "b", "encodedBytesSha256": "e" * 64, "rowId": "b2"},
            {"sourceGroup": "a", "encodedBytesSha256": "2" * 64, "rowId": "a2"},
        ]
        # Complete source rows are required for the production rank.  Supply a
        # deterministic rank-only adapter for this focused round-robin test.
        complete = []
        hashes = iter(_far_hashes(4))
        for index, row in enumerate(rows):
            complete.append(_fresh_row(
                100 + index, "real", "train", row["sourceGroup"], next(hashes),
                row_id=row["rowId"], encoded=row["encodedBytesSha256"],
            ))
        forward = [row["rowId"] for row in round_robin(complete, 4)]
        reverse = [row["rowId"] for row in round_robin(list(reversed(complete)), 4)]
        self.assertEqual(forward, reverse)
        self.assertEqual([row["sourceGroup"] for row in round_robin(complete, 4)], ["a", "b", "a", "b"])

    def test_dhash_index_matches_bruteforce(self):
        rng = random.Random(7)
        values = [rng.getrandbits(64) for _ in range(250)]
        index = DHashIndex()
        for value in values:
            index.add(value)
        queries = [*values, *[rng.getrandbits(64) for _ in range(100)]]
        for query in queries:
            expected = {(owner, (value ^ query).bit_count()) for owner, value in enumerate(values)
                        if (value ^ query).bit_count() <= 8}
            self.assertEqual(set(index.matches(query)), expected)

    def test_dhash_index_high_bit_distance_boundary(self):
        base = 0x0123456789ABCDEF
        index = DHashIndex()
        self.assertEqual(index.PROBES_PER_QUERY, 718)
        index.add(base)
        distance8 = base ^ sum(1 << bit for bit in (0, 8, 16, 24, 32, 40, 48, 63))
        distance9 = distance8 ^ (1 << 62)
        self.assertEqual(index.matches(distance8), [(0, 8)])
        self.assertEqual(index.matches(distance9), [])
        sparse = DHashIndex()
        sparse.add(base, owner=7)
        self.assertEqual(sparse.matches(base), [(7, 0)])

    def test_overlap_layers_are_audited_and_distance_nine_is_clean(self):
        hashes = iter(_far_hashes(20))
        base = _fresh_row(1, "real", "train", "COCO", next(hashes))
        identity_collision = _fresh_row(
            2, "real", "validation", "COCO", next(hashes),
            row_id=base["rowId"], filename=base["filename"],
        )
        encoded_collision = _fresh_row(
            3, "real", "validation", "FFHQ", next(hashes),
            encoded=base["encodedBytesSha256"],
        )
        decoded_collision = _fresh_row(
            4, "full_synthetic", "validation", "g1", next(hashes),
            decoded=base["decodedRgbSha256"],
        )
        base_value = int(base["dhash64"], 16)
        distance8 = _fresh_row(
            5, "full_synthetic", "validation", "g2",
            f"{base_value ^ sum(1 << bit for bit in range(8)):016x}",
        )
        distance9 = _fresh_row(
            6, "full_synthetic", "test", "ood",
            f"{base_value ^ sum(1 << bit for bit in range(9)):016x}",
            dataset=OOD_DATASET, revision=OOD_REVISION,
        )
        parts = {
            "set_train": [base],
            "set_validation": [identity_collision, encoded_collision, decoded_collision, distance8],
            "ood_test": [distance9],
        }
        clean, rejects = clean_source_rows(parts)
        self.assertIn(stable_identity(distance9), {stable_identity(row) for row in clean["ood_test"]})
        self.assertEqual(
            {row["layer"] for row in rejects},
            {"canonical-identity", "encoded-bytes-sha256", "decoded-rgb-sha256", "dhash-hamming<=8"},
        )
        for row in rejects:
            self.assertTrue(row["candidateArtifact"].endswith(".jsonl.gz"))
            self.assertIsInstance(row["candidateRowIndex"], int)
            self.assertTrue(row["comparatorArtifact"].endswith(".jsonl.gz"))
            self.assertIsInstance(row["comparatorRowIndex"], int)
            self.assertRegex(row["candidateIdentitySha256"], r"^[0-9a-f]{64}$")
            self.assertRegex(row["comparatorIdentitySha256"], r"^[0-9a-f]{64}$")
            if row["layer"] == "dhash-hamming<=8":
                self.assertEqual(row["dhashDistance"], 8)
            else:
                self.assertIsNone(row["dhashDistance"])

    def test_historical_rows_win_and_malformed_history_fails_closed(self):
        parts = _fixture_parts()
        candidate = parts["set_validation"][2]
        history = [_history_from(candidate)]
        history[0]["dhash64"] = None
        clean, rejects = clean_source_rows(parts, history)
        self.assertNotIn(stable_identity(candidate), {stable_identity(row) for row in clean["set_validation"]})
        self.assertEqual(rejects[0]["comparatorCohort"], "m4")
        broken = copy.deepcopy(history)
        broken[0]["encodedBytesSha256"] = "not-a-digest"
        with self.assertRaises(ValueError):
            clean_source_rows(parts, broken)
        fixture = fixture_history_bundle(history)
        fixture_rows, _, _ = validate_history_bundle(fixture, production=False)
        self.assertEqual(fixture_rows, history)

    def test_authoritative_history_reopens_all_fixed_metadata(self):
        bundle = build_history_bundle()
        self.assertEqual(bundle["receipt"]["cohortRowCounts"], {
            "h3": 600, "m2": 106878, "m3": 108978, "m4": 113162, "m5": 113162,
        })
        self.assertEqual(bundle["receipt"]["normalizedRows"], 442780)
        self.assertEqual(
            bundle["receipt"]["normalizedExpandedSha256"],
            "ea324b93c072332a19c9fc8256084fb2c7c2d82d74f31b10c5243b8c912661b7",
        )
        self.assertEqual(sum(row["dhash64"] is None for row in bundle["rows"]), 600)

    def test_source_shards_and_decoded_rgb_definition_are_frozen(self):
        inventory = load_source_shards()
        self.assertEqual(inventory["totalShards"], 85)
        self.assertEqual(inventory["totalBytes"], 64614604097)
        self.assertEqual(
            sha256(Path("benchmark/m6/source-shards.json").read_bytes()).hexdigest(),
            SOURCE_SHARDS_SHA256,
        )
        broken = copy.deepcopy(inventory)
        broken["sources"]["omniFakeSet"]["shards"][0]["size"] += 1
        with self.assertRaises(ValueError):
            validate_source_shards(broken)

        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAMAAAACCAIAAAASFvFNAAAAFElEQVR4nGP4z8DAAMaMXCJyEAAAMTMD7ybEdyAAAAAASUVORK5CYII="
        )
        self.assertEqual(image_facts(png), (
            "9c7330b7d83259d466401e33223b6d33fd7a769410da36a4070d5e59a863df46",
            "99bdbccd9aef19d940ae1186b76477d6916b6e24a34543a51509e3abd65712f3",
            "9e9e8e0e06004060", 3, 2,
        ))
        self.assertEqual(source_group(SET_DATASET, "train", "real", None, "train/real/COCO/a.jpg"), "COCO")
        self.assertEqual(source_group(OOD_DATASET, "test", "real", None, "real/a.jpg"), "OOD-real")
        self.assertEqual(source_group(SET_DATASET, "train", "full_synthetic", "flux", "train/full_synthetic/flux/a.png"), "flux")

    def test_source_lock_bundle_is_exact_atomic_and_order_independent(self):
        parts = _fixture_parts()
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        first = root / "first"
        result = _source_lock_fixture(
            parts, first, minimum_train=2, selector_each=2,
            set_per_batch=2, ood_per_batch=1, batch_count=2,
        )
        self.assertEqual(result, {
            "evaluationItems": 6,
            "h3PixelsRead": False,
            "overlapRejects": 0,
            "pixelsRead": False,
            "production": False,
            "selectorItems": 4,
            "status": "m6-source-lock-fixture",
            "trainItems": 4,
        })
        summary = json.loads((first / "source-lock-summary.json").read_text())
        self.assertFalse(summary["production"])
        self.assertEqual(summary["status"], "m6-source-lock-fixture")
        self.assertEqual(summary["protocolCommit"], "3b29ea2f9e1ad46e4cd78f47c9ccf5fe3a99877e")
        self.assertEqual(summary["quotas"], {
            "batchCount": 2, "minimumTrainPerClass": 2, "oodPerBatch": 1,
            "selectorPerClass": 2, "setValidationPerBatch": 2,
        })
        self.assertTrue(summary["roleDisjoint"])
        self.assertEqual(summary["rows"], {"evaluation": 6, "overlapRejects": 0, "selector": 4, "train": 4})
        for name, receipt in summary["artifacts"].items():
            raw = (first / name).read_bytes()
            expanded = gzip.decompress(raw) if name.endswith(".gz") else raw
            self.assertEqual(receipt["bytes"], len(raw))
            self.assertEqual(receipt["expandedBytes"], len(expanded))
            self.assertEqual(receipt["sha256"], sha256(raw).hexdigest())
            self.assertEqual(receipt["expandedSha256"], sha256(expanded).hexdigest())
        batches = json.loads((first / "evaluation-batches.json").read_text())
        self.assertEqual([(row["setValidation"], row["ood"]) for row in batches["batches"]], [(2, 1), (2, 1)])
        expanded_eval = gzip.decompress((first / "evaluation-manifest.jsonl.gz").read_bytes())
        self.assertEqual(batches["evaluationManifestExpandedSha256"], sha256(expanded_eval).hexdigest())
        manifest_payloads = {
            "train": gzip.decompress((first / "train-manifest.jsonl.gz").read_bytes()),
            "selector": (first / "selector-manifest.jsonl").read_bytes(),
            "evaluation": expanded_eval,
        }
        for expected_role, payload in manifest_payloads.items():
            rows = [json.loads(line) for line in payload.splitlines()]
            for role_index, row in enumerate(rows):
                validate_manifest_row(row)
                self.assertEqual(row["role"], expected_role)
                self.assertEqual(row["roleIndex"], role_index)

        shuffled = copy.deepcopy(parts)
        rng = random.Random(19)
        for rows in shuffled.values():
            rng.shuffle(rows)
        second = root / "second"
        _source_lock_fixture(
            shuffled, second, minimum_train=2, selector_each=2,
            set_per_batch=2, ood_per_batch=1, batch_count=2,
        )
        self.assertEqual(
            {path.name: path.read_bytes() for path in first.iterdir()},
            {path.name: path.read_bytes() for path in second.iterdir()},
        )
        with self.assertRaises(FileExistsError):
            _source_lock_fixture(parts, first)

    def test_source_lock_stale_partial_and_injected_failure_leave_no_output(self):
        parts = _fixture_parts()
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        output = root / "packet"
        partial = root / "packet.partial"
        partial.mkdir()
        with self.assertRaises(FileExistsError):
            _source_lock_fixture(parts, output)
        self.assertTrue(partial.is_dir())
        partial.rmdir()
        with self.assertRaisesRegex(RuntimeError, "injected source-lock write failure"):
            _source_lock_fixture(
                parts, output, minimum_train=2, selector_each=2,
                set_per_batch=2, ood_per_batch=1, batch_count=2,
                fail_after_artifacts=2,
            )
        self.assertFalse(output.exists())
        self.assertFalse(partial.exists())
        for stage in ("before-rename", "rename", "after-rename"):
            with self.assertRaises((RuntimeError, OSError)):
                _source_lock_fixture(
                    parts, output, minimum_train=2, selector_each=2,
                    set_per_batch=2, ood_per_batch=1, batch_count=2,
                    fail_stage=stage,
                )
            self.assertFalse(output.exists(), stage)
            self.assertFalse(partial.exists(), stage)
        with self.assertRaisesRegex(RuntimeError, "publication state unknown"):
            _source_lock_fixture(
                parts, output, minimum_train=2, selector_each=2,
                set_per_batch=2, ood_per_batch=1, batch_count=2,
                fail_stage="rollback-fsync",
            )
        self.assertFalse(output.exists())
        self.assertFalse(partial.exists())

    def test_history_receipt_is_mandatory_complete_and_hash_bound(self):
        hashes = iter(_far_hashes(8))
        rows = []
        for index, cohort in enumerate(("h3", "m2", "m3", "m4", "m5"), 1):
            fresh = _fresh_row(index, "real", "train", cohort, next(hashes))
            rows.append(_history_from(fresh, cohort))
        normalized = sorted(rows, key=lambda row: (row["cohort"], sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()))
        expanded = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in normalized)
        receipt = {
            "cohortRowCounts": {"h3": 1, "m2": 1, "m3": 1, "m4": 1, "m5": 1},
            "h3PixelsRead": False,
            "manifests": [
                {"bytes": 1, "cohort": cohort, "path": f"benchmark/evidence/{cohort}/manifest.jsonl", "rows": 1, "sha256": sha256(cohort.encode()).hexdigest()}
                for cohort in ("h3", "m2", "m3", "m4", "m5")
            ],
            "normalizedExpandedSha256": sha256(expanded).hexdigest(),
            "normalizedRows": 5,
            "schemaVersion": 1,
            "status": "m6-historical-metadata-locked",
        }
        bundle = {"receipt": receipt, "rows": rows}
        validated_rows, validated_receipt, validated_expanded = validate_history_bundle(bundle, production=True)
        self.assertEqual(len(validated_rows), 5)
        self.assertEqual(validated_receipt, receipt)
        self.assertEqual(sha256(validated_expanded).hexdigest(), receipt["normalizedExpandedSha256"])
        for mutation in (None, {}, {"receipt": receipt, "rows": []}):
            with self.assertRaises(ValueError):
                validate_history_bundle(mutation, production=True)
        broken = copy.deepcopy(bundle)
        broken["receipt"]["cohortRowCounts"].pop("h3")
        with self.assertRaises(ValueError):
            validate_history_bundle(broken, production=True)
        broken = copy.deepcopy(bundle)
        broken["receipt"]["normalizedExpandedSha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_history_bundle(broken, production=True)
        for key, value in (("schemaVersion", True), ("h3PixelsRead", 0)):
            broken = copy.deepcopy(bundle)
            broken["receipt"][key] = value
            with self.assertRaises(ValueError):
                validate_history_bundle(broken, production=True)
        broken = copy.deepcopy(bundle)
        broken["receipt"]["cohortRowCounts"]["m2"] = True
        with self.assertRaises(ValueError):
            validate_history_bundle(broken, production=True)
        broken = copy.deepcopy(bundle)
        broken["receipt"]["manifests"][0]["rows"] = 999
        with self.assertRaises(ValueError):
            validate_history_bundle(broken, production=True)
        broken = copy.deepcopy(bundle)
        broken["receipt"]["manifests"][0]["bytes"] = True
        with self.assertRaises(ValueError):
            validate_history_bundle(broken, production=True)
        broken = copy.deepcopy(bundle)
        broken["receipt"]["manifests"].pop()
        with self.assertRaises(ValueError):
            validate_history_bundle(broken, production=True)

    def test_production_source_lock_rejects_unreviewed_protocol_commit(self):
        with self.assertRaisesRegex(ValueError, "protocol commit changed"):
            source_lock(
                {}, Path("unused"), history_bundle={}, protocol_commit="0" * 40,
                materialization_summary_sha256="0" * 64, source_shards_sha256="1" * 64,
            )

    def test_canonical_gzip_golden_vector(self):
        payload = b'{"m6":"source-lock"}\n'
        compressed = canonical_gzip(payload)
        self.assertEqual(gzip.decompress(compressed), payload)
        self.assertEqual(compressed[:10].hex(), "1f8b08000000000002ff")
        self.assertEqual(sha256(compressed).hexdigest(), "787f54aacae190314ab9a8afe7fd7108550ec49fa2b6877f986e89239409f4fe")

    def test_duplicate_and_strict_utf8(self):
        with self.assertRaises(ValueError):
            parse_json_bytes(b'{"a":1,"a":2}', label="x")
        with self.assertRaises(ValueError):
            parse_json_bytes(b"\xff", label="x")

    def test_manifest_tampered_and_digest_rejected(self):
        row = {
            "dataset": SET_DATASET, "revision": SET_REVISION, "partition": "train",
            "rowId": "1", "filename": "x", "sourceGroup": "x", "label": "tampered",
            "encodedBytesSha256": "0" * 64, "decodedRgbSha256": "1" * 64,
            "dhash64": "0" * 16,
        }
        with self.assertRaises(ValueError):
            validate_manifest_row(row)
        validate_manifest_row({**row, "label": "real"})

    def test_preflight_aborts(self):
        with self.assertRaises(ValueError):
            validate_preflight({"pairedItemsPerSecond": 1})
        validate_preflight({
            "pairedItemsPerSecond": 100,
            "peakGpuMemoryBytes": 1,
            "freeRamBytes": 30_000_000_000,
            "freeDiskBytes": 300_000_000_000,
            "projectedPeakDiskBytes": 100_000_000_000,
            "hourlyUsd": 3,
            "projectedGpuUsd": 20,
            "allInUsd": 25,
            "projectedWallSeconds": 40_000,
        })

    def test_old_wall_bypass_is_red(self):
        with self.assertRaises(ValueError):
            project({
                "measuredBatchUniqueItems": 64,
                "sourceLockedUniqueItems": 100_000,
                "oneBatchSeconds": 150_000,
                "hourlyUsd": 1,
                "selectorSeconds": 0,
                "regressionSeconds": 0,
                "evalSeconds": 0,
            })

    def test_manifest_types_are_strict(self):
        row = {
            "dataset": SET_DATASET,
            "revision": SET_REVISION,
            "partition": "train",
            "rowId": "1",
            "filename": "ok/a.jpg",
            "sourceGroup": "src",
            "label": "real",
            "encodedBytesSha256": "0" * 64,
            "decodedRgbSha256": "1" * 64,
            "dhash64": "0" * 16,
        }
        validate_manifest_row(row)
        with self.assertRaises(ValueError):
            validate_manifest_row({**row, "filename": 1})
        with self.assertRaises(ValueError):
            validate_manifest_row({**row, "imageSha256": row["encodedBytesSha256"]})

    def test_pinned_downloader_mock_verifies_and_reuses(self):
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        source_file = root / "source.parquet"
        source_file.write_bytes(b"parquet-placeholder")
        payload = source_file.read_bytes()
        shard = {
            "path": "data/Image/train-00000-of-00019.parquet",
            "size": len(payload),
            "lfsSha256": sha256(payload).hexdigest(),
            "partition": "train",
        }
        source = {"dataset": SET_DATASET, "revision": SET_REVISION}
        calls = []

        def downloader(**kwargs):
            calls.append(kwargs)
            return str(source_file)

        receipt = _download_shard_fixture(
            cache_root=root / "cache", source_key="omniFakeSet", source=source,
            shard=shard, downloader=downloader,
        )
        self.assertEqual(receipt["sha256"], shard["lfsSha256"])
        self.assertEqual(calls, [{
            "filename": shard["path"],
            "local_dir": str(root / "cache/.hf-downloads/omniFakeSet"),
            "repo_id": SET_DATASET,
            "repo_type": "dataset",
            "revision": SET_REVISION,
            "token": False,
        }])
        destination = root / "cache/omni-fake-set" / shard["path"]
        self.assertEqual(destination.read_bytes(), payload)
        _download_shard_fixture(
            cache_root=root / "cache", source_key="omniFakeSet", source=source,
            shard=shard, downloader=lambda **_: self.fail("verified cache redownloaded"),
        )
        with self.assertRaisesRegex(ValueError, "source pin"):
            _download_shard_fixture(
                cache_root=root / "other", source_key="omniFakeSet",
                source={**source, "revision": "0" * 40}, shard=shard,
                downloader=downloader,
            )
        with self.assertRaisesRegex(ValueError, "shard pin"):
            _download_shard_fixture(
                cache_root=root / "other", source_key="omniFakeSet", source=source,
                shard={**shard, "path": "unexpected.parquet"}, downloader=downloader,
            )

    def test_downloader_rejects_wrong_bytes_and_symlink_cache(self):
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        source_file = root / "bad.parquet"
        source_file.write_bytes(b"bad")
        source = {"dataset": SET_DATASET, "revision": SET_REVISION}
        shard = {
            "path": "data/Image/train-00000-of-00019.parquet", "partition": "train",
            "size": 4, "lfsSha256": sha256(b"good").hexdigest(),
        }
        with self.assertRaises(ValueError):
            _download_shard_fixture(
                cache_root=root / "cache", source_key="omniFakeSet", source=source,
                shard=shard, downloader=lambda **_: str(source_file),
            )
        destination = root / "symlink/omni-fake-set" / shard["path"]
        destination.parent.mkdir(parents=True)
        destination.symlink_to(source_file)
        with self.assertRaisesRegex(ValueError, "symlink"):
            _download_shard_fixture(
                cache_root=root / "symlink", source_key="omniFakeSet", source=source,
                shard=shard, downloader=lambda **_: str(source_file),
            )

    def test_production_downloader_rejects_plausible_noninventory_shard(self):
        source = {"dataset": SET_DATASET, "revision": SET_REVISION}
        shard = {
            "path": "data/Image/train-99999-of-99999.parquet", "partition": "train",
            "size": 123, "lfsSha256": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "pinned inventory"):
            download_shard(
                cache_root=Path("unused"), source_key="omniFakeSet", source=source,
                shard=shard, downloader=lambda **_: self.fail("download should not run"),
            )
        with self.assertRaisesRegex(ValueError, "pinned inventory"):
            materialize_shard(
                cache_root=Path("unused"), output_root=Path("unused-output"),
                source_key="omniFakeSet", source=source, shard=shard, workers=1,
            )

    def test_shard_materialization_is_source_bound_atomic_and_resumable(self):
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        cache_root, source, shard = _fixture_download(root)
        output = root / "materialized"
        output.mkdir()
        receipt = _materialize_shard_fixture(
            cache_root=cache_root, output_root=output, source_key="omniFakeSet",
            source=source, shard=shard, workers=1,
        )
        final = output / "fragments/omniFakeSet" / PurePosixPath(shard["path"]).name
        self.assertEqual(set(os.listdir(final)), {"fragment.jsonl.gz", "receipt.json"})
        resumed = _materialize_shard_fixture(
            cache_root=cache_root, output_root=output, source_key="omniFakeSet",
            source=source, shard=shard, workers=1,
        )
        self.assertEqual(resumed, receipt)

        (final / "fragment.jsonl.gz").write_bytes(b"not-gzip")
        with self.assertRaises(ValueError):
            _materialize_shard_fixture(
                cache_root=cache_root, output_root=output, source_key="omniFakeSet",
                source=source, shard=shard, workers=1,
            )
        shutil.rmtree(final)
        _materialize_shard_fixture(
            cache_root=cache_root, output_root=output, source_key="omniFakeSet",
            source=source, shard=shard, workers=1,
        )
        expanded = gzip.decompress((final / "fragment.jsonl.gz").read_bytes())
        forged_row = json.loads(expanded)
        forged_row["sourceGroup"] = "forged-but-schema-valid"
        forged_expanded = canonical_json(forged_row)
        forged_fragment = canonical_gzip(forged_expanded)
        forged_receipt = parse_json_bytes(
            (final / "receipt.json").read_bytes(), label="fixture receipt",
        )
        forged_receipt.update({
            "expandedSha256": sha256(forged_expanded).hexdigest(),
            "fragmentBytes": len(forged_fragment),
            "fragmentSha256": sha256(forged_fragment).hexdigest(),
        })
        (final / "fragment.jsonl.gz").write_bytes(forged_fragment)
        (final / "receipt.json").write_bytes(canonical_json(forged_receipt))
        with self.assertRaisesRegex(ValueError, "source-derived"):
            _materialize_shard_fixture(
                cache_root=cache_root, output_root=output, source_key="omniFakeSet",
                source=source, shard=shard, workers=1,
            )

    def test_materializer_row_identity_uses_shard_and_ordinal(self):
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        cache_a, source, shard_a = _fixture_download(
            root, shard_name="train-00000-of-00019.parquet",
        )
        _, _, shard_b = _fixture_download(
            root, shard_name="train-00001-of-00019.parquet",
        )
        _, compressed_a, _ = _derive_shard_materialization(
            cache_root=cache_a, source_key="omniFakeSet", source=source,
            shard=shard_a, workers=1,
        )
        _, compressed_b, _ = _derive_shard_materialization(
            cache_root=cache_a, source_key="omniFakeSet", source=source,
            shard=shard_b, workers=1,
        )
        row_a = json.loads(gzip.decompress(compressed_a))
        row_b = json.loads(gzip.decompress(compressed_b))
        self.assertEqual(row_a["filename"], row_b["filename"])
        self.assertNotEqual(row_a["rowId"], row_b["rowId"])
        self.assertTrue(row_a["rowId"].endswith(":" + shard_a["path"] + ":0"))

    def test_materializer_publication_failures_are_retryable(self):
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        cache_root, source, shard = _fixture_download(root)
        stages = (
            "fragment-written", "receipt-written", "temporary-directory-fsynced",
            "before-rename", "after-rename", "parent-directory-fsynced",
        )
        for stage in stages:
            with self.subTest(stage=stage):
                output = root / ("out-" + stage)
                output.mkdir()

                def hook(current, wanted=stage):
                    if current == wanted:
                        raise OSError("injected " + wanted)

                with self.assertRaises(OSError):
                    _materialize_shard_fixture(
                        cache_root=cache_root, output_root=output,
                        source_key="omniFakeSet", source=source, shard=shard,
                        workers=1, failure_hook=hook,
                    )
                parent = output / "fragments/omniFakeSet"
                name = PurePosixPath(shard["path"]).name
                self.assertFalse((parent / name).exists())
                self.assertFalse((parent / (name + ".partial")).exists())
                _materialize_shard_fixture(
                    cache_root=cache_root, output_root=output,
                    source_key="omniFakeSet", source=source, shard=shard, workers=1,
                )

        output = root / "out-rollback"
        output.mkdir()

        def rollback_hook(current):
            if current in {"after-rename", "rollback-fsync"}:
                raise OSError("injected rollback")

        with self.assertRaisesRegex(RuntimeError, "state unknown"):
            _materialize_shard_fixture(
                cache_root=cache_root, output_root=output, source_key="omniFakeSet",
                source=source, shard=shard, workers=1, failure_hook=rollback_hook,
            )

    def test_verified_descriptor_survives_path_replacement(self):
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        cache_root, source, shard = _fixture_download(root)
        original_open = _open_local_shard
        local = _local_shard_path(cache_root, "omniFakeSet", shard)
        replacement = local.with_name(local.name + ".replacement")
        replacement.write_bytes(b"not parquet")

        def open_then_replace(*args, **kwargs):
            handle = original_open(*args, **kwargs)
            os.replace(replacement, local)
            return handle

        with mock.patch("benchmark.m6.materialize._open_local_shard", side_effect=open_then_replace):
            expanded, _, receipt = _derive_shard_materialization(
                cache_root=cache_root, source_key="omniFakeSet", source=source,
                shard=shard, workers=1,
            )
        self.assertTrue(expanded.endswith(b"\n"))
        self.assertEqual(receipt["rows"], 1)

    def test_historical_publication_rolls_back_after_rename(self):
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        fixture = {"rows": [], "receipt": {"schemaVersion": 1}}

        def renamed(current):
            if current == "renamed":
                raise OSError("post-rename")

        output = root / "history"
        with mock.patch("benchmark.m6.historical.build_history_bundle", return_value=fixture):
            with self.assertRaises(OSError):
                write_history_bundle(output, failure_hook=renamed)
        self.assertFalse(output.exists())

        def parent_fsynced(current):
            if current == "parent-fsynced":
                raise OSError("post-parent-fsync")

        with mock.patch("benchmark.m6.historical.build_history_bundle", return_value=fixture):
            with self.assertRaises(OSError):
                write_history_bundle(output, failure_hook=parent_fsynced)
        self.assertFalse(output.exists())

        def unknown(current):
            if current in {"renamed", "rollback-fsync"}:
                raise OSError("rollback")

        with mock.patch("benchmark.m6.historical.build_history_bundle", return_value=fixture):
            with self.assertRaisesRegex(RuntimeError, "state unknown"):
                write_history_bundle(output, failure_hook=unknown)

    def test_summary_publication_is_atomic_idempotent_and_symlink_safe(self):
        root = Path(tempfile.mkdtemp()).resolve()
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        expected = {"schemaVersion": 1, "status": "fixture-pass"}
        fields = set(expected)
        stages = (
            "summary-written", "before-summary-rename", "after-summary-rename",
            "summary-parent-fsynced",
        )
        for stage in stages:
            with self.subTest(stage=stage):
                parent = root / stage
                parent.mkdir()
                path = parent / "summary.json"

                def hook(current, wanted=stage):
                    if current == wanted:
                        raise OSError("injected " + wanted)

                with self.assertRaises(OSError):
                    _publish_summary(
                        path, expected, fields=fields, label="fixture summary",
                        failure_hook=hook,
                    )
                self.assertFalse(path.exists())
                self.assertFalse(path.with_suffix(".json.partial").exists())
                _publish_summary(path, expected, fields=fields, label="fixture summary")
                _publish_summary(path, expected, fields=fields, label="fixture summary")

        stale_parent = root / "stale"
        stale_parent.mkdir()
        stale = stale_parent / "summary.json"
        stale.with_suffix(".json.partial").write_bytes(b"interrupted")
        _publish_summary(stale, expected, fields=fields, label="fixture summary")
        self.assertFalse(stale.with_suffix(".json.partial").exists())
        stale.write_bytes(canonical_json({"schemaVersion": 1, "status": "changed"}))
        with self.assertRaises(ValueError):
            _publish_summary(stale, expected, fields=fields, label="fixture summary")

        symlink_parent = root / "symlink"
        symlink_parent.mkdir()
        target = symlink_parent / "target.json"
        target.write_bytes(canonical_json(expected))
        linked = symlink_parent / "summary.json"
        linked.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            _publish_summary(linked, expected, fields=fields, label="fixture summary")

        rollback_parent = root / "rollback"
        rollback_parent.mkdir()

        def rollback_hook(current):
            if current in {"after-summary-rename", "summary-rollback-fsync"}:
                raise OSError("injected rollback")

        with self.assertRaisesRegex(RuntimeError, "state unknown"):
            _publish_summary(
                rollback_parent / "summary.json", expected, fields=fields,
                label="fixture summary", failure_hook=rollback_hook,
            )

if __name__ == "__main__":
    unittest.main()
