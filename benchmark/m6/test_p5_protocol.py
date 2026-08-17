from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
import tempfile
import unittest

from PIL import Image

from benchmark.m6.historical import HISTORY_SPECS
from benchmark.m6.materialize import image_facts
from benchmark.m6.p5_protocol import (
    ALLOCATION_SOURCE_SPECS,
    BRANCHES,
    COHORT_ORDER,
    EPOCHS,
    EXPECTED_QUOTA_CENSUS,
    FRONTIER_SOURCE_SPECS,
    REAL_PREFIXES,
    SELECTOR_GENERATORS,
    SYNTHETIC_COMPONENTS,
    RUNPOD_IMAGE_DIGEST,
    TRAINING,
    VIEWS,
    _allocate_all_cohorts_core as allocate_all_cohorts,
    calibration_weights,
    candidate_only,
    canonical_identity,
    canonical_json,
    decision,
    expected_protocol,
    expected_synthetic_batch_counts,
    fit_platt,
    load_p5_protocol,
    load_p5_quota_census,
    platt_score,
    rank_passers,
    round_robin_counts,
    _training_plan_core as training_plan,
    training_view,
    _validate_calibration_receipt_core as validate_calibration_receipt,
    validate_historical_coverage,
    _validate_paid_receipt_core as validate_paid_receipt,
    _validate_real_acceptance_core as validate_real_acceptance,
    _validate_selector_counts_core as validate_selector_counts,
    _validate_synthetic_acceptance_core as validate_synthetic_acceptance,
    verify_encoded_row,
)
from benchmark.m6.p5_transform_fixture import GOLDENS, render


def png_bytes() -> bytes:
    image = Image.new("RGB", (9, 7))
    image.putdata([((x * 31) % 256, (x * 67) % 256, (x * 101) % 256) for x in range(63)])
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=6)
    return output.getvalue()


def encoded_records(source_key: str = "aigenimages2026-test", *, locator: int = 3) -> tuple[dict, dict, bytes]:
    spec = FRONTIER_SOURCE_SPECS[source_key]
    encoded = png_bytes()
    common = {
        "sourceKey": source_key,
        "dataset": spec["dataset"],
        "publisher": spec["publisher"],
        "revision": spec["revision"],
        "partition": spec["partition"],
        "role": spec["role"],
        "label": spec["label"],
        "sourceGroup": "GPT4o",
        "locatorKind": "ordinal",
        "locatorValue": locator,
    }
    inventory = {
        **common,
        "containerPath": spec["containerPath"],
        "containerBytes": spec["containerBytes"],
        "containerSha256": spec["containerSha256"],
        "encodedBytes": len(encoded),
        "encodedSha256": sha256(encoded).hexdigest(),
    }
    return inventory, dict(common), encoded


def admitted_row(source_key: str, source_group: str, role: str, locator: int) -> tuple[dict, str]:
    spec = ALLOCATION_SOURCE_SPECS[source_key]
    row = {
        "dataset": spec["dataset"],
        "revision": spec["revision"],
        "partition": spec["partition"],
        "sourceKey": source_key,
        "sourceGroup": source_group,
        "locatorKind": "ordinal",
        "locatorValue": locator,
        "label": spec["label"],
    }
    identity_sha = sha256(canonical_identity({key: row[key] for key in ("dataset", "revision", "partition", "sourceGroup", "locatorKind", "locatorValue")})).hexdigest()
    body = {
        "status": "admitted",
        "sourceKey": source_key,
        "role": role,
        "label": spec["label"],
        "sourceGroup": source_group,
        "identitySha256": identity_sha,
        "inventorySha256": "1" * 64,
        "publisherSha256": "2" * 64,
        "encodedSha256": "3" * 64,
        "decodedRgbSha256": "4" * 64,
        "dhash64": "5" * 16,
        "width": 384,
        "height": 384,
    }
    receipt_sha = sha256(canonical_json(body)).hexdigest()
    row["admission"] = {**body, "receiptSha256": receipt_sha}
    return row, receipt_sha


def history_receipt() -> dict:
    manifests = [
        {"bytes": byte_count, "cohort": cohort, "path": path, "rows": rows, "sha256": digest}
        for cohort in ("h3", "m2", "m3", "m4", "m5")
        for path, byte_count, digest, rows, _kind in HISTORY_SPECS[cohort]
    ]
    return {
        "cohortRowCounts": {"h3": 600, "m2": 106878, "m3": 108978, "m4": 113162, "m5": 113162},
        "h3PixelsRead": False,
        "manifests": manifests,
        "normalizedExpandedSha256": "ea324b93c072332a19c9fc8256084fb2c7c2d82d74f31b10c5243b8c912661b7",
        "normalizedRows": 442780,
        "schemaVersion": 1,
        "status": "m6-historical-metadata-locked",
    }


def overlap_receipt(ledger: set[str], rejects: set[str]) -> dict:
    body = {
        "schemaVersion": 1,
        "status": "m6-overlap-rejects-locked",
        "historyNormalizedExpandedSha256": "ea324b93c072332a19c9fc8256084fb2c7c2d82d74f31b10c5243b8c912661b7",
        "admissionLedgerSha256": sha256(canonical_json(sorted(ledger))).hexdigest(),
        "rejectIdentitySetSha256": sha256(canonical_json(sorted(rejects))).hexdigest(),
        "rejectManifestSha256": "c" * 64,
        "rejectManifestExpandedSha256": "d" * 64,
        "rejectCount": len(rejects),
        "layerCounts": {"canonical-identity": len(rejects), "encoded-bytes-sha256": 0, "decoded-rgb-sha256": 0, "dhash-hamming<=8": 0},
    }
    return {**body, "receiptSha256": sha256(canonical_json(body)).hexdigest()}


def selector_panel() -> dict:
    by_source = round_robin_counts(SELECTOR_GENERATORS, 2000)
    views = {
        view: {
            "overall": {"tp": 2000, "tn": 2000, "fp": 0, "fn": 0},
            "sources": {source: {"tp": count, "fn": 0} for source, count in by_source.items()},
        }
        for view in VIEWS
    }
    return {"binding": score_binding("m6-selector-scored", 28000, True, False), "realCount": 2000, "syntheticCount": 2000, "syntheticBySource": by_source, "duplicates": 0, "missing": 0, "errors": 0, "views": views}


def synthetic_batches() -> list[dict]:
    return [
        {
            "index": index,
            "sourceCounts": source_counts,
            "original": {"tp": 100, "fn": 0},
        }
        for index, source_counts in enumerate(expected_synthetic_batch_counts())
    ]


def synthetic_panel() -> dict:
    return {
        "binding": score_binding("m6-synthetic-acceptance-scored", 700000, False, True),
        "components": dict(SYNTHETIC_COMPONENTS),
        "duplicates": 0,
        "missing": 0,
        "errors": 0,
        "batches": synthetic_batches(),
        "views": {
            view: {"sources": {source: {"tp": count, "fn": 0} for source, count in SYNTHETIC_COMPONENTS.items()}}
            for view in VIEWS
        },
    }


def real_panel() -> dict:
    prefixes = round_robin_counts(REAL_PREFIXES, 5000)
    sizes = {"oodReddit": 5000, **prefixes}
    return {
        "binding": score_binding("m6-real-acceptance-scored", 70000, False, True),
        "components": {"oodReddit": 5000, "setRealPrefixes": prefixes},
        "duplicates": 0,
        "missing": 0,
        "errors": 0,
        "views": {view: {"cohorts": {cohort: {"tn": count, "fp": 0} for cohort, count in sizes.items()}} for view in VIEWS},
    }


def score_binding(status: str, score_count: int, selection_influence: bool, selection_lock: bool) -> dict:
    value = {
        "status": status,
        "sourceLockReceiptSha256": "6" * 64,
        "modelSha256": "7" * 64,
        "modelBytes": 87_000_000,
        "calibrationReceiptSha256": "8" * 64,
        "threshold": 65.0,
        "scoreSha256": "9" * 64,
        "scoreCount": score_count,
        "selectionInfluence": selection_influence,
        "h3PixelsRead": False,
    }
    if selection_lock:
        value["selectionLockReceiptSha256"] = "a" * 64
    return value


def paid_receipt() -> dict:
    return {
        "schemaVersion": 1,
        "status": "m6-one-attempt-authorized",
        "provider": "RunPod",
        "cloudType": "SECURE",
        "gpuProduct": "NVIDIA L40S",
        "gpuCount": 1,
        "imageDigest": RUNPOD_IMAGE_DIGEST,
        "volume": {"id": "seroslop-m5-training", "sizeGb": 300, "mount": "/workspace"},
        "sourceLock": {"commit": "b" * 40, "tree": "c" * 40, "receiptSha256": "d" * 64},
        "preflight": {"status": "preflight-pass", "receiptSha256": "e" * 64},
        "rates": {"hourlyUsd": 1.0, "preflightUsd": 0.5, "gpuUsd": 20.0, "storageUsd": 0.2, "allInUsd": 20.7},
        "createdAtUnix": 1_800_000_000,
        "deadlineUnix": 1_800_043_200,
        "maximumRuntimeSeconds": 43_200,
        "stop": {"operatorStopRequired": True, "safetySeconds": 300, "noRetry": True},
        "cleanup": {"stopPod": True, "deleteContainerDisk": True, "retainNetworkVolume": True},
        "attempt": 1,
    }


class P5ProtocolTests(unittest.TestCase):
    def test_p5_production_entrypoints_are_fail_closed_until_p6(self) -> None:
        from benchmark.m6 import p5_protocol
        for name in (
            "allocate_all_cohorts", "validate_calibration_receipt",
            "validate_selector_counts", "validate_synthetic_acceptance",
            "validate_real_acceptance", "training_plan", "validate_paid_receipt",
        ):
            with self.assertRaisesRegex(RuntimeError, "P6 authoritative"):
                getattr(p5_protocol, name)()

    def test_protocol_and_quota_are_closed_world(self) -> None:
        self.assertEqual(load_p5_protocol(), expected_protocol())
        self.assertEqual(load_p5_quota_census(), EXPECTED_QUOTA_CENSUS)
        for path, target in ((Path(__file__).with_name("p5-protocol.json"), "maximumIterations"), (Path(__file__).with_name("p5-quota-census.json"), "historyRows")):
            value = json.loads(path.read_text())
            if target == "maximumIterations":
                value["calibration"][target] += 1
            else:
                value[target] += 1
            with tempfile.NamedTemporaryFile() as output:
                output.write(canonical_json(value)); output.flush()
                with self.assertRaises(ValueError):
                    (load_p5_protocol if "protocol" in path.name else load_p5_quota_census)(Path(output.name))
        raw = Path(__file__).with_name("p5-protocol.json").read_bytes().replace(b'"schemaVersion":1', b'"schemaVersion":1,"schemaVersion":1', 1)
        with tempfile.NamedTemporaryFile() as output:
            output.write(raw); output.flush()
            with self.assertRaises(ValueError): load_p5_protocol(Path(output.name))

    def test_candidate_metadata_alone_never_admits(self) -> None:
        self.assertEqual(candidate_only({"label": "synthetic"}), (False, {"status": "candidate-unverified", "reason": "encoded-bytes-and-publisher-record-required"}))

    def test_encoded_admission_binds_source_publisher_and_bytes(self) -> None:
        inventory, publisher, encoded = encoded_records()
        ok, receipt = verify_encoded_row(inventory, publisher, encoded)
        self.assertTrue(ok)
        self.assertEqual(receipt["status"], "admitted")
        self.assertEqual(receipt["encodedSha256"], image_facts(encoded)[0])
        self.assertEqual(verify_encoded_row(None, publisher, encoded)[1]["reason"], "inventory-schema")
        for mutator, reason in (
            (lambda i, p: p.pop("publisher"), "publisher-schema"),
            (lambda i, p: p.__setitem__("partition", "train"), "source-pin-mismatch"),
            (lambda i, p: i.__setitem__("label", "real"), "source-pin-mismatch"),
            (lambda i, p: i.__setitem__("containerSha256", "0" * 64), "container-pin-mismatch"),
            (lambda i, p: i.__setitem__("encodedSha256", "0" * 64), "encoded-sha256-mismatch"),
        ):
            changed_inventory, changed_publisher = deepcopy(inventory), deepcopy(publisher)
            mutator(changed_inventory, changed_publisher)
            self.assertEqual(verify_encoded_row(changed_inventory, changed_publisher, encoded)[1]["reason"], reason)
        taste_inventory, taste_publisher, taste_encoded = encoded_records("taste")
        self.assertEqual(verify_encoded_row(taste_inventory, taste_publisher, taste_encoded)[1]["reason"], "p6-exact-container-inventory-required")

    def test_dhash_and_transform_goldens_are_literal(self) -> None:
        inventory, publisher, encoded = encoded_records()
        receipt = verify_encoded_row(inventory, publisher, encoded)[1]
        self.assertEqual(receipt["dhash64"], image_facts(encoded)[2])
        self.assertEqual(set(GOLDENS), set(VIEWS))
        for view, golden in GOLDENS.items():
            rendered, size = render(view)
            self.assertEqual(sha256(rendered).hexdigest(), golden["sha256"])
            self.assertEqual(list(size), golden["dimensions"])

    def test_historical_receipt_is_exact_not_shape_only(self) -> None:
        receipt = history_receipt()
        validate_historical_coverage(receipt)
        changed = deepcopy(receipt); changed["manifests"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError): validate_historical_coverage(changed)
        changed = deepcopy(receipt); changed["normalizedExpandedSha256"] = "0" * 64
        with self.assertRaises(ValueError): validate_historical_coverage(changed)

    def test_allocation_requires_verified_admissions_roles_and_no_reuse(self) -> None:
        definitions = (
            ("omni-set-train-real", "cal-real", "calibration"),
            ("omni-set-train-synthetic", "train-synth", "balanced-training"),
            ("omni-set-validation-real", "selector-real", "selector"),
            ("aigenimages2026-test", "frontier", "synthetic-acceptance"),
            ("omni-ood-test-real", "ood-real", "real-acceptance"),
        )
        rows, ledger, reservations = [], set(), {}
        for index, (source_key, group, role) in enumerate(definitions):
            row, digest = admitted_row(source_key, group, role, index)
            rows.append(row); ledger.add(digest); reservations[role] = {group: 1}
        allocation = allocate_all_cohorts(reversed(rows), ledger, set(), overlap_receipt(ledger, set()), reservations)
        self.assertEqual(tuple(allocation), COHORT_ORDER)
        self.assertEqual([allocation[role][0]["sourceGroup"] for role in COHORT_ORDER], [entry[1] for entry in definitions])
        with self.assertRaises(ValueError): allocate_all_cohorts(rows, set(), set(), overlap_receipt(ledger, set()), reservations)
        with self.assertRaises(ValueError): allocate_all_cohorts(rows + [rows[0]], ledger, set(), overlap_receipt(ledger, set()), reservations)
        rejected = {rows[0]["admission"]["identitySha256"]}
        with self.assertRaises(ValueError): allocate_all_cohorts(rows, ledger, rejected, overlap_receipt(ledger, rejected), reservations)
        with self.assertRaises(ValueError): allocate_all_cohorts(rows, ledger, rejected, overlap_receipt(ledger, set()), reservations)
        wrong_role = deepcopy(rows[0]); wrong_role["admission"]["role"] = "real-acceptance"
        with self.assertRaises(ValueError): allocate_all_cohorts([wrong_role], ledger, set(), overlap_receipt(ledger, set()), {role: ({"cal-real": 1} if role == "calibration" else {f"missing-{role}": 1}) for role in COHORT_ORDER})

    def test_calibration_weights_are_exactly_source_and_class_balanced(self) -> None:
        weights = calibration_weights(["real", "real", "synthetic"], ["photos", "photos", "frontier"])
        self.assertEqual(weights, [0.75, 0.75, 1.5])
        self.assertEqual(sum(weights), 3.0)
        with self.assertRaises(ValueError): calibration_weights(["real"], ["photos"])

    def test_platt_and_decision_are_finite_and_convergent(self) -> None:
        weights = calibration_weights(["real", "real", "synthetic", "synthetic"], ["photos", "photos", "frontier", "frontier"])
        a, b = fit_platt([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1], weights)
        self.assertGreater(a, 0.0)
        self.assertTrue(decision(65.0)); self.assertFalse(decision(64.999))
        self.assertGreater(platt_score(1.0, a, b), 99.0)
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError): decision(value)
            with self.assertRaises(ValueError): platt_score(value, 1.0, 0.0)

    def test_calibration_receipt_recomputes_all_4000_rows(self) -> None:
        logits = [-1.0] * 2000 + [1.0] * 2000
        labels = [0] * 2000 + [1] * 2000
        real_counts = round_robin_counts(REAL_PREFIXES, 2000)
        synthetic_counts = round_robin_counts(SELECTOR_GENERATORS, 2000)
        sources = [source for source, count in real_counts.items() for _ in range(count)] + [source for source, count in synthetic_counts.items() for _ in range(count)]
        weights = calibration_weights(["real"] * 2000 + ["synthetic"] * 2000, sources)
        a, b = fit_platt(logits, labels, weights)
        float_digest = lambda values: sha256(b"".join(struct.pack(">d", value) for value in values)).hexdigest()
        receipt = {
            "schemaVersion": 1, "status": "m6-calibration-fit",
            "sourceLockReceiptSha256": "b" * 64, "modelSha256": "c" * 64,
            "modelBytes": 87_000_000, "realCount": 2000, "syntheticCount": 2000,
            "logitsSha256": float_digest(logits), "labelsSha256": sha256(bytes(labels)).hexdigest(),
            "sourcesSha256": sha256(canonical_json(sources)).hexdigest(), "weightsSha256": float_digest(weights),
            "aHex": a.hex(), "bHex": b.hex(), "threshold": 65.0,
            "method": "seroslop-m6/platt/v1", "selectorPixelsRead": False, "h3PixelsRead": False,
        }
        validate_calibration_receipt(receipt, logits, labels, sources)
        changed = deepcopy(receipt); changed["aHex"] = 1.0.hex()
        with self.assertRaises(ValueError): validate_calibration_receipt(changed, logits, labels, sources)
        changed_sources = list(sources); changed_sources[0] = "invented-real-source"
        with self.assertRaises(ValueError): validate_calibration_receipt(receipt, logits, labels, changed_sources)

    def test_views_are_deterministic_and_domain_bound(self) -> None:
        observed = {(branch, epoch): training_view(TRAINING["seed"], branch, epoch, "row-1") for branch in BRANCHES for epoch in EPOCHS}
        self.assertEqual(observed, {("consistency", 4): "social-q75", ("consistency", 6): "provider-cdn", ("margin", 4): "forum-repost", ("margin", 6): "screenshot"})
        self.assertIn(training_view(TRAINING["seed"], "consistency", 1, "row-1"), VIEWS[1:])
        with self.assertRaises(ValueError): training_view(TRAINING["seed"], "evil", 4, "row-1")

    def test_selector_recomputes_all_views_and_sources(self) -> None:
        panel = selector_panel(); validate_selector_counts(panel)
        changed = deepcopy(panel); changed["syntheticBySource"] = {"evil": 2000}
        with self.assertRaises(ValueError): validate_selector_counts(changed)
        changed = deepcopy(panel); changed["views"]["original"]["overall"]["fp"] = 1; changed["views"]["original"]["overall"]["tn"] = 1999
        with self.assertRaises(ValueError): validate_selector_counts(changed)
        changed = deepcopy(panel); changed["views"]["original"]["sources"][SELECTOR_GENERATORS[0]]["fn"] = 1
        with self.assertRaises(ValueError): validate_selector_counts(changed)
        changed = deepcopy(panel); changed["binding"]["threshold"] = 64.0
        with self.assertRaises(ValueError): validate_selector_counts(changed)

    def test_synthetic_acceptance_reconciles_every_item_view_and_batch(self) -> None:
        panel = synthetic_panel(); validate_synthetic_acceptance(panel)
        changed = deepcopy(panel); changed["components"]["taste"] -= 1
        with self.assertRaises(ValueError): validate_synthetic_acceptance(changed)
        changed = deepcopy(panel); changed["views"]["social-heavy"]["sources"]["taste"] = {"tp": 0, "fn": SYNTHETIC_COMPONENTS["taste"]}
        with self.assertRaises(ValueError): validate_synthetic_acceptance(changed)
        changed = deepcopy(panel); changed["binding"]["selectionInfluence"] = True
        with self.assertRaises(ValueError): validate_synthetic_acceptance(changed)
        changed = deepcopy(panel)
        for batch in changed["batches"][:51]: batch["original"] = {"tp": 0, "fn": 100}
        with self.assertRaises(ValueError): validate_synthetic_acceptance(changed)
        changed = deepcopy(panel)
        for batch in changed["batches"]: batch["original"] = {"tp": 96, "fn": 4}
        with self.assertRaises(ValueError): validate_synthetic_acceptance(changed)

    def test_real_acceptance_reconciles_social_forum_search_and_provider_views(self) -> None:
        panel = real_panel(); validate_real_acceptance(panel)
        changed = deepcopy(panel); changed["components"]["setRealPrefixes"][REAL_PREFIXES[0]] -= 1
        with self.assertRaises(ValueError): validate_real_acceptance(changed)
        changed = deepcopy(panel); changed["views"]["forum-repost"]["cohorts"]["oodReddit"] = {"tn": 4700, "fp": 300}
        with self.assertRaises(ValueError): validate_real_acceptance(changed)
        changed = deepcopy(panel); changed["binding"]["modelBytes"] = 90_000_001
        with self.assertRaises(ValueError): validate_real_acceptance(changed)

    def test_training_plan_rejects_nonproduction_or_unverified_rows(self) -> None:
        real, real_digest = admitted_row("omni-set-train-real", "OpenImages", "balanced-training", 1)
        synthetic, synthetic_digest = admitted_row("omni-set-train-synthetic", "sdxl_lightning", "balanced-training", 2)
        ledger = {real_digest, synthetic_digest}
        with self.assertRaises(ValueError): training_plan([synthetic, real], ledger)
        with self.assertRaises(ValueError): training_plan([real, synthetic], {real_digest})
        wrong = deepcopy(real); wrong["admission"]["role"] = "calibration"
        with self.assertRaises(ValueError): training_plan([wrong, synthetic], ledger)

    def test_ranking_is_exact_and_nonfinite_values_fail(self) -> None:
        values = [
            {"worstViewBA": .9, "originalBA": .9, "worstViewSyntheticRecall": .9, "worstSourceViewRecall": .9, "branchIndex": 1, "epoch": 6},
            {"worstViewBA": .9, "originalBA": .9, "worstViewSyntheticRecall": .9, "worstSourceViewRecall": .9, "branchIndex": 0, "epoch": 4},
        ]
        self.assertEqual(rank_passers(values)[0]["branchIndex"], 0)
        values[0]["worstViewBA"] = float("nan")
        with self.assertRaises(ValueError): rank_passers(values)

    def test_paid_receipt_is_exact_and_cost_bounded(self) -> None:
        receipt = paid_receipt(); validate_paid_receipt(receipt)
        for mutate in (
            lambda value: value.__setitem__("imageDigest", "sha256:" + "z" * 64),
            lambda value: value["rates"].__setitem__("gpuUsd", -1.0),
            lambda value: value["rates"].__setitem__("allInUsd", 30.01),
            lambda value: value.__setitem__("deadlineUnix", value["createdAtUnix"]),
            lambda value: value["stop"].__setitem__("noRetry", False),
            lambda value: value["volume"].__setitem__("sizeGb", 299),
            lambda value: value["sourceLock"].__setitem__("commit", "0" * 40),
            lambda value: value["preflight"].__setitem__("receiptSha256", "0" * 64),
        ):
            changed = deepcopy(receipt); mutate(changed)
            with self.assertRaises(ValueError): validate_paid_receipt(changed)


if __name__ == "__main__":
    unittest.main()
