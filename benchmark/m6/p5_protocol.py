"""Fail-closed M6 P5 contracts.

P5 fixes data admission, allocation, calibration, views, selector gates,
acceptance panels, and paid-run evidence before P6 may acquire new pixels.
It deliberately provides no downloader, trainer, scorer, or provider client.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import math
from pathlib import Path
import re
import struct
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from benchmark.m6.contracts import parse_json_bytes
from benchmark.m6.materialize import image_facts


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "benchmark/m6/p5-protocol.json"
QUOTA_CENSUS_PATH = ROOT / "benchmark/m6/p5-quota-census.json"

ALLOCATION_DOMAIN = b"seroslop-m6/allocation/v1"
PLATT_DOMAIN = "seroslop-m6/platt/v1"
VIEW_DOMAIN = b"seroslop-m6/view/v1"
RANKING_DOMAIN = "seroslop-m6/ranking/v1"
VIEWS = (
    "original",
    "screenshot",
    "social-q75",
    "social-heavy",
    "forum-repost",
    "search-thumbnail",
    "provider-cdn",
)
STRESS_VIEWS = VIEWS[1:]
BRANCHES = ("consistency", "margin")
TRAINING_EPOCHS = tuple(range(1, 7))
SNAPSHOT_EPOCHS = (4, 6)
# Compatibility name for candidate snapshot validation.
EPOCHS = SNAPSHOT_EPOCHS
REAL_PREFIXES = (
    "COCO",
    "FFHQ",
    "OpenForensics_real",
    "OpenImages",
    "WIDER",
    "celebA",
    "flickr30k-images",
)
SELECTOR_GENERATORS = (
    "Absolute_Reality", "AttGAN", "BEGAN", "CramerGAN", "DiT10k",
    "FLUX_kon", "InfoMaxGAN", "MMDGAN", "OpenForensics_fake", "R3GAN",
    "RDDM10k", "RelGAN", "SNGAN", "STGAN", "SiT10k", "VQGAN",
    "codiff", "flux", "flux-face", "kandinsky3", "latent_diffusion",
    "latent_diffusion_1", "latent_diffusion_2", "latent_diffusion_3",
    "midjourney", "playground", "qwen-image", "realism_riiwa",
    "realistic_vision", "sd-3.5-face", "sdxl_lightning", "sg2", "sg3",
    "sgxl",
)
SYNTHETIC_COMPONENTS = {
    "setValidation": 58_228,
    "ood": 28_693,
    "aigenimages2026": 559,
    "taste": 644,
    "x-aigd": 2_419,
    "nano-banana": 9_457,
}
FRONTIER_COMPONENTS = {
    "aigenimages2026": 559,
    "taste": 644,
    "x-aigd": 2_419,
    "nano-banana": 9_457,
}
TRAINING_SOURCE_KEY_COUNTS = {
    "omni-set-train-real": 48_662,
    "omni-set-train-synthetic": 43_782,
    "aigenimages2026-train": 4_880,
}
TRAINING_LABEL_COUNTS = {"real": 48_662, "synthetic": 48_662}
RUNPOD_IMAGE_DIGEST = "sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385"
RUNPOD_VOLUME_ID = "seroslop-m5-training"

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX16 = re.compile(r"^[0-9a-f]{16}$")
RUNPOD_ID = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


FRONTIER_SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "aigenimages2026-train": {
        "dataset": "pthan12/AIGenImages2026",
        "publisher": "pthan12",
        "revision": "073e1924d9d0d85ac97a53b07947b6ac95ce241c",
        "partition": "train",
        "role": "balanced-training",
        "label": "synthetic",
        "containerPath": "aigenimages2026.tar.gz",
        "containerBytes": 11_138_511_098,
        "containerSha256": "67c6042712f783aebfdb29f8a8903dfc94fc7ac54fee5c154eaf6b880d0ec498",
        "rows": 4_880,
    },
    "aigenimages2026-test": {
        "dataset": "pthan12/AIGenImages2026",
        "publisher": "pthan12",
        "revision": "073e1924d9d0d85ac97a53b07947b6ac95ce241c",
        "partition": "test",
        "role": "synthetic-acceptance",
        "label": "synthetic",
        "containerPath": "aigenimages2026.tar.gz",
        "containerBytes": 11_138_511_098,
        "containerSha256": "67c6042712f783aebfdb29f8a8903dfc94fc7ac54fee5c154eaf6b880d0ec498",
        "rows": 559,
    },
    "taste": {
        "dataset": "purvanshi/TASTE",
        "publisher": "purvanshi",
        "revision": "731a7f588d433214c6d864d2e9f47978d91aed6b",
        "partition": "train",
        "role": "synthetic-acceptance",
        "label": "synthetic",
        "containerPath": None,
        "containerBytes": None,
        "containerSha256": None,
        "rows": 644,
    },
    "nano-banana": {
        "dataset": "bitmind/nano-banana",
        "publisher": "bitmind",
        "revision": "9ea8da32a5be03f4946e6cb10c2d2f8e90f0a0a4",
        "partition": "train",
        "role": "synthetic-acceptance",
        "label": "synthetic",
        "containerPath": None,
        "containerBytes": None,
        "containerSha256": None,
        "rows": 9_457,
    },
    "x-aigd": {
        "dataset": "Coxy7/X-AIGD",
        "publisher": "Coxy7",
        "revision": "92180f32030507ab54a40d6f1b88f39d6cec8178",
        "partition": "labeled_test",
        "role": "synthetic-acceptance",
        "label": "synthetic",
        "containerPath": "data/labeled_test-00000-of-00001.parquet",
        "containerBytes": 3_488_049_189,
        "containerSha256": "f86630ae51ef1103de204c879ad74d70bacaeca258489f2c32102851344a5c75",
        "rows": 2_419,
    },
}
FORBIDDEN_SOURCES = frozenset({
    "NTIRE", "Treasure", "FLUX.2 2M", "X-AIGD reconstructed real",
    "OpenFake", "RISE",
})
ALLOCATION_SOURCE_KEYS = frozenset({
    "omni-set-train-real", "omni-set-train-synthetic",
    "omni-set-validation-real", "omni-set-validation-synthetic",
    "omni-ood-test-real", "omni-ood-test-synthetic",
    *FRONTIER_SOURCE_SPECS,
})
OMNI_SET_REVISION = "724e97f5fc9f4b89f59631a8d4e6331712b7d441"
OMNI_OOD_REVISION = "9ed7e38bbdb4aeb2eb553896a5890680a9ffcf17"
ALLOCATION_SOURCE_SPECS: dict[str, dict[str, Any]] = {
    "omni-set-train-real": {
        "dataset": "JamalLee/Omni-Fake-SET", "revision": OMNI_SET_REVISION,
        "partition": "train", "label": "real", "roles": ("calibration", "balanced-training"),
    },
    "omni-set-train-synthetic": {
        "dataset": "JamalLee/Omni-Fake-SET", "revision": OMNI_SET_REVISION,
        "partition": "train", "label": "synthetic", "roles": ("calibration", "balanced-training"),
    },
    "omni-set-validation-real": {
        "dataset": "JamalLee/Omni-Fake-SET", "revision": OMNI_SET_REVISION,
        "partition": "validation", "label": "real", "roles": ("selector", "real-acceptance"),
    },
    "omni-set-validation-synthetic": {
        "dataset": "JamalLee/Omni-Fake-SET", "revision": OMNI_SET_REVISION,
        "partition": "validation", "label": "synthetic", "roles": ("selector", "synthetic-acceptance"),
    },
    "omni-ood-test-real": {
        "dataset": "JamalLee/Omni-Fake-OOD", "revision": OMNI_OOD_REVISION,
        "partition": "test", "label": "real", "roles": ("real-acceptance",),
    },
    "omni-ood-test-synthetic": {
        "dataset": "JamalLee/Omni-Fake-OOD", "revision": OMNI_OOD_REVISION,
        "partition": "test", "label": "synthetic", "roles": ("synthetic-acceptance",),
    },
    **{
        key: {
            "dataset": value["dataset"], "revision": value["revision"],
            "partition": value["partition"], "label": value["label"],
            "roles": (value["role"],),
        }
        for key, value in FRONTIER_SOURCE_SPECS.items()
    },
}
TRAINING_SOURCE_KEYS = frozenset({
    "omni-set-train-real", "omni-set-train-synthetic",
    "aigenimages2026-train",
})

TRAINING = {
    "branches": BRANCHES,
    "epochs": 6,
    "snapshots": SNAPSHOT_EPOCHS,
    "blocks": tuple(range(6, 12)),
    "batchSize": 64,
    "accumulation": 2,
    "seed": 20_260_816,
    "optimizer": "AdamW",
    "backboneLr": 2e-5,
    "classifierLr": 2e-4,
    "weightDecay": 0.05,
    "clip": 1.0,
    "warmup": 0.05,
    "scheduler": "cosine",
    "precision": "bfloat16",
    "bce": True,
    "consistencyMse": 0.05,
    "marginCoefficient": 0.2,
    "margin": 1.0,
    "threshold": 65.0,
}


def canonical_json(value: object) -> bytes:
    import json
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _require_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} key set changed")


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{label} must be an exact integer")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{label} is below its minimum")
    return number


def _hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{label} digest invalid")
    return value


EXPECTED_QUOTA_CENSUS = {
    "schemaVersion": 1,
    "status": "P5-score-blind-quota-census",
    "pixelsRead": True,
    "h3PixelsRead": False,
    "historyRows": 442_780,
    "rejects": 478_408,
    "rejectLayers": {
        "canonical-identity": 314_376,
        "encoded-bytes-sha256": 1_802,
        "dhash-hamming<=8": 162_230,
    },
    "clean": {"setTrain": 150_474, "setValidation": 148_107, "oodTest": 77_951},
    "labels": {
        "setTrain": {"real": 50_662, "synthetic": 62_804, "tampered": 37_008},
        "setValidation": {"real": 54_843, "synthetic": 61_289, "tampered": 31_975},
        "oodTest": {"real": 23_557, "synthetic": 28_693, "tampered": 25_701},
    },
    "acceptance": {
        "setValidationSynthetic": 58_228,
        "oodSynthetic": 28_693,
        "aigenimages2026": 559,
        "taste": 644,
        "x-aigd": 2_419,
        "nano-banana": 9_457,
        "total": 100_000,
    },
}


def load_p5_quota_census(path: Path = QUOTA_CENSUS_PATH) -> dict[str, Any]:
    value = parse_json_bytes(path.read_bytes(), label="P5 quota census")
    if value != EXPECTED_QUOTA_CENSUS:
        raise ValueError("P5 quota census content changed")
    return value


def load_p5_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    value = parse_json_bytes(path.read_bytes(), label="P5 protocol")
    if value != expected_protocol():
        raise ValueError("P5 protocol content changed")
    load_p5_quota_census()
    return value


def _json_training_contract() -> dict[str, Any]:
    return {
        "accumulation": 2, "backboneLearningRate": 2e-5,
        "batchSize": 64, "blocks": list(range(6, 12)),
        "branches": list(BRANCHES), "classifierLearningRate": 2e-4,
        "clip": 1.0, "consistencyMse": 0.05, "snapshots": list(SNAPSHOT_EPOCHS),
        "totalEpochs": 6,
        "loss": "source-class-balanced-BCE", "margin": 1.0,
        "marginCoefficient": 0.2, "optimizer": "AdamW",
        "precision": "bfloat16", "scheduler": "cosine",
        "seed": 20_260_816, "teacherFreshRowsOnly": True,
        "sourceKeyCounts": TRAINING_SOURCE_KEY_COUNTS,
        "labelCounts": TRAINING_LABEL_COUNTS,
        "warmup": 0.05, "weightDecay": 0.05,
    }


def _selector_contract() -> dict[str, Any]:
    return {
        "evidenceBinding": {"scoreCount": 28_000, "selectionInfluence": True, "selectionLockRequired": False},
        "items": {"real": 2_000, "synthetic": 2_000},
        "sources": list(SELECTOR_GENERATORS),
        "sourceAllocation": "UTF-8 lexical cyclic round-robin",
        "views": {
            "original": {"balancedAccuracy": 0.97, "realRecall": 1.0, "sourceRecall": 0.90, "syntheticRecall": 0.94},
            "search-thumbnail": {"balancedAccuracy": 0.90, "realRecall": 1.0, "sourceRecall": 0.70, "syntheticRecall": 0.80},
            **{view: {"balancedAccuracy": 0.95, "realRecall": 1.0, "sourceRecall": 0.85, "syntheticRecall": 0.90}
               for view in VIEWS if view not in ("original", "search-thumbnail")},
        },
        "zeroFalsePositivesEveryView": True,
        "ranking": ["worstViewBADescending", "originalBADescending", "worstViewSyntheticRecallDescending", "worstSourceViewRecallDescending", "branchIndexAscending", "epochAscending"],
        "noPasser": "terminal-failure-acceptance-unread",
    }


def _synthetic_acceptance_contract() -> dict[str, Any]:
    return {
        "evidenceBinding": {"scoreCount": 700_000, "selectionInfluence": False, "selectionLockRequired": True},
        "components": SYNTHETIC_COMPONENTS,
        "batches": 1_000,
        "batchSize": 100,
        "batchEnumeration": "UTF-8 lexical cyclic source round-robin then contiguous batches",
        "original": {"overallRecallStrictlyGreaterThan": 0.95, "meanBatchRecallStrictlyGreaterThan": 0.95, "medianBatchRecallStrictlyGreaterThan": 0.95},
        "frontier": {"components": FRONTIER_COMPONENTS, "items": 13_079, "aggregateRecallEveryView": 0.90, "cohortRecallEveryView": 0.85},
        "failureConsumesPanel": True,
    }


def _real_acceptance_contract() -> dict[str, Any]:
    return {
        "evidenceBinding": {"scoreCount": 70_000, "selectionInfluence": False, "selectionLockRequired": True},
        "components": {"oodReddit": 5_000, "setValidationReal": 5_000},
        "setPrefixes": list(REAL_PREFIXES),
        "views": list(VIEWS),
        "aggregate": {"maximumFpr": 0.01, "maximumWilson95Upper": 0.02},
        "cohort": {"maximumFpr": 0.03, "maximumWilson95Upper": 0.05},
        "failureConsumesPanel": True,
    }


def _paid_contract() -> dict[str, Any]:
    return {
        "provider": "RunPod", "cloudType": "SECURE", "gpu": "NVIDIA L40S",
        "gpuCount": 1, "volumeGb": 300, "volumeMount": "/workspace",
        "preflightMaximumUsd": 1.0, "gpuMaximumUsd": 24.0,
        "allInMaximumUsd": 30.0, "attempts": 1,
        "autoTrainOnlyAfterPreflightPass": True,
        "retryOrAlternativeGpuRequiresNewApproval": True,
    }


def expected_protocol() -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "name": "seroslop-m6-p5-public-green",
        "basePublicHead": "05a131b64fdef5f7fe8a6bdad4dac6d401e8193a",
        "status": "metadata-only; no acquisition or materialization",
        "activation": {
            "status": "P6-exact-public-trust-anchors-required",
            "required": ["sourceCommit", "sourceTree", "admissionLedger", "historicalCoverageReceipt", "overlapManifest", "allocationReceipt", "sourceLockReceipt"],
            "p5ProductionEntrypoints": "hard-stop",
        },
        "sources": {
            "frontier": FRONTIER_SOURCE_SPECS,
            "existing": {
                "omniFakeSet": {"dataset": "JamalLee/Omni-Fake-SET", "revision": "724e97f5fc9f4b89f59631a8d4e6331712b7d441", "roles": ["calibration", "balanced-training", "selector", "synthetic-acceptance", "real-acceptance"]},
                "omniFakeOOD": {"dataset": "JamalLee/Omni-Fake-OOD", "revision": "9ed7e38bbdb4aeb2eb553896a5890680a9ffcf17", "roles": ["synthetic-acceptance", "real-acceptance"], "training": False},
            },
            "excluded": sorted(FORBIDDEN_SOURCES),
        },
        "identity": {
            "fields": ["dataset", "pinnedRevision", "partition", "explicitSourceGroup", "publisherOrdinalOrPinnedMemberPath"],
            "encoding": "uint64-big-endian byte length followed by UTF-8 bytes",
            "collision": "quarantine-and-fail",
        },
        "allocation": {
            "sourceOrder": "UTF-8 byte lexical",
            "rowOrder": "canonical identity bytes",
            "algorithm": "cyclic source round-robin; remove exhausted source",
            "precedence": list(COHORT_ORDER),
            "historicalRejectsFirst": True,
            "shortage": "terminal",
            "manualOrScoreSelection": False,
        },
        "historical": {
            "rows": 442_780,
            "cohortRows": {"h3": 600, "m2": 106_878, "m3": 108_978, "m4": 113_162, "m5": 113_162},
            "requiredLayers": ["canonical-identity", "encoded-bytes-sha256"],
            "optionalWhenRecorded": ["decoded-rgb-sha256", "dhash-hamming<=8"],
            "rejectReceipt": "bind verified admission ledger, exact historical normalization, sorted reject identity set, layer counts, and manifest hashes",
            "h3PixelsRead": False,
        },
        "domains": {"allocation": ALLOCATION_DOMAIN.decode(), "platt": PLATT_DOMAIN, "ranking": RANKING_DOMAIN, "view": VIEW_DOMAIN.decode()},
        "views": list(VIEWS),
        "transforms": {
            "pillow": "11.3.0",
            "decode": "EXIF-transpose then RGB",
            "dhash": "LANCZOS 9x8 grayscale row-major MSB-first",
            "metricViews": "literal fixture hashes and dimensions in p5_transform_fixture.py",
            "viewPolicies": {
                "original": "decoded RGB unchanged",
                "screenshot": "8 percent light border",
                "social-q75": "JPEG quality 75 subsampling 4:2:0",
                "social-heavy": "75 percent LANCZOS downscale; JPEG quality 55 4:2:0; BICUBIC restore; sharpen",
                "forum-repost": "3 percent contrast; 4 percent light border; JPEG quality 82 4:2:0",
                "search-thumbnail": "50 percent LANCZOS thumbnail; BILINEAR restore",
                "provider-cdn": "JPEG quality 82 subsampling 4:2:0",
            },
            "trainingChoice": "SHA256 length-prefixed seed,branch,epoch,rowId; unsigned big-endian modulo six stress views",
            "googlePrivateDistributionClaimed": False,
        },
        "calibration": {
            "armijo": 1e-4, "dampingInitial": 1e-12, "dampingMaximum": 1e12,
            "dampingMultiplier": 10, "gradientInfTolerance": 1e-12,
            "initial": [1.0, 0.0], "maximumBacktracks": 32,
            "maximumIterations": 100, "regularization": 5e-7,
            "scoreThresholdInclusive": 65.0, "stepInfTolerance": 1e-12,
            "receipt": "recompute logits labels source weights and float-hex parameters; selector pixels unread",
        },
        "training": _json_training_contract(),
        "selector": _selector_contract(),
        "syntheticAcceptance": _synthetic_acceptance_contract(),
        "realAcceptance": _real_acceptance_contract(),
        "paid": _paid_contract(),
        "claims": {
            "publisherMetadataIsGroundTruthBoundary": True,
            "metaphysicalAuthenticityClaimed": False,
            "commercialRightsClearanceClaimed": False,
            "ambiguousRows": "quarantine",
            "sourcePixelsRedistributed": False,
            "h3PixelsRead": False,
        },
    }


def candidate_only(_: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    return False, {"status": "candidate-unverified", "reason": "encoded-bytes-and-publisher-record-required"}


def verify_encoded_row(inventory: Mapping[str, Any], publisher: Mapping[str, Any], encoded: bytes) -> tuple[bool, dict[str, Any]]:
    """Verify one extracted record. P6 must derive both records from its pinned container."""
    inventory_keys = {
        "sourceKey", "dataset", "publisher", "revision", "partition", "role",
        "label", "sourceGroup", "locatorKind", "locatorValue", "containerPath",
        "containerBytes", "containerSha256", "encodedBytes", "encodedSha256",
    }
    publisher_keys = {
        "sourceKey", "dataset", "publisher", "revision", "partition", "role",
        "label", "sourceGroup", "locatorKind", "locatorValue",
    }
    identity = {key: inventory.get(key) for key in ("sourceKey", "locatorKind", "locatorValue")} if isinstance(inventory, Mapping) else {}
    fail = lambda reason: (False, {"status": "quarantined", "identity": identity, "reason": reason})
    if not isinstance(inventory, Mapping) or set(inventory) != inventory_keys:
        return fail("inventory-schema")
    if not isinstance(publisher, Mapping) or set(publisher) != publisher_keys:
        return fail("publisher-schema")
    spec = FRONTIER_SOURCE_SPECS.get(inventory["sourceKey"])
    if spec is None:
        return fail("source-not-allowed")
    fixed = ("dataset", "publisher", "revision", "partition", "role", "label")
    if any(inventory[key] != spec[key] or publisher[key] != spec[key] for key in fixed):
        return fail("source-pin-mismatch")
    shared = ("sourceKey", "dataset", "publisher", "revision", "partition", "role", "label", "sourceGroup", "locatorKind", "locatorValue")
    if any(inventory[key] != publisher[key] for key in shared):
        return fail("publisher-inventory-conflict")
    if not isinstance(inventory["sourceGroup"], str) or not inventory["sourceGroup"]:
        return fail("source-group-missing")
    if inventory["locatorKind"] not in ("ordinal", "memberPath"):
        return fail("locator-kind")
    if inventory["locatorKind"] == "ordinal":
        if type(inventory["locatorValue"]) is not int or inventory["locatorValue"] < 0:
            return fail("locator-value")
    elif not isinstance(inventory["locatorValue"], str) or not inventory["locatorValue"] or inventory["locatorValue"].startswith("/") or ".." in inventory["locatorValue"].split("/"):
        return fail("locator-value")
    if spec["containerSha256"] is None:
        return fail("p6-exact-container-inventory-required")
    if any(inventory[key] != spec[key] for key in ("containerPath", "containerBytes", "containerSha256")):
        return fail("container-pin-mismatch")
    if type(encoded) is not bytes or not encoded:
        return fail("encoded-bytes-missing")
    if type(inventory["encodedBytes"]) is not int or inventory["encodedBytes"] <= 0 or len(encoded) != inventory["encodedBytes"]:
        return fail("encoded-size-mismatch")
    if not isinstance(inventory["encodedSha256"], str) or not HEX64.fullmatch(inventory["encodedSha256"]) or sha256(encoded).hexdigest() != inventory["encodedSha256"]:
        return fail("encoded-sha256-mismatch")
    try:
        encoded_sha, decoded_sha, dhash, width, height = image_facts(encoded)
    except Exception:
        return fail("image-decode-failed")
    identity_bytes_value = canonical_identity({
        "dataset": inventory["dataset"], "revision": inventory["revision"],
        "partition": inventory["partition"], "sourceGroup": inventory["sourceGroup"],
        "locatorKind": inventory["locatorKind"], "locatorValue": inventory["locatorValue"],
    })
    body = {
        "status": "admitted", "sourceKey": inventory["sourceKey"],
        "role": inventory["role"], "label": inventory["label"],
        "sourceGroup": inventory["sourceGroup"],
        "identitySha256": sha256(identity_bytes_value).hexdigest(),
        "inventorySha256": sha256(canonical_json(dict(inventory))).hexdigest(),
        "publisherSha256": sha256(canonical_json(dict(publisher))).hexdigest(),
        "encodedSha256": encoded_sha, "decodedRgbSha256": decoded_sha,
        "dhash64": dhash, "width": width, "height": height,
    }
    return True, {**body, "receiptSha256": sha256(canonical_json(body)).hexdigest()}


def canonical_identity(row: Mapping[str, Any]) -> bytes:
    _require_keys(row, {"dataset", "revision", "partition", "sourceGroup", "locatorKind", "locatorValue"}, "allocation identity")
    for key in ("dataset", "revision", "partition", "sourceGroup", "locatorKind"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError("allocation identity field invalid")
    if row["locatorKind"] == "ordinal":
        if type(row["locatorValue"]) is not int or row["locatorValue"] < 0:
            raise ValueError("allocation ordinal invalid")
        locator = f'ordinal:{row["locatorValue"]}'
    elif row["locatorKind"] == "memberPath":
        locator_value = row["locatorValue"]
        if not isinstance(locator_value, str) or not locator_value or locator_value.startswith("/") or ".." in locator_value.split("/"):
            raise ValueError("allocation member path invalid")
        locator = f"memberPath:{locator_value}"
    else:
        raise ValueError("allocation locator kind invalid")
    fields: Sequence[str] = (
        row["dataset"], row["revision"], row["partition"],
        row["sourceGroup"], locator,
    )
    if any(any(ord(character) < 32 or ord(character) == 127 for character in value) for value in fields):
        raise ValueError("allocation identity field empty")
    return ALLOCATION_DOMAIN + b"".join(struct.pack(">Q", len(value.encode("utf-8"))) + value.encode("utf-8") for value in fields)


def _verify_admitted_row(row: Mapping[str, Any], verified_receipt_sha256: set[str]) -> bytes:
    _require_keys(row, {"dataset", "revision", "partition", "sourceKey", "sourceGroup", "locatorKind", "locatorValue", "label", "admission"}, "allocation row")
    if row["sourceKey"] not in ALLOCATION_SOURCE_KEYS:
        raise ValueError("allocation source not allowed")
    source_spec = ALLOCATION_SOURCE_SPECS[row["sourceKey"]]
    if any(row[key] != source_spec[key] for key in ("dataset", "revision", "partition", "label")):
        raise ValueError("allocation source binding changed")
    receipt = row["admission"]
    receipt_keys = {"status", "sourceKey", "role", "label", "sourceGroup", "identitySha256", "inventorySha256", "publisherSha256", "encodedSha256", "decodedRgbSha256", "dhash64", "width", "height", "receiptSha256"}
    _require_keys(receipt, receipt_keys, "admission receipt")
    body = dict(receipt)
    receipt_sha = body.pop("receiptSha256")
    if receipt["status"] != "admitted" or not HEX64.fullmatch(str(receipt_sha)) or sha256(canonical_json(body)).hexdigest() != receipt_sha:
        raise ValueError("admission receipt digest/status invalid")
    if receipt_sha not in verified_receipt_sha256:
        raise ValueError("admission receipt is absent from verified ledger")
    for key in ("identitySha256", "inventorySha256", "publisherSha256", "encodedSha256", "decodedRgbSha256"):
        _hex(receipt[key], HEX64, f"admission {key}")
    _hex(receipt["dhash64"], HEX16, "admission dHash")
    _positive_int(receipt["width"], "admission width")
    _positive_int(receipt["height"], "admission height")
    if receipt["role"] not in source_spec["roles"]:
        raise ValueError("admission role changed")
    if receipt["sourceKey"] != row["sourceKey"] or receipt["sourceGroup"] != row["sourceGroup"] or receipt["label"] != row["label"]:
        raise ValueError("admission receipt row binding changed")
    identity = canonical_identity({key: row[key] for key in ("dataset", "revision", "partition", "sourceGroup", "locatorKind", "locatorValue")})
    if sha256(identity).hexdigest() != receipt["identitySha256"]:
        raise ValueError("admission identity binding changed")
    return identity


def validate_historical_coverage(receipt: Mapping[str, Any]) -> None:
    _require_keys(receipt, {"schemaVersion", "status", "cohortRowCounts", "h3PixelsRead", "manifests", "normalizedExpandedSha256", "normalizedRows"}, "historical receipt")
    if receipt["schemaVersion"] != 1 or receipt["status"] != "m6-historical-metadata-locked" or receipt["h3PixelsRead"] is not False:
        raise ValueError("historical receipt boundary changed")
    if receipt["cohortRowCounts"] != {"h3": 600, "m2": 106_878, "m3": 108_978, "m4": 113_162, "m5": 113_162}:
        raise ValueError("historical cohort counts changed")
    if receipt["normalizedRows"] != 442_780 or receipt["normalizedExpandedSha256"] != "ea324b93c072332a19c9fc8256084fb2c7c2d82d74f31b10c5243b8c912661b7":
        raise ValueError("historical normalized evidence changed")
    from benchmark.m6.historical import HISTORY_SPECS
    expected_manifests = [
        {"bytes": byte_count, "cohort": cohort, "path": path, "rows": rows, "sha256": digest}
        for cohort in ("h3", "m2", "m3", "m4", "m5")
        for path, byte_count, digest, rows, _kind in HISTORY_SPECS[cohort]
    ]
    if receipt["manifests"] != expected_manifests:
        raise ValueError("historical manifest inventory changed")


COHORT_ORDER = ("calibration", "balanced-training", "selector", "synthetic-acceptance", "real-acceptance")


def validate_overlap_reject_receipt(receipt: Mapping[str, Any], verified_receipt_sha256: set[str], historical_reject_identity_sha256: set[str]) -> None:
    _require_keys(receipt, {
        "schemaVersion", "status", "historyNormalizedExpandedSha256",
        "admissionLedgerSha256", "rejectIdentitySetSha256", "rejectManifestSha256",
        "rejectManifestExpandedSha256", "rejectCount", "layerCounts", "receiptSha256",
    }, "overlap reject receipt")
    if not isinstance(verified_receipt_sha256, set) or not verified_receipt_sha256 or any(not HEX64.fullmatch(value) for value in verified_receipt_sha256):
        raise ValueError("verified admission ledger invalid")
    if not isinstance(historical_reject_identity_sha256, set) or any(not HEX64.fullmatch(value) for value in historical_reject_identity_sha256):
        raise ValueError("historical reject ledger invalid")
    body = dict(receipt)
    receipt_sha = body.pop("receiptSha256")
    if (
        receipt["schemaVersion"] != 1
        or receipt["status"] != "m6-overlap-rejects-locked"
        or receipt["historyNormalizedExpandedSha256"] != "ea324b93c072332a19c9fc8256084fb2c7c2d82d74f31b10c5243b8c912661b7"
        or receipt["admissionLedgerSha256"] != sha256(canonical_json(sorted(verified_receipt_sha256))).hexdigest()
        or receipt["rejectIdentitySetSha256"] != sha256(canonical_json(sorted(historical_reject_identity_sha256))).hexdigest()
        or receipt["rejectCount"] != len(historical_reject_identity_sha256)
        or not HEX64.fullmatch(str(receipt_sha))
        or sha256(canonical_json(body)).hexdigest() != receipt_sha
    ):
        raise ValueError("overlap reject receipt binding changed")
    for key in ("rejectManifestSha256", "rejectManifestExpandedSha256"):
        _hex(receipt[key], HEX64, key)
    expected_layers = {"canonical-identity", "encoded-bytes-sha256", "decoded-rgb-sha256", "dhash-hamming<=8"}
    if set(receipt["layerCounts"]) != expected_layers or any(type(value) is not int or value < 0 for value in receipt["layerCounts"].values()) or sum(receipt["layerCounts"].values()) != receipt["rejectCount"]:
        raise ValueError("overlap reject layers changed")


def _allocate_all_cohorts_core(rows: Iterable[Mapping[str, Any]], verified_receipt_sha256: set[str], historical_reject_identity_sha256: set[str], reject_receipt: Mapping[str, Any], reservations: Mapping[str, Mapping[str, int]]) -> dict[str, list[Mapping[str, Any]]]:
    if set(reservations) != set(COHORT_ORDER):
        raise ValueError("allocation cohorts changed")
    validate_overlap_reject_receipt(reject_receipt, verified_receipt_sha256, historical_reject_identity_sha256)
    available: dict[tuple[str, str], list[tuple[bytes, Mapping[str, Any]]]] = defaultdict(list)
    seen: set[bytes] = set()
    for row in rows:
        identity = _verify_admitted_row(row, verified_receipt_sha256)
        if identity in seen:
            raise ValueError("allocation identity collision")
        seen.add(identity)
        if sha256(identity).hexdigest() in historical_reject_identity_sha256:
            continue
        available[(str(row["admission"]["role"]), str(row["sourceGroup"]))].append((identity, row))
    for bucket in available.values():
        bucket.sort(key=lambda pair: pair[0])
    output: dict[str, list[Mapping[str, Any]]] = {}
    for cohort in COHORT_ORDER:
        quotas = reservations[cohort]
        if not isinstance(quotas, Mapping) or not quotas or any(not isinstance(source, str) or not source or type(count) is not int or count <= 0 for source, count in quotas.items()):
            raise ValueError("allocation quota invalid")
        chosen: list[Mapping[str, Any]] = []
        source_order = sorted(quotas, key=lambda value: value.encode("utf-8"))
        used = Counter()
        while len(chosen) < sum(quotas.values()):
            progressed = False
            for source in source_order:
                if used[source] >= quotas[source]:
                    continue
                bucket = available.get((cohort, source))
                if bucket:
                    _, row = bucket.pop(0)
                    chosen.append(row)
                    used[source] += 1
                    progressed = True
            if not progressed:
                raise ValueError(f"allocation shortage in {cohort}")
        output[cohort] = chosen
    return output


def view_digest(seed: int, branch: str, epoch: int, row_id: str) -> bytes:
    if type(seed) is not int or seed < 0 or branch not in BRANCHES or epoch not in TRAINING_EPOCHS or not isinstance(row_id, str) or not row_id:
        raise ValueError("view selection input invalid")
    fields = (str(seed), branch, str(epoch), row_id)
    payload = VIEW_DOMAIN + b"".join(struct.pack(">Q", len(value.encode())) + value.encode() for value in fields)
    return sha256(payload).digest()


def training_view(seed: int, branch: str, epoch: int, row_id: str) -> str:
    return STRESS_VIEWS[int.from_bytes(view_digest(seed, branch, epoch, row_id), "big") % len(STRESS_VIEWS)]


def calibration_weights(labels: Iterable[str], sources: Iterable[str]) -> list[float]:
    label_values, source_values = list(labels), list(sources)
    if len(label_values) != len(source_values) or not label_values:
        raise ValueError("calibration rows invalid")
    if any(label not in ("real", "synthetic") for label in label_values) or any(not isinstance(source, str) or not source for source in source_values):
        raise ValueError("calibration label/source invalid")
    pair_counts = Counter(zip(label_values, source_values))
    sources_by_class = {label: {source for observed_label, source in pair_counts if observed_label == label} for label in ("real", "synthetic")}
    if not all(sources_by_class.values()):
        raise ValueError("calibration requires both classes")
    total = len(label_values)
    return [total / (2.0 * len(sources_by_class[label]) * pair_counts[(label, source)]) for label, source in zip(label_values, source_values)]


def fit_platt(logits: Iterable[float], labels: Iterable[int], weights: Iterable[float]) -> tuple[float, float]:
    xs, ys, ws = list(logits), list(labels), list(weights)
    if len(xs) != len(ys) or len(xs) != len(ws) or not xs:
        raise ValueError("calibration vector sizes changed")
    if any(type(value) is not float or not math.isfinite(value) for value in xs + ws) or any(weight <= 0 for weight in ws):
        raise ValueError("calibration numeric input invalid")
    if any(type(label) is not int or label not in (0, 1) for label in ys):
        raise ValueError("calibration labels invalid")
    a, b = 1.0, 0.0
    damping = 1e-12

    def objective(aa: float, bb: float) -> float:
        value = 0.5e-6 * ((aa - 1.0) ** 2 + bb ** 2)
        for x, y, weight in zip(xs, ys, ws):
            z = aa * x + bb
            value += weight * (max(z, 0.0) - y * z + math.log1p(math.exp(-abs(z))))
        return value

    for _ in range(100):
        ga, gb = 1e-6 * (a - 1.0), 1e-6 * b
        haa, hab, hbb = 1e-6, 0.0, 1e-6
        for x, y, weight in zip(xs, ys, ws):
            z = max(-709.0, min(709.0, a * x + b))
            probability = 1.0 / (1.0 + math.exp(-z))
            gradient = weight * (probability - y)
            curvature = weight * probability * (1.0 - probability)
            ga += gradient * x
            gb += gradient
            haa += curvature * x * x
            hab += curvature * x
            hbb += curvature
        gradient_inf = max(abs(ga), abs(gb))
        trial_damping = damping
        accepted_step: tuple[float, float, float] | None = None
        while trial_damping <= 1e12 and accepted_step is None:
            damped_aa, damped_bb = haa + trial_damping, hbb + trial_damping
            determinant = damped_aa * damped_bb - hab * hab
            if not math.isfinite(determinant) or determinant <= 0:
                trial_damping *= 10.0
                continue
            da = -(damped_bb * ga - hab * gb) / determinant
            db = -(-hab * ga + damped_aa * gb) / determinant
            directional = ga * da + gb * db
            if not math.isfinite(directional) or directional >= 0:
                trial_damping *= 10.0
                continue
            scale = 1.0
            current = objective(a, b)
            for _ in range(32):
                next_a, next_b = a + scale * da, b + scale * db
                if next_a > 0 and math.isfinite(next_a) and math.isfinite(next_b) and objective(next_a, next_b) <= current + 1e-4 * scale * directional:
                    accepted_step = (next_a, next_b, max(abs(scale * da), abs(scale * db)))
                    break
                scale *= 0.5
            if accepted_step is None:
                trial_damping *= 10.0
        if accepted_step is None:
            raise ValueError("Platt optimizer has no acceptable step")
        a, b, step_inf = accepted_step
        damping = trial_damping
        if gradient_inf <= 1e-12 and step_inf <= 1e-12:
            return a, b
    raise ValueError("Platt optimizer did not converge")


def platt_score(logit: float, a: float, b: float) -> float:
    for value in (logit, a, b):
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("Platt score input invalid")
    if a <= 0:
        raise ValueError("Platt slope must be positive")
    z = max(-709.0, min(709.0, a * logit + b))
    return 100.0 / (1.0 + math.exp(-z))


def _float64_sha256(values: Sequence[float]) -> str:
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("float64 evidence contains a nonfinite value")
    return sha256(b"".join(struct.pack(">d", value) for value in values)).hexdigest()


def _validate_calibration_receipt_core(receipt: Mapping[str, Any], logits: Iterable[float], labels: Iterable[int], sources: Iterable[str]) -> None:
    _require_keys(receipt, {
        "schemaVersion", "status", "sourceLockReceiptSha256", "modelSha256",
        "modelBytes", "realCount", "syntheticCount", "logitsSha256",
        "labelsSha256", "sourcesSha256", "weightsSha256", "aHex", "bHex",
        "threshold", "method", "selectorPixelsRead", "h3PixelsRead",
    }, "calibration receipt")
    xs, ys, source_values = list(logits), list(labels), list(sources)
    if len(xs) != 4_000 or len(ys) != 4_000 or len(source_values) != 4_000:
        raise ValueError("calibration production census changed")
    if Counter(ys) != Counter({0: 2_000, 1: 2_000}):
        raise ValueError("calibration class census changed")
    real_sources = Counter(source for label, source in zip(ys, source_values) if label == 0)
    synthetic_sources = Counter(source for label, source in zip(ys, source_values) if label == 1)
    if real_sources != Counter(round_robin_counts(REAL_PREFIXES, 2_000)) or synthetic_sources != Counter(round_robin_counts(SELECTOR_GENERATORS, 2_000)):
        raise ValueError("calibration source census changed")
    label_names = ["synthetic" if label == 1 else "real" for label in ys]
    weights = calibration_weights(label_names, source_values)
    fitted_a, fitted_b = fit_platt(xs, ys, weights)
    if (
        receipt["schemaVersion"] != 1
        or receipt["status"] != "m6-calibration-fit"
        or receipt["realCount"] != 2_000
        or receipt["syntheticCount"] != 2_000
        or receipt["threshold"] != 65.0
        or receipt["method"] != PLATT_DOMAIN
        or receipt["selectorPixelsRead"] is not False
        or receipt["h3PixelsRead"] is not False
    ):
        raise ValueError("calibration boundary changed")
    for key in ("sourceLockReceiptSha256", "modelSha256"):
        _hex(receipt[key], HEX64, key)
    if _positive_int(receipt["modelBytes"], "calibration model bytes") > 90_000_000:
        raise ValueError("calibration model size changed")
    if receipt["logitsSha256"] != _float64_sha256(xs):
        raise ValueError("calibration logits changed")
    if receipt["labelsSha256"] != sha256(bytes(ys)).hexdigest():
        raise ValueError("calibration labels changed")
    if receipt["sourcesSha256"] != sha256(canonical_json(source_values)).hexdigest():
        raise ValueError("calibration sources changed")
    if receipt["weightsSha256"] != _float64_sha256(weights):
        raise ValueError("calibration weights changed")
    if receipt["aHex"] != fitted_a.hex() or receipt["bHex"] != fitted_b.hex():
        raise ValueError("calibration fit changed")


def decision(score: float) -> bool:
    if type(score) is not float or not math.isfinite(score):
        raise ValueError("decision score invalid")
    return score >= 65.0


def _confusion(counts: Mapping[str, Any], *, require_both_classes: bool) -> dict[str, float]:
    _require_keys(counts, {"tp", "tn", "fp", "fn"}, "confusion counts")
    tp = _positive_int(counts["tp"], "tp", allow_zero=True)
    tn = _positive_int(counts["tn"], "tn", allow_zero=True)
    fp = _positive_int(counts["fp"], "fp", allow_zero=True)
    fn = _positive_int(counts["fn"], "fn", allow_zero=True)
    real, synthetic = tn + fp, tp + fn
    if require_both_classes and (real == 0 or synthetic == 0):
        raise ValueError("confusion matrix must contain both classes")
    return {
        "real": real, "synthetic": synthetic,
        "realRecall": tn / real if real else math.nan,
        "syntheticRecall": tp / synthetic if synthetic else math.nan,
        "balancedAccuracy": 0.5 * (tn / real + tp / synthetic) if real and synthetic else math.nan,
        "fpr": fp / real if real else math.nan,
    }


def wilson_upper(successes: int, total: int, z: float = 1.959963984540054) -> float:
    successes = _positive_int(successes, "Wilson successes", allow_zero=True)
    total = _positive_int(total, "Wilson total")
    if successes > total:
        raise ValueError("Wilson successes exceed total")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return min(1.0, center + half)


def round_robin_counts(names: Sequence[str], total: int) -> dict[str, int]:
    if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("round-robin source names invalid")
    total = _positive_int(total, "round-robin total")
    ordered = sorted(names, key=lambda value: value.encode("utf-8"))
    quotient, remainder = divmod(total, len(ordered))
    return {name: quotient + (index < remainder) for index, name in enumerate(ordered)}


def expected_synthetic_batch_counts() -> list[dict[str, int]]:
    remaining = Counter(SYNTHETIC_COMPONENTS)
    order = sorted(SYNTHETIC_COMPONENTS, key=lambda value: value.encode("utf-8"))
    sequence: list[str] = []
    while remaining:
        progressed = False
        for source in order:
            if remaining[source] > 0:
                sequence.append(source)
                remaining[source] -= 1
                progressed = True
                if remaining[source] == 0:
                    del remaining[source]
        if not progressed:
            raise AssertionError("synthetic batch enumeration stalled")
    return [dict(Counter(sequence[index:index + 100])) for index in range(0, len(sequence), 100)]


def _validate_score_binding(binding: Mapping[str, Any], *, status: str, score_count: int, selection_influence: bool, selection_lock: bool) -> None:
    expected = {
        "status", "sourceLockReceiptSha256", "modelSha256", "modelBytes",
        "calibrationReceiptSha256", "threshold", "scoreSha256", "scoreCount",
        "selectionInfluence", "h3PixelsRead",
    }
    if selection_lock:
        expected.add("selectionLockReceiptSha256")
    _require_keys(binding, expected, "score binding")
    if binding["status"] != status or binding["threshold"] != 65.0 or binding["scoreCount"] != score_count or binding["selectionInfluence"] is not selection_influence or binding["h3PixelsRead"] is not False:
        raise ValueError("score binding boundary changed")
    for key in ("sourceLockReceiptSha256", "modelSha256", "calibrationReceiptSha256", "scoreSha256"):
        _hex(binding[key], HEX64, key)
    if selection_lock:
        _hex(binding["selectionLockReceiptSha256"], HEX64, "selection lock receipt")
    model_bytes = _positive_int(binding["modelBytes"], "model bytes")
    if model_bytes > 90_000_000:
        raise ValueError("model size gate failed")


def _validate_selector_counts_core(panel: Mapping[str, Any]) -> None:
    _require_keys(panel, {"binding", "realCount", "syntheticCount", "syntheticBySource", "duplicates", "missing", "errors", "views"}, "selector panel")
    _validate_score_binding(panel["binding"], status="m6-selector-scored", score_count=28_000, selection_influence=True, selection_lock=False)
    if panel["realCount"] != 2_000 or panel["syntheticCount"] != 2_000 or any(panel[key] != 0 for key in ("duplicates", "missing", "errors")):
        raise ValueError("selector census changed")
    if panel["syntheticBySource"] != round_robin_counts(SELECTOR_GENERATORS, 2_000):
        raise ValueError("selector source census changed")
    if set(panel["views"]) != set(VIEWS):
        raise ValueError("selector views changed")
    contract = _selector_contract()["views"]
    for view in VIEWS:
        record = panel["views"][view]
        _require_keys(record, {"overall", "sources"}, "selector view")
        metrics = _confusion(record["overall"], require_both_classes=True)
        if metrics["real"] != 2_000 or metrics["synthetic"] != 2_000 or record["overall"]["fp"] != 0:
            raise ValueError("selector view census/false-positive changed")
        minimum = contract[view]
        if metrics["balancedAccuracy"] < minimum["balancedAccuracy"] or metrics["realRecall"] < 1.0 or metrics["syntheticRecall"] < minimum["syntheticRecall"]:
            raise ValueError("selector view gate failed")
        if set(record["sources"]) != set(SELECTOR_GENERATORS):
            raise ValueError("selector per-source evidence changed")
        source_tp = source_fn = 0
        for source in SELECTOR_GENERATORS:
            counts = record["sources"][source]
            _require_keys(counts, {"tp", "fn"}, "selector source counts")
            tp = _positive_int(counts["tp"], "selector source tp", allow_zero=True)
            fn = _positive_int(counts["fn"], "selector source fn", allow_zero=True)
            if tp + fn != panel["syntheticBySource"][source] or tp / (tp + fn) < minimum["sourceRecall"]:
                raise ValueError("selector source gate failed")
            source_tp += tp
            source_fn += fn
        if source_tp != record["overall"]["tp"] or source_fn != record["overall"]["fn"]:
            raise ValueError("selector source totals do not reconcile")


def _validate_synthetic_acceptance_core(panel: Mapping[str, Any]) -> None:
    _require_keys(panel, {"binding", "components", "duplicates", "missing", "errors", "batches", "views"}, "synthetic acceptance")
    _validate_score_binding(panel["binding"], status="m6-synthetic-acceptance-scored", score_count=700_000, selection_influence=False, selection_lock=True)
    if panel["components"] != SYNTHETIC_COMPONENTS or any(panel[key] != 0 for key in ("duplicates", "missing", "errors")):
        raise ValueError("synthetic acceptance census changed")
    if not isinstance(panel["batches"], list) or len(panel["batches"]) != 1_000:
        raise ValueError("synthetic batch count changed")
    expected_batches = expected_synthetic_batch_counts()
    accumulated = Counter()
    batch_recalls: list[float] = []
    for index, batch in enumerate(panel["batches"]):
        _require_keys(batch, {"index", "sourceCounts", "original"}, "synthetic batch")
        if batch["index"] != index or batch["sourceCounts"] != expected_batches[index]:
            raise ValueError("synthetic batch composition changed")
        accumulated.update(batch["sourceCounts"])
        original = batch["original"]
        _require_keys(original, {"tp", "fn"}, "synthetic batch original")
        tp = _positive_int(original["tp"], "batch tp", allow_zero=True)
        fn = _positive_int(original["fn"], "batch fn", allow_zero=True)
        if tp + fn != 100:
            raise ValueError("synthetic batch score count changed")
        batch_recalls.append(tp / 100.0)
    if dict(accumulated) != SYNTHETIC_COMPONENTS:
        raise ValueError("synthetic batch totals do not reconcile")
    if set(panel["views"]) != set(VIEWS):
        raise ValueError("synthetic acceptance views changed")
    original_overall: float | None = None
    for view in VIEWS:
        view_record = panel["views"][view]
        _require_keys(view_record, {"sources"}, "synthetic acceptance view")
        if set(view_record["sources"]) != set(SYNTHETIC_COMPONENTS):
            raise ValueError("synthetic source evidence changed")
        true_positive = false_negative = 0
        frontier_true = frontier_false = 0
        for source, expected in SYNTHETIC_COMPONENTS.items():
            counts = view_record["sources"][source]
            _require_keys(counts, {"tp", "fn"}, "synthetic source counts")
            tp = _positive_int(counts["tp"], "synthetic tp", allow_zero=True)
            fn = _positive_int(counts["fn"], "synthetic fn", allow_zero=True)
            if tp + fn != expected:
                raise ValueError("synthetic source count changed")
            true_positive += tp
            false_negative += fn
            if source in FRONTIER_COMPONENTS:
                if tp / expected < 0.85:
                    raise ValueError("frontier cohort gate failed")
                frontier_true += tp
                frontier_false += fn
        if true_positive + false_negative != 100_000:
            raise ValueError("synthetic view total changed")
        if frontier_true + frontier_false != 13_079 or frontier_true / 13_079 < 0.90:
            raise ValueError("frontier aggregate gate failed")
        if view == "original":
            original_overall = true_positive / 100_000
    if original_overall is None or original_overall <= 0.95 or mean(batch_recalls) <= 0.95 or median(batch_recalls) <= 0.95:
        raise ValueError("synthetic original/batch recall gate failed")
    original_sources = panel["views"]["original"]["sources"]
    if sum(batch["original"]["tp"] for batch in panel["batches"]) != sum(counts["tp"] for counts in original_sources.values()) or sum(batch["original"]["fn"] for batch in panel["batches"]) != sum(counts["fn"] for counts in original_sources.values()):
        raise ValueError("synthetic batch/original-view totals conflict")


def _validate_real_acceptance_core(panel: Mapping[str, Any]) -> None:
    _require_keys(panel, {"binding", "components", "duplicates", "missing", "errors", "views"}, "real acceptance")
    _validate_score_binding(panel["binding"], status="m6-real-acceptance-scored", score_count=70_000, selection_influence=False, selection_lock=True)
    components = panel["components"]
    _require_keys(components, {"oodReddit", "setRealPrefixes"}, "real components")
    if components["oodReddit"] != 5_000 or components["setRealPrefixes"] != round_robin_counts(REAL_PREFIXES, 5_000):
        raise ValueError("real component census changed")
    if any(panel[key] != 0 for key in ("duplicates", "missing", "errors")) or set(panel["views"]) != set(VIEWS):
        raise ValueError("real failure/view evidence changed")
    cohort_sizes = {"oodReddit": 5_000, **components["setRealPrefixes"]}
    for view in VIEWS:
        record = panel["views"][view]
        _require_keys(record, {"cohorts"}, "real view")
        if set(record["cohorts"]) != set(cohort_sizes):
            raise ValueError("real cohorts changed")
        total_fp = 0
        for cohort, expected in cohort_sizes.items():
            counts = record["cohorts"][cohort]
            _require_keys(counts, {"tn", "fp"}, "real cohort counts")
            tn = _positive_int(counts["tn"], "real tn", allow_zero=True)
            fp = _positive_int(counts["fp"], "real fp", allow_zero=True)
            if tn + fp != expected:
                raise ValueError("real cohort count changed")
            if fp / expected > 0.03 or wilson_upper(fp, expected) > 0.05:
                raise ValueError("real cohort gate failed")
            total_fp += fp
        if total_fp / 10_000 > 0.01 or wilson_upper(total_fp, 10_000) > 0.02:
            raise ValueError("real aggregate gate failed")


def row_weights(rows: Iterable[Mapping[str, Any]]) -> list[float]:
    values = list(rows)
    if not values:
        raise ValueError("training rows empty")
    labels = [str(row["label"]) for row in values]
    sources = [str(row["sourceGroup"]) for row in values]
    return calibration_weights(labels, sources)


def _training_plan_core(rows: Iterable[Mapping[str, Any]], verified_receipt_sha256: set[str]) -> dict[str, Any]:
    values = list(rows)
    identities = [_verify_admitted_row(row, verified_receipt_sha256) for row in values]
    if not values or any(row["sourceKey"] not in TRAINING_SOURCE_KEYS or row["label"] not in ("real", "synthetic") for row in values):
        raise ValueError("training source/label changed")
    if any(row["admission"]["role"] != "balanced-training" for row in values):
        raise ValueError("training admission role changed")
    if len(set(identities)) != len(identities):
        raise ValueError("training identity collision")
    if Counter(row["label"] for row in values) != Counter(TRAINING_LABEL_COUNTS) or Counter(row["sourceKey"] for row in values) != Counter(TRAINING_SOURCE_KEY_COUNTS):
        raise ValueError("training production census changed")
    ordered = [row for _, row in sorted(zip(identities, values), key=lambda pair: pair[0])]
    weights = row_weights(ordered)
    batches = [[sha256(_verify_admitted_row(row, verified_receipt_sha256)).hexdigest() for row in ordered[index:index + 64]] for index in range(0, len(ordered), 64)]
    schedule = {
        f"{branch}-e{epoch}": [training_view(TRAINING["seed"], branch, epoch, sha256(_verify_admitted_row(row, verified_receipt_sha256)).hexdigest()) for row in ordered]
        for branch in BRANCHES for epoch in TRAINING_EPOCHS
    }
    return {"weights": weights, "batches": batches, "views": schedule}


def rank_passers(passers: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    values = list(passers)
    required = {"worstViewBA", "originalBA", "worstViewSyntheticRecall", "worstSourceViewRecall", "branchIndex", "epoch"}
    for value in values:
        _require_keys(value, required, "candidate ranking")
        for key in required - {"branchIndex", "epoch"}:
            if _finite(value[key], key, minimum=0.0) > 1.0:
                raise ValueError("candidate ranking metric exceeds one")
        if value["branchIndex"] not in (0, 1) or value["epoch"] not in SNAPSHOT_EPOCHS:
            raise ValueError("candidate rank identity changed")
    return sorted(values, key=lambda value: (
        -value["worstViewBA"], -value["originalBA"],
        -value["worstViewSyntheticRecall"], -value["worstSourceViewRecall"],
        value["branchIndex"], value["epoch"],
    ))


def _validate_paid_receipt_core(receipt: Mapping[str, Any]) -> None:
    _require_keys(receipt, {
        "schemaVersion", "status", "provider", "cloudType", "gpuProduct",
        "gpuCount", "imageDigest", "volume", "sourceLock", "preflight",
        "rates", "createdAtUnix", "deadlineUnix", "maximumRuntimeSeconds",
        "stop", "cleanup", "attempt",
    }, "paid receipt")
    if receipt["schemaVersion"] != 1 or receipt["status"] != "m6-one-attempt-authorized" or receipt["provider"] != "RunPod" or receipt["cloudType"] != "SECURE" or receipt["gpuProduct"] != "NVIDIA L40S" or receipt["gpuCount"] != 1 or receipt["attempt"] != 1:
        raise ValueError("paid provider/attempt changed")
    if receipt["imageDigest"] != RUNPOD_IMAGE_DIGEST:
        raise ValueError("paid image digest invalid")
    volume = receipt["volume"]
    _require_keys(volume, {"id", "sizeGb", "mount"}, "paid volume")
    if volume["id"] != RUNPOD_VOLUME_ID or volume["sizeGb"] != 300 or volume["mount"] != "/workspace":
        raise ValueError("paid volume changed")
    source_lock = receipt["sourceLock"]
    _require_keys(source_lock, {"commit", "tree", "receiptSha256"}, "paid source lock")
    _hex(source_lock["commit"], HEX40, "source commit")
    _hex(source_lock["tree"], HEX40, "source tree")
    _hex(source_lock["receiptSha256"], HEX64, "source receipt")
    if source_lock["commit"] == "0" * 40 or source_lock["tree"] == "0" * 40 or source_lock["receiptSha256"] == "0" * 64:
        raise ValueError("source lock uses a null object")
    preflight = receipt["preflight"]
    _require_keys(preflight, {"status", "receiptSha256"}, "paid preflight")
    if preflight["status"] != "preflight-pass":
        raise ValueError("preflight did not pass")
    _hex(preflight["receiptSha256"], HEX64, "preflight receipt")
    if preflight["receiptSha256"] == "0" * 64:
        raise ValueError("preflight receipt uses a null digest")
    rates = receipt["rates"]
    _require_keys(rates, {"hourlyUsd", "preflightUsd", "gpuUsd", "storageUsd", "allInUsd"}, "paid rates")
    values = {key: _finite(value, key, minimum=0.0) for key, value in rates.items()}
    if values["hourlyUsd"] <= 0 or values["preflightUsd"] > 1 or values["gpuUsd"] > 24 or values["allInUsd"] > 30 or abs(values["allInUsd"] - (values["preflightUsd"] + values["gpuUsd"] + values["storageUsd"])) > 1e-9:
        raise ValueError("paid rates/caps changed")
    created = _positive_int(receipt["createdAtUnix"], "created time")
    deadline = _positive_int(receipt["deadlineUnix"], "deadline")
    if receipt["maximumRuntimeSeconds"] != 43_200 or deadline - created != 43_200:
        raise ValueError("paid deadline changed")
    stop = receipt["stop"]
    _require_keys(stop, {"operatorStopRequired", "safetySeconds", "noRetry"}, "paid stop")
    if stop != {"operatorStopRequired": True, "safetySeconds": 300, "noRetry": True}:
        raise ValueError("paid stop policy changed")
    cleanup = receipt["cleanup"]
    _require_keys(cleanup, {"stopPod", "deleteContainerDisk", "retainNetworkVolume"}, "paid cleanup")
    if cleanup != {"stopPod": True, "deleteContainerDisk": True, "retainNetworkVolume": True}:
        raise ValueError("paid cleanup policy changed")


def _p6_activation_required(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("P6 authoritative public source-lock activation is not present in P5")


# P5 is metadata-only. P6 must replace these hard stops with wrappers that pin
# its exact public commit/tree and recompute the real ledger/artifact bytes
# before invoking the corresponding private deterministic core.
allocate_all_cohorts = _p6_activation_required
validate_calibration_receipt = _p6_activation_required
validate_selector_counts = _p6_activation_required
validate_synthetic_acceptance = _p6_activation_required
validate_real_acceptance = _p6_activation_required
training_plan = _p6_activation_required
validate_paid_receipt = _p6_activation_required
