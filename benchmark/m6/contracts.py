"""Frozen, dependency-light contracts for the M6 RunPod protocol.

This module deliberately validates metadata only.  Pixel materialisation,
training and scoring are separate authenticated stages and fail closed until
an append-only source-lock is supplied.
"""
from __future__ import annotations

import json
import math
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
CENSUS_PATH = ROOT / "benchmark/m6/census-evidence.json"
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")
GENERATORS = ("Absolute_Reality", "AttGAN", "BEGAN", "CramerGAN", "DiT10k", "FLUX_kon", "InfoMaxGAN", "MMDGAN", "OpenForensics_fake", "R3GAN", "RDDM10k", "RelGAN", "SNGAN", "STGAN", "SiT10k", "VQGAN", "codiff", "flux", "flux-face", "kandinsky3", "latent_diffusion", "latent_diffusion_1", "latent_diffusion_2", "latent_diffusion_3", "midjourney", "playground", "qwen-image", "realism_riiwa", "realistic_vision", "sd-3.5-face", "sdxl_lightning", "sg2", "sg3", "sgxl")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def parse_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value.decode("utf-8", "strict"), object_pairs_hook=_unique,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"non-finite {x}")))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"M6 {label} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"M6 {label} must be an object")
    return parsed


def load_recipe(path: Path = ROOT / "benchmark/m6/recipe.json") -> dict[str, Any]:
    recipe = parse_json_bytes(path.read_bytes(), label="recipe")
    validate_recipe(recipe)
    validate_census_evidence(parse_json_bytes(CENSUS_PATH.read_bytes(), label="census evidence"))
    return recipe

def validate_census_evidence(value: Mapping[str, Any]) -> None:
    if set(value) != {"schemaVersion", "status", "pixelsRead", "h3PixelsRead", "sourceRights", "sources"} or value["schemaVersion"] != 1 or value["status"] != "P-metadata-census" or value["pixelsRead"] or value["h3PixelsRead"]:
        raise ValueError("M6 census evidence boundary changed")
    set_source = value["sources"]["omniFakeSet"]; ood_source = value["sources"]["omniFakeOOD"]
    if set_source["splitShards"] != {"train": {"count": 19, "bytes": 10447263801}, "validation": {"count": 47, "bytes": 37645535846}} or ood_source["splitShards"] != {"test": {"count": 19, "bytes": 16521804450}}:
        raise ValueError("M6 shard census changed")
    if set_source["realSourcePrefixes"] != {"COCO": 9000, "FFHQ": 4680, "OpenForensics_real": 8568, "OpenImages": 18000, "WIDER": 197, "celebA": 14656, "flickr30k-images": 2860}:
        raise ValueError("M6 SET real-prefix census changed")
    if ood_source["generatorCounts"] != {"Flux.1_pro": 3736, "GPT4o": 5332, "Hidream": 3465, "Ideogram2": 1413, "Ideogram3": 2583, "Imagen4": 3162, "Recraftv3": 3991, "Seedream3.0": 4997, "imagen3": 2489, "nanao_banana": 1301}:
        raise ValueError("M6 OOD generator census changed")
    if set_source["splits"]["train"]["rows"] != 198963 or set_source["splits"]["validation"]["rows"] != 224355 or ood_source["test"]["rows"] != 99722:
        raise ValueError("M6 row census changed")


def validate_recipe(recipe: Mapping[str, Any]) -> None:
    required = {"schemaVersion", "name", "baseCommit", "deliverable", "sources", "training", "selector", "evaluation", "overlap", "preflight", "claims", "gates"}
    if set(recipe) != required or recipe["schemaVersion"] != 1 or recipe["name"] != "prooflens-m6-runpod-score-blind":
        raise ValueError("M6 recipe identity/schema changed")
    if recipe["baseCommit"] != "76d0a807dcf240245830b8510e623d838e43cd4c":
        raise ValueError("M6 must begin at terminal M5 HEAD")
    d = recipe["deliverable"]
    if d != {"format": "ONNX FP32", "input": ["N", 3, 384, 384], "output": ["N", 1], "maximumBytes": 90_000_000, "browserExecution": ["wasm", "webgpu"], "networkAfterInstall": False}:
        raise ValueError("local deliverable boundary changed")
    sources = recipe["sources"]
    if set(sources) != {"omniFakeSet", "omniFakeOOD"}:
        raise ValueError("source set changed")
    expected = {
        "omniFakeSet": ("JamalLee/Omni-Fake-SET", "724e97f5fc9f4b89f59631a8d4e6331712b7d441", "CC-BY-4.0"),
        "omniFakeOOD": ("JamalLee/Omni-Fake-OOD", "9ed7e38bbdb4aeb2eb553896a5890680a9ffcf17", "CC-BY-4.0"),
    }
    for key, (dataset, rev, license_name) in expected.items():
        src = sources[key]
        if src.get("dataset") != dataset or src.get("revision") != rev or src.get("license") != license_name:
            raise ValueError(f"{key} pin changed")
        if src.get("rightsWarranty") != "none; constituent-rights verification remains required":
            raise ValueError("rights disclaimer missing")
    if sources["omniFakeSet"]["imageSplits"] != {"train": {"rows": 198963, "real": 57961, "full_synthetic": 83637, "tampered": 57365}, "validation": {"rows": 224355, "real": 66685, "full_synthetic": 92913, "tampered": 64757}}:
        raise ValueError("SET census changed")
    if sources["omniFakeOOD"]["imageSplits"] != {"test": {"rows": 99722, "real": 33207, "full_synthetic": 32469, "tampered": 34046}}:
        raise ValueError("OOD census changed")
    tr = recipe["training"]
    if tr != {"source": "Omni-Fake-SET:image/train", "classes": ["real", "full_synthetic"], "minimumPerClass": 40000, "selection": "all clean smaller class; source-round-robin same count larger class", "gradientSource": "M6 clean rows only; no M2-M5/H3 pixels or teacher anchor", "branches": [{"name": "last6-consistency", "trainableEncoderBlocks": [6,7,8,9,10,11], "epochs": 6, "snapshots": [4,6], "pairwiseMarginCoefficient": 0.0}, {"name": "last6-margin", "trainableEncoderBlocks": [6,7,8,9,10,11], "epochs": 6, "snapshots": [4,6], "pairwiseMarginCoefficient": 0.2, "pairwiseMargin": 1.0}], "precision": "bfloat16", "optimizer": "AdamW", "batchSize": 64, "gradientAccumulation": 2, "seed": 20260816, "backboneLearningRate": 2e-5, "classifierLearningRate": 2e-4, "weightDecay": 0.05, "gradientClipNorm": 1, "warmupRatio": 0.05, "scheduler": "cosine", "loss": {"sourceClassBalancedBCE": True, "pairedViewConsistencyMSE": 0.05, "pairwiseDefinition": "mean(softplus(1.0 + real_logit_mean - synthetic_logit_mean)); zero when minibatch lacks either class"}, "viewsPerRow": "canonical plus one deterministic stress view uniformly from three transforms", "teacherAnchor": False}:
        raise ValueError("training contract changed")
    sel = recipe["selector"]
    expected_gates = {"original": {"minimumBalancedAccuracy": 0.97, "minimumRealRecall": 1.0, "minimumSyntheticRecall": 0.94, "minimumSyntheticRecallBySource": 0.9}, "screenshot": {"minimumBalancedAccuracy": 0.95, "minimumRealRecall": 1.0, "minimumSyntheticRecall": 0.9, "minimumSyntheticRecallBySource": 0.85}, "social-q75": {"minimumBalancedAccuracy": 0.95, "minimumRealRecall": 1.0, "minimumSyntheticRecall": 0.9, "minimumSyntheticRecallBySource": 0.85}, "social-heavy": {"minimumBalancedAccuracy": 0.95, "minimumRealRecall": 1.0, "minimumSyntheticRecall": 0.9, "minimumSyntheticRecallBySource": 0.85}}
    if sel != {"source": "Omni-Fake-SET:image/validation", "baseItems": 4000, "real": 2000, "synthetic": 2000, "syntheticSelection": "source-round-robin; every one of 34 generators represented", "generators": list(GENERATORS), "views": list(VARIANTS), "zeroObservedFalsePositive": True, "wilsonConfidence": 0.95, "wilsonUpperBoundAtZero": 0.001917047281252934, "poolViews": False, "thresholdSearch": "exhaustive binary64; deterministic ranking/failure", "gates": expected_gates}:
        raise ValueError("selector contract/gates changed")
    ev = recipe["evaluation"]
    if ev != {"items": 100000, "batches": 1000, "batchSize": 100, "synthetic": {"omniFakeOOD": 30000, "omniFakeSETValidation": 70000}, "assignedBeforeSelectorScoring": True, "itemDisjoint": True, "selectionInfluence": False, "selectionExcludesSelector": True, "strictMeanRecallGreaterThan": 0.95, "strictMedianBatchRecallGreaterThan": 0.95, "failureConsumesPanel": True}:
        raise ValueError("evaluation precommit changed")
    overlap = recipe["overlap"]
    if overlap != {"layers": ["canonical dataset row ID/filename/source group", "decoded image SHA256", "EXIF-oriented RGB dHash64 Hamming<=8"], "against": ["cross-partition", "all M2-M5 metadata", "H3 metadata only"], "h3PixelsRead": False}:
        raise ValueError("overlap boundary changed")
    p = recipe["preflight"]
    expected_p = {"minimumPairedItemsPerSecond": 90, "maximumPeakGpuMemoryBytes": 44_000_000_000, "minimumFreeRamBytes": 20_000_000_000, "minimumFreeDiskBytes": 220_000_000_000, "projectedPeakDiskBytes": 180_000_000_000, "maximumL40SHourlyUsd": 3.50, "maximumProjectedGpuUsd": 24, "maximumAllInUsd": 30, "hardWallSeconds": 43200, "safetySeconds": 300, "reads": ["source-locked N", "paired items", "one-batch wall time", "provider rates"], "forbids": ["selector", "regressions", "evaluation", "H3"]}
    if p != expected_p:
        raise ValueError("preflight/cost boundary changed")
    if recipe["claims"] != {"freshness": "item-level only", "zeroFp": "observed only with Wilson bound", "comparative": False}:
        raise ValueError("claim boundary changed")
    if recipe["gates"] != {"sourceStage": "P metadata-only census", "sourceLockStage": "S append-only manifests before selector inference", "publicAuthorizationRequired": True, "paidRunRequires": ["S", "authorization", "preflight"], "publicationRequires": ["fresh 100K pass", "regressions", "critic review"]}:
        raise ValueError("stage gates changed")


def validate_manifest_row(row: Mapping[str, Any]) -> None:
    required = {"dataset", "revision", "partition", "rowId", "filename", "sourceGroup", "label", "imageSha256", "dhash64"}
    if set(row) != required or not isinstance(row["rowId"], str) or not row["rowId"]:
        raise ValueError("manifest row schema/key invalid")
    for key in ("dataset", "partition", "filename", "sourceGroup"):
        if not isinstance(row[key], str) or not row[key] or "\\" in row[key] or row[key].startswith("/") or ".." in row[key].split("/"):
            raise ValueError("manifest string/path field invalid")
    if row["dataset"] not in {"JamalLee/Omni-Fake-SET", "JamalLee/Omni-Fake-OOD"} or row["partition"] not in {"train", "validation", "test"}:
        raise ValueError("manifest dataset/partition invalid")
    if not HEX40.fullmatch(row["revision"]) or not HEX64.fullmatch(row["imageSha256"]):
        raise ValueError("manifest digest invalid")
    if not isinstance(row["dhash64"], str) or not re.fullmatch(r"[0-9a-f]{16}", row["dhash64"]):
        raise ValueError("dHash64 invalid")
    if row["label"] not in {"real", "full_synthetic"}:
        raise ValueError("tampered rows are forbidden")

def assign_evaluation(rows: list[Mapping[str, Any]], *, set_count: int = 70000, ood_count: int = 30000) -> list[dict[str, Any]]:
    """Assign eval membership before any score; OOD never enters selector/train."""
    eligible = [r for r in rows if r.get("label") == "full_synthetic" and r.get("role") == "eval" and ((r.get("dataset") == "JamalLee/Omni-Fake-SET" and r.get("partition") == "validation") or (r.get("dataset") == "JamalLee/Omni-Fake-OOD" and r.get("partition") == "test"))]
    set_rows = sorted((r for r in eligible if r.get("dataset") == "JamalLee/Omni-Fake-SET"), key=lambda r: r.get("imageSha256", ""))[:set_count]
    ood_rows = sorted((r for r in eligible if r.get("dataset") == "JamalLee/Omni-Fake-OOD"), key=lambda r: r.get("imageSha256", ""))[:ood_count]
    if len(set_rows) != set_count or len(ood_rows) != ood_count: raise ValueError("evaluation quotas unavailable")
    selected = set_rows + ood_rows
    identities = {(r.get("dataset"), r.get("partition"), r.get("rowId"), r.get("imageSha256")) for r in selected}
    if len(identities) != len(selected): raise ValueError("duplicate evaluation identity")
    return [{"evalIndex": i, "batch": i // 100, **dict(row)} for i, row in enumerate(selected)]

def overlap_reject(rows: list[Mapping[str, Any]], historical: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ids = {(r.get("dataset"), r.get("rowId"), r.get("filename"), r.get("sourceGroup")) for r in historical}
    shas = {r.get("imageSha256") for r in historical}; hashes = {r.get("dhash64") for r in historical}
    out = []; fresh = set()
    for row in rows:
        key = (row.get("dataset"), row.get("rowId"), row.get("filename"), row.get("sourceGroup"))
        if key in fresh or row.get("imageSha256") in {r.get("imageSha256") for r in out}: continue
        if key in ids or row.get("imageSha256") in shas: continue
        if any(isinstance(row.get("dhash64"), str) and isinstance(h, str) and (int(row["dhash64"], 16) ^ int(h, 16)).bit_count() <= 8 for h in hashes): continue
        out.append(dict(row))
        fresh.add(key)
    return out

def assign_train_selector(rows: list[Mapping[str, Any]], *, train_count: int = 40000, selector_each: int = 2000):
    clean = [r for r in rows if r.get("dataset") == "JamalLee/Omni-Fake-SET" and r.get("partition") in {"train", "validation"} and r.get("label") in {"real", "full_synthetic"} and r.get("role") not in {"eval", "history", "h3"}]
    real = sorted((r for r in clean if r.get("label") == "real" and r.get("partition") == "train"), key=lambda r: (r.get("sourceGroup", ""), r.get("imageSha256", "")))
    synth = sorted((r for r in clean if r.get("label") == "full_synthetic" and r.get("partition") == "train"), key=lambda r: (r.get("sourceGroup", ""), r.get("imageSha256", "")))
    if not real or not synth: raise ValueError("train quota unavailable")
    def rr(items, sources):
        buckets = {s: sorted((r for r in items if r.get("sourceGroup") == s), key=lambda r: r.get("imageSha256", "")) for s in sources}; out=[]
        while len(out) < selector_each:
            progressed = False
            for source in sources:
                if buckets[source]: out.append(buckets[source].pop(0)); progressed = True
                if len(out) >= selector_each: break
            if not progressed: break
        return out
    selector_real = rr([r for r in clean if r.get("partition") == "validation" and r.get("label") == "real"], sorted({r.get("sourceGroup") for r in clean if r.get("partition") == "validation" and r.get("label") == "real"}))
    selector_synth = rr([r for r in clean if r.get("partition") == "validation" and r.get("label") == "full_synthetic" and r.get("sourceGroup") in GENERATORS], list(GENERATORS))
    if len({r.get("sourceGroup") for r in selector_synth}) != len(GENERATORS): raise ValueError("selector generator representation incomplete")
    selector_ids = {r.get("rowId") for r in selector_real + selector_synth}
    count = min(len(real), len(synth), train_count)
    train = real[:count] + synth[:count]
    return train, selector_real + selector_synth


def validate_preflight(observed: Mapping[str, Any], recipe: Mapping[str, Any] | None = None) -> None:
    r = recipe or load_recipe()
    p = r["preflight"]
    checks = {"pairedItemsPerSecond": observed.get("pairedItemsPerSecond", 0) >= p["minimumPairedItemsPerSecond"], "peakGpuMemoryBytes": observed.get("peakGpuMemoryBytes", math.inf) <= p["maximumPeakGpuMemoryBytes"], "freeRamBytes": observed.get("freeRamBytes", 0) >= p["minimumFreeRamBytes"], "freeDiskBytes": observed.get("freeDiskBytes", 0) >= p["minimumFreeDiskBytes"], "projectedPeakDiskBytes": observed.get("projectedPeakDiskBytes", math.inf) <= p["projectedPeakDiskBytes"], "hourlyUsd": observed.get("hourlyUsd", math.inf) <= p["maximumL40SHourlyUsd"], "projectedGpuUsd": observed.get("projectedGpuUsd", math.inf) <= p["maximumProjectedGpuUsd"], "allInUsd": observed.get("allInUsd", math.inf) <= p["maximumAllInUsd"], "projectedWallSeconds": observed.get("projectedWallSeconds", math.inf) <= p["hardWallSeconds"] - p["safetySeconds"]}
    if not all(checks.values()):
        raise ValueError("M6 preflight failed: " + ", ".join(key for key, ok in checks.items() if not ok))
