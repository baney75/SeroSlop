"""CUDA throughput preflight contract; no model/data surfaces are read."""
from __future__ import annotations
import json, time, math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.m6.contracts import load_recipe, validate_preflight

def project(observed, recipe=None):
    recipe = recipe or load_recipe(); epochs = sum(b["epochs"] for b in recipe["training"]["branches"])
    required = ("measuredBatchUniqueItems","sourceLockedUniqueItems","oneBatchSeconds","hourlyUsd","selectorSeconds","regressionSeconds","evalSeconds")
    for key in required:
        value = observed.get(key)
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0: raise ValueError(f"invalid preflight value: {key}")
    if observed["measuredBatchUniqueItems"] <= 0 or observed["sourceLockedUniqueItems"] < 100000: raise ValueError("source-locked item count is insufficient")
    train_seconds = observed["oneBatchSeconds"] * math.ceil(observed["sourceLockedUniqueItems"] * 2 / observed["measuredBatchUniqueItems"]) * epochs
    total_seconds = train_seconds + observed["selectorSeconds"] + observed["regressionSeconds"] + observed["evalSeconds"]
    gpu = (train_seconds + observed["selectorSeconds"] + observed["regressionSeconds"] + observed["evalSeconds"]) / 3600 * observed["hourlyUsd"]
    projected = {"pairedItemsPerSecond": observed["measuredBatchUniqueItems"] / max(observed["oneBatchSeconds"], 1e-12), "trainingSeconds": train_seconds, "projectedWallSeconds": total_seconds, "projectedGpuUsd": gpu, "allInUsd": gpu + float(observed.get("stagingUsd", 0)) + float(observed.get("storageUsd", 0))}
    if projected["projectedWallSeconds"] > recipe["preflight"]["hardWallSeconds"] - recipe["preflight"]["safetySeconds"]: raise ValueError("projected hard wall exceeded")
    return projected

def receipt(observed):
    recipe = load_recipe(); projected = project(observed, recipe); merged = {**observed, **projected}; validate_preflight(merged, recipe)
    return {"schemaVersion": 1, "status": "m6-preflight-pass", "recipe": "benchmark/m6/recipe.json", "branches": [{"name": b["name"], "epochs": b["epochs"], "snapshots": b["snapshots"]} for b in recipe["training"]["branches"]], "projected": projected, "observed": observed}

def main():
    raise SystemExit("M6 preflight requires an operator-injected CUDA receipt; no implicit Torch/provider import")

if __name__ == "__main__": main()
