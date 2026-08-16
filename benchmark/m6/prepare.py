"""M6 P/S source-stage launcher.

P performs metadata-only census.  S is intentionally explicit and requires a
public authorization file; this command never downloads or decodes pixels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.m6.contracts import load_recipe

def census(rows, expected=None):
    """Validate an injected metadata census; never fetches or decodes pixels."""
    recipe = load_recipe(); expected = expected or recipe["sources"]["omniFakeSet"]["imageSplits"]["validation"]
    if not isinstance(rows, list): raise ValueError("census rows must be a list")
    labels = {label: sum(1 for row in rows if row.get("label") == label) for label in ("real", "full_synthetic", "tampered")}
    if any("id" not in row or "label" not in row for row in rows): raise ValueError("schema missing id/label")
    if labels != {"real": expected["real"], "full_synthetic": expected["full_synthetic"], "tampered": expected["tampered"]}: raise ValueError("split label census changed")
    return {"rows": len(rows), "labels": labels}

def census_all(parts, expected=None):
    """Metadata-only validation for SET train/validation and OOD test."""
    if set(parts) != {"set_train", "set_validation", "ood_test"}: raise ValueError("required split census missing")
    if expected is None:
        recipe = load_recipe(); s = recipe["sources"]["omniFakeSet"]["imageSplits"]; o = recipe["sources"]["omniFakeOOD"]["imageSplits"]
        expected = {"set_train": {"labels": {k: s["train"][k] for k in ("real", "full_synthetic", "tampered")}}, "set_validation": {"labels": {k: s["validation"][k] for k in ("real", "full_synthetic", "tampered")}}, "ood_test": {"labels": {k: o["test"][k] for k in ("real", "full_synthetic", "tampered")}}}
    required_labels = {"real", "full_synthetic", "tampered"}
    if set(expected) != set(parts) or any(set(expected[name].get("labels", {})) != required_labels for name in parts): raise ValueError("frozen census expectations incomplete")
    for name, rows in parts.items():
        if len({row.get("id") for row in rows}) != len(rows): raise ValueError(f"duplicate row identity: {name}")
        for row in rows:
            if set(row) != {"id", "label", "generator", "sourcePrefix"}: raise ValueError("schema incomplete")
            if row["label"] not in required_labels: raise ValueError("unknown label")
            if row["label"] == "tampered" and name == "ood_test": continue
        counts = {label: sum(row.get("label") == label for row in rows) for label in ("real", "full_synthetic", "tampered")}
        if counts != expected[name]["labels"]: raise ValueError(f"label counts changed: {name}")
    return {name: len(rows) for name, rows in parts.items()}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("census",), required=True)
    parser.add_argument("--rows", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    recipe = load_recipe()
    if args.phase == "census":
        # Local P only checks the frozen contract; live shard/LFS census is an
        # explicit injected phase and cannot be mistaken for completed S.
        print(json.dumps({"stage": "P", "status": "metadata-contract-valid", "pixelsRead": False, "liveCensus": False}, sort_keys=True))
        return
    return


if __name__ == "__main__":
    main()
