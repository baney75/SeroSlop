"""Replay the replacement-v2 evaluator and intervals against local pixels."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")
PROTOCOLS = {
    "confirmatory-v2": {
        "dataRoot": "benchmark/data/replacement-v2",
        "manifest": "benchmark/manifests/test-v2.jsonl",
        "manifestSha256": "773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8",
        "outputDir": "benchmark/evidence/evaluation/confirmatory-v2",
        "name": "prooflens-confirmatory-v2",
    },
    "web-negative-v2": {
        "dataRoot": "benchmark/data/replacement-v2",
        "manifest": "benchmark/manifests/web-negative-v2.jsonl",
        "manifestSha256": "6a1287bae6826811c81cbebab79a1bc6abb475fde70c9aa1529c390ed97014c9",
        "outputDir": "benchmark/evidence/evaluation/web-negative-v2",
        "name": "prooflens-web-negative-v2",
    },
}
FREEZE_PATH = Path("benchmark/evidence/evaluation/pre-score-freeze-v3.json")
RECIPE_PATH = Path("benchmark/large/recipe.json")
MODEL_LOCK_PATH = Path("model-lock.json")


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("weights/prooflens-cf384.onnx"))
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--calibration", type=Path, default=Path("benchmark/evidence/large/calibration.json"))
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument("--execution-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/evidence/evaluation/replay-verification-v2.json"),
    )
    parser.add_argument(
        "--verify-existing-receipt",
        action="store_true",
        help="Require an existing receipt to be byte-identical instead of overwriting it.",
    )
    args = parser.parse_args()

    model_lock = json.loads(MODEL_LOCK_PATH.read_text())
    training_evidence = model_lock.get("trainingEvidence", {})
    if (
        digest(args.model) != args.expected_model_sha256
        or model_lock.get("sha256") != args.expected_model_sha256
        or args.model.stat().st_size != model_lock.get("bytes")
    ):
        raise ValueError("Replay model does not match model-lock.json")
    if (
        digest(args.calibration) != args.expected_calibration_sha256
        or training_evidence.get("calibrationSha256") != args.expected_calibration_sha256
    ):
        raise ValueError("Replay calibration does not match model-lock.json")
    if digest(RECIPE_PATH) != training_evidence.get("recipeSha256"):
        raise ValueError("Replay recipe does not match model-lock.json")
    if not FREEZE_PATH.is_file():
        raise FileNotFoundError("Replacement replay requires the public V3 freeze receipt")

    commands: list[list[str]] = []
    for protocol, config in PROTOCOLS.items():
        command = [
            sys.executable,
            "benchmark/evaluate.py",
            "--model", str(args.model),
            "--expected-model-sha256", args.expected_model_sha256,
            "--data-root", config["dataRoot"],
            "--manifest", config["manifest"],
            "--expected-manifest-sha256", config["manifestSha256"],
            "--output-dir", config["outputDir"],
            "--protocol", protocol,
            "--batch-size", str(args.batch_size),
            "--execution-provider", args.execution_provider,
            "--calibration", str(args.calibration),
            "--expected-calibration-sha256", args.expected_calibration_sha256,
            "--verify-existing",
        ]
        subprocess.run(command, check=True)
        commands.append(command[1:])

    calibration = json.loads(args.calibration.read_text())
    threshold = str(calibration["rawProbabilityThreshold"])
    confirmatory = PROTOCOLS["confirmatory-v2"]
    confirmatory_predictions = [
        f"{confirmatory['outputDir']}/{confirmatory['name']}-{variant}-predictions.jsonl"
        for variant in VARIANTS
    ]
    bootstrap_command = [
        sys.executable,
        "benchmark/bootstrap_ci.py",
        "--predictions", *confirmatory_predictions,
        "--manifest", confirmatory["manifest"],
        "--expected-manifest-sha256", confirmatory["manifestSha256"],
        "--raw-threshold", threshold,
        "--seed", "20260813",
        "--replicates", "20000",
        "--output", f"{confirmatory['outputDir']}/bootstrap.json",
        "--verify-existing",
    ]
    subprocess.run(bootstrap_command, check=True)
    commands.append(bootstrap_command[1:])

    web_negative = PROTOCOLS["web-negative-v2"]
    web_predictions = [
        f"{web_negative['outputDir']}/{web_negative['name']}-{variant}-predictions.jsonl"
        for variant in VARIANTS
    ]
    wilson_command = [
        sys.executable,
        "benchmark/bootstrap_fpr.py",
        "--predictions", *web_predictions,
        "--manifest", web_negative["manifest"],
        "--expected-manifest-sha256", web_negative["manifestSha256"],
        "--raw-threshold", threshold,
        "--output", f"{web_negative['outputDir']}/wilson.json",
        "--verify-existing",
    ]
    subprocess.run(wilson_command, check=True)
    commands.append(wilson_command[1:])

    bound_files = [
        args.model,
        args.calibration,
        MODEL_LOCK_PATH,
        RECIPE_PATH,
        FREEZE_PATH,
        Path("benchmark/evaluate.py"),
        Path("benchmark/evaluation_contract.py"),
        Path("benchmark/bootstrap_ci.py"),
        Path("benchmark/bootstrap_fpr.py"),
        Path("benchmark/prediction_contract.py"),
        Path("benchmark/verify_evaluation_evidence.py"),
        Path("benchmark/run_release_replay.py"),
        Path("benchmark/manifests/test-v2.jsonl"),
        Path("benchmark/manifests/web-negative-v2.jsonl"),
        Path("benchmark/manifests/replacement-v2-selection.json"),
        Path("benchmark/manifests/parity-ids-v2.json"),
    ]
    for config in PROTOCOLS.values():
        root = Path(config["outputDir"])
        bound_files.extend(root / f"{config['name']}-{variant}-predictions.jsonl" for variant in VARIANTS)
        bound_files.extend([
            root / f"{config['name']}-summary.json",
            root / f"{config['name']}-complete.json",
        ])
    bound_files.extend([
        Path(f"{confirmatory['outputDir']}/bootstrap.json"),
        Path(f"{web_negative['outputDir']}/wilson.json"),
    ])
    receipt = {
        "schemaVersion": 2,
        "mode": "byte-identical replay of replacement-v2 confirmatory, web-negative, bootstrap, and Wilson evidence",
        "modelSha256": args.expected_model_sha256,
        "calibrationSha256": args.expected_calibration_sha256,
        "recipeSha256": digest(RECIPE_PATH),
        "freezeReceiptSha256": digest(FREEZE_PATH),
        "executionProvider": args.execution_provider,
        "batchSize": args.batch_size,
        "commands": commands,
        "files": {str(path): digest(path) for path in bound_files},
    }
    encoded = (json.dumps(receipt, indent=2) + "\n").encode()
    if args.output.exists():
        if not args.verify_existing_receipt:
            raise FileExistsError(f"Refusing to overwrite replay verification: {args.output}")
        if args.output.read_bytes() != encoded:
            raise ValueError("Existing replacement replay receipt is not byte-identical")
        print(json.dumps(receipt, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=f".{args.output.name}.", delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, args.output)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
