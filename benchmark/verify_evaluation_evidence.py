"""Replay every frozen evaluator and interval output against local source pixels."""

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
    "validation": {
        "dataRoot": "benchmark/data/modern-head",
        "manifest": "benchmark/manifests/validation.jsonl",
        "manifestSha256": "41be10ef876ecef0635744ed29677a1888a7759cc8060dc7a392f76f83ab263b",
        "outputDir": "benchmark/evidence/evaluation/validation",
        "name": "prooflens-validation",
    },
    "confirmatory": {
        "dataRoot": "benchmark/data",
        "manifest": "benchmark/manifests/test.jsonl",
        "manifestSha256": "28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae",
        "outputDir": "benchmark/evidence/evaluation/confirmatory",
        "name": "prooflens-confirmatory-test",
    },
    "web-negative": {
        "dataRoot": "benchmark/data/web-negative",
        "manifest": "benchmark/manifests/web-negative.jsonl",
        "manifestSha256": "ad8b3f30a37feb3b6b046683db2d4071e236e6878612c7d8733869699d7f7824",
        "outputDir": "benchmark/evidence/evaluation/web-negative",
        "name": "prooflens-web-negative",
    },
}


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
        default=Path("benchmark/evidence/evaluation/replay-verification.json"),
    )
    parser.add_argument(
        "--verify-existing-receipt",
        action="store_true",
        help="Require an existing receipt to be byte-identical instead of overwriting it.",
    )
    args = parser.parse_args()

    if digest(args.model) != args.expected_model_sha256:
        raise ValueError("Replay model SHA-256 changed")
    if digest(args.calibration) != args.expected_calibration_sha256:
        raise ValueError("Replay calibration SHA-256 changed")
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
    confirmatory_predictions = [
        f"benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-{variant}-predictions.jsonl"
        for variant in VARIANTS
    ]
    bootstrap_command = [
        sys.executable,
        "benchmark/bootstrap_ci.py",
        "--predictions", *confirmatory_predictions,
        "--manifest", PROTOCOLS["confirmatory"]["manifest"],
        "--expected-manifest-sha256", PROTOCOLS["confirmatory"]["manifestSha256"],
        "--raw-threshold", threshold,
        "--seed", "20260813",
        "--replicates", "20000",
        "--output", "benchmark/evidence/evaluation/confirmatory/bootstrap.json",
        "--verify-existing",
    ]
    subprocess.run(bootstrap_command, check=True)
    commands.append(bootstrap_command[1:])

    web_predictions = [
        f"benchmark/evidence/evaluation/web-negative/prooflens-web-negative-{variant}-predictions.jsonl"
        for variant in VARIANTS
    ]
    wilson_command = [
        sys.executable,
        "benchmark/bootstrap_fpr.py",
        "--predictions", *web_predictions,
        "--manifest", PROTOCOLS["web-negative"]["manifest"],
        "--expected-manifest-sha256", PROTOCOLS["web-negative"]["manifestSha256"],
        "--raw-threshold", threshold,
        "--output", "benchmark/evidence/evaluation/web-negative/wilson.json",
        "--verify-existing",
    ]
    subprocess.run(wilson_command, check=True)
    commands.append(wilson_command[1:])

    bound_files = [
        args.model,
        args.calibration,
        Path("benchmark/evaluate.py"),
        Path("benchmark/evaluation_contract.py"),
        Path("benchmark/bootstrap_ci.py"),
        Path("benchmark/bootstrap_fpr.py"),
        Path("benchmark/prediction_contract.py"),
        Path("benchmark/verify_evaluation_evidence.py"),
        Path("benchmark/evidence/evaluation/pre-score-freeze-v2.json"),
    ]
    for config in PROTOCOLS.values():
        root = Path(config["outputDir"])
        bound_files.extend(root / f"{config['name']}-{variant}-predictions.jsonl" for variant in VARIANTS)
        bound_files.extend([
            root / f"{config['name']}-summary.json",
            root / f"{config['name']}-complete.json",
        ])
    bound_files.extend([
        Path("benchmark/evidence/evaluation/confirmatory/bootstrap.json"),
        Path("benchmark/evidence/evaluation/web-negative/wilson.json"),
    ])
    receipt = {
        "schemaVersion": 1,
        "mode": "byte-identical replay of immutable validation, confirmatory, web-negative, bootstrap, and Wilson evidence",
        "modelSha256": args.expected_model_sha256,
        "calibrationSha256": args.expected_calibration_sha256,
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
            raise ValueError("Existing replay verification receipt is not byte-identical")
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
