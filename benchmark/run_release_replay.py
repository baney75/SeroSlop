"""Run the pixel-and-ONNX release replay using the repository's frozen lock."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    model_lock_path = Path("model-lock.json")
    model_lock = json.loads(model_lock_path.read_text())
    model = Path(model_lock["artifact"])
    calibration = Path("benchmark/evidence/large/calibration.json")
    if digest(model) != model_lock["sha256"] or model.stat().st_size != model_lock["bytes"]:
        raise ValueError("Release replay model does not match model-lock.json")
    command = [
        sys.executable,
        "benchmark/verify_evaluation_evidence.py",
        "--model", str(model),
        "--expected-model-sha256", model_lock["sha256"],
        "--calibration", str(calibration),
        "--expected-calibration-sha256", digest(calibration),
        "--execution-provider", "cpu",
        "--batch-size", "16",
        "--verify-existing-receipt",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
