"""Download and verify the exact corrected Community Forensics ONNX artifact."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import urllib.request


REVISION = "ac6ee457bea904a373065754107451793b56db00"
URL = f"https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/{REVISION}/onnx/model.onnx"
EXPECTED_SHA256 = "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmark/candidates/upstream-cf384.onnx"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with urllib.request.urlopen(URL, timeout=120) as source, partial.open("wb") as destination:
        while chunk := source.read(1024 * 1024):
            destination.write(chunk)
    actual = digest(partial)
    if actual != EXPECTED_SHA256:
        partial.unlink(missing_ok=True)
        raise ValueError(f"Upstream model SHA-256 mismatch: {actual}")
    partial.replace(args.output)
    print(f"verified {args.output} {actual}")


if __name__ == "__main__":
    main()
