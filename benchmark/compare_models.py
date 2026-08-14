"""Prove that a candidate ONNX differs from its pinned base only in the classifier."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import onnx


CLASSIFIER_INITIALIZERS = {"classifier.weight", "classifier.bias"}


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def message_digest(value: object) -> str:
    return sha256(value.SerializeToString(deterministic=True)).hexdigest()


def messages_digest(values: object) -> str:
    digest_value = sha256()
    for value in values:
        serialized = value.SerializeToString(deterministic=True)
        digest_value.update(len(serialized).to_bytes(8, "big"))
        digest_value.update(serialized)
    return digest_value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_hash = digest(args.base)
    candidate_hash = digest(args.candidate)
    if base_hash != args.expected_base_sha256:
        raise ValueError(f"Unexpected base SHA-256: {base_hash}")
    if candidate_hash != args.expected_candidate_sha256:
        raise ValueError(f"Unexpected candidate SHA-256: {candidate_hash}")
    base = onnx.load(args.base)
    candidate = onnx.load(args.candidate)
    onnx.checker.check_model(base)
    onnx.checker.check_model(candidate)

    if messages_digest(base.graph.node) != messages_digest(candidate.graph.node):
        raise ValueError("ONNX graph nodes differ")
    if messages_digest(base.graph.input) != messages_digest(candidate.graph.input):
        raise ValueError("ONNX graph inputs differ")
    if messages_digest(base.graph.output) != messages_digest(candidate.graph.output):
        raise ValueError("ONNX graph outputs differ")
    if messages_digest(base.opset_import) != messages_digest(candidate.opset_import):
        raise ValueError("ONNX opsets differ")

    base_initializers = {value.name: value for value in base.graph.initializer}
    candidate_initializers = {value.name: value for value in candidate.graph.initializer}
    if base_initializers.keys() != candidate_initializers.keys():
        raise ValueError("ONNX initializer names differ")
    changed: list[dict[str, object]] = []
    unchanged = 0
    for name in sorted(base_initializers):
        before = message_digest(base_initializers[name])
        after = message_digest(candidate_initializers[name])
        if before == after:
            unchanged += 1
            continue
        changed.append(
            {
                "name": name,
                "beforeSha256": before,
                "afterSha256": after,
                "dimensions": list(candidate_initializers[name].dims),
            }
        )
    if {row["name"] for row in changed} != CLASSIFIER_INITIALIZERS:
        raise ValueError(f"Unexpected changed initializers: {[row['name'] for row in changed]}")

    evidence = {
        "schemaVersion": 1,
        "base": {"path": str(args.base), "sha256": base_hash, "bytes": args.base.stat().st_size},
        "candidate": {
            "path": str(args.candidate),
            "sha256": candidate_hash,
            "bytes": args.candidate.stat().st_size,
        },
        "changedInitializers": changed,
        "unchangedInitializerCount": unchanged,
        "graphNodesSha256": messages_digest(base.graph.node),
        "graphInputsSha256": messages_digest(base.graph.input),
        "graphOutputsSha256": messages_digest(base.graph.output),
        "opsetsSha256": messages_digest(base.opset_import),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
