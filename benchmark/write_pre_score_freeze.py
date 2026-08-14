"""Create the receipt that must be committed alone and pushed before sealed inference."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

from evaluation_contract import require_anonymous_public_file


OUTPUT = Path("benchmark/evidence/evaluation/pre-score-freeze.json")
REPOSITORY_URL = "https://github.com/baney75/prooflens"
CANONICAL_ORIGIN_URLS = {
    REPOSITORY_URL,
    f"{REPOSITORY_URL}.git",
    "git@github.com:baney75/prooflens.git",
    "ssh://git@github.com/baney75/prooflens.git",
}
ALLOWED_POST_SCORE_PATHS = [
    "artifacts/**",
    "benchmark/evidence/evaluation/**",
    "BENCHMARK.md",
    "MODEL_CARD.md",
    "README.md",
    "docs/ACCEPTANCE.md",
]
IMMUTABLE_FILES = [
    "benchmark/bootstrap_ci.py",
    "benchmark/bootstrap_fpr.py",
    "benchmark/evaluate.py",
    "benchmark/evaluation_contract.py",
    "benchmark/large/recipe.json",
    "benchmark/manifests/test.jsonl",
    "benchmark/manifests/validation.jsonl",
    "benchmark/manifests/web-negative.jsonl",
    "benchmark/prediction_contract.py",
    "benchmark/run_release_replay.py",
    "benchmark/verify_evaluation_evidence.py",
    "benchmark/write_pre_score_freeze.py",
    "model-lock.json",
    "scripts/check-benchmark-evidence.mjs",
    "scripts/check-pre-score-freeze.mjs",
    "src/inference/calibration.ts",
    "src/inference/detector.ts",
    "src/shared/model-spec.ts",
    "weights/prooflens-cf384.onnx",
]
PROHIBITED_PRE_SCORE_PATH_PREFIXES = [
    "artifacts/browser-parity",
    "benchmark/evidence/evaluation/confirmatory/",
    "benchmark/evidence/evaluation/web-negative/",
]
PROHIBITED_PRE_SCORE_FILES = {
    "benchmark/evidence/evaluation/pre-score-freeze.json",
    "benchmark/evidence/evaluation/replay-verification.json",
}


def command(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True).strip()


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace pre-score freeze: {OUTPUT}")
    if command("git", "status", "--porcelain"):
        raise ValueError("Pre-score source commit must have a completely clean worktree")
    source_commit = command("git", "rev-parse", "HEAD")
    source_tree = command("git", "rev-parse", "HEAD^{tree}")
    branch = command("git", "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        raise ValueError("Pre-score freeze must be anchored from main")
    origin_fetch = command("git", "remote", "get-url", "origin")
    origin_push = command("git", "remote", "get-url", "--push", "origin")
    if origin_fetch not in CANONICAL_ORIGIN_URLS or origin_push not in CANONICAL_ORIGIN_URLS:
        raise ValueError("Pre-score freeze requires the canonical baney75/prooflens GitHub origin")
    remote_rows = command("git", "ls-remote", "origin", "refs/heads/main").split()
    if len(remote_rows) != 2 or remote_rows[0] != source_commit:
        raise ValueError("Public origin/main does not equal the clean local pre-score commit")
    model_lock_bytes = subprocess.check_output(["git", "show", f"{source_commit}:model-lock.json"])
    public_proof = require_anonymous_public_file(
        expected_head=source_commit,
        file_commit=source_commit,
        path=Path("model-lock.json"),
        expected_bytes=model_lock_bytes,
    )
    tree_paths = set(command("git", "ls-tree", "-r", "--name-only", source_commit).splitlines())
    if any(path in tree_paths for path in PROHIBITED_PRE_SCORE_FILES) or any(
        any(path.startswith(prefix) for prefix in PROHIBITED_PRE_SCORE_PATH_PREFIXES)
        for path in tree_paths
    ):
        raise ValueError("Public source commit already contains post-score evidence")
    missing = [path for path in IMMUTABLE_FILES if path not in tree_paths]
    if missing:
        raise ValueError(f"Pre-score source commit lacks immutable files: {missing}")
    immutable_hashes = {
        path: digest_bytes(subprocess.check_output(["git", "show", f"{source_commit}:{path}"]))
        for path in IMMUTABLE_FILES
    }
    payload = {
        "schemaVersion": 2,
        "mode": "public pre-score source freeze before any confirmatory or web-negative inference",
        "repository": REPOSITORY_URL,
        "branch": branch,
        "sourceCommit": source_commit,
        "sourceTree": source_tree,
        "remoteObservedHead": remote_rows[0],
        "remoteVerifiedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "publicCommitUrl": f"{REPOSITORY_URL}/commit/{source_commit}",
        "anonymousPublicProof": public_proof,
        "allowedPostScorePaths": ALLOWED_POST_SCORE_PATHS,
        "immutableFilesSha256": immutable_hashes,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(
        "Next: commit only benchmark/evidence/evaluation/pre-score-freeze.json, "
        "push that freeze-only child to public main, and do not score until origin/main equals it."
    )


if __name__ == "__main__":
    main()
