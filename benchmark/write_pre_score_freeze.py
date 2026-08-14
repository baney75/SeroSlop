"""Create the recovery receipt that must be committed alone before sealed inference."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evaluation_contract import (
    EXPECTED_ALLOWED_POST_SCORE_PATHS,
    EXPECTED_IMMUTABLE_FILES,
    digest_file,
    digest_git_blob,
    require_anonymous_public_file,
)


OUTPUT = Path("benchmark/evidence/evaluation/pre-score-freeze-v2.json")
LEGACY_FREEZE_PATH = Path("benchmark/evidence/evaluation/pre-score-freeze.json")
LEGACY_SOURCE_COMMIT = "0771a9422b552e2023e5150fb6c8b4238b811a74"
LEGACY_FREEZE_COMMIT = "2bd0c4757f6059c57879414a5dba77629d66460e"
LEGACY_FREEZE_SHA256 = "400fd2b7a7cd84b063f81799eaf3829f770220c41f58dff53e7caebd1a145c34"
FAILED_RECOVERY_SOURCE_COMMIT = "99861df575854511c685d7b8f90acdc7ed4e5923"
FAILED_RECOVERY_SOURCE_TREE = "204594f26118d8c1c3add9dfef3a6050949772e1"
FAILED_RECOVERY_ACTIONS_RUN_ID = 31846361076
FAILED_RECOVERY_ACTIONS_RUN_URL = "https://github.com/baney75/prooflens/actions/runs/31846361076"
FAILED_RECOVERY_REASON = "ci-missing-inference-dependencies-before-v2-guard"
RECOVERY_REASON = "ci-pre-input-guard-imported-inference-dependencies"
REPOSITORY_URL = "https://github.com/baney75/prooflens"
CANONICAL_ORIGIN_URLS = {
    REPOSITORY_URL,
    f"{REPOSITORY_URL}.git",
    "git@github.com:baney75/prooflens.git",
    "ssh://git@github.com/baney75/prooflens.git",
}
FAILED_RECOVERY_REPAIR_PATHS = [
    "BENCHMARK.md",
    "README.md",
    "benchmark/evaluate.py",
    "benchmark/evaluation_contract.py",
    "benchmark/test_integrity_contracts.py",
    "benchmark/verify_evaluation_evidence.py",
    "benchmark/write_pre_score_freeze.py",
    "package.json",
    "scripts/check-benchmark-evidence.mjs",
    "scripts/check-pre-score-freeze.mjs",
    "scripts/check-pre-score-stage.mjs",
    "scripts/git-object-digest.mjs",
    "scripts/release-stage-policy.mjs",
    "scripts/run-static-verification.mjs",
    "scripts/test-git-object-digest.mjs",
    "scripts/test-release-stage-policy.mjs",
]
RECOVERY_REPAIR_PATHS = [
    "BENCHMARK.md",
    "README.md",
    "benchmark/evaluate.py",
    "benchmark/evaluation_contract.py",
    "benchmark/test_integrity_contracts.py",
    "benchmark/write_pre_score_freeze.py",
    "scripts/check-pre-score-freeze.mjs",
    "scripts/check-pre-score-stage.mjs",
    "scripts/release-stage-policy.mjs",
]
PROHIBITED_PRE_SCORE_PATH_PREFIXES = [
    "artifacts/browser-parity",
    "benchmark/evidence/evaluation/confirmatory/",
    "benchmark/evidence/evaluation/web-negative/",
]
PROHIBITED_PRE_SCORE_FILES = {
    str(OUTPUT),
    "benchmark/evidence/evaluation/replay-verification.json",
}


def command(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True).strip()


def main() -> None:
    repository_root = Path.cwd()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace pre-score freeze: {OUTPUT}")
    if command("git", "status", "--porcelain"):
        raise ValueError("Pre-score source commit must have a completely clean worktree")
    source_commit = command("git", "rev-parse", "HEAD")
    source_tree = command("git", "rev-parse", "HEAD^{tree}")
    source_parent = command("git", "rev-parse", "HEAD^")
    if (
        source_parent != FAILED_RECOVERY_SOURCE_COMMIT
        or command("git", "rev-parse", f"{FAILED_RECOVERY_SOURCE_COMMIT}^") != LEGACY_FREEZE_COMMIT
        or command("git", "rev-parse", f"{LEGACY_FREEZE_COMMIT}^") != LEGACY_SOURCE_COMMIT
        or command("git", "rev-parse", f"{FAILED_RECOVERY_SOURCE_COMMIT}^{{tree}}")
        != FAILED_RECOVERY_SOURCE_TREE
    ):
        raise ValueError("Second recovery source lineage changed")
    legacy_paths = command(
        "git", "diff-tree", "--no-renames", "--no-commit-id", "--name-only", "-r", LEGACY_FREEZE_COMMIT
    ).splitlines()
    if legacy_paths != [str(LEGACY_FREEZE_PATH)]:
        raise ValueError("Legacy freeze commit shape changed")
    if not LEGACY_FREEZE_PATH.is_file() or digest_file(LEGACY_FREEZE_PATH) != LEGACY_FREEZE_SHA256:
        raise ValueError("Legacy freeze receipt changed in the recovery source")
    if digest_git_blob(repository_root, f"{LEGACY_FREEZE_COMMIT}:{LEGACY_FREEZE_PATH}") != LEGACY_FREEZE_SHA256:
        raise ValueError("Legacy public freeze receipt bytes changed")
    added_freeze_receipts = sorted({
        path
        for path in command(
            "git", "log", "--no-renames", "--diff-filter=A", "--format=", "--name-only", "HEAD"
        ).splitlines()
        if path.startswith("benchmark/evidence/evaluation/pre-score-freeze")
    })
    if added_freeze_receipts != [str(LEGACY_FREEZE_PATH)]:
        raise ValueError("Recovery source history contains an alternate pre-score freeze receipt")
    failed_repair_paths = sorted(command(
        "git", "diff", "--no-renames", "--name-only",
        f"{LEGACY_FREEZE_COMMIT}..{FAILED_RECOVERY_SOURCE_COMMIT}"
    ).splitlines())
    if failed_repair_paths != sorted(FAILED_RECOVERY_REPAIR_PATHS):
        raise ValueError(
            f"Failed first recovery changed outside its exact repair set: {failed_repair_paths}"
        )
    repair_paths = sorted(command(
        "git", "diff", "--no-renames", "--name-only", f"{FAILED_RECOVERY_SOURCE_COMMIT}..{source_commit}"
    ).splitlines())
    if repair_paths != sorted(RECOVERY_REPAIR_PATHS):
        raise ValueError(f"Recovery source changed outside its exact repair set: {repair_paths}")
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
    missing = [path for path in EXPECTED_IMMUTABLE_FILES if path not in tree_paths]
    if missing:
        raise ValueError(f"Pre-score source commit lacks immutable files: {missing}")
    immutable_hashes = {
        path: digest_git_blob(repository_root, f"{source_commit}:{path}")
        for path in EXPECTED_IMMUTABLE_FILES
    }
    if any(digest_file(Path(path)) != expected for path, expected in immutable_hashes.items()):
        raise ValueError("Recovery source worktree differs from its immutable Git blobs")
    payload = {
        "schemaVersion": 3,
        "generation": 2,
        "mode": "public second-recovery pre-score source freeze before any confirmatory or web-negative inference",
        "receiptPath": str(OUTPUT),
        "repository": REPOSITORY_URL,
        "branch": branch,
        "sourceCommit": source_commit,
        "sourceTree": source_tree,
        "remoteObservedHead": remote_rows[0],
        "remoteVerifiedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "publicCommitUrl": f"{REPOSITORY_URL}/commit/{source_commit}",
        "anonymousPublicProof": public_proof,
        "recovery": {
            "reason": RECOVERY_REASON,
            "legacySourceCommit": LEGACY_SOURCE_COMMIT,
            "legacyFreezeCommit": LEGACY_FREEZE_COMMIT,
            "legacyReceiptPath": str(LEGACY_FREEZE_PATH),
            "legacyReceiptSha256": LEGACY_FREEZE_SHA256,
            "failedRecoverySourceCommit": FAILED_RECOVERY_SOURCE_COMMIT,
            "failedRecoverySourceTree": FAILED_RECOVERY_SOURCE_TREE,
            "failedRecoveryActionsRunId": FAILED_RECOVERY_ACTIONS_RUN_ID,
            "failedRecoveryActionsRunUrl": FAILED_RECOVERY_ACTIONS_RUN_URL,
            "failedRecoveryReason": FAILED_RECOVERY_REASON,
            "failedRecoveryRepairPaths": FAILED_RECOVERY_REPAIR_PATHS,
            "repairPaths": RECOVERY_REPAIR_PATHS,
            "repositoryEvidenceLimitation": (
                "Repository history proves no canonical sealed output was committed before this recovery freeze; "
                "it cannot prove that no pixels were viewed outside the recorded workflow."
            ),
        },
        "allowedPostScorePaths": EXPECTED_ALLOWED_POST_SCORE_PATHS,
        "immutableFilesSha256": immutable_hashes,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(
        f"Next: commit only {OUTPUT}, "
        "push that freeze-only child to public main, and do not score until origin/main equals it."
    )


if __name__ == "__main__":
    main()
