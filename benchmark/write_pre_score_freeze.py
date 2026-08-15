"""Create the receipt-only V3 freeze after the public A4 source is green."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from evaluation_contract import (
    EXPECTED_ALLOWED_POST_SCORE_PATHS,
    EXPECTED_IMMUTABLE_FILES,
    FAILED_EVALUATION_ACTIONS_RUN_ID,
    FAILED_EVALUATION_COMMIT,
    FAILED_EVALUATION_PATHS,
    FAILED_EVALUATION_REASON,
    FAILED_EVALUATION_TREE,
    FAILED_RECOVERY_ACTIONS_RUN_ID,
    FAILED_RECOVERY_ACTIONS_RUN_URL,
    FAILED_RECOVERY_REASON,
    FAILED_RECOVERY_REPAIR_PATHS,
    FAILED_RECOVERY_SOURCE_COMMIT,
    FAILED_RECOVERY_SOURCE_TREE,
    FREEZE_PATH,
    LEGACY_FREEZE_ACTIONS_RUN_ID,
    LEGACY_FREEZE_COMMIT,
    LEGACY_FREEZE_PATH,
    LEGACY_FREEZE_SHA256,
    LEGACY_FREEZE_TREE,
    LEGACY_SOURCE_ACTIONS_RUN_ID,
    LEGACY_SOURCE_COMMIT,
    LEGACY_SOURCE_TREE,
    PARITY_IDS_V2_SHA256,
    POST_SCORE_FILES,
    POST_SCORE_PREFIXES,
    QUALITY_WORKFLOW_PATH,
    RECOVERY_REASON,
    RECOVERY_REPAIR_PATHS,
    REPLACEMENT_CONFIRMATORY_MANIFEST_SHA256,
    REPLACEMENT_SELECTION_ACTIONS_RUN_ID,
    REPLACEMENT_SELECTION_COMMIT,
    REPLACEMENT_SELECTION_PATHS,
    REPLACEMENT_SELECTION_REASON,
    REPLACEMENT_SELECTION_SHA256,
    REPLACEMENT_SELECTION_TREE,
    REPLACEMENT_WEB_NEGATIVE_MANIFEST_SHA256,
    SECOND_FREEZE_ACTIONS_RUN_ID,
    SECOND_FREEZE_COMMIT,
    SECOND_FREEZE_PATH,
    SECOND_FREEZE_SHA256,
    SECOND_FREEZE_TREE,
    SECOND_RECOVERY_ACTIONS_RUN_ID,
    SECOND_RECOVERY_REPAIR_PATHS,
    SECOND_RECOVERY_SOURCE_COMMIT,
    SECOND_RECOVERY_SOURCE_TREE,
    digest_file,
    digest_git_blob,
    require_anonymous_public_file,
    require_successful_public_quality_run,
)


OUTPUT = FREEZE_PATH
REPOSITORY_URL = "https://github.com/baney75/prooflens"
CANONICAL_ORIGIN_URLS = {
    REPOSITORY_URL,
    f"{REPOSITORY_URL}.git",
    "git@github.com:baney75/prooflens.git",
    "ssh://git@github.com/baney75/prooflens.git",
}


def command(*arguments: str) -> str:
    return subprocess.check_output(arguments, text=True).strip()


def action_url(run_id: int) -> str:
    return f"{REPOSITORY_URL}/actions/runs/{run_id}"


def changed_paths(older: str, newer: str) -> list[str]:
    return sorted(command("git", "diff", "--no-renames", "--name-only", f"{older}..{newer}").splitlines())


def added_path_hashes(repository_root: Path, commit: str, expected_paths: list[str]) -> dict[str, str]:
    rows = command(
        "git", "diff-tree", "--no-renames", "--no-commit-id", "--name-status", "-r", commit
    ).splitlines()
    expected_rows = [f"A\t{path}" for path in expected_paths]
    if sorted(rows) != sorted(expected_rows):
        raise ValueError(f"Historical evidence packet changed at {commit}")
    return {path: digest_git_blob(repository_root, f"{commit}:{path}") for path in expected_paths}


def require_parent(child: str, parent: str, label: str) -> None:
    if command("git", "rev-parse", f"{child}^") != parent:
        raise ValueError(f"{label} lineage changed")


def require_tree(commit: str, tree: str, label: str) -> None:
    if command("git", "rev-parse", f"{commit}^{{tree}}") != tree:
        raise ValueError(f"{label} tree changed")


def main() -> None:
    repository_root = Path.cwd()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace pre-score freeze: {OUTPUT}")
    if command("git", "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("V3 source commit must have a completely clean worktree")
    source_commit = command("git", "rev-parse", "HEAD")
    source_tree = command("git", "rev-parse", "HEAD^{tree}")
    require_parent(source_commit, REPLACEMENT_SELECTION_COMMIT, "Numeric recovery source")
    for commit, tree, label in (
        (LEGACY_SOURCE_COMMIT, LEGACY_SOURCE_TREE, "Legacy source"),
        (LEGACY_FREEZE_COMMIT, LEGACY_FREEZE_TREE, "Legacy freeze"),
        (FAILED_RECOVERY_SOURCE_COMMIT, FAILED_RECOVERY_SOURCE_TREE, "Failed recovery source"),
        (SECOND_RECOVERY_SOURCE_COMMIT, SECOND_RECOVERY_SOURCE_TREE, "Second recovery source"),
        (SECOND_FREEZE_COMMIT, SECOND_FREEZE_TREE, "Second freeze"),
        (FAILED_EVALUATION_COMMIT, FAILED_EVALUATION_TREE, "Failed evaluation"),
        (REPLACEMENT_SELECTION_COMMIT, REPLACEMENT_SELECTION_TREE, "Replacement selection"),
    ):
        require_tree(commit, tree, label)
    for child, parent, label in (
        (LEGACY_FREEZE_COMMIT, LEGACY_SOURCE_COMMIT, "Legacy freeze"),
        (FAILED_RECOVERY_SOURCE_COMMIT, LEGACY_FREEZE_COMMIT, "Failed recovery source"),
        (SECOND_RECOVERY_SOURCE_COMMIT, FAILED_RECOVERY_SOURCE_COMMIT, "Second recovery source"),
        (SECOND_FREEZE_COMMIT, SECOND_RECOVERY_SOURCE_COMMIT, "Second freeze"),
        (FAILED_EVALUATION_COMMIT, SECOND_FREEZE_COMMIT, "Failed evaluation"),
        (REPLACEMENT_SELECTION_COMMIT, FAILED_EVALUATION_COMMIT, "Replacement selection"),
    ):
        require_parent(child, parent, label)
    if changed_paths(LEGACY_FREEZE_COMMIT, FAILED_RECOVERY_SOURCE_COMMIT) != sorted(FAILED_RECOVERY_REPAIR_PATHS):
        raise ValueError("Failed first recovery repair surface changed")
    if changed_paths(FAILED_RECOVERY_SOURCE_COMMIT, SECOND_RECOVERY_SOURCE_COMMIT) != sorted(SECOND_RECOVERY_REPAIR_PATHS):
        raise ValueError("Second recovery repair surface changed")
    if changed_paths(REPLACEMENT_SELECTION_COMMIT, source_commit) != sorted(RECOVERY_REPAIR_PATHS):
        raise ValueError("Numeric recovery changed outside the declared A4 repair set")
    if digest_file(LEGACY_FREEZE_PATH) != LEGACY_FREEZE_SHA256 or (
        digest_git_blob(repository_root, f"{LEGACY_FREEZE_COMMIT}:{LEGACY_FREEZE_PATH}")
        != LEGACY_FREEZE_SHA256
    ):
        raise ValueError("Legacy freeze receipt changed")
    if digest_file(SECOND_FREEZE_PATH) != SECOND_FREEZE_SHA256 or (
        digest_git_blob(repository_root, f"{SECOND_FREEZE_COMMIT}:{SECOND_FREEZE_PATH}")
        != SECOND_FREEZE_SHA256
    ):
        raise ValueError("Second freeze receipt changed")
    added_receipts = sorted({
        path
        for path in command(
            "git", "log", "--no-renames", "--diff-filter=A", "--format=", "--name-only", "HEAD"
        ).splitlines()
        if path.startswith("benchmark/evidence/evaluation/pre-score-freeze")
    })
    if added_receipts != sorted([str(LEGACY_FREEZE_PATH), str(SECOND_FREEZE_PATH)]):
        raise ValueError("A4 history contains an alternate pre-score freeze receipt")
    if command("git", "rev-parse", "--abbrev-ref", "HEAD") != "main":
        raise ValueError("V3 freeze must be anchored from main")
    origin_fetch = command("git", "remote", "get-url", "origin")
    origin_push = command("git", "remote", "get-url", "--push", "origin")
    if origin_fetch not in CANONICAL_ORIGIN_URLS or origin_push not in CANONICAL_ORIGIN_URLS:
        raise ValueError("V3 freeze requires the canonical baney75/prooflens GitHub origin")
    remote_rows = command("git", "ls-remote", "origin", "refs/heads/main").split()
    if len(remote_rows) != 2 or remote_rows[0] != source_commit:
        raise ValueError("Public origin/main does not equal the clean A4 source")
    model_lock_bytes = subprocess.check_output(["git", "show", f"{source_commit}:model-lock.json"])
    public_proof = require_anonymous_public_file(
        expected_head=source_commit,
        file_commit=source_commit,
        path=Path("model-lock.json"),
        expected_bytes=model_lock_bytes,
    )
    source_ci = require_successful_public_quality_run(source_commit)
    source_ci["workflowFileSha256"] = digest_git_blob(
        repository_root, f"{source_commit}:{QUALITY_WORKFLOW_PATH}"
    )
    tree_paths = set(command("git", "ls-tree", "-r", "--name-only", source_commit).splitlines())
    if any(path in tree_paths for path in POST_SCORE_FILES) or any(
        any(path.startswith(prefix) for prefix in POST_SCORE_PREFIXES) for path in tree_paths
    ):
        raise ValueError("A4 source already contains replacement evaluation output")
    missing = [path for path in EXPECTED_IMMUTABLE_FILES if path not in tree_paths]
    if missing:
        raise ValueError(f"A4 source lacks immutable files: {missing}")
    immutable_hashes = {
        path: digest_git_blob(repository_root, f"{source_commit}:{path}")
        for path in EXPECTED_IMMUTABLE_FILES
    }
    if any(digest_file(Path(path)) != expected for path, expected in immutable_hashes.items()):
        raise ValueError("A4 worktree differs from its immutable Git blobs")
    failed_hashes = added_path_hashes(repository_root, FAILED_EVALUATION_COMMIT, FAILED_EVALUATION_PATHS)
    replacement_hashes = added_path_hashes(
        repository_root, REPLACEMENT_SELECTION_COMMIT, REPLACEMENT_SELECTION_PATHS
    )
    payload = {
        "schemaVersion": 4,
        "generation": 3,
        "mode": (
            "public replacement-v2 pre-score source freeze before any replacement "
            "confirmatory or web-negative inference"
        ),
        "receiptPath": str(OUTPUT),
        "repository": REPOSITORY_URL,
        "branch": "main",
        "sourceCommit": source_commit,
        "sourceTree": source_tree,
        "remoteObservedHead": remote_rows[0],
        "remoteVerifiedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "publicCommitUrl": f"{REPOSITORY_URL}/commit/{source_commit}",
        "anonymousPublicProof": public_proof,
        "sourcePublicCiProof": source_ci,
        "lineage": {
            "reason": RECOVERY_REASON,
            "legacySourceCommit": LEGACY_SOURCE_COMMIT,
            "legacySourceTree": LEGACY_SOURCE_TREE,
            "legacySourceActionsRunId": LEGACY_SOURCE_ACTIONS_RUN_ID,
            "legacySourceActionsRunUrl": action_url(LEGACY_SOURCE_ACTIONS_RUN_ID),
            "legacySourceActionsConclusion": "success",
            "legacyFreezeCommit": LEGACY_FREEZE_COMMIT,
            "legacyFreezeTree": LEGACY_FREEZE_TREE,
            "legacyFreezeReceiptPath": str(LEGACY_FREEZE_PATH),
            "legacyFreezeReceiptSha256": LEGACY_FREEZE_SHA256,
            "legacyFreezeActionsRunId": LEGACY_FREEZE_ACTIONS_RUN_ID,
            "legacyFreezeActionsRunUrl": action_url(LEGACY_FREEZE_ACTIONS_RUN_ID),
            "legacyFreezeActionsConclusion": "failure",
            "failedRecoverySourceCommit": FAILED_RECOVERY_SOURCE_COMMIT,
            "failedRecoverySourceTree": FAILED_RECOVERY_SOURCE_TREE,
            "failedRecoveryActionsRunId": FAILED_RECOVERY_ACTIONS_RUN_ID,
            "failedRecoveryActionsRunUrl": FAILED_RECOVERY_ACTIONS_RUN_URL,
            "failedRecoveryActionsConclusion": "failure",
            "failedRecoveryReason": FAILED_RECOVERY_REASON,
            "failedRecoveryRepairPaths": FAILED_RECOVERY_REPAIR_PATHS,
            "secondRecoverySourceCommit": SECOND_RECOVERY_SOURCE_COMMIT,
            "secondRecoverySourceTree": SECOND_RECOVERY_SOURCE_TREE,
            "secondRecoveryActionsRunId": SECOND_RECOVERY_ACTIONS_RUN_ID,
            "secondRecoveryActionsRunUrl": action_url(SECOND_RECOVERY_ACTIONS_RUN_ID),
            "secondRecoveryActionsConclusion": "success",
            "secondRecoveryRepairPaths": SECOND_RECOVERY_REPAIR_PATHS,
            "secondFreezeCommit": SECOND_FREEZE_COMMIT,
            "secondFreezeTree": SECOND_FREEZE_TREE,
            "secondFreezeReceiptPath": str(SECOND_FREEZE_PATH),
            "secondFreezeReceiptSha256": SECOND_FREEZE_SHA256,
            "secondFreezeActionsRunId": SECOND_FREEZE_ACTIONS_RUN_ID,
            "secondFreezeActionsRunUrl": action_url(SECOND_FREEZE_ACTIONS_RUN_ID),
            "secondFreezeActionsConclusion": "success",
            "failedEvaluationCommit": FAILED_EVALUATION_COMMIT,
            "failedEvaluationTree": FAILED_EVALUATION_TREE,
            "failedEvaluationActionsRunId": FAILED_EVALUATION_ACTIONS_RUN_ID,
            "failedEvaluationActionsRunUrl": action_url(FAILED_EVALUATION_ACTIONS_RUN_ID),
            "failedEvaluationActionsConclusion": "failure",
            "failedEvaluationReason": FAILED_EVALUATION_REASON,
            "failedEvaluationPathsSha256": failed_hashes,
            "replacementSelectionCommit": REPLACEMENT_SELECTION_COMMIT,
            "replacementSelectionTree": REPLACEMENT_SELECTION_TREE,
            "replacementSelectionActionsRunId": REPLACEMENT_SELECTION_ACTIONS_RUN_ID,
            "replacementSelectionActionsRunUrl": action_url(REPLACEMENT_SELECTION_ACTIONS_RUN_ID),
            "replacementSelectionActionsConclusion": "failure",
            "replacementSelectionReason": REPLACEMENT_SELECTION_REASON,
            "replacementSelectionPathsSha256": replacement_hashes,
            "numericRecoverySourceCommit": source_commit,
            "numericRecoverySourceTree": source_tree,
            "numericRecoveryRepairPaths": RECOVERY_REPAIR_PATHS,
            "repositoryEvidenceLimitation": (
                "Repository history proves the canonical replacement-v2 outputs were absent from the "
                "public source and freeze; it cannot prove that no pixels were viewed outside the recorded workflow."
            ),
        },
        "replacementProtocol": {
            "confirmatory": {
                "protocol": "confirmatory-v2",
                "manifest": "benchmark/manifests/test-v2.jsonl",
                "manifestSha256": REPLACEMENT_CONFIRMATORY_MANIFEST_SHA256,
                "items": 600,
                "labels": {"0": 300, "1": 300},
                "sources": {"coxy7-infinity": 300, "stockimages-cc0": 300},
                "dataRoot": "benchmark/data/replacement-v2",
                "outputDir": "benchmark/evidence/evaluation/confirmatory-v2",
                "name": "prooflens-confirmatory-v2",
            },
            "webNegative": {
                "protocol": "web-negative-v2",
                "manifest": "benchmark/manifests/web-negative-v2.jsonl",
                "manifestSha256": REPLACEMENT_WEB_NEGATIVE_MANIFEST_SHA256,
                "items": 319,
                "labels": {"0": 319},
                "sources": {"stockimages-cc0": 319},
                "dataRoot": "benchmark/data/replacement-v2",
                "outputDir": "benchmark/evidence/evaluation/web-negative-v2",
                "name": "prooflens-web-negative-v2",
            },
            "selection": {
                "path": "benchmark/manifests/replacement-v2-selection.json",
                "sha256": REPLACEMENT_SELECTION_SHA256,
            },
            "parity": {
                "path": "benchmark/manifests/parity-ids-v2.json",
                "sha256": PARITY_IDS_V2_SHA256,
                "items": 60,
                "sources": {"coxy7-infinity": 30, "stockimages-cc0": 30},
            },
            "replayReceiptPath": "benchmark/evidence/evaluation/replay-verification-v2.json",
        },
        "allowedPostScorePaths": EXPECTED_ALLOWED_POST_SCORE_PATHS,
        "immutableFilesSha256": immutable_hashes,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(
        f"Next: commit only {OUTPUT}, push that receipt-only child to public main, "
        "and do not score until its exact public quality run succeeds."
    )


if __name__ == "__main__":
    main()
