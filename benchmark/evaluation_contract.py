"""Fail-closed paths for the immutable ProofLens evaluation protocols."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from hashlib import sha256
from typing import Callable
from urllib.parse import quote
from urllib.request import ProxyHandler, Request, build_opener


CANONICAL_OUTPUT_DIRECTORIES = {
    "validation": Path("benchmark/evidence/evaluation/validation"),
    "confirmatory": Path("benchmark/evidence/evaluation/confirmatory"),
    "web-negative": Path("benchmark/evidence/evaluation/web-negative"),
}
FREEZE_PATH = Path("benchmark/evidence/evaluation/pre-score-freeze-v2.json")
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
EXPECTED_ALLOWED_POST_SCORE_PATHS = [
    "artifacts/**",
    "benchmark/evidence/evaluation/**",
    "BENCHMARK.md",
    "MODEL_CARD.md",
    "README.md",
    "docs/ACCEPTANCE.md",
]
EXPECTED_IMMUTABLE_FILES = [
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
    "scripts/git-object-digest.mjs",
    "scripts/release-stage-policy.mjs",
    "src/inference/calibration.ts",
    "src/inference/detector.ts",
    "src/shared/model-spec.ts",
    "weights/prooflens-cf384.onnx",
]
CANONICAL_ORIGIN_URLS = frozenset({
    "https://github.com/baney75/prooflens",
    "https://github.com/baney75/prooflens.git",
    "git@github.com:baney75/prooflens.git",
    "ssh://git@github.com/baney75/prooflens.git",
})
PUBLIC_GIT_URL = "https://github.com/baney75/prooflens.git"
PUBLIC_RAW_BASE_URL = "https://raw.githubusercontent.com/baney75/prooflens"
POST_SCORE_PREFIXES = (
    "artifacts/browser-parity",
    "benchmark/evidence/evaluation/confirmatory/",
    "benchmark/evidence/evaluation/web-negative/",
)
POST_SCORE_FILES = {"benchmark/evidence/evaluation/replay-verification.json"}


def require_canonical_output_directory(protocol: str, requested: Path, *, repository_root: Path) -> Path:
    try:
        relative = CANONICAL_OUTPUT_DIRECTORIES[protocol]
    except KeyError as error:
        raise ValueError(f"Unknown evaluation protocol: {protocol}") from error
    root = Path(os.path.abspath(repository_root))
    expected = Path(os.path.abspath(root / relative))
    requested_absolute = Path(os.path.abspath(requested))
    if requested_absolute != expected:
        raise ValueError(f"{protocol} evidence must be written to {relative}")
    try:
        components = expected.relative_to(root).parts
    except ValueError as error:  # pragma: no cover - constants are repository-owned
        raise ValueError("Canonical evaluation output escaped the repository") from error
    cursor = root
    for component in components:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"Canonical evaluation output cannot traverse a symlink: {cursor}")
    return expected


def git(repository_root: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.check_output(
        ["git", *arguments],
        cwd=repository_root,
        text=text,
    )
    return result.strip() if text else result


def digest_file(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest_git_blob(repository_root: Path, revision_path: str) -> str:
    with tempfile.TemporaryFile() as stderr_handle:
        process = subprocess.Popen(
            ["git", "cat-file", "blob", revision_path],
            cwd=repository_root,
            stdout=subprocess.PIPE,
            stderr=stderr_handle,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            raise ValueError(f"Could not stream Git blob {revision_path}")
        value = sha256()
        try:
            with process.stdout:
                for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
                    value.update(chunk)
        except OSError as error:
            process.kill()
            process.wait()
            raise ValueError(f"Git blob {revision_path} stream failed") from error
        return_code = process.wait()
        stderr_handle.seek(0)
        stderr = stderr_handle.read(64 * 1024).decode("utf8", errors="replace").strip()
    if return_code != 0:
        detail = f": {stderr}" if stderr else ""
        raise ValueError(f"Git blob {revision_path} failed with code {return_code}{detail}")
    return value.hexdigest()


def anonymous_public_head() -> str:
    """Resolve public main without Git config, helpers, tokens, prompts, or SSH."""
    environment = os.environ.copy()
    for key in tuple(environment):
        if key in {
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "GIT_ASKPASS",
            "SSH_ASKPASS",
            "GIT_CONFIG_COUNT",
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_COMMON_DIR",
        } or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            environment.pop(key, None)
    environment.update({
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    })
    command = [
        "git",
        "-c", "credential.helper=",
        "-c", "core.askPass=",
        "-c", "http.extraHeader=",
        "-c", "http.https://github.com/.extraheader=",
        "ls-remote",
        PUBLIC_GIT_URL,
        "refs/heads/main",
    ]
    with tempfile.TemporaryDirectory(prefix="prooflens-anonymous-git-") as temporary:
        result = subprocess.run(
            command,
            cwd=temporary,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    if result.returncode != 0:
        raise ValueError("Canonical GitHub repository is not anonymously readable")
    rows = result.stdout.split()
    if len(rows) != 2 or len(rows[0]) != 40 or any(character not in "0123456789abcdef" for character in rows[0]):
        raise ValueError("Could not resolve anonymous public main")
    return rows[0]


def anonymous_https_bytes(url: str) -> bytes:
    """Fetch a public byte surface without cookies, proxies, or auth headers."""
    request = Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "Cache-Control": "no-cache",
            "User-Agent": "ProofLens-public-freeze-verifier/1",
        },
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=30) as response:
            if response.status != 200:
                raise ValueError(f"Anonymous public file returned HTTP {response.status}")
            return response.read()
    except OSError as error:
        raise ValueError("Canonical GitHub file is not anonymously readable") from error


def require_anonymous_public_file(
    *,
    expected_head: str,
    file_commit: str,
    path: Path,
    expected_bytes: bytes,
    head_resolver: Callable[[], str] = anonymous_public_head,
    byte_fetcher: Callable[[str], bytes] = anonymous_https_bytes,
) -> dict[str, object]:
    """Bind an exact public main head and exact anonymously readable commit blob."""
    observed_head = head_resolver()
    if observed_head != expected_head:
        raise ValueError("Anonymous public main does not equal the required commit")
    relative = path.as_posix()
    raw_url = f"{PUBLIC_RAW_BASE_URL}/{file_commit}/{quote(relative, safe='/')}"
    observed_bytes = byte_fetcher(f"{raw_url}?prooflens={file_commit}")
    if observed_bytes != expected_bytes:
        raise ValueError(f"Anonymous public bytes differ from the committed {relative}")
    return {
        "method": "credential-free HTTPS git ls-remote plus unauthenticated raw byte match",
        "head": observed_head,
        "fileCommit": file_commit,
        "file": relative,
        "fileSha256": sha256(expected_bytes).hexdigest(),
        "rawUrl": raw_url,
    }


def allowed_post_score_path(path: str, allowed_patterns: list[str]) -> bool:
    if is_unexpected_freeze_receipt(path):
        return False
    return any(
        path.startswith(pattern[:-2]) if pattern.endswith("/**") else path == pattern
        for pattern in allowed_patterns
    )


def is_unexpected_freeze_receipt(path: str) -> bool:
    return path.startswith("benchmark/evidence/evaluation/pre-score-freeze") and path not in {
        str(LEGACY_FREEZE_PATH),
        str(FREEZE_PATH),
    }


def require_public_pre_score_freeze(
    *,
    repository_root: Path,
    allow_public_descendant: bool = False,
    canonical_origin_urls: frozenset[str] = CANONICAL_ORIGIN_URLS,
    anonymous_head_resolver: Callable[[], str] = anonymous_public_head,
    anonymous_byte_fetcher: Callable[[str], bytes] = anonymous_https_bytes,
    recovery_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    """Require the exact recovery freeze-only public commit before sealed inference."""
    contract = recovery_contract or {
        "freezePath": str(FREEZE_PATH),
        "legacyFreezePath": str(LEGACY_FREEZE_PATH),
        "legacySourceCommit": LEGACY_SOURCE_COMMIT,
        "legacyFreezeCommit": LEGACY_FREEZE_COMMIT,
        "legacyFreezeSha256": LEGACY_FREEZE_SHA256,
        "failedRecoverySourceCommit": FAILED_RECOVERY_SOURCE_COMMIT,
        "failedRecoverySourceTree": FAILED_RECOVERY_SOURCE_TREE,
        "failedRecoveryActionsRunId": FAILED_RECOVERY_ACTIONS_RUN_ID,
        "failedRecoveryActionsRunUrl": FAILED_RECOVERY_ACTIONS_RUN_URL,
        "failedRecoveryReason": FAILED_RECOVERY_REASON,
        "failedRecoveryRepairPaths": FAILED_RECOVERY_REPAIR_PATHS,
        "reason": RECOVERY_REASON,
        "repairPaths": RECOVERY_REPAIR_PATHS,
        "immutableFiles": EXPECTED_IMMUTABLE_FILES,
        "allowedPostScorePaths": EXPECTED_ALLOWED_POST_SCORE_PATHS,
    }
    active_freeze_path = Path(str(contract["freezePath"]))
    legacy_freeze_path = Path(str(contract["legacyFreezePath"]))
    legacy_source_commit = str(contract["legacySourceCommit"])
    legacy_freeze_commit = str(contract["legacyFreezeCommit"])
    legacy_freeze_sha256 = str(contract["legacyFreezeSha256"])
    failed_recovery_source_commit = str(contract["failedRecoverySourceCommit"])
    failed_recovery_source_tree = str(contract["failedRecoverySourceTree"])
    failed_recovery_actions_run_id = int(contract["failedRecoveryActionsRunId"])
    failed_recovery_actions_run_url = str(contract["failedRecoveryActionsRunUrl"])
    failed_recovery_reason = str(contract["failedRecoveryReason"])
    failed_repair_paths_expected = [
        str(path) for path in contract["failedRecoveryRepairPaths"]  # type: ignore[union-attr]
    ]
    recovery_reason = str(contract["reason"])
    repair_paths_expected = [str(path) for path in contract["repairPaths"]]  # type: ignore[union-attr]
    immutable_files_expected = [str(path) for path in contract["immutableFiles"]]  # type: ignore[union-attr]
    allowed_paths_expected = [str(path) for path in contract["allowedPostScorePaths"]]  # type: ignore[union-attr]
    freeze_path = repository_root / active_freeze_path
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    if (
        freeze.get("schemaVersion") != 3
        or freeze.get("generation") != 2
        or freeze.get("mode")
        != "public second-recovery pre-score source freeze before any confirmatory or web-negative inference"
        or freeze.get("receiptPath") != str(active_freeze_path)
    ):
        raise ValueError("Only the recovery pre-score freeze can authorize sealed inference")
    recovery = freeze.get("recovery")
    if not isinstance(recovery, dict) or (
        recovery.get("reason") != recovery_reason
        or recovery.get("legacySourceCommit") != legacy_source_commit
        or recovery.get("legacyFreezeCommit") != legacy_freeze_commit
        or recovery.get("legacyReceiptPath") != str(legacy_freeze_path)
        or recovery.get("legacyReceiptSha256") != legacy_freeze_sha256
        or recovery.get("failedRecoverySourceCommit") != failed_recovery_source_commit
        or recovery.get("failedRecoverySourceTree") != failed_recovery_source_tree
        or recovery.get("failedRecoveryActionsRunId") != failed_recovery_actions_run_id
        or recovery.get("failedRecoveryActionsRunUrl") != failed_recovery_actions_run_url
        or recovery.get("failedRecoveryReason") != failed_recovery_reason
        or recovery.get("failedRecoveryRepairPaths") != failed_repair_paths_expected
        or recovery.get("repairPaths") != repair_paths_expected
        or "cannot prove" not in str(recovery.get("repositoryEvidenceLimitation", ""))
    ):
        raise ValueError("Recovery pre-score lineage changed")
    source_commit = str(freeze.get("sourceCommit", ""))
    source_tree = str(freeze.get("sourceTree", ""))
    if source_tree != git(repository_root, "rev-parse", f"{source_commit}^{{tree}}"):
        raise ValueError("Pre-score freeze source commit/tree changed")
    if (
        git(repository_root, "rev-parse", f"{source_commit}^") != failed_recovery_source_commit
        or git(repository_root, "rev-parse", f"{failed_recovery_source_commit}^") != legacy_freeze_commit
        or git(repository_root, "rev-parse", f"{legacy_freeze_commit}^") != legacy_source_commit
        or git(repository_root, "rev-parse", f"{failed_recovery_source_commit}^{{tree}}")
        != failed_recovery_source_tree
    ):
        raise ValueError("Second recovery source lineage changed")
    legacy_receipt = repository_root / legacy_freeze_path
    if not legacy_receipt.is_file() or digest_file(legacy_receipt) != legacy_freeze_sha256:
        raise ValueError("Legacy freeze receipt changed in the recovery source")
    if digest_git_blob(repository_root, f"{legacy_freeze_commit}:{legacy_freeze_path}") != legacy_freeze_sha256:
        raise ValueError("Legacy freeze receipt changed in public history")
    legacy_paths = str(
        git(repository_root, "diff-tree", "--no-renames", "--no-commit-id", "--name-only", "-r", legacy_freeze_commit)
    ).splitlines()
    if legacy_paths != [str(legacy_freeze_path)]:
        raise ValueError("Legacy freeze commit shape changed")
    failed_repair_paths = sorted(
        str(git(
            repository_root,
            "diff", "--no-renames", "--name-only",
            f"{legacy_freeze_commit}..{failed_recovery_source_commit}",
        ))
        .splitlines()
    )
    if failed_repair_paths != sorted(failed_repair_paths_expected):
        raise ValueError("Failed first recovery source changed outside its exact repair set")
    repair_paths = sorted(
        str(git(
            repository_root,
            "diff", "--no-renames", "--name-only",
            f"{failed_recovery_source_commit}..{source_commit}",
        )).splitlines()
    )
    if repair_paths != sorted(repair_paths_expected):
        raise ValueError("Recovery source changed outside its exact repair set")
    additions = str(
        git(repository_root, "log", "--no-renames", "--diff-filter=A", "--format=%H", "--", str(active_freeze_path))
    ).splitlines()
    if len(additions) != 1:
        raise ValueError("Pre-score freeze must be added exactly once")
    freeze_commit = additions[0]
    ancestry = str(git(repository_root, "rev-list", "--parents", "-n", "1", freeze_commit)).split()
    if ancestry != [freeze_commit, source_commit]:
        raise ValueError("Pre-score freeze commit must be a freeze-only child of the source commit")
    committed_freeze = git(repository_root, "show", f"{freeze_commit}:{active_freeze_path}", text=False)
    if committed_freeze != freeze_bytes:
        raise ValueError("Pre-score freeze receipt changed after its public commit")
    changed = str(
        git(repository_root, "diff-tree", "--no-renames", "--no-commit-id", "--name-only", "-r", freeze_commit)
    ).splitlines()
    if changed != [str(active_freeze_path)]:
        raise ValueError("Pre-score freeze commit changed more than the freeze receipt")
    added_freeze_receipts = sorted({
        path
        for path in str(
            git(repository_root, "log", "--no-renames", "--diff-filter=A", "--format=", "--name-only", "HEAD")
        ).splitlines()
        if path.startswith("benchmark/evidence/evaluation/pre-score-freeze")
    })
    if added_freeze_receipts != sorted([str(legacy_freeze_path), str(active_freeze_path)]):
        raise ValueError("An alternate pre-score freeze receipt exists in public history")
    frozen_tree_paths = set(
        str(git(repository_root, "ls-tree", "-r", "--name-only", freeze_commit)).splitlines()
    )
    if any(any(path.startswith(prefix) for prefix in POST_SCORE_PREFIXES) for path in frozen_tree_paths) or any(
        path in POST_SCORE_FILES for path in frozen_tree_paths
    ):
        raise ValueError("Pre-score freeze commit already contains sealed evaluation output")
    status_rows = str(
        git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    ).splitlines()
    dirty_paths = [row[3:].split(" -> ")[-1] for row in status_rows if len(row) > 3]
    if any(
        not path.startswith("benchmark/evidence/evaluation/") or is_unexpected_freeze_receipt(path)
        for path in dirty_paths
    ):
        raise ValueError("Sealed evaluation refuses a dirty source worktree")
    origin_fetch = str(git(repository_root, "remote", "get-url", "origin"))
    origin_push = str(git(repository_root, "remote", "get-url", "--push", "origin"))
    if origin_fetch not in canonical_origin_urls or origin_push not in canonical_origin_urls:
        raise ValueError("Public freeze requires the canonical baney75/prooflens GitHub origin")
    local_head = str(git(repository_root, "rev-parse", "HEAD"))
    if allow_public_descendant:
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", freeze_commit, local_head],
            cwd=repository_root,
            capture_output=True,
            check=False,
        )
        if ancestor.returncode != 0:
            raise ValueError("Replay HEAD must descend from the public freeze-only commit")
    elif local_head != freeze_commit:
        raise ValueError("Local HEAD must equal the public freeze-only commit before first sealed inference")
    remote = str(git(repository_root, "ls-remote", "origin", "refs/heads/main")).split()
    if len(remote) != 2 or remote[0] != local_head:
        raise ValueError("Public origin/main must equal the local inference commit")
    public_proof = require_anonymous_public_file(
        expected_head=local_head,
        file_commit=freeze_commit,
        path=active_freeze_path,
        expected_bytes=freeze_bytes,
        head_resolver=anonymous_head_resolver,
        byte_fetcher=anonymous_byte_fetcher,
    )
    immutable_hashes = freeze.get("immutableFilesSha256")
    if not isinstance(immutable_hashes, dict) or sorted(immutable_hashes) != sorted(immutable_files_expected):
        raise ValueError("Pre-score freeze immutable-file list changed")
    for path, expected_hash in immutable_hashes.items():
        if not isinstance(path, str) or not isinstance(expected_hash, str) or (
            len(expected_hash) != 64 or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError("Pre-score freeze contains an invalid immutable hash")
        current = repository_root / str(path)
        if (
            not current.is_file()
            or digest_file(current) != expected_hash
            or digest_git_blob(repository_root, f"{source_commit}:{path}") != expected_hash
        ):
            raise ValueError(f"Immutable pre-score file changed: {path}")
    allowed_patterns = freeze.get("allowedPostScorePaths")
    if allowed_patterns != allowed_paths_expected:
        raise ValueError("Pre-score freeze post-score path policy changed")
    changed_paths = str(
        git(repository_root, "diff", "--no-renames", "--name-only", f"{source_commit}..{local_head}")
    ).splitlines()
    if any(not allowed_post_score_path(path, allowed_patterns) for path in changed_paths):
        raise ValueError("Public replay commit changed a frozen source path")
    return {**freeze, "freezeCommit": freeze_commit, "anonymousInferenceProof": public_proof}
