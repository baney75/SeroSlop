"""Fail-closed paths for the immutable ProofLens evaluation protocols."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from typing import Callable
from urllib.parse import quote, urlencode
from urllib.request import ProxyHandler, Request, build_opener


CANONICAL_OUTPUT_DIRECTORIES = {
    "confirmatory-v2": Path("benchmark/evidence/evaluation/confirmatory-v2"),
    "web-negative-v2": Path("benchmark/evidence/evaluation/web-negative-v2"),
}
FREEZE_PATH = Path("benchmark/evidence/evaluation/pre-score-freeze-v3.json")
SECOND_FREEZE_PATH = Path("benchmark/evidence/evaluation/pre-score-freeze-v2.json")
LEGACY_FREEZE_PATH = Path("benchmark/evidence/evaluation/pre-score-freeze.json")
LEGACY_SOURCE_COMMIT = "0771a9422b552e2023e5150fb6c8b4238b811a74"
LEGACY_SOURCE_TREE = "6b4575742ea847c747fe8bea216c3b2c7b357068"
LEGACY_FREEZE_COMMIT = "2bd0c4757f6059c57879414a5dba77629d66460e"
LEGACY_FREEZE_TREE = "ccf2de97024090245189ccbb7f308bce392565b9"
LEGACY_FREEZE_SHA256 = "400fd2b7a7cd84b063f81799eaf3829f770220c41f58dff53e7caebd1a145c34"
LEGACY_SOURCE_ACTIONS_RUN_ID = 31843811845
LEGACY_FREEZE_ACTIONS_RUN_ID = 31844088383
FAILED_RECOVERY_SOURCE_COMMIT = "99861df575854511c685d7b8f90acdc7ed4e5923"
FAILED_RECOVERY_SOURCE_TREE = "204594f26118d8c1c3add9dfef3a6050949772e1"
FAILED_RECOVERY_ACTIONS_RUN_ID = 31846361076
FAILED_RECOVERY_ACTIONS_RUN_URL = "https://github.com/baney75/prooflens/actions/runs/31846361076"
FAILED_RECOVERY_REASON = "ci-missing-inference-dependencies-before-v2-guard"
SECOND_RECOVERY_SOURCE_COMMIT = "17124df0bf390c2c2c27583ae81f06b65ead2e3f"
SECOND_RECOVERY_SOURCE_TREE = "d8ecc3e0a60399e955b90e96f792baa30f65045c"
SECOND_RECOVERY_ACTIONS_RUN_ID = 31847256279
SECOND_FREEZE_COMMIT = "2757a4ff267d580a7dd8ad4918885441fa887f1b"
SECOND_FREEZE_TREE = "38111c60c95542a0048e9e49328a6a7e44149a95"
SECOND_FREEZE_SHA256 = "0acd317dba772efd534b8900b627851fed0d04d54f814529b71c701465806017"
SECOND_FREEZE_ACTIONS_RUN_ID = 31847694896
FAILED_EVALUATION_COMMIT = "45400803a19b967c8cae0bbf4817fe984aea349a"
FAILED_EVALUATION_TREE = "17463eae7285e6292c8f3aeb4fc3c0f1803ef6fa"
FAILED_EVALUATION_ACTIONS_RUN_ID = 31848762781
FAILED_EVALUATION_REASON = "float32-sigmoid-probability-contract-violation"
REPLACEMENT_SELECTION_COMMIT = "baaf3eb0b7a22f635d2ec6a3cb2496b9e76313b8"
REPLACEMENT_SELECTION_TREE = "2ddfec1f346eaa0f6bf0e797635af2121fe83866"
REPLACEMENT_SELECTION_ACTIONS_RUN_ID = 31853385690
REPLACEMENT_SELECTION_REASON = "score-blind-replacement-selection-before-numeric-repair"
RECOVERY_REASON = "binary64-sigmoid-and-replacement-protocol-recovery"
QUALITY_WORKFLOW_NAME = "quality"
QUALITY_WORKFLOW_PATH = ".github/workflows/quality.yml"
REPLACEMENT_CONFIRMATORY_MANIFEST_SHA256 = "773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8"
REPLACEMENT_WEB_NEGATIVE_MANIFEST_SHA256 = "6a1287bae6826811c81cbebab79a1bc6abb475fde70c9aa1529c390ed97014c9"
REPLACEMENT_SELECTION_SHA256 = "4417bf34db53993c2ccf459a18947c92a52118b85a6f740740d981bb8e223f3c"
PARITY_IDS_V2_SHA256 = "0f0e72ac4bd91549af10a76c494138b6cf0c22328d904134b67be82d79badf99"
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
SECOND_RECOVERY_REPAIR_PATHS = [
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
RECOVERY_REPAIR_PATHS = [
    "BENCHMARK.md", "MODEL_CARD.md", "README.md",
    "benchmark/bootstrap_ci.py", "benchmark/bootstrap_fpr.py", "benchmark/evaluate.py",
    "benchmark/evaluation_contract.py", "benchmark/manifests/README.md",
    "benchmark/manifests/parity-ids-v2.json", "benchmark/prediction_contract.py",
    "benchmark/prepare_parity.py", "benchmark/recovery_v3/README.md",
    "benchmark/select_parity_ids.py", "benchmark/test_integrity_contracts.py",
    "benchmark/verify_evaluation_evidence.py", "benchmark/write_pre_score_freeze.py",
    "docs/ACCEPTANCE.md", "package.json", "scripts/check-benchmark-evidence.mjs",
    "scripts/check-pre-score-freeze.mjs", "scripts/check-pre-score-stage.mjs",
    "scripts/release-stage-policy.mjs", "scripts/test-release-stage-policy.mjs",
]
EXPECTED_ALLOWED_POST_SCORE_PATHS = [
    "artifacts/**",
    "benchmark/evidence/evaluation/confirmatory-v2/**",
    "benchmark/evidence/evaluation/web-negative-v2/**",
    "benchmark/evidence/evaluation/replay-verification-v2.json",
    "BENCHMARK.md",
    "MODEL_CARD.md",
    "README.md",
    "docs/ACCEPTANCE.md",
]
EXPECTED_IMMUTABLE_FILES = [
    ".github/workflows/quality.yml",
    "benchmark/bootstrap_ci.py",
    "benchmark/bootstrap_fpr.py",
    "benchmark/evaluate.py",
    "benchmark/evaluation_contract.py",
    "benchmark/evidence/evaluation/confirmatory/failed-evaluation.json",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-complete.json",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-original-predictions.jsonl",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-screenshot-predictions.jsonl",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-social-heavy-predictions.jsonl",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-social-q75-predictions.jsonl",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-summary.json",
    "benchmark/evidence/evaluation/validation/prooflens-validation-complete.json",
    "benchmark/evidence/evaluation/validation/prooflens-validation-original-predictions.jsonl",
    "benchmark/evidence/evaluation/validation/prooflens-validation-screenshot-predictions.jsonl",
    "benchmark/evidence/evaluation/validation/prooflens-validation-social-heavy-predictions.jsonl",
    "benchmark/evidence/evaluation/validation/prooflens-validation-social-q75-predictions.jsonl",
    "benchmark/evidence/evaluation/validation/prooflens-validation-summary.json",
    "benchmark/evidence/large/calibration.json",
    "benchmark/evidence/large/selection-summary.json",
    "benchmark/evidence/large/training-summary.json",
    "benchmark/large/recipe.json",
    "benchmark/manifests/historical-perceptual-exclusions-v1.json.gz",
    "benchmark/manifests/parity-ids-v2.json",
    "benchmark/manifests/replacement-v2-attribution.json",
    "benchmark/manifests/replacement-v2-perceptual-review.json",
    "benchmark/manifests/replacement-v2-selection.json",
    "benchmark/manifests/test-v2.jsonl",
    "benchmark/manifests/web-negative-v2.jsonl",
    "benchmark/manifests/README.md",
    "benchmark/prediction_contract.py",
    "benchmark/prepare_parity.py",
    "benchmark/recovery_v3/README.md",
    "benchmark/recovery_v3/build_historical_index.py",
    "benchmark/recovery_v3/legacy-test.jsonl.gz",
    "benchmark/recovery_v3/legacy-validation.jsonl.gz",
    "benchmark/recovery_v3/perceptual-review.json",
    "benchmark/recovery_v3/prepare.py",
    "benchmark/recovery_v3/recipe.json",
    "benchmark/recovery_v3/test_prepare.py",
    "benchmark/recovery_v3/verify.py",
    "benchmark/run_release_replay.py",
    "benchmark/select_parity_ids.py",
    "benchmark/verify_evaluation_evidence.py",
    "benchmark/write_pre_score_freeze.py",
    "model-lock.json",
    "scripts/check-benchmark-evidence.mjs",
    "scripts/check-pre-score-freeze.mjs",
    "scripts/check-pre-score-stage.mjs",
    "scripts/git-object-digest.mjs",
    "scripts/release-stage-policy.mjs",
    "scripts/test-release-stage-policy.mjs",
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
    "artifacts/",
    "benchmark/evidence/evaluation/confirmatory-v2/",
    "benchmark/evidence/evaluation/web-negative-v2/",
)
POST_SCORE_FILES = {"benchmark/evidence/evaluation/replay-verification-v2.json"}

FAILED_EVALUATION_PATHS = [
    "benchmark/evidence/evaluation/confirmatory/failed-evaluation.json",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-complete.json",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-original-predictions.jsonl",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-screenshot-predictions.jsonl",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-social-heavy-predictions.jsonl",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-social-q75-predictions.jsonl",
    "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-summary.json",
]
REPLACEMENT_SELECTION_PATHS = [
    "benchmark/manifests/historical-perceptual-exclusions-v1.json.gz",
    "benchmark/manifests/replacement-v2-attribution.json",
    "benchmark/manifests/replacement-v2-perceptual-review.json",
    "benchmark/manifests/replacement-v2-selection.json",
    "benchmark/manifests/test-v2.jsonl",
    "benchmark/manifests/web-negative-v2.jsonl",
    "benchmark/recovery_v3/README.md",
    "benchmark/recovery_v3/build_historical_index.py",
    "benchmark/recovery_v3/legacy-test.jsonl.gz",
    "benchmark/recovery_v3/legacy-validation.jsonl.gz",
    "benchmark/recovery_v3/perceptual-review.json",
    "benchmark/recovery_v3/prepare.py",
    "benchmark/recovery_v3/recipe.json",
    "benchmark/recovery_v3/test_prepare.py",
    "benchmark/recovery_v3/verify.py",
]


def require_canonical_repository_path(
    requested: Path,
    relative: Path,
    *,
    repository_root: Path,
    label: str,
) -> Path:
    """Require one repository-owned lexical path with no symlink component."""
    root = Path(os.path.abspath(repository_root))
    expected = Path(os.path.abspath(root / relative))
    requested_absolute = Path(os.path.abspath(requested))
    if requested_absolute != expected:
        raise ValueError(f"Canonical {label} must be {relative}")
    try:
        components = expected.relative_to(root).parts
    except ValueError as error:  # pragma: no cover - constants are repository-owned
        raise ValueError(f"Canonical {label} escaped the repository") from error
    cursor = root
    if cursor.is_symlink():
        raise ValueError(f"Canonical {label} cannot traverse a symlink: {cursor}")
    for component in components:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"Canonical {label} cannot traverse a symlink: {cursor}")
    return expected


def require_canonical_output_directory(protocol: str, requested: Path, *, repository_root: Path) -> Path:
    try:
        relative = CANONICAL_OUTPUT_DIRECTORIES[protocol]
    except KeyError as error:
        raise ValueError(f"Unknown evaluation protocol: {protocol}") from error
    return require_canonical_repository_path(
        requested,
        relative,
        repository_root=repository_root,
        label=f"{protocol} evidence output",
    )


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


def anonymous_public_actions_runs(head_sha: str) -> list[dict[str, object]]:
    """Read public push runs without credentials, cookies, proxies, or ambient auth."""
    if len(head_sha) != 40 or any(character not in "0123456789abcdef" for character in head_sha):
        raise ValueError("GitHub Actions lookup requires a full lowercase commit SHA")
    query = urlencode({"event": "push", "head_sha": head_sha, "per_page": 100})
    request = Request(
        f"https://api.github.com/repos/baney75/prooflens/actions/runs?{query}",
        headers={
            "Accept": "application/vnd.github+json",
            "Cache-Control": "no-cache",
            "User-Agent": "ProofLens-public-actions-verifier/1",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=30) as response:
            if response.status != 200:
                raise ValueError(f"Public GitHub Actions API returned HTTP {response.status}")
            payload = json.load(response)
    except OSError as error:
        raise ValueError("Public GitHub Actions state is unavailable") from error
    rows = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Public GitHub Actions API returned an invalid run list")
    return rows


def require_successful_public_quality_run(
    head_sha: str,
    *,
    runs_resolver: Callable[[str], list[dict[str, object]]] = anonymous_public_actions_runs,
) -> dict[str, object]:
    """Require a completed successful public quality push run for one exact head."""
    candidates = []
    for row in runs_resolver(head_sha):
        if (
            row.get("head_sha") == head_sha
            and row.get("name") == QUALITY_WORKFLOW_NAME
            and row.get("path") == QUALITY_WORKFLOW_PATH
            and row.get("event") == "push"
            and row.get("status") == "completed"
            and row.get("conclusion") == "success"
            and isinstance(row.get("id"), int)
            and isinstance(row.get("html_url"), str)
        ):
            candidates.append(row)
    if not candidates:
        raise ValueError("Exact public freeze head lacks a completed successful quality push run")
    selected = max(candidates, key=lambda row: int(row["id"]))
    return {
        "workflowName": QUALITY_WORKFLOW_NAME,
        "workflowPath": QUALITY_WORKFLOW_PATH,
        "event": "push",
        "headSha": head_sha,
        "status": "completed",
        "conclusion": "success",
        "runId": int(selected["id"]),
        "runUrl": str(selected["html_url"]),
        "checkedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


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
        str(SECOND_FREEZE_PATH),
        str(FREEZE_PATH),
    }


def _changed_paths(repository_root: Path, older: str, newer: str) -> list[str]:
    return sorted(
        str(git(repository_root, "diff", "--no-renames", "--name-only", f"{older}..{newer}"))
        .splitlines()
    )


def _added_path_map(repository_root: Path, commit: str) -> dict[str, str]:
    rows = str(
        git(
            repository_root,
            "diff-tree", "--no-renames", "--no-commit-id", "--name-status", "-r", commit,
        )
    ).splitlines()
    result: dict[str, str] = {}
    for row in rows:
        status, separator, path = row.partition("\t")
        if not separator or status != "A" or not path:
            raise ValueError(f"Commit {commit} is not an additions-only evidence packet")
        result[path] = digest_git_blob(repository_root, f"{commit}:{path}")
    return result


def _require_commit_tree(repository_root: Path, commit: str, expected_tree: str, label: str) -> None:
    if str(git(repository_root, "rev-parse", f"{commit}^{{tree}}")) != expected_tree:
        raise ValueError(f"{label} tree changed")


def _require_direct_parent(repository_root: Path, child: str, parent: str, label: str) -> None:
    if str(git(repository_root, "rev-parse", f"{child}^")) != parent:
        raise ValueError(f"{label} lineage changed")


def require_public_pre_score_freeze(
    *,
    repository_root: Path,
    allow_public_descendant: bool = False,
    require_freeze_ci_success: bool = False,
    canonical_origin_urls: frozenset[str] = CANONICAL_ORIGIN_URLS,
    anonymous_head_resolver: Callable[[], str] = anonymous_public_head,
    anonymous_byte_fetcher: Callable[[str], bytes] = anonymous_https_bytes,
    actions_runs_resolver: Callable[[str], list[dict[str, object]]] = anonymous_public_actions_runs,
    recovery_contract: dict[str, object] | None = None,
) -> dict[str, object]:
    """Require the exact public V3 freeze before replacement-v2 inference."""
    contract = recovery_contract or {
        "freezePath": str(FREEZE_PATH),
        "secondFreezePath": str(SECOND_FREEZE_PATH),
        "legacyFreezePath": str(LEGACY_FREEZE_PATH),
        "legacySourceCommit": LEGACY_SOURCE_COMMIT,
        "legacySourceTree": LEGACY_SOURCE_TREE,
        "legacySourceActionsRunId": LEGACY_SOURCE_ACTIONS_RUN_ID,
        "legacyFreezeCommit": LEGACY_FREEZE_COMMIT,
        "legacyFreezeTree": LEGACY_FREEZE_TREE,
        "legacyFreezeSha256": LEGACY_FREEZE_SHA256,
        "legacyFreezeActionsRunId": LEGACY_FREEZE_ACTIONS_RUN_ID,
        "failedRecoverySourceCommit": FAILED_RECOVERY_SOURCE_COMMIT,
        "failedRecoverySourceTree": FAILED_RECOVERY_SOURCE_TREE,
        "failedRecoveryActionsRunId": FAILED_RECOVERY_ACTIONS_RUN_ID,
        "failedRecoveryActionsRunUrl": FAILED_RECOVERY_ACTIONS_RUN_URL,
        "failedRecoveryReason": FAILED_RECOVERY_REASON,
        "failedRecoveryRepairPaths": FAILED_RECOVERY_REPAIR_PATHS,
        "secondRecoverySourceCommit": SECOND_RECOVERY_SOURCE_COMMIT,
        "secondRecoverySourceTree": SECOND_RECOVERY_SOURCE_TREE,
        "secondRecoveryActionsRunId": SECOND_RECOVERY_ACTIONS_RUN_ID,
        "secondRecoveryRepairPaths": SECOND_RECOVERY_REPAIR_PATHS,
        "secondFreezeCommit": SECOND_FREEZE_COMMIT,
        "secondFreezeTree": SECOND_FREEZE_TREE,
        "secondFreezeSha256": SECOND_FREEZE_SHA256,
        "secondFreezeActionsRunId": SECOND_FREEZE_ACTIONS_RUN_ID,
        "failedEvaluationCommit": FAILED_EVALUATION_COMMIT,
        "failedEvaluationTree": FAILED_EVALUATION_TREE,
        "failedEvaluationActionsRunId": FAILED_EVALUATION_ACTIONS_RUN_ID,
        "failedEvaluationReason": FAILED_EVALUATION_REASON,
        "failedEvaluationPaths": FAILED_EVALUATION_PATHS,
        "replacementSelectionCommit": REPLACEMENT_SELECTION_COMMIT,
        "replacementSelectionTree": REPLACEMENT_SELECTION_TREE,
        "replacementSelectionActionsRunId": REPLACEMENT_SELECTION_ACTIONS_RUN_ID,
        "replacementSelectionReason": REPLACEMENT_SELECTION_REASON,
        "replacementSelectionPaths": REPLACEMENT_SELECTION_PATHS,
        "reason": RECOVERY_REASON,
        "repairPaths": RECOVERY_REPAIR_PATHS,
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
        "replacementFilesSha256": {
            "benchmark/manifests/test-v2.jsonl": REPLACEMENT_CONFIRMATORY_MANIFEST_SHA256,
            "benchmark/manifests/web-negative-v2.jsonl": REPLACEMENT_WEB_NEGATIVE_MANIFEST_SHA256,
            "benchmark/manifests/replacement-v2-selection.json": REPLACEMENT_SELECTION_SHA256,
            "benchmark/manifests/parity-ids-v2.json": PARITY_IDS_V2_SHA256,
        },
        "immutableFiles": EXPECTED_IMMUTABLE_FILES,
        "allowedPostScorePaths": EXPECTED_ALLOWED_POST_SCORE_PATHS,
    }
    active_freeze_path = Path(str(contract["freezePath"]))
    second_freeze_path = Path(str(contract["secondFreezePath"]))
    legacy_freeze_path = Path(str(contract["legacyFreezePath"]))
    legacy_source_commit = str(contract["legacySourceCommit"])
    legacy_source_tree = str(contract["legacySourceTree"])
    legacy_source_actions_run_id = int(contract["legacySourceActionsRunId"])
    legacy_freeze_commit = str(contract["legacyFreezeCommit"])
    legacy_freeze_tree = str(contract["legacyFreezeTree"])
    legacy_freeze_sha256 = str(contract["legacyFreezeSha256"])
    legacy_freeze_actions_run_id = int(contract["legacyFreezeActionsRunId"])
    failed_recovery_source_commit = str(contract["failedRecoverySourceCommit"])
    failed_recovery_source_tree = str(contract["failedRecoverySourceTree"])
    failed_recovery_actions_run_id = int(contract["failedRecoveryActionsRunId"])
    failed_recovery_actions_run_url = str(contract["failedRecoveryActionsRunUrl"])
    failed_recovery_reason = str(contract["failedRecoveryReason"])
    failed_repair_paths_expected = [
        str(path) for path in contract["failedRecoveryRepairPaths"]  # type: ignore[union-attr]
    ]
    second_recovery_source_commit = str(contract["secondRecoverySourceCommit"])
    second_recovery_source_tree = str(contract["secondRecoverySourceTree"])
    second_recovery_actions_run_id = int(contract["secondRecoveryActionsRunId"])
    second_repair_paths_expected = [
        str(path) for path in contract["secondRecoveryRepairPaths"]  # type: ignore[union-attr]
    ]
    second_freeze_commit_expected = str(contract["secondFreezeCommit"])
    second_freeze_tree = str(contract["secondFreezeTree"])
    second_freeze_sha256 = str(contract["secondFreezeSha256"])
    second_freeze_actions_run_id = int(contract["secondFreezeActionsRunId"])
    failed_evaluation_commit = str(contract["failedEvaluationCommit"])
    failed_evaluation_tree = str(contract["failedEvaluationTree"])
    failed_evaluation_actions_run_id = int(contract["failedEvaluationActionsRunId"])
    failed_evaluation_reason = str(contract["failedEvaluationReason"])
    failed_evaluation_paths = [str(path) for path in contract["failedEvaluationPaths"]]  # type: ignore[union-attr]
    replacement_selection_commit = str(contract["replacementSelectionCommit"])
    replacement_selection_tree = str(contract["replacementSelectionTree"])
    replacement_selection_actions_run_id = int(contract["replacementSelectionActionsRunId"])
    replacement_selection_reason = str(contract["replacementSelectionReason"])
    replacement_selection_paths = [
        str(path) for path in contract["replacementSelectionPaths"]  # type: ignore[union-attr]
    ]
    recovery_reason = str(contract["reason"])
    repair_paths_expected = [str(path) for path in contract["repairPaths"]]  # type: ignore[union-attr]
    replacement_protocol_expected = contract["replacementProtocol"]
    replacement_files_sha256 = dict(contract["replacementFilesSha256"])  # type: ignore[arg-type]
    immutable_files_expected = [str(path) for path in contract["immutableFiles"]]  # type: ignore[union-attr]
    allowed_paths_expected = [str(path) for path in contract["allowedPostScorePaths"]]  # type: ignore[union-attr]
    freeze_path = repository_root / active_freeze_path
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    if (
        freeze.get("schemaVersion") != 4
        or freeze.get("generation") != 3
        or freeze.get("mode")
        != "public replacement-v2 pre-score source freeze before any replacement confirmatory or web-negative inference"
        or freeze.get("receiptPath") != str(active_freeze_path)
        or freeze.get("repository") != "https://github.com/baney75/prooflens"
        or freeze.get("branch") != "main"
    ):
        raise ValueError("Only the V3 replacement pre-score freeze can authorize inference")
    source_commit = str(freeze.get("sourceCommit", ""))
    source_tree = str(freeze.get("sourceTree", ""))
    if (
        freeze.get("remoteObservedHead") != source_commit
        or freeze.get("publicCommitUrl") != f"https://github.com/baney75/prooflens/commit/{source_commit}"
    ):
        raise ValueError("V3 source publication metadata changed")
    if source_tree != git(repository_root, "rev-parse", f"{source_commit}^{{tree}}"):
        raise ValueError("Pre-score freeze source commit/tree changed")
    _require_commit_tree(repository_root, legacy_source_commit, legacy_source_tree, "Legacy source")
    _require_commit_tree(repository_root, legacy_freeze_commit, legacy_freeze_tree, "Legacy freeze")
    _require_commit_tree(repository_root, failed_recovery_source_commit, failed_recovery_source_tree, "Failed recovery source")
    _require_commit_tree(repository_root, second_recovery_source_commit, second_recovery_source_tree, "Second recovery source")
    _require_commit_tree(repository_root, second_freeze_commit_expected, second_freeze_tree, "Second freeze")
    _require_commit_tree(repository_root, failed_evaluation_commit, failed_evaluation_tree, "Failed evaluation")
    _require_commit_tree(repository_root, replacement_selection_commit, replacement_selection_tree, "Replacement selection")
    _require_direct_parent(repository_root, legacy_freeze_commit, legacy_source_commit, "Legacy freeze")
    _require_direct_parent(repository_root, failed_recovery_source_commit, legacy_freeze_commit, "Failed recovery")
    _require_direct_parent(repository_root, second_recovery_source_commit, failed_recovery_source_commit, "Second recovery")
    _require_direct_parent(repository_root, second_freeze_commit_expected, second_recovery_source_commit, "Second freeze")
    _require_direct_parent(repository_root, failed_evaluation_commit, second_freeze_commit_expected, "Failed evaluation")
    _require_direct_parent(repository_root, replacement_selection_commit, failed_evaluation_commit, "Replacement selection")
    _require_direct_parent(repository_root, source_commit, replacement_selection_commit, "Numeric recovery source")
    legacy_receipt = repository_root / legacy_freeze_path
    if not legacy_receipt.is_file() or digest_file(legacy_receipt) != legacy_freeze_sha256:
        raise ValueError("Legacy freeze receipt changed in the recovery source")
    if digest_git_blob(repository_root, f"{legacy_freeze_commit}:{legacy_freeze_path}") != legacy_freeze_sha256:
        raise ValueError("Legacy freeze receipt changed in public history")
    second_receipt = repository_root / second_freeze_path
    if not second_receipt.is_file() or digest_file(second_receipt) != second_freeze_sha256:
        raise ValueError("Second freeze receipt changed in the numeric recovery source")
    if digest_git_blob(repository_root, f"{second_freeze_commit_expected}:{second_freeze_path}") != second_freeze_sha256:
        raise ValueError("Second freeze receipt changed in public history")
    legacy_paths = str(
        git(repository_root, "diff-tree", "--no-renames", "--no-commit-id", "--name-only", "-r", legacy_freeze_commit)
    ).splitlines()
    if legacy_paths != [str(legacy_freeze_path)]:
        raise ValueError("Legacy freeze commit shape changed")
    second_freeze_paths = str(
        git(repository_root, "diff-tree", "--no-renames", "--no-commit-id", "--name-only", "-r", second_freeze_commit_expected)
    ).splitlines()
    if second_freeze_paths != [str(second_freeze_path)]:
        raise ValueError("Second freeze commit shape changed")
    failed_repair_paths = _changed_paths(repository_root, legacy_freeze_commit, failed_recovery_source_commit)
    if failed_repair_paths != sorted(failed_repair_paths_expected):
        raise ValueError("Failed first recovery source changed outside its exact repair set")
    second_repair_paths = _changed_paths(repository_root, failed_recovery_source_commit, second_recovery_source_commit)
    if second_repair_paths != sorted(second_repair_paths_expected):
        raise ValueError("Second recovery source changed outside its exact repair set")
    if _added_path_map(repository_root, failed_evaluation_commit) != {
        path: digest_git_blob(repository_root, f"{failed_evaluation_commit}:{path}")
        for path in failed_evaluation_paths
    }:
        raise ValueError("Failed evaluation commit changed outside its exact evidence packet")
    if _added_path_map(repository_root, replacement_selection_commit) != {
        path: digest_git_blob(repository_root, f"{replacement_selection_commit}:{path}")
        for path in replacement_selection_paths
    }:
        raise ValueError("Replacement selection commit changed outside its exact packet")
    repair_paths = _changed_paths(repository_root, replacement_selection_commit, source_commit)
    if repair_paths != sorted(repair_paths_expected):
        raise ValueError("Numeric recovery source changed outside its exact repair set")

    failed_evaluation_hashes = {
        path: digest_git_blob(repository_root, f"{failed_evaluation_commit}:{path}")
        for path in failed_evaluation_paths
    }
    replacement_selection_hashes = {
        path: digest_git_blob(repository_root, f"{replacement_selection_commit}:{path}")
        for path in replacement_selection_paths
    }
    lineage = freeze.get("lineage")
    if not isinstance(lineage, dict) or (
        lineage.get("reason") != recovery_reason
        or lineage.get("legacySourceCommit") != legacy_source_commit
        or lineage.get("legacySourceTree") != legacy_source_tree
        or lineage.get("legacySourceActionsRunId") != legacy_source_actions_run_id
        or lineage.get("legacySourceActionsConclusion") != "success"
        or lineage.get("legacyFreezeCommit") != legacy_freeze_commit
        or lineage.get("legacyFreezeTree") != legacy_freeze_tree
        or lineage.get("legacyFreezeReceiptPath") != str(legacy_freeze_path)
        or lineage.get("legacyFreezeReceiptSha256") != legacy_freeze_sha256
        or lineage.get("legacyFreezeActionsRunId") != legacy_freeze_actions_run_id
        or lineage.get("legacyFreezeActionsConclusion") != "failure"
        or lineage.get("failedRecoverySourceCommit") != failed_recovery_source_commit
        or lineage.get("failedRecoverySourceTree") != failed_recovery_source_tree
        or lineage.get("failedRecoveryActionsRunId") != failed_recovery_actions_run_id
        or lineage.get("failedRecoveryActionsRunUrl") != failed_recovery_actions_run_url
        or lineage.get("failedRecoveryActionsConclusion") != "failure"
        or lineage.get("failedRecoveryReason") != failed_recovery_reason
        or lineage.get("failedRecoveryRepairPaths") != failed_repair_paths_expected
        or lineage.get("secondRecoverySourceCommit") != second_recovery_source_commit
        or lineage.get("secondRecoverySourceTree") != second_recovery_source_tree
        or lineage.get("secondRecoveryActionsRunId") != second_recovery_actions_run_id
        or lineage.get("secondRecoveryActionsConclusion") != "success"
        or lineage.get("secondRecoveryRepairPaths") != second_repair_paths_expected
        or lineage.get("secondFreezeCommit") != second_freeze_commit_expected
        or lineage.get("secondFreezeTree") != second_freeze_tree
        or lineage.get("secondFreezeReceiptPath") != str(second_freeze_path)
        or lineage.get("secondFreezeReceiptSha256") != second_freeze_sha256
        or lineage.get("secondFreezeActionsRunId") != second_freeze_actions_run_id
        or lineage.get("secondFreezeActionsConclusion") != "success"
        or lineage.get("failedEvaluationCommit") != failed_evaluation_commit
        or lineage.get("failedEvaluationTree") != failed_evaluation_tree
        or lineage.get("failedEvaluationActionsRunId") != failed_evaluation_actions_run_id
        or lineage.get("failedEvaluationActionsConclusion") != "failure"
        or lineage.get("failedEvaluationReason") != failed_evaluation_reason
        or lineage.get("failedEvaluationPathsSha256") != failed_evaluation_hashes
        or lineage.get("replacementSelectionCommit") != replacement_selection_commit
        or lineage.get("replacementSelectionTree") != replacement_selection_tree
        or lineage.get("replacementSelectionActionsRunId") != replacement_selection_actions_run_id
        or lineage.get("replacementSelectionActionsConclusion") != "failure"
        or lineage.get("replacementSelectionReason") != replacement_selection_reason
        or lineage.get("replacementSelectionPathsSha256") != replacement_selection_hashes
        or lineage.get("numericRecoverySourceCommit") != source_commit
        or lineage.get("numericRecoverySourceTree") != source_tree
        or lineage.get("numericRecoveryRepairPaths") != repair_paths_expected
        or "cannot prove" not in str(lineage.get("repositoryEvidenceLimitation", ""))
    ):
        raise ValueError("V3 pre-score lineage changed")

    replacement_protocol = freeze.get("replacementProtocol")
    if replacement_protocol != replacement_protocol_expected:
        raise ValueError("Replacement-v2 protocol binding changed")
    for path, expected_hash in replacement_files_sha256.items():
        if digest_git_blob(repository_root, f"{source_commit}:{path}") != expected_hash:
            raise ValueError(f"Replacement protocol file changed: {path}")

    source_ci = freeze.get("sourcePublicCiProof")
    if not isinstance(source_ci, dict) or (
        source_ci.get("workflowName") != QUALITY_WORKFLOW_NAME
        or source_ci.get("workflowPath") != QUALITY_WORKFLOW_PATH
        or source_ci.get("event") != "push"
        or source_ci.get("headSha") != source_commit
        or source_ci.get("status") != "completed"
        or source_ci.get("conclusion") != "success"
        or not isinstance(source_ci.get("runId"), int)
        or not isinstance(source_ci.get("runUrl"), str)
        or source_ci.get("workflowFileSha256")
        != digest_git_blob(repository_root, f"{source_commit}:{QUALITY_WORKFLOW_PATH}")
    ):
        raise ValueError("V3 source public CI proof changed")
    source_ci_revalidation = None
    if require_freeze_ci_success:
        source_ci_revalidation = require_successful_public_quality_run(
            source_commit,
            runs_resolver=actions_runs_resolver,
        )
        source_ci_identity_fields = (
            "workflowName",
            "workflowPath",
            "event",
            "headSha",
            "status",
            "conclusion",
            "runId",
            "runUrl",
        )
        if any(
            source_ci.get(field) != source_ci_revalidation.get(field)
            for field in source_ci_identity_fields
        ):
            raise ValueError("V3 source public CI proof does not match current public Actions evidence")
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
        git(repository_root, "diff-tree", "--no-renames", "--no-commit-id", "--name-status", "-r", freeze_commit)
    ).splitlines()
    if changed != [f"A\t{active_freeze_path}"]:
        raise ValueError("Pre-score freeze commit changed more than the freeze receipt")
    added_freeze_receipts = sorted({
        path
        for path in str(
            git(repository_root, "log", "--no-renames", "--diff-filter=A", "--format=", "--name-only", "HEAD")
        ).splitlines()
        if path.startswith("benchmark/evidence/evaluation/pre-score-freeze")
    })
    if added_freeze_receipts != sorted([str(legacy_freeze_path), str(second_freeze_path), str(active_freeze_path)]):
        raise ValueError("An alternate pre-score freeze receipt exists in public history")
    for revision in (source_commit, freeze_commit):
        frozen_tree_paths = set(
            str(git(repository_root, "ls-tree", "-r", "--name-only", revision)).splitlines()
        )
        if any(any(path.startswith(prefix) for prefix in POST_SCORE_PREFIXES) for path in frozen_tree_paths) or any(
            path in POST_SCORE_FILES for path in frozen_tree_paths
        ):
            raise ValueError("Pre-score source or freeze already contains replacement evaluation output")
    status_rows = str(
        git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    ).splitlines()
    dirty_paths = [row[3:].split(" -> ")[-1] for row in status_rows if len(row) > 3]
    allowed_patterns = freeze.get("allowedPostScorePaths")
    if allowed_patterns != allowed_paths_expected:
        raise ValueError("Pre-score freeze post-score path policy changed")
    if any(not allowed_post_score_path(path, allowed_paths_expected) for path in dirty_paths):
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
    freeze_ci_proof = None
    if require_freeze_ci_success:
        freeze_ci_proof = require_successful_public_quality_run(
            freeze_commit,
            runs_resolver=actions_runs_resolver,
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
    changed_paths = str(
        git(repository_root, "diff", "--no-renames", "--name-only", f"{source_commit}..{local_head}")
    ).splitlines()
    if any(
        path != str(active_freeze_path) and not allowed_post_score_path(path, allowed_paths_expected)
        for path in changed_paths
    ):
        raise ValueError("Public replay commit changed a frozen source path")
    return {
        **freeze,
        "freezeCommit": freeze_commit,
        "anonymousInferenceProof": public_proof,
        "sourcePublicCiRevalidation": source_ci_revalidation,
        "freezePublicCiProof": freeze_ci_proof,
    }
