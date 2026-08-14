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
FREEZE_PATH = Path("benchmark/evidence/evaluation/pre-score-freeze.json")
CANONICAL_ORIGIN_URLS = frozenset({
    "https://github.com/baney75/prooflens",
    "https://github.com/baney75/prooflens.git",
    "git@github.com:baney75/prooflens.git",
    "ssh://git@github.com/baney75/prooflens.git",
})
PUBLIC_GIT_URL = "https://github.com/baney75/prooflens.git"
PUBLIC_RAW_BASE_URL = "https://raw.githubusercontent.com/baney75/prooflens"
POST_SCORE_PREFIXES = (
    "benchmark/evidence/evaluation/confirmatory/",
    "benchmark/evidence/evaluation/web-negative/",
)


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
    return any(
        path.startswith(pattern[:-2]) if pattern.endswith("/**") else path == pattern
        for pattern in allowed_patterns
    )


def require_public_pre_score_freeze(
    *,
    repository_root: Path,
    allow_public_descendant: bool = False,
    canonical_origin_urls: frozenset[str] = CANONICAL_ORIGIN_URLS,
    anonymous_head_resolver: Callable[[], str] = anonymous_public_head,
    anonymous_byte_fetcher: Callable[[str], bytes] = anonymous_https_bytes,
) -> dict[str, object]:
    """Require the exact freeze-only public commit before sealed inference."""
    freeze_path = repository_root / FREEZE_PATH
    freeze_bytes = freeze_path.read_bytes()
    freeze = json.loads(freeze_bytes)
    source_commit = str(freeze.get("sourceCommit", ""))
    source_tree = str(freeze.get("sourceTree", ""))
    if source_tree != git(repository_root, "rev-parse", f"{source_commit}^{{tree}}"):
        raise ValueError("Pre-score freeze source commit/tree changed")
    additions = str(
        git(repository_root, "log", "--diff-filter=A", "--format=%H", "--", str(FREEZE_PATH))
    ).splitlines()
    if len(additions) != 1:
        raise ValueError("Pre-score freeze must be added exactly once")
    freeze_commit = additions[0]
    ancestry = str(git(repository_root, "rev-list", "--parents", "-n", "1", freeze_commit)).split()
    if ancestry != [freeze_commit, source_commit]:
        raise ValueError("Pre-score freeze commit must be a freeze-only child of the source commit")
    committed_freeze = git(repository_root, "show", f"{freeze_commit}:{FREEZE_PATH}", text=False)
    if committed_freeze != freeze_bytes:
        raise ValueError("Pre-score freeze receipt changed after its public commit")
    changed = str(
        git(repository_root, "diff-tree", "--no-commit-id", "--name-only", "-r", freeze_commit)
    ).splitlines()
    if changed != [str(FREEZE_PATH)]:
        raise ValueError("Pre-score freeze commit changed more than the freeze receipt")
    frozen_tree_paths = set(
        str(git(repository_root, "ls-tree", "-r", "--name-only", freeze_commit)).splitlines()
    )
    if any(any(path.startswith(prefix) for prefix in POST_SCORE_PREFIXES) for path in frozen_tree_paths):
        raise ValueError("Pre-score freeze commit already contains sealed evaluation output")
    status_rows = str(
        git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    ).splitlines()
    dirty_paths = [row[3:].split(" -> ")[-1] for row in status_rows if len(row) > 3]
    if any(not path.startswith("benchmark/evidence/evaluation/") for path in dirty_paths):
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
        path=FREEZE_PATH,
        expected_bytes=freeze_bytes,
        head_resolver=anonymous_head_resolver,
        byte_fetcher=anonymous_byte_fetcher,
    )
    immutable_hashes = freeze.get("immutableFilesSha256")
    if not isinstance(immutable_hashes, dict) or not immutable_hashes:
        raise ValueError("Pre-score freeze lacks immutable file hashes")
    for path, expected_hash in immutable_hashes.items():
        current = repository_root / str(path)
        if not current.is_file() or sha256(current.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Immutable pre-score file changed: {path}")
    allowed_patterns = freeze.get("allowedPostScorePaths")
    if not isinstance(allowed_patterns, list) or not all(isinstance(row, str) for row in allowed_patterns):
        raise ValueError("Pre-score freeze lacks its post-score path policy")
    changed_paths = str(
        git(repository_root, "diff", "--name-only", f"{source_commit}..{local_head}")
    ).splitlines()
    if any(not allowed_post_score_path(path, allowed_patterns) for path in changed_paths):
        raise ValueError("Public replay commit changed a frozen source path")
    return {**freeze, "freezeCommit": freeze_commit, "anonymousInferenceProof": public_proof}
