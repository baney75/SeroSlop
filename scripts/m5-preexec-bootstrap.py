#!/usr/bin/env python3
"""Stdlib-only gate that verifies the exact Git worktree before M5 source runs."""

from __future__ import annotations

from hashlib import sha1, sha256
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
from typing import Sequence


GIT = "/usr/bin/git"
GIT_ARGUMENTS = (
    "-c", "core.fsmonitor=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "core.pager=cat",
    "-c", "core.attributesFile=/dev/null",
)
RUNPOD_PATH = "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
LOCAL_PATH = "/usr/bin:/bin"
ALLOWED_IGNORED_PREFIXES = (
    ".verify-venv/",
    "benchmark/.venv/",
    "benchmark/candidates/",
    "benchmark/data/",
    "dist/",
    "node_modules/",
    "release/",
)
LARGE_EVALUATION_PATH = "benchmark/evidence/m5/large-synthetic-evaluation.json"
LOCAL_NODE = Path("/Users/baney/.local/node/bin/node")
LOCAL_NODE_VERSION = "v24.15.0"
LOCAL_NODE_SHA256 = "3200fbd9f7fd4410426dd541e10d1ab829d3472f270d743c7fabd1696c03fe32"


def clean_environment(path: str) -> dict[str, str]:
    return {
        "PATH": path,
        "HOME": "/nonexistent/seroslop-m5-preexec",
        "XDG_CONFIG_HOME": "/nonexistent/seroslop-m5-preexec/xdg",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "GIT_SSH_COMMAND": "/bin/false",
        "GIT_PAGER": "cat",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def parse_runpod_pod_id(payload: bytes) -> str:
    """Extract exactly one bounded ASCII Pod ID without exposing neighbors."""
    if not payload or len(payload) > 1024 * 1024 or not payload.endswith(b"\0"):
        raise ValueError("M5 RunPod PID-1 environment is malformed")
    matches = []
    for row in payload.split(b"\0"):
        if not row:
            continue
        key, separator, value = row.partition(b"=")
        if separator and key == b"RUNPOD_POD_ID":
            try:
                matches.append(value.decode("ascii", errors="strict"))
            except UnicodeDecodeError:
                raise ValueError("M5 RunPod PID-1 Pod ID is malformed") from None
    if len(matches) != 1:
        raise ValueError("M5 RunPod launch requires exactly one PID-1 Pod ID")
    pod_id = matches[0]
    if not 1 <= len(pod_id) <= 128 or any(not (character.isascii() and (character.isalnum() or character in "_-")) for character in pod_id):
        raise ValueError("M5 RunPod PID-1 Pod ID is malformed")
    return pod_id


def runpod_pod_id_from_init() -> str:
    """Read only RUNPOD_POD_ID from a bounded container-init environment."""
    try:
        with Path("/proc/1/environ").open("rb") as handle:
            payload = handle.read(1024 * 1024 + 1)
    except OSError:
        raise ValueError("M5 RunPod PID-1 environment is unavailable") from None
    pod_id = parse_runpod_pod_id(payload)
    caller_id = os.environ.get("RUNPOD_POD_ID")
    if caller_id is not None and caller_id != pod_id:
        raise ValueError("M5 caller and PID-1 RunPod Pod IDs differ")
    return pod_id


def runpod_environment() -> dict[str, str]:
    """Build the fixed RunPod child environment with only the verified Pod ID."""
    environment = clean_environment(RUNPOD_PATH)
    environment["RUNPOD_POD_ID"] = runpod_pod_id_from_init()
    return environment


def git_bytes(arguments: Sequence[str], root: Path) -> bytes:
    return subprocess.check_output(
        [GIT, *GIT_ARGUMENTS, *arguments],
        cwd=root,
        env=clean_environment(LOCAL_PATH),
    )


def records(payload: bytes) -> list[str]:
    return [part.decode("utf-8", errors="strict") for part in payload.split(b"\0") if part]


def file_sha256(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_exact_worktree(*, allowed_untracked: Sequence[str] = ()) -> Path:
    root = Path.cwd()
    if root.resolve(strict=True) != root:
        raise ValueError("M5 pre-exec repository root must not traverse a symlink")
    if git_bytes(["rev-parse", "--show-toplevel"], root).decode().strip() != str(root):
        raise ValueError("M5 pre-exec requires the physical repository root")
    if git_bytes(["rev-parse", "--show-object-format"], root).decode().strip() != "sha1":
        raise ValueError("M5 pre-exec requires the frozen SHA-1 Git object format")

    flags = records(git_bytes(["ls-files", "-v", "-z"], root))
    abnormal = [record for record in flags if not record.startswith("H ")]
    if abnormal:
        raise ValueError(f"M5 pre-exec rejects non-normal index flags: {abnormal}")

    index: dict[str, tuple[str, str]] = {}
    for record in records(git_bytes(["ls-files", "--stage", "-z"], root)):
        metadata, pathname = record.split("\t", maxsplit=1)
        mode, oid, stage_number = metadata.split(" ")
        if mode not in {"100644", "100755"} or stage_number != "0" or pathname in index:
            raise ValueError(f"M5 pre-exec index row changed: {record}")
        index[pathname] = (mode, oid)
    committed: dict[str, tuple[str, str]] = {}
    for record in records(git_bytes(["ls-tree", "-r", "-z", "--full-tree", "HEAD"], root)):
        metadata, pathname = record.split("\t", maxsplit=1)
        mode, object_type, oid = metadata.split(" ")
        if mode not in {"100644", "100755"} or object_type != "blob" or pathname in committed:
            raise ValueError(f"M5 pre-exec committed row changed: {record}")
        committed[pathname] = (mode, oid)
    if index != committed:
        raise ValueError("M5 pre-exec index differs from the committed HEAD tree")

    for pathname, (mode, oid) in index.items():
        path = root / pathname
        for parent in path.parents:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError(f"M5 pre-exec tracked path traverses a symlink: {pathname}")
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink() or root not in path.resolve(strict=True).parents:
            raise ValueError(f"M5 pre-exec tracked path is not an exact regular file: {pathname}")
        executable = bool(metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        if executable != (mode == "100755"):
            raise ValueError(f"M5 pre-exec tracked mode changed: {pathname}")
        payload = path.read_bytes()
        blob_oid = sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
        if blob_oid != oid:
            raise ValueError(f"M5 pre-exec tracked bytes changed: {pathname}")

    allowed = set(allowed_untracked)
    untracked = records(git_bytes(["ls-files", "--others", "--exclude-standard", "-z"], root))
    unexpected = [pathname for pathname in untracked if pathname not in allowed]
    if unexpected or allowed.difference(untracked):
        raise ValueError(f"M5 pre-exec untracked surface changed: {unexpected + sorted(allowed.difference(untracked))}")
    ignored = records(git_bytes(["ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"], root))
    unsafe_ignored = [pathname for pathname in ignored if not any(pathname.startswith(prefix) for prefix in ALLOWED_IGNORED_PREFIXES)]
    if unsafe_ignored:
        raise ValueError(f"M5 pre-exec ignored executable/import surface changed: {unsafe_ignored}")
    return root


def main() -> int:
    if len(sys.argv) < 2:
        raise ValueError("M5 pre-exec mode is required")
    mode, *arguments = sys.argv[1:]
    allowed_untracked = [LARGE_EVALUATION_PATH] if mode == "runpod" and arguments[:1] == ["finalize"] else []
    verify_exact_worktree(allowed_untracked=allowed_untracked)
    if mode == "verify-only":
        if arguments:
            raise ValueError("M5 pre-exec verify-only accepts no arguments")
        print('{"policy":"pass","stage":"pre-exec"}')
        return 0
    if mode == "authorize":
        if arguments or platform.system() != "Darwin" or platform.machine() != "arm64":
            raise ValueError("M5 authorization requires the fixed Darwin arm64 operator host and no arguments")
        if not LOCAL_NODE.is_file() or LOCAL_NODE.is_symlink() or file_sha256(LOCAL_NODE) != LOCAL_NODE_SHA256:
            raise ValueError("M5 authorization requires the exact pinned local Node bytes")
        environment = clean_environment(LOCAL_PATH)
        version = subprocess.check_output([LOCAL_NODE, "--version"], text=True, env=environment).strip()
        if version != LOCAL_NODE_VERSION:
            raise ValueError("M5 authorization local Node version changed")
        os.execve(LOCAL_NODE, [str(LOCAL_NODE), "scripts/m5-run-authorization.mjs"], environment)
    if mode == "runpod-install":
        if arguments or platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("M5 RunPod dependency installation requires fixed Linux x86_64")
        python = "/opt/conda/bin/python"
        os.execve(python, [
            python, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--require-hashes",
            "-r", "benchmark/m5/runpod-requirements.txt",
        ], clean_environment(RUNPOD_PATH))
    if mode == "runpod":
        if not arguments or platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ValueError("M5 RunPod execution requires fixed Linux x86_64 and a launch mode")
        os.execve("/bin/bash", [
            "/bin/bash", "--noprofile", "--norc", "scripts/m5-runpod-launch.sh", *arguments,
        ], runpod_environment())
    raise ValueError(f"Unknown M5 pre-exec mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
