#!/usr/bin/env python3
"""Run the frozen M5 train-and-select phase on one RunPod L40S.

This command deliberately has no H3 or terminal-regression arguments. It trains
the six frozen candidates, scores only the fresh M4 selector, and emits a
canonical selection-lock draft. Terminal regressions are a separate post-lock
command.
"""

from __future__ import annotations

import argparse
from base64 import b64encode
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha1, sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import ProxyHandler, Request, build_opener


def _require_runpod_launcher() -> None:
    if __name__ != "__main__":
        return
    expected = Path("/workspace/.seroslop/runtime/node-v24.18.1-linux-x64/bin/node")
    parent = Path(f"/proc/{os.getppid()}/exe")
    try:
        actual = parent.resolve(strict=True)
        digest = sha256()
        with actual.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        launcher = Path(__file__).resolve().parents[2] / "scripts/m5-python-launch.mjs"
        launcher_digest = sha256(launcher.read_bytes()).hexdigest()
        command = Path(f"/proc/{os.getppid()}/cmdline").read_bytes().split(b"\0")
    except OSError as error:
        raise RuntimeError("M5 production Python could not verify its pinned Node parent") from error
    if (
        sys.platform != "linux" or actual != expected or not expected.is_file() or expected.is_symlink()
        or digest.hexdigest() != "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a"
        or launcher_digest != "e8b3b37a79d9a71be1f2e4ff9b584d52da164eed8937a5608872895ab867834a"
        or len(command) < 4 or command[0] != str(expected).encode("utf-8")
        or command[1] != b"scripts/m5-python-launch.mjs" or command[2] not in {b"preflight", b"train"} or command[3] != b"--"
        or os.environ.get("SEROSLOP_M5_LAUNCH_NODE_VERSION") != "v24.18.1"
        or os.environ.get("SEROSLOP_M5_LAUNCH_NODE_SHA256") != "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a"
    ):
        raise RuntimeError("M5 production Python must start through the pinned RunPod launcher")


_require_runpod_launcher()

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark.m5.contracts import (
    VARIANTS,
    branch_candidate_ids,
    canonical_json,
    choose_selector_threshold,
    digest_file,
    load_recipe,
    parse_json_bytes,
    read_jsonl,
    source_balanced_weights,
    validate_environment_receipt,
    validate_manifest_rows,
    validate_parity_recovery_authorization,
    validate_cublas_recovery_authorization,
    validate_provisioning_receipt,
    ort_cuda_providers,
)


ROOT = REPOSITORY_ROOT
GIT = "/usr/bin/git"
GIT_FIXED_ARGUMENTS = ("-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null", "-c", "core.pager=cat")
RECIPE_PATH = ROOT / "benchmark/m5/recipe.json"
MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)
INPUT_SIZE = 384
RESIZE_SHORT_EDGE = 440
PROTECTED_NEW_SOURCES = {
    "british-library-plates",
    "rapidata-dalle-3",
    "rapidata-flux",
    "rapidata-midjourney",
    "rapidata-stable-diffusion",
}
M5_ORIGINAL_PROTOCOL_COMMIT = "89bd1c833abbaa23195d45cd9a82fc3e117bad88"
M5_ORIGINAL_PROTOCOL_TREE = "d16f4d1033416a2935427dc6ced6ceb4ffea4674"
M5_BASE_SOURCE_COMMIT = "5ab375fad2a744620b6ec75f09e6153c8a409049"
M5_BASE_SOURCE_TREE = "fc0afc8a746f3f41c29bbd8713f309856d2bdc53"
M5_PROTOCOL_RECOVERY_PATHS = frozenset({
    "benchmark/m5/README.md",
    "benchmark/m5/contracts.py",
    "benchmark/m5/evaluate_locked.py",
    "benchmark/m5/evaluate_large_synthetic.py",
    "benchmark/m5/finalize.py",
    "benchmark/m5/large_synthetic.py",
    "benchmark/m5/recipe.json",
    "benchmark/m5/test_contracts.py",
    "benchmark/m5/train_gpu.py",
    "package.json",
    "scripts/check-m5-failure-stage.mjs",
    "scripts/check-m5-final-stage.mjs",
    "scripts/check-m5-large-source-stage.mjs",
    "scripts/check-m5-protocol-stage.mjs",
    "scripts/check-m5-selection-lock.mjs",
    "scripts/m5-stage-policy.mjs",
    "scripts/m5-training-contract.mjs",
    "scripts/test-m5-stage-policy.mjs",
    "scripts/test-m5-training-contract.mjs",
})
M5_P2_COMMIT = "1c4ac973785f937fa9023018863941e6d89d8693"
M5_P2_TREE = "a56caae4291e275029076417fb2111be76b07a41"
M5_FAILED_SOURCE_COMMIT = "fba4b51ef5073e0a189ab6baaaf155fccf785dc6"
M5_FAILED_SOURCE_TREE = "9176b515dfe87a5d5136f0103ef2f8b81fab2938"
M5_CI_RECOVERY_COMMIT = "8e06b822f9d9fb4a3dd8adb2e4bf6dc2f19c51bb"
M5_CI_RECOVERY_TREE = "798e7c2ec55abd82de5bf0927f4c988f2e1c9a7c"
M5_P4_COMMIT = "99505818a95e494b3ad8ed10fbb22bff0ce798da"
M5_P4_TREE = "0ff0eff3b518397dc2c721099ed4597fc15eb17f"
M5_P4_AUTHORIZATION_SHA256 = "d1fcdc2fab96873d3860abfeb71d1edd74b14bfb080b1feadd837ce8d4e011d3"
M5_RUNTIME_RECOVERY_COMMIT = "f171aa983a1bd43737f5b931564e3c5b7e5798c6"
M5_RUNTIME_RECOVERY_TREE = "2983b3ab90d0a1097b457bb2d217a810f1067fd6"
M5_RUNTIME_AUTHORIZATION_COMMIT = "e75b16f94718c24ac98311bd6859f10f27b5f2bd"
M5_RUNTIME_AUTHORIZATION_TREE = "143e5e6c0ad694406d7d39da22cc3d4d56e743bd"
M5_RUNTIME_AUTHORIZATION_SHA256 = "eeee532d699705faef1e35d49748c8264387880e0e573d2b8c61e412944c9ce9"
M5_RUNPOD_ENV_RECOVERY_COMMIT = "49355e62437a8c4af9b5da6b4707577e4b0dad6f"
M5_RUNPOD_ENV_RECOVERY_TREE = "5e61796753780033dc1645fd850ce7539bfc9ec3"
M5_RUNPOD_ENV_AUTHORIZATION_COMMIT = "b8cfc6ba9634a2af12c4fec5f49f32b6024c4903"
M5_RUNPOD_ENV_AUTHORIZATION_TREE = "8431b086cb99c0f1281e748a169cdb25c561366b"
M5_RUNPOD_ENV_AUTHORIZATION_SHA256 = "031f8b85dca9362d7afe06bcd30dc400f9f76a82a314012259b4300b237a8662"
M5_RUN_AUTHORIZATION_PATH = "benchmark/evidence/m5/run-authorization.json"
M5_RUNTIME_AUTHORIZATION_PATH = "benchmark/evidence/m5/runtime-recovery-authorization.json"
M5_RUNPOD_ENV_AUTHORIZATION_PATH = "benchmark/evidence/m5/runpod-environment-authorization.json"
M5_NUMERIC_AUDIT_AUTHORIZATION_PATH = "benchmark/evidence/m5/numeric-audit-authorization.json"
M5_A4_COMMIT = "f3d86077cf5e7a124d09b593d69e9a1769d7e295"
M5_A4_TREE = "d93819ff013943ac48c6dafc659effc9cfbf3e95"
M5_A4_AUTHORIZATION_SHA256 = "8286dc24babe83a16fdf898fa5e70b6202a1da8c46ae2aeda8cf557134db0f03"
M5_A5_AUTHORIZATION_PATH = "benchmark/evidence/m5/parity-recovery-authorization.json"
M5_A5_STATUS = "m5-parity-recovery-authorized"
M5_A5_COMMIT = "adc2b06aef8b427c9efba918bb53eaba25c46b77"
M5_A5_TREE = "951fd5145156ab3fe5df3f4e4db0f09a3b06888d"
M5_A5_AUTHORIZATION_SHA256 = "6aa5e08d4b44b01e232c39084a8286704dc3c7d9491f9b02ca8b7b3f63dcaa4d"
M5_A6_AUTHORIZATION_PATH = "benchmark/evidence/m5/cublas-recovery-authorization.json"
M5_A6_STATUS = "m5-cublas-recovery-authorized"
M5_R6_ROWS = {
    "benchmark/m5/README.md": "M", "benchmark/m5/contracts.py": "M", "benchmark/m5/recipe.json": "M",
    "benchmark/m5/test_contracts.py": "M", "benchmark/m5/train_gpu.py": "M",
    "scripts/check-m5-cublas-authorized-chain.mjs": "A", "scripts/check-m5-authorized-chain.mjs": "M",
    "scripts/check-m5-run-authorization-stage.mjs": "M", "scripts/check-m5-source-recovery-stage.mjs": "M",
    "scripts/m5-preexec-bootstrap.py": "M", "scripts/m5-run-authorization.mjs": "M",
    "scripts/m5-runpod-launch.sh": "M", "scripts/m5-stage-policy.mjs": "M",
    "scripts/m5-training-contract.mjs": "M", "scripts/run-static-verification.mjs": "M",
    "scripts/test-m5-stage-policy.mjs": "M", "scripts/test-m5-training-contract.mjs": "M",
}
M5_R5_ROWS = {
    "benchmark/m5/README.md":"M", "benchmark/m5/contracts.py":"M", "benchmark/m5/evaluate_large_synthetic.py":"M", "benchmark/m5/evaluate_locked.py":"M", "benchmark/m5/recipe.json":"M", "benchmark/m5/test_contracts.py":"M", "benchmark/m5/train_gpu.py":"M", "benchmark/evidence/m5/initial-parity-diagnostic.json":"A", "scripts/check-m5-authorized-chain.mjs":"M", "scripts/check-m5-run-authorization-stage.mjs":"M", "scripts/check-m5-source-recovery-stage.mjs":"M", "scripts/m5-run-authorization.mjs":"M", "scripts/m5-stage-policy.mjs":"M", "scripts/m5-training-contract.mjs":"M", "scripts/run-static-verification.mjs":"M", "scripts/test-m5-stage-policy.mjs":"M", "scripts/test-m5-training-contract.mjs":"M",
}
M5_SOURCE_RECOVERY_ROWS = {
    "benchmark/m5/README.md": "M",
    "benchmark/m5/contracts.py": "M",
    "benchmark/m5/evaluate_locked.py": "M",
    "benchmark/m5/evaluate_large_synthetic.py": "M",
    "benchmark/m5/finalize.py": "M",
    "benchmark/m5/large_synthetic.py": "M",
    "benchmark/m5/test_contracts.py": "M",
    "benchmark/m5/train_gpu.py": "M",
    "package.json": "M",
    "scripts/check-m5-failure-stage.mjs": "M",
    "scripts/check-m5-final-stage.mjs": "M",
    "scripts/check-m5-large-source-stage.mjs": "M",
    "scripts/check-m5-protocol-stage.mjs": "M",
    "scripts/check-m5-selection-lock.mjs": "M",
    "scripts/m5-stage-policy.mjs": "M",
    "scripts/test-m5-stage-policy.mjs": "M",
    "scripts/m5-run-authorization.mjs": "A",
    "scripts/m5-preexec-bootstrap.py": "A",
    "scripts/m5-python-launch.mjs": "A",
    "scripts/m5-runpod-launch.sh": "A",
    "scripts/m5-safe-git.mjs": "A",
    "scripts/m5_node_bootstrap.py": "A",
    "scripts/check-m5-run-authorization-stage.mjs": "A",
    "scripts/check-m5-authorized-chain.mjs": "A",
    "scripts/check-m5-source-recovery-stage.mjs": "A",
    "scripts/run-static-verification.mjs": "M",
}
M5_SOURCE_RECOVERY_PATHS = frozenset(M5_SOURCE_RECOVERY_ROWS)
M5_SOURCE_CI_RECOVERY_ROWS = {
    "benchmark/m5/README.md": "M",
    "benchmark/m5/test_contracts.py": "M",
    "benchmark/m5/train_gpu.py": "M",
    "scripts/check-m5-authorized-chain.mjs": "M",
    "scripts/check-m5-source-recovery-stage.mjs": "M",
    "scripts/m5-run-authorization.mjs": "M",
    "scripts/m5-stage-policy.mjs": "M",
    "scripts/run-static-verification.mjs": "M",
    "scripts/test-m5-stage-policy.mjs": "M",
}
M5_RUNTIME_RECOVERY_ROWS = {
    "benchmark/m5/README.md": "M",
    "benchmark/m5/contracts.py": "M",
    "benchmark/m5/test_contracts.py": "M",
    "benchmark/m5/train_gpu.py": "M",
    "scripts/check-m5-authorized-chain.mjs": "M",
    "scripts/check-m5-run-authorization-stage.mjs": "M",
    "scripts/check-m5-source-recovery-stage.mjs": "M",
    "scripts/m5-run-authorization.mjs": "M",
    "scripts/m5-stage-policy.mjs": "M",
    "scripts/run-static-verification.mjs": "M",
    "scripts/test-m5-stage-policy.mjs": "M",
}
M5_RUNPOD_ENV_RECOVERY_ROWS = {
    "benchmark/m5/README.md": "M",
    "benchmark/m5/contracts.py": "M",
    "benchmark/m5/test_contracts.py": "M",
    "benchmark/m5/train_gpu.py": "M",
    "scripts/check-m5-authorized-chain.mjs": "M",
    "scripts/check-m5-run-authorization-stage.mjs": "M",
    "scripts/check-m5-source-recovery-stage.mjs": "M",
    "scripts/m5-preexec-bootstrap.py": "M",
    "scripts/m5-run-authorization.mjs": "M",
    "scripts/m5-stage-policy.mjs": "M",
    "scripts/run-static-verification.mjs": "M",
    "scripts/test-m5-stage-policy.mjs": "M",
}
M5_NUMERIC_AUDIT_RECOVERY_ROWS = {
    ".github/workflows/quality.yml": "M",
    "benchmark/m5/README.md": "M",
    "benchmark/m5/contracts.py": "M",
    "benchmark/m5/test_contracts.py": "M",
    "benchmark/m5/train_gpu.py": "M",
    "scripts/check-m5-authorized-chain.mjs": "M",
    "scripts/check-m5-run-authorization-stage.mjs": "M",
    "scripts/check-m5-source-recovery-stage.mjs": "M",
    "scripts/m5-run-authorization.mjs": "M",
    "scripts/m5-stage-policy.mjs": "M",
    "scripts/run-static-verification.mjs": "M",
    "scripts/test-m5-stage-policy.mjs": "M",
}


@dataclass(frozen=True)
class Item:
    id: str
    path: Path
    image_sha256: str
    label: int
    source: str
    row_index: int
    weight: float
    anchor: bool


class ModelLogits:
    """Stable ONNX export wrapper."""

    def __init__(self, module: Any) -> None:
        import torch

        class Wrapper(torch.nn.Module):
            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner

            def forward(self, pixel_values: Any) -> Any:
                return self.inner(pixel_values=pixel_values).logits

        self.module = Wrapper(module)


def git_environment() -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment.update({
        "PATH": "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/nonexistent/seroslop-m5-git",
        "XDG_CONFIG_HOME": "/nonexistent/seroslop-m5-git/xdg",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "GIT_SSH_COMMAND": "/bin/false",
        "GIT_PAGER": "cat",
    })
    return environment


def git_bytes(arguments: Sequence[str], *, cwd: Path = ROOT) -> bytes:
    return subprocess.check_output([GIT, *GIT_FIXED_ARGUMENTS, *arguments], cwd=cwd, env=git_environment())


def git_text(arguments: Sequence[str], *, cwd: Path = ROOT) -> str:
    return git_bytes(arguments, cwd=cwd).decode("utf-8", errors="strict").strip()


def _nul_records(payload: bytes) -> list[str]:
    return [record.decode("utf-8", errors="strict") for record in payload.split(b"\0") if record]


def assert_worktree_exact(*, allowed_untracked: Sequence[str] = (), cwd: Path = ROOT) -> None:
    if git_text(["rev-parse", "--show-object-format"], cwd=cwd) != "sha1":
        raise ValueError("M5 exact-worktree verification requires the frozen SHA-1 Git object format")
    abnormal = [record for record in _nul_records(git_bytes(["ls-files", "-v", "-z"], cwd=cwd)) if not record.startswith("H ")]
    if abnormal:
        raise ValueError(f"M5 exact-worktree verification rejects non-normal index flags: {abnormal}")
    index: dict[str, tuple[str, str]] = {}
    for record in _nul_records(git_bytes(["ls-files", "--stage", "-z"], cwd=cwd)):
        metadata, pathname = record.split("\t", maxsplit=1)
        mode, oid, stage = metadata.split(" ")
        if mode not in {"100644", "100755"} or stage != "0" or pathname in index:
            raise ValueError(f"M5 exact-worktree index row changed: {record}")
        index[pathname] = (mode, oid)
    committed: dict[str, tuple[str, str]] = {}
    for record in _nul_records(git_bytes(["ls-tree", "-r", "-z", "--full-tree", "HEAD"], cwd=cwd)):
        metadata, pathname = record.split("\t", maxsplit=1)
        mode, object_type, oid = metadata.split(" ")
        if mode not in {"100644", "100755"} or object_type != "blob" or pathname in committed:
            raise ValueError(f"M5 exact-worktree committed row changed: {record}")
        committed[pathname] = (mode, oid)
    if index != committed:
        raise ValueError("M5 exact-worktree index differs from the committed HEAD tree")
    root = cwd.resolve(strict=True)
    for pathname, (mode, oid) in index.items():
        path = root / pathname
        for parent in path.parents:
            if parent == root:
                break
            if parent.is_symlink():
                raise ValueError(f"M5 tracked file traverses a symlink: {pathname}")
        if path.is_symlink() or not path.is_file() or root not in path.resolve(strict=True).parents:
            raise ValueError(f"M5 tracked file is missing, non-regular, symlinked, or escaped: {pathname}")
        if bool(path.stat().st_mode & 0o111) != (mode == "100755"):
            raise ValueError(f"M5 tracked file mode changed: {pathname}")
        payload = path.read_bytes()
        blob_oid = sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()
        if blob_oid != oid:
            raise ValueError(f"M5 tracked file bytes changed: {pathname}")
    allowed = set(allowed_untracked)
    untracked = _nul_records(git_bytes(["ls-files", "--others", "--exclude-standard", "-z"], cwd=cwd))
    unexpected = [pathname for pathname in untracked if pathname not in allowed]
    observed = set(untracked)
    if unexpected or allowed - observed:
        raise ValueError(f"M5 exact-worktree untracked surface changed: {unexpected + sorted(allowed - observed)}")


def run(command: Sequence[str], *, cwd: Path = ROOT) -> str:
    if command and command[0] in {"git", GIT}:
        return git_text(command[1:], cwd=cwd)
    completed = subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return completed.stdout.strip()


def resolve_authorized_protocol_commit() -> str:
    head = run(["git", "rev-parse", "HEAD"])
    parents = run(["git", "rev-list", "--parents", "-n", "1", head]).split()[1:]
    if parents != [M5_ORIGINAL_PROTOCOL_COMMIT]:
        raise ValueError("M5 training requires the exact append-only protocol-recovery child")
    if run(["git", "rev-parse", f"{M5_ORIGINAL_PROTOCOL_COMMIT}^{{tree}}"]) != M5_ORIGINAL_PROTOCOL_TREE:
        raise ValueError("M5 original protocol tree changed")
    original_parents = run(["git", "rev-list", "--parents", "-n", "1", M5_ORIGINAL_PROTOCOL_COMMIT]).split()[1:]
    if original_parents != [M5_BASE_SOURCE_COMMIT] or run(["git", "rev-parse", f"{M5_BASE_SOURCE_COMMIT}^{{tree}}"]) != M5_BASE_SOURCE_TREE:
        raise ValueError("M5 original protocol ancestry changed")
    rows: dict[str, str] = {}
    for line in run(["git", "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", head]).splitlines():
        status, pathname = line.split("\t", maxsplit=1)
        if pathname in rows:
            raise ValueError("M5 protocol recovery contains a duplicate path")
        rows[pathname] = status
    if set(rows) != M5_PROTOCOL_RECOVERY_PATHS or any(status != "M" for status in rows.values()):
        raise ValueError("M5 protocol recovery changed outside its exact authorized surface")
    assert_worktree_exact()
    return head


def require_public_authorization_commit(authorization_commit: str) -> dict[str, Any]:
    """Require anonymous public main and exact-head green quality for A6."""
    opener = build_opener(ProxyHandler({}))
    reference_request = Request(
        "https://api.github.com/repos/baney75/prooflens/git/ref/heads/main",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "seroslop-m5-runtime"},
    )
    with opener.open(reference_request, timeout=30) as response:
        reference = json.loads(response.read().decode("utf-8", errors="strict"))
    if reference.get("object", {}).get("sha") != authorization_commit:
        raise ValueError("M5 runtime requires the exact public A6 main head")
    request = Request(
        f"https://api.github.com/repos/baney75/prooflens/actions/runs?event=push&head_sha={authorization_commit}&per_page=100",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "seroslop-m5-runtime"},
    )
    with opener.open(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8", errors="strict"))
    run_row = next((row for row in payload.get("workflow_runs", []) if (
        row.get("head_sha") == authorization_commit
        and row.get("event") == "push"
        and row.get("status") == "completed"
        and row.get("conclusion") == "success"
        and row.get("path") == ".github/workflows/quality.yml"
    )), None)
    if run_row is None:
        raise ValueError("M5 runtime requires exact-head successful public A6 quality CI")
    return {
        "conclusion": "success",
        "event": "push",
        "headSha": authorization_commit,
        "runId": int(run_row["id"]),
        "status": "completed",
        "url": str(run_row["html_url"]),
        "workflowPath": ".github/workflows/quality.yml",
    }


def validate_cublas_authorization_commit(authorization: str) -> tuple[str, str, str]:
    if commit_rows(authorization) != {M5_A6_AUTHORIZATION_PATH: "A"}:
        raise ValueError("M5 A6 authorization must be receipt-only")
    authorization_parents = run(["git", "rev-list", "--parents", "-n", "1", authorization]).split()[1:]
    if len(authorization_parents) != 1:
        raise ValueError("M5 A6 authorization must have one R6 parent")
    source = authorization_parents[0]
    if run(["git", "rev-list", "--parents", "-n", "1", source]).split()[1:] != [M5_A5_COMMIT] or commit_rows(source) != M5_R6_ROWS:
        raise ValueError("M5 R6 recovery lineage or surface changed")
    if (
        run(["git", "rev-parse", f"{M5_A5_COMMIT}^{{tree}}"] ) != M5_A5_TREE
        or commit_rows(M5_A5_COMMIT) != {M5_A5_AUTHORIZATION_PATH: "A"}
        or sha256(git_bytes(["show", f"{M5_A5_COMMIT}:{M5_A5_AUTHORIZATION_PATH}"])).hexdigest() != M5_A5_AUTHORIZATION_SHA256
    ):
        raise ValueError("M5 immutable A5 authorization binding changed")
    raw = git_bytes(["show", f"{authorization}:{M5_A6_AUTHORIZATION_PATH}"])
    if (ROOT / M5_A6_AUTHORIZATION_PATH).read_bytes() != raw:
        raise ValueError("M5 A6 authorization bytes differ from the committed receipt")
    receipt = parse_json_bytes(raw, label="M5 A6 cuBLAS authorization")
    if raw != canonical_json(receipt):
        raise ValueError("M5 A6 authorization is not canonical JSON")
    source_tree = run(["git", "rev-parse", f"{source}^{{tree}}"])
    expected_map = [
        {"path": path, "sha256": sha256(git_bytes(["show", f"{source}:{path}"])).hexdigest()}
        for path in sorted(M5_R6_ROWS)
    ]
    validate_cublas_recovery_authorization(
        receipt,
        protocol_commit=M5_P2_COMMIT,
        protocol_tree=M5_P2_TREE,
        prior_authorization_commit=M5_A5_COMMIT,
        prior_authorization_tree=M5_A5_TREE,
        prior_authorization_sha256=M5_A5_AUTHORIZATION_SHA256,
        source_commit=source,
        source_tree=source_tree,
        source_path_map=expected_map,
    )
    return source, source_tree, sha256(raw).hexdigest()


def resolve_authorized_run() -> tuple[str, str, str, str, dict[str, Any]]:
    """Require the clean public A6 deterministic-CUDA authorization."""
    head = run(["git", "rev-parse", "HEAD"])
    if not (ROOT / M5_A6_AUTHORIZATION_PATH).is_file():
        raise ValueError("M5 runtime requires the exact A6 authorization receipt")
    source, source_tree, receipt_sha256 = validate_cublas_authorization_commit(head)
    assert_worktree_exact()
    return source, source_tree, head, receipt_sha256, require_public_authorization_commit(head)


def commit_rows(commit: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in run(["git", "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit]).splitlines():
        status, pathname = line.split("\t", maxsplit=1)
        if pathname in rows:
            raise ValueError(f"M5 commit contains a duplicate path: {commit}")
        rows[pathname] = status
    return rows


def validate_source_recovery_history(source: str) -> None:
    # R5 is the exact sole child of public A4; older recovery descendants are
    # cryptographically bound by A4's fixed tree and receipt digest.
    if run(["git", "rev-list", "--parents", "-n", "1", source]).split()[1:] == [M5_A4_COMMIT] and commit_rows(source) == M5_R5_ROWS:
        if (
            run(["git", "rev-parse", f"{M5_A4_COMMIT}^{{tree}}"]) != M5_A4_TREE
            or commit_rows(M5_A4_COMMIT) != {M5_NUMERIC_AUDIT_AUTHORIZATION_PATH: "A"}
            or sha256(git_bytes(["show", f"{M5_A4_COMMIT}:{M5_NUMERIC_AUDIT_AUTHORIZATION_PATH}"])).hexdigest() != M5_A4_AUTHORIZATION_SHA256
        ):
            raise ValueError("M5 A4 tree or receipt changed")
        return
    raise ValueError("M5 runtime requires exact A4 -> R5 recovery history")


def validate_authorization_commit(authorization_commit: str) -> tuple[str, str, str]:
    """Validate A5 authorization and return (recovered source, tree, receipt SHA-256)."""
    authorization_parents = run(["git", "rev-list", "--parents", "-n", "1", authorization_commit]).split()[1:]
    if len(authorization_parents) != 1:
        raise ValueError("M5 authorization must have one source parent")
    source = authorization_parents[0]
    authorization_rows: dict[str, str] = {}
    for line in run(["git", "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", authorization_commit]).splitlines():
        status, pathname = line.split("\t", maxsplit=1)
        if pathname in authorization_rows:
            raise ValueError("M5 A5 authorization contains a duplicate path")
        authorization_rows[pathname] = status
    if authorization_rows != {M5_A5_AUTHORIZATION_PATH: "A"}:
        raise ValueError("M5 A5 authorization must be an exact receipt-only commit")
    if source == M5_A4_COMMIT or run(["git", "rev-parse", f"{M5_A4_COMMIT}^{{tree}}"]) != M5_A4_TREE:
        raise ValueError("M5 A4 parent changed")
    validate_source_recovery_history(source)
    prior_raw = git_bytes(["show", f"{M5_A4_COMMIT}:{M5_NUMERIC_AUDIT_AUTHORIZATION_PATH}"])
    if sha256(prior_raw).hexdigest() != M5_A4_AUTHORIZATION_SHA256:
        raise ValueError("M5 prior numeric authorization bytes changed")
    auth_path = ROOT / M5_A5_AUTHORIZATION_PATH
    raw = auth_path.read_bytes() if auth_path.is_file() and not auth_path.is_symlink() else b""
    if not raw or raw != canonical_json(parse_json_bytes(raw, label="M5 A5 parity authorization")):
        raise ValueError("M5 runtime requires canonical A5 parity authorization")
    receipt = parse_json_bytes(raw, label="M5 A5 parity authorization")
    committed_raw = git_bytes(["show", f"{authorization_commit}:{M5_A5_AUTHORIZATION_PATH}"])
    if committed_raw != raw:
        raise ValueError("M5 A5 authorization bytes changed")
    source_tree = run(["git", "rev-parse", f"{source}^{{tree}}"])
    expected_map = [
        {"path": path, "sha256": sha256(git_bytes(["show", f"{source}:{path}"])).hexdigest()}
        for path in sorted(M5_R5_ROWS)
    ]
    validate_parity_recovery_authorization(
        receipt,
        protocol_commit=M5_P2_COMMIT,
        protocol_tree=M5_P2_TREE,
        prior_authorization_commit=M5_A4_COMMIT,
        prior_authorization_tree=M5_A4_TREE,
        prior_authorization_sha256=M5_A4_AUTHORIZATION_SHA256,
        source_commit=source,
        source_tree=source_tree,
        source_path_map=expected_map,
    )
    return source, source_tree, sha256(raw).hexdigest()


def require_canonical_path(requested: str, expected: str, *, label: str, must_exist: bool = True) -> Path:
    requested_path = Path(requested)
    expected_path = ROOT / expected
    if os.path.abspath(requested_path) != os.path.abspath(expected_path):
        raise ValueError(f"M5 {label} must be {expected}")
    current = expected_path
    while current != ROOT.parent:
        if current.is_symlink():
            raise ValueError(f"M5 {label} traverses a symlink")
        if current == ROOT:
            break
        current = current.parent
    if must_exist and not expected_path.exists():
        raise ValueError(f"M5 {label} is missing")
    return expected_path


def resolve_item_path(data_root: Path, relative: str) -> Path:
    lexical = data_root / relative
    if lexical.is_symlink():
        raise ValueError(f"M5 image path is a symlink: {relative}")
    current = lexical.parent
    while current != data_root.parent:
        if current.is_symlink():
            raise ValueError(f"M5 image parent is a symlink: {relative}")
        if current == data_root:
            break
        current = current.parent
    resolved_root = data_root.resolve(strict=True)
    resolved = lexical.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"M5 image path escapes data root: {relative}")
    return resolved


def load_items(rows: Sequence[dict[str, Any]], data_root: Path, weights: Sequence[float]) -> list[Item]:
    return [
        Item(
            id=str(row["id"]),
            path=resolve_item_path(data_root, str(row["path"])),
            image_sha256=str(row["imageSha256"]),
            label=int(row["label"]),
            source=str(row["source"]),
            row_index=int(row["rowIndex"]),
            weight=float(weight),
            anchor=str(row["source"]) not in PROTECTED_NEW_SOURCES,
        )
        for row, weight in zip(rows, weights, strict=True)
    ]


def verify_item(item: Item) -> None:
    if digest_file(item.path) != item.image_sha256:
        raise ValueError(f"M5 image integrity mismatch: {item.id}")
    with Image.open(item.path) as opened:
        opened.verify()


def verify_all_items(items: Sequence[Item], workers: int) -> None:
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(verify_item, items, chunksize=32))
    print(json.dumps({"event": "pixel-preflight", "items": len(items), "seconds": round(time.monotonic() - started, 3)}), flush=True)


def deterministic_rng(*parts: object) -> random.Random:
    digest = sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def resize_long_edge(image: Image.Image, maximum: int) -> Image.Image:
    scale = maximum / max(image.size)
    if scale >= 1:
        return image
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def transform_view(image: Image.Image, item: Item, variant: str, *, training: bool, branch: str, epoch: int) -> Image.Image:
    rng = deterministic_rng(20260815, branch, epoch, item.id, variant)
    if variant == "original":
        transformed = image
    elif variant == "screenshot":
        frame = Image.new("RGB", (1170, 1400), (238, 241, 244))
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle((46, 46, 1124, 1354), radius=24, fill=(255, 255, 255), outline=(217, 222, 229), width=2)
        draw.ellipse((74, 69, 122, 117), fill=(72, 132, 220))
        draw.rounded_rectangle((138, 84, 318, 99), radius=8, fill=(200, 206, 214))
        left, top, width, height = 47, 140, 1076, 1110
        draw.rectangle((left, top, left + width - 1, top + height - 1), fill=(17, 21, 26))
        scale = min(width / image.width, height / image.height)
        if training:
            scale *= rng.uniform(0.94, 1.0)
        rendered = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        frame.paste(rendered, (left + (width - rendered.width) // 2, top + (height - rendered.height) // 2))
        for x in (78, 124, 170):
            draw.ellipse((x, 1291, x + 22, 1313), outline=(136, 145, 155), width=3)
        transformed = frame
    elif variant == "social-q75":
        edge = rng.randint(800, 1280) if training else 1080
        quality = rng.randint(60, 88) if training else 75
        transformed = resize_long_edge(image, edge)
        if training and rng.random() < 0.5:
            transformed = transformed.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.65)))
        transformed = jpeg_roundtrip(transformed, quality)
    elif variant == "social-heavy":
        first_edge = rng.randint(600, 900) if training else 720
        second_edge = rng.randint(480, 720) if training else 640
        first_quality = rng.randint(38, 62) if training else 50
        second_quality = rng.randint(28, 48) if training else 38
        transformed = jpeg_roundtrip(resize_long_edge(image, first_edge), first_quality)
        transformed = jpeg_roundtrip(resize_long_edge(transformed, second_edge), second_quality)
    else:
        raise ValueError(f"Unknown M5 variant: {variant}")
    if training:
        if rng.random() < 0.5:
            transformed = ImageOps.mirror(transformed)
        transformed = ImageEnhance.Color(transformed).enhance(rng.uniform(0.90, 1.10))
        transformed = ImageEnhance.Contrast(transformed).enhance(rng.uniform(0.92, 1.08))
    return transformed


def preprocess_image(image: Image.Image, item: Item, variant: str, *, training: bool, branch: str, epoch: int) -> np.ndarray:
    transformed = transform_view(image, item, variant, training=training, branch=branch, epoch=epoch)
    scale = RESIZE_SHORT_EDGE / min(transformed.size)
    resized = transformed.resize(
        (max(INPUT_SIZE, round(transformed.width * scale)), max(INPUT_SIZE, round(transformed.height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - INPUT_SIZE) // 2
    top = (resized.height - INPUT_SIZE) // 2
    pixels = np.asarray(resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE)), dtype=np.float32) / 255.0
    return np.transpose((pixels - MEAN) / STD, (2, 0, 1)).astype(np.float32)


class ImageDataset:
    def __init__(self, items: Sequence[Item], *, branch: str, epoch: int, variant: str | None, training: bool) -> None:
        self.items = list(items)
        self.branch = branch
        self.epoch = epoch
        self.variant = variant
        self.training = training

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, float, float, float, int]:
        item = self.items[index]
        variant = self.variant
        if variant is None:
            variant = VARIANTS[int.from_bytes(sha256(f"20260815:{self.branch}:{self.epoch}:{item.id}".encode()).digest()[:2], "big") % len(VARIANTS)]
        with Image.open(item.path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        pixels = preprocess_image(image, item, variant, training=self.training, branch=self.branch, epoch=self.epoch)
        return pixels, float(item.label), item.weight, float(item.anchor), item.row_index


def collate(batch: Sequence[tuple[np.ndarray, float, float, float, int]]) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    pixels, labels, weights, anchors, indexes = zip(*batch, strict=True)
    return (
        torch.from_numpy(np.stack(pixels)),
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
        torch.tensor(anchors, dtype=torch.float32),
        torch.tensor(indexes, dtype=torch.int64),
    )


def load_provisioning_receipt(path: Path, recipe: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("M5 requires the canonical operator-recorded RunPod provisioning receipt")
    raw = path.read_bytes()
    receipt = parse_json_bytes(raw, label="RunPod provisioning receipt")
    if raw != canonical_json(receipt):
        raise ValueError("M5 RunPod provisioning receipt is not canonical JSON")
    validate_provisioning_receipt(receipt, recipe)
    return receipt, sha256(raw).hexdigest()


def environment_receipt(
    provisioning: Mapping[str, Any],
    provisioning_sha256: str,
    recipe: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    import transformers

    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if not pod_id:
        raise ValueError("M5 must run inside an identified RunPod pod")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("M5 requires exactly one CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    driver = run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0].strip()
    return {
        "provider": "RunPod Secure Cloud (operator-recorded control-plane receipt)",
        "gpuProduct": torch.cuda.get_device_name(0),
        "gpuMemoryBytes": int(properties.total_memory),
        "cudaAvailable": True,
        "cudaVersion": str(torch.version.cuda),
        "driverVersion": driver,
        "torchVersion": torch.__version__,
        "transformersVersion": transformers.__version__,
        "pythonVersion": platform.python_version(),
        "launchNodeVersion": os.environ.get("SEROSLOP_M5_LAUNCH_NODE_VERSION", ""),
        "launchNodeSha256": os.environ.get("SEROSLOP_M5_LAUNCH_NODE_SHA256", ""),
        "runpodPodIdSha256": sha256(pod_id.encode()).hexdigest(),
        "provisioningReceiptSha256": provisioning_sha256,
        "containerImage": provisioning["containerImage"],
        "requirementsSha256": digest_file(ROOT / recipe["training"]["requirementsPath"]),
        "providerEvidenceBoundary": "operator-recorded-not-cryptographic-attestation",
        "providerIdentityEvidence": recipe["training"]["providerIdentityEvidence"],
        "providerSignedAttestation": False,
        "runtimeConsistencyEvidence": recipe["training"]["runtimeConsistencyEvidence"],
        "cublasWorkspaceConfig": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }


def validate_environment_matches_provisioning(environment: Mapping[str, Any], provisioning: Mapping[str, Any]) -> None:
    if environment["runpodPodIdSha256"] != provisioning["podIdSha256"]:
        raise ValueError("M5 runtime pod identity does not match the provisioning receipt")
    if environment["gpuProduct"] != provisioning["gpuProduct"]:
        raise ValueError("M5 runtime GPU does not match the provisioning receipt")


def prepare_runtime(recipe: Mapping[str, Any], cache_dir: Path) -> tuple[Any, Any, dict[str, str]]:
    import torch
    import onnx
    from onnx import numpy_helper
    from huggingface_hub import snapshot_download
    from transformers import ViTForImageClassification

    snapshot = Path(snapshot_download(
        repo_id=recipe["upstream"]["repository"],
        revision=recipe["upstream"]["revision"],
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
        cache_dir=cache_dir,
    ))
    source_hashes: dict[str, str] = {}
    for key in ("config", "preprocessor", "pytorchWeights"):
        lock = recipe["upstream"][key]
        path = snapshot / lock["path"]
        if path.stat().st_size != lock["bytes"] or digest_file(path) != lock["sha256"]:
            raise ValueError(f"M5 pinned upstream {key} changed")
        source_hashes[lock["path"]] = lock["sha256"]
    model = ViTForImageClassification.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation=recipe["training"]["attentionImplementation"],
    )
    initial_onnx = ROOT / recipe["initialModel"]["path"]
    if initial_onnx.stat().st_size != recipe["initialModel"]["bytes"] or digest_file(initial_onnx) != recipe["initialModel"]["sha256"]:
        raise ValueError("M5 initial ONNX changed")
    graph = onnx.load(initial_onnx, load_external_data=False)
    initializers = {value.name: value for value in graph.graph.initializer}
    weight = numpy_helper.to_array(initializers["classifier.weight"]).astype(np.float32)
    bias = numpy_helper.to_array(initializers["classifier.bias"]).astype(np.float32)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.from_numpy(weight))
        model.classifier.bias.copy_(torch.from_numpy(bias))
    teacher = __import__("copy").deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return model, teacher, source_hashes


def verify_initial_model_parity(model: Any, items: Sequence[Item], recipe: Mapping[str, Any]) -> float:
    import onnxruntime as ort
    import torch

    arrays: list[np.ndarray] = []
    for item in items[:16]:
        with Image.open(item.path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        arrays.append(preprocess_image(image, item, "original", training=False, branch="parity", epoch=0))
    pixels = np.stack(arrays).astype(np.float32)
    with torch.inference_mode():
        reference = model(pixel_values=torch.from_numpy(pixels)).logits.detach().cpu().numpy()
    providers = ort_cuda_providers(ort)
    provider = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    session = ort.InferenceSession(str(ROOT / recipe["initialModel"]["path"]), providers=providers)
    actual = session.run(["logits"], {"pixel_values": pixels})[0]
    maximum = float(np.max(np.abs(reference - actual)))
    if maximum > recipe["initialModel"]["maximumPytorchOnnxParityError"]:
        raise ValueError(f"M5 initial PyTorch/packaged ONNX parity failed: {maximum}")
    return maximum


def configure_branch(model: Any, branch: Mapping[str, Any]) -> list[dict[str, Any]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = set(int(index) for index in branch["trainableEncoderBlocks"])
    parameter_groups: list[dict[str, Any]] = []
    for index, layer in enumerate(model.vit.encoder.layer):
        if index not in trainable:
            continue
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
        depth = 11 - index
        learning_rate = float(branch["backboneLearningRate"]) * float(branch["layerwiseLearningRateDecay"]) ** depth
        parameter_groups.append({"params": list(layer.parameters()), "lr": learning_rate, "name": f"encoder-{index}"})
    for parameter in model.vit.layernorm.parameters():
        parameter.requires_grad_(True)
    parameter_groups.append({"params": list(model.vit.layernorm.parameters()), "lr": float(branch["backboneLearningRate"]), "name": "layernorm"})
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)
    parameter_groups.append({"params": list(model.classifier.parameters()), "lr": float(branch["classifierLearningRate"]), "name": "classifier"})
    if trainable == set(range(12)):
        for parameter in model.vit.embeddings.parameters():
            parameter.requires_grad_(True)
        embedding_lr = float(branch["backboneLearningRate"]) * float(branch["layerwiseLearningRateDecay"]) ** 12
        parameter_groups.append({"params": list(model.vit.embeddings.parameters()), "lr": embedding_lr, "name": "embeddings"})
    if not parameter_groups or not any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("M5 branch has no trainable parameters")
    return parameter_groups


def cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _checkpoint_epoch(path: Path) -> int:
    try:
        return int(path.stem.rpartition("-epoch-")[2])
    except ValueError as error:
        raise ValueError(f"M5 checkpoint filename is invalid: {path.name}") from error


def _validate_checkpoint_payload(payload: Mapping[str, Any], branch_name: str, epoch: int, *, resume: bool) -> None:
    required = {"model", "branch", "epoch"}
    if resume:
        required |= {"optimizer", "scheduler", "globalStep", "epochReceipts"}
    if set(payload) != required or payload["branch"] != branch_name or payload["epoch"] != epoch:
        raise ValueError("M5 checkpoint payload identity changed")
    if resume:
        receipts = payload["epochReceipts"]
        if not isinstance(receipts, list) or len(receipts) != epoch or [row.get("epoch") for row in receipts] != list(range(1, epoch + 1)):
            raise ValueError("M5 resume receipt history is not contiguous")


def _write_epoch_seal(
    output_dir: Path,
    branch_name: str,
    epoch: int,
    resume_path: Path,
    candidate_path: Path | None,
    payload: Mapping[str, Any],
) -> Path:
    seal_path = output_dir / "seals" / f"{branch_name}-epoch-{epoch}.json"
    seal = {
        "schemaVersion": 1,
        "branch": branch_name,
        "epoch": epoch,
        "globalStep": int(payload["globalStep"]),
        "epochReceiptsSha256": sha256(canonical_json(payload["epochReceipts"])).hexdigest(),
        "resume": {"path": resume_path.relative_to(ROOT).as_posix(), "sha256": digest_file(resume_path)},
        "candidate": None if candidate_path is None else {
            "path": candidate_path.relative_to(ROOT).as_posix(), "sha256": digest_file(candidate_path),
        },
    }
    write_json(seal_path, seal)
    return seal_path


def load_or_recover_branch_history(
    output_dir: Path,
    branch: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, list[dict[str, Any]]]:
    import torch

    branch_name = str(branch["name"])
    resume_dir = output_dir / "resume"
    candidate_dir = output_dir / "checkpoints"
    seal_dir = output_dir / "seals"
    for directory in (resume_dir, candidate_dir, seal_dir):
        directory.mkdir(parents=True, exist_ok=True)
    partials = [path for directory in (resume_dir, candidate_dir, seal_dir) for path in directory.glob(f"{branch_name}-epoch-*.partial")]
    if partials:
        raise ValueError("M5 checkpoint history contains a partial file")
    resumes = {_checkpoint_epoch(path): path for path in resume_dir.glob(f"{branch_name}-epoch-*.pt")}
    candidates = {_checkpoint_epoch(path): path for path in candidate_dir.glob(f"{branch_name}-epoch-*.pt")}
    seals = {_checkpoint_epoch(path): path for path in seal_dir.glob(f"{branch_name}-epoch-*.json")}
    if len(resumes) != len(list(resume_dir.glob(f"{branch_name}-epoch-*.pt"))) or len(candidates) != len(list(candidate_dir.glob(f"{branch_name}-epoch-*.pt"))):
        raise ValueError("M5 checkpoint history contains duplicate epochs")
    sealed_epochs = sorted(seals)
    if sealed_epochs != list(range(1, len(sealed_epochs) + 1)):
        raise ValueError("M5 checkpoint seals are not contiguous")
    candidate_epochs = set(int(value) for value in branch["candidateEpochs"])
    latest_payload: Mapping[str, Any] | None = None
    epoch_receipts: list[dict[str, Any]] = []
    for epoch in sealed_epochs:
        if epoch not in resumes:
            raise ValueError("M5 sealed resume checkpoint is missing")
        seal = json.loads(seals[epoch].read_text(encoding="utf-8"))
        expected_candidate = candidates.get(epoch) if epoch in candidate_epochs else None
        if set(seal) != {"schemaVersion", "branch", "epoch", "globalStep", "epochReceiptsSha256", "resume", "candidate"}:
            raise ValueError("M5 checkpoint seal schema changed")
        if seal["schemaVersion"] != 1 or seal["branch"] != branch_name or seal["epoch"] != epoch:
            raise ValueError("M5 checkpoint seal identity changed")
        if seal["resume"] != {"path": resumes[epoch].relative_to(ROOT).as_posix(), "sha256": digest_file(resumes[epoch])}:
            raise ValueError("M5 sealed resume digest changed")
        expected_candidate_receipt = None if expected_candidate is None else {
            "path": expected_candidate.relative_to(ROOT).as_posix(), "sha256": digest_file(expected_candidate),
        }
        if seal["candidate"] != expected_candidate_receipt:
            raise ValueError("M5 sealed candidate digest changed")
        payload = torch.load(resumes[epoch], map_location="cpu", weights_only=True)
        _validate_checkpoint_payload(payload, branch_name, epoch, resume=True)
        if seal["epochReceiptsSha256"] != sha256(canonical_json(payload["epochReceipts"])).hexdigest():
            raise ValueError("M5 sealed epoch receipt digest changed")
        latest_payload = payload
        epoch_receipts = list(payload["epochReceipts"])
    next_epoch = len(sealed_epochs) + 1
    unsealed_resumes = sorted(set(resumes) - set(sealed_epochs))
    unsealed_candidates = sorted(set(candidates) - set(sealed_epochs))
    if unsealed_resumes:
        if unsealed_resumes != [next_epoch] or any(epoch != next_epoch for epoch in unsealed_candidates):
            raise ValueError("M5 checkpoint history contains a gap or extra unsealed epoch")
        resume_path = resumes[next_epoch]
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        _validate_checkpoint_payload(payload, branch_name, next_epoch, resume=True)
        candidate_path: Path | None = None
        if next_epoch in candidate_epochs:
            candidate_path = candidate_dir / f"{branch_name}-epoch-{next_epoch}.pt"
            if candidate_path.exists():
                candidate_payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
                _validate_checkpoint_payload(candidate_payload, branch_name, next_epoch, resume=False)
            else:
                atomic_torch_save(candidate_path, {"model": payload["model"], "branch": branch_name, "epoch": next_epoch})
            candidates[next_epoch] = candidate_path
        elif unsealed_candidates:
            raise ValueError("M5 checkpoint history has an unexpected candidate")
        _write_epoch_seal(output_dir, branch_name, next_epoch, resume_path, candidate_path, payload)
        latest_payload = payload
        epoch_receipts = list(payload["epochReceipts"])
        sealed_epochs.append(next_epoch)
    elif unsealed_candidates:
        raise ValueError("M5 candidate checkpoint has no matching resume state")
    if set(resumes) != set(sealed_epochs) or set(candidates) != candidate_epochs.intersection(sealed_epochs):
        raise ValueError("M5 checkpoint inventory does not match its seals")
    return latest_payload, epoch_receipts


def train_branch(
    base_model: Any,
    teacher: Any,
    branch: Mapping[str, Any],
    items: Sequence[Item],
    recipe: Mapping[str, Any],
    output_dir: Path,
    budget: PaidTimeBudget,
) -> tuple[list[Path], list[dict[str, Any]]]:
    import copy
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    branch_name = str(branch["name"])
    torch.manual_seed(int(recipe["seed"]))
    torch.cuda.manual_seed_all(int(recipe["seed"]))
    model = copy.deepcopy(base_model).to("cuda", dtype=torch.float32)
    teacher = teacher.to("cuda", dtype=torch.float32)
    parameter_groups = configure_branch(model, branch)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(branch["weightDecay"]),
    )
    batch_size = int(recipe["training"]["perGpuBatchSize"])
    accumulation = int(recipe["training"]["gradientAccumulationSteps"])
    batches = math.ceil(len(items) / batch_size)
    steps_per_epoch = math.ceil(batches / accumulation)
    total_steps = steps_per_epoch * int(recipe["training"]["epochs"])
    warmup_steps = round(total_steps * float(recipe["training"]["warmupRatio"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_schedule(step, total_steps, warmup_steps))
    checkpoints: list[Path] = []
    epoch_receipts: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 1
    resume_dir = output_dir / "resume"
    latest_payload, epoch_receipts = load_or_recover_branch_history(output_dir, branch)
    if latest_payload is not None:
        payload = latest_payload
        resumed_epoch = int(payload["epoch"])
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        global_step = int(payload["globalStep"])
        start_epoch = resumed_epoch + 1
        print(json.dumps({"event": "resume", "branch": branch_name, "epoch": resumed_epoch}), flush=True)
    for epoch in range(start_epoch, int(recipe["training"]["epochs"]) + 1):
        dataset = ImageDataset(items, branch=branch_name, epoch=epoch, variant=None, training=True)
        generator = torch.Generator().manual_seed(int(recipe["seed"]) + epoch + (0 if branch_name == "last4" else 1000))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=int(recipe["training"]["workers"]),
            pin_memory=True,
            persistent_workers=False,
            collate_fn=collate,
            drop_last=False,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_bce = 0.0
        total_anchor = 0.0
        started = time.monotonic()
        for batch_index, (pixels, labels, weights, anchors, _indexes) in enumerate(loader, start=1):
            budget.check(f"train-{branch_name}-epoch-{epoch}-batch-{batch_index}")
            pixels = pixels.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            weights = weights.to("cuda", non_blocking=True)
            anchors = anchors.to("cuda", non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=pixels).logits.reshape(-1).float()
                with torch.no_grad():
                    teacher_logits = teacher(pixel_values=pixels).logits.reshape(-1).float()
                bce_values = functional.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights
                anchor_values = ((logits - teacher_logits) ** 2) * anchors
                denominator = accumulation_window_samples(len(items), batch_size, accumulation, batch_index)
                loss = (bce_values.sum() + float(branch["teacherAnchorCoefficient"]) * anchor_values.sum()) / denominator
                bce = bce_values.mean()
                anchor_loss = anchor_values.mean()
            loss.backward()
            total_bce += float(bce.detach().cpu()) * len(labels)
            total_anchor += float(anchor_loss.detach().cpu()) * len(labels)
            should_step = batch_index % accumulation == 0 or batch_index == len(loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=float(recipe["training"]["gradientClipNorm"]),
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
        elapsed = time.monotonic() - started
        receipt = {
            "branch": branch_name,
            "epoch": epoch,
            "globalStep": global_step,
            "seconds": round(elapsed, 3),
            "images": len(items),
            "meanWeightedBce": total_bce / len(items),
            "meanMaskedTeacherMse": total_anchor / len(items),
            "learningRates": {group["name"]: group["lr"] for group in optimizer.param_groups},
        }
        epoch_receipts.append(receipt)
        print(json.dumps({"event": "epoch", **receipt}, sort_keys=True), flush=True)
        resume_path = resume_dir / f"{branch_name}-epoch-{epoch}.pt"
        state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
        resume_payload = {
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "branch": branch_name,
            "epoch": epoch,
            "globalStep": global_step,
            "epochReceipts": epoch_receipts,
        }
        atomic_torch_save(resume_path, resume_payload)
        candidate_path: Path | None = None
        if epoch in branch["candidateEpochs"]:
            candidate_path = output_dir / "checkpoints" / f"{branch_name}-epoch-{epoch}.pt"
            atomic_torch_save(candidate_path, {"model": state, "branch": branch_name, "epoch": epoch})
        _write_epoch_seal(output_dir, branch_name, epoch, resume_path, candidate_path, resume_payload)
        budget.check(f"train-{branch_name}-epoch-{epoch}-sealed", force_persist=True)
    for epoch in branch["candidateEpochs"]:
        checkpoint = output_dir / "checkpoints" / f"{branch_name}-epoch-{epoch}.pt"
        if not checkpoint.is_file():
            raise ValueError(f"M5 candidate checkpoint is missing after training: {checkpoint.name}")
        checkpoints.append(checkpoint)
    teacher.to("cpu")
    model.to("cpu")
    torch.cuda.empty_cache()
    return checkpoints, epoch_receipts


def load_checkpoint_model(base_model: Any, checkpoint: Path) -> Any:
    import copy
    import torch

    model = copy.deepcopy(base_model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"], strict=True)
    return model


def export_onnx(
    model: Any,
    destination: Path,
    recipe: Mapping[str, Any],
    *,
    parity_pixels: np.ndarray | None = None,
    providers: list[Any] | None = None,
) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ModelLogits(model.eval()).module
    dummy = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        destination,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    graph = onnx.load(destination, load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    if destination.stat().st_size > recipe["deliverable"]["maximumBytes"]:
        raise ValueError("M5 candidate exceeds the browser model-size gate")
    providers = providers if providers is not None else ort_cuda_providers(ort)
    if not providers:
        raise ValueError("M5 ONNX export parity requires an execution provider")
    provider = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    session = ort.InferenceSession(str(destination), providers=providers)
    with torch.no_grad():
        reference = wrapper(dummy).detach().cpu().numpy()
    actual = session.run(["logits"], {"pixel_values": dummy.numpy()})[0]
    parity = float(np.max(np.abs(reference - actual)))
    if parity_pixels is not None:
        pixels = np.asarray(parity_pixels, dtype=np.float32)
        if pixels.ndim != 4 or pixels.shape[1:] != (3, INPUT_SIZE, INPUT_SIZE) or not np.any(pixels != 0):
            raise ValueError("M5 export parity probe must be a nonzero NCHW image batch")
        with torch.no_grad():
            reference = wrapper(torch.from_numpy(pixels)).detach().cpu().numpy()
        actual = session.run(["logits"], {"pixel_values": pixels})[0]
        parity = max(parity, float(np.max(np.abs(reference - actual))))
    if parity > recipe["initialModel"]["maximumPytorchOnnxParityError"]:
        raise ValueError(f"M5 ONNX parity failed: {parity}")
    return {
        "path": destination.relative_to(ROOT).as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": digest_file(destination),
        "parityMaximumAbsoluteError": parity,
        "parityProvider": provider,
        "parityProviderOptions": {"use_tf32": "0"} if provider == "CUDAExecutionProvider" else {},
    }


def predict_onnx_variant(
    session: Any,
    items: Sequence[Item],
    variant: str,
    recipe: Mapping[str, Any],
    budget: PaidTimeBudget,
) -> list[float]:
    from torch.utils.data import DataLoader

    dataset = ImageDataset(items, branch="selector", epoch=0, variant=variant, training=False)
    loader = DataLoader(
        dataset,
        batch_size=int(recipe["training"]["perGpuBatchSize"]),
        shuffle=False,
        num_workers=int(recipe["training"]["workers"]),
        pin_memory=True,
        collate_fn=collate,
    )
    values: list[float] = []
    for batch_index, (pixels, _labels, _weights, _anchors, _indexes) in enumerate(loader, start=1):
        budget.check(f"selector-onnx-{variant}-batch-{batch_index}")
        logits = session.run(["logits"], {"pixel_values": pixels.numpy()})[0]
        values.extend(float(value) for value in np.asarray(logits, dtype=np.float32).reshape(-1))
    return values


def pack_float32(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype="<f4")
    payload = array.tobytes(order="C")
    return {
        "dtype": "float32-little-endian",
        "count": len(values),
        "sha256": sha256(payload).hexdigest(),
        "base64": b64encode(payload).decode("ascii"),
    }


def serializable_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        variant: {
            "balancedAccuracy": value.balanced_accuracy,
            "realRecall": value.real_recall,
            "syntheticRecall": value.synthetic_recall,
            "syntheticRecallBySource": value.synthetic_recall_by_source,
            "falsePositives": value.false_positives,
            "falsePositiveTrials": value.false_positive_trials,
            "falsePositiveRate": value.false_positive_rate,
            "falsePositiveWilson95": value.false_positive_wilson95,
        }
        for variant, value in metrics.items()
    }


def evaluate_candidates(
    base_model: Any,
    checkpoints: Sequence[Path],
    selector_items: Sequence[Item],
    selector_rows: Sequence[dict[str, Any]],
    recipe: Mapping[str, Any],
    output_dir: Path,
    budget: PaidTimeBudget,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    expected = branch_candidate_ids(recipe)
    found = [checkpoint.stem for checkpoint in checkpoints]
    if found != expected:
        raise ValueError(f"M5 candidate checkpoint order changed: {found}")
    grid: list[dict[str, Any]] = []
    winning: tuple[tuple[float, ...], dict[str, Any]] | None = None
    for branch_order, checkpoint in enumerate(checkpoints):
        budget.check(f"candidate-{checkpoint.stem}-export")
        model = load_checkpoint_model(base_model, checkpoint)
        candidate_id = checkpoint.stem
        onnx_path = output_dir / "models" / f"{candidate_id}.onnx"
        with Image.open(selector_items[0].path) as opened:
            probe_image = ImageOps.exif_transpose(opened).convert("RGB")
        probe = preprocess_image(probe_image, selector_items[0], "original", training=False, branch="selector-parity", epoch=0)
        model_receipt = export_onnx(model, onnx_path, recipe, parity_pixels=np.expand_dims(probe, axis=0))
        import onnxruntime as ort
        session = ort.InferenceSession(str(onnx_path), providers=ort_cuda_providers(ort))
        logits = {variant: predict_onnx_variant(session, selector_items, variant, recipe, budget) for variant in VARIANTS}
        selected = choose_selector_threshold(logits, selector_rows, recipe["selection"]["gates"])
        row: dict[str, Any] = {
            "candidateId": candidate_id,
            "checkpoint": {
                "path": checkpoint.relative_to(ROOT).as_posix(),
                "bytes": checkpoint.stat().st_size,
                "sha256": digest_file(checkpoint),
            },
            "model": model_receipt,
            "selectorLogits": {variant: pack_float32(logits[variant]) for variant in VARIANTS},
            "accepted": selected is not None,
        }
        if selected is not None:
            threshold, metrics, key = selected
            branch_name, _, epoch_text = candidate_id.rpartition("-epoch-")
            ranking = (*key[:-1], -branch_order, -int(epoch_text), key[-1])
            row["rawThreshold"] = threshold
            row["metrics"] = serializable_metrics(metrics)
            row["selectionKey"] = list(ranking)
            if winning is None or ranking > winning[0]:
                winning = (ranking, row)
        grid.append(row)
        del model
    return grid, None if winning is None else winning[1]


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise ValueError(f"M5 partial output already exists: {temporary.name}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json(path: Path, value: object) -> None:
    atomic_write_bytes(path, canonical_json(value))


def atomic_torch_save(path: Path, value: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise ValueError(f"M5 partial checkpoint already exists: {temporary.name}")
    torch.save(value, temporary)
    descriptor = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def accumulation_window_samples(total_items: int, batch_size: int, accumulation: int, batch_index: int) -> int:
    if min(total_items, batch_size, accumulation, batch_index) <= 0:
        raise ValueError("M5 accumulation dimensions must be positive")
    total_batches = math.ceil(total_items / batch_size)
    if batch_index > total_batches:
        raise ValueError("M5 accumulation batch index is out of range")
    window_first_batch = ((batch_index - 1) // accumulation) * accumulation + 1
    first_item = (window_first_batch - 1) * batch_size
    return min(total_items - first_item, batch_size * accumulation)


class PaidTimeBudget:
    def __init__(
        self,
        receipt: Mapping[str, Any],
        recipe: Mapping[str, Any],
        state_path: Path,
        *,
        clock: Any = time.time,
    ) -> None:
        self.created = int(receipt["createdAtUnix"])
        self.workload_stop = int(receipt["workloadStopAtUnix"])
        self.deadline = self.workload_stop - int(recipe["training"]["deadlineSafetySeconds"])
        self.maximum = int(recipe["training"]["maximumPaidWallClockSeconds"])
        self.state_path = state_path
        self.clock = clock
        self.last_persisted = 0.0
        self.check("budget-initialized", force_persist=True)

    def check(self, phase: str, *, force_persist: bool = False) -> None:
        now = float(self.clock())
        if now < self.created - 60:
            raise ValueError("M5 paid-time clock precedes the provisioned pod")
        elapsed = max(0.0, now - self.created)
        if elapsed >= self.maximum or now >= self.deadline:
            self._persist(now, phase, exhausted=True)
            raise TimeoutError(f"M5 paid-time deadline reached before {phase}")
        if force_persist or now - self.last_persisted >= 60.0:
            self._persist(now, phase, exhausted=False)

    def _persist(self, now: float, phase: str, *, exhausted: bool) -> None:
        write_json(self.state_path, {
            "schemaVersion": 1,
            "createdAtUnix": self.created,
            "workloadStopAtUnix": self.workload_stop,
            "maximumPaidWallClockSeconds": self.maximum,
            "lastObservedUnix": now,
            "cumulativePaidWallClockSeconds": max(0.0, now - self.created),
            "phase": phase,
            "exhausted": exhausted,
        })
        self.last_persisted = now


def ensure_run_marker(
    output_dir: Path,
    recipe: Mapping[str, Any],
    protocol_commit: str,
    environment: Mapping[str, Any],
    provisioning: Mapping[str, Any],
    provisioning_sha256: str,
) -> None:
    marker_path = output_dir / "run-marker.json"
    marker = {
        "schemaVersion": 1,
        "protocolCommit": protocol_commit,
        "recipeSha256": digest_file(RECIPE_PATH),
        "trainingManifestSha256": recipe["sourceEvidence"]["trainingManifest"]["compressedSha256"],
        "selectorManifestSha256": recipe["sourceEvidence"]["selectorManifest"]["sha256"],
        "initialModelSha256": recipe["initialModel"]["sha256"],
        "environment": environment,
        "provisioningReceiptSha256": provisioning_sha256,
        "paidDeadline": {
            "createdAtUnix": provisioning["createdAtUnix"],
            "workloadStopAtUnix": provisioning["workloadStopAtUnix"],
            "maximumPaidWallClockSeconds": provisioning["maximumRuntimeSeconds"],
            "operatorStopRequired": provisioning["operatorStopRequired"],
            "providerAutoStopAvailable": provisioning["providerAutoStopAvailable"],
        },
        "h3PixelsRead": False,
    }
    receipt_path = output_dir / "preflight" / "preflight-receipt.json"
    if not receipt_path.is_file() or any(path.name.endswith(".partial") for path in (output_dir / "preflight").rglob("*")):
        raise ValueError("M5 candidate output has an incomplete preflight")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schemaVersion") != 1
        or receipt.get("status") != "preflight-pass"
        or receipt.get("protocolCommit") != protocol_commit
        or receipt.get("recipeSha256") != digest_file(RECIPE_PATH)
        or receipt.get("environment") != environment
        or receipt.get("provisioningReceiptSha256") != provisioning_sha256
        or receipt.get("selectorRead") is not False
        or receipt.get("terminalRegressionsRead") is not False
        or receipt.get("h3PixelsRead") is not False
    ):
        raise ValueError("M5 preflight receipt does not bind this full run")
    if marker_path.exists():
        existing = json.loads(marker_path.read_text(encoding="utf-8"))
        if existing != marker:
            raise ValueError("M5 resume marker changed")
        return
    existing = {path.name for path in output_dir.iterdir()}
    expected = {"runpod-provisioning-receipt.json", "preflight"}
    if existing != expected:
        raise ValueError("M5 full training requires exactly the provisioning receipt and completed preflight")
    write_json(marker_path, marker)


def run_preflight(
    items: Sequence[Item],
    recipe: Mapping[str, Any],
    output_dir: Path,
    protocol_commit: str,
    environment: Mapping[str, Any],
    provisioning: Mapping[str, Any],
) -> int:
    """Exercise one exact L40S batch without opening selector or regression data."""
    import copy
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    preflight_dir = output_dir / "preflight"
    receipt_path = preflight_dir / "preflight-receipt.json"
    if {path.name for path in output_dir.iterdir()} != {"runpod-provisioning-receipt.json"}:
        raise ValueError("M5 preflight requires a clean output root containing only its provisioning receipt")
    if receipt_path.exists():
        raise ValueError("M5 preflight receipt already exists; reuse is forbidden")
    preflight_dir.mkdir(parents=True, exist_ok=True)
    budget = PaidTimeBudget(provisioning, recipe, preflight_dir / "paid-time.json")
    branch = recipe["training"]["branches"][0]
    batch_size = int(recipe["training"]["perGpuBatchSize"])
    if len(items) < batch_size:
        raise ValueError("M5 preflight does not have one complete training batch")
    budget.check("preflight-pixel-verification")
    verify_all_items(items[:batch_size], workers=int(recipe["training"]["workers"]))
    budget.check("preflight-runtime-load")
    base_model, teacher, upstream_hashes = prepare_runtime(recipe, preflight_dir / "hf-cache")
    initial_parity = verify_initial_model_parity(base_model, items[:16], recipe)
    model = copy.deepcopy(base_model).to("cuda", dtype=torch.float32)
    teacher = teacher.to("cuda", dtype=torch.float32)
    parameter_groups = configure_branch(model, branch)
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=float(branch["weightDecay"]),
    )
    dataset = ImageDataset(items[:batch_size], branch="preflight", epoch=1, variant=None, training=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(recipe["training"]["workers"]),
        pin_memory=True,
        collate_fn=collate,
    )
    budget.check("preflight-training-batch")
    pixels, labels, weights, anchors, _indexes = next(iter(loader))
    pixels = pixels.to("cuda", non_blocking=True)
    labels = labels.to("cuda", non_blocking=True)
    weights = weights.to("cuda", non_blocking=True)
    anchors = anchors.to("cuda", non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(pixel_values=pixels).logits.reshape(-1).float()
        with torch.no_grad():
            teacher_logits = teacher(pixel_values=pixels).logits.reshape(-1).float()
        bce = (functional.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean()
        anchor_loss = (((logits - teacher_logits) ** 2) * anchors).mean()
        loss = bce + float(branch["teacherAnchorCoefficient"]) * anchor_loss
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        max_norm=float(recipe["training"]["gradientClipNorm"]),
    ).detach().cpu())
    optimizer.step()
    model = model.to("cpu")
    teacher.to("cpu")
    torch.cuda.empty_cache()
    budget.check("preflight-export")
    model_receipt = export_onnx(
        model,
        preflight_dir / "model.onnx",
        recipe,
        parity_pixels=pixels.detach().cpu().numpy()[:1],
    )
    receipt = {
        "schemaVersion": 1,
        "status": "preflight-pass",
        "protocolCommit": protocol_commit,
        "recipeSha256": digest_file(RECIPE_PATH),
        "environment": environment,
        "provisioningReceiptSha256": environment["provisioningReceiptSha256"],
        "branch": branch["name"],
        "batchSize": batch_size,
        "mixedPrecision": recipe["training"]["mixedPrecision"],
        "weightedBce": float(bce.detach().cpu()),
        "maskedTeacherMse": float(anchor_loss.detach().cpu()),
        "gradientNormBeforeClip": gradient_norm,
        "initialPytorchOnnxParityMaximumAbsoluteError": initial_parity,
        "upstreamSourceSha256": upstream_hashes,
        "postStepModel": model_receipt,
        "selectorRead": False,
        "terminalRegressionsRead": False,
        "h3PixelsRead": False,
        "paidTime": json.loads((preflight_dir / "paid-time.json").read_text(encoding="utf-8")),
    }
    write_json(receipt_path, receipt)
    print(json.dumps({
        "event": "preflight-pass",
        "batchSize": batch_size,
        "modelSha256": model_receipt["sha256"],
        "selectorRead": False,
        "h3PixelsRead": False,
    }, sort_keys=True), flush=True)
    return 0


def require_cuda_determinism_environment(recipe: Mapping[str, Any]) -> None:
    expected = recipe["training"]["deterministicCudaRuntime"]["cublasWorkspaceConfig"]
    if expected != ":4096:8" or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected:
        raise ValueError("M5 requires the frozen deterministic cuBLAS workspace before Torch import")


def execute(args: argparse.Namespace) -> int:
    recipe = load_recipe(RECIPE_PATH)
    require_cuda_determinism_environment(recipe)
    protocol_commit, source_tree, authorization_commit, authorization_receipt_sha256, authorization_public_ci = resolve_authorized_run()
    import torch
    import transformers

    data_root = require_canonical_path(args.data_root, recipe["sourceEvidence"]["dataRoot"], label="data root")
    train_manifest = require_canonical_path(
        args.train_manifest, recipe["sourceEvidence"]["trainingManifest"]["trackedPath"], label="training manifest",
    )
    output_dir = require_canonical_path(
        args.output_dir, recipe["output"]["candidateRoot"], label="candidate output", must_exist=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    provisioning_path = require_canonical_path(
        args.runpod_provisioning_receipt,
        recipe["training"]["provisioningReceiptPath"],
        label="RunPod provisioning receipt",
    )
    provisioning, provisioning_sha256 = load_provisioning_receipt(provisioning_path, recipe)
    if digest_file(train_manifest) != recipe["sourceEvidence"]["trainingManifest"]["compressedSha256"]:
        raise ValueError("M5 training manifest bytes changed")
    rows = read_jsonl(train_manifest)
    validate_manifest_rows(
        rows,
        expected_items=recipe["sourceEvidence"]["trainingManifest"]["items"],
        expected_class_counts=recipe["sourceEvidence"]["trainingManifest"]["classCounts"],
    )
    environment = environment_receipt(provisioning, provisioning_sha256, recipe)
    environment["sourceCommit"] = protocol_commit
    environment["sourceTree"] = source_tree
    environment["authorizationCommit"] = authorization_commit
    environment["authorizationReceiptSha256"] = authorization_receipt_sha256
    environment["authorizationPublicCi"] = authorization_public_ci
    validate_environment_receipt(environment, recipe)
    validate_environment_matches_provisioning(environment, provisioning)
    print(json.dumps({"event": "environment", **environment}, sort_keys=True), flush=True)
    torch.manual_seed(recipe["seed"])
    np.random.seed(recipe["seed"] % (2**32))
    random.seed(recipe["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    weights = source_balanced_weights(rows)
    items = load_items(rows, data_root, weights)
    if args.preflight_only:
        return run_preflight(items, recipe, output_dir, protocol_commit, environment, provisioning)
    ensure_run_marker(
        output_dir, recipe, protocol_commit, environment, provisioning, provisioning_sha256,
    )
    budget = PaidTimeBudget(provisioning, recipe, output_dir / "paid-time.json")
    budget.check("selector-manifest-preflight")
    selector_manifest = require_canonical_path(
        args.selector_manifest, recipe["sourceEvidence"]["selectorManifest"]["path"], label="selector manifest",
    )
    if digest_file(selector_manifest) != recipe["sourceEvidence"]["selectorManifest"]["sha256"]:
        raise ValueError("M5 selector manifest bytes changed")
    selector_rows = read_jsonl(selector_manifest)
    validate_manifest_rows(
        selector_rows,
        expected_items=recipe["sourceEvidence"]["selectorManifest"]["items"],
        expected_class_counts=recipe["sourceEvidence"]["selectorManifest"]["classCounts"],
        expected_source_counts=recipe["sourceEvidence"]["selectorManifest"]["sourceCounts"],
    )
    train_ids = {row["id"] for row in rows}
    train_hashes = {row["imageSha256"] for row in rows}
    if train_ids.intersection(row["id"] for row in selector_rows) or train_hashes.intersection(row["imageSha256"] for row in selector_rows):
        raise ValueError("M5 training and selector overlap")
    selector_items = load_items(selector_rows, data_root, [1.0] * len(selector_rows))
    budget.check("full-pixel-verification")
    verify_all_items([*items, *selector_items], workers=recipe["training"]["workers"])
    budget.check("full-runtime-load")
    base_model, teacher, upstream_hashes = prepare_runtime(recipe, output_dir / "hf-cache")
    initial_parity = verify_initial_model_parity(base_model, selector_items, recipe)
    checkpoints: list[Path] = []
    epoch_receipts: list[dict[str, Any]] = []
    for branch in recipe["training"]["branches"]:
        budget.check(f"branch-{branch['name']}-start")
        branch_checkpoints, branch_receipts = train_branch(
            base_model, teacher, branch, items, recipe, output_dir, budget,
        )
        checkpoints.extend(branch_checkpoints)
        epoch_receipts.extend(branch_receipts)
    budget.check("selector-evaluation-start")
    grid, selected = evaluate_candidates(
        base_model, checkpoints, selector_items, selector_rows, recipe, output_dir, budget,
    )
    grid_path = output_dir / "candidate-grid.json"
    grid_packet = {
        "schemaVersion": 1,
        "recipeSha256": digest_file(RECIPE_PATH),
        "protocolCommit": protocol_commit,
        "candidates": grid,
        "selectorManifestSha256": recipe["sourceEvidence"]["selectorManifest"]["sha256"],
        "h3PixelsRead": False,
    }
    write_json(grid_path, grid_packet)
    summary = {
        "schemaVersion": 1,
        "status": "selector-pass" if selected is not None else "selector-fail",
        "recipeSha256": digest_file(RECIPE_PATH),
        "protocolCommit": protocol_commit,
        "environment": environment,
        "upstreamSourceSha256": upstream_hashes,
        "initialPytorchOnnxParityMaximumAbsoluteError": initial_parity,
        "trainingManifestSha256": recipe["sourceEvidence"]["trainingManifest"]["compressedSha256"],
        "selectorManifestSha256": recipe["sourceEvidence"]["selectorManifest"]["sha256"],
        "trainingItems": len(items),
        "selectorItems": len(selector_items),
        "epochReceipts": epoch_receipts,
        "candidateGrid": {"path": grid_path.relative_to(ROOT).as_posix(), "sha256": digest_file(grid_path)},
        "selectedCandidateId": None if selected is None else selected["candidateId"],
        "h3PixelsRead": False,
        "terminalRegressionsRead": False,
    }
    summary_path = output_dir / "training-summary.json"
    write_json(summary_path, summary)
    if selected is None:
        failure = {
            "schemaVersion": 1,
            "status": "failed-m5-selector",
            "acceptanceEligible": False,
            "recipeSha256": digest_file(RECIPE_PATH),
            "protocolCommit": protocol_commit,
            "trainingSummary": summary,
            "trainingSummarySha256": digest_file(summary_path),
            "candidateGrid": grid_packet,
            "candidateGridSha256": digest_file(grid_path),
            "h3PixelsRead": False,
            "terminalRegressionsRead": False,
            "reason": "No predeclared candidate and exhaustive raw threshold passed every fresh-selector gate.",
        }
        write_json(output_dir / "failed-training-attempt-1.json", failure)
        print(json.dumps({"event": "selector-fail", "policy": "stop"}), flush=True)
        return 2
    raw_threshold = float(selected["rawThreshold"])
    display_logit = math.log(0.65 / 0.35)
    lock = {
        "schemaVersion": 1,
        "status": "m5-selected-pre-regression",
        "acceptanceEligible": False,
        "recipeSha256": digest_file(RECIPE_PATH),
        "protocolCommit": protocol_commit,
        "trainingSummary": summary,
        "trainingSummarySha256": digest_file(summary_path),
        "candidateGrid": grid_packet,
        "candidateGridSha256": digest_file(grid_path),
        "selectedCandidateId": selected["candidateId"],
        "selectedModel": selected["model"],
        "rawThreshold": raw_threshold,
        "calibration": {"slope": 1.0, "intercept": display_logit - raw_threshold, "displayThreshold": 0.65},
        "selectorMetrics": selected["metrics"],
        "selectionKey": selected["selectionKey"],
        "selectionInfluencedByRegression": False,
        "terminalRegressionsRead": False,
        "h3PixelsRead": False,
    }
    write_json(output_dir / "selection-lock-draft.json", lock)
    print(json.dumps({
        "event": "selector-pass",
        "candidate": selected["candidateId"],
        "modelSha256": selected["model"]["sha256"],
        "rawThreshold": raw_threshold,
        "h3PixelsRead": False,
    }, sort_keys=True), flush=True)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", default="benchmark/data/m4-head")
    value.add_argument("--train-manifest", default="benchmark/evidence/m4/train-manifest.jsonl.gz")
    value.add_argument("--selector-manifest", default="benchmark/evidence/m4/validation-manifest.jsonl")
    value.add_argument("--output-dir", default="benchmark/candidates/prooflens-cf384-m5")
    value.add_argument(
        "--runpod-provisioning-receipt",
        default="benchmark/candidates/prooflens-cf384-m5/runpod-provisioning-receipt.json",
    )
    value.add_argument("--preflight-only", action="store_true")
    return value


if __name__ == "__main__":
    try:
        raise SystemExit(execute(parser().parse_args()))
    except Exception as error:
        print(f"M5 train-select failed: {error}", file=sys.stderr)
        raise
