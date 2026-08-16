#!/opt/conda/bin/python
"""Install and verify the exact Node runtime used by the M5 RunPod launcher."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
from urllib.request import ProxyHandler, Request, build_opener


VERSION = "v24.18.1"
NPM_VERSION = "11.16.0"
ARCHIVE_NAME = "node-v24.18.1-linux-x64.tar.xz"
ARCHIVE_URL = f"https://nodejs.org/download/release/{VERSION}/{ARCHIVE_NAME}"
ARCHIVE_BYTES = 31_525_884
ARCHIVE_SHA256 = "d6c664df3f3f61458e8c277585571328522d705166723a7c7823a9253a4d15a0"
NODE_SHA256 = "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a"
NPM_CLI_SHA256 = "8e5f6f3429f8cdbe693cdc29904e9d5a7b127a494bd15c804bd54c7403bfcbe7"
RUNTIME_ROOT = Path("/workspace/.seroslop/runtime")
INSTALL_ROOT = RUNTIME_ROOT / "node-v24.18.1-linux-x64"
ARCHIVE_PATH = RUNTIME_ROOT / ARCHIVE_NAME


def reject_symlinked_runtime_roots() -> None:
    for path in (Path("/workspace"), Path("/workspace/.seroslop"), RUNTIME_ROOT, INSTALL_ROOT):
        if path.exists() and path.is_symlink():
            raise ValueError(f"M5 Node runtime root is a symlink: {path}")


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def runtime_lock() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "version": VERSION,
        "npmVersion": NPM_VERSION,
        "platform": "linux",
        "architecture": "x64",
        "archive": {
            "url": ARCHIVE_URL,
            "bytes": ARCHIVE_BYTES,
            "sha256": ARCHIVE_SHA256,
        },
        "nodeSha256": NODE_SHA256,
        "npmCliSha256": NPM_CLI_SHA256,
        "installRoot": str(INSTALL_ROOT),
    }


def validate_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    expected = INSTALL_ROOT.name
    if path.is_absolute() or not path.parts or path.parts[0] != expected or ".." in path.parts:
        raise ValueError(f"Unsafe Node archive member: {member.name}")
    if member.issym() or member.islnk():
        target = PurePosixPath(member.linkname)
        if target.is_absolute():
            raise ValueError(f"Unsafe Node archive link: {member.name}")
        parts = list(path.parent.parts if member.issym() else ())
        for part in target.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                if len(parts) <= 1:
                    raise ValueError(f"Unsafe Node archive link: {member.name}")
                parts.pop()
            else:
                parts.append(part)
        if not parts or parts[0] != expected:
            raise ValueError(f"Unsafe Node archive link: {member.name}")


def verify_runtime() -> dict[str, object]:
    reject_symlinked_runtime_roots()
    node = INSTALL_ROOT / "bin/node"
    npm_cli = INSTALL_ROOT / "lib/node_modules/npm/bin/npm-cli.js"
    if not node.is_file() or node.is_symlink() or digest(node) != NODE_SHA256:
        raise ValueError("Pinned M5 Node executable is missing or changed")
    if not npm_cli.is_file() or npm_cli.is_symlink() or digest(npm_cli) != NPM_CLI_SHA256:
        raise ValueError("Pinned M5 npm CLI is missing or changed")
    node_version = subprocess.check_output([node, "--version"], text=True).strip()
    npm_version = subprocess.check_output([node, npm_cli, "--version"], text=True).strip()
    if node_version != VERSION or npm_version != NPM_VERSION:
        raise ValueError("Pinned M5 Node/npm versions changed")
    return {**runtime_lock(), "status": "verified", "nodePath": str(node)}


def download_archive() -> None:
    if ARCHIVE_PATH.exists():
        if ARCHIVE_PATH.is_symlink() or ARCHIVE_PATH.stat().st_size != ARCHIVE_BYTES or digest(ARCHIVE_PATH) != ARCHIVE_SHA256:
            raise ValueError("Existing M5 Node archive is not the pinned artifact")
        return
    partial = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".partial")
    if partial.exists():
        raise ValueError("Partial M5 Node runtime download exists; inspect it before retrying")
    request = Request(ARCHIVE_URL, headers={"User-Agent": "seroslop-m5-node-bootstrap"})
    with build_opener(ProxyHandler({})).open(request, timeout=60) as response, partial.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != ARCHIVE_BYTES or digest(partial) != ARCHIVE_SHA256:
        raise ValueError("Downloaded M5 Node runtime failed its byte lock")
    os.replace(partial, ARCHIVE_PATH)


def install_runtime() -> dict[str, object]:
    reject_symlinked_runtime_roots()
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if INSTALL_ROOT.exists():
        return verify_runtime()
    download_archive()
    stage = Path(tempfile.mkdtemp(prefix="node-stage-", dir=RUNTIME_ROOT))
    with tarfile.open(ARCHIVE_PATH, "r:xz") as archive:
        members = archive.getmembers()
        for member in members:
            validate_member(member)
        archive.extractall(stage, members=members)
    extracted = stage / INSTALL_ROOT.name
    if not extracted.is_dir() or extracted.is_symlink():
        raise ValueError("Pinned M5 Node archive did not contain its expected root")
    os.replace(extracted, INSTALL_ROOT)
    stage.rmdir()
    return verify_runtime()


if __name__ == "__main__":
    print(json.dumps(install_runtime(), sort_keys=True, separators=(",", ":")))
