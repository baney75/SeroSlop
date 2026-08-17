"""Freeze and verify the fixed-M2 submission proxy without running inference.

Browser inference is deliberately isolated in ``scripts/bounty-proxy-browser.mjs``.
The freeze phase reads source metadata and fixed model metadata only. It never
opens a selected image or imports Pillow, ONNX Runtime, or Torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "benchmark/evidence/bounty-proxy-m2-v1"
H3_ROOT = ROOT / "benchmark/data/h3-met-holdout-v1"
H3_MANIFEST = H3_ROOT / "manifest.jsonl"
TASTE_ROOT = ROOT / "benchmark/data/m6-frontier-cache/taste"
TASTE_ASSETS = TASTE_ROOT / "assets.parquet"
MODEL = ROOT / "weights/prooflens-cf384.onnx"
MODEL_LOCK = ROOT / "model-lock.json"
CALIBRATION = ROOT / "benchmark/evidence/m2/calibration.json"

LABEL = "M2 fixed-threshold H3 Met / TASTE submission proxy"
SELECTION_NAMESPACE = "seroslop-bounty-proxy-m2-v1"
TASTE_REVISION = "731a7f588d433214c6d864d2e9f47978d91aed6b"
DISPLAY_THRESHOLD = 0.65
RAW_LOGIT_THRESHOLD = -0.993612766265869
RAW_PROBABILITY_THRESHOLD = 0.27019907955040323
CALIBRATION_INTERCEPT = 1.6126519746720926
TASTE_GROUPS = (
    "FLUX.2 [max]",
    "GPT Image 1.5",
    "Nano Banana 2",
    "Seedream 5.0 Lite",
)

EXPECTED = {
    "h3Manifest": (989_390, "50574778ab0d58f839f1dccc3c99da5f6dca98150186f13aeca8d9ba052e9547"),
    "tasteAssets": (26_772, "326e9300bac89f5ed884de7a9a59dccfc7d5aa203f6d2844f604000dc4e32bf1"),
    "model": (87_442_080, "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47"),
    "calibration": (688, "06d2452a8db9de26d42285cdc9dad0d233d397a6015583604c64480aec560e2c"),
    "modelLock": (None, "2a818b7b2582bc9614f02f178d9af997f46628734ea078a79415c3d68d3061f0"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ) + "\n").encode("utf-8")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise ValueError("JSON is not UTF-8") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")),
        )
    except json.JSONDecodeError as error:
        raise ValueError("malformed JSON") from error


def _safe_relative(value: str, *, prefix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value or any(ord(c) < 32 for c in value):
        raise ValueError("unsafe relative path")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("unsafe relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("unsafe relative path")
    normalized = path.as_posix()
    if prefix is not None and not normalized.startswith(prefix):
        raise ValueError(f"path must begin with {prefix}")
    return normalized


def _physical_file(root: Path, relative: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"unsafe root: {root}")
    relative = _safe_relative(relative)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlink input: {cursor}")
    try:
        mode = cursor.stat().st_mode
    except FileNotFoundError as error:
        raise ValueError(f"missing input: {cursor}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"input is not a regular file: {cursor}")
    resolved_root = root.resolve(strict=True)
    resolved = cursor.resolve(strict=True)
    if resolved.parent != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("input escaped fixed root")
    return cursor


def _require_fixed_file(path: Path, expected: str) -> dict[str, Any]:
    expected_bytes, expected_sha = EXPECTED[expected]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe fixed input: {path}")
    actual_bytes = path.stat().st_size
    actual_sha = sha256(path)
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(f"{expected} byte count changed")
    if actual_sha != expected_sha:
        raise ValueError(f"{expected} SHA-256 changed")
    return {"bytes": actual_bytes, "path": path.relative_to(ROOT).as_posix(), "sha256": actual_sha}


def _read_h3_rows() -> list[dict[str, Any]]:
    _require_fixed_file(H3_MANIFEST, "h3Manifest")
    rows: list[dict[str, Any]] = []
    for raw_line in H3_MANIFEST.read_bytes().splitlines():
        if not raw_line:
            continue
        row = parse_json_bytes(raw_line)
        if not isinstance(row, dict):
            raise ValueError("H3 row is not an object")
        rows.append(row)
    if len(rows) != 600:
        raise ValueError("H3 panel must contain exactly 600 rows")
    identities: set[str] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        if row.get("label") != 0 or row.get("source") != "met-open-access" or row.get("split") != "confirmatory-reserved":
            raise ValueError("H3 label/source/split changed")
        identity = row.get("id")
        image_hash = row.get("imageSha256")
        path = _safe_relative(row.get("path"), prefix="real/met-open-access/")
        if not isinstance(identity, str) or not identity.startswith("met-open-access:"):
            raise ValueError("H3 identity changed")
        if not isinstance(image_hash, str) or len(image_hash) != 64 or any(c not in "0123456789abcdef" for c in image_hash):
            raise ValueError("H3 image digest changed")
        if identity in identities or path in paths or image_hash in hashes:
            raise ValueError("H3 duplicate identity/path/image")
        identities.add(identity); paths.add(path); hashes.add(image_hash)
    return rows


def _read_taste_rows() -> list[dict[str, Any]]:
    _require_fixed_file(TASTE_ASSETS, "tasteAssets")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("freeze requires pinned pyarrow for TASTE metadata") from error
    expected_schema = pa.schema([
        ("asset_id", pa.int64()),
        ("model", pa.string()),
        ("image_url", pa.string()),
        ("track", pa.string()),
        ("image_path", pa.string()),
    ])
    actual_schema = pq.ParquetFile(TASTE_ASSETS).schema_arrow.remove_metadata()
    if not actual_schema.equals(expected_schema):
        raise ValueError("TASTE schema changed")
    rows = pq.read_table(TASTE_ASSETS).to_pylist()
    if len(rows) != 644 or {row.get("asset_id") for row in rows} != set(range(1, 645)):
        raise ValueError("TASTE asset census changed")
    if Counter(row.get("model") for row in rows) != Counter({group: 161 for group in TASTE_GROUPS}):
        raise ValueError("TASTE model census changed")
    if Counter(row.get("track") for row in rows) != Counter({"descriptions": 320, "aesthetics": 324}):
        raise ValueError("TASTE track census changed")
    paths: set[str] = set()
    for row in rows:
        path = _safe_relative(row.get("image_path"), prefix="images/")
        if path in paths:
            raise ValueError("duplicate TASTE image path")
        paths.add(path)
        if not isinstance(row.get("image_url"), str) or not row["image_url"].startswith("https://"):
            raise ValueError("TASTE publisher URL changed")
    return rows


def _selection_digest(model: str, asset_id: int) -> str:
    value = f"{SELECTION_NAMESPACE}\0{model}\0{asset_id:d}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_taste(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    groups: dict[str, list[dict[str, Any]]] = {group: [] for group in TASTE_GROUPS}
    for row in rows:
        model = row.get("model")
        if model not in groups:
            raise ValueError("unknown TASTE model")
        groups[model].append(row)
    if any(len(group) != 161 for group in groups.values()):
        raise ValueError("TASTE group size changed")
    selected: list[dict[str, Any]] = []
    omitted: dict[str, list[int]] = {}
    for model in TASTE_GROUPS:
        ranked = sorted(groups[model], key=lambda row: (_selection_digest(model, int(row["asset_id"])), int(row["asset_id"])))
        omitted[model] = sorted(int(row["asset_id"]) for row in ranked[150:])
        for rank, row in enumerate(ranked[:150]):
            asset_id = int(row["asset_id"])
            selected.append({
                "assetId": asset_id,
                "id": f"taste:{TASTE_REVISION}:{asset_id}",
                "label": 1,
                "path": _safe_relative(row["image_path"], prefix="images/"),
                "root": "taste",
                "selection": {
                    "algorithm": "sha256-ranked-150-per-taste-model",
                    "namespace": SELECTION_NAMESPACE,
                    "rank": rank,
                },
                "source": "taste",
                "sourceGroup": model,
                "track": row["track"],
            })
    return sorted(selected, key=lambda row: row["id"].encode("utf-8")), omitted


def _frozen_rows() -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    h3_rows = _read_h3_rows()
    real = []
    for rank, row in enumerate(h3_rows):
        real.append({
            "id": row["id"],
            "imageSha256": row["imageSha256"],
            "label": 0,
            "path": _safe_relative(row["path"], prefix="real/met-open-access/"),
            "root": "h3",
            "selection": {
                "algorithm": "all-h3-met-600",
                "namespace": SELECTION_NAMESPACE,
                "rank": rank,
            },
            "source": "met-open-access",
            "sourceGroup": "Met Open Access",
        })
    synthetic, omitted = select_taste(_read_taste_rows())
    rows = real + synthetic
    if len(rows) != 1200 or Counter(row["label"] for row in rows) != Counter({0: 600, 1: 600}):
        raise ValueError("proxy class allocation changed")
    if len({row["id"] for row in rows}) != 1200 or len({(row["root"], row["path"]) for row in rows}) != 1200:
        raise ValueError("proxy identity/path collision")
    return rows, omitted


def _publish_directory(
    parent: Path,
    name: str,
    files: dict[str, bytes],
    *,
    failure_hook: Callable[[str], None] | None = None,
) -> Path:
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("evidence parent must be a physical directory")
    if "/" in name or name in ("", ".", ".."):
        raise ValueError("unsafe bundle name")
    final = parent / name
    partial = parent / f".{name}.partial"
    if final.exists() or final.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(f"bundle target already exists: {final}")
    installed = False
    try:
        partial.mkdir(mode=0o700)
        for relative, raw in sorted(files.items()):
            if PurePosixPath(relative).parent != PurePosixPath("."):
                raise ValueError("bundle files must be flat")
            target = partial / relative
            with target.open("xb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
        directory_fd = os.open(partial, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if failure_hook: failure_hook("before-rename")
        os.replace(partial, final)
        installed = True
        if failure_hook: failure_hook("after-rename")
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return final
    except Exception as error:
        if installed and final.exists():
            try:
                shutil.rmtree(final)
                parent_fd = os.open(parent, os.O_RDONLY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            except Exception as rollback_error:
                raise RuntimeError("bundle publication entered an unknown state") from rollback_error
        elif partial.exists():
            shutil.rmtree(partial)
        raise error


def _manifest_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json(row) for row in rows)


def freeze_manifest(evidence_root: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    rows, omitted = _frozen_rows()
    manifest = _manifest_bytes(rows)
    lock = {
        "bountyAcceptanceClaimed": False,
        "calibration": {
            **_require_fixed_file(CALIBRATION, "calibration"),
            "intercept": CALIBRATION_INTERCEPT,
            "rawLogitThreshold": RAW_LOGIT_THRESHOLD,
            "rawProbabilityThreshold": RAW_PROBABILITY_THRESHOLD,
        },
        "classCounts": {"real": 600, "synthetic": 600},
        "decision": {"displayThreshold": DISPLAY_THRESHOLD, "inclusive": True},
        "h3Manifest": {**_require_fixed_file(H3_MANIFEST, "h3Manifest"), "rows": 600},
        "inferenceRun": False,
        "label": LABEL,
        "manifestSha256": hashlib.sha256(manifest).hexdigest(),
        "model": _require_fixed_file(MODEL, "model"),
        "modelLock": _require_fixed_file(MODEL_LOCK, "modelLock"),
        "pixelsReadAtFreeze": False,
        "rows": 1200,
        "schemaVersion": 1,
        "selection": {
            "namespace": SELECTION_NAMESPACE,
            "tasteGroups": 4,
            "tasteOmittedAssetIds": omitted,
            "tasteSelectedPerGroup": 150,
            "tasteUnselectedPerGroup": 11,
        },
        "sourceGroupCounts": {"Met Open Access": 600, **{group: 150 for group in TASTE_GROUPS}},
        "status": "m2-bounty-proxy-pre-score-locked",
        "tasteAssets": {**_require_fixed_file(TASTE_ASSETS, "tasteAssets"), "rows": 644},
    }
    _publish_directory(evidence_root, "frozen", {
        "manifest.jsonl": manifest,
        "source-lock.json": canonical_json(lock),
    })
    return lock


def reopen_frozen(evidence_root: Path = EVIDENCE_ROOT) -> tuple[list[dict[str, Any]], dict[str, Any], bytes, bytes]:
    bundle = evidence_root / "frozen"
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("frozen bundle is missing or unsafe")
    if sorted(path.name for path in bundle.iterdir()) != ["manifest.jsonl", "source-lock.json"]:
        raise ValueError("frozen bundle inventory changed")
    manifest_path = _physical_file(bundle, "manifest.jsonl")
    lock_path = _physical_file(bundle, "source-lock.json")
    manifest = manifest_path.read_bytes()
    lock_raw = lock_path.read_bytes()
    lock = parse_json_bytes(lock_raw)
    if not isinstance(lock, dict) or canonical_json(lock) != lock_raw:
        raise ValueError("source lock is not canonical")
    rows: list[dict[str, Any]] = []
    for raw_line in manifest.splitlines(keepends=True):
        row = parse_json_bytes(raw_line)
        if not isinstance(row, dict) or canonical_json(row) != raw_line:
            raise ValueError("manifest row is not canonical")
        rows.append(row)
    if hashlib.sha256(manifest).hexdigest() != lock.get("manifestSha256"):
        raise ValueError("manifest digest does not match source lock")
    expected_keys = {
        "bountyAcceptanceClaimed", "calibration", "classCounts", "decision", "h3Manifest",
        "inferenceRun", "label", "manifestSha256", "model", "modelLock", "pixelsReadAtFreeze",
        "rows", "schemaVersion", "selection", "sourceGroupCounts", "status", "tasteAssets",
    }
    if set(lock) != expected_keys or lock.get("schemaVersion") != 1 or lock.get("status") != "m2-bounty-proxy-pre-score-locked":
        raise ValueError("source-lock schema changed")
    if lock.get("rows") != 1200 or lock.get("classCounts") != {"real": 600, "synthetic": 600}:
        raise ValueError("source-lock counts changed")
    if lock.get("pixelsReadAtFreeze") is not False or lock.get("inferenceRun") is not False or lock.get("bountyAcceptanceClaimed") is not False:
        raise ValueError("source-lock boundary changed")
    if lock.get("decision") != {"displayThreshold": DISPLAY_THRESHOLD, "inclusive": True}:
        raise ValueError("decision rule changed")
    if lock.get("model") != _require_fixed_file(MODEL, "model") or lock.get("modelLock") != _require_fixed_file(MODEL_LOCK, "modelLock"):
        raise ValueError("model binding changed")
    expected_calibration = {
        **_require_fixed_file(CALIBRATION, "calibration"),
        "intercept": CALIBRATION_INTERCEPT,
        "rawLogitThreshold": RAW_LOGIT_THRESHOLD,
        "rawProbabilityThreshold": RAW_PROBABILITY_THRESHOLD,
    }
    if lock.get("calibration") != expected_calibration:
        raise ValueError("calibration binding changed")
    if len(rows) != 1200 or Counter(row.get("label") for row in rows) != Counter({0: 600, 1: 600}):
        raise ValueError("manifest allocation changed")
    if Counter(row.get("sourceGroup") for row in rows) != Counter(lock.get("sourceGroupCounts")):
        raise ValueError("manifest source-group allocation changed")
    return rows, lock, manifest, lock_raw


def verify_inputs(evidence_root: Path = EVIDENCE_ROOT) -> dict[str, Any]:
    rows, lock, manifest_raw, lock_raw = reopen_frozen(evidence_root)
    inputs: list[dict[str, Any]] = []
    image_hashes: set[str] = set()
    for row in rows:
        root_name = row.get("root")
        if root_name == "h3" and row.get("source") == "met-open-access" and row.get("label") == 0:
            root = H3_ROOT
        elif root_name == "taste" and row.get("source") == "taste" and row.get("label") == 1:
            root = TASTE_ROOT
        else:
            raise ValueError("manifest root/source/label binding changed")
        path = _physical_file(root, row.get("path"))
        actual_sha = sha256(path)
        declared_sha = row.get("imageSha256")
        if declared_sha is not None and declared_sha != actual_sha:
            raise ValueError(f"declared image digest changed for {row.get('id')}")
        if actual_sha in image_hashes:
            raise ValueError("duplicate selected image bytes")
        image_hashes.add(actual_sha)
        inputs.append({
            "bytes": path.stat().st_size,
            "id": row["id"],
            "path": row["path"],
            "root": root_name,
            "sha256": actual_sha,
        })
    inputs_raw = _manifest_bytes(inputs)
    receipt = {
        "calibrationSha256": lock["calibration"]["sha256"],
        "inputManifestSha256": hashlib.sha256(inputs_raw).hexdigest(),
        "label": LABEL,
        "manifestSha256": hashlib.sha256(manifest_raw).hexdigest(),
        "modelSha256": lock["model"]["sha256"],
        "rows": 1200,
        "schemaVersion": 1,
        "sourceLockSha256": hashlib.sha256(lock_raw).hexdigest(),
        "status": "m2-bounty-proxy-inputs-verified",
    }
    _publish_directory(evidence_root, "verified-inputs", {
        "input-manifest.jsonl": inputs_raw,
        "verification.json": canonical_json(receipt),
    })
    return receipt


def metrics(rows: list[dict[str, Any]], *, probability_field: str = "displayScore") -> dict[str, Any]:
    if not rows or any(row.get("label") not in (0, 1) for row in rows):
        raise ValueError("metrics require binary labelled rows")
    if any(not isinstance(row.get(probability_field), (int, float)) or isinstance(row.get(probability_field), bool) or not math.isfinite(row[probability_field]) for row in rows):
        raise ValueError("metrics require finite scores")
    real = [row for row in rows if row["label"] == 0]
    synthetic = [row for row in rows if row["label"] == 1]
    if not real or not synthetic:
        raise ValueError("metrics require both classes")
    flagged = lambda row: row[probability_field] >= DISPLAY_THRESHOLD
    tn = sum(not flagged(row) for row in real)
    fp = len(real) - tn
    tp = sum(flagged(row) for row in synthetic)
    fn = len(synthetic) - tp
    real_recall = tn / len(real)
    synthetic_recall = tp / len(synthetic)
    groups = {}
    for group in TASTE_GROUPS:
        members = [row for row in synthetic if row.get("sourceGroup") == group]
        if members:
            groups[group] = sum(flagged(row) for row in members) / len(members)
    return {
        "balancedAccuracy": (real_recall + synthetic_recall) / 2,
        "confusion": {"fn": fn, "fp": fp, "tn": tn, "tp": tp},
        "realRecall": real_recall,
        "syntheticGroupRecalls": groups,
        "syntheticRecall": synthetic_recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("freeze-manifest", "verify-inputs"))
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    arguments = parser.parse_args()
    if arguments.evidence_root.resolve() != EVIDENCE_ROOT.resolve():
        raise SystemExit("production CLI requires the fixed evidence root")
    if arguments.phase == "freeze-manifest":
        freeze_manifest(arguments.evidence_root)
    else:
        verify_inputs(arguments.evidence_root)


if __name__ == "__main__":
    main()
