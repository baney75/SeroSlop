"""Build the complete pixel-bound perceptual exclusion index.

The replacement holdout must be screened against every image that influenced an
earlier training, validation, confirmation, or web-negative decision.  Most of
those dHashes are already committed.  The legacy 1,200-image evaluation predates
that evidence contract, so this builder recovers each byte from its pinned
upstream locator, verifies the existing SHA-256 lock, and only then computes the
dHash.  Pixels remain under the ignored benchmark data directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request
import zlib


ROOT = Path(__file__).resolve().parents[2]
RECIPE_PATH = Path(__file__).with_name("recipe.json")
DHASH_PATTERN = re.compile(r"^[0-9a-f]{16}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def deterministic_gzip(value: bytes) -> bytes:
    """Emit gzip with a fixed ten-byte header, raw DEFLATE body, and trailer."""

    compressor = zlib.compressobj(level=9, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
    body = compressor.compress(value) + compressor.flush()
    header = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"
    trailer = (zlib.crc32(value) & 0xFFFFFFFF).to_bytes(4, "little") + (len(value) & 0xFFFFFFFF).to_bytes(4, "little")
    return header + body + trailer


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def require_safe_data_path(path: Path) -> Path:
    """Reject symlinked or escaped writable paths before any mutation."""

    lexical_root = (ROOT / "benchmark" / "data").absolute()
    lexical = path.absolute()
    try:
        lexical.relative_to(lexical_root)
    except ValueError as error:
        raise ValueError(f"Data path escapes benchmark/data: {path}") from error
    current = lexical_root
    if current.is_symlink():
        raise ValueError(f"Symlinked data path component: {current}")
    for component in lexical.relative_to(lexical_root).parts:
        current /= component
        if current.exists() and current.is_symlink():
            raise ValueError(f"Symlinked data path component: {current}")
    if lexical_root.resolve() not in lexical.resolve(strict=False).parents and lexical.resolve(strict=False) != lexical_root.resolve():
        raise ValueError(f"Resolved data path escapes benchmark/data: {path}")
    return lexical


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def image_dhash(value: bytes) -> str:
    # Importing here keeps the source-recovery helpers usable in lean tests.
    from prepare import image_facts

    return image_facts(value)[2]


def legacy_url(identifier: str, recipe: dict[str, Any]) -> tuple[str, str]:
    config = recipe["historicalExclusions"]["legacyPixelSources"]
    if identifier.startswith("open-images:"):
        image_id = identifier.rsplit(":", 1)[-1]
        return f"{config['openImagesBaseUrl']}/{image_id}.jpg", ".jpg"
    if identifier.startswith("qwen-image-bench:"):
        parts = identifier.split(":", 3)
        if len(parts) != 4:
            raise ValueError(f"Malformed legacy Qwen identifier: {identifier}")
        _, revision, source, filename = parts
        if revision != config["qwenRevision"] or not re.fullmatch(r"[A-Za-z0-9._+-]+", source):
            raise ValueError(f"Unexpected legacy Qwen source binding: {identifier}")
        if Path(filename).name != filename or not re.search(r"\.(?:jpe?g|png)$", filename, re.IGNORECASE):
            raise ValueError(f"Unsafe legacy Qwen filename: {identifier}")
        dataset = config["qwenDataset"]
        url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/images/{source}/{filename}?download=true"
        return url, Path(filename).suffix.lower()
    raise ValueError(f"Unsupported legacy evaluation identifier: {identifier}")


def recover_one(
    identifier: str,
    expected_sha256: str,
    recipe: dict[str, Any],
    cache_root: Path,
    *,
    allow_download: bool,
) -> tuple[str, bytes, str]:
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError(f"Malformed legacy pixel lock: {identifier}")
    url, extension = legacy_url(identifier, recipe)
    destination = require_safe_data_path(cache_root / f"{digest_bytes(identifier.encode())}{extension}")
    if destination.is_file():
        value = destination.read_bytes()
        if digest_bytes(value) == expected_sha256:
            return identifier, value, url
        if not allow_download:
            raise ValueError(f"Locked legacy pixel changed during offline verification: {identifier}")
        destination.unlink()
    elif not allow_download:
        raise ValueError(f"Locked legacy pixel is unavailable during offline verification: {identifier}")
    partial = require_safe_data_path(destination.with_suffix(destination.suffix + ".partial"))
    if partial.exists():
        partial.unlink()
    last_error: Exception | None = None
    for attempt in range(6):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ProofLens/1.0 (https://github.com/baney75/prooflens)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                value = response.read()
            if digest_bytes(value) != expected_sha256:
                raise ValueError(f"Recovered legacy pixel failed its existing SHA-256 lock: {identifier}")
            destination = require_safe_data_path(destination)
            with tempfile.NamedTemporaryFile(dir=cache_root, prefix=f".{destination.name}.", delete=False) as handle:
                handle.write(value)
                temporary = Path(handle.name)
            require_safe_data_path(temporary)
            destination = require_safe_data_path(destination)
            os.replace(temporary, destination)
            return identifier, value, url
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError) as error:
            last_error = error
            if attempt == 5:
                break
            time.sleep(min(2**attempt, 20))
    raise ValueError(f"Could not recover locked legacy pixel {identifier}: {last_error}")


def canonical_group(identifier: str, row: dict[str, Any], docci_clusters: dict[str, str]) -> str:
    if identifier.startswith("docci:"):
        example_id = identifier.rsplit(":", 1)[-1]
        cluster_id = docci_clusters.get(example_id)
        if not cluster_id:
            raise ValueError(f"DOCCI metadata is missing historical ID {identifier}")
        revision = str(row.get("datasetRevision", identifier.split(":", 2)[1]))
        return f"docci:{revision}:cluster:{cluster_id}"
    if identifier.startswith("qwen-image-bench:"):
        revision = identifier.split(":", 3)[1]
        filename = identifier.rsplit(":", 1)[-1]
        match = re.match(r"^(\d+)_", filename)
        group_id = str(row.get("sourceGroupId", row.get("groupId", match.group(1) if match else "")))
        if not group_id:
            raise ValueError(f"Qwen row has no prompt group: {identifier}")
        return f"qwen-image-bench:{revision}:prompt:{group_id}"
    group_id = row.get("sourceGroupId", row.get("groupId"))
    if group_id:
        return f"{row.get('datasetRevision', row.get('dataset', 'dataset'))}:group:{group_id}"
    return identifier


def load_docci_clusters(recipe: dict[str, Any], *, allow_download: bool) -> dict[str, str]:
    config = recipe["historicalExclusions"]["docciMetadata"]
    from prepare import download, validate_data_root

    data_root = validate_data_root(ROOT / "benchmark" / "data")
    path = require_safe_data_path(ROOT / config["path"])
    download(
        str(config["url"]),
        path,
        int(config["bytes"]),
        str(config["sha256"]),
        allow_download=allow_download,
        allowed_root=data_root,
    )
    path = require_safe_data_path(path)
    clusters: dict[str, str] = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        identifier = str(row.get("example_id", ""))
        cluster = str(row.get("cluster_id", ""))
        if not identifier or not cluster or identifier in clusters:
            raise ValueError("Official DOCCI cluster metadata is malformed or duplicated")
        clusters[identifier] = cluster
    if len(clusters) != int(config["items"]):
        raise ValueError("Official DOCCI cluster metadata row count changed")
    return clusters


def build_index(recipe: dict[str, Any], cache_root: Path, workers: int, *, allow_download: bool = True) -> dict[str, Any]:
    exclusions = recipe["historicalExclusions"]
    docci_clusters = load_docci_clusters(recipe, allow_download=allow_download)

    large_path = ROOT / exclusions["largeTrainManifestPath"]
    if digest(large_path) != exclusions["largeTrainManifestSha256"]:
        raise ValueError("Large training manifest changed")
    with gzip.open(large_path, "rt") as handle:
        large_rows = [json.loads(line) for line in handle if line.strip()]

    evaluation_path = ROOT / exclusions["evaluationPerceptualPath"]
    if digest(evaluation_path) != exclusions["evaluationPerceptualCompressedSha256"]:
        raise ValueError("Evaluation perceptual packet changed")
    expanded = gzip.decompress(evaluation_path.read_bytes())
    if digest_bytes(expanded) != exclusions["evaluationPerceptualExpandedSha256"]:
        raise ValueError("Evaluation perceptual packet expansion changed")
    evaluation_packet = json.loads(expanded)
    if (
        evaluation_packet.get("algorithm") != recipe["overlapPolicy"]["perceptualDhash64"]["algorithm"]
        or int(evaluation_packet.get("hammingThreshold", -1))
        != int(recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"])
    ):
        raise ValueError("Evaluation perceptual packet uses the wrong algorithm")
    evaluation_dhash = {
        str(row["id"]): (str(row["imageSha256"]), str(row["perceptualDhash64"]))
        for row in evaluation_packet["items"]
    }

    manifest_rows: list[tuple[str, dict[str, Any]]] = []
    for relative, expected in exclusions["manifestPathsSha256"].items():
        path = ROOT / relative
        if digest(path) != expected:
            raise ValueError(f"Historical manifest changed: {relative}")
        manifest_rows.extend((relative, row) for row in load_jsonl(path))

    items: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any], dhash: str, origin: str, source_manifest: str) -> None:
        identifier = str(row["id"])
        image_sha = str(row["imageSha256"])
        if not SHA256_PATTERN.fullmatch(image_sha) or not DHASH_PATTERN.fullmatch(dhash):
            raise ValueError(f"Malformed historical perceptual evidence: {identifier}")
        item = {
            "id": identifier,
            "imageSha256": image_sha,
            "perceptualDhash64": dhash,
            "sourceGroupId": canonical_group(identifier, row, docci_clusters),
            "origin": origin,
            "sourceManifest": source_manifest,
        }
        previous = items.get(identifier)
        if previous is not None and (
            previous["imageSha256"] != image_sha
            or previous["perceptualDhash64"] != dhash
            or previous["sourceGroupId"] != item["sourceGroupId"]
        ):
            raise ValueError(f"Conflicting historical evidence for {identifier}")
        items[identifier] = item

    for row in large_rows:
        add(row, str(row["perceptualDhash64"]), "large-train", exclusions["largeTrainManifestPath"])

    for relative, row in manifest_rows:
        identifier = str(row["id"])
        if identifier in items:
            if items[identifier]["imageSha256"] != str(row["imageSha256"]):
                raise ValueError(f"Historical manifest byte lock conflicts with large training: {identifier}")
            continue
        inline = row.get("perceptualDhash64")
        evidence = evaluation_dhash.get(identifier)
        dhash = str(inline) if inline else (evidence[1] if evidence else "")
        if evidence and evidence[0] != str(row["imageSha256"]):
            raise ValueError(f"Evaluation perceptual byte lock changed: {identifier}")
        if not dhash:
            raise ValueError(f"Historical manifest lacks perceptual evidence: {identifier}")
        add(row, dhash, "current-evaluation", relative)

    legacy_path = ROOT / exclusions["legacyExclusionsPath"]
    if digest(legacy_path) != exclusions["legacyExclusionsSha256"]:
        raise ValueError("Legacy evaluation exclusions changed")
    legacy = json.loads(legacy_path.read_text())
    legacy_ids = {str(value) for value in legacy["evaluationIds"]}
    legacy_hashes = {str(value) for value in legacy["evaluationImageSha256"]}
    if len(legacy_ids) != int(exclusions["legacyItems"]) or len(legacy_hashes) != int(exclusions["legacyItems"]):
        raise ValueError("Legacy evaluation exclusion cardinality changed")
    legacy_rows: dict[str, dict[str, Any]] = {}
    for manifest in exclusions["legacyManifests"]:
        path = ROOT / manifest["path"]
        compressed = path.read_bytes()
        if digest_bytes(compressed) != manifest["compressedSha256"]:
            raise ValueError(f"Legacy manifest compression changed: {manifest['path']}")
        expanded_manifest = gzip.decompress(compressed)
        if digest_bytes(expanded_manifest) != manifest["expandedSha256"]:
            raise ValueError(f"Legacy manifest expansion changed: {manifest['path']}")
        for line in expanded_manifest.splitlines():
            if not line:
                continue
            row = json.loads(line)
            identifier = str(row["id"])
            if identifier in legacy_rows:
                raise ValueError(f"Duplicate legacy evaluation ID: {identifier}")
            row["sourceManifest"] = manifest["sourcePath"]
            legacy_rows[identifier] = row
    if set(legacy_rows) != legacy_ids or {str(row["imageSha256"]) for row in legacy_rows.values()} != legacy_hashes:
        raise ValueError("Legacy source manifests do not match the public exclusion locks")
    expected_by_id = {identifier: str(row["imageSha256"]) for identifier, row in legacy_rows.items()}
    cache_root = require_safe_data_path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_root = require_safe_data_path(cache_root)
    recovered: dict[str, tuple[bytes, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                recover_one,
                identifier,
                image_sha,
                recipe,
                cache_root,
                allow_download=allow_download,
            ): identifier
            for identifier, image_sha in expected_by_id.items()
        }
        for future in as_completed(futures):
            identifier, value, url = future.result()
            recovered[identifier] = (value, url)
    if set(recovered) != set(expected_by_id):
        raise ValueError("Legacy recovery did not cover the frozen exclusion set")
    for identifier in sorted(recovered):
        value, url = recovered[identifier]
        row = legacy_rows[identifier]
        add(row, image_dhash(value), "legacy-evaluation", f"{row['sourceManifest']} @ {legacy['sourceCommit']} | {url}")

    if len(items) != int(exclusions["expectedUniqueItems"]):
        raise ValueError(f"Historical perceptual index has {len(items)} items, expected {exclusions['expectedUniqueItems']}")
    image_hashes = {str(row["imageSha256"]) for row in items.values()}
    if len(image_hashes) != len(items):
        raise ValueError("Historical perceptual index contains duplicate pixel bytes")
    output_items = [items[identifier] for identifier in sorted(items)]
    return {
        "schemaVersion": 1,
        "algorithm": recipe["overlapPolicy"]["perceptualDhash64"]["algorithm"],
        "maximumHammingDistance": recipe["overlapPolicy"]["perceptualDhash64"]["maximumHammingDistance"],
        "inputs": {
            "largeTrainManifestSha256": exclusions["largeTrainManifestSha256"],
            "evaluationPerceptualCompressedSha256": exclusions["evaluationPerceptualCompressedSha256"],
            "evaluationPerceptualExpandedSha256": exclusions["evaluationPerceptualExpandedSha256"],
            "legacyExclusionsSha256": exclusions["legacyExclusionsSha256"],
            "legacySourceCommit": legacy["sourceCommit"],
            "docciMetadataSha256": exclusions["docciMetadata"]["sha256"],
        },
        "counts": {
            "items": len(output_items),
            "largeTrain": sum(row["origin"] == "large-train" for row in output_items),
            "currentEvaluation": sum(row["origin"] == "current-evaluation" for row in output_items),
            "legacyEvaluation": sum(row["origin"] == "legacy-evaluation" for row in output_items),
        },
        "items": output_items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("benchmark/data/replacement-v2/source/legacy-evaluation"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/manifests/historical-perceptual-exclusions-v1.json.gz"),
    )
    parser.add_argument("--download-workers", type=int, default=16)
    parser.add_argument("--offline", action="store_true", help="Require all pinned source bytes to exist locally")
    args = parser.parse_args()
    if not 1 <= args.download_workers <= 32:
        raise ValueError("Download workers must be in [1, 32]")
    recipe = json.loads(RECIPE_PATH.read_text())
    cache_root = require_safe_data_path(ROOT / args.cache_root)
    packet = build_index(recipe, cache_root, args.download_workers, allow_download=not args.offline)
    expanded = canonical_json(packet)
    compressed = deterministic_gzip(expanded)
    output = (ROOT / args.output).absolute()
    try:
        output.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("Historical index output escapes the repository") from error
    atomic_write(output, compressed)
    if gzip.decompress(output.read_bytes()) != expanded:
        raise ValueError("Historical perceptual index gzip did not round-trip")
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "items": packet["counts"]["items"],
                "expandedSha256": digest_bytes(expanded),
                "compressedSha256": digest_bytes(compressed),
                "compressedBytes": len(compressed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
