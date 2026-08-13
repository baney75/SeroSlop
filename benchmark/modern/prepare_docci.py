"""Add a deterministic CC BY 4.0 DOCCI-train real-image stratum.

The official release is a single gzip archive, so this script keeps the archive
locally but extracts only the selected training members. DOCCI test images are
not eligible; the frontier regression audit uses that held-out split.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request


DATA_ROOT = Path("benchmark/data/modern-head")
SOURCE_ROOT = DATA_ROOT / "source" / "docci"
DESCRIPTIONS = SOURCE_ROOT / "docci_descriptions.jsonlines"
ARCHIVE = SOURCE_ROOT / "docci_images.tar.gz"
TRAIN_MANIFEST = DATA_ROOT / "train-manifest.jsonl"
SELECTION = DATA_ROOT / "selection.json"
AUDIT_MANIFEST = Path("benchmark/data/frontier-original/manifest.jsonl")
REVISION = "a0a43eaf34676ffd008fb6565dd8c2ba00d09100"
TARGET = 1200
DESCRIPTIONS_URL = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"
DESCRIPTIONS_SHA256 = "c9df4819963883af35ddd2cf257949892fd8c6d88b33a012094352df60719800"
ARCHIVE_URL = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"
ARCHIVE_BYTES = 7_592_938_768
ARCHIVE_MD5 = "57493de5075fe508d51e77500748f4da"


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download(url: str, output: Path, expected_bytes: int | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    if expected_bytes is not None and existing > expected_bytes:
        raise ValueError(f"Partial download is larger than expected: {partial}")
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("ab" if existing else "wb") as handle:
        completed = existing
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            completed += len(chunk)
            if completed // (128 * 1024 * 1024) != (completed - len(chunk)) // (128 * 1024 * 1024):
                total = expected_bytes or completed
                print(f"Downloaded {completed / 1024**3:.2f}/{total / 1024**3:.2f} GiB", flush=True)
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        raise ValueError(f"Downloaded {partial.stat().st_size} bytes; expected {expected_bytes}")
    partial.replace(output)


def json_lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    if not DESCRIPTIONS.exists():
        download(DESCRIPTIONS_URL, DESCRIPTIONS)
    if file_digest(DESCRIPTIONS) != DESCRIPTIONS_SHA256:
        raise ValueError("DOCCI descriptions failed SHA-256 verification")
    descriptions = [
        row for row in json_lines(DESCRIPTIONS)
        if row["split"] == "train" and str(row["example_id"]).startswith("train")
    ]
    candidates = sorted(
        descriptions,
        key=lambda row: hashlib.sha256(f"{REVISION}:{row['example_id']}".encode()).hexdigest(),
    )
    selected = candidates[:TARGET]
    if len(selected) != TARGET:
        raise ValueError(f"Selected only {len(selected)} DOCCI training images")
    wanted = {str(row["image_file"]): row for row in selected}

    if not ARCHIVE.exists():
        download(ARCHIVE_URL, ARCHIVE, ARCHIVE_BYTES)
    if ARCHIVE.stat().st_size != ARCHIVE_BYTES or file_digest(ARCHIVE, "md5") != ARCHIVE_MD5:
        raise ValueError("DOCCI archive failed size or MD5 verification")

    output_root = DATA_ROOT / "train" / "real" / "docci"
    output_root.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            basename = Path(member.name).name
            if basename not in wanted or not member.isfile():
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Could not extract {member.name}")
            with (output_root / basename).open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.add(basename)
            if len(extracted) % 100 == 0:
                print(f"Extracted DOCCI {len(extracted)}/{TARGET}", flush=True)
    missing = set(wanted) - extracted
    if missing:
        raise ValueError(f"DOCCI archive is missing {len(missing)} selected images")

    audit_hashes = {str(row["imageSha256"]) for row in json_lines(AUDIT_MANIFEST)} if AUDIT_MANIFEST.exists() else set()
    existing = [row for row in json_lines(TRAIN_MANIFEST) if row["dataset"] != "google/docci"]
    additions: list[dict[str, object]] = []
    for index, row in enumerate(sorted(selected, key=lambda item: str(item["example_id"]))):
        filename = str(row["image_file"])
        relative_path = f"train/real/docci/{filename}"
        image_hash = file_digest(DATA_ROOT / relative_path)
        if image_hash in audit_hashes:
            raise ValueError(f"Frontier audit leakage detected for DOCCI {row['example_id']}")
        additions.append({
            "id": f"docci:{REVISION}:train:{row['example_id']}",
            "dataset": "google/docci",
            "datasetRevision": REVISION,
            "split": "train",
            "rowIndex": index,
            "path": relative_path,
            "imageSha256": image_hash,
            "label": 0,
            "source": "docci-train",
        })
    combined = sorted(existing + additions, key=lambda row: str(row["id"]))
    TRAIN_MANIFEST.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in combined) + "\n")

    selection = json.loads(SELECTION.read_text())
    selection["docci"] = {
        "dataset": "google/docci",
        "revision": REVISION,
        "split": "train",
        "license": "CC-BY-4.0",
        "target": TARGET,
        "strategy": "lowest SHA-256 priorities over official train example IDs",
        "descriptionsSha256": DESCRIPTIONS_SHA256,
        "archiveBytes": ARCHIVE_BYTES,
        "archiveMd5": ARCHIVE_MD5,
    }
    selection["counts"]["train"] = len(combined)
    selection["counts"]["real"] = (
        sum(int(row["label"]) == 0 for row in combined)
        + int(selection["counts"]["validation"]) // 2
        + int(selection["counts"].get("test", 0)) // 2
    )
    SELECTION.write_text(json.dumps(selection, indent=2) + "\n")
    (DATA_ROOT / "docci-attribution.json").write_text(json.dumps({
        "dataset": "DOCCI: Descriptions of Connected and Contrasting Images",
        "revision": REVISION,
        "license": "CC BY 4.0",
        "homepage": "https://google.github.io/docci/",
        "citation": "Onoe et al., DOCCI: Descriptions of Connected and Contrasting Images, ECCV 2024",
        "selectedExampleIds": sorted(str(row["example_id"]) for row in selected),
    }, indent=2) + "\n")
    print(f"Added {len(additions)} DOCCI train images; train manifest now has {len(combined)} items", flush=True)


if __name__ == "__main__":
    main()
