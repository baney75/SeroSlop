"""Build the exhaustive modern-training/evaluation dHash review packet.

The output contains no pixels. Existing human decisions are retained only when
every bound ID, byte hash, dHash, path, manifest, and Hamming distance is still
identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import prepare


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, default=Path("benchmark/large/recipe.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/manifests/training-evaluation-perceptual-review.json"),
    )
    parser.add_argument(
        "--modern-data-root", type=Path, default=Path("benchmark/data/modern-head")
    )
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    threshold = int(recipe["perceptualDuplicateHammingThreshold"])
    _, _, evaluation_index, _, _ = prepare.load_evaluation_exclusions(recipe)

    exclusion_manifests = [
        *recipe["evaluationManifests"],
        *recipe.get("additionalTrainingExclusionManifests", []),
    ]
    evaluation_by_id: dict[str, dict[str, str]] = {}
    exclusion_hashes: dict[str, str] = {}
    for manifest in exclusion_manifests:
        relative = str(manifest["path"])
        manifest_path = prepare.REPOSITORY_ROOT / relative
        data_root = prepare.REPOSITORY_ROOT / str(manifest["dataRoot"])
        exclusion_hashes[relative] = prepare.digest(manifest_path)
        for row in manifest_rows(manifest_path):
            data = (data_root / str(row["path"])).read_bytes()
            if prepare.sha256(data).hexdigest() != str(row["imageSha256"]):
                raise ValueError(f"Evaluation image integrity mismatch: {row['id']}")
            _, _, dhash = prepare.validate_image(data, 1)
            evaluation_by_id[str(row["id"])] = {
                "evaluationId": str(row["id"]),
                "evaluationImageSha256": str(row["imageSha256"]),
                "evaluationPerceptualDhash64": dhash,
                "evaluationManifest": relative,
                "evaluationDataRoot": str(manifest["dataRoot"]),
                "evaluationPath": str(row["path"]),
            }

    training_manifest = prepare.REPOSITORY_ROOT / str(recipe["modernTrainingManifest"])
    previous: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.exists():
        previous_packet = json.loads(args.output.read_text())
        previous = {
            (str(row["trainingId"]), str(row["evaluationId"])): row
            for row in previous_packet.get("items", [])
        }

    items: list[dict[str, Any]] = []
    immutable_keys = {
        "trainingId",
        "trainingImageSha256",
        "trainingPerceptualDhash64",
        "trainingPath",
        "evaluationId",
        "evaluationImageSha256",
        "evaluationPerceptualDhash64",
        "evaluationManifest",
        "evaluationDataRoot",
        "evaluationPath",
        "hammingDistance",
    }
    for row in manifest_rows(training_manifest):
        data = (args.modern_data_root / str(row["path"])).read_bytes()
        if prepare.sha256(data).hexdigest() != str(row["imageSha256"]):
            raise ValueError(f"Modern training image integrity mismatch: {row['id']}")
        _, _, dhash = prepare.validate_image(data, 1)
        for match in prepare.perceptual_matches(dhash, evaluation_index, threshold):
            evaluation = evaluation_by_id[str(match["id"])]
            item: dict[str, Any] = {
                "trainingId": str(row["id"]),
                "trainingImageSha256": str(row["imageSha256"]),
                "trainingPerceptualDhash64": dhash,
                "trainingPath": str(row["path"]),
                **evaluation,
                "hammingDistance": int(match["hammingDistance"]),
            }
            old = previous.get((item["trainingId"], item["evaluationId"]))
            if old is not None and all(old.get(key) == item.get(key) for key in immutable_keys):
                item.update(
                    {
                        "decision": old.get("decision", "pending"),
                        "reviewedAt": old.get("reviewedAt", ""),
                        "reviewer": old.get("reviewer", ""),
                        "rationale": old.get("rationale", ""),
                    }
                )
            else:
                item.update(
                    {"decision": "pending", "reviewedAt": "", "reviewer": "", "rationale": ""}
                )
            items.append(item)

    items.sort(key=lambda row: (row["trainingId"], row["evaluationId"]))
    packet = {
        "schemaVersion": 1,
        "hammingThreshold": threshold,
        "trainingManifest": str(recipe["modernTrainingManifest"]),
        "trainingManifestSha256": prepare.digest(training_manifest),
        "evaluationExclusionSha256ByPath": exclusion_hashes,
        "reviewMethod": "Direct side-by-side visual inspection of every dHash candidate pair",
        "items": items,
    }
    prepare.write_atomic(args.output, prepare.json_bytes(packet, pretty=True))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": prepare.digest(args.output),
                "pairs": len(items),
                "pending": sum(row["decision"] == "pending" for row in items),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
