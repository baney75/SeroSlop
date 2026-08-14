"""Freeze and materialize the evaluation-only web-negative challenge.

The Library of Congress catalog is consulted only during explicit planning.
Every later run consumes exact catalog IDs and image hashes from the committed,
pixel-free plan. Dataset pixels remain local and outside Git.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageOps
import pyarrow.parquet as parquet


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLAN_PATH = REPOSITORY_ROOT / "benchmark/manifests/web-negative-plan.json"
REVIEW_PATH = REPOSITORY_ROOT / "benchmark/manifests/web-negative-review.json"


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_bytes(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return (json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n").encode()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_bytes(data)
    partial.replace(path)


def priority(seed: int, value: str) -> str:
    return sha256(f"{seed}:{value}".encode()).hexdigest()


def request_json(url: str, attempts: int = 8) -> dict[str, Any]:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ProofLens/1.0 (https://github.com/baney75/prooflens)"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read())
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def bounded_download(url: str, maximum: int, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "ProofLens/1.0 (https://github.com/baney75/prooflens)"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > maximum:
                    raise ValueError(f"response exceeds {maximum} bytes")
                data = response.read(maximum + 1)
            if len(data) > maximum:
                raise ValueError(f"response exceeds {maximum} bytes")
            return data
        except ValueError:
            raise
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def image_facts(data: bytes, minimum: int) -> tuple[int, int, str]:
    with Image.open(BytesIO(data)) as opened:
        opened.verify()
    with Image.open(BytesIO(data)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        resized = image.resize((9, 8), Image.Resampling.LANCZOS).convert("L")
        pixels = list(resized.getdata())
    if min(width, height) < minimum:
        raise ValueError("image is below the minimum short edge")
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return int(width), int(height), f"{bits:016x}"


def best_loc_image_url(row: dict[str, Any]) -> str:
    urls = [str(value).split("#", 1)[0] for value in row.get("image_url", [])]
    jpegs = [value for value in urls if value.lower().endswith((".jpg", ".jpeg"))]
    if not jpegs:
        raise ValueError("catalog item has no JPEG image")
    return jpegs[-1]


def loc_candidates(recipe: dict[str, Any]) -> list[dict[str, object]]:
    config = recipe["libraryOfCongress"]
    query = urllib.parse.urlencode({"fo": "json", "c": config["catalogPageSize"], "at": "results,pagination"})
    api_url = f"{config['collectionUrl']}?{query}"
    catalog = request_json(api_url)
    results = catalog.get("results", [])
    if len(results) < int(config["targetCount"]):
        raise ValueError("Library of Congress collection returned too few catalog rows")
    candidates: list[dict[str, object]] = []
    for row in results:
        item = row.get("item") or {}
        control_number = str(item.get("control_number", ""))
        year_text = str(row.get("date", ""))[:4]
        rights = str(item.get("rights_advisory", ""))
        part_of = str(item.get("part_of", ""))
        if (
            not control_number
            or not year_text.isdigit()
            or int(year_text) > int(config["latestCreationYear"])
            or rights not in set(config["acceptedRightsAdvisories"])
            or config["collectionTitle"] not in part_of
        ):
            continue
        try:
            image_url = best_loc_image_url(row)
        except ValueError:
            continue
        candidates.append(
            {
                "controlNumber": control_number,
                "catalogId": str(row["id"]),
                "catalogUrl": str(row["url"]),
                "resourceUrls": [str(value) for value in item.get("resource_links", [])],
                "imageUrl": image_url,
                "title": str(row.get("title", "")),
                "date": str(row.get("date", "")),
                "createdPublished": str(item.get("created_published", "")),
                "photographers": [str(value) for value in item.get("contributors", [])],
                "rightsAdvisory": rights,
                "rightsSource": config["rightsSource"],
                "rightsStatus": config["rightsStatus"],
                "callNumber": str(item.get("call_number", "")),
                "reproductionNumber": str(item.get("reproduction_number", "")),
                "medium": [str(value) for value in item.get("medium", [])],
                "subjects": [str(value) for value in item.get("subjects", [])],
                "locations": [str(value) for value in row.get("location", [])],
                "description": [str(value) for value in row.get("description", [])],
                "catalogExtractTimestamp": str(row.get("extract_timestamp", "")),
                "selectionPriority": priority(int(recipe["seed"]), control_number),
                "nonGenerativeEvidence": {
                    "kind": "documented historical color transparency",
                    "latestCreationYear": int(config["latestCreationYear"]),
                    "catalogDate": str(row.get("date", "")),
                    "collection": config["collectionTitle"],
                },
            }
        )
    return sorted(candidates, key=lambda row: (str(row["selectionPriority"]), str(row["controlNumber"])))


def fetch_loc_candidate(
    row: dict[str, object], data_root: Path, maximum: int, minimum: int
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    try:
        relative = Path("library-of-congress") / f"{row['controlNumber']}.jpg"
        destination = data_root / relative
        data = destination.read_bytes() if destination.exists() else bounded_download(str(row["imageUrl"]), maximum)
        width, height, perceptual_hash = image_facts(data, minimum)
        if not destination.exists():
            write_atomic(destination, data)
        return (
            {
                **row,
                "path": relative.as_posix(),
                "imageSha256": sha256(data).hexdigest(),
                "imageBytes": len(data),
                "width": width,
                "height": height,
                "perceptualDhash64": perceptual_hash,
            },
            None,
        )
    except (OSError, urllib.error.URLError, ValueError) as error:
        return None, {"controlNumber": row.get("controlNumber"), "reason": str(error)}


def build_plan(recipe_path: Path, recipe: dict[str, Any], data_root: Path) -> dict[str, object]:
    config = recipe["libraryOfCongress"]
    candidates = loc_candidates(recipe)
    selected: list[dict[str, object]] = []
    rejects: list[dict[str, object]] = []
    batch_size = int(config["downloadBatchSize"])
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        with ThreadPoolExecutor(max_workers=int(config["maximumDownloadWorkers"])) as executor:
            results = list(
                executor.map(
                    lambda row: fetch_loc_candidate(
                        row,
                        data_root,
                        int(recipe["maximumImageBytes"]),
                        int(config["minimumShortEdge"]),
                    ),
                    batch,
                )
            )
        for item, rejection in results:
            if item is not None:
                selected.append(item)
            if rejection is not None:
                rejects.append(rejection)
        print(f"Library of Congress plan: {len(selected)}/{config['targetCount']} valid", flush=True)
        if len(selected) >= int(config["targetCount"]):
            break
    selected = selected[: int(config["targetCount"])]
    if len(selected) != int(config["targetCount"]):
        raise ValueError(f"Only {len(selected)} deterministic historical photographs were usable")
    plan = {
        "schemaVersion": 1,
        "recipeSha256": digest(recipe_path),
        "selectionSource": {
            "collectionUrl": config["collectionUrl"],
            "collectionTitle": config["collectionTitle"],
            "rightsSource": config["rightsSource"],
        },
        "items": selected,
        "rejections": rejects,
    }
    write_atomic(PLAN_PATH, json_bytes(plan, pretty=True))
    review = {
        "schemaVersion": 1,
        "planSha256": digest(PLAN_PATH),
        "reviewMethod": "Row-level visual inspection backed by the catalog's 1939-1944 color-transparency provenance",
        "items": [
            {"controlNumber": row["controlNumber"], "decision": "pending", "note": ""}
            for row in selected
        ],
    }
    write_atomic(REVIEW_PATH, json_bytes(review, pretty=True))
    print(f"wrote {PLAN_PATH} ({digest(PLAN_PATH)})", flush=True)
    print(f"wrote pending review template {REVIEW_PATH}", flush=True)
    return plan


def validate_plan(recipe_path: Path, recipe: dict[str, Any], plan: dict[str, Any]) -> None:
    if plan.get("schemaVersion") != 1 or plan.get("recipeSha256") != digest(recipe_path):
        raise ValueError("Web-negative plan does not match the frozen recipe")
    items = plan.get("items")
    if not isinstance(items, list) or len(items) != int(recipe["libraryOfCongress"]["targetCount"]):
        raise ValueError("Web-negative plan count mismatch")
    ids = [str(row["controlNumber"]) for row in items]
    hashes = [str(row["imageSha256"]) for row in items]
    paths = [str(row["path"]) for row in items]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes) or len(set(paths)) != len(paths):
        raise ValueError("Web-negative plan contains duplicate IDs, bytes, or paths")
    config = recipe["libraryOfCongress"]
    if any(
        int(str(row["date"])[:4]) > int(config["latestCreationYear"])
        or row["rightsAdvisory"] not in set(config["acceptedRightsAdvisories"])
        for row in items
    ):
        raise ValueError("Web-negative plan violates the historical-photo or rights boundary")


def load_approved_review(plan: dict[str, Any]) -> dict[str, dict[str, object]]:
    review = json.loads(REVIEW_PATH.read_text())
    if review.get("schemaVersion") != 1 or review.get("planSha256") != digest(PLAN_PATH):
        raise ValueError("Web-negative visual review is not bound to the frozen plan")
    rows = review.get("items")
    if not isinstance(rows, list):
        raise ValueError("Web-negative review is missing item decisions")
    by_id = {str(row["controlNumber"]): row for row in rows}
    expected = {str(row["controlNumber"]) for row in plan["items"]}
    if set(by_id) != expected or any(row.get("decision") != "include" for row in rows):
        raise ValueError("Every frozen historical photograph requires an explicit include decision")
    return by_id


def materialize_loc(
    recipe: dict[str, Any], plan: dict[str, Any], data_root: Path
) -> list[dict[str, object]]:
    review = load_approved_review(plan)
    output: list[dict[str, object]] = []
    for row in plan["items"]:
        destination = data_root / str(row["path"])
        data = destination.read_bytes() if destination.exists() else bounded_download(
            str(row["imageUrl"]), int(recipe["maximumImageBytes"])
        )
        if sha256(data).hexdigest() != row["imageSha256"] or len(data) != int(row["imageBytes"]):
            raise ValueError(f"Library of Congress image identity mismatch: {row['controlNumber']}")
        width, height, perceptual_hash = image_facts(
            data, int(recipe["libraryOfCongress"]["minimumShortEdge"])
        )
        if perceptual_hash != row["perceptualDhash64"]:
            raise ValueError(f"Library of Congress perceptual hash mismatch: {row['controlNumber']}")
        if not destination.exists():
            write_atomic(destination, data)
        output.append(
            {
                "id": f"loc-fsa-owi-color:{row['controlNumber']}",
                "dataset": recipe["libraryOfCongress"]["collectionTitle"],
                "datasetRevision": str(row["catalogExtractTimestamp"]),
                "split": "test",
                "path": str(row["path"]),
                "imageSha256": str(row["imageSha256"]),
                "perceptualDhash64": str(row["perceptualDhash64"]),
                "label": 0,
                "source": "library-of-congress-fsa-owi-color",
                "groupId": str(row["controlNumber"]),
                "width": width,
                "height": height,
                "attribution": {
                    key: value for key, value in row.items()
                    if key not in {"imageUrl", "selectionPriority"}
                },
                "humanReview": review[str(row["controlNumber"])],
            }
        )
    return output


def chartography_rows(recipe: dict[str, Any], root: Path) -> list[dict[str, object]]:
    config = recipe["chartography"]
    metadata_url = f"https://huggingface.co/datasets/{config['dataset']}/resolve/{config['revision']}/data.parquet"
    metadata_path = root / "chartography-metadata.parquet"
    if not metadata_path.exists():
        write_atomic(metadata_path, bounded_download(metadata_url, 1_000_000))
    if digest(metadata_path) != config["metadataSha256"]:
        raise ValueError("Chartography metadata SHA-256 mismatch")
    selected = [
        row for row in parquet.read_table(metadata_path).to_pylist()
        if row["source_type"] == config["sourceType"]
    ]
    if len(selected) != int(config["expectedCount"]):
        raise ValueError("Unexpected Chartography expert-created row count")
    relative_paths = [str(row["chart_path"]) for row in selected]
    basenames = [Path(value).name for value in relative_paths]
    if len(set(relative_paths)) != len(relative_paths) or len(set(basenames)) != len(basenames):
        raise ValueError("Chartography selection contains duplicate source paths or basenames")
    output: list[dict[str, object]] = []
    for row in sorted(selected, key=lambda value: value["task_id"]):
        relative_source = str(row["chart_path"])
        url = (
            f"https://huggingface.co/datasets/{config['dataset']}/resolve/"
            f"{config['revision']}/{urllib.parse.quote(relative_source)}"
        )
        relative = Path("chartography") / Path(relative_source).name
        destination = root / relative
        data = destination.read_bytes() if destination.exists() else bounded_download(
            url, int(recipe["maximumImageBytes"])
        )
        width, height, perceptual_hash = image_facts(data, 64)
        if not destination.exists():
            write_atomic(destination, data)
        output.append(
            {
                "id": f"chartography:{config['revision']}:{row['task_id']}",
                "dataset": config["dataset"],
                "datasetRevision": config["revision"],
                "split": "web-negative-challenge",
                "path": relative.as_posix(),
                "imageSha256": sha256(data).hexdigest(),
                "perceptualDhash64": perceptual_hash,
                "label": 0,
                "source": "chartography-expert-created",
                "groupId": str(row["task_id"]),
                "width": width,
                "height": height,
                "license": config["license"],
                "domain": row["domain_combined"],
            }
        )
    return output


def write_final_manifests(
    recipe_path: Path,
    recipe: dict[str, Any],
    data_root: Path,
    plan: dict[str, Any],
    historical: list[dict[str, object]],
    charts: list[dict[str, object]],
) -> None:
    web_rows = sorted([*historical, *charts], key=lambda row: str(row["id"]))
    if len(web_rows) != int(recipe["expectedTotalCount"]):
        raise ValueError("Web-negative challenge count mismatch")
    if len({str(row["imageSha256"]) for row in web_rows}) != len(web_rows):
        raise ValueError("Web-negative challenge contains duplicate image bytes")
    for index, row in enumerate(web_rows):
        row["rowIndex"] = index
    manifest = data_root / "manifest.jsonl"
    write_atomic(manifest, b"".join(json_bytes(row) for row in web_rows))

    synthetic_path = REPOSITORY_ROOT / str(recipe["confirmatoryTest"]["syntheticManifest"])
    synthetic = [json.loads(line) for line in synthetic_path.read_text().splitlines() if line]
    if len(synthetic) != int(recipe["confirmatoryTest"]["syntheticCount"]):
        raise ValueError("Confirmatory synthetic manifest count mismatch")
    confirmatory = [
        {**row, "path": f"modern-head/{row['path']}", "split": "test"}
        for row in synthetic
    ] + [
        {**row, "path": f"web-negative/{row['path']}", "split": "test"}
        for row in historical
    ]
    if len(confirmatory) != int(recipe["confirmatoryTest"]["expectedCount"]):
        raise ValueError("Confirmatory test count mismatch")
    confirmatory.sort(key=lambda row: str(row["id"]))
    if len({str(row["imageSha256"]) for row in confirmatory}) != len(confirmatory):
        raise ValueError("Confirmatory test contains duplicate image bytes")
    for index, row in enumerate(confirmatory):
        row["rowIndex"] = index
    confirmatory_path = data_root.parent / "confirmatory-test-manifest.jsonl"
    write_atomic(confirmatory_path, b"".join(json_bytes(row) for row in confirmatory))

    summary = {
        "schemaVersion": 1,
        "recipeSha256": digest(recipe_path),
        "planSha256": digest(PLAN_PATH),
        "reviewSha256": digest(REVIEW_PATH),
        "manifestSha256": digest(manifest),
        "confirmatoryTestManifestSha256": digest(confirmatory_path),
        "counts": {
            "libraryOfCongressHistoricalPhotos": len(historical),
            "chartographyExpertCreated": len(charts),
            "webNegativeTotal": len(web_rows),
            "confirmatorySynthetic": len(synthetic),
            "confirmatoryReal": len(historical),
            "confirmatoryTotal": len(confirmatory),
        },
        "planRecipeSha256": plan["recipeSha256"],
    }
    write_atomic(data_root / "summary.json", json_bytes(summary, pretty=True))
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, default=Path("benchmark/web-negative/recipe.json"))
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data/web-negative"))
    parser.add_argument("--phase", choices=("plan", "materialize", "all"), default="all")
    parser.add_argument("--replan", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    args.data_root.mkdir(parents=True, exist_ok=True)
    plan: dict[str, Any]
    if args.phase in {"plan", "all"}:
        if PLAN_PATH.exists() and not args.replan:
            plan = json.loads(PLAN_PATH.read_text())
            validate_plan(args.recipe, recipe, plan)
            print(f"reused {PLAN_PATH} ({digest(PLAN_PATH)})", flush=True)
        else:
            plan = build_plan(args.recipe, recipe, args.data_root)
            validate_plan(args.recipe, recipe, plan)
    else:
        if not PLAN_PATH.exists():
            raise FileNotFoundError("Create the frozen web-negative plan before materialization")
        plan = json.loads(PLAN_PATH.read_text())
        validate_plan(args.recipe, recipe, plan)
    if args.phase in {"materialize", "all"}:
        historical = materialize_loc(recipe, plan, args.data_root)
        charts = chartography_rows(recipe, args.data_root)
        write_final_manifests(args.recipe, recipe, args.data_root, plan, historical, charts)


if __name__ == "__main__":
    main()
