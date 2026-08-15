"""Pure, dependency-light contracts for the score-blind M4 protocol."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable


HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
MODELS = ("dalle-3", "flux", "mj", "stable_diffusion")
VARIANTS = ("original", "screenshot", "social-q75", "social-heavy")
ADAPTER_WEIGHT_DECAYS = (0.003, 0.01, 0.03)
ADAPTER_ANCHOR_COEFFICIENTS = (0.01, 0.03, 0.1, 0.3)
ALLOWED_BRITISH_DECADES = ("1800", "1820", "1830", "1840", "1860", "1870", "1880", "1890")
SOURCE_LOCKS_RAW_SHA256 = "bf44ceba6f32d322de04f9fae994c0fed7fdcd00e2bcfff9de39c6d852a01394"
SOURCE_LOCKS_CANONICAL_SHA256 = "095e06677cc77e82e5ed3777f80e9a21585efe401f6f5c7f658372c1987da622"
RECIPE_RAW_SHA256 = "344ced4ee8e68325bd0217391e4e5745d554b8140586b51d7aa98ef6bb441b34"
RECIPE_CANONICAL_SHA256 = "c567578f254bc8c4793eac9f413354caaac5153a6d1897e8ad2f096ced7d9668"


def canonical_json(value: object, *, pretty: bool = False) -> bytes:
    separators = None if pretty else (",", ":")
    payload = json.dumps(
        value, indent=2 if pretty else None, separators=separators, sort_keys=True, allow_nan=False,
    )
    return (payload + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def priority(namespace: str, value: str) -> str:
    return digest_bytes(f"{namespace}{value}".encode("utf-8"))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"M4 JSON contains a duplicate key: {key}")
        value[key] = item
    return value


def parse_json_bytes(value: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = value.decode("utf-8", errors="strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"M4 {label} contains non-finite JSON: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"M4 {label} is not strict UTF-8 JSON") from error
    except ValueError as error:
        raise ValueError(f"M4 {label} is invalid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"M4 {label} must be a JSON object")
    return parsed


def load_frozen_protocol(recipe_path: Path, locks_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    recipe_bytes = recipe_path.read_bytes()
    locks_bytes = locks_path.read_bytes()
    if digest_bytes(recipe_bytes) != RECIPE_RAW_SHA256:
        raise ValueError("M4 recipe bytes changed")
    if digest_bytes(locks_bytes) != SOURCE_LOCKS_RAW_SHA256:
        raise ValueError("M4 source-lock bytes changed")
    recipe = parse_json_bytes(recipe_bytes, label="recipe")
    locks = parse_json_bytes(locks_bytes, label="source locks")
    validate_recipe(recipe, locks)
    return recipe, locks


def british_book_id(fname: str) -> str:
    identifier = fname.split("_", 1)[0]
    if not identifier.isdigit() or len(identifier) < 6:
        raise ValueError(f"British Library filename has no valid system number: {fname}")
    return identifier


def british_decade(value: str) -> str:
    try:
        year = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"British Library date is not an integer: {value}") from error
    decade = str((year // 10) * 10)
    if decade not in ALLOWED_BRITISH_DECADES:
        raise ValueError(f"British Library date is outside the frozen decade strata: {year}")
    return decade


def classify_british_date(value: object) -> tuple[str | None, str]:
    """Return the frozen decade or an auditable raw-value rejection key."""
    raw = "null" if value is None else str(value)
    if isinstance(value, bool):
        return None, raw
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None, raw
    decade = str((year // 10) * 10)
    return (decade, raw) if decade in ALLOWED_BRITISH_DECADES else (None, raw)


def prompt_group_id(prompt: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Rapidata prompt is empty")
    return digest_bytes(prompt.encode("utf-8"))


def validate_source_locks(value: dict[str, Any]) -> None:
    if digest_bytes(canonical_json(value)) != SOURCE_LOCKS_CANONICAL_SHA256:
        raise ValueError("M4 source-lock packet changed")
    if set(value) != {"schemaVersion", "britishLibrary", "rapidata"} or value["schemaVersion"] != 1:
        raise ValueError("M4 source-lock top-level schema changed")
    british = value["britishLibrary"]
    rapidata = value["rapidata"]
    if set(british) != {
        "dataset", "revision", "repositoryFileCount", "repositoryBytes", "selectedShardCount",
        "repositoryInventoryEndpoint", "repositoryInventoryCanonicalSha256", "selectedShardBytes",
        "selectedFilesCanonicalSha256", "selection", "files",
    }:
        raise ValueError("British Library source-lock schema changed")
    if set(rapidata) != {
        "dataset", "revision", "fileCount", "bytes", "repositoryInventoryEndpoint",
        "repositoryInventoryCanonicalSha256", "files",
    }:
        raise ValueError("Rapidata source-lock schema changed")
    if british["dataset"] != "biglam/british-library-book-images" or not HEX40.fullmatch(british["revision"]):
        raise ValueError("British Library source identity changed")
    if rapidata["dataset"] != "Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3" or not HEX40.fullmatch(rapidata["revision"]):
        raise ValueError("Rapidata source identity changed")
    expected = ((british, 32, 20_742_409_321), (rapidata, 26, 1_989_350_951))
    for packet, count, size in expected:
        files = packet["files"]
        if len(files) != count or len({row["path"] for row in files}) != count:
            raise ValueError("M4 source-lock file count or uniqueness changed")
        if sum(int(row["bytes"]) for row in files) != size:
            raise ValueError("M4 source-lock byte total changed")
        for row in files:
            if set(row) != {"path", "bytes", "sha256", "gitOid"}:
                raise ValueError("M4 source-lock file schema changed")
            if not isinstance(row["bytes"], int) or row["bytes"] <= 0:
                raise ValueError("M4 source-lock file size is invalid")
            if not HEX64.fullmatch(row["sha256"]) or not HEX40.fullmatch(row["gitOid"]):
                raise ValueError("M4 source-lock digest is invalid")
    if british["selectedShardCount"] != 32 or british["selectedShardBytes"] != 20_742_409_321:
        raise ValueError("British Library selected-shard totals changed")
    if (
        british["repositoryFileCount"] != 647
        or british["repositoryBytes"] != 424_911_645_071
        or british["repositoryInventoryCanonicalSha256"]
        != "efe0ab238ba74f03be93e8cbcfa4ce65f51f0ec2fe2452290113590276e1cb1b"
        or british["selectedFilesCanonicalSha256"]
        != "f270cdcb15dc5395d8eecd860f0995fc559780e0578e03ff98f1148bda4739bc"
        or british["selection"] != "lowest SHA-256 of prooflens:m4:british-library-shard-v1:<path>"
    ):
        raise ValueError("British Library repository-selection proof changed")
    if rapidata["fileCount"] != 26 or rapidata["bytes"] != 1_989_350_951:
        raise ValueError("Rapidata totals changed")
    if rapidata["repositoryInventoryCanonicalSha256"] != "c9253532f715fb78ac61186de6747fc2c63126e1363f7051f85a18efc8211b79":
        raise ValueError("Rapidata repository inventory changed")


def validate_recipe(recipe: dict[str, Any], locks: dict[str, Any]) -> None:
    validate_source_locks(locks)
    if digest_bytes(canonical_json(recipe)) != RECIPE_CANONICAL_SHA256:
        raise ValueError("M4 recipe changed")
    if recipe.get("schemaVersion") != 1 or recipe.get("name") != "prooflens-m4-residual-adapter":
        raise ValueError("M4 recipe identity changed")
    if recipe.get("baseCommit") != "439b2481dc88a887f8317be669096495760fbeb1":
        raise ValueError("M4 base commit changed")
    if recipe.get("baseTree") != "440931a595c87ca3d293f5a6f980c75169ddb899":
        raise ValueError("M4 base tree changed")
    if recipe.get("seed") != 20260815 or recipe.get("sourceLocksSha256") != SOURCE_LOCKS_RAW_SHA256:
        raise ValueError("M4 seed or source-lock binding changed")
    if recipe["upstreamModel"] != {
        "path": "weights/prooflens-cf384.onnx",
        "sha256": "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47",
        "bytes": 87_442_080,
        "modelLock": "model-lock.json",
        "modelLockSha256": "2a818b7b2582bc9614f02f178d9af997f46628734ea078a79415c3d68d3061f0",
    }:
        raise ValueError("M4 upstream model binding changed")
    if recipe["expectedTraining"] != {
        "images": 112_698,
        "featureViews": 150_792,
        "classCounts": {"real": 59_578, "synthetic": 53_120},
        "newSourceCounts": {
            "british-library-plates": 2_400,
            "rapidata-dalle-3": 480,
            "rapidata-flux": 480,
            "rapidata-midjourney": 480,
            "rapidata-stable-diffusion": 480,
        },
        "singleViewSources": ["diffusiondb-stable-diffusion", "open-images-train"],
    }:
        raise ValueError("M4 expected training composition changed")
    training = recipe["training"]
    if tuple(training["weightDecays"]) != ADAPTER_WEIGHT_DECAYS or tuple(training["anchorCoefficients"]) != ADAPTER_ANCHOR_COEFFICIENTS:
        raise ValueError("M4 candidate grid changed")
    if (
        training["candidateCount"] != 12
        or training["epochs"] != 12
        or training["batchSize"] != 2048
        or training["learningRate"] != 0.003
        or training["executionProvider"] != "cpu"
        or training["freshFeatureExtractionRequired"] is not True
        or training["displayThreshold"] != 0.65
    ):
        raise ValueError("M4 training protocol changed")
    if recipe["adapter"]["width"] != 64 or recipe["adapter"]["classifierFrozen"] is not True:
        raise ValueError("M4 adapter architecture changed")
    if recipe["h3Exclusion"]["pixelsMayBeRead"] is not False:
        raise ValueError("M4 H3 boundary changed")
    if len(recipe["regressions"]) != 2 or [row["name"] for row in recipe["regressions"]] != [
        "m3-selector-regression", "m2-development-regression",
    ]:
        raise ValueError("M4 regression order changed")
    if tuple(recipe["britishLibrary"]["allowedDecades"]) != ALLOWED_BRITISH_DECADES:
        raise ValueError("British Library decade strata changed")
    if (
        recipe["britishLibrary"].get("expectedSourceRows") != 19_060
        or recipe["britishLibrary"].get("expectedCandidateRows") != 18_451
        or recipe["britishLibrary"].get("expectedRejectedSourceRows") != 609
        or recipe["britishLibrary"].get("expectedRejectedDates")
        != {"1754": 1, "1777": 13, "Unknown": 595}
        or recipe["britishLibrary"].get("sourceRowCountsCanonicalSha256")
        != "726c7a22f13c2949e8f37a1ed5275c70a8334a6bc68eb9beed429e1d8fd8a2f3"
    ):
        raise ValueError("British Library source eligibility boundary changed")
    if sum(recipe["britishLibrary"]["selectorDecadeQuotas"].values()) != 300:
        raise ValueError("British Library selector quotas changed")
    if sum(recipe["britishLibrary"]["trainingDecadeQuotas"].values()) != 2400:
        raise ValueError("British Library training quotas changed")
    if recipe["rapidata"]["models"] != {
        "dalle-3": "rapidata-dalle-3",
        "flux": "rapidata-flux",
        "mj": "rapidata-midjourney",
        "stable_diffusion": "rapidata-stable-diffusion",
    }:
        raise ValueError("Rapidata model-family map changed")


def _require_source_row(row: dict[str, Any], *, label: str) -> tuple[str, int]:
    shard = row.get("shard")
    row_index = row.get("rowIndex")
    if not isinstance(shard, str) or not shard or PurePosixPath(shard).is_absolute() or ".." in PurePosixPath(shard).parts:
        raise ValueError(f"{label} shard path is invalid")
    if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index < 0:
        raise ValueError(f"{label} row index is invalid")
    return shard, row_index


def rank_british_candidates(
    rows: Iterable[dict[str, Any]],
    *,
    selector_namespace: str,
    training_namespace: str,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group source rows by decade/book and pre-rank rows for both partitions."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen_rows: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("imageType") != "plates":
            raise ValueError("British Library source row is not a plate")
        if not isinstance(row.get("date"), (str, int)) or isinstance(row.get("date"), bool):
            raise ValueError("British Library date has the wrong type")
        if not isinstance(row.get("fname"), str):
            raise ValueError("British Library filename has the wrong type")
        shard, row_index = _require_source_row(row, label="British Library")
        identity = (shard, row_index)
        if identity in seen_rows:
            raise ValueError("British Library source row is duplicated")
        seen_rows.add(identity)
        decade = british_decade(str(row["date"]))
        book = british_book_id(row["fname"])
        grouped[decade][book].append(dict(row))
    for books in grouped.values():
        for book, values in books.items():
            values.sort(key=lambda row: (
                priority(selector_namespace, f"{book}:{row['shard']}:{row['rowIndex']}"),
                str(row["shard"]), int(row["rowIndex"]),
            ))
            for row in values:
                row["selectorPriority"] = priority(
                    selector_namespace, f"{book}:{row['shard']}:{row['rowIndex']}"
                )
                row["trainingPriority"] = priority(
                    training_namespace, f"{book}:{row['shard']}:{row['rowIndex']}"
                )
    return {decade: dict(books) for decade, books in grouped.items()}


def select_british_partitions(
    rows: Iterable[dict[str, Any]],
    *,
    selector_quotas: dict[str, int],
    training_quotas: dict[str, int],
    selector_namespace: str,
    training_namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    if tuple(selector_quotas) != ALLOWED_BRITISH_DECADES or tuple(training_quotas) != ALLOWED_BRITISH_DECADES:
        raise ValueError("British Library decade quota keys or order changed")
    grouped = rank_british_candidates(
        rows,
        selector_namespace=selector_namespace,
        training_namespace=training_namespace,
    )
    selector: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    selector_books: set[str] = set()
    for decade in ALLOWED_BRITISH_DECADES:
        books = grouped.get(decade, {})
        selector_order = sorted(
            books,
            key=lambda book: (priority(selector_namespace, book), book),
        )
        chosen_selector = selector_order[: selector_quotas[decade]]
        if len(chosen_selector) != selector_quotas[decade]:
            raise ValueError(f"British Library selector capacity is insufficient for {decade}")
        selector_books.update(chosen_selector)
        for book in chosen_selector:
            chosen = min(
                books[book],
                key=lambda row: (row["selectorPriority"], str(row["shard"]), int(row["rowIndex"])),
            )
            selector.append({**chosen, "bookId": book, "decade": decade})
        training_order = sorted(
            (book for book in books if book not in selector_books),
            key=lambda book: (priority(training_namespace, book), book),
        )
        chosen_training = training_order[: training_quotas[decade]]
        if len(chosen_training) != training_quotas[decade]:
            raise ValueError(f"British Library training capacity is insufficient for {decade}")
        for book in chosen_training:
            chosen = min(
                books[book],
                key=lambda row: (row["trainingPriority"], str(row["shard"]), int(row["rowIndex"])),
            )
            training.append({**chosen, "bookId": book, "decade": decade})
    if len(selector) != sum(selector_quotas.values()) or len(training) != sum(training_quotas.values()):
        raise AssertionError("British Library partition totals changed")
    if {row["bookId"] for row in selector} & {row["bookId"] for row in training}:
        raise AssertionError("British Library books crossed partitions")
    return selector, training, selector_books


def collect_rapidata_groups(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse repeated vote rows into immutable prompt/model/path groups."""
    paths: dict[str, tuple[str, str]] = {}
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("Rapidata prompt has the wrong type")
        group = prompt_group_id(prompt)
        entry = groups.setdefault(group, {"promptSha256": group, "models": defaultdict(set)})
        for side in (1, 2):
            model = row.get(f"model{side}")
            pathname = row.get(f"image{side}Path")
            if not isinstance(model, str) or not isinstance(pathname, str):
                raise ValueError("Rapidata model/path has the wrong type")
            path_parts = PurePosixPath(pathname)
            if model not in MODELS or not pathname.startswith(f"{model}/") or not pathname.endswith(".jpg"):
                raise ValueError("Rapidata model/path ownership changed")
            if path_parts.is_absolute() or ".." in path_parts.parts:
                raise ValueError("Rapidata image path is unsafe")
            prior = paths.get(pathname)
            if prior is not None and prior != (group, model):
                raise ValueError("Rapidata path belongs to multiple prompt/model groups")
            paths[pathname] = (group, model)
            entry["models"][model].add(pathname)
    output: dict[str, dict[str, Any]] = {}
    for group, entry in groups.items():
        output[group] = {
            "promptSha256": group,
            "models": {model: sorted(entry["models"].get(model, set())) for model in MODELS},
        }
    return output


def select_rapidata_partitions(
    groups: dict[str, dict[str, Any]],
    *,
    selector_groups: int,
    training_groups: int,
    selector_namespace: str,
    training_namespace: str,
    image_namespace: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:
    complete = [
        group for group, row in groups.items()
        if all(len(row["models"].get(model, [])) == 4 for model in MODELS)
    ]
    selector_ids = sorted(complete, key=lambda group: (priority(selector_namespace, group), group))[:selector_groups]
    selector_set = set(selector_ids)
    remaining = [group for group in complete if group not in selector_set]
    training_ids = sorted(remaining, key=lambda group: (priority(training_namespace, group), group))[:training_groups]
    if len(selector_ids) != selector_groups or len(training_ids) != training_groups:
        raise ValueError("Rapidata prompt-group capacity is insufficient")
    selector: list[dict[str, Any]] = []
    excluded_selector_paths: set[str] = set()
    for group in selector_ids:
        for model in MODELS:
            paths = groups[group]["models"][model]
            selected = min(paths, key=lambda path: (priority(image_namespace, f"{group}:{model}:{path}"), path))
            selector.append({"promptSha256": group, "model": model, "path": selected})
            excluded_selector_paths.update(paths)
    training: list[dict[str, Any]] = []
    for group in training_ids:
        for model in MODELS:
            for pathname in groups[group]["models"][model]:
                training.append({"promptSha256": group, "model": model, "path": pathname})
    if len(selector) != selector_groups * len(MODELS):
        raise AssertionError("Rapidata selector row count changed")
    if len(training) != training_groups * len(MODELS) * 4:
        raise AssertionError("Rapidata training row count changed")
    if {row["promptSha256"] for row in selector} & {row["promptSha256"] for row in training}:
        raise AssertionError("Rapidata prompt groups crossed partitions")
    return selector, training, excluded_selector_paths


def source_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["source"]) for row in rows).items()))
