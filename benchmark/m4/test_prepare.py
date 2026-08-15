from __future__ import annotations

import copy
import os
from io import BytesIO
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m4 import prepare, verify  # noqa: E402
from benchmark.m4.contracts import ALLOWED_BRITISH_DECADES, MODELS, prompt_group_id  # noqa: E402


def candidate(identifier: str, group: str, perceptual: str, *, label: int, source: str) -> dict[str, object]:
    return {
        "id": identifier,
        "imageSha256": __import__("hashlib").sha256(identifier.encode()).hexdigest(),
        "sourceGroupId": group,
        "perceptualDhash64": perceptual,
        "label": label,
        "source": source,
        "width": 10,
        "height": 10,
        "extension": ".jpg",
    }


def rapidata_fixture(prompt: str, perceptual_prefix: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    group_id = prompt_group_id(prompt)
    model_paths: dict[str, list[str]] = {}
    images: list[dict[str, object]] = []
    for model_index, model in enumerate(MODELS):
        paths = [f"{model}/{prompt}-{index}.jpg" for index in range(4)]
        model_paths[model] = paths
        for index, path in enumerate(paths):
            value = candidate(
                f"rapidata:{prompt}:{model}:{index}",
                f"rapidata:prompt:{group_id}",
                f"{perceptual_prefix + model_index * 16 + index:016x}",
                label=1,
                source=f"rapidata-{model}",
            )
            value.update({"imagePath": path, "model": model, "promptSha256": group_id})
            images.append(value)
    return {"promptSha256": group_id, "models": model_paths}, images


class M4PreparationTests(unittest.TestCase):
    def test_deterministic_gzip_has_canonical_header(self) -> None:
        raw = b"prooflens-m4\n" * 100
        first = prepare.deterministic_gzip(raw)
        self.assertEqual(first, prepare.deterministic_gzip(raw))
        self.assertEqual(first[9], 0xFF)
        self.assertEqual(__import__("gzip").decompress(first), raw)

    def test_inventory_canonicalization_ignores_input_order_but_not_content(self) -> None:
        rows = [
            {"path": "b.parquet", "bytes": 2, "sha256": "b" * 64, "gitOid": "b" * 40},
            {"path": "a.parquet", "bytes": 1, "sha256": "a" * 64, "gitOid": "a" * 40},
        ]
        self.assertEqual(prepare.canonical_inventory(rows), prepare.canonical_inventory(reversed(rows)))
        changed = [dict(row) for row in rows]
        changed[0]["bytes"] = 3
        self.assertNotEqual(prepare.canonical_inventory(rows), prepare.canonical_inventory(changed))

    def test_british_scan_accounts_for_every_raw_row_before_pixel_decode(self) -> None:
        from PIL import Image

        encoded = BytesIO()
        Image.new("RGB", (16, 16), (127, 64, 32)).save(encoded, format="PNG")
        rows = [
            {"image": {"bytes": b"not-an-image", "path": None}, "date": "Unknown", "fname": "100000_bad.png", "image_type": "plates"},
            {"image": {"bytes": b"not-an-image", "path": None}, "date": "1754", "fname": "100001_bad.png", "image_type": "plates"},
            {"image": {"bytes": b"not-an-image", "path": None}, "date": "1777", "fname": "100002_bad.png", "image_type": "plates"},
            {"image": {"bytes": encoded.getvalue(), "path": None}, "date": "1887", "fname": "100003_good.png", "image_type": "plates"},
        ]

        class FakeBatch:
            def to_pylist(self) -> list[dict[str, object]]:
                return rows

        class FakeParquetFile:
            def __init__(self, _path: Path) -> None:
                pass

            def iter_batches(self, *, batch_size: int, columns: list[str]) -> list[FakeBatch]:
                self.assertions = (batch_size, columns)
                return [FakeBatch()]

        fake_pyarrow = ModuleType("pyarrow")
        fake_parquet = ModuleType("pyarrow.parquet")
        fake_parquet.ParquetFile = FakeParquetFile  # type: ignore[attr-defined]
        fake_pyarrow.parquet = fake_parquet  # type: ignore[attr-defined]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part-00001.parquet"
            path.touch()
            source_counts = {"plates/part-00001.parquet": 4}
            recipe = {
                "britishLibrary": {
                    "expectedSourceRows": 4,
                    "expectedCandidateRows": 1,
                    "expectedRejectedSourceRows": 3,
                    "expectedRejectedDates": {"1754": 1, "1777": 1, "Unknown": 1},
                    "sourceRowCountsCanonicalSha256": prepare.digest_bytes(prepare.canonical_json(source_counts)),
                    "expectedDistinctBooksByDecade": {
                        decade: int(decade == "1880") for decade in ALLOWED_BRITISH_DECADES
                    },
                    "expectedDistinctBooks": 1,
                    "sourceReportedLicense": "fixture",
                },
            }
            locks = {
                "britishLibrary": {
                    "dataset": "fixture/british",
                    "revision": "1" * 40,
                    "files": [{"path": "plates/part-00001.parquet", "sha256": "2" * 64}],
                },
            }
            with patch.dict(sys.modules, {"pyarrow": fake_pyarrow, "pyarrow.parquet": fake_parquet}):
                candidates, eligibility = prepare.scan_british([path], locks, recipe)
        self.assertEqual([row["sourceRow"] for row in candidates], [3])
        self.assertEqual(eligibility["sourceRows"], 4)
        self.assertEqual(eligibility["eligibleCandidateRows"], 1)
        self.assertEqual(eligibility["rejectedDateCounts"], {"1754": 1, "1777": 1, "Unknown": 1})
        self.assertEqual([row["sourceRow"] for row in eligibility["rejectedItems"]], [0, 1, 2])

    def test_public_british_eligibility_verifier_rejects_incomplete_or_ambiguous_rows(self) -> None:
        shard = "plates/part-00001.parquet"
        shard_sha = "2" * 64
        row_counts = {shard: 4}
        recipe = {
            "britishLibrary": {
                "expectedSourceRows": 4,
                "expectedRejectedSourceRows": 3,
                "expectedRejectedDates": {"1754": 1, "1777": 1, "Unknown": 1},
                "sourceRowCountsCanonicalSha256": prepare.digest_bytes(prepare.canonical_json(row_counts)),
            },
        }
        eligible = [{"sourceShard": shard, "sourceRow": 3}]
        rejected = [
            {
                "sourceShard": shard,
                "sourceShardSha256": shard_sha,
                "sourceRow": index,
                "rawDate": raw_date,
                "reason": "date-not-in-frozen-strata",
            }
            for index, raw_date in enumerate(("Unknown", "1754", "1777"))
        ]
        eligibility = {
            "sourceRows": 4,
            "eligibleCandidateRows": 1,
            "rejectedSourceRows": 3,
            "rejectedDateCounts": {"1754": 1, "1777": 1, "Unknown": 1},
            "sourceRowCounts": row_counts,
            "rejectedItems": rejected,
        }
        locks = {shard: {"sha256": shard_sha}}
        verify._validate_british_source_eligibility(eligible, eligibility, recipe, locks)

        cases: dict[str, tuple[dict[str, object], dict[str, object]]] = {}

        missing = copy.deepcopy(eligibility)
        missing["sourceRows"] = 5
        missing["sourceRowCounts"][shard] = 5
        missing_recipe = copy.deepcopy(recipe)
        missing_recipe["britishLibrary"]["expectedSourceRows"] = 5
        missing_recipe["britishLibrary"]["sourceRowCountsCanonicalSha256"] = prepare.digest_bytes(
            prepare.canonical_json(missing["sourceRowCounts"]),
        )
        cases["missing position"] = (missing, missing_recipe)

        overlap = copy.deepcopy(eligibility)
        overlap["rejectedItems"][0]["sourceRow"] = 3
        cases["eligible/rejected overlap"] = (overlap, copy.deepcopy(recipe))

        duplicate = copy.deepcopy(eligibility)
        duplicate["rejectedItems"][2]["sourceRow"] = 0
        cases["duplicate rejected position"] = (duplicate, copy.deepcopy(recipe))

        out_of_range = copy.deepcopy(eligibility)
        out_of_range["rejectedItems"][2]["sourceRow"] = 4
        cases["out-of-range position"] = (out_of_range, copy.deepcopy(recipe))

        wrong_dates = copy.deepcopy(eligibility)
        wrong_dates["rejectedItems"][1]["rawDate"] = "Unknown"
        cases["rejected date counts"] = (wrong_dates, copy.deepcopy(recipe))

        wrong_row_hash = copy.deepcopy(eligibility)
        wrong_row_hash["sourceRowCounts"][shard] = 5
        cases["source row-count hash"] = (wrong_row_hash, copy.deepcopy(recipe))

        for label, (changed_eligibility, changed_recipe) in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    verify._validate_british_source_eligibility(
                        eligible, changed_eligibility, changed_recipe, locks,
                    )

    def test_overlap_state_is_group_atomic_and_rejects_cross_pool_near_matches(self) -> None:
        state = prepare.OverlapState(8)
        group = [
            candidate("a", "prompt:a", "0000000000000000", label=1, source="one"),
            candidate("b", "prompt:a", "0000000000000001", label=1, source="two"),
        ]
        self.assertIsNone(state.admit_group(group))
        rejected = candidate("c", "prompt:c", "0000000000000003", label=1, source="three")
        self.assertEqual(state.issue(rejected)["reason"], "perceptualDhash64")
        duplicate_group = candidate("d", "prompt:a", "ffffffffffffffff", label=1, source="four")
        self.assertEqual(state.issue(duplicate_group)["reason"], "sourceGroupId")
        duplicate_bytes = dict(candidate("e", "prompt:e", "ffffffffffffffff", label=1, source="five"))
        duplicate_bytes["imageSha256"] = group[0]["imageSha256"]
        self.assertEqual(state.issue(duplicate_bytes)["reason"], "imageSha256")

    def test_british_selector_is_admitted_before_disjoint_training(self) -> None:
        candidates: list[dict[str, object]] = []
        perceptual_values = (
            "0000000000000000", "ffffffffffffffff", "aaaaaaaaaaaaaaaa", "5555555555555555",
        )
        for index, book in enumerate(("100000", "100001", "100002", "100003")):
            row = candidate(
                f"british:{book}", f"british:book:{book}", perceptual_values[index],
                label=0, source="british-library-plates",
            )
            row.update({"bookId": book, "decade": "1800", "sourceShard": "plates/a.parquet", "sourceRow": index})
            candidates.append(row)
        selector_quotas = {decade: int(decade == "1800") for decade in ALLOWED_BRITISH_DECADES}
        training_quotas = {decade: 2 * int(decade == "1800") for decade in ALLOWED_BRITISH_DECADES}
        state = prepare.OverlapState(8)
        rejects: list[dict[str, object]] = []
        selector, considered = prepare.select_british_phase(
            candidates,
            phase="validation-real",
            quotas=selector_quotas,
            namespace="selector:",
            excluded_books=set(),
            state=state,
            rejects=rejects,
        )
        training, _ = prepare.select_british_phase(
            candidates,
            phase="train-real",
            quotas=training_quotas,
            namespace="training:",
            excluded_books=considered,
            state=state,
            rejects=rejects,
        )
        self.assertEqual(len(selector), 1)
        self.assertEqual(len(training), 2)
        self.assertFalse({row["bookId"] for row in selector} & {row["bookId"] for row in training})
        self.assertTrue({row["bookId"] for row in selector} <= considered)

    def test_rapidata_selector_prompt_is_excluded_from_training(self) -> None:
        alpha_group, alpha_images = rapidata_fixture("alpha", 0x0000000000000000)
        beta_group, beta_images = rapidata_fixture("beta", 0xFFFFFFFFFFFFFF00)
        groups = {
            str(alpha_group["promptSha256"]): alpha_group,
            str(beta_group["promptSha256"]): beta_group,
        }
        state = prepare.OverlapState(8)
        rejects: list[dict[str, object]] = []
        selector, considered = prepare.select_rapidata_phase(
            groups,
            [*alpha_images, *beta_images],
            phase="validation-synthetic",
            target_groups=1,
            namespace="selector:",
            image_namespace="image:",
            excluded_groups=set(),
            state=state,
            rejects=rejects,
        )
        training, _ = prepare.select_rapidata_phase(
            groups,
            [*alpha_images, *beta_images],
            phase="train-synthetic",
            target_groups=1,
            namespace="training:",
            image_namespace="image:",
            excluded_groups=considered,
            state=state,
            rejects=rejects,
        )
        self.assertEqual(len(selector), 4)
        self.assertEqual(len(training), 16)
        self.assertFalse({row["promptSha256"] for row in selector} & {row["promptSha256"] for row in training})
        self.assertTrue({row["promptSha256"] for row in selector} <= considered)

    def test_rapidata_training_retains_clean_images_when_every_family_remains(self) -> None:
        group, images = rapidata_fixture("partial", 0)
        for row in images:
            row["perceptualDhash64"] = __import__("hashlib").sha256(str(row["id"]).encode()).hexdigest()[:16]
        blocker = candidate(
            "frozen:blocker", "frozen:blocker", str(images[0]["perceptualDhash64"]),
            label=0, source="frozen",
        )
        state = prepare.OverlapState(8)
        state.add_frozen(blocker)
        rejects: list[dict[str, object]] = []
        selected, considered = prepare.select_rapidata_phase(
            {str(group["promptSha256"]): group}, images,
            phase="train-synthetic", target_groups=1, namespace="training:",
            image_namespace="image:", excluded_groups=set(), state=state, rejects=rejects,
        )
        self.assertEqual(len(selected), 15)
        self.assertEqual({str(row["model"]) for row in selected}, set(MODELS))
        self.assertEqual(considered, {str(group["promptSha256"])})
        self.assertEqual(len(rejects), 1)
        self.assertEqual(rejects[0]["reason"], "perceptualDhash64")

    def test_rapidata_training_rejects_group_when_a_clean_family_is_missing(self) -> None:
        group, images = rapidata_fixture("missing", 0)
        for row in images:
            row["perceptualDhash64"] = __import__("hashlib").sha256(str(row["id"]).encode()).hexdigest()[:16]
        state = prepare.OverlapState(8)
        blocked_model = MODELS[0]
        for index, row in enumerate(item for item in images if item["model"] == blocked_model):
            state.add_frozen(candidate(
                f"frozen:blocker:{index}", f"frozen:blocker:{index}", str(row["perceptualDhash64"]),
                label=0, source="frozen",
            ))
        rejects: list[dict[str, object]] = []
        with self.assertRaisesRegex(ValueError, "capacity"):
            prepare.select_rapidata_phase(
                {str(group["promptSha256"]): group}, images,
                phase="train-synthetic", target_groups=1, namespace="training:",
                image_namespace="image:", excluded_groups=set(), state=state, rejects=rejects,
            )
        self.assertEqual(sum(row["reason"] == "perceptualDhash64" for row in rejects), 4)
        missing = [row for row in rejects if row["reason"] == "missingCleanFamily"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["missingFamilies"], [blocked_model])

    def test_rapidata_capacity_failure_does_not_reuse_selector_group(self) -> None:
        group, images = rapidata_fixture("only", 0x1000000000000000)
        groups = {str(group["promptSha256"]): group}
        state = prepare.OverlapState(8)
        rejects: list[dict[str, object]] = []
        _, considered = prepare.select_rapidata_phase(
            groups, images, phase="validation-synthetic", target_groups=1,
            namespace="selector:", image_namespace="image:", excluded_groups=set(),
            state=state, rejects=rejects,
        )
        with self.assertRaisesRegex(ValueError, "capacity"):
            prepare.select_rapidata_phase(
                groups, images, phase="train-synthetic", target_groups=1,
                namespace="training:", image_namespace="image:", excluded_groups=considered,
                state=state, rejects=rejects,
            )

    def test_data_root_rejects_child_symlink_escape(self) -> None:
        data = ROOT / "benchmark/data"
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=data) as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            (root / "escape").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlinked"):
                prepare.safe_output_path(root, "escape/pixel.jpg")

    def test_source_is_score_blind_and_has_no_h3_data_argument(self) -> None:
        source = (ROOT / "benchmark/m4/prepare.py").read_text()
        for forbidden in ("onnxruntime", "rawProbability", "model.onnx", "--h3", "h3-data-root"):
            self.assertNotIn(forbidden, source)
        self.assertIn('"h3PixelsRead": False', source)
        self.assertNotIn("benchmark/data/h3-met-holdout-v1", source)


if __name__ == "__main__":
    unittest.main()
