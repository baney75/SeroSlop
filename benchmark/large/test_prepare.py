"""Offline regression tests for the large-corpus preparation contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare  # noqa: E402


def fixture_recipe(path: Path) -> dict[str, object]:
    recipe: dict[str, object] = {
        "schemaVersion": 1,
        "seed": 7,
        "diffusionDb": {
            "revision": "diff-revision",
            "targetCount": 1,
            "reserveCount": 1,
            "candidatePartCount": 1,
        },
        "openImages": {
            "revision": "open-revision",
            "targetCount": 1,
            "reserveCount": 1,
        },
        "resourceEnvelope": {
            "minimumFreeBytesBeforeMaterialization": 1,
            "maximumOpenImageBytes": 1_000_000,
            "maximumArchiveWorkers": 2,
            "maximumOpenImageWorkers": 2,
        },
    }
    path.write_text(json.dumps(recipe, sort_keys=True) + "\n")
    return recipe


def fixture_plan(recipe_path: Path, recipe: dict[str, object]) -> dict[str, object]:
    diffusion = recipe["diffusionDb"]
    open_images = recipe["openImages"]
    seed = int(recipe["seed"])
    diffusion_revision = str(diffusion["revision"])
    part_id = min(
        range(1, 2001),
        key=lambda value: prepare.selection_priority(seed, diffusion_revision, "part", str(value)),
    )
    diffusion_rows = [
        {
            "id": f"image-{index}.png",
            "partId": part_id,
            "priority": prepare.selection_priority(seed, diffusion_revision, "image", f"image-{index}.png"),
        }
        for index in range(2)
    ]
    diffusion_rows.sort(key=lambda row: (row["priority"], row["id"]))
    open_revision = str(open_images["revision"])
    open_rows = [
        {
            "id": f"open-{index}",
            "priority": prepare.selection_priority(seed, open_revision, "image", f"open-{index}"),
        }
        for index in range(2)
    ]
    open_rows.sort(key=lambda row: (row["priority"], row["id"]))
    return {
        "schemaVersion": 1,
        "recipeSha256": prepare.digest(recipe_path),
        "diffusionDb": {
            "partIds": [part_id],
            "archives": [
                {"partId": part_id, "path": f"images/part-{part_id:06d}.zip", "sha256": "a" * 64, "bytes": 1}
            ],
            "candidates": diffusion_rows,
        },
        "openImages": {"candidates": open_rows},
    }


class LargePreparationTests(unittest.TestCase):
    def test_plan_validation_rejects_duplicate_candidate(self) -> None:
        with TemporaryDirectory() as temporary:
            recipe_path = Path(temporary) / "recipe.json"
            recipe = fixture_recipe(recipe_path)
            plan = fixture_plan(recipe_path, recipe)
            prepare.validate_plan(recipe_path, recipe, plan)
            plan["diffusionDb"]["candidates"][1]["id"] = plan["diffusionDb"]["candidates"][0]["id"]
            with self.assertRaisesRegex(ValueError, "duplicate IDs"):
                prepare.validate_plan(recipe_path, recipe, plan)

    def test_selection_excludes_held_out_id_and_hash(self) -> None:
        rows = [
            {"id": "blocked-id", "imageSha256": "1" * 64, "perceptualDhash64": "1" * 16, "dataset": "fixture"},
            {"id": "blocked-hash", "imageSha256": "2" * 64, "perceptualDhash64": "2" * 16, "dataset": "fixture"},
            {"id": "selected", "imageSha256": "3" * 64, "perceptualDhash64": "3" * 16, "dataset": "fixture"},
        ]
        rejects: list[dict[str, object]] = []
        selected = prepare.select_unique(
            rows,
            1,
            set(),
            set(),
            {"blocked-id"},
            {"2" * 64},
            prepare.build_perceptual_index([]),
            8,
            rejects,
        )
        self.assertEqual([row["id"] for row in selected], ["selected"])
        self.assertEqual(len(rejects), 2)

    def test_resource_preflight_fails_before_materialization(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = {
                "resourceEnvelope": {
                    "minimumFreeBytesBeforeMaterialization": 10**30,
                    "maximumOpenImageBytes": 10,
                    "maximumArchiveWorkers": 1,
                    "maximumOpenImageWorkers": 1,
                }
            }
            plan = {
                "diffusionDb": {"archives": [], "candidates": []},
                "openImages": {"candidates": []},
            }
            with self.assertRaisesRegex(OSError, "requires at least"):
                prepare.preflight_resources(recipe, plan, root)
            self.assertTrue((root / "resource-report.json").exists())

    def test_perceptual_index_finds_threshold_match(self) -> None:
        index = prepare.build_perceptual_index(
            [{"id": "held-out", "perceptualDhash64": "0000000000000000"}]
        )
        matches = prepare.perceptual_matches("000000000000000f", index, 4)
        self.assertEqual(matches, [{"id": "held-out", "hammingDistance": 4}])
        self.assertEqual(prepare.perceptual_matches("ffffffffffffffff", index, 4), [])

    def test_perceptual_review_must_be_exact_and_approved(self) -> None:
        exact = {
            ("train", "eval"): {
                "trainingId": "train",
                "evaluationId": "eval",
                "hammingDistance": 5,
            }
        }
        approved = [{
            "trainingId": "train",
            "evaluationId": "eval",
            "hammingDistance": 5,
            "decision": "visually-distinct",
            "rationale": "Different scenes",
            "reviewer": "reviewer",
            "reviewedAt": "2026-08-13",
        }]
        self.assertEqual(
            set(prepare.require_exact_visually_distinct_pairs(approved, exact)),
            {("train", "eval")},
        )
        pending = [{**approved[0], "decision": "pending"}]
        with self.assertRaisesRegex(ValueError, "not approved exactly"):
            prepare.require_exact_visually_distinct_pairs(pending, exact)
        with self.assertRaisesRegex(ValueError, "stale or incomplete"):
            prepare.require_exact_visually_distinct_pairs([], exact)


if __name__ == "__main__":
    unittest.main()
