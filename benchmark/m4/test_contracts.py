from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m4 import contracts  # noqa: E402


RECIPE_PATH = ROOT / "benchmark/m4/recipe.json"
LOCKS_PATH = ROOT / "benchmark/m4/source-locks.json"


def rapidata_rows(prompt: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model in contracts.MODELS:
        for pair in range(2):
            rows.append({
                "prompt": prompt,
                "model1": model,
                "image1Path": f"{model}/{prompt}-{pair * 2}.jpg",
                "model2": model,
                "image2Path": f"{model}/{prompt}-{pair * 2 + 1}.jpg",
            })
    return rows


class M4FrozenContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.recipe = json.loads(RECIPE_PATH.read_text())
        self.locks = json.loads(LOCKS_PATH.read_text())

    def test_current_recipe_and_source_locks_are_exact(self) -> None:
        recipe, locks = contracts.load_frozen_protocol(RECIPE_PATH, LOCKS_PATH)
        self.assertEqual(recipe["name"], "prooflens-m4-residual-adapter")
        self.assertEqual(len(locks["britishLibrary"]["files"]), 32)
        self.assertEqual(len(locks["rapidata"]["files"]), 26)

    def test_recipe_mutations_reject_across_every_frozen_section(self) -> None:
        mutations = [
            lambda value: value.__setitem__("seed", 1),
            lambda value: value["freshSelector"]["sourceCounts"].__setitem__("rapidata-flux", 74),
            lambda value: value["regressions"][0]["gates"].__setitem__("minimumBalancedAccuracyPerVariant", 0),
            lambda value: value["regressions"][1].__setitem__("sha256", "0" * 64),
            lambda value: value["h3Exclusion"].__setitem__("manifest", "benchmark/data/h3-secret.jsonl"),
            lambda value: value["britishLibrary"].__setitem__("selectorDecadeQuotas", {"1900": 300}),
            lambda value: value["rapidata"].__setitem__("expectedFourPerFamilyGroups", 237),
            lambda value: value["adapter"].__setitem__("width", 65),
            lambda value: value["training"].__setitem__("featureBatchSize", 1),
            lambda value: value["validationGates"].__setitem__("minimumSyntheticRecallPerVariant", 0),
            lambda value: value["onnxContract"].__setitem__("opset", 17),
            lambda value: value["selectionPolicy"].__setitem__("h3InputPermitted", True),
            lambda value: value["output"].__setitem__("dataRoot", "benchmark/data/h3-met-holdout-v1"),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = deepcopy(self.recipe)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "recipe changed"):
                    contracts.validate_recipe(changed, self.locks)

    def test_source_lock_mutations_reject_even_when_shape_and_totals_stay_valid(self) -> None:
        mutations = [
            lambda value: value["britishLibrary"].__setitem__("revision", "0" * 40),
            lambda value: value["britishLibrary"].__setitem__("selection", "first 32 files"),
            lambda value: value["britishLibrary"].__setitem__("repositoryBytes", 1),
            lambda value: value["britishLibrary"]["files"][0].__setitem__("path", "plates/part-00000.parquet"),
            lambda value: value["britishLibrary"]["files"][0].__setitem__("sha256", "0" * 64),
            lambda value: value["britishLibrary"]["files"][0].__setitem__("gitOid", "0" * 40),
            lambda value: value["rapidata"].__setitem__("repositoryInventoryCanonicalSha256", "0" * 64),
            lambda value: value["rapidata"]["files"].reverse(),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = deepcopy(self.locks)
                mutate(changed)
                with self.assertRaisesRegex(ValueError, "source-lock packet changed"):
                    contracts.validate_source_locks(changed)

    def test_raw_protocol_bytes_duplicate_keys_and_invalid_utf8_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = root / "recipe.json"
            locks = root / "locks.json"
            recipe.write_bytes(RECIPE_PATH.read_bytes() + b" ")
            locks.write_bytes(LOCKS_PATH.read_bytes())
            with self.assertRaisesRegex(ValueError, "recipe bytes changed"):
                contracts.load_frozen_protocol(recipe, locks)
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            contracts.parse_json_bytes(b'{"x":1,"x":2}\n', label="probe")
        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            contracts.parse_json_bytes(b'{"x":"\xff"}\n', label="probe")

    def test_british_book_partition_is_selector_first_disjoint_and_deterministic(self) -> None:
        rows = []
        for index, book in enumerate(("100000", "100001", "100002", "100003")):
            rows.append({
                "imageType": "plates",
                "date": "1800",
                "fname": f"{book}_plate_{index}.jpg",
                "shard": "plates/part-00001.parquet",
                "rowIndex": index,
            })
        selector_quotas = {decade: int(decade == "1800") for decade in contracts.ALLOWED_BRITISH_DECADES}
        training_quotas = {decade: 2 * int(decade == "1800") for decade in contracts.ALLOWED_BRITISH_DECADES}
        first = contracts.select_british_partitions(
            rows,
            selector_quotas=selector_quotas,
            training_quotas=training_quotas,
            selector_namespace="selector:",
            training_namespace="training:",
        )
        second = contracts.select_british_partitions(
            reversed(rows),
            selector_quotas=selector_quotas,
            training_quotas=training_quotas,
            selector_namespace="selector:",
            training_namespace="training:",
        )
        self.assertEqual(first, second)
        selector, training, selector_books = first
        self.assertEqual(len(selector), 1)
        self.assertEqual(len(training), 2)
        self.assertFalse({row["bookId"] for row in selector} & {row["bookId"] for row in training})
        self.assertEqual(selector_books, {selector[0]["bookId"]})

    def test_british_date_classification_is_explicit_and_score_blind(self) -> None:
        self.assertEqual(contracts.classify_british_date("1887"), ("1880", "1887"))
        self.assertEqual(contracts.classify_british_date("Unknown"), (None, "Unknown"))
        self.assertEqual(contracts.classify_british_date("1754"), (None, "1754"))
        self.assertEqual(contracts.classify_british_date(None), (None, "null"))
        self.assertEqual(contracts.classify_british_date(True), (None, "True"))

    def test_british_ties_and_capacity_fail_closed(self) -> None:
        rows = [
            {
                "imageType": "plates", "date": "1800", "fname": f"{book}_x.jpg",
                "shard": "plates/part-00001.parquet", "rowIndex": index,
            }
            for index, book in enumerate(("100000", "100001", "100002"))
        ]
        selector_quotas = {decade: int(decade == "1800") for decade in contracts.ALLOWED_BRITISH_DECADES}
        training_quotas = {decade: int(decade == "1800") for decade in contracts.ALLOWED_BRITISH_DECADES}
        with patch("benchmark.m4.contracts.priority", return_value="0" * 64):
            first = contracts.select_british_partitions(
                rows, selector_quotas=selector_quotas, training_quotas=training_quotas,
                selector_namespace="s:", training_namespace="t:",
            )
            second = contracts.select_british_partitions(
                reversed(rows), selector_quotas=selector_quotas, training_quotas=training_quotas,
                selector_namespace="s:", training_namespace="t:",
            )
        self.assertEqual(first, second)
        too_many = dict(training_quotas)
        too_many["1800"] = 3
        with self.assertRaisesRegex(ValueError, "training capacity"):
            contracts.select_british_partitions(
                rows, selector_quotas=selector_quotas, training_quotas=too_many,
                selector_namespace="s:", training_namespace="t:",
            )
        with self.assertRaisesRegex(ValueError, "duplicated"):
            contracts.rank_british_candidates(rows + [rows[0]], selector_namespace="s:", training_namespace="t:")

    def test_rapidata_vote_collapse_and_prompt_partition_are_deterministic(self) -> None:
        rows = rapidata_rows("alpha") + rapidata_rows("beta") + rapidata_rows("gamma")
        groups = contracts.collect_rapidata_groups(rows + rows[:2])
        self.assertEqual(len(groups), 3)
        first = contracts.select_rapidata_partitions(
            groups,
            selector_groups=1,
            training_groups=1,
            selector_namespace="selector:",
            training_namespace="training:",
            image_namespace="image:",
        )
        reverse_groups = dict(reversed(list(groups.items())))
        second = contracts.select_rapidata_partitions(
            reverse_groups,
            selector_groups=1,
            training_groups=1,
            selector_namespace="selector:",
            training_namespace="training:",
            image_namespace="image:",
        )
        self.assertEqual(first, second)
        selector, training, excluded_paths = first
        self.assertEqual(len(selector), 4)
        self.assertEqual(len(training), 16)
        self.assertEqual(len(excluded_paths), 16)
        self.assertFalse({row["promptSha256"] for row in selector} & {row["promptSha256"] for row in training})

    def test_rapidata_ties_conflicts_types_and_capacity_fail_closed(self) -> None:
        rows = rapidata_rows("alpha") + rapidata_rows("beta")
        groups = contracts.collect_rapidata_groups(rows)
        with patch("benchmark.m4.contracts.priority", return_value="0" * 64):
            first = contracts.select_rapidata_partitions(
                groups, selector_groups=1, training_groups=1,
                selector_namespace="s:", training_namespace="t:", image_namespace="i:",
            )
            second = contracts.select_rapidata_partitions(
                dict(reversed(list(groups.items()))), selector_groups=1, training_groups=1,
                selector_namespace="s:", training_namespace="t:", image_namespace="i:",
            )
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "capacity"):
            contracts.select_rapidata_partitions(
                groups, selector_groups=2, training_groups=1,
                selector_namespace="s:", training_namespace="t:", image_namespace="i:",
            )
        conflict = deepcopy(rows[0])
        conflict["prompt"] = "other"
        with self.assertRaisesRegex(ValueError, "multiple prompt"):
            contracts.collect_rapidata_groups(rows + [conflict])
        wrong_type = deepcopy(rows[0])
        wrong_type["prompt"] = None
        with self.assertRaisesRegex(ValueError, "wrong type"):
            contracts.collect_rapidata_groups([wrong_type])


if __name__ == "__main__":
    unittest.main()
