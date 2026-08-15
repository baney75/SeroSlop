from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.m4 import publication_contract as contract  # noqa: E402
from benchmark.m4.contracts import canonical_json  # noqa: E402
from benchmark.m4.select_model_state_fixtures import validate_previous_manifest  # noqa: E402
from benchmark.fresh_feature_run import marker_bytes  # noqa: E402


class M4PublicationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packet = {
            "candidateHashes": {
                "model": "1" * 64,
                "summary": "2" * 64,
                "calibration": "3" * 64,
                "grid": "4" * 64,
                "selectionLock": "5" * 64,
                "tensorSeal": "6" * 64,
                "freshMarker": "7" * 64,
            },
            "modelBytes": 87_700_000,
            "selectionLock": {"selectedCandidateId": "wd-0.003-anchor-0.01"},
        }
        self.arguments = {
            "source_commit": "a" * 40,
            "source_tree": "b" * 40,
            "packet": self.packet,
            "comparison_sha256": "c" * 64,
            "adapter_patch": {"schemaVersion": 1, "candidateSha256": "1" * 64},
            "candidate_evidence_json": {"training-summary.json": "{}\n"},
            "public_document_hashes": {
                "README.md": "d" * 64,
                "MODEL_CARD.md": "e" * 64,
                "BENCHMARK.md": "f" * 64,
            },
            "fixture_manifest_sha256": "8" * 64,
        }

    def test_publication_surface_is_exact_and_receipt_last(self) -> None:
        self.assertEqual(len(contract.PUBLICATION_ROWS), 12)
        self.assertEqual(contract.TRANSACTIONAL_ROWS, contract.PUBLICATION_ROWS)
        self.assertEqual(contract.PUBLICATION_ROWS[-1], ("weights/prooflens-cf384.onnx", "M"))
        self.assertEqual(
            {path for path, _status in contract.PUBLICATION_ROWS},
            {
                "BENCHMARK.md", "MODEL_CARD.md", "README.md",
                "benchmark/evidence/m4/calibration.json",
                "benchmark/evidence/m4/candidate-grid.json",
                "benchmark/evidence/m4/finalization-receipt.json",
                "benchmark/evidence/m4/model-comparison.json",
                "benchmark/evidence/m4/training-summary.json",
                "model-lock.json", "tests/fixtures/model-states/fixture-manifest.json",
                "weights/README.md", "weights/prooflens-cf384.onnx",
            },
        )

    def test_publication_lock_is_exact_and_mutation_sensitive(self) -> None:
        with patch.object(contract, "digest", return_value="9" * 64):
            lock = contract.build_publication_lock(**self.arguments)
            contract.validate_publication_lock(lock, **self.arguments)
            self.assertEqual(lock["profile"], "m4")
            self.assertEqual(lock["publicationRows"], [
                {"path": path, "status": status} for path, status in contract.PUBLICATION_ROWS
            ])
            for field in lock:
                with self.subTest(field=field):
                    changed = deepcopy(lock)
                    changed[field] = "changed"
                    with self.assertRaisesRegex(ValueError, "publication lock"):
                        contract.validate_publication_lock(changed, **self.arguments)

    def test_publication_lock_rejects_duplicate_keys_invalid_utf8_and_noncanonical_bytes(self) -> None:
        value = {"schemaVersion": 1, "h3HoldoutScored": False}
        raw = canonical_json(value, pretty=True)
        self.assertEqual(contract.parse_canonical_publication_lock(raw), value)
        duplicate = raw.replace(
            b'  "h3HoldoutScored": false',
            b'  "h3HoldoutScored": true,\n  "h3HoldoutScored": false',
        )
        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            contract.parse_canonical_publication_lock(duplicate)
        with self.assertRaisesRegex(ValueError, "strict UTF-8"):
            contract.parse_canonical_publication_lock(b'{"x":"\xff"}\n')
        with self.assertRaisesRegex(ValueError, "canonical"):
            contract.parse_canonical_publication_lock(b'{"schemaVersion":1,"h3HoldoutScored":false}\n')

    def test_exact_selector_and_regression_source_sets(self) -> None:
        row = {
            "balancedAccuracy": 1.0,
            "realRecall": 1.0,
            "syntheticRecall": 1.0,
            "syntheticRecallBySource": {source: 1.0 for source in sorted(contract.SELECTOR_SYNTHETIC_SOURCES)},
            "realRecallBySource": {"british-library-plates": 1.0},
        }
        metrics = {variant: deepcopy(row) for variant in contract.VARIANTS}
        contract.require_source_sets(metrics, contract.SELECTOR_SYNTHETIC_SOURCES,
                                     contract.SELECTOR_REAL_SOURCES, "selector")
        metrics["original"]["syntheticRecallBySource"] = {"wrong-family": 1.0}
        with self.assertRaisesRegex(ValueError, "synthetic source set"):
            contract.require_source_sets(metrics, contract.SELECTOR_SYNTHETIC_SOURCES,
                                         contract.SELECTOR_REAL_SOURCES, "selector")

    def test_adapter_graph_contract_is_exact(self) -> None:
        self.assertEqual(len(contract.ADDED_INITIALIZERS), 6)
        self.assertEqual(len(contract.ADDED_NODES), 7)
        self.assertEqual(contract.ADDED_NODES[-1], (
            "m4_add_residual", "Add", ["/Gather_output_0", "m4.residual"], ["m4.adapted"],
        ))
        self.assertEqual(contract.ADDED_INITIALIZER_SHAPES["m4.adapter_in.weight"], [64, 384])
        self.assertEqual(contract.ADDED_INITIALIZER_SHAPES["m4.adapter_out.weight"], [384, 64])

    def test_candidate_paths_and_grid_tensor_records_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "benchmark/candidates/prooflens-cf384-m4/candidates"
            candidates.mkdir(parents=True)
            identifier = "wd-0.003-anchor-0.01"
            candidate = candidates / f"{identifier}.npz"
            candidate.write_bytes(b"candidate")
            with (
                patch.object(contract, "ROOT", root),
                patch.object(contract, "CANDIDATE_DIR", candidates.parent),
            ):
                self.assertEqual(contract.candidate_artifact_path(identifier), candidate)
                for changed in ("../../outside", "/tmp/outside", "wd-0.003-anchor-9.99"):
                    with self.assertRaisesRegex(ValueError, "frozen hyperparameter grid"):
                        contract.candidate_artifact_path(changed)
                candidate.unlink()
                outside = root / "outside.npz"
                outside.write_bytes(b"outside")
                candidate.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, "direct regular file"):
                    contract.candidate_artifact_path(identifier)

        sealed = []
        for identifier, (weight_decay, anchor) in contract.EXPECTED_CANDIDATE_PAIRS.items():
            sealed.append({
                "candidateId": identifier,
                "weightDecay": weight_decay,
                "anchorCoefficient": anchor,
                "trainableParameters": 49_600,
                "tensorSha256": {"tensor": "1" * 64},
                "tensorShapes": {"tensor": [1]},
                "tensorDtypes": {"tensor": "float32"},
                "tensorFloat32Base64": {"tensor": "AAAAAA=="},
            })
        grid = {"candidates": deepcopy(sealed)}
        tensor_seal = {"candidates": sealed}
        contract.require_grid_tensor_bindings(grid, tensor_seal)
        grid["candidates"][0]["tensorFloat32Base64"]["tensor"] = "AQAAAA=="
        with self.assertRaisesRegex(ValueError, "tensor binding"):
            contract.require_grid_tensor_bindings(grid, tensor_seal)

    def test_adapter_patch_byte_reconstructs_in_python_and_javascript(self) -> None:
        from benchmark.m4.train_adapter import AdapterCandidate, export_adapter_model

        candidate = AdapterCandidate(
            candidate_id="wd-0.003-anchor-0.01",
            weight_decay=0.003,
            anchor_coefficient=0.01,
            mean=np.zeros(384, dtype=np.float32),
            std=np.ones(384, dtype=np.float32),
            input_weight=np.zeros((64, 384), dtype=np.float32),
            input_bias=np.zeros(64, dtype=np.float32),
            output_weight=np.zeros((384, 64), dtype=np.float32),
            output_bias=np.zeros(384, dtype=np.float32),
        )
        recipe = json.loads((ROOT / "benchmark/m4/recipe.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "candidate.onnx"
            export_adapter_model(ROOT / "weights/prooflens-cf384.onnx", model, candidate, recipe)
            comparison, adapter_patch = contract.compare_adapter_models(
                ROOT / "weights/prooflens-cf384.onnx", model,
            )
            patch_path = root / "patch.json"
            comparison_path = root / "comparison.json"
            patch_path.write_bytes(canonical_json(adapter_patch, pretty=True))
            comparison_path.write_bytes(canonical_json(comparison, pretty=True))
            script = """
import { readFileSync } from 'node:fs';
import { reconstructM4Candidate, validateM4AdapterModel } from './scripts/m4-candidate-patch.mjs';
const base = readFileSync(process.env.M4_TEST_BASE);
const expected = readFileSync(process.env.M4_TEST_CANDIDATE);
const patch = JSON.parse(readFileSync(process.env.M4_TEST_PATCH, 'utf8'));
const comparison = JSON.parse(readFileSync(process.env.M4_TEST_COMPARISON, 'utf8'));
const rebuilt = reconstructM4Candidate({ baseBytes: base, adapterPatch: patch });
if (!rebuilt.equals(expected)) throw new Error('JavaScript reconstruction changed candidate bytes');
validateM4AdapterModel({ baseBytes: base, candidateBytes: rebuilt, comparison });
"""
            environment = {
                **os.environ,
                "M4_TEST_BASE": str(ROOT / "weights/prooflens-cf384.onnx"),
                "M4_TEST_CANDIDATE": str(model),
                "M4_TEST_PATCH": str(patch_path),
                "M4_TEST_COMPARISON": str(comparison_path),
            }
            subprocess.run(
                ["node", "--input-type=module", "--eval", script],
                cwd=ROOT,
                env=environment,
                check=True,
            )

    def test_training_command_has_no_h3_input(self) -> None:
        arguments = contract.expected_training_arguments()
        self.assertNotIn("h3", " ".join(arguments).lower())
        self.assertEqual(arguments[-2:], ["--output-dir", "benchmark/candidates/prooflens-cf384-m4"])
        self.assertLess(arguments.index("--m3-regression-manifest"), arguments.index("--m2-regression-manifest"))

    def test_fixture_identity_and_path_are_rejected_before_inference_dependencies(self) -> None:
        reviewed = json.loads((ROOT / "tests/fixtures/model-states/fixture-manifest.json").read_text())
        self.assertEqual(len(validate_previous_manifest(reviewed)), 2)
        extra = deepcopy(reviewed)
        extra["items"].append(deepcopy(extra["items"][0]))
        with self.assertRaisesRegex(ValueError, "exactly two"):
            validate_previous_manifest(extra)
        traversal = deepcopy(reviewed)
        traversal["items"][0]["asset"] = "../h3-met-holdout-v1/secret.png"
        with self.assertRaisesRegex(ValueError, "direct-child"):
            validate_previous_manifest(traversal)

    def test_fresh_feature_evidence_reopens_pixels_caches_and_exact_context(self) -> None:
        from benchmark.m4 import train_adapter

        pipeline = train_adapter.feature_pipeline
        run_id = "0123456789abcdef0123456789abcdef"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "benchmark/candidates/prooflens-cf384-m4"
            features = candidate / "features"
            training_root = root / "benchmark/data/m4-head"
            selector_root = root / "benchmark/data/m4-head"
            protocol_root = root / "benchmark/m4"
            evidence_root = root / "benchmark/evidence/m4"
            for path in (features, training_root, protocol_root, evidence_root):
                path.mkdir(parents=True, exist_ok=True)
            (protocol_root / "train_adapter.py").write_text("# frozen trainer\n")
            recipe_path = protocol_root / "recipe.json"
            recipe_path.write_text("{}\n")
            selection_summary_path = evidence_root / "selection-summary.json"
            selection_summary_path.write_text("{}\n")
            train_asset = training_root / "train.bin"
            selector_asset = selector_root / "selector.bin"
            train_asset.write_bytes(b"train")
            selector_asset.write_bytes(b"selector")
            train_manifest = training_root / "train-manifest.jsonl"
            selector_manifest = evidence_root / "validation-manifest.jsonl"
            train_manifest.write_text(json.dumps({
                "id": "train", "path": "train.bin", "imageSha256": contract.digest(train_asset),
                "label": 1, "source": "diffusiondb-stable-diffusion",
            }) + "\n")
            selector_manifest.write_text(json.dumps({
                "id": "selector", "path": "selector.bin", "imageSha256": contract.digest(selector_asset),
                "label": 0, "source": "british-library-plates",
            }) + "\n")
            recipe = {
                "expectedTraining": {"singleViewSources": [
                    "diffusiondb-stable-diffusion", "open-images-train",
                ]},
                "freshSelector": {
                    "manifest": "benchmark/evidence/m4/validation-manifest.jsonl",
                    "dataRoot": "benchmark/data/m4-head",
                },
                "regressions": [],
            }
            training_hash = pipeline.feature_configuration_hash(
                training=True,
                single_view_sources=frozenset(recipe["expectedTraining"]["singleViewSources"]),
                providers=("CPUExecutionProvider",),
                feature_batch_size=24,
            )
            evaluation_hash = pipeline.feature_configuration_hash(
                training=False,
                single_view_sources=frozenset(),
                providers=("CPUExecutionProvider",),
                feature_batch_size=24,
            )
            context = {
                "pipelineVersion": 1,
                "trainerSha256": contract.digest(protocol_root / "train_adapter.py"),
                "recipeSha256": contract.digest(recipe_path),
                "modelSha256": contract.UPSTREAM_MODEL_SHA256,
                "trainManifestSha256": contract.digest(train_manifest),
                "selectorManifestSha256": contract.digest(selector_manifest),
                "m3RegressionManifestSha256": "3" * 64,
                "m2RegressionManifestSha256": "4" * 64,
                "selectionSummarySha256": contract.digest(selection_summary_path),
                "featureConfigurationHashes": {"training": training_hash, "evaluation": evaluation_hash},
                "featureBatchSize": 24,
                "featureShardImages": 2_000,
                "singleViewSources": ["diffusiondb-stable-diffusion", "open-images-train"],
            }
            marker = {"schemaVersion": 1, "runId": run_id, "state": "extracting", "context": context}
            (candidate / "fresh-feature-run.json").write_bytes(marker_bytes(marker))
            train_items = pipeline.load_manifest(train_manifest, training_root)
            selector_items = pipeline.load_manifest(selector_manifest, selector_root)
            train_arrays = (
                np.zeros((1, 384), dtype=np.float32), np.asarray([1], dtype=np.float32),
                np.asarray([0], dtype=np.int64), np.asarray(["diffusiondb-stable-diffusion"]),
            )
            selector_arrays = (
                np.zeros((4, 384), dtype=np.float32), np.zeros(4, dtype=np.float32),
                np.arange(4, dtype=np.int64), np.asarray(["british-library-plates"] * 4),
            )
            rows = []
            for name, manifest, items, arrays, training, configuration in (
                ("train", train_manifest, train_items, train_arrays, True, training_hash),
                ("selector", selector_manifest, selector_items, selector_arrays, False, evaluation_hash),
            ):
                cache = features / f"{name}-00000.npz"
                item_ids_hash = __import__("hashlib").sha256(items[0].id.encode()).hexdigest()
                pipeline.save_feature_shard(
                    cache,
                    arrays,
                    manifest_hash=contract.digest(manifest),
                    model_hash=contract.UPSTREAM_MODEL_SHA256,
                    item_ids_hash=item_ids_hash,
                    feature_configuration_hash=configuration,
                    training=training,
                    fresh_feature_run_id=run_id,
                )
                rows.append({
                    "cache": f"benchmark/candidates/prooflens-cf384-m4/features/{name}-00000.npz",
                    "cacheSha256": contract.digest(cache),
                    "replacedCacheSha256": None,
                    "freshFeatureRunId": run_id,
                    "freshlyExtractedThisRun": True,
                    "freshlyExtractedThisProcess": True,
                    "items": 1,
                    "views": int(arrays[0].shape[0]),
                    "itemIdsSha256": item_ids_hash,
                    "featureConfigurationSha256": configuration,
                    "arraySha256": {
                        "features": pipeline.array_digest(arrays[0]),
                        "labels": pipeline.array_digest(arrays[1]),
                        "variants": pipeline.array_digest(arrays[2]),
                        "sources": pipeline.array_digest(arrays[3]),
                    },
                })
            summary = {
                "freshFeatureRunId": run_id,
                "freshFeatureMarkerSha256": pipeline.marker_sha256(marker),
                "featureConfigurationHashes": context["featureConfigurationHashes"],
                "trainManifestSha256": context["trainManifestSha256"],
                "selectorManifestSha256": context["selectorManifestSha256"],
                "m3RegressionManifestSha256": context["m3RegressionManifestSha256"],
                "m2RegressionManifestSha256": context["m2RegressionManifestSha256"],
                "featureShardEvidence": rows,
            }
            with (
                patch.object(contract, "ROOT", root),
                patch.object(contract, "CANDIDATE_DIR", candidate),
                patch.object(contract, "RECIPE_PATH", recipe_path),
                patch.object(contract, "SELECTION_SUMMARY_PATH", selection_summary_path),
            ):
                self.assertEqual(contract.validate_fresh_feature_evidence(
                    summary, recipe, marker_state="extracting", completed_regressions=0,
                ), marker)
                changed = deepcopy(summary)
                changed["featureShardEvidence"][0]["cacheSha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "feature-shard evidence"):
                    contract.validate_fresh_feature_evidence(
                        changed, recipe, marker_state="extracting", completed_regressions=0,
                    )
                bad_marker = deepcopy(marker)
                bad_marker["context"]["featureBatchSize"] = 1
                (candidate / "fresh-feature-run.json").write_bytes(marker_bytes(bad_marker))
                changed = deepcopy(summary)
                changed["freshFeatureMarkerSha256"] = pipeline.marker_sha256(bad_marker)
                with self.assertRaisesRegex(ValueError, "marker"):
                    contract.validate_fresh_feature_evidence(
                        changed, recipe, marker_state="extracting", completed_regressions=0,
                    )


if __name__ == "__main__":
    unittest.main()
