export const M3_BASE_COMMIT = "0adbd55d8cdc25ad3d20e773a315ec57d14c7973";
export const M3_SOURCE_COMMIT = "2e6de2187d1cff5aea48e57ad9a30f15541fc4df";
export const M3_PUBLICATION_LOCK_PATH = "benchmark/evidence/m3/publication-lock.json";
export const M3_FAILURE_PATH = "benchmark/evidence/m3/failed-training-attempt-1.json";

// This map is finalized with the source-only commit before any M3 feature
// extraction or candidate fitting begins.
export const M3_SOURCE_EXPECTED = new Map([
  ["benchmark/evidence/m3/attribution.json", "A"],
  ["benchmark/evidence/m3/flux-source-index.json.gz", "A"],
  ["benchmark/evidence/m3/h3-met-holdout-manifest.jsonl", "A"],
  ["benchmark/evidence/m3/perceptual-review.json", "A"],
  ["benchmark/evidence/m3/rejects.jsonl.gz", "A"],
  ["benchmark/evidence/m3/selection-summary.json", "A"],
  ["benchmark/evidence/m3/train-manifest.jsonl.gz", "A"],
  ["benchmark/evidence/m3/training-evaluation-perceptual-review.json", "A"],
  ["benchmark/evidence/m3/validation-manifest.jsonl", "A"],
  ["benchmark/m3/README.md", "A"],
  ["benchmark/m3/contracts.py", "A"],
  ["benchmark/m3/finalize.py", "A"],
  ["benchmark/m3/prepare.py", "A"],
  ["benchmark/m3/publication_contract.py", "A"],
  ["benchmark/m3/recipe.json", "A"],
  ["benchmark/m3/select_model_state_fixtures.py", "A"],
  ["benchmark/m3/test_prepare.py", "A"],
  ["benchmark/m3/test_publication_contract.py", "A"],
  ["benchmark/m3/test_trainer_contracts.py", "A"],
  ["benchmark/m3/verify.py", "A"],
  ["benchmark/manifests/met-development-probe-v1.jsonl", "A"],
  ["benchmark/modern/train_rehead.py", "M"],
  ["package.json", "M"],
  ["scripts/check-m3-selection-evidence.mjs", "A"],
  ["scripts/check-m3-source-stage.mjs", "A"],
  ["scripts/check-m3-publication-lock.mjs", "A"],
  ["scripts/check-m3-training-evidence.mjs", "A"],
  ["scripts/check-m2-training-evidence.mjs", "M"],
  ["scripts/m3-candidate-patch.mjs", "A"],
  ["scripts/m3-stage-policy.mjs", "A"],
  ["scripts/m3-training-contract.mjs", "A"],
  ["scripts/render-m3-public-docs.mjs", "A"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m3-stage-policy.mjs", "A"],
  ["scripts/test-m3-training-contract.mjs", "A"],
]);

export const M3_LOCK_EXPECTED = new Map([
  [M3_PUBLICATION_LOCK_PATH, "A"],
]);

export const M3_FAILURE_EXPECTED = new Map([
  ["benchmark/evidence/m3/failed-selector-diagnostic-1.json", "A"],
  [M3_FAILURE_PATH, "A"],
  ["benchmark/m3/diagnose_failed_training.py", "A"],
  ["benchmark/m3/test_failure_diagnostic.py", "A"],
  ["package.json", "M"],
  ["scripts/check-m2-training-evidence.mjs", "M"],
  ["scripts/check-m3-failure-stage.mjs", "A"],
  ["scripts/m3-failure-contract.mjs", "A"],
  ["scripts/m3-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m3-failure-contract.mjs", "A"],
  ["scripts/test-m3-stage-policy.mjs", "M"],
]);

export const M3_PUBLICATION_EXPECTED = new Map([
  ["BENCHMARK.md", "M"],
  ["MODEL_CARD.md", "M"],
  ["README.md", "M"],
  ["benchmark/evidence/m3/calibration.json", "A"],
  ["benchmark/evidence/m3/candidate-grid.json", "A"],
  ["benchmark/evidence/m3/finalization-receipt.json", "A"],
  ["benchmark/evidence/m3/model-comparison.json", "A"],
  ["benchmark/evidence/m3/training-summary.json", "A"],
  ["model-lock.json", "M"],
  ["tests/fixtures/model-states/fixture-manifest.json", "M"],
  ["weights/README.md", "M"],
  ["weights/prooflens-cf384.onnx", "M"],
]);

export function classifyM3Stage({ selectionExists, failureExists, lockExists, trainingExists }) {
  if (!selectionExists && (failureExists || lockExists || trainingExists)) {
    throw new Error("M3 failure, lock, or training evidence cannot exist without M3 selection evidence");
  }
  if (!selectionExists) return null;
  if (failureExists && (lockExists || trainingExists)) {
    throw new Error("M3 failed evidence cannot coexist with pinned or final M3 evidence");
  }
  if (failureExists) return "m3-failed";
  if (trainingExists && !lockExists) {
    throw new Error("M3 training evidence cannot be published before its output lock");
  }
  if (trainingExists) return "m3-final";
  return lockExists ? "m3-pinned" : "m3-source";
}

export function matchesExpectedRows(rows, expected) {
  if (rows.length !== expected.size) return false;
  const seen = new Set();
  for (const [pathname, status] of rows) {
    if (seen.has(pathname) || expected.get(pathname) !== status) return false;
    seen.add(pathname);
  }
  return seen.size === expected.size && [...expected.keys()].every((pathname) => seen.has(pathname));
}
