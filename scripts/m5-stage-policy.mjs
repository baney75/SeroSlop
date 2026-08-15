export const M5_BASE_SOURCE_COMMIT = "5ab375fad2a744620b6ec75f09e6153c8a409049";
export const M5_BASE_SOURCE_TREE = "fc0afc8a746f3f41c29bbd8713f309856d2bdc53";
export const M5_SELECTION_LOCK_PATH = "benchmark/evidence/m5/selection-lock.json";
export const M5_FAILURE_PATH = "benchmark/evidence/m5/failed-training-attempt-1.json";
export const M5_FINAL_RECEIPT_PATH = "benchmark/evidence/m5/finalization-receipt.json";
export const M5_LARGE_MANIFEST_PATH = "benchmark/evidence/m5/large-synthetic/manifest.jsonl.gz";
export const M5_LARGE_BATCHES_PATH = "benchmark/evidence/m5/large-synthetic/batches.json";
export const M5_LARGE_SOURCE_LOCK_PATH = "benchmark/evidence/m5/large-synthetic/source-lock.json";
export const M5_LARGE_ATTRIBUTION_PATH = "benchmark/evidence/m5/large-synthetic/attribution.json";
export const M5_LARGE_EVALUATION_PATH = "benchmark/evidence/m5/large-synthetic-evaluation.json";

export const M5_PROTOCOL_EXPECTED = new Map([
  ["DESIGN.md", "M"],
  ["MODEL_CARD.md", "M"],
  ["PRIVACY.md", "M"],
  ["README.md", "M"],
  ["SECURITY.md", "M"],
  ["THIRD_PARTY_NOTICES.md", "M"],
  ["benchmark/m5/README.md", "A"],
  ["benchmark/m5/contracts.py", "A"],
  ["benchmark/m5/evaluate_locked.py", "A"],
  ["benchmark/m5/evaluate_large_synthetic.py", "A"],
  ["benchmark/m5/finalize.py", "A"],
  ["benchmark/m5/large_synthetic.py", "A"],
  ["benchmark/m5/recipe.json", "A"],
  ["benchmark/m5/runpod-requirements.txt", "A"],
  ["benchmark/m5/test_contracts.py", "A"],
  ["benchmark/m5/train_gpu.py", "A"],
  ["package.json", "M"],
  ["scripts/check-m5-failure-stage.mjs", "A"],
  ["scripts/check-m5-final-stage.mjs", "A"],
  ["scripts/check-m5-large-source-stage.mjs", "A"],
  ["scripts/check-m5-protocol-stage.mjs", "A"],
  ["scripts/check-m5-selection-lock.mjs", "A"],
  ["scripts/check-package.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m5-stage-policy.mjs", "A"],
  ["scripts/m5-training-contract.mjs", "A"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m5-stage-policy.mjs", "A"],
  ["scripts/test-m5-training-contract.mjs", "A"],
  ["src/content.ts", "M"],
  ["src/popup.ts", "M"],
  ["src/setup.ts", "M"],
  ["src/shared/model-spec.ts", "M"],
  ["src/static/manifest.json", "M"],
  ["src/static/offscreen.html", "M"],
  ["src/static/popup.html", "M"],
  ["src/static/setup.html", "M"],
  ["tests/model-spec.test.ts", "M"],
]);

export const M5_LOCK_EXPECTED = new Map([[M5_SELECTION_LOCK_PATH, "A"]]);
export const M5_FAILURE_EXPECTED = new Map([[M5_FAILURE_PATH, "A"]]);
export const M5_LARGE_SOURCE_EXPECTED = new Map([
  [M5_LARGE_ATTRIBUTION_PATH, "A"],
  [M5_LARGE_BATCHES_PATH, "A"],
  [M5_LARGE_MANIFEST_PATH, "A"],
  [M5_LARGE_SOURCE_LOCK_PATH, "A"],
]);

export const M5_FINAL_EXPECTED = new Map([
  ["BENCHMARK.md", "M"],
  ["MODEL_CARD.md", "M"],
  ["README.md", "M"],
  ["benchmark/evidence/m5/calibration.json", "A"],
  [M5_LARGE_EVALUATION_PATH, "A"],
  [M5_FINAL_RECEIPT_PATH, "A"],
  ["benchmark/evidence/m5/model-comparison.json", "A"],
  ["benchmark/evidence/m5/regression-summary.json", "A"],
  ["benchmark/evidence/m5/training-summary.json", "A"],
  ["model-lock.json", "M"],
  ["tests/fixtures/model-states/fixture-manifest.json", "M"],
  ["weights/README.md", "M"],
  ["weights/prooflens-cf384.onnx", "M"],
]);

export function matchesExpectedRows(rows, expected) {
  if (rows.length !== expected.size) return false;
  const seen = new Set();
  for (const [pathname, status] of rows) {
    if (seen.has(pathname) || expected.get(pathname) !== status) return false;
    seen.add(pathname);
  }
  return seen.size === expected.size && [...expected.keys()].every((pathname) => seen.has(pathname));
}

export function matchesM5ProtocolCommit({ parents, rows, parentTree }) {
  return parents.length === 1 &&
    parents[0] === M5_BASE_SOURCE_COMMIT &&
    parentTree === M5_BASE_SOURCE_TREE &&
    matchesExpectedRows(rows, M5_PROTOCOL_EXPECTED);
}

export function classifyM5Stage({ protocolExists, lockExists, failureExists, largeSourceLockExists, finalExists }) {
  if (!protocolExists && (lockExists || failureExists || largeSourceLockExists || finalExists)) {
    throw new Error("M5 evidence cannot exist without the M5 protocol");
  }
  if (!protocolExists) return null;
  if (failureExists && (lockExists || largeSourceLockExists || finalExists)) {
    throw new Error("M5 failure cannot coexist with a selection lock or final evidence");
  }
  if (largeSourceLockExists && !lockExists) throw new Error("M5 100K source lock requires the public selection lock");
  if (finalExists && !largeSourceLockExists) throw new Error("M5 final evidence requires the public 100K source lock");
  if (failureExists) return "m5-failed";
  if (finalExists) return "m5-final";
  if (largeSourceLockExists) return "m5-eval-locked";
  if (lockExists) return "m5-pinned";
  return "m5-protocol";
}
