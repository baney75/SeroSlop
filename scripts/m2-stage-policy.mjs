export const M2_RECOVERY_COMMIT = "31b2ea718016b9aac13e62de64e3457d2e2b3008";

export const M2_FINALIZER_SOURCE_EXPECTED = new Map([
  ["benchmark/finalize_training_evidence.py", "M"],
  ["benchmark/test_integrity_contracts.py", "M"],
  ["package.json", "M"],
  ["scripts/check-large-training-evidence.mjs", "M"],
  ["scripts/check-m2-source-stage.mjs", "M"],
  ["scripts/check-m2-training-evidence.mjs", "A"],
  ["scripts/m2-stage-policy.mjs", "A"],
  ["scripts/m2-training-contract.mjs", "A"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m2-stage-policy.mjs", "A"],
  ["scripts/test-m2-training-contract.mjs", "A"],
]);

export const M2_PUBLICATION_EXPECTED = new Map([
  ["benchmark/evidence/m2/calibration.json", "A"],
  ["benchmark/evidence/m2/candidate-grid.json", "A"],
  ["benchmark/evidence/m2/finalization-receipt.json", "A"],
  ["benchmark/evidence/m2/model-comparison.json", "A"],
  ["benchmark/evidence/m2/training-summary.json", "A"],
  ["model-lock.json", "M"],
  ["weights/README.md", "M"],
  ["weights/prooflens-cf384.onnx", "M"],
]);

export function classifyM2Stage({ selectionExists, trainingExists }) {
  if (trainingExists && !selectionExists) {
    throw new Error("M2 training evidence cannot exist without M2 selection evidence");
  }
  if (!selectionExists) return null;
  return trainingExists ? "m2-final" : "m2-source";
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
