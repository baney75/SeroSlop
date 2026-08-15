export const M4_BASE_COMMIT = "439b2481dc88a887f8317be669096495760fbeb1";
export const M4_BASE_TREE = "440931a595c87ca3d293f5a6f980c75169ddb899";
export const M4_FAILED_PROTOCOL_COMMIT = "82b06b49d44bedd54aba22a09d6a96b44e89d303";
export const M4_FAILED_PROTOCOL_TREE = "9e1de15031f83145ba40c8b1a2470b0833854fd8";
export const M4_PROTOCOL_RECOVERY_COMMIT = "6fed0d0ad0e9b9bdf50e17cc0463d8c845abc64b";
export const M4_PROTOCOL_RECOVERY_TREE = "96f8ccd610cb9362fff88bbaacb5a050937c259d";
export const M4_PUBLICATION_LOCK_PATH = "benchmark/evidence/m4/publication-lock.json";
export const M4_FAILURE_PATH = "benchmark/evidence/m4/failed-training-attempt-1.json";

// P: the complete executable protocol, committed before source materialization.
export const M4_PROTOCOL_EXPECTED = new Map([
  ["benchmark/m4/README.md", "A"],
  ["benchmark/m4/contracts.py", "A"],
  ["benchmark/m4/finalize.py", "A"],
  ["benchmark/m4/prepare.py", "A"],
  ["benchmark/m4/publication_contract.py", "A"],
  ["benchmark/m4/recipe.json", "A"],
  ["benchmark/m4/select_model_state_fixtures.py", "A"],
  ["benchmark/m4/source-locks.json", "A"],
  ["benchmark/m4/test_contracts.py", "A"],
  ["benchmark/m4/test_prepare.py", "A"],
  ["benchmark/m4/test_publication_contract.py", "A"],
  ["benchmark/m4/test_trainer_contracts.py", "A"],
  ["benchmark/m4/train_adapter.py", "A"],
  ["benchmark/m4/verify.py", "A"],
  ["package.json", "M"],
  ["scripts/check-m4-failure-stage.mjs", "A"],
  ["scripts/check-m4-protocol-stage.mjs", "A"],
  ["scripts/check-m4-publication-lock.mjs", "A"],
  ["scripts/check-m4-selection-evidence.mjs", "A"],
  ["scripts/check-m4-source-stage.mjs", "A"],
  ["scripts/check-m4-training-evidence.mjs", "A"],
  ["scripts/m4-candidate-patch.mjs", "A"],
  ["scripts/m4-failure-contract.mjs", "A"],
  ["scripts/m4-stage-policy.mjs", "A"],
  ["scripts/m4-training-contract.mjs", "A"],
  ["scripts/render-m4-public-docs.mjs", "A"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m4-failure-contract.mjs", "A"],
  ["scripts/test-m4-stage-policy.mjs", "A"],
  ["scripts/test-m4-training-contract.mjs", "A"],
]);

// P2: append-only recovery from the public P commit whose exact-head quality
// run 31899816870 failed before any source materialization because the static
// package scripts referenced an intentionally untracked local virtualenv.
export const M4_PROTOCOL_RECOVERY_EXPECTED = new Map([
  ["benchmark/verify-requirements.txt", "M"],
  ["package.json", "M"],
  ["scripts/check-m4-failure-stage.mjs", "M"],
  ["scripts/check-m4-protocol-stage.mjs", "M"],
  ["scripts/check-m4-publication-lock.mjs", "M"],
  ["scripts/check-m4-source-stage.mjs", "M"],
  ["scripts/check-m4-training-evidence.mjs", "M"],
  ["scripts/m4-stage-policy.mjs", "M"],
  ["scripts/run-benchmark-python.mjs", "A"],
  ["scripts/test-m4-stage-policy.mjs", "M"],
]);

// P3: append-only source-eligibility recovery after the exact P2 head was
// public and green. The locked British Library shards contain 19,060 raw rows,
// including 609 rows whose date is outside the predeclared 1800--1899 policy.
// This repair freezes that rejection surface before selection/materialization.
export const M4_DATE_RECOVERY_EXPECTED = new Map([
  ["benchmark/m4/README.md", "M"],
  ["benchmark/m4/contracts.py", "M"],
  ["benchmark/m4/prepare.py", "M"],
  ["benchmark/m4/recipe.json", "M"],
  ["benchmark/m4/test_contracts.py", "M"],
  ["benchmark/m4/test_prepare.py", "M"],
  ["benchmark/m4/verify.py", "M"],
  ["scripts/check-m4-failure-stage.mjs", "M"],
  ["scripts/check-m4-protocol-stage.mjs", "M"],
  ["scripts/check-m4-publication-lock.mjs", "M"],
  ["scripts/check-m4-source-stage.mjs", "M"],
  ["scripts/check-m4-training-evidence.mjs", "M"],
  ["scripts/m4-stage-policy.mjs", "M"],
  ["scripts/m4-training-contract.mjs", "M"],
  ["scripts/test-m4-failure-contract.mjs", "M"],
  ["scripts/test-m4-stage-policy.mjs", "M"],
]);

// S: pixel-free, score-free source evidence. Source pixels remain ignored.
export const M4_SOURCE_EXPECTED = new Map([
  ["benchmark/evidence/m4/attribution.json", "A"],
  ["benchmark/evidence/m4/british-source-index.json.gz", "A"],
  ["benchmark/evidence/m4/perceptual-review.json", "A"],
  ["benchmark/evidence/m4/rapidata-source-index.json.gz", "A"],
  ["benchmark/evidence/m4/rejects.jsonl.gz", "A"],
  ["benchmark/evidence/m4/selection-summary.json", "A"],
  ["benchmark/evidence/m4/train-manifest.jsonl.gz", "A"],
  ["benchmark/evidence/m4/validation-manifest.jsonl", "A"],
]);

export const M4_LOCK_EXPECTED = new Map([[M4_PUBLICATION_LOCK_PATH, "A"]]);

// E: the only terminal alternative to L/F. The diagnostic embeds the sealed
// candidate/selector/regression evidence needed for a cache-independent audit.
export const M4_FAILURE_EXPECTED = new Map([
  ["benchmark/evidence/m4/failed-selector-diagnostic-1.json", "A"],
  [M4_FAILURE_PATH, "A"],
]);

export const M4_PUBLICATION_EXPECTED = new Map([
  ["BENCHMARK.md", "M"],
  ["MODEL_CARD.md", "M"],
  ["README.md", "M"],
  ["benchmark/evidence/m4/calibration.json", "A"],
  ["benchmark/evidence/m4/candidate-grid.json", "A"],
  ["benchmark/evidence/m4/finalization-receipt.json", "A"],
  ["benchmark/evidence/m4/model-comparison.json", "A"],
  ["benchmark/evidence/m4/training-summary.json", "A"],
  ["model-lock.json", "M"],
  ["tests/fixtures/model-states/fixture-manifest.json", "M"],
  ["weights/README.md", "M"],
  ["weights/prooflens-cf384.onnx", "M"],
]);

export function classifyM4Stage({ protocolExists, selectionExists, failureExists, lockExists, trainingExists }) {
  if (!protocolExists && (selectionExists || failureExists || lockExists || trainingExists)) {
    throw new Error("M4 evidence cannot exist without the M4 protocol");
  }
  if (!protocolExists) return null;
  if (!selectionExists && (failureExists || lockExists || trainingExists)) {
    throw new Error("M4 failure, lock, or training evidence cannot exist without M4 source evidence");
  }
  if (!selectionExists) return "m4-protocol";
  if (failureExists && (lockExists || trainingExists)) {
    throw new Error("M4 failed evidence cannot coexist with pinned or final M4 evidence");
  }
  if (failureExists) return "m4-failed";
  if (trainingExists && !lockExists) {
    throw new Error("M4 training evidence cannot be published before its output lock");
  }
  if (trainingExists) return "m4-final";
  return lockExists ? "m4-pinned" : "m4-source";
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

export function matchesM4ProtocolRecoveryLineage({
  protocolParents,
  protocolRows,
  recoveryProtocolParents,
  recoveryProtocolRows,
  recoveryProtocolTree,
  failedProtocolParents,
  failedProtocolRows,
  failedProtocolTree,
  baseTree,
}) {
  return protocolParents.length === 1 && protocolParents[0] === M4_PROTOCOL_RECOVERY_COMMIT &&
    matchesExpectedRows(protocolRows, M4_DATE_RECOVERY_EXPECTED) &&
    recoveryProtocolParents.length === 1 && recoveryProtocolParents[0] === M4_FAILED_PROTOCOL_COMMIT &&
    recoveryProtocolTree === M4_PROTOCOL_RECOVERY_TREE &&
    matchesExpectedRows(recoveryProtocolRows, M4_PROTOCOL_RECOVERY_EXPECTED) &&
    failedProtocolParents.length === 1 && failedProtocolParents[0] === M4_BASE_COMMIT &&
    failedProtocolTree === M4_FAILED_PROTOCOL_TREE && baseTree === M4_BASE_TREE &&
    matchesExpectedRows(failedProtocolRows, M4_PROTOCOL_EXPECTED);
}
