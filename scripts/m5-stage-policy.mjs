export const M5_BASE_SOURCE_COMMIT = "5ab375fad2a744620b6ec75f09e6153c8a409049";
export const M5_BASE_SOURCE_TREE = "fc0afc8a746f3f41c29bbd8713f309856d2bdc53";
export const M5_ORIGINAL_PROTOCOL_COMMIT = "89bd1c833abbaa23195d45cd9a82fc3e117bad88";
export const M5_ORIGINAL_PROTOCOL_TREE = "d16f4d1033416a2935427dc6ced6ceb4ffea4674";
export const M5_P2_COMMIT = "1c4ac973785f937fa9023018863941e6d89d8693";
export const M5_P2_TREE = "a56caae4291e275029076417fb2111be76b07a41";
export const M5_FAILED_SOURCE_COMMIT = "fba4b51ef5073e0a189ab6baaaf155fccf785dc6";
export const M5_FAILED_SOURCE_TREE = "9176b515dfe87a5d5136f0103ef2f8b81fab2938";
export const M5_CI_RECOVERY_COMMIT = "8e06b822f9d9fb4a3dd8adb2e4bf6dc2f19c51bb";
export const M5_CI_RECOVERY_TREE = "798e7c2ec55abd82de5bf0927f4c988f2e1c9a7c";
export const M5_P4_COMMIT = "99505818a95e494b3ad8ed10fbb22bff0ce798da";
export const M5_P4_TREE = "0ff0eff3b518397dc2c721099ed4597fc15eb17f";
export const M5_P4_AUTHORIZATION_SHA256 = "d1fcdc2fab96873d3860abfeb71d1edd74b14bfb080b1feadd837ce8d4e011d3";
export const M5_RUNTIME_RECOVERY_COMMIT = "f171aa983a1bd43737f5b931564e3c5b7e5798c6";
export const M5_RUNTIME_RECOVERY_TREE = "2983b3ab90d0a1097b457bb2d217a810f1067fd6";
export const M5_RUNTIME_AUTHORIZATION_COMMIT = "e75b16f94718c24ac98311bd6859f10f27b5f2bd";
export const M5_RUNTIME_AUTHORIZATION_TREE = "143e5e6c0ad694406d7d39da22cc3d4d56e743bd";
export const M5_RUNTIME_AUTHORIZATION_SHA256 = "eeee532d699705faef1e35d49748c8264387880e0e573d2b8c61e412944c9ce9";
export const M5_RUN_AUTHORIZATION_PATH = "benchmark/evidence/m5/run-authorization.json";
export const M5_RUNTIME_AUTHORIZATION_PATH = "benchmark/evidence/m5/runtime-recovery-authorization.json";
export const M5_RUNPOD_ENV_AUTHORIZATION_PATH = "benchmark/evidence/m5/runpod-environment-authorization.json";
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

export const M5_PROTOCOL_RECOVERY_EXPECTED = new Map([
  ["benchmark/m5/README.md", "M"],
  ["benchmark/m5/contracts.py", "M"],
  ["benchmark/m5/evaluate_locked.py", "M"],
  ["benchmark/m5/evaluate_large_synthetic.py", "M"],
  ["benchmark/m5/finalize.py", "M"],
  ["benchmark/m5/large_synthetic.py", "M"],
  ["benchmark/m5/recipe.json", "M"],
  ["benchmark/m5/test_contracts.py", "M"],
  ["benchmark/m5/train_gpu.py", "M"],
  ["package.json", "M"],
  ["scripts/check-m5-failure-stage.mjs", "M"],
  ["scripts/check-m5-final-stage.mjs", "M"],
  ["scripts/check-m5-large-source-stage.mjs", "M"],
  ["scripts/check-m5-protocol-stage.mjs", "M"],
  ["scripts/check-m5-selection-lock.mjs", "M"],
  ["scripts/m5-stage-policy.mjs", "M"],
  ["scripts/m5-training-contract.mjs", "M"],
  ["scripts/test-m5-stage-policy.mjs", "M"],
  ["scripts/test-m5-training-contract.mjs", "M"],
]);

// P3 is the source-recovery commit.  Its exact surface is intentionally
// pinned here; the receipt-only P4 child is the only later authorization
// surface permitted before runtime.
export const M5_SOURCE_RECOVERY_EXPECTED = new Map([
  ["benchmark/m5/README.md", "M"],
  ["benchmark/m5/contracts.py", "M"],
  ["benchmark/m5/evaluate_locked.py", "M"],
  ["benchmark/m5/evaluate_large_synthetic.py", "M"],
  ["benchmark/m5/finalize.py", "M"],
  ["benchmark/m5/large_synthetic.py", "M"],
  ["benchmark/m5/test_contracts.py", "M"],
  ["benchmark/m5/train_gpu.py", "M"],
  ["package.json", "M"],
  ["scripts/check-m5-failure-stage.mjs", "M"],
  ["scripts/check-m5-final-stage.mjs", "M"],
  ["scripts/check-m5-large-source-stage.mjs", "M"],
  ["scripts/check-m5-protocol-stage.mjs", "M"],
  ["scripts/check-m5-selection-lock.mjs", "M"],
  ["scripts/m5-stage-policy.mjs", "M"],
  ["scripts/test-m5-stage-policy.mjs", "M"],
  ["scripts/m5-run-authorization.mjs", "A"],
  ["scripts/m5-preexec-bootstrap.py", "A"],
  ["scripts/m5-python-launch.mjs", "A"],
  ["scripts/m5-runpod-launch.sh", "A"],
  ["scripts/m5-safe-git.mjs", "A"],
  ["scripts/m5_node_bootstrap.py", "A"],
  ["scripts/check-m5-run-authorization-stage.mjs", "A"],
  ["scripts/check-m5-authorized-chain.mjs", "A"],
  ["scripts/check-m5-source-recovery-stage.mjs", "A"],
  ["scripts/run-static-verification.mjs", "M"],
]);

// The first P3 packet is immutable public history. Its GitHub Actions run
// failed only because a host-specific test expected the macOS exit code on
// Linux. The sole authorized repair is this exact append-only surface.
export const M5_SOURCE_CI_RECOVERY_EXPECTED = new Map([
  ["benchmark/m5/README.md", "M"],
  ["benchmark/m5/test_contracts.py", "M"],
  ["benchmark/m5/train_gpu.py", "M"],
  ["scripts/check-m5-authorized-chain.mjs", "M"],
  ["scripts/check-m5-source-recovery-stage.mjs", "M"],
  ["scripts/m5-run-authorization.mjs", "M"],
  ["scripts/m5-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m5-stage-policy.mjs", "M"],
]);

// P4 was correctly authorized but its Python runtime history map omitted the
// already-frozen pre-exec bootstrap row.  The only permitted runtime repair is
// this exact append-only child; a second receipt must authorize these bytes.
export const M5_RUNTIME_RECOVERY_EXPECTED = new Map([
  ["benchmark/m5/README.md", "M"],
  ["benchmark/m5/contracts.py", "M"],
  ["benchmark/m5/test_contracts.py", "M"],
  ["benchmark/m5/train_gpu.py", "M"],
  ["scripts/check-m5-authorized-chain.mjs", "M"],
  ["scripts/check-m5-run-authorization-stage.mjs", "M"],
  ["scripts/check-m5-source-recovery-stage.mjs", "M"],
  ["scripts/m5-run-authorization.mjs", "M"],
  ["scripts/m5-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m5-stage-policy.mjs", "M"],
]);

// The runtime recovery correctly repaired lineage, but the trusted bootstrap's
// minimal environment omitted RunPod's own Pod ID.  This exact child may pass
// only that single provider variable from PID 1; a third receipt authorizes it.
export const M5_RUNPOD_ENV_RECOVERY_EXPECTED = new Map([
  ["benchmark/m5/README.md", "M"],
  ["benchmark/m5/contracts.py", "M"],
  ["benchmark/m5/test_contracts.py", "M"],
  ["benchmark/m5/train_gpu.py", "M"],
  ["scripts/check-m5-authorized-chain.mjs", "M"],
  ["scripts/check-m5-run-authorization-stage.mjs", "M"],
  ["scripts/check-m5-source-recovery-stage.mjs", "M"],
  ["scripts/m5-preexec-bootstrap.py", "M"],
  ["scripts/m5-run-authorization.mjs", "M"],
  ["scripts/m5-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m5-stage-policy.mjs", "M"],
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

export function matchesM5ProtocolLineage({
  recoveryParents,
  recoveryRows,
  originalTree,
  originalParents,
  originalRows,
  baseTree,
}) {
  return recoveryParents.length === 1 &&
    recoveryParents[0] === M5_ORIGINAL_PROTOCOL_COMMIT &&
    matchesExpectedRows(recoveryRows, M5_PROTOCOL_RECOVERY_EXPECTED) &&
    originalTree === M5_ORIGINAL_PROTOCOL_TREE &&
    originalParents.length === 1 &&
    originalParents[0] === M5_BASE_SOURCE_COMMIT &&
    matchesExpectedRows(originalRows, M5_PROTOCOL_EXPECTED) &&
    baseTree === M5_BASE_SOURCE_TREE;
}

export function matchesM5SourceRecoveryCommit({ sourceParents, sourceRows, failedSourceTree, failedSourceParents, failedSourceRows }) {
  return sourceParents.length === 1 && sourceParents[0] === M5_FAILED_SOURCE_COMMIT &&
    matchesExpectedRows(sourceRows, M5_SOURCE_CI_RECOVERY_EXPECTED) &&
    failedSourceTree === M5_FAILED_SOURCE_TREE &&
    failedSourceParents.length === 1 && failedSourceParents[0] === M5_P2_COMMIT &&
    matchesExpectedRows(failedSourceRows, M5_SOURCE_RECOVERY_EXPECTED);
}

export function matchesM5RuntimeRecoveryCommit({ runtimeParents, runtimeRows, p4Tree, p4Parents, p4Rows, sourceCommit, sourceParents, sourceRows, failedSourceTree, failedSourceParents, failedSourceRows }) {
  return runtimeParents.length === 1 && runtimeParents[0] === M5_P4_COMMIT &&
    matchesExpectedRows(runtimeRows, M5_RUNTIME_RECOVERY_EXPECTED) &&
    p4Tree === M5_P4_TREE &&
    p4Parents.length === 1 && p4Parents[0] === sourceCommit &&
    matchesExpectedRows(p4Rows, new Map([[M5_RUN_AUTHORIZATION_PATH, "A"]])) &&
    sourceCommit === M5_CI_RECOVERY_COMMIT &&
    matchesM5SourceRecoveryCommit({ sourceParents, sourceRows, failedSourceTree, failedSourceParents, failedSourceRows });
}

export function matchesM5AuthorizedChain({ authorizationCommit, authorizationParents, authorizationRows, environmentCommit, environmentParents, environmentRows, runtimeAuthorizationTree, runtimeAuthorizationParents, runtimeAuthorizationRows, runtimeCommit, runtimeParents, runtimeRows, p4Tree, p4Parents, p4Rows, sourceCommit, sourceParents, sourceRows, failedSourceTree, failedSourceParents, failedSourceRows }) {
  return authorizationCommit && authorizationParents.length === 1 && authorizationParents[0] === environmentCommit &&
    matchesExpectedRows(authorizationRows, new Map([[M5_RUNPOD_ENV_AUTHORIZATION_PATH, "A"]])) &&
    matchesM5RunpodEnvironmentRecoveryCommit({ environmentParents, environmentRows, runtimeAuthorizationTree, runtimeAuthorizationParents, runtimeAuthorizationRows, runtimeCommit, runtimeParents, runtimeRows, p4Tree, p4Parents, p4Rows, sourceCommit, sourceParents, sourceRows, failedSourceTree, failedSourceParents, failedSourceRows });
}

export function matchesM5RunpodEnvironmentRecoveryCommit({ environmentParents, environmentRows, runtimeAuthorizationTree, runtimeAuthorizationParents, runtimeAuthorizationRows, runtimeCommit, runtimeParents, runtimeRows, p4Tree, p4Parents, p4Rows, sourceCommit, sourceParents, sourceRows, failedSourceTree, failedSourceParents, failedSourceRows }) {
  return environmentParents.length === 1 && environmentParents[0] === M5_RUNTIME_AUTHORIZATION_COMMIT &&
    matchesExpectedRows(environmentRows, M5_RUNPOD_ENV_RECOVERY_EXPECTED) &&
    runtimeAuthorizationTree === M5_RUNTIME_AUTHORIZATION_TREE &&
    runtimeAuthorizationParents.length === 1 && runtimeAuthorizationParents[0] === runtimeCommit &&
    matchesExpectedRows(runtimeAuthorizationRows, new Map([[M5_RUNTIME_AUTHORIZATION_PATH, "A"]])) &&
    runtimeCommit === M5_RUNTIME_RECOVERY_COMMIT &&
    matchesM5RuntimeRecoveryCommit({ runtimeParents, runtimeRows, p4Tree, p4Parents, p4Rows, sourceCommit, sourceParents, sourceRows, failedSourceTree, failedSourceParents, failedSourceRows });
}

export function classifyM5Stage({ protocolExists, lockExists, failureExists, largeSourceLockExists, finalExists, sourceRecoveryExists = false, authorizationExists = false }) {
  if (!protocolExists && (lockExists || failureExists || largeSourceLockExists || finalExists)) {
    throw new Error("M5 evidence cannot exist without the M5 protocol");
  }
  if (!protocolExists) return null;
  if (failureExists && (lockExists || largeSourceLockExists || finalExists)) {
    throw new Error("M5 failure cannot coexist with a selection lock or final evidence");
  }
  if (largeSourceLockExists && !lockExists) throw new Error("M5 100K source lock requires the public selection lock");
  if (finalExists && !largeSourceLockExists) throw new Error("M5 final evidence requires the public 100K source lock");
  if ((lockExists || failureExists || largeSourceLockExists || finalExists) && !authorizationExists) {
    throw new Error("M5 result evidence requires the public run authorization");
  }
  if (failureExists) return "m5-failed";
  if (finalExists) return "m5-final";
  if (largeSourceLockExists) return "m5-eval-locked";
  if (lockExists) return "m5-pinned";
  if (sourceRecoveryExists) return "m5-source-recovery";
  if (authorizationExists) return "m5-authorized";
  return "m5-protocol";
}
