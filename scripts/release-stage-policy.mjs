export const LEGACY_SOURCE_COMMIT = "0771a9422b552e2023e5150fb6c8b4238b811a74";
export const LEGACY_FREEZE_COMMIT = "2bd0c4757f6059c57879414a5dba77629d66460e";
export const LEGACY_FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze.json";
export const LEGACY_FREEZE_SHA256 = "400fd2b7a7cd84b063f81799eaf3829f770220c41f58dff53e7caebd1a145c34";
export const FAILED_RECOVERY_SOURCE_COMMIT = "99861df575854511c685d7b8f90acdc7ed4e5923";
export const FAILED_RECOVERY_SOURCE_TREE = "204594f26118d8c1c3add9dfef3a6050949772e1";
export const FAILED_RECOVERY_ACTIONS_RUN_ID = 31846361076;
export const FAILED_RECOVERY_ACTIONS_RUN_URL =
  "https://github.com/baney75/prooflens/actions/runs/31846361076";
export const FAILED_RECOVERY_REASON = "ci-missing-inference-dependencies-before-v2-guard";
export const FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze-v2.json";
export const FREEZE_PATH_PREFIX = "benchmark/evidence/evaluation/pre-score-freeze";

export const FAILED_RECOVERY_REPAIR_PATHS = [
  "BENCHMARK.md",
  "README.md",
  "benchmark/evaluate.py",
  "benchmark/evaluation_contract.py",
  "benchmark/test_integrity_contracts.py",
  "benchmark/verify_evaluation_evidence.py",
  "benchmark/write_pre_score_freeze.py",
  "package.json",
  "scripts/check-benchmark-evidence.mjs",
  "scripts/check-pre-score-freeze.mjs",
  "scripts/check-pre-score-stage.mjs",
  "scripts/git-object-digest.mjs",
  "scripts/release-stage-policy.mjs",
  "scripts/run-static-verification.mjs",
  "scripts/test-git-object-digest.mjs",
  "scripts/test-release-stage-policy.mjs",
];

export const RECOVERY_REPAIR_PATHS = [
  "BENCHMARK.md",
  "README.md",
  "benchmark/evaluate.py",
  "benchmark/evaluation_contract.py",
  "benchmark/test_integrity_contracts.py",
  "benchmark/write_pre_score_freeze.py",
  "scripts/check-pre-score-freeze.mjs",
  "scripts/check-pre-score-stage.mjs",
  "scripts/release-stage-policy.mjs",
];

export const PROHIBITED_PRE_SCORE_PREFIXES = [
  "artifacts/browser-parity",
  "benchmark/evidence/evaluation/confirmatory/",
  "benchmark/evidence/evaluation/web-negative/",
];

export const PROHIBITED_PRE_SCORE_FILES = new Set([
  "benchmark/evidence/evaluation/replay-verification.json",
]);

export function classifyReleaseStage({ freezeExists, head, freezeCommit, legacyRecoverySource = false }) {
  const committed = /^[a-f0-9]{40}$/u.test(freezeCommit ?? "");
  if (!freezeExists && !committed) {
    return legacyRecoverySource ? "pre-score-recovery-source" : "pre-score-source";
  }
  if (!freezeExists && committed) return "final";
  if (!committed) {
    throw new Error("The pre-score freeze exists but has no unique committed addition");
  }
  return head === freezeCommit ? "pre-score-freeze" : "final";
}

export function isProhibitedPreScorePath(path) {
  return PROHIBITED_PRE_SCORE_FILES.has(path) ||
    isUnexpectedFreezeReceipt(path) ||
    PROHIBITED_PRE_SCORE_PREFIXES.some((prefix) => path.startsWith(prefix));
}

export function isUnexpectedFreezeReceipt(path) {
  return path.startsWith(FREEZE_PATH_PREFIX) &&
    path !== LEGACY_FREEZE_PATH && path !== FREEZE_PATH;
}

export function freezeReceiptAdditions(paths) {
  return [...new Set(paths.filter((path) => path.startsWith(FREEZE_PATH_PREFIX)))].sort();
}
