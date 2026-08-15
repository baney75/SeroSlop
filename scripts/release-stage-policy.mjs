export const LEGACY_SOURCE_COMMIT = "0771a9422b552e2023e5150fb6c8b4238b811a74";
export const LEGACY_SOURCE_TREE = "6b4575742ea847c747fe8bea216c3b2c7b357068";
export const LEGACY_FREEZE_COMMIT = "2bd0c4757f6059c57879414a5dba77629d66460e";
export const LEGACY_FREEZE_TREE = "ccf2de97024090245189ccbb7f308bce392565b9";
export const LEGACY_FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze.json";
export const LEGACY_FREEZE_SHA256 = "400fd2b7a7cd84b063f81799eaf3829f770220c41f58dff53e7caebd1a145c34";
export const FAILED_RECOVERY_SOURCE_COMMIT = "99861df575854511c685d7b8f90acdc7ed4e5923";
export const FAILED_RECOVERY_SOURCE_TREE = "204594f26118d8c1c3add9dfef3a6050949772e1";
export const SECOND_RECOVERY_SOURCE_COMMIT = "17124df0bf390c2c2c27583ae81f06b65ead2e3f";
export const SECOND_RECOVERY_SOURCE_TREE = "d8ecc3e0a60399e955b90e96f792baa30f65045c";
export const SECOND_FREEZE_COMMIT = "2757a4ff267d580a7dd8ad4918885441fa887f1b";
export const SECOND_FREEZE_TREE = "38111c60c95542a0048e9e49328a6a7e44149a95";
export const SECOND_FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze-v2.json";
export const SECOND_FREEZE_SHA256 = "0acd317dba772efd534b8900b627851fed0d04d54f814529b71c701465806017";
export const FAILED_EVALUATION_COMMIT = "45400803a19b967c8cae0bbf4817fe984aea349a";
export const FAILED_EVALUATION_TREE = "17463eae7285e6292c8f3aeb4fc3c0f1803ef6fa";
export const REPLACEMENT_SELECTION_COMMIT = "baaf3eb0b7a22f635d2ec6a3cb2496b9e76313b8";
export const REPLACEMENT_SELECTION_TREE = "2ddfec1f346eaa0f6bf0e797635af2121fe83866";
export const FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze-v3.json";
export const FREEZE_PATH_PREFIX = "benchmark/evidence/evaluation/pre-score-freeze";

export const FAILED_RECOVERY_REPAIR_PATHS = [
  "BENCHMARK.md", "README.md", "benchmark/evaluate.py", "benchmark/evaluation_contract.py",
  "benchmark/test_integrity_contracts.py", "benchmark/verify_evaluation_evidence.py",
  "benchmark/write_pre_score_freeze.py", "package.json", "scripts/check-benchmark-evidence.mjs",
  "scripts/check-pre-score-freeze.mjs", "scripts/check-pre-score-stage.mjs",
  "scripts/git-object-digest.mjs", "scripts/release-stage-policy.mjs",
  "scripts/run-static-verification.mjs", "scripts/test-git-object-digest.mjs",
  "scripts/test-release-stage-policy.mjs",
];

export const SECOND_RECOVERY_REPAIR_PATHS = [
  "BENCHMARK.md", "README.md", "benchmark/evaluate.py", "benchmark/evaluation_contract.py",
  "benchmark/test_integrity_contracts.py", "benchmark/write_pre_score_freeze.py",
  "scripts/check-pre-score-freeze.mjs", "scripts/check-pre-score-stage.mjs",
  "scripts/release-stage-policy.mjs",
];

export const FAILED_EVALUATION_PATHS = [
  "benchmark/evidence/evaluation/confirmatory/failed-evaluation.json",
  "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-complete.json",
  "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-original-predictions.jsonl",
  "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-screenshot-predictions.jsonl",
  "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-social-heavy-predictions.jsonl",
  "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-social-q75-predictions.jsonl",
  "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-summary.json",
];

export const REPLACEMENT_SELECTION_PATHS = [
  "benchmark/manifests/historical-perceptual-exclusions-v1.json.gz",
  "benchmark/manifests/replacement-v2-attribution.json",
  "benchmark/manifests/replacement-v2-perceptual-review.json",
  "benchmark/manifests/replacement-v2-selection.json",
  "benchmark/manifests/test-v2.jsonl",
  "benchmark/manifests/web-negative-v2.jsonl",
  "benchmark/recovery_v3/README.md",
  "benchmark/recovery_v3/build_historical_index.py",
  "benchmark/recovery_v3/legacy-test.jsonl.gz",
  "benchmark/recovery_v3/legacy-validation.jsonl.gz",
  "benchmark/recovery_v3/perceptual-review.json",
  "benchmark/recovery_v3/prepare.py",
  "benchmark/recovery_v3/recipe.json",
  "benchmark/recovery_v3/test_prepare.py",
  "benchmark/recovery_v3/verify.py",
];

export const RECOVERY_REPAIR_PATHS = [
  "BENCHMARK.md", "MODEL_CARD.md", "README.md", "benchmark/bootstrap_ci.py",
  "benchmark/bootstrap_fpr.py", "benchmark/evaluate.py", "benchmark/evaluation_contract.py",
  "benchmark/manifests/README.md", "benchmark/manifests/parity-ids-v2.json",
  "benchmark/prediction_contract.py", "benchmark/prepare_parity.py",
  "benchmark/recovery_v3/README.md", "benchmark/select_parity_ids.py",
  "benchmark/test_integrity_contracts.py", "benchmark/verify_evaluation_evidence.py",
  "benchmark/write_pre_score_freeze.py", "docs/ACCEPTANCE.md", "package.json",
  "scripts/check-benchmark-evidence.mjs", "scripts/check-pre-score-freeze.mjs",
  "scripts/check-pre-score-stage.mjs", "scripts/release-stage-policy.mjs",
  "scripts/test-release-stage-policy.mjs",
];

export const PROHIBITED_PRE_SCORE_PREFIXES = [
  "artifacts/",
  "benchmark/evidence/evaluation/confirmatory-v2/",
  "benchmark/evidence/evaluation/web-negative-v2/",
];

export const PROHIBITED_PRE_SCORE_FILES = new Set([
  "benchmark/evidence/evaluation/replay-verification-v2.json",
]);

export function classifyReleaseStage({ freezeExists, head, freezeCommit, legacyRecoverySource = false }) {
  const committed = /^[a-f0-9]{40}$/u.test(freezeCommit ?? "");
  if (!freezeExists && !committed) {
    return legacyRecoverySource ? "pre-score-recovery-source" : "pre-score-source";
  }
  if (!freezeExists && committed) return "final";
  if (!committed) throw new Error("The pre-score freeze exists but has no unique committed addition");
  return head === freezeCommit ? "pre-score-freeze" : "final";
}

export function isProhibitedPreScorePath(path) {
  return PROHIBITED_PRE_SCORE_FILES.has(path) || isUnexpectedFreezeReceipt(path) ||
    PROHIBITED_PRE_SCORE_PREFIXES.some((prefix) => path.startsWith(prefix));
}

export function isUnexpectedFreezeReceipt(path) {
  return path.startsWith(FREEZE_PATH_PREFIX) &&
    ![LEGACY_FREEZE_PATH, SECOND_FREEZE_PATH, FREEZE_PATH].includes(path);
}

export function freezeReceiptAdditions(paths) {
  return [...new Set(paths.filter((path) => path.startsWith(FREEZE_PATH_PREFIX)))].sort();
}
