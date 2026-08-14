export const FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze.json";

export const PROHIBITED_PRE_SCORE_PREFIXES = [
  "artifacts/browser-parity",
  "benchmark/evidence/evaluation/confirmatory/",
  "benchmark/evidence/evaluation/web-negative/",
];

export const PROHIBITED_PRE_SCORE_FILES = new Set([
  "benchmark/evidence/evaluation/replay-verification.json",
]);

export function classifyReleaseStage({ freezeExists, head, freezeCommit }) {
  const committed = /^[a-f0-9]{40}$/u.test(freezeCommit ?? "");
  if (!freezeExists && !committed) return "pre-score-source";
  if (!freezeExists && committed) return "final";
  if (!committed) {
    throw new Error("The pre-score freeze exists but has no unique committed addition");
  }
  return head === freezeCommit ? "pre-score-freeze" : "final";
}

export function isProhibitedPreScorePath(path) {
  return PROHIBITED_PRE_SCORE_FILES.has(path) ||
    PROHIBITED_PRE_SCORE_PREFIXES.some((prefix) => path.startsWith(prefix));
}
