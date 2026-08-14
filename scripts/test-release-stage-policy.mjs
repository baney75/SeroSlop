import assert from "node:assert/strict";
import {
  classifyReleaseStage,
  freezeReceiptAdditions,
  isProhibitedPreScorePath,
  isUnexpectedFreezeReceipt,
} from "./release-stage-policy.mjs";

const source = "a".repeat(40);
const freeze = "b".repeat(40);
const final = "c".repeat(40);

assert.equal(classifyReleaseStage({ freezeExists: false, head: source }), "pre-score-source");
assert.equal(classifyReleaseStage({ freezeExists: false, head: source, legacyRecoverySource: true }),
  "pre-score-recovery-source");
assert.equal(classifyReleaseStage({ freezeExists: true, head: freeze, freezeCommit: freeze }), "pre-score-freeze");
assert.equal(classifyReleaseStage({ freezeExists: true, head: final, freezeCommit: freeze }), "final");
assert.equal(classifyReleaseStage({ freezeExists: false, head: final, freezeCommit: freeze }), "final");
assert.throws(() => classifyReleaseStage({ freezeExists: true, head: freeze }), /unique committed addition/u);
assert.equal(isProhibitedPreScorePath("artifacts/browser-parity.json"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/confirmatory/predictions.jsonl"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/web-negative/wilson.json"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/replay-verification.json"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/pre-score-freeze-v3.json"), true);
assert.equal(isProhibitedPreScorePath("artifacts/chrome-e2e-wasm.json"), false);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/validation/summary.json"), false);
assert.equal(isUnexpectedFreezeReceipt("benchmark/evidence/evaluation/pre-score-freeze.json"), false);
assert.equal(isUnexpectedFreezeReceipt("benchmark/evidence/evaluation/pre-score-freeze-v2.json"), false);
assert.equal(isUnexpectedFreezeReceipt("benchmark/evidence/evaluation/pre-score-freezes/alternate.json"), true);
assert.deepEqual(freezeReceiptAdditions([
  "benchmark/evidence/evaluation/pre-score-freeze.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v2.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v3.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v3.json",
  "README.md",
]), [
  "benchmark/evidence/evaluation/pre-score-freeze-v2.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v3.json",
  "benchmark/evidence/evaluation/pre-score-freeze.json",
]);

console.log(JSON.stringify({ cases: 17, policy: "pass" }));
