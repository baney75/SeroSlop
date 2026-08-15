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
assert.equal(isProhibitedPreScorePath("artifacts/chrome-e2e-wasm.json"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/confirmatory-v2/predictions.jsonl"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/web-negative-v2/wilson.json"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/replay-verification-v2.json"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/confirmatory/failed-evaluation.json"), false);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/validation/summary.json"), false);
assert.equal(isUnexpectedFreezeReceipt("benchmark/evidence/evaluation/pre-score-freeze.json"), false);
assert.equal(isUnexpectedFreezeReceipt("benchmark/evidence/evaluation/pre-score-freeze-v2.json"), false);
assert.equal(isUnexpectedFreezeReceipt("benchmark/evidence/evaluation/pre-score-freeze-v3.json"), false);
assert.equal(isUnexpectedFreezeReceipt("benchmark/evidence/evaluation/pre-score-freeze-v4.json"), true);
assert.equal(isUnexpectedFreezeReceipt("benchmark/evidence/evaluation/pre-score-freezes/alternate.json"), true);
assert.deepEqual(freezeReceiptAdditions([
  "benchmark/evidence/evaluation/pre-score-freeze.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v2.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v3.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v4.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v4.json",
  "README.md",
]), [
  "benchmark/evidence/evaluation/pre-score-freeze-v2.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v3.json",
  "benchmark/evidence/evaluation/pre-score-freeze-v4.json",
  "benchmark/evidence/evaluation/pre-score-freeze.json",
]);

console.log(JSON.stringify({ cases: 18, policy: "pass" }));
