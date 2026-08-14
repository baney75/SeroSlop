import assert from "node:assert/strict";
import { classifyReleaseStage, isProhibitedPreScorePath } from "./release-stage-policy.mjs";

const source = "a".repeat(40);
const freeze = "b".repeat(40);
const final = "c".repeat(40);

assert.equal(classifyReleaseStage({ freezeExists: false, head: source }), "pre-score-source");
assert.equal(classifyReleaseStage({ freezeExists: true, head: freeze, freezeCommit: freeze }), "pre-score-freeze");
assert.equal(classifyReleaseStage({ freezeExists: true, head: final, freezeCommit: freeze }), "final");
assert.equal(classifyReleaseStage({ freezeExists: false, head: final, freezeCommit: freeze }), "final");
assert.throws(() => classifyReleaseStage({ freezeExists: true, head: freeze }), /unique committed addition/u);
assert.equal(isProhibitedPreScorePath("artifacts/browser-parity.json"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/confirmatory/predictions.jsonl"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/web-negative/wilson.json"), true);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/replay-verification.json"), true);
assert.equal(isProhibitedPreScorePath("artifacts/chrome-e2e-wasm.json"), false);
assert.equal(isProhibitedPreScorePath("benchmark/evidence/evaluation/validation/summary.json"), false);

console.log(JSON.stringify({ cases: 11, policy: "pass" }));
