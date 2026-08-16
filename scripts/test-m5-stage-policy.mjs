import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  M5_BASE_SOURCE_COMMIT,
  M5_BASE_SOURCE_TREE,
  M5_ORIGINAL_PROTOCOL_COMMIT,
  M5_ORIGINAL_PROTOCOL_TREE,
  M5_PROTOCOL_EXPECTED,
  M5_PROTOCOL_RECOVERY_EXPECTED,
  classifyM5Stage,
  matchesExpectedRows,
  matchesM5ProtocolLineage,
} from "./m5-stage-policy.mjs";

const rows = [...M5_PROTOCOL_EXPECTED.entries()];
const recoveryRows = [...M5_PROTOCOL_RECOVERY_EXPECTED.entries()];
assert.equal(matchesExpectedRows(rows, M5_PROTOCOL_EXPECTED), true);
assert.equal(matchesExpectedRows([...rows.slice(1), rows[1]], M5_PROTOCOL_EXPECTED), false);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: [M5_ORIGINAL_PROTOCOL_COMMIT], recoveryRows,
  originalTree: M5_ORIGINAL_PROTOCOL_TREE,
  originalParents: [M5_BASE_SOURCE_COMMIT], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), true);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: ["0".repeat(40)], recoveryRows,
  originalTree: M5_ORIGINAL_PROTOCOL_TREE,
  originalParents: [M5_BASE_SOURCE_COMMIT], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), false);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: [M5_ORIGINAL_PROTOCOL_COMMIT], recoveryRows: recoveryRows.slice(1),
  originalTree: M5_ORIGINAL_PROTOCOL_TREE,
  originalParents: [M5_BASE_SOURCE_COMMIT], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), false);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: [M5_ORIGINAL_PROTOCOL_COMMIT], recoveryRows,
  originalTree: "0".repeat(40),
  originalParents: [M5_BASE_SOURCE_COMMIT], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), false);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: [M5_ORIGINAL_PROTOCOL_COMMIT], recoveryRows,
  originalTree: M5_ORIGINAL_PROTOCOL_TREE,
  originalParents: ["0".repeat(40)], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), false);
assert.equal(classifyM5Stage({ protocolExists: false, lockExists: false, failureExists: false, largeSourceLockExists: false, finalExists: false }), null);
assert.equal(classifyM5Stage({ protocolExists: true, lockExists: false, failureExists: false, largeSourceLockExists: false, finalExists: false }), "m5-protocol");
assert.equal(classifyM5Stage({ protocolExists: true, lockExists: true, failureExists: false, largeSourceLockExists: false, finalExists: false }), "m5-pinned");
assert.equal(classifyM5Stage({ protocolExists: true, lockExists: true, failureExists: false, largeSourceLockExists: true, finalExists: false }), "m5-eval-locked");
assert.equal(classifyM5Stage({ protocolExists: true, lockExists: false, failureExists: true, largeSourceLockExists: false, finalExists: false }), "m5-failed");
assert.equal(classifyM5Stage({ protocolExists: true, lockExists: true, failureExists: false, largeSourceLockExists: true, finalExists: true }), "m5-final");
assert.throws(() => classifyM5Stage({ protocolExists: true, lockExists: true, failureExists: true, largeSourceLockExists: false, finalExists: false }));
assert.throws(() => classifyM5Stage({ protocolExists: true, lockExists: false, failureExists: false, largeSourceLockExists: true, finalExists: false }));
assert.throws(() => classifyM5Stage({ protocolExists: true, lockExists: true, failureExists: false, largeSourceLockExists: false, finalExists: true }));
const dispatcher = readFileSync("scripts/run-static-verification.mjs", "utf8");
const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
for (const [stage, script] of [["m5-failed", "verify:m5-failed"], ["m5-eval-locked", "verify:m5-eval-locked"], ["m5-final", "verify:m5-final"]]) {
  assert.ok(dispatcher.includes(`["${stage}", "${script}"]`));
  assert.equal(typeof packageJson.scripts[script], "string");
}
assert.ok(packageJson.scripts["benchmark:m5:preflight"].includes("--preflight-only"));
assert.ok(!packageJson.scripts["benchmark:m5:preflight"].includes("--protocol-commit"));
assert.ok(!packageJson.scripts["benchmark:m5:train"].includes("M5_PROTOCOL_COMMIT"));
console.log(JSON.stringify({ cases: 23, policy: "pass" }));
