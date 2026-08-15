import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  M4_FAILURE_EXPECTED,
  M4_LOCK_EXPECTED,
  M4_PROTOCOL_EXPECTED,
  M4_PUBLICATION_EXPECTED,
  M4_SOURCE_EXPECTED,
  classifyM4Stage,
  matchesExpectedRows,
} from "./m4-stage-policy.mjs";

const state = (overrides = {}) => ({
  protocolExists: true,
  selectionExists: false,
  failureExists: false,
  lockExists: false,
  trainingExists: false,
  ...overrides,
});

assert.equal(classifyM4Stage(state({ protocolExists: false })), null);
assert.equal(classifyM4Stage(state()), "m4-protocol");
assert.equal(classifyM4Stage(state({ selectionExists: true })), "m4-source");
assert.equal(classifyM4Stage(state({ selectionExists: true, failureExists: true })), "m4-failed");
assert.equal(classifyM4Stage(state({ selectionExists: true, lockExists: true })), "m4-pinned");
assert.equal(classifyM4Stage(state({ selectionExists: true, lockExists: true, trainingExists: true })), "m4-final");
assert.throws(() => classifyM4Stage(state({ protocolExists: false, selectionExists: true })), /without the M4 protocol/u);
assert.throws(() => classifyM4Stage(state({ failureExists: true })), /without M4 source evidence/u);
assert.throws(() => classifyM4Stage(state({ lockExists: true })), /without M4 source evidence/u);
assert.throws(() => classifyM4Stage(state({ trainingExists: true })), /without M4 source evidence/u);
assert.throws(() => classifyM4Stage(state({ selectionExists: true, failureExists: true, lockExists: true })), /cannot coexist/u);
assert.throws(() => classifyM4Stage(state({ selectionExists: true, failureExists: true, trainingExists: true })), /cannot coexist/u);
assert.throws(() => classifyM4Stage(state({ selectionExists: true, trainingExists: true })), /before its output lock/u);

for (const expected of [M4_PROTOCOL_EXPECTED, M4_SOURCE_EXPECTED, M4_LOCK_EXPECTED,
  M4_FAILURE_EXPECTED, M4_PUBLICATION_EXPECTED]) {
  const rows = [...expected];
  assert.equal(matchesExpectedRows(rows, expected), true);
  assert.equal(matchesExpectedRows(rows.slice(1), expected), false);
  assert.equal(matchesExpectedRows([...rows, rows[0]], expected), false);
  assert.equal(matchesExpectedRows(rows.map((row, index) => index === 0 ? [row[0], row[1] === "A" ? "M" : "A"] : row), expected), false);
}

const scripts = JSON.parse(readFileSync("package.json", "utf8")).scripts;
for (const stage of ["protocol", "source", "failed", "pinned", "final"]) {
  assert.equal(typeof scripts[`verify:m4-${stage}`], "string");
}
const router = readFileSync("scripts/run-static-verification.mjs", "utf8");
assert.match(router, /classifyM4Stage/u);
assert.ok(router.indexOf("classifyM4Stage") < router.indexOf("classifyM3Stage"));
assert.match(router, /\["m4-final", "verify:m4-final"\]/u);
assert.match(router, /\["m4-failed", "verify:m4-failed"\]/u);

console.log(JSON.stringify({ cases: 33, policy: "pass" }));
