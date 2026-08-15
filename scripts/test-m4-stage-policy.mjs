import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  M4_FAILURE_EXPECTED,
  M4_LOCK_EXPECTED,
  M4_DATE_RECOVERY_EXPECTED,
  M4_PROTOCOL_EXPECTED,
  M4_PROTOCOL_RECOVERY_EXPECTED,
  M4_PUBLICATION_EXPECTED,
  M4_SOURCE_EXPECTED,
  classifyM4Stage,
  matchesM4ProtocolRecoveryLineage,
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

for (const expected of [M4_PROTOCOL_EXPECTED, M4_PROTOCOL_RECOVERY_EXPECTED, M4_DATE_RECOVERY_EXPECTED,
  M4_SOURCE_EXPECTED, M4_LOCK_EXPECTED,
  M4_FAILURE_EXPECTED, M4_PUBLICATION_EXPECTED]) {
  const rows = [...expected];
  assert.equal(matchesExpectedRows(rows, expected), true);
  assert.equal(matchesExpectedRows(rows.slice(1), expected), false);
  assert.equal(matchesExpectedRows([...rows, rows[0]], expected), false);
  assert.equal(matchesExpectedRows(rows.map((row, index) => index === 0 ? [row[0], row[1] === "A" ? "M" : "A"] : row), expected), false);
}

const lineage = {
  protocolParents: ["6fed0d0ad0e9b9bdf50e17cc0463d8c845abc64b"],
  protocolRows: [...M4_DATE_RECOVERY_EXPECTED],
  recoveryProtocolParents: ["82b06b49d44bedd54aba22a09d6a96b44e89d303"],
  recoveryProtocolRows: [...M4_PROTOCOL_RECOVERY_EXPECTED],
  recoveryProtocolTree: "96f8ccd610cb9362fff88bbaacb5a050937c259d",
  failedProtocolParents: ["439b2481dc88a887f8317be669096495760fbeb1"],
  failedProtocolRows: [...M4_PROTOCOL_EXPECTED],
  failedProtocolTree: "9e1de15031f83145ba40c8b1a2470b0833854fd8",
  baseTree: "440931a595c87ca3d293f5a6f980c75169ddb899",
};
assert.equal(matchesM4ProtocolRecoveryLineage(lineage), true);
for (const field of ["protocolParents", "protocolRows", "recoveryProtocolParents", "recoveryProtocolRows",
  "recoveryProtocolTree", "failedProtocolParents", "failedProtocolRows",
  "failedProtocolTree", "baseTree"]) {
  const current = lineage[field];
  const changed = { ...lineage, [field]: Array.isArray(current) ? current.slice(1) : "0".repeat(40) };
  assert.equal(matchesM4ProtocolRecoveryLineage(changed), false, field);
}

const scripts = JSON.parse(readFileSync("package.json", "utf8")).scripts;
for (const stage of ["protocol", "source", "failed", "pinned", "final"]) {
  assert.equal(typeof scripts[`verify:m4-${stage}`], "string");
}
for (const name of ["check:m4-pipeline", "check:m4-publication-contract"]) {
  assert.match(scripts[name], /^node scripts\/run-benchmark-python\.mjs /u);
  assert.doesNotMatch(scripts[name], /benchmark\/\.venv\/bin\/python/u);
}
const router = readFileSync("scripts/run-static-verification.mjs", "utf8");
assert.match(router, /classifyM4Stage/u);
assert.ok(router.indexOf("classifyM4Stage") < router.indexOf("classifyM3Stage"));
assert.match(router, /\["m4-final", "verify:m4-final"\]/u);
assert.match(router, /\["m4-failed", "verify:m4-failed"\]/u);

console.log(JSON.stringify({ cases: 47, policy: "pass" }));
