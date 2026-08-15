import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  M3_FAILURE_EXPECTED,
  M3_LOCK_EXPECTED,
  M3_PUBLICATION_EXPECTED,
  M3_SOURCE_EXPECTED,
  classifyM3Stage,
  matchesExpectedRows,
} from "./m3-stage-policy.mjs";

assert.equal(classifyM3Stage({ selectionExists: false, failureExists: false, lockExists: false, trainingExists: false }), null);
assert.equal(classifyM3Stage({ selectionExists: true, failureExists: false, lockExists: false, trainingExists: false }), "m3-source");
assert.equal(classifyM3Stage({ selectionExists: true, failureExists: true, lockExists: false, trainingExists: false }), "m3-failed");
assert.equal(classifyM3Stage({ selectionExists: true, failureExists: false, lockExists: true, trainingExists: false }), "m3-pinned");
assert.equal(classifyM3Stage({ selectionExists: true, failureExists: false, lockExists: true, trainingExists: true }), "m3-final");
assert.throws(
  () => classifyM3Stage({ selectionExists: false, failureExists: true, lockExists: false, trainingExists: false }),
  /without M3 selection evidence/u,
);
assert.throws(
  () => classifyM3Stage({ selectionExists: false, failureExists: false, lockExists: true, trainingExists: false }),
  /without M3 selection evidence/u,
);
assert.throws(
  () => classifyM3Stage({ selectionExists: false, failureExists: false, lockExists: false, trainingExists: true }),
  /without M3 selection evidence/u,
);
assert.throws(
  () => classifyM3Stage({ selectionExists: true, failureExists: true, lockExists: true, trainingExists: false }),
  /cannot coexist/u,
);
assert.throws(
  () => classifyM3Stage({ selectionExists: true, failureExists: true, lockExists: false, trainingExists: true }),
  /cannot coexist/u,
);
assert.throws(
  () => classifyM3Stage({ selectionExists: true, failureExists: false, lockExists: false, trainingExists: true }),
  /before its output lock/u,
);

const lockRows = [...M3_LOCK_EXPECTED].map(([path, status]) => [path, status]);
assert.equal(matchesExpectedRows(lockRows, M3_LOCK_EXPECTED), true);
assert.equal(matchesExpectedRows([...lockRows, lockRows[0]], M3_LOCK_EXPECTED), false);
const finalRows = [...M3_PUBLICATION_EXPECTED].map(([path, status]) => [path, status]);
assert.equal(matchesExpectedRows(finalRows, M3_PUBLICATION_EXPECTED), true);
assert.equal(matchesExpectedRows(finalRows.slice(1), M3_PUBLICATION_EXPECTED), false);
assert.equal(matchesExpectedRows(finalRows.map((row, index) => index === 0 ? [row[0], "A"] : row), M3_PUBLICATION_EXPECTED), false);
const failureRows = [...M3_FAILURE_EXPECTED].map(([path, status]) => [path, status]);
assert.equal(matchesExpectedRows(failureRows, M3_FAILURE_EXPECTED), true);
assert.equal(matchesExpectedRows(failureRows.slice(1), M3_FAILURE_EXPECTED), false);
assert.equal(matchesExpectedRows([...failureRows, failureRows[0]], M3_FAILURE_EXPECTED), false);
assert.equal(matchesExpectedRows(failureRows.map((row, index) => index === 0 ? [row[0], "M"] : row),
  M3_FAILURE_EXPECTED), false);

const scripts = JSON.parse(readFileSync("package.json", "utf8")).scripts;
for (const stage of ["source", "failed", "pinned", "final"]) {
  assert.equal(typeof scripts[`verify:m3-${stage}`], "string");
}
assert.match(readFileSync("scripts/run-static-verification.mjs", "utf8"), /\["m3-final", "verify:m3-final"\]/u);
assert.match(readFileSync("scripts/run-static-verification.mjs", "utf8"), /\["m3-failed", "verify:m3-failed"\]/u);
assert.equal(M3_SOURCE_EXPECTED.get("scripts/check-m2-training-evidence.mjs"), "M");
assert.match(readFileSync("scripts/check-m2-training-evidence.mjs", "utf8"),
  /exact M3 source\/lock\/failure descendants/u);

console.log(JSON.stringify({ cases: 27, policy: "pass" }));
