import assert from "node:assert/strict";
import {
  M2_CHECKER_RECOVERY_EXPECTED,
  M2_FINALIZER_SOURCE_EXPECTED,
  M2_PUBLICATION_EXPECTED,
  classifyM2Stage,
  matchesExpectedRows,
} from "./m2-stage-policy.mjs";

assert.equal(classifyM2Stage({ selectionExists: false, trainingExists: false }), null);
assert.equal(classifyM2Stage({ selectionExists: true, trainingExists: false }), "m2-source");
assert.equal(classifyM2Stage({ selectionExists: true, trainingExists: true }), "m2-final");
assert.throws(
  () => classifyM2Stage({ selectionExists: false, trainingExists: true }),
  /cannot exist without M2 selection evidence/u,
);

const sourceRows = [...M2_FINALIZER_SOURCE_EXPECTED].map(([pathname, status]) => [pathname, status]);
const publicationRows = [...M2_PUBLICATION_EXPECTED].map(([pathname, status]) => [pathname, status]);
const checkerRecoveryRows = [...M2_CHECKER_RECOVERY_EXPECTED].map(([pathname, status]) => [pathname, status]);
assert.equal(matchesExpectedRows(sourceRows, M2_FINALIZER_SOURCE_EXPECTED), true);
assert.equal(matchesExpectedRows(sourceRows.slice(1), M2_FINALIZER_SOURCE_EXPECTED), false);
assert.equal(matchesExpectedRows([...sourceRows, ["README.md", "M"]], M2_FINALIZER_SOURCE_EXPECTED), false);
assert.equal(matchesExpectedRows(
  sourceRows.map(([pathname, status], index) => [pathname, index === 0 ? "A" : status]),
  M2_FINALIZER_SOURCE_EXPECTED,
), false);
assert.equal(matchesExpectedRows(
  [sourceRows[0], sourceRows[0], ...sourceRows.slice(2)],
  M2_FINALIZER_SOURCE_EXPECTED,
), false);
assert.equal(matchesExpectedRows(publicationRows, M2_PUBLICATION_EXPECTED), true);
assert.equal(matchesExpectedRows(checkerRecoveryRows, M2_CHECKER_RECOVERY_EXPECTED), true);
assert.equal(matchesExpectedRows(checkerRecoveryRows.slice(1), M2_CHECKER_RECOVERY_EXPECTED), false);
assert.equal(matchesExpectedRows(
  publicationRows.map(([pathname, status]) => [
    pathname === "weights/prooflens-cf384.onnx" ? "benchmark/evidence/large/training-summary.json" : pathname,
    status,
  ]),
  M2_PUBLICATION_EXPECTED,
), false);

console.log(JSON.stringify({ cases: 13, policy: "pass" }));
