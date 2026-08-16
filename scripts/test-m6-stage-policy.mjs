import assert from "node:assert/strict";
import {
  classifyM6Stage,
  isM6ProtocolHead,
  isM6ProtocolLineageHead,
  M6_BASE_COMMIT,
  M6_P_COMMIT,
  M6_PROTOCOL_PATHS,
  M6_PROTOCOL_RECOVERY_EXPECTED,
  matchesM6ProtocolRecovery,
  matchesProspectiveP,
  parseM6Recipe,
} from "./m6-stage-policy.mjs";
const recipe = parseM6Recipe();
assert.equal(recipe.baseCommit, M6_BASE_COMMIT);
assert.equal(classifyM6Stage({}), "m6-protocol");
assert.throws(() => classifyM6Stage({ sourceLock: true }), /executable/);
assert.throws(() => classifyM6Stage({ trained: true }), /executable/);
assert.throws(() => classifyM6Stage({ evaluated: true }), /executable/);
const paths = [...M6_PROTOCOL_PATHS];
const statuses = Object.fromEntries(paths.map((path) => [path, ["package.json", "scripts/run-static-verification.mjs"].includes(path) ? "M" : "A"]));
assert.equal(matchesProspectiveP({ head: "a".repeat(40), parents: [M6_BASE_COMMIT], paths, statuses }), true);
assert.equal(matchesProspectiveP({ head: "a".repeat(40), parents: [M6_BASE_COMMIT], paths: paths.slice(1), statuses }), false);
assert.equal(matchesProspectiveP({ head: "a".repeat(40), parents: ["b".repeat(40)], paths, statuses }), false);
assert.equal(matchesProspectiveP({ head: "a".repeat(40), parents: [M6_BASE_COMMIT], paths, statuses: { ...statuses, [paths[0]]: "M" } }), false);
assert.equal(isM6ProtocolHead({ head: "a".repeat(40), parent: M6_BASE_COMMIT, treePaths: paths }), true);
assert.equal(isM6ProtocolHead({ head: "a".repeat(40), parent: "b".repeat(40), treePaths: paths }), false);
const recoveryRows = M6_PROTOCOL_RECOVERY_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_P_COMMIT, rows: recoveryRows }), true);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_BASE_COMMIT, rows: recoveryRows }), false);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_P_COMMIT, rows: recoveryRows.slice(1) }), false);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_P_COMMIT, rows: [...recoveryRows, ["extra", "M"]] }), false);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_P_COMMIT, rows: recoveryRows.map(([path, status], index) => [path, index === 0 ? "A" : status]) }), false);
assert.equal(isM6ProtocolLineageHead({ head: "a".repeat(40), parent: M6_BASE_COMMIT, treePaths: paths, rows: [] }), true);
assert.equal(isM6ProtocolLineageHead({ head: "c".repeat(40), parent: M6_P_COMMIT, treePaths: paths, rows: recoveryRows }), true);
console.log("M6 stage policy PASS");
