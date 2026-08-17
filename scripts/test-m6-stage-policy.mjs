import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  classifyM6Stage,
  isM6ProtocolHead,
  isM6ProtocolLineageHead,
  M6_BASE_COMMIT,
  M6_CI_RECOVERY_EXPECTED,
  M6_P_COMMIT,
  M6_P2_COMMIT,
  M6_P3_COMMIT,
  M6_MATERIALIZER_RECOVERY_EXPECTED,
  M6_PROTOCOL_PATHS,
  M6_PROTOCOL_RECOVERY_EXPECTED,
  matchesM6CiRecovery,
  matchesM6MaterializerRecovery,
  matchesM6ProtocolRecovery,
  matchesProspectiveP,
  parseM6Recipe,
  validateM6VerifyRequirements,
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
const materializerRows = M6_MATERIALIZER_RECOVERY_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6MaterializerRecovery({ head: "d".repeat(40), parent: M6_P2_COMMIT, rows: materializerRows }), true);
assert.equal(matchesM6MaterializerRecovery({ head: "d".repeat(40), parent: M6_P_COMMIT, rows: materializerRows }), false);
assert.equal(matchesM6MaterializerRecovery({ head: "d".repeat(40), parent: M6_P2_COMMIT, rows: materializerRows.slice(1) }), false);
assert.equal(matchesM6MaterializerRecovery({ head: "d".repeat(40), parent: M6_P2_COMMIT, rows: [...materializerRows, ["extra", "A"]] }), false);
assert.equal(isM6ProtocolLineageHead({ head: "d".repeat(40), parent: M6_P2_COMMIT, treePaths: paths, rows: materializerRows }), true);
const ciRows = M6_CI_RECOVERY_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6CiRecovery({ head: "e".repeat(40), parent: M6_P3_COMMIT, rows: ciRows }), true);
assert.equal(matchesM6CiRecovery({ head: "e".repeat(40), parent: M6_P2_COMMIT, rows: ciRows }), false);
assert.equal(matchesM6CiRecovery({ head: "e".repeat(40), parent: M6_P3_COMMIT, rows: ciRows.slice(1) }), false);
assert.equal(matchesM6CiRecovery({ head: "e".repeat(40), parent: M6_P3_COMMIT, rows: [...ciRows, ["extra", "M"]] }), false);
assert.equal(isM6ProtocolLineageHead({ head: "e".repeat(40), parent: M6_P3_COMMIT, treePaths: paths, rows: ciRows }), true);
const requirements = readFileSync("benchmark/verify-requirements.txt");
assert.equal(validateM6VerifyRequirements(requirements), true);
const requirementsText = requirements.toString("utf8");
assert.throws(() => validateM6VerifyRequirements(Buffer.from(requirementsText.replace("pyarrow==20.0.0\n", ""))), /bytes changed/);
assert.throws(() => validateM6VerifyRequirements(Buffer.from(requirementsText.replace("numpy==2.2.6", "numpy==0.0.0"))), /bytes changed/);
assert.throws(() => validateM6VerifyRequirements(Buffer.from(requirementsText + "requests==2.0.0\n")), /bytes changed/);
assert.throws(() => validateM6VerifyRequirements(Buffer.from(requirementsText + "pyarrow==20.0.0\n")), /bytes changed/);
console.log("M6 stage policy PASS");
