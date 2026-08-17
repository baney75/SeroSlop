import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { classifyReleaseStage, FREEZE_PATH, LEGACY_FREEZE_PATH } from "./release-stage-policy.mjs";
import { classifyM2Stage } from "./m2-stage-policy.mjs";
import { classifyM4Stage, M4_FAILURE_PATH, M4_PUBLICATION_LOCK_PATH } from "./m4-stage-policy.mjs";
import { classifyM3Stage, M3_FAILURE_PATH, M3_PUBLICATION_LOCK_PATH } from "./m3-stage-policy.mjs";
import { classifyM5Stage, M5_FAILURE_PATH, M5_FINAL_RECEIPT_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_A5_AUTHORIZATION_PATH, M5_A5_COMMIT, M5_A6_AUTHORIZATION_PATH, M5_A6_COMMIT, M5_A7_AUTHORIZATION_PATH, M5_NUMERIC_AUDIT_RECOVERY_EXPECTED, M5_R5_EXPECTED, M5_R6_EXPECTED, M5_R7_EXPECTED, M5_A4_COMMIT, M5_RUNPOD_ENV_AUTHORIZATION_COMMIT, M5_SELECTION_LOCK_PATH, matchesExpectedRows } from "./m5-stage-policy.mjs";
import { m5Git } from "./m5-safe-git.mjs";
import { isM6ProtocolLineageHead, matchesM6SubmissionProxyRHead } from "./m6-stage-policy.mjs";
// M6 P5/P7/P8 prospective direct-child routing is handled by isM6ProtocolLineageHead.
// P8 R/A remain receipt-bound and never imply source-lock or training authority.

function git(arguments_) {
  return m5Git(arguments_);
}

const freezeExists = existsSync(FREEZE_PATH);
const legacyRecoverySource = !freezeExists && existsSync(LEGACY_FREEZE_PATH);
const m2SelectionExists = existsSync("benchmark/evidence/m2/selection-summary.json");
const m2TrainingExists = existsSync("benchmark/evidence/m2/training-summary.json");
const m3SelectionExists = existsSync("benchmark/evidence/m3/selection-summary.json");
const m3FailureExists = existsSync(M3_FAILURE_PATH);
const m3LockExists = existsSync(M3_PUBLICATION_LOCK_PATH);
const m3TrainingExists = existsSync("benchmark/evidence/m3/training-summary.json");
const m4ProtocolExists = existsSync("benchmark/m4/recipe.json");
const m4SelectionExists = existsSync("benchmark/evidence/m4/selection-summary.json");
const m4FailureExists = existsSync(M4_FAILURE_PATH);
const m4LockExists = existsSync(M4_PUBLICATION_LOCK_PATH);
const m4TrainingExists = existsSync("benchmark/evidence/m4/training-summary.json");
const m5ProtocolExists = existsSync("benchmark/m5/recipe.json");
const head = git(["rev-parse", "HEAD"]);
const submissionProxyExists = (() => {
  try {
    const parents = git(["rev-list", "--parents", "-n", "1", head]).split(" ").slice(1);
    const rows = git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", head])
      .split("\n").filter(Boolean).map((line) => {
        const [status, path] = line.split("\t");
        return [path, status];
      });
    return parents.length === 1 && matchesM6SubmissionProxyRHead({ head, parent: parents[0], rows });
  } catch { return false; }
})();
const m6ProtocolExists = (() => {
  try {
    const parent = git(["rev-list", "--parents", "-n", "1", head]).split(" ").slice(1);
    const treePaths = git(["ls-tree", "-r", "--name-only", head]).split("\n");
    const rows = git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", head])
      .split("\n").filter(Boolean).map((line) => {
        const [status, path] = line.split("\t");
        return [path, status];
      });
    // The lineage predicate includes the exact append-only P6 frontier child.
    return parent.length === 1 && isM6ProtocolLineageHead({ head, parent: parent[0], treePaths, rows });
  } catch { return false; }
})();
const m5LockExists = existsSync(M5_SELECTION_LOCK_PATH);
const m5FailureExists = existsSync(M5_FAILURE_PATH);
const m5LargeSourceLockExists = existsSync(M5_LARGE_SOURCE_LOCK_PATH);
const m5FinalExists = existsSync(M5_FINAL_RECEIPT_PATH);
const m5A7Exists = existsSync(M5_A7_AUTHORIZATION_PATH);
if (m5A7Exists && (m5LockExists || m5LargeSourceLockExists || m5FinalExists)) throw new Error("M5 A7 is failure-verifier-only and cannot authorize success evidence");
const m5AuthorizationExists = m5A7Exists || existsSync(M5_A6_AUTHORIZATION_PATH) || (head === M5_A5_COMMIT && existsSync(M5_A5_AUTHORIZATION_PATH));
const m5HeadRows = git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", head]).split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
const m5HeadParent = git(["rev-list", "--parents", "-n", "1", head]).split(" ")[1];
const m5SourceRecoveryExists = (m5HeadParent === M5_A4_COMMIT && matchesExpectedRows(m5HeadRows, M5_R5_EXPECTED)) ||
  (m5HeadParent === M5_RUNPOD_ENV_AUTHORIZATION_COMMIT && matchesExpectedRows(m5HeadRows, M5_NUMERIC_AUDIT_RECOVERY_EXPECTED)) ||
  (m5HeadParent === M5_A5_COMMIT && matchesExpectedRows(m5HeadRows, M5_R6_EXPECTED)) ||
  (m5HeadParent === M5_A6_COMMIT && matchesExpectedRows(m5HeadRows, M5_R7_EXPECTED));
const additions = git(["log", "--no-renames", "--diff-filter=A", "--format=%H", "--", FREEZE_PATH])
  .split("\n").filter(Boolean);
if (additions.length > 1 || (freezeExists && additions.length !== 1)) {
  throw new Error("The pre-score freeze must have exactly one committed addition");
}
const stage = classifyReleaseStage({ freezeExists, head, freezeCommit: additions[0], legacyRecoverySource });
const m2Stage = classifyM2Stage({
  selectionExists: m2SelectionExists,
  trainingExists: m2TrainingExists,
});
const m4Stage = classifyM4Stage({
  protocolExists: m4ProtocolExists,
  selectionExists: m4SelectionExists,
  failureExists: m4FailureExists,
  lockExists: m4LockExists,
  trainingExists: m4TrainingExists,
});
const m5Stage = classifyM5Stage({
  protocolExists: m5ProtocolExists,
  lockExists: m5LockExists,
  failureExists: m5FailureExists,
  largeSourceLockExists: m5LargeSourceLockExists,
  finalExists: m5FinalExists,
  sourceRecoveryExists: m5SourceRecoveryExists,
  authorizationExists: m5AuthorizationExists,
  a5Exists: m5AuthorizationExists,
});
const effectiveStage = m5Stage ?? m4Stage ?? classifyM3Stage({
  selectionExists: m3SelectionExists,
  failureExists: m3FailureExists,
  lockExists: m3LockExists,
  trainingExists: m3TrainingExists,
}) ?? m2Stage ?? stage;
const scripts = new Map([
  ["m6-protocol", "verify:m6-protocol"],
  ["m5-protocol", "verify:m5-protocol"],
  ["m5-source-recovery", "verify:m5-source-recovery"],
  ["m5-authorized", "verify:m5-authorized"],
  ["m5-pinned", "verify:m5-pinned"],
  ["m5-eval-locked", "verify:m5-eval-locked"],
  ["m5-failed", "verify:m5-failed"],
  ["m5-final", "verify:m5-final"],
  ["m4-protocol", "verify:m4-protocol"],
  ["m4-source", "verify:m4-source"],
  ["m4-failed", "verify:m4-failed"],
  ["m4-pinned", "verify:m4-pinned"],
  ["m4-final", "verify:m4-final"],
  ["m3-failed", "verify:m3-failed"],
  ["m3-source", "verify:m3-source"],
  ["m3-pinned", "verify:m3-pinned"],
  ["m3-final", "verify:m3-final"],
  ["m2-source", "verify:m2-source"],
  ["m2-final", "verify:m2-final"],
  ["final", "verify:final"],
  ["pre-score", "verify:pre-score"],
]);
if (submissionProxyExists) {
  const result = spawnSync("npm", ["run", "verify:submission-proxy"], { stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
  process.exit(0);
}
if (m6ProtocolExists) {
  // Route P6 S/R/A/R2 and prospective P7 frontier states through the protocol checker.
  const result = spawnSync("npm", ["run", "verify:m6-protocol"], { stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
  process.exit(0);
}
const script = scripts.get(effectiveStage);
if (script === undefined) throw new Error(`Unknown release stage: ${effectiveStage}`);
console.log(JSON.stringify({ stage: effectiveStage, script }));
const result = spawnSync("npm", ["run", script], { stdio: "inherit" });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
