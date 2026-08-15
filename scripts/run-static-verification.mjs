import { execFileSync, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { classifyReleaseStage, FREEZE_PATH, LEGACY_FREEZE_PATH } from "./release-stage-policy.mjs";
import { classifyM2Stage } from "./m2-stage-policy.mjs";
import { classifyM4Stage, M4_FAILURE_PATH, M4_PUBLICATION_LOCK_PATH } from "./m4-stage-policy.mjs";
import { classifyM3Stage, M3_FAILURE_PATH, M3_PUBLICATION_LOCK_PATH } from "./m3-stage-policy.mjs";
import { classifyM5Stage, M5_FAILURE_PATH, M5_FINAL_RECEIPT_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_SELECTION_LOCK_PATH } from "./m5-stage-policy.mjs";

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8" }).trim();
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
const m5LockExists = existsSync(M5_SELECTION_LOCK_PATH);
const m5FailureExists = existsSync(M5_FAILURE_PATH);
const m5LargeSourceLockExists = existsSync(M5_LARGE_SOURCE_LOCK_PATH);
const m5FinalExists = existsSync(M5_FINAL_RECEIPT_PATH);
const head = git(["rev-parse", "HEAD"]);
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
});
const effectiveStage = m5Stage ?? m4Stage ?? classifyM3Stage({
  selectionExists: m3SelectionExists,
  failureExists: m3FailureExists,
  lockExists: m3LockExists,
  trainingExists: m3TrainingExists,
}) ?? m2Stage ?? stage;
const scripts = new Map([
  ["m5-protocol", "verify:m5-protocol"],
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
const script = scripts.get(effectiveStage);
if (script === undefined) throw new Error(`Unknown release stage: ${effectiveStage}`);
console.log(JSON.stringify({ stage: effectiveStage, script }));
const result = spawnSync("npm", ["run", script], { stdio: "inherit" });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
