import { execFileSync, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { classifyReleaseStage, FREEZE_PATH, LEGACY_FREEZE_PATH } from "./release-stage-policy.mjs";
import { classifyM2Stage } from "./m2-stage-policy.mjs";
import { classifyM3Stage, M3_FAILURE_PATH, M3_PUBLICATION_LOCK_PATH } from "./m3-stage-policy.mjs";

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
const effectiveStage = classifyM3Stage({
  selectionExists: m3SelectionExists,
  failureExists: m3FailureExists,
  lockExists: m3LockExists,
  trainingExists: m3TrainingExists,
}) ?? m2Stage ?? stage;
const scripts = new Map([
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
