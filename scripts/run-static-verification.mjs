import { execFileSync, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { classifyReleaseStage, FREEZE_PATH, LEGACY_FREEZE_PATH } from "./release-stage-policy.mjs";
import { classifyM2Stage } from "./m2-stage-policy.mjs";

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8" }).trim();
}

const freezeExists = existsSync(FREEZE_PATH);
const legacyRecoverySource = !freezeExists && existsSync(LEGACY_FREEZE_PATH);
const m2SelectionExists = existsSync("benchmark/evidence/m2/selection-summary.json");
const m2TrainingExists = existsSync("benchmark/evidence/m2/training-summary.json");
const head = git(["rev-parse", "HEAD"]);
const additions = git(["log", "--no-renames", "--diff-filter=A", "--format=%H", "--", FREEZE_PATH])
  .split("\n").filter(Boolean);
if (additions.length > 1 || (freezeExists && additions.length !== 1)) {
  throw new Error("The pre-score freeze must have exactly one committed addition");
}
const stage = classifyReleaseStage({ freezeExists, head, freezeCommit: additions[0], legacyRecoverySource });
const effectiveStage = classifyM2Stage({
  selectionExists: m2SelectionExists,
  trainingExists: m2TrainingExists,
}) ?? stage;
const script = effectiveStage === "m2-source"
  ? "verify:m2-source"
  : (effectiveStage === "m2-final"
    ? "verify:m2-final"
    : (effectiveStage === "final" ? "verify:final" : "verify:pre-score"));
console.log(JSON.stringify({ stage: effectiveStage, script }));
const result = spawnSync("npm", ["run", script], { stdio: "inherit" });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
