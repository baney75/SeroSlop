import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import {
  classifyReleaseStage,
  FAILED_EVALUATION_COMMIT,
  FAILED_EVALUATION_PATHS,
  FAILED_EVALUATION_TREE,
  FAILED_RECOVERY_REPAIR_PATHS,
  FAILED_RECOVERY_SOURCE_COMMIT,
  FAILED_RECOVERY_SOURCE_TREE,
  FREEZE_PATH,
  LEGACY_FREEZE_COMMIT,
  LEGACY_FREEZE_PATH,
  LEGACY_FREEZE_SHA256,
  LEGACY_FREEZE_TREE,
  LEGACY_SOURCE_COMMIT,
  LEGACY_SOURCE_TREE,
  RECOVERY_REPAIR_PATHS,
  REPLACEMENT_SELECTION_COMMIT,
  REPLACEMENT_SELECTION_PATHS,
  REPLACEMENT_SELECTION_TREE,
  SECOND_FREEZE_COMMIT,
  SECOND_FREEZE_PATH,
  SECOND_FREEZE_SHA256,
  SECOND_FREEZE_TREE,
  SECOND_RECOVERY_REPAIR_PATHS,
  SECOND_RECOVERY_SOURCE_COMMIT,
  SECOND_RECOVERY_SOURCE_TREE,
  freezeReceiptAdditions,
  isProhibitedPreScorePath,
} from "./release-stage-policy.mjs";

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8" }).trim();
}

function gitBytes(arguments_) {
  return execFileSync("git", arguments_);
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function sortedDiff(older, newer) {
  return git(["diff", "--no-renames", "--name-only", `${older}..${newer}`])
    .split("\n").filter(Boolean).sort();
}

function requireTree(commit, tree, label) {
  requireCondition(git(["rev-parse", `${commit}^{tree}`]) === tree, `${label} tree changed`);
}

function requireParent(child, parent, label) {
  requireCondition(git(["rev-parse", `${child}^`]) === parent, `${label} lineage changed`);
}

function requireAddedPacket(commit, paths, label) {
  const rows = git(["diff-tree", "--no-renames", "--no-commit-id", "--name-status", "-r", commit])
    .split("\n").filter(Boolean).sort();
  const expected = paths.map((path) => `A\t${path}`).sort();
  requireCondition(JSON.stringify(rows) === JSON.stringify(expected), `${label} packet changed`);
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "Pre-score verification requires a completely clean worktree");
const freezeExists = existsSync(FREEZE_PATH);
const head = git(["rev-parse", "HEAD"]);
const additions = git(["log", "--no-renames", "--diff-filter=A", "--format=%H", "--", FREEZE_PATH])
  .split("\n").filter(Boolean);
requireCondition(additions.length <= 1 && (!freezeExists || additions.length === 1),
  "The V3 freeze must have exactly one committed addition");
const legacyRecoverySource = !freezeExists && additions.length === 0 && existsSync(SECOND_FREEZE_PATH);
const stage = classifyReleaseStage({ freezeExists, head, freezeCommit: additions[0], legacyRecoverySource });
requireCondition(stage !== "final", "A descendant of V3 must pass the complete final evidence gate");
const addedReceipts = freezeReceiptAdditions(git([
  "log", "--no-renames", "--diff-filter=A", "--format=", "--name-only", "HEAD",
]).split("\n"));
const expectedReceipts = stage === "pre-score-freeze"
  ? [LEGACY_FREEZE_PATH, SECOND_FREEZE_PATH, FREEZE_PATH].sort()
  : [LEGACY_FREEZE_PATH, SECOND_FREEZE_PATH].sort();
requireCondition(JSON.stringify(addedReceipts) === JSON.stringify(expectedReceipts),
  "Pre-score history contains an alternate freeze receipt");

if (stage === "pre-score-recovery-source") {
  for (const [commit, tree, label] of [
    [LEGACY_SOURCE_COMMIT, LEGACY_SOURCE_TREE, "Legacy source"],
    [LEGACY_FREEZE_COMMIT, LEGACY_FREEZE_TREE, "Legacy freeze"],
    [FAILED_RECOVERY_SOURCE_COMMIT, FAILED_RECOVERY_SOURCE_TREE, "Failed recovery"],
    [SECOND_RECOVERY_SOURCE_COMMIT, SECOND_RECOVERY_SOURCE_TREE, "Second recovery"],
    [SECOND_FREEZE_COMMIT, SECOND_FREEZE_TREE, "Second freeze"],
    [FAILED_EVALUATION_COMMIT, FAILED_EVALUATION_TREE, "Failed evaluation"],
    [REPLACEMENT_SELECTION_COMMIT, REPLACEMENT_SELECTION_TREE, "Replacement selection"],
  ]) requireTree(commit, tree, label);
  for (const [child, parent, label] of [
    [LEGACY_FREEZE_COMMIT, LEGACY_SOURCE_COMMIT, "Legacy freeze"],
    [FAILED_RECOVERY_SOURCE_COMMIT, LEGACY_FREEZE_COMMIT, "Failed recovery"],
    [SECOND_RECOVERY_SOURCE_COMMIT, FAILED_RECOVERY_SOURCE_COMMIT, "Second recovery"],
    [SECOND_FREEZE_COMMIT, SECOND_RECOVERY_SOURCE_COMMIT, "Second freeze"],
    [FAILED_EVALUATION_COMMIT, SECOND_FREEZE_COMMIT, "Failed evaluation"],
    [REPLACEMENT_SELECTION_COMMIT, FAILED_EVALUATION_COMMIT, "Replacement selection"],
    [head, REPLACEMENT_SELECTION_COMMIT, "Numeric recovery"],
  ]) requireParent(child, parent, label);
  requireCondition(digest(readFileSync(LEGACY_FREEZE_PATH)) === LEGACY_FREEZE_SHA256 &&
    digest(gitBytes(["show", `${LEGACY_FREEZE_COMMIT}:${LEGACY_FREEZE_PATH}`])) === LEGACY_FREEZE_SHA256,
  "Legacy freeze receipt bytes changed");
  requireCondition(digest(readFileSync(SECOND_FREEZE_PATH)) === SECOND_FREEZE_SHA256 &&
    digest(gitBytes(["show", `${SECOND_FREEZE_COMMIT}:${SECOND_FREEZE_PATH}`])) === SECOND_FREEZE_SHA256,
  "Second freeze receipt bytes changed");
  requireCondition(JSON.stringify(sortedDiff(LEGACY_FREEZE_COMMIT, FAILED_RECOVERY_SOURCE_COMMIT)) ===
    JSON.stringify([...FAILED_RECOVERY_REPAIR_PATHS].sort()), "Failed recovery repair surface changed");
  requireCondition(JSON.stringify(sortedDiff(FAILED_RECOVERY_SOURCE_COMMIT, SECOND_RECOVERY_SOURCE_COMMIT)) ===
    JSON.stringify([...SECOND_RECOVERY_REPAIR_PATHS].sort()), "Second recovery repair surface changed");
  requireAddedPacket(FAILED_EVALUATION_COMMIT, FAILED_EVALUATION_PATHS, "Failed evaluation");
  requireAddedPacket(REPLACEMENT_SELECTION_COMMIT, REPLACEMENT_SELECTION_PATHS, "Replacement selection");
  const repairPaths = sortedDiff(REPLACEMENT_SELECTION_COMMIT, head);
  requireCondition(JSON.stringify(repairPaths) === JSON.stringify([...RECOVERY_REPAIR_PATHS].sort()),
    `A4 changed outside the exact repair set: ${repairPaths.join(", ")}`);
}

const treePaths = git(["ls-tree", "-r", "--name-only", "HEAD"]).split("\n").filter(Boolean);
const prohibited = treePaths.filter(isProhibitedPreScorePath);
requireCondition(prohibited.length === 0,
  `Pre-score commit contains replacement output: ${prohibited.join(", ")}`);

if (stage === "pre-score-freeze") {
  const result = spawnSync(process.execPath, ["scripts/check-pre-score-freeze.mjs"], { stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

console.log(JSON.stringify({ stage, head, replacementSelectionCommit: REPLACEMENT_SELECTION_COMMIT,
  postScoreEvidence: "absent", policy: "pass" }));
