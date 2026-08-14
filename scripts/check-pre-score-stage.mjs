import { execFileSync, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import {
  classifyReleaseStage,
  FAILED_RECOVERY_REPAIR_PATHS,
  FAILED_RECOVERY_SOURCE_COMMIT,
  FAILED_RECOVERY_SOURCE_TREE,
  FREEZE_PATH,
  LEGACY_FREEZE_COMMIT,
  LEGACY_FREEZE_PATH,
  LEGACY_FREEZE_SHA256,
  LEGACY_SOURCE_COMMIT,
  RECOVERY_REPAIR_PATHS,
  freezeReceiptAdditions,
  isProhibitedPreScorePath,
} from "./release-stage-policy.mjs";

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8" }).trim();
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "Pre-score verification requires a completely clean worktree");
const freezeExists = existsSync(FREEZE_PATH);
const legacyFreezeExists = existsSync(LEGACY_FREEZE_PATH);
const head = git(["rev-parse", "HEAD"]);
const additions = git(["log", "--no-renames", "--diff-filter=A", "--format=%H", "--", FREEZE_PATH])
  .split("\n").filter(Boolean);
requireCondition(additions.length <= 1 && (!freezeExists || additions.length === 1),
  "The pre-score freeze must have exactly one committed addition");
const legacyRecoverySource = !freezeExists && additions.length === 0 && legacyFreezeExists;
const stage = classifyReleaseStage({ freezeExists, head, freezeCommit: additions[0], legacyRecoverySource });
requireCondition(stage !== "final", "A descendant of the public freeze must pass the complete final evidence gate");
const addedFreezeReceipts = freezeReceiptAdditions(git([
  "log", "--no-renames", "--diff-filter=A", "--format=", "--name-only", "HEAD",
]).split("\n"));
const expectedFreezeReceipts = stage === "pre-score-freeze"
  ? [LEGACY_FREEZE_PATH, FREEZE_PATH].sort()
  : (legacyRecoverySource ? [LEGACY_FREEZE_PATH] : []);
requireCondition(JSON.stringify(addedFreezeReceipts) === JSON.stringify(expectedFreezeReceipts),
  "Pre-score history contains an alternate freeze receipt");

if (stage === "pre-score-recovery-source") {
  requireCondition(git(["rev-parse", `${LEGACY_FREEZE_COMMIT}^`]) === LEGACY_SOURCE_COMMIT &&
    git(["rev-parse", `${FAILED_RECOVERY_SOURCE_COMMIT}^`]) === LEGACY_FREEZE_COMMIT &&
    git(["rev-parse", `${FAILED_RECOVERY_SOURCE_COMMIT}^{tree}`]) === FAILED_RECOVERY_SOURCE_TREE &&
    git(["rev-parse", `${head}^`]) === FAILED_RECOVERY_SOURCE_COMMIT,
  "Second recovery source lineage changed");
  const legacyFreezePaths = git(["diff-tree", "--no-renames", "--no-commit-id", "--name-only", "-r", LEGACY_FREEZE_COMMIT])
    .split("\n").filter(Boolean);
  requireCondition(JSON.stringify(legacyFreezePaths) === JSON.stringify([LEGACY_FREEZE_PATH]),
    "Legacy freeze history changed");
  const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
  requireCondition(hash(readFileSync(LEGACY_FREEZE_PATH)) === LEGACY_FREEZE_SHA256 &&
    hash(execFileSync("git", ["show", `${LEGACY_FREEZE_COMMIT}:${LEGACY_FREEZE_PATH}`])) === LEGACY_FREEZE_SHA256,
  "Legacy freeze receipt bytes changed");
  const failedRepairPaths = git([
    "diff", "--no-renames", "--name-only", `${LEGACY_FREEZE_COMMIT}..${FAILED_RECOVERY_SOURCE_COMMIT}`,
  ]).split("\n").filter(Boolean).sort();
  requireCondition(JSON.stringify(failedRepairPaths) ===
    JSON.stringify([...FAILED_RECOVERY_REPAIR_PATHS].sort()),
  `Failed first recovery changed outside the exact repair set: ${failedRepairPaths.join(", ")}`);
  const repairPaths = git(["diff", "--no-renames", "--name-only", `${FAILED_RECOVERY_SOURCE_COMMIT}..${head}`])
    .split("\n").filter(Boolean).sort();
  requireCondition(JSON.stringify(repairPaths) === JSON.stringify([...RECOVERY_REPAIR_PATHS].sort()),
    `Recovery source changed outside the exact repair set: ${repairPaths.join(", ")}`);
}

const treePaths = git(["ls-tree", "-r", "--name-only", "HEAD"]).split("\n").filter(Boolean);
const prohibited = treePaths.filter(isProhibitedPreScorePath);
requireCondition(prohibited.length === 0,
  `Pre-score commit contains post-score evidence: ${prohibited.join(", ")}`);

if (stage === "pre-score-freeze") {
  const result = spawnSync(process.execPath, ["scripts/check-pre-score-freeze.mjs"], { stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

console.log(JSON.stringify({ stage, head,
  failedRecoverySourceCommit: legacyRecoverySource ? FAILED_RECOVERY_SOURCE_COMMIT : undefined,
  legacyFreezeCommit: legacyRecoverySource ? LEGACY_FREEZE_COMMIT : undefined,
  postScoreEvidence: "absent", policy: "pass" }));
