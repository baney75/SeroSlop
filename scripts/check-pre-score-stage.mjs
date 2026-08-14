import { execFileSync, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import {
  classifyReleaseStage,
  FREEZE_PATH,
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
const head = git(["rev-parse", "HEAD"]);
const additions = git(["log", "--diff-filter=A", "--format=%H", "--", FREEZE_PATH])
  .split("\n").filter(Boolean);
requireCondition(additions.length <= 1 && (!freezeExists || additions.length === 1),
  "The pre-score freeze must have exactly one committed addition");
const stage = classifyReleaseStage({ freezeExists, head, freezeCommit: additions[0] });
requireCondition(stage !== "final", "A descendant of the public freeze must pass the complete final evidence gate");

const treePaths = git(["ls-tree", "-r", "--name-only", "HEAD"]).split("\n").filter(Boolean);
const prohibited = treePaths.filter(isProhibitedPreScorePath);
requireCondition(prohibited.length === 0,
  `Pre-score commit contains post-score evidence: ${prohibited.join(", ")}`);

if (stage === "pre-score-freeze") {
  const result = spawnSync(process.execPath, ["scripts/check-pre-score-freeze.mjs"], { stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}

console.log(JSON.stringify({ stage, head, postScoreEvidence: "absent", policy: "pass" }));
