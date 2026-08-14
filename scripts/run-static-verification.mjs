import { execFileSync, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { classifyReleaseStage, FREEZE_PATH } from "./release-stage-policy.mjs";

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8" }).trim();
}

const freezeExists = existsSync(FREEZE_PATH);
const head = git(["rev-parse", "HEAD"]);
const additions = git(["log", "--diff-filter=A", "--format=%H", "--", FREEZE_PATH])
  .split("\n").filter(Boolean);
if (additions.length > 1 || (freezeExists && additions.length !== 1)) {
  throw new Error("The pre-score freeze must have exactly one committed addition");
}
const stage = classifyReleaseStage({ freezeExists, head, freezeCommit: additions[0] });
const script = stage === "final" ? "verify:final" : "verify:pre-score";
console.log(JSON.stringify({ stage, script }));
const result = spawnSync("npm", ["run", script], { stdio: "inherit" });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
