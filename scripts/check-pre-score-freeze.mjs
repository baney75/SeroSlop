import { execFileSync, spawnSync } from "node:child_process";
import { FREEZE_PATH } from "./release-stage-policy.mjs";

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8" }).trim();
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "Freeze verification requires a completely clean index, worktree, and untracked-file set");
const head = git(["rev-parse", "HEAD"]);
const additions = git(["log", "--no-renames", "--diff-filter=A", "--format=%H", "--", FREEZE_PATH])
  .split("\n").filter(Boolean);
requireCondition(additions.length === 1, "V3 must have one immutable receipt addition");
const freezeCommit = additions[0];
const allowDescendant = head !== freezeCommit;
const source = [
  "import json, pathlib, sys",
  "sys.path.insert(0, str(pathlib.Path('benchmark').resolve()))",
  "from evaluation_contract import require_public_pre_score_freeze",
  `proof = require_public_pre_score_freeze(repository_root=pathlib.Path.cwd(), allow_public_descendant=${allowDescendant ? "True" : "False"})`,
  "print(json.dumps({'sourceCommit': proof['sourceCommit'], 'freezeCommit': proof['freezeCommit'], 'policy': 'pass'}))",
].join("; ");
const result = spawnSync("python3", ["-c", source], { encoding: "utf8", timeout: 180_000 });
if (result.error) throw result.error;
if (result.status !== 0) {
  process.stderr.write(result.stderr);
  process.exit(result.status ?? 1);
}
process.stdout.write(result.stdout);
