import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import {
  M5_BASE_SOURCE_COMMIT,
  M5_BASE_SOURCE_TREE,
  M5_FAILURE_PATH,
  M5_FINAL_RECEIPT_PATH,
  M5_LARGE_EVALUATION_PATH,
  M5_LARGE_SOURCE_EXPECTED,
  M5_LARGE_SOURCE_LOCK_PATH,
  M5_LOCK_EXPECTED,
  M5_ORIGINAL_PROTOCOL_COMMIT,
  M5_ORIGINAL_PROTOCOL_TREE,
  matchesExpectedRows,
  matchesM5ProtocolLineage,
} from "./m5-stage-policy.mjs";

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }).trim();
}

function rows(commit) {
  return git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, pathname, extra] = line.split("\t");
      if (!status || !pathname || extra !== undefined) throw new Error(`Malformed M5 100K source row: ${line}`);
      return [pathname, status];
    });
}

function parents(commit) {
  return git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
}

const head = git(["rev-parse", "HEAD"]);
const sourceParents = parents(head);
if (sourceParents.length !== 1 || !matchesExpectedRows(rows(head), M5_LARGE_SOURCE_EXPECTED)) {
  throw new Error("M5 100K source lock is not the exact four-file packet");
}
const lockCommit = sourceParents[0];
const lockParents = parents(lockCommit);
if (lockParents.length !== 1 || !matchesExpectedRows(rows(lockCommit), M5_LOCK_EXPECTED)) {
  throw new Error("M5 100K source lock is not the direct child of the one-file selection lock");
}
const protocol = lockParents[0];
const recoveryParents = parents(protocol);
const originalParents = parents(M5_ORIGINAL_PROTOCOL_COMMIT);
const originalTree = git(["rev-parse", `${M5_ORIGINAL_PROTOCOL_COMMIT}^{tree}`]);
const baseTree = git(["rev-parse", `${M5_BASE_SOURCE_COMMIT}^{tree}`]);
if (!matchesM5ProtocolLineage({
  recoveryParents,
  recoveryRows: rows(protocol),
  originalTree,
  originalParents,
  originalRows: rows(M5_ORIGINAL_PROTOCOL_COMMIT),
  baseTree,
}) || originalTree !== M5_ORIGINAL_PROTOCOL_TREE || baseTree !== M5_BASE_SOURCE_TREE) {
  throw new Error("M5 100K source lock has the wrong protocol ancestry");
}
if (git(["status", "--porcelain=v1", "--untracked-files=all"])) {
  throw new Error("M5 100K source verification requires a completely clean repository");
}
for (const forbidden of [M5_FAILURE_PATH, M5_FINAL_RECEIPT_PATH, M5_LARGE_EVALUATION_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 100K source lock contains forbidden output: ${forbidden}`);
}
execFileSync("python3", ["benchmark/m5/large_synthetic.py", "--verify-public"], { stdio: "inherit" });
const sourceLock = JSON.parse(readFileSync(M5_LARGE_SOURCE_LOCK_PATH, "utf8"));
if (sourceLock.lockCommit !== lockCommit || sourceLock.protocolCommit !== protocol ||
    JSON.stringify(sourceLock.scoreBlindness) !== JSON.stringify({
      repositoryScoreArtifactsPresentAtSourceLock: false,
      publicSourceLockPrecedesEvaluationReceipt: true,
      firstInferenceAfterLock: "operator-attested",
      privatePriorScoringAbsenceProven: false,
      trainingExclusionClaim: "not-used-in-seroslop-m2-through-m5-gradients-or-selection",
    }) || sourceLock.selectionInfluence !== false || sourceLock.h3PixelsRead !== false) {
  throw new Error("M5 100K source-lock ancestry or score-blind boundary changed");
}
console.log(JSON.stringify({ head, lockCommit, protocol, paths: M5_LARGE_SOURCE_EXPECTED.size, policy: "pass" }));
