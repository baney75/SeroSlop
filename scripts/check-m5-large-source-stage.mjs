import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { validateM5AuthorizedChain } from "./check-m5-authorized-chain.mjs";
import { assertM5WorktreeExact, m5Git } from "./m5-safe-git.mjs";
import {
  M5_FAILURE_PATH,
  M5_FINAL_RECEIPT_PATH,
  M5_LARGE_EVALUATION_PATH,
  M5_LARGE_SOURCE_EXPECTED,
  M5_LARGE_SOURCE_LOCK_PATH,
  M5_LOCK_EXPECTED,
  M5_SELECTION_LOCK_PATH,
  matchesExpectedRows,
} from "./m5-stage-policy.mjs";

function git(arguments_) {
  return m5Git(arguments_);
}
const authorized = validateM5AuthorizedChain();

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
if (lockParents[0] !== authorized.authorization) throw new Error("M5 selection lock skipped P4 authorization");
const protocol = authorized.source;
assertM5WorktreeExact();
for (const forbidden of [M5_FAILURE_PATH, M5_FINAL_RECEIPT_PATH, M5_LARGE_EVALUATION_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 100K source lock contains forbidden output: ${forbidden}`);
}
execFileSync("python3", ["benchmark/m5/large_synthetic.py", "--verify-public"], { stdio: "inherit" });
const sourceLock = JSON.parse(readFileSync(M5_LARGE_SOURCE_LOCK_PATH, "utf8"));
const selectionLock = JSON.parse(readFileSync(M5_SELECTION_LOCK_PATH, "utf8"));
const environment = selectionLock.trainingSummary?.environment;
if (sourceLock.lockCommit !== lockCommit || sourceLock.protocolCommit !== protocol ||
    selectionLock.protocolCommit !== protocol || environment?.sourceCommit !== authorized.source ||
    environment?.sourceTree !== authorized.sourceTree || environment?.authorizationCommit !== authorized.authorization ||
    environment?.authorizationReceiptSha256 !== authorized.authorizationReceiptSha256 ||
    JSON.stringify(sourceLock.scoreBlindness) !== JSON.stringify({
      repositoryScoreArtifactsPresentAtSourceLock: false,
      publicSourceLockPrecedesEvaluationReceipt: true,
      firstInferenceAfterLock: "operator-attested",
      privatePriorScoringAbsenceProven: false,
      trainingExclusionClaim: "not-used-in-seroslop-m2-through-m5-gradients-or-selection",
    }) || sourceLock.selectionInfluence !== false || sourceLock.h3PixelsRead !== false) {
  throw new Error("M5 100K source-lock ancestry or score-blind boundary changed");
}
console.log(JSON.stringify({ head, lockCommit, protocol, authorization: authorized.authorization, paths: M5_LARGE_SOURCE_EXPECTED.size, policy: "pass" }));
