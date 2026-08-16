import { existsSync } from "node:fs";
import { M5_FAILED_SOURCE_COMMIT, M5_FAILED_SOURCE_TREE, M5_FAILURE_PATH, M5_FINAL_RECEIPT_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_P2_COMMIT, M5_P2_TREE, M5_RUN_AUTHORIZATION_PATH, M5_SELECTION_LOCK_PATH, M5_SOURCE_CI_RECOVERY_EXPECTED, M5_SOURCE_RECOVERY_EXPECTED, matchesExpectedRows } from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git } from "./m5-safe-git.mjs";
const git = (args) => m5Git(args);
const head = git(["rev-parse", "HEAD"]);
const parent = git(["rev-list", "--parents", "-n", "1", head]).split(" ")[1];
const commitRows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit]).split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
const failedParents = git(["rev-list", "--parents", "-n", "1", M5_FAILED_SOURCE_COMMIT]).split(" ").slice(1);
if (parent !== M5_FAILED_SOURCE_COMMIT ||
    git(["rev-parse", `${M5_FAILED_SOURCE_COMMIT}^{tree}`]) !== M5_FAILED_SOURCE_TREE ||
    failedParents.length !== 1 || failedParents[0] !== M5_P2_COMMIT ||
    git(["rev-parse", `${M5_P2_COMMIT}^{tree}`]) !== M5_P2_TREE ||
    !matchesExpectedRows(commitRows(M5_FAILED_SOURCE_COMMIT), M5_SOURCE_RECOVERY_EXPECTED) ||
    !matchesExpectedRows(commitRows(head), M5_SOURCE_CI_RECOVERY_EXPECTED)) {
  throw new Error("M5 append-only P3 CI recovery stage is not exact");
}
assertM5WorktreeExact();
for (const forbidden of [M5_RUN_AUTHORIZATION_PATH, M5_SELECTION_LOCK_PATH, M5_FAILURE_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_FINAL_RECEIPT_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 P3 contains forbidden later evidence: ${forbidden}`);
}
console.log(JSON.stringify({ head, parent, failedSource: M5_FAILED_SOURCE_COMMIT, policy: "pass" }));
