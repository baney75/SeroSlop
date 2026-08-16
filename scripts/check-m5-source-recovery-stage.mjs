import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { M5_CI_RECOVERY_COMMIT, M5_CI_RECOVERY_TREE, M5_FAILED_SOURCE_COMMIT, M5_FAILED_SOURCE_TREE, M5_FAILURE_PATH, M5_FINAL_RECEIPT_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_P2_COMMIT, M5_P2_TREE, M5_P4_AUTHORIZATION_SHA256, M5_P4_COMMIT, M5_P4_TREE, M5_RUN_AUTHORIZATION_PATH, M5_RUNTIME_AUTHORIZATION_PATH, M5_RUNTIME_RECOVERY_EXPECTED, M5_SELECTION_LOCK_PATH, M5_SOURCE_CI_RECOVERY_EXPECTED, M5_SOURCE_RECOVERY_EXPECTED, matchesExpectedRows } from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git, m5GitBytes } from "./m5-safe-git.mjs";
const git = (args) => m5Git(args);
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const head = git(["rev-parse", "HEAD"]);
const parent = git(["rev-list", "--parents", "-n", "1", head]).split(" ")[1];
const commitRows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit]).split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
const p4Parents = git(["rev-list", "--parents", "-n", "1", M5_P4_COMMIT]).split(" ").slice(1);
const ciRecoveryParents = git(["rev-list", "--parents", "-n", "1", M5_CI_RECOVERY_COMMIT]).split(" ").slice(1);
const failedParents = git(["rev-list", "--parents", "-n", "1", M5_FAILED_SOURCE_COMMIT]).split(" ").slice(1);
if (parent !== M5_P4_COMMIT ||
    git(["rev-parse", `${M5_P4_COMMIT}^{tree}`]) !== M5_P4_TREE ||
    p4Parents.length !== 1 || p4Parents[0] !== M5_CI_RECOVERY_COMMIT ||
    !matchesExpectedRows(commitRows(M5_P4_COMMIT), new Map([[M5_RUN_AUTHORIZATION_PATH, "A"]])) ||
    digest(m5GitBytes(["show", `${M5_P4_COMMIT}:${M5_RUN_AUTHORIZATION_PATH}`])) !== M5_P4_AUTHORIZATION_SHA256 ||
    git(["rev-parse", `${M5_CI_RECOVERY_COMMIT}^{tree}`]) !== M5_CI_RECOVERY_TREE ||
    ciRecoveryParents.length !== 1 || ciRecoveryParents[0] !== M5_FAILED_SOURCE_COMMIT ||
    !matchesExpectedRows(commitRows(M5_CI_RECOVERY_COMMIT), M5_SOURCE_CI_RECOVERY_EXPECTED) ||
    git(["rev-parse", `${M5_FAILED_SOURCE_COMMIT}^{tree}`]) !== M5_FAILED_SOURCE_TREE ||
    failedParents.length !== 1 || failedParents[0] !== M5_P2_COMMIT ||
    git(["rev-parse", `${M5_P2_COMMIT}^{tree}`]) !== M5_P2_TREE ||
    !matchesExpectedRows(commitRows(M5_FAILED_SOURCE_COMMIT), M5_SOURCE_RECOVERY_EXPECTED) ||
    !matchesExpectedRows(commitRows(head), M5_RUNTIME_RECOVERY_EXPECTED)) {
  throw new Error("M5 append-only runtime recovery stage is not exact");
}
assertM5WorktreeExact();
if (!existsSync(M5_RUN_AUTHORIZATION_PATH)) throw new Error("M5 runtime recovery lost the inherited P4 receipt");
for (const forbidden of [M5_RUNTIME_AUTHORIZATION_PATH, M5_SELECTION_LOCK_PATH, M5_FAILURE_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_FINAL_RECEIPT_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 runtime recovery contains forbidden later evidence: ${forbidden}`);
}
console.log(JSON.stringify({ head, parent, priorAuthorization: M5_P4_COMMIT, policy: "pass" }));
