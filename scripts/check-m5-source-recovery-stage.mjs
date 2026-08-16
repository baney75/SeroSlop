import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import {
  M5_CI_RECOVERY_COMMIT,
  M5_CI_RECOVERY_TREE,
  M5_FAILED_SOURCE_COMMIT,
  M5_FAILED_SOURCE_TREE,
  M5_FAILURE_PATH,
  M5_FINAL_RECEIPT_PATH,
  M5_LARGE_SOURCE_LOCK_PATH,
  M5_P2_COMMIT,
  M5_P2_TREE,
  M5_P4_AUTHORIZATION_SHA256,
  M5_P4_COMMIT,
  M5_P4_TREE,
  M5_NUMERIC_AUDIT_AUTHORIZATION_PATH,
  M5_NUMERIC_AUDIT_RECOVERY_EXPECTED,
  M5_RUN_AUTHORIZATION_PATH,
  M5_RUNPOD_ENV_AUTHORIZATION_COMMIT,
  M5_RUNPOD_ENV_AUTHORIZATION_PATH,
  M5_RUNPOD_ENV_AUTHORIZATION_SHA256,
  M5_RUNPOD_ENV_AUTHORIZATION_TREE,
  M5_RUNPOD_ENV_RECOVERY_COMMIT,
  M5_RUNPOD_ENV_RECOVERY_EXPECTED,
  M5_RUNPOD_ENV_RECOVERY_TREE,
  M5_RUNTIME_AUTHORIZATION_COMMIT,
  M5_RUNTIME_AUTHORIZATION_PATH,
  M5_RUNTIME_AUTHORIZATION_SHA256,
  M5_RUNTIME_AUTHORIZATION_TREE,
  M5_RUNTIME_RECOVERY_COMMIT,
  M5_RUNTIME_RECOVERY_EXPECTED,
  M5_RUNTIME_RECOVERY_TREE,
  M5_SELECTION_LOCK_PATH,
  M5_SOURCE_CI_RECOVERY_EXPECTED,
  M5_SOURCE_RECOVERY_EXPECTED,
  matchesExpectedRows,
} from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git, m5GitBytes } from "./m5-safe-git.mjs";

const git = (args) => m5Git(args);
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const commitRows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
const parents = (commit) => git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
const head = git(["rev-parse", "HEAD"]);

if (parents(head).length !== 1 || parents(head)[0] !== M5_RUNPOD_ENV_AUTHORIZATION_COMMIT ||
    !matchesExpectedRows(commitRows(head), M5_NUMERIC_AUDIT_RECOVERY_EXPECTED) ||
    git(["rev-parse", `${M5_RUNPOD_ENV_AUTHORIZATION_COMMIT}^{tree}`]) !== M5_RUNPOD_ENV_AUTHORIZATION_TREE ||
    parents(M5_RUNPOD_ENV_AUTHORIZATION_COMMIT).length !== 1 || parents(M5_RUNPOD_ENV_AUTHORIZATION_COMMIT)[0] !== M5_RUNPOD_ENV_RECOVERY_COMMIT ||
    !matchesExpectedRows(commitRows(M5_RUNPOD_ENV_AUTHORIZATION_COMMIT), new Map([[M5_RUNPOD_ENV_AUTHORIZATION_PATH, "A"]])) ||
    digest(m5GitBytes(["show", `${M5_RUNPOD_ENV_AUTHORIZATION_COMMIT}:${M5_RUNPOD_ENV_AUTHORIZATION_PATH}`])) !== M5_RUNPOD_ENV_AUTHORIZATION_SHA256 ||
    git(["rev-parse", `${M5_RUNPOD_ENV_RECOVERY_COMMIT}^{tree}`]) !== M5_RUNPOD_ENV_RECOVERY_TREE ||
    parents(M5_RUNPOD_ENV_RECOVERY_COMMIT).length !== 1 || parents(M5_RUNPOD_ENV_RECOVERY_COMMIT)[0] !== M5_RUNTIME_AUTHORIZATION_COMMIT ||
    !matchesExpectedRows(commitRows(M5_RUNPOD_ENV_RECOVERY_COMMIT), M5_RUNPOD_ENV_RECOVERY_EXPECTED) ||
    git(["rev-parse", `${M5_RUNTIME_AUTHORIZATION_COMMIT}^{tree}`]) !== M5_RUNTIME_AUTHORIZATION_TREE ||
    parents(M5_RUNTIME_AUTHORIZATION_COMMIT).length !== 1 || parents(M5_RUNTIME_AUTHORIZATION_COMMIT)[0] !== M5_RUNTIME_RECOVERY_COMMIT ||
    !matchesExpectedRows(commitRows(M5_RUNTIME_AUTHORIZATION_COMMIT), new Map([[M5_RUNTIME_AUTHORIZATION_PATH, "A"]])) ||
    digest(m5GitBytes(["show", `${M5_RUNTIME_AUTHORIZATION_COMMIT}:${M5_RUNTIME_AUTHORIZATION_PATH}`])) !== M5_RUNTIME_AUTHORIZATION_SHA256 ||
    git(["rev-parse", `${M5_RUNTIME_RECOVERY_COMMIT}^{tree}`]) !== M5_RUNTIME_RECOVERY_TREE ||
    parents(M5_RUNTIME_RECOVERY_COMMIT).length !== 1 || parents(M5_RUNTIME_RECOVERY_COMMIT)[0] !== M5_P4_COMMIT ||
    !matchesExpectedRows(commitRows(M5_RUNTIME_RECOVERY_COMMIT), M5_RUNTIME_RECOVERY_EXPECTED) ||
    git(["rev-parse", `${M5_P4_COMMIT}^{tree}`]) !== M5_P4_TREE ||
    parents(M5_P4_COMMIT).length !== 1 || parents(M5_P4_COMMIT)[0] !== M5_CI_RECOVERY_COMMIT ||
    !matchesExpectedRows(commitRows(M5_P4_COMMIT), new Map([[M5_RUN_AUTHORIZATION_PATH, "A"]])) ||
    digest(m5GitBytes(["show", `${M5_P4_COMMIT}:${M5_RUN_AUTHORIZATION_PATH}`])) !== M5_P4_AUTHORIZATION_SHA256 ||
    git(["rev-parse", `${M5_CI_RECOVERY_COMMIT}^{tree}`]) !== M5_CI_RECOVERY_TREE ||
    parents(M5_CI_RECOVERY_COMMIT).length !== 1 || parents(M5_CI_RECOVERY_COMMIT)[0] !== M5_FAILED_SOURCE_COMMIT ||
    !matchesExpectedRows(commitRows(M5_CI_RECOVERY_COMMIT), M5_SOURCE_CI_RECOVERY_EXPECTED) ||
    git(["rev-parse", `${M5_FAILED_SOURCE_COMMIT}^{tree}`]) !== M5_FAILED_SOURCE_TREE ||
    parents(M5_FAILED_SOURCE_COMMIT).length !== 1 || parents(M5_FAILED_SOURCE_COMMIT)[0] !== M5_P2_COMMIT ||
    git(["rev-parse", `${M5_P2_COMMIT}^{tree}`]) !== M5_P2_TREE ||
    !matchesExpectedRows(commitRows(M5_FAILED_SOURCE_COMMIT), M5_SOURCE_RECOVERY_EXPECTED)) {
  throw new Error("M5 append-only numeric audit recovery stage is not exact");
}
assertM5WorktreeExact();
if (!existsSync(M5_RUN_AUTHORIZATION_PATH) || !existsSync(M5_RUNTIME_AUTHORIZATION_PATH) || !existsSync(M5_RUNPOD_ENV_AUTHORIZATION_PATH)) {
  throw new Error("M5 numeric audit recovery lost an inherited authorization receipt");
}
for (const forbidden of [M5_NUMERIC_AUDIT_AUTHORIZATION_PATH, M5_SELECTION_LOCK_PATH, M5_FAILURE_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_FINAL_RECEIPT_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 numeric audit recovery contains forbidden later evidence: ${forbidden}`);
}
console.log(JSON.stringify({ head, parent: M5_RUNPOD_ENV_AUTHORIZATION_COMMIT, priorAuthorization: M5_RUNPOD_ENV_AUTHORIZATION_COMMIT, policy: "pass" }));
