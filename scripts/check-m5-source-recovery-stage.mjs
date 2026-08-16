import { existsSync } from "node:fs";
import { M5_FAILURE_PATH, M5_FINAL_RECEIPT_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_P2_COMMIT, M5_P2_TREE, M5_RUN_AUTHORIZATION_PATH, M5_SELECTION_LOCK_PATH, M5_SOURCE_RECOVERY_EXPECTED, matchesExpectedRows } from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git } from "./m5-safe-git.mjs";
const git = (args) => m5Git(args);
const head = git(["rev-parse", "HEAD"]);
const parent = git(["rev-list", "--parents", "-n", "1", head]).split(" ")[1];
const rows = git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", head]).split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
if (parent !== M5_P2_COMMIT || git(["rev-parse", `${M5_P2_COMMIT}^{tree}`]) !== M5_P2_TREE || !matchesExpectedRows(rows, M5_SOURCE_RECOVERY_EXPECTED)) throw new Error("M5 P3 source recovery stage is not exact");
assertM5WorktreeExact();
for (const forbidden of [M5_RUN_AUTHORIZATION_PATH, M5_SELECTION_LOCK_PATH, M5_FAILURE_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_FINAL_RECEIPT_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 P3 contains forbidden later evidence: ${forbidden}`);
}
console.log(JSON.stringify({ head, parent, policy: "pass" }));
