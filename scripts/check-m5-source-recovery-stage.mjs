import { existsSync } from "node:fs";
import { validateM5A5AuthorizedChain } from "./check-m5-authorized-chain.mjs";
import { M5_A5_COMMIT, M5_A5_TREE, M5_A6_AUTHORIZATION_PATH, M5_R6_EXPECTED, matchesExpectedRows } from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git } from "./m5-safe-git.mjs";

const git = (arguments_) => m5Git(arguments_);
const rows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, pathname] = line.split("\t"); return [pathname, status]; });
const parents = (commit) => git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
const head = git(["rev-parse", "HEAD"]);
const inherited = validateM5A5AuthorizedChain();
if (inherited.authorization !== M5_A5_COMMIT || parents(head).length !== 1 || parents(head)[0] !== M5_A5_COMMIT ||
    git(["rev-parse", `${M5_A5_COMMIT}^{tree}`]) !== M5_A5_TREE || !matchesExpectedRows(rows(head), M5_R6_EXPECTED) ||
    existsSync(M5_A6_AUTHORIZATION_PATH)) throw new Error("M5 R6 source recovery lineage or surface changed");
assertM5WorktreeExact();
console.log(JSON.stringify({ head, parent: M5_A5_COMMIT, paths: M5_R6_EXPECTED.size, policy: "pass" }));
