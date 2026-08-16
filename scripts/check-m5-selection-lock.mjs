import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { validateM5AuthorizedChain } from "./check-m5-authorized-chain.mjs";
import { assertM5WorktreeExact, m5Git } from "./m5-safe-git.mjs";
import {
  M5_FAILURE_PATH,
  M5_FINAL_RECEIPT_PATH,
  M5_LARGE_EVALUATION_PATH,
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
      if (!status || !pathname || extra !== undefined) throw new Error(`Malformed M5 diff row: ${line}`);
      return [pathname, status];
    });
}

const head = git(["rev-parse", "HEAD"]);
const headParents = git(["rev-list", "--parents", "-n", "1", head]).split(" ").slice(1);
if (headParents.length !== 1 || !matchesExpectedRows(rows(head), M5_LOCK_EXPECTED)) {
  throw new Error("M5 selection lock must be an exact one-file commit");
}
if (headParents[0] !== authorized.authorization) throw new Error("M5 selection lock must directly follow P4 authorization");
const protocol = authorized.source;
assertM5WorktreeExact();
for (const forbidden of [M5_FAILURE_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_LARGE_EVALUATION_PATH, M5_FINAL_RECEIPT_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 selection-lock stage contains forbidden output: ${forbidden}`);
}
execFileSync("python3", ["-c", [
  "from pathlib import Path",
  "from benchmark.m5.contracts import load_recipe,parse_json_bytes,read_jsonl,validate_selection_lock",
  "r=load_recipe(Path('benchmark/m5/recipe.json'))",
  `l=parse_json_bytes(Path('${M5_SELECTION_LOCK_PATH}').read_bytes(),label='selection lock')`,
  "rows=read_jsonl(Path(r['sourceEvidence']['selectorManifest']['path']))",
  "validate_selection_lock(l,r,rows)",
  `assert l['protocolCommit']=='${protocol}'`,
  "env=l['trainingSummary']['environment']",
  `assert env['sourceCommit']=='${authorized.source}' and env['sourceTree']=='${authorized.sourceTree}'`,
  `assert env['authorizationCommit']=='${authorized.authorization}' and env['authorizationReceiptSha256']=='${authorized.authorizationReceiptSha256}'`,
].join(";")], { stdio: "inherit" });
console.log(JSON.stringify({ head, protocol, authorization: authorized.authorization, policy: "pass" }));
