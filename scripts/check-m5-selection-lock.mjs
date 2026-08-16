import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import {
  M5_BASE_SOURCE_COMMIT,
  M5_BASE_SOURCE_TREE,
  M5_FAILURE_PATH,
  M5_FINAL_RECEIPT_PATH,
  M5_LARGE_EVALUATION_PATH,
  M5_LARGE_SOURCE_LOCK_PATH,
  M5_LOCK_EXPECTED,
  M5_ORIGINAL_PROTOCOL_COMMIT,
  M5_ORIGINAL_PROTOCOL_TREE,
  M5_PROTOCOL_EXPECTED,
  M5_PROTOCOL_RECOVERY_EXPECTED,
  M5_SELECTION_LOCK_PATH,
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
      if (!status || !pathname || extra !== undefined) throw new Error(`Malformed M5 diff row: ${line}`);
      return [pathname, status];
    });
}

const head = git(["rev-parse", "HEAD"]);
const headParents = git(["rev-list", "--parents", "-n", "1", head]).split(" ").slice(1);
if (headParents.length !== 1 || !matchesExpectedRows(rows(head), M5_LOCK_EXPECTED)) {
  throw new Error("M5 selection lock must be the exact one-file direct child of the protocol commit");
}
const protocol = headParents[0];
const recoveryParents = git(["rev-list", "--parents", "-n", "1", protocol]).split(" ").slice(1);
const originalParents = git(["rev-list", "--parents", "-n", "1", M5_ORIGINAL_PROTOCOL_COMMIT]).split(" ").slice(1);
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
  throw new Error("M5 selection lock has the wrong protocol ancestry");
}
if (git(["status", "--porcelain=v1", "--untracked-files=all"])) {
  throw new Error("M5 selection-lock verification requires a completely clean repository");
}
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
].join(";")], { stdio: "inherit" });
console.log(JSON.stringify({ head, protocol, originalProtocol: M5_ORIGINAL_PROTOCOL_COMMIT, paths: M5_PROTOCOL_EXPECTED.size + M5_PROTOCOL_RECOVERY_EXPECTED.size, policy: "pass" }));
