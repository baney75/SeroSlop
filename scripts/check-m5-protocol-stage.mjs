import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import {
  M5_BASE_SOURCE_COMMIT,
  M5_BASE_SOURCE_TREE,
  M5_FAILURE_PATH,
  M5_FINAL_RECEIPT_PATH,
  M5_LARGE_EVALUATION_PATH,
  M5_LARGE_SOURCE_LOCK_PATH,
  M5_ORIGINAL_PROTOCOL_COMMIT,
  M5_ORIGINAL_PROTOCOL_TREE,
  M5_PROTOCOL_EXPECTED,
  M5_PROTOCOL_RECOVERY_EXPECTED,
  M5_SELECTION_LOCK_PATH,
  matchesM5ProtocolLineage,
} from "./m5-stage-policy.mjs";
import { loadAndValidateM5Recipe } from "./m5-training-contract.mjs";
import { assertM5WorktreeExact, m5Git } from "./m5-safe-git.mjs";

function git(arguments_) {
  return m5Git(arguments_);
}

function digest(pathname) {
  return createHash("sha256").update(readFileSync(pathname)).digest("hex");
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
const recoveryParents = git(["rev-list", "--parents", "-n", "1", head]).split(" ").slice(1);
const originalParents = git(["rev-list", "--parents", "-n", "1", M5_ORIGINAL_PROTOCOL_COMMIT]).split(" ").slice(1);
const originalTree = git(["rev-parse", `${M5_ORIGINAL_PROTOCOL_COMMIT}^{tree}`]);
const baseTree = git(["rev-parse", `${M5_BASE_SOURCE_COMMIT}^{tree}`]);
if (!matchesM5ProtocolLineage({
  recoveryParents,
  recoveryRows: rows(head),
  originalTree,
  originalParents,
  originalRows: rows(M5_ORIGINAL_PROTOCOL_COMMIT),
  baseTree,
})) {
  throw new Error("M5 protocol recovery is not the exact append-only child of the original protocol");
}
if (originalTree !== M5_ORIGINAL_PROTOCOL_TREE || baseTree !== M5_BASE_SOURCE_TREE) throw new Error("M5 protocol lineage tree changed");
assertM5WorktreeExact();
for (const forbidden of [M5_SELECTION_LOCK_PATH, M5_FAILURE_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_LARGE_EVALUATION_PATH, M5_FINAL_RECEIPT_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 protocol stage contains forbidden output: ${forbidden}`);
}
const recipe = loadAndValidateM5Recipe();
const evidence = [
  [recipe.sourceEvidence.trainingManifest.trackedPath, recipe.sourceEvidence.trainingManifest.compressedSha256],
  [recipe.sourceEvidence.selectorManifest.path, recipe.sourceEvidence.selectorManifest.sha256],
  [recipe.sourceEvidence.selectionSummary.path, recipe.sourceEvidence.selectionSummary.sha256],
  [recipe.initialModel.path, recipe.initialModel.sha256],
  [recipe.initialModel.modelLock, recipe.initialModel.modelLockSha256],
  [recipe.h3Boundary.manifest, recipe.h3Boundary.sha256],
];
for (const [pathname, expected] of evidence) {
  if (digest(pathname) !== expected) throw new Error(`M5 frozen input changed: ${pathname}`);
}
if (M5_PROTOCOL_EXPECTED.size !== 38 || M5_PROTOCOL_RECOVERY_EXPECTED.size !== 19) throw new Error("M5 protocol inventory changed");
console.log(JSON.stringify({
  head,
  originalProtocol: M5_ORIGINAL_PROTOCOL_COMMIT,
  base: M5_BASE_SOURCE_COMMIT,
  originalPaths: M5_PROTOCOL_EXPECTED.size,
  recoveryPaths: M5_PROTOCOL_RECOVERY_EXPECTED.size,
  policy: "pass",
}));
