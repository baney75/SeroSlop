import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import {
  M3_BASE_COMMIT,
  M3_PUBLICATION_LOCK_PATH,
  M3_SOURCE_EXPECTED,
  matchesExpectedRows,
} from "./m3-stage-policy.mjs";

const M3_BASE_TREE = "32aa561cfb12ccda59aee919d3ca2b3761b07d9c";
const M2_MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47";
const FORBIDDEN_OUTPUTS = [
  M3_PUBLICATION_LOCK_PATH,
  "benchmark/evidence/m3/calibration.json",
  "benchmark/evidence/m3/candidate-grid.json",
  "benchmark/evidence/m3/finalization-receipt.json",
  "benchmark/evidence/m3/model-comparison.json",
  "benchmark/evidence/m3/training-summary.json",
];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8" }).trim();
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function commitRows(commit) {
  return git(["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, path] = line.split("\t");
      return [path, status];
    });
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M3 source verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
const parents = git(["show", "-s", "--format=%P", head]).split(" ").filter(Boolean);
requireCondition(parents.length === 1 && parents[0] === M3_BASE_COMMIT,
  "The M3 source freeze must be the base release's direct single-parent child");
requireCondition(git(["rev-parse", `${M3_BASE_COMMIT}^{tree}`]) === M3_BASE_TREE,
  "The frozen M3 base tree changed");
requireCondition(matchesExpectedRows(commitRows(head), M3_SOURCE_EXPECTED),
  "The M3 source freeze changed outside its exact source-only packet");
for (const path of FORBIDDEN_OUTPUTS) {
  requireCondition(!existsSync(path), `M3 source freeze must precede candidate/output publication: ${path}`);
}
requireCondition(digest(readFileSync("weights/prooflens-cf384.onnx")) === M2_MODEL_SHA256,
  "M3 source freeze must retain the reviewed M2 model bytes");
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");
requireCondition(git(["ls-files", "benchmark/data/m3-head", "benchmark/data/m3-source", "benchmark/data/h3-met-holdout-v1"]) === "",
  "M3 source pixels must remain outside Git");

console.log(JSON.stringify({
  head,
  parent: M3_BASE_COMMIT,
  paths: M3_SOURCE_EXPECTED.size,
  modelSha256: M2_MODEL_SHA256,
  trainingOutputsPresent: false,
  stage: "m3-source",
  policy: "pass",
}));
