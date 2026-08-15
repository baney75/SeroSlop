import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import {
  M4_BASE_COMMIT,
  M4_BASE_TREE,
  M4_PROTOCOL_EXPECTED,
  matchesExpectedRows,
} from "./m4-stage-policy.mjs";

const MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47";
const FORBIDDEN_OUTPUTS = [
  "benchmark/evidence/m4/attribution.json",
  "benchmark/evidence/m4/british-source-index.json.gz",
  "benchmark/evidence/m4/perceptual-review.json",
  "benchmark/evidence/m4/rapidata-source-index.json.gz",
  "benchmark/evidence/m4/rejects.jsonl.gz",
  "benchmark/evidence/m4/selection-summary.json",
  "benchmark/evidence/m4/train-manifest.jsonl.gz",
  "benchmark/evidence/m4/validation-manifest.jsonl",
  "benchmark/evidence/m4/publication-lock.json",
  "benchmark/evidence/m4/failed-selector-diagnostic-1.json",
  "benchmark/evidence/m4/failed-training-attempt-1.json",
  "benchmark/evidence/m4/calibration.json",
  "benchmark/evidence/m4/candidate-grid.json",
  "benchmark/evidence/m4/finalization-receipt.json",
  "benchmark/evidence/m4/model-comparison.json",
  "benchmark/evidence/m4/training-summary.json",
];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }).trim();
}

function commitRows(commit) {
  return git(["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, pathname] = line.split("\t");
      return [pathname, status];
    });
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M4 protocol verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
const parents = git(["show", "-s", "--format=%P", head]).split(" ").filter(Boolean);
requireCondition(parents.length === 1 && parents[0] === M4_BASE_COMMIT,
  "The M4 protocol must be the M3 failure commit's direct single-parent child");
requireCondition(git(["rev-parse", `${M4_BASE_COMMIT}^{tree}`]) === M4_BASE_TREE,
  "The frozen M4 base tree changed");
requireCondition(matchesExpectedRows(commitRows(head), M4_PROTOCOL_EXPECTED),
  "The M4 protocol commit changed outside its exact pre-materialization packet");
for (const pathname of FORBIDDEN_OUTPUTS) {
  requireCondition(!existsSync(pathname), `M4 protocol freeze must precede materialization/training: ${pathname}`);
}
for (const pathname of ["benchmark/data/m4-head", "benchmark/data/m4-source",
  "benchmark/candidates/prooflens-cf384-m4"]) {
  requireCondition(!existsSync(pathname), `M4 protocol freeze must precede ignored materialization: ${pathname}`);
}
requireCondition(digest(readFileSync("weights/prooflens-cf384.onnx")) === MODEL_SHA256,
  "M4 protocol freeze must retain the reviewed M2 model bytes");
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");
requireCondition(git(["ls-files", "benchmark/data/m4-head", "benchmark/data/m4-source",
  "benchmark/candidates/prooflens-cf384-m4", "benchmark/data/h3-met-holdout-v1"]) === "",
"M4 source/candidate or H3 pixels must remain outside Git");

console.log(JSON.stringify({
  stage: "m4-protocol",
  head,
  parent: M4_BASE_COMMIT,
  paths: M4_PROTOCOL_EXPECTED.size,
  modelSha256: MODEL_SHA256,
  materializationOutputsPresent: false,
  h3AcceptedAsInput: false,
  policy: "pass",
}));
