import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import {
  M4_BASE_COMMIT,
  M4_FAILED_PROTOCOL_COMMIT,
  M4_FAILED_PROTOCOL_TREE,
  M4_DATE_CI_RECOVERY_COMMIT,
  M4_DATE_CI_RECOVERY_TREE,
  M4_DATE_RECOVERY_COMMIT,
  M4_DATE_RECOVERY_TREE,
  M4_DATE_CI_RECOVERY_EXPECTED,
  M4_PROTOCOL_RECOVERY_COMMIT,
  M4_PROTOCOL_RECOVERY_TREE,
  M4_DATE_RECOVERY_EXPECTED,
  M4_PROTOCOL_EXPECTED,
  M4_PROTOCOL_RECOVERY_EXPECTED,
  matchesM4ProtocolRecoveryLineage,
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
const failedProtocolParents = git(["show", "-s", "--format=%P", M4_FAILED_PROTOCOL_COMMIT])
  .split(" ").filter(Boolean);
const recoveryProtocolParents = git(["show", "-s", "--format=%P", M4_PROTOCOL_RECOVERY_COMMIT])
  .split(" ").filter(Boolean);
const dateRecoveryParents = git(["show", "-s", "--format=%P", M4_DATE_RECOVERY_COMMIT])
  .split(" ").filter(Boolean);
const dateCiRecoveryParents = git(["show", "-s", "--format=%P", M4_DATE_CI_RECOVERY_COMMIT])
  .split(" ").filter(Boolean);
requireCondition(matchesM4ProtocolRecoveryLineage({
  protocolParents: parents,
  protocolRows: commitRows(head),
  dateCiRecoveryParents,
  dateCiRecoveryRows: commitRows(M4_DATE_CI_RECOVERY_COMMIT),
  dateCiRecoveryTree: git(["rev-parse", `${M4_DATE_CI_RECOVERY_COMMIT}^{tree}`]),
  dateRecoveryParents,
  dateRecoveryRows: commitRows(M4_DATE_RECOVERY_COMMIT),
  dateRecoveryTree: git(["rev-parse", `${M4_DATE_RECOVERY_COMMIT}^{tree}`]),
  recoveryProtocolParents,
  recoveryProtocolRows: commitRows(M4_PROTOCOL_RECOVERY_COMMIT),
  recoveryProtocolTree: git(["rev-parse", `${M4_PROTOCOL_RECOVERY_COMMIT}^{tree}`]),
  failedProtocolParents,
  failedProtocolRows: commitRows(M4_FAILED_PROTOCOL_COMMIT),
  failedProtocolTree: git(["rev-parse", `${M4_FAILED_PROTOCOL_COMMIT}^{tree}`]),
  baseTree: git(["rev-parse", `${M4_BASE_COMMIT}^{tree}`]),
}), "The M4 protocol recovery lineage or exact packet changed");
for (const pathname of FORBIDDEN_OUTPUTS) {
  requireCondition(!existsSync(pathname), `M4 protocol freeze must precede materialization/training: ${pathname}`);
}
for (const pathname of ["benchmark/data/m4-head", "benchmark/candidates/prooflens-cf384-m4"]) {
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
  parent: M4_DATE_CI_RECOVERY_COMMIT,
  failedProtocolTree: M4_FAILED_PROTOCOL_TREE,
  originalProtocolPaths: M4_PROTOCOL_EXPECTED.size,
  recoveryPaths: M4_PROTOCOL_RECOVERY_EXPECTED.size,
  dateEligibilityRecoveryPaths: M4_DATE_RECOVERY_EXPECTED.size,
  ciRecoveryPaths: M4_DATE_CI_RECOVERY_EXPECTED.size,
  ciRecoveryCommit: M4_DATE_CI_RECOVERY_COMMIT,
  ciRecoveryTree: M4_DATE_CI_RECOVERY_TREE,
  dateRecoveryCommit: M4_DATE_RECOVERY_COMMIT,
  dateRecoveryTree: M4_DATE_RECOVERY_TREE,
  recoveryCommit: M4_PROTOCOL_RECOVERY_COMMIT,
  recoveryTree: M4_PROTOCOL_RECOVERY_TREE,
  modelSha256: MODEL_SHA256,
  materializationOutputsPresent: false,
  sourceArchiveMayExist: true,
  recoveryReason: "rapidata-overlap-clean-group-capacity-before-selection",
  h3AcceptedAsInput: false,
  policy: "pass",
}));
