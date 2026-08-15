import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";

import {
  M4_BASE_COMMIT,
  M4_DATE_CI_RECOVERY_COMMIT,
  M4_DATE_RECOVERY_COMMIT,
  M4_FAILED_PROTOCOL_COMMIT,
  M4_PROTOCOL_RECOVERY_COMMIT,
  M4_SOURCE_EXPECTED,
  matchesM4ProtocolRecoveryLineage,
  matchesExpectedRows,
} from "./m4-stage-policy.mjs";


const MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47";
const FORBIDDEN = [
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

function parents(commit) {
  return git(["show", "-s", "--format=%P", commit]).split(" ").filter(Boolean);
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M4 source verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
const sourceParents = parents(head);
requireCondition(sourceParents.length === 1, "The M4 source commit must have one parent");
const protocol = sourceParents[0];
const protocolParents = parents(protocol);
requireCondition(matchesM4ProtocolRecoveryLineage({
  protocolParents,
  protocolRows: commitRows(protocol),
  dateCiRecoveryParents: parents(M4_DATE_CI_RECOVERY_COMMIT),
  dateCiRecoveryRows: commitRows(M4_DATE_CI_RECOVERY_COMMIT),
  dateCiRecoveryTree: git(["rev-parse", `${M4_DATE_CI_RECOVERY_COMMIT}^{tree}`]),
  dateRecoveryParents: parents(M4_DATE_RECOVERY_COMMIT),
  dateRecoveryRows: commitRows(M4_DATE_RECOVERY_COMMIT),
  dateRecoveryTree: git(["rev-parse", `${M4_DATE_RECOVERY_COMMIT}^{tree}`]),
  recoveryProtocolParents: parents(M4_PROTOCOL_RECOVERY_COMMIT),
  recoveryProtocolRows: commitRows(M4_PROTOCOL_RECOVERY_COMMIT),
  recoveryProtocolTree: git(["rev-parse", `${M4_PROTOCOL_RECOVERY_COMMIT}^{tree}`]),
  failedProtocolParents: parents(M4_FAILED_PROTOCOL_COMMIT),
  failedProtocolRows: commitRows(M4_FAILED_PROTOCOL_COMMIT),
  failedProtocolTree: git(["rev-parse", `${M4_FAILED_PROTOCOL_COMMIT}^{tree}`]),
  baseTree: git(["rev-parse", `${M4_BASE_COMMIT}^{tree}`]),
}), "The M4 source packet is not the recovered protocol's direct child");
requireCondition(matchesExpectedRows(commitRows(head), M4_SOURCE_EXPECTED),
  "The M4 source commit changed outside its exact score-blind packet");
for (const pathname of FORBIDDEN) {
  requireCondition(!existsSync(pathname), `M4 source stage contains terminal or training output: ${pathname}`);
}
requireCondition(digest(readFileSync("weights/prooflens-cf384.onnx")) === MODEL_SHA256,
  "M4 source stage changed the shipped M2 model");
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");

execFileSync("node", ["scripts/check-m4-selection-evidence.mjs"], { stdio: "inherit" });
console.log(JSON.stringify({
  stage: "m4-source",
  head,
  protocol,
  paths: M4_SOURCE_EXPECTED.size,
  modelSha256: MODEL_SHA256,
  h3AcceptedAsInput: false,
  policy: "pass",
}));
