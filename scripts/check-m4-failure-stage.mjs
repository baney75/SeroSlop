import { execFileSync } from "node:child_process";
import { existsSync, lstatSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { TextDecoder } from "node:util";

import { validateM4FailurePacket } from "./m4-failure-contract.mjs";
import { digest, jsonEqual, parseManifestMetadata, requireCondition } from "./m4-training-contract.mjs";
import {
  M4_BASE_COMMIT,
  M4_DATE_CI_RECOVERY_COMMIT,
  M4_DATE_RECOVERY_COMMIT,
  M4_FAILED_PROTOCOL_COMMIT,
  M4_PROTOCOL_RECOVERY_COMMIT,
  M4_FAILURE_EXPECTED,
  M4_FAILURE_PATH,
  M4_PUBLICATION_LOCK_PATH,
  M4_SOURCE_EXPECTED,
  matchesM4ProtocolRecoveryLineage,
  matchesExpectedRows,
} from "./m4-stage-policy.mjs";


const MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47";
const DIAGNOSTIC_PATH = "benchmark/evidence/m4/failed-selector-diagnostic-1.json";
const SUCCESS_OUTPUTS = [
  M4_PUBLICATION_LOCK_PATH,
  "benchmark/evidence/m4/calibration.json",
  "benchmark/evidence/m4/candidate-grid.json",
  "benchmark/evidence/m4/finalization-receipt.json",
  "benchmark/evidence/m4/model-comparison.json",
  "benchmark/evidence/m4/training-summary.json",
];

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }).trim();
}

function sourceBytes(commit, pathname) {
  return execFileSync("git", ["show", `${commit}:${pathname}`], { encoding: null, maxBuffer: 256 * 1024 * 1024 });
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

function inventory(root) {
  const rows = [];
  const visit = (directory) => {
    for (const name of readdirSync(directory).sort()) {
      const pathname = path.join(directory, name);
      const details = lstatSync(pathname);
      requireCondition(!details.isSymbolicLink(), `M4 candidate cache contains a symlink: ${pathname}`);
      if (details.isDirectory()) visit(pathname);
      else if (details.isFile()) {
        const bytes = readFileSync(pathname);
        rows.push({ path: pathname.split(path.sep).join("/"), bytes: bytes.length, sha256: digest(bytes) });
      }
    }
  };
  visit(root);
  return rows.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M4 failure verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
const failureParents = parents(head);
requireCondition(failureParents.length === 1, "The M4 failure commit must have one parent");
const source = failureParents[0];
const sourceParents = parents(source);
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
}), "The M4 failure lineage changed before the recovered protocol commit");
requireCondition(matchesExpectedRows(commitRows(source), M4_SOURCE_EXPECTED) &&
  matchesExpectedRows(commitRows(head), M4_FAILURE_EXPECTED),
"The M4 failure lineage changed outside an exact stage packet");
for (const pathname of SUCCESS_OUTPUTS) {
  requireCondition(!existsSync(pathname), `M4 failure packet contains successful publication evidence: ${pathname}`);
}
requireCondition(digest(readFileSync("weights/prooflens-cf384.onnx")) === MODEL_SHA256,
  "M4 terminal failure changed the shipped M2 model");
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");
requireCondition(git(["ls-files", "benchmark/data/m4-head", "benchmark/data/m4-source",
  "benchmark/candidates/prooflens-cf384-m4", "benchmark/data/h3-met-holdout-v1"]) === "",
"M4 source, candidate, or H3 pixels entered Git");

const recipeBytes = sourceBytes(source, "benchmark/m4/recipe.json");
const recipe = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(recipeBytes));
const selectionSummaryBytes = sourceBytes(source, "benchmark/evidence/m4/selection-summary.json");
const selectionSummary = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(selectionSummaryBytes));
const selectorBytes = sourceBytes(source, "benchmark/evidence/m4/validation-manifest.jsonl");
const m3Bytes = sourceBytes(source, "benchmark/evidence/m3/validation-manifest.jsonl");
const m2Bytes = sourceBytes(source, "benchmark/evidence/m2/validation-manifest.jsonl");
const selectorMetadata = parseManifestMetadata(selectorBytes, {
  label: "M4 selector", items: 600,
  sources: {
    "british-library-plates": { label: 0, count: 300 },
    "rapidata-dalle-3": { label: 1, count: 75 },
    "rapidata-flux": { label: 1, count: 75 },
    "rapidata-midjourney": { label: 1, count: 75 },
    "rapidata-stable-diffusion": { label: 1, count: 75 },
  },
});
const regressionMetadata = {
  m3: parseManifestMetadata(m3Bytes, {
    label: "M3 regression", items: 600,
    sources: { "met-open-access": { label: 0, count: 300 }, "flux-1-dev-development": { label: 1, count: 300 } },
  }),
  m2: parseManifestMetadata(m2Bytes, {
    label: "M2 regression", items: 900,
    sources: {
      "open-images": { label: 0, count: 300 }, "stockimages-cc0": { label: 0, count: 300 },
      "GLM-Image": { label: 1, count: 150 }, "HunyuanImage-3.0": { label: 1, count: 150 },
    },
  }),
};
const packet = validateM4FailurePacket({
  diagnosticBytes: readFileSync(DIAGNOSTIC_PATH),
  receiptBytes: readFileSync(M4_FAILURE_PATH),
  recipe,
  selectionSummary,
  selectorMetadata,
  regressionMetadata,
  hashes: {
    sourceCommit: source,
    sourceTree: git(["rev-parse", `${source}^{tree}`]),
    trainer: digest(sourceBytes(source, "benchmark/m4/train_adapter.py")),
    recipe: digest(recipeBytes),
    sourceLocks: digest(sourceBytes(source, "benchmark/m4/source-locks.json")),
    selectionSummary: digest(selectionSummaryBytes),
    selectorManifest: digest(selectorBytes),
    m3RegressionManifest: digest(m3Bytes),
    m2RegressionManifest: digest(m2Bytes),
  },
});
const cacheRoot = "benchmark/candidates/prooflens-cf384-m4";
if (existsSync(cacheRoot)) {
  requireCondition(jsonEqual(inventory(cacheRoot), packet.receipt.candidateCacheSnapshot.inventory),
    "M4 retained candidate cache no longer matches the terminal receipt");
}
execFileSync("node", ["scripts/check-m4-selection-evidence.mjs"], { stdio: "inherit" });
console.log(JSON.stringify({
  stage: "m4-failed",
  head,
  source,
  terminalOutcome: packet.diagnostic.terminalOutcome,
  candidateCachePresent: existsSync(cacheRoot),
  h3HoldoutScored: false,
  policy: "pass",
}));
