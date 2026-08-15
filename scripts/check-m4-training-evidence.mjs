import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";

import { validateM4LockedPacket } from "./check-m4-publication-lock.mjs";
import { digest, parseCanonicalJson, requireCondition } from "./m4-training-contract.mjs";
import {
  M4_BASE_COMMIT,
  M4_FAILED_PROTOCOL_COMMIT,
  M4_LOCK_EXPECTED,
  M4_PUBLICATION_EXPECTED,
  M4_PUBLICATION_LOCK_PATH,
  M4_SOURCE_EXPECTED,
  matchesM4ProtocolRecoveryLineage,
  matchesExpectedRows,
} from "./m4-stage-policy.mjs";


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

function bytes(pathname) {
  return readFileSync(pathname);
}

function equal(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M4 final verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
const finalParents = parents(head);
requireCondition(finalParents.length === 1, "The M4 publication commit must have one parent");
const lockCommit = finalParents[0];
const lockParents = parents(lockCommit);
requireCondition(lockParents.length === 1, "The M4 lock commit must have one parent");
const source = lockParents[0];
const sourceParents = parents(source);
requireCondition(sourceParents.length === 1, "The M4 source commit must have one parent");
const protocol = sourceParents[0];
const protocolParents = parents(protocol);
requireCondition(matchesM4ProtocolRecoveryLineage({
  protocolParents,
  protocolRows: commitRows(protocol),
  failedProtocolParents: parents(M4_FAILED_PROTOCOL_COMMIT),
  failedProtocolRows: commitRows(M4_FAILED_PROTOCOL_COMMIT),
  failedProtocolTree: git(["rev-parse", `${M4_FAILED_PROTOCOL_COMMIT}^{tree}`]),
  baseTree: git(["rev-parse", `${M4_BASE_COMMIT}^{tree}`]),
}), "The M4 final lineage changed before the recovered protocol commit");
requireCondition(matchesExpectedRows(commitRows(source), M4_SOURCE_EXPECTED) &&
  matchesExpectedRows(commitRows(lockCommit), M4_LOCK_EXPECTED) &&
  matchesExpectedRows(commitRows(head), M4_PUBLICATION_EXPECTED),
"The M4 final lineage changed outside an exact stage packet");
requireCondition(!existsSync("benchmark/evidence/m4/failed-selector-diagnostic-1.json") &&
  !existsSync("benchmark/evidence/m4/failed-training-attempt-1.json"),
"M4 final evidence cannot coexist with terminal failure evidence");

const locked = validateM4LockedPacket(bytes(M4_PUBLICATION_LOCK_PATH));
const { lock, candidateBytes, summary, calibration, grid } = locked;
requireCondition(lock.sourceCommit === source && lock.sourceTree === git(["rev-parse", `${source}^{tree}`]),
  "The M4 final lock does not bind the exact source commit");
const publicEvidence = {
  "training-summary.json": bytes("benchmark/evidence/m4/training-summary.json"),
  "calibration.json": bytes("benchmark/evidence/m4/calibration.json"),
  "candidate-grid.json": bytes("benchmark/evidence/m4/candidate-grid.json"),
  "model-comparison.json": bytes("benchmark/evidence/m4/model-comparison.json"),
};
for (const [name, value] of Object.entries(publicEvidence)) {
  requireCondition(value.equals(Buffer.from(lock.candidateEvidenceJson[name], "utf8")),
    `M4 published candidate evidence changed: ${name}`);
}
requireCondition(bytes("weights/prooflens-cf384.onnx").equals(candidateBytes),
  "M4 shipped model does not equal the locked reconstructed candidate");
for (const pathname of ["README.md", "MODEL_CARD.md", "BENCHMARK.md"]) {
  requireCondition(digest(bytes(pathname)) === lock.publicDocumentHashes[pathname],
    `M4 published document changed: ${pathname}`);
}
requireCondition(bytes("tests/fixtures/model-states/fixture-manifest.json").equals(
  Buffer.from(lock.candidateEvidenceJson["fixture-manifest.json"], "utf8")),
"M4 published fixture manifest changed");

const sourceModelLock = JSON.parse(execFileSync("git", ["show", `${source}:model-lock.json`], { encoding: "utf8" }));
const modelLockBytes = bytes("model-lock.json");
const modelLock = parseCanonicalJson(modelLockBytes, "M4 model lock");
const expectedModelLock = {
  schemaVersion: 2,
  artifact: sourceModelLock.artifact,
  bytes: lock.candidateModelBytes,
  sha256: lock.candidateHashes.model,
  format: sourceModelLock.format,
  input: sourceModelLock.input,
  output: sourceModelLock.output,
  upstream: sourceModelLock.upstream,
  trainingRecipe: `prooflens-m4-residual-adapter-v1:${lock.recipeSha256}:${lock.selectionSummarySha256}`,
  trainingEvidence: {
    recipe: "benchmark/m4/recipe.json",
    recipeSha256: lock.recipeSha256,
    sourceLocks: "benchmark/m4/source-locks.json",
    sourceLocksSha256: lock.sourceLocksSha256,
    selectionSummary: "benchmark/evidence/m4/selection-summary.json",
    selectionSummarySha256: lock.selectionSummarySha256,
    trainManifestSha256: summary.trainManifestSha256,
    selectorManifestSha256: summary.selectorManifestSha256,
    m3RegressionManifestSha256: summary.m3RegressionManifestSha256,
    m2RegressionManifestSha256: summary.m2RegressionManifestSha256,
    trainingSummarySha256: lock.candidateHashes.summary,
    calibrationSha256: lock.candidateHashes.calibration,
    candidateGridSha256: lock.candidateHashes.grid,
    selectionLockSha256: lock.candidateHashes.selectionLock,
    candidateTensorSealSha256: lock.candidateHashes.tensorSeal,
    freshFeatureRunId: summary.freshFeatureRunId,
    publicationLockSha256: digest(bytes(M4_PUBLICATION_LOCK_PATH)),
    upstreamModelSha256: lock.upstreamModelSha256,
    architecture: "frozen M2 backbone/classifier plus 384-to-64-to-384 residual feature adapter",
  },
  calibration: {
    slope: calibration.slope,
    intercept: calibration.intercept,
    displayThreshold: calibration.displayThreshold,
    validationThresholdLogit: calibration.rawThreshold,
  },
};
requireCondition(equal(modelLock, expectedModelLock), "M4 published model lock changed");

const expectedWeightsReadme = `# Packaged detector\n\n` +
  "`prooflens-cf384.onnx` is the exact FP32 artifact described by the repository root [model-lock.json](../model-lock.json).\n\n" +
  "```text\n" +
  `bytes    ${lock.candidateModelBytes.toLocaleString("en-US")}\n` +
  `sha256   ${lock.candidateHashes.model}\n` +
  "input    pixel_values [N,3,384,384] float32\n" +
  "output   logits [N,1] float32\n" +
  "```\n\n" +
  "M4 preserves the M2 backbone and classifier tensors byte-for-byte and inserts one 384→64→384 residual feature adapter. The extension loads this one packaged model locally; it does not download a model or call a detector service after installation.\n";
requireCondition(bytes("weights/README.md").equals(Buffer.from(expectedWeightsReadme, "utf8")),
  "M4 packaged-model README changed");

const receiptBytes = bytes("benchmark/evidence/m4/finalization-receipt.json");
const receipt = parseCanonicalJson(receiptBytes, "M4 finalization receipt");
const publicationRows = [...M4_PUBLICATION_EXPECTED].map(([path, status]) => ({ path, status }));
const expectedReceipt = {
  schemaVersion: 1,
  profile: "m4",
  sourceCommit: source,
  sourceTree: lock.sourceTree,
  lockCommit,
  publicationLockSha256: digest(bytes(M4_PUBLICATION_LOCK_PATH)),
  candidateDirectory: "benchmark/candidates/prooflens-cf384-m4",
  upstreamSha256: lock.upstreamModelSha256,
  shippedModel: { path: "weights/prooflens-cf384.onnx", sha256: lock.candidateHashes.model, bytes: lock.candidateModelBytes },
  sourceEvidenceSha256: {
    "training-summary.json": lock.candidateHashes.summary,
    "calibration.json": lock.candidateHashes.calibration,
    "candidate-grid.json": lock.candidateHashes.grid,
  },
  publishedEvidenceSha256: {
    "training-summary.json": digest(publicEvidence["training-summary.json"]),
    "calibration.json": digest(publicEvidence["calibration.json"]),
    "candidate-grid.json": digest(publicEvidence["candidate-grid.json"]),
    "model-comparison.json": digest(publicEvidence["model-comparison.json"]),
  },
  publishedRepositorySha256: {
    "weights/prooflens-cf384.onnx": lock.candidateHashes.model,
    "model-lock.json": digest(modelLockBytes),
    "weights/README.md": digest(bytes("weights/README.md")),
    "README.md": digest(bytes("README.md")),
    "MODEL_CARD.md": digest(bytes("MODEL_CARD.md")),
    "BENCHMARK.md": digest(bytes("BENCHMARK.md")),
    "tests/fixtures/model-states/fixture-manifest.json": digest(bytes("tests/fixtures/model-states/fixture-manifest.json")),
  },
  selectorGatesPassed: true,
  m3RegressionGatesPassed: true,
  m2RegressionGatesPassed: true,
  selectionInfluencedByRegression: false,
  h3HoldoutScored: false,
  h3PixelsRead: false,
  transactionalRows: publicationRows,
  requiredFinalCommitRows: publicationRows,
};
requireCondition(equal(receipt, expectedReceipt), "M4 finalization receipt changed");
requireCondition(grid.validCandidateCount > 0 && summary.status === "accepted-development-candidate",
  "M4 final packet is not an accepted development candidate");

execFileSync("node", ["scripts/check-m4-selection-evidence.mjs"], { stdio: "inherit" });
console.log(JSON.stringify({
  stage: "m4-final", head, lockCommit, source,
  modelSha256: lock.candidateHashes.model,
  modelBytes: lock.candidateModelBytes,
  candidateId: lock.selectionLock.selectedCandidateId,
  h3HoldoutScored: false,
  policy: "pass",
}));
