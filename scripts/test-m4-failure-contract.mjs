import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";

import { validateM4FailurePacket } from "./m4-failure-contract.mjs";
import { digest } from "./m4-training-contract.mjs";
import { buildTrainingFixture, canonicalBytes, clone } from "./test-m4-training-contract.mjs";


const SOURCE_COMMIT = "1".repeat(40);
const SOURCE_TREE = "2".repeat(40);
const H = (character) => character.repeat(64);

function failureMetrics(metadata, realSources, syntheticSources, passed) {
  const metrics = {};
  for (const variant of ["original", "screenshot", "social-q75", "social-heavy"]) {
    metrics[variant] = {
      balancedAccuracy: passed ? 1 : 0.5,
      realRecall: passed ? 1 : 0,
      syntheticRecall: 1,
      syntheticRecallBySource: Object.fromEntries(syntheticSources.map((source) => [source, 1])),
      realRecallBySource: Object.fromEntries(realSources.map((source) => [source, passed ? 1 : 0])),
    };
  }
  const values = metadata.labels.map((label) => passed ? (label === 0 ? -2 : 2) : 0);
  const bytes = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  const hash = arrayDigest(bytes, values.length);
  return { metrics, bytes, hash };
}

function arrayDigest(bytes, count) {
  return createHash("sha256").update("<f4").update(JSON.stringify([count])).update(bytes).digest("hex");
}

function regressionRow(name, metadata, realSources, syntheticSources, passed) {
  const values = failureMetrics(metadata, realSources, syntheticSources, passed);
  return {
    name,
    metrics: values.metrics,
    passed,
    logitsSha256: values.hash,
    logitsFloat32Base64: values.bytes.toString("base64"),
    logitCount: metadata.labels.length,
  };
}

function buildPacket(status) {
  const selectorFeasible = status !== "failed-selector";
  const fixture = buildTrainingFixture({ selectorFeasible });
  const completed = status === "failed-selector" ? 0 : status === "failed-m3-selector-regression" ? 1 : 2;
  fixture.marker.state = "extracting";
  fixture.hashes.freshMarker = digest(canonicalBytes(fixture.marker));
  fixture.summary.status = status;
  fixture.summary.freshFeatureMarkerSha256 = fixture.hashes.freshMarker;
  fixture.summary.featureShardEvidence = fixture.summary.featureShardEvidence.slice(0, 58 + completed);
  for (const key of ["freshFeatureRunComplete", "modelSha256", "modelBytes",
    "zeroAdapterImageParityMaximumAbsoluteError", "exportedCandidateImageParityMaximumAbsoluteError",
    "calibrationSha256"]) delete fixture.summary[key];
  if (completed === 0) {
    for (const key of ["selectionLockSha256", "selectionLock", "selectedCandidate", "regressions"]) {
      delete fixture.summary[key];
    }
  } else {
    const m3 = regressionRow("m3-selector-regression", fixture.regressionMetadata.m3,
      ["met-open-access"], ["flux-1-dev-development"], completed === 2);
    const regressions = [m3];
    if (completed === 2) regressions.push(regressionRow("m2-development-regression", fixture.regressionMetadata.m2,
      ["open-images", "stockimages-cc0"], ["GLM-Image", "HunyuanImage-3.0"], false));
    fixture.summary.regressions = regressions;
  }
  const artifacts = {
    "validation-summary.json": digest(canonicalBytes(fixture.summary)),
    "candidate-grid.json": digest(canonicalBytes(fixture.grid)),
    "candidate-tensor-seal.json": digest(canonicalBytes(fixture.tensorSeal)),
    "fresh-feature-run.json": digest(canonicalBytes(fixture.marker)),
    ...(completed === 0 ? {} : { "selection-lock.json": digest(canonicalBytes(fixture.selectionLock)) }),
  };
  const diagnostic = {
    schemaVersion: 1,
    profile: "m4",
    sourceCommit: SOURCE_COMMIT,
    sourceTree: SOURCE_TREE,
    baseCommit: fixture.recipe.baseCommit,
    recipeSha256: fixture.hashes.recipe ?? "344ced4ee8e68325bd0217391e4e5745d554b8140586b51d7aa98ef6bb441b34",
    sourceLocksSha256: fixture.hashes.sourceLocks ?? "bf44ceba6f32d322de04f9fae994c0fed7fdcd00e2bcfff9de39c6d852a01394",
    trainerSha256: fixture.hashes.trainer,
    upstreamModelSha256: fixture.summary.upstreamModelSha256,
    commandArguments: fixture.summary.commandArguments,
    candidateDirectory: "benchmark/candidates/prooflens-cf384-m4",
    terminalOutcome: status,
    attemptStatus: status,
    candidateArtifactSha256: artifacts,
    selectionLockSha256: completed === 0 ? null : digest(canonicalBytes(fixture.selectionLock)),
    selectionLock: completed === 0 ? null : fixture.selectionLock,
    candidateGrid: fixture.grid,
    candidateTensorSeal: fixture.tensorSeal,
    freshFeatureMarkerJson: canonicalBytes(fixture.marker).toString("utf8"),
    trainingSummary: fixture.summary,
    selectedCandidate: completed === 0 ? null : fixture.winner,
    completedRegressions: fixture.summary.regressions ?? [],
    notRunRegressions: fixture.recipe.selectionPolicy.regressionOrder.slice(completed),
    freshFeatureRunId: fixture.summary.freshFeatureRunId,
    featureShardEvidence: fixture.summary.featureShardEvidence,
    publishedModel: false,
    publicationLockCreated: false,
    successfulM4PublicationEvidencePresent: false,
    h3HoldoutScored: false,
    h3PixelsRead: false,
    terminality: {
      reselectionPermitted: false,
      thresholdChangePermitted: false,
      gateChangePermitted: false,
      retrySameSelectorPermitted: false,
    },
  };
  const inventory = [];
  for (const [name, sha256] of Object.entries(artifacts)) {
    inventory.push({
      path: `benchmark/candidates/prooflens-cf384-m4/${name}`,
      bytes: 1,
      sha256,
    });
  }
  for (const candidate of fixture.tensorSeal.candidates) inventory.push({
    path: `benchmark/candidates/prooflens-cf384-m4/candidates/${candidate.candidateId}.npz`,
    bytes: 1,
    sha256: H("a"),
  });
  for (const shard of fixture.summary.featureShardEvidence) inventory.push({
    path: shard.cache,
    bytes: 1,
    sha256: shard.cacheSha256,
  });
  inventory.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
  const diagnosticBytes = canonicalBytes(diagnostic);
  const receipt = {
    schemaVersion: 1,
    profile: "m4",
    sourceCommit: SOURCE_COMMIT,
    sourceTree: SOURCE_TREE,
    diagnosticPath: "benchmark/evidence/m4/failed-selector-diagnostic-1.json",
    diagnosticSha256: digest(diagnosticBytes),
    terminalOutcome: status,
    candidateCacheSnapshot: {
      fileCount: inventory.length,
      bytes: inventory.reduce((total, row) => total + row.bytes, 0),
      inventory,
    },
    successfulM4PublicationEvidencePresent: false,
    h3HoldoutScored: false,
    h3PixelsRead: false,
    requiredFailureCommitRows: [
      { path: "benchmark/evidence/m4/failed-selector-diagnostic-1.json", status: "A" },
      { path: "benchmark/evidence/m4/failed-training-attempt-1.json", status: "A" },
    ],
  };
  const hashes = {
    ...fixture.hashes,
    sourceCommit: SOURCE_COMMIT,
    sourceTree: SOURCE_TREE,
    recipe: diagnostic.recipeSha256,
    sourceLocks: diagnostic.sourceLocksSha256,
  };
  return {
    ...fixture,
    diagnostic,
    receipt,
    diagnosticBytes,
    receiptBytes: canonicalBytes(receipt),
    hashes,
  };
}

function validate(packet) {
  return validateM4FailurePacket({
    diagnosticBytes: packet.diagnosticBytes,
    receiptBytes: packet.receiptBytes,
    recipe: packet.recipe,
    selectionSummary: packet.selectionSummary,
    selectorMetadata: packet.selectorMetadata,
    regressionMetadata: packet.regressionMetadata,
    hashes: packet.hashes,
  });
}

function mutate(packet, callback) {
  const changed = clone(packet);
  callback(changed);
  changed.diagnosticBytes = canonicalBytes(changed.diagnostic);
  changed.receipt.diagnosticSha256 = digest(changed.diagnosticBytes);
  changed.receiptBytes = canonicalBytes(changed.receipt);
  return changed;
}

function main() {
  for (const status of ["failed-selector", "failed-m3-selector-regression", "failed-m2-development-regression"]) {
    const packet = buildPacket(status);
    assert.equal(validate(packet).diagnostic.attemptStatus, status);
  }

  const gridLogit = mutate(buildPacket("failed-selector"), (packet) => {
    packet.diagnostic.candidateGrid.candidates[0].selectorLogitsFloat32Base64 = "AAAA";
  });
  assert.throws(() => validate(gridLogit), /candidate artifacts changed|bytes or digest changed/u);

  const tensor = mutate(buildPacket("failed-selector"), (packet) => {
    const name = Object.keys(packet.diagnostic.candidateTensorSeal.candidates[0].tensorFloat32Base64)[0];
    packet.diagnostic.candidateTensorSeal.candidates[0].tensorFloat32Base64[name] = "AAAA";
  });
  assert.throws(() => validate(tensor), /candidate artifacts changed|tensor bytes changed/u);

  const marker = mutate(buildPacket("failed-selector"), (packet) => {
    const parsed = JSON.parse(packet.diagnostic.freshFeatureMarkerJson);
    parsed.state = "complete";
    packet.diagnostic.freshFeatureMarkerJson = canonicalBytes(parsed).toString("utf8");
  });
  assert.throws(() => validate(marker), /marker context changed|artifact/u);

  const regression = mutate(buildPacket("failed-m3-selector-regression"), (packet) => {
    packet.diagnostic.trainingSummary.regressions[0].passed = true;
    packet.diagnostic.completedRegressions[0].passed = true;
  });
  assert.throws(() => validate(regression), /evidence changed|result changed/u);

  const terminality = mutate(buildPacket("failed-selector"), (packet) => {
    packet.diagnostic.terminality.reselectionPermitted = true;
  });
  assert.throws(() => validate(terminality), /terminal boundary changed/u);

  const inventory = mutate(buildPacket("failed-selector"), (packet) => {
    packet.receipt.candidateCacheSnapshot.bytes += 1;
  });
  assert.throws(() => validate(inventory), /byte total changed/u);

  const missingFeature = mutate(buildPacket("failed-selector"), (packet) => {
    const cache = packet.diagnostic.trainingSummary.featureShardEvidence[0].cache;
    packet.receipt.candidateCacheSnapshot.inventory =
      packet.receipt.candidateCacheSnapshot.inventory.filter((row) => row.path !== cache);
    packet.receipt.candidateCacheSnapshot.fileCount -= 1;
    packet.receipt.candidateCacheSnapshot.bytes -= 1;
  });
  assert.throws(() => validate(missingFeature), /does not bind feature shard/u);

  const changedFeature = mutate(buildPacket("failed-selector"), (packet) => {
    const cache = packet.diagnostic.trainingSummary.featureShardEvidence[0].cache;
    packet.receipt.candidateCacheSnapshot.inventory.find((row) => row.path === cache).sha256 = H("0");
  });
  assert.throws(() => validate(changedFeature), /does not bind feature shard/u);

  const extraFeature = mutate(buildPacket("failed-selector"), (packet) => {
    packet.receipt.candidateCacheSnapshot.inventory.push({
      path: "benchmark/candidates/prooflens-cf384-m4/features/stale-99999.npz",
      bytes: 1,
      sha256: H("a"),
    });
    packet.receipt.candidateCacheSnapshot.inventory.sort((left, right) =>
      left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
    packet.receipt.candidateCacheSnapshot.fileCount += 1;
    packet.receipt.candidateCacheSnapshot.bytes += 1;
  });
  assert.throws(() => validate(extraFeature), /unbound feature shard/u);

  const duplicate = buildPacket("failed-selector");
  const duplicateBytes = Buffer.from(duplicate.diagnosticBytes.toString("utf8").replace(/^\{/u,
    '{\n  "schemaVersion": 9,'));
  assert.throws(() => validateM4FailurePacket({
    diagnosticBytes: duplicateBytes,
    receiptBytes: duplicate.receiptBytes,
    recipe: duplicate.recipe,
    selectionSummary: duplicate.selectionSummary,
    selectorMetadata: duplicate.selectorMetadata,
    regressionMetadata: duplicate.regressionMetadata,
    hashes: duplicate.hashes,
  }), /not canonical JSON/u);

  console.log(JSON.stringify({ cases: 13, statuses: 3, policy: "pass" }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
