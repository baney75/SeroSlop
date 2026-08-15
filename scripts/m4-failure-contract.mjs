import {
  M4,
  M4_COMMON_SUMMARY_KEYS,
  M4_EXPECTED_ARGUMENTS,
  digest,
  exactKeys,
  jsonEqual,
  parseCanonicalJson,
  recomputeM4Grid,
  requireCondition,
  validateM4FreshRunEvidence,
  validateM4Regression,
  validateM4TensorSeal,
} from "./m4-training-contract.mjs";


const HEX64 = /^[a-f0-9]{64}$/u;
const FAILURE_ROWS = [
  { path: "benchmark/evidence/m4/failed-selector-diagnostic-1.json", status: "A" },
  { path: "benchmark/evidence/m4/failed-training-attempt-1.json", status: "A" },
];
const STATUS = Object.freeze({
  "failed-selector": 0,
  "failed-m3-selector-regression": 1,
  "failed-m2-development-regression": 2,
});

function canonicalBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function validateSelectionSummary(summary, recipe) {
  exactKeys(summary, ["schemaVersion", "recipeSha256", "sourceLocksSha256", "scoreBlind",
    "modelOutputsRead", "h3PixelsRead", "h3ManifestSha256", "selectionOrder", "training",
    "freshSelector", "partitionGroups", "overlap", "publicArtifacts"], "M4 failure source selection summary");
  requireCondition(summary.schemaVersion === 1 && summary.recipeSha256 === M4.recipeSha256 &&
    summary.sourceLocksSha256 === M4.sourceLocksSha256 && summary.scoreBlind === true &&
    summary.modelOutputsRead === false && summary.h3PixelsRead === false &&
    summary.h3ManifestSha256 === recipe.h3Exclusion.sha256 &&
    jsonEqual(summary.selectionOrder, ["british-selector", "rapidata-selector", "british-training",
      "rapidata-training"]), "M4 failure source-selection boundary changed");
}

function validateSummaryCommon({ summary, recipe, selectionSummary, hashes, marker, completed }) {
  const status = summary.status;
  const extra = completed === 0 ? [] : ["selectionLockSha256", "selectionLock", "selectedCandidate", "regressions"];
  exactKeys(summary, [...M4_COMMON_SUMMARY_KEYS, ...extra], "M4 failed training summary");
  requireCondition(summary.schemaVersion === 1 && summary.pipelineVersion === M4.pipelineVersion &&
    summary.seed === M4.seed && jsonEqual(summary.commandArguments, M4_EXPECTED_ARGUMENTS) &&
    summary.trainerSha256 === hashes.trainer && summary.recipeSha256 === hashes.recipe &&
    summary.sourceLocksSha256 === hashes.sourceLocks &&
    summary.selectionSummarySha256 === hashes.selectionSummary &&
    summary.upstreamModelSha256 === M4.upstreamSha256 &&
    summary.trainManifestSha256 === selectionSummary.publicArtifacts["train-manifest.jsonl"].expandedSha256 &&
    summary.selectorManifestSha256 === hashes.selectorManifest &&
    summary.m3RegressionManifestSha256 === hashes.m3RegressionManifest &&
    summary.m2RegressionManifestSha256 === hashes.m2RegressionManifest,
  "M4 failed training input bindings changed");
  const expectedSources = Object.fromEntries(Object.entries({
    ...recipe.baseTraining.sourceCounts,
    ...recipe.expectedTraining.newSourceCounts,
  }).sort());
  requireCondition(summary.trainImages === M4.trainImages && summary.trainFeatureViews === M4.trainViews &&
    jsonEqual(summary.trainSourceCounts, expectedSources) &&
    jsonEqual(summary.trainClassCounts, recipe.expectedTraining.classCounts) &&
    summary.selectorImages === M4.selectorImages && summary.selectorFeatureViews === M4.selectorViews &&
    jsonEqual(summary.selectorSourceCounts, recipe.freshSelector.sourceCounts) &&
    jsonEqual(summary.selectorClassCounts, recipe.freshSelector.classCounts) &&
    summary.sourceBalancedLoss === true &&
    jsonEqual(summary.anchorLossProtectedSources, recipe.adapter.protectedAnchorSources) &&
    summary.candidateCount === M4.candidateCount &&
    jsonEqual(summary.regressionOrder, recipe.selectionPolicy.regressionOrder) &&
    summary.zeroAdapterFeatureParityMaximumAbsoluteError === 0 &&
    summary.h3HoldoutScored === false && summary.h3PixelsRead === false &&
    summary.selectionInfluencedByRegression === false,
  "M4 failed training composition or policy changed");
  validateM4FreshRunEvidence(summary, marker, {
    state: "extracting",
    completedRegressions: recipe.selectionPolicy.regressionOrder.slice(0, completed),
  });
  requireCondition(summary.freshFeatureMarkerSha256 === digest(canonicalBytes(marker)),
    "M4 failed fresh feature marker digest changed");
  requireCondition(Object.prototype.hasOwnProperty.call(STATUS, status) && STATUS[status] === completed,
    "M4 terminal status changed");
}

function validateGridSeal({ summary, grid, tensorSeal, selectorMetadata, recipe, hashes, requireWinner }) {
  const sealedById = validateM4TensorSeal(tensorSeal);
  requireCondition(summary.candidateTensorSealSha256 === digest(canonicalBytes(tensorSeal)) &&
    summary.candidateGridSha256 === digest(canonicalBytes(grid)) &&
    summary.candidateTensorSealSha256 === hashes.tensorSeal && summary.candidateGridSha256 === hashes.grid,
  "M4 failed candidate artifacts changed");
  const winner = recomputeM4Grid(grid, selectorMetadata, recipe.validationGates, { requireWinner });
  requireCondition(grid.candidateTensorSealSha256 === hashes.tensorSeal &&
    grid.selectorManifestSha256 === hashes.selectorManifest &&
    summary.validCandidateCount === grid.validCandidateCount &&
    grid.candidates.every((row) => jsonEqual(sealedById.get(row.candidateId), {
      candidateId: row.candidateId,
      weightDecay: row.weightDecay,
      anchorCoefficient: row.anchorCoefficient,
      trainableParameters: row.trainableParameters,
      tensorSha256: row.tensorSha256,
      tensorShapes: row.tensorShapes,
      tensorDtypes: row.tensorDtypes,
      tensorFloat32Base64: row.tensorFloat32Base64,
    })), "M4 failed grid is not bound to the tensor seal");
  return winner;
}

function expectedSelectionLock(winner, hashes) {
  return {
    schemaVersion: 1,
    candidateTensorSealSha256: hashes.tensorSeal,
    candidateGridSha256: hashes.grid,
    selectedCandidateId: winner.candidateId,
    selectedTensorSha256: winner.tensorSha256,
    rawThreshold: winner.rawThreshold,
    selectorMetrics: winner.selectorMetrics,
    candidateSelectionKey: winner.candidateSelectionKey,
    selectorManifestSha256: hashes.selectorManifest,
    createdBeforeRegressionEvaluation: true,
    selectionInfluencedByRegression: false,
    h3HoldoutScored: false,
  };
}

function validateInventory(snapshot, diagnostic) {
  exactKeys(snapshot, ["fileCount", "bytes", "inventory"], "M4 failed cache snapshot");
  requireCondition(Number.isInteger(snapshot.fileCount) && snapshot.fileCount >= 0 &&
    Number.isInteger(snapshot.bytes) && snapshot.bytes >= 0 && Array.isArray(snapshot.inventory) &&
    snapshot.inventory.length === snapshot.fileCount, "M4 failed cache snapshot header changed");
  let total = 0;
  let previous = "";
  const byPath = new Map();
  for (const row of snapshot.inventory) {
    exactKeys(row, ["path", "bytes", "sha256"], "M4 failed cache inventory row");
    requireCondition(typeof row.path === "string" &&
      row.path.startsWith("benchmark/candidates/prooflens-cf384-m4/") &&
      !row.path.includes("..") && row.path > previous && Number.isInteger(row.bytes) && row.bytes >= 0 &&
      HEX64.test(row.sha256), "M4 failed cache inventory changed");
    previous = row.path;
    total += row.bytes;
    byPath.set(row.path, row);
  }
  requireCondition(total === snapshot.bytes, "M4 failed cache byte total changed");
  const required = {
    "validation-summary.json": "validation-summary.json",
    "candidate-grid.json": "candidate-grid.json",
    "candidate-tensor-seal.json": "candidate-tensor-seal.json",
    "fresh-feature-run.json": "fresh-feature-run.json",
    ...(diagnostic.selectionLock === null ? {} : { "selection-lock.json": "selection-lock.json" }),
  };
  for (const [name, suffix] of Object.entries(required)) {
    const row = byPath.get(`benchmark/candidates/prooflens-cf384-m4/${suffix}`);
    requireCondition(row?.sha256 === diagnostic.candidateArtifactSha256[name],
      `M4 failed cache inventory does not bind ${name}`);
  }
  for (const candidate of diagnostic.candidateTensorSeal.candidates) {
    requireCondition(byPath.has(`benchmark/candidates/prooflens-cf384-m4/candidates/${candidate.candidateId}.npz`),
      `M4 failed cache inventory omits candidate ${candidate.candidateId}`);
  }
  const featureCaches = new Set();
  for (const shard of diagnostic.trainingSummary.featureShardEvidence) {
    const row = byPath.get(shard.cache);
    requireCondition(row?.sha256 === shard.cacheSha256,
      `M4 failed cache inventory does not bind feature shard ${shard.cache}`);
    featureCaches.add(shard.cache);
  }
  for (const pathname of byPath.keys()) {
    if (pathname.startsWith("benchmark/candidates/prooflens-cf384-m4/features/")) {
      requireCondition(featureCaches.has(pathname),
        `M4 failed cache inventory contains an unbound feature shard ${pathname}`);
    }
  }
  return snapshot;
}

export function validateM4FailurePacket({
  diagnosticBytes,
  receiptBytes,
  recipe,
  selectionSummary,
  selectorMetadata,
  regressionMetadata,
  hashes,
}) {
  const diagnostic = parseCanonicalJson(diagnosticBytes, "M4 failure diagnostic");
  const receipt = parseCanonicalJson(receiptBytes, "M4 failure receipt");
  exactKeys(diagnostic, ["schemaVersion", "profile", "sourceCommit", "sourceTree", "baseCommit",
    "recipeSha256", "sourceLocksSha256", "trainerSha256", "upstreamModelSha256", "commandArguments",
    "candidateDirectory", "terminalOutcome", "attemptStatus", "candidateArtifactSha256",
    "selectionLockSha256", "selectionLock", "candidateGrid", "candidateTensorSeal",
    "freshFeatureMarkerJson", "trainingSummary", "selectedCandidate", "completedRegressions",
    "notRunRegressions", "freshFeatureRunId", "featureShardEvidence", "publishedModel",
    "publicationLockCreated", "successfulM4PublicationEvidencePresent", "h3HoldoutScored", "h3PixelsRead",
    "terminality"], "M4 failure diagnostic");
  exactKeys(diagnostic.terminality, ["reselectionPermitted", "thresholdChangePermitted", "gateChangePermitted",
    "retrySameSelectorPermitted"], "M4 failure terminality");
  const completed = STATUS[diagnostic.attemptStatus];
  requireCondition(Number.isInteger(completed) && diagnostic.schemaVersion === 1 && diagnostic.profile === "m4" &&
    diagnostic.sourceCommit === hashes.sourceCommit && diagnostic.sourceTree === hashes.sourceTree &&
    diagnostic.baseCommit === M4.baseCommit && diagnostic.recipeSha256 === hashes.recipe &&
    diagnostic.sourceLocksSha256 === hashes.sourceLocks && diagnostic.trainerSha256 === hashes.trainer &&
    diagnostic.upstreamModelSha256 === M4.upstreamSha256 &&
    jsonEqual(diagnostic.commandArguments, M4_EXPECTED_ARGUMENTS) &&
    diagnostic.candidateDirectory === "benchmark/candidates/prooflens-cf384-m4" &&
    diagnostic.terminalOutcome === diagnostic.attemptStatus && diagnostic.publishedModel === false &&
    diagnostic.publicationLockCreated === false && diagnostic.successfulM4PublicationEvidencePresent === false &&
    diagnostic.h3HoldoutScored === false && diagnostic.h3PixelsRead === false &&
    Object.values(diagnostic.terminality).every((value) => value === false),
  "M4 failure identity or terminal boundary changed");
  validateSelectionSummary(selectionSummary, recipe);
  const markerBytes = Buffer.from(diagnostic.freshFeatureMarkerJson, "utf8");
  const marker = parseCanonicalJson(markerBytes, "M4 failed fresh feature marker");
  const summary = diagnostic.trainingSummary;
  const grid = diagnostic.candidateGrid;
  const tensorSeal = diagnostic.candidateTensorSeal;
  validateSummaryCommon({ summary, recipe, selectionSummary, hashes, marker, completed });
  const winner = validateGridSeal({
    summary, grid, tensorSeal, selectorMetadata, recipe, hashes, requireWinner: completed > 0,
  });
  const summaryBytes = canonicalBytes(summary);
  const gridBytes = canonicalBytes(grid);
  const sealBytes = canonicalBytes(tensorSeal);
  const expectedArtifacts = {
    "validation-summary.json": digest(summaryBytes),
    "candidate-grid.json": digest(gridBytes),
    "candidate-tensor-seal.json": digest(sealBytes),
    "fresh-feature-run.json": digest(markerBytes),
  };
  if (completed === 0) {
    requireCondition(winner === null && grid.validCandidateCount === 0 && summary.validCandidateCount === 0 &&
      diagnostic.selectionLockSha256 === null && diagnostic.selectionLock === null &&
      diagnostic.selectedCandidate === null && diagnostic.completedRegressions.length === 0,
    "M4 selector failure contains a winner or regression");
  } else {
    requireCondition(winner !== null, "M4 regression failure lacks a selector winner");
    const selectionLock = expectedSelectionLock(winner, hashes);
    const selectionLockBytes = canonicalBytes(selectionLock);
    expectedArtifacts["selection-lock.json"] = digest(selectionLockBytes);
    requireCondition(diagnostic.selectionLockSha256 === digest(selectionLockBytes) &&
      jsonEqual(diagnostic.selectionLock, selectionLock) && jsonEqual(summary.selectionLock, selectionLock) &&
      summary.selectionLockSha256 === digest(selectionLockBytes) &&
      jsonEqual(summary.selectedCandidate, winner) && jsonEqual(diagnostic.selectedCandidate, winner) &&
      Array.isArray(summary.regressions) && summary.regressions.length === completed,
    "M4 failed pre-regression selection lock changed");
    const specifications = [
      {
        name: "m3-selector-regression", featureViews: 2_400, gates: recipe.regressions[0].gates,
        sources: { synthetic: ["flux-1-dev-development"], real: ["met-open-access"] },
        metadata: regressionMetadata.m3,
      },
      {
        name: "m2-development-regression", featureViews: 3_600, gates: recipe.regressions[1].gates,
        sources: { synthetic: ["GLM-Image", "HunyuanImage-3.0"], real: ["open-images", "stockimages-cc0"] },
        metadata: regressionMetadata.m2,
      },
    ];
    for (let index = 0; index < completed; index += 1) {
      const passed = validateM4Regression(summary.regressions[index], specifications[index].metadata,
        specifications[index], winner.rawThreshold, { requirePassed: false });
      requireCondition(index === completed - 1 ? passed === false : passed === true,
        "M4 terminal regression order or result changed");
    }
  }
  requireCondition(jsonEqual(diagnostic.candidateArtifactSha256, expectedArtifacts) &&
    jsonEqual(diagnostic.completedRegressions, summary.regressions ?? []) &&
    jsonEqual(diagnostic.notRunRegressions, recipe.selectionPolicy.regressionOrder.slice(completed)) &&
    diagnostic.freshFeatureRunId === summary.freshFeatureRunId &&
    jsonEqual(diagnostic.featureShardEvidence, summary.featureShardEvidence),
  "M4 failure diagnostic evidence binding changed");

  exactKeys(receipt, ["schemaVersion", "profile", "sourceCommit", "sourceTree", "diagnosticPath",
    "diagnosticSha256", "terminalOutcome", "candidateCacheSnapshot",
    "successfulM4PublicationEvidencePresent", "h3HoldoutScored", "h3PixelsRead",
    "requiredFailureCommitRows"], "M4 failure receipt");
  requireCondition(receipt.schemaVersion === 1 && receipt.profile === "m4" &&
    receipt.sourceCommit === hashes.sourceCommit && receipt.sourceTree === hashes.sourceTree &&
    receipt.diagnosticPath === "benchmark/evidence/m4/failed-selector-diagnostic-1.json" &&
    receipt.diagnosticSha256 === digest(diagnosticBytes) && receipt.terminalOutcome === diagnostic.terminalOutcome &&
    receipt.successfulM4PublicationEvidencePresent === false && receipt.h3HoldoutScored === false &&
    receipt.h3PixelsRead === false && jsonEqual(receipt.requiredFailureCommitRows, FAILURE_ROWS),
  "M4 failure receipt binding changed");
  validateInventory(receipt.candidateCacheSnapshot, diagnostic);
  return { diagnostic, receipt, marker, summary, grid, tensorSeal, winner };
}
