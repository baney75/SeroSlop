import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

import {
  M4,
  M4_EXPECTED_ARGUMENTS,
  M4_TENSOR_SHAPES,
  digest,
  parseCanonicalJson,
  validateM4TrainingPacket,
} from "./m4-training-contract.mjs";


const VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];
const DECAYS = [0.003, 0.01, 0.03];
const ANCHORS = [0.01, 0.03, 0.1, 0.3];
const H = (character) => character.repeat(64);

export function canonicalBytes(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`, "utf8");
}

export function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function arrayDigest(bytes, shape) {
  return createHash("sha256").update("<f4").update(JSON.stringify(shape)).update(bytes).digest("hex");
}

function encodedFloat32(values) {
  const bytes = Buffer.alloc(values.length * 4);
  values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
  return {
    base64: bytes.toString("base64"),
    sha256: arrayDigest(bytes, [values.length]),
    count: values.length,
  };
}

function tensorRecord(weightDecay, anchorCoefficient) {
  const tensorSha256 = {};
  const tensorShapes = {};
  const tensorDtypes = {};
  const tensorFloat32Base64 = {};
  for (const [name, shape] of Object.entries(M4_TENSOR_SHAPES)) {
    const count = shape.reduce((product, value) => product * value, 1);
    const bytes = Buffer.alloc(count * 4);
    if (name === "m4.feature_std") {
      for (let index = 0; index < count; index += 1) bytes.writeFloatLE(1, index * 4);
    }
    tensorSha256[name] = arrayDigest(bytes, shape);
    tensorShapes[name] = shape;
    tensorDtypes[name] = "float32";
    tensorFloat32Base64[name] = bytes.toString("base64");
  }
  return {
    candidateId: `wd-${weightDecay.toFixed(3)}-anchor-${anchorCoefficient.toFixed(2)}`,
    weightDecay,
    anchorCoefficient,
    trainableParameters: 49_600,
    tensorSha256,
    tensorShapes,
    tensorDtypes,
    tensorFloat32Base64,
  };
}

function selectorMetadata() {
  const labels = [];
  const variants = [];
  const sources = [];
  const rows = [
    ["british-library-plates", 0, 300],
    ["rapidata-dalle-3", 1, 75],
    ["rapidata-flux", 1, 75],
    ["rapidata-midjourney", 1, 75],
    ["rapidata-stable-diffusion", 1, 75],
  ];
  for (const [source, label, count] of rows) {
    for (let item = 0; item < count; item += 1) {
      for (let variant = 0; variant < 4; variant += 1) {
        labels.push(label); variants.push(variant); sources.push(source);
      }
    }
  }
  return { labels, variants, sources };
}

function regressionMetadata(realSources, syntheticSources, counts) {
  const labels = [];
  const variants = [];
  const sources = [];
  for (const [source, count] of realSources.map((source, index) => [source, counts.real[index]])) {
    for (let item = 0; item < count; item += 1) for (let variant = 0; variant < 4; variant += 1) {
      labels.push(0); variants.push(variant); sources.push(source);
    }
  }
  for (const [source, count] of syntheticSources.map((source, index) => [source, counts.synthetic[index]])) {
    for (let item = 0; item < count; item += 1) for (let variant = 0; variant < 4; variant += 1) {
      labels.push(1); variants.push(variant); sources.push(source);
    }
  }
  return { labels, variants, sources };
}

function perfectMetrics(realSources, syntheticSources) {
  return Object.fromEntries(VARIANTS.map((variant) => [variant, {
    balancedAccuracy: 1,
    realRecall: 1,
    syntheticRecall: 1,
    syntheticRecallBySource: Object.fromEntries(syntheticSources.map((source) => [source, 1])),
    realRecallBySource: Object.fromEntries(realSources.map((source) => [source, 1])),
  }]));
}

function regressionRow(name, metadata, realSources, syntheticSources) {
  const logits = metadata.labels.map((label) => label === 0 ? -2 : 2);
  const encoded = encodedFloat32(logits);
  return {
    name,
    metrics: perfectMetrics(realSources, syntheticSources),
    passed: true,
    logitsSha256: encoded.sha256,
    logitsFloat32Base64: encoded.base64,
    logitCount: encoded.count,
  };
}

function featureEvidence(runId, trainingHash, evaluationHash) {
  const output = [];
  const add = (prefix, count, itemCount, viewCount, configuration, viewFn) => {
    let remainingItems = itemCount;
    let remainingViews = viewCount;
    for (let index = 0; index < count; index += 1) {
      const items = Math.min(2_000, remainingItems);
      const views = viewFn ? viewFn(index, items) : (index === count - 1 ? remainingViews : items);
      output.push({
        cache: `benchmark/candidates/prooflens-cf384-m4/features/${prefix}-${String(index).padStart(5, "0")}.npz`,
        cacheSha256: H("a"),
        replacedCacheSha256: null,
        freshFeatureRunId: runId,
        freshlyExtractedThisRun: true,
        freshlyExtractedThisProcess: true,
        items,
        views,
        itemIdsSha256: H("b"),
        featureConfigurationSha256: configuration,
        arraySha256: { features: H("c"), labels: H("d"), variants: H("e"), sources: H("f") },
      });
      remainingItems -= items;
      remainingViews -= views;
    }
    assert.equal(remainingItems, 0);
    assert.equal(remainingViews, 0);
  };
  add("train", 57, 112_562, 150_248, trainingHash,
    (index, items) => index < 50 ? items : items * 4);
  add("selector", 1, 600, 2_400, evaluationHash);
  add("regression-m3", 1, 600, 2_400, evaluationHash);
  add("regression-m2", 1, 900, 3_600, evaluationHash);
  return output;
}

export function buildTrainingFixture({ selectorFeasible = true } = {}) {
  const recipe = JSON.parse(readFileSync("benchmark/m4/recipe.json", "utf8"));
  const selector = selectorMetadata();
  const m3 = regressionMetadata(["met-open-access"], ["flux-1-dev-development"], {
    real: [300], synthetic: [300],
  });
  const m2 = regressionMetadata(["open-images", "stockimages-cc0"], ["GLM-Image", "HunyuanImage-3.0"], {
    real: [300, 300], synthetic: [150, 150],
  });
  const selectorLogits = selector.labels.map((label) => selectorFeasible ? (label === 0 ? -2 : 2) : 0);
  const selectorEncoded = encodedFloat32(selectorLogits);
  const candidates = [];
  const seals = [];
  for (const decay of DECAYS) for (const anchor of ANCHORS) {
    const tensors = tensorRecord(decay, anchor);
    seals.push(tensors);
    const row = {
      ...tensors,
      selectorLogitsFloat32Base64: selectorEncoded.base64,
      selectorLogitsSha256: selectorEncoded.sha256,
      selectorLogitCount: selectorEncoded.count,
      thresholdPartitions: selectorFeasible ? 3 : 2,
      valid: selectorFeasible,
    };
    if (selectorFeasible) Object.assign(row, {
      rawThreshold: -1.9999999999999998,
      selectorMetrics: perfectMetrics(["british-library-plates"], [
        "rapidata-dalle-3", "rapidata-flux", "rapidata-midjourney", "rapidata-stable-diffusion",
      ]),
      selectorKey: [1, 1, 1, 1, 1],
      candidateSelectionKey: [1, 1, 1, 1, 1, -decay, -anchor],
    });
    candidates.push(row);
  }
  const tensorSeal = {
    schemaVersion: 1, createdBeforeSelectorEvaluation: true, candidateCount: 12, candidates: seals,
  };
  const hashes = {
    trainer: H("1"), selectionSummary: H("2"), selectorManifest: H("3"),
    m3RegressionManifest: H("4"), m2RegressionManifest: H("5"),
    model: H("6"), modelBytes: 87_644_354,
  };
  hashes.tensorSeal = digest(canonicalBytes(tensorSeal));
  const grid = {
    schemaVersion: 1,
    candidateTensorSealSha256: hashes.tensorSeal,
    selectorManifestSha256: hashes.selectorManifest,
    candidateCount: 12,
    validCandidateCount: selectorFeasible ? 12 : 0,
    candidates,
  };
  hashes.grid = digest(canonicalBytes(grid));
  const winner = selectorFeasible ? candidates[0] : null;
  const selectionLock = selectorFeasible ? {
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
  } : null;
  hashes.selectionLock = selectorFeasible ? digest(canonicalBytes(selectionLock)) : null;
  const trainingHash = H("7");
  const evaluationHash = H("8");
  const runId = "0123456789abcdef0123456789abcdef";
  const selectionSummary = {
    schemaVersion: 1,
    recipeSha256: M4.recipeSha256,
    sourceLocksSha256: M4.sourceLocksSha256,
    scoreBlind: true,
    modelOutputsRead: false,
    h3PixelsRead: false,
    h3ManifestSha256: recipe.h3Exclusion.sha256,
    selectionOrder: ["british-selector", "rapidata-selector", "british-training", "rapidata-training"],
    training: {}, freshSelector: {}, partitionGroups: {}, overlap: {},
    publicArtifacts: { "train-manifest.jsonl": { expandedSha256: H("9") } },
  };
  const marker = {
    schemaVersion: 1,
    runId,
    state: "complete",
    context: {
      pipelineVersion: 1,
      trainerSha256: hashes.trainer,
      recipeSha256: M4.recipeSha256,
      modelSha256: M4.upstreamSha256,
      trainManifestSha256: H("9"),
      selectorManifestSha256: hashes.selectorManifest,
      m3RegressionManifestSha256: hashes.m3RegressionManifest,
      m2RegressionManifestSha256: hashes.m2RegressionManifest,
      selectionSummarySha256: hashes.selectionSummary,
      featureConfigurationHashes: { training: trainingHash, evaluation: evaluationHash },
      featureBatchSize: 24,
      featureShardImages: 2_000,
      singleViewSources: ["diffusiondb-stable-diffusion", "open-images-train"],
    },
  };
  hashes.freshMarker = digest(canonicalBytes(marker));
  const calibration = selectorFeasible ? {
    schemaVersion: 1,
    mode: "threshold-alignment-not-probability-calibration",
    slope: 1,
    intercept: Math.log(0.65 / 0.35) - winner.rawThreshold,
    rawThreshold: winner.rawThreshold,
    displayThreshold: 0.65,
    rawProbabilityAtThreshold: 1 / (1 + Math.exp(-winner.rawThreshold)),
    displayProbabilityAtRawThreshold: 1 / (1 + Math.exp(-(winner.rawThreshold +
      (Math.log(0.65 / 0.35) - winner.rawThreshold)))),
    modelSha256: hashes.model,
    selectionLockSha256: hashes.selectionLock,
    selectorManifestSha256: hashes.selectorManifest,
  } : null;
  hashes.calibration = calibration ? digest(canonicalBytes(calibration)) : null;
  const regressions = selectorFeasible ? [
    regressionRow("m3-selector-regression", m3, ["met-open-access"], ["flux-1-dev-development"]),
    regressionRow("m2-development-regression", m2, ["open-images", "stockimages-cc0"],
      ["GLM-Image", "HunyuanImage-3.0"]),
  ] : [];
  const summary = {
    schemaVersion: 1,
    pipelineVersion: 1,
    status: selectorFeasible ? "accepted-development-candidate" : "failed-selector",
    seed: M4.seed,
    commandArguments: M4_EXPECTED_ARGUMENTS,
    trainerSha256: hashes.trainer,
    recipeSha256: M4.recipeSha256,
    sourceLocksSha256: M4.sourceLocksSha256,
    selectionSummarySha256: hashes.selectionSummary,
    upstreamModelSha256: M4.upstreamSha256,
    trainManifestSha256: H("9"),
    selectorManifestSha256: hashes.selectorManifest,
    m3RegressionManifestSha256: hashes.m3RegressionManifest,
    m2RegressionManifestSha256: hashes.m2RegressionManifest,
    trainImages: M4.trainImages,
    trainFeatureViews: M4.trainViews,
    trainSourceCounts: Object.fromEntries(Object.entries({
      ...recipe.baseTraining.sourceCounts, ...recipe.expectedTraining.newSourceCounts,
    }).sort()),
    trainClassCounts: recipe.expectedTraining.classCounts,
    selectorImages: M4.selectorImages,
    selectorFeatureViews: M4.selectorViews,
    selectorSourceCounts: recipe.freshSelector.sourceCounts,
    selectorClassCounts: recipe.freshSelector.classCounts,
    candidateTensorSealSha256: hashes.tensorSeal,
    candidateGridSha256: hashes.grid,
    zeroAdapterFeatureParityMaximumAbsoluteError: 0,
    featureConfigurationHashes: marker.context.featureConfigurationHashes,
    featureShardEvidence: featureEvidence(runId, trainingHash, evaluationHash),
    freshFeatureRunId: runId,
    freshFeatureMarkerSha256: hashes.freshMarker,
    sourceBalancedLoss: true,
    anchorLossProtectedSources: recipe.adapter.protectedAnchorSources,
    candidateCount: 12,
    validCandidateCount: grid.validCandidateCount,
    regressionOrder: recipe.selectionPolicy.regressionOrder,
    h3HoldoutScored: false,
    h3PixelsRead: false,
    selectionInfluencedByRegression: false,
    environment: {
      python: "3.12.0", numpy: "2.2.6", torch: "2.7.0", onnxRuntime: "1.22.0",
      pillow: "11.3.0", platform: "test", providers: ["CPUExecutionProvider"],
    },
    ...(selectorFeasible ? {
      selectionLockSha256: hashes.selectionLock,
      selectionLock,
      selectedCandidate: winner,
      regressions,
      freshFeatureRunComplete: true,
      modelSha256: hashes.model,
      modelBytes: hashes.modelBytes,
      zeroAdapterImageParityMaximumAbsoluteError: 0,
      exportedCandidateImageParityMaximumAbsoluteError: 0,
      calibrationSha256: hashes.calibration,
    } : {}),
  };
  return {
    recipe, selectionSummary, selectorMetadata: selector,
    regressionMetadata: { m3, m2 }, tensorSeal, grid, selectionLock, marker, calibration,
    summary, hashes, winner,
  };
}

function validateFixture(fixture) {
  return validateM4TrainingPacket({
    summary: fixture.summary,
    calibration: fixture.calibration,
    grid: fixture.grid,
    recipe: fixture.recipe,
    selectionSummary: fixture.selectionSummary,
    hashes: fixture.hashes,
    selectorMetadata: fixture.selectorMetadata,
    regressionMetadata: fixture.regressionMetadata,
    tensorSeal: fixture.tensorSeal,
    selectionLock: fixture.selectionLock,
    freshMarker: fixture.marker,
  });
}

function main() {
  const fixture = buildTrainingFixture();
  assert.equal(validateFixture(fixture).candidateId, fixture.winner.candidateId);

  const badCalibration = clone(fixture);
  badCalibration.calibration.selectionLock = badCalibration.selectionLock;
  assert.throws(() => validateFixture(badCalibration), /calibration.*keys changed/u);

  const badMarker = clone(fixture);
  badMarker.marker.context.featureBatchSize = 1;
  assert.throws(() => validateFixture(badMarker), /marker context changed/u);

  const badCache = clone(fixture);
  badCache.summary.featureShardEvidence[0].cache = "benchmark/candidates/prooflens-cf384-m4/features/wrong.npz";
  assert.throws(() => validateFixture(badCache), /feature-shard evidence changed/u);

  const badTensor = clone(fixture);
  const name = Object.keys(badTensor.tensorSeal.candidates[0].tensorFloat32Base64)[0];
  const bytes = Buffer.from(badTensor.tensorSeal.candidates[0].tensorFloat32Base64[name], "base64");
  bytes.writeFloatLE(2, 0);
  badTensor.tensorSeal.candidates[0].tensorFloat32Base64[name] = bytes.toString("base64");
  assert.throws(() => validateFixture(badTensor), /tensor bytes changed/u);

  const badFamily = clone(fixture);
  badFamily.regressionMetadata.m2.sources[2_400] = "wrong-family";
  assert.throws(() => validateFixture(badFamily), /source set changed/u);

  const wrongWinner = clone(fixture);
  wrongWinner.summary.selectedCandidate = wrongWinner.grid.candidates.at(-1);
  assert.throws(() => validateFixture(wrongWinner), /winner changed/u);

  const canonical = canonicalBytes({ schemaVersion: 1, value: "ok" });
  assert.deepEqual(parseCanonicalJson(canonical), { schemaVersion: 1, value: "ok" });
  assert.throws(() => parseCanonicalJson(Buffer.from('{"x":1,"x":2}\n')), /not canonical JSON/u);
  assert.throws(() => parseCanonicalJson(Buffer.from([0xff])), /canonical UTF-8/u);

  console.log(JSON.stringify({ cases: 9, candidates: 12, selectorViews: 2_400, policy: "pass" }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
