import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  M2,
  M2_EXPECTED_ARGUMENTS,
  M2_EXPECTED_PARAMETERS,
  M2_EXPECTED_SHARDS,
  M2_EXPECTED_SOURCES,
  M2_SINGLE_VIEW_SOURCES,
  M2_VARIANTS,
  digest,
  validateM2PublicationMetadata,
  validateM2OnnxEvidence,
  validateM2PublicDocumentation,
  validateM2TrainingPacket,
} from "./m2-training-contract.mjs";

const recipeBytes = readFileSync("benchmark/m2/recipe.json");
const selectionBytes = readFileSync("benchmark/evidence/m2/selection-summary.json");
assert.equal(digest(recipeBytes), M2.recipeSha256);
assert.equal(digest(selectionBytes), M2.selectionSummarySha256);
const recipe = JSON.parse(recipeBytes);
const selectionSummary = JSON.parse(selectionBytes);

function hash(value) {
  return digest(Buffer.from(value));
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function metrics(stockRecall = 0.95) {
  return Object.fromEntries(M2_VARIANTS.map((variant) => [variant, {
    balancedAccuracy: 0.95,
    realRecall: 0.95,
    syntheticRecall: 0.95,
    syntheticRecallBySource: { "GLM-Image": 0.9, "HunyuanImage-3.0": 1 },
    realRecallBySource: { "open-images": 0.95, "stockimages-cc0": stockRecall },
  }]));
}

function buildFixture() {
  const selectedVariants = metrics(0.95);
  const grid = M2_EXPECTED_PARAMETERS.map((parameters, index) => {
    if (parameters.weightDecay === 0.1 && parameters.upstreamBlendAlpha === 0.4) {
      return {
        parameters,
        status: "rejected",
        reason: "No threshold satisfies the frozen validation gates",
      };
    }
    const selected = parameters.weightDecay === 0.1 && parameters.upstreamBlendAlpha === 0.85;
    return {
      parameters,
      thresholdLogit: selected ? M2.thresholdLogit : -2 - index / 100,
      selectionKey: selected ? [...M2.selectionKey] : [0.9, 0.91, 0.92, 0.93, 0.7],
      variants: selected ? selectedVariants : metrics(0.95),
    };
  });
  const configuration = { training: hash("training-config"), validation: hash("validation-config") };
  const freshFeatureRun = {
    schemaVersion: 1,
    runId: M2.freshRunId,
    state: "complete",
    context: {
      pipelineVersion: M2.pipelineVersion,
      featureExtractorContract: "cf384-static-batch24-preprocess-v1",
      upstreamModelSha256: M2.upstreamSha256,
      trainManifestSha256: M2.trainManifestSha256,
      validationManifestSha256: M2.validationManifestSha256,
      selectionSummarySha256: M2.selectionSummarySha256,
      featureBatchSize: 24,
      featureShardImages: 2_000,
      singleViewTrainingSources: [...M2_SINGLE_VIEW_SOURCES],
      featureConfigurationHashes: configuration,
    },
  };
  const featureShardEvidence = M2_EXPECTED_SHARDS.map((cache, index) => {
    const training = index < M2.trainShards;
    const items = training ? (index < 52 ? 2_000 : 1_978) : 900;
    let views = items;
    if (index === 0 || index === 1) views = 8_000;
    if (index === 2) views = 7_934;
    if (!training) views = 3_600;
    return {
      cache,
      cacheSha256: hash(`cache-${index}`),
      replacedCacheSha256: null,
      freshFeatureRunId: M2.freshRunId,
      freshlyExtractedThisRun: true,
      freshlyExtractedThisProcess: true,
      items,
      views,
      itemIdsSha256: hash(`items-${index}`),
      featureConfigurationSha256: configuration[training ? "training" : "validation"],
      arraySha256: Object.fromEntries(["features", "labels", "variants", "sources"]
        .map((name) => [name, hash(`${name}-${index}`)])),
    };
  });
  const summary = {
    schemaVersion: 1,
    pipelineVersion: M2.pipelineVersion,
    seed: M2.seed,
    trainerSha256: M2.trainerSha256,
    commandArguments: [...M2_EXPECTED_ARGUMENTS],
    recipeSha256: M2.recipeSha256,
    selectionSummarySha256: M2.selectionSummarySha256,
    upstreamModelSha256: M2.upstreamSha256,
    trainManifestSha256: M2.trainManifestSha256,
    validationManifestSha256: M2.validationManifestSha256,
    trainImages: M2.trainImages,
    trainFeatureViews: M2.trainViews,
    validationImages: M2.validationImages,
    validationFeatureViews: M2.validationViews,
    uniqueTrainingImagesCovered: M2.trainImages,
    uniqueTrainingFeatureViewsCovered: M2.trainViews,
    viewsPerImage: 4,
    sourceBalancedSampling: false,
    sourceBalancedLoss: true,
    trainingEpochs: 12,
    trainingBatchSize: 2_048,
    featureBatchSize: 24,
    featureShardImages: 2_000,
    cachedFeatureSourcePixelsReverified: true,
    cachedFeatureArraysValidated: true,
    cachedFeatureValuesReextracted: true,
    cachedFeatureDtypes: { features: "float32", labels: "float32", variants: "int64", sources: "unicode" },
    singleViewTrainingSources: [...M2_SINGLE_VIEW_SOURCES],
    trainingSourceCounts: { ...M2_EXPECTED_SOURCES },
    featureConfigurationHashes: configuration,
    freshFeatureRun,
    freshFeatureRunMarkerSha256: digest(Buffer.from(`${JSON.stringify(freshFeatureRun, null, 2)}\n`)),
    featureShardEvidence,
    environment: {
      executionProvider: "cpu",
      providers: ["CPUExecutionProvider"],
      torchDeterministicAlgorithms: true,
      cuda: null,
      gpu: null,
      numpy: "2.2.6",
      onnxRuntime: "1.22.0",
      pillow: "11.3.0",
      python: "3.12.13",
      platform: "test",
      torch: "2.8.0",
    },
    validationGates: clone(recipe.validationGates),
    validationGatesPassed: true,
    candidateCount: M2.candidateCount,
    validCandidateCount: M2.validCandidateCount,
    selectedParameters: { ...M2.selectedParameters },
    selectionKey: [...M2.selectionKey],
    thresholdLogit: M2.thresholdLogit,
    variants: selectedVariants,
    model: {
      path: "benchmark/candidates/prooflens-cf384-m2/model.onnx",
      sha256: M2.modelSha256,
      bytes: M2.modelBytes,
      maxAbsParityError: 0.00001,
    },
  };
  const calibration = {
    schemaVersion: 1,
    method: "Validation-selected logit alignment to fixed 65/100 display threshold; not probability calibration",
    modelSha256: M2.modelSha256,
    trainManifestSha256: M2.trainManifestSha256,
    validationManifestSha256: M2.validationManifestSha256,
    selectionSummarySha256: M2.selectionSummarySha256,
    slope: 1,
    intercept: 1.6126519746720926,
    displayThreshold: M2.displayThreshold,
    validationThresholdLogit: M2.thresholdLogit,
    rawProbabilityThreshold: 0.27019907955040323,
  };
  return {
    summary,
    calibration,
    grid,
    recipe: clone(recipe),
    selectionSummary: clone(selectionSummary),
    fileHashes: {
      recipe: M2.recipeSha256,
      selectionSummary: M2.selectionSummarySha256,
      trainingSummary: M2.trainingSummarySha256,
      calibration: M2.calibrationSha256,
      candidateGrid: M2.candidateGridSha256,
    },
    model: { sha256: M2.modelSha256, bytes: M2.modelBytes },
  };
}

function expectFailure(mutate, pattern) {
  const fixture = buildFixture();
  mutate(fixture);
  assert.throws(() => validateM2TrainingPacket(fixture), pattern);
}

validateM2TrainingPacket(buildFixture());
expectFailure((fixture) => { fixture.model.sha256 = "0".repeat(64); }, /model bytes changed/u);
expectFailure((fixture) => { fixture.fileHashes.trainingSummary = "0".repeat(64); }, /byte hashes changed/u);
expectFailure((fixture) => { fixture.summary.commandArguments[15] = "benchmark/evidence/m2/selection-summary.json"; },
  /input bindings changed/u);
expectFailure((fixture) => { fixture.grid[1].parameters = { ...fixture.grid[2].parameters }; },
  /parameter coverage changed/u);
expectFailure((fixture) => { fixture.grid.find((row) => Array.isArray(row.selectionKey)).selectionKey.pop(); },
  /selection evidence is malformed/u);
expectFailure((fixture) => { fixture.summary.variants.original.balancedAccuracy = Number.NaN; },
  /failed minimumBalancedAccuracy/u);
expectFailure((fixture) => { fixture.summary.variants.original.realRecallBySource["stockimages-cc0"] = 0.9299; },
  /failed stockimages-cc0 real recall/u);
expectFailure((fixture) => { fixture.summary.featureShardEvidence[0].freshlyExtractedThisProcess = false; },
  /feature shard evidence changed/u);
expectFailure((fixture) => { fixture.summary.featureShardEvidence[1].cache = fixture.summary.featureShardEvidence[0].cache; },
  /feature shard paths changed/u);
expectFailure((fixture) => { fixture.grid.find((row) => row.parameters.upstreamBlendAlpha === 0.55).selectionKey[0] = 1; },
  /not the frozen deterministic maximum/u);

const publication = {
  modelLock: {
    schemaVersion: 2,
    artifact: "weights/prooflens-cf384.onnx",
    bytes: M2.modelBytes,
    sha256: M2.modelSha256,
    trainingRecipe: `${M2.identity}:${M2.recipeSha256}:${M2.selectionSummarySha256}`,
    trainingEvidence: {
      recipe: "benchmark/m2/recipe.json",
      recipeSha256: M2.recipeSha256,
      selectionSummary: "benchmark/evidence/m2/selection-summary.json",
      selectionSummarySha256: M2.selectionSummarySha256,
      trainManifestSha256: M2.trainManifestSha256,
      trainingSummarySha256: M2.trainingSummarySha256,
      calibrationSha256: M2.calibrationSha256,
      candidateGridSha256: M2.candidateGridSha256,
    },
    calibration: { displayThreshold: M2.displayThreshold, validationThresholdLogit: M2.thresholdLogit },
  },
  weightsReadme: `${M2.modelSha256}\n${M2.modelBytes.toLocaleString("en-US")}`,
  comparison: {
    schemaVersion: 1,
    base: { path: "benchmark/candidates/upstream-cf384.onnx", sha256: M2.upstreamSha256, bytes: M2.modelBytes },
    candidate: {
      path: "benchmark/candidates/prooflens-cf384-m2/model.onnx",
      sha256: M2.modelSha256,
      bytes: M2.modelBytes,
    },
    unchangedInitializerCount: 198,
    changedInitializers: [{ name: "classifier.bias" }, { name: "classifier.weight" }],
  },
  repositoryHashes: {
    "weights/prooflens-cf384.onnx": M2.modelSha256,
    "model-lock.json": hash("lock"),
    "weights/README.md": hash("readme"),
  },
};
publication.receipt = {
  schemaVersion: 3,
  profile: "m2",
  candidateDirectory: "benchmark/candidates/prooflens-cf384-m2",
  upstreamSha256: M2.upstreamSha256,
  shippedModel: { path: "weights/prooflens-cf384.onnx", sha256: M2.modelSha256, bytes: M2.modelBytes },
  sourceEvidenceSha256: {
    "training-summary.json": M2.trainingSummarySha256,
    "calibration.json": M2.calibrationSha256,
    "candidate-grid.json": M2.candidateGridSha256,
  },
  publishedEvidenceSha256: {
    "training-summary.json": M2.trainingSummarySha256,
    "calibration.json": M2.calibrationSha256,
    "candidate-grid.json": M2.candidateGridSha256,
    "model-comparison.json": M2.modelComparisonSha256,
  },
  publishedRepositorySha256: publication.repositoryHashes,
};
validateM2PublicationMetadata(publication);
const currentStart = "<!-- PROOFLENS_CURRENT_M2_START -->";
const currentEnd = "<!-- PROOFLENS_CURRENT_M2_END -->";
const historyStart = "<!-- PROOFLENS_HISTORICAL_M1_START -->";
const historyEnd = "<!-- PROOFLENS_HISTORICAL_M1_END -->";
const readmeFacts = `${M2.modelSha256} 105,978 public images 54,778 non-AI 123,912 feature views ` +
  "94.08% balanced accuracy on originals benchmark/evidence/m2/";
const modelCardFacts = `${M2.modelSha256} 105,978 unique public images 2,378 StockImages-CC0 ` +
  "123,912 feature views 24 of 25 candidate configurations npm run check:m2-training-evidence";
const benchmarkFacts = `${M2.modelSha256} 105,978 123,912 24 of 25 candidates ` +
  "94.08% balanced accuracy on originals benchmark/evidence/m2/";
const historyFacts = "941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c " +
  "103,600 training images 114,400 feature views 25 candidate configurations were valid; " +
  "v1 and replacement-v2 are consumed and acceptance-ineligible; acceptanceEligible: false";
const publicDocumentation = {
  readme: `# ProofLens\n\n${currentStart}\n## Current M2 model\n${readmeFacts}\n${currentEnd}`,
  modelCard: `# Model\n\n${currentStart}\n## Current M2 head-training data\n${modelCardFacts}\n` +
    `${currentEnd}\n\n${historyStart}\n## Historical M1 and evaluation evidence\n${historyFacts}\n${historyEnd}`,
  benchmark: `# Benchmark\n\n${currentStart}\n## Current M2 model-selection boundary\n${benchmarkFacts}\n` +
    `${currentEnd}\n\n${historyStart}\n## Historical M1 training and evaluation evidence\n` +
    `${historyFacts}\n${historyEnd}`,
};
validateM2PublicDocumentation(publicDocumentation);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  modelCard: publicDocumentation.modelCard.replace(M2.modelSha256, "0".repeat(64)),
}), /MODEL_CARD\.md current M2 section does not describe/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  readme: `# ProofLens\n\n${currentStart}\n## Current M2 model\nNo facts here.\n${currentEnd}` +
    `\n\n ## Appendix\n${readmeFacts}`,
}), /README\.md current M2 section does not describe/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  readme: publicDocumentation.readme.replace("benchmark/evidence/m2/", "benchmark/evidence/m2/\n" +
    "941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c"),
}), /README\.md current M2 section retains a stale M1/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  modelCard: publicDocumentation.modelCard.replace("npm run check:m2-training-evidence",
    "npm run check:m2-training-evidence All 25 candidate configurations were valid"),
}), /MODEL_CARD\.md current M2 section retains a stale M1/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  modelCard: publicDocumentation.modelCard.replace("npm run check:m2-training-evidence",
    "npm run check:m2-training-evidence " +
    "941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c"),
}), /MODEL_CARD\.md current M2 section retains a stale M1/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  modelCard: publicDocumentation.modelCard.replace("103,600 training images", "103,601 training images"),
}), /MODEL_CARD\.md historical M1 section changed/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  modelCard: `# Model\n\n${currentStart}\n## Current M2 head-training data\n${modelCardFacts}\n` +
    `${currentEnd}\n\n${historyStart}\n## Historical M1 and evaluation evidence\nNo history.\n` +
    `${historyEnd}\n\nHistorical appendix\n---\n${historyFacts}`,
}), /MODEL_CARD\.md historical M1 section changed/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  benchmark: publicDocumentation.benchmark.replace("benchmark/evidence/m2/",
    "benchmark/evidence/m2/ Fresh extraction covered all 103,600 images and 114,400 configured views"),
}), /BENCHMARK\.md current M2 section retains a stale M1/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  benchmark: publicDocumentation.benchmark.replace("benchmark/evidence/m2/",
    "benchmark/evidence/m2/ shipped artifact " +
    "941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c"),
}), /BENCHMARK\.md current M2 section retains a stale M1/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  benchmark: publicDocumentation.benchmark.replace("v1 and replacement-v2 are consumed and acceptance-ineligible",
    "v1 and replacement-v2 are pending"),
}), /BENCHMARK\.md historical M1 section changed/u);
assert.throws(() => validateM2PublicDocumentation({
  ...publicDocumentation,
  benchmark: `${publicDocumentation.benchmark}\n\n## Stale appendix\n` +
    "The finalized ONNX SHA-256 is `941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c`.",
}), /BENCHMARK\.md (historical M1 section changed|retains an unscoped stale M1)/u);
const staleMetadata = clone(publication);
staleMetadata.modelLock.trainingEvidence.selectionSummary = "benchmark/data/m2-head/selection-summary.json";
assert.throws(() => validateM2PublicationMetadata(staleMetadata), /model lock changed/u);
const missingComparison = clone(publication);
delete missingComparison.receipt.publishedEvidenceSha256["model-comparison.json"];
assert.throws(() => validateM2PublicationMetadata(missingComparison), /receipt changed/u);

const upstreamInitializers = Array.from({ length: 198 }, (_, index) => ({
  name: `encoder.${index}`,
  dimensions: [1],
  sha256: hash(`upstream-${index}`),
})).concat([
  { name: "classifier.bias", dimensions: [1], sha256: hash("js-upstream-bias") },
  { name: "classifier.weight", dimensions: [1, 384], sha256: hash("js-upstream-weight") },
]);
const graphDigests = {
  graphNodesSha256: hash("js-graph-nodes"),
  graphInputsSha256: hash("js-graph-inputs"),
  graphOutputsSha256: hash("js-graph-outputs"),
  opsetsSha256: hash("js-opsets"),
};
const upstreamStructure = { ...graphDigests, initializers: upstreamInitializers };
const shippedStructure = {
  ...graphDigests,
  initializers: upstreamInitializers.map((row) => ({
    ...row,
    sha256: row.name.startsWith("classifier.") ? hash(`js-candidate-${row.name}`) : row.sha256,
  })),
};
const crossDomainComparison = {
  graphNodesSha256: hash("python-graph-nodes"),
  graphInputsSha256: hash("python-graph-inputs"),
  graphOutputsSha256: hash("python-graph-outputs"),
  opsetsSha256: hash("python-opsets"),
  changedInitializers: [
    {
      name: "classifier.bias", dimensions: [1],
      beforeSha256: hash("python-upstream-bias"), afterSha256: hash("python-candidate-bias"),
    },
    {
      name: "classifier.weight", dimensions: [1, 384],
      beforeSha256: hash("python-upstream-weight"), afterSha256: hash("python-candidate-weight"),
    },
  ],
};
validateM2OnnxEvidence({ upstreamStructure, shippedStructure, comparison: crossDomainComparison });
const changedGraph = clone(shippedStructure);
changedGraph.graphNodesSha256 = hash("changed-js-graph");
assert.throws(() => validateM2OnnxEvidence({
  upstreamStructure, shippedStructure: changedGraph, comparison: crossDomainComparison,
}), /graphNodesSha256 changed/u);
const extraInitializer = clone(shippedStructure);
extraInitializer.initializers[0].sha256 = hash("changed-encoder");
assert.throws(() => validateM2OnnxEvidence({
  upstreamStructure, shippedStructure: extraInitializer, comparison: crossDomainComparison,
}), /outside the classifier head/u);
const wrongPythonShape = clone(crossDomainComparison);
wrongPythonShape.changedInitializers[0].dimensions = [2];
assert.throws(() => validateM2OnnxEvidence({
  upstreamStructure, shippedStructure, comparison: wrongPythonShape,
}), /comparison is malformed/u);

console.log(JSON.stringify({ cases: 31, policy: "pass" }));
