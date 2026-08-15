import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  M3,
  M3_EXPECTED_ARGUMENTS,
  M3_VARIANTS,
  digest,
  expectedM3CurrentSections,
  parseCanonicalM3PublicationLock,
  validateM3BrowserFixtureManifest,
  validateM3OnnxEvidence,
  validateM3PublicDocumentation,
  validateM3PublicationLockShape,
  validateM3TrainingPacket,
} from "./m3-training-contract.mjs";
import { M3_PUBLICATION_EXPECTED } from "./m3-stage-policy.mjs";
import { renderM3PublicDocuments } from "./render-m3-public-docs.mjs";
import {
  inspectM3ClassifierInitializers,
  reconstructM3CandidateModel,
} from "./m3-candidate-patch.mjs";


const recipe = JSON.parse(readFileSync("benchmark/m3/recipe.json", "utf8"));
const hex = (label) => digest(Buffer.from(label));
const clone = (value) => JSON.parse(JSON.stringify(value));
const metrics = (partition) => ({
  balancedAccuracy: 1,
  realRecall: 1,
  syntheticRecall: 1,
  syntheticRecallBySource: partition === "selector"
    ? { "flux-1-dev-development": 1 }
    : { "GLM-Image": 1, "HunyuanImage-3.0": 1 },
  realRecallBySource: partition === "selector"
    ? { "met-open-access": 1 }
    : { "open-images": 1, "stockimages-cc0": 1 },
});
const variants = (partition) => Object.fromEntries(M3_VARIANTS.map((variant) => [variant, metrics(partition)]));

function trainingFixture() {
  const hashes = {
    trainer: hex("trainer"),
    recipe: hex("recipe"),
    selectionSummary: hex("selection"),
    validationManifest: hex("validation"),
    regressionManifest: hex("regression"),
    model: hex("model"),
  };
  const configurations = {
    training: hex("training-config"),
    validation: hex("evaluation-config"),
    regression: hex("evaluation-config"),
  };
  const fresh = {
    schemaVersion: 1,
    runId: "a".repeat(32),
    state: "complete",
    context: {
      pipelineVersion: M3.pipelineVersion,
      featureExtractorContract: "cf384-static-batch24-preprocess-v1",
      upstreamModelSha256: M3.upstreamSha256,
      trainManifestSha256: hex("train-manifest"),
      validationManifestSha256: hashes.validationManifest,
      regressionManifestSha256: hashes.regressionManifest,
      regressionDataRoot: "benchmark/data/m2-head",
      regressionFeatureViews: M3.regressionViews,
      selectionSummarySha256: hashes.selectionSummary,
      featureBatchSize: 24,
      featureShardImages: 2_000,
      singleViewTrainingSources: ["diffusiondb-stable-diffusion", "open-images-train"],
      featureConfigurationHashes: configurations,
    },
  };
  const shards = [];
  let remainingExtraViews = M3.trainViews - M3.trainImages;
  for (let index = 0; index < M3.trainShards; index += 1) {
    const items = index < 54 ? 2_000 : 378;
    const extra = Math.min(remainingExtraViews, items * 3);
    remainingExtraViews -= extra;
    shards.push({
      cache: `benchmark/candidates/prooflens-cf384-m3/features/train-${String(index).padStart(5, "0")}.npz`,
      items,
      views: items + extra,
      freshFeatureRunId: fresh.runId,
      freshlyExtractedThisRun: true,
      freshlyExtractedThisProcess: true,
      replacedCacheSha256: null,
      featureConfigurationSha256: configurations.training,
      cacheSha256: hex(`cache-${index}`),
      itemIdsSha256: hex(`items-${index}`),
      arraySha256: Object.fromEntries(["features", "labels", "variants", "sources"].map((name) =>
        [name, hex(`${name}-${index}`)])),
    });
  }
  for (const [partition, items, views, config] of [
    ["validation", M3.selectorImages, M3.selectorViews, configurations.validation],
    ["regression", M3.regressionImages, M3.regressionViews, configurations.regression],
  ]) {
    const index = shards.length;
    shards.push({
      cache: `benchmark/candidates/prooflens-cf384-m3/features/${partition}-00000.npz`,
      items,
      views,
      freshFeatureRunId: fresh.runId,
      freshlyExtractedThisRun: true,
      freshlyExtractedThisProcess: true,
      replacedCacheSha256: null,
      featureConfigurationSha256: config,
      cacheSha256: hex(`cache-${index}`),
      itemIdsSha256: hex(`items-${index}`),
      arraySha256: Object.fromEntries(["features", "labels", "variants", "sources"].map((name) =>
        [name, hex(`${name}-${index}`)])),
    });
  }
  assert.equal(remainingExtraViews, 0);
  const grid = [0.1, 0.03, 0.01, 0.003, 0.001].flatMap((weightDecay, decayIndex) =>
    [0.4, 0.55, 0.7, 0.85, 1].map((upstreamBlendAlpha, alphaIndex) => {
      const rank = decayIndex * 5 + alphaIndex;
      return {
        parameters: { weightDecay, upstreamBlendAlpha },
        selectionKey: [rank, 1, 1, 1, 1],
        thresholdLogit: 0,
        variants: variants("selector"),
      };
    }));
  const selected = grid.at(-1);
  const summary = {
    schemaVersion: 2,
    seed: M3.seed,
    pipelineVersion: M3.pipelineVersion,
    trainerSha256: hashes.trainer,
    commandArguments: M3_EXPECTED_ARGUMENTS,
    recipeSha256: hashes.recipe,
    selectionSummarySha256: hashes.selectionSummary,
    upstreamModelSha256: M3.upstreamSha256,
    trainManifestSha256: fresh.context.trainManifestSha256,
    validationManifestSha256: hashes.validationManifest,
    regressionManifestSha256: hashes.regressionManifest,
    trainImages: M3.trainImages,
    trainFeatureViews: M3.trainViews,
    validationImages: M3.selectorImages,
    validationFeatureViews: M3.selectorViews,
    regressionImages: M3.regressionImages,
    regressionFeatureViews: M3.regressionViews,
    uniqueTrainingImagesCovered: M3.trainImages,
    uniqueTrainingFeatureViewsCovered: M3.trainViews,
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
    singleViewTrainingSources: ["diffusiondb-stable-diffusion", "open-images-train"],
    trainingSourceCounts: recipe.expectedSourceCounts,
    featureConfigurationHashes: configurations,
    freshFeatureRun: fresh,
    freshFeatureRunMarkerSha256: digest(Buffer.from(`${JSON.stringify(fresh, null, 2)}\n`)),
    featureShardEvidence: shards,
    selector: {
      manifestSha256: hashes.validationManifest,
      role: "fresh-m3-selection-validation",
      images: M3.selectorImages,
      featureViews: M3.selectorViews,
      gates: recipe.validationGates,
      gatesPassed: true,
      thresholdLogit: 0,
      variants: selected.variants,
    },
    regression: {
      manifestSha256: hashes.regressionManifest,
      dataRoot: "benchmark/data/m2-head",
      role: "consumed-m2-post-selection-regression",
      images: M3.regressionImages,
      featureViews: M3.regressionViews,
      gates: recipe.regressionGates,
      gatesPassed: true,
      thresholdLogitFromSelector: 0,
      variants: variants("regression"),
      selectionInfluenced: false,
    },
    selectedParameters: selected.parameters,
    selectionKey: selected.selectionKey,
    thresholdLogit: 0,
    variants: selected.variants,
    candidateCount: 25,
    validCandidateCount: 25,
    environment: {
      executionProvider: "cpu",
      providers: ["CPUExecutionProvider"],
      torchDeterministicAlgorithms: true,
      cuda: null,
      gpu: null,
    },
    model: {
      sha256: hashes.model,
      bytes: 87_442_080,
      maxAbsParityError: 0.00001,
      maxAbsParityErrorByPartition: { selector: 0.00001, regression: 0.00001 },
    },
  };
  const intercept = Math.log(0.65 / 0.35);
  const calibration = {
    schemaVersion: 1,
    slope: 1,
    intercept,
    validationThresholdLogit: 0,
    rawProbabilityThreshold: 0.5,
    displayThreshold: 0.65,
    modelSha256: hashes.model,
    trainManifestSha256: summary.trainManifestSha256,
    validationManifestSha256: hashes.validationManifest,
    regressionManifestSha256: hashes.regressionManifest,
    selectionSummarySha256: hashes.selectionSummary,
  };
  return {
    summary,
    calibration,
    grid,
    recipe,
    selectionSummary: { manifestSha256: summary.trainManifestSha256 },
    hashes,
    model: { sha256: hashes.model, bytes: 87_442_080 },
  };
}

const valid = trainingFixture();
validateM3TrainingPacket(valid);
let cases = 1;
for (const mutate of [
  (value) => { value.summary.regression.selectionInfluenced = true; },
  (value) => { value.grid[0].regression = { leaked: true }; },
  (value) => { value.summary.featureShardEvidence[0].freshlyExtractedThisProcess = false; },
  (value) => { value.summary.selector.variants.original.realRecallBySource["met-open-access"] = 0.9299; },
  (value) => { value.summary.regression.variants.original.syntheticRecallBySource = { "wrong-family": 1 }; },
  (value) => { value.summary.selectedParameters = value.grid[0].parameters; },
  (value) => { value.calibration.intercept += 0.01; },
]) {
  const changed = clone(valid);
  mutate(changed);
  assert.throws(() => validateM3TrainingPacket(changed));
  cases += 1;
}

const comparison = {
  schemaVersion: 1,
  base: { path: "weights/prooflens-cf384.onnx", sha256: M3.upstreamSha256 },
  candidate: { path: "benchmark/candidates/prooflens-cf384-m3/model.onnx", sha256: hex("new-model"), bytes: 100 },
  unchangedInitializerCount: 1,
  changedInitializers: [
    { name: "classifier.weight", dimensions: [1, 384], beforeSha256: hex("bw"), afterSha256: hex("aw") },
    { name: "classifier.bias", dimensions: [1], beforeSha256: hex("bb"), afterSha256: hex("ab") },
  ],
};
const baseStructure = {
  graphNodesSha256: hex("nodes"), graphInputsSha256: hex("inputs"),
  graphOutputsSha256: hex("outputs"), opsetsSha256: hex("opsets"),
  initializers: [
    { name: "classifier.bias", dimensions: [1], sha256: hex("bias-before") },
    { name: "classifier.weight", dimensions: [1, 384], sha256: hex("weight-before") },
    { name: "other", dimensions: [1], sha256: hex("other") },
  ],
};
const shippedStructure = clone(baseStructure);
shippedStructure.initializers[0].sha256 = hex("bias-after");
shippedStructure.initializers[1].sha256 = hex("weight-after");
validateM3OnnxEvidence({
  baseStructure,
  shippedStructure,
  comparison,
  model: { sha256: hex("new-model"), bytes: 100 },
});
cases += 1;
const graphChanged = clone(shippedStructure);
graphChanged.graphNodesSha256 = hex("changed-nodes");
assert.throws(() => validateM3OnnxEvidence({
  baseStructure,
  shippedStructure: graphChanged,
  comparison,
  model: { sha256: hex("new-model"), bytes: 100 },
}), /graphNodesSha256/u);
cases += 1;

const expectedSections = expectedM3CurrentSections({ summary: valid.summary, modelSha256: valid.model.sha256 });
const documentFor = (label) => [
  "<!-- PROOFLENS_CURRENT_M3_START -->",
  expectedSections[label],
  "<!-- PROOFLENS_CURRENT_M3_END -->",
  "<!-- PROOFLENS_HISTORICAL_M2_START -->",
  `${M3.m2ModelSha256} 105,978 123,912 acceptanceEligible: false`,
  "<!-- PROOFLENS_HISTORICAL_M2_END -->",
].join("\n");
const documents = {
  readme: documentFor("README"),
  modelCard: documentFor("MODEL_CARD"),
  benchmark: documentFor("BENCHMARK"),
};
validateM3PublicDocumentation({
  ...documents,
  summary: valid.summary, modelSha256: valid.model.sha256,
});
cases += 1;
assert.throws(() => validateM3PublicDocumentation({
  ...documents,
  readme: documents.readme.replace("600-image", "600 item"),
  summary: valid.summary, modelSha256: valid.model.sha256,
}), /canonical/u);
cases += 1;
assert.throws(() => validateM3PublicDocumentation({
  ...documents,
  readme: documents.readme.replace("The shipped local ONNX", "M3 IS AN ACCEPTANCE RESULT. The shipped local ONNX"),
  summary: valid.summary, modelSha256: valid.model.sha256,
}), /canonical|contradictory/u);
cases += 1;

const rendered = renderM3PublicDocuments({
  readme: readFileSync("README.md", "utf8"),
  modelCard: readFileSync("MODEL_CARD.md", "utf8"),
  benchmark: readFileSync("BENCHMARK.md", "utf8"),
  summary: valid.summary,
  modelSha256: valid.model.sha256,
  modelBytes: valid.model.bytes,
});
validateM3PublicDocumentation({
  readme: rendered.README,
  modelCard: rendered.MODEL_CARD,
  benchmark: rendered.BENCHMARK,
  summary: valid.summary,
  modelSha256: valid.model.sha256,
});
const renderedHashes = Object.fromEntries(Object.entries(rendered).map(([name, value]) =>
  [`${name}.md`, digest(Buffer.from(value))]));
assert.deepEqual(Object.keys(renderedHashes).sort(), ["BENCHMARK.md", "MODEL_CARD.md", "README.md"]);
const contradictedReadme = `${rendered.README}\nM3 has passed acceptance and is approved for production.\n`;
assert.notEqual(digest(Buffer.from(contradictedReadme)), renderedHashes["README.md"]);
cases += 1;

const patchBase = readFileSync("weights/prooflens-cf384.onnx");
const patchCandidate = Buffer.from(patchBase);
const replacements = inspectM3ClassifierInitializers(patchBase).map((row) => {
  const after = Buffer.from(row.rawData);
  after[0] ^= 1;
  after.copy(patchCandidate, row.offset);
  return {
    name: row.name,
    dimensions: row.dimensions,
    offset: row.offset,
    bytes: row.bytes,
    beforeSha256: row.sha256,
    afterSha256: digest(after),
    afterBase64: after.toString("base64"),
  };
});
const classifierPatch = {
  schemaVersion: 1,
  baseSha256: digest(patchBase),
  candidateSha256: digest(patchCandidate),
  candidateBytes: patchCandidate.length,
  replacements,
};
assert.equal(reconstructM3CandidateModel({ baseBytes: patchBase, patch: classifierPatch }).equals(patchCandidate), true);
cases += 1;
const wrongCandidateDigest = clone(classifierPatch);
wrongCandidateDigest.candidateSha256 = "f".repeat(64);
assert.throws(() => reconstructM3CandidateModel({ baseBytes: patchBase, patch: wrongCandidateDigest }),
  /does not reconstruct/u);
cases += 1;
const publicationLockShape = {
  schemaVersion: 1,
  profile: "m3",
  sourceCommit: "a".repeat(40),
  sourceTree: hex("source-tree"),
  upstreamModelSha256: M3.upstreamSha256,
  trainerSha256: hex("trainer"),
  recipeSha256: hex("recipe"),
  selectionSummarySha256: hex("selection"),
  candidateHashes: {
    "training-summary.json": hex("summary"),
    "calibration.json": hex("calibration"),
    "candidate-grid.json": hex("grid"),
    "model.onnx": classifierPatch.candidateSha256,
  },
  candidateModelBytes: classifierPatch.candidateBytes,
  modelComparisonSha256: hex("comparison"),
  freshRunId: "a".repeat(32),
  finalizerSha256: hex("finalizer"),
  publicationContractSha256: hex("publication-contract"),
  fixtureSelectorSha256: hex("fixture-selector"),
  documentationRendererSha256: hex("renderer"),
  publicDocumentHashes: {
    "README.md": hex("readme"),
    "MODEL_CARD.md": hex("model-card"),
    "BENCHMARK.md": hex("benchmark"),
  },
  fixtureManifestSha256: hex("fixture"),
  candidateEvidenceJson: {
    "training-summary.json": "{}\n",
    "calibration.json": "{}\n",
    "candidate-grid.json": "[]\n",
    "model-comparison.json": "{}\n",
    "fixture-manifest.json": "{}\n",
  },
  classifierPatch,
  publicationRows: [...M3_PUBLICATION_EXPECTED].map(([path, status]) => ({ path, status })),
  selectionInfluencedByRegression: false,
  h3HoldoutScored: false,
};
validateM3PublicationLockShape(publicationLockShape);
cases += 1;
const canonicalLock = `${JSON.stringify(publicationLockShape, null, 2)}\n`;
assert.deepEqual(parseCanonicalM3PublicationLock(canonicalLock), publicationLockShape);
cases += 1;
const duplicateLockKey = canonicalLock.replace(
  '  "h3HoldoutScored": false',
  '  "h3HoldoutScored": true,\n  "h3HoldoutScored": false',
);
assert.throws(() => parseCanonicalM3PublicationLock(duplicateLockKey), /canonical JSON/u);
cases += 1;
const replacementCharacterLock = clone(publicationLockShape);
replacementCharacterLock.profile = "m3�";
const malformedUtf8 = Buffer.from(`${JSON.stringify(replacementCharacterLock, null, 2)}\n`, "utf8");
const replacementOffset = malformedUtf8.indexOf(Buffer.from("�", "utf8"));
assert.notEqual(replacementOffset, -1);
malformedUtf8[replacementOffset] = 0xff;
assert.throws(() => parseCanonicalM3PublicationLock(malformedUtf8), /canonical UTF-8/u);
cases += 1;
const authorityClaim = clone(publicationLockShape);
authorityClaim.acceptanceEligible = true;
assert.throws(() => validateM3PublicationLockShape(authorityClaim), /top-level schema/u);
cases += 1;
const nestedAuthorityClaim = clone(publicationLockShape);
nestedAuthorityClaim.classifierPatch.replacements[0].approved = true;
assert.throws(() => validateM3PublicationLockShape(nestedAuthorityClaim), /nested schema/u);
cases += 1;

const fixtureCalibration = { sha256: hex("calibration"), value: { intercept: Math.log(0.65 / 0.35) } };
const fixture = {
  schemaVersion: 3,
  modelSha256: valid.model.sha256,
  calibrationSha256: fixtureCalibration.sha256,
  trainingSummarySha256: hex("summary"),
  inferenceProvider: "CPUExecutionProvider",
  assetsUnchangedFromM2: true,
  items: [
    {
      role: "likely-ai", asset: "likely-ai.png", assetSha256: hex("likely"), referenceLogit: 4,
      referenceRawProbability: 1 / (1 + Math.exp(-4)),
      referenceDisplayScore: 1 / (1 + Math.exp(-(4 + fixtureCalibration.value.intercept))),
    },
    {
      role: "below-threshold", asset: "below-threshold.jpg", assetSha256: hex("below"), referenceLogit: -4,
      referenceRawProbability: 1 / (1 + Math.exp(4)),
      referenceDisplayScore: 1 / (1 + Math.exp(-(-4 + fixtureCalibration.value.intercept))),
    },
  ],
};
validateM3BrowserFixtureManifest({
  manifest: fixture,
  manifestSha256: hex("fixture"),
  assets: { "likely-ai.png": hex("likely"), "below-threshold.jpg": hex("below") },
  calibration: fixtureCalibration,
  model: valid.model,
  summarySha256: hex("summary"),
});
cases += 1;
const badFixture = clone(fixture);
badFixture.items[0].referenceRawProbability += 0.01;
assert.throws(() => validateM3BrowserFixtureManifest({
  manifest: badFixture,
  manifestSha256: hex("fixture"),
  assets: { "likely-ai.png": hex("likely"), "below-threshold.jpg": hex("below") },
  calibration: fixtureCalibration,
  model: valid.model,
  summarySha256: hex("summary"),
}));
cases += 1;

console.log(JSON.stringify({ cases, policy: "pass" }));
