import { createHash } from "node:crypto";

export const M2 = Object.freeze({
  profile: "m2",
  identity: "prooflens-cf384-m2-head-v1",
  seed: 20_260_813,
  pipelineVersion: 9,
  upstreamSha256: "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1",
  trainerSha256: "b1948f678e32f30e36cf7543e402b8dd42d96ef73dad40232052301d3493c5b9",
  recipeSha256: "6c215ffbce297b5405b516629fc3fb16e2d7d6711163767c2e3517fdedbf98ca",
  selectionSummarySha256: "f619f334d3263c609262b90844d6dcefacbb4333a1b7c215b6d386d5f517bd0c",
  trainManifestSha256: "0d180f36fa8e0c2f099743a325b5f25fb4ad7aa0565223abdc89ddf47190710a",
  validationManifestSha256: "a63953148040e1a4223f16fa04ebf4b85c4022da65531ead0b25ce46434eab93",
  modelSha256: "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47",
  modelBytes: 87_442_080,
  trainingSummarySha256: "c3d49719e50b1fbf5fdc9ba5b8c1df57712910af0f0284a3c3acdf6bad931c04",
  calibrationSha256: "06d2452a8db9de26d42285cdc9dad0d233d397a6015583604c64480aec560e2c",
  candidateGridSha256: "7ac1028543607a94af88a95af585bdd973205849aa20bbf841f656598b4afe1c",
  modelComparisonSha256: "7e037912f28a69ac7ea9620471f1410b7b1ab445b7bb30ce9d7bdbe0c24f96ac",
  freshRunId: "add5d5306942c5c729c97556bd61cabd",
  trainImages: 105_978,
  trainViews: 123_912,
  validationImages: 900,
  validationViews: 3_600,
  trainShards: 53,
  totalShards: 54,
  candidateCount: 25,
  validCandidateCount: 24,
  displayThreshold: 0.65,
  maxParityError: 2e-4,
  selectedParameters: Object.freeze({ weightDecay: 0.1, upstreamBlendAlpha: 0.85 }),
  selectionKey: Object.freeze([
    0.9366666666666668,
    0.9445833333333333,
    0.9433333333333334,
    0.94,
    0.84,
  ]),
  thresholdLogit: -0.993612766265869,
});

export const M2_VARIANTS = Object.freeze(["original", "screenshot", "social-q75", "social-heavy"]);
export const M2_VALIDATION_SYNTHETIC_SOURCES = Object.freeze(["GLM-Image", "HunyuanImage-3.0"]);
export const M2_VALIDATION_REAL_SOURCES = Object.freeze(["open-images", "stockimages-cc0"]);
export const M2_SINGLE_VIEW_SOURCES = Object.freeze(["diffusiondb-stable-diffusion", "open-images-train"]);

export const M2_EXPECTED_SOURCES = Object.freeze({
  "FLUX.2-pro": 80,
  "FLUX.2_max": 80,
  "GPT-Image-1": 80,
  "GPT-Image-1.5": 80,
  "Imagen-4.0": 80,
  "Imagen-4.0-Ultra": 80,
  "Qwen-Image": 80,
  "Qwen-Image-2.0-pro": 80,
  "Qwen-Image-2512": 80,
  "Seedream-4.0": 80,
  "Seedream-4.5": 80,
  "Seedream-5.0": 80,
  "diffusiondb-stable-diffusion": 50_000,
  "docci-train": 1_200,
  "gpt-image-2": 80,
  "nano-banana-2.0": 80,
  "nano-banana-pro": 80,
  "open-images": 1_200,
  "open-images-train": 50_000,
  "stockimages-cc0": 2_378,
});

export const M2_EXPECTED_CLASS_COUNTS = Object.freeze({ real: 54_778, synthetic: 51_200 });

export const M2_EXPECTED_ARGUMENTS = Object.freeze([
  "--model", "benchmark/candidates/upstream-cf384.onnx",
  "--expected-model-sha256", M2.upstreamSha256,
  "--data-root", "benchmark/data/m2-head",
  "--train-manifest", "benchmark/data/m2-head/train-manifest.jsonl",
  "--validation-data-root", "benchmark/data/m2-head",
  "--validation-manifest", "benchmark/evidence/m2/validation-manifest.jsonl",
  "--recipe", "benchmark/m2/recipe.json",
  "--selection-summary", "benchmark/data/m2-head/selection-summary.json",
  "--single-view-source", "diffusiondb-stable-diffusion",
  "--single-view-source", "open-images-train",
  "--execution-provider", "cpu",
  "--batch-size", "24",
  "--feature-shard-images", "2000",
  "--reextract-cached-features",
  "--output-dir", "benchmark/candidates/prooflens-cf384-m2",
]);

export const M2_EXPECTED_PARAMETERS = Object.freeze(
  [0.1, 0.03, 0.01, 0.003, 0.001].flatMap((weightDecay) =>
    [0.4, 0.55, 0.7, 0.85, 1].map((upstreamBlendAlpha) => ({ weightDecay, upstreamBlendAlpha }))),
);

export const M2_EXPECTED_SHARDS = Object.freeze([
  ...Array.from({ length: M2.trainShards }, (_, index) =>
    `benchmark/candidates/prooflens-cf384-m2/features/train-${String(index).padStart(5, "0")}.npz`),
  "benchmark/candidates/prooflens-cf384-m2/features/validation-00000.npz",
]);

const HEX64 = /^[a-f0-9]{64}$/u;

export function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

export function jsonEqual(left, right) {
  const canonical = (value) => {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
    }
    return value;
  };
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

export function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function close(left, right, label, tolerance = 2e-12) {
  requireCondition(finiteNumber(left) && finiteNumber(right) && Math.abs(left - right) <= tolerance,
    `${label} changed`);
}

function exactKeys(value, expected, label) {
  requireCondition(value && typeof value === "object" && !Array.isArray(value) &&
    jsonEqual(Object.keys(value).sort(), [...expected].sort()), `${label} keys changed`);
}

export function compareSelectionKeys(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return 0;
}

export function requireVariantGates(variants, gates, label = "candidate") {
  exactKeys(variants, M2_VARIANTS, `${label} variants`);
  for (const variant of M2_VARIANTS) {
    const metrics = variants[variant];
    requireCondition(metrics && typeof metrics === "object", `${label} ${variant} metrics are malformed`);
    for (const [metric, gate] of [
      ["balancedAccuracy", "minimumBalancedAccuracyPerVariant"],
      ["realRecall", "minimumRealRecallPerVariant"],
      ["syntheticRecall", "minimumSyntheticRecallPerVariant"],
    ]) {
      requireCondition(finiteNumber(metrics[metric]) && finiteNumber(gates[gate]) && metrics[metric] >= gates[gate],
        `${label} ${variant} failed ${gate}`);
    }
    exactKeys(metrics.syntheticRecallBySource, M2_VALIDATION_SYNTHETIC_SOURCES,
      `${label} ${variant} synthetic sources`);
    for (const source of M2_VALIDATION_SYNTHETIC_SOURCES) {
      requireCondition(finiteNumber(metrics.syntheticRecallBySource[source]) &&
        metrics.syntheticRecallBySource[source] >= gates.minimumSyntheticRecallPerFamily,
      `${label} ${variant} failed ${source} synthetic recall`);
    }
    exactKeys(metrics.realRecallBySource, M2_VALIDATION_REAL_SOURCES, `${label} ${variant} real sources`);
    for (const [source, minimum] of Object.entries(gates.minimumRealRecallBySource ?? {})) {
      requireCondition(finiteNumber(metrics.realRecallBySource[source]) && finiteNumber(minimum) &&
        metrics.realRecallBySource[source] >= minimum,
      `${label} ${variant} failed ${source} real recall`);
    }
  }
}

export function validateM2TrainingPacket({
  summary,
  calibration,
  grid,
  recipe,
  selectionSummary,
  fileHashes,
  model,
}) {
  requireCondition(jsonEqual(fileHashes, {
    recipe: M2.recipeSha256,
    selectionSummary: M2.selectionSummarySha256,
    trainingSummary: M2.trainingSummarySha256,
    calibration: M2.calibrationSha256,
    candidateGrid: M2.candidateGridSha256,
  }), "M2 evidence byte hashes changed");
  requireCondition(model.sha256 === M2.modelSha256 && model.bytes === M2.modelBytes,
    "M2 shipped model bytes changed");
  requireCondition(recipe.schemaVersion === 2 && recipe.expectedTotalCount === M2.trainImages &&
    recipe.expectedValidationCount === M2.validationImages &&
    recipe.expectedTrainingFeatureViews === M2.trainViews &&
    recipe.expectedValidationFeatureViews === M2.validationViews &&
    jsonEqual(recipe.expectedSourceCounts, M2_EXPECTED_SOURCES) &&
    jsonEqual(recipe.expectedClassCounts, M2_EXPECTED_CLASS_COUNTS) &&
    jsonEqual(recipe.singleViewTrainingSources, M2_SINGLE_VIEW_SOURCES),
  "M2 recipe counts or source contract changed");
  requireCondition(selectionSummary.schemaVersion === 2 &&
    selectionSummary.recipeSha256 === M2.recipeSha256 &&
    selectionSummary.manifestSha256 === M2.trainManifestSha256 &&
    selectionSummary.validationManifestSha256 === M2.validationManifestSha256 &&
    jsonEqual(selectionSummary.sourceCounts, M2_EXPECTED_SOURCES) &&
    jsonEqual(selectionSummary.classCounts, M2_EXPECTED_CLASS_COUNTS),
  "M2 selection summary changed");

  requireCondition(summary.schemaVersion === 1 && summary.pipelineVersion === M2.pipelineVersion &&
    summary.seed === M2.seed && summary.trainerSha256 === M2.trainerSha256 &&
    summary.recipeSha256 === M2.recipeSha256 &&
    summary.selectionSummarySha256 === M2.selectionSummarySha256 &&
    summary.upstreamModelSha256 === M2.upstreamSha256 &&
    summary.trainManifestSha256 === M2.trainManifestSha256 &&
    summary.validationManifestSha256 === M2.validationManifestSha256 &&
    jsonEqual(summary.commandArguments, M2_EXPECTED_ARGUMENTS),
  "M2 training input bindings changed");
  requireCondition(summary.trainImages === M2.trainImages && summary.trainFeatureViews === M2.trainViews &&
    summary.validationImages === M2.validationImages && summary.validationFeatureViews === M2.validationViews &&
    summary.uniqueTrainingImagesCovered === M2.trainImages &&
    summary.uniqueTrainingFeatureViewsCovered === M2.trainViews && summary.viewsPerImage === 4,
  "M2 training coverage changed");
  requireCondition(summary.sourceBalancedSampling === false && summary.sourceBalancedLoss === true &&
    summary.trainingEpochs === 12 && summary.trainingBatchSize === 2_048 &&
    summary.featureBatchSize === 24 && summary.featureShardImages === 2_000 &&
    summary.cachedFeatureSourcePixelsReverified === true && summary.cachedFeatureArraysValidated === true &&
    summary.cachedFeatureValuesReextracted === true &&
    jsonEqual(summary.cachedFeatureDtypes, {
      features: "float32", labels: "float32", variants: "int64", sources: "unicode",
    }) && jsonEqual(summary.singleViewTrainingSources, M2_SINGLE_VIEW_SOURCES) &&
    jsonEqual(summary.trainingSourceCounts, M2_EXPECTED_SOURCES),
  "M2 training procedure changed");
  requireCondition(HEX64.test(summary.featureConfigurationHashes?.training ?? "") &&
    HEX64.test(summary.featureConfigurationHashes?.validation ?? "") &&
    summary.featureConfigurationHashes.training !== summary.featureConfigurationHashes.validation,
  "M2 feature-configuration hashes are missing or collapsed");
  requireCondition(jsonEqual(summary.validationGates, recipe.validationGates) &&
    summary.validationGatesPassed === true, "M2 validation gates changed or failed");

  const fresh = summary.freshFeatureRun;
  requireCondition(fresh?.schemaVersion === 1 && fresh.runId === M2.freshRunId && fresh.state === "complete" &&
    jsonEqual(fresh.context, {
      pipelineVersion: M2.pipelineVersion,
      featureExtractorContract: "cf384-static-batch24-preprocess-v1",
      upstreamModelSha256: M2.upstreamSha256,
      trainManifestSha256: M2.trainManifestSha256,
      validationManifestSha256: M2.validationManifestSha256,
      selectionSummarySha256: M2.selectionSummarySha256,
      featureBatchSize: 24,
      featureShardImages: 2_000,
      singleViewTrainingSources: [...M2_SINGLE_VIEW_SOURCES],
      featureConfigurationHashes: summary.featureConfigurationHashes,
    }) && summary.freshFeatureRunMarkerSha256 ===
      digest(Buffer.from(`${JSON.stringify(fresh, null, 2)}\n`)),
  "M2 fresh-feature run marker or context changed");
  const shards = summary.featureShardEvidence;
  requireCondition(Array.isArray(shards) && shards.length === M2.totalShards &&
    jsonEqual(shards.map((row) => row.cache), M2_EXPECTED_SHARDS), "M2 feature shard paths changed");
  const cacheHashes = new Set();
  const itemHashes = new Set();
  for (let index = 0; index < shards.length; index += 1) {
    const row = shards[index];
    const training = index < M2.trainShards;
    const expectedItems = training ? (index < 52 ? 2_000 : 1_978) : 900;
    requireCondition(row.items === expectedItems && Number.isInteger(row.views) &&
      row.views >= row.items && row.views <= row.items * 4 && row.freshFeatureRunId === M2.freshRunId &&
      row.freshlyExtractedThisRun === true && row.freshlyExtractedThisProcess === true &&
      row.replacedCacheSha256 === null && HEX64.test(row.cacheSha256 ?? "") &&
      HEX64.test(row.itemIdsSha256 ?? "") && row.featureConfigurationSha256 ===
        summary.featureConfigurationHashes[training ? "training" : "validation"] &&
      ["features", "labels", "variants", "sources"].every((name) => HEX64.test(row.arraySha256?.[name] ?? "")),
    "M2 feature shard evidence changed");
    cacheHashes.add(row.cacheSha256);
    itemHashes.add(row.itemIdsSha256);
  }
  requireCondition(cacheHashes.size === M2.totalShards && itemHashes.size === M2.totalShards &&
    shards.reduce((total, row) => total + row.items, 0) === M2.trainImages + M2.validationImages &&
    shards.reduce((total, row) => total + row.views, 0) === M2.trainViews + M2.validationViews,
  "M2 feature shards are duplicated or incomplete");
  requireCondition(summary.environment?.executionProvider === "cpu" &&
    jsonEqual(summary.environment.providers, ["CPUExecutionProvider"]) &&
    summary.environment.torchDeterministicAlgorithms === true && summary.environment.cuda === null &&
    summary.environment.gpu === null && ["numpy", "onnxRuntime", "pillow", "python", "platform", "torch"]
      .every((name) => typeof summary.environment[name] === "string" && summary.environment[name].length > 0),
  "M2 CPU environment evidence changed");

  requireCondition(Array.isArray(grid) && grid.length === M2.candidateCount &&
    summary.candidateCount === M2.candidateCount, "M2 candidate grid count changed");
  const expectedParameters = new Set(M2_EXPECTED_PARAMETERS.map((value) => JSON.stringify(value)));
  const seenParameters = new Set(grid.map((candidate) => JSON.stringify(candidate.parameters)));
  requireCondition(seenParameters.size === M2.candidateCount &&
    [...expectedParameters].every((value) => seenParameters.has(value)), "M2 candidate parameter coverage changed");
  const valid = grid.filter((candidate) => Array.isArray(candidate.selectionKey));
  const rejected = grid.filter((candidate) => !Array.isArray(candidate.selectionKey));
  requireCondition(valid.length === M2.validCandidateCount && summary.validCandidateCount === M2.validCandidateCount &&
    rejected.length === 1 && jsonEqual(rejected[0].parameters, { weightDecay: 0.1, upstreamBlendAlpha: 0.4 }) &&
    rejected[0].status === "rejected" && rejected[0].reason === "No threshold satisfies the frozen validation gates",
  "M2 valid/rejected candidate contract changed");
  for (const candidate of valid) {
    requireCondition(candidate.selectionKey.length === 5 && candidate.selectionKey.every(finiteNumber) &&
      finiteNumber(candidate.thresholdLogit), "M2 candidate selection evidence is malformed");
    requireVariantGates(candidate.variants, recipe.validationGates,
      `candidate ${JSON.stringify(candidate.parameters)}`);
  }
  const selected = valid.find((candidate) => jsonEqual(candidate.parameters, M2.selectedParameters));
  const best = valid.reduce((current, candidate) =>
    compareSelectionKeys(candidate.selectionKey, current.selectionKey) > 0 ? candidate : current);
  requireCondition(selected === best && jsonEqual(selected.selectionKey, M2.selectionKey) &&
    selected.thresholdLogit === M2.thresholdLogit && jsonEqual(selected.variants, summary.variants) &&
    jsonEqual(summary.selectedParameters, M2.selectedParameters) &&
    jsonEqual(summary.selectionKey, M2.selectionKey) && summary.thresholdLogit === M2.thresholdLogit,
  "M2 selected candidate is not the frozen deterministic maximum");
  requireVariantGates(summary.variants, recipe.validationGates, "selected M2 candidate");

  requireCondition(summary.model?.path === "benchmark/candidates/prooflens-cf384-m2/model.onnx" &&
    summary.model.sha256 === M2.modelSha256 && summary.model.bytes === M2.modelBytes &&
    finiteNumber(summary.model.maxAbsParityError) && summary.model.maxAbsParityError <= M2.maxParityError,
  "M2 model export evidence changed");
  requireCondition(calibration.schemaVersion === 1 &&
    calibration.method === "Validation-selected logit alignment to fixed 65/100 display threshold; not probability calibration" &&
    calibration.modelSha256 === M2.modelSha256 && calibration.trainManifestSha256 === M2.trainManifestSha256 &&
    calibration.validationManifestSha256 === M2.validationManifestSha256 &&
    calibration.selectionSummarySha256 === M2.selectionSummarySha256 && calibration.slope === 1 &&
    calibration.displayThreshold === M2.displayThreshold &&
    calibration.validationThresholdLogit === M2.thresholdLogit &&
    [calibration.intercept, calibration.rawProbabilityThreshold].every(finiteNumber) &&
    calibration.rawProbabilityThreshold > 0 && calibration.rawProbabilityThreshold < 1,
  "M2 calibration bindings changed");
  close(1 / (1 + Math.exp(-calibration.validationThresholdLogit)), calibration.rawProbabilityThreshold,
    "M2 raw calibration threshold");
  close(1 / (1 + Math.exp(-(calibration.validationThresholdLogit + calibration.intercept))),
    M2.displayThreshold, "M2 display calibration threshold");
  return { selected, freshRun: fresh };
}

export function validateM2PublicationMetadata({
  modelLock,
  weightsReadme,
  receipt,
  comparison,
  repositoryHashes,
}) {
  requireCondition(modelLock.schemaVersion === 2 && modelLock.artifact === "weights/prooflens-cf384.onnx" &&
    modelLock.bytes === M2.modelBytes && modelLock.sha256 === M2.modelSha256 &&
    modelLock.trainingRecipe === `${M2.identity}:${M2.recipeSha256}:${M2.selectionSummarySha256}` &&
    jsonEqual(modelLock.trainingEvidence, {
      recipe: "benchmark/m2/recipe.json",
      recipeSha256: M2.recipeSha256,
      selectionSummary: "benchmark/evidence/m2/selection-summary.json",
      selectionSummarySha256: M2.selectionSummarySha256,
      trainManifestSha256: M2.trainManifestSha256,
      trainingSummarySha256: M2.trainingSummarySha256,
      calibrationSha256: M2.calibrationSha256,
      candidateGridSha256: M2.candidateGridSha256,
    }) && modelLock.calibration?.displayThreshold === M2.displayThreshold &&
    modelLock.calibration.validationThresholdLogit === M2.thresholdLogit,
  "M2 model lock changed");
  requireCondition(weightsReadme.includes(M2.modelSha256) &&
    weightsReadme.includes(M2.modelBytes.toLocaleString("en-US")), "M2 weights README is stale");
  requireCondition(comparison.schemaVersion === 1 &&
    comparison.base?.path === "benchmark/candidates/upstream-cf384.onnx" &&
    comparison.base.sha256 === M2.upstreamSha256 && comparison.base.bytes === M2.modelBytes &&
    comparison.candidate?.path === "benchmark/candidates/prooflens-cf384-m2/model.onnx" &&
    comparison.candidate.sha256 === M2.modelSha256 && comparison.candidate.bytes === M2.modelBytes &&
    comparison.unchangedInitializerCount === 198 &&
    jsonEqual(comparison.changedInitializers.map((row) => row.name).sort(),
      ["classifier.bias", "classifier.weight"]),
  "M2 classifier comparison changed");
  requireCondition(receipt.schemaVersion === 3 && receipt.profile === M2.profile &&
    receipt.candidateDirectory === "benchmark/candidates/prooflens-cf384-m2" &&
    receipt.upstreamSha256 === M2.upstreamSha256 && receipt.shippedModel?.path === "weights/prooflens-cf384.onnx" &&
    receipt.shippedModel.sha256 === M2.modelSha256 && receipt.shippedModel.bytes === M2.modelBytes &&
    jsonEqual(receipt.sourceEvidenceSha256, {
      "training-summary.json": M2.trainingSummarySha256,
      "calibration.json": M2.calibrationSha256,
      "candidate-grid.json": M2.candidateGridSha256,
    }) && ["training-summary.json", "calibration.json", "candidate-grid.json"].every((name) =>
      receipt.publishedEvidenceSha256?.[name] === receipt.sourceEvidenceSha256[name]) &&
    receipt.publishedEvidenceSha256?.["model-comparison.json"] === M2.modelComparisonSha256 &&
    jsonEqual(receipt.publishedRepositorySha256, repositoryHashes),
  "M2 finalization receipt changed");
}

export function validateM2OnnxEvidence({ upstreamStructure, shippedStructure, comparison }) {
  for (const name of ["graphNodesSha256", "graphInputsSha256", "graphOutputsSha256", "opsetsSha256"]) {
    requireCondition(shippedStructure[name] === upstreamStructure[name],
      `M2 shipped ONNX ${name} changed`);
    requireCondition(HEX64.test(comparison[name] ?? ""),
      `M2 Python comparison ${name} is malformed`);
  }
  requireCondition(Array.isArray(upstreamStructure.initializers) &&
    Array.isArray(shippedStructure.initializers) && upstreamStructure.initializers.length === 200 &&
    shippedStructure.initializers.length === 200, "M2 ONNX initializer count changed");
  const upstreamByName = new Map(upstreamStructure.initializers.map((row) => [row.name, row]));
  requireCondition(upstreamByName.size === 200, "M2 upstream ONNX contains duplicate initializers");
  const independentlyChanged = [];
  for (const initializer of shippedStructure.initializers) {
    const upstream = upstreamByName.get(initializer.name);
    requireCondition(upstream && jsonEqual(initializer.dimensions, upstream.dimensions),
      `M2 initializer shape changed: ${initializer.name}`);
    if (initializer.sha256 !== upstream.sha256) independentlyChanged.push({
      name: initializer.name,
      dimensions: initializer.dimensions,
    });
  }
  requireCondition(jsonEqual(independentlyChanged.map((row) => row.name).sort(),
    ["classifier.bias", "classifier.weight"]), "M2 changed initializers outside the classifier head");
  requireCondition(Array.isArray(comparison.changedInitializers) && comparison.changedInitializers.length === 2,
    "M2 Python comparison changed-initializer count is malformed");
  const comparisonByName = new Map(comparison.changedInitializers.map((row) => [row.name, row]));
  requireCondition(comparisonByName.size === 2, "M2 Python comparison contains duplicate initializers");
  for (const row of independentlyChanged) {
    const recorded = comparisonByName.get(row.name);
    requireCondition(recorded && jsonEqual(recorded.dimensions, row.dimensions) &&
      HEX64.test(recorded.beforeSha256 ?? "") && HEX64.test(recorded.afterSha256 ?? "") &&
      recorded.beforeSha256 !== recorded.afterSha256,
    `M2 Python comparison is malformed: ${row.name}`);
  }
  // Python and onnxruntime-web use different protobuf serialization domains.
  // Each domain proves base-to-candidate equality independently; their digests
  // must not be compared byte-for-byte across implementations.
  return independentlyChanged;
}
