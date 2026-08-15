import { createHash } from "node:crypto";
import { TextDecoder } from "node:util";


export const M3 = Object.freeze({
  upstreamSha256: "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47",
  seed: 20260813,
  pipelineVersion: 9,
  trainImages: 108_378,
  trainViews: 133_512,
  selectorImages: 600,
  selectorViews: 2_400,
  regressionImages: 900,
  regressionViews: 3_600,
  trainShards: 55,
  totalShards: 57,
  m2ModelSha256: "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47",
  m2TrainImages: 105_978,
  m2TrainViews: 123_912,
});

export const M3_VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];
export const M3_SINGLE_VIEW_SOURCES = ["diffusiondb-stable-diffusion", "open-images-train"];
export const M3_EXPECTED_ARGUMENTS = [
  "--model", "weights/prooflens-cf384.onnx",
  "--expected-model-sha256", M3.upstreamSha256,
  "--data-root", "benchmark/data/m3-head",
  "--train-manifest", "benchmark/data/m3-head/train-manifest.jsonl",
  "--validation-data-root", "benchmark/data/m3-head",
  "--validation-manifest", "benchmark/evidence/m3/validation-manifest.jsonl",
  "--regression-data-root", "benchmark/data/m2-head",
  "--regression-manifest", "benchmark/evidence/m2/validation-manifest.jsonl",
  "--recipe", "benchmark/m3/recipe.json",
  "--selection-summary", "benchmark/evidence/m3/selection-summary.json",
  "--single-view-source", "diffusiondb-stable-diffusion",
  "--single-view-source", "open-images-train",
  "--execution-provider", "cpu",
  "--batch-size", "24",
  "--feature-shard-images", "2000",
  "--reextract-cached-features",
  "--output-dir", "benchmark/candidates/prooflens-cf384-m3",
];

const HEX64 = /^[a-f0-9]{64}$/;
const RUN_ID = /^[a-f0-9]{32}$/;


export function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function jsonEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

export function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function hasExactKeys(value, expected) {
  return value && typeof value === "object" && !Array.isArray(value) &&
    jsonEqual(Object.keys(value).sort(), [...expected].sort());
}

export function validateM3PublicationLockShape(lock) {
  requireCondition(hasExactKeys(lock, [
    "schemaVersion",
    "profile",
    "sourceCommit",
    "sourceTree",
    "upstreamModelSha256",
    "trainerSha256",
    "recipeSha256",
    "selectionSummarySha256",
    "candidateHashes",
    "candidateModelBytes",
    "modelComparisonSha256",
    "freshRunId",
    "finalizerSha256",
    "publicationContractSha256",
    "fixtureSelectorSha256",
    "documentationRendererSha256",
    "publicDocumentHashes",
    "fixtureManifestSha256",
    "candidateEvidenceJson",
    "classifierPatch",
    "publicationRows",
    "selectionInfluencedByRegression",
    "h3HoldoutScored",
  ]), "M3 publication lock top-level schema changed");
  requireCondition(hasExactKeys(lock.candidateHashes, [
    "training-summary.json", "calibration.json", "candidate-grid.json", "model.onnx",
  ]) && hasExactKeys(lock.publicDocumentHashes, ["README.md", "MODEL_CARD.md", "BENCHMARK.md"]) &&
    hasExactKeys(lock.candidateEvidenceJson, [
      "training-summary.json", "calibration.json", "candidate-grid.json",
      "model-comparison.json", "fixture-manifest.json",
    ]) && hasExactKeys(lock.classifierPatch, [
      "schemaVersion", "baseSha256", "candidateSha256", "candidateBytes", "replacements",
    ]) && Array.isArray(lock.classifierPatch.replacements) && lock.classifierPatch.replacements.length === 2 &&
    lock.classifierPatch.replacements.every((row) => hasExactKeys(row, [
      "name", "dimensions", "offset", "bytes", "beforeSha256", "afterSha256", "afterBase64",
    ])) && Array.isArray(lock.publicationRows) &&
    lock.publicationRows.every((row) => hasExactKeys(row, ["path", "status"])),
  "M3 publication lock nested schema changed");
}

export function parseCanonicalM3PublicationLock(value) {
  const raw = Buffer.isBuffer(value) ? Buffer.from(value) : Buffer.from(String(value), "utf8");
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  } catch (error) {
    throw new Error("M3 publication lock is not canonical UTF-8", { cause: error });
  }
  let lock;
  try {
    lock = JSON.parse(text);
  } catch (error) {
    throw new Error("M3 publication lock is not valid JSON", { cause: error });
  }
  validateM3PublicationLockShape(lock);
  requireCondition(raw.equals(Buffer.from(`${JSON.stringify(lock, null, 2)}\n`, "utf8")),
    "M3 publication lock bytes are not canonical JSON");
  return lock;
}

function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function sigmoid(value) {
  return value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
}

function expectedShards() {
  return [
    ...Array.from({ length: M3.trainShards }, (_, index) =>
      `benchmark/candidates/prooflens-cf384-m3/features/train-${String(index).padStart(5, "0")}.npz`),
    "benchmark/candidates/prooflens-cf384-m3/features/validation-00000.npz",
    "benchmark/candidates/prooflens-cf384-m3/features/regression-00000.npz",
  ];
}

function requireVariantGates(variants, gates, label, expectedSources) {
  requireCondition(variants && typeof variants === "object" && !Array.isArray(variants) &&
    jsonEqual(Object.keys(variants).sort(), [...M3_VARIANTS].sort()), `${label} variant set changed`);
  for (const variant of M3_VARIANTS) {
    const metrics = variants[variant];
    requireCondition(metrics && typeof metrics === "object" && !Array.isArray(metrics),
      `${label} ${variant} metrics are malformed`);
    for (const [metric, gate] of [
      ["balancedAccuracy", "minimumBalancedAccuracyPerVariant"],
      ["realRecall", "minimumRealRecallPerVariant"],
      ["syntheticRecall", "minimumSyntheticRecallPerVariant"],
    ]) {
      requireCondition(finite(metrics[metric]) && finite(gates[gate]) && metrics[metric] >= gates[gate],
        `${label} ${variant} failed ${gate}`);
    }
    const families = metrics.syntheticRecallBySource;
    requireCondition(families && typeof families === "object" && Object.keys(families).length > 0 &&
      jsonEqual(Object.keys(families).sort(), [...expectedSources.synthetic].sort()) &&
      Object.values(families).every((value) => finite(value) && value >= gates.minimumSyntheticRecallPerFamily),
    `${label} ${variant} failed the synthetic-family gate`);
    const realSources = metrics.realRecallBySource;
    requireCondition(realSources && typeof realSources === "object" &&
      jsonEqual(Object.keys(realSources).sort(), [...expectedSources.real].sort()),
    `${label} ${variant} real-source set changed`);
    for (const [source, minimum] of Object.entries(gates.minimumRealRecallBySource ?? {})) {
      requireCondition(finite(realSources[source]) && finite(minimum) && realSources[source] >= minimum,
        `${label} ${variant} failed the ${source} real-recall gate`);
    }
  }
}

function compareSelectionKeys(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return 0;
}

export function validateM3TrainingPacket({ summary, calibration, grid, recipe, selectionSummary, hashes, model }) {
  requireCondition(summary?.schemaVersion === 2 && summary.pipelineVersion === M3.pipelineVersion &&
    summary.seed === M3.seed && summary.upstreamModelSha256 === M3.upstreamSha256 &&
    summary.trainerSha256 === hashes.trainer && summary.recipeSha256 === hashes.recipe &&
    summary.selectionSummarySha256 === hashes.selectionSummary &&
    summary.trainManifestSha256 === selectionSummary.manifestSha256 &&
    summary.validationManifestSha256 === hashes.validationManifest &&
    summary.regressionManifestSha256 === hashes.regressionManifest &&
    jsonEqual(summary.commandArguments, M3_EXPECTED_ARGUMENTS), "M3 training input bindings changed");
  requireCondition(summary.trainImages === M3.trainImages && summary.trainFeatureViews === M3.trainViews &&
    summary.validationImages === M3.selectorImages && summary.validationFeatureViews === M3.selectorViews &&
    summary.regressionImages === M3.regressionImages && summary.regressionFeatureViews === M3.regressionViews &&
    summary.uniqueTrainingImagesCovered === M3.trainImages &&
    summary.uniqueTrainingFeatureViewsCovered === M3.trainViews, "M3 training coverage changed");
  requireCondition(summary.sourceBalancedSampling === false && summary.sourceBalancedLoss === true &&
    summary.trainingEpochs === 12 && summary.trainingBatchSize === 2_048 &&
    summary.featureBatchSize === 24 && summary.featureShardImages === 2_000 &&
    summary.cachedFeatureSourcePixelsReverified === true && summary.cachedFeatureArraysValidated === true &&
    summary.cachedFeatureValuesReextracted === true &&
    jsonEqual(summary.cachedFeatureDtypes, {
      features: "float32", labels: "float32", variants: "int64", sources: "unicode",
    }) && jsonEqual(summary.singleViewTrainingSources, M3_SINGLE_VIEW_SOURCES) &&
    jsonEqual(summary.trainingSourceCounts, recipe.expectedSourceCounts), "M3 training procedure changed");

  const configurations = summary.featureConfigurationHashes;
  requireCondition(configurations && jsonEqual(Object.keys(configurations).sort(), ["regression", "training", "validation"]) &&
    Object.values(configurations).every((value) => HEX64.test(value)) &&
    configurations.training !== configurations.validation && configurations.validation === configurations.regression,
  "M3 feature-configuration hashes changed");
  const fresh = summary.freshFeatureRun;
  requireCondition(fresh?.schemaVersion === 1 && RUN_ID.test(fresh.runId ?? "") && fresh.state === "complete" &&
    jsonEqual(fresh.context, {
      pipelineVersion: M3.pipelineVersion,
      featureExtractorContract: "cf384-static-batch24-preprocess-v1",
      upstreamModelSha256: M3.upstreamSha256,
      trainManifestSha256: summary.trainManifestSha256,
      validationManifestSha256: summary.validationManifestSha256,
      regressionManifestSha256: summary.regressionManifestSha256,
      regressionDataRoot: "benchmark/data/m2-head",
      regressionFeatureViews: M3.regressionViews,
      selectionSummarySha256: summary.selectionSummarySha256,
      featureBatchSize: 24,
      featureShardImages: 2_000,
      singleViewTrainingSources: M3_SINGLE_VIEW_SOURCES,
      featureConfigurationHashes: configurations,
    }) && summary.freshFeatureRunMarkerSha256 === digest(Buffer.from(`${JSON.stringify(fresh, null, 2)}\n`)),
  "M3 fresh-feature marker changed");
  const shards = summary.featureShardEvidence;
  requireCondition(Array.isArray(shards) && shards.length === M3.totalShards &&
    jsonEqual(shards.map((row) => row.cache), expectedShards()), "M3 shard paths changed");
  const cacheHashes = new Set();
  const itemHashes = new Set();
  for (let index = 0; index < shards.length; index += 1) {
    const row = shards[index];
    const partition = index < M3.trainShards ? "training" : (index === M3.trainShards ? "validation" : "regression");
    const expectedItems = index < 54 ? 2_000 : (index === 54 ? 378 : (index === 55 ? 600 : 900));
    requireCondition(row.items === expectedItems && Number.isInteger(row.views) && row.views >= row.items &&
      row.views <= row.items * 4 && row.freshFeatureRunId === fresh.runId &&
      row.freshlyExtractedThisRun === true && row.freshlyExtractedThisProcess === true &&
      row.replacedCacheSha256 === null && row.featureConfigurationSha256 === configurations[partition] &&
      HEX64.test(row.cacheSha256 ?? "") && HEX64.test(row.itemIdsSha256 ?? "") &&
      row.arraySha256 && ["features", "labels", "variants", "sources"].every((name) =>
        HEX64.test(row.arraySha256[name] ?? "")), "M3 shard evidence changed");
    cacheHashes.add(row.cacheSha256);
    itemHashes.add(row.itemIdsSha256);
  }
  requireCondition(cacheHashes.size === M3.totalShards && itemHashes.size === M3.totalShards &&
    shards.reduce((total, row) => total + row.items, 0) === M3.trainImages + M3.selectorImages + M3.regressionImages &&
    shards.reduce((total, row) => total + row.views, 0) === M3.trainViews + M3.selectorViews + M3.regressionViews,
  "M3 feature shards are duplicated or incomplete");
  requireCondition(summary.environment?.executionProvider === "cpu" &&
    jsonEqual(summary.environment.providers, ["CPUExecutionProvider"]) &&
    summary.environment.torchDeterministicAlgorithms === true && summary.environment.cuda === null &&
    summary.environment.gpu === null, "M3 execution environment changed");

  requireCondition(summary.selector?.manifestSha256 === hashes.validationManifest &&
    summary.selector.role === "fresh-m3-selection-validation" &&
    summary.selector.images === M3.selectorImages && summary.selector.featureViews === M3.selectorViews &&
    jsonEqual(summary.selector.gates, recipe.validationGates) && summary.selector.gatesPassed === true &&
    summary.selector.thresholdLogit === summary.thresholdLogit &&
    jsonEqual(summary.selector.variants, summary.variants), "M3 selector evidence changed");
  requireCondition(summary.regression?.manifestSha256 === hashes.regressionManifest &&
    summary.regression.dataRoot === "benchmark/data/m2-head" &&
    summary.regression.role === "consumed-m2-post-selection-regression" &&
    summary.regression.images === M3.regressionImages && summary.regression.featureViews === M3.regressionViews &&
    jsonEqual(summary.regression.gates, recipe.regressionGates) && summary.regression.gatesPassed === true &&
    summary.regression.thresholdLogitFromSelector === summary.thresholdLogit &&
    summary.regression.selectionInfluenced === false, "M3 regression evidence changed");
  const selectorSources = {
    synthetic: ["flux-1-dev-development"],
    real: ["met-open-access"],
  };
  const regressionSources = {
    synthetic: ["GLM-Image", "HunyuanImage-3.0"],
    real: ["open-images", "stockimages-cc0"],
  };
  requireVariantGates(summary.selector.variants, recipe.validationGates, "M3 selector", selectorSources);
  requireVariantGates(summary.regression.variants, recipe.regressionGates, "M2 regression", regressionSources);

  requireCondition(Array.isArray(grid) && grid.length === 25 && summary.candidateCount === 25 &&
    grid.every((row) => row && typeof row === "object" && !Array.isArray(row) && !("regression" in row)),
  "M3 candidate grid changed or leaks regression data");
  const parameters = new Set(grid.map((row) =>
    `${row.parameters?.weightDecay}:${row.parameters?.upstreamBlendAlpha}`));
  const expectedParameters = new Set([0.1, 0.03, 0.01, 0.003, 0.001].flatMap((decay) =>
    [0.4, 0.55, 0.7, 0.85, 1].map((alpha) => `${decay}:${alpha}`)));
  requireCondition(parameters.size === expectedParameters.size &&
    [...parameters].every((value) => expectedParameters.has(value)), "M3 grid parameter coverage changed");
  const valid = grid.filter((row) => Array.isArray(row.selectionKey));
  const rejected = grid.filter((row) => !Array.isArray(row.selectionKey));
  requireCondition(valid.length > 0 && valid.length === summary.validCandidateCount &&
    rejected.every((row) => row.status === "rejected" && typeof row.reason === "string" && row.reason.length > 0),
  "M3 grid candidate status changed");
  for (const row of valid) {
    requireCondition(row.selectionKey.length === 5 && row.selectionKey.every(finite),
      "M3 grid selection key changed");
    requireVariantGates(row.variants, recipe.validationGates, "M3 grid candidate", selectorSources);
  }
  const best = valid.reduce((left, right) => compareSelectionKeys(left.selectionKey, right.selectionKey) >= 0 ? left : right);
  requireCondition(jsonEqual(best.parameters, summary.selectedParameters) &&
    jsonEqual(best.selectionKey, summary.selectionKey) && best.thresholdLogit === summary.thresholdLogit &&
    jsonEqual(best.variants, summary.variants), "M3 selected candidate is not the deterministic selector maximum");

  requireCondition(model.sha256 === summary.model?.sha256 && model.bytes === summary.model?.bytes &&
    model.sha256 === hashes.model && finite(summary.model.maxAbsParityError) &&
    summary.model.maxAbsParityError <= 2e-4 &&
    jsonEqual(Object.keys(summary.model.maxAbsParityErrorByPartition ?? {}).sort(), ["regression", "selector"]) &&
    Object.values(summary.model.maxAbsParityErrorByPartition).every((value) => finite(value) && value <= 2e-4),
  "M3 model or export parity changed");
  requireCondition(calibration?.schemaVersion === 1 && calibration.modelSha256 === model.sha256 &&
    calibration.trainManifestSha256 === summary.trainManifestSha256 &&
    calibration.validationManifestSha256 === summary.validationManifestSha256 &&
    calibration.regressionManifestSha256 === summary.regressionManifestSha256 &&
    calibration.selectionSummarySha256 === summary.selectionSummarySha256 && calibration.slope === 1 &&
    calibration.displayThreshold === 0.65 && calibration.validationThresholdLogit === summary.thresholdLogit &&
    finite(calibration.intercept) && finite(calibration.rawProbabilityThreshold) &&
    calibration.rawProbabilityThreshold > 0 && calibration.rawProbabilityThreshold < 1 &&
    Math.abs(sigmoid(calibration.validationThresholdLogit) - calibration.rawProbabilityThreshold) <= 2e-12 &&
    Math.abs(sigmoid(calibration.validationThresholdLogit + calibration.intercept) - 0.65) <= 2e-12,
  "M3 calibration changed");
}

export function validateM3OnnxEvidence({ baseStructure, shippedStructure, comparison, model }) {
  for (const name of ["graphNodesSha256", "graphInputsSha256", "graphOutputsSha256", "opsetsSha256"]) {
    requireCondition(baseStructure[name] === shippedStructure[name], `M3 ONNX ${name} changed`);
  }
  requireCondition(Array.isArray(baseStructure.initializers) && Array.isArray(shippedStructure.initializers) &&
    baseStructure.initializers.length === shippedStructure.initializers.length, "M3 initializer inventory changed");
  const base = new Map(baseStructure.initializers.map((row) => [row.name, row]));
  const shipped = new Map(shippedStructure.initializers.map((row) => [row.name, row]));
  const changed = [];
  for (const [name, row] of base) {
    const current = shipped.get(name);
    requireCondition(current && jsonEqual(current.dimensions, row.dimensions), `M3 initializer ${name} changed shape`);
    if (current.sha256 !== row.sha256) changed.push(name);
  }
  requireCondition(jsonEqual(changed.sort(), ["classifier.bias", "classifier.weight"]),
    "M3 changed an initializer outside the classifier head");
  requireCondition(comparison?.schemaVersion === 1 && comparison.base?.sha256 === M3.upstreamSha256 &&
    comparison.base.path === "weights/prooflens-cf384.onnx" && comparison.candidate?.sha256 === model.sha256 &&
    comparison.candidate.bytes === model.bytes &&
    comparison.candidate.path === "benchmark/candidates/prooflens-cf384-m3/model.onnx" &&
    comparison.unchangedInitializerCount === base.size - 2 && Array.isArray(comparison.changedInitializers) &&
    jsonEqual(comparison.changedInitializers.map((row) => row.name).sort(), changed.sort()) &&
    comparison.changedInitializers.every((row) => HEX64.test(row.beforeSha256 ?? "") &&
      HEX64.test(row.afterSha256 ?? "") && row.beforeSha256 !== row.afterSha256 &&
      jsonEqual(row.dimensions, base.get(row.name)?.dimensions)), "M3 Python comparison evidence changed");
}

function boundedSection(document, start, end, label) {
  requireCondition(document.split(start).length === 2 && document.split(end).length === 2,
    `${label} markers must occur exactly once`);
  const startIndex = document.indexOf(start) + start.length;
  const endIndex = document.indexOf(end);
  requireCondition(startIndex < endIndex, `${label} markers are out of order`);
  return document.slice(startIndex, endIndex);
}

function requireTokens(value, tokens, label) {
  for (const token of tokens) requireCondition(value.includes(token), `${label} is missing ${token}`);
}

export function expectedM3CurrentSections({ summary, modelSha256 }) {
  const percentages = Object.fromEntries(M3_VARIANTS.map((variant) =>
    [variant, (summary.variants[variant].balancedAccuracy * 100).toFixed(2) + "%"]));
  const metricSentence = `Selector balanced accuracy was ${percentages.original} on originals, ` +
    `${percentages.screenshot} on screenshots, ${percentages["social-q75"]} on JPEG-75, and ` +
    `${percentages["social-heavy"]} on heavy double-JPEG.`;
  return {
    README: `## Current M3 model

The shipped local ONNX keeps the Community Forensics ViT-S/16 backbone frozen and changes only its 384-to-1 classifier weight and bias. Its SHA-256 is \`${modelSha256}\`.

The M3 head used 108,378 public training images and 133,512 feature views: 57,178 non-AI images and 51,200 synthetic images. It preserves the complete M2 corpus and adds 2,400 score-blind Met Open Access hard negatives.

The 600-image fresh Met/FLUX selector alone chose the candidate and raw threshold. ${metricSentence} The consumed 900-image M2 packet ran afterward as a post-selection regression gate and could only fail the run; it could not alter the candidate or threshold.

These are development-selection results, not an untouched generalization estimate and not an acceptance result. The reserved H3 packet remains unscored. The pixel-free training receipt, fresh-feature evidence, candidate grid, calibration, and classifier-only comparison are under \`benchmark/evidence/m3/\`.`,
    MODEL_CARD: `## Current M3 head-training data

The shipped artifact SHA-256 is \`${modelSha256}\`. Independent comparison permits changes only to \`classifier.weight [1,384]\` and \`classifier.bias [1]\`; the graph contract and every other initializer remain frozen.

The M3 head used 108,378 public training images and 133,512 feature views: 57,178 non-AI images and 51,200 synthetic images. The only addition to the complete M2 corpus is 2,400 score-blind Met Open Access public-domain cultural-heritage images. Source labels and rights statements are provenance evidence, not independent rights clearance.

The 600-image fresh selector contains 300 different Met images and 300 FLUX.1-dev images. It alone chose the candidate and threshold. ${metricSentence} The consumed 900-image M2 development packet was evaluated only as a post-selection regression gate and had no selection influence.

These figures are development-selection evidence, not an untouched generalization estimate and not an acceptance result. The reserved H3 packet remains unscored. The complete pixel-free contract is under \`benchmark/evidence/m3/\`.`,
    BENCHMARK: `## Current M3 model-selection boundary

M3 keeps the frozen ViT-S/16 backbone, preprocessing, fixed 65/100 display threshold, and 25-pair head grid. Training used 108,378 public images and 133,512 feature views: 57,178 non-AI images and 51,200 synthetic images. The only new training stratum is 2,400 Met Open Access hard negatives.

The shipped ONNX SHA-256 is \`${modelSha256}\`; only the classifier weight and bias may differ from M2.

The 600-image fresh Met/FLUX selector alone chose the candidate and threshold. ${metricSentence} The consumed 900-image M2 development packet then ran as a post-selection regression gate. Failure was terminal; regression metrics could not select another candidate or threshold.

These are development-selection results, not an untouched generalization estimate and not an acceptance result. The reserved H3 packet remains unscored. Reproducible evidence is under \`benchmark/evidence/m3/\`.`,
  };
}

export function validateM3PublicDocumentation({ readme, modelCard, benchmark, summary, modelSha256 }) {
  const expectedCurrent = expectedM3CurrentSections({ summary, modelSha256 });
  const documents = [
    ["README", readme],
    ["MODEL_CARD", modelCard],
    ["BENCHMARK", benchmark],
  ];
  for (const [label, document] of documents) {
    requireCondition(!document.includes("PROOFLENS_CURRENT_M2"), `${label} retains a current-M2 marker`);
    const current = boundedSection(document,
      "<!-- PROOFLENS_CURRENT_M3_START -->", "<!-- PROOFLENS_CURRENT_M3_END -->", `${label} current M3`);
    const historical = boundedSection(document,
      "<!-- PROOFLENS_HISTORICAL_M2_START -->", "<!-- PROOFLENS_HISTORICAL_M2_END -->", `${label} historical M2`);
    requireCondition(current.trim() === expectedCurrent[label], `${label} current M3 section is not canonical`);
    requireTokens(current, [
      modelSha256,
      "108,378",
      "133,512",
      "57,178",
      "51,200",
      "600-image",
      "post-selection regression",
      "benchmark/evidence/m3/",
      "not an untouched generalization estimate",
      "not an acceptance result",
    ], `${label} current M3 section`);
    for (const variant of M3_VARIANTS) {
      requireTokens(current, [(summary.variants[variant].balancedAccuracy * 100).toFixed(2) + "%"],
        `${label} current M3 metrics`);
    }
    requireTokens(historical, [M3.m2ModelSha256, "105,978", "123,912", "acceptanceEligible: false"],
      `${label} historical M2 section`);
    requireCondition(!current.includes(M3.m2ModelSha256), `${label} labels M2 as current M3`);
    requireCondition(document.split(M3.m2ModelSha256).length === 2,
      `${label} must contain the historical M2 SHA exactly once`);
    requireCondition(!/M3\s+IS\s+AN\s+ACCEPTANCE\s+RESULT/iu.test(document),
      `${label} contains a contradictory M3 acceptance claim`);
  }
}

export function validateM3BrowserFixtureManifest({ manifest, manifestSha256, assets, calibration, model, summarySha256 }) {
  requireCondition(manifest?.schemaVersion === 3 && manifest.modelSha256 === model.sha256 &&
    manifest.calibrationSha256 === calibration.sha256 && manifest.trainingSummarySha256 === summarySha256 &&
    manifest.inferenceProvider === "CPUExecutionProvider" && manifest.assetsUnchangedFromM2 === true &&
    Array.isArray(manifest.items) && manifest.items.length === 2 && HEX64.test(manifestSha256),
  "M3 browser fixture manifest changed");
  const byRole = new Map(manifest.items.map((row) => [row.role, row]));
  requireCondition(byRole.size === 2 && byRole.has("likely-ai") && byRole.has("below-threshold"),
    "M3 browser fixture roles changed");
  for (const [role, row] of byRole) {
    requireCondition(assets[row.asset] === row.assetSha256 && HEX64.test(row.assetSha256 ?? "") &&
      finite(row.referenceLogit) && finite(row.referenceRawProbability) && finite(row.referenceDisplayScore) &&
      Math.abs(sigmoid(row.referenceLogit) - row.referenceRawProbability) <= 2e-12 &&
      Math.abs(sigmoid(row.referenceLogit + calibration.value.intercept) - row.referenceDisplayScore) <= 2e-12,
    `M3 ${role} fixture evidence changed`);
  }
  requireCondition(byRole.get("likely-ai").referenceDisplayScore >= 0.8 &&
    byRole.get("below-threshold").referenceDisplayScore <= 0.45,
  "M3 browser fixtures no longer demonstrate both visible states");
}
