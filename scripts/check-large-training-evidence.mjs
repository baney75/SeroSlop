import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { posix as path } from "node:path";
import { gunzipSync } from "node:zlib";
import { readFile } from "node:fs/promises";
import { inspectOnnxStructure } from "./onnx-structure.mjs";

const ROOT = "benchmark/evidence/large";
const HEX64 = /^[a-f0-9]{64}$/u;
const DHASH64 = /^[a-f0-9]{16}$/u;
const VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];
const VALIDATION_SYNTHETIC_SOURCES = ["GLM-Image", "HunyuanImage-3.0"];
const UPSTREAM_MODEL_SHA256 = "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1";
const M1_TRAINER_SHA256 = "6a4199d2be6fead9a10c483ce4c9648951d87ae18721499cc42a98db5f05bf56";
const M1_SOURCE_COMMIT = "1323c10a151bdd0b96640962b447607371607b90";
const EXPECTED_TRAINING_ARGUMENTS = [
  "--model", "benchmark/candidates/upstream-cf384.onnx",
  "--expected-model-sha256", UPSTREAM_MODEL_SHA256,
  "--data-root", "benchmark/data/large-head",
  "--train-manifest", "benchmark/data/large-head/train-manifest.jsonl",
  "--validation-data-root", "benchmark/data/modern-head",
  "--validation-manifest", "benchmark/manifests/validation.jsonl",
  "--recipe", "benchmark/large/recipe.json",
  "--selection-summary", "benchmark/data/large-head/selection-summary.json",
  "--single-view-source", "diffusiondb-stable-diffusion",
  "--single-view-source", "open-images-train",
  "--execution-provider", "cpu",
  "--batch-size", "24",
  "--feature-shard-images", "2000",
  "--reextract-cached-features",
  "--output-dir", "benchmark/candidates/prooflens-cf384-large",
];
const EXPECTED_SOURCES = {
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
};
const EXPECTED_EXCLUSIONS = [
  {
    path: "benchmark/manifests/validation.jsonl",
    rows: 600,
    dataRoot: "benchmark/data/modern-head",
    role: "validation-or-confirmatory-test",
  },
  {
    path: "benchmark/manifests/test.jsonl",
    rows: 600,
    dataRoot: "benchmark/data",
    role: "validation-or-confirmatory-test",
  },
  {
    path: "benchmark/manifests/web-negative-chartography.jsonl",
    rows: 19,
    dataRoot: "benchmark/data/web-negative",
    role: "web-negative-training-exclusion",
  },
];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function committedBytes(pathname) {
  return execFileSync("git", ["show", `${M1_SOURCE_COMMIT}:${pathname}`], {
    encoding: null,
    maxBuffer: 128 * 1024 * 1024,
  });
}

function parseJson(bytes, file) {
  try {
    return JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error(`${file} is not valid JSON`, { cause: error });
  }
}

function parseJsonLines(bytes, file) {
  try {
    return bytes.toString("utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
  } catch (error) {
    throw new Error(`${file} is not valid JSONL`, { cause: error });
  }
}

async function readGzip(pathname) {
  return gunzipSync(await readFile(pathname));
}

function jsonEqual(left, right) {
  const canonical = (value) => {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
    }
    return value;
  };
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function close(actual, expected, label, tolerance = 1e-12) {
  requireCondition(finiteNumber(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, received ${actual}`);
}

const POPCOUNT_16 = new Uint8Array(65_536);
for (let value = 1; value < POPCOUNT_16.length; value += 1) {
  POPCOUNT_16[value] = POPCOUNT_16[value >> 1] + (value & 1);
}

function blocks(dhash) {
  return [0, 4, 8, 12].map((offset) => Number.parseInt(dhash.slice(offset, offset + 4), 16));
}

function hamming64(left, right) {
  const a = blocks(left);
  const b = blocks(right);
  return a.reduce((total, value, index) => total + POPCOUNT_16[value ^ b[index]], 0);
}

const neighborCache = new Map();
function neighborsWithinTwo(value) {
  const cached = neighborCache.get(value);
  if (cached) return cached;
  const values = [value];
  for (let first = 0; first < 16; first += 1) {
    values.push(value ^ (1 << first));
    for (let second = first + 1; second < 16; second += 1) {
      values.push(value ^ (1 << first) ^ (1 << second));
    }
  }
  neighborCache.set(value, values);
  return values;
}

function findNearPairs(leftRows, rightRows, threshold) {
  const indexes = Array.from({ length: 4 }, () => new Map());
  for (let index = 0; index < rightRows.length; index += 1) {
    const values = blocks(rightRows[index].perceptualDhash64);
    for (let blockIndex = 0; blockIndex < 4; blockIndex += 1) {
      const bucket = indexes[blockIndex].get(values[blockIndex]) ?? [];
      bucket.push(index);
      indexes[blockIndex].set(values[blockIndex], bucket);
    }
  }
  const pairs = [];
  for (const left of leftRows) {
    const candidates = new Set();
    for (const [blockIndex, value] of blocks(left.perceptualDhash64).entries()) {
      for (const neighbor of neighborsWithinTwo(value)) {
        for (const index of indexes[blockIndex].get(neighbor) ?? []) candidates.add(index);
      }
    }
    for (const index of candidates) {
      const right = rightRows[index];
      const distance = hamming64(left.perceptualDhash64, right.perceptualDhash64);
      if (distance <= threshold) pairs.push({ left, right, distance });
    }
  }
  return pairs;
}

function requireVariantGates(variants, gates, label) {
  requireCondition(jsonEqual(Object.keys(variants).sort(), [...VARIANTS].sort()), `${label} variants are incomplete`);
  for (const variant of VARIANTS) {
    const row = variants[variant];
    requireCondition(finiteNumber(row?.balancedAccuracy) && row.balancedAccuracy >= gates.minimumBalancedAccuracyPerVariant,
      `${label} ${variant} balanced accuracy failed its frozen gate`);
    requireCondition(finiteNumber(row.realRecall) && row.realRecall >= gates.minimumRealRecallPerVariant,
      `${label} ${variant} real recall failed its frozen gate`);
    requireCondition(finiteNumber(row.syntheticRecall) && row.syntheticRecall >= gates.minimumSyntheticRecallPerVariant,
      `${label} ${variant} synthetic recall failed its frozen gate`);
    requireCondition(jsonEqual(Object.keys(row.syntheticRecallBySource ?? {}).sort(), VALIDATION_SYNTHETIC_SOURCES),
      `${label} ${variant} validation-family coverage changed`);
    const familyRecalls = Object.values(row.syntheticRecallBySource ?? {});
    requireCondition(familyRecalls.length > 0 && familyRecalls.every((recall) =>
      finiteNumber(recall) && recall >= gates.minimumSyntheticRecallPerFamily),
    `${label} ${variant} family recall failed its frozen gate`);
  }
}

const recipeBytes = await readFile("benchmark/large/recipe.json");
const recipe = parseJson(recipeBytes, "benchmark/large/recipe.json");
const summaryBytes = await readFile(`${ROOT}/selection-summary.json`);
const selectionSummary = parseJson(summaryBytes, `${ROOT}/selection-summary.json`);
requireCondition(selectionSummary.schemaVersion === 1, "Unsupported large-corpus selection summary");
requireCondition(selectionSummary.recipeSha256 === digest(recipeBytes), "Selection summary recipe hash mismatch");

const planBytes = await readGzip(`${ROOT}/selection-plan.json.gz`);
requireCondition(digest(planBytes) === selectionSummary.planSha256, "Expanded selection plan hash mismatch");
const plan = parseJson(planBytes, `${ROOT}/selection-plan.json.gz`);
requireCondition(plan.schemaVersion === 1 && plan.recipeSha256 === selectionSummary.recipeSha256,
  "Selection plan does not bind the recipe");
requireCondition(plan.diffusionDb.archives.length === 85 && plan.diffusionDb.candidates.length === 58_247,
  "Selection plan does not contain the frozen DiffusionDB envelope");
requireCondition(plan.openImages.candidates.length === 55_000,
  "Selection plan does not contain the frozen Open Images reserve");
const partIds = plan.diffusionDb.archives.map((row) => row.partId);
requireCondition(new Set(partIds).size === 85 && partIds.every((partId, index) => index === 0 || partId > partIds[index - 1]),
  "DiffusionDB archive locks are not sorted and unique");
requireCondition(plan.diffusionDb.archives.every((row) => Number.isInteger(row.bytes) && row.bytes > 0 &&
  HEX64.test(row.sha256) && row.path === `images/part-${String(row.partId).padStart(6, "0")}.zip`),
"DiffusionDB archive lock is malformed");
requireCondition(plan.diffusionDb.archives.reduce((total, row) => total + row.bytes, 0) === 53_761_314_024,
  "DiffusionDB archive byte total changed");
requireCondition(new Set(plan.diffusionDb.candidates.map((row) => row.id)).size === 58_247 &&
  new Set(plan.openImages.candidates.map((row) => row.id)).size === 55_000,
"Selection plan contains duplicate candidates");
requireCondition(recipe.diffusionDb.dataset === "poloclub/diffusiondb" &&
  recipe.diffusionDb.revision === "fb620fbe49fa4420e0734bd9c0df11f51176b61f" &&
  recipe.diffusionDb.license === "CC0-1.0", "DiffusionDB source identity changed");
requireCondition(recipe.openImages.dataset === "Open Images V7" &&
  recipe.openImages.revision === "v7-train-boxable-2018-04" &&
  recipe.openImages.license === "CC-BY-2.0", "Open Images source identity changed");

const trainBytes = await readGzip(`${ROOT}/train-manifest.jsonl.gz`);
requireCondition(digest(trainBytes) === selectionSummary.manifestSha256, "Expanded training manifest hash mismatch");
const trainingRows = parseJsonLines(trainBytes, `${ROOT}/train-manifest.jsonl.gz`);
requireCondition(trainingRows.length === 103_600, `Expected 103600 training rows, received ${trainingRows.length}`);
const trainIds = new Set();
const trainHashes = new Set();
const sourceCounts = {};
const classCounts = { real: 0, synthetic: 0 };
for (const [index, row] of trainingRows.entries()) {
  requireCondition(row.rowIndex === index && row.split === "train", `Training row ${index} has invalid split/index`);
  requireCondition(typeof row.id === "string" && !trainIds.has(row.id), `Duplicate training ID: ${row.id}`);
  requireCondition(HEX64.test(row.imageSha256) && !trainHashes.has(row.imageSha256),
    `Duplicate or invalid training image hash: ${row.id}`);
  requireCondition(DHASH64.test(row.perceptualDhash64), `Invalid training dHash: ${row.id}`);
  requireCondition(typeof row.path === "string" && row.path === path.normalize(row.path) &&
    !path.isAbsolute(row.path) && row.path !== ".." && !row.path.startsWith("../"), `Unsafe training path: ${row.id}`);
  requireCondition(row.label === 0 || row.label === 1, `Invalid label: ${row.id}`);
  trainIds.add(row.id);
  trainHashes.add(row.imageSha256);
  sourceCounts[row.source] = (sourceCounts[row.source] ?? 0) + 1;
  classCounts[row.label === 0 ? "real" : "synthetic"] += 1;
  if (["open-images-train", "open-images", "docci-train"].includes(row.source)) {
    requireCondition(row.label === 0, `Real source has a synthetic label: ${row.id}`);
  } else {
    requireCondition(row.label === 1, `Synthetic source has a real label: ${row.id}`);
  }
}
requireCondition(jsonEqual(sourceCounts, EXPECTED_SOURCES) && jsonEqual(sourceCounts, selectionSummary.sourceCounts),
  "Training source counts changed");
requireCondition(jsonEqual(classCounts, { real: 52_400, synthetic: 51_200 }) &&
  jsonEqual(classCounts, selectionSummary.classCounts), "Training class counts changed");
requireCondition(jsonEqual(selectionSummary.counts, {
  diffusionDb: 50_000, modern: 3_600, openImages: 50_000, rejected: selectionSummary.counts.rejected, total: 103_600,
}), "Selection summary counts changed");

const attributionBytes = await readGzip(`${ROOT}/open-images-attribution.jsonl.gz`);
requireCondition(digest(attributionBytes) === selectionSummary.attributionSha256,
  "Expanded Open Images attribution hash mismatch");
const attributionRows = parseJsonLines(attributionBytes, `${ROOT}/open-images-attribution.jsonl.gz`);
requireCondition(attributionRows.length === 50_000, "Open Images attribution must contain 50000 rows");
const openImageTrainingRows = new Map(trainingRows.filter((row) => row.source === "open-images-train")
  .map((row) => [row.id, row.imageSha256]));
const attributionIds = new Set();
const attributionHashes = new Set();
for (const row of attributionRows) {
  requireCondition(!attributionIds.has(row.manifestId) && !attributionHashes.has(row.selectedImageSha256),
    `Duplicate Open Images attribution: ${row.manifestId}`);
  requireCondition(openImageTrainingRows.get(row.manifestId) === row.selectedImageSha256,
    `Open Images attribution does not bind its selected training row: ${row.manifestId}`);
  requireCondition(row.license === "https://creativecommons.org/licenses/by/2.0/" &&
    [row.author, row.title, row.landingUrl, row.originalUrl].every((value) => typeof value === "string" && value.length > 0),
  `Open Images attribution is incomplete: ${row.manifestId}`);
  attributionIds.add(row.manifestId);
  attributionHashes.add(row.selectedImageSha256);
}
requireCondition(attributionIds.size === openImageTrainingRows.size,
  "Open Images attribution is not one-to-one with selected training images");

const rejectsBytes = await readGzip(`${ROOT}/rejects.jsonl.gz`);
requireCondition(digest(rejectsBytes) === selectionSummary.rejectsSha256, "Expanded rejects hash mismatch");
const rejectRows = parseJsonLines(rejectsBytes, `${ROOT}/rejects.jsonl.gz`);
requireCondition(rejectRows.length === selectionSummary.counts.rejected, "Reject count does not match selection summary");

const exclusionRows = [];
const exclusionHashByPath = {};
const expectedSummaryExclusions = [];
for (const expected of EXPECTED_EXCLUSIONS) {
  const bytes = await readFile(expected.path);
  const rows = parseJsonLines(bytes, expected.path);
  requireCondition(rows.length === expected.rows, `${expected.path} row count changed`);
  const sha256 = digest(bytes);
  exclusionHashByPath[expected.path] = sha256;
  expectedSummaryExclusions.push({ ...expected, sha256 });
  for (const row of rows) {
    requireCondition(!trainIds.has(row.id) && !trainHashes.has(row.imageSha256),
      `Training overlaps frozen exclusion: ${row.id}`);
    exclusionRows.push({ ...row, manifest: expected.path });
  }
}
requireCondition(jsonEqual(selectionSummary.evaluationExclusions, expectedSummaryExclusions),
  "Selection summary exclusion bindings changed");
requireCondition(selectionSummary.evaluationExcludedIds === 1_219 &&
  selectionSummary.evaluationExcludedImageHashes === 1_219,
"Selection summary exclusion counts changed");

const evaluationDhashBytes = await readGzip(`${ROOT}/evaluation-perceptual-hashes.json.gz`);
requireCondition(digest(evaluationDhashBytes) === selectionSummary.evaluationPerceptualHashesSha256,
  "Expanded evaluation dHash packet mismatch");
const evaluationDhash = parseJson(evaluationDhashBytes, `${ROOT}/evaluation-perceptual-hashes.json.gz`);
requireCondition(evaluationDhash.schemaVersion === 1 && evaluationDhash.hammingThreshold === 8 &&
  evaluationDhash.algorithm === "EXIF-oriented RGB dHash 64-bit, LANCZOS 9x8 grayscale" &&
  evaluationDhash.items.length === 1_219, "Evaluation dHash packet contract changed");
const exclusionById = new Map(exclusionRows.map((row) => [row.id, row]));
const evaluationDhashById = new Map();
for (const row of evaluationDhash.items) {
  const excluded = exclusionById.get(row.id);
  requireCondition(excluded && excluded.imageSha256 === row.imageSha256 && excluded.manifest === row.manifest &&
    DHASH64.test(row.perceptualDhash64), `Evaluation dHash entry is stale: ${row.id}`);
  requireCondition(!evaluationDhashById.has(row.id), `Duplicate evaluation dHash entry: ${row.id}`);
  evaluationDhashById.set(row.id, row);
}
requireCondition(evaluationDhashById.size === exclusionById.size,
  "Evaluation dHash packet is not exhaustive");

const primaryReviewPath = "benchmark/manifests/perceptual-overlap-review.json";
const primaryReviewBytes = await readFile(primaryReviewPath);
const primaryReview = parseJson(primaryReviewBytes, primaryReviewPath);
requireCondition(jsonEqual(selectionSummary.perceptualOverlapReview, {
  path: primaryReviewPath,
  sha256: digest(primaryReviewBytes),
  reviewedPairCount: primaryReview.items.length,
  hammingThreshold: 8,
}), "Primary perceptual review binding changed");
const validationDhashRows = evaluationDhash.items.filter((row) => row.manifest === EXPECTED_EXCLUSIONS[0].path);
const testDhashRows = evaluationDhash.items.filter((row) => row.manifest === EXPECTED_EXCLUSIONS[1].path);
const primaryPairs = findNearPairs(validationDhashRows, testDhashRows, 8);
const reviewedPrimaryPairs = new Set(primaryReview.items.map((row) => row.ids.join("\0")));
requireCondition(primaryReview.schemaVersion === 1 && primaryReview.hammingThreshold === 8 &&
  primaryPairs.length === primaryReview.items.length && primaryPairs.every(({ left, right, distance }) => {
    const review = primaryReview.items.find((row) => row.ids[0] === left.id && row.ids[1] === right.id);
    return review?.decision === "visually-distinct" && review.hammingDistance === distance && reviewedPrimaryPairs.has(`${left.id}\0${right.id}`);
  }), "Primary validation/test dHash review is not exhaustive");

const trainingReviewPath = "benchmark/manifests/training-evaluation-perceptual-review.json";
const trainingReviewBytes = await readFile(trainingReviewPath);
const trainingReview = parseJson(trainingReviewBytes, trainingReviewPath);
requireCondition(jsonEqual(selectionSummary.trainingPerceptualOverlapReview, {
  path: trainingReviewPath,
  sha256: digest(trainingReviewBytes),
  reviewedPairCount: trainingReview.items.length,
  hammingThreshold: 8,
}), "Training/evaluation perceptual review binding changed");
requireCondition(trainingReview.schemaVersion === 1 && trainingReview.hammingThreshold === 8 &&
  jsonEqual(trainingReview.evaluationExclusionSha256ByPath, exclusionHashByPath) &&
  trainingReview.trainingManifestSha256 === digest(await readFile("benchmark/manifests/train.jsonl")),
"Training/evaluation perceptual review inputs changed");
const trainingPairs = findNearPairs(trainingRows, evaluationDhash.items, 8);
const reviewByPair = new Map(trainingReview.items.map((row) => [`${row.trainingId}\0${row.evaluationId}`, row]));
requireCondition(trainingPairs.length === 100 && reviewByPair.size === 100 && trainingPairs.every(({ left, right, distance }) => {
  const review = reviewByPair.get(`${left.id}\0${right.id}`);
  return review?.decision === "visually-distinct" && review.hammingDistance === distance &&
    review.trainingImageSha256 === left.imageSha256 && review.evaluationImageSha256 === right.imageSha256 &&
    review.trainingPerceptualDhash64 === left.perceptualDhash64 &&
    review.evaluationPerceptualDhash64 === right.perceptualDhash64 &&
    [review.reviewer, review.reviewedAt, review.rationale].every((value) => typeof value === "string" && value.length > 0);
}), "Training/evaluation dHash review is stale or incomplete");
requireCondition(jsonEqual(selectionSummary.overlapWithEvaluation, {
  ids: 0,
  imageHashes: 0,
  perceptualHammingThreshold: 8,
  reviewedVisuallyDistinctDhashPairsAtOrBelowThreshold: 100,
  unreviewedPerceptualDhashPairsAtOrBelowThreshold: 0,
}), "Selection summary overlap result changed");

const trainingSummaryBytes = await readFile(`${ROOT}/training-summary.json`);
const calibrationEvidenceBytes = await readFile(`${ROOT}/calibration.json`);
const candidateGridBytes = await readFile(`${ROOT}/candidate-grid.json`);
const modelComparisonBytes = await readFile(`${ROOT}/model-comparison.json`);
const trainingSummary = parseJson(trainingSummaryBytes, `${ROOT}/training-summary.json`);
const calibration = parseJson(calibrationEvidenceBytes, `${ROOT}/calibration.json`);
const candidateGrid = parseJson(candidateGridBytes, `${ROOT}/candidate-grid.json`);
const modelComparison = parseJson(modelComparisonBytes, `${ROOT}/model-comparison.json`);
const finalizationReceipt = parseJson(
  await readFile(`${ROOT}/finalization-receipt.json`),
  `${ROOT}/finalization-receipt.json`,
);
const validationManifestBytes = await readFile("benchmark/manifests/validation.jsonl");
const selectionSummarySha256 = digest(summaryBytes);
requireCondition(trainingSummary.schemaVersion === 1 && trainingSummary.seed === 20_260_813 &&
  trainingSummary.pipelineVersion === 8 && trainingSummary.recipeSha256 === digest(recipeBytes) &&
  trainingSummary.trainerSha256 === M1_TRAINER_SHA256 &&
  jsonEqual(trainingSummary.commandArguments, EXPECTED_TRAINING_ARGUMENTS) &&
  trainingSummary.selectionSummarySha256 === selectionSummarySha256 &&
  trainingSummary.upstreamModelSha256 === UPSTREAM_MODEL_SHA256 &&
  trainingSummary.trainManifestSha256 === digest(trainBytes) &&
  trainingSummary.validationManifestSha256 === digest(validationManifestBytes),
"Training summary input bindings changed");
requireCondition(trainingSummary.trainImages === 103_600 && trainingSummary.validationImages === 600 &&
  trainingSummary.viewsPerImage === 4 && trainingSummary.trainFeatureViews === 114_400 &&
  trainingSummary.validationFeatureViews === 2_400 && trainingSummary.uniqueTrainingImagesCovered === 103_600 &&
  trainingSummary.uniqueTrainingFeatureViewsCovered === 114_400, "Training coverage changed");
requireCondition(trainingSummary.sourceBalancedSampling === false && trainingSummary.sourceBalancedLoss === true &&
  trainingSummary.trainingEpochs === 12 && trainingSummary.trainingBatchSize === 2_048 &&
  trainingSummary.featureBatchSize === 24 && trainingSummary.featureShardImages === 2_000 &&
  trainingSummary.cachedFeatureSourcePixelsReverified === true && trainingSummary.cachedFeatureArraysValidated === true &&
  trainingSummary.cachedFeatureValuesReextracted === true &&
  jsonEqual(trainingSummary.cachedFeatureDtypes, {
    features: "float32", labels: "float32", variants: "int64", sources: "unicode",
  }) &&
  jsonEqual(trainingSummary.singleViewTrainingSources, ["diffusiondb-stable-diffusion", "open-images-train"]) &&
  jsonEqual(trainingSummary.trainingSourceCounts, EXPECTED_SOURCES), "Training procedure changed");
requireCondition(HEX64.test(trainingSummary.featureConfigurationHashes?.training ?? "") &&
  HEX64.test(trainingSummary.featureConfigurationHashes?.validation ?? "") &&
  trainingSummary.featureConfigurationHashes.training !== trainingSummary.featureConfigurationHashes.validation,
"Training feature-configuration hashes are missing or collapsed");
const freshFeatureRun = trainingSummary.freshFeatureRun;
requireCondition(freshFeatureRun?.schemaVersion === 1 && /^[a-f0-9]{32}$/u.test(freshFeatureRun.runId ?? "") &&
  freshFeatureRun.state === "complete" && jsonEqual(freshFeatureRun.context, {
    pipelineVersion: 8,
    featureExtractorContract: "cf384-static-batch24-preprocess-v1",
    upstreamModelSha256: UPSTREAM_MODEL_SHA256,
    trainManifestSha256: digest(trainBytes),
    validationManifestSha256: digest(validationManifestBytes),
    selectionSummarySha256,
    featureBatchSize: 24,
    featureShardImages: 2_000,
    singleViewTrainingSources: ["diffusiondb-stable-diffusion", "open-images-train"],
    featureConfigurationHashes: trainingSummary.featureConfigurationHashes,
  }) && trainingSummary.freshFeatureRunMarkerSha256 ===
    digest(Buffer.from(`${JSON.stringify(freshFeatureRun, null, 2)}\n`)),
"Fresh-feature run marker is missing or does not bind the release extraction context");
const expectedShardNames = [
  ...Array.from({ length: Math.ceil(103_600 / 2_000) }, (_, index) =>
    `benchmark/candidates/prooflens-cf384-large/features/train-${String(index).padStart(5, "0")}.npz`),
  "benchmark/candidates/prooflens-cf384-large/features/validation-00000.npz",
];
requireCondition(Array.isArray(trainingSummary.featureShardEvidence) &&
  trainingSummary.featureShardEvidence.length === expectedShardNames.length &&
  jsonEqual(trainingSummary.featureShardEvidence.map((row) => row.cache), expectedShardNames) &&
  trainingSummary.featureShardEvidence.every((row) =>
    row.freshFeatureRunId === freshFeatureRun.runId && row.freshlyExtractedThisRun === true &&
    typeof row.freshlyExtractedThisProcess === "boolean" && HEX64.test(row.cacheSha256) &&
    (row.replacedCacheSha256 === null || HEX64.test(row.replacedCacheSha256)) &&
    HEX64.test(row.itemIdsSha256) &&
    row.featureConfigurationSha256 === (row.cache.includes("/train-")
      ? trainingSummary.featureConfigurationHashes.training
      : trainingSummary.featureConfigurationHashes.validation) &&
    ["features", "labels", "variants", "sources"].every((name) => HEX64.test(row.arraySha256?.[name] ?? "")) &&
    Number.isInteger(row.items) && row.items > 0 && Number.isInteger(row.views) && row.views > 0) &&
  trainingSummary.featureShardEvidence.reduce((total, row) => total + row.items, 0) === 104_200 &&
  trainingSummary.featureShardEvidence.reduce((total, row) => total + row.views, 0) === 116_800,
"Final training did not freshly extract, resume, and bind every feature shard to one run");
requireCondition(trainingSummary.environment?.executionProvider === "cpu" &&
  jsonEqual(trainingSummary.environment.providers, ["CPUExecutionProvider"]) &&
  trainingSummary.environment.torchDeterministicAlgorithms === true &&
  [trainingSummary.environment.numpy, trainingSummary.environment.onnxRuntime,
    trainingSummary.environment.pillow, trainingSummary.environment.torch,
    trainingSummary.environment.python, trainingSummary.environment.platform].every((value) =>
    typeof value === "string" && value.length > 0), "Training environment evidence is incomplete");
requireCondition(jsonEqual(trainingSummary.validationGates, recipe.validationGates) &&
  trainingSummary.validationGatesPassed === true, "Training summary validation gates changed");
requireCondition(candidateGrid.length === 25 && trainingSummary.candidateCount === 25,
  "Candidate grid must contain all 25 predeclared parameter pairs");
const expectedParameters = new Set();
for (const weightDecay of [0.1, 0.03, 0.01, 0.003, 0.001]) {
  for (const upstreamBlendAlpha of [0.4, 0.55, 0.7, 0.85, 1]) {
    expectedParameters.add(JSON.stringify({ weightDecay, upstreamBlendAlpha }));
  }
}
const seenParameters = new Set(candidateGrid.map((candidate) => JSON.stringify(candidate.parameters)));
requireCondition(seenParameters.size === 25 && [...expectedParameters].every((key) => seenParameters.has(key)),
  "Candidate grid parameter coverage changed");
const validCandidates = candidateGrid.filter((candidate) => candidate.selectionKey);
requireCondition(validCandidates.length === trainingSummary.validCandidateCount && validCandidates.length > 0,
  "Candidate grid valid-candidate count changed");
for (const candidate of validCandidates) {
  requireCondition(Array.isArray(candidate.selectionKey) && candidate.selectionKey.length === 4 &&
    candidate.selectionKey.every(finiteNumber) && finiteNumber(candidate.thresholdLogit),
  "Candidate selection evidence is malformed");
  requireVariantGates(candidate.variants, recipe.validationGates, `candidate ${JSON.stringify(candidate.parameters)}`);
}
for (const candidate of candidateGrid.filter((row) => !row.selectionKey)) {
  requireCondition(candidate.status === "rejected" && typeof candidate.reason === "string" && candidate.reason.length > 0,
    "Rejected candidate lacks a reason");
}
const selected = validCandidates.find((candidate) => jsonEqual(candidate.parameters, trainingSummary.selectedParameters));
requireCondition(selected && jsonEqual(selected.selectionKey, trainingSummary.selectionKey) &&
  selected.thresholdLogit === trainingSummary.thresholdLogit && jsonEqual(selected.variants, trainingSummary.variants),
"Selected candidate does not match the training summary");
const compareSelectionKeys = (left, right) => {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return 0;
};
const deterministicBest = validCandidates.reduce((best, candidate) =>
  compareSelectionKeys(candidate.selectionKey, best.selectionKey) > 0 ? candidate : best);
requireCondition(selected === deterministicBest,
  "Selected candidate is not the deterministic lexicographic maximum (stable grid order breaks ties)");
requireVariantGates(trainingSummary.variants, recipe.validationGates, "selected candidate");

// M1 becomes immutable historical evidence after M2 promotion. Read its shipped
// bytes from the public v3 source commit instead of reinterpreting the current model.
const shippedModelBytes = committedBytes("weights/prooflens-cf384.onnx");
const shippedModelSha256 = digest(shippedModelBytes);
const modelLockBytes = committedBytes("model-lock.json");
const modelLock = parseJson(modelLockBytes, "model-lock.json");
const weightsReadmeBytes = committedBytes("weights/README.md");
const expectedTrainingRecipe = `prooflens-cf384-large-head-v1:${digest(recipeBytes)}:${selectionSummarySha256}`;
requireCondition(modelLock.schemaVersion === 2 && modelLock.artifact === "weights/prooflens-cf384.onnx" &&
  modelLock.bytes === shippedModelBytes.length && modelLock.sha256 === shippedModelSha256 &&
  modelLock.trainingRecipe === expectedTrainingRecipe &&
  jsonEqual(modelLock.trainingEvidence, {
    recipe: "benchmark/large/recipe.json",
    recipeSha256: digest(recipeBytes),
    selectionSummary: "benchmark/evidence/large/selection-summary.json",
    selectionSummarySha256,
    trainManifestSha256: digest(trainBytes),
    trainingSummarySha256: digest(trainingSummaryBytes),
    calibrationSha256: digest(calibrationEvidenceBytes),
    candidateGridSha256: digest(candidateGridBytes),
  }), "Model lock does not bind the final large-training evidence");
requireCondition(modelLock.calibration?.slope === calibration.slope &&
  modelLock.calibration.intercept === calibration.intercept &&
  modelLock.calibration.displayThreshold === calibration.displayThreshold &&
  modelLock.calibration.validationThresholdLogit === calibration.validationThresholdLogit,
"Model lock calibration does not match the validated candidate");
requireCondition(weightsReadmeBytes.toString("utf8").includes(shippedModelSha256) &&
  weightsReadmeBytes.toString("utf8").includes(shippedModelBytes.length.toLocaleString("en-US")),
"Packaged model README is stale");
requireCondition(finalizationReceipt.schemaVersion === 2 &&
  finalizationReceipt.candidateDirectory === "benchmark/candidates/prooflens-cf384-large" &&
  finalizationReceipt.upstreamSha256 === UPSTREAM_MODEL_SHA256 &&
  finalizationReceipt.shippedModel?.path === "weights/prooflens-cf384.onnx" &&
  finalizationReceipt.shippedModel.sha256 === shippedModelSha256 &&
  finalizationReceipt.shippedModel.bytes === shippedModelBytes.length &&
  ["training-summary.json", "calibration.json", "candidate-grid.json"].every((name) =>
    finalizationReceipt.sourceEvidenceSha256?.[name] === digest(
      name === "training-summary.json" ? trainingSummaryBytes :
        name === "calibration.json" ? calibrationEvidenceBytes : candidateGridBytes,
    ) && finalizationReceipt.publishedEvidenceSha256?.[name] ===
      finalizationReceipt.sourceEvidenceSha256[name]) &&
  finalizationReceipt.publishedRepositorySha256?.["weights/prooflens-cf384.onnx"] === shippedModelSha256 &&
  finalizationReceipt.publishedRepositorySha256?.["model-lock.json"] === digest(modelLockBytes) &&
  finalizationReceipt.publishedRepositorySha256?.["weights/README.md"] === digest(weightsReadmeBytes),
"Finalization receipt does not bind the published candidate/evidence packet");
requireCondition(trainingSummary.model?.sha256 === shippedModelSha256 &&
  trainingSummary.model.bytes === shippedModelBytes.length && finiteNumber(trainingSummary.model.maxAbsParityError) &&
  trainingSummary.model.maxAbsParityError <= 2e-4, "Shipped model does not match validated training output");
requireCondition(calibration.schemaVersion === 1 &&
  calibration.method === "Validation-selected logit alignment to fixed 65/100 display threshold; not probability calibration" &&
  calibration.slope === 1 && calibration.displayThreshold === 0.65 &&
  calibration.modelSha256 === shippedModelSha256 && calibration.trainManifestSha256 === digest(trainBytes) &&
  calibration.validationManifestSha256 === digest(validationManifestBytes) &&
  calibration.selectionSummarySha256 === selectionSummarySha256,
"Calibration input bindings changed");
requireCondition([calibration.intercept, calibration.validationThresholdLogit,
  calibration.rawProbabilityThreshold].every(finiteNumber), "Calibration contains a non-finite value");
requireCondition(calibration.rawProbabilityThreshold > 0 && calibration.rawProbabilityThreshold < 1,
  "Calibration raw threshold must be strictly between zero and one");
close(1 / (1 + Math.exp(-calibration.validationThresholdLogit)), calibration.rawProbabilityThreshold,
  "calibration raw threshold");
close(1 / (1 + Math.exp(-(calibration.validationThresholdLogit + calibration.intercept))), 0.65,
  "calibration display threshold");

requireCondition(modelComparison.schemaVersion === 1 &&
  modelComparison.base?.sha256 === UPSTREAM_MODEL_SHA256 &&
  modelComparison.candidate?.sha256 === shippedModelSha256 &&
  modelComparison.candidate.bytes === shippedModelBytes.length &&
  modelComparison.unchangedInitializerCount > 0 &&
  jsonEqual(modelComparison.changedInitializers.map((row) => row.name).sort(), ["classifier.bias", "classifier.weight"]) &&
  [modelComparison.graphNodesSha256, modelComparison.graphInputsSha256,
    modelComparison.graphOutputsSha256, modelComparison.opsetsSha256].every((value) => HEX64.test(value)),
"Model comparison does not prove classifier-only adaptation");
requireCondition(finalizationReceipt.publishedEvidenceSha256?.["model-comparison.json"] ===
  digest(modelComparisonBytes),
"Finalization receipt does not bind the model comparison");

const upstreamStructure = parseJson(
  await readFile(`${ROOT}/upstream-model-structure.json`),
  `${ROOT}/upstream-model-structure.json`,
);
requireCondition(upstreamStructure.schemaVersion === 1 &&
  upstreamStructure.model?.sha256 === UPSTREAM_MODEL_SHA256 && upstreamStructure.model.bytes === 87_442_080 &&
  upstreamStructure.protobufReader === "onnxruntime-web@1.22.0 generated ONNX schema" &&
  upstreamStructure.initializers.length === 200,
"Pinned upstream model-structure lock changed");
const shippedStructure = inspectOnnxStructure(shippedModelBytes);
for (const key of ["graphNodesSha256", "graphInputsSha256", "graphOutputsSha256", "opsetsSha256"]) {
  requireCondition(shippedStructure[key] === upstreamStructure[key], `Shipped ONNX ${key} differs from the pinned upstream`);
}
const upstreamInitializerByName = new Map(upstreamStructure.initializers.map((row) => [row.name, row]));
requireCondition(upstreamInitializerByName.size === 200 && shippedStructure.initializers.length === 200 &&
  shippedStructure.initializers.every((row) => upstreamInitializerByName.has(row.name)),
"Shipped ONNX initializer names differ from the pinned upstream");
const independentlyChangedInitializers = [];
for (const initializer of shippedStructure.initializers) {
  const upstream = upstreamInitializerByName.get(initializer.name);
  requireCondition(jsonEqual(initializer.dimensions, upstream.dimensions),
    `Shipped ONNX initializer dimensions changed: ${initializer.name}`);
  if (initializer.sha256 !== upstream.sha256) independentlyChangedInitializers.push(initializer.name);
}
requireCondition(jsonEqual(independentlyChangedInitializers.sort(), ["classifier.bias", "classifier.weight"]),
  "Independent ONNX parse found changes outside the classifier head");

console.log(JSON.stringify({
  corpus: {
    images: trainingRows.length,
    real: classCounts.real,
    synthetic: classCounts.synthetic,
    sources: Object.keys(sourceCounts).length,
    openImagesAttributions: attributionRows.length,
  },
  exclusions: {
    images: exclusionRows.length,
    exactIdOverlap: 0,
    exactByteOverlap: 0,
    reviewedNearPairs: trainingPairs.length,
    unreviewedNearPairs: 0,
  },
  training: {
    featureViews: trainingSummary.trainFeatureViews,
    candidates: candidateGrid.length,
    validCandidates: validCandidates.length,
    modelSha256: shippedModelSha256,
    classifierOnly: true,
    historicalSourceCommit: M1_SOURCE_COMMIT,
  },
  policy: "pass",
}));
