import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pngDimensions, validateBrowserGeometryEvidence } from "./browser-geometry-contract.mjs";

const VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];
const EXPECTED_MANIFESTS = {
  train: {
    path: "benchmark/manifests/train.jsonl",
    sha256: "03b88b3804244018fbdf532b2b7d451db91dad7c3229c9013ee4ede9fa798015",
    rows: 3_600,
  },
  validation: {
    path: "benchmark/manifests/validation.jsonl",
    sha256: "41be10ef876ecef0635744ed29677a1888a7759cc8060dc7a392f76f83ab263b",
    rows: 600,
  },
  test: {
    path: "benchmark/manifests/test.jsonl",
    sha256: "28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae",
    rows: 600,
  },
  webNegative: {
    path: "benchmark/manifests/web-negative.jsonl",
    sha256: "ad8b3f30a37feb3b6b046683db2d4071e236e6878612c7d8733869699d7f7824",
    rows: 319,
  },
};
const CONFIRMATORY_PREDICTIONS = Object.fromEntries(VARIANTS.map((variant) => [
  variant,
  `benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-${variant}-predictions.jsonl`,
]));
const WEB_NEGATIVE_PREDICTIONS = Object.fromEntries(VARIANTS.map((variant) => [
  variant,
  `benchmark/evidence/evaluation/web-negative/prooflens-web-negative-${variant}-predictions.jsonl`,
]));
const CONFIRMATORY_SUMMARY = "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-summary.json";
const CONFIRMATORY_COMPLETION = "benchmark/evidence/evaluation/confirmatory/prooflens-confirmatory-test-complete.json";
const CONFIRMATORY_BOOTSTRAP = "benchmark/evidence/evaluation/confirmatory/bootstrap.json";
const WEB_NEGATIVE_SUMMARY = "benchmark/evidence/evaluation/web-negative/prooflens-web-negative-summary.json";
const WEB_NEGATIVE_COMPLETION = "benchmark/evidence/evaluation/web-negative/prooflens-web-negative-complete.json";
const WEB_NEGATIVE_WILSON = "benchmark/evidence/evaluation/web-negative/wilson.json";
const VALIDATION_PREDICTIONS = Object.fromEntries(VARIANTS.map((variant) => [
  variant,
  `benchmark/evidence/evaluation/validation/prooflens-validation-${variant}-predictions.jsonl`,
]));
const VALIDATION_SUMMARY = "benchmark/evidence/evaluation/validation/prooflens-validation-summary.json";
const VALIDATION_COMPLETION = "benchmark/evidence/evaluation/validation/prooflens-validation-complete.json";
const CHROME_E2E = {
  wasm: "artifacts/chrome-e2e-wasm.json",
  webgpu: "artifacts/chrome-e2e-webgpu.json",
};

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function parseJsonLines(bytes, file) {
  try {
    return bytes.toString("utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
  } catch (error) {
    throw new Error(`${file} is not valid JSONL`, { cause: error });
  }
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  }
  return value;
}

function jsonEqual(left, right) {
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

function close(actual, expected, label, tolerance = 1e-12) {
  requireCondition(typeof actual === "number" && Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, received ${actual}`);
}

function sigmoid(logit) {
  if (logit >= 0) return 1 / (1 + Math.exp(-logit));
  const exponential = Math.exp(logit);
  return exponential / (1 + exponential);
}

function predictionIsNumericallyConsistent(row) {
  return Number.isFinite(row.logit) && Number.isFinite(row.rawProbability) &&
    row.rawProbability >= 0 && row.rawProbability <= 1 &&
    Math.abs(sigmoid(row.logit) - row.rawProbability) <= 2e-12;
}

function countsBy(rows, field) {
  return Object.fromEntries([...new Set(rows.map((row) => String(row[field])))].sort().map((value) => [
    value,
    rows.filter((row) => String(row[field]) === value).length,
  ]));
}

function predictionMetrics(rows, threshold) {
  const real = rows.filter((row) => row.label === 0);
  const synthetic = rows.filter((row) => row.label === 1);
  const realRecall = real.filter((row) => row.rawProbability < threshold).length / real.length;
  const syntheticRecall = synthetic.filter((row) => row.rawProbability >= threshold).length / synthetic.length;
  return {
    balancedAccuracy: (realRecall + syntheticRecall) / 2,
    realRecall,
    syntheticRecall,
    syntheticRecallBySource: Object.fromEntries([...new Set(synthetic.map((row) => row.source))].sort().map((source) => {
      const selected = synthetic.filter((row) => row.source === source);
      return [source, selected.filter((row) => row.rawProbability >= threshold).length / selected.length];
    })),
    count: rows.length,
  };
}

function negativeMetrics(rows, threshold) {
  const falsePositives = rows.filter((row) => row.rawProbability >= threshold).length;
  return {
    count: rows.length,
    realRecall: 1 - falsePositives / rows.length,
    falsePositiveRate: falsePositives / rows.length,
    bySource: Object.fromEntries([...new Set(rows.map((row) => row.source))].sort().map((source) => {
      const selected = rows.filter((row) => row.source === source);
      const count = selected.filter((row) => row.rawProbability >= threshold).length;
      return [source, { count: selected.length, falsePositiveRate: count / selected.length }];
    })),
  };
}

function wilson(successes, count) {
  const z = 1.959963984540054;
  const probability = successes / count;
  const denominator = 1 + z * z / count;
  const center = (probability + z * z / (2 * count)) / denominator;
  const margin = z * Math.sqrt(probability * (1 - probability) / count + z * z / (4 * count * count)) / denominator;
  return [Math.max(0, center - margin), Math.min(1, center + margin)];
}

function priority(identifier) {
  return createHash("sha256").update(`20260813:browser-parity:${identifier}`).digest("hex");
}

const manifests = {};
for (const [name, expected] of Object.entries(EXPECTED_MANIFESTS)) {
  const bytes = await readFile(expected.path);
  requireCondition(digest(bytes) === expected.sha256, `${expected.path} SHA-256 changed`);
  const rows = parseJsonLines(bytes, expected.path);
  requireCondition(rows.length === expected.rows, `${expected.path} row count changed`);
  requireCondition(new Set(rows.map((row) => row.id)).size === rows.length,
    `${expected.path} contains duplicate IDs`);
  requireCondition(new Set(rows.map((row) => row.imageSha256)).size === rows.length,
    `${expected.path} contains duplicate image bytes`);
  manifests[name] = rows;
}

const splitRows = [manifests.train, manifests.validation, manifests.test];
const splitNames = ["train", "validation", "test"];
for (const [index, rows] of splitRows.entries()) {
  requireCondition(rows.every((row) => row.split === splitNames[index] && /^[a-f0-9]{64}$/u.test(row.imageSha256)),
    `${splitNames[index]} manifest has invalid split/hash fields`);
}
for (let left = 0; left < splitRows.length; left += 1) {
  for (let right = left + 1; right < splitRows.length; right += 1) {
    const rightIds = new Set(splitRows[right].map((row) => row.id));
    const rightHashes = new Set(splitRows[right].map((row) => row.imageSha256));
    requireCondition(splitRows[left].every((row) => !rightIds.has(row.id) && !rightHashes.has(row.imageSha256)),
      `${splitNames[left]} and ${splitNames[right]} overlap`);
  }
}

const syntheticBySplit = Object.fromEntries(splitNames.map((name) => [name, manifests[name].filter((row) => row.label === 1)]));
const expectedSynthetic = {
  train: { rows: 1_200, groups: 80, groupSize: 15, sources: 15, sourceSize: 80 },
  validation: { rows: 300, groups: 150, groupSize: 2, sources: 2, sourceSize: 150 },
  test: { rows: 300, groups: 300, groupSize: 1, sources: 1, sourceSize: 300 },
};
const groupSets = {};
const sourceSets = {};
for (const name of splitNames) {
  const rows = syntheticBySplit[name];
  const groups = countsBy(rows, "groupId");
  const sources = countsBy(rows, "source");
  const expected = expectedSynthetic[name];
  requireCondition(rows.length === expected.rows && Object.keys(groups).length === expected.groups &&
    Object.values(groups).every((count) => count === expected.groupSize) &&
    Object.keys(sources).length === expected.sources && Object.values(sources).every((count) => count === expected.sourceSize),
  `${name} synthetic prompt/source allocation changed`);
  groupSets[name] = new Set(Object.keys(groups));
  sourceSets[name] = new Set(Object.keys(sources));
}
for (const [left, right] of [["train", "validation"], ["train", "test"], ["validation", "test"]]) {
  requireCondition([...groupSets[left]].every((value) => !groupSets[right].has(value)),
    `${left} and ${right} share a synthetic prompt group`);
  requireCondition([...sourceSets[left]].every((value) => !sourceSets[right].has(value)),
    `${left} and ${right} share a synthetic generator source`);
}
requireCondition(jsonEqual(countsBy(manifests.train.filter((row) => row.label === 0), "source"), {
  "docci-train": 1_200, "open-images": 1_200,
}) && jsonEqual(countsBy(manifests.validation.filter((row) => row.label === 0), "source"), {
  "open-images": 300,
}) && jsonEqual(countsBy(manifests.test.filter((row) => row.label === 0), "source"), {
  "library-of-congress-fsa-owi-color": 300,
}), "Real-source split allocation changed");
requireCondition(jsonEqual(countsBy(syntheticBySplit.test, "source"), { kling_v2_1: 300 }),
  "Confirmatory synthetic source changed");

const webIds = new Set(manifests.webNegative.map((row) => row.id));
const webHashes = new Set(manifests.webNegative.map((row) => row.imageSha256));
const intentionalTestOverlap = manifests.test.filter((row) => webIds.has(row.id) && webHashes.has(row.imageSha256));
requireCondition(intentionalTestOverlap.length === 300 && intentionalTestOverlap.every((row) =>
  row.source === "library-of-congress-fsa-owi-color") &&
  manifests.webNegative.filter((row) => row.source === "chartography-expert-created").length === 19,
"Web-negative/test overlap is not exactly the disclosed 300 historical photographs");
const trainAndValidationIds = new Set([...manifests.train, ...manifests.validation].map((row) => row.id));
const trainAndValidationHashes = new Set([...manifests.train, ...manifests.validation].map((row) => row.imageSha256));
requireCondition(manifests.webNegative.every((row) => !trainAndValidationIds.has(row.id) &&
  !trainAndValidationHashes.has(row.imageSha256)), "Web-negative pixels leaked into training/validation");

const legacyBytes = await readFile("benchmark/manifests/legacy-evaluation-exclusions.json");
const legacy = JSON.parse(legacyBytes);
const legacyIds = new Set(legacy.evaluationIds);
const legacyHashes = new Set(legacy.evaluationImageSha256);
const legacyGroups = new Set(legacy.qwenPromptGroups);
requireCondition([...manifests.train, ...manifests.validation, ...manifests.test].every((row) =>
  !legacyIds.has(row.id) && !legacyHashes.has(row.imageSha256) &&
  (row.label !== 1 || !legacyGroups.has(row.groupId))), "Fresh splits overlap legacy evaluation evidence");
const selection = JSON.parse(await readFile("benchmark/manifests/selection.json", "utf8"));
requireCondition(selection.schemaVersion === 1 && selection.seed === 20_260_813 &&
  selection.legacyEvaluationExclusions.sha256 === digest(legacyBytes) &&
  selection.legacyEvaluationExclusions.overlapIds === 0 &&
  selection.legacyEvaluationExclusions.overlapImageSha256 === 0 &&
  jsonEqual([...selection.qwenImageBench.trainSources].sort(), [...sourceSets.train].sort()) &&
  jsonEqual([...selection.qwenImageBench.validationSources].sort(), [...sourceSets.validation].sort()) &&
  jsonEqual([...selection.qwenImageBench.testSources].sort(), [...sourceSets.test].sort()),
"Modern selection evidence changed");

const openImagesAttribution = JSON.parse(await readFile("benchmark/manifests/open-images-attribution.json", "utf8"));
const modernOpenImages = [...manifests.train, ...manifests.validation].filter((row) => row.dataset === "Open Images V7");
const attributionById = new Map(openImagesAttribution.map((row) => [row.imageId, row]));
requireCondition(openImagesAttribution.length === 1_500 && attributionById.size === 1_500 &&
  modernOpenImages.length === 1_500 && modernOpenImages.every((row) => {
    const identifier = row.id.split(":").at(-1);
    const attribution = attributionById.get(identifier);
    return attribution?.license === "https://creativecommons.org/licenses/by/2.0/" &&
      [attribution.author, attribution.title, attribution.landingUrl].every((value) =>
        typeof value === "string" && value.length > 0);
  }), "Modern Open Images attribution is incomplete");
const docciAttribution = JSON.parse(await readFile("benchmark/manifests/docci-attribution.json", "utf8"));
requireCondition(Array.isArray(docciAttribution.selectedExampleIds) &&
  docciAttribution.selectedExampleIds.length === 1_200 &&
  new Set(docciAttribution.selectedExampleIds).size === 1_200,
"DOCCI attribution is incomplete");

const webSelection = JSON.parse(await readFile("benchmark/results/web-negative-selection.json", "utf8"));
const webPlanBytes = await readFile("benchmark/manifests/web-negative-plan.json");
const webReviewBytes = await readFile("benchmark/manifests/web-negative-review.json");
const webRecipeBytes = await readFile("benchmark/web-negative/recipe.json");
const webPlan = JSON.parse(webPlanBytes);
const webReview = JSON.parse(webReviewBytes);
requireCondition(webSelection.schemaVersion === 1 &&
  webSelection.confirmatoryTestManifestSha256 === EXPECTED_MANIFESTS.test.sha256 &&
  webSelection.manifestSha256 === EXPECTED_MANIFESTS.webNegative.sha256 &&
  webSelection.planSha256 === digest(webPlanBytes) && webSelection.reviewSha256 === digest(webReviewBytes) &&
  webSelection.recipeSha256 === digest(webRecipeBytes) && webPlan.items.length === 300 &&
  webReview.items.length === 300 && webReview.items.every((row) => row.decision === "include"),
"Web-negative source/review evidence changed");

const modelLock = JSON.parse(await readFile("model-lock.json", "utf8"));
const shippedModelBytes = await readFile("weights/prooflens-cf384.onnx");
requireCondition(digest(shippedModelBytes) === modelLock.sha256 && shippedModelBytes.length === modelLock.bytes,
  "Packaged model bytes do not match model-lock.json");
const calibrationBytes = await readFile("benchmark/evidence/large/calibration.json");
const calibration = JSON.parse(calibrationBytes);
const rawThreshold = calibration.rawProbabilityThreshold;
requireCondition(calibration.modelSha256 === modelLock.sha256 && calibration.displayThreshold === 0.65 &&
  calibration.slope === 1 && typeof rawThreshold === "number" && rawThreshold > 0 && rawThreshold < 1 &&
  modelLock.calibration.intercept === calibration.intercept &&
  modelLock.calibration.validationThresholdLogit === calibration.validationThresholdLogit,
"Model lock and frozen calibration disagree");

const evaluationRecipe = JSON.parse(await readFile("benchmark/large/recipe.json", "utf8"));
requireCondition(jsonEqual(evaluationRecipe.evaluationProtocols?.validation, {
  name: "prooflens-validation",
  manifest: EXPECTED_MANIFESTS.validation.path,
  manifestSha256: EXPECTED_MANIFESTS.validation.sha256,
  items: 600,
  variants: VARIANTS,
}) && evaluationRecipe.evaluationProtocols?.confirmatory?.name === "prooflens-confirmatory-test" &&
  evaluationRecipe.evaluationProtocols.confirmatory.manifest === EXPECTED_MANIFESTS.test.path &&
  evaluationRecipe.evaluationProtocols.confirmatory.manifestSha256 === EXPECTED_MANIFESTS.test.sha256 &&
  evaluationRecipe.evaluationProtocols.confirmatory.items === 600 &&
  jsonEqual(evaluationRecipe.evaluationProtocols.confirmatory.variants, VARIANTS) &&
  evaluationRecipe.evaluationProtocols.confirmatory.postScorePolicy ===
    "No model, threshold, preprocessing, corpus, gate, or runtime changes. Any runtime defect requires a new untouched holdout and a new public freeze before source changes." &&
  evaluationRecipe.evaluationProtocols?.webNegative?.manifestSha256 === EXPECTED_MANIFESTS.webNegative.sha256 &&
  evaluationRecipe.evaluationProtocols.webNegative.items === 319 &&
  evaluationRecipe.evaluationProtocols.webNegative.sharesConfirmatoryRows === 300 &&
  evaluationRecipe.evaluationProtocols.webNegative.independentEstimate === false &&
  jsonEqual(evaluationRecipe.evaluationProtocols.webNegative.variants, VARIANTS),
"Frozen evaluation protocol changed");
const testGates = evaluationRecipe.testGates;
const webNegativeGates = evaluationRecipe.webNegativeGates;
requireCondition(testGates.method ===
  "Class-stratified one-image-cluster bootstrap with variant-derived deterministic RNG" &&
  testGates.confidenceLevel === 0.95 && testGates.seed === 20_260_813 && testGates.replicates === 20_000 &&
  testGates.minimumBalancedAccuracyLower95 === 0.75 && testGates.minimumRealRecallLower95 === 0.75 &&
  testGates.minimumSyntheticRecallLower95 === 0.75 && testGates.minimumSyntheticRecallPerFamily === 0.60 &&
  webNegativeGates.method === "Wilson score interval for false-positive proportions" &&
  webNegativeGates.confidenceLevel === 0.95 &&
  webNegativeGates.maximumOverallFalsePositiveRateUpper95 === 0.10 &&
  webNegativeGates.maximumPerSourceFalsePositiveRateUpper95 === 0.20,
"Frozen statistical acceptance gates changed");

const validationSummary = JSON.parse(await readFile(VALIDATION_SUMMARY, "utf8"));
const validationCompletion = JSON.parse(await readFile(VALIDATION_COMPLETION, "utf8"));
requireCondition(validationSummary.schemaVersion === 2 && validationSummary.protocol === "validation" &&
  validationSummary.model.sha256 === modelLock.sha256 && validationSummary.model.bytes === modelLock.bytes &&
  validationSummary.dataset.sha256 === EXPECTED_MANIFESTS.validation.sha256 && validationSummary.dataset.items === 600 &&
  validationSummary.threshold.raw === rawThreshold &&
  validationSummary.threshold.calibrationSha256 === digest(calibrationBytes) &&
  validationCompletion.schemaVersion === 1 && validationCompletion.protocol === "validation" &&
  validationCompletion.modelSha256 === modelLock.sha256 &&
  validationCompletion.manifestSha256 === EXPECTED_MANIFESTS.validation.sha256 &&
  validationCompletion.calibrationSha256 === digest(calibrationBytes),
"Validation evaluation input bindings changed");
const validationById = new Map(manifests.validation.map((row) => [row.id, row]));
for (const variant of VARIANTS) {
  const file = VALIDATION_PREDICTIONS[variant];
  const bytes = await readFile(file);
  const rows = parseJsonLines(bytes, file);
  requireCondition(validationCompletion.files?.[path.basename(file)] === digest(bytes) &&
    rows.length === 600 && new Set(rows.map((row) => row.id)).size === 600 && rows.every((row) => {
      const item = validationById.get(row.id);
      return item && row.variant === variant && row.label === item.label && row.source === item.source &&
        row.groupId === item.groupId && predictionIsNumericallyConsistent(row);
    }), `${file} does not match the frozen validation manifest`);
  requireCondition(jsonEqual(predictionMetrics(rows, rawThreshold), validationSummary.variants[variant]),
    `${variant} validation metrics do not recompute`);
}
requireCondition(validationCompletion.files?.[path.basename(VALIDATION_SUMMARY)] === digest(await readFile(VALIDATION_SUMMARY)),
  "Validation summary is not completion-marker-bound");

const confirmatorySummary = JSON.parse(await readFile(CONFIRMATORY_SUMMARY, "utf8"));
const confirmatoryCompletion = JSON.parse(await readFile(CONFIRMATORY_COMPLETION, "utf8"));
const bootstrapBytes = await readFile(CONFIRMATORY_BOOTSTRAP);
const bootstrap = JSON.parse(bootstrapBytes);
requireCondition(confirmatorySummary.schemaVersion === 2 && confirmatorySummary.protocol === "confirmatory" &&
  confirmatorySummary.commitMarkerPublication === true && confirmatorySummary.allVariantsRequired === true &&
  confirmatorySummary.model.sha256 === modelLock.sha256 &&
  confirmatorySummary.model.bytes === modelLock.bytes &&
  confirmatorySummary.dataset.sha256 === EXPECTED_MANIFESTS.test.sha256 &&
  confirmatorySummary.dataset.items === 600 && confirmatorySummary.threshold.display === 0.65 &&
  confirmatorySummary.threshold.raw === rawThreshold &&
  confirmatorySummary.threshold.calibrationSha256 === digest(calibrationBytes),
"Confirmatory summary input bindings changed");
requireCondition(bootstrap.schemaVersion === 3 && bootstrap.seed === testGates.seed && bootstrap.replicates === testGates.replicates &&
  bootstrap.manifestSha256 === EXPECTED_MANIFESTS.test.sha256 &&
  bootstrap.rawProbabilityThreshold === rawThreshold &&
  bootstrap.method === testGates.method,
"Confirmatory bootstrap contract changed");
requireCondition(confirmatoryCompletion.schemaVersion === 1 && confirmatoryCompletion.protocol === "confirmatory" &&
  confirmatoryCompletion.modelSha256 === modelLock.sha256 &&
  confirmatoryCompletion.manifestSha256 === EXPECTED_MANIFESTS.test.sha256 &&
  confirmatoryCompletion.calibrationSha256 === digest(calibrationBytes),
"Confirmatory completion marker input bindings changed");
const testById = new Map(manifests.test.map((row) => [row.id, row]));
for (const variant of VARIANTS) {
  const file = CONFIRMATORY_PREDICTIONS[variant];
  const bytes = await readFile(file);
  requireCondition(confirmatoryCompletion.files?.[path.basename(file)] === digest(bytes),
    `${variant} confirmatory predictions are not completion-marker-bound`);
  requireCondition(bootstrap.variants[variant]?.predictionsSha256 === digest(bytes),
    `${variant} confirmatory predictions are not bootstrap-bound`);
  const rows = parseJsonLines(bytes, file);
  requireCondition(rows.length === 600 && new Set(rows.map((row) => row.id)).size === 600,
    `${file} does not contain exactly one prediction per test image`);
  for (const row of rows) {
    const item = testById.get(row.id);
    requireCondition(item && row.variant === variant && row.label === item.label && row.source === item.source &&
      row.groupId === item.groupId && predictionIsNumericallyConsistent(row),
    `${file} contains a stale prediction: ${row.id}`);
  }
  const metrics = predictionMetrics(rows, rawThreshold);
  requireCondition(jsonEqual(metrics, confirmatorySummary.variants[variant]),
    `${variant} confirmatory summary metrics do not recompute`);
  const interval = bootstrap.variants[variant];
  close(interval.balancedAccuracy, metrics.balancedAccuracy, `${variant} bootstrap balanced accuracy`);
  close(interval.realRecall, metrics.realRecall, `${variant} bootstrap real recall`);
  close(interval.syntheticRecall, metrics.syntheticRecall, `${variant} bootstrap synthetic recall`);
  requireCondition(interval.realCount === 300 && interval.realClusters === 300 && interval.realClusterSize === 1 &&
    interval.syntheticCount === 300 && interval.syntheticClusters === 300 && interval.syntheticClusterSize === 1,
  `${variant} bootstrap cluster design changed`);
  requireCondition(interval.lower95 >= testGates.minimumBalancedAccuracyLower95 &&
    interval.realRecallLower95 >= testGates.minimumRealRecallLower95 &&
    interval.syntheticRecallLower95 >= testGates.minimumSyntheticRecallLower95 &&
    Object.values(metrics.syntheticRecallBySource).every((recall) =>
      recall >= testGates.minimumSyntheticRecallPerFamily),
  `${variant} failed the frozen confirmatory acceptance gates`);
}
requireCondition(confirmatoryCompletion.files?.[path.basename(CONFIRMATORY_SUMMARY)] ===
  digest(await readFile(CONFIRMATORY_SUMMARY)), "Confirmatory summary is not completion-marker-bound");

const bootstrapReplayRoot = await mkdtemp(path.join(os.tmpdir(), "prooflens-bootstrap-replay-"));
try {
  const replayPath = path.join(bootstrapReplayRoot, "bootstrap.json");
  execFileSync(process.env.PROOFLENS_PYTHON ?? "python3", [
    "benchmark/bootstrap_ci.py",
    "--predictions", ...VARIANTS.map((variant) => CONFIRMATORY_PREDICTIONS[variant]),
    "--manifest", EXPECTED_MANIFESTS.test.path,
    "--expected-manifest-sha256", EXPECTED_MANIFESTS.test.sha256,
    "--raw-threshold", String(rawThreshold),
    "--seed", String(testGates.seed),
    "--replicates", String(testGates.replicates),
    "--output", replayPath,
  ], { stdio: "pipe", maxBuffer: 8 * 1024 * 1024 });
  requireCondition(jsonEqual(JSON.parse(await readFile(replayPath, "utf8")), bootstrap),
    "Committed confirmatory bootstrap intervals do not recompute deterministically");
} finally {
  await rm(bootstrapReplayRoot, { recursive: true, force: true });
}

const webSummary = JSON.parse(await readFile(WEB_NEGATIVE_SUMMARY, "utf8"));
const webCompletion = JSON.parse(await readFile(WEB_NEGATIVE_COMPLETION, "utf8"));
const webWilson = JSON.parse(await readFile(WEB_NEGATIVE_WILSON, "utf8"));
requireCondition(webSummary.schemaVersion === 2 && webSummary.protocol === "web-negative" &&
  webSummary.commitMarkerPublication === true && webSummary.allVariantsRequired === true &&
  webSummary.model.sha256 === modelLock.sha256 &&
  webSummary.dataset.sha256 === EXPECTED_MANIFESTS.webNegative.sha256 && webSummary.dataset.items === 319 &&
  webSummary.threshold.raw === rawThreshold && webSummary.threshold.calibrationSha256 === digest(calibrationBytes),
"Web-negative summary input bindings changed");
requireCondition(webWilson.schemaVersion === 2 && webWilson.confidenceLevel === webNegativeGates.confidenceLevel &&
  webWilson.manifestSha256 === EXPECTED_MANIFESTS.webNegative.sha256 && webWilson.rawProbabilityThreshold === rawThreshold,
"Web-negative Wilson contract changed");
requireCondition(webCompletion.schemaVersion === 1 && webCompletion.protocol === "web-negative" &&
  webCompletion.modelSha256 === modelLock.sha256 &&
  webCompletion.manifestSha256 === EXPECTED_MANIFESTS.webNegative.sha256 &&
  webCompletion.calibrationSha256 === digest(calibrationBytes),
"Web-negative completion marker input bindings changed");
const webById = new Map(manifests.webNegative.map((row) => [row.id, row]));
for (const variant of VARIANTS) {
  const file = WEB_NEGATIVE_PREDICTIONS[variant];
  const bytes = await readFile(file);
  requireCondition(webCompletion.files?.[path.basename(file)] === digest(bytes),
    `${variant} web-negative predictions are not completion-marker-bound`);
  requireCondition(webWilson.variants[variant]?.predictionsSha256 === digest(bytes),
    `${variant} web-negative predictions are not interval-bound`);
  const rows = parseJsonLines(bytes, file);
  requireCondition(rows.length === 319 && new Set(rows.map((row) => row.id)).size === 319 &&
    rows.every((row) => {
    const item = webById.get(row.id);
    return item && row.variant === variant && row.label === 0 && row.source === item.source &&
      row.groupId === item.groupId && predictionIsNumericallyConsistent(row);
  }), `${file} does not match the frozen web-negative manifest`);
  const metrics = negativeMetrics(rows, rawThreshold);
  requireCondition(jsonEqual(metrics, webSummary.variants[variant]),
    `${variant} web-negative summary metrics do not recompute`);
  const falsePositives = rows.filter((row) => row.rawProbability >= rawThreshold).length;
  const [lower, upper] = wilson(falsePositives, rows.length);
  const interval = webWilson.variants[variant];
  close(interval.falsePositiveRate, metrics.falsePositiveRate, `${variant} web-negative FPR`);
  close(interval.lower95, lower, `${variant} web-negative lower interval`);
  close(interval.upper95, upper, `${variant} web-negative upper interval`);
  requireCondition(interval.upper95 <= webNegativeGates.maximumOverallFalsePositiveRateUpper95,
    `${variant} overall web-negative upper FPR exceeds its frozen gate`);
  for (const [source, sourceMetrics] of Object.entries(metrics.bySource)) {
    const selected = rows.filter((row) => row.source === source);
    const sourceFalsePositives = selected.filter((row) => row.rawProbability >= rawThreshold).length;
    const [sourceLower, sourceUpper] = wilson(sourceFalsePositives, selected.length);
    const sourceInterval = interval.bySource[source];
    requireCondition(sourceInterval.count === selected.length, `${variant}/${source} count changed`);
    close(sourceInterval.falsePositiveRate, sourceMetrics.falsePositiveRate, `${variant}/${source} FPR`);
    close(sourceInterval.lower95, sourceLower, `${variant}/${source} lower interval`);
    close(sourceInterval.upper95, sourceUpper, `${variant}/${source} upper interval`);
    requireCondition(sourceInterval.upper95 <= webNegativeGates.maximumPerSourceFalsePositiveRateUpper95,
      `${variant}/${source} upper FPR exceeds its frozen gate`);
  }
}
requireCondition(webCompletion.files?.[path.basename(WEB_NEGATIVE_SUMMARY)] === digest(await readFile(WEB_NEGATIVE_SUMMARY)),
  "Web-negative summary is not completion-marker-bound");

const replayVerification = JSON.parse(await readFile(
  "benchmark/evidence/evaluation/replay-verification.json",
  "utf8",
));
const replayBoundFiles = [
  "weights/prooflens-cf384.onnx",
  "benchmark/evidence/large/calibration.json",
  "benchmark/evaluate.py",
  "benchmark/evaluation_contract.py",
  "benchmark/bootstrap_ci.py",
  "benchmark/bootstrap_fpr.py",
  "benchmark/prediction_contract.py",
  "benchmark/verify_evaluation_evidence.py",
  "benchmark/evidence/evaluation/pre-score-freeze-v2.json",
  ...Object.values(VALIDATION_PREDICTIONS), VALIDATION_SUMMARY, VALIDATION_COMPLETION,
  ...Object.values(CONFIRMATORY_PREDICTIONS), CONFIRMATORY_SUMMARY, CONFIRMATORY_COMPLETION,
  ...Object.values(WEB_NEGATIVE_PREDICTIONS), WEB_NEGATIVE_SUMMARY, WEB_NEGATIVE_COMPLETION,
  CONFIRMATORY_BOOTSTRAP,
  WEB_NEGATIVE_WILSON,
];
const calibrationSha256 = digest(calibrationBytes);
const evaluatorCommand = (protocol, dataRoot, manifest, manifestSha256, outputDir) => [
  "benchmark/evaluate.py",
  "--model", "weights/prooflens-cf384.onnx",
  "--expected-model-sha256", modelLock.sha256,
  "--data-root", dataRoot,
  "--manifest", manifest,
  "--expected-manifest-sha256", manifestSha256,
  "--output-dir", outputDir,
  "--protocol", protocol,
  "--batch-size", "16",
  "--execution-provider", "cpu",
  "--calibration", "benchmark/evidence/large/calibration.json",
  "--expected-calibration-sha256", calibrationSha256,
  "--verify-existing",
];
const expectedReplayCommands = [
  evaluatorCommand(
    "validation", "benchmark/data/modern-head", EXPECTED_MANIFESTS.validation.path,
    EXPECTED_MANIFESTS.validation.sha256, "benchmark/evidence/evaluation/validation",
  ),
  evaluatorCommand(
    "confirmatory", "benchmark/data", EXPECTED_MANIFESTS.test.path,
    EXPECTED_MANIFESTS.test.sha256, "benchmark/evidence/evaluation/confirmatory",
  ),
  evaluatorCommand(
    "web-negative", "benchmark/data/web-negative", EXPECTED_MANIFESTS.webNegative.path,
    EXPECTED_MANIFESTS.webNegative.sha256, "benchmark/evidence/evaluation/web-negative",
  ),
  [
    "benchmark/bootstrap_ci.py",
    "--predictions", ...VARIANTS.map((variant) => CONFIRMATORY_PREDICTIONS[variant]),
    "--manifest", EXPECTED_MANIFESTS.test.path,
    "--expected-manifest-sha256", EXPECTED_MANIFESTS.test.sha256,
    "--raw-threshold", String(rawThreshold),
    "--seed", "20260813",
    "--replicates", "20000",
    "--output", CONFIRMATORY_BOOTSTRAP,
    "--verify-existing",
  ],
  [
    "benchmark/bootstrap_fpr.py",
    "--predictions", ...VARIANTS.map((variant) => WEB_NEGATIVE_PREDICTIONS[variant]),
    "--manifest", EXPECTED_MANIFESTS.webNegative.path,
    "--expected-manifest-sha256", EXPECTED_MANIFESTS.webNegative.sha256,
    "--raw-threshold", String(rawThreshold),
    "--output", WEB_NEGATIVE_WILSON,
    "--verify-existing",
  ],
];
requireCondition(replayVerification.schemaVersion === 1 &&
  replayVerification.mode ===
    "byte-identical replay of immutable validation, confirmatory, web-negative, bootstrap, and Wilson evidence" &&
  replayVerification.modelSha256 === modelLock.sha256 &&
  replayVerification.calibrationSha256 === calibrationSha256 &&
  replayVerification.executionProvider === "cpu" && replayVerification.batchSize === 16 &&
  jsonEqual(replayVerification.commands, expectedReplayCommands) &&
  jsonEqual(Object.keys(replayVerification.files).sort(), [...replayBoundFiles].sort()) &&
  replayBoundFiles.every((file) => /^[a-f0-9]{64}$/u.test(replayVerification.files?.[file] ?? "")),
"Local evaluation replay receipt contract changed");
for (const file of replayBoundFiles) {
  requireCondition(replayVerification.files?.[file] === digest(await readFile(file)),
    `Local evaluation replay receipt is stale: ${file}`);
}

const parityIds = JSON.parse(await readFile("benchmark/manifests/parity-ids.json", "utf8"));
const expectedParityIds = [];
for (const label of [0, 1]) {
  const candidates = manifests.test.filter((row) => row.label === label)
    .sort((left, right) => priority(left.id).localeCompare(priority(right.id)) || left.id.localeCompare(right.id));
  expectedParityIds.push(...candidates.slice(0, 30).map((row) => row.id));
}
requireCondition(jsonEqual(parityIds, expectedParityIds), "Browser-parity IDs are not the deterministic 30-per-class subset");
const parity = JSON.parse(await readFile("artifacts/browser-parity.json", "utf8"));
requireCondition(parity.schemaVersion >= 2 && parity.modelSha256 === modelLock.sha256 && parity.samples === 60 &&
  parity.classCounts.real === 30 && parity.classCounts.synthetic === 30 && parity.cleanProfile === true &&
  parity.offline === true && parity.networkRequests.length === 0 &&
  parity.providerCounts.webgpu === 60 && parity.browserMetrics.balancedAccuracy >= 0.75 &&
  parity.decisionAgreement >= 0.95 && parity.meanAbsoluteProbabilityDifference <= 0.05 &&
  parity.maximumAbsoluteProbabilityDifference <= 0.25,
"Browser parity failed its frozen runtime-agreement gates");

const browserEvidence = {};
for (const [provider, file] of Object.entries(CHROME_E2E)) {
  const report = JSON.parse(await readFile(file, "utf8"));
  requireCondition(report.schemaVersion === 6 && report.modelSha256 === modelLock.sha256 &&
    report.provider === provider && report.cleanProfile === true && report.persistedModelAfterRestart === true &&
    report.serverStoppedBeforeAnalysis === true && report.browserOfflineBeforeAnalysis === true &&
    jsonEqual(report.postCutoffNetworkRequests, []) && report.setupProgressAccessibleName === true &&
    report.setupProgressAdvanced === true && report.popupCaveatVisible === true &&
    report.setupInitialFailureRecovered === true &&
    report.popupUnsupportedGuard === true && report.popupSupportedPageControls === true &&
    report.popupTemporaryLabelsReset === true && report.popupSavedSiteStatePersisted === true &&
    report.popupRescanFeedback === true && report.popupRescanWork?.acceptedAfter > report.popupRescanWork?.acceptedBefore &&
    report.popupFailureStateTruthful === true && report.popupCrossOriginMutationRejected === true &&
    report.popupInitializationNavigationRejected === true &&
    report.numericScore === true &&
    report.modelStateFixtures?.likelyAi?.classification === "likely-ai" &&
    report.modelStateFixtures?.belowThreshold?.classification === "not-flagged" &&
    report.reducedMotionSuppressed === true && report.closedShadowRoot === true &&
    report.overlayRemovalRecovered === true && report.cssomInFlightStaleResponseRejected === true &&
    report.boundedTargetAdmission?.lateTargetAfter5000Recovered === true &&
    report.boundedTargetAdmission?.staticTargetAfter5000Discovered === true &&
    report.boundedTargetAdmission?.staticFullDocumentTraversalCompleted === true &&
    report.boundedTargetAdmission?.visibleTargetEvictedOffscreenAtCap === true &&
    report.boundedTargetAdmission?.viableTargetEvictedBrokenVisibleAtCap === true &&
    report.boundedTargetAdmission?.linearAdmissionPriorityBounded === true &&
    report.boundedTargetAdmission?.pendingDeferredReconciliations === 0 &&
    report.boundedTargetAdmission?.mutationOverflowRecoveryPending === false &&
    report.boundedTargetAdmission?.synchronousMutationReconciliations === 0 &&
    report.trackedSourceWorktreeDirty === false && /^[a-f0-9]{40}$/u.test(report.testedGitHead) &&
    /^[a-f0-9]{40}$/u.test(report.testedGitTree) && /^[a-f0-9]{64}$/u.test(report.archiveSha256),
  `${provider} Chrome E2E receipt failed its release contract`);
  const testedTree = execFileSync("git", ["rev-parse", `${report.testedGitHead}^{tree}`], { encoding: "utf8" }).trim();
  requireCondition(testedTree === report.testedGitTree, `${provider} Chrome E2E source tree binding changed`);
  const actualScreenshots = {};
  for (const [kind, suffix] of Object.entries({
    modelState: "states",
    narrow: "narrow",
    smallTarget: "small-target",
  })) {
    const bytes = await readFile(`artifacts/chrome-e2e-${provider}-${suffix}.png`);
    actualScreenshots[kind] = { sha256: digest(bytes), ...pngDimensions(bytes) };
  }
  validateBrowserGeometryEvidence(report.geometryEvidence, actualScreenshots);
  browserEvidence[provider] = report;
}
requireCondition(browserEvidence.wasm.testedGitHead === browserEvidence.webgpu.testedGitHead &&
  browserEvidence.wasm.testedGitTree === browserEvidence.webgpu.testedGitTree &&
  browserEvidence.wasm.archiveSha256 === browserEvidence.webgpu.archiveSha256,
"WASM and WebGPU Chrome evidence do not bind the same source and archive");
requireCondition(digest(await readFile("release/prooflens.zip")) === browserEvidence.wasm.archiveSha256,
  "Chrome E2E receipts do not bind the current release archive");
const currentHead = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
if (currentHead !== browserEvidence.wasm.testedGitHead) {
  const evidenceDelta = execFileSync(
    "git",
    ["diff", "--no-renames", "--name-only", `${browserEvidence.wasm.testedGitHead}..${currentHead}`],
    { encoding: "utf8" },
  ).trim().split("\n").filter(Boolean);
  requireCondition(evidenceDelta.length > 0 && evidenceDelta.every((file) => file.startsWith("artifacts/")),
    "Chrome evidence was not committed as an artifacts-only child of its tested source");
}

console.log(JSON.stringify({
  manifests: Object.fromEntries(Object.entries(EXPECTED_MANIFESTS).map(([name, value]) => [name, {
    rows: value.rows, sha256: value.sha256,
  }])),
  familyOverlap: 0,
  promptGroupOverlap: 0,
  legacyEvaluationOverlap: 0,
  confirmatoryLower95Gates: "pass",
  webNegativeUpper95Gates: "pass",
  browserParity: "pass",
  chromeProviders: Object.keys(browserEvidence),
  policy: "pass",
}));
