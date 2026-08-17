import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { pngDimensions, validateBrowserGeometryEvidence } from "./browser-geometry-contract.mjs";

const VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];
const MODEL_PATH = "weights/prooflens-cf384.onnx";
const CALIBRATION_PATH = "benchmark/evidence/large/calibration.json";
const RECIPE_PATH = "benchmark/large/recipe.json";
const FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze-v3.json";
const REPLAY_PATH = "benchmark/evidence/evaluation/replay-verification-v2.json";
const PARITY_IDS_PATH = "benchmark/manifests/parity-ids-v2.json";
const HISTORICAL_INDEX_PATH = "benchmark/manifests/historical-perceptual-exclusions-v1.json.gz";
const SELECTION_PATH = "benchmark/manifests/replacement-v2-selection.json";
const EXPECTED = {
  confirmatory: {
    protocol: "confirmatory-v2",
    manifest: "benchmark/manifests/test-v2.jsonl",
    manifestSha256: "773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8",
    outputDir: "benchmark/evidence/evaluation/confirmatory-v2",
    name: "prooflens-confirmatory-v2",
    rows: 600,
    labels: { 0: 300, 1: 300 },
    sources: { "coxy7-infinity": 300, "stockimages-cc0": 300 },
  },
  webNegative: {
    protocol: "web-negative-v2",
    manifest: "benchmark/manifests/web-negative-v2.jsonl",
    manifestSha256: "6a1287bae6826811c81cbebab79a1bc6abb475fde70c9aa1529c390ed97014c9",
    outputDir: "benchmark/evidence/evaluation/web-negative-v2",
    name: "prooflens-web-negative-v2",
    rows: 319,
    labels: { 0: 319 },
    sources: { "stockimages-cc0": 319 },
  },
};
const FAILED_ROOT = "benchmark/evidence/evaluation/confirmatory";
const FAILED_RECEIPT = `${FAILED_ROOT}/failed-evaluation.json`;
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

function predictionIsNumericallyConsistent(row, threshold) {
  const expected = sigmoid(row.logit);
  return Number.isFinite(row.logit) && Number.isFinite(row.rawProbability) &&
    row.rawProbability >= 0 && row.rawProbability <= 1 &&
    Math.abs(expected - row.rawProbability) <= 2e-12 &&
    (expected >= threshold) === (row.rawProbability >= threshold);
}

function countsBy(rows, field) {
  return Object.fromEntries([...new Set(rows.map((row) => String(row[field])))].sort().map((value) => [
    value, rows.filter((row) => String(row[field]) === value).length,
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

function predictionPaths(config) {
  return Object.fromEntries(VARIANTS.map((variant) => [
    variant, `${config.outputDir}/${config.name}-${variant}-predictions.jsonl`,
  ]));
}

const modelLockBytes = await readFile("model-lock.json");
const modelLock = JSON.parse(modelLockBytes);
const modelBytes = await readFile(MODEL_PATH);
const calibrationBytes = await readFile(CALIBRATION_PATH);
const calibration = JSON.parse(calibrationBytes);
const recipeBytes = await readFile(RECIPE_PATH);
const recipe = JSON.parse(recipeBytes);
const freezeBytes = await readFile(FREEZE_PATH);
const freeze = JSON.parse(freezeBytes);
requireCondition(digest(modelBytes) === modelLock.sha256 && modelBytes.length === modelLock.bytes,
  "Packaged model bytes do not match model-lock.json");
requireCondition(digest(calibrationBytes) === modelLock.trainingEvidence?.calibrationSha256 &&
  digest(recipeBytes) === modelLock.trainingEvidence?.recipeSha256 &&
  calibration.modelSha256 === modelLock.sha256 && calibration.displayThreshold === 0.65 &&
  calibration.slope === 1 && Number.isFinite(calibration.rawProbabilityThreshold) &&
  calibration.rawProbabilityThreshold > 0 && calibration.rawProbabilityThreshold < 1,
"Model, calibration, and training recipe bindings changed");
requireCondition(freeze.schemaVersion === 4 && freeze.generation === 3 &&
  freeze.replacementProtocol?.confirmatory?.manifestSha256 === EXPECTED.confirmatory.manifestSha256 &&
  freeze.replacementProtocol?.webNegative?.manifestSha256 === EXPECTED.webNegative.manifestSha256,
"V3 replacement protocol receipt changed");
const rawThreshold = calibration.rawProbabilityThreshold;

const failed = JSON.parse(await readFile(FAILED_RECEIPT, "utf8"));
requireCondition(failed.schemaVersion === 1 && failed.status === "failed-numeric-contract" &&
  failed.acceptanceEligible === false && failed.holdoutConsumed === true &&
  failed.modelSha256 === modelLock.sha256 && failed.calibrationSha256 === digest(calibrationBytes) &&
  failed.numericContract?.rows === 2400 && failed.numericContract?.violations === 2231 &&
  failed.numericContract?.tolerance === 2e-12 && failed.failure?.bootstrapPublished === false &&
  failed.failure?.webNegativeInferenceStarted === false,
"Consumed v1 failure disclosure changed or became acceptance-eligible");
for (const [name, expectedHash] of Object.entries(failed.files)) {
  requireCondition(digest(await readFile(`${FAILED_ROOT}/${name}`)) === expectedHash,
    `Consumed v1 evidence changed: ${name}`);
}
requireCondition(!existsSync(`${FAILED_ROOT}/bootstrap.json`) &&
  !existsSync("benchmark/evidence/evaluation/web-negative"),
"The failed v1 run acquired acceptance intervals or web-negative output");

const manifests = {};
for (const [key, expected] of Object.entries(EXPECTED)) {
  const bytes = await readFile(expected.manifest);
  requireCondition(digest(bytes) === expected.manifestSha256, `${expected.manifest} SHA-256 changed`);
  const rows = parseJsonLines(bytes, expected.manifest);
  requireCondition(rows.length === expected.rows && new Set(rows.map((row) => row.id)).size === rows.length &&
    new Set(rows.map((row) => row.imageSha256)).size === rows.length &&
    jsonEqual(countsBy(rows, "label"), expected.labels) && jsonEqual(countsBy(rows, "source"), expected.sources),
  `${expected.manifest} allocation changed`);
  const expectedSplit = key === "confirmatory" ? "confirmatory-v2" : "web-negative-v2";
  requireCondition(rows.every((row) => row.split === expectedSplit && /^[a-f0-9]{64}$/u.test(row.imageSha256) &&
    /^[a-f0-9]{16}$/u.test(row.perceptualDhash64)), `${expected.manifest} fields changed`);
  manifests[key] = rows;
}
const confirmIds = new Set(manifests.confirmatory.map((row) => row.id));
const confirmHashes = new Set(manifests.confirmatory.map((row) => row.imageSha256));
requireCondition(manifests.webNegative.every((row) => !confirmIds.has(row.id) && !confirmHashes.has(row.imageSha256)),
  "Replacement confirmatory and web-negative pixels overlap");
const historicalBytes = await readFile(HISTORICAL_INDEX_PATH);
const historical = JSON.parse(gunzipSync(historicalBytes));
const historicalIds = new Set(historical.items.map((row) => row.id));
const historicalHashes = new Set(historical.items.map((row) => row.imageSha256));
requireCondition(historical.schemaVersion === 1 && historical.items.length === 106_019 &&
  historical.maximumHammingDistance === 8 && [...manifests.confirmatory, ...manifests.webNegative].every((row) =>
    !historicalIds.has(row.id) && !historicalHashes.has(row.imageSha256)),
"Replacement rows overlap the frozen historical index");
const selectionBytes = await readFile(SELECTION_PATH);
const selection = JSON.parse(selectionBytes);
requireCondition(digest(selectionBytes) === "4417bf34db53993c2ccf459a18947c92a52118b85a6f740740d981bb8e223f3c" &&
  selection.schemaVersion === 2 && selection.scoreBlindSelection === true &&
  selection.observedV1ResultsUsed === false && selection.confirmatory.manifestSha256 === EXPECTED.confirmatory.manifestSha256 &&
  selection.webNegative.manifestSha256 === EXPECTED.webNegative.manifestSha256 &&
  selection.overlap?.retainedPerceptualMatches === 0,
"Score-blind replacement selection evidence changed");

const testGates = recipe.testGates;
const webNegativeGates = recipe.webNegativeGates;
requireCondition(testGates.method ===
  "Class-stratified one-image-cluster bootstrap with variant-derived deterministic RNG" &&
  testGates.confidenceLevel === 0.95 && testGates.seed === 20_260_813 && testGates.replicates === 20_000 &&
  testGates.minimumBalancedAccuracyLower95 === 0.75 && testGates.minimumRealRecallLower95 === 0.75 &&
  testGates.minimumSyntheticRecallLower95 === 0.75 && testGates.minimumSyntheticRecallPerFamily === 0.60 &&
  webNegativeGates.method === "Wilson score interval for false-positive proportions" &&
  webNegativeGates.confidenceLevel === 0.95 &&
  webNegativeGates.maximumOverallFalsePositiveRateUpper95 === 0.10 &&
  webNegativeGates.maximumPerSourceFalsePositiveRateUpper95 === 0.20,
"Frozen statistical gates changed");

const confirmPredictions = predictionPaths(EXPECTED.confirmatory);
const confirmSummaryPath = `${EXPECTED.confirmatory.outputDir}/${EXPECTED.confirmatory.name}-summary.json`;
const confirmCompletionPath = `${EXPECTED.confirmatory.outputDir}/${EXPECTED.confirmatory.name}-complete.json`;
const confirmBootstrapPath = `${EXPECTED.confirmatory.outputDir}/bootstrap.json`;
const confirmSummary = JSON.parse(await readFile(confirmSummaryPath, "utf8"));
const confirmCompletion = JSON.parse(await readFile(confirmCompletionPath, "utf8"));
const confirmBootstrap = JSON.parse(await readFile(confirmBootstrapPath, "utf8"));
requireCondition(confirmSummary.schemaVersion === 3 && confirmSummary.protocol === EXPECTED.confirmatory.protocol &&
  confirmSummary.runtime?.probabilityArithmetic === "binary64 sigmoid from the recorded ONNX logit" &&
  confirmSummary.model.sha256 === modelLock.sha256 && confirmSummary.model.bytes === modelLock.bytes &&
  confirmSummary.dataset.sha256 === EXPECTED.confirmatory.manifestSha256 && confirmSummary.dataset.items === 600 &&
  confirmSummary.threshold.raw === rawThreshold && confirmSummary.threshold.calibrationSha256 === digest(calibrationBytes),
"Replacement confirmatory summary binding changed");
requireCondition(confirmCompletion.schemaVersion === 1 && confirmCompletion.protocol === EXPECTED.confirmatory.protocol &&
  confirmCompletion.modelSha256 === modelLock.sha256 &&
  confirmCompletion.manifestSha256 === EXPECTED.confirmatory.manifestSha256 &&
  confirmCompletion.calibrationSha256 === digest(calibrationBytes),
"Replacement confirmatory completion marker changed");
requireCondition(confirmBootstrap.schemaVersion === 3 && confirmBootstrap.manifestSha256 === EXPECTED.confirmatory.manifestSha256 &&
  confirmBootstrap.seed === testGates.seed && confirmBootstrap.replicates === testGates.replicates &&
  confirmBootstrap.rawProbabilityThreshold === rawThreshold && confirmBootstrap.method === testGates.method,
"Replacement confirmatory bootstrap contract changed");
const confirmById = new Map(manifests.confirmatory.map((row) => [row.id, row]));
for (const variant of VARIANTS) {
  const file = confirmPredictions[variant];
  const bytes = await readFile(file);
  const rows = parseJsonLines(bytes, file);
  requireCondition(confirmCompletion.files?.[path.basename(file)] === digest(bytes) &&
    confirmBootstrap.variants?.[variant]?.predictionsSha256 === digest(bytes) && rows.length === 600 &&
    new Set(rows.map((row) => row.id)).size === 600 && rows.every((row) => {
      const item = confirmById.get(row.id);
      return item && row.variant === variant && row.label === item.label && row.source === item.source &&
        row.groupId === item.groupId && predictionIsNumericallyConsistent(row, rawThreshold);
    }), `${file} is stale, incomplete, or numerically invalid`);
  const metrics = predictionMetrics(rows, rawThreshold);
  requireCondition(jsonEqual(metrics, confirmSummary.variants[variant]),
    `${variant} replacement confirmatory metrics do not recompute`);
  const interval = confirmBootstrap.variants[variant];
  close(interval.balancedAccuracy, metrics.balancedAccuracy, `${variant} balanced accuracy`);
  close(interval.realRecall, metrics.realRecall, `${variant} real recall`);
  close(interval.syntheticRecall, metrics.syntheticRecall, `${variant} synthetic recall`);
  requireCondition(interval.realCount === 300 && interval.realClusters === 300 && interval.realClusterSize === 1 &&
    interval.syntheticCount === 300 && interval.syntheticClusters === 300 && interval.syntheticClusterSize === 1 &&
    interval.lower95 >= testGates.minimumBalancedAccuracyLower95 &&
    interval.realRecallLower95 >= testGates.minimumRealRecallLower95 &&
    interval.syntheticRecallLower95 >= testGates.minimumSyntheticRecallLower95 &&
    Object.values(metrics.syntheticRecallBySource).every((recall) => recall >= testGates.minimumSyntheticRecallPerFamily),
  `${variant} failed the frozen replacement confirmatory gates`);
}
requireCondition(confirmCompletion.files?.[path.basename(confirmSummaryPath)] === digest(await readFile(confirmSummaryPath)),
  "Replacement confirmatory summary is not completion-marker-bound");

const bootstrapReplayRoot = await mkdtemp(path.join(os.tmpdir(), "prooflens-bootstrap-v2-replay-"));
try {
  const replayPath = path.join(bootstrapReplayRoot, "bootstrap.json");
  execFileSync(process.env.PROOFLENS_PYTHON ?? "python3", [
    "benchmark/bootstrap_ci.py", "--predictions", ...VARIANTS.map((variant) => confirmPredictions[variant]),
    "--manifest", EXPECTED.confirmatory.manifest,
    "--expected-manifest-sha256", EXPECTED.confirmatory.manifestSha256,
    "--raw-threshold", String(rawThreshold), "--seed", String(testGates.seed),
    "--replicates", String(testGates.replicates), "--output", replayPath,
  ], { stdio: "pipe", maxBuffer: 8 * 1024 * 1024 });
  requireCondition(jsonEqual(JSON.parse(await readFile(replayPath, "utf8")), confirmBootstrap),
    "Replacement bootstrap intervals do not recompute deterministically");
} finally {
  await rm(bootstrapReplayRoot, { recursive: true, force: true });
}

const webPredictions = predictionPaths(EXPECTED.webNegative);
const webSummaryPath = `${EXPECTED.webNegative.outputDir}/${EXPECTED.webNegative.name}-summary.json`;
const webCompletionPath = `${EXPECTED.webNegative.outputDir}/${EXPECTED.webNegative.name}-complete.json`;
const webWilsonPath = `${EXPECTED.webNegative.outputDir}/wilson.json`;
const webSummary = JSON.parse(await readFile(webSummaryPath, "utf8"));
const webCompletion = JSON.parse(await readFile(webCompletionPath, "utf8"));
const webWilson = JSON.parse(await readFile(webWilsonPath, "utf8"));
requireCondition(webSummary.schemaVersion === 3 && webSummary.protocol === EXPECTED.webNegative.protocol &&
  webSummary.runtime?.probabilityArithmetic === "binary64 sigmoid from the recorded ONNX logit" &&
  webSummary.model.sha256 === modelLock.sha256 && webSummary.dataset.sha256 === EXPECTED.webNegative.manifestSha256 &&
  webSummary.dataset.items === 319 && webSummary.threshold.raw === rawThreshold &&
  webSummary.threshold.calibrationSha256 === digest(calibrationBytes),
"Replacement web-negative summary binding changed");
requireCondition(webCompletion.schemaVersion === 1 && webCompletion.protocol === EXPECTED.webNegative.protocol &&
  webCompletion.modelSha256 === modelLock.sha256 && webCompletion.manifestSha256 === EXPECTED.webNegative.manifestSha256 &&
  webCompletion.calibrationSha256 === digest(calibrationBytes),
"Replacement web-negative completion marker changed");
requireCondition(webWilson.schemaVersion === 2 && webWilson.manifestSha256 === EXPECTED.webNegative.manifestSha256 &&
  webWilson.confidenceLevel === webNegativeGates.confidenceLevel && webWilson.rawProbabilityThreshold === rawThreshold,
"Replacement web-negative Wilson contract changed");
const webById = new Map(manifests.webNegative.map((row) => [row.id, row]));
for (const variant of VARIANTS) {
  const file = webPredictions[variant];
  const bytes = await readFile(file);
  const rows = parseJsonLines(bytes, file);
  requireCondition(webCompletion.files?.[path.basename(file)] === digest(bytes) &&
    webWilson.variants?.[variant]?.predictionsSha256 === digest(bytes) && rows.length === 319 &&
    new Set(rows.map((row) => row.id)).size === 319 && rows.every((row) => {
      const item = webById.get(row.id);
      return item && row.variant === variant && row.label === 0 && row.source === item.source &&
        row.groupId === item.groupId && predictionIsNumericallyConsistent(row, rawThreshold);
    }), `${file} is stale, incomplete, or numerically invalid`);
  const metrics = negativeMetrics(rows, rawThreshold);
  requireCondition(jsonEqual(metrics, webSummary.variants[variant]),
    `${variant} replacement web-negative metrics do not recompute`);
  const falsePositives = rows.filter((row) => row.rawProbability >= rawThreshold).length;
  const [lower, upper] = wilson(falsePositives, rows.length);
  const interval = webWilson.variants[variant];
  close(interval.falsePositiveRate, metrics.falsePositiveRate, `${variant} web FPR`);
  close(interval.lower95, lower, `${variant} web lower interval`);
  close(interval.upper95, upper, `${variant} web upper interval`);
  requireCondition(interval.upper95 <= webNegativeGates.maximumOverallFalsePositiveRateUpper95,
    `${variant} replacement web-negative upper FPR failed`);
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
      `${variant}/${source} replacement web-negative upper FPR failed`);
  }
}
requireCondition(webCompletion.files?.[path.basename(webSummaryPath)] === digest(await readFile(webSummaryPath)),
  "Replacement web-negative summary is not completion-marker-bound");

const replay = JSON.parse(await readFile(REPLAY_PATH, "utf8"));
const replayBoundFiles = [
  MODEL_PATH, CALIBRATION_PATH, "model-lock.json", RECIPE_PATH, FREEZE_PATH,
  "benchmark/evaluate.py", "benchmark/evaluation_contract.py", "benchmark/bootstrap_ci.py",
  "benchmark/bootstrap_fpr.py", "benchmark/prediction_contract.py",
  "benchmark/verify_evaluation_evidence.py", "benchmark/run_release_replay.py",
  EXPECTED.confirmatory.manifest, EXPECTED.webNegative.manifest, SELECTION_PATH, PARITY_IDS_PATH,
  ...Object.values(confirmPredictions), confirmSummaryPath, confirmCompletionPath,
  ...Object.values(webPredictions), webSummaryPath, webCompletionPath,
  confirmBootstrapPath, webWilsonPath,
];
const evaluatorCommand = (config) => [
  "benchmark/evaluate.py", "--model", MODEL_PATH, "--expected-model-sha256", modelLock.sha256,
  "--data-root", "benchmark/data/replacement-v2", "--manifest", config.manifest,
  "--expected-manifest-sha256", config.manifestSha256, "--output-dir", config.outputDir,
  "--protocol", config.protocol, "--batch-size", "16", "--execution-provider", "cpu",
  "--calibration", CALIBRATION_PATH, "--expected-calibration-sha256", digest(calibrationBytes),
  "--verify-existing",
];
const expectedReplayCommands = [
  evaluatorCommand(EXPECTED.confirmatory),
  evaluatorCommand(EXPECTED.webNegative),
  [
    "benchmark/bootstrap_ci.py", "--predictions", ...VARIANTS.map((variant) => confirmPredictions[variant]),
    "--manifest", EXPECTED.confirmatory.manifest,
    "--expected-manifest-sha256", EXPECTED.confirmatory.manifestSha256,
    "--raw-threshold", String(rawThreshold), "--seed", "20260813", "--replicates", "20000",
    "--output", confirmBootstrapPath, "--verify-existing",
  ],
  [
    "benchmark/bootstrap_fpr.py", "--predictions", ...VARIANTS.map((variant) => webPredictions[variant]),
    "--manifest", EXPECTED.webNegative.manifest,
    "--expected-manifest-sha256", EXPECTED.webNegative.manifestSha256,
    "--raw-threshold", String(rawThreshold), "--output", webWilsonPath, "--verify-existing",
  ],
];
requireCondition(replay.schemaVersion === 2 && replay.mode ===
  "byte-identical replay of replacement-v2 confirmatory, web-negative, bootstrap, and Wilson evidence" &&
  replay.modelSha256 === modelLock.sha256 && replay.calibrationSha256 === digest(calibrationBytes) &&
  replay.recipeSha256 === digest(recipeBytes) && replay.freezeReceiptSha256 === digest(freezeBytes) &&
  replay.executionProvider === "cpu" && replay.batchSize === 16 &&
  jsonEqual(replay.commands, expectedReplayCommands) &&
  jsonEqual(Object.keys(replay.files).sort(), [...replayBoundFiles].sort()),
"Replacement release replay receipt contract changed");
for (const file of replayBoundFiles) {
  requireCondition(replay.files[file] === digest(await readFile(file)), `Replacement replay is stale: ${file}`);
}

const parityIdsBytes = await readFile(PARITY_IDS_PATH);
requireCondition(digest(parityIdsBytes) === "0f0e72ac4bd91549af10a76c494138b6cf0c22328d904134b67be82d79badf99",
  "Replacement parity ID packet changed");
const parityIds = JSON.parse(parityIdsBytes);
const expectedParityIds = [];
for (const label of [0, 1]) {
  const candidates = manifests.confirmatory.filter((row) => row.label === label)
    .sort((left, right) => priority(left.id).localeCompare(priority(right.id)) || left.id.localeCompare(right.id));
  expectedParityIds.push(...candidates.slice(0, 30).map((row) => row.id));
}
requireCondition(jsonEqual(parityIds, expectedParityIds),
  "Browser parity IDs are not the pre-score deterministic replacement subset");
const confirmOriginalById = new Map(parseJsonLines(await readFile(confirmPredictions.original), confirmPredictions.original)
  .map((row) => [row.id, row]));
const parity = JSON.parse(await readFile("artifacts/browser-parity.json", "utf8"));
requireCondition(parity.schemaVersion === 2 && parity.modelSha256 === modelLock.sha256 && parity.samples === 60 &&
  parity.classCounts.real === 30 && parity.classCounts.synthetic === 30 && parity.cleanProfile === true &&
  parity.offline === true && parity.networkRequests.length === 0 && /^[a-f0-9]{64}$/u.test(parity.fixtureManifestSha256) &&
  parity.providerCounts.webgpu === 60 && parity.browserMetrics.balancedAccuracy >= 0.75 &&
  parity.decisionAgreement >= 0.95 && parity.meanAbsoluteProbabilityDifference <= 0.05 &&
  parity.maximumAbsoluteProbabilityDifference <= 0.25 && Array.isArray(parity.predictions) &&
  parity.predictions.length === 60 && parity.predictions.map((row) => row.id).join("\n") === parityIds.join("\n") &&
  parity.predictions.every((row) => {
    const item = confirmById.get(row.id);
    const reference = confirmOriginalById.get(row.id);
    return item && reference && row.label === item.label && row.source === item.source &&
      row.imageSha256 === item.imageSha256 && row.referenceRawProbability === reference.rawProbability;
  }), "Browser parity failed its replacement-v2 runtime-agreement gates");

const browserEvidence = {};
for (const [provider, file] of Object.entries(CHROME_E2E)) {
  const report = JSON.parse(await readFile(file, "utf8"));
  requireCondition(report.schemaVersion === 6 && report.modelSha256 === modelLock.sha256 &&
    report.provider === provider && report.cleanProfile === true && report.persistedModelAfterRestart === true &&
    report.serverStoppedBeforeAnalysis === true && report.browserOfflineBeforeAnalysis === true &&
    jsonEqual(report.postCutoffNetworkRequests, []) && report.setupProgressAccessibleName === true &&
    report.setupProgressAdvanced === true && report.popupCaveatVisible === false &&
    report.setupInitialFailureRecovered === true && report.popupUnsupportedGuard === true &&
    report.popupSupportedPageControls === true && report.popupTemporaryLabelsReset === true &&
    report.popupSavedSiteStatePersisted === true && report.popupRescanFeedback === true &&
    report.popupRescanWork?.acceptedAfter > report.popupRescanWork?.acceptedBefore &&
    report.popupFailureStateTruthful === true && report.popupCrossOriginMutationRejected === true &&
    report.popupInitializationNavigationRejected === true && report.numericScore === true &&
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
  for (const [kind, suffix] of Object.entries({ modelState: "states", narrow: "narrow", smallTarget: "small-target" })) {
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
    "git", ["diff", "--no-renames", "--name-only", `${browserEvidence.wasm.testedGitHead}..${currentHead}`],
    { encoding: "utf8" },
  ).trim().split("\n").filter(Boolean);
  requireCondition(evidenceDelta.length > 0 && evidenceDelta.every((file) => file.startsWith("artifacts/")),
    "Chrome evidence was not committed as an artifacts-only child of its tested source");
}

console.log(JSON.stringify({
  replacementConfirmatoryRows: manifests.confirmatory.length,
  replacementWebNegativeRows: manifests.webNegative.length,
  historicalExclusions: historical.items.length,
  v1FailurePreserved: true,
  confirmatoryLower95Gates: "pass",
  webNegativeUpper95Gates: "pass",
  browserParity: "pass",
  chromeProviders: Object.keys(browserEvidence),
  policy: "pass",
}));
