import { createHash } from "node:crypto";
import { TextDecoder } from "node:util";
import { completeDecisionThresholds } from "./m3-failure-contract.mjs";


export const M4 = Object.freeze({
  baseCommit: "439b2481dc88a887f8317be669096495760fbeb1",
  baseTree: "440931a595c87ca3d293f5a6f980c75169ddb899",
  upstreamSha256: "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47",
  upstreamBytes: 87_442_080,
  recipeSha256: "975c1588b1be33b126b63c49a8ab7623eedb7db07ef8e9d788c6cc64e7a9473d",
  sourceLocksSha256: "bf44ceba6f32d322de04f9fae994c0fed7fdcd00e2bcfff9de39c6d852a01394",
  seed: 20260815,
  pipelineVersion: 1,
  trainImages: 112_698,
  trainViews: 150_792,
  selectorImages: 600,
  selectorViews: 2_400,
  trainShards: 57,
  totalShards: 60,
  candidateCount: 12,
});
export const M4_VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];
export const M4_EXPECTED_ARGUMENTS = [
  "--model", "weights/prooflens-cf384.onnx",
  "--data-root", "benchmark/data/m4-head",
  "--train-manifest", "benchmark/data/m4-head/train-manifest.jsonl",
  "--selector-manifest", "benchmark/evidence/m4/validation-manifest.jsonl",
  "--m3-regression-data-root", "benchmark/data/m3-head",
  "--m3-regression-manifest", "benchmark/evidence/m3/validation-manifest.jsonl",
  "--m2-regression-data-root", "benchmark/data/m2-head",
  "--m2-regression-manifest", "benchmark/evidence/m2/validation-manifest.jsonl",
  "--selection-summary", "benchmark/evidence/m4/selection-summary.json",
  "--single-view-source", "diffusiondb-stable-diffusion",
  "--single-view-source", "open-images-train",
  "--execution-provider", "cpu",
  "--batch-size", "24",
  "--feature-shard-images", "2000",
  "--reextract-cached-features",
  "--output-dir", "benchmark/candidates/prooflens-cf384-m4",
];
export const M4_PUBLICATION_ROWS = [
  ["BENCHMARK.md", "M"], ["MODEL_CARD.md", "M"], ["README.md", "M"],
  ["benchmark/evidence/m4/calibration.json", "A"],
  ["benchmark/evidence/m4/candidate-grid.json", "A"],
  ["benchmark/evidence/m4/finalization-receipt.json", "A"],
  ["benchmark/evidence/m4/model-comparison.json", "A"],
  ["benchmark/evidence/m4/training-summary.json", "A"],
  ["model-lock.json", "M"], ["tests/fixtures/model-states/fixture-manifest.json", "M"],
  ["weights/README.md", "M"], ["weights/prooflens-cf384.onnx", "M"],
];

const HEX64 = /^[a-f0-9]{64}$/u;
const RUN_ID = /^[a-f0-9]{32}$/u;
const DECAYS = [0.003, 0.01, 0.03];
const ANCHORS = [0.01, 0.03, 0.1, 0.3];
export const M4_TENSOR_SHAPES = Object.freeze({
  "m4.adapter_in.bias": [64],
  "m4.adapter_in.weight": [64, 384],
  "m4.adapter_out.bias": [384],
  "m4.adapter_out.weight": [384, 64],
  "m4.feature_mean": [384],
  "m4.feature_std": [384],
});
export const M4_COMMON_SUMMARY_KEYS = [
  "schemaVersion", "pipelineVersion", "status", "seed", "commandArguments", "trainerSha256",
  "recipeSha256", "sourceLocksSha256", "selectionSummarySha256", "upstreamModelSha256",
  "trainManifestSha256", "selectorManifestSha256", "m3RegressionManifestSha256",
  "m2RegressionManifestSha256", "trainImages", "trainFeatureViews", "trainSourceCounts",
  "trainClassCounts", "selectorImages", "selectorFeatureViews", "selectorSourceCounts",
  "selectorClassCounts", "candidateTensorSealSha256", "candidateGridSha256",
  "zeroAdapterFeatureParityMaximumAbsoluteError", "featureConfigurationHashes",
  "featureShardEvidence", "freshFeatureRunId", "freshFeatureMarkerSha256", "sourceBalancedLoss",
  "anchorLossProtectedSources", "candidateCount", "validCandidateCount", "regressionOrder",
  "h3HoldoutScored", "h3PixelsRead", "selectionInfluencedByRegression", "environment",
];
const SUMMARY_KEYS = [
  ...M4_COMMON_SUMMARY_KEYS,
  "selectionLockSha256", "selectionLock", "selectedCandidate", "regressions",
  "freshFeatureRunComplete", "modelSha256", "modelBytes",
  "zeroAdapterImageParityMaximumAbsoluteError", "exportedCandidateImageParityMaximumAbsoluteError",
  "calibrationSha256",
];

export function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

export function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

export function jsonEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

export function exactKeys(value, keys, label) {
  requireCondition(value !== null && typeof value === "object" && !Array.isArray(value) &&
    jsonEqual(Object.keys(value).sort(), [...keys].sort()), `${label} keys changed`);
}

function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function validateTensorRecord(row, label) {
  exactKeys(row, ["candidateId", "weightDecay", "anchorCoefficient", "trainableParameters",
    "tensorSha256", "tensorShapes", "tensorDtypes", "tensorFloat32Base64"], label);
  requireCondition(DECAYS.includes(row.weightDecay) && ANCHORS.includes(row.anchorCoefficient) &&
    row.candidateId === `wd-${row.weightDecay.toFixed(3)}-anchor-${row.anchorCoefficient.toFixed(2)}` &&
    row.trainableParameters === 49_600, `${label} identity changed`);
  const names = Object.keys(M4_TENSOR_SHAPES);
  for (const [key, value] of [["tensorSha256", row.tensorSha256], ["tensorShapes", row.tensorShapes],
    ["tensorDtypes", row.tensorDtypes], ["tensorFloat32Base64", row.tensorFloat32Base64]]) {
    exactKeys(value, names, `${label} ${key}`);
  }
  for (const name of names) {
    const shape = M4_TENSOR_SHAPES[name];
    requireCondition(jsonEqual(row.tensorShapes[name], shape) && row.tensorDtypes[name] === "float32" &&
      HEX64.test(row.tensorSha256[name]) && typeof row.tensorFloat32Base64[name] === "string",
    `${label} tensor metadata changed: ${name}`);
    const bytes = Buffer.from(row.tensorFloat32Base64[name], "base64");
    const count = shape.reduce((product, value) => product * value, 1);
    const hash = createHash("sha256").update("<f4").update(JSON.stringify(shape)).update(bytes).digest("hex");
    requireCondition(bytes.toString("base64") === row.tensorFloat32Base64[name] && bytes.length === count * 4 &&
      hash === row.tensorSha256[name], `${label} tensor bytes changed: ${name}`);
    for (let offset = 0; offset < bytes.length; offset += 4) {
      const value = bytes.readFloatLE(offset);
      requireCondition(Number.isFinite(value) && (name !== "m4.feature_std" || value >= 1e-5),
        `${label} tensor value changed: ${name}`);
    }
  }
  return row;
}

export function validateM4TensorSeal(tensorSeal) {
  exactKeys(tensorSeal, ["schemaVersion", "createdBeforeSelectorEvaluation", "candidateCount", "candidates"],
    "M4 candidate tensor seal");
  requireCondition(tensorSeal.schemaVersion === 1 && tensorSeal.createdBeforeSelectorEvaluation === true &&
    tensorSeal.candidateCount === M4.candidateCount && Array.isArray(tensorSeal.candidates) &&
    tensorSeal.candidates.length === M4.candidateCount, "M4 candidate tensor seal changed");
  const sealedById = new Map();
  for (const row of tensorSeal.candidates) {
    validateTensorRecord(row, "M4 sealed candidate");
    requireCondition(!sealedById.has(row.candidateId), "M4 sealed candidate repeated");
    sealedById.set(row.candidateId, row);
  }
  requireCondition(sealedById.size === M4.candidateCount, "M4 sealed candidate count changed");
  return sealedById;
}

export function parseCanonicalJson(value, label = "M4 JSON") {
  const bytes = Buffer.isBuffer(value) ? Buffer.from(value) : Buffer.from(String(value), "utf8");
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${label} is not canonical UTF-8`, { cause: error });
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} is not valid JSON`, { cause: error });
  }
  requireCondition(bytes.equals(Buffer.from(`${JSON.stringify(parsed, null, 2)}\n`, "utf8")),
    `${label} bytes are not canonical JSON`);
  return parsed;
}

export function validateM4PublicationLockShape(lock) {
  exactKeys(lock, [
    "schemaVersion", "profile", "sourceCommit", "sourceTree", "baseCommit", "baseTree",
    "upstreamModelSha256", "upstreamModelBytes", "upstreamModelLockSha256", "trainerSha256",
    "recipeSha256", "sourceLocksSha256", "selectionSummarySha256", "candidateHashes",
    "candidateModelBytes", "selectionLock", "modelComparisonSha256", "adapterPatch",
    "finalizerSha256", "publicationContractSha256", "fixtureSelectorSha256",
    "documentationRendererSha256", "publicDocumentHashes", "fixtureManifestSha256",
    "candidateEvidenceJson", "publicationRows", "selectionInfluencedByRegression",
    "h3HoldoutScored", "h3PixelsRead",
  ], "M4 publication lock");
  exactKeys(lock.candidateHashes, ["model", "summary", "calibration", "grid", "selectionLock", "tensorSeal", "freshMarker"],
    "M4 publication lock candidate hashes");
  exactKeys(lock.publicDocumentHashes, ["README.md", "MODEL_CARD.md", "BENCHMARK.md"],
    "M4 publication lock document hashes");
  exactKeys(lock.candidateEvidenceJson, [
    "training-summary.json", "calibration.json", "candidate-grid.json", "selection-lock.json",
    "candidate-tensor-seal.json", "fresh-feature-run.json", "model-comparison.json", "fixture-manifest.json",
  ], "M4 publication lock candidate evidence");
  exactKeys(lock.adapterPatch, [
    "schemaVersion", "baseSha256", "candidateSha256", "candidateBytes", "featureTensor",
    "classifierNodeName", "classifierInputBefore", "classifierInputAfter", "addedInitializers", "addedNodes",
    "classifierNodeProtoSha256", "classifierNodeProtoBase64", "reconstructedCandidateSha256",
  ], "M4 adapter patch");
  requireCondition(Array.isArray(lock.adapterPatch.addedInitializers) && lock.adapterPatch.addedInitializers.length === 6 &&
    lock.adapterPatch.addedInitializers.every((row) => {
      exactKeys(row, ["name", "dimensions", "dataType", "tensorProtoSha256", "tensorProtoBase64",
        "rawDataSha256", "rawDataBase64"],
        "M4 adapter initializer patch");
      return true;
    }) && Array.isArray(lock.adapterPatch.addedNodes) && lock.adapterPatch.addedNodes.length === 7 &&
    lock.adapterPatch.addedNodes.every((row) => {
      exactKeys(row, ["name", "opType", "inputs", "outputs", "attributes", "nodeProtoSha256", "nodeProtoBase64"],
        "M4 adapter node patch");
      return true;
    }), "M4 adapter patch arrays changed");
  requireCondition(Array.isArray(lock.publicationRows) && lock.publicationRows.length === 12 &&
    lock.publicationRows.every((row) => {
      exactKeys(row, ["path", "status"], "M4 publication row");
      return true;
  }) && jsonEqual(lock.publicationRows.map((row) => [row.path, row.status]), M4_PUBLICATION_ROWS),
  "M4 publication rows changed");
  requireCondition(lock.schemaVersion === 1 && lock.profile === "m4" &&
    lock.baseCommit === M4.baseCommit && lock.baseTree === M4.baseTree &&
    lock.upstreamModelSha256 === M4.upstreamSha256 && lock.upstreamModelBytes === M4.upstreamBytes &&
    lock.recipeSha256 === M4.recipeSha256 && lock.sourceLocksSha256 === M4.sourceLocksSha256 &&
    lock.selectionInfluencedByRegression === false && lock.h3HoldoutScored === false && lock.h3PixelsRead === false,
  "M4 publication lock frozen boundary changed");
}

export function parseCanonicalM4PublicationLock(value) {
  const lock = parseCanonicalJson(value, "M4 publication lock");
  validateM4PublicationLockShape(lock);
  return lock;
}

function arrayDigestFloat32(bytes, count) {
  const hash = createHash("sha256");
  hash.update("<f4");
  hash.update(`[${count}]`);
  hash.update(bytes);
  return hash.digest("hex");
}

function decodeFloat32(row, base64Key, countKey, digestKey, label) {
  requireCondition(typeof row[base64Key] === "string" && Number.isInteger(row[countKey]) && row[countKey] > 0,
    `${label} header changed`);
  const bytes = Buffer.from(row[base64Key], "base64");
  requireCondition(bytes.toString("base64") === row[base64Key] && bytes.length === row[countKey] * 4 &&
    arrayDigestFloat32(bytes, row[countKey]) === row[digestKey], `${label} bytes or digest changed`);
  const values = Array.from({ length: row[countKey] }, (_, index) => bytes.readFloatLE(index * 4));
  requireCondition(values.every(Number.isFinite), `${label} contains non-finite values`);
  return values;
}

export function parseManifestMetadata(bytes, expected) {
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${expected.label} manifest is not UTF-8`, { cause: error });
  }
  const rows = text.split("\n").filter(Boolean).map((line) => JSON.parse(line));
  requireCondition(rows.length === expected.items && new Set(rows.map((row) => row.id)).size === rows.length &&
    new Set(rows.map((row) => row.imageSha256)).size === rows.length, `${expected.label} manifest identity changed`);
  const sourceCounts = {};
  const labels = [];
  const sources = [];
  const variants = [];
  rows.forEach((row, index) => {
    requireCondition(row.rowIndex === index && row.label === (expected.sources[row.source]?.label ?? null),
      `${expected.label} row order, source, or label changed`);
    sourceCounts[row.source] = (sourceCounts[row.source] ?? 0) + 1;
    for (let variant = 0; variant < M4_VARIANTS.length; variant += 1) {
      labels.push(row.label);
      sources.push(row.source);
      variants.push(variant);
    }
  });
  requireCondition(jsonEqual(sourceCounts, Object.fromEntries(Object.entries(expected.sources)
    .map(([source, value]) => [source, value.count]))), `${expected.label} source counts changed`);
  return { rows, labels, sources, variants };
}

export function validateM4FreshRunEvidence(summary, marker, { state, completedRegressions }) {
  exactKeys(marker, ["schemaVersion", "runId", "state", "context"], "M4 fresh feature marker");
  exactKeys(marker.context, ["pipelineVersion", "trainerSha256", "recipeSha256", "modelSha256",
    "trainManifestSha256", "selectorManifestSha256", "m3RegressionManifestSha256",
    "m2RegressionManifestSha256", "selectionSummarySha256", "featureConfigurationHashes",
    "featureBatchSize", "featureShardImages", "singleViewSources"], "M4 fresh feature context");
  exactKeys(summary.featureConfigurationHashes, ["training", "evaluation"],
    "M4 feature configuration hashes");
  requireCondition(marker.schemaVersion === 1 && RUN_ID.test(marker.runId) && marker.state === state &&
    marker.runId === summary.freshFeatureRunId && marker.context.pipelineVersion === M4.pipelineVersion &&
    marker.context.trainerSha256 === summary.trainerSha256 &&
    marker.context.recipeSha256 === summary.recipeSha256 &&
    marker.context.modelSha256 === summary.upstreamModelSha256 &&
    marker.context.trainManifestSha256 === summary.trainManifestSha256 &&
    marker.context.selectorManifestSha256 === summary.selectorManifestSha256 &&
    marker.context.m3RegressionManifestSha256 === summary.m3RegressionManifestSha256 &&
    marker.context.m2RegressionManifestSha256 === summary.m2RegressionManifestSha256 &&
    marker.context.selectionSummarySha256 === summary.selectionSummarySha256 &&
    jsonEqual(marker.context.featureConfigurationHashes, summary.featureConfigurationHashes) &&
    marker.context.featureBatchSize === 24 && marker.context.featureShardImages === 2_000 &&
    jsonEqual(marker.context.singleViewSources, ["diffusiondb-stable-diffusion", "open-images-train"]) &&
    HEX64.test(summary.featureConfigurationHashes.training) && HEX64.test(summary.featureConfigurationHashes.evaluation) &&
    summary.featureConfigurationHashes.training !== summary.featureConfigurationHashes.evaluation,
  "M4 fresh feature marker context changed");
  exactKeys(summary.environment, ["python", "numpy", "torch", "onnxRuntime", "pillow", "platform", "providers"],
    "M4 training environment");
  requireCondition(jsonEqual(summary.environment.providers, ["CPUExecutionProvider"]) &&
    ["python", "numpy", "torch", "onnxRuntime", "pillow", "platform"].every((key) =>
      typeof summary.environment[key] === "string" && summary.environment[key].length > 0),
  "M4 CPU training environment changed");
  const prefixes = [
    ["train", 57, M4.trainImages, M4.trainViews, summary.featureConfigurationHashes.training],
    ["selector", 1, M4.selectorImages, M4.selectorViews, summary.featureConfigurationHashes.evaluation],
    ...completedRegressions.slice(0, 1).map(() => ["regression-m3", 1, 600, 2_400,
      summary.featureConfigurationHashes.evaluation]),
    ...completedRegressions.slice(1, 2).map(() => ["regression-m2", 1, 900, 3_600,
      summary.featureConfigurationHashes.evaluation]),
  ];
  const expectedCount = prefixes.reduce((total, row) => total + row[1], 0);
  requireCondition(Array.isArray(summary.featureShardEvidence) && summary.featureShardEvidence.length === expectedCount,
    "M4 feature-shard count changed");
  let offset = 0;
  for (const [prefix, count, expectedItems, expectedViews, configuration] of prefixes) {
    let items = 0;
    let views = 0;
    for (let index = 0; index < count; index += 1) {
      const row = summary.featureShardEvidence[offset];
      exactKeys(row, ["cache", "cacheSha256", "replacedCacheSha256", "freshFeatureRunId",
        "freshlyExtractedThisRun", "freshlyExtractedThisProcess", "items", "views", "itemIdsSha256",
        "featureConfigurationSha256", "arraySha256"], "M4 feature-shard evidence");
      exactKeys(row.arraySha256, ["features", "labels", "variants", "sources"],
        "M4 feature-shard array hashes");
      const expectedCache = `benchmark/candidates/prooflens-cf384-m4/features/${prefix}-${String(index).padStart(5, "0")}.npz`;
      requireCondition(row.cache === expectedCache && HEX64.test(row.cacheSha256) &&
        row.replacedCacheSha256 === null && row.freshFeatureRunId === marker.runId &&
        row.freshlyExtractedThisRun === true && typeof row.freshlyExtractedThisProcess === "boolean" &&
        Number.isInteger(row.items) && row.items > 0 && row.items <= 2_000 &&
        Number.isInteger(row.views) && row.views >= row.items && row.views <= row.items * 4 &&
        HEX64.test(row.itemIdsSha256) &&
        row.featureConfigurationSha256 === configuration &&
        Object.values(row.arraySha256).every((value) => HEX64.test(value)),
      `M4 feature-shard evidence changed: ${expectedCache}`);
      items += row.items;
      views += row.views;
      offset += 1;
    }
    requireCondition(items === expectedItems && views === expectedViews,
      `M4 feature-shard totals changed: ${prefix}`);
  }
  return true;
}

function variantMetrics(logits, metadata, threshold) {
  const output = {};
  for (let variant = 0; variant < M4_VARIANTS.length; variant += 1) {
    let real = 0; let realCorrect = 0; let synthetic = 0; let syntheticCorrect = 0;
    const syntheticSources = new Map();
    const realSources = new Map();
    for (let index = 0; index < logits.length; index += 1) {
      if (metadata.variants[index] !== variant) continue;
      const label = metadata.labels[index];
      const source = metadata.sources[index];
      if (label === 0) {
        real += 1;
        const passed = logits[index] < threshold ? 1 : 0;
        realCorrect += passed;
        const row = realSources.get(source) ?? [0, 0]; row[0] += passed; row[1] += 1; realSources.set(source, row);
      } else {
        synthetic += 1;
        const passed = logits[index] >= threshold ? 1 : 0;
        syntheticCorrect += passed;
        const row = syntheticSources.get(source) ?? [0, 0]; row[0] += passed; row[1] += 1; syntheticSources.set(source, row);
      }
    }
    requireCondition(real > 0 && synthetic > 0, "M4 metric partition lost a class");
    const realRecall = realCorrect / real;
    const syntheticRecall = syntheticCorrect / synthetic;
    output[M4_VARIANTS[variant]] = {
      balancedAccuracy: (realRecall + syntheticRecall) / 2,
      realRecall,
      syntheticRecall,
      syntheticRecallBySource: Object.fromEntries([...syntheticSources].sort().map(([source, [correct, total]]) => [source, correct / total])),
      realRecallBySource: Object.fromEntries([...realSources].sort().map(([source, [correct, total]]) => [source, correct / total])),
    };
  }
  return output;
}

function passesGates(metrics, gates, expectedSources, label) {
  requireCondition(jsonEqual(Object.keys(metrics).sort(), [...M4_VARIANTS].sort()), `${label} variant set changed`);
  for (const variant of M4_VARIANTS) {
    const row = metrics[variant];
    requireCondition(jsonEqual(Object.keys(row.syntheticRecallBySource).sort(), [...expectedSources.synthetic].sort()) &&
      jsonEqual(Object.keys(row.realRecallBySource).sort(), [...expectedSources.real].sort()),
    `${label} source set changed`);
  }
  const rows = Object.values(metrics);
  return rows.every((row) => row.balancedAccuracy >= gates.minimumBalancedAccuracyPerVariant &&
    row.realRecall >= gates.minimumRealRecallPerVariant && row.syntheticRecall >= gates.minimumSyntheticRecallPerVariant) &&
    Math.min(...rows.flatMap((row) => Object.values(row.syntheticRecallBySource))) >= gates.minimumSyntheticRecallPerFamily &&
    rows.every((row) => Object.entries(gates.minimumRealRecallBySource ?? {})
      .every(([source, minimum]) => row.realRecallBySource[source] >= minimum));
}

function selectionKey(metrics, gates, decay, anchor) {
  const rows = M4_VARIANTS.map((variant) => metrics[variant]);
  return [
    Math.min(...rows.map((row) => row.balancedAccuracy)),
    Math.min(...rows.map((row) => row.realRecall)),
    Math.min(...rows.map((row) => row.syntheticRecall)),
    Math.min(...rows.flatMap((row) => Object.values(row.syntheticRecallBySource))),
    Math.min(...rows.flatMap((row) => Object.keys(gates.minimumRealRecallBySource ?? {})
      .map((source) => row.realRecallBySource[source]))),
    -decay,
    -anchor,
  ];
}

function compareKeys(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] > right[index] ? 1 : -1;
  }
  return 0;
}

export function recomputeM4Grid(grid, selectorMetadata, gates, { requireWinner = true } = {}) {
  exactKeys(grid, ["schemaVersion", "candidateTensorSealSha256", "selectorManifestSha256",
    "candidateCount", "validCandidateCount", "candidates"], "M4 candidate grid");
  requireCondition(grid.schemaVersion === 1 && grid.candidateCount === M4.candidateCount &&
    Array.isArray(grid.candidates) && grid.candidates.length === M4.candidateCount,
  "M4 candidate grid count changed");
  const pairs = new Set();
  const valid = [];
  for (const row of grid.candidates) {
    const expectedBaseKeys = ["candidateId", "weightDecay", "anchorCoefficient", "trainableParameters",
      "tensorSha256", "tensorShapes", "tensorDtypes", "tensorFloat32Base64", "selectorLogitsFloat32Base64",
      "selectorLogitsSha256", "selectorLogitCount", "thresholdPartitions", "valid"];
    const expectedKeys = row.valid ? [...expectedBaseKeys, "rawThreshold", "selectorMetrics", "selectorKey", "candidateSelectionKey"] : expectedBaseKeys;
    exactKeys(row, expectedKeys, "M4 candidate row");
    validateTensorRecord(Object.fromEntries(expectedBaseKeys.slice(0, 8).map((key) => [key, row[key]])),
      "M4 grid candidate");
    const pair = `${row.weightDecay}:${row.anchorCoefficient}`;
    requireCondition(!pairs.has(pair), "M4 candidate hyperparameter pair repeated"); pairs.add(pair);
    const logits = decodeFloat32(row, "selectorLogitsFloat32Base64", "selectorLogitCount", "selectorLogitsSha256",
      "M4 selector logits");
    requireCondition(logits.length === M4.selectorViews, "M4 selector logit count changed");
    const thresholds = completeDecisionThresholds(logits);
    let best = null;
    for (const threshold of thresholds) {
      const metrics = variantMetrics(logits, selectorMetadata, threshold);
      const passes = passesGates(metrics, gates, {
        synthetic: ["rapidata-dalle-3", "rapidata-flux", "rapidata-midjourney", "rapidata-stable-diffusion"],
        real: ["british-library-plates"],
      }, "M4 selector");
      if (!passes) continue;
      const key = selectionKey(metrics, gates, row.weightDecay, row.anchorCoefficient);
      if (best === null || compareKeys(key, best.key) > 0) best = { threshold, metrics, key };
    }
    requireCondition(row.thresholdPartitions === thresholds.length && row.valid === (best !== null),
      "M4 candidate threshold result changed");
    if (best) {
      requireCondition(row.rawThreshold === best.threshold && jsonEqual(row.selectorMetrics, best.metrics) &&
        jsonEqual(row.selectorKey, best.key.slice(0, 5)) && jsonEqual(row.candidateSelectionKey, best.key),
      "M4 candidate selector metrics or key changed");
      valid.push(row);
    }
  }
  requireCondition(pairs.size === M4.candidateCount && valid.length === grid.validCandidateCount &&
    (!requireWinner || valid.length > 0),
    "M4 valid candidate count changed");
  valid.sort((left, right) => compareKeys(left.candidateSelectionKey, right.candidateSelectionKey) ||
    left.candidateId.localeCompare(right.candidateId));
  return valid.at(-1) ?? null;
}

export function validateM4Regression(row, metadata, spec, rawThreshold, { requirePassed = true } = {}) {
  exactKeys(row, ["name", "metrics", "passed", "logitsSha256", "logitsFloat32Base64", "logitCount"],
    `M4 ${spec.name} regression`);
  const logits = decodeFloat32(row, "logitsFloat32Base64", "logitCount", "logitsSha256", `M4 ${spec.name} logits`);
  requireCondition(logits.length === spec.featureViews, `M4 ${spec.name} logit count changed`);
  const metrics = variantMetrics(logits, metadata, rawThreshold);
  const passed = passesGates(metrics, spec.gates, spec.sources, spec.name);
  requireCondition(row.name === spec.name && jsonEqual(row.metrics, metrics) && row.passed === passed &&
    (!requirePassed || passed),
    `M4 ${spec.name} regression evidence changed or failed`);
  return passed;
}

export function validateM4TrainingPacket({ summary, calibration, grid, recipe, selectionSummary, hashes,
  selectorMetadata, regressionMetadata, tensorSeal, selectionLock, freshMarker }) {
  exactKeys(summary, SUMMARY_KEYS, "M4 training summary");
  exactKeys(calibration, ["schemaVersion", "mode", "slope", "intercept", "rawThreshold",
    "displayThreshold", "rawProbabilityAtThreshold", "displayProbabilityAtRawThreshold",
    "modelSha256", "selectionLockSha256", "selectorManifestSha256"], "M4 calibration");
  exactKeys(selectionSummary, ["schemaVersion", "recipeSha256", "sourceLocksSha256", "scoreBlind",
    "modelOutputsRead", "h3PixelsRead", "h3ManifestSha256", "selectionOrder", "training",
    "freshSelector", "partitionGroups", "overlap", "publicArtifacts"], "M4 source selection summary");
  requireCondition(selectionSummary.schemaVersion === 1 && selectionSummary.recipeSha256 === M4.recipeSha256 &&
    selectionSummary.sourceLocksSha256 === M4.sourceLocksSha256 && selectionSummary.scoreBlind === true &&
    selectionSummary.modelOutputsRead === false && selectionSummary.h3PixelsRead === false &&
    selectionSummary.h3ManifestSha256 === recipe.h3Exclusion.sha256 &&
    jsonEqual(selectionSummary.selectionOrder, ["british-selector", "rapidata-selector",
      "british-training", "rapidata-training"]), "M4 source selection boundary changed");
  requireCondition(summary?.schemaVersion === 1 && summary.pipelineVersion === M4.pipelineVersion &&
    summary.status === "accepted-development-candidate" && summary.seed === M4.seed &&
    summary.commandArguments && jsonEqual(summary.commandArguments, M4_EXPECTED_ARGUMENTS) &&
    summary.trainerSha256 === hashes.trainer && summary.recipeSha256 === M4.recipeSha256 &&
    summary.sourceLocksSha256 === M4.sourceLocksSha256 &&
    summary.selectionSummarySha256 === hashes.selectionSummary &&
    summary.upstreamModelSha256 === M4.upstreamSha256 &&
    summary.trainManifestSha256 === selectionSummary.publicArtifacts["train-manifest.jsonl"].expandedSha256 &&
    summary.selectorManifestSha256 === hashes.selectorManifest &&
    summary.m3RegressionManifestSha256 === hashes.m3RegressionManifest &&
    summary.m2RegressionManifestSha256 === hashes.m2RegressionManifest &&
    summary.candidateTensorSealSha256 === hashes.tensorSeal &&
    summary.candidateGridSha256 === hashes.grid && summary.selectionLockSha256 === hashes.selectionLock &&
    summary.calibrationSha256 === hashes.calibration,
  "M4 training input bindings changed");
  requireCondition(summary.trainImages === M4.trainImages && summary.trainFeatureViews === M4.trainViews &&
    summary.selectorImages === M4.selectorImages && summary.selectorFeatureViews === M4.selectorViews &&
    summary.trainClassCounts?.real === 59_578 && summary.trainClassCounts?.synthetic === 53_120 &&
    summary.selectorClassCounts?.real === 300 && summary.selectorClassCounts?.synthetic === 300,
  "M4 training or selector counts changed");
  const expectedTrainSources = { ...recipe.baseTraining.sourceCounts, ...recipe.expectedTraining.newSourceCounts };
  requireCondition(jsonEqual(summary.trainSourceCounts, Object.fromEntries(Object.entries(expectedTrainSources).sort())) &&
    jsonEqual(summary.selectorSourceCounts, recipe.freshSelector.sourceCounts) &&
    jsonEqual(summary.anchorLossProtectedSources, recipe.adapter.protectedAnchorSources),
  "M4 training or selector source counts changed");
  requireCondition(summary.sourceBalancedLoss === true && summary.anchorLossProtectedSources &&
    summary.freshFeatureRunComplete === true && RUN_ID.test(summary.freshFeatureRunId),
  "M4 fresh feature-run evidence changed");
  validateM4FreshRunEvidence(summary, freshMarker, {
    state: "complete", completedRegressions: ["m3-selector-regression", "m2-development-regression"],
  });
  requireCondition(summary.freshFeatureMarkerSha256 === hashes.freshMarker &&
    summary.environment !== null && typeof summary.environment === "object" &&
    jsonEqual(summary.environment.providers, ["CPUExecutionProvider"]) &&
    summary.featureConfigurationHashes && HEX64.test(summary.featureConfigurationHashes.training) &&
    HEX64.test(summary.featureConfigurationHashes.evaluation) &&
    summary.featureConfigurationHashes.training !== summary.featureConfigurationHashes.evaluation,
  "M4 feature environment or configuration changed");
  requireCondition(summary.h3HoldoutScored === false && summary.h3PixelsRead === false &&
    summary.selectionInfluencedByRegression === false &&
    jsonEqual(summary.regressionOrder, ["m3-selector-regression", "m2-development-regression"]),
  "M4 selection or H3 boundary changed");

  const sealedById = validateM4TensorSeal(tensorSeal);
  const winner = recomputeM4Grid(grid, selectorMetadata, recipe.validationGates);
  requireCondition(grid.candidateTensorSealSha256 === hashes.tensorSeal &&
    grid.selectorManifestSha256 === hashes.selectorManifest &&
    grid.candidates.every((row) => jsonEqual(sealedById.get(row.candidateId), {
      candidateId: row.candidateId, weightDecay: row.weightDecay, anchorCoefficient: row.anchorCoefficient,
      trainableParameters: row.trainableParameters, tensorSha256: row.tensorSha256,
      tensorShapes: row.tensorShapes, tensorDtypes: row.tensorDtypes,
      tensorFloat32Base64: row.tensorFloat32Base64,
    })), "M4 grid is not bound to the candidate tensor seal");
  requireCondition(jsonEqual(summary.selectedCandidate, winner) && summary.candidateCount === 12 &&
    summary.validCandidateCount === grid.validCandidateCount && summary.candidateGridSha256 === hashes.grid,
  "M4 deterministic selector winner changed");
  exactKeys(selectionLock, ["schemaVersion", "candidateTensorSealSha256", "candidateGridSha256",
    "selectedCandidateId", "selectedTensorSha256", "rawThreshold", "selectorMetrics", "candidateSelectionKey",
    "selectorManifestSha256", "createdBeforeRegressionEvaluation", "selectionInfluencedByRegression",
    "h3HoldoutScored"], "M4 selection lock");
  const expectedSelectionLock = {
    schemaVersion: 1, candidateTensorSealSha256: hashes.tensorSeal, candidateGridSha256: hashes.grid,
    selectedCandidateId: winner.candidateId, selectedTensorSha256: winner.tensorSha256,
    rawThreshold: winner.rawThreshold, selectorMetrics: winner.selectorMetrics,
    candidateSelectionKey: winner.candidateSelectionKey, selectorManifestSha256: hashes.selectorManifest,
    createdBeforeRegressionEvaluation: true, selectionInfluencedByRegression: false, h3HoldoutScored: false,
  };
  requireCondition(jsonEqual(selectionLock, expectedSelectionLock) && jsonEqual(summary.selectionLock, expectedSelectionLock),
  "M4 pre-regression selection lock changed");
  requireCondition(Array.isArray(summary.regressions) && summary.regressions.length === 2,
    "M4 regression result count changed");
  validateM4Regression(summary.regressions[0], regressionMetadata.m3, {
    name: "m3-selector-regression", featureViews: 2400, gates: recipe.regressions[0].gates,
    sources: { synthetic: ["flux-1-dev-development"], real: ["met-open-access"] },
  }, winner.rawThreshold);
  validateM4Regression(summary.regressions[1], regressionMetadata.m2, {
    name: "m2-development-regression", featureViews: 3600, gates: recipe.regressions[1].gates,
    sources: { synthetic: ["GLM-Image", "HunyuanImage-3.0"], real: ["open-images", "stockimages-cc0"] },
  }, winner.rawThreshold);
  requireCondition(summary.modelSha256 === hashes.model && summary.modelBytes === hashes.modelBytes &&
    summary.zeroAdapterFeatureParityMaximumAbsoluteError === 0 &&
    finite(summary.zeroAdapterImageParityMaximumAbsoluteError) && summary.zeroAdapterImageParityMaximumAbsoluteError <= 0.000002 &&
    finite(summary.exportedCandidateImageParityMaximumAbsoluteError) && summary.exportedCandidateImageParityMaximumAbsoluteError <= 0.0002,
  "M4 model or parity evidence changed");
  const intercept = Math.log(0.65 / 0.35) - winner.rawThreshold;
  const sigmoid = (value) => value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
  requireCondition(calibration.schemaVersion === 1 && calibration.mode === "threshold-alignment-not-probability-calibration" &&
    calibration.slope === 1 && calibration.intercept === intercept && calibration.rawThreshold === winner.rawThreshold &&
    calibration.displayThreshold === 0.65 && calibration.rawProbabilityAtThreshold === sigmoid(winner.rawThreshold) &&
    calibration.displayProbabilityAtRawThreshold === sigmoid(winner.rawThreshold + intercept) &&
    calibration.modelSha256 === hashes.model && calibration.selectionLockSha256 === hashes.selectionLock &&
    calibration.selectorManifestSha256 === hashes.selectorManifest,
  "M4 calibration evidence changed");
  return winner;
}
