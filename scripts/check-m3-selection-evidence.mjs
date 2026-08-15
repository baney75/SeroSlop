import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";

const RECIPE_PATH = "benchmark/m3/recipe.json";
const ROOT = "benchmark/evidence/m3";
const HISTORICAL_PATH = "benchmark/manifests/historical-perceptual-exclusions-v1.json.gz";
const M2_TRAIN_PATH = "benchmark/evidence/m2/train-manifest.jsonl.gz";
const M2_REVIEW_PATH = "benchmark/manifests/training-evaluation-perceptual-review.json";
const HEX64 = /^[a-f0-9]{64}$/u;
const DHASH64 = /^[a-f0-9]{16}$/u;
const FORBIDDEN_RESULT_FIELDS = new Set([
  "displayScore", "logit", "prediction", "rawProbability",
]);

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function json(value, file) {
  try {
    return JSON.parse(value.toString("utf8"));
  } catch (error) {
    throw new Error(`${file} is not valid JSON`, { cause: error });
  }
}

function jsonl(value, file) {
  try {
    return value.toString("utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
  } catch (error) {
    throw new Error(`${file} is not valid JSONL`, { cause: error });
  }
}

function equal(left, right) {
  const canonical = (value) => {
    if (Array.isArray(value)) return value.map(canonical);
    if (value && typeof value === "object") {
      return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
    }
    return value;
  };
  return JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));
}

function requireCanonicalGzip(value, file) {
  requireCondition(value.length >= 10 && value[3] === 0 && value.readUInt32LE(4) === 0 && value[9] === 0xff,
    `${file} does not use the canonical gzip header`);
}

function requireSafeRow(row, index, split, file, indexField = "rowIndex") {
  requireCondition(row[indexField] === index && row.split === split, `${file} row ordering or split changed`);
  requireCondition(typeof row.id === "string" && row.id.length > 0 && HEX64.test(row.imageSha256) &&
    DHASH64.test(row.perceptualDhash64), `${file} contains a malformed identity or pixel hash`);
  requireCondition(typeof row.path === "string" && !row.path.startsWith("/") && !row.path.split("/").includes(".."),
    `${file} contains an unsafe path`);
  requireCondition(!Object.keys(row).some((key) => FORBIDDEN_RESULT_FIELDS.has(key)),
    `${file} contains model-result fields`);
  requireCondition(!("pixelBytes" in row) && !("imageTransport" in row) && !("detailTransport" in row),
    `${file} contains private materialization fields`);
}

function unique(values) {
  return new Set(values).size === values.length;
}

const POPCOUNT = new Uint8Array(65_536);
for (let value = 1; value < POPCOUNT.length; value += 1) {
  POPCOUNT[value] = POPCOUNT[value >> 1] + (value & 1);
}

function blocks(value) {
  return [0, 4, 8, 12].map((offset) => Number.parseInt(value.slice(offset, offset + 4), 16));
}

function hamming(left, right) {
  const leftBlocks = blocks(left);
  const rightBlocks = blocks(right);
  return leftBlocks.reduce((total, value, index) => total + POPCOUNT[value ^ rightBlocks[index]], 0);
}

const neighborCache = new Map();
function neighborsWithinTwo(value) {
  if (neighborCache.has(value)) return neighborCache.get(value);
  const output = [value];
  for (let first = 0; first < 16; first += 1) {
    output.push(value ^ (1 << first));
    for (let second = first + 1; second < 16; second += 1) output.push(value ^ (1 << first) ^ (1 << second));
  }
  neighborCache.set(value, output);
  return output;
}

class PerceptualIndex {
  constructor() {
    this.rows = new Map();
    this.buckets = Array.from({ length: 4 }, () => new Map());
  }

  add(row) {
    if (this.rows.has(row.id)) return;
    this.rows.set(row.id, row);
    blocks(row.perceptualDhash64).forEach((value, blockIndex) => {
      const bucket = this.buckets[blockIndex].get(value) ?? [];
      bucket.push(row.id);
      this.buckets[blockIndex].set(value, bucket);
    });
  }

  matches(row, threshold = 8) {
    const candidateIds = new Set();
    blocks(row.perceptualDhash64).forEach((value, blockIndex) => {
      for (const neighbor of neighborsWithinTwo(value)) {
        for (const id of this.buckets[blockIndex].get(neighbor) ?? []) candidateIds.add(id);
      }
    });
    return [...candidateIds]
      .map((id) => this.rows.get(id))
      .filter((candidate) => candidate.id !== row.id && hamming(row.perceptualDhash64, candidate.perceptualDhash64) <= threshold);
  }
}

const recipeBytes = await readFile(RECIPE_PATH);
const recipe = json(recipeBytes, RECIPE_PATH);
const summaryBytes = await readFile(`${ROOT}/selection-summary.json`);
const summary = json(summaryBytes, `${ROOT}/selection-summary.json`);
const trainGzip = await readFile(`${ROOT}/train-manifest.jsonl.gz`);
const fluxGzip = await readFile(`${ROOT}/flux-source-index.json.gz`);
const rejectsGzip = await readFile(`${ROOT}/rejects.jsonl.gz`);
for (const [value, file] of [
  [trainGzip, `${ROOT}/train-manifest.jsonl.gz`],
  [fluxGzip, `${ROOT}/flux-source-index.json.gz`],
  [rejectsGzip, `${ROOT}/rejects.jsonl.gz`],
]) requireCanonicalGzip(value, file);
const trainBytes = gunzipSync(trainGzip);
const fluxBytes = gunzipSync(fluxGzip);
const rejectsBytes = gunzipSync(rejectsGzip);
const validationBytes = await readFile(`${ROOT}/validation-manifest.jsonl`);
const holdoutBytes = await readFile(`${ROOT}/h3-met-holdout-manifest.jsonl`);
const probeBytes = await readFile("benchmark/manifests/met-development-probe-v1.jsonl");
const attributionBytes = await readFile(`${ROOT}/attribution.json`);
const reviewBytes = await readFile(`${ROOT}/perceptual-review.json`);
const trainingReviewBytes = await readFile(`${ROOT}/training-evaluation-perceptual-review.json`);

requireCondition(recipe.schemaVersion === 3 && recipe.name === "prooflens-m3-cultural-heritage-hard-negative-head",
  "M3 recipe identity changed");
requireCondition(summary.schemaVersion === 3 && summary.recipeSha256 === digest(recipeBytes),
  "M3 summary does not bind the recipe");
const artifacts = summary.publicArtifacts;
const expectedArtifactKeys = [
  "attribution", "developmentProbeManifest", "fluxSourceIndex", "h3MetHoldoutManifest",
  "perceptualReview", "rejects", "trainManifest", "trainingEvaluationPerceptualReview",
  "validationManifest",
];
requireCondition(equal(Object.keys(artifacts).sort(), expectedArtifactKeys), "M3 public artifact surface changed");
requireCondition(artifacts.trainManifest.expandedSha256 === digest(trainBytes) &&
  artifacts.trainManifest.compressedSha256 === digest(trainGzip) &&
  artifacts.validationManifest.sha256 === digest(validationBytes) &&
  artifacts.h3MetHoldoutManifest.sha256 === digest(holdoutBytes) &&
  artifacts.developmentProbeManifest.sha256 === digest(probeBytes) &&
  artifacts.fluxSourceIndex.expandedSha256 === digest(fluxBytes) &&
  artifacts.fluxSourceIndex.compressedSha256 === digest(fluxGzip) &&
  artifacts.rejects.expandedSha256 === digest(rejectsBytes) &&
  artifacts.rejects.compressedSha256 === digest(rejectsGzip) &&
  artifacts.attribution.sha256 === digest(attributionBytes) &&
  artifacts.perceptualReview.sha256 === digest(reviewBytes) &&
  artifacts.trainingEvaluationPerceptualReview.sha256 === digest(trainingReviewBytes),
"M3 summary does not bind every public artifact");

const train = jsonl(trainBytes, `${ROOT}/train-manifest.jsonl.gz`);
const validation = jsonl(validationBytes, `${ROOT}/validation-manifest.jsonl`);
const holdout = jsonl(holdoutBytes, `${ROOT}/h3-met-holdout-manifest.jsonl`);
const probe = jsonl(probeBytes, "benchmark/manifests/met-development-probe-v1.jsonl");
const baseGzip = await readFile(M2_TRAIN_PATH);
requireCondition(digest(baseGzip) === recipe.baseTraining.compressedSha256, "M2 base training gzip changed");
const baseBytes = gunzipSync(baseGzip);
requireCondition(digest(baseBytes) === recipe.baseTraining.expandedSha256 &&
  trainBytes.subarray(0, baseBytes.length).equals(baseBytes),
"M3 training manifest does not preserve the exact M2 byte prefix");
const newTraining = jsonl(trainBytes.subarray(baseBytes.length), "M3 Met training suffix");
requireCondition(train.length === 108_378 && newTraining.length === 2_400 && validation.length === 600 &&
  holdout.length === 600 && probe.length === 100, "M3 partition counts changed");
train.forEach((row, index) => requireSafeRow(row, index, "train", "M3 training manifest"));
validation.forEach((row, index) => requireSafeRow(row, index, "validation", "M3 selector manifest"));
holdout.forEach((row, index) => requireSafeRow(row, index, "confirmatory-reserved", "H3 Met reserve"));
probe.forEach((row, index) => requireSafeRow(row, index, "consumed-development", "consumed Met probe", "probeRank"));
for (const [label, rows] of [
  ["M3 training", train], ["M3 selector", validation], ["H3 Met reserve", holdout], ["consumed Met probe", probe],
]) requireCondition(unique(rows.map((row) => row.id)) && unique(rows.map((row) => row.imageSha256)),
  `${label} contains duplicate IDs or image bytes`);
requireCondition(newTraining.every((row) => row.source === "met-open-access" && row.label === 0),
  "M3 training suffix is not exactly Met real images");
requireCondition(validation.filter((row) => row.source === "met-open-access" && row.label === 0).length === 300 &&
  validation.filter((row) => row.source === "flux-1-dev-development" && row.label === 1).length === 300,
"M3 selector source balance changed");
requireCondition(holdout.every((row) => row.source === "met-open-access" && row.label === 0),
  "H3 reserve is not exactly Met real images");

const expectedSources = Object.fromEntries(Object.keys(recipe.expectedSourceCounts).sort().map((source) => [
  source, train.filter((row) => row.source === source).length,
]));
requireCondition(equal(expectedSources, recipe.expectedSourceCounts), "M3 training source counts changed");
requireCondition(train.filter((row) => row.label === 0).length === 57_178 &&
  train.filter((row) => row.label === 1).length === 51_200, "M3 class counts changed");

const historicalGzip = await readFile(HISTORICAL_PATH);
requireCondition(digest(historicalGzip) === recipe.historicalExclusions.historicalPerceptualIndexSha256,
  "Historical perceptual exclusion index changed");
const prior = json(gunzipSync(historicalGzip), HISTORICAL_PATH).items;
const ids = new Set(prior.map((row) => row.id));
const hashes = new Set(prior.map((row) => row.imageSha256));
const groups = new Set(prior.map((row) => row.sourceGroupId ?? row.groupId ?? row.id));
const perceptual = new PerceptualIndex();
for (const row of prior) perceptual.add(row);
const frozenRows = [
  ...jsonl(baseBytes, M2_TRAIN_PATH),
  ...jsonl(await readFile(recipe.regressionValidation.manifest), recipe.regressionValidation.manifest),
];
for (const config of recipe.consumedEvaluationExclusions) {
  const bytes = await readFile(config.path);
  requireCondition(digest(bytes) === config.sha256, `Consumed evaluation manifest changed: ${config.path}`);
  frozenRows.push(...jsonl(bytes, config.path));
}
for (const row of frozenRows) {
  if (!ids.has(row.id) && DHASH64.test(row.perceptualDhash64 ?? "")) perceptual.add(row);
  ids.add(row.id);
  hashes.add(row.imageSha256);
  groups.add(row.sourceGroupId ?? row.groupId ?? row.id);
}
for (const row of probe) {
  if (!ids.has(row.id)) perceptual.add(row);
  ids.add(row.id);
  hashes.add(row.imageSha256);
  groups.add(row.sourceGroupId ?? row.id);
}
for (const [label, rows] of [["H3 Met reserve", holdout], ["M3 selector", validation], ["new M3 training", newTraining]]) {
  for (const row of rows) {
    const group = row.sourceGroupId ?? row.groupId ?? row.id;
    requireCondition(!ids.has(row.id) && !hashes.has(row.imageSha256) && !groups.has(group),
      `${label} overlaps a frozen ID, image hash, or source group: ${row.id}`);
    requireCondition(perceptual.matches(row).length === 0,
      `${label} has a frozen dHash match at distance 8 or lower: ${row.id}`);
    ids.add(row.id);
    hashes.add(row.imageSha256);
    groups.add(group);
    perceptual.add(row);
  }
}

const flux = json(fluxBytes, `${ROOT}/flux-source-index.json.gz`);
requireCondition(flux.schemaVersion === 1 && flux.dataset === recipe.syntheticDevelopmentSource.dataset &&
  flux.revision === recipe.syntheticDevelopmentSource.revision && flux.items.length === 1_000 &&
  unique(flux.items.map((row) => row.path)) && flux.items.every((row) => HEX64.test(row.lfsSha256)),
"M3 FLUX source index changed");
const rejects = jsonl(rejectsBytes, `${ROOT}/rejects.jsonl.gz`);
const rejectCounts = Object.fromEntries([...new Set(rejects.map((row) => row.reason))].sort().map((reason) => [
  reason, rejects.filter((row) => row.reason === reason).length,
]));
requireCondition(rejects.length === 845 && unique(rejects.map((row) => `${row.phase}:${row.candidateId}`)) &&
  equal(rejectCounts, summary.materializationRejectCounts) &&
  rejects.every((row) => !("detail" in row)), "M3 rejection evidence is incomplete or environment-dependent");
requireCondition(trainingReviewBytes.equals(await readFile(M2_REVIEW_PATH)) &&
  summary.carriedForwardM2PerceptualReview.reviewedPairCount === 100 &&
  summary.newM3OverlapWithEvaluation.perceptualDhashPairsAtOrBelowThreshold === 0 &&
  summary.scoresReadDuringSelection === false && summary.consumedDevelopmentRowsUsedForSelection === false,
"M3 carried-forward review or score-blind boundary changed");
const attribution = json(attributionBytes, `${ROOT}/attribution.json`);
requireCondition(attribution.met.sourceReportedLicense.includes("CC0-1.0") &&
  attribution.fluxDevelopment.developmentOnly === true &&
  attribution.fluxDevelopment.neverAcceptanceEvidence === true,
"M3 attribution or development-only status changed");
requireCondition(execFileSync("git", ["ls-files", "benchmark/data/m3-head", "benchmark/data/m3-source", "benchmark/data/h3-met-holdout-v1"],
  { encoding: "utf8" }).trim() === "", "M3 source pixels must remain outside Git");

console.log(JSON.stringify({
  trainingImages: train.length,
  trainingViews: summary.counts.trainingFeatureViews,
  selectorImages: validation.length,
  regressionImages: summary.counts.regression,
  reservedH3MetImages: holdout.length,
  rejectedCandidates: rejects.length,
  newPerceptualExceptions: 0,
  policy: "pass",
}));
