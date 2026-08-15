import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { gunzipSync } from "node:zlib";

const RECIPE_PATH = "benchmark/m2/recipe.json";
const ROOT = "benchmark/evidence/m2";
const HISTORICAL_PATH = "benchmark/manifests/historical-perceptual-exclusions-v1.json.gz";
const HEX64 = /^[a-f0-9]{64}$/u;
const DHASH64 = /^[a-f0-9]{16}$/u;

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

function sorted(values) {
  return [...values].sort();
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
  requireCondition(value.length >= 10 && value[9] === 0xff, `${file} does not use the canonical gzip OS byte`);
}

const POPCOUNT = new Uint8Array(65_536);
for (let value = 1; value < POPCOUNT.length; value += 1) {
  POPCOUNT[value] = POPCOUNT[value >> 1] + (value & 1);
}

function blocks(value) {
  return [0, 4, 8, 12].map((offset) => Number.parseInt(value.slice(offset, offset + 4), 16));
}

function hamming(left, right) {
  const a = blocks(left);
  const b = blocks(right);
  return a.reduce((total, value, index) => total + POPCOUNT[value ^ b[index]], 0);
}

const neighborCache = new Map();
function neighborsWithinTwo(value) {
  if (neighborCache.has(value)) return neighborCache.get(value);
  const output = [value];
  for (let first = 0; first < 16; first += 1) {
    output.push(value ^ (1 << first));
    for (let second = first + 1; second < 16; second += 1) {
      output.push(value ^ (1 << first) ^ (1 << second));
    }
  }
  neighborCache.set(value, output);
  return output;
}

function perceptualIndex(rows) {
  const indexes = Array.from({ length: 4 }, () => new Map());
  rows.forEach((row, rowIndex) => {
    requireCondition(DHASH64.test(row.perceptualDhash64), `Invalid dHash for ${row.id}`);
    blocks(row.perceptualDhash64).forEach((value, blockIndex) => {
      const bucket = indexes[blockIndex].get(value) ?? [];
      bucket.push(rowIndex);
      indexes[blockIndex].set(value, bucket);
    });
  });
  return { rows, indexes };
}

function nearMatches(row, index, threshold = 8) {
  const candidates = new Set();
  blocks(row.perceptualDhash64).forEach((value, blockIndex) => {
    for (const neighbor of neighborsWithinTwo(value)) {
      for (const candidate of index.indexes[blockIndex].get(neighbor) ?? []) candidates.add(candidate);
    }
  });
  return [...candidates]
    .map((candidate) => index.rows[candidate])
    .filter((candidate) => candidate.id !== row.id && hamming(row.perceptualDhash64, candidate.perceptualDhash64) <= threshold);
}

const recipeBytes = await readFile(RECIPE_PATH);
const recipe = json(recipeBytes, RECIPE_PATH);
const summaryBytes = await readFile(`${ROOT}/selection-summary.json`);
const summary = json(summaryBytes, `${ROOT}/selection-summary.json`);
const trainGzip = await readFile(`${ROOT}/train-manifest.jsonl.gz`);
const selectionGzip = await readFile(`${ROOT}/stock-selection.json.gz`);
const rejectsGzip = await readFile(`${ROOT}/rejects.jsonl.gz`);
for (const [value, file] of [
  [trainGzip, `${ROOT}/train-manifest.jsonl.gz`],
  [selectionGzip, `${ROOT}/stock-selection.json.gz`],
  [rejectsGzip, `${ROOT}/rejects.jsonl.gz`],
]) requireCanonicalGzip(value, file);
const trainBytes = gunzipSync(trainGzip);
const selectionBytes = gunzipSync(selectionGzip);
const rejectsBytes = gunzipSync(rejectsGzip);
const validationBytes = await readFile(`${ROOT}/validation-manifest.jsonl`);
const attributionBytes = await readFile(`${ROOT}/attribution.json`);
const reviewBytes = await readFile(`${ROOT}/perceptual-review.json`);
const selection = json(selectionBytes, `${ROOT}/stock-selection.json.gz`);
const rejects = jsonl(rejectsBytes, `${ROOT}/rejects.jsonl.gz`);
const validation = jsonl(validationBytes, `${ROOT}/validation-manifest.jsonl`);
const attribution = json(attributionBytes, `${ROOT}/attribution.json`);
const review = json(reviewBytes, `${ROOT}/perceptual-review.json`);

requireCondition(recipe.schemaVersion === 2 && recipe.name === "prooflens-m2-stock-hard-negative-head",
  "M2 recipe identity changed");
requireCondition(digest(recipeBytes) === summary.recipeSha256 && summary.schemaVersion === 2,
  "M2 selection summary does not bind the recipe");
requireCondition(digest(trainBytes) === summary.manifestSha256 && digest(validationBytes) === summary.validationManifestSha256 &&
  digest(selectionBytes) === summary.stockSelectionSha256 && digest(rejectsBytes) === summary.rejectsSha256 &&
  digest(attributionBytes) === summary.attributionSha256 && digest(reviewBytes) === summary.newPerceptualReviewSha256,
"M2 selection summary does not bind every public artifact");

const baseGzip = await readFile(recipe.baseTraining.manifest);
requireCondition(digest(baseGzip) === recipe.baseTraining.compressedSha256,
  "M1 base training gzip changed");
const baseBytes = gunzipSync(baseGzip);
requireCondition(digest(baseBytes) === recipe.baseTraining.expandedSha256 &&
  trainBytes.subarray(0, baseBytes.length).equals(baseBytes),
"M2 training manifest does not preserve the exact M1 manifest prefix");
const stockTraining = jsonl(trainBytes.subarray(baseBytes.length), "M2 StockImages training suffix");
requireCondition(stockTraining.length === 2_378 && trainBytes.toString("utf8").split("\n").filter(Boolean).length === 105_978,
  "M2 training row counts changed");

const baseValidation = jsonl(await readFile(recipe.baseValidation.manifest), recipe.baseValidation.manifest);
requireCondition(validation.length === 900 && baseValidation.length === 600,
  "M2 validation row counts changed");
for (let index = 0; index < baseValidation.length; index += 1) {
  const expected = { ...baseValidation[index], rowIndex: index };
  requireCondition(equal(validation[index], expected), `M2 base validation row ${index} changed`);
}
const stockDevelopment = validation.slice(600);
const selectedStock = [...stockTraining, ...stockDevelopment];
requireCondition(selectedStock.every((row) => row.source === "stockimages-cc0" && row.label === 0 &&
  HEX64.test(row.imageSha256) && DHASH64.test(row.perceptualDhash64) &&
  !row.path.startsWith("/") && !row.path.split("/").includes("..")),
"M2 selected StockImages rows are malformed");
requireCondition(new Set(selectedStock.map((row) => row.id)).size === 2_678 &&
  new Set(selectedStock.map((row) => row.imageSha256)).size === 2_678 &&
  new Set(selectedStock.map((row) => row.sourceGroupId)).size === 2_678,
"M2 selected StockImages rows are not unique");

requireCondition(selection.schemaVersion === 1 && selection.scoreIndependent === true &&
  selection.consumedV2RowsUsedForGradients === false && selection.consumedV2RowsUsedForDevelopmentMetrics === false &&
  selection.candidateRows === 3_999 && selection.eligibleRows === 2_699 &&
  selection.trainingRows === 2_378 && selection.developmentRows === 300 &&
  selection.crossCandidateRejectedRows === 21 && selection.rejectedRows === 1_321,
"M2 selection policy or counts changed");
requireCondition(equal(selection.trainingIds, sorted(stockTraining.map((row) => row.id))) &&
  equal(selection.developmentIds, sorted(stockDevelopment.map((row) => row.id))),
"M2 selection ID lists do not match the manifests");
requireCondition(rejects.length === 1_321 && new Set(rejects.map((row) => row.candidateId)).size === rejects.length &&
  selectedStock.every((row) => !new Set(rejects.map((item) => item.candidateId)).has(row.id)),
"M2 rejection evidence is incomplete or overlaps selected rows");

const historicalGzip = await readFile(HISTORICAL_PATH);
requireCondition(digest(historicalGzip) === recipe.historicalExclusions.historicalPerceptualIndexSha256,
  "Historical perceptual exclusion index changed");
const historical = json(gunzipSync(historicalGzip), HISTORICAL_PATH).items;
const consumed = [];
for (const config of recipe.consumedEvaluationExclusions) {
  const bytes = await readFile(config.path);
  requireCondition(digest(bytes) === config.sha256, `Consumed manifest changed: ${config.path}`);
  consumed.push(...jsonl(bytes, config.path).filter((row) => DHASH64.test(row.perceptualDhash64 ?? "")));
}
const forbiddenIds = new Set([...historical, ...consumed].map((row) => row.id));
const forbiddenHashes = new Set([...historical, ...consumed].map((row) => row.imageSha256));
const forbiddenGroups = new Set([...historical, ...consumed].map((row) => row.sourceGroupId ?? row.groupId ?? row.id));
requireCondition(selectedStock.every((row) => !forbiddenIds.has(row.id) && !forbiddenHashes.has(row.imageSha256) &&
  !forbiddenGroups.has(row.sourceGroupId)),
"M2 selected rows overlap historical or consumed rows by ID, bytes, or group");
const forbiddenPerceptual = perceptualIndex([...historical, ...consumed]);
for (const row of selectedStock) {
  requireCondition(nearMatches(row, forbiddenPerceptual).length === 0,
    `M2 row ${row.id} perceptually overlaps historical or consumed evidence`);
}
const selectedPerceptual = perceptualIndex(selectedStock);
for (const row of selectedStock) {
  requireCondition(nearMatches(row, selectedPerceptual).length === 0,
    `M2 row ${row.id} perceptually overlaps another selected row`);
}

requireCondition(summary.counts.total === 105_978 && summary.counts.trainingFeatureViews === 123_912 &&
  summary.counts.validation === 900 && summary.counts.validationFeatureViews === 3_600 &&
  summary.classCounts.real === 54_778 && summary.classCounts.synthetic === 51_200 &&
  summary.sourceCounts["stockimages-cc0"] === 2_378,
"M2 summary counts changed");
requireCondition(equal(summary.evaluationExclusions.map((row) => row.role), [
  "development-validation", "consumed-v1-confirmatory", "frozen-v1-web-negative",
  "consumed-v2-confirmatory", "consumed-v2-web-negative",
]), "M2 evaluation exclusion roles changed");
requireCondition(review.retainedPairs.length === 0 && review.maximumHammingDistance === 8 &&
  attribution.sourceReportedLicense === "CC0-1.0" && attribution.uses.training === 2_378 &&
  attribution.uses.development === 300,
"M2 review or attribution evidence changed");
requireCondition(execFileSync("git", ["ls-files", "benchmark/data/m2-head"], { encoding: "utf8" }).trim() === "",
  "M2 source pixels must remain outside Git");

console.log(JSON.stringify({
  trainingImages: 105_978,
  trainingViews: 123_912,
  validationImages: 900,
  stockTraining: 2_378,
  stockDevelopment: 300,
  crossCandidateRejects: 21,
  historicalExclusions: historical.length,
  policy: "pass",
}));
