import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

const RAW_THRESHOLD = 0.5781767196773971;
const expectedManifests = new Map([
  ["train", { path: "benchmark/manifests/train.jsonl", sha256: "8207d373a6abec32e4a687361199de43568fe982a5ff2e045987d3a029b85626", count: 3_600 }],
  ["validation", { path: "benchmark/manifests/validation.jsonl", sha256: "651704a6d3fb86b9324f13bea721c40854535925ce33b49b8749978ee2bed915", count: 600 }],
  ["test", { path: "benchmark/manifests/test.jsonl", sha256: "cd4d09fbb59d695ebc0cc4dc96f0dd17caea9e0e8865d7658c6100ef723e977f", count: 600 }],
]);
const expectedPredictions = new Map([
  ["original", { path: "benchmark/predictions/cf384-rehead-sealed-test-original-predictions.jsonl", sha256: "47a237f2ee70a128b3be2b88641ad3f2bed8bdb6622443c0bd31511914f2a2e2" }],
  ["screenshot", { path: "benchmark/predictions/cf384-rehead-sealed-test-screenshot-predictions.jsonl", sha256: "a977b9a0c0d03a37284a28630fa00614e0bd0339068ea66c99a38858e9b4a826" }],
  ["social-heavy", { path: "benchmark/predictions/cf384-rehead-sealed-test-social-heavy-predictions.jsonl", sha256: "ce26277babb6343344d188f4c76e892c30a970ca54a281f376407907c6ea07ee" }],
  ["social-q75", { path: "benchmark/predictions/cf384-rehead-sealed-test-social-q75-predictions.jsonl", sha256: "5383151040a3ba07bcb44196a4a283512e1cee19220d07763a83ff27ed658d07" }],
]);

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function parseJsonLines(bytes, path) {
  try {
    return bytes.toString("utf8").trim().split("\n").filter(Boolean).map((line) => JSON.parse(line));
  } catch (error) {
    throw new Error(`${path} is not valid JSONL`, { cause: error });
  }
}

function close(actual, expected, label) {
  if (Math.abs(actual - expected) > 1e-12) throw new Error(`${label}: expected ${expected}, received ${actual}`);
}

const manifests = new Map();
const allIds = new Set();
const allImageHashes = new Map();
for (const [split, expected] of expectedManifests) {
  const bytes = await readFile(expected.path);
  const actualHash = digest(bytes);
  if (actualHash !== expected.sha256) throw new Error(`${expected.path} SHA-256 mismatch: ${actualHash}`);
  const rows = parseJsonLines(bytes, expected.path);
  if (rows.length !== expected.count) throw new Error(`${expected.path} expected ${expected.count} rows, received ${rows.length}`);
  for (const row of rows) {
    if (row.split !== split) throw new Error(`${row.id} is assigned to ${row.split}, not ${split}`);
    if (allIds.has(row.id)) throw new Error(`Duplicate benchmark id: ${row.id}`);
    allIds.add(row.id);
    if (!/^[a-f0-9]{64}$/u.test(row.imageSha256)) throw new Error(`Invalid image SHA-256 for ${row.id}`);
    const hashOwner = allImageHashes.get(row.imageSha256);
    if (hashOwner) throw new Error(`Duplicate image bytes across splits: ${hashOwner} and ${row.id}`);
    allImageHashes.set(row.imageSha256, row.id);
  }
  manifests.set(split, rows);
}

const testRows = manifests.get("test");
const testById = new Map(testRows.map((row) => [row.id, row]));
const summary = JSON.parse(await readFile("benchmark/results/sealed-test-summary.json", "utf8"));
const bootstrap = JSON.parse(await readFile("benchmark/results/sealed-test-bootstrap.json", "utf8"));
if (summary.dataset.sha256 !== expectedManifests.get("test").sha256 || summary.dataset.items !== testRows.length) {
  throw new Error("Sealed summary does not target the committed test manifest");
}

for (const [variant, expected] of expectedPredictions) {
  const bytes = await readFile(expected.path);
  const actualHash = digest(bytes);
  if (actualHash !== expected.sha256) throw new Error(`${expected.path} SHA-256 mismatch: ${actualHash}`);
  if (bootstrap.variants[variant]?.predictionsSha256 !== actualHash) {
    throw new Error(`Bootstrap evidence does not target ${expected.path}`);
  }
  const rows = parseJsonLines(bytes, expected.path);
  if (rows.length !== testRows.length || new Set(rows.map((row) => row.id)).size !== testRows.length) {
    throw new Error(`${expected.path} does not contain exactly one prediction per test item`);
  }
  for (const row of rows) {
    const item = testById.get(row.id);
    if (!item || row.label !== item.label || row.source !== item.source || row.variant !== variant) {
      throw new Error(`${expected.path} contains a prediction that does not match the frozen test manifest: ${row.id}`);
    }
    if (!Number.isFinite(row.logit) || !Number.isFinite(row.rawProbability) || row.rawProbability < 0 || row.rawProbability > 1) {
      throw new Error(`${expected.path} contains an invalid prediction for ${row.id}`);
    }
  }
  const real = rows.filter((row) => row.label === 0);
  const synthetic = rows.filter((row) => row.label === 1);
  const realRecall = real.filter((row) => row.rawProbability < RAW_THRESHOLD).length / real.length;
  const syntheticRecall = synthetic.filter((row) => row.rawProbability >= RAW_THRESHOLD).length / synthetic.length;
  const balancedAccuracy = (realRecall + syntheticRecall) / 2;
  const published = summary.variants[variant];
  close(realRecall, published.realRecall, `${variant} real recall`);
  close(syntheticRecall, published.syntheticRecall, `${variant} synthetic recall`);
  close(balancedAccuracy, published.balancedAccuracy, `${variant} balanced accuracy`);
  close(balancedAccuracy, bootstrap.variants[variant].balancedAccuracy, `${variant} bootstrap source accuracy`);
}

const selection = JSON.parse(await readFile("benchmark/manifests/selection.json", "utf8"));
if ("auditExclusionsManifest" in selection || !String(selection.strategy).includes("duplicate bytes")) {
  throw new Error("Frozen selection evidence contains an unsupported audit-exclusion claim");
}
const parityIds = JSON.parse(await readFile("benchmark/manifests/parity-ids.json", "utf8"));
if (!Array.isArray(parityIds) || parityIds.length !== 60 || new Set(parityIds).size !== 60 ||
  parityIds.some((id) => !testById.has(id))) {
  throw new Error("Browser-parity IDs do not identify exactly 60 unique frozen test items");
}
const parityItems = parityIds.map((id) => testById.get(id));
const parityReal = parityItems.filter((row) => row.label === 0);
const paritySources = [...new Set(parityItems.filter((row) => row.label === 1).map((row) => row.source))];
if (parityReal.length !== 30 || paritySources.length !== 6 ||
  paritySources.some((source) => parityItems.filter((row) => row.source === source).length !== 5)) {
  throw new Error("Browser-parity IDs are not balanced 30 real / 5 per held-out synthetic family");
}
const docciAttribution = JSON.parse(await readFile("benchmark/manifests/docci-attribution.json", "utf8"));
if (!Array.isArray(docciAttribution.selectedExampleIds) || docciAttribution.selectedExampleIds.length !== 1_200) {
  throw new Error("DOCCI attribution does not preserve all 1,200 selected public example IDs");
}
console.log(JSON.stringify({
  manifests: Object.fromEntries([...expectedManifests].map(([split, value]) => [split, { rows: value.count, sha256: value.sha256 }])),
  predictions: Object.fromEntries([...expectedPredictions].map(([variant, value]) => [variant, value.sha256])),
  duplicateImageHashes: 0,
  parityItems: parityIds.length,
  policy: "pass",
}));
