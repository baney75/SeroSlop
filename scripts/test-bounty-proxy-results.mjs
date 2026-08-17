import assert from "node:assert/strict";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

import { createHash } from "node:crypto";

import { canonicalJson, computeMetrics, DISPLAY_THRESHOLD, rawProbability } from "./bounty-proxy-browser.mjs";
import { verifyPacket } from "./check-bounty-proxy-results.mjs";

const source = path.resolve("benchmark/evidence/bounty-proxy-m2-v1");
const summary = JSON.parse(await readFile(path.join(source, "results/summary.json"), "utf8"));
const liveRun = {
  conclusion: "success",
  event: "push",
  head_sha: summary.testedGitHead,
  html_url: summary.publicCi.url,
  id: summary.publicCi.runId,
  path: summary.publicCi.workflowPath,
  status: "completed",
};
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
let fetchedUrl;
const fetchRun = async (url) => {
  fetchedUrl = url;
  return liveRun;
};
const verify = (root) => verifyPacket(root, { fetchRun, requireCurrentArchive: false });

const positive = await verify(source);
assert.equal(positive.rows, 1200);
assert.equal(positive.balancedAccuracy, 0.8583333333333333);
assert.equal(fetchedUrl, `https://api.github.com/repos/baney75/SeroSlop/actions/runs/${summary.publicCi.runId}`);

async function mutated(label, mutate, pattern) {
  const parent = await mkdtemp(path.join(os.tmpdir(), "seroslop-result-red-"));
  const root = path.join(parent, "packet");
  try {
    await cp(source, root, { recursive: true });
    await mutate(root);
    await assert.rejects(() => verify(root), pattern, label);
  } finally {
    await rm(parent, { force: true, recursive: true });
  }
}

await mutated("extra result file", async (root) => {
  await writeFile(path.join(root, "results/extra.json"), "{}\n");
}, /inventory changed/u);

await mutated("noncanonical summary", async (root) => {
  const file = path.join(root, "results/summary.json");
  const text = await readFile(file, "utf8");
  await writeFile(file, ` ${text}`);
}, /noncanonical/u);

await mutated("duplicate top-level key", async (root) => {
  const file = path.join(root, "results/summary.json");
  const text = await readFile(file, "utf8");
  await writeFile(file, text.replace("{", '{"rows":1200,'));
}, /duplicate-key/u);

await mutated("boolean score", async (root) => {
  const predictionsFile = path.join(root, "results/predictions.jsonl");
  const rows = (await readFile(predictionsFile, "utf8")).trimEnd().split("\n").map(JSON.parse);
  rows[0].displayScore = true;
  const raw = rows.map((row) => canonicalJson(row)).join("");
  await writeFile(predictionsFile, raw);
  const summaryFile = path.join(root, "results/summary.json");
  const value = JSON.parse(await readFile(summaryFile, "utf8"));
  value.predictionsSha256 = sha256(raw);
  await writeFile(summaryFile, canonicalJson(value));
}, /reviewed result packet digest/u);

await mutated("duplicate image hash", async (root) => {
  const predictionsFile = path.join(root, "results/predictions.jsonl");
  const rows = (await readFile(predictionsFile, "utf8")).trimEnd().split("\n").map(JSON.parse);
  rows[1].imageSha256 = rows[0].imageSha256;
  const raw = rows.map((row) => canonicalJson(row)).join("");
  await writeFile(predictionsFile, raw);
  const summaryFile = path.join(root, "results/summary.json");
  const value = JSON.parse(await readFile(summaryFile, "utf8"));
  value.predictionsSha256 = sha256(raw);
  await writeFile(summaryFile, canonicalJson(value));
}, /reviewed result packet digest/u);

await mutated("metric mutation", async (root) => {
  const file = path.join(root, "results/summary.json");
  const value = JSON.parse(await readFile(file, "utf8"));
  value.metrics.balancedAccuracy = 1;
  await writeFile(file, canonicalJson(value));
}, /reviewed result packet digest/u);

await mutated("completion mutation", async (root) => {
  const file = path.join(root, "results/summary.json");
  const value = JSON.parse(await readFile(file, "utf8"));
  value.completionSha256 = "0".repeat(64);
  await writeFile(file, canonicalJson(value));
}, /reviewed result packet digest/u);

await mutated("network request", async (root) => {
  const file = path.join(root, "results/summary.json");
  const value = JSON.parse(await readFile(file, "utf8"));
  value.networkRequestsAfterCutoff = ["https://example.com/"];
  await writeFile(file, canonicalJson(value));
}, /reviewed result packet digest/u);

await mutated("bounty acceptance claim", async (root) => {
  const file = path.join(root, "results/summary.json");
  const value = JSON.parse(await readFile(file, "utf8"));
  value.bountyAcceptanceClaimed = true;
  await writeFile(file, canonicalJson(value));
}, /reviewed result packet digest/u);

await mutated("coherent score and receipt forgery", async (root) => {
  const predictionsFile = path.join(root, "results/predictions.jsonl");
  const rows = (await readFile(predictionsFile, "utf8")).trimEnd().split("\n").map(JSON.parse);
  for (const row of rows) {
    row.displayScore = row.label === 0 ? 0.1 : 0.9;
    row.flagged = row.displayScore >= DISPLAY_THRESHOLD;
    row.rawProbability = rawProbability(row.displayScore);
  }
  const predictionsRaw = rows.map((row) => canonicalJson(row)).join("");
  await writeFile(predictionsFile, predictionsRaw);
  const summaryFile = path.join(root, "results/summary.json");
  const value = JSON.parse(await readFile(summaryFile, "utf8"));
  value.metrics = computeMetrics(rows);
  value.predictionsSha256 = sha256(predictionsRaw);
  const evidenceCore = { ...value };
  for (const key of ["bountyAcceptanceClaimed", "completedAt", "completionSha256", "schemaVersion", "status"]) {
    delete evidenceCore[key];
  }
  value.completionSha256 = sha256(Buffer.concat([Buffer.from(canonicalJson(evidenceCore)), Buffer.from(predictionsRaw)]));
  await writeFile(summaryFile, canonicalJson(value));
}, /reviewed result packet digest/u);

await mutated("prediction provider rewrite", async (root) => {
  const predictionsFile = path.join(root, "results/predictions.jsonl");
  const rows = (await readFile(predictionsFile, "utf8")).trimEnd().split("\n").map(JSON.parse);
  rows[0].provider = "wasm";
  const raw = rows.map((row) => canonicalJson(row)).join("");
  await writeFile(predictionsFile, raw);
}, /reviewed result packet digest/u);

await mutated("raw pixel field in manifest", async (root) => {
  const file = path.join(root, "frozen/manifest.jsonl");
  const rows = (await readFile(file, "utf8")).trimEnd().split("\n").map(JSON.parse);
  rows[0].rawPixels = "data:image/png;base64,AAAA";
  await writeFile(file, rows.map((row) => canonicalJson(row)).join(""));
}, /reviewed result packet digest/u);

assert.equal(0.65 >= DISPLAY_THRESHOLD, true, "inclusive threshold edge must flag");
assert.equal((DISPLAY_THRESHOLD - Number.EPSILON) >= DISPLAY_THRESHOLD, false, "value below threshold must not flag");
console.log("bounty proxy result verifier tests: PASS");
