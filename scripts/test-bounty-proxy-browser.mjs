import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  CALIBRATION_INTERCEPT,
  canonicalJson,
  computeMetrics,
  DISPLAY_THRESHOLD,
  MAX_SNAPSHOT_EDGE,
  MODEL_SHA256,
  PUBLIC_REPOSITORY,
  PUBLIC_REPOSITORY_URL,
  publicCiProof,
  RAW_PROBABILITY_THRESHOLD,
  renderedPixelsUrl,
  rawProbability,
  TASTE_GROUPS,
} from "./bounty-proxy-browser.mjs";

assert.equal(DISPLAY_THRESHOLD, 0.65);
assert.equal(CALIBRATION_INTERCEPT, 1.6126519746720926);
assert.equal(MODEL_SHA256, "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47");
assert.equal(PUBLIC_REPOSITORY, "baney75/SeroSlop");
assert.equal(PUBLIC_REPOSITORY_URL, "https://github.com/baney75/SeroSlop");
const productSource = readFileSync("src/content.ts", "utf8");
const productEdgeMatch = productSource.match(/const MAX_SNAPSHOT_EDGE = ([\d_]+);/u);
assert.ok(productEdgeMatch);
assert.equal(MAX_SNAPSHOT_EDGE, Number(productEdgeMatch[1].replaceAll("_", "")));
assert.ok(Math.abs(rawProbability(DISPLAY_THRESHOLD) - RAW_PROBABILITY_THRESHOLD) < 1e-15);
assert.equal(canonicalJson({ z: 1, a: { y: 2, b: 3 } }), '{"a":{"b":3,"y":2},"z":1}\n');
assert.throws(() => canonicalJson({ score: Number.NaN }), /nonfinite/u);
assert.throws(() => rawProbability(-1), /invalid display score/u);

const rows = [
  { displayScore: DISPLAY_THRESHOLD - 1e-12, label: 0, sourceGroup: "Met Open Access" },
  { displayScore: DISPLAY_THRESHOLD, label: 0, sourceGroup: "Met Open Access" },
  ...TASTE_GROUPS.flatMap((sourceGroup) => [
    { displayScore: DISPLAY_THRESHOLD, label: 1, sourceGroup },
    { displayScore: DISPLAY_THRESHOLD - 1e-12, label: 1, sourceGroup },
  ]),
];
const metric = computeMetrics(rows);
assert.deepEqual(metric.confusion, { fn: 4, fp: 1, tn: 1, tp: 4 });
assert.equal(metric.realRecall, 0.5);
assert.equal(metric.syntheticRecall, 0.5);
assert.equal(metric.balancedAccuracy, 0.5);
assert.deepEqual(metric.syntheticGroupRecalls, Object.fromEntries(TASTE_GROUPS.map((group) => [group, 0.5])));
assert.throws(() => computeMetrics([{ displayScore: Number.NaN, label: 0 }, { displayScore: 1, label: 1 }]), /finite/u);

const ciRun = {
  conclusion: "success",
  event: "push",
  head_sha: "a".repeat(40),
  html_url: "https://github.com/baney75/SeroSlop/actions/runs/123",
  id: 123,
  path: ".github/workflows/quality.yml",
  status: "completed",
};
assert.deepEqual(publicCiProof(ciRun, "a".repeat(40)), {
  conclusion: "success",
  event: "push",
  headSha: "a".repeat(40),
  runId: 123,
  status: "completed",
  url: "https://github.com/baney75/SeroSlop/actions/runs/123",
  workflowPath: ".github/workflows/quality.yml",
});
assert.throws(() => publicCiProof({ ...ciRun, head_sha: "b".repeat(40) }, "a".repeat(40)), /exact successful Quality/u);
assert.throws(() => publicCiProof({ ...ciRun, html_url: "https://github.com/baney75/prooflens/actions/runs/123" }, "a".repeat(40)), /exact successful Quality/u);
assert.equal(typeof renderedPixelsUrl, "function");

const source = readFileSync("scripts/bounty-proxy-browser.mjs", "utf8");
for (const pattern of [
  /PL_INFER/u,
  /setOffline\(true\)/u,
  /offlineCutoff = true/u,
  /networkRequestsAfterCutoff/u,
  /headless: false/u,
  /rename\(partial, final\)/u,
  /tracked product worktree is dirty/u,
  /scoring requires exact public main/u,
  /scoring requires the canonical public SeroSlop origin/u,
  /scoring requires exact-head public Quality success/u,
  /publicRemoteUrl/u,
  /publicMainSha/u,
  /publicCi/u,
  /MAX_SNAPSHOT_EDGE = 1024/u,
  /imageSmoothingQuality = "high"/u,
  /product-rendered pixels exceed the local image budget/u,
  /proxyThresholdCleared/u,
  /bountyAcceptanceClaimed: false/u,
]) assert.match(source, pattern);

console.log(JSON.stringify({ cases: 32, policy: "pass" }));
