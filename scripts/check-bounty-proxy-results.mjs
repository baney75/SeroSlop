/* global AbortSignal, fetch */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { lstat, readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  CALIBRATION_SHA256,
  DISPLAY_THRESHOLD,
  LABEL,
  MODEL_LOCK_SHA256,
  MODEL_SHA256,
  PUBLIC_REPOSITORY_URL,
  RAW_PROBABILITY_THRESHOLD,
  TASTE_GROUPS,
  canonicalJson,
  computeMetrics,
  publicCiProof,
  rawProbability,
} from "./bounty-proxy-browser.mjs";

const DEFAULT_ROOT = path.resolve("benchmark/evidence/bounty-proxy-m2-v1");
const PUBLIC_REPOSITORY = "baney75/SeroSlop";
const SCORED_HEAD = "57b356448608840f5b2ec3f2919799576719b539";
const SCORED_TREE = "a29ff072232dcefd23fc9cb409b98d7d5aacfad6";
const SCORED_RUN_ID = 32038187824;
const EXPECTED_DIGESTS = Object.freeze({
  inputManifest: "9961890a176b9b461092829ac97c4221da5b4de038f2545f2aacb02f81a90a57",
  manifest: "2f713788a8813c185fdea450db46fa4d5243467fd3cd405953a3130bf06b62a5",
  predictions: "01847dfde84c589a4285ee7c40e869227537506b7696c6a3ea95de9a34f6447c",
  sourceLock: "fe49a561efc95618411931188668fa0180797a2b1afac22985475dc6fb24dd48",
  summary: "61a06be31a48c01fd2534db8d656d2dccee1e4eb5107386ccc8a5e06e99a286e",
  verification: "776d1561329e082dc9b0a2913f24b6f70cc643805be4b1db17f7832ac6726bb1",
});
const EXPECTED_FILES = Object.freeze({
  frozen: ["manifest.jsonl", "source-lock.json"],
  results: ["predictions.jsonl", "summary.json"],
  "verified-inputs": ["input-manifest.jsonl", "verification.json"],
});
const SUMMARY_KEYS = Object.freeze([
  "archiveSha256", "bountyAcceptanceClaimed", "browserVersion", "classCounts", "cleanProfile",
  "completedAt", "completionSha256", "decision", "inputManifestSha256", "label", "manifestSha256",
  "metrics", "modelSha256", "networkRequestsAfterCutoff", "offlineBeforeInference", "predictionsSha256",
  "providerCounts", "proxyThresholdCleared", "publicCi", "publicMainSha", "publicRemoteUrl", "rows",
  "schemaVersion", "sourceLockSha256", "status", "testedGitHead", "testedGitTree",
  "trackedProductWorktreeDirty", "verificationSha256",
]);
const PREDICTION_KEYS = Object.freeze([
  "displayScore", "flagged", "id", "imageSha256", "label", "provider", "rawProbability", "source",
  "sourceGroup",
]);

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

function exactKeys(value, keys, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value) ||
      JSON.stringify(Object.keys(value).sort()) !== JSON.stringify([...keys].sort())) {
    throw new Error(`${label} schema changed`);
  }
}

async function physicalFile(file) {
  const info = await lstat(file);
  if (info.isSymbolicLink() || !info.isFile()) throw new Error(`unsafe evidence file: ${file}`);
  return readFile(file);
}

async function readCanonicalJson(file) {
  const raw = await physicalFile(file);
  const text = raw.toString("utf8");
  if (!text.endsWith("\n") || Buffer.from(text, "utf8").compare(raw) !== 0) {
    throw new Error(`noncanonical UTF-8 JSON: ${file}`);
  }
  const value = JSON.parse(text);
  if (canonicalJson(value) !== text) throw new Error(`noncanonical or duplicate-key JSON: ${file}`);
  return { raw, value };
}

async function readCanonicalJsonl(file) {
  const raw = await physicalFile(file);
  const text = raw.toString("utf8");
  if (!text.endsWith("\n") || Buffer.from(text, "utf8").compare(raw) !== 0) {
    throw new Error(`noncanonical UTF-8 JSONL: ${file}`);
  }
  const rows = text.slice(0, -1).split("\n").map((line, index) => {
    const value = JSON.parse(line);
    if (canonicalJson(value) !== `${line}\n`) {
      throw new Error(`noncanonical or duplicate-key JSONL: ${file}:${index + 1}`);
    }
    return value;
  });
  return { raw, rows };
}

async function defaultFetchRun(url) {
  const response = await fetch(url, {
    headers: {
      Accept: "application/vnd.github+json",
      "User-Agent": "SeroSlop-bounty-result-verifier/1.0",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    redirect: "error",
    signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok) throw new Error(`GitHub Actions refetch failed (${response.status})`);
  return response.json();
}

function exactMetric(actual, expected) {
  if (canonicalJson(actual) !== canonicalJson(expected)) throw new Error("recomputed metrics changed");
}

function countBy(rows, key) {
  const counts = {};
  for (const row of rows) counts[row[key]] = (counts[row[key]] ?? 0) + 1;
  return Object.fromEntries(Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)));
}

function validateManifestRow(row) {
  exactKeys(row.selection, ["algorithm", "namespace", "rank"], "manifest selection");
  if (row.selection.namespace !== "seroslop-bounty-proxy-m2-v1" ||
      !Number.isSafeInteger(row.selection.rank) || row.selection.rank < 0) {
    throw new Error("manifest selection changed");
  }
  if (row.label === 0) {
    exactKeys(row, ["id", "imageSha256", "label", "path", "root", "selection", "source", "sourceGroup"], "real manifest row");
    if (row.root !== "h3" || row.source !== "met-open-access" || row.sourceGroup !== "Met Open Access" ||
        row.selection.algorithm !== "all-h3-met-600" || !/^[0-9a-f]{64}$/u.test(row.imageSha256) ||
        !/^real\/met-open-access\/[0-9]+-[0-9a-f]{16}\.(?:jpg|jpeg|png)$/u.test(row.path)) {
      throw new Error("real manifest row changed");
    }
    return;
  }
  if (row.label === 1) {
    exactKeys(row, ["assetId", "id", "label", "path", "root", "selection", "source", "sourceGroup", "track"], "synthetic manifest row");
    if (row.root !== "taste" || row.source !== "taste" || !TASTE_GROUPS.includes(row.sourceGroup) ||
        row.selection.algorithm !== "sha256-ranked-150-per-taste-model" ||
        !Number.isSafeInteger(row.assetId) || row.assetId < 1 || row.assetId > 644 ||
        (row.track !== "descriptions" && row.track !== "aesthetics") ||
        !/^images\/[0-9]{7}\.(?:jpg|png)$/u.test(row.path)) {
      throw new Error("synthetic manifest row changed");
    }
    return;
  }
  throw new Error("manifest label changed");
}

export async function verifyPacket(root = DEFAULT_ROOT, { fetchRun = defaultFetchRun, requireCurrentArchive = true } = {}) {
  const resolvedRoot = path.resolve(root);
  const rootInfo = await lstat(resolvedRoot);
  if (rootInfo.isSymbolicLink() || !rootInfo.isDirectory()) throw new Error("unsafe evidence root");
  for (const [directory, expected] of Object.entries(EXPECTED_FILES)) {
    const actual = (await readdir(path.join(resolvedRoot, directory))).sort();
    if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(`${directory} evidence inventory changed`);
  }

  const manifest = await readCanonicalJsonl(path.join(resolvedRoot, "frozen/manifest.jsonl"));
  const sourceLock = await readCanonicalJson(path.join(resolvedRoot, "frozen/source-lock.json"));
  const inputManifest = await readCanonicalJsonl(path.join(resolvedRoot, "verified-inputs/input-manifest.jsonl"));
  const verification = await readCanonicalJson(path.join(resolvedRoot, "verified-inputs/verification.json"));
  const predictions = await readCanonicalJsonl(path.join(resolvedRoot, "results/predictions.jsonl"));
  const summary = await readCanonicalJson(path.join(resolvedRoot, "results/summary.json"));

  const packetDigests = {
    inputManifest: sha256(inputManifest.raw),
    manifest: sha256(manifest.raw),
    predictions: sha256(predictions.raw),
    sourceLock: sha256(sourceLock.raw),
    summary: sha256(summary.raw),
    verification: sha256(verification.raw),
  };
  if (canonicalJson(packetDigests) !== canonicalJson(EXPECTED_DIGESTS)) {
    throw new Error("reviewed result packet digest changed");
  }

  exactKeys(summary.value, SUMMARY_KEYS, "summary");
  exactKeys(verification.value, [
    "calibrationSha256", "inputManifestSha256", "label", "manifestSha256", "modelSha256", "rows",
    "schemaVersion", "sourceLockSha256", "status",
  ], "input verification");

  if (manifest.rows.length !== 1200 || inputManifest.rows.length !== 1200 || predictions.rows.length !== 1200 ||
      summary.value.rows !== 1200 || verification.value.rows !== 1200) {
    throw new Error("proxy evidence requires exactly 1,200 rows");
  }
  const hashBindings = [
    [manifest.raw, sourceLock.value.manifestSha256, "source-lock manifest"],
    [manifest.raw, verification.value.manifestSha256, "verification manifest"],
    [sourceLock.raw, verification.value.sourceLockSha256, "verification source lock"],
    [inputManifest.raw, verification.value.inputManifestSha256, "verification input manifest"],
    [manifest.raw, summary.value.manifestSha256, "summary manifest"],
    [sourceLock.raw, summary.value.sourceLockSha256, "summary source lock"],
    [inputManifest.raw, summary.value.inputManifestSha256, "summary input manifest"],
    [verification.raw, summary.value.verificationSha256, "summary verification"],
    [predictions.raw, summary.value.predictionsSha256, "summary predictions"],
  ];
  for (const [raw, expected, label] of hashBindings) {
    if (sha256(raw) !== expected) throw new Error(`${label} digest changed`);
  }

  if (sourceLock.value.status !== "m2-bounty-proxy-pre-score-locked" || sourceLock.value.label !== LABEL ||
      sourceLock.value.rows !== 1200 || sourceLock.value.pixelsReadAtFreeze !== false ||
      sourceLock.value.inferenceRun !== false || sourceLock.value.bountyAcceptanceClaimed !== false ||
      sourceLock.value.decision?.displayThreshold !== DISPLAY_THRESHOLD || sourceLock.value.decision?.inclusive !== true ||
      sourceLock.value.model?.sha256 !== MODEL_SHA256 || sourceLock.value.model?.bytes !== 87442080 ||
      sourceLock.value.calibration?.sha256 !== CALIBRATION_SHA256 ||
      sourceLock.value.calibration?.rawProbabilityThreshold !== RAW_PROBABILITY_THRESHOLD ||
      sourceLock.value.modelLock?.sha256 !== MODEL_LOCK_SHA256) {
    throw new Error("pre-score source-lock boundary changed");
  }
  if (verification.value.schemaVersion !== 1 || verification.value.status !== "m2-bounty-proxy-inputs-verified" ||
      verification.value.label !== LABEL || verification.value.modelSha256 !== MODEL_SHA256 ||
      verification.value.calibrationSha256 !== CALIBRATION_SHA256) {
    throw new Error("input verification boundary changed");
  }

  const manifestIds = new Set();
  const manifests = new Map();
  for (const row of manifest.rows) {
    validateManifestRow(row);
    if (typeof row.id !== "string" || manifestIds.has(row.id)) throw new Error("duplicate manifest ID");
    manifestIds.add(row.id);
    manifests.set(row.id, row);
  }
  const inputIds = new Set();
  const inputHashes = new Set();
  const inputs = new Map();
  for (const row of inputManifest.rows) {
    exactKeys(row, ["bytes", "id", "path", "root", "sha256"], "input row");
    if (typeof row.id !== "string" || typeof row.path !== "string" || typeof row.root !== "string" ||
        !Number.isSafeInteger(row.bytes) || row.bytes <= 0 || !/^[0-9a-f]{64}$/u.test(row.sha256) ||
        inputIds.has(row.id) || inputHashes.has(row.sha256)) {
      throw new Error("input row changed or duplicated");
    }
    inputIds.add(row.id);
    inputHashes.add(row.sha256);
    inputs.set(row.id, row);
  }

  const predictionIds = new Set();
  const predictionHashes = new Set();
  for (const [index, row] of predictions.rows.entries()) {
    exactKeys(row, PREDICTION_KEYS, "prediction row");
    const manifestRow = manifests.get(row.id);
    const inputRow = inputs.get(row.id);
    if (manifest.rows[index]?.id !== row.id || inputManifest.rows[index]?.id !== row.id || !manifestRow || !inputRow ||
        row.label !== manifestRow.label || row.source !== manifestRow.source || row.sourceGroup !== manifestRow.sourceGroup ||
        row.imageSha256 !== inputRow.sha256 || inputRow.root !== manifestRow.root || inputRow.path !== manifestRow.path ||
        (manifestRow.imageSha256 !== undefined && manifestRow.imageSha256 !== row.imageSha256)) {
      throw new Error(`prediction does not bind fixed row: ${row.id}`);
    }
    if (predictionIds.has(row.id) || predictionHashes.has(row.imageSha256)) {
      throw new Error("duplicate prediction ID or image hash");
    }
    if ((row.label !== 0 && row.label !== 1) || typeof row.displayScore !== "number" ||
        !Number.isFinite(row.displayScore) || row.displayScore < 0 || row.displayScore > 1 ||
        typeof row.rawProbability !== "number" || !Number.isFinite(row.rawProbability) ||
        typeof row.flagged !== "boolean" || typeof row.provider !== "string" || row.provider.length === 0 ||
        !/^[0-9a-f]{64}$/u.test(row.imageSha256)) {
      throw new Error("invalid prediction row");
    }
    if (row.flagged !== (row.displayScore >= DISPLAY_THRESHOLD)) throw new Error("inclusive display decision changed");
    if (Math.abs(row.rawProbability - rawProbability(row.displayScore)) > 2e-12) {
      throw new Error("raw probability does not match calibrated display score");
    }
    predictionIds.add(row.id);
    predictionHashes.add(row.imageSha256);
  }
  if (predictionIds.size !== manifestIds.size || predictionIds.size !== inputIds.size) throw new Error("row coverage changed");

  const metrics = computeMetrics(predictions.rows);
  exactMetric(summary.value.metrics, metrics);
  const providerCounts = countBy(predictions.rows, "provider");
  if (canonicalJson(providerCounts) !== canonicalJson({ webgpu: 1200 }) ||
      canonicalJson(providerCounts) !== canonicalJson(summary.value.providerCounts)) {
    throw new Error("recomputed provider counts changed");
  }
  const expectedGroups = Object.fromEntries(TASTE_GROUPS.map((group) => [group, 150]));
  const actualGroups = Object.fromEntries(TASTE_GROUPS.map((group) => [
    group,
    predictions.rows.filter((row) => row.label === 1 && row.sourceGroup === group).length,
  ]));
  if (canonicalJson(actualGroups) !== canonicalJson(expectedGroups)) throw new Error("synthetic group allocation changed");

  if (summary.value.schemaVersion !== 1 || summary.value.status !== "m2-bounty-proxy-complete" ||
      summary.value.label !== LABEL || summary.value.modelSha256 !== MODEL_SHA256 ||
      summary.value.decision?.displayThreshold !== DISPLAY_THRESHOLD || summary.value.decision?.inclusive !== true ||
      summary.value.decision?.rawProbabilityThreshold !== RAW_PROBABILITY_THRESHOLD ||
      canonicalJson(summary.value.classCounts) !== canonicalJson({ real: 600, synthetic: 600 }) ||
      summary.value.offlineBeforeInference !== true || summary.value.cleanProfile !== true ||
      summary.value.trackedProductWorktreeDirty !== false ||
      !Array.isArray(summary.value.networkRequestsAfterCutoff) || summary.value.networkRequestsAfterCutoff.length !== 0 ||
      summary.value.proxyThresholdCleared !== (metrics.balancedAccuracy >= 0.75) ||
      summary.value.proxyThresholdCleared !== true || summary.value.bountyAcceptanceClaimed !== false ||
      !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/u.test(summary.value.completedAt)) {
    throw new Error("completed result boundary changed");
  }

  const completionSha256 = summary.value.completionSha256;
  const evidenceCore = { ...summary.value };
  for (const key of ["bountyAcceptanceClaimed", "completedAt", "completionSha256", "schemaVersion", "status"]) {
    delete evidenceCore[key];
  }
  const recomputedCompletion = sha256(Buffer.concat([Buffer.from(canonicalJson(evidenceCore)), predictions.raw]));
  if (completionSha256 !== recomputedCompletion) throw new Error("completion digest changed");

  if (summary.value.testedGitHead !== SCORED_HEAD || summary.value.testedGitTree !== SCORED_TREE ||
      summary.value.publicMainSha !== SCORED_HEAD || summary.value.publicRemoteUrl !== PUBLIC_REPOSITORY_URL) {
    throw new Error("scored source binding changed");
  }
  const localTree = execFileSync("git", ["rev-parse", `${SCORED_HEAD}^{tree}`], { encoding: "utf8" }).trim();
  if (localTree !== SCORED_TREE) throw new Error("local scored Git object changed");
  const recordedCi = summary.value.publicCi;
  exactKeys(recordedCi, ["conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"], "public CI");
  if (recordedCi.runId !== SCORED_RUN_ID || recordedCi.headSha !== SCORED_HEAD) throw new Error("recorded CI binding changed");
  const runUrl = `https://api.github.com/repos/${PUBLIC_REPOSITORY}/actions/runs/${SCORED_RUN_ID}`;
  const liveRun = await fetchRun(runUrl);
  const liveProof = publicCiProof(liveRun, SCORED_HEAD);
  if (canonicalJson(liveProof) !== canonicalJson(recordedCi)) throw new Error("recorded CI proof no longer matches GitHub");

  const localArtifacts = [
    ["weights/prooflens-cf384.onnx", MODEL_SHA256],
    ["benchmark/evidence/m2/calibration.json", CALIBRATION_SHA256],
    ["model-lock.json", MODEL_LOCK_SHA256],
  ];
  if (requireCurrentArchive) localArtifacts.push(["release/prooflens.zip", summary.value.archiveSha256]);
  for (const [file, expected] of localArtifacts) {
    if (sha256(await physicalFile(path.resolve(file))) !== expected) throw new Error(`fixed artifact changed: ${file}`);
  }

  return {
    balancedAccuracy: metrics.balancedAccuracy,
    completionSha256: recomputedCompletion,
    predictionsSha256: sha256(predictions.raw),
    rows: predictions.rows.length,
    summarySha256: sha256(summary.raw),
  };
}

const isMain = process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const historicalRelease = process.argv.includes("--historical-release");
  const rootArgument = process.argv.slice(2).find((value) => !value.startsWith("--"));
  const result = await verifyPacket(rootArgument ?? DEFAULT_ROOT, { requireCurrentArchive: !historicalRelease });
  console.log(`bounty proxy result check: PASS (${result.rows} rows, ${(result.balancedAccuracy * 100).toFixed(2)}% balanced accuracy)`);
}
