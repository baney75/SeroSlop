/* global AbortSignal, clearInterval, fetch, setInterval */
import { createHash, randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  lstat,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  realpath,
  rename,
  rm,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

export const LABEL = "M2 fixed-threshold H3 Met / TASTE submission proxy";
export const DISPLAY_THRESHOLD = 0.65;
export const RAW_PROBABILITY_THRESHOLD = 0.27019907955040323;
export const CALIBRATION_INTERCEPT = 1.6126519746720926;
export const MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47";
export const CALIBRATION_SHA256 = "06d2452a8db9de26d42285cdc9dad0d233d397a6015583604c64480aec560e2c";
export const MODEL_LOCK_SHA256 = "2a818b7b2582bc9614f02f178d9af997f46628734ea078a79415c3d68d3061f0";
export const PUBLIC_REPOSITORY = "baney75/SeroSlop";
export const PUBLIC_REPOSITORY_URL = `https://github.com/${PUBLIC_REPOSITORY}`;
export const TASTE_GROUPS = Object.freeze([
  "FLUX.2 [max]",
  "GPT Image 1.5",
  "Nano Banana 2",
  "Seedream 5.0 Lite",
]);

const REPOSITORY_ROOT = path.resolve(".");
const DEFAULT_EVIDENCE_ROOT = path.join(REPOSITORY_ROOT, "benchmark/evidence/bounty-proxy-m2-v1");
const H3_ROOT = path.join(REPOSITORY_ROOT, "benchmark/data/h3-met-holdout-v1");
const TASTE_ROOT = path.join(REPOSITORY_ROOT, "benchmark/data/m6-frontier-cache/taste");
const EXTENSION_ROOT = path.join(REPOSITORY_ROOT, "dist");
const PUBLIC_ORIGIN_URLS = new Set([
  `${PUBLIC_REPOSITORY_URL}.git`,
  `git@github.com:${PUBLIC_REPOSITORY}.git`,
  `ssh://git@github.com/${PUBLIC_REPOSITORY}.git`,
]);

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function stable(value) {
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("nonfinite JSON value");
    return value;
  }
  if (Array.isArray(value)) return value.map(stable);
  if (typeof value === "object" && Object.getPrototypeOf(value) === Object.prototype) {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, stable(value[key])]));
  }
  throw new Error("unsupported JSON value");
}

export function canonicalJson(value) {
  return `${JSON.stringify(stable(value))}\n`;
}

async function physicalFile(root, relative) {
  if (typeof relative !== "string" || relative.length === 0 || relative.includes("\\") || relative.includes("\0")) {
    throw new Error("unsafe input path");
  }
  const parts = relative.split("/");
  if (path.posix.isAbsolute(relative) || parts.some((part) => part === "" || part === "." || part === "..")) {
    throw new Error("unsafe input path");
  }
  const rootStat = await lstat(root);
  if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) throw new Error("unsafe input root");
  let cursor = root;
  for (const part of parts) {
    cursor = path.join(cursor, part);
    const info = await lstat(cursor);
    if (info.isSymbolicLink()) throw new Error(`symlink input: ${relative}`);
  }
  const info = await lstat(cursor);
  if (!info.isFile()) throw new Error(`input is not a regular file: ${relative}`);
  const actualRoot = await realpath(root);
  const actual = await realpath(cursor);
  if (actual !== actualRoot && !actual.startsWith(`${actualRoot}${path.sep}`)) throw new Error("input escaped fixed root");
  return { bytes: info.size, path: cursor };
}

async function readCanonicalJson(file) {
  const root = path.dirname(file);
  const { path: physical } = await physicalFile(root, path.basename(file));
  const raw = await readFile(physical);
  const text = raw.toString("utf8");
  if (!text.endsWith("\n") || Buffer.from(text, "utf8").compare(raw) !== 0) throw new Error(`noncanonical UTF-8 JSON: ${file}`);
  const value = JSON.parse(text);
  if (canonicalJson(value) !== text) throw new Error(`noncanonical or duplicate-key JSON: ${file}`);
  return { raw, value };
}

async function readCanonicalJsonl(file) {
  const root = path.dirname(file);
  const { path: physical } = await physicalFile(root, path.basename(file));
  const raw = await readFile(physical);
  const text = raw.toString("utf8");
  if (!text.endsWith("\n") || Buffer.from(text, "utf8").compare(raw) !== 0) throw new Error(`noncanonical UTF-8 JSONL: ${file}`);
  const rows = text.slice(0, -1).split("\n").map((line) => {
    const value = JSON.parse(line);
    if (canonicalJson(value) !== `${line}\n`) throw new Error(`noncanonical or duplicate-key JSONL: ${file}`);
    return value;
  });
  return { raw, rows };
}

function exactKeys(value, keys, name) {
  if (value === null || typeof value !== "object" || Array.isArray(value) ||
      JSON.stringify(Object.keys(value).sort()) !== JSON.stringify([...keys].sort())) {
    throw new Error(`${name} schema changed`);
  }
}

function countBy(rows, key) {
  const result = {};
  for (const row of rows) result[row[key]] = (result[row[key]] ?? 0) + 1;
  return Object.fromEntries(Object.entries(result).sort(([left], [right]) => left.localeCompare(right)));
}

async function githubJson(url) {
  const response = await fetch(url, {
    headers: {
      Accept: "application/vnd.github+json",
      "User-Agent": "SeroSlop-bounty-proxy/1.0",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    redirect: "error",
    signal: AbortSignal.timeout(20_000),
  });
  if (!response.ok) throw new Error(`GitHub API request failed (${response.status})`);
  return response.json();
}

export function publicCiProof(run, expectedHead) {
  if (run === null || typeof run !== "object" || Array.isArray(run) ||
      !Number.isSafeInteger(run.id) || run.id <= 0 || run.head_sha !== expectedHead ||
      run.event !== "push" || run.status !== "completed" || run.conclusion !== "success" ||
      run.path !== ".github/workflows/quality.yml" ||
      run.html_url !== `${PUBLIC_REPOSITORY_URL}/actions/runs/${run.id}`) {
    throw new Error("public CI run is not an exact successful Quality run");
  }
  return {
    conclusion: run.conclusion,
    event: run.event,
    headSha: run.head_sha,
    runId: run.id,
    status: run.status,
    url: run.html_url,
    workflowPath: run.path,
  };
}

async function verifyPublicMainAndCi(testedGitHead) {
  const origin = execFileSync("git", ["remote", "get-url", "origin"], { encoding: "utf8" }).trim();
  if (!PUBLIC_ORIGIN_URLS.has(origin)) throw new Error("scoring requires the canonical public SeroSlop origin");
  const ref = await githubJson(`https://api.github.com/repos/${PUBLIC_REPOSITORY}/git/ref/heads/main`);
  if (ref?.object?.sha !== testedGitHead) throw new Error("scoring requires exact public main");
  const runs = await githubJson(
    `https://api.github.com/repos/${PUBLIC_REPOSITORY}/actions/runs?event=push&head_sha=${testedGitHead}&per_page=100`,
  );
  const candidates = Array.isArray(runs?.workflow_runs)
    ? runs.workflow_runs.filter((run) => run?.head_sha === testedGitHead && run?.event === "push" &&
      run?.status === "completed" && run?.conclusion === "success" && run?.path === ".github/workflows/quality.yml")
    : [];
  if (candidates.length === 0) throw new Error("scoring requires exact-head public Quality success");
  candidates.sort((left, right) => right.id - left.id);
  const exactRun = await githubJson(`https://api.github.com/repos/${PUBLIC_REPOSITORY}/actions/runs/${candidates[0].id}`);
  return {
    publicCi: publicCiProof(exactRun, testedGitHead),
    publicMainSha: testedGitHead,
    publicRemoteUrl: PUBLIC_REPOSITORY_URL,
  };
}

export function rawProbability(displayScore) {
  if (!Number.isFinite(displayScore) || displayScore < 0 || displayScore > 1) throw new Error("invalid display score");
  const clipped = Math.min(1 - 1e-7, Math.max(1e-7, displayScore));
  const rawLogit = Math.log(clipped / (1 - clipped)) - CALIBRATION_INTERCEPT;
  return 1 / (1 + Math.exp(-rawLogit));
}

export function computeMetrics(rows) {
  if (!Array.isArray(rows) || rows.length === 0 || rows.some((row) => row.label !== 0 && row.label !== 1)) {
    throw new Error("metrics require binary labelled rows");
  }
  if (rows.some((row) => !Number.isFinite(row.displayScore))) throw new Error("metrics require finite scores");
  const real = rows.filter((row) => row.label === 0);
  const synthetic = rows.filter((row) => row.label === 1);
  if (real.length === 0 || synthetic.length === 0) throw new Error("metrics require both classes");
  const flagged = (row) => row.displayScore >= DISPLAY_THRESHOLD;
  const tn = real.filter((row) => !flagged(row)).length;
  const fp = real.length - tn;
  const tp = synthetic.filter(flagged).length;
  const fn = synthetic.length - tp;
  const realRecall = tn / real.length;
  const syntheticRecall = tp / synthetic.length;
  const syntheticGroupRecalls = Object.fromEntries(TASTE_GROUPS.map((group) => {
    const members = synthetic.filter((row) => row.sourceGroup === group);
    if (members.length === 0) throw new Error(`missing synthetic group: ${group}`);
    return [group, members.filter(flagged).length / members.length];
  }));
  return {
    balancedAccuracy: (realRecall + syntheticRecall) / 2,
    confusion: { fn, fp, tn, tp },
    realRecall,
    syntheticGroupRecalls,
    syntheticRecall,
  };
}

async function validateEvidence(evidenceRoot) {
  if (path.resolve(evidenceRoot) !== DEFAULT_EVIDENCE_ROOT) throw new Error("scorer requires the fixed evidence root");
  const frozen = path.join(evidenceRoot, "frozen");
  const verified = path.join(evidenceRoot, "verified-inputs");
  if (JSON.stringify((await readdir(frozen)).sort()) !== JSON.stringify(["manifest.jsonl", "source-lock.json"])) {
    throw new Error("frozen bundle inventory changed");
  }
  if (JSON.stringify((await readdir(verified)).sort()) !== JSON.stringify(["input-manifest.jsonl", "verification.json"])) {
    throw new Error("verified-input bundle inventory changed");
  }
  const manifest = await readCanonicalJsonl(path.join(frozen, "manifest.jsonl"));
  const sourceLock = await readCanonicalJson(path.join(frozen, "source-lock.json"));
  const inputManifest = await readCanonicalJsonl(path.join(verified, "input-manifest.jsonl"));
  const verification = await readCanonicalJson(path.join(verified, "verification.json"));
  if (manifest.rows.length !== 1200 || inputManifest.rows.length !== 1200) throw new Error("proxy requires exactly 1,200 rows");
  if (sha256(manifest.raw) !== sourceLock.value.manifestSha256 || sha256(manifest.raw) !== verification.value.manifestSha256) {
    throw new Error("manifest digest binding changed");
  }
  if (sha256(sourceLock.raw) !== verification.value.sourceLockSha256 || sha256(inputManifest.raw) !== verification.value.inputManifestSha256) {
    throw new Error("verification digest binding changed");
  }
  exactKeys(verification.value, [
    "calibrationSha256", "inputManifestSha256", "label", "manifestSha256", "modelSha256",
    "rows", "schemaVersion", "sourceLockSha256", "status",
  ], "input verification");
  if (verification.value.schemaVersion !== 1 || verification.value.status !== "m2-bounty-proxy-inputs-verified" ||
      verification.value.label !== LABEL || verification.value.rows !== 1200 ||
      verification.value.modelSha256 !== MODEL_SHA256 || verification.value.calibrationSha256 !== CALIBRATION_SHA256) {
    throw new Error("input verification boundary changed");
  }
  if (sourceLock.value.status !== "m2-bounty-proxy-pre-score-locked" || sourceLock.value.label !== LABEL ||
      sourceLock.value.rows !== 1200 || sourceLock.value.pixelsReadAtFreeze !== false ||
      sourceLock.value.inferenceRun !== false || sourceLock.value.bountyAcceptanceClaimed !== false ||
      sourceLock.value.decision?.displayThreshold !== DISPLAY_THRESHOLD || sourceLock.value.decision?.inclusive !== true ||
      sourceLock.value.model?.sha256 !== MODEL_SHA256 || sourceLock.value.model?.bytes !== 87442080 ||
      sourceLock.value.calibration?.sha256 !== CALIBRATION_SHA256 ||
      sourceLock.value.calibration?.rawProbabilityThreshold !== RAW_PROBABILITY_THRESHOLD ||
      sourceLock.value.modelLock?.sha256 !== MODEL_LOCK_SHA256) {
    throw new Error("source-lock boundary changed");
  }
  const expectedGroups = { "Met Open Access": 600, ...Object.fromEntries(TASTE_GROUPS.map((group) => [group, 150])) };
  if (JSON.stringify(countBy(manifest.rows, "label")) !== JSON.stringify({ 0: 600, 1: 600 }) ||
      JSON.stringify(countBy(manifest.rows, "sourceGroup")) !== JSON.stringify(Object.fromEntries(Object.entries(expectedGroups).sort(([a], [b]) => a.localeCompare(b))))) {
    throw new Error("manifest allocation changed");
  }
  const inputs = new Map();
  const imageHashes = new Set();
  for (const input of inputManifest.rows) {
    exactKeys(input, ["bytes", "id", "path", "root", "sha256"], "input row");
    if (!Number.isSafeInteger(input.bytes) || input.bytes <= 0 || typeof input.id !== "string" ||
        typeof input.sha256 !== "string" || !/^[0-9a-f]{64}$/u.test(input.sha256) || inputs.has(input.id) || imageHashes.has(input.sha256)) {
      throw new Error("input row changed or duplicated");
    }
    inputs.set(input.id, input);
    imageHashes.add(input.sha256);
  }
  for (const row of manifest.rows) {
    const input = inputs.get(row.id);
    if (!input || input.root !== row.root || input.path !== row.path ||
        (row.imageSha256 !== undefined && row.imageSha256 !== input.sha256)) {
      throw new Error(`input receipt does not bind manifest row ${row.id}`);
    }
  }
  return {
    inputManifest,
    inputs,
    manifest,
    sourceLock,
    verification,
  };
}

function mimeFor(file) {
  if (/\.png$/iu.test(file)) return "image/png";
  if (/\.webp$/iu.test(file)) return "image/webp";
  if (/\.avif$/iu.test(file)) return "image/avif";
  if (/\.gif$/iu.test(file)) return "image/gif";
  return "image/jpeg";
}

async function fsyncDirectory(directory) {
  const handle = await open(directory, "r");
  try {
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function publishResults(evidenceRoot, files) {
  const final = path.join(evidenceRoot, "results");
  const partial = path.join(evidenceRoot, ".results.partial");
  for (const target of [final, partial]) {
    try {
      await lstat(target);
      throw new Error(`result target already exists: ${target}`);
    } catch (error) {
      if (error?.code !== "ENOENT") throw error;
    }
  }
  let installed = false;
  try {
    await mkdir(partial, { mode: 0o700 });
    for (const [name, raw] of Object.entries(files).sort(([left], [right]) => left.localeCompare(right))) {
      if (path.basename(name) !== name) throw new Error("unsafe result file name");
      const handle = await open(path.join(partial, name), "wx", 0o600);
      try {
        await handle.writeFile(raw);
        await handle.sync();
      } finally {
        await handle.close();
      }
    }
    await fsyncDirectory(partial);
    await rename(partial, final);
    installed = true;
    await fsyncDirectory(evidenceRoot);
  } catch (error) {
    if (installed) {
      try {
        await rm(final, { recursive: true });
        await fsyncDirectory(evidenceRoot);
      } catch (rollbackError) {
        throw new Error("result publication entered an unknown state", { cause: rollbackError });
      }
    } else {
      await rm(partial, { force: true, recursive: true });
    }
    throw error;
  }
}

async function score(evidenceRoot = DEFAULT_EVIDENCE_ROOT) {
  const evidence = await validateEvidence(path.resolve(evidenceRoot));
  const model = await readFile(path.join(REPOSITORY_ROOT, "weights/prooflens-cf384.onnx"));
  const bundledModel = await readFile(path.join(EXTENSION_ROOT, "weights/prooflens-cf384.onnx"));
  const calibration = await readFile(path.join(REPOSITORY_ROOT, "benchmark/evidence/m2/calibration.json"));
  const modelLock = await readFile(path.join(REPOSITORY_ROOT, "model-lock.json"));
  if (model.byteLength !== 87442080 || sha256(model) !== MODEL_SHA256 || sha256(bundledModel) !== MODEL_SHA256 ||
      sha256(calibration) !== CALIBRATION_SHA256 || sha256(modelLock) !== MODEL_LOCK_SHA256) {
    throw new Error("fixed model/calibration/package binding changed");
  }
  const testedGitHead = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
  const testedGitTree = execFileSync("git", ["rev-parse", "HEAD^{tree}"], { encoding: "utf8" }).trim();
  const publicState = await verifyPublicMainAndCi(testedGitHead);
  const dirty = execFileSync("git", [
    "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)contributor/**",
  ], { encoding: "utf8" }).trim();
  if (dirty) throw new Error(`tracked product worktree is dirty:\n${dirty}`);
  const archive = await readFile(path.join(REPOSITORY_ROOT, "release/prooflens.zip"));

  const profile = await mkdtemp(path.join(os.tmpdir(), "seroslop-bounty-proxy-"));
  let context;
  const requestsAfterCutoff = [];
  const predictions = [];
  let offlineCutoff = false;
  try {
    context = await chromium.launchPersistentContext(profile, {
      headless: false,
      args: [
        `--disable-extensions-except=${EXTENSION_ROOT}`,
        `--load-extension=${EXTENSION_ROOT}`,
        "--no-default-browser-check",
        "--no-first-run",
      ],
    });
    context.on("request", (request) => {
      if (offlineCutoff && /^https?:/u.test(request.url())) requestsAfterCutoff.push(request.url());
    });
    const worker = context.serviceWorkers()[0] ?? await context.waitForEvent("serviceworker", { timeout: 30_000 });
    const extensionId = new URL(worker.url()).host;
    const page = await context.newPage();
    await page.goto(`chrome-extension://${extensionId}/setup.html`);
    await page.getByRole("heading", { name: "Offline ready" }).waitFor({ timeout: 300_000 });
    await context.setOffline(true);
    offlineCutoff = true;

    let completed = 0;
    const progress = setInterval(() => console.log(`bounty proxy ${completed}/1200`), 10_000);
    try {
      for (const row of evidence.manifest.rows) {
        const input = evidence.inputs.get(row.id);
        const fixedRoot = input.root === "h3" ? H3_ROOT : input.root === "taste" ? TASTE_ROOT : undefined;
        if (fixedRoot === undefined) throw new Error(`unknown fixed root for ${row.id}`);
        const physical = await physicalFile(fixedRoot, input.path);
        const bytes = await readFile(physical.path);
        if (physical.bytes !== input.bytes || sha256(bytes) !== input.sha256) throw new Error(`input changed after verification: ${row.id}`);
        const requestId = randomUUID();
        const response = await page.evaluate(
          async ({ id, url }) => chrome.runtime.sendMessage({
            type: "PL_INFER",
            requestId: id,
            source: { kind: "rendered-pixels", url },
          }),
          { id: requestId, url: `data:${mimeFor(input.path)};base64,${bytes.toString("base64")}` },
        );
        if (!response?.ok || response.requestId !== requestId || !response.result ||
            !Number.isFinite(response.result.aiLikelihood) || typeof response.result.provider !== "string") {
          throw new Error(`PL_INFER failed for ${row.id}: ${JSON.stringify(response)}`);
        }
        const displayScore = response.result.aiLikelihood;
        const flagged = displayScore >= DISPLAY_THRESHOLD;
        if (response.result.classification !== (flagged ? "likely-ai" : "not-flagged")) {
          throw new Error(`runtime classification disagrees with inclusive threshold for ${row.id}`);
        }
        predictions.push({
          displayScore,
          flagged,
          id: row.id,
          imageSha256: input.sha256,
          label: row.label,
          provider: response.result.provider,
          rawProbability: rawProbability(displayScore),
          source: row.source,
          sourceGroup: row.sourceGroup,
        });
        completed += 1;
      }
    } finally {
      clearInterval(progress);
    }
    if (requestsAfterCutoff.length !== 0) throw new Error(`HTTP(S) request after offline cutoff: ${requestsAfterCutoff.join(", ")}`);
    const metric = computeMetrics(predictions);
    const predictionsRaw = Buffer.from(predictions.map((row) => canonicalJson(row)).join(""), "utf8");
    const providerCounts = countBy(predictions, "provider");
    const evidenceCore = {
      archiveSha256: sha256(archive),
      browserVersion: await context.browser()?.version(),
      classCounts: { real: 600, synthetic: 600 },
      cleanProfile: true,
      decision: {
        displayThreshold: DISPLAY_THRESHOLD,
        inclusive: true,
        rawProbabilityThreshold: RAW_PROBABILITY_THRESHOLD,
      },
      inputManifestSha256: sha256(evidence.inputManifest.raw),
      label: LABEL,
      manifestSha256: sha256(evidence.manifest.raw),
      metrics: metric,
      modelSha256: MODEL_SHA256,
      networkRequestsAfterCutoff: requestsAfterCutoff,
      offlineBeforeInference: true,
      predictionsSha256: sha256(predictionsRaw),
      providerCounts,
      publicCi: publicState.publicCi,
      publicMainSha: publicState.publicMainSha,
      publicRemoteUrl: publicState.publicRemoteUrl,
      proxyThresholdCleared: metric.balancedAccuracy >= 0.75,
      rows: 1200,
      sourceLockSha256: sha256(evidence.sourceLock.raw),
      testedGitHead,
      testedGitTree,
      trackedProductWorktreeDirty: false,
      verificationSha256: sha256(evidence.verification.raw),
    };
    const summary = {
      ...evidenceCore,
      bountyAcceptanceClaimed: false,
      completedAt: new Date().toISOString(),
      completionSha256: sha256(Buffer.concat([Buffer.from(canonicalJson(evidenceCore)), predictionsRaw])),
      schemaVersion: 1,
      status: "m2-bounty-proxy-complete",
    };
    await publishResults(path.resolve(evidenceRoot), {
      "predictions.jsonl": predictionsRaw,
      "summary.json": Buffer.from(canonicalJson(summary), "utf8"),
    });
    console.log(JSON.stringify(summary, null, 2));
    return summary;
  } finally {
    await context?.close().catch(() => undefined);
    await rm(profile, { force: true, recursive: true });
  }
}

const isMain = process.argv[1] !== undefined && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  const argument = process.argv[2] ?? DEFAULT_EVIDENCE_ROOT;
  await score(argument);
}
