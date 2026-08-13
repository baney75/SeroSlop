/* global clearInterval, crypto, setInterval */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright-core";

const fixtureRoot = path.resolve(process.argv[2] ?? "");
if (!fixtureRoot || fixtureRoot === process.cwd()) {
  throw new Error("Usage: node scripts/browser-parity.mjs /path/to/prooflens-parity");
}
const extensionPath = path.resolve("dist");
const profilePath = await mkdtemp(path.join(os.tmpdir(), "prooflens-parity-chrome-"));
const manifestPath = path.join(fixtureRoot, "manifest.json");
const manifestBytes = await readFile(manifestPath);
const manifest = JSON.parse(manifestBytes.toString("utf8"));
if (!Array.isArray(manifest) || !manifest.length) throw new Error("Parity manifest is empty");
const modelSha256 = "29545a1da0cfe2bf0149448334fd45a21f48074c57296db3b84437dd66f80a43";
const testedGitHead = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const testedGitTree = execFileSync("git", ["rev-parse", "HEAD^{tree}"], { encoding: "utf8" }).trim();
const trackedSourceWorktreeDirty = Boolean(execFileSync(
  "git",
  ["status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)artifacts/**"],
  { encoding: "utf8" },
).trim());
const RAW_THRESHOLD = 0.5781767196773971;
const CALIBRATION_INTERCEPT = 0.30374610239790173;

function mimeFor(file) {
  if (/\.png$/iu.test(file)) return "image/png";
  if (/\.webp$/iu.test(file)) return "image/webp";
  return "image/jpeg";
}

function rawProbability(displayed) {
  const clipped = Math.min(1 - 1e-7, Math.max(1e-7, displayed));
  const rawLogit = Math.log(clipped / (1 - clipped)) - CALIBRATION_INTERCEPT;
  return 1 / (1 + Math.exp(-rawLogit));
}

function metric(rows, probabilityField) {
  const real = rows.filter((row) => row.label === 0);
  const synthetic = rows.filter((row) => row.label === 1);
  const realRecall = real.filter((row) => row[probabilityField] < RAW_THRESHOLD).length / real.length;
  const syntheticRecall = synthetic.filter((row) => row[probabilityField] >= RAW_THRESHOLD).length / synthetic.length;
  return { balancedAccuracy: (realRecall + syntheticRecall) / 2, realRecall, syntheticRecall };
}

let context;
try {
  context = await chromium.launchPersistentContext(profilePath, {
    headless: false,
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      "--no-first-run",
      "--no-default-browser-check",
    ],
  });
  const worker = context.serviceWorkers()[0] ?? await context.waitForEvent("serviceworker", { timeout: 30_000 });
  const extensionId = new URL(worker.url()).host;
  const page = await context.newPage();
  await page.goto(`chrome-extension://${extensionId}/setup.html`);
  await page.getByRole("heading", { name: "Offline ready" }).waitFor({ timeout: 300_000 });
  await context.setOffline(true);
  const requests = [];
  context.on("request", (request) => {
    if (/^https?:/u.test(request.url())) requests.push(request.url());
  });

  let completed = 0;
  const progress = setInterval(() => console.log(`browser parity ${completed}/${manifest.length}`), 10_000);
  const predictions = [];
  try {
    for (const row of manifest) {
      const file = path.join(fixtureRoot, row.path);
      const bytes = await readFile(file);
      const actualImageHash = createHash("sha256").update(bytes).digest("hex");
      if (actualImageHash !== row.imageSha256) throw new Error(`Parity image integrity mismatch for ${row.id}`);
      const dataUrl = `data:${mimeFor(file)};base64,${bytes.toString("base64")}`;
      const requestId = crypto.randomUUID();
      const response = await page.evaluate(
        ({ id, url }) => chrome.runtime.sendMessage({
          type: "PL_INFER",
          requestId: id,
          source: { kind: "rendered-pixels", url },
        }),
        { id: requestId, url: dataUrl },
      );
      if (!response?.ok || response.requestId !== requestId || !response.result) {
        throw new Error(`Browser parity inference failed for ${row.id}: ${JSON.stringify(response)}`);
      }
      const browserRawProbability = rawProbability(response.result.aiLikelihood);
      predictions.push({
        id: row.id,
        label: row.label,
        source: row.source,
        imageSha256: row.imageSha256,
        referenceRawProbability: row.referenceRawProbability,
        browserRawProbability,
        absoluteDifference: Math.abs(browserRawProbability - row.referenceRawProbability),
        referenceClassification: row.referenceRawProbability >= RAW_THRESHOLD ? 1 : 0,
        browserClassification: browserRawProbability >= RAW_THRESHOLD ? 1 : 0,
        provider: response.result.provider,
      });
      completed += 1;
    }
  } finally {
    clearInterval(progress);
  }
  if (requests.length) throw new Error(`Network requests occurred during offline parity: ${requests.join(", ")}`);

  const differences = predictions.map((row) => row.absoluteDifference);
  const agreement = predictions.filter((row) => row.referenceClassification === row.browserClassification).length / predictions.length;
  const report = {
    schemaVersion: 2,
    generatedAt: new Date().toISOString(),
    testedGitHead,
    testedGitTree,
    trackedSourceWorktreeDirty,
    modelSha256,
    archiveSha256: createHash("sha256").update(await readFile("release/prooflens.zip")).digest("hex"),
    browserVersion: await context.browser()?.version(),
    cleanProfile: true,
    offline: true,
    networkRequests: requests,
    fixtureManifestSha256: createHash("sha256").update(manifestBytes).digest("hex"),
    samples: predictions.length,
    classCounts: {
      real: predictions.filter((row) => row.label === 0).length,
      synthetic: predictions.filter((row) => row.label === 1).length,
    },
    providerCounts: Object.fromEntries(
      [...new Set(predictions.map((row) => row.provider))].sort().map((provider) => [
        provider,
        predictions.filter((row) => row.provider === provider).length,
      ]),
    ),
    referenceMetrics: metric(predictions, "referenceRawProbability"),
    browserMetrics: metric(predictions, "browserRawProbability"),
    decisionAgreement: agreement,
    meanAbsoluteProbabilityDifference: differences.reduce((sum, value) => sum + value, 0) / differences.length,
    maximumAbsoluteProbabilityDifference: Math.max(...differences),
    predictions,
  };
  await writeFile("artifacts/browser-parity.json", `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ ...report, predictions: undefined }, null, 2));
} finally {
  await context?.close().catch(() => undefined);
  await rm(profilePath, { recursive: true, force: true });
}
