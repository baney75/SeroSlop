/* global clearInterval, clearTimeout, document, setInterval, setTimeout, window */
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright-core";
import sharp from "sharp";

const extensionPath = path.resolve("dist");
const expectedProvider = process.env.PROOFLENS_E2E_PROVIDER === "webgpu" ? "webgpu" : "wasm";
const profilePath = await mkdtemp(path.join(os.tmpdir(), "prooflens-chrome-"));
const artifactsPath = path.resolve("artifacts");
await mkdir(artifactsPath, { recursive: true });
const modelSha256 = "29545a1da0cfe2bf0149448334fd45a21f48074c57296db3b84437dd66f80a43";
const watchdogMs = 20 * 60_000;
const testedGitHead = execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const testedGitTree = execFileSync("git", ["rev-parse", "HEAD^{tree}"], { encoding: "utf8" }).trim();
const trackedSourceWorktreeDirty = Boolean(execFileSync(
  "git",
  ["status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)artifacts/**"],
  { encoding: "utf8" },
).trim());
let currentStage = "bootstrap";
const watchdog = setTimeout(() => {
  console.error(`Chrome E2E watchdog expired after ${watchdogMs}ms at stage: ${currentStage}`);
  process.exit(1);
}, watchdogMs);
watchdog.unref();

function stage(name) {
  currentStage = name;
  console.log(`Chrome E2E stage: ${name}`);
}

async function withTimeout(promise, timeoutMs, label) {
  let timeout;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timeout);
  }
}

function svgData(background, accent, label) {
  const source = `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="${background}"/><circle cx="205" cy="190" r="115" fill="${accent}"/><path d="M0 430L170 285l130 105 100-90 240 130" fill="#263552"/><text x="22" y="42" font-family="sans-serif" font-size="24" fill="white">${label}</text></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(source)}`;
}

const fixtureA = svgData("#174c3c", "#f5d97b", "fixture A");
const fixtureB = svgData("#5a284d", "#90d8f4", "fixture B");
const fixtureRace = svgData("#102c57", "#ff9f43", "in-flight race");
const fixtureC = `data:image/png;base64,${(
  await sharp({ create: { width: 640, height: 480, channels: 3, background: "#243c70" } })
    .composite([{ input: Buffer.from('<svg width="640" height="480"><circle cx="230" cy="190" r="120" fill="#f49b79"/><path d="M0 430L170 285l130 105 100-90 240 130" fill="#263552"/></svg>') }])
    .png()
    .toBuffer()
).toString("base64")}`;
const html = `<!doctype html>
<html><head><meta charset="utf-8"><style>
  body{font:16px system-ui;margin:24px;background:#f2f3f6;color:#171b2c}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}figure,.background{margin:0;min-height:260px;background:#fff;border-radius:14px;padding:10px}.background{background-image:url("${fixtureC}"),url("${fixtureA}");background-size:cover}.unavailable{background-image:url("file:///prooflens-unavailable.png");background-size:cover}img{display:block;width:100%;height:260px;object-fit:cover}</style></head>
<body><h1>ProofLens offline browser contract</h1><main id="grid">
<figure><img id="normal" width="640" height="480" alt="normal fixture" src="${fixtureA}"></figure>
<figure><img id="duplicate" width="640" height="480" alt="duplicate fixture" src="${fixtureA}"></figure>
<figure><picture><source media="(min-width:1px)" srcset="${fixtureB}"><img id="responsive" width="640" height="480" alt="responsive fixture" src="${fixtureA}"></picture></figure>
<div id="css-background" class="background" role="img" aria-label="CSS background fixture"></div>
<div id="unavailable-background" class="background unavailable" role="img" aria-label="Unavailable background fixture"></div>
</main><script>window.addDynamicFixture=()=>{const figure=document.createElement('figure');const image=document.createElement('img');image.id='dynamic';image.width=640;image.height=480;image.alt='dynamic fixture';image.src=${JSON.stringify(fixtureB)};figure.append(image);document.querySelector('#grid').append(figure);};</script></body></html>`;

const server = createServer((request, response) => {
  if (request.url === "/") {
    response.writeHead(200, {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
      connection: "close",
    });
    response.end(html);
    return;
  }
  response.writeHead(404, { connection: "close" }).end();
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
let serverOpen = true;
const address = server.address();
if (!address || typeof address === "string") throw new Error("Smoke server did not start");
const pageUrl = `http://127.0.0.1:${address.port}/`;
const pageOrigin = new URL(pageUrl).origin;

async function closeServer() {
  if (!serverOpen) return;
  serverOpen = false;
  const closing = new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  server.closeAllConnections();
  await withTimeout(closing, 5_000, "fixture server shutdown");
}

async function launch() {
  const providerArgs = expectedProvider === "wasm"
    ? ["--disable-webgpu", "--disable-gpu", "--disable-software-rasterizer"]
    : [];
  return chromium.launchPersistentContext(profilePath, {
    headless: false,
    ignoreDefaultArgs: expectedProvider === "wasm" ? ["--enable-unsafe-swiftshader"] : [],
    args: [
      `--disable-extensions-except=${extensionPath}`,
      `--load-extension=${extensionPath}`,
      ...providerArgs,
      "--no-first-run",
      "--no-default-browser-check",
    ],
  });
}

async function extensionWorker(context) {
  return context.serviceWorkers()[0] ?? context.waitForEvent("serviceworker", { timeout: 30_000 });
}

async function contentSnapshot(extensionPage, tabId) {
  return extensionPage.evaluate(
    (id) => chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" }),
    tabId,
  );
}

async function badgeSnapshot(extensionPage, tabId) {
  return (await contentSnapshot(extensionPage, tabId)).badges;
}

let context;
const diagnostics = [];
const postCutoffNetworkRequests = [];
let setupProgressAccessibleName = false;
try {
  stage("initial browser launch");
  context = await launch();
  let worker = await extensionWorker(context);
  const extensionId = new URL(worker.url()).host;
  const setup = await context.newPage();
  stage("model setup");
  await setup.goto(`chrome-extension://${extensionId}/setup.html`);
  await setup.getByRole("progressbar", { name: "Model preparation progress" }).waitFor();
  setupProgressAccessibleName = true;
  await setup.getByRole("heading", { name: "Offline ready" }).waitFor({ timeout: 300_000 });
  await setup.screenshot({ path: path.join(artifactsPath, `setup-${expectedProvider}.png`), fullPage: true });
  stage("initial browser close");
  await context.close();

  stage("restart browser launch");
  context = await launch();
  worker = await extensionWorker(context);
  const restartedExtensionId = new URL(worker.url()).host;
  if (restartedExtensionId !== extensionId) {
    throw new Error(`Extension identity changed across restart: ${extensionId} -> ${restartedExtensionId}`);
  }
  const statusPage = await context.newPage();
  const popupDiagnostics = [];
  statusPage.on("console", (message) => popupDiagnostics.push(`console:${message.type()}:${message.text()}`));
  statusPage.on("pageerror", (error) => popupDiagnostics.push(`pageerror:${error.message}`));
  await statusPage.goto(`chrome-extension://${extensionId}/popup.html`);
  stage("restart persistence check");
  try {
    await statusPage.locator("#model-status").filter({ hasText: "Offline ready" }).waitFor({ timeout: 60_000 });
  } catch (error) {
    const popupText = await statusPage.locator("body").innerText().catch(() => "<unavailable>");
    const directStatus = await statusPage.evaluate(async () => Promise.race([
      chrome.runtime.sendMessage({ type: "PL_GET_MODEL_STATUS" }),
      new Promise((resolve) => window.setTimeout(() => resolve({ state: "timeout" }), 10_000)),
    ])).catch((statusError) => ({ state: "probe-error", error: String(statusError) }));
    throw new Error(`Restarted popup did not report Offline ready: ${JSON.stringify({ popupText, directStatus, popupDiagnostics })}`, { cause: error });
  }
  await statusPage.screenshot({ path: path.join(artifactsPath, `popup-${expectedProvider}.png`), fullPage: true });
  await worker.evaluate((origin) => chrome.storage.local.set({ disabledOrigins: [origin] }), pageOrigin);
  const page = await context.newPage();
  await page.emulateMedia({ reducedMotion: "reduce" });
  stage("fixture navigation");
  await page.goto(pageUrl, { waitUntil: "load", timeout: 30_000 });
  stage("fixture image readiness");
  await page.locator("#normal").waitFor({ timeout: 30_000 });
  await page.waitForFunction(() => [...document.images].every((image) => image.complete), undefined, { timeout: 30_000 });
  stage("fixture server shutdown");
  await closeServer();
  stage("fixture offline transition");
  await withTimeout(context.setOffline(true), 10_000, "browser offline transition");
  context.on("request", (request) => {
    if (/^https?:/u.test(request.url())) postCutoffNetworkRequests.push(request.url());
  });

  stage("fixture tab discovery");
  const tabId = await withTimeout(worker.evaluate(async (url) => {
    const tabs = await chrome.tabs.query({ url });
    return tabs[0]?.id;
  }, pageUrl), 10_000, "fixture tab discovery");
  if (typeof tabId !== "number") throw new Error("Could not identify offline test tab");
  stage("fixture analysis enable");
  await withTimeout(worker.evaluate(
    async (id) => {
      await chrome.storage.local.set({ disabledOrigins: [] });
      await chrome.tabs.sendMessage(id, { type: "PL_SITE_STATE_CHANGED", enabled: true });
    },
    tabId,
  ), 30_000, "fixture analysis enable");
  stage("dynamic fixture admission");
  await withTimeout(page.evaluate(() => window.addDynamicFixture()), 10_000, "dynamic fixture admission");

  stage("initial offline inference");
  let progressBusy = false;
  const progressTimer = setInterval(async () => {
    if (progressBusy) return;
    progressBusy = true;
    try {
      console.log(`E2E progress ${JSON.stringify(await badgeSnapshot(statusPage, tabId))}`);
    } catch {
      // The final assertion will report authoritative diagnostics.
    } finally {
      progressBusy = false;
    }
  }, 10_000);
  try {
    const deadline = Date.now() + 600_000;
    while (Date.now() < deadline) {
      const current = await badgeSnapshot(statusPage, tabId);
      if (current.filter((badge) => badge.state === "complete").length === 5 &&
        current.filter((badge) => badge.state === "unavailable").length === 1) break;
      await page.waitForTimeout(500);
    }
    const settled = await badgeSnapshot(statusPage, tabId);
    if (settled.filter((badge) => badge.state === "complete").length !== 5 ||
      settled.filter((badge) => badge.state === "unavailable").length !== 1) {
      throw new Error("Timed out waiting for bounded content-script results");
    }
  } catch (error) {
    diagnostics.push(`badges: ${JSON.stringify(await badgeSnapshot(statusPage, tabId))}`);
    diagnostics.push(`storage: ${JSON.stringify(await worker.evaluate(() => chrome.storage.local.get()))}`);
    throw new Error(`${error instanceof Error ? error.message : String(error)}\n${diagnostics.join("\n")}`);
  } finally {
    clearInterval(progressTimer);
  }

  const badges = await badgeSnapshot(statusPage, tabId);
  const completed = badges.filter((badge) => badge.state === "complete");
  const unavailable = badges.filter((badge) => badge.state === "unavailable");
  if (badges.length !== 6 || completed.length !== 5 || unavailable.length !== 1) {
    throw new Error(`Unexpected target counts: ${JSON.stringify(badges)}`);
  }
  for (const badge of completed) {
    if (!/^(?:Likely AI|Not flagged) · \d{1,3}\.\d%$/u.test(badge.text ?? "")) {
      throw new Error(`Missing numeric confidence: ${JSON.stringify(badge)}`);
    }
    const runtimeLabel = expectedProvider === "webgpu" ? "WebGPU" : "WASM";
    if (badge.provider !== expectedProvider || !badge.title?.includes(`locally with ${runtimeLabel}`)) {
      throw new Error(`${runtimeLabel} provider was not observable: ${JSON.stringify(badge)}`);
    }
  }
  if (!unavailable[0]?.text?.includes("unavailable")) throw new Error("Unavailable target fabricated a confidence");
  if (postCutoffNetworkRequests.length) {
    throw new Error(`Network request occurred after server shutdown/offline cutoff: ${postCutoffNetworkRequests.join(", ")}`);
  }

  const cssBadge = (badges, elementId = "css-background") => badges.find(
    (badge) => badge.slot === "background:composite" && badge.elementId === elementId,
  );
  const originalCssBadge = cssBadge(badges);
  if (!originalCssBadge || originalCssBadge.acceptedResultCount !== 1) {
    throw new Error(`CSS badge slot was not observable: ${JSON.stringify(badges)}`);
  }

  // CSS-background race regression: admit a fresh valid target, wait until its
  // request is in flight, then change its source to one that cannot be read.
  // The lifetime acceptance counter proves the old valid response was never
  // displayed, even transiently, after invalidation.
  stage("CSS stale-response race");
  await page.evaluate((source) => {
    const target = document.createElement("div");
    target.id = "css-race";
    target.setAttribute("role", "img");
    target.setAttribute("aria-label", "CSS stale-response race fixture");
    target.style.cssText = `position:fixed;inset:8px auto auto 8px;z-index:2;width:640px;height:480px;background-image:url("${source}");background-size:cover`;
    document.body.append(target);
  }, fixtureRace);
  let raceStarted;
  for (let attempt = 0; attempt < 240; attempt += 1) {
    raceStarted = cssBadge(await badgeSnapshot(statusPage, tabId), "css-race");
    if (raceStarted?.state === "analyzing") break;
    await page.waitForTimeout(10);
  }
  if (raceStarted?.state !== "analyzing" || raceStarted.acceptedResultCount !== 0) {
    throw new Error(`CSS race request did not enter the analyzing state: ${JSON.stringify(raceStarted)}`);
  }
  if (raceStarted.animationName !== "none") {
    throw new Error(`Reduced motion did not suppress the analyzing animation: ${JSON.stringify(raceStarted)}`);
  }
  await page.evaluate(() => {
    const target = document.querySelector("#css-race");
    if (!target) throw new Error("CSS race fixture missing");
    target.style.backgroundImage = "url(\"file:///prooflens-race-invalidated.png\")";
  });
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const current = await badgeSnapshot(statusPage, tabId);
    const race = cssBadge(current, "css-race");
    if (race?.acceptedResultCount !== 0 || race?.state === "complete") {
      throw new Error(`A stale CSS response was accepted after its source changed: ${JSON.stringify(race)}`);
    }
    if (race?.state === "unavailable") break;
    await page.waitForTimeout(50);
  }
  const invalidated = await badgeSnapshot(statusPage, tabId);
  const unavailableRace = cssBadge(invalidated, "css-race");
  if (unavailableRace?.state !== "unavailable" || unavailableRace.acceptedResultCount !== 0) {
    throw new Error(`CSS source mutation did not reject the stale result: ${JSON.stringify(invalidated)}`);
  }

  await page.evaluate((source) => {
    const target = document.querySelector("#css-race");
    if (!target) throw new Error("CSS race fixture missing");
    target.style.backgroundImage = `url("${source}")`;
  }, fixtureB);
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const current = await badgeSnapshot(statusPage, tabId);
    if (cssBadge(current, "css-race")?.state === "complete") break;
    await page.waitForTimeout(50);
  }
  const reanalyzedRace = cssBadge(await badgeSnapshot(statusPage, tabId), "css-race");
  if (reanalyzedRace?.state !== "complete" || reanalyzedRace.acceptedResultCount !== 1) {
    throw new Error(`CSS replacement source was not freshly analyzed: ${JSON.stringify(reanalyzedRace)}`);
  }
  await page.evaluate(() => document.querySelector("#css-race")?.remove());
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if (!cssBadge(await badgeSnapshot(statusPage, tabId), "css-race")) break;
    await page.waitForTimeout(25);
  }
  if (cssBadge(await badgeSnapshot(statusPage, tabId), "css-race")) {
    throw new Error("Removed CSS race fixture retained a label");
  }

  // CSSOM-only mutation regression: no DOM attribute mutation occurs; the
  // bounded periodic reconciliation must still notice and reanalyze it.
  stage("CSSOM reconciliation");
  await page.evaluate(() => {
    const style = document.querySelector("style");
    if (!style?.sheet) throw new Error("CSSOM fixture missing");
    const rule = [...style.sheet.cssRules].find((candidate) => candidate.selectorText === ".background");
    if (!rule || !("style" in rule)) throw new Error("CSSOM background rule missing");
    rule.style.backgroundImage = "url(\"file:///prooflens-cssom-invalidated.png\")";
  });
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const current = await badgeSnapshot(statusPage, tabId);
    if (cssBadge(current)?.state === "unavailable") break;
    await page.waitForTimeout(50);
  }
  const cssomInvalidated = await badgeSnapshot(statusPage, tabId);
  if (cssBadge(cssomInvalidated)?.state !== "unavailable" || cssBadge(cssomInvalidated)?.acceptedResultCount !== 1) {
    throw new Error(`CSSOM change was not reconciled: ${JSON.stringify(cssomInvalidated)}`);
  }

  await page.evaluate(({ first, second }) => {
    const style = document.querySelector("style");
    if (!style?.sheet) throw new Error("CSSOM fixture missing");
    const rule = [...style.sheet.cssRules].find((candidate) => candidate.selectorText === ".background");
    if (!rule || !("style" in rule)) throw new Error("CSSOM background rule missing");
    rule.style.backgroundImage = `url("${first}"),url("${second}")`;
  }, { first: fixtureC, second: fixtureA });
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const current = await badgeSnapshot(statusPage, tabId);
    if (cssBadge(current)?.state === "complete" && cssBadge(current)?.acceptedResultCount === 2) break;
    await page.waitForTimeout(50);
  }
  const reanalyzed = await badgeSnapshot(statusPage, tabId);
  if (cssBadge(reanalyzed)?.state !== "complete" || cssBadge(reanalyzed)?.acceptedResultCount !== 2) {
    throw new Error(`CSS replacement source was not analyzed: ${JSON.stringify(reanalyzed)}`);
  }
  let summary;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    summary = await statusPage.evaluate(
      (id) => chrome.runtime.sendMessage({ type: "PL_GET_TAB_SUMMARY", tabId: id }),
      tabId,
    );
    const currentBadges = await badgeSnapshot(statusPage, tabId);
    const final = summary?.stats?.complete === 5 && summary?.stats?.unavailable === 1 &&
      currentBadges.filter((badge) => badge.state === "complete").length === 5 &&
      currentBadges.filter((badge) => badge.state === "unavailable").length === 1;
    if (final) {
      await page.waitForTimeout(750);
      const confirmed = await statusPage.evaluate(
        (id) => chrome.runtime.sendMessage({ type: "PL_GET_TAB_SUMMARY", tabId: id }),
        tabId,
      );
      const confirmedBadges = await badgeSnapshot(statusPage, tabId);
      if (confirmed?.stats?.complete === 5 && confirmed?.stats?.unavailable === 1 &&
        confirmedBadges.filter((badge) => badge.state === "complete").length === 5 &&
        confirmedBadges.filter((badge) => badge.state === "unavailable").length === 1) {
        summary = confirmed;
        break;
      }
    }
    await page.waitForTimeout(250);
  }
  if (summary?.stats?.complete !== 5 || summary?.stats?.unavailable !== 1) {
    throw new Error(`Background summary did not settle: ${JSON.stringify(summary)}; badges=${JSON.stringify(await badgeSnapshot(statusPage, tabId))}`);
  }
  await page.screenshot({ path: path.join(artifactsPath, `chrome-e2e-${expectedProvider}.png`), fullPage: true });

  stage("overlay recovery");
  const closedShadowRoot = await page.evaluate(() => {
    const host = document.querySelector("#prooflens-overlay");
    if (!host) return false;
    const closed = host.shadowRoot === null;
    host.remove();
    return closed;
  });
  if (!closedShadowRoot) throw new Error("Page context could access the trusted label shadow root");
  await page.locator("#prooflens-overlay").waitFor({ state: "attached", timeout: 5_000 });
  const repairedOverlay = await contentSnapshot(statusPage, tabId);
  if (!repairedOverlay.overlayAttached || repairedOverlay.badges.length !== 6) {
    throw new Error(`Overlay removal was not recovered: ${JSON.stringify(repairedOverlay)}`);
  }

  // Exhaust the mutation budget, let the admitted targets reconcile, then
  // mutate a CSS source again before the one-second budget window resets.
  // The second overflow must advance the invalidation epoch and reject the
  // response that was in flight after the first recovery.
  stage("repeated overflow stale-response race");
  await page.waitForTimeout(1_100);
  const overflowRaceBaseline = await contentSnapshot(statusPage, tabId);
  const overflowRaceAccepted = cssBadge(overflowRaceBaseline.badges)?.acceptedResultCount;
  if (typeof overflowRaceAccepted !== "number") throw new Error("Overflow race CSS target was not admitted");
  await page.evaluate(() => {
    const target = document.querySelector("#normal");
    for (let index = 0; index < 500; index += 1) target?.setAttribute("class", `prooflens-overflow-prime-${index}`);
  });
  let overflowRaceReanalysis;
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const current = await contentSnapshot(statusPage, tabId);
    const target = cssBadge(current.badges);
    if (current.mutationBudgetOverflows > overflowRaceBaseline.mutationBudgetOverflows &&
      !current.mutationOverflowRecoveryPending && target?.state === "analyzing") {
      overflowRaceReanalysis = current;
      break;
    }
    await page.waitForTimeout(5);
  }
  if (!overflowRaceReanalysis) throw new Error("First overflow did not enter bounded CSS reanalysis");
  await page.evaluate(() => {
    const target = document.querySelector("#css-background");
    if (!target) throw new Error("Overflow race CSS target missing");
    target.style.backgroundImage = "url(\"file:///prooflens-second-overflow.png\")";
  });
  let secondOverflowInvalidated;
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const current = await contentSnapshot(statusPage, tabId);
    const target = cssBadge(current.badges);
    if ((target?.acceptedResultCount ?? 0) > overflowRaceAccepted) {
      throw new Error(`Second overflow accepted a stale result: ${JSON.stringify(current)}`);
    }
    if (current.mutationBudgetOverflows >= overflowRaceBaseline.mutationBudgetOverflows + 2 &&
      !current.mutationOverflowRecoveryPending && target?.state === "unavailable") {
      secondOverflowInvalidated = current;
      break;
    }
    await page.waitForTimeout(25);
  }
  if (!secondOverflowInvalidated) throw new Error("Second same-window overflow did not invalidate the in-flight CSS result");
  await page.evaluate(() => document.querySelector("#css-background")?.style.removeProperty("background-image"));
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const target = cssBadge(await badgeSnapshot(statusPage, tabId));
    if (target?.state === "complete" && target.acceptedResultCount === overflowRaceAccepted + 1) break;
    await page.waitForTimeout(25);
  }
  const restoredOverflowRace = cssBadge(await badgeSnapshot(statusPage, tabId));
  if (restoredOverflowRace?.state !== "complete" || restoredOverflowRace.acceptedResultCount !== overflowRaceAccepted + 1) {
    throw new Error(`Overflow race CSS target did not recover exactly once: ${JSON.stringify(restoredOverflowRace)}`);
  }

  stage("hostile-page bounds");
  await page.evaluate((source) => {
    const deep = document.createElement("section");
    deep.id = "prooflens-deep-fixture";
    deep.hidden = true;
    for (let index = 0; index < 5_200; index += 1) deep.append(document.createElement("span"));
    const holder = document.createElement("div");
    holder.id = "prooflens-stress-fixture";
    holder.style.cssText = "position:absolute;left:-100000px;top:0;display:grid;grid-template-columns:repeat(32,64px);background:white";
    for (let index = 0; index < 505; index += 1) {
      const image = document.createElement("img");
      image.width = 64;
      image.height = 64;
      image.src = source;
      holder.append(image);
    }
    document.body.append(deep, holder);
  }, fixtureA);
  let preLate;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    preLate = await contentSnapshot(statusPage, tabId);
    if (preLate.recordCount === 511) break;
    await page.waitForTimeout(50);
  }
  if (preLate?.recordCount !== 511) throw new Error(`Deep fixture did not admit the bounded stress set: ${JSON.stringify(preLate)}`);
  await page.evaluate((source) => {
    const image = document.createElement("img");
    image.id = "late-after-5000";
    image.alt = "late target after 5000 inert elements";
    image.width = 640;
    image.height = 480;
    image.src = source;
    image.style.cssText = "position:fixed;inset:12px auto auto 12px;width:320px;height:240px;z-index:1";
    document.body.append(image);
  }, fixtureB);
  let bounded;
  for (let attempt = 0; attempt < 240; attempt += 1) {
    bounded = await contentSnapshot(statusPage, tabId);
    const late = bounded.badges.find((badge) => badge.elementId === "late-after-5000");
    if (bounded.recordCount === 512 && late?.state === "complete") break;
    await page.waitForTimeout(50);
  }
  if (!bounded) throw new Error("Hostile-page snapshot was unavailable");
  if (bounded.recordCount > bounded.targetLimit || bounded.targetLimit !== 512 || bounded.pendingCount > 32 || bounded.activeAnalyses > 1 ||
    bounded.lastScanVisited > bounded.maxElementsPerScan || bounded.fullScanIntervalMs !== 1_000 ||
    bounded.pendingMutationRoots > bounded.maxPendingMutationRoots || bounded.maxPendingMutationRoots !== 256 ||
    bounded.pendingDeferredReconciliations > bounded.maxDeferredReconciliations || bounded.maxDeferredReconciliations !== 512 ||
    bounded.maxDeferredReconciliationsPerPass !== 64 ||
    bounded.cssReconciliationIntervalMs !== 1_000 || bounded.maxCssReconciliationRecords !== 512) {
    throw new Error(`Hostile-page admission was not bounded: ${JSON.stringify(bounded)}`);
  }
  const scanPassesBeforeChurn = bounded.scanPasses;
  const mutationOverflowsBeforeChurn = bounded.mutationBudgetOverflows;
  await page.evaluate(() => {
    const target = document.querySelector("#normal");
    for (let index = 0; index < 500; index += 1) target?.setAttribute("class", `prooflens-churn-${index}`);
    for (let rootIndex = 0; rootIndex < 20; rootIndex += 1) {
      const root = document.createElement("section");
      root.dataset.prooflensHostileRoot = "true";
      for (let childIndex = 0; childIndex < 500; childIndex += 1) root.append(document.createElement("span"));
      document.body.append(root);
    }
  });
  let churned;
  let lateTargetEnteredRefreshing = false;
  for (let attempt = 0; attempt < 480; attempt += 1) {
    churned = await contentSnapshot(statusPage, tabId);
    const late = churned.badges.find((badge) => badge.elementId === "late-after-5000");
    if (late?.text?.includes("refreshing")) lateTargetEnteredRefreshing = true;
    if (lateTargetEnteredRefreshing && !churned.mutationOverflowRecoveryPending &&
      churned.pendingDeferredReconciliations === 0 && late?.state === "complete") break;
    await page.waitForTimeout(50);
  }
  if (!churned) throw new Error("Hostile mutation recovery snapshot was unavailable");
  const maxRecoveryPasses = Math.ceil(churned.maxDeferredReconciliations / churned.maxDeferredReconciliationsPerPass) + 3;
  if (!lateTargetEnteredRefreshing || churned.scanPasses - scanPassesBeforeChurn > maxRecoveryPasses ||
    churned.lastScanVisited > churned.maxElementsPerScan ||
    churned.pendingMutationRoots > churned.maxPendingMutationRoots ||
    churned.pendingDeferredReconciliations !== 0 || churned.mutationOverflowRecoveryPending ||
    churned.mutationBudgetWindowMs !== 1_000 || churned.maxMutationUnitsPerWindow !== 256 ||
    churned.maxObservedMutationUnitsInWindow > churned.maxMutationUnitsPerWindow ||
    churned.mutationBudgetOverflows <= mutationOverflowsBeforeChurn ||
    churned.synchronousMutationReconciliations !== 0) {
    throw new Error(`Hostile mutation traversal exceeded its aggregate budget: ${JSON.stringify(churned)}`);
  }
  await page.evaluate(() => {
    document.querySelectorAll("[data-prooflens-hostile-root]").forEach((root) => root.remove());
    document.querySelector("#prooflens-deep-fixture")?.remove();
    document.querySelector("#late-after-5000")?.remove();
  });
  await page.waitForTimeout(500);
  await worker.evaluate((id) => chrome.tabs.sendMessage(id, { type: "PL_SITE_STATE_CHANGED", enabled: false }), tabId);
  const replacementSource = fixtureB;
  await page.evaluate((source) => {
    const old = document.querySelector("#prooflens-stress-fixture");
    const replacement = document.createElement("div");
    replacement.id = "prooflens-replacement-fixture";
    replacement.style.cssText = "position:fixed;inset:0;display:grid;grid-template-columns:repeat(32,64px);overflow:hidden;background:white;z-index:-1";
    for (let index = 0; index < 20; index += 1) {
      const image = document.createElement("img");
      image.width = 64;
      image.height = 64;
      image.src = source;
      replacement.append(image);
    }
    old?.replaceWith(replacement);
  }, replacementSource);
  await page.waitForTimeout(1_500);
  const replaced = await contentSnapshot(statusPage, tabId);
  if (replaced.recordCount !== 26 || replaced.pendingCount > 32 || replaced.activeAnalyses > 1) {
    throw new Error(`Target-cap replacement did not recover: ${JSON.stringify(replaced)}`);
  }
  await page.evaluate(() => document.querySelector("#prooflens-replacement-fixture")?.remove());
  await worker.evaluate((id) => chrome.tabs.sendMessage(id, { type: "PL_SITE_STATE_CHANGED", enabled: true }), tabId);
  await page.setViewportSize({ width: 480, height: 720 });
  await page.waitForTimeout(500);
  const narrow = await contentSnapshot(statusPage, tabId);
  const visibleNarrowLabels = narrow.badges.filter((badge) => !badge.hidden).length;
  if (visibleNarrowLabels < 1) throw new Error(`Narrow viewport exposed no eligible label: ${JSON.stringify(narrow)}`);
  await page.screenshot({ path: path.join(artifactsPath, `chrome-e2e-${expectedProvider}-narrow.png`), fullPage: true });
  const archiveSha256 = createHash("sha256").update(await readFile("release/prooflens.zip")).digest("hex");
  const evidence = {
    schemaVersion: 2,
    generatedAt: new Date().toISOString(),
    testedGitHead,
    testedGitTree,
    trackedSourceWorktreeDirty,
    modelSha256,
    archiveSha256,
    browserVersion: await context.browser()?.version(),
    extensionId,
    cleanProfile: true,
    persistedModelAfterRestart: true,
    serverStoppedBeforeAnalysis: true,
    browserOfflineBeforeAnalysis: true,
    postCutoffNetworkRequests,
    provider: expectedProvider,
    setupProgressAccessibleName,
    reducedMotionSuppressed: raceStarted.animationName === "none",
    targets: { total: badges.length, complete: completed.length, unavailable: unavailable.length },
    numericConfidence: true,
    dynamicImage: true,
    responsivePicture: true,
    cssCompositeBackground: true,
    cssStaleResponseRejected: unavailableRace.acceptedResultCount === 0 && reanalyzedRace.acceptedResultCount === 1,
    cssomBackgroundReconciled: cssBadge(cssomInvalidated)?.acceptedResultCount === 1 && cssBadge(reanalyzed)?.acceptedResultCount === 2,
    repeatedOverflowStaleResponseRejected:
      secondOverflowInvalidated.mutationBudgetOverflows >= overflowRaceBaseline.mutationBudgetOverflows + 2 &&
      cssBadge(secondOverflowInvalidated.badges)?.acceptedResultCount === overflowRaceAccepted,
    narrowViewport: { width: 480, height: 720, visibleLabels: visibleNarrowLabels },
    closedShadowRoot,
    overlayRemovalRecovered: true,
    boundedTargetAdmission: {
      observedRecords: bounded.recordCount,
      targetLimit: bounded.targetLimit,
      pendingLimitObserved: bounded.pendingCount,
      activeLimitObserved: bounded.activeAnalyses,
      scanPassesDuringChurn: churned.scanPasses - scanPassesBeforeChurn,
      maxElementsPerScan: churned.maxElementsPerScan,
      maxPendingMutationRoots: churned.maxPendingMutationRoots,
      lateTargetAfter5000Recovered: lateTargetEnteredRefreshing,
      pendingDeferredReconciliations: churned.pendingDeferredReconciliations,
      maxDeferredReconciliations: churned.maxDeferredReconciliations,
      maxDeferredReconciliationsPerPass: churned.maxDeferredReconciliationsPerPass,
      mutationBudgetWindowMs: churned.mutationBudgetWindowMs,
      maxMutationUnitsPerWindow: churned.maxMutationUnitsPerWindow,
      maxObservedMutationUnitsInWindow: churned.maxObservedMutationUnitsInWindow,
      mutationBudgetOverflows: churned.mutationBudgetOverflows,
      mutationInvalidationEpoch: churned.mutationInvalidationEpoch,
      mutationOverflowRecoveryPending: churned.mutationOverflowRecoveryPending,
      synchronousMutationReconciliations: churned.synchronousMutationReconciliations,
      cssReconciliationIntervalMs: bounded.cssReconciliationIntervalMs,
      maxCssReconciliationRecords: bounded.maxCssReconciliationRecords,
      replacementRecords: replaced.recordCount,
    },
  };
  await writeFile(path.join(artifactsPath, `chrome-e2e-${expectedProvider}.json`), `${JSON.stringify(evidence, null, 2)}\n`);
  stage("complete");
  console.log(JSON.stringify(evidence, null, 2));
} finally {
  stage("cleanup");
  await context?.close().catch(() => undefined);
  await closeServer().catch(() => undefined);
  await rm(profilePath, { recursive: true, force: true });
  clearTimeout(watchdog);
}
