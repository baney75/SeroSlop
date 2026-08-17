/* global clearInterval, clearTimeout, document, getComputedStyle, HTMLProgressElement, setInterval, setTimeout, window */
import { createServer } from "node:http";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright-core";
import sharp from "sharp";
import {
  badgeAssociationError,
  browserGeometryBadgeRecord,
  BROWSER_GEOMETRY_POSITION_MARGIN,
  pngDimensions,
  rectsIntersect,
  validateBrowserGeometryEvidence,
} from "./browser-geometry-contract.mjs";

const extensionPath = path.resolve("dist");
const expectedProvider = process.env.PROOFLENS_E2E_PROVIDER === "webgpu" ? "webgpu" : "wasm";
const profilePath = await mkdtemp(path.join(os.tmpdir(), "prooflens-chrome-"));
const artifactsPath = path.resolve("artifacts");
await mkdir(artifactsPath, { recursive: true });
const modelLock = JSON.parse(await readFile("model-lock.json", "utf8"));
const modelSha256 = modelLock.sha256;
const modelStateFixtureRoot = path.resolve("tests/fixtures/model-states");
const modelStateManifestBytes = await readFile(path.join(modelStateFixtureRoot, "fixture-manifest.json"));
const modelStateManifest = JSON.parse(modelStateManifestBytes.toString("utf8"));
if (modelStateManifest.modelSha256 !== modelSha256 || modelStateManifest.items?.length !== 2 ||
  modelStateManifest.minimumLikelyAiScore !== 0.80 || modelStateManifest.maximumBelowThresholdScore !== 0.45) {
  throw new Error("Model-state fixtures do not target the packaged model");
}
const modelStateData = {};
for (const item of modelStateManifest.items) {
  const bytes = await readFile(path.join(modelStateFixtureRoot, item.asset));
  if (createHash("sha256").update(bytes).digest("hex") !== item.assetSha256) {
    throw new Error(`Model-state fixture integrity mismatch: ${item.role}`);
  }
  const mime = /\.png$/iu.test(item.asset) ? "image/png" : /\.webp$/iu.test(item.asset) ? "image/webp" : "image/jpeg";
  modelStateData[item.role] = {
    ...item,
    dataUrl: `data:${mime};base64,${bytes.toString("base64")}`,
  };
}
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

async function waitForEvaluation(
  page,
  predicate,
  argument,
  { timeout = 30_000, interval = 50, label = "browser condition" } = {},
) {
  const deadline = Date.now() + timeout;
  let lastError;
  while (Date.now() < deadline) {
    try {
      if (await page.evaluate(predicate, argument)) return;
      lastError = undefined;
    } catch (error) {
      lastError = error;
    }
    await page.waitForTimeout(interval);
  }
  const suffix = lastError instanceof Error ? `: ${lastError.message}` : "";
  throw new Error(`${label} timed out after ${timeout}ms${suffix}`);
}

async function screenshotRecord(file) {
  const bytes = await readFile(file);
  return {
    sha256: createHash("sha256").update(bytes).digest("hex"),
    ...pngDimensions(bytes),
  };
}

const UI_EVIDENCE_WIDTHS = Object.freeze([320, 375, 414, 768]);
async function captureUiMatrix(page, surface) {
  const originalViewport = page.viewportSize() ?? { width: 1280, height: 720 };
  const records = [];
  for (const theme of ["light", "dark"]) {
    await page.emulateMedia({ colorScheme: theme, reducedMotion: "reduce" });
    for (const width of UI_EVIDENCE_WIDTHS) {
      await page.setViewportSize({ width, height: 900 });
      await page.evaluate(() => { document.body.tabIndex = -1; document.body.focus(); });
      await page.keyboard.press("Tab");
      const layout = await page.evaluate(() => {
        const controls = [...document.querySelectorAll("button, a, summary, label.mode-option")]
          .filter((element) => {
            const rect = element.getBoundingClientRect();
            const style = getComputedStyle(element);
            return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
          });
        const clipped = controls.filter((element) => {
          const rect = element.getBoundingClientRect();
          return rect.left < -1 || rect.right > document.documentElement.clientWidth + 1;
        });
        const active = document.activeElement;
        const activeStyle = active ? getComputedStyle(active) : undefined;
        const labelStyle = active?.matches?.('input[type="radio"]') && active.closest("label")
          ? getComputedStyle(active.closest("label")) : undefined;
        const activeOutline = Number.parseFloat(activeStyle?.outlineWidth ?? "0");
        const labelOutline = Number.parseFloat(labelStyle?.outlineWidth ?? "0");
        const focusStyle = labelOutline > activeOutline ? labelStyle : activeStyle;
        return {
          horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
          clippedControlCount: clipped.length,
          minimumControlHeight: controls.length ? Math.min(...controls.map((element) => element.getBoundingClientRect().height)) : 0,
          focusOutlineWidth: Math.max(activeOutline, labelOutline),
          focusOutlineStyle: focusStyle?.outlineStyle ?? "none",
        };
      });
      if (layout.horizontalOverflow || layout.clippedControlCount !== 0 || layout.minimumControlHeight < 44 ||
        layout.focusOutlineWidth < 2 || layout.focusOutlineStyle === "none") {
        throw new Error(`${surface} ${theme} ${width}px UI contract failed: ${JSON.stringify(layout)}`);
      }
      const file = path.join(artifactsPath, `${surface}-${expectedProvider}-${theme}-${width}.png`);
      await page.screenshot({ path: file, fullPage: true });
      records.push({ theme, width, ...layout, screenshot: await screenshotRecord(file) });
    }
  }
  await page.emulateMedia({ colorScheme: "light", reducedMotion: "reduce" });
  await page.setViewportSize(originalViewport);
  return records;
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
const deepStaticNodes = "<span></span>".repeat(5_200);
const html = `<!doctype html>
<html><head><meta charset="utf-8"><style>
  body{font:16px system-ui;margin:24px;background:#f2f3f6;color:#171b2c}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}figure,.background{margin:0;min-height:260px;background:#fff;border-radius:14px;padding:10px}.background{background-image:url("${fixtureC}"),url("${fixtureA}");background-size:cover}.unavailable{background-image:url("file:///prooflens-unavailable.png");background-size:cover}img{display:block;width:100%;height:260px;object-fit:cover}</style></head>
<body><h1>SeroSlop offline browser contract</h1><main id="grid">
<figure><img id="normal" width="640" height="480" alt="normal fixture" src="${fixtureA}"></figure>
<figure><img id="duplicate" width="640" height="480" alt="duplicate fixture" src="${fixtureA}"></figure>
<figure><picture><source media="(min-width:1px)" srcset="${fixtureB}"><img id="responsive" width="640" height="480" alt="responsive fixture" src="${fixtureA}"></picture></figure>
<div id="css-background" class="background" role="img" aria-label="CSS background fixture"></div>
<div id="unavailable-background" class="background unavailable" role="img" aria-label="Unavailable background fixture"></div>
  </main><section id="prooflens-static-deep-prefix" hidden>${deepStaticNodes}</section>
  <img id="static-after-5000" width="640" height="480" alt="static target after 5000 inert elements" src="${fixtureB}" style="position:fixed;left:12px;bottom:12px;width:96px;height:72px;z-index:1">
  <script>window.addDynamicFixture=()=>{const figure=document.createElement('figure');const image=document.createElement('img');image.id='dynamic';image.width=640;image.height=480;image.alt='dynamic fixture';image.src=${JSON.stringify(fixtureB)};figure.append(image);document.querySelector('#grid').append(figure);};</script></body></html>`;
const controlsHtml = "<!doctype html><html><head><meta charset=utf-8><title>SeroSlop controls fixture</title></head><body><h1>Controls fixture</h1><p>No eligible images are present on this page.</p></body></html>";

function fixtureRequestHandler(request, response) {
  const pathname = request.url?.split("?", 1)[0];
  if (pathname === "/" || pathname === "/controls") {
    response.writeHead(200, {
      "content-type": "text/html; charset=utf-8",
      "cache-control": "no-store",
      connection: "close",
    });
    response.end(pathname === "/controls" ? controlsHtml : html);
    return;
  }
  response.writeHead(404, { connection: "close" }).end();
}

const server = createServer(fixtureRequestHandler);
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
let serverOpen = true;
const address = server.address();
if (!address || typeof address === "string") {
  throw new Error("Smoke servers did not start");
}
const pageUrl = `http://127.0.0.1:${address.port}/`;
const controlsUrl = `http://127.0.0.1:${address.port}/controls`;
const pageOrigin = new URL(pageUrl).origin;

async function closeHttpServer(instance, label) {
  const closing = new Promise((resolve, reject) => {
    instance.close((error) => error ? reject(error) : resolve());
    instance.closeAllConnections();
  });
  await withTimeout(closing, 5_000, label);
}

async function closeServer() {
  const closings = [];
  if (serverOpen) {
    serverOpen = false;
    closings.push(closeHttpServer(server, "fixture server shutdown"));
  }
  await Promise.all(closings);
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
let setupProgressAdvanced = false;
let setupPreparingState;
let setupInitialFailureRecovered = false;
let setupExplicitModeSelected = false;
let setupVisualEvidence;
let popupUnsupportedVisualEvidence;
let popupErrorVisualEvidence;
let popupPickVisualEvidence;
let popupSupportedVisualEvidence;
let popupModeSheetVisualEvidence;
let popupSupportedPageControls = false;
let popupRescanFeedback = false;
let popupRescanWork;
let modeSaveFailureTruthful = false;
let sameOriginNavigationRejected = false;
let pickModeNoAutomaticAnalysis = false;
let pickerTargetOutlined = false;
let pickerEscapeCancelled = false;
let pickerExactOneTarget = false;
let pickerHostileImageCountBounded = false;
let mainModeScoped = false;
let allModePersisted = false;
let modelStateFixtures;
let modelStateGeometryEvidence;
let narrowGeometryEvidence;
let smallTargetGeometryEvidence;
try {
  stage("initial browser launch");
  context = await launch();
  let worker = await extensionWorker(context);
  const extensionId = new URL(worker.url()).host;
  const setup = context.pages().find((candidate) => candidate.url().includes("/setup.html")) ?? await context.newPage();
  stage("model setup");
  if (!setup.url().includes("/setup.html")) await setup.goto(`chrome-extension://${extensionId}/setup.html`);
  await setup.getByRole("progressbar", { name: "Model preparation progress" }).waitFor({ timeout: 30_000 });
  setupProgressAccessibleName = true;
  const progress = setup.getByRole("progressbar", { name: "Model preparation progress" });
  await setup.getByRole("heading", { name: "Verifying model" }).waitFor({ timeout: 30_000 });
  const preparingButton = setup.locator("#prepare-model");
  if (!await preparingButton.isDisabled() || await preparingButton.isVisible()) {
    throw new Error("Setup preparation exposed an inactive action button");
  }
  const initialProgress = await progress.evaluate(
    (element) => Number(element.dataset.minimumObservedBytes ?? element.value),
  );
  await waitForEvaluation(
    setup,
    (initial) => {
      const element = document.querySelector("#setup-progress");
      return element instanceof HTMLProgressElement && element.value > initial;
    },
    initialProgress,
    { timeout: 30_000, label: "setup progress advancement" },
  );
  setupPreparingState = await progress.evaluate((element, initialBytes) => ({
    initialBytes,
    observedBytes: Number(element.value),
    totalBytes: Number(element.max),
    visibleText: document.querySelector("#setup-progress-text")?.textContent,
    buttonDisabled: document.querySelector("#prepare-model")?.disabled === true,
  }), initialProgress);
  setupProgressAdvanced = setupPreparingState.observedBytes > setupPreparingState.initialBytes &&
    setupPreparingState.observedBytes <= setupPreparingState.totalBytes && setupPreparingState.buttonDisabled;
  if (!setupProgressAdvanced || !/MB$/u.test(setupPreparingState.visibleText ?? "")) {
    throw new Error(`Setup did not expose determinate progress: ${JSON.stringify(setupPreparingState)}`);
  }
  await setup.screenshot({ path: path.join(artifactsPath, `setup-${expectedProvider}-preparing.png`), fullPage: true });
  await setup.getByRole("heading", { name: "Offline ready" }).waitFor({ timeout: 300_000 });
  // A fresh install must not silently choose a scan mode. Select one explicitly
  // before restart so the persistence check below exercises the real onboarding path.
  await setup.locator('input[name="scan-mode"][value="all"]').check();
  await setup.getByRole("button", { name: "Save mode" }).click();
  await setup.getByRole("status").filter({ hasText: "Saved: Every image" }).waitFor({ timeout: 10_000 });
  setupExplicitModeSelected = true;
  if (await setup.locator("#prepare-model").isVisible()) throw new Error("Ready setup retained an inactive action button");
  const observedProgress = await progress.evaluate((element) => ({
    minimumBytes: Number(element.dataset.minimumObservedBytes),
    maximumBytes: Number(element.dataset.maximumObservedBytes),
    renderCount: Number(element.dataset.renderCount),
    value: Number(element.value),
    max: Number(element.max),
  }));
  if (!(observedProgress.minimumBytes < observedProgress.maximumBytes) ||
    observedProgress.maximumBytes !== observedProgress.max || observedProgress.value !== observedProgress.max ||
    observedProgress.renderCount < 3) {
    throw new Error(`Setup progress history was not determinate: ${JSON.stringify(observedProgress)}`);
  }
  setupVisualEvidence = await captureUiMatrix(setup, "setup");
  // The matrix already captures the ready state at four widths in both themes.
  // Reuse its 375px light image for the legacy artifact name instead of asking
  // Chromium/Xvfb to encode a redundant full-page frame before the viewport is
  // initialized for visual evidence.
  await copyFile(
    path.join(artifactsPath, `setup-${expectedProvider}-light-375.png`),
    path.join(artifactsPath, `setup-${expectedProvider}.png`),
  );
  stage("setup initial-status failure recovery");
  const failureSetup = await context.newPage();
  await failureSetup.addInitScript(() => {
    const original = chrome.runtime.sendMessage.bind(chrome.runtime);
    let failed = false;
    Object.defineProperty(chrome.runtime, "sendMessage", {
      configurable: true,
      value(...arguments_) {
        if (!failed && arguments_[0]?.type === "PL_GET_MODEL_STATUS") {
          failed = true;
          return Promise.reject(new Error("Injected initial model-status failure"));
        }
        return original(...arguments_);
      },
    });
  });
  await failureSetup.goto(`chrome-extension://${extensionId}/setup.html`);
  await failureSetup.getByRole("heading", { name: "Setup failed" }).waitFor({ timeout: 30_000 });
  const retryVerification = failureSetup.getByRole("button", { name: "Retry verification" });
  if (await retryVerification.isDisabled()) throw new Error("Initial setup failure did not enable recovery");
  await failureSetup.screenshot({ path: path.join(artifactsPath, `setup-${expectedProvider}-failure.png`), fullPage: true });
  await retryVerification.click();
  await failureSetup.getByRole("heading", { name: "Offline ready" }).waitFor({ timeout: 300_000 });
  setupInitialFailureRecovered = true;
  await failureSetup.close();
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
  const popupCaveatVisible = await statusPage.locator(".caveat").count() > 0;
  const popupUnsupportedGuard = await statusPage.getByRole("button", { name: "This page can’t be scanned" }).isDisabled();
  if (popupCaveatVisible || !popupUnsupportedGuard) throw new Error("Popup retained the removed caveat or exposed an action on an unsupported page");
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
  const restartedMode = await worker.evaluate(() => chrome.storage.local.get("scanMode"));
  if (restartedMode.scanMode !== "all") throw new Error(`Setup mode did not persist across restart: ${JSON.stringify(restartedMode)}`);
  await statusPage.screenshot({ path: path.join(artifactsPath, `popup-${expectedProvider}.png`), fullPage: true });
  popupUnsupportedVisualEvidence = await captureUiMatrix(statusPage, "popup-unsupported");
  await worker.evaluate(() => chrome.storage.local.set({ disabledOrigins: [], scanMode: "all" }));
  const page = await context.newPage();
  await page.emulateMedia({ reducedMotion: "reduce" });
  stage("supported-page control fixture navigation");
  await page.goto(controlsUrl, { waitUntil: "load", timeout: 30_000 });
  stage("supported-page tab discovery");
  const tabId = await withTimeout(worker.evaluate(async (url) => {
    const tabs = await chrome.tabs.query({ url });
    return tabs[0]?.id;
  }, controlsUrl), 10_000, "supported-page tab discovery");
  if (typeof tabId !== "number") throw new Error("Could not identify the supported test tab");
  stage("supported-page popup contract");
  await statusPage.getByRole("button", { name: "This page can’t be scanned" }).waitFor();
  if (!await statusPage.getByRole("button", { name: "This page can’t be scanned" }).isDisabled()) {
    throw new Error("Unsupported page action was not disabled");
  }
  popupSupportedPageControls = true;
  stage("three scan modes and explicit picker");
  await page.goto(pageUrl, { waitUntil: "load", timeout: 30_000 });
  await worker.evaluate((id) => chrome.tabs.update(id, { active: true }), tabId);
  await statusPage.reload({ waitUntil: "load", timeout: 30_000 });
  await page.locator("#normal").waitFor({ timeout: 30_000 });
  await waitForEvaluation(page, () => [...document.images].every((image) => image.complete), undefined,
    { timeout: 30_000, label: "mode fixture image loading" });
  stage("same-origin popup document guard");
  const modeBeforeNavigation = await worker.evaluate(() => chrome.storage.local.get("scanMode"));
  const snapshotBeforeNavigation = await contentSnapshot(statusPage, tabId);
  if (!snapshotBeforeNavigation.documentToken) throw new Error("Document token was unavailable before navigation");
  const sameOriginNavigationUrl = new URL(pageUrl);
  sameOriginNavigationUrl.searchParams.set("document", "changed");
  await page.goto(sameOriginNavigationUrl.href, { waitUntil: "load", timeout: 30_000 });
  await waitForEvaluation(statusPage, async ({ id, token }) => {
    const snapshot = await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" });
    return Boolean(snapshot.documentToken && snapshot.documentToken !== token);
  }, { id: tabId, token: snapshotBeforeNavigation.documentToken }, { timeout: 30_000, label: "same-origin document replacement" });
  const rejectedNavigationMutation = await worker.evaluate(({ id, origin, token }) => chrome.tabs.sendMessage(id, {
    type: "PL_SCAN_MODE_CHANGED",
    scanMode: "main",
    expectedOrigin: origin,
    expectedDocumentToken: token,
  }), { id: tabId, origin: pageOrigin, token: snapshotBeforeNavigation.documentToken });
  if (rejectedNavigationMutation?.pageChanged !== true || rejectedNavigationMutation?.scanMode) {
    throw new Error(`Replaced same-origin document accepted stale popup authority: ${JSON.stringify(rejectedNavigationMutation)}`);
  }
  const modeAfterNavigation = await worker.evaluate(() => chrome.storage.local.get("scanMode"));
  if (JSON.stringify(modeAfterNavigation) !== JSON.stringify(modeBeforeNavigation)) throw new Error("Same-origin navigation mutated the saved mode");
  sameOriginNavigationRejected = true;
  await page.goto(pageUrl, { waitUntil: "load", timeout: 30_000 });
  await statusPage.reload({ waitUntil: "load", timeout: 30_000 });

  stage("mode-save failure recovery");
  await statusPage.getByRole("button", { name: "Change" }).click();
  await statusPage.locator('input[name="popup-mode"][value="main"]').check();
  await statusPage.evaluate(() => {
    const original = chrome.runtime.sendMessage.bind(chrome.runtime);
    Object.defineProperty(chrome.runtime, "sendMessage", {
      configurable: true,
      value: (...arguments_) => arguments_[0]?.type === "PL_SET_SCAN_MODE"
        ? Promise.reject(new Error("Injected mode storage failure"))
        : original(...arguments_),
    });
  });
  await statusPage.getByRole("button", { name: "Save mode" }).click();
  await statusPage.locator("#mode-error").filter({ hasText: "Couldn’t save your mode" }).waitFor({ timeout: 10_000 });
  const modeAfterFailedSave = await worker.evaluate(() => chrome.storage.local.get("scanMode"));
  if (modeAfterFailedSave.scanMode !== "all") throw new Error("Failed mode save changed persisted state");
  modeSaveFailureTruthful = true;
  popupErrorVisualEvidence = await captureUiMatrix(statusPage, "popup-mode-error");
  await statusPage.reload({ waitUntil: "load", timeout: 30_000 });
  const setMode = async (mode) => {
    const titles = { pick: "Choose an image", main: "Main images", all: "Every image" };
    await statusPage.getByRole("button", { name: "Change" }).click();
    await statusPage.locator(`input[name="popup-mode"][value="${mode}"]`).check();
    await statusPage.getByRole("button", { name: "Save mode" }).click();
    await statusPage.locator("#mode-summary").filter({ hasText: titles[mode] }).waitFor({ timeout: 10_000 });
    await statusPage.locator("#mode-sheet").waitFor({ state: "hidden", timeout: 10_000 });
  };
  await setMode("pick");
  await page.locator("#normal").focus();
  await page.locator("#normal").evaluate((element, source) => {
    element.style.backgroundImage = `url(${JSON.stringify(source)})`;
  }, fixtureB);
  const pickBefore = await contentSnapshot(statusPage, tabId);
  if (pickBefore.badges.length !== 0) throw new Error("Pick mode performed automatic analysis");
  pickModeNoAutomaticAnalysis = true;
  const pickActionLabel = await statusPage.locator("#primary-action").textContent();
  if (pickActionLabel !== "Choose image") {
    throw new Error(`Pick action was not ready: ${JSON.stringify({ pickActionLabel, pickBefore, popup: await statusPage.locator("body").innerText() })}`);
  }
  popupPickVisualEvidence = await captureUiMatrix(statusPage, "popup-pick");
  await statusPage.getByRole("button", { name: "Choose image", exact: true }).click();
  await waitForEvaluation(statusPage, async (id) => {
    const snapshot = await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" });
    return snapshot.pickerActive === true && Boolean(snapshot.pickerTarget);
  }, tabId, { label: "picker outline" });
  const pickerGeometry = (await contentSnapshot(statusPage, tabId)).pickerTarget;
  if (!pickerGeometry?.outline.includes("rgb(239, 51, 59)") || pickerGeometry.outlineOffset !== "2px" ||
    pickerGeometry.pointerEvents !== "none" || pickerGeometry.controllerFocused !== true ||
    !pickerGeometry.instruction.includes("normal fixture") || !pickerGeometry.instruction.includes(" of ") ||
    !pickerGeometry.instruction.includes("Tab") || !pickerGeometry.instruction.includes("Enter") ||
    !pickerGeometry.instruction.includes("Esc")) {
    throw new Error(`Picker target was not unambiguously outlined: ${JSON.stringify(pickerGeometry)}`);
  }
  pickerTargetOutlined = true;
  await page.keyboard.press("Escape");
  await waitForEvaluation(statusPage, async (id) => {
    const snapshot = await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" });
    return snapshot.pickerActive === false && !snapshot.pickerTarget && snapshot.lastPickerFocusRestored === true;
  }, tabId, { label: "picker escape cancellation" });
  pickerEscapeCancelled = true;
  await page.evaluate((source) => {
    const image = document.createElement("img");
    image.id = "picker-initially-below-fold";
    image.alt = "initially below fold picker fixture";
    image.src = source;
    image.style.cssText = "display:block;width:320px;height:240px;margin-top:1600px";
    document.body.append(image);
    const hostileImages = document.createElement("div");
    hostileImages.id = "picker-hostile-image-count";
    hostileImages.hidden = true;
    const fragment = document.createDocumentFragment();
    for (let index = 0; index < 5_200; index += 1) fragment.append(document.createElement("img"));
    hostileImages.append(fragment);
    document.body.append(hostileImages);
  }, fixtureB);
  await page.locator("#picker-initially-below-fold").waitFor();
  if (await page.locator("img").count() < 5_200) throw new Error("Picker hostile-image fixture was not installed");
  await statusPage.getByRole("button", { name: "Choose image", exact: true }).click();
  await page.locator("#picker-initially-below-fold").scrollIntoViewIfNeeded();
  await page.locator("#picker-initially-below-fold").click();
  await waitForEvaluation(statusPage, async (id) => {
    const badges = (await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" })).badges;
    return badges.length === 1 && badges[0]?.elementId === "picker-initially-below-fold";
  }, tabId, { timeout: 30_000, label: "single picker target" });
  const pickResult = await badgeSnapshot(statusPage, tabId);
  if (pickResult.length !== 1 || pickResult[0].elementId !== "picker-initially-below-fold") throw new Error("Pick mode selected more than one target");
  await page.locator("#picker-initially-below-fold").evaluate((element) => { element.style.backgroundPosition = "1px 1px"; });
  await waitForEvaluation(statusPage, async (id) => {
    const snapshot = await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" });
    return snapshot.recordCount === 1 && snapshot.badges[0]?.elementId === "picker-initially-below-fold";
  }, tabId, { timeout: 30_000, label: "picker exact-one reconciliation" });
  pickerExactOneTarget = true;
  pickerHostileImageCountBounded = true;
  await page.locator("#picker-initially-below-fold").evaluate((element) => element.remove());
  await page.locator("#picker-hostile-image-count").evaluate((element) => element.remove());

  await setMode("main");
  await waitForEvaluation(statusPage, async (id) => {
    const badges = (await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" })).badges;
    return badges.length > 0 && badges.every((badge) => badge.elementId !== "static-after-5000");
  }, tabId, { timeout: 60_000, label: "main-content mode scope" });
  const mainResult = await badgeSnapshot(statusPage, tabId);
  if (mainResult.some((badge) => badge.elementId === "static-after-5000")) throw new Error("Main mode fell back to all images");
  await page.evaluate(() => {
    const normal = document.querySelector("#normal");
    const main = normal?.parentElement;
    if (!normal || !main) throw new Error("Main fixture missing");
    window.__seroslopMainHome = main;
    const aside = document.createElement("aside");
    aside.id = "mode-excluded-aside";
    const article = document.createElement("article");
    const nested = document.createElement("img");
    nested.id = "mode-nested-excluded";
    nested.alt = "nested excluded fixture";
    nested.src = normal.tagName === "IMG" ? (normal.currentSrc || normal.getAttribute("src") || "") : "";
    nested.style.cssText = "width:96px;height:72px";
    article.append(nested);
    aside.append(article);
    document.body.append(aside);
  });
  await waitForEvaluation(page, () => document.querySelector("#mode-nested-excluded")?.complete === true, undefined,
    { timeout: 30_000, label: "nested excluded image loading" });
  await page.waitForTimeout(1_200);
  if ((await badgeSnapshot(statusPage, tabId)).some((badge) => badge.elementId === "mode-nested-excluded")) {
    throw new Error("Main mode admitted an article nested inside an aside");
  }
  await page.evaluate(() => document.querySelector("#mode-excluded-aside")?.append(document.querySelector("#normal")));
  await waitForEvaluation(statusPage, async (id) => {
    const badges = (await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" })).badges;
    return badges.every((badge) => badge.elementId !== "normal");
  }, tabId, { timeout: 30_000, label: "main target removal after excluded move" });
  await page.evaluate(() => window.__seroslopMainHome?.prepend(document.querySelector("#normal")));
  await waitForEvaluation(statusPage, async (id) => {
    const badges = (await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" })).badges;
    return badges.some((badge) => badge.elementId === "normal");
  }, tabId, { timeout: 30_000, label: "main target readmission" });
  await page.evaluate(() => document.querySelector("#mode-excluded-aside")?.remove());
  mainModeScoped = true;

  await setMode("all");
  const persistedMode = await worker.evaluate(() => chrome.storage.local.get("scanMode"));
  if (persistedMode.scanMode !== "all") throw new Error(`Mode did not persist: ${JSON.stringify(persistedMode)}`);
  allModePersisted = true;

  stage("fixture navigation while analysis is disabled");
  await page.goto(pageUrl, { waitUntil: "load", timeout: 30_000 });
  await page.locator("#normal").waitFor({ timeout: 30_000 });
  await waitForEvaluation(
    page,
    () => [...document.images].every((image) => image.complete),
    undefined,
    { timeout: 30_000, label: "fixture image loading" },
  );
  await statusPage.reload({ waitUntil: "load", timeout: 30_000 });
  await statusPage.getByRole("button", { name: "Re-scan every image" }).waitFor({ timeout: 30_000 });
  stage("fixture server shutdown");
  await closeServer();
  stage("fixture offline transition");
  await withTimeout(context.setOffline(true), 10_000, "browser offline transition");
  context.on("request", (request) => {
    if (/^https?:/u.test(request.url())) postCutoffNetworkRequests.push(request.url());
  });
  stage("fixture analysis enabled in all mode");
  await waitForEvaluation(statusPage, async (id) => {
    try {
      const snapshot = await chrome.tabs.sendMessage(id, { type: "PL_GET_CONTENT_SNAPSHOT" });
      return snapshot.enabled === true && snapshot.scanMode === "all";
    }
    catch { return false; }
  }, tabId, { label: "offline enabled-site state" });
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
      if (current.filter((badge) => badge.state === "complete").length === 6 &&
        current.filter((badge) => badge.state === "unavailable").length === 1) break;
      await page.waitForTimeout(500);
    }
    const settled = await badgeSnapshot(statusPage, tabId);
    if (settled.filter((badge) => badge.state === "complete").length !== 6 ||
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
  if (badges.length !== 7 || completed.length !== 6 || unavailable.length !== 1) {
    throw new Error(`Unexpected target counts: ${JSON.stringify(badges)}`);
  }
  for (const badge of completed) {
    if (!/^(?:Likely AI|Below flag threshold) · \d{1,3}\.\d\/100$/u.test(badge.text ?? "")) {
      throw new Error(`Missing numeric model score: ${JSON.stringify(badge)}`);
    }
    const runtimeLabel = expectedProvider === "webgpu" ? "WebGPU" : "WASM";
    if (badge.provider !== expectedProvider || !badge.title?.includes(`Ran with ${runtimeLabel}`)) {
      throw new Error(`${runtimeLabel} provider was not observable: ${JSON.stringify(badge)}`);
    }
  }
  const normalBadge = completed.find((badge) => badge.elementId === "normal");
  if (!normalBadge?.accessibleName?.includes("image “normal fixture”")) {
    throw new Error(`Completed result lacks target-specific accessible context: ${JSON.stringify(normalBadge)}`);
  }
  const accessibleCssBadge = completed.find((badge) => badge.elementId === "css-background");
  if (!accessibleCssBadge?.accessibleName?.includes("CSS background “CSS background fixture”")) {
    throw new Error(`CSS result lacks target-specific accessible context: ${JSON.stringify(accessibleCssBadge)}`);
  }
  if (!unavailable[0]?.text?.includes("unavailable")) throw new Error("Unavailable target fabricated a score");
  const staticAfter5000 = completed.find((badge) => badge.elementId === "static-after-5000");
  if (!staticAfter5000 || staticAfter5000.state !== "complete") {
    throw new Error(`A static target beyond the first 5,000 DOM elements was starved: ${JSON.stringify(badges)}`);
  }
  const initialTraversal = await contentSnapshot(statusPage, tabId);
  if (initialTraversal.fullDocumentScanRequired || initialTraversal.fullDocumentRestartRequested) {
    throw new Error(`The bounded initial traversal did not reach document EOF: ${JSON.stringify(initialTraversal)}`);
  }
  if (postCutoffNetworkRequests.length) {
    throw new Error(`Network request occurred after server shutdown/offline cutoff: ${postCutoffNetworkRequests.join(", ")}`);
  }

  stage("supported-page popup interactions");
  await statusPage.screenshot({
    path: path.join(artifactsPath, `popup-${expectedProvider}-supported.png`),
    fullPage: true,
  });
  popupSupportedVisualEvidence = await captureUiMatrix(statusPage, "popup-supported");
  await statusPage.getByRole("button", { name: "Change" }).click();
  popupModeSheetVisualEvidence = await captureUiMatrix(statusPage, "popup-mode-sheet");
  await statusPage.getByRole("button", { name: "Back" }).click();
  const offlineRescan = statusPage.getByRole("button", { name: "Re-scan every image" });
  const rescanAcceptedBefore = normalBadge.acceptedResultCount;
  await offlineRescan.click();
  const scanningAgain = statusPage.getByRole("button", { name: "Scanning again…" });
  await scanningAgain.waitFor();
  if (!await scanningAgain.isDisabled()) throw new Error("Re-scan feedback did not expose a disabled busy state");
  await statusPage.screenshot({
    path: path.join(artifactsPath, `popup-${expectedProvider}-supported-scanning.png`),
    fullPage: true,
  });
  await statusPage.getByRole("button", { name: "Re-scan every image" }).waitFor({ timeout: 10_000 });
  let rescannedNormal;
  for (let attempt = 0; attempt < 480; attempt += 1) {
    const current = (await badgeSnapshot(statusPage, tabId)).find((badge) => badge.elementId === "normal");
    if (current?.state === "complete" && current.acceptedResultCount > rescanAcceptedBefore) {
      rescannedNormal = current;
      break;
    }
    await page.waitForTimeout(25);
  }
  if (!rescannedNormal) throw new Error("Re-scan control did not produce a fresh completed target result");
  let rescanSummary;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    rescanSummary = await statusPage.evaluate(
      (id) => chrome.runtime.sendMessage({ type: "PL_GET_TAB_SUMMARY", tabId: id }),
      tabId,
    );
    if (rescanSummary?.stats?.complete === 6 && rescanSummary?.stats?.unavailable === 1) break;
    await page.waitForTimeout(25);
  }
  if (rescanSummary?.stats?.complete !== 6 || rescanSummary?.stats?.unavailable !== 1) {
    throw new Error(`Re-scan did not settle the page summary: ${JSON.stringify(rescanSummary)}`);
  }
  await statusPage.locator("#page-summary").filter({ hasText: /complete/u }).waitFor({ timeout: 10_000 });
  popupRescanFeedback = true;
  popupRescanWork = {
    target: "normal",
    acceptedBefore: rescanAcceptedBefore,
    acceptedAfter: rescannedNormal.acceptedResultCount,
    finalState: rescannedNormal.state,
  };

  stage("high-margin model-state fixtures");
  await page.evaluate(({ likely, below }) => {
    const container = document.createElement("section");
    container.id = "prooflens-model-state-fixtures";
    container.style.cssText = "display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:18px";
    for (const [id, alt, source] of [
      ["likely-ai-fixture", "high-margin likely AI fixture", likely],
      ["below-threshold-fixture", "high-margin below threshold fixture", below],
    ]) {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.id = id;
      image.alt = alt;
      image.src = source;
      figure.append(image);
      container.append(figure);
    }
    document.body.append(container);
  }, {
    likely: modelStateData["likely-ai"].dataUrl,
    below: modelStateData["below-threshold"].dataUrl,
  });
  const modelStateContainer = page.locator("#prooflens-model-state-fixtures");
  await modelStateContainer.scrollIntoViewIfNeeded({ timeout: 30_000 });
  await waitForEvaluation(
    page,
    () => [...document.querySelectorAll("#prooflens-model-state-fixtures img")]
      .every((image) => image.complete && image.naturalWidth > 0 && image.naturalHeight > 0),
    undefined,
    { timeout: 30_000, label: "model-state fixture loading" },
  );
  let stateBadges;
  for (let attempt = 0; attempt < 480; attempt += 1) {
    const current = await badgeSnapshot(statusPage, tabId);
    const likely = current.find((badge) => badge.elementId === "likely-ai-fixture");
    const below = current.find((badge) => badge.elementId === "below-threshold-fixture");
    if (likely?.state === "complete" && below?.state === "complete") {
      stateBadges = { likely, below };
      break;
    }
    await page.waitForTimeout(250);
  }
  if (!stateBadges) {
    const failureSnapshot = await contentSnapshot(statusPage, tabId);
    throw new Error(`High-margin model-state fixtures did not complete: ${JSON.stringify(failureSnapshot)}`);
  }
  const scoreFromBadge = (badge) => Number.parseFloat(String(badge.text).match(/(\d{1,3}\.\d)\/100/u)?.[1] ?? "NaN");
  const likelyScore = scoreFromBadge(stateBadges.likely);
  const belowScore = scoreFromBadge(stateBadges.below);
  const likelyReferenceScore = modelStateData["likely-ai"].referenceDisplayScore * 100;
  const belowReferenceScore = modelStateData["below-threshold"].referenceDisplayScore * 100;
  const runtimeLabel = expectedProvider === "webgpu" ? "WebGPU" : "WASM";
  if (stateBadges.likely.classification !== "likely-ai" ||
    !String(stateBadges.likely.text).startsWith("Likely AI · ") || likelyScore < 80 ||
    !stateBadges.likely.accessibleName?.includes("image “high-margin likely AI fixture”") ||
    stateBadges.likely.provider !== expectedProvider || !stateBadges.likely.title?.includes(`Ran with ${runtimeLabel}`)) {
    throw new Error(`Likely-AI fixture rendered the wrong state: ${JSON.stringify(stateBadges.likely)}`);
  }
  if (stateBadges.below.classification !== "not-flagged" ||
    !String(stateBadges.below.text).startsWith("Below flag threshold · ") || belowScore > 45 ||
    !stateBadges.below.accessibleName?.includes("image “high-margin below threshold fixture”") ||
    stateBadges.below.provider !== expectedProvider || !stateBadges.below.title?.includes(`Ran with ${runtimeLabel}`)) {
    throw new Error(`Below-threshold fixture rendered the wrong state: ${JSON.stringify(stateBadges.below)}`);
  }
  if (Math.abs(likelyScore - likelyReferenceScore) > 0.2 || Math.abs(belowScore - belowReferenceScore) > 0.2) {
    throw new Error(`Browser fixture scores diverged from validation reference: ${JSON.stringify({
      likelyScore, likelyReferenceScore, belowScore, belowReferenceScore,
    })}`);
  }
  modelStateFixtures = {
    fixtureManifestSha256: createHash("sha256").update(modelStateManifestBytes).digest("hex"),
    likelyAi: {
      id: modelStateData["likely-ai"].id,
      referenceScore: likelyReferenceScore,
      observedScore: likelyScore,
      classification: stateBadges.likely.classification,
    },
    belowThreshold: {
      id: modelStateData["below-threshold"].id,
      referenceScore: belowReferenceScore,
      observedScore: belowScore,
      classification: stateBadges.below.classification,
    },
  };
  const stateGeometry = await contentSnapshot(statusPage, tabId);
  const stateViewport = page.viewportSize();
  if (!stateViewport) throw new Error("Model-state viewport geometry was unavailable");
  const stateVisibleTargets = stateGeometry.badges
    .map((badge) => badge.targetRect)
    .filter((rect) => rect.right > 0 && rect.bottom > 0 && rect.left < stateViewport.width && rect.top < stateViewport.height);
  if (stateGeometry.badges.some((badge) => badge.hidden && badge.display !== "none")) {
    throw new Error(`Collision-hidden model-state labels remained visually rendered: ${JSON.stringify(stateGeometry.badges)}`);
  }
  for (const badge of stateGeometry.badges.filter((candidate) => !candidate.hidden && candidate.display !== "none")) {
    const associationError = badgeAssociationError(badge, stateVisibleTargets);
    if (associationError) {
      throw new Error(`Model-state label lost its target association (${associationError}): ${JSON.stringify(badge)}`);
    }
  }
  await page.evaluate(() => { document.querySelector("#static-after-5000")?.style.setProperty("opacity", "0"); });
  await page.waitForTimeout(50);
  const modelStateScreenshotPath = path.join(artifactsPath, `chrome-e2e-${expectedProvider}-states.png`);
  await page.screenshot({ path: modelStateScreenshotPath, fullPage: false });
  modelStateGeometryEvidence = {
    viewport: { width: stateViewport.width, height: stateViewport.height },
    screenshot: await screenshotRecord(modelStateScreenshotPath),
    badges: stateGeometry.badges.map(browserGeometryBadgeRecord),
  };
  await page.evaluate(() => { document.querySelector("#static-after-5000")?.style.removeProperty("opacity"); });
  await page.evaluate(() => document.querySelector("#prooflens-model-state-fixtures")?.remove());
  let modelStateFixturesRemoved = false;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    const current = await badgeSnapshot(statusPage, tabId);
    if (!current.some((badge) => ["likely-ai-fixture", "below-threshold-fixture"].includes(badge.elementId))) {
      modelStateFixturesRemoved = true;
      break;
    }
    await page.waitForTimeout(50);
  }
  if (!modelStateFixturesRemoved) throw new Error("Model-state fixtures were not removed cleanly");

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

  // A CSSStyleRule mutation is invisible to MutationObserver. Change the rule
  // while capture is rate-limited and require the acceptance-time descriptor
  // check to reject the old response before periodic reconciliation runs.
  stage("CSSOM stale-response race");
  await page.evaluate((source) => {
    const style = document.createElement("style");
    style.id = "prooflens-cssom-race-style";
    style.textContent = `.prooflens-cssom-race{background-image:url("${source}");background-size:cover}`;
    const target = document.createElement("div");
    target.id = "cssom-race";
    target.className = "prooflens-cssom-race";
    target.setAttribute("role", "img");
    target.setAttribute("aria-label", "CSSOM stale-response race fixture");
    target.style.cssText = "position:fixed;inset:10px 10px auto auto;z-index:2;width:640px;height:480px";
    document.head.append(style);
    document.body.append(target);
  }, fixtureRace);
  let cssomRaceStarted;
  for (let attempt = 0; attempt < 240; attempt += 1) {
    cssomRaceStarted = cssBadge(await badgeSnapshot(statusPage, tabId), "cssom-race");
    if (cssomRaceStarted?.state === "analyzing") break;
    await page.waitForTimeout(10);
  }
  if (cssomRaceStarted?.state !== "analyzing" || cssomRaceStarted.acceptedResultCount !== 0) {
    throw new Error(`CSSOM race request did not enter the analyzing state: ${JSON.stringify(cssomRaceStarted)}`);
  }
  await page.evaluate(() => {
    const style = document.querySelector("#prooflens-cssom-race-style");
    const rule = style?.sheet?.cssRules[0];
    if (!rule || !("style" in rule)) throw new Error("CSSOM race rule missing");
    rule.style.backgroundImage = "url(\"file:///prooflens-cssom-race-invalidated.png\")";
  });
  let cssomRaceInvalidated;
  for (let attempt = 0; attempt < 160; attempt += 1) {
    cssomRaceInvalidated = cssBadge(await badgeSnapshot(statusPage, tabId), "cssom-race");
    if ((cssomRaceInvalidated?.acceptedResultCount ?? 0) > 0 || cssomRaceInvalidated?.state === "complete") {
      throw new Error(`A stale CSSOM response was accepted: ${JSON.stringify(cssomRaceInvalidated)}`);
    }
    if (cssomRaceInvalidated?.state === "unavailable") break;
    await page.waitForTimeout(25);
  }
  if (cssomRaceInvalidated?.state !== "unavailable" || cssomRaceInvalidated.acceptedResultCount !== 0) {
    throw new Error(`CSSOM in-flight source change was not rejected: ${JSON.stringify(cssomRaceInvalidated)}`);
  }
  await page.evaluate((source) => {
    const style = document.querySelector("#prooflens-cssom-race-style");
    const rule = style?.sheet?.cssRules[0];
    if (!rule || !("style" in rule)) throw new Error("CSSOM race rule missing");
    rule.style.backgroundImage = `url("${source}")`;
  }, fixtureB);
  for (let attempt = 0; attempt < 160; attempt += 1) {
    const current = cssBadge(await badgeSnapshot(statusPage, tabId), "cssom-race");
    if (current?.state === "complete" && current.acceptedResultCount === 1) break;
    await page.waitForTimeout(25);
  }
  const cssomRaceRecovered = cssBadge(await badgeSnapshot(statusPage, tabId), "cssom-race");
  if (cssomRaceRecovered?.state !== "complete" || cssomRaceRecovered.acceptedResultCount !== 1) {
    throw new Error(`CSSOM race target did not recover exactly once: ${JSON.stringify(cssomRaceRecovered)}`);
  }
  await page.evaluate(() => {
    document.querySelector("#cssom-race")?.remove();
    document.querySelector("#prooflens-cssom-race-style")?.remove();
  });
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if (!cssBadge(await badgeSnapshot(statusPage, tabId), "cssom-race")) break;
    await page.waitForTimeout(25);
  }
  if (cssBadge(await badgeSnapshot(statusPage, tabId), "cssom-race")) {
    throw new Error("Removed CSSOM race fixture retained a label");
  }

  // CSSOM-only mutation regression: no DOM attribute mutation occurs; the
  // bounded periodic reconciliation must still notice and reanalyze it.
  stage("CSSOM reconciliation");
  const cssomBaseline = cssBadge(await badgeSnapshot(statusPage, tabId));
  if (cssomBaseline?.state !== "complete") {
    throw new Error(`CSSOM baseline was not complete: ${JSON.stringify(cssomBaseline)}`);
  }
  const cssomAcceptedBefore = cssomBaseline.acceptedResultCount;
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
  if (cssBadge(cssomInvalidated)?.state !== "unavailable" ||
    cssBadge(cssomInvalidated)?.acceptedResultCount !== cssomAcceptedBefore) {
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
    if (cssBadge(current)?.state === "complete" &&
      cssBadge(current)?.acceptedResultCount === cssomAcceptedBefore + 1) break;
    await page.waitForTimeout(50);
  }
  const reanalyzed = await badgeSnapshot(statusPage, tabId);
  if (cssBadge(reanalyzed)?.state !== "complete" ||
    cssBadge(reanalyzed)?.acceptedResultCount !== cssomAcceptedBefore + 1) {
    throw new Error(`CSS replacement source was not analyzed: ${JSON.stringify(reanalyzed)}`);
  }
  let summary;
  for (let attempt = 0; attempt < 120; attempt += 1) {
    summary = await statusPage.evaluate(
      (id) => chrome.runtime.sendMessage({ type: "PL_GET_TAB_SUMMARY", tabId: id }),
      tabId,
    );
    const currentBadges = await badgeSnapshot(statusPage, tabId);
    const final = summary?.stats?.complete === 6 && summary?.stats?.unavailable === 1 &&
      currentBadges.filter((badge) => badge.state === "complete").length === 6 &&
      currentBadges.filter((badge) => badge.state === "unavailable").length === 1;
    if (final) {
      await page.waitForTimeout(750);
      const confirmed = await statusPage.evaluate(
        (id) => chrome.runtime.sendMessage({ type: "PL_GET_TAB_SUMMARY", tabId: id }),
        tabId,
      );
      const confirmedBadges = await badgeSnapshot(statusPage, tabId);
      if (confirmed?.stats?.complete === 6 && confirmed?.stats?.unavailable === 1 &&
        confirmedBadges.filter((badge) => badge.state === "complete").length === 6 &&
        confirmedBadges.filter((badge) => badge.state === "unavailable").length === 1) {
        summary = confirmed;
        break;
      }
    }
    await page.waitForTimeout(250);
  }
  if (summary?.stats?.complete !== 6 || summary?.stats?.unavailable !== 1) {
    throw new Error(`Background summary did not settle: ${JSON.stringify(summary)}; badges=${JSON.stringify(await badgeSnapshot(statusPage, tabId))}`);
  }
  await page.screenshot({ path: path.join(artifactsPath, `chrome-e2e-${expectedProvider}.png`), fullPage: false });

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
  if (!repairedOverlay.overlayAttached || repairedOverlay.badges.length !== 7) {
    throw new Error(`Overlay removal was not recovered: ${JSON.stringify(repairedOverlay)}`);
  }
  await page.screenshot({
    path: path.join(artifactsPath, `chrome-e2e-${expectedProvider}-recovered.png`),
    fullPage: false,
  });

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
    if (current.mutationBudgetOverflows > overflowRaceBaseline.mutationBudgetOverflows &&
      !current.mutationOverflowRecoveryPending && current.pendingDeferredReconciliations === 0) {
      overflowRaceReanalysis = current;
      break;
    }
    await page.waitForTimeout(5);
  }
  if (!overflowRaceReanalysis) throw new Error("First overflow did not complete bounded reconciliation");
  await page.evaluate(() => {
    const target = document.querySelector("#css-background");
    if (!target) throw new Error("Overflow race CSS target missing");
    target.style.backgroundImage = "url(\"file:///prooflens-second-overflow.png\")";
  });
  let secondOverflowInvalidated;
  let secondOverflowAcceptedAtInvalidation;
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const current = await contentSnapshot(statusPage, tabId);
    const target = cssBadge(current.badges);
    if (current.mutationBudgetOverflows >= overflowRaceBaseline.mutationBudgetOverflows + 2 &&
      secondOverflowAcceptedAtInvalidation === undefined) {
      secondOverflowAcceptedAtInvalidation = target?.acceptedResultCount;
    }
    if (typeof secondOverflowAcceptedAtInvalidation === "number" &&
      (target?.acceptedResultCount ?? 0) > secondOverflowAcceptedAtInvalidation) {
      throw new Error(`Second overflow accepted a stale result: ${JSON.stringify(current)}`);
    }
    if (current.mutationBudgetOverflows >= overflowRaceBaseline.mutationBudgetOverflows + 2 &&
      !current.mutationOverflowRecoveryPending && target?.state === "unavailable") {
      secondOverflowInvalidated = current;
      break;
    }
    await page.waitForTimeout(25);
  }
  if (!secondOverflowInvalidated || typeof secondOverflowAcceptedAtInvalidation !== "number") {
    throw new Error("Second same-window overflow did not invalidate the CSS result epoch");
  }
  await page.evaluate(() => document.querySelector("#css-background")?.style.removeProperty("background-image"));
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const target = cssBadge(await badgeSnapshot(statusPage, tabId));
    if (target?.state === "complete" && target.acceptedResultCount === secondOverflowAcceptedAtInvalidation + 1) break;
    await page.waitForTimeout(25);
  }
  const restoredOverflowRace = cssBadge(await badgeSnapshot(statusPage, tabId));
  if (restoredOverflowRace?.state !== "complete" || restoredOverflowRace.acceptedResultCount !== secondOverflowAcceptedAtInvalidation + 1) {
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
    for (let index = 0; index < 4_999; index += 1) {
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
    if (preLate.recordCount === 512) break;
    await page.waitForTimeout(50);
  }
  if (preLate?.recordCount !== 512) throw new Error(`Deep fixture did not fill the bounded stress set: ${JSON.stringify(preLate)}`);
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
    bounded.cssReconciliationIntervalMs !== 1_000 || bounded.maxCssReconciliationRecords !== 512 ||
    bounded.maxAdmissionPriorityEvaluationsInPass > bounded.maxElementsPerScan + 2 * bounded.targetLimit) {
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
      churned.pendingDeferredReconciliations === 0 && !churned.fullDocumentScanRequired &&
      !churned.fullDocumentRestartRequested && late?.state === "complete") break;
    await page.waitForTimeout(50);
  }
  if (!churned) throw new Error("Hostile mutation recovery snapshot was unavailable");
  const maxRecoveryPasses = Math.ceil(churned.maxDeferredReconciliations / churned.maxDeferredReconciliationsPerPass) + 3;
  if (!lateTargetEnteredRefreshing || churned.scanPasses - scanPassesBeforeChurn > maxRecoveryPasses ||
    churned.lastScanVisited > churned.maxElementsPerScan ||
    churned.pendingMutationRoots > churned.maxPendingMutationRoots ||
    churned.pendingDeferredReconciliations !== 0 || churned.mutationOverflowRecoveryPending ||
    churned.fullDocumentScanRequired || churned.fullDocumentRestartRequested ||
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
  await worker.evaluate(({ id, origin }) => chrome.tabs.sendMessage(id, {
    type: "PL_SITE_STATE_CHANGED",
    enabled: false,
    expectedOrigin: origin,
  }), { id: tabId, origin: pageOrigin });
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
  if (replaced.recordCount !== 27 || replaced.pendingCount > 32 || replaced.activeAnalyses > 1) {
    throw new Error(`Target-cap replacement did not recover: ${JSON.stringify(replaced)}`);
  }
  await page.evaluate(() => document.querySelector("#prooflens-replacement-fixture")?.remove());
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if ((await contentSnapshot(statusPage, tabId)).recordCount === 7) break;
    await page.waitForTimeout(25);
  }
  stage("visible viable target replaces broken visible cap");
  await page.evaluate(() => {
    const holder = document.createElement("div");
    holder.id = "prooflens-broken-cap";
    holder.style.cssText = "position:fixed;inset:0;z-index:-1";
    for (let index = 0; index < 505; index += 1) {
      const image = document.createElement("img");
      image.width = 64;
      image.height = 64;
      image.src = `file:///prooflens-broken-cap-${index}.png`;
      image.style.cssText = "position:fixed;left:0;top:0;width:64px;height:64px";
      holder.append(image);
    }
    document.body.append(holder);
  });
  await waitForEvaluation(
    page,
    () => [...document.querySelectorAll("#prooflens-broken-cap img")]
      .every((image) => image.complete),
    undefined,
    { timeout: 30_000, label: "broken-cap fixture loading" },
  );
  let brokenCap;
  for (let attempt = 0; attempt < 160; attempt += 1) {
    brokenCap = await contentSnapshot(statusPage, tabId);
    if (brokenCap.recordCount === 512) break;
    await page.waitForTimeout(25);
  }
  if (brokenCap?.recordCount !== 512) {
    throw new Error(`Broken visible fixture did not fill the target cap: ${JSON.stringify(brokenCap)}`);
  }
  await page.evaluate((source) => {
    const image = document.createElement("img");
    image.id = "valid-after-broken-cap";
    image.alt = "valid target after broken visible cap";
    image.width = 640;
    image.height = 480;
    image.src = source;
    image.style.cssText = "position:fixed;left:96px;top:96px;width:320px;height:240px;z-index:1";
    document.body.append(image);
  }, fixtureB);
  await worker.evaluate(({ id, origin }) => chrome.tabs.sendMessage(id, {
    type: "PL_SITE_STATE_CHANGED",
    enabled: true,
    expectedOrigin: origin,
  }), { id: tabId, origin: pageOrigin });
  let viableCapReplacement;
  for (let attempt = 0; attempt < 160; attempt += 1) {
    viableCapReplacement = await contentSnapshot(statusPage, tabId);
    if (viableCapReplacement.recordCount === 512 &&
      viableCapReplacement.badges.some((badge) =>
        badge.elementId === "valid-after-broken-cap" && badge.state === "complete")) break;
    await page.waitForTimeout(25);
  }
  if (!viableCapReplacement?.badges.some((badge) =>
    badge.elementId === "valid-after-broken-cap" && badge.state === "complete")) {
    throw new Error(`Valid visible target was locked out by broken visible targets: ${JSON.stringify(viableCapReplacement)}`);
  }
  await page.evaluate(() => {
    document.querySelector("#prooflens-broken-cap")?.remove();
    document.querySelector("#valid-after-broken-cap")?.remove();
  });
  for (let attempt = 0; attempt < 120; attempt += 1) {
    if ((await contentSnapshot(statusPage, tabId)).recordCount === 7) break;
    await page.waitForTimeout(25);
  }
  await worker.evaluate(({ id, origin }) => chrome.tabs.sendMessage(id, {
    type: "PL_SITE_STATE_CHANGED",
    enabled: true,
    expectedOrigin: origin,
  }), { id: tabId, origin: pageOrigin });
  await page.setViewportSize({ width: 480, height: 720 });
  const deviceScaleFactor = 1.5;
  const cdp = await context.newCDPSession(page);
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 480,
    height: 720,
    deviceScaleFactor,
    mobile: false,
  });
  await page.waitForTimeout(500);
  const narrow = await contentSnapshot(statusPage, tabId);
  if (narrow.badges.some((badge) => badge.hidden && badge.display !== "none")) {
    throw new Error(`Collision-hidden narrow labels remained visually rendered: ${JSON.stringify(narrow.badges)}`);
  }
  const visibleNarrowBadges = narrow.badges.filter((badge) => !badge.hidden && badge.display !== "none" &&
    badge.badgeRect.right > 0 && badge.badgeRect.bottom > 0 && badge.badgeRect.left < 480 && badge.badgeRect.top < 720);
  if (visibleNarrowBadges.length < 1) throw new Error(`Narrow viewport exposed no eligible label: ${JSON.stringify(narrow)}`);
  const visibleNarrowTargets = narrow.badges
    .map((badge) => badge.targetRect)
    .filter((rect) => rect.right > 0 && rect.bottom > 0 && rect.left < 480 && rect.top < 720);
  for (const badge of visibleNarrowBadges) {
    const currentBadge = badge.badgeRect;
    const currentTarget = badge.targetRect;
    const overlaps = currentBadge.left < currentTarget.right && currentBadge.right > currentTarget.left &&
      currentBadge.top < currentTarget.bottom && currentBadge.bottom > currentTarget.top;
    if (badge.pointerEvents !== "none" || currentBadge.left < 0 || currentBadge.top < 0 ||
      currentBadge.right > 480 || currentBadge.bottom > 720 ||
      (String(badge.placement).startsWith("outside-") && overlaps) ||
      typeof badge.accessibleName !== "string" || badge.accessibleName.length <= String(badge.text ?? "").length) {
      throw new Error(`Narrow/zoomed label contract failed: ${JSON.stringify(badge)}`);
    }
    const associationError = badgeAssociationError(badge, visibleNarrowTargets);
    if (associationError) {
      throw new Error(`Narrow/zoomed label lost its target association (${associationError}): ${JSON.stringify(badge)}`);
    }
  }
  for (let leftIndex = 0; leftIndex < visibleNarrowBadges.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < visibleNarrowBadges.length; rightIndex += 1) {
      const leftBadge = visibleNarrowBadges[leftIndex];
      const rightBadge = visibleNarrowBadges[rightIndex];
      const intersects = leftBadge.badgeRect.left < rightBadge.badgeRect.right &&
        leftBadge.badgeRect.right > rightBadge.badgeRect.left &&
        leftBadge.badgeRect.top < rightBadge.badgeRect.bottom &&
        leftBadge.badgeRect.bottom > rightBadge.badgeRect.top;
      if (intersects) {
        throw new Error(`Narrow/zoomed labels overlap: ${JSON.stringify({ leftBadge, rightBadge })}`);
      }
    }
  }
  await page.evaluate(() => { document.querySelector("#static-after-5000")?.style.setProperty("opacity", "0"); });
  await page.waitForTimeout(50);
  const narrowScreenshotPath = path.join(artifactsPath, `chrome-e2e-${expectedProvider}-narrow.png`);
  await page.screenshot({ path: narrowScreenshotPath, fullPage: false });
  narrowGeometryEvidence = {
    viewport: { width: 480, height: 720 },
    deviceScaleFactor,
    screenshot: await screenshotRecord(narrowScreenshotPath),
    badges: narrow.badges.map(browserGeometryBadgeRecord),
  };
  await page.evaluate(() => { document.querySelector("#static-after-5000")?.style.removeProperty("opacity"); });

  // A crowded page may honestly hide a small target's chip. Isolate one 64px
  // target to prove the available outside placement remains noninteractive,
  // viewport-safe, and immediately associated with the image it describes.
  await page.evaluate((source) => {
    document.querySelectorAll("main img, #static-after-5000, #css-background, #unavailable-background")
      .forEach((element) => { element.style.display = "none"; });
    const image = document.createElement("img");
    image.id = "prooflens-small-target";
    image.alt = "small target fixture";
    image.src = source;
    image.style.cssText = "position:fixed;left:120px;top:300px;width:64px;height:64px;z-index:3";
    document.body.append(image);
  }, fixtureB);
  let smallTargetBadge;
  for (let attempt = 0; attempt < 240; attempt += 1) {
    smallTargetBadge = (await badgeSnapshot(statusPage, tabId)).find(
      (badge) => badge.elementId === "prooflens-small-target",
    );
    if (smallTargetBadge?.state === "complete" && !smallTargetBadge.hidden && smallTargetBadge.display !== "none") break;
    await page.waitForTimeout(250);
  }
  if (smallTargetBadge?.state !== "complete" || smallTargetBadge.hidden || smallTargetBadge.display === "none" ||
    smallTargetBadge.pointerEvents !== "none") {
    throw new Error(`Isolated small target did not produce a visible noninteractive result: ${JSON.stringify(smallTargetBadge)}`);
  }
  const badgeRect = smallTargetBadge.badgeRect;
  const targetRect = smallTargetBadge.targetRect;
  const overlapsTarget = rectsIntersect(badgeRect, targetRect);
  const smallTargetAssociationError = badgeAssociationError(smallTargetBadge, [targetRect]);
  if (badgeRect.left < 0 || badgeRect.top < 0 || badgeRect.right > 480 || badgeRect.bottom > 720 ||
    overlapsTarget || smallTargetAssociationError) {
    throw new Error(`Small-target label geometry is unsafe (${smallTargetAssociationError ?? "geometry"}): ${JSON.stringify(smallTargetBadge)}`);
  }
  const smallTargetScreenshotPath = path.join(artifactsPath, `chrome-e2e-${expectedProvider}-small-target.png`);
  await page.screenshot({ path: smallTargetScreenshotPath, fullPage: false });
  smallTargetGeometryEvidence = {
    viewport: { width: 480, height: 720 },
    screenshot: await screenshotRecord(smallTargetScreenshotPath),
    badges: [browserGeometryBadgeRecord(smallTargetBadge)],
  };
  const archiveSha256 = createHash("sha256").update(await readFile("release/prooflens.zip")).digest("hex");
  const evidence = {
    schemaVersion: 7,
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
    setupProgressAdvanced,
    setupPreparingState,
    setupInitialFailureRecovered,
    setupExplicitModeSelected,
    visualEvidence: {
      setup: setupVisualEvidence,
      popupUnsupported: popupUnsupportedVisualEvidence,
      popupError: popupErrorVisualEvidence,
      popupPick: popupPickVisualEvidence,
      popupSupported: popupSupportedVisualEvidence,
      popupModeSheet: popupModeSheetVisualEvidence,
    },
    popupCaveatVisible,
    popupUnsupportedGuard,
    popupSupportedPageControls,
    popupRescanFeedback,
    popupRescanWork,
    modeSaveFailureTruthful,
    sameOriginNavigationRejected,
    pickModeNoAutomaticAnalysis,
    pickerTargetOutlined,
    pickerEscapeCancelled,
    pickerExactOneTarget,
    pickerHostileImageCountBounded,
    mainModeScoped,
    allModePersisted,
    targetSpecificAccessibleNames: true,
    reducedMotionSuppressed: raceStarted.animationName === "none",
    targets: { total: badges.length, complete: completed.length, unavailable: unavailable.length },
    numericScore: true,
    modelStateFixtures,
    geometryEvidence: {
      schemaVersion: 1,
      positionMargin: BROWSER_GEOMETRY_POSITION_MARGIN,
      modelState: modelStateGeometryEvidence,
      narrow: narrowGeometryEvidence,
      smallTarget: smallTargetGeometryEvidence,
    },
    dynamicImage: true,
    responsivePicture: true,
    cssCompositeBackground: true,
    cssStaleResponseRejected: unavailableRace.acceptedResultCount === 0 && reanalyzedRace.acceptedResultCount === 1,
    cssomBackgroundReconciled:
      cssBadge(cssomInvalidated)?.acceptedResultCount === cssomAcceptedBefore &&
      cssBadge(reanalyzed)?.acceptedResultCount === cssomAcceptedBefore + 1,
    repeatedOverflowStaleResponseRejected:
      secondOverflowInvalidated.mutationBudgetOverflows >= overflowRaceBaseline.mutationBudgetOverflows + 2 &&
      cssBadge(secondOverflowInvalidated.badges)?.acceptedResultCount === secondOverflowAcceptedAtInvalidation,
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
      maxAdmissionPriorityEvaluationsInPass: bounded.maxAdmissionPriorityEvaluationsInPass,
      linearAdmissionPriorityBounded:
        bounded.maxAdmissionPriorityEvaluationsInPass <= bounded.maxElementsPerScan + 2 * bounded.targetLimit,
      staticTargetAfter5000Discovered: staticAfter5000.state === "complete",
      staticFullDocumentTraversalCompleted:
        !initialTraversal.fullDocumentScanRequired && !initialTraversal.fullDocumentRestartRequested,
      visibleTargetEvictedOffscreenAtCap: bounded.badges.some((badge) => badge.elementId === "late-after-5000" && badge.state === "complete"),
      viableTargetEvictedBrokenVisibleAtCap:
        viableCapReplacement.badges.some((badge) =>
          badge.elementId === "valid-after-broken-cap" && badge.state === "complete"),
    },
    cssomInFlightStaleResponseRejected:
      cssomRaceInvalidated.acceptedResultCount === 0 && cssomRaceRecovered.acceptedResultCount === 1,
  };
  validateBrowserGeometryEvidence(evidence.geometryEvidence, {
    modelState: modelStateGeometryEvidence.screenshot,
    narrow: narrowGeometryEvidence.screenshot,
    smallTarget: smallTargetGeometryEvidence.screenshot,
  });
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
