/* global document, PointerEvent, window */
import { createServer } from "node:http";
import { cp, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright-core";

const sourceExtensionPath = path.resolve("contributor/dist");
const profilePath = await mkdtemp(path.join(os.tmpdir(), "seroslop-contributor-"));
const extensionPath = path.join(profilePath, "extension");
const fixture = `<!doctype html><html><body style="margin:0;background:#fff"><img id="decoy" alt="visible decoy" width="320" height="240" style="display:block;width:160px;height:120px" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='320' height='240'%3E%3Crect width='320' height='240' fill='%235a284d'/%3E%3C/svg%3E"><div style="height:1200px"></div><img id="target" alt="known AI test image" width="600" height="400" style="display:block;width:300px;height:200px" src="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='400'%3E%3Crect width='600' height='400' fill='%23174c3c'/%3E%3C/svg%3E"></body></html>`;
const server = createServer((_request, response) => {
  response.writeHead(200, { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" });
  response.end(fixture);
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const address = server.address();
if (!address || typeof address === "string") throw new Error("Contributor fixture server did not start");
const fixtureUrl = `http://127.0.0.1:${address.port}/`;
await cp(sourceExtensionPath, extensionPath, { recursive: true });
const testManifestPath = path.join(extensionPath, "manifest.json");
const testManifest = JSON.parse(await readFile(testManifestPath, "utf8"));
if (testManifest.host_permissions || testManifest.permissions.includes("<all_urls>")) throw new Error("Production contributor manifest broadened host access");
// Opening a popup document as a Playwright tab cannot create Chrome's real
// user-gesture activeTab grant. This disposable test copy uses capture-only
// host access; source-manifest assertions above keep the shipped extension on
// activeTab with no persistent host permission.
testManifest.host_permissions = ["<all_urls>"];
await writeFile(testManifestPath, `${JSON.stringify(testManifest, null, 2)}\n`);
let context;
try {
  context = await chromium.launchPersistentContext(profilePath, {
    headless: false,
    acceptDownloads: true,
    args: [`--disable-extensions-except=${extensionPath}`, `--load-extension=${extensionPath}`, "--no-first-run", "--no-default-browser-check"],
  });
  const worker = context.serviceWorkers()[0] ?? await context.waitForEvent("serviceworker", { timeout: 30_000 });
  const extensionId = new URL(worker.url()).host;
  const page = await context.newPage();
  await page.goto(fixtureUrl, { waitUntil: "load" });
  await page.evaluate(() => {
    const hostileImages = document.createElement("div");
    hostileImages.hidden = true;
    const fragment = document.createDocumentFragment();
    for (let index = 0; index < 6_000; index += 1) fragment.append(document.createElement("img"));
    hostileImages.append(fragment);
    document.body.append(hostileImages);
  });
  const tabId = await worker.evaluate(async () => (await chrome.tabs.query({ active: true, currentWindow: true }))[0]?.id);
  if (typeof tabId !== "number") throw new Error("Contributor fixture tab was not found");
  const remoteAfterPickerStart = [];
  context.on("request", (request) => { if (/^https?:/u.test(request.url())) remoteAfterPickerStart.push(request.url()); });

  await worker.evaluate((id) => chrome.tabs.update(id, { active: true }), tabId);
  const popup = await context.newPage();
  await popup.goto(`chrome-extension://${extensionId}/popup.html`);
  await worker.evaluate((id) => chrome.tabs.update(id, { active: true }), tabId);
  await popup.evaluate(() => document.querySelector("#select")?.click());
  await page.locator("#target").scrollIntoViewIfNeeded();
  await page.waitForTimeout(100);
  await page.keyboard.press("Tab");
  const tip = page.getByRole("status").filter({ hasText: /Tab moves\. Enter selects\. Esc cancels\./u });
  await tip.waitFor({ timeout: 10_000 });
  const originalDocumentState = await worker.evaluate((id) => chrome.tabs.sendMessage(id, { type: "CONTRIBUTOR_GET_DOCUMENT_STATE" }), tabId);
  if (!/^[a-f0-9]{32}$/u.test(originalDocumentState?.documentToken) || originalDocumentState?.origin !== new URL(fixtureUrl).origin) {
    throw new Error("Contributor document binding was not established");
  }
  await page.evaluate(async () => {
    const host = document.createElement("div");
    Object.assign(host.style, { position: "fixed", inset: "0", pointerEvents: "none" });
    const source = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24'%3E%3Crect width='24' height='24' fill='%23174c3c'/%3E%3C/svg%3E";
    const images = [];
    for (let index = 0; index < 5_100; index += 1) {
      const image = document.createElement("img");
      image.alt = `hostile candidate ${index}`;
      image.src = source;
      Object.assign(image.style, { position: "absolute", left: "0", top: "0", width: "24px", height: "24px" });
      host.append(image);
      images.push(image);
    }
    document.body.append(host);
    await Promise.all(images.map((image) => image.decode()));
    for (const image of images) image.dispatchEvent(new PointerEvent("pointermove", { bubbles: true }));
    host.remove();
    document.querySelector("#target")?.dispatchEvent(new PointerEvent("pointermove", { bubbles: true }));
  });
  const pickerBounds = await worker.evaluate((id) => chrome.tabs.sendMessage(id, { type: "CONTRIBUTOR_GET_DOCUMENT_STATE" }), tabId);
  if (pickerBounds?.pickerCandidateLimit !== 5_000 || pickerBounds.pickerCandidateCount > 5_000 ||
    pickerBounds.pickerMaxCandidateCountObserved > 5_000) {
    throw new Error(`Contributor picker exceeded its hard candidate bound: ${JSON.stringify(pickerBounds)}`);
  }
  await page.keyboard.press("Enter");
  let capturedSelection;
  for (let attempt = 0; attempt < 200; attempt += 1) {
    capturedSelection = (await worker.evaluate(() => chrome.storage.session.get("contributorSelection"))).contributorSelection;
    if (capturedSelection?.thumbnail) break;
    await page.waitForTimeout(50);
  }
  if (!capturedSelection?.thumbnail || capturedSelection.screenshot) {
    const lastResponse = await worker.evaluate(async (id) => {
      const [execution] = await chrome.scripting.executeScript({ target: { tabId: id }, func: () => globalThis.__seroslopContributorLastResponse });
      return execution.result;
    }, tabId);
    throw new Error(`Contributor selection was not minimized at capture: ${JSON.stringify(lastResponse)}`);
  }

  const expiryReview = await context.newPage();
  await worker.evaluate(async (selection) => {
    await chrome.storage.session.set({ contributorSelection: { ...selection, expiresAt: Date.now() + 150 } });
  }, capturedSelection);
  await expiryReview.goto(`chrome-extension://${extensionId}/popup.html`);
  await expiryReview.getByText("Image captured locally.").waitFor({ timeout: 10_000 });
  await expiryReview.getByText("The image capture expired. Choose it again.").waitFor({ timeout: 10_000 });
  const expiredState = await worker.evaluate(() => chrome.storage.session.get("contributorSelection"));
  if (expiredState.contributorSelection || await expiryReview.locator("#thumb").getAttribute("src")) {
    throw new Error("Contributor review material survived its expiry while the popup was open");
  }
  await expiryReview.close();
  await worker.evaluate((selection) => chrome.storage.session.set({ contributorSelection: selection }), capturedSelection);

  const review = await context.newPage();
  await review.goto(`chrome-extension://${extensionId}/popup.html`);
  await review.getByText("Image captured locally.").waitFor({ timeout: 10_000 });
  const previewSource = await review.locator("#thumb").getAttribute("src");
  if (!previewSource?.startsWith("data:image/jpeg;base64,")) throw new Error("Contributor preview was not a local captured image");
  const preview = await review.locator("#thumb").evaluate((image) => {
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let redOverlayPixels = 0;
    let red = 0; let green = 0; let blue = 0;
    for (let offset = 0; offset < pixels.length; offset += 4) {
      red += pixels[offset]; green += pixels[offset + 1]; blue += pixels[offset + 2];
      if (pixels[offset] > 180 && pixels[offset + 1] < 90 && pixels[offset + 2] < 100) redOverlayPixels += 1;
    }
    const count = pixels.length / 4;
    return { width: image.naturalWidth, height: image.naturalHeight, src: image.src, redOverlayPixels, mean: [red / count, green / count, blue / count] };
  });
  if (preview.width < 24 || preview.height < 24 || Math.abs(preview.width / preview.height - 3 / 2) > 0.05 ||
    preview.redOverlayPixels !== 0 || preview.mean[0] > 45 || preview.mean[1] < 55 || preview.mean[1] > 100 || preview.mean[2] < 40 || preview.mean[2] > 90) {
    throw new Error(`Contributor crop did not preserve the selected target: ${JSON.stringify(preview)}`);
  }
  await review.getByLabel("AI image — detector missed it").check();
  await review.getByLabel("SeroSlop score shown for this image").fill("70");
  await review.getByLabel(/What makes you confident/u).fill("Publisher source and generation record confirm this image is AI-generated.");
  await review.getByLabel(/I personally know this label is correct/u).check();
  if (!await review.getByRole("button", { name: "Prepare review file" }).isDisabled()) throw new Error("Threshold-inconsistent false negative was accepted");
  await review.getByLabel("SeroSlop score shown for this image").fill("20");
  await review.getByRole("button", { name: "Prepare review file" }).click();
  await review.getByText("Prepared locally. Nothing was uploaded.").waitFor();
  const beforeExport = await worker.evaluate(() => chrome.storage.session.get(["contributorSelection", "contributorPreparedReview"]));
  if (!beforeExport.contributorSelection || beforeExport.contributorPreparedReview) throw new Error("Prepared review was persisted before export");
  const downloadPromise = review.waitForEvent("download", { timeout: 10_000 });
  await review.getByRole("button", { name: "Download prepared file" }).click();
  const download = await downloadPromise;
  const downloadPath = await download.path();
  if (!downloadPath) throw new Error("Contributor review download has no local path");
  const payload = JSON.parse(await readFile(downloadPath, "utf8"));
  if (payload?.score !== 20 || payload?.label !== "ai_false_negative" || !/^[a-f0-9]{64}$/u.test(payload?.image?.sha256) ||
    !payload?.image?.thumbnail?.startsWith("data:image/jpeg;base64,") || payload?.image?.width !== 600 || payload?.image?.height !== 400 || payload?.source !== fixtureUrl) {
    throw new Error(`Contributor export contract failed: ${JSON.stringify(payload)}`);
  }
  await review.getByText("Review file downloaded.").waitFor({ timeout: 10_000 });
  const stored = await worker.evaluate(() => chrome.storage.session.get(["contributorSelection", "contributorPreparedReview"]));
  if (stored.contributorSelection || stored.contributorPreparedReview) throw new Error("Contributor review material was retained after export");
  if (remoteAfterPickerStart.length !== 0) throw new Error(`Contributor caused network requests: ${remoteAfterPickerStart.join(", ")}`);
  await worker.evaluate((id) => chrome.tabs.update(id, { active: true }), tabId);
  await page.reload({ waitUntil: "load" });
  await worker.evaluate((id) => chrome.scripting.executeScript({ target: { tabId: id }, files: ["content.js"] }), tabId);
  remoteAfterPickerStart.length = 0;
  const staleResult = await worker.evaluate(async ({ id, state, source }) => {
    const [execution] = await chrome.scripting.executeScript({
      target: { tabId: id },
      func: async (oldState, pageUrl) => chrome.runtime.sendMessage({
        type: "CONTRIBUTOR_IMAGE_SELECTED",
        image: {
          alt: "stale target", width: 600, height: 400, pageUrl,
          origin: oldState.origin, documentToken: oldState.documentToken,
          rect: { left: 0, top: 0, width: 300, height: 200 },
          viewport: { width: window.innerWidth, height: window.innerHeight }
        }
      }),
      args: [state, source]
    });
    return execution.result;
  }, { id: tabId, state: originalDocumentState, source: fixtureUrl });
  if (staleResult?.stored !== false) throw new Error("Stale contributor document was accepted");
  const afterStale = await worker.evaluate(() => chrome.storage.session.get("contributorSelection"));
  if (afterStale.contributorSelection || remoteAfterPickerStart.length !== 0) throw new Error("Stale contributor capture retained data or caused network access");
  console.log(JSON.stringify({ status: "contributor-chrome-pass", cleanProfile: true, keyboardPicker: true, hostileImageCountBounded: true, localCapture: true, exactTargetPixels: true, documentBound: true, scoreBound: true, exported: true, cleaned: true, networkRequests: [] }));
} finally {
  await context?.close().catch(() => undefined);
  await new Promise((resolve) => server.close(resolve));
  server.closeAllConnections();
  await rm(profilePath, { recursive: true, force: true });
}
