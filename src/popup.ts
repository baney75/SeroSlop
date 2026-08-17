import type { ModelStatus, TabSummaryResponse } from "./shared/contracts";
import { isScanMode, SCAN_MODE_COPY, type ScanMode } from "./shared/scan-mode";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Popup element ${selector} is missing`);
  return element;
}

const statusElement = requireElement<HTMLElement>("#model-status");
const pageElement = requireElement<HTMLElement>("#page-summary");
const feedbackElement = requireElement<HTMLElement>("#control-feedback");
const siteContextElement = requireElement<HTMLElement>("#site-context");
const summaryView = requireElement<HTMLElement>("#summary-view");
const modeSummary = requireElement<HTMLElement>("#mode-summary");
const modeSheet = requireElement<HTMLElement>("#mode-sheet");
const changeModeButton = requireElement<HTMLButtonElement>("#change-mode");
const modeBackButton = requireElement<HTMLButtonElement>("#mode-back");
const saveModeButton = requireElement<HTMLButtonElement>("#save-mode");
const modeError = requireElement<HTMLElement>("#mode-error");
const modeInputs = [...document.querySelectorAll<HTMLInputElement>('input[name="popup-mode"]')];
const primaryAction = requireElement<HTMLButtonElement>("#primary-action");
const setupLink = requireElement<HTMLAnchorElement>("#open-setup");

let activeTab: chrome.tabs.Tab | undefined;
let origin = "";
let modelReady = false;
let scanMode: ScanMode | undefined;
let contentAvailable = false;
let pickerActive = false;
let pickResultExists = false;
let targetInvalidated = false;
let refreshPending = false;
let actionPending = false;
let savePending = false;
let summaryTimer: number | undefined;

interface ContentSnapshot {
  pickerActive?: boolean;
  scanMode?: ScanMode;
  documentToken?: string;
  recordCount?: number;
}

let documentToken = "";

function supportedOrigin(url: string | undefined): string {
  if (!url || !/^https?:/u.test(url)) return "";
  try { return new URL(url).origin; } catch { return ""; }
}

function setFeedback(message: string, state: "info" | "error" = "info"): void {
  feedbackElement.textContent = message;
  feedbackElement.dataset.state = state;
}

function describeModel(status: ModelStatus): string {
  if (status.state === "ready") return "Offline ready";
  if (status.state === "preparing") return "Verifying model…";
  if (status.state === "error") return "Setup required";
  return "Setup required";
}

function describePage(stats: TabSummaryResponse["stats"]): string {
  const progress = [
    stats.analyzing ? `${stats.analyzing} analyzing` : "",
    stats.queued ? `${stats.queued} queued` : "",
  ].filter(Boolean).join(" · ");
  const settled = `${stats.complete} complete · ${stats.flagged} flagged · ${stats.unavailable} unavailable`;
  return progress ? `${progress} · ${settled}` : settled;
}

function exposeChangedPage(): void {
  targetInvalidated = true;
  contentAvailable = false;
  pickerActive = false;
  pageElement.textContent = "Page changed · reopen SeroSlop";
  siteContextElement.textContent = "The selected page changed";
  setFeedback("The page changed. Reopen SeroSlop to use its controls.", "error");
  renderPrimaryAction();
}

async function requireCurrentTarget(): Promise<{ tabId: number; origin: string } | undefined> {
  if (targetInvalidated || activeTab?.id === undefined || !origin) return undefined;
  const [current] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (current?.id !== activeTab.id || supportedOrigin(current.url) !== origin) {
    exposeChangedPage();
    return undefined;
  }
  if (documentToken) {
    const confirmed = await chrome.tabs.sendMessage(activeTab.id, {
      type: "PL_CONFIRM_DOCUMENT",
      expectedOrigin: origin,
      expectedDocumentToken: documentToken,
    }).catch(() => undefined) as { documentToken?: string } | undefined;
    if (confirmed?.documentToken !== documentToken) {
      exposeChangedPage();
      return undefined;
    }
  }
  return { tabId: activeTab.id, origin };
}

function renderPrimaryAction(): void {
  if (actionPending) return;
  if (targetInvalidated) {
    primaryAction.textContent = "Reopen SeroSlop to scan";
    primaryAction.disabled = true;
    return;
  }
  if (!modelReady || !scanMode) {
    primaryAction.textContent = "Finish setup";
    primaryAction.disabled = false;
    return;
  }
  if (!contentAvailable) {
    primaryAction.textContent = "This page can’t be scanned";
    primaryAction.disabled = true;
    return;
  }
  primaryAction.textContent = scanMode === "pick" && pickerActive
    ? "Cancel picker"
    : scanMode === "pick" && pickResultExists
      ? "Choose another image"
      : SCAN_MODE_COPY[scanMode].action;
  primaryAction.disabled = false;
}

function renderMode(): void {
  modeSummary.textContent = scanMode ? SCAN_MODE_COPY[scanMode].title : "Choose a mode";
  changeModeButton.disabled = savePending || !modelReady;
  for (const input of modeInputs) input.checked = input.value === scanMode;
  renderPrimaryAction();
}

async function refreshPageState(): Promise<void> {
  if (refreshPending || activeTab?.id === undefined || !origin) return;
  refreshPending = true;
  try {
    const target = await requireCurrentTarget();
    if (!target) return;
    const summary = await chrome.runtime.sendMessage({ type: "PL_GET_TAB_SUMMARY", tabId: target.tabId }) as TabSummaryResponse;
    pageElement.textContent = describePage(summary.stats);
    const snapshot = await chrome.tabs.sendMessage(target.tabId, { type: "PL_GET_CONTENT_SNAPSHOT" }).catch(() => undefined) as ContentSnapshot | undefined;
    if (documentToken && snapshot?.documentToken !== documentToken) {
      exposeChangedPage();
      return;
    }
    if (snapshot?.documentToken) documentToken = snapshot.documentToken;
    contentAvailable = Boolean(snapshot);
    pickerActive = snapshot?.pickerActive === true;
    if (isScanMode(snapshot?.scanMode) && snapshot.scanMode !== scanMode) scanMode = snapshot.scanMode;
    pickResultExists = scanMode === "pick" && Number(snapshot?.recordCount ?? 0) > 0;
  } catch {
    contentAvailable = false;
    pickerActive = false;
    pageElement.textContent = "Page state unavailable";
  } finally {
    refreshPending = false;
    renderMode();
  }
}

function openSetup(): void {
  void chrome.tabs.create({ url: chrome.runtime.getURL("setup.html") });
}

async function initialize(): Promise<void> {
  [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
  origin = supportedOrigin(activeTab?.url);
  if (origin && activeTab?.url) {
    const host = new URL(activeTab.url).hostname;
    siteContextElement.textContent = host;
    siteContextElement.title = host;
    siteContextElement.setAttribute("aria-label", `Current site: ${host}`);
  } else {
    siteContextElement.textContent = "No supported site selected";
    pageElement.textContent = "No supported page is active";
  }

  const model = await chrome.runtime.sendMessage({ type: "PL_GET_MODEL_STATUS" }).catch(() => undefined) as ModelStatus | undefined;
  modelReady = model?.state === "ready";
  statusElement.textContent = model ? describeModel(model) : "Setup status unavailable";
  statusElement.dataset.state = modelReady ? "ready" : "error";

  const modeResponse = await chrome.runtime.sendMessage({ type: "PL_GET_SCAN_MODE" }).catch(() => undefined) as { scanMode?: ScanMode } | undefined;
  scanMode = isScanMode(modeResponse?.scanMode) ? modeResponse.scanMode : undefined;
  renderMode();

  if (activeTab?.id !== undefined && origin) {
    await refreshPageState();
    summaryTimer = window.setInterval(() => void refreshPageState(), 500);
  } else {
    contentAvailable = false;
    renderPrimaryAction();
  }
}

function showModeSheet(): void {
  summaryView.hidden = true;
  modeSheet.hidden = false;
  modeError.textContent = "";
  delete modeError.dataset.state;
  for (const input of modeInputs) input.checked = input.value === scanMode;
  saveModeButton.disabled = true;
  (modeInputs.find((input) => input.checked) ?? modeInputs[0])?.focus();
}

function hideModeSheet(): void {
  modeSheet.hidden = true;
  summaryView.hidden = false;
  changeModeButton.focus();
}

changeModeButton.addEventListener("click", showModeSheet);
modeBackButton.addEventListener("click", hideModeSheet);
for (const input of modeInputs) {
  input.addEventListener("change", () => {
    modeError.textContent = "";
    delete modeError.dataset.state;
    saveModeButton.disabled = !isScanMode(input.value) || input.value === scanMode;
  });
}

saveModeButton.addEventListener("click", () => {
  void (async () => {
    if (savePending) return;
    const selected = modeInputs.find((input) => input.checked)?.value;
    if (!isScanMode(selected)) return;
    const previous = scanMode;
    savePending = true;
    saveModeButton.disabled = true;
    saveModeButton.textContent = "Saving mode…";
    saveModeButton.setAttribute("aria-busy", "true");
    modeInputs.forEach((input) => { input.disabled = true; });
    modeError.textContent = "";
    delete modeError.dataset.state;
    try {
      const before = origin ? await requireCurrentTarget() : undefined;
      if (origin && !before) throw new Error("page-changed");
      const saved = await chrome.runtime.sendMessage({ type: "PL_SET_SCAN_MODE", scanMode: selected }) as { scanMode?: ScanMode };
      if (saved?.scanMode !== selected) throw new Error("save-not-confirmed");
      scanMode = selected;
      if (before) {
        const relayed = await chrome.tabs.sendMessage(before.tabId, {
          type: "PL_SCAN_MODE_CHANGED",
          scanMode: selected,
          expectedOrigin: before.origin,
          expectedDocumentToken: documentToken,
        }).catch(() => undefined) as { scanMode?: ScanMode; pageChanged?: boolean } | undefined;
        if (relayed?.scanMode !== selected) {
          setFeedback("Mode saved. It will apply after reload.");
        } else {
          setFeedback(`Scanning mode: ${SCAN_MODE_COPY[selected].title}.`);
        }
      }
      hideModeSheet();
      await refreshPageState();
    } catch {
      scanMode = previous;
      for (const input of modeInputs) input.checked = input.value === previous;
      modeError.textContent = "Couldn’t save your mode. Try again.";
      modeError.dataset.state = "error";
      (modeInputs.find((input) => input.value === selected) ?? modeInputs[0])?.focus();
    } finally {
      savePending = false;
      saveModeButton.textContent = "Save mode";
      saveModeButton.removeAttribute("aria-busy");
      modeInputs.forEach((input) => { input.disabled = false; });
      renderMode();
    }
  })();
});

primaryAction.addEventListener("click", () => {
  void (async () => {
    if (actionPending) return;
    if (!modelReady || !scanMode) { openSetup(); return; }
    const target = await requireCurrentTarget();
    if (!target) return;
    actionPending = true;
    primaryAction.disabled = true;
    primaryAction.setAttribute("aria-busy", "true");
    try {
      if (scanMode === "pick") {
        const type = pickerActive ? "PL_CANCEL_PICKER" : "PL_START_PICKER";
        const response = await chrome.tabs.sendMessage(target.tabId, { type, expectedOrigin: target.origin, expectedDocumentToken: documentToken }) as { started?: boolean; cancelled?: boolean; noCandidates?: boolean };
        if (response.noCandidates) {
          setFeedback("No supported images are available on this page.", "error");
          return;
        }
        if (type === "PL_START_PICKER" && response.started !== true) throw new Error("picker-not-started");
        if (type === "PL_CANCEL_PICKER" && response.cancelled !== true) throw new Error("picker-not-cancelled");
        pickerActive = type === "PL_START_PICKER";
        pageElement.textContent = pickerActive ? "Choose one image on this page. Esc cancels." : "Image picker cancelled";
        setFeedback(pickerActive ? "Choose an image on the page." : "Picker cancelled.");
      } else {
        primaryAction.textContent = "Scanning again…";
        const [response] = await Promise.all([
          chrome.tabs.sendMessage(target.tabId, { type: "PL_RESCAN", expectedOrigin: target.origin, expectedDocumentToken: documentToken }) as Promise<{ rescanned?: boolean }>,
          new Promise((resolve) => window.setTimeout(resolve, 500)),
        ]);
        if (response.rescanned !== true) throw new Error("rescan-not-confirmed");
        pageElement.textContent = "Fresh scan queued";
        setFeedback("Scan started.");
      }
    } catch {
      contentAvailable = false;
      pageElement.textContent = "This page can’t be scanned";
      setFeedback("The page changed. Reopen SeroSlop and try again.", "error");
    } finally {
      actionPending = false;
      primaryAction.removeAttribute("aria-busy");
      renderPrimaryAction();
      window.setTimeout(() => void refreshPageState(), 250);
    }
  })();
});

setupLink.addEventListener("click", (event) => { event.preventDefault(); openSetup(); });
window.addEventListener("unload", () => { if (summaryTimer !== undefined) window.clearInterval(summaryTimer); });

void initialize();
