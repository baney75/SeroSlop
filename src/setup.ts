import type { ModelStatus } from "./shared/contracts";
import { MODEL_SPEC } from "./shared/model-spec";
import { isScanMode, SCAN_MODE_COPY, type ScanMode } from "./shared/scan-mode";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Setup element ${selector} is missing`);
  return element;
}

const stateElement = requireElement<HTMLElement>("#setup-state");
const detailElement = requireElement<HTMLElement>("#setup-detail");
const statusPanel = requireElement<HTMLElement>("#setup-status-panel");
const announcementElement = requireElement<HTMLElement>("#setup-announcement");
const progressElement = requireElement<HTMLProgressElement>("#setup-progress");
const progressTextElement = requireElement<HTMLElement>("#setup-progress-text");
const progressRegion = requireElement<HTMLElement>("#setup-progress-region");
const prepareButton = requireElement<HTMLButtonElement>("#prepare-model");
const modeSection = requireElement<HTMLElement>("#mode-section");
const modeInputs = [...document.querySelectorAll<HTMLInputElement>('input[name="scan-mode"]')];
const modeHeading = requireElement<HTMLElement>("#mode-heading");
const modeIntro = requireElement<HTMLElement>("#mode-intro");
const modeFeedback = requireElement<HTMLElement>("#mode-feedback");
const saveModeButton = requireElement<HTMLButtonElement>("#save-mode");
const modelElement = requireElement<HTMLElement>("#model-name");
const sizeElement = requireElement<HTMLElement>("#model-size");
const hashElement = requireElement<HTMLElement>("#model-hash");
let announcedState = "";
let persistedMode: ScanMode | undefined;
let modeLoad: Promise<void> | undefined;
let modeSavePending = false;

modelElement.textContent = MODEL_SPEC.displayName;
sizeElement.textContent = `${(MODEL_SPEC.weightsBytes / 1_000_000).toFixed(1)} MB`;
hashElement.textContent = MODEL_SPEC.weightsSha256;

function render(status: ModelStatus): void {
  document.body.dataset.state = status.state;
  statusPanel.setAttribute("aria-busy", String(status.state === "preparing"));
  const previousMinimum = Number(progressElement.dataset.minimumObservedBytes ?? status.completedBytes);
  const previousMaximum = Number(progressElement.dataset.maximumObservedBytes ?? status.completedBytes);
  progressElement.dataset.minimumObservedBytes = String(Math.min(previousMinimum, status.completedBytes));
  progressElement.dataset.maximumObservedBytes = String(Math.max(previousMaximum, status.completedBytes));
  progressElement.dataset.renderCount = String(Number(progressElement.dataset.renderCount ?? 0) + 1);
  progressElement.max = status.totalBytes || MODEL_SPEC.weightsBytes;
  progressElement.value = status.completedBytes;
  const completedMb = (status.completedBytes / 1_000_000).toFixed(1);
  const totalMb = ((status.totalBytes || MODEL_SPEC.weightsBytes) / 1_000_000).toFixed(1);
  progressTextElement.textContent = `${completedMb} of ${totalMb} MB`;
  progressElement.setAttribute("aria-valuetext", progressTextElement.textContent);
  if (status.state === "ready") {
    stateElement.textContent = "Offline ready";
    detailElement.textContent = "Model verified.";
    prepareButton.textContent = "Model verified";
    prepareButton.disabled = true;
    prepareButton.hidden = true;
    progressRegion.hidden = false;
    modeLoad ??= showMode();
  } else if (status.state === "preparing") {
    stateElement.textContent = "Verifying model";
    detailElement.textContent = "Keep this tab open until verification finishes.";
    prepareButton.textContent = "Verifying model";
    prepareButton.disabled = true;
    prepareButton.hidden = true;
    progressRegion.hidden = false;
    modeSection.hidden = true;
  } else if (status.state === "error") {
    stateElement.textContent = "Setup failed";
    detailElement.textContent = status.error ?? "The packaged model could not be verified.";
    prepareButton.textContent = "Retry verification";
    prepareButton.disabled = false;
    prepareButton.hidden = false;
    progressRegion.hidden = true;
    modeSection.hidden = true;
  } else {
    stateElement.textContent = "Model not verified";
    detailElement.textContent = "Verify the model before scanning pages.";
    prepareButton.textContent = "Verify model";
    prepareButton.disabled = false;
    prepareButton.hidden = false;
    progressRegion.hidden = true;
    modeSection.hidden = true;
  }
  if (announcedState !== status.state) {
    announcedState = status.state;
    announcementElement.textContent = `${stateElement.textContent}. ${detailElement.textContent}`;
  }
}

async function showMode(): Promise<void> {
  modeSection.hidden = false;
  try {
    const response = await chrome.runtime.sendMessage({ type: "PL_GET_SCAN_MODE" }) as { scanMode?: ScanMode };
    persistedMode = isScanMode(response?.scanMode) ? response.scanMode : undefined;
    modeHeading.textContent = persistedMode ? "How SeroSlop scans pages" : "Choose how SeroSlop scans pages";
    modeIntro.textContent = persistedMode ? "Change the saved mode below." : "You can change this anytime from the extension.";
    for (const input of modeInputs) input.checked = input.value === persistedMode;
    delete modeFeedback.dataset.state;
  } catch {
    persistedMode = undefined;
    modeHeading.textContent = "Choose how SeroSlop scans pages";
    modeIntro.textContent = "The saved mode couldn’t be read. Choose a mode and try again.";
    modeFeedback.textContent = "Couldn’t read the saved mode.";
    modeFeedback.dataset.state = "error";
  }
  updateModeAction();
}

function selectedMode(): ScanMode | undefined {
  const value = modeInputs.find((input) => input.checked)?.value;
  return isScanMode(value) ? value : undefined;
}

function updateModeAction(): void {
  const selected = selectedMode();
  saveModeButton.hidden = Boolean(persistedMode && selected === persistedMode);
  saveModeButton.disabled = modeSavePending || !selected || selected === persistedMode;
  saveModeButton.textContent = selected ? "Save mode" : "Choose a mode";
}

async function applyModeToOpenPages(scanMode: ScanMode): Promise<void> {
  const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*"] });
  await Promise.allSettled(tabs.map(async (tab) => {
    if (tab.id === undefined || !tab.url) return;
    const origin = new URL(tab.url).origin;
    const snapshot = await chrome.tabs.sendMessage(tab.id, { type: "PL_GET_CONTENT_SNAPSHOT" }) as { documentToken?: string };
    if (!snapshot.documentToken) return;
    await chrome.tabs.sendMessage(tab.id, {
      type: "PL_SCAN_MODE_CHANGED",
      scanMode,
      expectedOrigin: origin,
      expectedDocumentToken: snapshot.documentToken,
    });
  }));
}

for (const input of modeInputs) {
  input.addEventListener("change", () => {
    modeFeedback.textContent = "";
    delete modeFeedback.dataset.state;
    updateModeAction();
  });
}

saveModeButton.addEventListener("click", () => {
  void (async () => {
    const chosen = selectedMode();
    if (!chosen || modeSavePending) return;
    const previous = persistedMode;
    modeSavePending = true;
    saveModeButton.hidden = false;
    saveModeButton.disabled = true;
    saveModeButton.textContent = "Saving mode…";
    saveModeButton.setAttribute("aria-busy", "true");
    modeInputs.forEach((input) => { input.disabled = true; });
    modeFeedback.textContent = "";
    try {
      const result = await chrome.runtime.sendMessage({ type: "PL_SET_SCAN_MODE", scanMode: chosen }) as { scanMode?: ScanMode };
      if (result?.scanMode !== chosen) throw new Error("Mode save was not confirmed");
      persistedMode = chosen;
      await applyModeToOpenPages(chosen);
      stateElement.textContent = "Offline ready";
      detailElement.textContent = "Mode saved. You can close this tab.";
      modeHeading.textContent = "How SeroSlop scans pages";
      modeIntro.textContent = "Change the saved mode below.";
      modeFeedback.textContent = `Saved: ${SCAN_MODE_COPY[chosen].title}.`;
      delete modeFeedback.dataset.state;
      announcementElement.textContent = `Model verified. Scanning mode: ${SCAN_MODE_COPY[chosen].title}.`;
    } catch {
      persistedMode = previous;
      for (const input of modeInputs) input.checked = input.value === previous;
      modeFeedback.textContent = "Couldn’t save your mode. Try again.";
      modeFeedback.dataset.state = "error";
      (modeInputs.find((input) => input.value === chosen) ?? modeInputs[0])?.focus();
    } finally {
      modeSavePending = false;
      modeInputs.forEach((input) => { input.disabled = false; });
      saveModeButton.removeAttribute("aria-busy");
      updateModeAction();
    }
  })();
});

function renderFailure(error: unknown): void {
  render({
    state: "error",
    modelId: MODEL_SPEC.id,
    completedBytes: 0,
    totalBytes: MODEL_SPEC.weightsBytes,
    error: error instanceof Error ? error.message : String(error),
  });
}

async function prepare(): Promise<void> {
  const preparingStarted = performance.now();
  render({
    state: "preparing",
    modelId: MODEL_SPEC.id,
    completedBytes: 0,
    totalBytes: MODEL_SPEC.weightsBytes,
  });
  // Keep the initial determinate state perceptible before model streaming begins.
  await new Promise((resolve) => window.setTimeout(resolve, 180));
  try {
    const status = (await chrome.runtime.sendMessage({ type: "PL_PREPARE_MODEL" })) as ModelStatus;
    const remainingVisibleMs = Math.max(0, 1_200 - (performance.now() - preparingStarted));
    if (remainingVisibleMs) await new Promise((resolve) => window.setTimeout(resolve, remainingVisibleMs));
    render(status);
  } catch (error) {
    renderFailure(error);
  }
}

prepareButton.addEventListener("click", () => void prepare());

chrome.runtime.onMessage.addListener(
  (message: { type: string; completedBytes?: number; totalBytes?: number }) => {
    if (message.type !== "PL_SETUP_PROGRESS") return;
    render({
      state: "preparing",
      modelId: MODEL_SPEC.id,
      completedBytes: message.completedBytes ?? 0,
      totalBytes: message.totalBytes ?? MODEL_SPEC.weightsBytes,
    });
  },
);

async function initialize(): Promise<void> {
  try {
    const status = (await chrome.runtime.sendMessage({ type: "PL_GET_MODEL_STATUS" })) as ModelStatus;
    render(status);
    if (status.state === "not-installed") await prepare();
  } catch (error) {
    renderFailure(error);
  }
}

void initialize();
