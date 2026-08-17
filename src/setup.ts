import type { ModelStatus } from "./shared/contracts";
import { MODEL_SPEC } from "./shared/model-spec";

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
const modelElement = requireElement<HTMLElement>("#model-name");
const sizeElement = requireElement<HTMLElement>("#model-size");
const hashElement = requireElement<HTMLElement>("#model-hash");
let announcedState = "";

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
    detailElement.textContent = "Model verified. You can close this tab.";
    prepareButton.textContent = "Model verified";
    prepareButton.disabled = true;
    prepareButton.hidden = true;
    progressRegion.hidden = false;
  } else if (status.state === "preparing") {
    stateElement.textContent = "Verifying model";
    detailElement.textContent = "Keep this tab open until verification finishes.";
    prepareButton.textContent = "Verifying model";
    prepareButton.disabled = true;
    prepareButton.hidden = true;
    progressRegion.hidden = false;
  } else if (status.state === "error") {
    stateElement.textContent = "Setup failed";
    detailElement.textContent = status.error ?? "The packaged model could not be verified.";
    prepareButton.textContent = "Retry verification";
    prepareButton.disabled = false;
    prepareButton.hidden = false;
    progressRegion.hidden = true;
  } else {
    stateElement.textContent = "Model not verified";
    detailElement.textContent = "Verify the model before scanning pages.";
    prepareButton.textContent = "Verify model";
    prepareButton.disabled = false;
    prepareButton.hidden = false;
    progressRegion.hidden = true;
  }
  if (announcedState !== status.state) {
    announcedState = status.state;
    announcementElement.textContent = `${stateElement.textContent}. ${detailElement.textContent}`;
  }
}

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
