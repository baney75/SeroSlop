import type { ModelStatus } from "./shared/contracts";
import { MODEL_SPEC } from "./shared/model-spec";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Setup element ${selector} is missing`);
  return element;
}

const stateElement = requireElement<HTMLElement>("#setup-state");
const detailElement = requireElement<HTMLElement>("#setup-detail");
const progressElement = requireElement<HTMLProgressElement>("#setup-progress");
const prepareButton = requireElement<HTMLButtonElement>("#prepare-model");
const modelElement = requireElement<HTMLElement>("#model-name");
const sizeElement = requireElement<HTMLElement>("#model-size");
const hashElement = requireElement<HTMLElement>("#model-hash");

modelElement.textContent = MODEL_SPEC.displayName;
sizeElement.textContent = `${(MODEL_SPEC.weightsBytes / 1_000_000).toFixed(1)} MB`;
hashElement.textContent = MODEL_SPEC.weightsSha256;

function render(status: ModelStatus): void {
  progressElement.max = status.totalBytes || MODEL_SPEC.weightsBytes;
  progressElement.value = status.completedBytes;
  if (status.state === "ready") {
    stateElement.textContent = "Offline ready";
    detailElement.textContent = "The packaged model passed its SHA-256 check and is stored locally. No server is used.";
    prepareButton.textContent = "Model verified";
    prepareButton.disabled = true;
  } else if (status.state === "preparing") {
    stateElement.textContent = "Verifying local model…";
    detailElement.textContent = "Keep this page open while ProofLens verifies and prepares the packaged model.";
    prepareButton.textContent = "Preparing…";
    prepareButton.disabled = true;
  } else if (status.state === "error") {
    stateElement.textContent = "Setup failed";
    detailElement.textContent = status.error ?? "The packaged model could not be verified.";
    prepareButton.textContent = "Retry verification";
    prepareButton.disabled = false;
  } else {
    stateElement.textContent = "Preparing local detector";
    detailElement.textContent = "ProofLens needs to verify its packaged model once before scanning pages.";
    prepareButton.textContent = "Prepare verified model";
    prepareButton.disabled = false;
  }
}

async function prepare(): Promise<void> {
  render({
    state: "preparing",
    modelId: MODEL_SPEC.id,
    completedBytes: 0,
    totalBytes: MODEL_SPEC.weightsBytes,
  });
  try {
    const status = (await chrome.runtime.sendMessage({ type: "PL_PREPARE_MODEL" })) as ModelStatus;
    render(status);
  } catch (error) {
    render({
      state: "error",
      modelId: MODEL_SPEC.id,
      completedBytes: 0,
      totalBytes: MODEL_SPEC.weightsBytes,
      error: error instanceof Error ? error.message : String(error),
    });
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
  const status = (await chrome.runtime.sendMessage({ type: "PL_GET_MODEL_STATUS" })) as ModelStatus;
  render(status);
  if (status.state === "not-installed") await prepare();
}

void initialize();
