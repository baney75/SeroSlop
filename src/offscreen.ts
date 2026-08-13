import { BrowserDetector } from "./inference/detector";
import type {
  BackgroundToOffscreenMessage,
  InferenceFailure,
  InferenceResponse,
  ModelStatus,
} from "./shared/contracts";

const detector = new BrowserDetector();
const MAX_PENDING_INFERENCES = 8;
let inferenceTail: Promise<void> = Promise.resolve();
let pendingInferences = 0;

function failureFor(error: unknown): InferenceFailure {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("not prepared")) return { code: "model-not-ready", message };
  if (message.includes("fetch") || message.includes("URL") || message.includes("network")) {
    return { code: "fetch-failed", message };
  }
  if (message.includes("image") || message.includes("Image") || message.includes("decode")) {
    return { code: "decode-failed", message };
  }
  return { code: "inference-failed", message };
}

async function prepareModel(): Promise<ModelStatus> {
  return detector.prepare(({ completedBytes, totalBytes }) => {
    void chrome.runtime.sendMessage({ type: "PL_SETUP_PROGRESS", completedBytes, totalBytes });
  });
}

function enqueueInference(
  message: Extract<BackgroundToOffscreenMessage, { type: "PL_OFFSCREEN_INFER" }>,
): Promise<InferenceResponse> {
  if (pendingInferences >= MAX_PENDING_INFERENCES) {
    return Promise.resolve({
      ok: false,
      requestId: message.requestId,
      error: { code: "inference-failed", message: "The bounded offscreen inference queue is full" },
    });
  }
  pendingInferences += 1;
  const operation = async (): Promise<InferenceResponse> => {
    try {
      return { ok: true, requestId: message.requestId, result: await detector.infer(message.source) };
    } catch (error) {
      return { ok: false, requestId: message.requestId, error: failureFor(error) };
    }
  };
  const result = inferenceTail.then(operation, operation);
  inferenceTail = result.then(
    () => undefined,
    () => undefined,
  );
  return result.finally(() => {
    pendingInferences -= 1;
  });
}

chrome.runtime.onMessage.addListener((message: BackgroundToOffscreenMessage, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || !sender.url?.startsWith(chrome.runtime.getURL("background.js"))) return false;
  if (message.type === "PL_OFFSCREEN_STATUS") {
    void detector.getStatus().then(sendResponse).catch((error: unknown) => {
      sendResponse({
        state: "error",
        modelId: "unknown",
        completedBytes: 0,
        totalBytes: 0,
        error: error instanceof Error ? error.message : String(error),
      } satisfies ModelStatus);
    });
    return true;
  }
  if (message.type === "PL_OFFSCREEN_PREPARE_MODEL") {
    void prepareModel().then(sendResponse).catch((error: unknown) => {
      sendResponse({
        state: "error",
        modelId: "unknown",
        completedBytes: 0,
        totalBytes: 0,
        error: error instanceof Error ? error.message : String(error),
      } satisfies ModelStatus);
    });
    return true;
  }
  if (message.type === "PL_OFFSCREEN_INFER") {
    void enqueueInference(message).then(sendResponse);
    return true;
  }
  return false;
});
