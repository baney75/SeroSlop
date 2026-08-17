import {
  type InferenceResponse,
  type InferenceSource,
  type ModelStatus,
  type RuntimeMessage,
  type SiteStateResponse,
  type TabSummaryResponse,
  type ViewportCrop,
} from "./shared/contracts";
import { MinimumIntervalGate } from "./shared/minimum-interval-gate";
import { PageStatsStore } from "./shared/page-stats-store";
import { captureForExactDocument } from "./shared/document-bound-capture";
import { isScanMode, type ScanMode } from "./shared/scan-mode";

const OFFSCREEN_PATH = "offscreen.html";
const MAX_INFERENCE_REQUESTS = 8;
const MAX_LOCAL_IMAGE_CHARACTERS = 8 * 1024 * 1024;
const OFFSCREEN_READY_ATTEMPTS = 40;
const OFFSCREEN_READY_RETRY_MS = 50;
const MIN_VIEWPORT_CAPTURE_INTERVAL_MS = 600;
const pageStats = new PageStatsStore();
const viewportCaptureGate = new MinimumIntervalGate(MIN_VIEWPORT_CAPTURE_INTERVAL_MS);
let creatingOffscreen: Promise<void> | undefined;
let inferenceRequests = 0;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function validRequestId(value: unknown): value is string {
  return typeof value === "string" && value.length >= 8 && value.length <= 128;
}

function validViewportCrop(value: unknown): value is ViewportCrop {
  if (!isObject(value)) return false;
  const fields = ["left", "top", "width", "height", "viewportWidth", "viewportHeight"] as const;
  if (!fields.every((field) => typeof value[field] === "number" && Number.isFinite(value[field]))) return false;
  const { left, top, width, height, viewportWidth, viewportHeight } = value as Record<typeof fields[number], number>;
  return left >= 0 && top >= 0 && width > 0 && height > 0 && viewportWidth > 0 && viewportHeight > 0 &&
    viewportWidth <= 100_000 && viewportHeight <= 100_000 && left + width <= viewportWidth + 1 && top + height <= viewportHeight + 1;
}

function validRenderedPixels(value: unknown): value is InferenceSource {
  return isObject(value) && value.kind === "rendered-pixels" && typeof value.url === "string" &&
    value.url.length <= MAX_LOCAL_IMAGE_CHARACTERS && value.url.toLowerCase().startsWith("data:image/");
}

function internalSender(sender: chrome.runtime.MessageSender): boolean {
  return sender.id === chrome.runtime.id;
}

function extensionPageSender(sender: chrome.runtime.MessageSender): boolean {
  return internalSender(sender) && Boolean(sender.url?.startsWith(chrome.runtime.getURL("")));
}

function contentSender(sender: chrome.runtime.MessageSender): boolean {
  return internalSender(sender) && sender.tab?.id !== undefined && !extensionPageSender(sender);
}

function inferenceFailure(requestId: string, message: string): InferenceResponse {
  return { ok: false, requestId, error: { code: "inference-failed", message } };
}

function urlOrigin(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    return new URL(value).origin;
  } catch {
    return undefined;
  }
}

async function confirmCaptureDocument(tabId: number, documentId: string, expectedOrigin: string): Promise<void> {
  const response = await chrome.tabs.sendMessage(tabId, {
    type: "PL_CONFIRM_DOCUMENT",
    expectedOrigin,
  }, { documentId }) as { origin?: string } | undefined;
  if (response?.origin !== expectedOrigin) {
    throw new Error("The requesting document changed during viewport capture");
  }
}

async function captureViewport(sender: chrome.runtime.MessageSender, crop: InferenceSource["crop"]): Promise<InferenceSource> {
  const tabId = sender.tab?.id;
  const windowId = sender.tab?.windowId;
  const documentId = sender.documentId;
  const expectedOrigin = urlOrigin(sender.url);
  if (tabId === undefined || windowId === undefined || !crop || typeof documentId !== "string" || !expectedOrigin) {
    throw new Error("Viewport capture requires the exact sending document");
  }
  return viewportCaptureGate.run(async () => {
    const url = await captureForExactDocument({
      tabId,
      expectedOrigin,
      before: async () => (await chrome.tabs.query({ active: true, windowId }))[0] ?? {},
      confirmDocument: () => confirmCaptureDocument(tabId, documentId, expectedOrigin),
      capture: () => chrome.tabs.captureVisibleTab(windowId, { format: "png" }),
      after: async () => (await chrome.tabs.query({ active: true, windowId }))[0] ?? {},
    });
    if (url.length > MAX_LOCAL_IMAGE_CHARACTERS || !url.toLowerCase().startsWith("data:image/png")) {
      throw new Error("Captured viewport exceeds the local image budget");
    }
    return { kind: "captured-viewport", url, crop };
  });
}

async function ensureOffscreen(): Promise<void> {
  const offscreenUrl = chrome.runtime.getURL(OFFSCREEN_PATH);
  const contexts = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
    documentUrls: [offscreenUrl],
  });
  if (contexts.length > 0) return;
  if (!creatingOffscreen) {
    creatingOffscreen = chrome.offscreen
      .createDocument({
        url: OFFSCREEN_PATH,
        reasons: [chrome.offscreen.Reason.BLOBS],
        justification: "Decode webpage images and run the packaged ONNX detector outside service-worker lifetime",
      })
      .finally(() => {
        creatingOffscreen = undefined;
      });
  }
  await creatingOffscreen;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function offscreenMessage<T>(message: RuntimeMessage): Promise<T> {
  await ensureOffscreen();
  let lastError: unknown = new Error("The offscreen document did not respond");
  for (let attempt = 0; attempt < OFFSCREEN_READY_ATTEMPTS; attempt += 1) {
    try {
      // createDocument can resolve before the new document has registered its message
      // listener. A null response means no listener accepted the message, so retrying is
      // safe; every accepted offscreen request returns a non-null response.
      const response = await chrome.runtime.sendMessage(message) as T | null | undefined;
      if (response !== null && response !== undefined) return response;
      lastError = new Error("The offscreen document is not ready");
    } catch (error) {
      lastError = error;
    }
    await delay(OFFSCREEN_READY_RETRY_MS);
  }
  throw lastError;
}

async function disabledOrigins(): Promise<string[]> {
  const stored = await chrome.storage.local.get("disabledOrigins");
  return Array.isArray(stored.disabledOrigins) ? (stored.disabledOrigins as string[]) : [];
}

async function notifyModelReady(): Promise<void> {
  const tabs = await chrome.tabs.query({});
  await Promise.all(
    tabs
      .filter((tab) => tab.id !== undefined)
      .map((tab) => chrome.tabs.sendMessage(tab.id as number, { type: "PL_MODEL_READY" }).catch(() => undefined)),
  );
}

async function handleMessage(message: RuntimeMessage, sender: chrome.runtime.MessageSender): Promise<unknown> {
  switch (message.type) {
    case "PL_INFER": {
      if (!validRequestId(message.requestId)) return inferenceFailure("invalid-request", "Inference request ID is invalid");
      const renderedPixels = validRenderedPixels(message.source);
      const viewportCrop = isObject(message.source) && message.source.kind === "viewport-crop" &&
        validViewportCrop(message.source.crop) && contentSender(sender) &&
        typeof sender.documentId === "string" && typeof sender.url === "string";
      if (!renderedPixels && !viewportCrop) {
        return inferenceFailure(message.requestId, "Inference accepts only local rendered pixels or an active-tab viewport crop");
      }
      if (inferenceRequests >= MAX_INFERENCE_REQUESTS) {
        return inferenceFailure(message.requestId, "The bounded inference queue is full; rescan after current work finishes");
      }
      inferenceRequests += 1;
      try {
        const source = renderedPixels
          ? message.source as InferenceSource
          : await captureViewport(sender, (message.source as { crop: ViewportCrop }).crop);
        return await offscreenMessage<InferenceResponse>({
          type: "PL_OFFSCREEN_INFER",
          requestId: message.requestId,
          source,
        });
      } catch (error) {
        return inferenceFailure(message.requestId, error instanceof Error ? error.message : String(error));
      } finally {
        inferenceRequests -= 1;
      }
    }
    case "PL_PAGE_STATS":
      if (contentSender(sender) && sender.tab?.id !== undefined && typeof sender.documentId === "string" &&
        Object.values(message.stats).every((value) => Number.isSafeInteger(value) && value >= 0) &&
        message.stats.complete + message.stats.queued + message.stats.analyzing + message.stats.unavailable === message.stats.total &&
        message.stats.flagged <= message.stats.complete) {
        pageStats.update(sender.tab.id, sender.documentId, message.stats);
      }
      return { ok: true };
    case "PL_GET_SITE_STATE": {
      if (!contentSender(sender) && !extensionPageSender(sender)) throw new Error("Site state request has an invalid sender");
      if (contentSender(sender) && sender.url && new URL(sender.url).origin !== message.origin) {
        throw new Error("Site origin does not match the sending frame");
      }
      const disabled = await disabledOrigins();
      const stored = await chrome.storage.local.get(["scanMode"]);
      return { enabled: !disabled.includes(message.origin), scanMode: isScanMode(stored.scanMode) ? stored.scanMode : undefined } as SiteStateResponse & { scanMode?: ScanMode };
    }
    case "PL_GET_MODEL_STATUS":
      if (!extensionPageSender(sender)) throw new Error("Model status request has an invalid sender");
      return offscreenMessage<ModelStatus>({ type: "PL_OFFSCREEN_STATUS" });
    case "PL_PREPARE_MODEL": {
      if (!extensionPageSender(sender)) throw new Error("Model preparation request has an invalid sender");
      const status = await offscreenMessage<ModelStatus>({ type: "PL_OFFSCREEN_PREPARE_MODEL" });
      if (status.state === "ready") await notifyModelReady();
      return status;
    }
    case "PL_GET_TAB_SUMMARY":
      if (!extensionPageSender(sender)) throw new Error("Tab summary request has an invalid sender");
      return { stats: pageStats.get(message.tabId) } satisfies TabSummaryResponse;
    case "PL_SET_SITE_STATE": {
      if (!extensionPageSender(sender)) throw new Error("Site state mutation has an invalid sender");
      const disabled = new Set(await disabledOrigins());
      if (message.enabled) disabled.delete(message.origin);
      else disabled.add(message.origin);
      await chrome.storage.local.set({ disabledOrigins: [...disabled] });
      return { enabled: message.enabled } satisfies SiteStateResponse;
    }
    case "PL_GET_SCAN_MODE": {
      if (!extensionPageSender(sender) && !contentSender(sender)) throw new Error("Scan mode request has an invalid sender");
      const stored = await chrome.storage.local.get("scanMode");
      return { scanMode: isScanMode(stored.scanMode) ? stored.scanMode : undefined };
    }
    case "PL_SET_SCAN_MODE": {
      if (!extensionPageSender(sender)) throw new Error("Scan mode mutation has an invalid sender");
      if (!isScanMode(message.scanMode)) throw new Error("A scan mode must be selected");
      await chrome.storage.local.set({ scanMode: message.scanMode });
      return { scanMode: message.scanMode };
    }
    case "PL_OFFSCREEN_STATUS":
    case "PL_OFFSCREEN_PREPARE_MODEL":
    case "PL_OFFSCREEN_INFER":
      return undefined;
  }
}

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
  if (!internalSender(sender) || !isObject(message) || typeof message.type !== "string") return false;
  if (message.type.startsWith("PL_OFFSCREEN_") || message.type === "PL_SETUP_PROGRESS") return false;
  const knownTypes = new Set([
    "PL_INFER",
    "PL_PAGE_STATS",
    "PL_GET_SITE_STATE",
    "PL_GET_MODEL_STATUS",
    "PL_PREPARE_MODEL",
    "PL_GET_TAB_SUMMARY",
    "PL_SET_SITE_STATE",
    "PL_GET_SCAN_MODE",
    "PL_SET_SCAN_MODE",
  ]);
  if (!knownTypes.has(message.type)) return false;
  void handleMessage(message as RuntimeMessage, sender)
    .then(sendResponse)
    .catch((error: unknown) => {
      sendResponse({ ok: false, error: error instanceof Error ? error.message : String(error) });
    });
  return true;
});

chrome.tabs.onRemoved.addListener((tabId) => pageStats.delete(tabId));
chrome.tabs.onUpdated.addListener((tabId, changeInfo) => {
  if (changeInfo.status === "loading" || changeInfo.url !== undefined) pageStats.delete(tabId);
});

chrome.runtime.onInstalled.addListener(({ reason }) => {
  if (reason !== "install") return;
  void chrome.tabs.create({ url: chrome.runtime.getURL("setup.html") });
});
