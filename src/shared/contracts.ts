export const AI_THRESHOLD = 0.65;

export function classifyLikelihood(value: number): "likely-ai" | "not-flagged" {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error("Likelihood must be a probability");
  return value >= AI_THRESHOLD ? "likely-ai" : "not-flagged";
}

export type AnalysisState = "queued" | "analyzing" | "complete" | "unavailable";
export type RuntimeProvider = "webgpu" | "wasm";

export interface ViewportCrop {
  left: number;
  top: number;
  width: number;
  height: number;
  viewportWidth: number;
  viewportHeight: number;
}

/** A local image payload that is safe to pass into the offscreen detector. */
export interface InferenceSource {
  kind: "rendered-pixels" | "captured-viewport";
  /** A lossless image data URL created by the content script or captureVisibleTab. */
  url: string;
  crop?: ViewportCrop;
}

/** A content-script request. Viewport capture is performed by the service worker. */
export type PageInferenceSource =
  | { kind: "rendered-pixels"; url: string }
  | { kind: "viewport-crop"; crop: ViewportCrop };

export interface InferenceResult {
  aiLikelihood: number;
  classification: "likely-ai" | "not-flagged";
  modelId: string;
  provider: RuntimeProvider;
  contentSha256: string;
  durationMs: number;
}

export interface InferenceFailure {
  code: "model-not-ready" | "fetch-failed" | "decode-failed" | "inference-failed";
  message: string;
}

export interface ModelStatus {
  state: "not-installed" | "preparing" | "ready" | "error";
  modelId: string;
  completedBytes: number;
  totalBytes: number;
  error?: string;
}

export interface PageStats {
  revision: number;
  total: number;
  queued: number;
  analyzing: number;
  complete: number;
  flagged: number;
  unavailable: number;
}

export type ContentToBackgroundMessage =
  | { type: "PL_INFER"; requestId: string; source: PageInferenceSource }
  | { type: "PL_PAGE_STATS"; stats: PageStats }
  | { type: "PL_GET_SITE_STATE"; origin: string };

export type UiToBackgroundMessage =
  | { type: "PL_GET_MODEL_STATUS" }
  | { type: "PL_PREPARE_MODEL" }
  | { type: "PL_GET_TAB_SUMMARY"; tabId: number }
  | { type: "PL_SET_SITE_STATE"; origin: string; enabled: boolean };

export type BackgroundToOffscreenMessage =
  | { type: "PL_OFFSCREEN_STATUS" }
  | { type: "PL_OFFSCREEN_PREPARE_MODEL" }
  | { type: "PL_OFFSCREEN_INFER"; requestId: string; source: InferenceSource };

export type RuntimeMessage = ContentToBackgroundMessage | UiToBackgroundMessage | BackgroundToOffscreenMessage;

export interface InferenceResponse {
  ok: boolean;
  requestId: string;
  result?: InferenceResult;
  error?: InferenceFailure;
}

export interface SiteStateResponse {
  enabled: boolean;
}

export interface TabSummaryResponse {
  stats: PageStats;
}

export const EMPTY_PAGE_STATS: PageStats = {
  revision: 0,
  total: 0,
  queued: 0,
  analyzing: 0,
  complete: 0,
  flagged: 0,
  unavailable: 0,
};
