import {
  classifyLikelihood,
  type AnalysisState,
  type InferenceResponse,
  type PageInferenceSource,
  type PageStats,
  type SiteStateResponse,
} from "./shared/contracts";
import { extractCssImageUrls, formatLikelihood, type TargetDescriptor } from "./shared/content-targets";

const MIN_DIMENSION = 64;
const MAX_SNAPSHOT_EDGE = 1_024;
const MAX_LOCAL_IMAGE_CHARACTERS = 8 * 1024 * 1024;
const MAX_TARGETS_PER_DOCUMENT = 512;
const MAX_PENDING_ANALYSES = 32;
const MAX_ELEMENTS_PER_SCAN = 5_000;
const MAX_CONCURRENT_ANALYSES = 1;
const FULL_SCAN_INTERVAL_MS = 1_000;
const OVERLAY_REPAIR_DELAY_MS = 250;
const POSITION_MARGIN = 6;

interface TargetRecord extends TargetDescriptor {
  element: HTMLElement;
  badge: HTMLButtonElement;
  state: AnalysisState;
  flagged: boolean;
  unavailable: boolean;
  requestId?: string;
}

const recordsByElement = new Map<HTMLElement, Map<string, TargetRecord>>();
const records = new Set<TargetRecord>();
const pendingRecords: TargetRecord[] = [];
const pendingRecordSet = new Set<TargetRecord>();
let enabled = true;
let labelsVisible = true;
let activeAnalyses = 0;
let positionFrame: number | undefined;
let fullScanFrame: number | undefined;
let fullScanTimer: number | undefined;
let lastFullScanAt = -FULL_SCAN_INTERVAL_MS;
let overlayRepairTimer: number | undefined;
let captureHidingOverlay = false;

const overlayHost = document.createElement("div");
const overlayStyles = {
  all: "initial",
  position: "fixed",
  inset: "0",
  width: "0",
  height: "0",
  pointerEvents: "none",
  zIndex: "2147483647",
} as const;
const overlayStyleTemplate = document.createElement("div");
for (const [property, value] of Object.entries(overlayStyles)) {
  overlayStyleTemplate.style.setProperty(
    property.replace(/[A-Z]/gu, (letter) => `-${letter.toLowerCase()}`),
    value,
    "important",
  );
}
const expectedOverlayStyle = overlayStyleTemplate.style.cssText;

function ensureOverlayIntegrity(): void {
  if (overlayHost.id !== "prooflens-overlay") overlayHost.id = "prooflens-overlay";
  if (overlayHost.getAttribute("aria-label") !== "ProofLens image analysis labels") {
    overlayHost.setAttribute("aria-label", "ProofLens image analysis labels");
  }
  overlayHost.removeAttribute("hidden");
  overlayHost.removeAttribute("inert");
  const desiredStyle = captureHidingOverlay ? `${expectedOverlayStyle} visibility: hidden !important;` : expectedOverlayStyle;
  if (overlayHost.style.cssText !== desiredStyle) {
    overlayHost.style.cssText = desiredStyle;
  }
}
ensureOverlayIntegrity();
const shadow = overlayHost.attachShadow({ mode: "closed" });
const style = document.createElement("style");
style.textContent = `
  button {
    all: initial; align-items: center; background: #283049; border: 1px solid #ffffff66;
    border-radius: 999px; box-shadow: 0 3px 12px #0006; box-sizing: border-box;
    color: #fff; cursor: help; display: block; font: 700 12px/1.2 system-ui, sans-serif;
    left: 0; letter-spacing: .01em; max-width: min(230px, calc(100vw - 12px));
    overflow: hidden; padding: 7px 10px; pointer-events: auto; position: fixed; top: 0;
    text-overflow: ellipsis; white-space: nowrap;
  }
  button[data-state="analyzing"] { background: #3e4778; }
  button[data-state="complete"][data-classification="likely-ai"] { background: #7a2e24; }
  button[data-state="complete"][data-classification="not-flagged"] { background: #176044; }
  button[data-state="unavailable"] { background: #5e3440; }
  button:focus-visible { outline: 3px solid #67b7ff; outline-offset: 2px; }
  @media (prefers-reduced-motion: no-preference) {
    button[data-state="analyzing"] { animation: prooflens-pulse 1.2s ease-in-out infinite alternate; }
  }
  @keyframes prooflens-pulse { to { opacity: .72; } }
`;
shadow.append(style);
document.documentElement.append(overlayHost);

function descriptorsFor(element: HTMLElement): TargetDescriptor[] {
  const descriptors: TargetDescriptor[] = [];
  if (element instanceof HTMLImageElement) {
    const current = element.currentSrc || element.src;
    if (current && current.length <= MAX_LOCAL_IMAGE_CHARACTERS) {
      descriptors.push({ slot: "image", kind: "image", source: current });
    }
  }
  if (element !== overlayHost) {
    const background = getComputedStyle(element).backgroundImage;
    extractCssImageUrls(background, document.baseURI).forEach((source, index) => {
      if (source.length <= MAX_LOCAL_IMAGE_CHARACTERS) {
        descriptors.push({ slot: `background:${index}`, kind: "background", source });
      }
    });
  }
  return descriptors;
}

function eligible(record: TargetRecord): boolean {
  if (!record.element.isConnected || !record.source) return false;
  const rect = record.element.getBoundingClientRect();
  if (rect.width < MIN_DIMENSION || rect.height < MIN_DIMENSION || record.element.getClientRects().length === 0) {
    return false;
  }
  if (record.kind === "image") {
    const image = record.element as HTMLImageElement;
    return image.complete && image.naturalWidth >= MIN_DIMENSION && image.naturalHeight >= MIN_DIMENSION;
  }
  return true;
}

function makeBadge(): HTMLButtonElement {
  const badge = document.createElement("button");
  badge.type = "button";
  badge.textContent = "ProofLens · queued";
  badge.setAttribute("aria-label", "ProofLens image analysis queued");
  badge.dataset.state = "queued";
  badge.hidden = !labelsVisible;
  shadow.append(badge);
  return badge;
}

function updateBadge(record: TargetRecord, label: string, detail: string): void {
  record.badge.dataset.state = record.state;
  record.badge.textContent = label;
  record.badge.title = detail;
  record.badge.setAttribute("aria-label", `${label}. ${detail}`);
  record.badge.hidden = !labelsVisible || !enabled;
  schedulePositions();
  reportStats();
}

function resetRecord(record: TargetRecord, detail = "Waiting for this image to enter the viewport"): void {
  record.requestId = undefined;
  record.state = "queued";
  record.flagged = false;
  record.unavailable = false;
  delete record.badge.dataset.classification;
  delete record.badge.dataset.provider;
  updateBadge(record, "ProofLens · queued", detail);
  intersectionObserver.observe(record.element);
  queueAnalysis(record);
}

function removeRecord(record: TargetRecord): void {
  record.requestId = undefined;
  record.badge.remove();
  pendingRecordSet.delete(record);
  records.delete(record);
}

function syncElement(element: HTMLElement): void {
  if (element === overlayHost || overlayHost.contains(element)) return;
  const descriptors = descriptorsFor(element);
  let slots = recordsByElement.get(element);
  if (!slots && descriptors.length && records.size < MAX_TARGETS_PER_DOCUMENT) {
    slots = new Map();
    recordsByElement.set(element, slots);
    intersectionObserver.observe(element);
  }
  if (!slots) return;

  const liveSlots = new Set(descriptors.map((descriptor) => descriptor.slot));
  for (const [slot, record] of slots) {
    if (liveSlots.has(slot)) continue;
    removeRecord(record);
    slots.delete(slot);
  }
  for (const descriptor of descriptors) {
    const existing = slots.get(descriptor.slot);
    if (existing) {
      if (existing.source !== descriptor.source) {
        existing.source = descriptor.source;
        resetRecord(existing, "The displayed image changed; analysis was restarted");
      }
      continue;
    }
    if (records.size >= MAX_TARGETS_PER_DOCUMENT) break;
    const record: TargetRecord = {
      ...descriptor,
      element,
      badge: makeBadge(),
      state: "queued",
      flagged: false,
      unavailable: false,
    };
    slots.set(descriptor.slot, record);
    records.add(record);
  }
  if (!slots.size) {
    intersectionObserver.unobserve(element);
    recordsByElement.delete(element);
  }
  schedulePositions();
  reportStats();
}

function scan(root: ParentNode = document): void {
  if (root instanceof HTMLElement) syncElement(root);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  let visited = 0;
  while (visited < MAX_ELEMENTS_PER_SCAN && records.size < MAX_TARGETS_PER_DOCUMENT) {
    const next = walker.nextNode();
    if (!next) break;
    if (next instanceof HTMLElement) syncElement(next);
    visited += 1;
  }
}

function scheduleFullScan(): void {
  if (fullScanFrame !== undefined || fullScanTimer !== undefined) return;
  const delay = Math.max(0, FULL_SCAN_INTERVAL_MS - (performance.now() - lastFullScanAt));
  const scheduleFrame = (): void => {
    fullScanTimer = undefined;
    fullScanFrame = requestAnimationFrame(() => {
      fullScanFrame = undefined;
      lastFullScanAt = performance.now();
      scan();
    });
  };
  if (delay > 0) fullScanTimer = window.setTimeout(scheduleFrame, delay);
  else scheduleFrame();
}

function viewportSource(record: TargetRecord): PageInferenceSource | undefined {
  let protocol: string;
  try {
    protocol = new URL(record.source).protocol;
  } catch {
    return undefined;
  }
  if (!["data:", "blob:", "http:", "https:"].includes(protocol)) return undefined;
  if (record.kind === "background" && ["http:", "https:"].includes(protocol) &&
    performance.getEntriesByName(record.source, "resource").length === 0) return undefined;
  const rect = record.element.getBoundingClientRect();
  const left = Math.max(0, rect.left);
  const top = Math.max(0, rect.top);
  const right = Math.min(innerWidth, rect.right);
  const bottom = Math.min(innerHeight, rect.bottom);
  if (right <= left || bottom <= top || innerWidth <= 0 || innerHeight <= 0) return undefined;
  return {
    kind: "viewport-crop",
    crop: { left, top, width: right - left, height: bottom - top, viewportWidth: innerWidth, viewportHeight: innerHeight },
  };
}

function inferenceSource(record: TargetRecord): PageInferenceSource | undefined {
  if (record.kind !== "image") return viewportSource(record);
  const image = record.element as HTMLImageElement;
  try {
    const scale = Math.min(1, MAX_SNAPSHOT_EDGE / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) return viewportSource(record);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const url = canvas.toDataURL("image/png");
    return url.length <= MAX_LOCAL_IMAGE_CHARACTERS ? { kind: "rendered-pixels", url } : viewportSource(record);
  } catch {
    return viewportSource(record);
  }
}

function nearViewport(record: TargetRecord): boolean {
  const rect = record.element.getBoundingClientRect();
  return rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
}

function queueAnalysis(record: TargetRecord): void {
  if (!enabled || record.state !== "queued" || pendingRecordSet.has(record) || !eligible(record) || !nearViewport(record)) return;
  if (pendingRecords.length >= MAX_PENDING_ANALYSES) return;
  pendingRecordSet.add(record);
  pendingRecords.push(record);
  pumpAnalysisQueue();
}

function refillPendingAnalyses(): void {
  if (!enabled) return;
  for (const record of records) {
    if (pendingRecords.length >= MAX_PENDING_ANALYSES) break;
    if (record.state === "queued" && !pendingRecordSet.has(record) && eligible(record) && nearViewport(record)) {
      pendingRecordSet.add(record);
      pendingRecords.push(record);
    }
  }
}

function pumpAnalysisQueue(): void {
  if (activeAnalyses >= MAX_CONCURRENT_ANALYSES || !enabled) return;
  refillPendingAnalyses();
  while (pendingRecords.length) {
    const record = pendingRecords.shift();
    if (!record) return;
    pendingRecordSet.delete(record);
    if (record.state !== "queued" || !eligible(record) || !nearViewport(record)) continue;
    activeAnalyses += 1;
    void analyze(record).finally(() => {
      activeAnalyses -= 1;
      pumpAnalysisQueue();
    });
    return;
  }
}

function nextPaint(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve())));
}

async function sendInference(requestId: string, source: PageInferenceSource): Promise<InferenceResponse> {
  if (source.kind !== "viewport-crop") {
    return chrome.runtime.sendMessage({ type: "PL_INFER", requestId, source }) as Promise<InferenceResponse>;
  }
  captureHidingOverlay = true;
  overlayHost.style.setProperty("visibility", "hidden", "important");
  await nextPaint();
  try {
    return await chrome.runtime.sendMessage({ type: "PL_INFER", requestId, source }) as InferenceResponse;
  } finally {
    captureHidingOverlay = false;
    overlayHost.style.removeProperty("visibility");
    ensureOverlayIntegrity();
  }
}

async function analyze(record: TargetRecord): Promise<void> {
  if (!enabled || record.state === "analyzing" || !eligible(record)) return;
  record.state = "analyzing";
  record.flagged = false;
  record.unavailable = false;
  const requestId = crypto.randomUUID();
  const expectedSource = record.source;
  record.requestId = requestId;
  updateBadge(record, "ProofLens · analyzing", "Analysis runs privately on this device");

  try {
    const source = inferenceSource(record);
    if (!source) throw new Error("Rendered pixels are unavailable");
    const response = await sendInference(requestId, source);
    if (record.requestId !== requestId || record.source !== expectedSource) return;
    if (!response || response.requestId !== requestId) throw new Error("Inference response did not match its request");
    if (!response.ok || !response.result) {
      record.state = "unavailable";
      record.unavailable = true;
      updateBadge(record, "ProofLens · unavailable", response.error?.message ?? "This image could not be analyzed");
      return;
    }
    record.state = "complete";
    const classification = classifyLikelihood(response.result.aiLikelihood);
    record.flagged = classification === "likely-ai";
    record.badge.dataset.classification = classification;
    record.badge.dataset.provider = response.result.provider;
    const percentage = formatLikelihood(response.result.aiLikelihood);
    const runtime = response.result.provider === "webgpu" ? "WebGPU" : "WASM";
    updateBadge(
      record,
      `AI likelihood · ${percentage}`,
      record.flagged
        ? `Likely AI-generated at the inclusive 65% threshold. Processed locally with ${runtime}. This estimate is not proof.`
        : `Not flagged at the inclusive 65% threshold. Processed locally with ${runtime}. This estimate is not proof of authenticity.`,
    );
  } catch {
    if (record.requestId !== requestId) return;
    record.state = "unavailable";
    record.unavailable = true;
    updateBadge(record, "ProofLens · unavailable", "Image unavailable or blocked");
  }
}

function stats(): Omit<PageStats, "revision"> {
  const values = [...records];
  return {
    total: values.length,
    queued: values.filter((record) => record.state === "queued").length,
    analyzing: values.filter((record) => record.state === "analyzing").length,
    complete: values.filter((record) => record.state === "complete").length,
    flagged: values.filter((record) => record.flagged).length,
    unavailable: values.filter((record) => record.unavailable).length,
  };
}

let statsTimer: number | undefined;
let statsRevision = 0;
function reportStats(): void {
  if (statsTimer !== undefined) return;
  statsTimer = window.setTimeout(() => {
    statsTimer = undefined;
    void chrome.runtime.sendMessage({ type: "PL_PAGE_STATS", stats: { ...stats(), revision: ++statsRevision } });
  }, 100);
}

function positionBadges(): void {
  positionFrame = undefined;
  for (const [element, slots] of recordsByElement) {
    if (!element.isConnected) {
      for (const record of slots.values()) removeRecord(record);
      recordsByElement.delete(element);
      continue;
    }
    const rect = element.getBoundingClientRect();
    const visible = rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
    let stack = 0;
    for (const record of slots.values()) {
      record.badge.hidden = !visible || !enabled || !labelsVisible;
      if (!visible) continue;
      const width = record.badge.offsetWidth || 180;
      const height = record.badge.offsetHeight || 30;
      const left = Math.max(POSITION_MARGIN, Math.min(innerWidth - width - POSITION_MARGIN, rect.right - width - POSITION_MARGIN));
      const top = Math.max(POSITION_MARGIN, Math.min(innerHeight - height - POSITION_MARGIN, rect.top + POSITION_MARGIN + stack * (height + 4)));
      record.badge.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
      stack += 1;
    }
  }
  reportStats();
}

function schedulePositions(): void {
  if (positionFrame !== undefined) return;
  positionFrame = requestAnimationFrame(positionBadges);
}

const intersectionObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      const elementRecords = recordsByElement.get(entry.target as HTMLElement);
      for (const record of elementRecords?.values() ?? []) queueAnalysis(record);
    }
  },
  { rootMargin: "0px" },
);

function scheduleOverlayRepair(): void {
  if (overlayRepairTimer !== undefined) return;
  overlayRepairTimer = window.setTimeout(() => {
    overlayRepairTimer = undefined;
    ensureOverlayIntegrity();
    if (!overlayHost.isConnected) document.documentElement.append(overlayHost);
    schedulePositions();
  }, OVERLAY_REPAIR_DELAY_MS);
}

const mutationObserver = new MutationObserver((mutations) => {
  let fullScanRequired = false;
  for (const mutation of mutations) {
    if (mutation.type === "attributes" && mutation.target instanceof HTMLElement) {
      if (mutation.target === overlayHost) scheduleOverlayRepair();
      else syncElement(mutation.target);
      if (mutation.target instanceof HTMLSourceElement) {
        const image = mutation.target.parentElement?.querySelector("img");
        if (image) syncElement(image);
      }
    }
    mutation.addedNodes.forEach((node) => {
      if (node instanceof HTMLElement) scan(node);
      if (node instanceof HTMLStyleElement || node instanceof HTMLLinkElement) fullScanRequired = true;
    });
  }
  if (!overlayHost.isConnected) scheduleOverlayRepair();
  if (fullScanRequired) scheduleFullScan();
  schedulePositions();
});

chrome.runtime.onMessage.addListener((
  message: { type: string; enabled?: boolean; visible?: boolean },
  sender,
  sendResponse,
) => {
  if (sender.id !== chrome.runtime.id) return false;
  if (message.type === "PL_GET_CONTENT_SNAPSHOT") {
    sendResponse({
      badges: [...records].map((record) => ({
        text: record.badge.textContent,
        title: record.badge.title,
        state: record.badge.dataset.state,
        classification: record.badge.dataset.classification,
        provider: record.badge.dataset.provider,
        hidden: record.badge.hidden,
      })),
      recordCount: records.size,
      pendingCount: pendingRecords.length,
      activeAnalyses,
      targetLimit: MAX_TARGETS_PER_DOCUMENT,
      overlayAttached: overlayHost.isConnected,
    });
    return false;
  }
  if (message.type === "PL_SITE_STATE_CHANGED" && typeof message.enabled === "boolean") {
    enabled = message.enabled;
    for (const record of records) record.badge.hidden = !enabled || !labelsVisible;
    if (enabled) {
      scan();
      pumpAnalysisQueue();
    } else {
      pendingRecords.length = 0;
      pendingRecordSet.clear();
      for (const record of records) {
        record.requestId = undefined;
        if (record.state === "analyzing") {
          record.state = "queued";
          record.flagged = false;
          record.unavailable = false;
          updateBadge(record, "ProofLens · queued", "Analysis paused while ProofLens is disabled for this site");
        }
      }
    }
  }
  if (message.type === "PL_LABEL_VISIBILITY" && typeof message.visible === "boolean") {
    labelsVisible = message.visible;
    for (const record of records) record.badge.hidden = !enabled || !labelsVisible;
  }
  if (message.type === "PL_RESCAN" || message.type === "PL_MODEL_READY") {
    for (const record of records) resetRecord(record, "Queued for fresh local analysis");
    scan();
    pumpAnalysisQueue();
  }
  return false;
});

document.addEventListener(
  "load",
  (event) => {
    if (event.target instanceof HTMLImageElement) {
      syncElement(event.target);
      for (const record of recordsByElement.get(event.target)?.values() ?? []) queueAnalysis(record);
    }
  },
  true,
);

async function start(): Promise<void> {
  const state = (await chrome.runtime.sendMessage({
    type: "PL_GET_SITE_STATE",
    origin: location.origin,
  })) as SiteStateResponse;
  enabled = state.enabled;
  if (enabled) scan();
  mutationObserver.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["src", "srcset", "sizes", "media", "type", "style", "class", "id", "hidden", "inert", "aria-label"],
  });
  window.addEventListener("load", scheduleFullScan, { once: true });
  window.addEventListener("scroll", schedulePositions, { passive: true });
  window.addEventListener("resize", () => {
    schedulePositions();
    scheduleFullScan();
  }, { passive: true });
  scheduleOverlayRepair();
}

void start();
