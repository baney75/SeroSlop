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
const MAX_MUTATIONS_PER_CALLBACK = 1_000;
const MAX_PENDING_MUTATION_ROOTS = 256;
const MUTATION_BUDGET_WINDOW_MS = 1_000;
const MAX_MUTATION_UNITS_PER_WINDOW = 256;
const FULL_SCAN_INTERVAL_MS = 1_000;
const CSS_RECONCILIATION_INTERVAL_MS = 1_000;
const MAX_CSS_RECONCILIATION_RECORDS = 512;
const OVERLAY_REPAIR_DELAY_MS = 250;
const POSITION_MARGIN = 6;

interface TargetRecord extends TargetDescriptor {
  element: HTMLElement;
  badge: HTMLDivElement;
  state: AnalysisState;
  flagged: boolean;
  unavailable: boolean;
  acceptedResultCount: number;
  needsReconciliation: boolean;
  requestId?: string;
}

const recordsByElement = new Map<HTMLElement, Map<string, TargetRecord>>();
const records = new Set<TargetRecord>();
const pendingRecords: TargetRecord[] = [];
const pendingRecordSet = new Set<TargetRecord>();
const pendingMutationRoots = new Set<HTMLElement>();
let enabled = true;
let labelsVisible = true;
let activeAnalyses = 0;
let positionFrame: number | undefined;
let fullScanFrame: number | undefined;
let fullScanTimer: number | undefined;
let lastFullScanAt = -FULL_SCAN_INTERVAL_MS;
let overlayRepairTimer: number | undefined;
let captureHidingOverlay = false;
let scanPasses = 0;
let lastScanVisited = 0;
let fullDocumentScanRequired = false;
let mutationWindowStartedAt = performance.now();
let mutationUnitsInWindow = 0;
let maxObservedMutationUnitsInWindow = 0;
let mutationBudgetOverflows = 0;
let mutationOverflowHandledInWindow = false;
let synchronousMutationReconciliations = 0;
let mutationCallbackActive = false;

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
  [role="status"] {
    all: initial; align-items: center; background: #283049; border: 1px solid #ffffff66;
    border-radius: 999px; box-shadow: 0 3px 12px #0006; box-sizing: border-box;
    color: #fff; cursor: default; display: block; font: 700 12px/1.2 system-ui, sans-serif;
    left: 0; letter-spacing: .01em; max-width: min(230px, calc(100vw - 12px));
    overflow: hidden; padding: 7px 10px; pointer-events: none; position: fixed; top: 0;
    text-overflow: ellipsis; white-space: nowrap;
  }
  [role="status"][data-state="analyzing"] { background: #3e4778; }
  [role="status"][data-state="complete"][data-classification="likely-ai"] { background: #7a2e24; }
  [role="status"][data-state="complete"][data-classification="not-flagged"] { background: #176044; }
  [role="status"][data-state="unavailable"] { background: #5e3440; }
  @media (prefers-reduced-motion: no-preference) {
    [role="status"][data-state="analyzing"] { animation: prooflens-pulse 1.2s ease-in-out infinite alternate; }
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
    const sources = extractCssImageUrls(background, document.baseURI)
      .filter((source) => source.length <= MAX_LOCAL_IMAGE_CHARACTERS);
    if (sources.length) {
      // The viewport crop contains the rendered composite, so expose one result rather than
      // implying that individual CSS layers were classified separately.
      descriptors.push({ slot: "background:composite", kind: "background", source: `composite:${JSON.stringify(sources)}` });
    }
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

function makeBadge(): HTMLDivElement {
  const badge = document.createElement("div");
  badge.setAttribute("role", "status");
  badge.setAttribute("aria-live", "off");
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

function removePendingRecord(record: TargetRecord): void {
  pendingRecordSet.delete(record);
  for (let index = pendingRecords.length - 1; index >= 0; index -= 1) {
    if (pendingRecords[index] === record) pendingRecords.splice(index, 1);
  }
}

function resetRecord(record: TargetRecord, detail = "Waiting for this image to enter the viewport"): void {
  record.requestId = undefined;
  record.needsReconciliation = false;
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
  removePendingRecord(record);
  records.delete(record);
}

function invalidateRecordForDeferredSync(record: TargetRecord): void {
  record.requestId = undefined;
  if (record.needsReconciliation) return;
  record.needsReconciliation = true;
  record.state = "queued";
  record.flagged = false;
  record.unavailable = false;
  delete record.badge.dataset.classification;
  delete record.badge.dataset.provider;
  removePendingRecord(record);
  updateBadge(record, "ProofLens · refreshing", "The displayed image changed; bounded reconciliation is pending");
}

function invalidateElementForDeferredSync(element: HTMLElement): void {
  for (const record of recordsByElement.get(element)?.values() ?? []) {
    invalidateRecordForDeferredSync(record);
  }
}

function invalidateAllRecordsForDeferredSync(): void {
  for (const record of records) invalidateRecordForDeferredSync(record);
}

function syncElement(element: HTMLElement): void {
  if (mutationCallbackActive) synchronousMutationReconciliations += 1;
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
      if (existing.source !== descriptor.source || existing.needsReconciliation) {
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
      acceptedResultCount: 0,
      needsReconciliation: false,
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

function reconcileVisibleCssBackgrounds(): void {
  if (!enabled) return;
  let checked = 0;
  for (const record of records) {
    if (checked >= MAX_CSS_RECONCILIATION_RECORDS) break;
    if (record.kind !== "background" || !record.element.isConnected || !nearViewport(record)) continue;
    checked += 1;
    syncElement(record.element);
  }
}

function scanWithinBudget(root: ParentNode, budget: number): number {
  let visited = 0;
  if (root instanceof HTMLElement && visited < budget) {
    syncElement(root);
    visited += 1;
  }
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
  while (visited < budget) {
    const next = walker.nextNode();
    if (!next) break;
    if (next instanceof HTMLElement) syncElement(next);
    visited += 1;
  }
  return visited;
}

function scan(root: ParentNode = document): void {
  scanPasses += 1;
  lastFullScanAt = performance.now();
  const visited = scanWithinBudget(root, MAX_ELEMENTS_PER_SCAN);
  lastScanVisited = visited;
}

function queueMutationRoot(root: HTMLElement): void {
  if (root === overlayHost || overlayHost.contains(root)) return;
  for (const queued of pendingMutationRoots) {
    if (queued.contains(root)) return;
    if (root.contains(queued)) pendingMutationRoots.delete(queued);
  }
  if (pendingMutationRoots.size >= MAX_PENDING_MUTATION_ROOTS) {
    fullDocumentScanRequired = true;
    scheduleBoundedScan();
    return;
  }
  pendingMutationRoots.add(root);
  scheduleBoundedScan();
}

function refreshMutationBudgetWindow(now = performance.now()): void {
  if (now - mutationWindowStartedAt < MUTATION_BUDGET_WINDOW_MS) return;
  mutationWindowStartedAt = now;
  mutationUnitsInWindow = 0;
  mutationOverflowHandledInWindow = false;
}

function consumeMutationUnit(): boolean {
  refreshMutationBudgetWindow();
  if (mutationUnitsInWindow >= MAX_MUTATION_UNITS_PER_WINDOW) return false;
  mutationUnitsInWindow += 1;
  maxObservedMutationUnitsInWindow = Math.max(maxObservedMutationUnitsInWindow, mutationUnitsInWindow);
  return true;
}

function handleMutationBudgetOverflow(): void {
  if (!mutationOverflowHandledInWindow) {
    mutationOverflowHandledInWindow = true;
    mutationBudgetOverflows += 1;
    // Any skipped record might be the source mutation for an in-flight result.
    // Invalidate every admitted record once, without forcing synchronous style work.
    invalidateAllRecordsForDeferredSync();
  }
  scheduleFullScan();
}

function scheduleFullScan(): void {
  fullDocumentScanRequired = true;
  scheduleBoundedScan();
}

function scheduleBoundedScan(): void {
  if (fullScanFrame !== undefined || fullScanTimer !== undefined) return;
  const delay = Math.max(0, FULL_SCAN_INTERVAL_MS - (performance.now() - lastFullScanAt));
  const scheduleFrame = (): void => {
    fullScanTimer = undefined;
    fullScanFrame = requestAnimationFrame(() => {
      fullScanFrame = undefined;
      lastFullScanAt = performance.now();
      scanPasses += 1;
      let visited = 0;
      for (const root of pendingMutationRoots) {
        if (visited >= MAX_ELEMENTS_PER_SCAN) break;
        pendingMutationRoots.delete(root);
        if (root.isConnected) visited += scanWithinBudget(root, MAX_ELEMENTS_PER_SCAN - visited);
      }
      if (fullDocumentScanRequired && visited < MAX_ELEMENTS_PER_SCAN) {
        fullDocumentScanRequired = false;
        visited += scanWithinBudget(document, MAX_ELEMENTS_PER_SCAN - visited);
      }
      lastScanVisited = visited;
      if (pendingMutationRoots.size || fullDocumentScanRequired) scheduleBoundedScan();
    });
  };
  if (delay > 0) fullScanTimer = window.setTimeout(scheduleFrame, delay);
  else scheduleFrame();
}

function viewportSource(record: TargetRecord): PageInferenceSource | undefined {
  let sources: string[];
  try {
    sources = record.kind === "background"
      ? JSON.parse(record.source.replace(/^composite:/u, "")) as string[]
      : [record.source];
  } catch {
    return undefined;
  }
  if (!sources.length || sources.some((source) => {
    try {
      return !["data:", "blob:", "http:", "https:"].includes(new URL(source).protocol);
    } catch {
      return true;
    }
  })) return undefined;
  if (record.kind === "background") {
    const unloadedRemoteSource = sources.some((source) => {
      const protocol = new URL(source).protocol;
      return ["http:", "https:"].includes(protocol) && performance.getEntriesByName(source, "resource").length === 0;
    });
    if (unloadedRemoteSource) return undefined;
  }
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
  if (!enabled || record.needsReconciliation || record.state !== "queued" || pendingRecordSet.has(record) || !eligible(record) || !nearViewport(record)) return;
  if (pendingRecords.length >= MAX_PENDING_ANALYSES) return;
  pendingRecordSet.add(record);
  pendingRecords.push(record);
  pumpAnalysisQueue();
}

function refillPendingAnalyses(): void {
  if (!enabled) return;
  for (const record of records) {
    if (pendingRecords.length >= MAX_PENDING_ANALYSES) break;
    if (!record.needsReconciliation && record.state === "queued" && !pendingRecordSet.has(record) && eligible(record) && nearViewport(record)) {
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
    if (record.needsReconciliation || record.state !== "queued" || !eligible(record) || !nearViewport(record)) continue;
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
  if (!enabled || record.needsReconciliation || record.state === "analyzing" || !eligible(record)) return;
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
    record.acceptedResultCount += 1;
    const classification = classifyLikelihood(response.result.aiLikelihood);
    record.flagged = classification === "likely-ai";
    record.badge.dataset.classification = classification;
    record.badge.dataset.provider = response.result.provider;
    const percentage = formatLikelihood(response.result.aiLikelihood);
    const runtime = response.result.provider === "webgpu" ? "WebGPU" : "WASM";
    updateBadge(
      record,
      record.flagged ? `Likely AI · ${percentage}` : `Not flagged · ${percentage}`,
      record.flagged
        ? `65.0% and above is flagged. Processed locally with ${runtime}. This estimate is not proof.`
        : `Below the inclusive 65.0% threshold. Processed locally with ${runtime}. This estimate is not proof of authenticity.`,
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
  let needsFullScan = false;
  let processedMutations = 0;
  let processedNodes = 0;
  mutationCallbackActive = true;
  try {
    for (const mutation of mutations) {
      if (processedMutations >= MAX_MUTATIONS_PER_CALLBACK || !consumeMutationUnit()) {
        needsFullScan = true;
        handleMutationBudgetOverflow();
        break;
      }
      processedMutations += 1;
      if (mutation.type === "attributes" && mutation.target instanceof HTMLElement) {
        if (mutation.target === overlayHost) scheduleOverlayRepair();
        else {
          // Reject an in-flight result immediately, but defer computed-style work
          // to the coalesced, per-window bounded scan.
          invalidateElementForDeferredSync(mutation.target);
          queueMutationRoot(mutation.target);
        }
        if (mutation.target instanceof HTMLSourceElement) {
          const image = mutation.target.parentElement?.querySelector("img");
          if (image) {
            invalidateElementForDeferredSync(image);
            queueMutationRoot(image);
          }
        }
      }
      for (const node of mutation.removedNodes) {
        if (processedNodes >= MAX_MUTATIONS_PER_CALLBACK || !consumeMutationUnit()) {
          needsFullScan = true;
          handleMutationBudgetOverflow();
          break;
        }
        processedNodes += 1;
        if (node instanceof HTMLElement) needsFullScan = true;
      }
      if (needsFullScan && mutation.removedNodes.length) break;
      for (const node of mutation.addedNodes) {
        if (processedNodes >= MAX_MUTATIONS_PER_CALLBACK || !consumeMutationUnit()) {
          needsFullScan = true;
          handleMutationBudgetOverflow();
          break;
        }
        processedNodes += 1;
        if (node instanceof HTMLElement) queueMutationRoot(node);
        if (node instanceof HTMLStyleElement || node instanceof HTMLLinkElement) needsFullScan = true;
      }
      if (needsFullScan && processedNodes >= MAX_MUTATIONS_PER_CALLBACK) break;
    }
  } finally {
    mutationCallbackActive = false;
  }
  if (!overlayHost.isConnected) scheduleOverlayRepair();
  if (needsFullScan) scheduleFullScan();
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
        slot: record.slot,
        kind: record.kind,
        elementId: record.element.id || undefined,
        acceptedResultCount: record.acceptedResultCount,
        text: record.badge.textContent,
        title: record.badge.title,
        state: record.badge.dataset.state,
        classification: record.badge.dataset.classification,
        provider: record.badge.dataset.provider,
        animationName: getComputedStyle(record.badge).animationName,
        hidden: record.badge.hidden,
      })),
      recordCount: records.size,
      pendingCount: pendingRecords.length,
      activeAnalyses,
      targetLimit: MAX_TARGETS_PER_DOCUMENT,
      scanPasses,
      lastScanVisited,
      maxElementsPerScan: MAX_ELEMENTS_PER_SCAN,
      pendingMutationRoots: pendingMutationRoots.size,
      maxPendingMutationRoots: MAX_PENDING_MUTATION_ROOTS,
      mutationBudgetWindowMs: MUTATION_BUDGET_WINDOW_MS,
      maxMutationUnitsPerWindow: MAX_MUTATION_UNITS_PER_WINDOW,
      mutationUnitsInWindow,
      maxObservedMutationUnitsInWindow,
      mutationBudgetOverflows,
      synchronousMutationReconciliations,
      fullScanIntervalMs: FULL_SCAN_INTERVAL_MS,
      cssReconciliationIntervalMs: CSS_RECONCILIATION_INTERVAL_MS,
      maxCssReconciliationRecords: MAX_CSS_RECONCILIATION_RECORDS,
      overlayAttached: overlayHost.isConnected,
      labelsVisible,
      enabled,
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
    if (event.isTrusted && event.target instanceof HTMLImageElement) {
      invalidateElementForDeferredSync(event.target);
      queueMutationRoot(event.target);
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
  window.setInterval(reconcileVisibleCssBackgrounds, CSS_RECONCILIATION_INTERVAL_MS);
}

void start();
