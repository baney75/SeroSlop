import {
  classifyLikelihood,
  type AnalysisState,
  type InferenceResponse,
  type PageInferenceSource,
  type PageStats,
  type SiteStateResponse,
} from "./shared/contracts";
import { extractCssImageUrls, formatAiScore, type TargetDescriptor } from "./shared/content-targets";

const MIN_DIMENSION = 64;
const MAX_SNAPSHOT_EDGE = 1_024;
const MAX_LOCAL_IMAGE_CHARACTERS = 8 * 1024 * 1024;
const MAX_TARGETS_PER_DOCUMENT = 512;
const MAX_PENDING_ANALYSES = 32;
const MAX_ELEMENTS_PER_SCAN = 5_000;
const MIN_FULL_DOCUMENT_ELEMENTS_PER_PASS = 2_500;
const MAX_CONCURRENT_ANALYSES = 1;
const MAX_MUTATIONS_PER_CALLBACK = 1_000;
const MAX_PENDING_MUTATION_ROOTS = 256;
const MUTATION_BUDGET_WINDOW_MS = 1_000;
const MAX_MUTATION_UNITS_PER_WINDOW = 256;
const MAX_DEFERRED_RECONCILIATIONS_PER_PASS = 64;
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
  reconciledMutationEpoch: number;
  admissionSequence: number;
  requestId?: string;
}

interface AdmissionCandidate {
  record: TargetRecord;
  priority: number;
}

const recordsByElement = new Map<HTMLElement, Map<string, TargetRecord>>();
const records = new Set<TargetRecord>();
const pendingRecords: TargetRecord[] = [];
const pendingRecordSet = new Set<TargetRecord>();
const pendingMutationRoots = new Set<HTMLElement>();
const pendingDeferredReconciliationElements = new Set<HTMLElement>();
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
let fullDocumentWalker: TreeWalker | undefined;
let fullDocumentPendingNode: Node | undefined;
let fullDocumentRestartRequested = false;
let mutationWindowStartedAt = performance.now();
let mutationUnitsInWindow = 0;
let maxObservedMutationUnitsInWindow = 0;
let mutationBudgetOverflows = 0;
let mutationInvalidationEpoch = 0;
let mutationOverflowRecoveryPending = false;
let synchronousMutationReconciliations = 0;
let mutationCallbackActive = false;
let nextAdmissionSequence = 0;
let admissionPassDepth = 0;
let admissionCandidates: AdmissionCandidate[] | undefined;
let admissionPriorityEvaluationsInPass = 0;
let lastAdmissionPriorityEvaluations = 0;
let maxAdmissionPriorityEvaluationsInPass = 0;
let admissionPasses = 0;
let documentLifecycleEpoch = 0;
let staleDocumentResponsesRejected = 0;

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
  if (overlayHost.getAttribute("aria-label") !== "SeroSlop image analysis labels") {
    overlayHost.setAttribute("aria-label", "SeroSlop image analysis labels");
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
  [role="status"][hidden] { display: none !important; }
  [role="status"][data-state="analyzing"] { background: #3e4778; }
  [role="status"][data-state="complete"][data-classification="likely-ai"] { background: #7a2e24; }
  [role="status"][data-state="complete"][data-classification="not-flagged"] { background: #334155; }
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
  badge.textContent = "SeroSlop · queued";
  badge.setAttribute("aria-label", "SeroSlop image analysis queued");
  badge.dataset.state = "queued";
  badge.hidden = !labelsVisible || mutationOverflowRecoveryPending;
  shadow.append(badge);
  return badge;
}

function boundedAccessibleName(value: string): string {
  const normalized = value.replace(/\s+/gu, " ").trim();
  return normalized.length > 120 ? `${normalized.slice(0, 117)}…` : normalized;
}

function describeTarget(record: TargetRecord): string {
  if (record.kind === "background") {
    const pageName = boundedAccessibleName(record.element.getAttribute("aria-label") ?? "");
    return pageName ? `CSS background “${pageName}”` : "CSS background image";
  }
  const image = record.element as HTMLImageElement;
  const pageName = boundedAccessibleName(image.alt || image.getAttribute("aria-label") || "");
  return pageName ? `image “${pageName}”` : "image";
}

function updateBadge(record: TargetRecord, label: string, detail: string): void {
  record.badge.dataset.state = record.state;
  record.badge.textContent = label;
  record.badge.title = detail;
  record.badge.setAttribute("aria-label", `${label} for ${describeTarget(record)}. ${detail}`);
  record.badge.hidden = !labelsVisible || !enabled || mutationOverflowRecoveryPending;
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
  record.reconciledMutationEpoch = mutationInvalidationEpoch;
  record.state = "queued";
  record.flagged = false;
  record.unavailable = false;
  delete record.badge.dataset.classification;
  delete record.badge.dataset.provider;
  updateBadge(record, "SeroSlop · queued", detail);
  intersectionObserver.observe(record.element);
  queueAnalysis(record);
}

function removeRecord(record: TargetRecord): void {
  untrackAdmissionCandidate(record);
  record.requestId = undefined;
  record.badge.remove();
  removePendingRecord(record);
  records.delete(record);
}

function admissionPriority(element: HTMLElement, descriptor: TargetDescriptor): number {
  admissionPriorityEvaluationsInPass += 1;
  if (!element.isConnected) return Number.NEGATIVE_INFINITY;
  const rect = element.getBoundingClientRect();
  if (rect.width < MIN_DIMENSION || rect.height < MIN_DIMENSION || element.getClientRects().length === 0) {
    return -3e12;
  }
  const isVisible = rect.bottom > 0 && rect.right > 0 && rect.top < innerHeight && rect.left < innerWidth;
  if (descriptor.kind === "image" && element instanceof HTMLImageElement) {
    if (element.complete && (element.naturalWidth < MIN_DIMENSION || element.naturalHeight < MIN_DIMENSION)) {
      return -2e12;
    }
    if (!element.complete) return isVisible ? 2e12 : -2e12;
  } else if (descriptor.kind === "background") {
    try {
      const sources = JSON.parse(descriptor.source.replace(/^composite:/u, "")) as string[];
      if (!sources.length || sources.some((source) => !["data:", "blob:", "http:", "https:"].includes(new URL(source).protocol))) {
        return -2e12;
      }
    } catch {
      return -2e12;
    }
  }
  if (isVisible) return 3e12;
  const horizontalDistance = rect.right <= 0 ? -rect.right : rect.left >= innerWidth ? rect.left - innerWidth : 0;
  const verticalDistance = rect.bottom <= 0 ? -rect.bottom : rect.top >= innerHeight ? rect.top - innerHeight : 0;
  return 1e12 - Math.hypot(horizontalDistance, verticalDistance);
}

function compareAdmissionCandidates(left: AdmissionCandidate, right: AdmissionCandidate): number {
  return left.priority - right.priority || right.record.admissionSequence - left.record.admissionSequence;
}

function beginAdmissionPass(): void {
  admissionPassDepth += 1;
  if (admissionPassDepth !== 1) return;
  admissionCandidates = undefined;
  admissionPriorityEvaluationsInPass = 0;
  admissionPasses += 1;
}

function endAdmissionPass(): void {
  admissionPassDepth -= 1;
  if (admissionPassDepth !== 0) return;
  lastAdmissionPriorityEvaluations = admissionPriorityEvaluationsInPass;
  maxAdmissionPriorityEvaluationsInPass = Math.max(
    maxAdmissionPriorityEvaluationsInPass,
    admissionPriorityEvaluationsInPass,
  );
  admissionCandidates = undefined;
}

function initializeAdmissionCandidates(): AdmissionCandidate[] {
  if (admissionCandidates) return admissionCandidates;
  admissionCandidates = [...records]
    .map((record) => ({ record, priority: admissionPriority(record.element, record) }))
    .sort(compareAdmissionCandidates);
  return admissionCandidates;
}

function trackAdmissionCandidate(record: TargetRecord): void {
  if (!admissionCandidates) return;
  const candidate = { record, priority: admissionPriority(record.element, record) };
  let lower = 0;
  let upper = admissionCandidates.length;
  while (lower < upper) {
    const middle = (lower + upper) >>> 1;
    if (compareAdmissionCandidates(admissionCandidates[middle]!, candidate) <= 0) lower = middle + 1;
    else upper = middle;
  }
  admissionCandidates.splice(lower, 0, candidate);
}

function untrackAdmissionCandidate(record: TargetRecord): void {
  if (!admissionCandidates) return;
  const index = admissionCandidates.findIndex((candidate) => candidate.record === record);
  if (index >= 0) admissionCandidates.splice(index, 1);
}

function removeIndexedRecord(record: TargetRecord): void {
  const slots = recordsByElement.get(record.element);
  removeRecord(record);
  slots?.delete(record.slot);
  if (slots && slots.size === 0) {
    intersectionObserver.unobserve(record.element);
    recordsByElement.delete(record.element);
  }
}

function ensureAdmissionCapacity(element: HTMLElement, descriptor: TargetDescriptor): boolean {
  if (records.size < MAX_TARGETS_PER_DOCUMENT) return true;
  const ownsPass = admissionPassDepth === 0;
  if (ownsPass) beginAdmissionPass();
  try {
    const candidates = initializeAdmissionCandidates();
    const eviction = candidates.find((candidate) => candidate.record.element !== element);
    const incomingPriority = admissionPriority(element, descriptor);
    if (!eviction || incomingPriority <= eviction.priority) return false;
    removeIndexedRecord(eviction.record);
    return true;
  } finally {
    if (ownsPass) endAdmissionPass();
  }
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
  updateBadge(record, "SeroSlop · refreshing", "The displayed image changed; bounded reconciliation is pending");
}

function invalidateElementForDeferredSync(element: HTMLElement): void {
  let admitted = false;
  for (const record of recordsByElement.get(element)?.values() ?? []) {
    admitted = true;
    invalidateRecordForDeferredSync(record);
  }
  if (admitted) {
    pendingDeferredReconciliationElements.add(element);
    scheduleBoundedScan();
  }
}

function invalidateAllRecordsForDeferredSync(): void {
  for (const record of records) {
    invalidateRecordForDeferredSync(record);
    pendingDeferredReconciliationElements.add(record.element);
  }
}

function syncElement(element: HTMLElement): void {
  if (mutationCallbackActive) synchronousMutationReconciliations += 1;
  if (element === overlayHost || overlayHost.contains(element)) return;
  const descriptors = descriptorsFor(element);
  let slots = recordsByElement.get(element);
  if (!slots && descriptors.length) {
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
    if (!ensureAdmissionCapacity(element, descriptor)) break;
    const record: TargetRecord = {
      ...descriptor,
      element,
      badge: makeBadge(),
      state: "queued",
      flagged: false,
      unavailable: false,
      acceptedResultCount: 0,
      needsReconciliation: false,
      reconciledMutationEpoch: mutationInvalidationEpoch,
      admissionSequence: ++nextAdmissionSequence,
    };
    slots.set(descriptor.slot, record);
    records.add(record);
    trackAdmissionCandidate(record);
    updateBadge(record, "SeroSlop · queued", "Waiting to analyze this image");
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
  beginAdmissionPass();
  let checked = 0;
  try {
    for (const record of records) {
      if (checked >= MAX_CSS_RECONCILIATION_RECORDS) break;
      if (record.kind !== "background" || !record.element.isConnected || !nearViewport(record)) continue;
      checked += 1;
      syncElement(record.element);
    }
  } finally {
    endAdmissionPass();
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

function beginFullDocumentScan(): void {
  fullDocumentWalker = document.createTreeWalker(document, NodeFilter.SHOW_ELEMENT);
  fullDocumentPendingNode = undefined;
  fullDocumentScanRequired = true;
}

function requestFullDocumentScan(): void {
  if (fullDocumentScanRequired && fullDocumentWalker) {
    fullDocumentRestartRequested = true;
    return;
  }
  beginFullDocumentScan();
}

function finishFullDocumentPass(): void {
  fullDocumentWalker = undefined;
  fullDocumentPendingNode = undefined;
  if (fullDocumentRestartRequested) {
    fullDocumentRestartRequested = false;
    beginFullDocumentScan();
  } else {
    fullDocumentScanRequired = false;
  }
}

function scanFullDocumentWithinBudget(budget: number): number {
  if (!fullDocumentWalker) beginFullDocumentScan();
  let visited = 0;
  while (visited < budget) {
    const next = fullDocumentPendingNode ?? fullDocumentWalker?.nextNode();
    fullDocumentPendingNode = undefined;
    if (!next) {
      finishFullDocumentPass();
      break;
    }
    if (next instanceof HTMLElement) syncElement(next);
    visited += 1;
  }
  if (visited === budget && fullDocumentWalker) {
    const next = fullDocumentWalker.nextNode();
    if (next) fullDocumentPendingNode = next;
    else finishFullDocumentPass();
  }
  return visited;
}

function scan(root: ParentNode = document): void {
  beginAdmissionPass();
  scanPasses += 1;
  lastFullScanAt = performance.now();
  try {
    let visited: number;
    if (root === document) {
      requestFullDocumentScan();
      visited = scanFullDocumentWithinBudget(MAX_ELEMENTS_PER_SCAN);
      if (fullDocumentScanRequired) scheduleBoundedScan();
    } else {
      visited = scanWithinBudget(root, MAX_ELEMENTS_PER_SCAN);
    }
    lastScanVisited = visited;
  } finally {
    endAdmissionPass();
  }
}

function queueMutationRoot(root: HTMLElement): void {
  if (root === overlayHost || overlayHost.contains(root)) return;
  for (const queued of pendingMutationRoots) {
    if (queued.contains(root)) return;
    if (root.contains(queued)) pendingMutationRoots.delete(queued);
  }
  if (pendingMutationRoots.size >= MAX_PENDING_MUTATION_ROOTS) {
    requestFullDocumentScan();
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
}

function consumeMutationUnit(): boolean {
  refreshMutationBudgetWindow();
  if (mutationUnitsInWindow >= MAX_MUTATION_UNITS_PER_WINDOW) return false;
  mutationUnitsInWindow += 1;
  maxObservedMutationUnitsInWindow = Math.max(maxObservedMutationUnitsInWindow, mutationUnitsInWindow);
  return true;
}

function handleMutationBudgetOverflow(): void {
  mutationInvalidationEpoch += 1;
  mutationBudgetOverflows += 1;
  if (!mutationOverflowRecoveryPending) {
    mutationOverflowRecoveryPending = true;
    // Any skipped record might be the source mutation for an in-flight result.
    // Invalidate every admitted record once, without forcing synchronous style work.
    invalidateAllRecordsForDeferredSync();
  }
  scheduleFullScan();
  schedulePositions();
}

function finishMutationOverflowRecoveryIfStable(): void {
  if (!mutationOverflowRecoveryPending || pendingDeferredReconciliationElements.size) return;
  for (const record of records) {
    if (record.reconciledMutationEpoch === mutationInvalidationEpoch) continue;
    invalidateRecordForDeferredSync(record);
    pendingDeferredReconciliationElements.add(record.element);
  }
  if (pendingDeferredReconciliationElements.size) return;
  mutationOverflowRecoveryPending = false;
  schedulePositions();
  pumpAnalysisQueue();
}

function scheduleFullScan(): void {
  requestFullDocumentScan();
  scheduleBoundedScan();
}

function scheduleBoundedScan(): void {
  if (fullScanFrame !== undefined || fullScanTimer !== undefined) return;
  const delay = Math.max(0, FULL_SCAN_INTERVAL_MS - (performance.now() - lastFullScanAt));
  const scheduleFrame = (): void => {
    fullScanTimer = undefined;
    fullScanFrame = requestAnimationFrame(() => {
      fullScanFrame = undefined;
      beginAdmissionPass();
      try {
        lastFullScanAt = performance.now();
        scanPasses += 1;
        let visited = 0;
        let reconciled = 0;
        for (const element of pendingDeferredReconciliationElements) {
          if (reconciled >= MAX_DEFERRED_RECONCILIATIONS_PER_PASS) break;
          pendingDeferredReconciliationElements.delete(element);
          if (recordsByElement.has(element)) syncElement(element);
          reconciled += 1;
        }
        const mutationRootBudget = fullDocumentScanRequired
          ? MAX_ELEMENTS_PER_SCAN - MIN_FULL_DOCUMENT_ELEMENTS_PER_PASS
          : MAX_ELEMENTS_PER_SCAN;
        for (const root of pendingMutationRoots) {
          if (visited >= mutationRootBudget) break;
          pendingMutationRoots.delete(root);
          if (root.isConnected) visited += scanWithinBudget(root, mutationRootBudget - visited);
        }
        if (fullDocumentScanRequired && visited < MAX_ELEMENTS_PER_SCAN) {
          visited += scanFullDocumentWithinBudget(MAX_ELEMENTS_PER_SCAN - visited);
        }
        lastScanVisited = visited;
        finishMutationOverflowRecoveryIfStable();
        if (pendingDeferredReconciliationElements.size || pendingMutationRoots.size || fullDocumentScanRequired) scheduleBoundedScan();
      } finally {
        endAdmissionPass();
      }
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
  if (!enabled || mutationOverflowRecoveryPending || record.needsReconciliation || record.state !== "queued" || pendingRecordSet.has(record) || !eligible(record) || !nearViewport(record)) return;
  if (pendingRecords.length >= MAX_PENDING_ANALYSES) return;
  pendingRecordSet.add(record);
  pendingRecords.push(record);
  pumpAnalysisQueue();
}

function refillPendingAnalyses(): void {
  if (!enabled || mutationOverflowRecoveryPending) return;
  for (const record of records) {
    if (pendingRecords.length >= MAX_PENDING_ANALYSES) break;
    if (!record.needsReconciliation && record.state === "queued" && !pendingRecordSet.has(record) && eligible(record) && nearViewport(record)) {
      pendingRecordSet.add(record);
      pendingRecords.push(record);
    }
  }
}

function pumpAnalysisQueue(): void {
  if (activeAnalyses >= MAX_CONCURRENT_ANALYSES || !enabled || mutationOverflowRecoveryPending) return;
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
  const expectedMutationEpoch = mutationInvalidationEpoch;
  const expectedDocumentLifecycleEpoch = documentLifecycleEpoch;
  record.requestId = requestId;
  updateBadge(record, "SeroSlop · analyzing", "Analyzing this image");

  try {
    const source = inferenceSource(record);
    if (!source) throw new Error("Rendered pixels are unavailable");
    const response = await sendInference(requestId, source);
    if (expectedDocumentLifecycleEpoch !== documentLifecycleEpoch) {
      staleDocumentResponsesRejected += 1;
      return;
    }
    if (expectedMutationEpoch !== mutationInvalidationEpoch) {
      invalidateElementForDeferredSync(record.element);
      return;
    }
    if (record.requestId !== requestId || record.source !== expectedSource) return;
    if (!record.element.isConnected) {
      removeIndexedRecord(record);
      reportStats();
      return;
    }
    if (!eligible(record)) {
      invalidateElementForDeferredSync(record.element);
      return;
    }
    const currentDescriptor = descriptorsFor(record.element).find((descriptor) => descriptor.slot === record.slot);
    if (!currentDescriptor || currentDescriptor.source !== expectedSource) {
      invalidateElementForDeferredSync(record.element);
      return;
    }
    if (!response || response.requestId !== requestId) throw new Error("Inference response did not match its request");
    if (!response.ok || !response.result) {
      record.state = "unavailable";
      record.unavailable = true;
      updateBadge(record, "SeroSlop · unavailable", response.error?.message ?? "This image could not be analyzed");
      return;
    }
    record.state = "complete";
    record.acceptedResultCount += 1;
    const classification = classifyLikelihood(response.result.aiLikelihood);
    record.flagged = classification === "likely-ai";
    record.badge.dataset.classification = classification;
    record.badge.dataset.provider = response.result.provider;
    const score = formatAiScore(response.result.aiLikelihood);
    const runtime = response.result.provider === "webgpu" ? "WebGPU" : "WASM";
    updateBadge(
      record,
      record.flagged ? `Likely AI · ${score}` : `Below flag threshold · ${score}`,
      record.flagged
        ? `Flagged at 65.0/100 and above. Ran with ${runtime}.`
        : `Below 65.0/100. Ran with ${runtime}.`,
    );
  } catch {
    if (record.requestId !== requestId) return;
    record.state = "unavailable";
    record.unavailable = true;
    updateBadge(record, "SeroSlop · unavailable", "Image unavailable or blocked");
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
  interface PositionRect { left: number; top: number; right: number; bottom: number }
  interface VisibleTarget { element: HTMLElement; rect: PositionRect }
  interface PositionCandidate { left: number; top: number; placement: string; inside: boolean }
  const occupied: PositionRect[] = [];
  const visibleTargets: VisibleTarget[] = [];
  for (const element of recordsByElement.keys()) {
    if (!element.isConnected) continue;
    const rect = element.getBoundingClientRect();
    if (rect.bottom <= 0 || rect.right <= 0 || rect.top >= innerHeight || rect.left >= innerWidth) continue;
    visibleTargets.push({
      element,
      rect: { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom },
    });
  }
  const overlaps = (
    left: number,
    top: number,
    width: number,
    height: number,
    rectangles: PositionRect[],
  ): boolean => rectangles.some((rectangle) =>
    left < rectangle.right + 2 && left + width > rectangle.left - 2 &&
    top < rectangle.bottom + 2 && top + height > rectangle.top - 2
  );
  const contains = (outer: PositionRect, left: number, top: number, width: number, height: number): boolean =>
    left >= outer.left && top >= outer.top && left + width <= outer.right && top + height <= outer.bottom;
  const withinViewport = (left: number, top: number, width: number, height: number): boolean =>
    left >= POSITION_MARGIN && top >= POSITION_MARGIN &&
    left + width <= innerWidth - POSITION_MARGIN && top + height <= innerHeight - POSITION_MARGIN;
  const uniqueCandidates = (candidates: PositionCandidate[]): PositionCandidate[] => candidates.filter(
    (candidate, index, values) => values.findIndex((value) =>
      value.left === candidate.left && value.top === candidate.top && value.placement === candidate.placement
    ) === index,
  );
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
      record.badge.hidden = !visible || !enabled || !labelsVisible || mutationOverflowRecoveryPending;
      if (!visible) continue;
      const visibleTarget: PositionRect = {
        left: Math.max(POSITION_MARGIN, rect.left),
        top: Math.max(POSITION_MARGIN, rect.top),
        right: Math.min(innerWidth - POSITION_MARGIN, rect.right),
        bottom: Math.min(innerHeight - POSITION_MARGIN, rect.bottom),
      };
      const compactInside = visibleTarget.right - visibleTarget.left >= 132 &&
        visibleTarget.bottom - visibleTarget.top >= 44;
      const maxBadgeWidth = Math.max(
        96,
        Math.floor(Math.min(
          compactInside ? visibleTarget.right - visibleTarget.left - POSITION_MARGIN * 2 : 180,
          230,
          innerWidth - POSITION_MARGIN * 2,
        )),
      );
      record.badge.style.maxWidth = `${maxBadgeWidth}px`;
      const width = record.badge.offsetWidth || 180;
      const height = record.badge.offsetHeight || 30;
      const stackOffset = stack * (height + 4);
      const candidates: PositionCandidate[] = [];
      if (contains(visibleTarget, visibleTarget.right - width - POSITION_MARGIN,
        visibleTarget.top + POSITION_MARGIN + stackOffset, width, height)) {
        candidates.push({
          left: visibleTarget.right - width - POSITION_MARGIN,
          top: visibleTarget.top + POSITION_MARGIN + stackOffset,
          placement: "inside-top-right",
          inside: true,
        });
      }
      const verticalPositions = [
        rect.top + stackOffset,
        rect.bottom - height - stackOffset,
        rect.top + (rect.height - height) / 2,
      ];
      const horizontalPositions = [
        rect.left + stackOffset,
        rect.right - width - stackOffset,
        rect.left + (rect.width - width) / 2,
      ];
      for (const top of verticalPositions) {
        candidates.push(
          { left: rect.right + POSITION_MARGIN, top, placement: "outside-right", inside: false },
          { left: rect.left - POSITION_MARGIN - width, top, placement: "outside-left", inside: false },
        );
      }
      for (const left of horizontalPositions) {
        candidates.push(
          { left, top: rect.bottom + POSITION_MARGIN, placement: "outside-bottom", inside: false },
          { left, top: rect.top - POSITION_MARGIN - height, placement: "outside-top", inside: false },
        );
      }
      const selected = uniqueCandidates(candidates).find((candidate) => {
        if (!withinViewport(candidate.left, candidate.top, width, height)) return false;
        if (overlaps(candidate.left, candidate.top, width, height, occupied)) return false;
        if (candidate.inside) {
          if (!contains({ left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom },
            candidate.left, candidate.top, width, height)) return false;
          return !visibleTargets.some((target) => target.element !== element &&
            overlaps(candidate.left, candidate.top, width, height, [target.rect]));
        }
        return !visibleTargets.some((target) =>
          overlaps(candidate.left, candidate.top, width, height, [target.rect]));
      });
      if (!selected) {
        record.badge.hidden = true;
        record.badge.dataset.placement = "collision-hidden";
        continue;
      }
      const left = Math.round(selected.left);
      const top = Math.round(selected.top);
      record.badge.dataset.placement = selected.placement;
      record.badge.style.transform = `translate(${left}px, ${top}px)`;
      occupied.push({ left, top, right: left + width, bottom: top + height });
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
  message: { type: string; enabled?: boolean; visible?: boolean; expectedOrigin?: string },
  sender,
  sendResponse,
) => {
  if (sender.id !== chrome.runtime.id) return false;
  if (message.type === "PL_CONFIRM_DOCUMENT") {
    if (message.expectedOrigin === location.origin) sendResponse({ origin: location.origin });
    else sendResponse({ pageChanged: true });
    return false;
  }
  if (message.type === "PL_GET_CONTENT_SNAPSHOT") {
    sendResponse({
      badges: [...records].map((record) => {
        const badgeRect = record.badge.getBoundingClientRect();
        const targetRect = record.element.getBoundingClientRect();
        const computedBadgeStyle = getComputedStyle(record.badge);
        return {
          slot: record.slot,
          kind: record.kind,
          elementId: record.element.id || undefined,
          acceptedResultCount: record.acceptedResultCount,
          text: record.badge.textContent,
          accessibleName: record.badge.getAttribute("aria-label"),
          title: record.badge.title,
          state: record.badge.dataset.state,
          classification: record.badge.dataset.classification,
          provider: record.badge.dataset.provider,
          animationName: computedBadgeStyle.animationName,
          pointerEvents: computedBadgeStyle.pointerEvents,
          display: computedBadgeStyle.display,
          placement: record.badge.dataset.placement,
          badgeRect: {
            left: badgeRect.left,
            top: badgeRect.top,
            right: badgeRect.right,
            bottom: badgeRect.bottom,
            width: badgeRect.width,
            height: badgeRect.height,
          },
          targetRect: {
            left: targetRect.left,
            top: targetRect.top,
            right: targetRect.right,
            bottom: targetRect.bottom,
            width: targetRect.width,
            height: targetRect.height,
          },
          hidden: record.badge.hidden,
        };
      }),
      recordCount: records.size,
      pendingCount: pendingRecords.length,
      activeAnalyses,
      targetLimit: MAX_TARGETS_PER_DOCUMENT,
      scanPasses,
      lastScanVisited,
      maxElementsPerScan: MAX_ELEMENTS_PER_SCAN,
      pendingMutationRoots: pendingMutationRoots.size,
      maxPendingMutationRoots: MAX_PENDING_MUTATION_ROOTS,
      pendingDeferredReconciliations: pendingDeferredReconciliationElements.size,
      maxDeferredReconciliations: MAX_TARGETS_PER_DOCUMENT,
      maxDeferredReconciliationsPerPass: MAX_DEFERRED_RECONCILIATIONS_PER_PASS,
      mutationBudgetWindowMs: MUTATION_BUDGET_WINDOW_MS,
      maxMutationUnitsPerWindow: MAX_MUTATION_UNITS_PER_WINDOW,
      mutationUnitsInWindow,
      maxObservedMutationUnitsInWindow,
      mutationBudgetOverflows,
      mutationInvalidationEpoch,
      mutationOverflowRecoveryPending,
      synchronousMutationReconciliations,
      fullScanIntervalMs: FULL_SCAN_INTERVAL_MS,
      fullDocumentScanRequired,
      fullDocumentRestartRequested,
      cssReconciliationIntervalMs: CSS_RECONCILIATION_INTERVAL_MS,
      maxCssReconciliationRecords: MAX_CSS_RECONCILIATION_RECORDS,
      admissionPasses,
      lastAdmissionPriorityEvaluations,
      maxAdmissionPriorityEvaluationsInPass,
      documentLifecycleEpoch,
      staleDocumentResponsesRejected,
      overlayAttached: overlayHost.isConnected,
      labelsVisible,
      enabled,
    });
    return false;
  }
  if (["PL_SITE_STATE_CHANGED", "PL_LABEL_VISIBILITY", "PL_RESCAN"].includes(message.type) &&
    message.expectedOrigin !== location.origin) {
    sendResponse({ pageChanged: true });
    return false;
  }
  if (message.type === "PL_SITE_STATE_CHANGED" && typeof message.enabled === "boolean") {
    enabled = message.enabled;
    for (const record of records) record.badge.hidden = !enabled || !labelsVisible || mutationOverflowRecoveryPending;
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
          updateBadge(record, "SeroSlop · queued", "Analysis paused while SeroSlop is disabled for this site");
        }
      }
    }
    sendResponse({ enabled });
    return false;
  }
  if (message.type === "PL_LABEL_VISIBILITY" && typeof message.visible === "boolean") {
    labelsVisible = message.visible;
    for (const record of records) record.badge.hidden = !enabled || !labelsVisible || mutationOverflowRecoveryPending;
    sendResponse({ labelsVisible });
    return false;
  }
  if (message.type === "PL_RESCAN" || message.type === "PL_MODEL_READY") {
    for (const record of records) resetRecord(record, "Queued for a fresh scan");
    scan();
    pumpAnalysisQueue();
    sendResponse({ rescanned: message.type === "PL_RESCAN" });
    return false;
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

window.addEventListener("pagehide", () => {
  documentLifecycleEpoch += 1;
  pendingRecords.length = 0;
  pendingRecordSet.clear();
  for (const record of records) {
    record.requestId = undefined;
    if (record.state === "analyzing") record.state = "queued";
  }
});

window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  documentLifecycleEpoch += 1;
  for (const record of records) resetRecord(record, "Queued after page navigation");
  scheduleFullScan();
});

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
    attributeFilter: ["src", "srcset", "sizes", "media", "type", "style", "class", "id", "hidden", "inert", "alt", "aria-label"],
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
