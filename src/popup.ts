import type { ModelStatus, SiteStateResponse, TabSummaryResponse } from "./shared/contracts";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Popup element ${selector} is missing`);
  return element;
}

const statusElement = requireElement<HTMLElement>("#model-status");
const pageElement = requireElement<HTMLElement>("#page-summary");
const feedbackElement = requireElement<HTMLElement>("#control-feedback");
const siteContextElement = requireElement<HTMLElement>("#site-context");
const siteToggle = requireElement<HTMLInputElement>("#site-enabled");
const labelToggle = requireElement<HTMLInputElement>("#labels-visible");
const rescanButton = requireElement<HTMLButtonElement>("#rescan");
const setupLink = requireElement<HTMLAnchorElement>("#open-setup");

let activeTab: chrome.tabs.Tab | undefined;
let origin = "";
let summaryTimer: number | undefined;
let refreshPending = false;
let contentAvailable = false;
let siteMutationPending = false;
let labelMutationPending = false;
let rescanPending = false;
let targetInvalidated = false;

interface ContentSnapshot {
  labelsVisible?: boolean;
  enabled?: boolean;
}

interface TargetContext {
  tabId: number;
  origin: string;
}

function setFeedback(message: string, state: "info" | "error" = "info"): void {
  feedbackElement.textContent = message;
  feedbackElement.dataset.state = state;
}

function describeModel(status: ModelStatus): string {
  if (status.state === "ready") return "Offline ready";
  if (status.state === "preparing") return "Verifying model…";
  if (status.state === "error") return `Setup error: ${status.error ?? "unknown error"}`;
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

function supportedOrigin(url: string | undefined): string {
  if (!url || !/^https?:/u.test(url)) return "";
  try {
    return new URL(url).origin;
  } catch {
    return "";
  }
}

function exposeChangedPage(message = "The page changed. Reopen SeroSlop to use its controls."): void {
  targetInvalidated = true;
  contentAvailable = false;
  pageElement.textContent = "Page changed · reopen SeroSlop";
  siteContextElement.textContent = "The selected page changed";
  siteToggle.disabled = true;
  labelToggle.disabled = true;
  rescanButton.disabled = true;
  rescanButton.textContent = "Reopen SeroSlop to scan";
  setFeedback(message, "error");
}

async function requireCurrentTarget(): Promise<TargetContext | undefined> {
  if (targetInvalidated || !origin || activeTab?.id === undefined) return undefined;
  const [current] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (current?.id !== activeTab.id || supportedOrigin(current.url) !== origin) {
    exposeChangedPage();
    return undefined;
  }
  return { tabId: activeTab.id, origin };
}

async function refreshPageState(): Promise<void> {
  if (activeTab?.id === undefined || refreshPending) return;
  refreshPending = true;
  try {
    const target = await requireCurrentTarget();
    if (!target) return;
    const summary = (await chrome.runtime.sendMessage({
      type: "PL_GET_TAB_SUMMARY",
      tabId: target.tabId,
    })) as TabSummaryResponse;
    pageElement.textContent = describePage(summary.stats);
    const content = await chrome.tabs.sendMessage(target.tabId, { type: "PL_GET_CONTENT_SNAPSHOT" }).catch(() => undefined) as
      | ContentSnapshot
      | undefined;
    if (typeof content?.labelsVisible === "boolean") {
      contentAvailable = true;
      if (!labelMutationPending) {
        labelToggle.checked = content.labelsVisible;
        labelToggle.disabled = false;
        labelToggle.removeAttribute("aria-busy");
      }
      if (!rescanPending) {
        rescanButton.disabled = false;
        rescanButton.textContent = "Re-scan page";
        rescanButton.removeAttribute("aria-busy");
      }
    } else {
      contentAvailable = false;
      if (!labelMutationPending) labelToggle.disabled = true;
      if (!rescanPending) {
        rescanButton.disabled = true;
        rescanButton.textContent = "This page can’t be scanned";
      }
    }
  } catch {
    contentAvailable = false;
    pageElement.textContent = "Page state unavailable";
    labelToggle.disabled = true;
    rescanButton.disabled = true;
    rescanButton.textContent = "This page can’t be scanned";
  } finally {
    refreshPending = false;
  }
}

async function initialize(): Promise<void> {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTab = tabs[0];
  origin = supportedOrigin(activeTab?.url);
  if (origin && activeTab?.url) {
    siteContextElement.textContent = `Automatic analysis on ${new URL(activeTab.url).hostname}`;
  }

  const model = (await chrome.runtime.sendMessage({ type: "PL_GET_MODEL_STATUS" })) as ModelStatus | null;
  if (model && typeof model.state === "string") {
    statusElement.textContent = describeModel(model);
    statusElement.dataset.state = model.state;
  } else {
    statusElement.textContent = "Setup status unavailable";
    statusElement.dataset.state = "error";
  }

  if (activeTab?.id !== undefined && origin) {
    await refreshPageState();
    summaryTimer = window.setInterval(() => void refreshPageState(), 500);
  } else {
    pageElement.textContent = "No supported page is active";
    siteContextElement.textContent = "No supported site selected";
  }

  if (origin) {
    try {
      const siteState = (await chrome.runtime.sendMessage({
        type: "PL_GET_SITE_STATE",
        origin,
      })) as SiteStateResponse;
      if (typeof siteState?.enabled !== "boolean") throw new Error("Saved site state was not available");
      if (!await requireCurrentTarget()) return;
      siteToggle.checked = siteState.enabled;
      siteToggle.disabled = false;
    } catch {
      siteToggle.disabled = true;
      if (!targetInvalidated) {
        setFeedback("Saved site setting unavailable. Reopen SeroSlop to retry.", "error");
      }
    }
  } else {
    siteToggle.disabled = true;
  }
}

siteToggle.addEventListener("change", () => {
  void (async () => {
    if (!origin || activeTab?.id === undefined || siteMutationPending) return;
    const desired = siteToggle.checked;
    siteMutationPending = true;
    siteToggle.disabled = true;
    siteToggle.setAttribute("aria-busy", "true");
    setFeedback("Saving this site setting…");
    try {
      const target = await requireCurrentTarget();
      if (!target) {
        siteToggle.checked = !desired;
        return;
      }
      let stored: SiteStateResponse;
      try {
        stored = (await chrome.runtime.sendMessage({
          type: "PL_SET_SITE_STATE",
          origin: target.origin,
          enabled: desired,
        })) as SiteStateResponse;
        if (typeof stored?.enabled !== "boolean" || stored.enabled !== desired) {
          throw new Error("Saved site state was not confirmed");
        }
      } catch {
        const current = await chrome.runtime.sendMessage({
          type: "PL_GET_SITE_STATE",
          origin: target.origin,
        }).catch(() => undefined) as
          | SiteStateResponse
          | undefined;
        siteToggle.checked = typeof current?.enabled === "boolean" ? current.enabled : !desired;
        setFeedback("Couldn’t save this site setting. Try again.", "error");
        return;
      }
      siteToggle.checked = stored.enabled;
      try {
        const relayed = (await chrome.tabs.sendMessage(target.tabId, {
          type: "PL_SITE_STATE_CHANGED",
          enabled: stored.enabled,
          expectedOrigin: target.origin,
        })) as ContentSnapshot;
        if (relayed?.enabled !== stored.enabled) throw new Error("Page did not confirm the saved site state");
      } catch {
        setFeedback("Saved. The current page changed; this setting will apply after reload.");
        return;
      }
      setFeedback(stored.enabled ? "Analysis enabled for this site." : "Analysis disabled for this site.");
    } finally {
      siteMutationPending = false;
      siteToggle.disabled = targetInvalidated || !origin;
      siteToggle.removeAttribute("aria-busy");
    }
  })();
});

labelToggle.addEventListener("change", () => {
  void (async () => {
    if (activeTab?.id === undefined || labelMutationPending) return;
    const desired = labelToggle.checked;
    const previous = !desired;
    labelMutationPending = true;
    labelToggle.disabled = true;
    labelToggle.setAttribute("aria-busy", "true");
    setFeedback("Updating labels on this page…");
    try {
      const target = await requireCurrentTarget();
      if (!target) {
        labelToggle.checked = previous;
        return;
      }
      const relayed = (await chrome.tabs.sendMessage(target.tabId, {
        type: "PL_LABEL_VISIBILITY",
        visible: desired,
        expectedOrigin: target.origin,
      })) as ContentSnapshot;
      if (relayed?.labelsVisible !== desired) throw new Error("Page did not confirm label visibility");
      const snapshot = (await chrome.tabs.sendMessage(target.tabId, {
        type: "PL_GET_CONTENT_SNAPSHOT",
      })) as ContentSnapshot;
      if (snapshot?.labelsVisible !== desired) throw new Error("Label visibility did not settle");
      labelToggle.checked = desired;
      setFeedback(desired ? "Labels shown on this page." : "Labels hidden on this page.");
    } catch {
      const target = await requireCurrentTarget();
      const snapshot = target ? await chrome.tabs.sendMessage(target.tabId, {
        type: "PL_GET_CONTENT_SNAPSHOT",
      }).catch(() => undefined) as ContentSnapshot | undefined : undefined;
      labelToggle.checked = typeof snapshot?.labelsVisible === "boolean" ? snapshot.labelsVisible : previous;
      if (!targetInvalidated) {
        setFeedback("Couldn’t update labels because the page changed. Try again.", "error");
      }
    } finally {
      labelMutationPending = false;
      labelToggle.disabled = !contentAvailable || targetInvalidated;
      labelToggle.removeAttribute("aria-busy");
    }
  })();
});

window.addEventListener("unload", () => {
  if (summaryTimer !== undefined) window.clearInterval(summaryTimer);
});

rescanButton.addEventListener("click", async () => {
  if (activeTab?.id === undefined || !contentAvailable || rescanPending) return;
  rescanPending = true;
  rescanButton.disabled = true;
  rescanButton.setAttribute("aria-busy", "true");
  rescanButton.textContent = "Scanning again…";
  setFeedback("Starting a fresh scan…");
  try {
    const target = await requireCurrentTarget();
    if (!target) return;
    const [response] = await Promise.all([
      chrome.tabs.sendMessage(target.tabId, {
        type: "PL_RESCAN",
        expectedOrigin: target.origin,
      }) as Promise<{ rescanned?: boolean }>,
      new Promise((resolve) => window.setTimeout(resolve, 500)),
    ]);
    if (response?.rescanned !== true) throw new Error("Page did not confirm the re-scan");
    pageElement.textContent = "Fresh scan queued";
    setFeedback("Scan started.");
    window.setTimeout(() => void refreshPageState(), 300);
  } catch {
    contentAvailable = false;
    pageElement.textContent = "This page can’t be scanned";
    setFeedback("Couldn’t re-scan because the page changed.", "error");
  } finally {
    rescanPending = false;
    rescanButton.disabled = !contentAvailable || targetInvalidated;
    rescanButton.removeAttribute("aria-busy");
    rescanButton.textContent = targetInvalidated
      ? "Reopen SeroSlop to scan"
      : contentAvailable ? "Re-scan page" : "This page can’t be scanned";
  }
});

setupLink.addEventListener("click", (event) => {
  event.preventDefault();
  void chrome.tabs.create({ url: chrome.runtime.getURL("setup.html") });
});

void initialize();
