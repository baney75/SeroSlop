import type { ModelStatus, SiteStateResponse, TabSummaryResponse } from "./shared/contracts";

function requireElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Popup element ${selector} is missing`);
  return element;
}

const statusElement = requireElement<HTMLElement>("#model-status");
const pageElement = requireElement<HTMLElement>("#page-summary");
const siteToggle = requireElement<HTMLInputElement>("#site-enabled");
const labelToggle = requireElement<HTMLInputElement>("#labels-visible");
const rescanButton = requireElement<HTMLButtonElement>("#rescan");
const setupLink = requireElement<HTMLAnchorElement>("#open-setup");

let activeTab: chrome.tabs.Tab | undefined;
let origin = "";
let summaryTimer: number | undefined;
let refreshPending = false;

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

async function refreshPageState(): Promise<void> {
  if (activeTab?.id === undefined || refreshPending) return;
  refreshPending = true;
  try {
    const summary = (await chrome.runtime.sendMessage({
      type: "PL_GET_TAB_SUMMARY",
      tabId: activeTab.id,
    })) as TabSummaryResponse;
    pageElement.textContent = describePage(summary.stats);
    const content = await chrome.tabs.sendMessage(activeTab.id, { type: "PL_GET_CONTENT_SNAPSHOT" }).catch(() => undefined) as
      | { labelsVisible?: boolean }
      | undefined;
    if (typeof content?.labelsVisible === "boolean") {
      labelToggle.checked = content.labelsVisible;
      labelToggle.disabled = false;
    } else {
      labelToggle.disabled = true;
    }
  } catch {
    pageElement.textContent = "Page state unavailable";
    labelToggle.disabled = true;
  } finally {
    refreshPending = false;
  }
}

async function initialize(): Promise<void> {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTab = tabs[0];
  if (activeTab?.url && /^https?:/u.test(activeTab.url)) {
    try {
      origin = new URL(activeTab.url).origin;
    } catch {
      origin = "";
    }
  }

  const model = (await chrome.runtime.sendMessage({ type: "PL_GET_MODEL_STATUS" })) as ModelStatus | null;
  if (model && typeof model.state === "string") {
    statusElement.textContent = describeModel(model);
    statusElement.dataset.state = model.state;
  } else {
    statusElement.textContent = "Setup status unavailable";
    statusElement.dataset.state = "error";
  }

  if (activeTab?.id !== undefined) {
    await refreshPageState();
    summaryTimer = window.setInterval(() => void refreshPageState(), 500);
  } else {
    pageElement.textContent = "No supported page is active";
  }

  if (origin) {
    const siteState = (await chrome.runtime.sendMessage({
      type: "PL_GET_SITE_STATE",
      origin,
    })) as SiteStateResponse;
    siteToggle.checked = siteState.enabled;
    siteToggle.disabled = false;
  } else {
    siteToggle.disabled = true;
  }
}

siteToggle.addEventListener("change", () => {
  if (!origin || activeTab?.id === undefined) return;
  void chrome.runtime.sendMessage({ type: "PL_SET_SITE_STATE", origin, enabled: siteToggle.checked });
  void chrome.tabs.sendMessage(activeTab.id, { type: "PL_SITE_STATE_CHANGED", enabled: siteToggle.checked });
});

labelToggle.addEventListener("change", () => {
  if (activeTab?.id === undefined) return;
  void chrome.tabs.sendMessage(activeTab.id, { type: "PL_LABEL_VISIBILITY", visible: labelToggle.checked });
});

window.addEventListener("unload", () => {
  if (summaryTimer !== undefined) window.clearInterval(summaryTimer);
});

rescanButton.addEventListener("click", () => {
  if (activeTab?.id === undefined) return;
  void chrome.tabs.sendMessage(activeTab.id, { type: "PL_RESCAN" });
  window.close();
});

setupLink.addEventListener("click", (event) => {
  event.preventDefault();
  void chrome.tabs.create({ url: chrome.runtime.getURL("setup.html") });
});

void initialize();
