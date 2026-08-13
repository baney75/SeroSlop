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

function describeModel(status: ModelStatus): string {
  if (status.state === "ready") return "Offline ready";
  if (status.state === "preparing") return "Verifying model…";
  if (status.state === "error") return `Setup error: ${status.error ?? "unknown error"}`;
  return "Setup required";
}

async function initialize(): Promise<void> {
  const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTab = tabs[0];
  if (activeTab?.url) {
    try {
      origin = new URL(activeTab.url).origin;
    } catch {
      origin = "";
    }
  }

  const model = (await chrome.runtime.sendMessage({ type: "PL_GET_MODEL_STATUS" })) as ModelStatus;
  statusElement.textContent = describeModel(model);
  statusElement.dataset.state = model.state;

  if (activeTab?.id !== undefined) {
    const summary = (await chrome.runtime.sendMessage({
      type: "PL_GET_TAB_SUMMARY",
      tabId: activeTab.id,
    })) as TabSummaryResponse;
    pageElement.textContent = `${summary.stats.complete} analyzed · ${summary.stats.flagged} flagged · ${summary.stats.unavailable} unavailable`;
  } else {
    pageElement.textContent = "No supported page is active";
  }

  if (origin) {
    const siteState = (await chrome.runtime.sendMessage({
      type: "PL_GET_SITE_STATE",
      origin,
    })) as SiteStateResponse;
    siteToggle.checked = siteState.enabled;
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
