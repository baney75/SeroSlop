export const SCAN_MODES = ["pick", "main", "all"] as const;
export type ScanMode = (typeof SCAN_MODES)[number];

export const SCAN_MODE_COPY: Record<ScanMode, { title: string; description: string; action: string }> = {
  pick: {
    title: "Choose an image",
    description: "Pick one image when you want a score.",
    action: "Choose image",
  },
  main: {
    title: "Main images",
    description: "Scan images in the page’s main content.",
    action: "Re-scan main images",
  },
  all: {
    title: "Every image",
    description: "Scan all supported page images.",
    action: "Re-scan every image",
  },
};

export function isScanMode(value: unknown): value is ScanMode {
  return value === "pick" || value === "main" || value === "all";
}

/** Deterministic semantic predicate for the Main images mode. */
export function isMainContentElement(element: Element): boolean {
  if (element.closest('header, nav, aside, footer, [role="navigation"]')) return false;
  const main = element.closest('main, article, [role="main"]');
  return Boolean(main);
}
