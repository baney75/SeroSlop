export interface ActiveTabSnapshot {
  id?: number;
  url?: string;
}

interface DocumentBoundCaptureOptions<T> {
  tabId: number;
  expectedOrigin: string;
  before: () => Promise<ActiveTabSnapshot>;
  confirmDocument: () => Promise<void>;
  capture: () => Promise<T>;
  after: () => Promise<ActiveTabSnapshot>;
}

function origin(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    return new URL(value).origin;
  } catch {
    return undefined;
  }
}

function requireExpectedTab(
  snapshot: ActiveTabSnapshot,
  tabId: number,
  expectedOrigin: string,
  phase: "before" | "after",
): void {
  if (snapshot.id !== tabId || origin(snapshot.url) !== expectedOrigin) {
    throw new Error(`The active document changed ${phase} viewport capture`);
  }
}

/** Capture only while one exact sender document remains active before and after the browser API. */
export async function captureForExactDocument<T>(options: DocumentBoundCaptureOptions<T>): Promise<T> {
  requireExpectedTab(await options.before(), options.tabId, options.expectedOrigin, "before");
  await options.confirmDocument();
  const captured = await options.capture();
  requireExpectedTab(await options.after(), options.tabId, options.expectedOrigin, "after");
  await options.confirmDocument();
  return captured;
}
