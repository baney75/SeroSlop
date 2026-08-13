import { EMPTY_PAGE_STATS, type PageStats } from "./contracts";

interface StoredPageStats {
  documentId: string;
  stats: PageStats;
}

/** Keeps monotonic page-summary revisions scoped to one document navigation. */
export class PageStatsStore {
  private readonly entries = new Map<number, StoredPageStats>();

  update(tabId: number, documentId: string, stats: PageStats): void {
    const current = this.entries.get(tabId);
    if (current?.documentId === documentId && stats.revision < current.stats.revision) return;
    this.entries.set(tabId, { documentId, stats });
  }

  get(tabId: number): PageStats {
    return this.entries.get(tabId)?.stats ?? EMPTY_PAGE_STATS;
  }

  delete(tabId: number): void {
    this.entries.delete(tabId);
  }
}
