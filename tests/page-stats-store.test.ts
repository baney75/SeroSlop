import { describe, expect, it } from "vitest";
import { EMPTY_PAGE_STATS, type PageStats } from "../src/shared/contracts";
import { PageStatsStore } from "../src/shared/page-stats-store";

function stats(revision: number, flagged: number): PageStats {
  return {
    revision,
    total: 1,
    queued: 0,
    analyzing: 0,
    complete: 1,
    flagged,
    unavailable: 0,
  };
}

describe("page stats store", () => {
  it("replaces a high-revision prior document with the current navigation", () => {
    const store = new PageStatsStore();
    store.update(7, "document-a", stats(100, 1));
    store.update(7, "document-b", stats(1, 0));

    expect(store.get(7)).toEqual(stats(1, 0));
  });

  it("rejects a late lower revision only within the same document", () => {
    const store = new PageStatsStore();
    store.update(7, "document-a", stats(5, 1));
    store.update(7, "document-a", stats(4, 0));
    expect(store.get(7)).toEqual(stats(5, 1));

    store.delete(7);
    expect(store.get(7)).toEqual(EMPTY_PAGE_STATS);
  });
});
