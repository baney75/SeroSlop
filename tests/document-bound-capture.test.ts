import { describe, expect, it } from "vitest";
import { captureForExactDocument } from "../src/shared/document-bound-capture";

describe("document-bound viewport capture", () => {
  it("returns bytes only when the exact document survives both confirmations", async () => {
    const order: string[] = [];
    const result = await captureForExactDocument({
      tabId: 7,
      expectedOrigin: "https://example.test",
      before: async () => { order.push("before"); return { id: 7, url: "https://example.test/a" }; },
      confirmDocument: async () => { order.push("confirm"); },
      capture: async () => { order.push("capture"); return "local-png"; },
      after: async () => { order.push("after"); return { id: 7, url: "https://example.test/b" }; },
    });
    expect(result).toBe("local-png");
    expect(order).toEqual(["before", "confirm", "capture", "after", "confirm"]);
  });

  it("rejects a same-tab cross-origin navigation during capture", async () => {
    await expect(captureForExactDocument({
      tabId: 7,
      expectedOrigin: "https://example.test",
      before: async () => ({ id: 7, url: "https://example.test/a" }),
      confirmDocument: async () => undefined,
      capture: async () => "wrong-document-png",
      after: async () => ({ id: 7, url: "https://other.test/" }),
    })).rejects.toThrow("changed after viewport capture");
  });

  it("rejects when document-targeted confirmation no longer reaches the sender", async () => {
    let confirmations = 0;
    await expect(captureForExactDocument({
      tabId: 7,
      expectedOrigin: "https://example.test",
      before: async () => ({ id: 7, url: "https://example.test/a" }),
      confirmDocument: async () => {
        confirmations += 1;
        if (confirmations === 2) throw new Error("No document with the given ID");
      },
      capture: async () => "discarded-png",
      after: async () => ({ id: 7, url: "https://example.test/a" }),
    })).rejects.toThrow("No document with the given ID");
  });
});
