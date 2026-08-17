import { describe, expect, it } from "vitest";
import { SCAN_MODE_COPY, isMainContentElement, isScanMode } from "../src/shared/scan-mode";

describe("scan modes", () => {
  it("accepts only the persisted enum", () => {
    expect(["pick", "main", "all"].every(isScanMode)).toBe(true);
    expect(isScanMode("legacy")).toBe(false);
    expect(isScanMode(undefined)).toBe(false);
    expect(isScanMode(true)).toBe(false);
    expect(Object.keys(SCAN_MODE_COPY)).toEqual(["pick", "main", "all"]);
  });
  it("limits main mode to semantic main content and excludes chrome", () => {
    const fake = (inMain: boolean, excludedInsideMain = false) => {
      const main = { contains: (value: unknown) => excludedInsideMain && value === excluded };
      const excluded = {};
      return {
        closest: (selector: string) => selector.startsWith("main")
          ? (inMain ? main : null)
          : (excludedInsideMain ? excluded : null),
      } as unknown as Element;
    };
    expect(isMainContentElement(fake(true))).toBe(true);
    expect(isMainContentElement(fake(true, true))).toBe(false);
    expect(isMainContentElement(fake(false))).toBe(false);
  });
});
