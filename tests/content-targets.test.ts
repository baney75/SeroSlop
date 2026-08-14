import { describe, expect, it } from "vitest";
import { extractCssImageUrls, formatAiScore } from "../src/shared/content-targets";

describe("ordinary webpage target helpers", () => {
  it("extracts, resolves, and deduplicates CSS background URLs", () => {
    expect(
      extractCssImageUrls(
        'linear-gradient(red, blue), url("/hero.webp"), url(\'tile.png\'), url("/hero.webp")',
        "https://example.test/articles/page",
      ),
    ).toEqual(["https://example.test/hero.webp", "https://example.test/articles/tile.png"]);
  });

  it("renders a stable bounded model score without probability semantics", () => {
    expect(formatAiScore(0.65)).toBe("65.0/100");
    expect(formatAiScore(0.9999)).toBe("100.0/100");
    expect(() => formatAiScore(Number.NaN)).toThrow("bounded");
  });
});
