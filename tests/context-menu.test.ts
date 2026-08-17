import { describe, expect, it } from "vitest";
import {
  IMAGE_CONTEXT_MENU_ID,
  IMAGE_CONTEXT_MENU_TITLE,
  supportedContextPage,
} from "../src/shared/context-menu";

describe("image context menu", () => {
  it("has one stable, literal image-picker action", () => {
    expect(IMAGE_CONTEXT_MENU_ID).toBe("seroslop-choose-image");
    expect(IMAGE_CONTEXT_MENU_TITLE).toBe("Analyze this image with SeroSlop");
  });

  it("is limited to ordinary HTTP(S) pages", () => {
    expect(supportedContextPage("https://example.test/gallery")).toBe(true);
    expect(supportedContextPage("http://127.0.0.1/image")).toBe(true);
    expect(supportedContextPage("chrome://extensions")).toBe(false);
    expect(supportedContextPage("file:///tmp/image.png")).toBe(false);
    expect(supportedContextPage(undefined)).toBe(false);
  });
});
