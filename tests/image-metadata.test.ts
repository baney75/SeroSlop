import { describe, expect, it } from "vitest";
import { assertRasterWithinLimits, inspectRasterDimensions } from "../src/shared/image-metadata";

function png(width: number, height: number): ArrayBuffer {
  const bytes = new Uint8Array(24);
  bytes.set([137, 80, 78, 71, 13, 10, 26, 10], 0);
  const view = new DataView(bytes.buffer);
  view.setUint32(16, width);
  view.setUint32(20, height);
  return bytes.buffer;
}

function jpeg(width: number, height: number): ArrayBuffer {
  const bytes = Uint8Array.from([
    0xff, 0xd8,
    0xff, 0xc0, 0x00, 0x11, 0x08,
    (height >> 8) & 0xff, height & 0xff,
    (width >> 8) & 0xff, width & 0xff,
    0x03, 0x01, 0x11, 0x00, 0x02, 0x11, 0x00, 0x03, 0x11, 0x00,
  ]);
  return bytes.buffer;
}

describe("pre-decode raster budgets", () => {
  it("reads PNG and JPEG dimensions without decoding", () => {
    expect(inspectRasterDimensions(png(640, 480))).toEqual({ format: "png", width: 640, height: 480 });
    expect(inspectRasterDimensions(jpeg(1_280, 720))).toEqual({ format: "jpeg", width: 1_280, height: 720 });
  });

  it("rejects excessive pixels, edges, and aspect ratios", () => {
    const limits = { maxEdge: 8_192, maxPixels: 25_000_000, maxAspectRatio: 16 };
    expect(() => assertRasterWithinLimits({ format: "png", width: 4_000, height: 3_000 }, limits)).not.toThrow();
    expect(() => assertRasterWithinLimits({ format: "png", width: 8_193, height: 1_000 }, limits)).toThrow();
    expect(() => assertRasterWithinLimits({ format: "png", width: 8_000, height: 4_000 }, limits)).toThrow();
    expect(() => assertRasterWithinLimits({ format: "png", width: 8_000, height: 100 }, limits)).toThrow();
  });

  it("fails closed for formats without safe dimension metadata", () => {
    expect(() => inspectRasterDimensions(Uint8Array.from([1, 2, 3, 4]).buffer)).toThrow("format");
  });
});
