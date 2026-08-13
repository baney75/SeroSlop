import { describe, expect, it } from "vitest";
import {
  centerCropGeometry,
  imageDataToNormalizedChw,
  rgbBytesToNormalizedChw,
  resizeShortEdgeGeometry,
  sigmoidLogit,
  softmaxSynthetic,
} from "../src/inference/preprocess";

describe("image preprocessing", () => {
  it("normalizes RGB pixels into planar CHW order", () => {
    const imageData = {
      width: 2,
      height: 1,
      data: new Uint8ClampedArray([0, 127, 255, 255, 255, 127, 0, 255]),
    } as ImageData;
    const result = imageDataToNormalizedChw(imageData, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);

    const expected = [
      -1,
      1,
      127 / 255 / 0.5 - 1,
      127 / 255 / 0.5 - 1,
      1,
      -1,
    ];
    Array.from(result).forEach((value, index) => expect(value).toBeCloseTo(expected[index]!, 7));
  });

  it("computes a centered square crop without stretching", () => {
    expect(centerCropGeometry(400, 200)).toEqual({
      sourceX: 112.72727272727273,
      sourceY: 12.727272727272734,
      sourceSize: 174.54545454545453,
      targetSize: 384,
    });
  });

  it("resizes the short edge while preserving aspect ratio", () => {
    expect(resizeShortEdgeGeometry(400, 200)).toEqual({ width: 880, height: 440 });
    expect(resizeShortEdgeGeometry(200, 400)).toEqual({ width: 440, height: 880 });
    expect(resizeShortEdgeGeometry(8_192, 512)).toEqual({ width: 7_040, height: 440 });
    expect(resizeShortEdgeGeometry(512, 8_192)).toEqual({ width: 440, height: 7_040 });
  });

  it("produces the same tensor from equivalent RGB and RGBA buffers", () => {
    const rgb = rgbBytesToNormalizedChw(
      new Uint8Array([10, 20, 30, 200, 210, 220]),
      2,
      1,
      3,
      [0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5],
    );
    const rgba = rgbBytesToNormalizedChw(
      new Uint8ClampedArray([10, 20, 30, 255, 200, 210, 220, 128]),
      2,
      1,
      4,
      [0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5],
    );
    expect(rgb).toEqual(rgba);
  });
});

describe("output calibration primitives", () => {
  it("converts a finite single logit to a probability", () => {
    expect(sigmoidLogit(0)).toBe(0.5);
    expect(sigmoidLogit(Math.log(3))).toBeCloseTo(0.75, 8);
    expect(() => sigmoidLogit(Number.NaN)).toThrow("invalid logit");
  });

  it("maps the configured synthetic logit through softmax", () => {
    expect(softmaxSynthetic([0, Math.log(3)], 1)).toBeCloseTo(0.75, 8);
  });

  it("rejects malformed model output", () => {
    expect(() => softmaxSynthetic([1], 0)).toThrow("invalid logits");
  });
});
