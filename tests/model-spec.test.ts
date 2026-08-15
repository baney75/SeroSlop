import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import lock from "../model-lock.json";
import { MODEL_SPEC } from "../src/shared/model-spec";

describe("pinned detector model", () => {
  it("locks the exact corrected upstream revision and local artifact", () => {
    expect(MODEL_SPEC.displayName).toBe("SeroSlop Community Forensics ViT-S/16 (FP32)");
    expect(MODEL_SPEC.id).toMatch(/^ProofLens\//u);
    expect(MODEL_SPEC.upstreamRevision).toMatch(/^[0-9a-f]{40}$/u);
    expect(MODEL_SPEC.weightsSha256).toMatch(/^[0-9a-f]{64}$/u);
    expect(MODEL_SPEC.weightsSha256).toBe(lock.sha256);
    expect(MODEL_SPEC.weightsBytes).toBe(lock.bytes);
    expect(MODEL_SPEC.upstreamRevision).toBe(lock.upstream.revision);
  });

  it("matches the corrected CF384 single-logit preprocessing contract", () => {
    expect(MODEL_SPEC.imageMean).toEqual([0.48145466, 0.4578275, 0.40821073]);
    expect(MODEL_SPEC.imageStd).toEqual([0.26862954, 0.26130258, 0.27577711]);
    expect(MODEL_SPEC.inputSize).toBe(384);
    expect(MODEL_SPEC.resizeShortEdge).toBe(440);
    expect(MODEL_SPEC.calibration).toEqual({ slope: lock.calibration.slope, intercept: lock.calibration.intercept });
  });

  it("ships the byte-for-byte artifact in the model lock", async () => {
    const bytes = await readFile("weights/prooflens-cf384.onnx");
    expect((await stat("weights/prooflens-cf384.onnx")).size).toBe(lock.bytes);
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(lock.sha256);
  });
});
