import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import { verifyModelRecord } from "../src/shared/model-integrity";

function bytes(value: string): ArrayBuffer {
  return Uint8Array.from(Buffer.from(value)).buffer;
}

describe("persisted model integrity", () => {
  const payload = bytes("verified model bytes");
  const sha256 = createHash("sha256").update(Buffer.from(payload)).digest("hex");
  const lock = { modelId: "fixture", sha256, bytes: payload.byteLength };

  it("accepts only the locked model ID, size, metadata hash, and byte hash", async () => {
    await expect(verifyModelRecord({ modelId: "fixture", sha256, bytes: payload }, lock)).resolves.toBe(payload);
  });

  it("rejects tampered stored bytes even when metadata still claims the locked hash", async () => {
    const tampered = bytes("tampered model bytes");
    await expect(verifyModelRecord({ modelId: "fixture", sha256, bytes: tampered }, { ...lock, bytes: tampered.byteLength }))
      .rejects.toThrow("integrity");
  });

  it("rejects metadata for a different artifact", async () => {
    await expect(verifyModelRecord({ modelId: "other", sha256, bytes: payload }, lock)).rejects.toThrow("not prepared");
  });
});
