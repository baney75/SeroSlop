import { describe, expect, it } from "vitest";
import { MinimumIntervalGate } from "../src/shared/minimum-interval-gate";

describe("minimum interval gate", () => {
  it("serializes operations and spaces their start times", async () => {
    let now = 1_000;
    const starts: number[] = [];
    const gate = new MinimumIntervalGate(600, () => now, async (milliseconds) => {
      now += milliseconds;
    });

    const results = await Promise.all([
      gate.run(async () => {
        starts.push(now);
        return "first";
      }),
      gate.run(async () => {
        starts.push(now);
        return "second";
      }),
      gate.run(async () => {
        starts.push(now);
        return "third";
      }),
    ]);

    expect(results).toEqual(["first", "second", "third"]);
    expect(starts).toEqual([1_000, 1_600, 2_200]);
  });

  it("continues spacing work after an operation rejects", async () => {
    let now = 5_000;
    const starts: number[] = [];
    const gate = new MinimumIntervalGate(600, () => now, async (milliseconds) => {
      now += milliseconds;
    });

    const rejected = gate.run(async () => {
      starts.push(now);
      throw new Error("capture failed");
    });
    const recovered = gate.run(async () => {
      starts.push(now);
      return "recovered";
    });

    await expect(rejected).rejects.toThrow("capture failed");
    await expect(recovered).resolves.toBe("recovered");
    expect(starts).toEqual([5_000, 5_600]);
  });
});
