import { describe, expect, it } from "vitest";
import { calculateMetrics } from "../benchmark/metrics";
import type { Prediction } from "../benchmark/types";

function prediction(label: 0 | 1, aiLikelihood: number, source: string): Prediction {
  return {
    id: `${source}-${aiLikelihood}`,
    dataset: "fixture",
    datasetRevision: "fixture",
    split: "test",
    rowIndex: 0,
    path: "fixture",
    imageSha256: "0".repeat(64),
    label,
    source,
    aiLikelihood,
    predictedLabel: aiLikelihood >= 0.65 ? 1 : 0,
    durationMs: 1,
  };
}

describe("balanced accuracy", () => {
  it("weights real and synthetic recall equally at the fixed threshold", () => {
    const metrics = calculateMetrics(
      [
        prediction(0, 0.1, "real"),
        prediction(0, 0.7, "real"),
        prediction(1, 0.9, "sdxl"),
        prediction(1, 0.8, "sdxl"),
        prediction(1, 0.2, "dalle3"),
        prediction(1, 0.1, "dalle3"),
      ],
      0.65,
    );
    expect(metrics.trueRealRate).toBe(0.5);
    expect(metrics.trueSyntheticRate).toBe(0.5);
    expect(metrics.balancedAccuracy).toBe(0.5);
    expect(metrics.accuracy).toBe(0.5);
    expect(metrics.bySource).toHaveLength(3);
  });

  it("rejects a benchmark without both classes", () => {
    expect(() => calculateMetrics([prediction(1, 0.9, "sdxl")], 0.65)).toThrow("empty class");
  });
});
