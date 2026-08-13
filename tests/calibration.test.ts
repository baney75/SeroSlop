import { describe, expect, it } from "vitest";
import { calibrateAiLikelihood } from "../src/inference/calibration";
import { classifyLikelihood } from "../src/shared/contracts";
import { MODEL_SPEC } from "../src/shared/model-spec";

describe("frozen score policy", () => {
  it("maps the validation logit threshold to the displayed 65% cutoff", () => {
    const raw = 1 / (1 + Math.exp(-0.3152931060083219));
    expect(calibrateAiLikelihood(raw, MODEL_SPEC.calibration)).toBeCloseTo(0.65, 12);
  });

  it("uses an inclusive boundary", () => {
    expect(classifyLikelihood(0.649999)).toBe("not-flagged");
    expect(classifyLikelihood(0.65)).toBe("likely-ai");
    expect(classifyLikelihood(0.650001)).toBe("likely-ai");
  });

  it("is monotonic and finite at saturation", () => {
    expect(calibrateAiLikelihood(0.2, MODEL_SPEC.calibration)).toBeLessThan(
      calibrateAiLikelihood(0.8, MODEL_SPEC.calibration),
    );
    expect(calibrateAiLikelihood(0, MODEL_SPEC.calibration)).toBeGreaterThan(0);
    expect(calibrateAiLikelihood(1, MODEL_SPEC.calibration)).toBeLessThanOrEqual(1);
  });
});
