import type { Prediction } from "./types";

export interface SourceMetrics {
  source: string;
  count: number;
  accuracy: number;
  meanAiLikelihood: number;
}

export interface BenchmarkMetrics {
  threshold: number;
  count: number;
  realCount: number;
  syntheticCount: number;
  trueRealRate: number;
  trueSyntheticRate: number;
  balancedAccuracy: number;
  accuracy: number;
  bySource: SourceMetrics[];
}

function mean(values: readonly number[]): number {
  if (values.length === 0) throw new Error("Cannot calculate a metric for an empty class");
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function calculateMetrics(predictions: readonly Prediction[], threshold: number): BenchmarkMetrics {
  if (predictions.length === 0) throw new Error("Benchmark contains no predictions");
  if (threshold < 0 || threshold > 1) throw new Error("Threshold must be between zero and one");

  const evaluated = predictions.map((prediction) => ({
    ...prediction,
    predictedLabel: prediction.aiLikelihood >= threshold ? (1 as const) : (0 as const),
  }));
  const real = evaluated.filter((prediction) => prediction.label === 0);
  const synthetic = evaluated.filter((prediction) => prediction.label === 1);
  const trueRealRate = mean(real.map((prediction) => Number(prediction.predictedLabel === 0)));
  const trueSyntheticRate = mean(synthetic.map((prediction) => Number(prediction.predictedLabel === 1)));
  const sources = [...new Set(evaluated.map((prediction) => prediction.source))].sort();

  return {
    threshold,
    count: evaluated.length,
    realCount: real.length,
    syntheticCount: synthetic.length,
    trueRealRate,
    trueSyntheticRate,
    balancedAccuracy: (trueRealRate + trueSyntheticRate) / 2,
    accuracy: mean(evaluated.map((prediction) => Number(prediction.predictedLabel === prediction.label))),
    bySource: sources.map((source) => {
      const group = evaluated.filter((prediction) => prediction.source === source);
      return {
        source,
        count: group.length,
        accuracy: mean(group.map((prediction) => Number(prediction.predictedLabel === prediction.label))),
        meanAiLikelihood: mean(group.map((prediction) => prediction.aiLikelihood)),
      };
    }),
  };
}
