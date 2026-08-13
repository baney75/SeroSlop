export interface BenchmarkItem {
  id: string;
  dataset: string;
  datasetRevision: string;
  split: string;
  rowIndex: number;
  path: string;
  imageSha256: string;
  label: 0 | 1;
  source: string;
}

export interface Prediction extends BenchmarkItem {
  rawAiLikelihood?: number;
  aiLikelihood: number;
  predictedLabel: 0 | 1;
  durationMs: number;
}
