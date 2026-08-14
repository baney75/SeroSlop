import modelLock from "../../model-lock.json";

export interface ModelSpec {
  id: string;
  displayName: string;
  bundledWeightsPath: string;
  weightsSha256: string;
  weightsBytes: number;
  license: string;
  upstreamRepository: string;
  upstreamRevision: string;
  trainingRecipeVersion: string;
  inputSize: number;
  resizeShortEdge: number;
  inputName: string;
  outputName: string;
  calibration: { slope: number; intercept: number };
  imageMean: readonly [number, number, number];
  imageStd: readonly [number, number, number];
}

/**
 * Immutable contract for the exact artifact evaluated in research/results.
 * The upstream revision identifies the corrected July 2026 CF384 release.
 */
export const MODEL_SPEC: ModelSpec = {
  id: `ProofLens/${modelLock.trainingRecipe}:${modelLock.sha256.slice(0, 12)}:fp32`,
  displayName: "ProofLens Community Forensics ViT-S/16 (FP32)",
  bundledWeightsPath: modelLock.artifact,
  weightsSha256: modelLock.sha256,
  weightsBytes: modelLock.bytes,
  license: modelLock.upstream.license,
  upstreamRepository: modelLock.upstream.repository,
  upstreamRevision: modelLock.upstream.revision,
  trainingRecipeVersion: modelLock.trainingRecipe,
  inputSize: 384,
  resizeShortEdge: 440,
  inputName: modelLock.input.name,
  outputName: modelLock.output.name,
  calibration: { slope: modelLock.calibration.slope, intercept: modelLock.calibration.intercept },
  imageMean: [0.48145466, 0.4578275, 0.40821073],
  imageStd: [0.26862954, 0.26130258, 0.27577711],
};
