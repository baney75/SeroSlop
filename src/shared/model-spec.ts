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
  id: "ProofLens/community-forensics-cf384-modern-rehead@20260813:fp32",
  displayName: "ProofLens Community Forensics ViT-S/16 (FP32)",
  bundledWeightsPath: "weights/prooflens-cf384.onnx",
  weightsSha256: "29545a1da0cfe2bf0149448334fd45a21f48074c57296db3b84437dd66f80a43",
  weightsBytes: 87_442_080,
  license: "MIT",
  upstreamRepository: "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT",
  upstreamRevision: "ac6ee457bea904a373065754107451793b56db00",
  trainingRecipeVersion: "prooflens-cf384-rehead-v1-20260813",
  inputSize: 384,
  resizeShortEdge: 440,
  inputName: "pixel_values",
  outputName: "logits",
  calibration: { slope: 1, intercept: 0.30374610239790173 },
  imageMean: [0.48145466, 0.4578275, 0.40821073],
  imageStd: [0.26862954, 0.26130258, 0.27577711],
};
