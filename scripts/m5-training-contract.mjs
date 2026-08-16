import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

const VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];

export function validateM5Recipe(recipe) {
  const requirementsSha256 = createHash("sha256").update(readFileSync("benchmark/m5/runpod-requirements.txt")).digest("hex");
  if (recipe.schemaVersion !== 1 || recipe.name !== "prooflens-m5-runpod-vit-finetune") {
    throw new Error("M5 recipe identity changed");
  }
  if (recipe.baseSource.commit !== "5ab375fad2a744620b6ec75f09e6153c8a409049" ||
      recipe.baseSource.tree !== "fc0afc8a746f3f41c29bbd8713f309856d2bdc53") {
    throw new Error("M5 source boundary changed");
  }
  if (recipe.deliverable.format !== "ONNX FP32" || recipe.deliverable.maximumBytes !== 90_000_000 ||
      recipe.deliverable.networkAfterInstall !== false ||
      JSON.stringify(recipe.deliverable.browserExecution) !== JSON.stringify(["wasm", "webgpu"])) {
    throw new Error("M5 local model boundary changed");
  }
  if (recipe.upstream.repository !== "buildborderless/CommunityForensics-DeepfakeDet-ViT" ||
      recipe.upstream.revision !== "ac6ee457bea904a373065754107451793b56db00" ||
      recipe.upstream.pytorchWeights.sha256 !== "275ba982236ddd6afddf7131f8133e89f537574b964cf8fa5825b4956d741692") {
    throw new Error("M5 upstream model changed");
  }
  if (recipe.initialModel.sha256 !== "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47") {
    throw new Error("M5 initial model changed");
  }
  if (recipe.sourceEvidence.trainingManifest.items !== 112_562 ||
      recipe.sourceEvidence.trainingManifest.compressedSha256 !== "e5dfc79869541ae5c6703b60de250930f1fb8247790f55b67bb1805f5ac73a93" ||
      recipe.sourceEvidence.selectorManifest.items !== 600 ||
      recipe.sourceEvidence.selectorManifest.sha256 !== "643eb365a603309b94b112403ef4250b565b9863d2ec61a5cc48aa80d5f85caa" ||
      recipe.sourceEvidence.initialParityDiagnostic.path !== "benchmark/evidence/m5/initial-parity-diagnostic.json" ||
      recipe.sourceEvidence.initialParityDiagnostic.sha256 !== "c9c673efa0b1a6e4ea79b195ec16c71ae8ac91f962390a49c4e570b6d8de5c11" ||
      createHash("sha256").update(readFileSync(recipe.sourceEvidence.initialParityDiagnostic.path)).digest("hex") !== recipe.sourceEvidence.initialParityDiagnostic.sha256 ||
      recipe.sourceEvidence.h3PixelsRead !== false) {
    throw new Error("M5 score-blind source packet changed");
  }
  if (JSON.stringify(recipe.training.onnxRuntimeProviderPolicy) !== JSON.stringify({ provider: "CUDAExecutionProvider", useTf32: false }) ||
      recipe.training.provider !== "RunPod Secure Cloud (operator-recorded control-plane receipt)" ||
      recipe.training.providerIdentityEvidence !== "operator-attested-control-plane-observation" ||
      recipe.training.providerSignedAttestation !== false ||
      recipe.training.runtimeConsistencyEvidence !== "RUNPOD_POD_ID hash and locally observed GPU match the operator-authored receipt" ||
      recipe.training.requiredGpuProduct !== "NVIDIA L40S" ||
      recipe.training.containerImage !== "pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385" ||
      recipe.training.requirementsPath !== "benchmark/m5/runpod-requirements.txt" ||
      recipe.training.requirementsSha256 !== "ec87953539172609d20e1a969b8acdbf34e98a3cc8a71a6df08212c30cd41f11" ||
      requirementsSha256 !== recipe.training.requirementsSha256 ||
      recipe.training.provisioningReceiptPath !== "benchmark/candidates/prooflens-cf384-m5/runpod-provisioning-receipt.json" ||
      recipe.training.maximumPaidWallClockSeconds !== 86_400 || recipe.training.deadlineSafetySeconds !== 300 ||
      recipe.training.providerAutoStopAvailable !== false || recipe.training.providerAutoStopRequired !== false ||
      recipe.training.stopControl !== "trainer-deadline-plus-authenticated-operator-stop" ||
      recipe.training.cudaRequired !== true || recipe.training.mixedPrecision !== "bfloat16" ||
      recipe.training.attentionImplementation !== "eager" ||
      recipe.training.candidateCount !== 6 || recipe.training.effectiveBatchSize !== 128) {
    throw new Error("M5 RunPod training boundary changed");
  }
  const branches = recipe.training.branches;
  if (JSON.stringify(branches.map((row) => row.name)) !== JSON.stringify(["last4", "full"]) ||
      JSON.stringify(branches.flatMap((row) => row.candidateEpochs)) !== JSON.stringify([4, 6, 8, 4, 6, 8])) {
    throw new Error("M5 candidate grid changed");
  }
  if (JSON.stringify(Object.keys(recipe.selection.gates)) !== JSON.stringify(VARIANTS) || recipe.selection.displayThreshold !== 0.65) {
    throw new Error("M5 selector variants or display threshold changed");
  }
  for (const variant of VARIANTS) {
    if (recipe.selection.gates[variant].minimumRealRecall !== 1) {
      throw new Error("M5 zero-observed-false-positive gate changed");
    }
  }
  if (JSON.stringify(recipe.selection.falsePositiveConfidence) !== JSON.stringify({
    method: "Wilson score interval for false-positive proportions",
    confidenceLevel: 0.95,
    sampleUnit: "base-real-image",
    trialsPerVariant: 300,
    poolAcrossVariants: false,
    sharedBaseImagesAcrossVariants: true,
  })) {
    throw new Error("M5 selector false-positive confidence contract changed");
  }
  if (recipe.selection.gates.original.minimumBalancedAccuracy !== 0.97 ||
      recipe.selection.gates.original.minimumSyntheticRecall !== 0.94) {
    throw new Error("M5 original accuracy target changed");
  }
  if (JSON.stringify(recipe.terminalRegressions.map((row) => row.name)) !==
      JSON.stringify(["m3-selector-regression", "m2-development-regression"])) {
    throw new Error("M5 terminal regression order changed");
  }
  const largeEvaluation = recipe.largeSyntheticEvaluation;
  if (largeEvaluation.role !== "post-training-fresh-synthetic-recall-only" ||
      largeEvaluation.sourceStatus !== "must-be-fixed-and-public-before-first-score" ||
      largeEvaluation.minimumItems !== 100_000 || largeEvaluation.batchSize !== 100 ||
      largeEvaluation.minimumBatches !== 1_000 ||
      largeEvaluation.minimumMeanBatchRecallExclusive !== 0.95 ||
      largeEvaluation.minimumMedianBatchRecallExclusive !== 0.95 ||
      largeEvaluation.trainingOverlapAllowed !== false || largeEvaluation.selectorOverlapAllowed !== false ||
      largeEvaluation.regressionOverlapAllowed !== false || largeEvaluation.selectionInfluence !== false) {
    throw new Error("M5 100,000-image evaluation boundary changed");
  }
  if (JSON.stringify(largeEvaluation.scoreBlindnessEvidence) !== JSON.stringify({
    repositoryScoreArtifactsPresentAtSourceLock: false,
    publicSourceLockPrecedesEvaluationReceipt: true,
    firstInferenceAfterLock: "operator-attested",
    privatePriorScoringAbsenceProven: false,
    trainingExclusionClaim: "not-used-in-seroslop-m2-through-m5-gradients-or-selection",
  })) {
    throw new Error("M5 100,000-image score-blindness evidence boundary changed");
  }
  const source = largeEvaluation.source;
  if (source.repository !== "JamalLee/Omni-Fake-SET" ||
      source.revision !== "724e97f5fc9f4b89f59631a8d4e6331712b7d441" ||
      source.configuration !== "image" || source.sourceReportedLicense !== "CC-BY-4.0" ||
      source.expectedParquetShards !== 71 || source.expectedParquetBytes !== 49_751_776_056 ||
      source.eligibleLabel !== "full_synthetic" || source.selectionNamespace !== "seroslop:m5:synthetic-eval:v1" ||
      JSON.stringify(source.splits) !== JSON.stringify(["train", "validation"]) ||
      JSON.stringify(source.excludedGeneratorFamilies) !==
        JSON.stringify(["DALL-E", "FLUX.1-dev", "Midjourney", "Stable Diffusion"]) ||
      largeEvaluation.manifest !== "benchmark/evidence/m5/large-synthetic/manifest.jsonl.gz" ||
      largeEvaluation.batchAssignment !== "benchmark/evidence/m5/large-synthetic/batches.json" ||
      largeEvaluation.sourceLock !== "benchmark/evidence/m5/large-synthetic/source-lock.json" ||
      largeEvaluation.attribution !== "benchmark/evidence/m5/large-synthetic/attribution.json" ||
      largeEvaluation.evaluationReceipt !== "benchmark/evidence/m5/large-synthetic-evaluation.json") {
    throw new Error("M5 100,000-image source lock changed");
  }
  if (recipe.h3Boundary.pixelsMayBeRead !== false || recipe.h3Boundary.acceptedArguments !== false ||
      recipe.h3Boundary.scoreArtifactsBeforeFinalLock !== false) {
    throw new Error("M5 H3 boundary changed");
  }
  return true;
}

export function loadAndValidateM5Recipe(path = "benchmark/m5/recipe.json") {
  const recipe = JSON.parse(readFileSync(path, "utf8"));
  validateM5Recipe(recipe);
  return recipe;
}
