import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { inspectOnnxStructure } from "./onnx-structure.mjs";
import { reconstructM3CandidateModel } from "./m3-candidate-patch.mjs";
import {
  M3_BASE_COMMIT,
  M3_LOCK_EXPECTED,
  M3_PUBLICATION_EXPECTED,
  M3_SOURCE_EXPECTED,
  matchesExpectedRows,
} from "./m3-stage-policy.mjs";
import {
  M3,
  digest,
  jsonEqual,
  parseCanonicalM3PublicationLock,
  requireCondition,
  validateM3BrowserFixtureManifest,
  validateM3OnnxEvidence,
  validateM3PublicDocumentation,
  validateM3TrainingPacket,
} from "./m3-training-contract.mjs";


function git(arguments_, options = {}) {
  return execFileSync("git", arguments_, {
    encoding: options.encoding ?? "utf8",
    maxBuffer: options.maxBuffer ?? 128 * 1024 * 1024,
  }).trim();
}

function gitBytes(arguments_) {
  return execFileSync("git", arguments_, { encoding: null, maxBuffer: 128 * 1024 * 1024 });
}

function commitRows(commit) {
  return git(["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, pathname] = line.split("\t");
      return [pathname, status];
    });
}

function parseJson(bytes, pathname) {
  try {
    return JSON.parse(bytes.toString("utf8"));
  } catch (error) {
    throw new Error(`${pathname} is not valid JSON`, { cause: error });
  }
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M3 final verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
const lockCommit = git(["rev-parse", "HEAD^"]);
const sourceCommit = git(["rev-parse", "HEAD^^"]);
requireCondition([head, lockCommit, sourceCommit].every((commit) =>
  git(["rev-list", "--parents", "-n", "1", commit]).split(" ").length === 2),
"M3 source, lock, and final commits must each have one parent");
requireCondition(git(["rev-parse", `${sourceCommit}^`]) === M3_BASE_COMMIT &&
  matchesExpectedRows(commitRows(sourceCommit), M3_SOURCE_EXPECTED) &&
  git(["rev-parse", `${lockCommit}^`]) === sourceCommit &&
  matchesExpectedRows(commitRows(lockCommit), M3_LOCK_EXPECTED) &&
  git(["rev-parse", `${head}^`]) === lockCommit &&
  matchesExpectedRows(commitRows(head), M3_PUBLICATION_EXPECTED),
"M3 source, lock, or final publication changed outside its exact packet");
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");

const paths = {
  recipe: "benchmark/m3/recipe.json",
  selectionSummary: "benchmark/evidence/m3/selection-summary.json",
  validationManifest: "benchmark/evidence/m3/validation-manifest.jsonl",
  regressionManifest: "benchmark/evidence/m2/validation-manifest.jsonl",
  trainingSummary: "benchmark/evidence/m3/training-summary.json",
  calibration: "benchmark/evidence/m3/calibration.json",
  candidateGrid: "benchmark/evidence/m3/candidate-grid.json",
  comparison: "benchmark/evidence/m3/model-comparison.json",
  receipt: "benchmark/evidence/m3/finalization-receipt.json",
  publicationLock: "benchmark/evidence/m3/publication-lock.json",
  model: "weights/prooflens-cf384.onnx",
  modelLock: "model-lock.json",
  weightsReadme: "weights/README.md",
  readme: "README.md",
  modelCard: "MODEL_CARD.md",
  benchmark: "BENCHMARK.md",
  fixtureManifest: "tests/fixtures/model-states/fixture-manifest.json",
  likelyAsset: "tests/fixtures/model-states/likely-ai.png",
  belowAsset: "tests/fixtures/model-states/below-threshold.jpg",
  trainer: "benchmark/modern/train_rehead.py",
};
const entries = Object.fromEntries(await Promise.all(Object.entries(paths).map(async ([name, pathname]) =>
  [name, await readFile(pathname)])));
const recipe = parseJson(entries.recipe, paths.recipe);
const selectionSummary = parseJson(entries.selectionSummary, paths.selectionSummary);
const trainingSummary = parseJson(entries.trainingSummary, paths.trainingSummary);
const calibration = parseJson(entries.calibration, paths.calibration);
const candidateGrid = parseJson(entries.candidateGrid, paths.candidateGrid);
const comparison = parseJson(entries.comparison, paths.comparison);
const receipt = parseJson(entries.receipt, paths.receipt);
const lock = parseCanonicalM3PublicationLock(entries.publicationLock);
const modelLock = parseJson(entries.modelLock, paths.modelLock);
const fixtureManifest = parseJson(entries.fixtureManifest, paths.fixtureManifest);
const model = { sha256: digest(entries.model), bytes: entries.model.length };
const hashes = {
  trainer: digest(entries.trainer),
  recipe: digest(entries.recipe),
  selectionSummary: digest(entries.selectionSummary),
  validationManifest: digest(entries.validationManifest),
  regressionManifest: digest(entries.regressionManifest),
  model: model.sha256,
};

validateM3TrainingPacket({
  summary: trainingSummary,
  calibration,
  grid: candidateGrid,
  recipe,
  selectionSummary,
  hashes,
  model,
});
requireCondition(lock.schemaVersion === 1 && lock.profile === "m3" && lock.sourceCommit === sourceCommit &&
  lock.sourceTree === git(["rev-parse", `${sourceCommit}^{tree}`]) &&
  lock.upstreamModelSha256 === M3.upstreamSha256 && lock.trainerSha256 === hashes.trainer &&
  lock.recipeSha256 === hashes.recipe && lock.selectionSummarySha256 === hashes.selectionSummary &&
  lock.candidateModelBytes === model.bytes && lock.candidateHashes?.["model.onnx"] === model.sha256 &&
  lock.candidateHashes?.["training-summary.json"] === digest(entries.trainingSummary) &&
  lock.candidateHashes?.["calibration.json"] === digest(entries.calibration) &&
  lock.candidateHashes?.["candidate-grid.json"] === digest(entries.candidateGrid) &&
  lock.modelComparisonSha256 === digest(entries.comparison) &&
  lock.freshRunId === trainingSummary.freshFeatureRun.runId &&
  lock.finalizerSha256 === digest(await readFile("benchmark/m3/finalize.py")) &&
  lock.publicationContractSha256 === digest(await readFile("benchmark/m3/publication_contract.py")) &&
  lock.fixtureSelectorSha256 === digest(await readFile("benchmark/m3/select_model_state_fixtures.py")) &&
  lock.documentationRendererSha256 === digest(await readFile("scripts/render-m3-public-docs.mjs")) &&
  jsonEqual(lock.publicDocumentHashes, {
    "README.md": digest(entries.readme),
    "MODEL_CARD.md": digest(entries.modelCard),
    "BENCHMARK.md": digest(entries.benchmark),
  }) && lock.fixtureManifestSha256 === digest(entries.fixtureManifest) &&
  jsonEqual(lock.publicationRows, [...M3_PUBLICATION_EXPECTED].map(([pathname, status]) => ({ path: pathname, status }))) &&
  lock.selectionInfluencedByRegression === false && lock.h3HoldoutScored === false,
"M3 publication lock does not match the final packet");
requireCondition(jsonEqual(lock.candidateEvidenceJson, {
  "training-summary.json": entries.trainingSummary.toString("utf8"),
  "calibration.json": entries.calibration.toString("utf8"),
  "candidate-grid.json": entries.candidateGrid.toString("utf8"),
  "model-comparison.json": entries.comparison.toString("utf8"),
  "fixture-manifest.json": entries.fixtureManifest.toString("utf8"),
}), "M3 final evidence bytes differ from the output-lock packet");
const baseModelBytes = gitBytes(["show", `${M3_BASE_COMMIT}:weights/prooflens-cf384.onnx`]);
const reconstructedModelBytes = reconstructM3CandidateModel({
  baseBytes: baseModelBytes,
  patch: lock.classifierPatch,
});
requireCondition(reconstructedModelBytes.equals(entries.model),
  "M3 final model differs from the classifier-only bytes frozen in the output lock");

const historicalModelLock = parseJson(
  gitBytes(["show", `${M3_BASE_COMMIT}:model-lock.json`]), `${M3_BASE_COMMIT}:model-lock.json`);
for (const name of ["artifact", "format", "input", "output", "upstream"]) {
  requireCondition(jsonEqual(modelLock[name], historicalModelLock[name]),
    `M3 model lock changed the immutable ${name} interface`);
}
requireCondition(modelLock.schemaVersion === 2 && modelLock.bytes === model.bytes && modelLock.sha256 === model.sha256 &&
  modelLock.trainingRecipe === `prooflens-cf384-m3-cultural-heritage-head-v1:${hashes.recipe}:${hashes.selectionSummary}` &&
  jsonEqual(modelLock.calibration, {
    slope: calibration.slope,
    intercept: calibration.intercept,
    displayThreshold: calibration.displayThreshold,
    validationThresholdLogit: calibration.validationThresholdLogit,
  }) && jsonEqual(modelLock.trainingEvidence, {
    recipe: paths.recipe,
    recipeSha256: hashes.recipe,
    selectionSummary: paths.selectionSummary,
    selectionSummarySha256: hashes.selectionSummary,
    trainManifestSha256: trainingSummary.trainManifestSha256,
    trainingSummarySha256: digest(entries.trainingSummary),
    calibrationSha256: digest(entries.calibration),
    candidateGridSha256: digest(entries.candidateGrid),
    validationManifestSha256: hashes.validationManifest,
    regressionManifestSha256: hashes.regressionManifest,
    upstreamModelSha256: M3.upstreamSha256,
    freshFeatureRunId: trainingSummary.freshFeatureRun.runId,
    publicationLockSha256: digest(entries.publicationLock),
  }), "M3 model lock does not bind the final training packet");

requireCondition(receipt.schemaVersion === 1 && receipt.profile === "m3" &&
  receipt.sourceCommit === sourceCommit && receipt.sourceTree === lock.sourceTree &&
  receipt.lockCommit === lockCommit && receipt.publicationLockSha256 === digest(entries.publicationLock) &&
  receipt.candidateDirectory === "benchmark/candidates/prooflens-cf384-m3" &&
  receipt.upstreamSha256 === M3.upstreamSha256 &&
  jsonEqual(receipt.shippedModel, { path: paths.model, sha256: model.sha256, bytes: model.bytes }) &&
  jsonEqual(receipt.sourceEvidenceSha256, {
    "training-summary.json": digest(entries.trainingSummary),
    "calibration.json": digest(entries.calibration),
    "candidate-grid.json": digest(entries.candidateGrid),
  }) && jsonEqual(receipt.publishedEvidenceSha256, {
    "training-summary.json": digest(entries.trainingSummary),
    "calibration.json": digest(entries.calibration),
    "candidate-grid.json": digest(entries.candidateGrid),
    "model-comparison.json": digest(entries.comparison),
  }) && jsonEqual(receipt.publishedRepositorySha256, {
    [paths.model]: model.sha256,
    [paths.modelLock]: digest(entries.modelLock),
    [paths.weightsReadme]: digest(entries.weightsReadme),
    [paths.readme]: digest(entries.readme),
    [paths.modelCard]: digest(entries.modelCard),
    [paths.benchmark]: digest(entries.benchmark),
    [paths.fixtureManifest]: digest(entries.fixtureManifest),
  }) && receipt.selectorGatesPassed === true && receipt.regressionGatesPassed === true &&
  receipt.selectionInfluencedByRegression === false && receipt.h3HoldoutScored === false &&
  jsonEqual(receipt.transactionalRows, [
    ...lock.publicationRows,
  ]) && jsonEqual(receipt.requiredFinalCommitRows, lock.publicationRows),
"M3 finalization receipt changed");
requireCondition(entries.weightsReadme.toString("utf8").includes(model.sha256) &&
  entries.weightsReadme.toString("utf8").includes(model.bytes.toLocaleString("en-US")),
"M3 weights README does not identify the shipped artifact");

const baseStructure = inspectOnnxStructure(baseModelBytes);
const shippedStructure = inspectOnnxStructure(entries.model);
validateM3OnnxEvidence({ baseStructure, shippedStructure, comparison, model });
validateM3PublicDocumentation({
  readme: entries.readme.toString("utf8"),
  modelCard: entries.modelCard.toString("utf8"),
  benchmark: entries.benchmark.toString("utf8"),
  summary: trainingSummary,
  modelSha256: model.sha256,
});
validateM3BrowserFixtureManifest({
  manifest: fixtureManifest,
  manifestSha256: digest(entries.fixtureManifest),
  assets: {
    "likely-ai.png": digest(entries.likelyAsset),
    "below-threshold.jpg": digest(entries.belowAsset),
  },
  calibration: { sha256: digest(entries.calibration), value: calibration },
  model,
  summarySha256: digest(entries.trainingSummary),
});
requireCondition(fixtureManifest.selectorSha256 === lock.fixtureSelectorSha256,
  "M3 browser fixture manifest does not bind the frozen selector");
requireCondition(git(["ls-files", "benchmark/data/m3-head", "benchmark/data/m3-source", "benchmark/data/h3-met-holdout-v1"]) === "",
  "M3 source or H3 holdout pixels entered Git");

console.log(JSON.stringify({
  stage: "m3-final",
  head,
  lockCommit,
  sourceCommit,
  modelSha256: model.sha256,
  modelBytes: model.bytes,
  trainingImages: M3.trainImages,
  trainingViews: M3.trainViews,
  selectorImages: M3.selectorImages,
  regressionImages: M3.regressionImages,
  h3HoldoutScored: false,
  classifierOnly: true,
  policy: "pass",
}));
