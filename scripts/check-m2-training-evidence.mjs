import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { inspectOnnxStructure } from "./onnx-structure.mjs";
import {
  M2_CHECKER_RECOVERY_EXPECTED,
  M2_FINALIZER_SOURCE_COMMIT,
  M2_FINALIZER_SOURCE_EXPECTED,
  M2_PUBLICATION_EXPECTED,
  M2_RECOVERY_COMMIT,
  matchesExpectedRows,
} from "./m2-stage-policy.mjs";
import {
  M2,
  digest,
  jsonEqual,
  requireCondition,
  validateM2PublicationMetadata,
  validateM2OnnxEvidence,
  validateM2TrainingPacket,
} from "./m2-training-contract.mjs";

const ROOT = "benchmark/evidence/m2";

function git(arguments_, options = {}) {
  return execFileSync("git", arguments_, {
    encoding: "utf8",
    maxBuffer: options.maxBuffer ?? 128 * 1024 * 1024,
  }).trim();
}

function gitBytes(arguments_) {
  return execFileSync("git", arguments_, { encoding: null, maxBuffer: 128 * 1024 * 1024 });
}

function rowsForCommit(commit) {
  const output = git(["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", commit]);
  return output.split("\n").filter(Boolean).map((line) => {
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
  "M2 final verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
const checkerRecoveryCommit = git(["rev-parse", "HEAD^"]);
requireCondition(git(["rev-list", "--parents", "-n", "1", head]).split(" ").length === 2 &&
  git(["rev-list", "--parents", "-n", "1", checkerRecoveryCommit]).split(" ").length === 2 &&
  git(["rev-list", "--parents", "-n", "1", M2_FINALIZER_SOURCE_COMMIT]).split(" ").length === 2,
"M2 source, checker recovery, and publication commits must each have one direct parent");
requireCondition(git(["rev-parse", `${M2_FINALIZER_SOURCE_COMMIT}^`]) === M2_RECOVERY_COMMIT,
  "M2 finalizer source is not the repaired M2 source commit's direct child");
requireCondition(matchesExpectedRows(rowsForCommit(M2_FINALIZER_SOURCE_COMMIT), M2_FINALIZER_SOURCE_EXPECTED),
  "M2 finalizer source changed outside its frozen source-only packet");
requireCondition(git(["rev-parse", `${checkerRecoveryCommit}^`]) === M2_FINALIZER_SOURCE_COMMIT &&
  matchesExpectedRows(rowsForCommit(checkerRecoveryCommit), M2_CHECKER_RECOVERY_EXPECTED),
"M2 checker recovery changed outside its direct six-path packet");
requireCondition(matchesExpectedRows(rowsForCommit(head), M2_PUBLICATION_EXPECTED),
  "M2 publication changed outside its exact eight-path packet");
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");

const paths = {
  recipe: "benchmark/m2/recipe.json",
  selectionSummary: `${ROOT}/selection-summary.json`,
  trainingSummary: `${ROOT}/training-summary.json`,
  calibration: `${ROOT}/calibration.json`,
  candidateGrid: `${ROOT}/candidate-grid.json`,
  comparison: `${ROOT}/model-comparison.json`,
  receipt: `${ROOT}/finalization-receipt.json`,
  model: "weights/prooflens-cf384.onnx",
  modelLock: "model-lock.json",
  weightsReadme: "weights/README.md",
  upstreamStructure: "benchmark/evidence/large/upstream-model-structure.json",
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
const modelLock = parseJson(entries.modelLock, paths.modelLock);
const upstreamStructure = parseJson(entries.upstreamStructure, paths.upstreamStructure);

validateM2TrainingPacket({
  summary: trainingSummary,
  calibration,
  grid: candidateGrid,
  recipe,
  selectionSummary,
  fileHashes: {
    recipe: digest(entries.recipe),
    selectionSummary: digest(entries.selectionSummary),
    trainingSummary: digest(entries.trainingSummary),
    calibration: digest(entries.calibration),
    candidateGrid: digest(entries.candidateGrid),
  },
  model: { sha256: digest(entries.model), bytes: entries.model.length },
});

const historicalModelLock = parseJson(
  gitBytes(["show", `${M2_RECOVERY_COMMIT}:model-lock.json`]),
  `${M2_RECOVERY_COMMIT}:model-lock.json`,
);
for (const name of ["artifact", "format", "input", "output", "upstream"]) {
  requireCondition(jsonEqual(modelLock[name], historicalModelLock[name]),
    `M2 model lock changed the immutable ${name} interface`);
}
requireCondition(jsonEqual(modelLock.calibration, {
  slope: calibration.slope,
  intercept: calibration.intercept,
  displayThreshold: calibration.displayThreshold,
  validationThresholdLogit: calibration.validationThresholdLogit,
}), "M2 model lock calibration does not match the published calibration");
const repositoryHashes = {
  "weights/prooflens-cf384.onnx": digest(entries.model),
  "model-lock.json": digest(entries.modelLock),
  "weights/README.md": digest(entries.weightsReadme),
};
validateM2PublicationMetadata({
  modelLock,
  weightsReadme: entries.weightsReadme.toString("utf8"),
  receipt,
  comparison,
  repositoryHashes,
});
requireCondition(digest(entries.comparison) === M2.modelComparisonSha256,
  "M2 model-comparison bytes changed");

requireCondition(upstreamStructure.schemaVersion === 1 &&
  upstreamStructure.model?.sha256 === M2.upstreamSha256 && upstreamStructure.model.bytes === M2.modelBytes &&
  Array.isArray(upstreamStructure.initializers) && upstreamStructure.initializers.length === 200,
"Pinned upstream ONNX structure changed");
const shippedStructure = inspectOnnxStructure(entries.model);
validateM2OnnxEvidence({ upstreamStructure, shippedStructure, comparison });

console.log(JSON.stringify({
  stage: "m2-final",
  head,
  finalizerSourceCommit: M2_FINALIZER_SOURCE_COMMIT,
  checkerRecoveryCommit,
  modelSha256: M2.modelSha256,
  trainingSummarySha256: M2.trainingSummarySha256,
  calibrationSha256: M2.calibrationSha256,
  candidateGridSha256: M2.candidateGridSha256,
  modelComparisonSha256: M2.modelComparisonSha256,
  classifierOnly: true,
  policy: "pass",
}));
