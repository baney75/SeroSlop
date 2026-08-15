import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { inspectOnnxStructure } from "./onnx-structure.mjs";
import {
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
const sourceCommit = git(["rev-parse", "HEAD^"]);
requireCondition(git(["rev-list", "--parents", "-n", "1", head]).split(" ").length === 2 &&
  git(["rev-list", "--parents", "-n", "1", sourceCommit]).split(" ").length === 2,
"M2 source and publication commits must each have one direct parent");
requireCondition(git(["rev-parse", `${sourceCommit}^`]) === M2_RECOVERY_COMMIT,
  "M2 finalizer source is not the repaired M2 source commit's direct child");
requireCondition(matchesExpectedRows(rowsForCommit(sourceCommit), M2_FINALIZER_SOURCE_EXPECTED),
  "M2 finalizer source changed outside its frozen source-only packet");
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
for (const name of ["graphNodesSha256", "graphInputsSha256", "graphOutputsSha256", "opsetsSha256"]) {
  requireCondition(shippedStructure[name] === upstreamStructure[name] && comparison[name] === shippedStructure[name],
    `M2 shipped ONNX ${name} changed`);
}
const upstreamByName = new Map(upstreamStructure.initializers.map((row) => [row.name, row]));
requireCondition(upstreamByName.size === 200 && shippedStructure.initializers.length === 200,
  "M2 shipped ONNX initializer count changed");
const independentlyChanged = [];
for (const initializer of shippedStructure.initializers) {
  const upstream = upstreamByName.get(initializer.name);
  requireCondition(upstream && jsonEqual(initializer.dimensions, upstream.dimensions),
    `M2 initializer shape changed: ${initializer.name}`);
  if (initializer.sha256 !== upstream.sha256) independentlyChanged.push({
    name: initializer.name,
    dimensions: initializer.dimensions,
    beforeSha256: upstream.sha256,
    afterSha256: initializer.sha256,
  });
}
requireCondition(jsonEqual(independentlyChanged.map((row) => row.name).sort(),
  ["classifier.bias", "classifier.weight"]), "M2 changed initializers outside the classifier head");
const comparisonByName = new Map(comparison.changedInitializers.map((row) => [row.name, row]));
for (const row of independentlyChanged) {
  const recorded = comparisonByName.get(row.name);
  requireCondition(recorded && jsonEqual(recorded.dimensions, row.dimensions) &&
    recorded.beforeSha256 === row.beforeSha256 && recorded.afterSha256 === row.afterSha256,
  `M2 comparison does not match independent ONNX evidence: ${row.name}`);
}

console.log(JSON.stringify({
  stage: "m2-final",
  head,
  sourceCommit,
  modelSha256: M2.modelSha256,
  trainingSummarySha256: M2.trainingSummarySha256,
  calibrationSha256: M2.calibrationSha256,
  candidateGridSha256: M2.candidateGridSha256,
  modelComparisonSha256: M2.modelComparisonSha256,
  classifierOnly: true,
  policy: "pass",
}));
