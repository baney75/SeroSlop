import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { inspectOnnxStructure } from "./onnx-structure.mjs";
import { reconstructM3CandidateModel } from "./m3-candidate-patch.mjs";
import {
  M3_BASE_COMMIT,
  M3_LOCK_EXPECTED,
  M3_PUBLICATION_EXPECTED,
  M3_PUBLICATION_LOCK_PATH,
  M3_SOURCE_EXPECTED,
  matchesExpectedRows,
} from "./m3-stage-policy.mjs";
import {
  jsonEqual,
  parseCanonicalM3PublicationLock,
  validateM3BrowserFixtureManifest,
  validateM3OnnxEvidence,
  validateM3TrainingPacket,
} from "./m3-training-contract.mjs";
import { renderM3PublicDocuments } from "./render-m3-public-docs.mjs";


const M3_BASE_TREE = "32aa561cfb12ccda59aee919d3ca2b3761b07d9c";
const M2_MODEL_SHA256 = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47";
const HEX64 = /^[a-f0-9]{64}$/;
const FINAL_OUTPUTS = [
  "benchmark/evidence/m3/calibration.json",
  "benchmark/evidence/m3/candidate-grid.json",
  "benchmark/evidence/m3/finalization-receipt.json",
  "benchmark/evidence/m3/model-comparison.json",
  "benchmark/evidence/m3/training-summary.json",
];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function git(arguments_, options = {}) {
  return execFileSync("git", arguments_, {
    encoding: options.encoding ?? "utf8",
    maxBuffer: options.maxBuffer ?? 128 * 1024 * 1024,
  }).trim();
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function parseJson(value, label) {
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`${label} is not valid JSON`, { cause: error });
  }
}

function commitRows(commit) {
  return git(["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, pathname] = line.split("\t");
      return [pathname, status];
    });
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M3 publication-lock verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
const sourceCommit = git(["rev-parse", "HEAD^"]);
requireCondition(git(["rev-list", "--parents", "-n", "1", head]).split(" ").length === 2 &&
  git(["rev-list", "--parents", "-n", "1", sourceCommit]).split(" ").length === 2,
"M3 source and lock commits must each have one parent");
requireCondition(git(["rev-parse", `${sourceCommit}^`]) === M3_BASE_COMMIT &&
  git(["rev-parse", `${M3_BASE_COMMIT}^{tree}`]) === M3_BASE_TREE &&
  matchesExpectedRows(commitRows(sourceCommit), M3_SOURCE_EXPECTED),
"M3 source commit changed outside its exact source-only packet");
requireCondition(matchesExpectedRows(commitRows(head), M3_LOCK_EXPECTED),
  "M3 output-lock commit must add exactly one lock file");
requireCondition(existsSync(M3_PUBLICATION_LOCK_PATH), "M3 publication lock is missing");
for (const pathname of FINAL_OUTPUTS) {
  requireCondition(!existsSync(pathname), `M3 output lock must precede final publication: ${pathname}`);
}
const lockBytes = readFileSync(M3_PUBLICATION_LOCK_PATH);
const lock = parseCanonicalM3PublicationLock(lockBytes);
const sourceTree = git(["rev-parse", `${sourceCommit}^{tree}`]);
requireCondition(lock?.schemaVersion === 1 && lock.profile === "m3" &&
  lock.sourceCommit === sourceCommit && lock.sourceTree === sourceTree &&
  lock.upstreamModelSha256 === M2_MODEL_SHA256 &&
  lock.trainerSha256 === digest(readFileSync("benchmark/modern/train_rehead.py")) &&
  lock.recipeSha256 === digest(readFileSync("benchmark/m3/recipe.json")) &&
  lock.selectionSummarySha256 === digest(readFileSync("benchmark/evidence/m3/selection-summary.json")) &&
  lock.finalizerSha256 === digest(readFileSync("benchmark/m3/finalize.py")) &&
  lock.publicationContractSha256 === digest(readFileSync("benchmark/m3/publication_contract.py")) &&
  lock.fixtureSelectorSha256 === digest(readFileSync("benchmark/m3/select_model_state_fixtures.py")) &&
  lock.documentationRendererSha256 === digest(readFileSync("scripts/render-m3-public-docs.mjs")) &&
  lock.selectionInfluencedByRegression === false && lock.h3HoldoutScored === false,
"M3 publication lock does not bind the frozen source commit");
requireCondition(lock.candidateHashes && !Array.isArray(lock.candidateHashes) &&
  JSON.stringify(Object.keys(lock.candidateHashes)) ===
    JSON.stringify(["training-summary.json", "calibration.json", "candidate-grid.json", "model.onnx"]) &&
  Object.values(lock.candidateHashes).every((value) => HEX64.test(value)) &&
  HEX64.test(lock.modelComparisonSha256 ?? "") && /^[a-f0-9]{32}$/.test(lock.freshRunId ?? "") &&
  Number.isInteger(lock.candidateModelBytes) && lock.candidateModelBytes > 0 &&
  lock.candidateModelBytes < 100 * 1024 * 1024,
"M3 publication lock candidate bindings are malformed");
requireCondition(lock.publicDocumentHashes &&
  JSON.stringify(Object.keys(lock.publicDocumentHashes).sort()) ===
    JSON.stringify(["BENCHMARK.md", "MODEL_CARD.md", "README.md"]) &&
  Object.values(lock.publicDocumentHashes).every((value) => HEX64.test(value)) &&
  HEX64.test(lock.fixtureManifestSha256 ?? ""),
"M3 publication lock document or fixture hashes are malformed");
const evidenceNames = [
  "training-summary.json",
  "calibration.json",
  "candidate-grid.json",
  "model-comparison.json",
  "fixture-manifest.json",
];
requireCondition(lock.candidateEvidenceJson && !Array.isArray(lock.candidateEvidenceJson) &&
  jsonEqual(Object.keys(lock.candidateEvidenceJson), evidenceNames) &&
  evidenceNames.every((name) => typeof lock.candidateEvidenceJson[name] === "string"),
"M3 publication lock lacks its exact candidate evidence bytes");
for (const name of ["training-summary.json", "calibration.json", "candidate-grid.json"]) {
  requireCondition(digest(Buffer.from(lock.candidateEvidenceJson[name])) === lock.candidateHashes[name],
    `M3 publication lock ${name} bytes do not match their digest`);
}
requireCondition(digest(Buffer.from(lock.candidateEvidenceJson["model-comparison.json"])) ===
  lock.modelComparisonSha256 && digest(Buffer.from(lock.candidateEvidenceJson["fixture-manifest.json"])) ===
  lock.fixtureManifestSha256, "M3 publication lock derived evidence bytes do not match their digests");

const baseModelBytes = readFileSync("weights/prooflens-cf384.onnx");
const candidateModelBytes = reconstructM3CandidateModel({ baseBytes: baseModelBytes, patch: lock.classifierPatch });
const model = { sha256: digest(candidateModelBytes), bytes: candidateModelBytes.length };
requireCondition(model.sha256 === lock.candidateHashes["model.onnx"] && model.bytes === lock.candidateModelBytes,
  "M3 publication lock classifier patch does not reconstruct the candidate model");
const summary = parseJson(lock.candidateEvidenceJson["training-summary.json"], "M3 training summary");
const calibration = parseJson(lock.candidateEvidenceJson["calibration.json"], "M3 calibration");
const grid = parseJson(lock.candidateEvidenceJson["candidate-grid.json"], "M3 candidate grid");
const comparison = parseJson(lock.candidateEvidenceJson["model-comparison.json"], "M3 model comparison");
const fixture = parseJson(lock.candidateEvidenceJson["fixture-manifest.json"], "M3 browser fixture manifest");
const recipeBytes = readFileSync("benchmark/m3/recipe.json");
const selectionBytes = readFileSync("benchmark/evidence/m3/selection-summary.json");
const validationBytes = readFileSync("benchmark/evidence/m3/validation-manifest.jsonl");
const regressionBytes = readFileSync("benchmark/evidence/m2/validation-manifest.jsonl");
validateM3TrainingPacket({
  summary,
  calibration,
  grid,
  recipe: parseJson(recipeBytes, "M3 recipe"),
  selectionSummary: parseJson(selectionBytes, "M3 selection summary"),
  hashes: {
    trainer: digest(readFileSync("benchmark/modern/train_rehead.py")),
    recipe: digest(recipeBytes),
    selectionSummary: digest(selectionBytes),
    validationManifest: digest(validationBytes),
    regressionManifest: digest(regressionBytes),
    model: model.sha256,
  },
  model,
});
validateM3OnnxEvidence({
  baseStructure: inspectOnnxStructure(baseModelBytes),
  shippedStructure: inspectOnnxStructure(candidateModelBytes),
  comparison,
  model,
});
const rendered = renderM3PublicDocuments({
  readme: readFileSync("README.md", "utf8"),
  modelCard: readFileSync("MODEL_CARD.md", "utf8"),
  benchmark: readFileSync("BENCHMARK.md", "utf8"),
  summary,
  modelSha256: model.sha256,
  modelBytes: model.bytes,
});
requireCondition(jsonEqual(lock.publicDocumentHashes, {
  "README.md": digest(Buffer.from(rendered.README)),
  "MODEL_CARD.md": digest(Buffer.from(rendered.MODEL_CARD)),
  "BENCHMARK.md": digest(Buffer.from(rendered.BENCHMARK)),
}), "M3 publication lock does not bind deterministic complete public documents");
validateM3BrowserFixtureManifest({
  manifest: fixture,
  manifestSha256: lock.fixtureManifestSha256,
  assets: {
    "likely-ai.png": digest(readFileSync("tests/fixtures/model-states/likely-ai.png")),
    "below-threshold.jpg": digest(readFileSync("tests/fixtures/model-states/below-threshold.jpg")),
  },
  calibration: { sha256: lock.candidateHashes["calibration.json"], value: calibration },
  model,
  summarySha256: lock.candidateHashes["training-summary.json"],
});
requireCondition(fixture.selectorSha256 === lock.fixtureSelectorSha256,
  "M3 publication lock browser fixture does not bind the frozen selector");
requireCondition(JSON.stringify(lock.publicationRows) === JSON.stringify(
  [...M3_PUBLICATION_EXPECTED].map(([pathname, status]) => ({ path: pathname, status }))),
"M3 publication lock path surface changed");
requireCondition(digest(baseModelBytes) === M2_MODEL_SHA256,
  "M3 lock stage must retain the reviewed M2 model bytes");
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");
requireCondition(git(["ls-files", "benchmark/data/m3-head", "benchmark/data/m3-source", "benchmark/data/h3-met-holdout-v1"]) === "",
  "M3 source pixels must remain outside Git");

console.log(JSON.stringify({
  stage: "m3-pinned",
  head,
  sourceCommit,
  sourceTree,
  publicationLockSha256: digest(lockBytes),
  candidateModelSha256: lock.candidateHashes["model.onnx"],
  publicationPaths: lock.publicationRows.length,
  h3HoldoutScored: false,
  policy: "pass",
}));
