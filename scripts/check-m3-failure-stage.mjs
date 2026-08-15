import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import {
  M3_BASE_COMMIT,
  M3_FAILURE_EXPECTED,
  M3_SOURCE_COMMIT,
  M3_SOURCE_EXPECTED,
  matchesExpectedRows,
} from "./m3-stage-policy.mjs";
import {
  validateM3FailureDiagnostic,
  validateM3FailureReceipt,
} from "./m3-failure-contract.mjs";

const SOURCE_TREE = "a6ab771f27b1efd108caa0b08128118fd7465334";
const MODEL_SHA = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47";
const RUN_ID = "447053b4b6f924488653d237e2372230";
const DIAGNOSTIC_SHA = "1238ae1cf341ce257c2e3e9f5e44cd499ac62b3a37e47010d3522e8f2f018eb5";
const RECEIPT_SHA = "bbdaea92db8628db880000bf0682b177d78bc8b77e6e8d800f8754dbf4a47ba3";
const GENERATOR_SHA = "429ba93e675f2ebf0b42fca4d50f3916021fc5abcb4cc6761be42afb5ec37746";
const CANDIDATE_OUTPUTS = [
  "benchmark/candidates/prooflens-cf384-m3/model.onnx",
  "benchmark/candidates/prooflens-cf384-m3/calibration.json",
  "benchmark/candidates/prooflens-cf384-m3/candidate-grid.json",
  "benchmark/candidates/prooflens-cf384-m3/validation-summary.json",
];
const PUBLICATION_OUTPUTS = [
  "benchmark/evidence/m3/publication-lock.json",
  "benchmark/evidence/m3/calibration.json",
  "benchmark/evidence/m3/candidate-grid.json",
  "benchmark/evidence/m3/finalization-receipt.json",
  "benchmark/evidence/m3/model-comparison.json",
  "benchmark/evidence/m3/training-summary.json",
];
const COMMAND_ARGUMENTS = [
  "--model", "weights/prooflens-cf384.onnx",
  "--expected-model-sha256", MODEL_SHA,
  "--data-root", "benchmark/data/m3-head",
  "--train-manifest", "benchmark/data/m3-head/train-manifest.jsonl",
  "--validation-data-root", "benchmark/data/m3-head",
  "--validation-manifest", "benchmark/evidence/m3/validation-manifest.jsonl",
  "--regression-data-root", "benchmark/data/m2-head",
  "--regression-manifest", "benchmark/evidence/m2/validation-manifest.jsonl",
  "--recipe", "benchmark/m3/recipe.json",
  "--selection-summary", "benchmark/evidence/m3/selection-summary.json",
  "--single-view-source", "diffusiondb-stable-diffusion",
  "--single-view-source", "open-images-train",
  "--execution-provider", "cpu",
  "--batch-size", "24",
  "--feature-shard-images", "2000",
  "--reextract-cached-features",
  "--output-dir", "benchmark/candidates/prooflens-cf384-m3",
];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }).trim();
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function commitRows(commit) {
  return git(["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, pathname] = line.split("\t");
      return [pathname, status];
    });
}

async function localCacheInventory(root) {
  const output = [];
  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      const pathname = path.join(directory, entry.name);
      const relative = pathname.split(path.sep).join("/");
      requireCondition(!entry.isSymbolicLink(), `M3 local cache contains a symlink: ${relative}`);
      if (entry.isDirectory()) await visit(pathname);
      else if (entry.isFile()) {
        const bytes = await readFile(pathname);
        output.push({ path: relative, bytes: bytes.length, sha256: digest(bytes) });
      }
    }
  }
  await visit(root);
  return output.sort((left, right) => left.path < right.path ? -1 : left.path > right.path ? 1 : 0);
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M3 failed-stage verification requires a completely clean tracked repository");
const head = git(["rev-parse", "HEAD"]);
const parents = git(["show", "-s", "--format=%P", head]).split(" ").filter(Boolean);
requireCondition(parents.length === 1 && parents[0] === M3_SOURCE_COMMIT,
  "The M3 failure record must be the frozen source commit's direct single-parent child");
requireCondition(git(["rev-parse", `${M3_SOURCE_COMMIT}^`]) === M3_BASE_COMMIT &&
  git(["rev-parse", `${M3_SOURCE_COMMIT}^{tree}`]) === SOURCE_TREE &&
  matchesExpectedRows(commitRows(M3_SOURCE_COMMIT), M3_SOURCE_EXPECTED),
"The frozen M3 source lineage changed");
requireCondition(matchesExpectedRows(commitRows(head), M3_FAILURE_EXPECTED),
  "The M3 failure commit changed outside its exact append-only packet");

const paths = {
  diagnostic: "benchmark/evidence/m3/failed-selector-diagnostic-1.json",
  receipt: "benchmark/evidence/m3/failed-training-attempt-1.json",
  generator: "benchmark/m3/diagnose_failed_training.py",
  trainer: "benchmark/modern/train_rehead.py",
  recipe: "benchmark/m3/recipe.json",
  selectionSummary: "benchmark/evidence/m3/selection-summary.json",
  selectorManifest: "benchmark/evidence/m3/validation-manifest.jsonl",
  regressionManifest: "benchmark/evidence/m2/validation-manifest.jsonl",
  model: "weights/prooflens-cf384.onnx",
};
const entries = Object.fromEntries(await Promise.all(Object.entries(paths).map(async ([name, pathname]) =>
  [name, await readFile(pathname)])));
requireCondition(digest(entries.diagnostic) === DIAGNOSTIC_SHA && digest(entries.receipt) === RECEIPT_SHA &&
  digest(entries.generator) === GENERATOR_SHA && digest(entries.model) === MODEL_SHA,
"M3 failure evidence, generator, or retained model bytes changed");
const expected = {
  sourceCommit: M3_SOURCE_COMMIT,
  sourceTree: SOURCE_TREE,
  baseCommit: M3_BASE_COMMIT,
  inputBindings: {
    upstreamModelSha256: MODEL_SHA,
    trainerSha256: digest(entries.trainer),
    diagnosticGeneratorSha256: GENERATOR_SHA,
    recipeSha256: digest(entries.recipe),
    selectionSummarySha256: digest(entries.selectionSummary),
    trainManifestSha256: "a0cb9994018b44fa958e312816460b15b5fbab43d489e75b1a9f5647bd70d261",
    selectorManifestSha256: digest(entries.selectorManifest),
    regressionManifestSha256: digest(entries.regressionManifest),
    runId: RUN_ID,
  },
  commandArguments: COMMAND_ARGUMENTS,
  cacheBytes: 286_698_318,
  markerSha256: "4560e455d813e371856093566651ac6cc96769f761f746d210eb026d6329b916",
  candidateOutputs: CANDIDATE_OUTPUTS,
  publicationOutputs: PUBLICATION_OUTPUTS,
};
const diagnostic = validateM3FailureDiagnostic({
  bytes: entries.diagnostic,
  selectorManifestBytes: entries.selectorManifest,
  recipeBytes: entries.recipe,
  expected,
});
const receipt = validateM3FailureReceipt({ bytes: entries.receipt, diagnosticBytes: entries.diagnostic, expected });

for (const pathname of [...CANDIDATE_OUTPUTS, ...PUBLICATION_OUTPUTS]) {
  requireCondition(!existsSync(pathname), `M3 failure stage cannot contain a successful output: ${pathname}`);
}
if (existsSync("benchmark/candidates/prooflens-cf384-m3")) {
  const actual = await localCacheInventory("benchmark/candidates/prooflens-cf384-m3");
  requireCondition(JSON.stringify(actual) === JSON.stringify(receipt.cacheSnapshot.inventory),
    "Remaining local M3 cache differs from the committed failure snapshot");
}
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");
requireCondition(git(["ls-files", "benchmark/candidates", "benchmark/data/m3-head",
  "benchmark/data/m3-source", "benchmark/data/h3-met-holdout-v1"]) === "",
"M3 cache, source pixels, or H3 pixels entered Git");

console.log(JSON.stringify({
  stage: "m3-failed",
  head,
  parent: M3_SOURCE_COMMIT,
  paths: M3_FAILURE_EXPECTED.size,
  candidateCount: diagnostic.aggregate.candidateCount,
  feasibleCandidateCount: diagnostic.aggregate.feasibleCandidateCount,
  modelSha256: MODEL_SHA,
  cachePresent: existsSync("benchmark/candidates/prooflens-cf384-m3"),
  h3AcceptedAsInput: false,
  policy: "pass",
}));
