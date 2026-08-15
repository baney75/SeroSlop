import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";

const F2_COMMIT = "163a4c8e0e56888b506be6ab3f2ed3f6d888f45b";
const F2_FAILURE_RECEIPT = "benchmark/evidence/evaluation/web-negative-v2/failed-evaluation.json";
const F2_FAILURE_SHA256 = "614d09b3ecb97e83be76d19b8919f91d499e00f91bb5410c4c866dd753c3494b";
const EXPECTED = new Map([
  ["benchmark/evidence/m2/attribution.json", "A"],
  ["benchmark/evidence/m2/perceptual-review.json", "A"],
  ["benchmark/evidence/m2/rejects.jsonl.gz", "A"],
  ["benchmark/evidence/m2/selection-summary.json", "A"],
  ["benchmark/evidence/m2/stock-selection.json.gz", "A"],
  ["benchmark/evidence/m2/train-manifest.jsonl.gz", "A"],
  ["benchmark/evidence/m2/validation-manifest.jsonl", "A"],
  ["benchmark/m2/README.md", "A"],
  ["benchmark/m2/contracts.py", "A"],
  ["benchmark/m2/prepare.py", "A"],
  ["benchmark/m2/recipe.json", "A"],
  ["benchmark/m2/test_prepare.py", "A"],
  ["benchmark/m2/test_trainer_contracts.py", "A"],
  ["benchmark/m2/verify.py", "A"],
  ["benchmark/modern/train_rehead.py", "M"],
  ["package.json", "M"],
  ["scripts/check-large-training-evidence.mjs", "M"],
  ["scripts/check-m2-selection-evidence.mjs", "A"],
  ["scripts/check-m2-source-stage.mjs", "A"],
  ["scripts/run-static-verification.mjs", "M"],
]);

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8" }).trim();
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
  "M2 source verification requires a completely clean repository");
const head = git(["rev-parse", "HEAD"]);
requireCondition(git(["rev-parse", "HEAD^"]) === F2_COMMIT,
  "The M2 source freeze must be F2's direct child");
const rows = git(["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", head])
  .split("\n").filter(Boolean).map((line) => {
    const [status, path] = line.split("\t");
    return [path, status];
  });
requireCondition(rows.length === EXPECTED.size && rows.every(([path, status]) => EXPECTED.get(path) === status),
  "The M2 source freeze changed an unexpected path or status");
requireCondition(digest(readFileSync(F2_FAILURE_RECEIPT)) === F2_FAILURE_SHA256,
  "The consumed v2 failure receipt changed");
for (const path of [
  "benchmark/evidence/m2/training-summary.json",
  "benchmark/evidence/m2/calibration.json",
  "benchmark/evidence/m2/candidate-grid.json",
  "benchmark/evidence/m2/model-comparison.json",
  "benchmark/evidence/m2/finalization-receipt.json",
]) requireCondition(!existsSync(path), `M2 source freeze must precede training output: ${path}`);
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");

console.log(JSON.stringify({ head, parent: F2_COMMIT, changedPaths: rows.length, stage: "m2-source", policy: "pass" }));
