import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";

import { M4 } from "./m4-training-contract.mjs";


const ROOT = "benchmark/evidence/m4";
const REQUIRED = [
  "attribution.json",
  "british-source-index.json.gz",
  "perceptual-review.json",
  "rapidata-source-index.json.gz",
  "rejects.jsonl.gz",
  "selection-summary.json",
  "train-manifest.jsonl.gz",
  "validation-manifest.jsonl",
];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

for (const name of REQUIRED) {
  requireCondition(existsSync(`${ROOT}/${name}`), `M4 public source artifact is missing: ${name}`);
}
requireCondition(digest(readFileSync("benchmark/m4/recipe.json")) === M4.recipeSha256,
  "M4 recipe bytes changed");
requireCondition(digest(readFileSync("benchmark/m4/source-locks.json")) === M4.sourceLocksSha256,
  "M4 source-lock bytes changed");

const output = execFileSync("python3", ["benchmark/m4/verify.py", "--public-only"], {
  encoding: "utf8",
  maxBuffer: 256 * 1024 * 1024,
}).trim();
let receipt;
try {
  receipt = JSON.parse(output);
} catch (error) {
  throw new Error("M4 public source verifier did not return JSON", { cause: error });
}
requireCondition(receipt.policy === "pass" && receipt.isolatedPublicRederivation === "pass" &&
  receipt.trainingItems === M4.trainImages && receipt.selectorItems === M4.selectorImages &&
  receipt.h3PixelsRead === false, "M4 public source replay changed");

const trackedPrivate = execFileSync("git", ["ls-files", "benchmark/data/m4-head",
  "benchmark/data/m4-source", "benchmark/candidates/prooflens-cf384-m4",
  "benchmark/data/h3-met-holdout-v1"], { encoding: "utf8" }).trim();
requireCondition(trackedPrivate === "", "M4 source, candidate, or H3 pixels entered Git");
requireCondition(!existsSync("docs/COMPETITOR_AUDIT.md"), "Competitor audit must remain absent");

console.log(JSON.stringify({
  stage: "m4-source-evidence",
  artifacts: REQUIRED.length,
  trainingItems: receipt.trainingItems,
  selectorItems: receipt.selectorItems,
  isolatedPublicRederivation: receipt.isolatedPublicRederivation,
  h3PixelsRead: false,
  policy: "pass",
}));
