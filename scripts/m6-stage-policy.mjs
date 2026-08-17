import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

export const M6_BASE_COMMIT = "76d0a807dcf240245830b8510e623d838e43cd4c";
export const M6_BASE_TREE = "5c10012b9520e3936efc86e08ff0a53adecec868";
export const M6_P_COMMIT = "3b29ea2f9e1ad46e4cd78f47c9ccf5fe3a99877e";
export const M6_P_TREE = "dfd29cd86f4f746d403b14994055a575d82f83c4";
export const M6_P2_COMMIT = "0777710c89cd0fa02e2f4bd063ec51664e3fc26a";
export const M6_P2_TREE = "30d8338382033caacd400e1b29c37ed287f9de43";
export const M6_CENSUS_SHA256 = "61f494f09fe256d771bacb809712b5c645e5b25f63cffca52dfc40d0e0ac7adf";
export const M6_P_RECIPE_SHA256 = "56bfe2487760c833c796289e3d4c5e8ef0eb65e62493229f9d62631a573ab613";
export const M6_P2_RECIPE_SHA256 = "42f594fd26ac4949f191eb5c773c977ec8e5bee586f766c9a648afce85bc2984";
export const M6_RECIPE_SHA256 = "a1c1700acbfbed19ef73e3cc4224c994eadef17e81ddb8d6d8040c8a3d5a5e88";
export const M6_SOURCE_SHARDS_PATH = "benchmark/m6/source-shards.json";
export const M6_SOURCE_SHARDS_SHA256 = "a86c7209e76248edddd61537f397379194a7aaa908405e0cede7c8f5a3d7fbfe";
export const M6_RECIPE_PATH = "benchmark/m6/recipe.json";
export const M6_CENSUS_PATH = "benchmark/m6/census-evidence.json";
export const M6_PROTOCOL_PATHS = Object.freeze([
  "benchmark/m6/README.md", "benchmark/m6/THIRD_PARTY_NOTICES.md", "benchmark/m6/__init__.py", "benchmark/m6/census-evidence.json", "benchmark/m6/contracts.py", "benchmark/m6/preflight.py", "benchmark/m6/prepare.py", "benchmark/m6/recipe.json", "benchmark/m6/test_contracts.py",
  "package.json", "scripts/check-m6-protocol-stage.mjs", "scripts/m6-stage-policy.mjs", "scripts/run-static-verification.mjs", "scripts/test-m6-stage-policy.mjs",
]);
export const M6_PROTOCOL_RECOVERY_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"],
  ["benchmark/m6/contracts.py", "M"],
  ["benchmark/m6/prepare.py", "M"],
  ["benchmark/m6/recipe.json", "M"],
  ["benchmark/m6/test_contracts.py", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_MATERIALIZER_RECOVERY_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"],
  ["benchmark/m6/contracts.py", "M"],
  ["benchmark/m6/historical.py", "A"],
  ["benchmark/m6/materialize.py", "A"],
  ["benchmark/m6/prepare.py", "M"],
  ["benchmark/m6/recipe.json", "M"],
  ["benchmark/m6/source-shards.json", "A"],
  ["benchmark/m6/test_contracts.py", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_STAGES = Object.freeze(["m6-protocol", "m6-source-lock", "m6-preflight", "m6-trained", "m6-evaluated"]);

function rejectDuplicateKeys(text) {
  const stack = []; let inString = false; let escaped = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (inString) { if (escaped) escaped = false; else if (c === "\\") escaped = true; else if (c === '"') inString = false; continue; }
    if (c === '"') {
      let j = i + 1; let e = false;
      for (; j < text.length; j += 1) { if (e) { e = false; continue; } if (text[j] === "\\") { e = true; continue; } if (text[j] === '"') break; }
      let k = j + 1; while (/\s/.test(text[k] ?? "")) k += 1;
      if (text[k] === ":" && stack.length) { const key = JSON.parse(text.slice(i, j + 1)); if (stack.at(-1).has(key)) throw new Error(`duplicate M6 key: ${key}`); stack.at(-1).add(key); }
      i = j; continue;
    }
    if (c === "{") stack.push(new Set()); else if (c === "}") stack.pop();
  }
}

export function parseM6Recipe(bytes = readFileSync(M6_RECIPE_PATH)) {
  const text = Buffer.from(bytes).toString("utf8");
  if (Buffer.from(text, "utf8").compare(Buffer.from(bytes)) !== 0) throw new Error("M6 recipe must be strict UTF-8");
  const value = JSON.parse(text);
  rejectDuplicateKeys(text);
  if (createHash("sha256").update(bytes).digest("hex") !== M6_RECIPE_SHA256) throw new Error("M6 recipe digest changed");
  if (value.baseCommit !== M6_BASE_COMMIT) throw new Error("M6 base commit changed");
  if (value.evaluation?.items !== 100000 || value.evaluation?.batches !== 1000) throw new Error("M6 evaluation precommit changed");
  if (value.selector?.source !== "Omni-Fake-SET:image/validation") throw new Error("M6 selector source changed");
  if (value.selector?.generators?.length !== 34 || value.selector.generators.includes("Flux.1_pro")) throw new Error("M6 selector generator census changed");
  const allowed = new Map([["deliverable", new Set(["format","input","output","maximumBytes","browserExecution","networkAfterInstall"])],["sources", new Set(["omniFakeSet","omniFakeOOD"])],["selector", new Set(["source","baseItems","real","synthetic","syntheticSelection","generators","views","zeroObservedFalsePositive","wilsonConfidence","wilsonUpperBoundAtZero","poolViews","thresholdSearch","gates"])],["evaluation", new Set(["items","batches","batchSize","synthetic","assignedBeforeSelectorScoring","itemDisjoint","selectionInfluence","selectionExcludesSelector","strictMeanRecallGreaterThan","strictMedianBatchRecallGreaterThan","failureConsumesPanel"])]]);
  for (const [name, keys] of allowed) if (value[name] && Object.keys(value[name]).some((key) => !keys.has(key))) throw new Error(`unknown M6 ${name} key`);
  return value;
}

export function classifyM6Stage({ protocol = true, sourceLock = false, preflight = false, trained = false, evaluated = false } = {}) {
  if (sourceLock || preflight || trained || evaluated) throw new Error("M6 progressed stages require executable Git/artifact checker; booleans are not trusted");
  if (evaluated && !trained) throw new Error("evaluation requires training");
  if (trained && !preflight) throw new Error("training requires preflight");
  if (preflight && !sourceLock) throw new Error("preflight requires source lock");
  if (sourceLock && !protocol) throw new Error("source lock requires protocol");
  if (evaluated) return "m6-evaluated";
  if (trained) return "m6-trained";
  if (preflight) return "m6-preflight";
  if (sourceLock) return "m6-source-lock";
  return protocol ? "m6-protocol" : null;
}

export function matchesProspectiveP({ head, parents = [], paths = [], statuses = {} } = {}) {
  const required = [...M6_PROTOCOL_PATHS].sort(); const actual = [...new Set(paths)].sort();
  const expectedStatuses = Object.fromEntries(required.map((path) => [path, ["package.json", "scripts/run-static-verification.mjs"].includes(path) ? "M" : "A"]));
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_BASE_COMMIT && parents.length === 1 && parents[0] === M6_BASE_COMMIT && JSON.stringify(actual) === JSON.stringify(required) && JSON.stringify(statuses) === JSON.stringify(expectedStatuses);
}

export function isM6ProtocolHead({ head, parent, treePaths = [] } = {}) {
  return parent === M6_BASE_COMMIT && M6_PROTOCOL_PATHS.every((path) => treePaths.includes(path)) && head !== M6_BASE_COMMIT;
}

function normalizedRows(rows = []) {
  return [...rows].map(([path, status]) => [path, status]).sort((a, b) => a[0].localeCompare(b[0]));
}

export function matchesM6ProtocolRecovery({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P_COMMIT &&
    parent === M6_P_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_PROTOCOL_RECOVERY_EXPECTED));
}

export function matchesM6MaterializerRecovery({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P2_COMMIT &&
    parent === M6_P2_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_MATERIALIZER_RECOVERY_EXPECTED));
}

export function isM6ProtocolLineageHead({ head, parent, treePaths = [], rows = [] } = {}) {
  return isM6ProtocolHead({ head, parent, treePaths }) ||
    matchesM6ProtocolRecovery({ head, parent, rows }) ||
    matchesM6MaterializerRecovery({ head, parent, rows });
}

export function recipeSha256() { return createHash("sha256").update(readFileSync(M6_RECIPE_PATH)).digest("hex"); }
