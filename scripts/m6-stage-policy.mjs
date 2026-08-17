import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

export const M6_BASE_COMMIT = "76d0a807dcf240245830b8510e623d838e43cd4c";
export const M6_BASE_TREE = "5c10012b9520e3936efc86e08ff0a53adecec868";
export const M6_P_COMMIT = "3b29ea2f9e1ad46e4cd78f47c9ccf5fe3a99877e";
export const M6_P_TREE = "dfd29cd86f4f746d403b14994055a575d82f83c4";
export const M6_P2_COMMIT = "0777710c89cd0fa02e2f4bd063ec51664e3fc26a";
export const M6_P2_TREE = "30d8338382033caacd400e1b29c37ed287f9de43";
export const M6_P3_COMMIT = "fa9f002a2f9805b59d7955bf4c4f9992bbfb22ce";
export const M6_P3_TREE = "74f84b12a5a1a126fbb79142f6530be996ce4990";
export const M6_CENSUS_SHA256 = "61f494f09fe256d771bacb809712b5c645e5b25f63cffca52dfc40d0e0ac7adf";
export const M6_P_RECIPE_SHA256 = "56bfe2487760c833c796289e3d4c5e8ef0eb65e62493229f9d62631a573ab613";
export const M6_P2_RECIPE_SHA256 = "42f594fd26ac4949f191eb5c773c977ec8e5bee586f766c9a648afce85bc2984";
export const M6_RECIPE_SHA256 = "a1c1700acbfbed19ef73e3cc4224c994eadef17e81ddb8d6d8040c8a3d5a5e88";
export const M6_SOURCE_SHARDS_PATH = "benchmark/m6/source-shards.json";
export const M6_SOURCE_SHARDS_SHA256 = "a86c7209e76248edddd61537f397379194a7aaa908405e0cede7c8f5a3d7fbfe";
export const M6_VERIFY_REQUIREMENTS_PATH = "benchmark/verify-requirements.txt";
export const M6_P3_VERIFY_REQUIREMENTS_SHA256 = "f34d97a0c10c23d3dba50f8bbcf4df8dad9e5b3cf80510b20c6df14bdc06af75";
export const M6_VERIFY_REQUIREMENTS_SHA256 = "00ea11478746fdb02c445e31c084e496e08f8fc6cc49f313fd229d34d70396ed";
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
export const M6_CI_RECOVERY_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"],
  ["benchmark/verify-requirements.txt", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_STAGES = Object.freeze(["m6-protocol", "m6-source-lock", "m6-preflight", "m6-trained", "m6-evaluated"]);
export const M6_P5_PARENT = "05a131b64fdef5f7fe8a6bdad4dac6d401e8193a";
export const M6_P5_COMMIT = "c878c2dc7ecbb49edb1cac4395aa20649471a330";
export const M6_P5_TREE = "4657121b4ba5c99006b8f0df8f0c4b629c78dc2d";
export const M6_P5_CI_RECOVERY_COMMIT = "57beb67fbfd9a140565b3b83436ba2893fd0fd82";
export const M6_P5_CI_RECOVERY_TREE = "bd75e0203f3b7156ee4cd3e870d22f1787d60b05";
export const M6_P5_PROTOCOL_PATHS = Object.freeze(["benchmark/m6/DATA_PROVENANCE.md", "benchmark/m6/p5-protocol.json", "benchmark/m6/p5-quota-census.json", "benchmark/m6/p5_protocol.py", "benchmark/m6/p5_transform_fixture.py", "benchmark/m6/test_p5_protocol.py", "benchmark/m6/README.md", "benchmark/m6/THIRD_PARTY_NOTICES.md", "package.json", "scripts/m6-stage-policy.mjs", "scripts/check-m6-protocol-stage.mjs", "scripts/test-m6-stage-policy.mjs", "scripts/run-static-verification.mjs"]);
export const M6_P5_ARTIFACT_SHA256 = Object.freeze({
  "benchmark/m6/DATA_PROVENANCE.md": "858f02abaf94445387f1cbd91f8495b0f0e179bfc3fd63e6f678e32f6fa90523",
  "benchmark/m6/p5-protocol.json": "ebee34dab243eb320679fad9dcd2166f1da4af1bc2ac46986c11d0d024f4582b",
  "benchmark/m6/p5-quota-census.json": "373bb0c6aad9980c1f9860e98e16286d27e8a0bc54962c5d02a9209cfb41d47e",
  "benchmark/m6/p5_protocol.py": "ac3f64057071cdb9a059498f1ef0b6fcbf55bd8f0c27bda026f3d41589a3caa9",
  "benchmark/m6/p5_transform_fixture.py": "17076323643442cf104d23260b19968d07dfd5cb1156aae73ecbfd3d1ccf357b",
  "benchmark/m6/test_p5_protocol.py": "33ebd1400ee9e88daf80682b8e34b1fab342d434dd511ad9312706b878eaaad4",
  "benchmark/m6/README.md": "da12842e6c7b8f48bc54e60f9b5bde901ea640ac105148cd7dac2db180a6acb6",
  "benchmark/m6/THIRD_PARTY_NOTICES.md": "e66693fd6b18d089f127dfd5199bc88fd2e3ddf5b85924aea661bb15db730c27",
  "package.json": "fd0c21754713ad1bda2dd22c520133bd118fc46e35c551832b878e5e61d96f4e",
});
export const M6_P5_CI_RECOVERY_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"],
  ["benchmark/m6/p5_transform_fixture.py", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_P5_RECOVERY_ARTIFACT_SHA256 = Object.freeze({
  ...M6_P5_ARTIFACT_SHA256,
  "benchmark/m6/p5_transform_fixture.py": "5cfbb8c3df33887aea2740003f8ef7ea39b2c691ca9464214f7eb739b399f73f",
  "benchmark/m6/README.md": "6fd0e35cf55fcfd580d2e634e22626c1910aacf4246e402fc67280bc3ca0e1ac",
});
export const M6_SUBMISSION_UI_EXPECTED = Object.freeze([
  ["scripts/build.mjs", "M"],
  ["scripts/check-benchmark-evidence.mjs", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
  ["src/static/popup.html", "M"],
  ["src/static/seroslop.svg", "A"],
]);
export const M6_SUBMISSION_UI_ARTIFACT_SHA256 = Object.freeze({
  "scripts/build.mjs": "bba20c2e37c0d1c8479c7ee0d3fa1b98956e033c9858871994a72bebadd64b3c",
  "scripts/check-benchmark-evidence.mjs": "5343e28c755c46d6b034d06991c3d0120d75aa77b69d933401d2c01d3ec15a93",
  "scripts/chrome-smoke.mjs": "a8f4c21b36e474a5c5b55ff349df110ef0336824b893a7f12104a8ef090fc33d",
  "src/static/popup.html": "747150fc7fc2e22de4a9cbf2e3afacd410d975380255abb5d84c5909062166bb",
  "src/static/seroslop.svg": "3305b345c480a6ea2f3ed7e7ae907c7ff2ebbffa9dd221342070cbe294ab9c9f",
});
export function validateM6P5Artifacts(artifactBytes = {}, expectedDigests = M6_P5_ARTIFACT_SHA256) {
  const expectedPaths = Object.keys(expectedDigests).sort();
  if (JSON.stringify(Object.keys(artifactBytes).sort()) !== JSON.stringify(expectedPaths)) throw new Error("M6 P5 artifact inventory changed");
  for (const path of expectedPaths) {
    const bytes = Buffer.from(artifactBytes[path]);
    if (createHash("sha256").update(bytes).digest("hex") !== expectedDigests[path]) throw new Error(`M6 P5 artifact bytes changed: ${path}`);
  }
  return true;
}
export function matchesProspectiveP5({ head, parent, paths = [], statuses = {} } = {}) {
  const expected = [...M6_P5_PROTOCOL_PATHS].sort(); const actual = [...new Set(paths)].sort();
  const additions = new Set(["benchmark/m6/DATA_PROVENANCE.md", "benchmark/m6/p5-protocol.json", "benchmark/m6/p5-quota-census.json", "benchmark/m6/p5_protocol.py", "benchmark/m6/p5_transform_fixture.py", "benchmark/m6/test_p5_protocol.py"]);
  return /^[0-9a-f]{40}$/.test(head ?? "") && parent === M6_P5_PARENT && JSON.stringify(expected) === JSON.stringify(actual) && M6_P5_PROTOCOL_PATHS.every((p) => statuses[p] === (additions.has(p) ? "A" : "M"));
}
export function matchesM6P5Head({ head, parent, rows = [], treePaths = [] } = {}) {
  const statuses = Object.fromEntries(rows.map(([path, status]) => [path, status]));
  return matchesProspectiveP5({ head, parent, paths: rows.map(([p]) => p), statuses }) && M6_P5_PROTOCOL_PATHS.every((p) => treePaths.includes(p));
}

export function matchesM6P5CiRecovery({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P5_COMMIT && parent === M6_P5_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_P5_CI_RECOVERY_EXPECTED));
}

export function matchesM6SubmissionUiHead({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P5_CI_RECOVERY_COMMIT &&
    parent === M6_P5_CI_RECOVERY_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_SUBMISSION_UI_EXPECTED));
}

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

export function validateM6VerifyRequirements(bytes = readFileSync(M6_VERIFY_REQUIREMENTS_PATH)) {
  const value = Buffer.from(bytes);
  if (createHash("sha256").update(value).digest("hex") !== M6_VERIFY_REQUIREMENTS_SHA256) {
    throw new Error("M6 verification requirements bytes changed");
  }
  const text = value.toString("utf8");
  if (Buffer.from(text, "utf8").compare(value) !== 0 || !text.endsWith("\n")) {
    throw new Error("M6 verification requirements must be canonical UTF-8 text");
  }
  const lines = text.trimEnd().split("\n");
  if (lines.filter((line) => line === "pyarrow==20.0.0").length !== 1) {
    throw new Error("M6 verification requirements must add exactly pyarrow 20.0.0");
  }
  return true;
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

export function matchesM6CiRecovery({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P3_COMMIT &&
    parent === M6_P3_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_CI_RECOVERY_EXPECTED));
}

export function isM6ProtocolLineageHead({ head, parent, treePaths = [], rows = [] } = {}) {
  return isM6ProtocolHead({ head, parent, treePaths }) ||
    matchesM6P5Head({ head, parent, rows, treePaths }) ||
    matchesM6P5CiRecovery({ head, parent, rows }) ||
    matchesM6SubmissionUiHead({ head, parent, rows }) ||
    matchesM6ProtocolRecovery({ head, parent, rows }) ||
    matchesM6MaterializerRecovery({ head, parent, rows }) ||
    matchesM6CiRecovery({ head, parent, rows });
}

export function recipeSha256() { return createHash("sha256").update(readFileSync(M6_RECIPE_PATH)).digest("hex"); }
