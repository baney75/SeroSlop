import { createHash } from "node:crypto";
import { m5Git, m5GitBytes } from "./m5-safe-git.mjs";
import {
  M6_BASE_COMMIT,
  M6_BASE_TREE,
  M6_CENSUS_PATH,
  M6_CENSUS_SHA256,
  M6_P_COMMIT,
  M6_P2_COMMIT,
  M6_P2_RECIPE_SHA256,
  M6_P2_TREE,
  M6_P3_COMMIT,
  M6_P3_TREE,
  M6_P3_VERIFY_REQUIREMENTS_SHA256,
  M6_P_RECIPE_SHA256,
  M6_P_TREE,
  M6_RECIPE_PATH,
  M6_RECIPE_SHA256,
  M6_SOURCE_SHARDS_PATH,
  M6_SOURCE_SHARDS_SHA256,
  M6_VERIFY_REQUIREMENTS_PATH,
  matchesM6CiRecovery,
  matchesM6MaterializerRecovery,
  matchesM6ProtocolRecovery,
  matchesProspectiveP,
  parseM6Recipe,
  validateM6VerifyRequirements,
  M6_P5_PARENT,
  M6_P5_ARTIFACT_SHA256,
  matchesM6P5Head,
  validateM6P5Artifacts,
} from "./m6-stage-policy.mjs";

const git = (args) => m5Git(args);
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const parentsOf = (commit) => git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
const rowsOf = (commit) => git([
  "diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit,
]).split("\n").filter(Boolean).map((line) => {
  const [status, path] = line.split("\t");
  return { path, status };
});

if (git(["rev-parse", `${M6_BASE_COMMIT}^{tree}`]) !== M6_BASE_TREE) {
  throw new Error("M6 base tree changed");
}
if (git(["rev-parse", `${M6_P_COMMIT}^{tree}`]) !== M6_P_TREE) {
  throw new Error("M6 immutable P tree changed");
}
const pParents = parentsOf(M6_P_COMMIT);
const pRows = rowsOf(M6_P_COMMIT);
const pStatuses = Object.fromEntries(pRows.map(({ path, status }) => [path, status]));
if (!matchesProspectiveP({
  head: M6_P_COMMIT,
  parents: pParents,
  paths: pRows.map(({ path }) => path),
  statuses: pStatuses,
})) {
  throw new Error("M6 immutable P lineage/path map changed");
}
const pRecipeBytes = m5GitBytes(["show", `${M6_P_COMMIT}:${M6_RECIPE_PATH}`]);
if (digest(pRecipeBytes) !== M6_P_RECIPE_SHA256) throw new Error("M6 immutable P recipe changed");
const pCensusBytes = m5GitBytes(["show", `${M6_P_COMMIT}:${M6_CENSUS_PATH}`]);
if (digest(pCensusBytes) !== M6_CENSUS_SHA256) throw new Error("M6 immutable P census changed");

if (git(["rev-parse", `${M6_P2_COMMIT}^{tree}`]) !== M6_P2_TREE) {
  throw new Error("M6 immutable P2 tree changed");
}
const p2Parents = parentsOf(M6_P2_COMMIT);
const p2Rows = rowsOf(M6_P2_COMMIT);
if (p2Parents.length !== 1 || !matchesM6ProtocolRecovery({
  head: M6_P2_COMMIT,
  parent: p2Parents[0],
  rows: p2Rows.map(({ path, status }) => [path, status]),
})) {
  throw new Error("M6 immutable P2 lineage/path map changed");
}
const p2RecipeBytes = m5GitBytes(["show", `${M6_P2_COMMIT}:${M6_RECIPE_PATH}`]);
if (digest(p2RecipeBytes) !== M6_P2_RECIPE_SHA256) throw new Error("M6 immutable P2 recipe changed");
const p2CensusBytes = m5GitBytes(["show", `${M6_P2_COMMIT}:${M6_CENSUS_PATH}`]);
if (digest(p2CensusBytes) !== M6_CENSUS_SHA256) throw new Error("M6 census changed at P2");

if (git(["rev-parse", `${M6_P3_COMMIT}^{tree}`]) !== M6_P3_TREE) {
  throw new Error("M6 immutable P3 tree changed");
}
const p3Parents = parentsOf(M6_P3_COMMIT);
const p3Rows = rowsOf(M6_P3_COMMIT);
if (p3Parents.length !== 1 || !matchesM6MaterializerRecovery({
  head: M6_P3_COMMIT,
  parent: p3Parents[0],
  rows: p3Rows.map(({ path, status }) => [path, status]),
})) {
  throw new Error("M6 immutable P3 lineage/path map changed");
}
const p3RecipeBytes = m5GitBytes(["show", `${M6_P3_COMMIT}:${M6_RECIPE_PATH}`]);
if (digest(p3RecipeBytes) !== M6_RECIPE_SHA256) throw new Error("M6 immutable P3 recipe changed");
const p3CensusBytes = m5GitBytes(["show", `${M6_P3_COMMIT}:${M6_CENSUS_PATH}`]);
if (digest(p3CensusBytes) !== M6_CENSUS_SHA256) throw new Error("M6 census changed at P3");
const p3SourceShardBytes = m5GitBytes(["show", `${M6_P3_COMMIT}:${M6_SOURCE_SHARDS_PATH}`]);
if (digest(p3SourceShardBytes) !== M6_SOURCE_SHARDS_SHA256) throw new Error("M6 source-shard inventory changed at P3");
const p3RequirementsBytes = m5GitBytes(["show", `${M6_P3_COMMIT}:${M6_VERIFY_REQUIREMENTS_PATH}`]);
if (digest(p3RequirementsBytes) !== M6_P3_VERIFY_REQUIREMENTS_SHA256) throw new Error("M6 immutable P3 verification requirements changed");

const head = git(["rev-parse", "HEAD"]);
const headParents = parentsOf(head);
const headRows = rowsOf(head).map(({ path, status }) => [path, status]);
const headTreePaths = git(["ls-tree", "-r", "--name-only", head]).split("\n").filter(Boolean);
if (headParents.length === 1 && headParents[0] === M6_P5_PARENT && matchesM6P5Head({ head, parent: headParents[0], rows: headRows, treePaths: headTreePaths })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_P5_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])));
  console.log(JSON.stringify({ status: "m6-p5-protocol-pass", head, parent: headParents[0], rows: headRows }));
  process.exit(0);
}
if (head === M6_P_COMMIT) {
  console.log(JSON.stringify({ status: "m6-protocol-pass", head, parent: pParents[0], rows: pRows }));
  process.exit(0);
}
if (head === M6_P2_COMMIT) {
  console.log(JSON.stringify({ status: "m6-protocol-recovery-pass", head, parent: p2Parents[0], rows: p2Rows }));
  process.exit(0);
}
if (head === M6_P3_COMMIT) {
  console.log(JSON.stringify({ status: "m6-materializer-recovery-pass", head, parent: p3Parents[0], rows: p3Rows }));
  process.exit(0);
}

const recoveryRows = rowsOf(head);
if (headParents.length !== 1 || !matchesM6CiRecovery({
  head,
  parent: headParents[0],
  rows: recoveryRows.map(({ path, status }) => [path, status]),
})) {
  throw new Error("M6 CI recovery lineage/path map mismatch");
}
const recipeBytes = m5GitBytes(["show", `${head}:${M6_RECIPE_PATH}`]);
if (digest(recipeBytes) !== M6_RECIPE_SHA256) throw new Error("M6 corrected recipe HEAD bytes changed");
parseM6Recipe(recipeBytes);
const censusBytes = m5GitBytes(["show", `${head}:${M6_CENSUS_PATH}`]);
if (digest(censusBytes) !== M6_CENSUS_SHA256) throw new Error("M6 census changed across recovery");
const sourceShardBytes = m5GitBytes(["show", `${head}:${M6_SOURCE_SHARDS_PATH}`]);
if (digest(sourceShardBytes) !== M6_SOURCE_SHARDS_SHA256) throw new Error("M6 source-shard inventory changed");
const requirementsBytes = m5GitBytes(["show", `${head}:${M6_VERIFY_REQUIREMENTS_PATH}`]);
validateM6VerifyRequirements(requirementsBytes);
console.log(JSON.stringify({
  status: "m6-ci-recovery-pass",
  head,
  parent: headParents[0],
  rows: recoveryRows,
}));
