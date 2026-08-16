import { createHash } from "node:crypto";
import { m5Git, m5GitBytes } from "./m5-safe-git.mjs";
import {
  M6_BASE_COMMIT,
  M6_BASE_TREE,
  M6_CENSUS_PATH,
  M6_CENSUS_SHA256,
  M6_P_COMMIT,
  M6_P_RECIPE_SHA256,
  M6_P_TREE,
  M6_RECIPE_PATH,
  M6_RECIPE_SHA256,
  matchesM6ProtocolRecovery,
  matchesProspectiveP,
  parseM6Recipe,
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

const head = git(["rev-parse", "HEAD"]);
if (head === M6_P_COMMIT) {
  console.log(JSON.stringify({ status: "m6-protocol-pass", head, parent: pParents[0], rows: pRows }));
  process.exit(0);
}

const headParents = parentsOf(head);
const recoveryRows = rowsOf(head);
if (headParents.length !== 1 || !matchesM6ProtocolRecovery({
  head,
  parent: headParents[0],
  rows: recoveryRows.map(({ path, status }) => [path, status]),
})) {
  throw new Error("M6 protocol recovery lineage/path map mismatch");
}
const recipeBytes = m5GitBytes(["show", `${head}:${M6_RECIPE_PATH}`]);
if (digest(recipeBytes) !== M6_RECIPE_SHA256) throw new Error("M6 corrected recipe HEAD bytes changed");
parseM6Recipe(recipeBytes);
const censusBytes = m5GitBytes(["show", `${head}:${M6_CENSUS_PATH}`]);
if (digest(censusBytes) !== M6_CENSUS_SHA256) throw new Error("M6 census changed across recovery");
console.log(JSON.stringify({
  status: "m6-protocol-recovery-pass",
  head,
  parent: headParents[0],
  rows: recoveryRows,
}));
