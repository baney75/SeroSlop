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
  M6_P5_COMMIT,
  M6_P5_TREE,
  M6_P5_CI_RECOVERY_COMMIT,
  M6_P5_CI_RECOVERY_TREE,
  M6_SUBMISSION_UI_COMMIT,
  M6_SUBMISSION_UI_TREE,
  M6_P5_ARTIFACT_SHA256,
  M6_P5_RECOVERY_ARTIFACT_SHA256,
  M6_SUBMISSION_UI_ARTIFACT_SHA256,
  M6_NO_SLOP_UI_ARTIFACT_SHA256,
  M6_NO_SLOP_UI_COMMIT,
  M6_NO_SLOP_UI_TREE,
  M6_BETA1_ARTIFACT_SHA256,
  M6_BETA1_COMMIT,
  M6_BETA1_TREE,
  M6_BETA1_RECOVERY_ARTIFACT_SHA256,
  M6_BETA1_RECOVERY_COMMIT,
  M6_BETA1_RECOVERY_TREE,
  M6_BETA1_RECOVERY2_ARTIFACT_SHA256,
  M6_BETA1_RECOVERY2_COMMIT,
  M6_BETA1_RECOVERY2_TREE,
  M6_BETA1_RECOVERY3_ARTIFACT_SHA256,
  M6_BETA1_AUTHORIZATION_PATH,
  M6_BETA1_EXPECTED,
  matchesM6P5Head,
  matchesM6P5CiRecovery,
  matchesM6SubmissionUiHead,
  matchesM6NoSlopUiHead,
  matchesM6Beta1Head,
  matchesM6Beta1RecoveryHead,
  matchesM6Beta1Recovery2Head,
  matchesM6Beta1Recovery3Head,
  matchesM6Beta1AuthorizationHead,
  validateM6Beta1Authorization,
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

if (git(["rev-parse", `${M6_P5_COMMIT}^{tree}`]) !== M6_P5_TREE) throw new Error("M6 immutable P5 tree changed");
const p5Parents = parentsOf(M6_P5_COMMIT);
const p5Rows = rowsOf(M6_P5_COMMIT).map(({ path, status }) => [path, status]);
const p5TreePaths = git(["ls-tree", "-r", "--name-only", M6_P5_COMMIT]).split("\n").filter(Boolean);
if (p5Parents.length !== 1 || !matchesM6P5Head({ head: M6_P5_COMMIT, parent: p5Parents[0], rows: p5Rows, treePaths: p5TreePaths })) throw new Error("M6 immutable P5 lineage/path map changed");
validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_P5_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_P5_COMMIT}:${path}`])])));

if (git(["rev-parse", `${M6_P5_CI_RECOVERY_COMMIT}^{tree}`]) !== M6_P5_CI_RECOVERY_TREE) throw new Error("M6 immutable P5 CI recovery tree changed");
const p5RecoveryParents = parentsOf(M6_P5_CI_RECOVERY_COMMIT);
const p5RecoveryRows = rowsOf(M6_P5_CI_RECOVERY_COMMIT).map(({ path, status }) => [path, status]);
if (p5RecoveryParents.length !== 1 || !matchesM6P5CiRecovery({ head: M6_P5_CI_RECOVERY_COMMIT, parent: p5RecoveryParents[0], rows: p5RecoveryRows })) throw new Error("M6 immutable P5 CI recovery lineage/path map changed");
validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_P5_RECOVERY_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_P5_CI_RECOVERY_COMMIT}:${path}`])])), M6_P5_RECOVERY_ARTIFACT_SHA256);

if (git(["rev-parse", `${M6_SUBMISSION_UI_COMMIT}^{tree}`]) !== M6_SUBMISSION_UI_TREE) throw new Error("M6 immutable submission UI tree changed");
const submissionUiParents = parentsOf(M6_SUBMISSION_UI_COMMIT);
const submissionUiRows = rowsOf(M6_SUBMISSION_UI_COMMIT).map(({ path, status }) => [path, status]);
if (submissionUiParents.length !== 1 || !matchesM6SubmissionUiHead({ head: M6_SUBMISSION_UI_COMMIT, parent: submissionUiParents[0], rows: submissionUiRows })) throw new Error("M6 immutable submission UI lineage/path map changed");
validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_SUBMISSION_UI_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_SUBMISSION_UI_COMMIT}:${path}`])])), M6_SUBMISSION_UI_ARTIFACT_SHA256);

if (git(["rev-parse", `${M6_NO_SLOP_UI_COMMIT}^{tree}`]) !== M6_NO_SLOP_UI_TREE) throw new Error("M6 immutable no-slop UI tree changed");
const noSlopUiParents = parentsOf(M6_NO_SLOP_UI_COMMIT);
const noSlopUiRows = rowsOf(M6_NO_SLOP_UI_COMMIT).map(({ path, status }) => [path, status]);
if (noSlopUiParents.length !== 1 || !matchesM6NoSlopUiHead({ head: M6_NO_SLOP_UI_COMMIT, parent: noSlopUiParents[0], rows: noSlopUiRows })) throw new Error("M6 immutable no-slop UI lineage/path map changed");
validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_NO_SLOP_UI_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_NO_SLOP_UI_COMMIT}:${path}`])])), M6_NO_SLOP_UI_ARTIFACT_SHA256);

if (git(["rev-parse", `${M6_BETA1_COMMIT}^{tree}`]) !== M6_BETA1_TREE) throw new Error("M6 immutable Beta1 tree changed");
const beta1Parents = parentsOf(M6_BETA1_COMMIT);
const beta1Rows = rowsOf(M6_BETA1_COMMIT).map(({ path, status }) => [path, status]);
if (beta1Parents.length !== 1 || !matchesM6Beta1Head({ head: M6_BETA1_COMMIT, parent: beta1Parents[0], rows: beta1Rows })) throw new Error("M6 immutable Beta1 lineage/path map changed");
validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_BETA1_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_BETA1_COMMIT}:${path}`])])), M6_BETA1_ARTIFACT_SHA256);

if (git(["rev-parse", `${M6_BETA1_RECOVERY_COMMIT}^{tree}`]) !== M6_BETA1_RECOVERY_TREE) throw new Error("M6 immutable Beta1 recovery tree changed");
const beta1RecoveryParents = parentsOf(M6_BETA1_RECOVERY_COMMIT);
const beta1RecoveryRows = rowsOf(M6_BETA1_RECOVERY_COMMIT).map(({ path, status }) => [path, status]);
if (beta1RecoveryParents.length !== 1 || !matchesM6Beta1RecoveryHead({ head: M6_BETA1_RECOVERY_COMMIT, parent: beta1RecoveryParents[0], rows: beta1RecoveryRows })) throw new Error("M6 immutable Beta1 recovery lineage/path map changed");
validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_BETA1_RECOVERY_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_BETA1_RECOVERY_COMMIT}:${path}`])])), M6_BETA1_RECOVERY_ARTIFACT_SHA256);

if (git(["rev-parse", `${M6_BETA1_RECOVERY2_COMMIT}^{tree}`]) !== M6_BETA1_RECOVERY2_TREE) throw new Error("M6 immutable Beta1 recovery2 tree changed");
const beta1Recovery2Parents = parentsOf(M6_BETA1_RECOVERY2_COMMIT);
const beta1Recovery2Rows = rowsOf(M6_BETA1_RECOVERY2_COMMIT).map(({ path, status }) => [path, status]);
if (beta1Recovery2Parents.length !== 1 || !matchesM6Beta1Recovery2Head({ head: M6_BETA1_RECOVERY2_COMMIT, parent: beta1Recovery2Parents[0], rows: beta1Recovery2Rows })) throw new Error("M6 immutable Beta1 recovery2 lineage/path map changed");
validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_BETA1_RECOVERY2_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_BETA1_RECOVERY2_COMMIT}:${path}`])])), M6_BETA1_RECOVERY2_ARTIFACT_SHA256);

const head = git(["rev-parse", "HEAD"]);
const headParents = parentsOf(head);
const headRows = rowsOf(head).map(({ path, status }) => [path, status]);
const headTreePaths = git(["ls-tree", "-r", "--name-only", head]).split("\n").filter(Boolean);
if (headParents.length === 1 && matchesM6Beta1AuthorizationHead({ head, parent: headParents[0], rows: headRows })) {
  const sourceCommit = headParents[0];
  const sourceParents = parentsOf(sourceCommit);
  const sourceRows = rowsOf(sourceCommit).map(({ path, status }) => [path, status]);
  if (sourceParents.length !== 1 || !matchesM6Beta1Recovery3Head({ head: sourceCommit, parent: sourceParents[0], rows: sourceRows })) {
    throw new Error("M6 Beta1 authorization source lineage changed");
  }
  const sourceTree = git(["rev-parse", `${sourceCommit}^{tree}`]);
  const sourcePathMap = Object.fromEntries(M6_BETA1_EXPECTED.map(([path]) => [path, digest(m5GitBytes(["show", `${sourceCommit}:${path}`]))]));
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_BETA1_RECOVERY3_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${sourceCommit}:${path}`])])), M6_BETA1_RECOVERY3_ARTIFACT_SHA256);
  validateM6Beta1Authorization(m5GitBytes(["show", `${head}:${M6_BETA1_AUTHORIZATION_PATH}`]), { sourceCommit, sourceTree, sourcePathMap });
  console.log(JSON.stringify({ status: "m6-beta1-authorized-pass", head, sourceCommit, sourceTree, rows: headRows }));
  process.exit(0);
}
if (headParents.length === 1 && headParents[0] === M6_P5_PARENT && matchesM6P5Head({ head, parent: headParents[0], rows: headRows, treePaths: headTreePaths })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_P5_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])));
  console.log(JSON.stringify({ status: "m6-p5-protocol-pass", head, parent: headParents[0], rows: headRows }));
  process.exit(0);
}
if (headParents.length === 1 && matchesM6P5CiRecovery({ head, parent: headParents[0], rows: headRows })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_P5_RECOVERY_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])), M6_P5_RECOVERY_ARTIFACT_SHA256);
  console.log(JSON.stringify({ status: "m6-p5-ci-recovery-pass", head, parent: headParents[0], rows: headRows }));
  process.exit(0);
}
if (headParents.length === 1 && matchesM6SubmissionUiHead({ head, parent: headParents[0], rows: headRows })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_SUBMISSION_UI_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])), M6_SUBMISSION_UI_ARTIFACT_SHA256);
  console.log(JSON.stringify({ status: "m6-submission-ui-pass", head, parent: headParents[0], rows: headRows }));
  process.exit(0);
}
if (headParents.length === 1 && matchesM6NoSlopUiHead({ head, parent: headParents[0], rows: headRows })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_NO_SLOP_UI_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])), M6_NO_SLOP_UI_ARTIFACT_SHA256);
  console.log(JSON.stringify({ status: "m6-no-slop-ui-pass", head, parent: headParents[0], rows: headRows }));
  process.exit(0);
}
if (headParents.length === 1 && matchesM6Beta1Head({ head, parent: headParents[0], rows: headRows })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_BETA1_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])), M6_BETA1_ARTIFACT_SHA256);
  console.log(JSON.stringify({ status: "m6-beta1-pass", head, parent: headParents[0], rows: headRows }));
  process.exit(0);
}
if (headParents.length === 1 && matchesM6Beta1RecoveryHead({ head, parent: headParents[0], rows: headRows })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_BETA1_RECOVERY_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])), M6_BETA1_RECOVERY_ARTIFACT_SHA256);
  console.log(JSON.stringify({ status: "m6-beta1-ci-recovery-pass", head, parent: headParents[0], rows: headRows }));
  process.exit(0);
}
if (headParents.length === 1 && matchesM6Beta1Recovery2Head({ head, parent: headParents[0], rows: headRows })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_BETA1_RECOVERY2_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])), M6_BETA1_RECOVERY2_ARTIFACT_SHA256);
  console.log(JSON.stringify({ status: "m6-beta1-ci-recovery2-pass", head, parent: headParents[0], rows: headRows }));
  process.exit(0);
}
if (headParents.length === 1 && matchesM6Beta1Recovery3Head({ head, parent: headParents[0], rows: headRows })) {
  validateM6P5Artifacts(Object.fromEntries(Object.keys(M6_BETA1_RECOVERY3_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${head}:${path}`])])), M6_BETA1_RECOVERY3_ARTIFACT_SHA256);
  console.log(JSON.stringify({ status: "m6-beta1-ci-recovery3-pass", head, parent: headParents[0], rows: headRows }));
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
