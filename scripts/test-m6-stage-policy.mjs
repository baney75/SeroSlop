import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { m5GitBytes } from "./m5-safe-git.mjs";
import {
  classifyM6Stage,
  isM6ProtocolHead,
  isM6ProtocolLineageHead,
  M6_BASE_COMMIT,
  M6_CI_RECOVERY_EXPECTED,
  M6_P_COMMIT,
  M6_P2_COMMIT,
  M6_P3_COMMIT,
  M6_MATERIALIZER_RECOVERY_EXPECTED,
  M6_PROTOCOL_PATHS,
  M6_PROTOCOL_RECOVERY_EXPECTED,
  matchesM6CiRecovery,
  matchesM6MaterializerRecovery,
  matchesM6ProtocolRecovery,
  matchesProspectiveP,
  parseM6Recipe,
  validateM6VerifyRequirements,
  M6_P5_PARENT,
  M6_P5_COMMIT,
  M6_P5_CI_RECOVERY_COMMIT,
  M6_SUBMISSION_UI_COMMIT,
  M6_P5_CI_RECOVERY_EXPECTED,
  M6_P5_ARTIFACT_SHA256,
  M6_P5_RECOVERY_ARTIFACT_SHA256,
  M6_P5_PROTOCOL_PATHS,
  M6_SUBMISSION_UI_EXPECTED,
  M6_SUBMISSION_UI_ARTIFACT_SHA256,
  M6_NO_SLOP_UI_EXPECTED,
  M6_NO_SLOP_UI_ARTIFACT_SHA256,
  M6_NO_SLOP_UI_COMMIT,
  M6_BETA1_EXPECTED,
  M6_P6_PARENT,
  M6_P6_CHECK_STATUS,
  M6_P6_S_COMMIT,
  M6_P6_S_TREE,
  M6_P6_R_EXPECTED,
  M6_P6_R_STATUS,
  M6_P6_AUTHORIZATION_PATH,
  M6_P6_S_ARTIFACT_SHA256,
  matchesM6P6RHead,
  validateM6P6Authorization,
  M6_P6_A_COMMIT,
  M6_P6_R_COMMIT,
  M6_P6_R2_STATUS,
  matchesM6P6R2Head,
  matchesM6P7Head,
  M6_P7_PARENT,
  M6_P7_EXPECTED,
  validateM6P7Protocol,
  M6_P7_S_COMMIT,
  M6_P7_S_TREE,
  M6_P7_S_ROWS,
  M6_P7_S_ARTIFACT_SHA256,
  M6_P7_R_EXPECTED,
  M6_P7_R_STATUS,
  M6_P7_A_STATUS,
  M6_P7_AUTHORIZATION_PATH,
  matchesM6P7RHead,
  matchesM6P7AuthorizationHead,
  validateM6P7Authorization,
  M6_P6_EXPECTED,
  M6_P6_ARTIFACT_SHA256,
  matchesM6P6Head,
  validateM6P6Artifacts,
  validateM6P6Inventory,
  M6_BETA1_ARTIFACT_SHA256,
  M6_BETA1_COMMIT,
  M6_BETA1_RECOVERY_EXPECTED,
  M6_BETA1_RECOVERY_ARTIFACT_SHA256,
  M6_BETA1_RECOVERY_COMMIT,
  M6_BETA1_RECOVERY2_EXPECTED,
  M6_BETA1_RECOVERY2_ARTIFACT_SHA256,
  M6_BETA1_RECOVERY2_COMMIT,
  M6_BETA1_RECOVERY3_EXPECTED,
  M6_BETA1_RECOVERY3_ARTIFACT_SHA256,
  M6_BETA1_RECOVERY3_COMMIT,
  M6_BETA1_RECOVERY4_EXPECTED,
  M6_BETA1_RECOVERY4_ARTIFACT_SHA256,
  M6_BETA1_RECOVERY4_COMMIT,
  M6_BETA1_RECOVERY5_EXPECTED,
  M6_BETA1_RECOVERY5_ARTIFACT_SHA256,
  M6_BETA1_AUTHORIZATION_PATH,
  M6_BETA1_AUTHORIZATION_STATUS,
  canonicalM6Json,
  matchesProspectiveP5,
  matchesM6P5Head,
  matchesM6P5CiRecovery,
  matchesM6SubmissionUiHead,
  matchesM6NoSlopUiHead,
  matchesM6Beta1Head,
  matchesM6Beta1RecoveryHead,
  matchesM6Beta1Recovery2Head,
  matchesM6Beta1Recovery3Head,
  matchesM6Beta1Recovery4Head,
  matchesM6Beta1Recovery5Head,
  matchesM6Beta1AuthorizationHead,
  validateM6Beta1Authorization,
  validateM6P5Artifacts,
} from "./m6-stage-policy.mjs";
const recipe = parseM6Recipe();
assert.equal(recipe.baseCommit, M6_BASE_COMMIT);
assert.equal(classifyM6Stage({}), "m6-protocol");
const p5Statuses = Object.fromEntries(M6_P5_PROTOCOL_PATHS.map((path) => [path, ["benchmark/m6/DATA_PROVENANCE.md", "benchmark/m6/p5-protocol.json", "benchmark/m6/p5-quota-census.json", "benchmark/m6/p5_protocol.py", "benchmark/m6/p5_transform_fixture.py", "benchmark/m6/test_p5_protocol.py"].includes(path) ? "A" : "M"]));
assert.equal(matchesProspectiveP5({ head: "f".repeat(40), parent: M6_P5_PARENT, paths: M6_P5_PROTOCOL_PATHS, statuses: p5Statuses }), true);
assert.equal(matchesProspectiveP5({ head: "f".repeat(40), parent: M6_P5_PARENT, paths: M6_P5_PROTOCOL_PATHS.slice(1), statuses: p5Statuses }), false);
assert.equal(matchesM6P5Head({ head: "f".repeat(40), parent: M6_P5_PARENT, rows: M6_P5_PROTOCOL_PATHS.map((p) => [p, ["benchmark/m6/DATA_PROVENANCE.md","benchmark/m6/p5-protocol.json","benchmark/m6/p5-quota-census.json","benchmark/m6/p5_protocol.py","benchmark/m6/p5_transform_fixture.py","benchmark/m6/test_p5_protocol.py"].includes(p) ? "A" : "M"]), treePaths: M6_P5_PROTOCOL_PATHS }), true);
const p5Artifacts = Object.fromEntries(Object.keys(M6_P5_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_P5_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(p5Artifacts), true);
assert.throws(() => validateM6P5Artifacts({ ...p5Artifacts, "benchmark/m6/p5-protocol.json": Buffer.from("{}\n") }), /bytes changed/);
const missingP5Artifact = { ...p5Artifacts };
delete missingP5Artifact[Object.keys(M6_P5_ARTIFACT_SHA256)[0]];
assert.throws(() => validateM6P5Artifacts(missingP5Artifact), /inventory changed/);
const p5RecoveryRows = M6_P5_CI_RECOVERY_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6P5CiRecovery({ head: "1".repeat(40), parent: M6_P5_COMMIT, rows: p5RecoveryRows }), true);
assert.equal(matchesM6P5CiRecovery({ head: "1".repeat(40), parent: M6_P5_PARENT, rows: p5RecoveryRows }), false);
assert.equal(matchesM6P5CiRecovery({ head: "1".repeat(40), parent: M6_P5_COMMIT, rows: p5RecoveryRows.slice(1) }), false);
assert.equal(matchesM6P5CiRecovery({ head: "1".repeat(40), parent: M6_P5_COMMIT, rows: [...p5RecoveryRows, ["extra", "M"]] }), false);
const p5RecoveryArtifacts = Object.fromEntries(Object.keys(M6_P5_RECOVERY_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_P5_CI_RECOVERY_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(p5RecoveryArtifacts, M6_P5_RECOVERY_ARTIFACT_SHA256), true);
const submissionUiRows = M6_SUBMISSION_UI_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6SubmissionUiHead({ head: "2".repeat(40), parent: M6_P5_CI_RECOVERY_COMMIT, rows: submissionUiRows }), true);
assert.equal(matchesM6SubmissionUiHead({ head: "2".repeat(40), parent: M6_P5_COMMIT, rows: submissionUiRows }), false);
assert.equal(matchesM6SubmissionUiHead({ head: "2".repeat(40), parent: M6_P5_CI_RECOVERY_COMMIT, rows: submissionUiRows.slice(1) }), false);
assert.equal(matchesM6SubmissionUiHead({ head: "2".repeat(40), parent: M6_P5_CI_RECOVERY_COMMIT, rows: [...submissionUiRows, ["extra", "M"]] }), false);
const submissionUiArtifacts = Object.fromEntries(Object.keys(M6_SUBMISSION_UI_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_SUBMISSION_UI_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(submissionUiArtifacts, M6_SUBMISSION_UI_ARTIFACT_SHA256), true);
assert.throws(() => validateM6P5Artifacts({ ...submissionUiArtifacts, "src/static/popup.html": Buffer.from("bad") }, M6_SUBMISSION_UI_ARTIFACT_SHA256), /bytes changed/);
const noSlopUiRows = M6_NO_SLOP_UI_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6NoSlopUiHead({ head: "3".repeat(40), parent: M6_SUBMISSION_UI_COMMIT, rows: noSlopUiRows }), true);
assert.equal(matchesM6NoSlopUiHead({ head: "3".repeat(40), parent: M6_P5_CI_RECOVERY_COMMIT, rows: noSlopUiRows }), false);
assert.equal(matchesM6NoSlopUiHead({ head: "3".repeat(40), parent: M6_SUBMISSION_UI_COMMIT, rows: noSlopUiRows.slice(1) }), false);
assert.equal(matchesM6NoSlopUiHead({ head: "3".repeat(40), parent: M6_SUBMISSION_UI_COMMIT, rows: [...noSlopUiRows, ["extra", "M"]] }), false);
const noSlopUiArtifacts = Object.fromEntries(Object.keys(M6_NO_SLOP_UI_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_NO_SLOP_UI_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(noSlopUiArtifacts, M6_NO_SLOP_UI_ARTIFACT_SHA256), true);
assert.throws(() => validateM6P5Artifacts({ ...noSlopUiArtifacts, "src/static/setup.html": Buffer.from("bad") }, M6_NO_SLOP_UI_ARTIFACT_SHA256), /bytes changed/);
const beta1Rows = M6_BETA1_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6Beta1Head({ head: "4".repeat(40), parent: M6_NO_SLOP_UI_COMMIT, rows: beta1Rows }), true);
assert.equal(matchesM6Beta1Head({ head: "4".repeat(40), parent: M6_SUBMISSION_UI_COMMIT, rows: beta1Rows }), false);
assert.equal(matchesM6Beta1Head({ head: "4".repeat(40), parent: M6_NO_SLOP_UI_COMMIT, rows: beta1Rows.slice(1) }), false);
assert.equal(matchesM6Beta1Head({ head: "4".repeat(40), parent: M6_NO_SLOP_UI_COMMIT, rows: [...beta1Rows, ["extra", "M"]] }), false);
assert.equal(matchesM6Beta1Head({ head: "4".repeat(40), parent: M6_NO_SLOP_UI_COMMIT, rows: beta1Rows.map(([path, status], index) => [path, index === 0 ? "A" : status]) }), false);
const beta1Artifacts = Object.fromEntries(Object.keys(M6_BETA1_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_BETA1_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(beta1Artifacts, M6_BETA1_ARTIFACT_SHA256), true);
assert.throws(() => validateM6P5Artifacts({ ...beta1Artifacts, "src/content.ts": Buffer.from("bad") }, M6_BETA1_ARTIFACT_SHA256), /bytes changed/);
const beta1RecoveryRows = M6_BETA1_RECOVERY_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6Beta1RecoveryHead({ head: "7".repeat(40), parent: M6_BETA1_COMMIT, rows: beta1RecoveryRows }), true);
assert.equal(matchesM6Beta1RecoveryHead({ head: "7".repeat(40), parent: M6_NO_SLOP_UI_COMMIT, rows: beta1RecoveryRows }), false);
assert.equal(matchesM6Beta1RecoveryHead({ head: "7".repeat(40), parent: M6_BETA1_COMMIT, rows: beta1RecoveryRows.slice(1) }), false);
assert.equal(matchesM6Beta1RecoveryHead({ head: "7".repeat(40), parent: M6_BETA1_COMMIT, rows: [...beta1RecoveryRows, ["extra", "M"]] }), false);
const beta1RecoveryArtifacts = Object.fromEntries(Object.keys(M6_BETA1_RECOVERY_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_BETA1_RECOVERY_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(beta1RecoveryArtifacts, M6_BETA1_RECOVERY_ARTIFACT_SHA256), true);
const beta1Recovery2Rows = M6_BETA1_RECOVERY2_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6Beta1Recovery2Head({ head: "8".repeat(40), parent: M6_BETA1_RECOVERY_COMMIT, rows: beta1Recovery2Rows }), true);
assert.equal(matchesM6Beta1Recovery2Head({ head: "8".repeat(40), parent: M6_BETA1_COMMIT, rows: beta1Recovery2Rows }), false);
assert.equal(matchesM6Beta1Recovery2Head({ head: "8".repeat(40), parent: M6_BETA1_RECOVERY_COMMIT, rows: beta1Recovery2Rows.slice(1) }), false);
assert.equal(matchesM6Beta1Recovery2Head({ head: "8".repeat(40), parent: M6_BETA1_RECOVERY_COMMIT, rows: [...beta1Recovery2Rows, ["extra", "M"]] }), false);
const beta1Recovery2Artifacts = Object.fromEntries(Object.keys(M6_BETA1_RECOVERY2_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_BETA1_RECOVERY2_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(beta1Recovery2Artifacts, M6_BETA1_RECOVERY2_ARTIFACT_SHA256), true);
const beta1Recovery3Rows = M6_BETA1_RECOVERY3_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6Beta1Recovery3Head({ head: "9".repeat(40), parent: M6_BETA1_RECOVERY2_COMMIT, rows: beta1Recovery3Rows }), true);
assert.equal(matchesM6Beta1Recovery3Head({ head: "9".repeat(40), parent: M6_BETA1_RECOVERY_COMMIT, rows: beta1Recovery3Rows }), false);
assert.equal(matchesM6Beta1Recovery3Head({ head: "9".repeat(40), parent: M6_BETA1_RECOVERY2_COMMIT, rows: beta1Recovery3Rows.slice(1) }), false);
assert.equal(matchesM6Beta1Recovery3Head({ head: "9".repeat(40), parent: M6_BETA1_RECOVERY2_COMMIT, rows: [...beta1Recovery3Rows, ["extra", "M"]] }), false);
const beta1Recovery3Artifacts = Object.fromEntries(Object.keys(M6_BETA1_RECOVERY3_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_BETA1_RECOVERY3_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(beta1Recovery3Artifacts, M6_BETA1_RECOVERY3_ARTIFACT_SHA256), true);
const beta1Recovery4Rows = M6_BETA1_RECOVERY4_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6Beta1Recovery4Head({ head: "a".repeat(40), parent: M6_BETA1_RECOVERY3_COMMIT, rows: beta1Recovery4Rows }), true);
assert.equal(matchesM6Beta1Recovery4Head({ head: "a".repeat(40), parent: M6_BETA1_RECOVERY2_COMMIT, rows: beta1Recovery4Rows }), false);
assert.equal(matchesM6Beta1Recovery4Head({ head: "a".repeat(40), parent: M6_BETA1_RECOVERY3_COMMIT, rows: beta1Recovery4Rows.slice(1) }), false);
assert.equal(matchesM6Beta1Recovery4Head({ head: "a".repeat(40), parent: M6_BETA1_RECOVERY3_COMMIT, rows: [...beta1Recovery4Rows, ["extra", "M"]] }), false);
const beta1Recovery4Artifacts = Object.fromEntries(Object.keys(M6_BETA1_RECOVERY4_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_BETA1_RECOVERY4_COMMIT}:${path}`])]));
assert.equal(validateM6P5Artifacts(beta1Recovery4Artifacts, M6_BETA1_RECOVERY4_ARTIFACT_SHA256), true);
const beta1Recovery5Rows = M6_BETA1_RECOVERY5_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6Beta1Recovery5Head({ head: "b".repeat(40), parent: M6_BETA1_RECOVERY4_COMMIT, rows: beta1Recovery5Rows }), true);
assert.equal(matchesM6Beta1Recovery5Head({ head: "b".repeat(40), parent: M6_BETA1_RECOVERY3_COMMIT, rows: beta1Recovery5Rows }), false);
assert.equal(matchesM6Beta1Recovery5Head({ head: "b".repeat(40), parent: M6_BETA1_RECOVERY4_COMMIT, rows: beta1Recovery5Rows.slice(1) }), false);
assert.equal(matchesM6Beta1Recovery5Head({ head: "b".repeat(40), parent: M6_BETA1_RECOVERY4_COMMIT, rows: [...beta1Recovery5Rows, ["extra", "M"]] }), false);
// Historical recovery5 validation must read its immutable source commit, not the descendant worktree.
const beta1Recovery5Artifacts = Object.fromEntries(Object.keys(M6_BETA1_RECOVERY5_ARTIFACT_SHA256).map((path) => [path, m5GitBytes(["show", `${M6_P6_PARENT}:${path}`])]));
assert.equal(validateM6P5Artifacts(beta1Recovery5Artifacts, M6_BETA1_RECOVERY5_ARTIFACT_SHA256), true);
assert.equal(matchesM6Beta1AuthorizationHead({ head: "5".repeat(40), parent: "4".repeat(40), rows: [[M6_BETA1_AUTHORIZATION_PATH, "A"]] }), true);
assert.equal(matchesM6Beta1AuthorizationHead({ head: "5".repeat(40), parent: "4".repeat(40), rows: [[M6_BETA1_AUTHORIZATION_PATH, "M"]] }), false);
const beta1SourcePathMap = Object.fromEntries(M6_BETA1_EXPECTED.map(([path]) => [path, "a".repeat(64)]));
const beta1Authorization = {
  authorizationPath: M6_BETA1_AUTHORIZATION_PATH,
  benchmarkAcceptanceEligible: false,
  contributorUploadEnabled: false,
  h3PixelsRead: false,
  modelSha256: "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47",
  schemaVersion: 1,
  sourceCommit: "4".repeat(40),
  sourcePathMap: beta1SourcePathMap,
  sourcePublicCi: { conclusion: "success", event: "push", headSha: "4".repeat(40), runId: 123, status: "completed", url: "https://github.com/baney75/prooflens/actions/runs/123", workflowPath: ".github/workflows/quality.yml" },
  sourceTree: "6".repeat(40),
  status: M6_BETA1_AUTHORIZATION_STATUS,
};
assert.equal(validateM6Beta1Authorization(Buffer.from(canonicalM6Json(beta1Authorization)), { sourceCommit: "4".repeat(40), sourceTree: "6".repeat(40), sourcePathMap: beta1SourcePathMap }).status, M6_BETA1_AUTHORIZATION_STATUS);
assert.throws(() => validateM6Beta1Authorization(Buffer.from(`${JSON.stringify(beta1Authorization, null, 2)}\n`), { sourceCommit: "4".repeat(40), sourceTree: "6".repeat(40), sourcePathMap: beta1SourcePathMap }), /canonical/);
assert.throws(() => validateM6Beta1Authorization(Buffer.from(canonicalM6Json({ ...beta1Authorization, benchmarkAcceptanceEligible: true })), { sourceCommit: "4".repeat(40), sourceTree: "6".repeat(40), sourcePathMap: beta1SourcePathMap }), /boundary/);
const productCopy = ["src/content.ts", "src/popup.ts", "src/setup.ts", "src/static/manifest.json", "src/static/setup.html"]
  .map((path) => readFileSync(path, "utf8")).join("\n");
for (const forbidden of ["Private by design", "Images never go to a detector service", "Runs offline after setup", "not proof", "No server is used"]) {
  assert.equal(productCopy.includes(forbidden), false, `product copy retained: ${forbidden}`);
}
assert.match(readFileSync("src/static/setup.html", "utf8"), /<button id="prepare-model" type="button" disabled hidden>/u);
assert.throws(() => classifyM6Stage({ sourceLock: true }), /executable/);
assert.throws(() => classifyM6Stage({ trained: true }), /executable/);
assert.throws(() => classifyM6Stage({ evaluated: true }), /executable/);
const paths = [...M6_PROTOCOL_PATHS];
const statuses = Object.fromEntries(paths.map((path) => [path, ["package.json", "scripts/run-static-verification.mjs"].includes(path) ? "M" : "A"]));
assert.equal(matchesProspectiveP({ head: "a".repeat(40), parents: [M6_BASE_COMMIT], paths, statuses }), true);
assert.equal(matchesProspectiveP({ head: "a".repeat(40), parents: [M6_BASE_COMMIT], paths: paths.slice(1), statuses }), false);
assert.equal(matchesProspectiveP({ head: "a".repeat(40), parents: ["b".repeat(40)], paths, statuses }), false);
assert.equal(matchesProspectiveP({ head: "a".repeat(40), parents: [M6_BASE_COMMIT], paths, statuses: { ...statuses, [paths[0]]: "M" } }), false);
assert.equal(isM6ProtocolHead({ head: "a".repeat(40), parent: M6_BASE_COMMIT, treePaths: paths }), true);
assert.equal(isM6ProtocolHead({ head: "a".repeat(40), parent: "b".repeat(40), treePaths: paths }), false);
const recoveryRows = M6_PROTOCOL_RECOVERY_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_P_COMMIT, rows: recoveryRows }), true);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_BASE_COMMIT, rows: recoveryRows }), false);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_P_COMMIT, rows: recoveryRows.slice(1) }), false);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_P_COMMIT, rows: [...recoveryRows, ["extra", "M"]] }), false);
assert.equal(matchesM6ProtocolRecovery({ head: "c".repeat(40), parent: M6_P_COMMIT, rows: recoveryRows.map(([path, status], index) => [path, index === 0 ? "A" : status]) }), false);
assert.equal(isM6ProtocolLineageHead({ head: "a".repeat(40), parent: M6_BASE_COMMIT, treePaths: paths, rows: [] }), true);
assert.equal(isM6ProtocolLineageHead({ head: "c".repeat(40), parent: M6_P_COMMIT, treePaths: paths, rows: recoveryRows }), true);
const materializerRows = M6_MATERIALIZER_RECOVERY_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6MaterializerRecovery({ head: "d".repeat(40), parent: M6_P2_COMMIT, rows: materializerRows }), true);
assert.equal(matchesM6MaterializerRecovery({ head: "d".repeat(40), parent: M6_P_COMMIT, rows: materializerRows }), false);
assert.equal(matchesM6MaterializerRecovery({ head: "d".repeat(40), parent: M6_P2_COMMIT, rows: materializerRows.slice(1) }), false);
assert.equal(matchesM6MaterializerRecovery({ head: "d".repeat(40), parent: M6_P2_COMMIT, rows: [...materializerRows, ["extra", "A"]] }), false);
assert.equal(isM6ProtocolLineageHead({ head: "d".repeat(40), parent: M6_P2_COMMIT, treePaths: paths, rows: materializerRows }), true);
const ciRows = M6_CI_RECOVERY_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6CiRecovery({ head: "e".repeat(40), parent: M6_P3_COMMIT, rows: ciRows }), true);
assert.equal(matchesM6CiRecovery({ head: "e".repeat(40), parent: M6_P2_COMMIT, rows: ciRows }), false);
assert.equal(matchesM6CiRecovery({ head: "e".repeat(40), parent: M6_P3_COMMIT, rows: ciRows.slice(1) }), false);
assert.equal(matchesM6CiRecovery({ head: "e".repeat(40), parent: M6_P3_COMMIT, rows: [...ciRows, ["extra", "M"]] }), false);
assert.equal(isM6ProtocolLineageHead({ head: "e".repeat(40), parent: M6_P3_COMMIT, treePaths: paths, rows: ciRows }), true);
const requirements = readFileSync("benchmark/verify-requirements.txt");
assert.equal(validateM6VerifyRequirements(requirements), true);
const requirementsText = requirements.toString("utf8");
assert.throws(() => validateM6VerifyRequirements(Buffer.from(requirementsText.replace("pyarrow==20.0.0\n", ""))), /bytes changed/);
assert.throws(() => validateM6VerifyRequirements(Buffer.from(requirementsText.replace("numpy==2.2.6", "numpy==0.0.0"))), /bytes changed/);
assert.throws(() => validateM6VerifyRequirements(Buffer.from(requirementsText + "requests==2.0.0\n")), /bytes changed/);
assert.throws(() => validateM6VerifyRequirements(Buffer.from(requirementsText + "pyarrow==20.0.0\n")), /bytes changed/);
const p6Rows = M6_P6_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6P6Head({ head: "f".repeat(40), parent: M6_P6_PARENT, rows: p6Rows, treePaths: p6Rows.map(([path]) => path) }), true);
assert.equal(matchesM6P6Head({ head: "f".repeat(40), parent: "0".repeat(40), rows: p6Rows, treePaths: p6Rows.map(([path]) => path) }), false);
assert.equal(matchesM6P6Head({ head: "f".repeat(40), parent: M6_P6_PARENT, rows: [...p6Rows, ["extra", "A"]], treePaths: p6Rows.map(([path]) => path) }), false);
const p6Artifacts = Object.fromEntries(Object.keys(M6_P6_ARTIFACT_SHA256).map((path) => [path, path === "package.json" ? m5GitBytes(["show", `${M6_P6_A_COMMIT}:${path}`]) : readFileSync(path)]));
assert.equal(validateM6P6Artifacts(p6Artifacts), true);
assert.throws(() => validateM6P6Artifacts({ ...p6Artifacts, "package.json": Buffer.from("mutated") }), /bytes changed/);
assert.equal(validateM6P6Inventory(p6Artifacts["benchmark/m6/p6-frontier-inventory.json"]).status, "metadata-only; no acquisition or materialization");
assert.equal(M6_P6_CHECK_STATUS, "m6-p6-metadata-only-unverified-pass");
const rRows = M6_P6_R_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6P6RHead({ head: "a".repeat(40), parent: M6_P6_S_COMMIT, rows: rRows }), true);
assert.equal(matchesM6P6RHead({ head: "a".repeat(40), parent: "0".repeat(40), rows: rRows }), false);
assert.equal(matchesM6P6RHead({ head: "a".repeat(40), parent: M6_P6_S_COMMIT, rows: [...rRows, ["extra", "M"]] }), false);
assert.equal(M6_P6_R_STATUS, "m6-p6-verifier-ready");
const checkerSource = readFileSync("scripts/check-m6-protocol-stage.mjs", "utf8");
assert.equal(checkerSource.includes('status: "m6-p6-authorization-created"'), true);
assert.equal(checkerSource.includes('status: "m6-p6-authorization-created", head'), true);
const ciProof = { conclusion: "success", event: "push", headSha: "a".repeat(40), runId: 123, status: "completed", url: "https://github.com/baney75/prooflens/actions/runs/123", workflowPath: ".github/workflows/quality.yml" };
const auth = { acceptanceEligible: false, authorizationPath: M6_P6_AUTHORIZATION_PATH, commercialRightsClearanceClaimed: false, h3PixelsRead: false, independentOriginProofClaimed: false, metadataOnly: true, publisherAssertionOnly: true, schemaVersion: 1, protocolCommit: M6_P6_S_COMMIT, protocolParent: M6_P6_PARENT, protocolPathMap: M6_P6_S_ARTIFACT_SHA256, protocolRows: p6Rows, protocolTree: M6_P6_S_TREE, sourceLockAuthorized: false, status: "m6-p6-protocol-verified", trainingAuthorized: false, verifierCommit: "a".repeat(40), verifierTree: "u".repeat(40), verifierRows: rRows, verifierPublicCi: ciProof };
assert.equal(validateM6P6Authorization(Buffer.from(canonicalM6Json(auth)), { sourceCommit: auth.protocolCommit, sourceTree: auth.protocolTree, sourceRows: p6Rows, sourcePathMap: M6_P6_S_ARTIFACT_SHA256, verifierCommit: auth.verifierCommit, verifierTree: auth.verifierTree, verifierRows: rRows, publicCi: ciProof }).status, "m6-p6-protocol-verified");
const authArgs = { sourceCommit: auth.protocolCommit, sourceTree: auth.protocolTree, sourceRows: p6Rows, sourcePathMap: M6_P6_S_ARTIFACT_SHA256, verifierCommit: auth.verifierCommit, verifierTree: auth.verifierTree, verifierRows: rRows, publicCi: ciProof };
assert.throws(() => validateM6P6Authorization(Buffer.from(canonicalM6Json({ ...auth, trainingAuthorized: true })), authArgs), /boundary/);
assert.throws(() => validateM6P6Authorization(Buffer.from(canonicalM6Json({ ...auth, verifierTree: "v".repeat(40) })), authArgs), /source binding/);
assert.throws(() => validateM6P6Authorization(Buffer.from(canonicalM6Json({ ...auth, verifierCommit: "b".repeat(40) })), authArgs), /source binding/);
assert.throws(() => validateM6P6Authorization(Buffer.from(canonicalM6Json({ ...auth, protocolRows: rRows })), authArgs), /source binding/);
const zeroRunCi = { ...ciProof, runId: 0, url: "https://github.com/baney75/prooflens/actions/runs/0" };
assert.throws(() => validateM6P6Authorization(Buffer.from(canonicalM6Json({ ...auth, verifierPublicCi: zeroRunCi })), { ...authArgs, publicCi: zeroRunCi }), /CI schema/);
const duplicateAuth = canonicalM6Json(auth).replace('{"acceptanceEligible":false', '{"acceptanceEligible":false,"acceptanceEligible":false');
assert.throws(() => validateM6P6Authorization(Buffer.from(duplicateAuth), authArgs), /duplicate/);
assert.throws(() => validateM6P6Authorization(Buffer.from(`${canonicalM6Json(auth).trim()}\n\n`), authArgs), /canonical/);
assert.throws(() => validateM6P6Inventory(Buffer.from(`${readFileSync("benchmark/m6/p6-frontier-inventory.json", "utf8").trim()}\n\n`)), /canonical/);
const r2Rows = M6_P6_R_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6P6R2Head({ head: "c".repeat(40), parent: M6_P6_A_COMMIT, rows: r2Rows }), true);
assert.equal(M6_P6_R_COMMIT, "9d04c7fb49d79dae572007adbd917510daf26001");
assert.equal(matchesM6P6R2Head({ head: "c".repeat(40), parent: "0".repeat(40), rows: r2Rows }), false);
assert.equal(matchesM6P6R2Head({ head: "c".repeat(40), parent: M6_P6_A_COMMIT, rows: [...r2Rows, ["extra", "M"]] }), false);
assert.equal(M6_P6_R2_STATUS, "m6-p6-protocol-ci-recovery-ready");
const p7Rows = M6_P7_EXPECTED.map(([path, status]) => [path, status]);
assert.equal(matchesM6P7Head({ head: "d".repeat(40), parent: M6_P7_PARENT, rows: p7Rows, treePaths: p7Rows.map(([path]) => path) }), true);
assert.equal(matchesM6P7Head({ head: "d".repeat(40), parent: "0".repeat(40), rows: p7Rows, treePaths: p7Rows.map(([path]) => path) }), false);
const p7Protocol = readFileSync("benchmark/m6/p7-protocol.json");
assert.equal(validateM6P7Protocol(p7Protocol), true);
assert.throws(() => validateM6P7Protocol(Buffer.from(p7Protocol.toString().replace("p7-phase1-taste-unverified", "forged"))), /bytes changed/);
const p7RRows = M6_P7_R_EXPECTED.map(([path, status]) => [path, status]);
assert.deepEqual(Object.fromEntries(Object.keys(M6_P7_S_ARTIFACT_SHA256).map((path) => [path, createHash("sha256").update(m5GitBytes(["show", `${M6_P7_S_COMMIT}:${path}`])).digest("hex")])), M6_P7_S_ARTIFACT_SHA256);
assert.equal(matchesM6P7RHead({ head: "e".repeat(40), parent: M6_P7_S_COMMIT, rows: p7RRows }), true);
assert.equal(matchesM6P7RHead({ head: "e".repeat(40), parent: M6_P7_S_COMMIT, rows: M6_P7_S_ROWS }), false);
assert.equal(matchesM6P7RHead({ head: "e".repeat(40), parent: "0".repeat(40), rows: p7RRows }), false);
assert.equal(matchesM6P7AuthorizationHead({ head: "f".repeat(40), parent: "e".repeat(40), rows: [[M6_P7_AUTHORIZATION_PATH, "A"]] }), true);
assert.equal(matchesM6P7AuthorizationHead({ head: "f".repeat(40), parent: "e".repeat(40), rows: [[M6_P7_AUTHORIZATION_PATH, "M"]] }), false);
assert.equal(M6_P7_R_STATUS, "p7-phase1-taste-verifier-ready");
assert.equal(M6_P7_A_STATUS, "p7-phase1-taste-authorized");
const p7Ci = { conclusion: "success", event: "push", headSha: "e".repeat(40), runId: 123, status: "completed", url: "https://github.com/baney75/prooflens/actions/runs/123", workflowPath: ".github/workflows/quality.yml" };
const p7Auth = { acceptanceEligible:false, authorizationPath:M6_P7_AUTHORIZATION_PATH, commercialRightsClearanceClaimed:false, h3PixelsRead:false, independentOriginProofClaimed:false, phase1InputsAuthorized:true, protocolCommit:M6_P7_S_COMMIT, protocolParent:M6_P7_PARENT, protocolPathMap:M6_P7_S_ARTIFACT_SHA256, protocolRows:M6_P7_S_ROWS, protocolTree:M6_P7_S_TREE, publisherAssertionOnly:true, schemaVersion:1, sourceLockAuthorized:false, status:M6_P7_A_STATUS, tasteMaterializationAuthorized:true, trainingAuthorized:false, verifierCommit:"e".repeat(40), verifierPublicCi:p7Ci, verifierRows:p7RRows, verifierTree:"u".repeat(40) };
const p7Args = { sourceCommit:M6_P7_S_COMMIT, sourceTree:M6_P7_S_TREE, sourceParent:M6_P7_PARENT, sourceRows:M6_P7_S_ROWS, sourcePathMap:M6_P7_S_ARTIFACT_SHA256, verifierCommit:p7Auth.verifierCommit, verifierTree:p7Auth.verifierTree, verifierRows:p7RRows, publicCi:p7Ci };
assert.equal(validateM6P7Authorization(Buffer.from(canonicalM6Json(p7Auth)), p7Args).status, M6_P7_A_STATUS);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, trainingAuthorized:true})), p7Args), /boundary/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, sourceLockAuthorized:true})), p7Args), /boundary/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, acceptanceEligible:true})), p7Args), /boundary/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, phase1InputsAuthorized:false})), p7Args), /boundary/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, verifierTree:"v".repeat(40)})), p7Args), /source binding/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, verifierCommit:"f".repeat(40)})), p7Args), /source binding/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, protocolRows:p7RRows})), p7Args), /source binding/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, protocolPathMap:{...M6_P7_S_ARTIFACT_SHA256,"package.json":"0".repeat(64)}})), p7Args), /source binding/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, verifierPublicCi:{...p7Ci,runId:0}})), {...p7Args, publicCi:{...p7Ci,runId:0}}), /CI proof/);
assert.throws(() => validateM6P7Authorization(Buffer.from(canonicalM6Json({...p7Auth, verifierPublicCi:{...p7Ci,url:"https://example.com/forged"}})), {...p7Args, publicCi:{...p7Ci,url:"https://example.com/forged"}}), /CI proof/);
const p7Duplicate = canonicalM6Json(p7Auth).replace('{"acceptanceEligible":false', '{"acceptanceEligible":false,"acceptanceEligible":false');
assert.throws(() => validateM6P7Authorization(Buffer.from(p7Duplicate), p7Args), /duplicate/);
assert.equal(checkerSource.includes("authorize-p7"), true);
assert.equal(checkerSource.includes("p7-phase1-taste-authorization-created"), true);
assert.equal(m5GitBytes(["rev-parse", "285bc3eefcaff35a6ae8a6cc9b23b2d0abdd4f90^{tree}"]).toString("utf8").trim(), "2457bb455d05fcef86aee07fb0c38cccd6ba289e");
console.log("M6 stage policy PASS");
