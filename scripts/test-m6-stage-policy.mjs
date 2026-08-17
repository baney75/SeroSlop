import assert from "node:assert/strict";
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
const beta1Recovery4Artifacts = Object.fromEntries(Object.keys(M6_BETA1_RECOVERY4_ARTIFACT_SHA256).map((path) => [path, readFileSync(path)]));
assert.equal(validateM6P5Artifacts(beta1Recovery4Artifacts, M6_BETA1_RECOVERY4_ARTIFACT_SHA256), true);
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
console.log("M6 stage policy PASS");
