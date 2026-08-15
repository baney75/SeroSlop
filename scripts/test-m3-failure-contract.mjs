import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import {
  completeDecisionThresholds,
  evaluateM3SelectorLogits,
  nextUp,
  parseCanonicalFailureJson,
  parseM3SelectorManifest,
  validateM3FailureDiagnostic,
  validateM3FailureReceipt,
} from "./m3-failure-contract.mjs";

const SOURCE_COMMIT = "2e6de2187d1cff5aea48e57ad9a30f15541fc4df";
const SOURCE_TREE = "a6ab771f27b1efd108caa0b08128118fd7465334";
const BASE_COMMIT = "0adbd55d8cdc25ad3d20e773a315ec57d14c7973";
const MODEL_SHA = "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47";
const RUN_ID = "447053b4b6f924488653d237e2372230";
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

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function canonical(value) {
  return Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
}

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

const diagnosticBytes = readFileSync("benchmark/evidence/m3/failed-selector-diagnostic-1.json");
const receiptBytes = readFileSync("benchmark/evidence/m3/failed-training-attempt-1.json");
const selectorManifestBytes = readFileSync("benchmark/evidence/m3/validation-manifest.jsonl");
const recipeBytes = readFileSync("benchmark/m3/recipe.json");
const expected = {
  sourceCommit: SOURCE_COMMIT,
  sourceTree: SOURCE_TREE,
  baseCommit: BASE_COMMIT,
  inputBindings: {
    upstreamModelSha256: MODEL_SHA,
    trainerSha256: "6973877bd16d360c10622beb8c8048a8753229fdfd6226a6556427c8c00d365b",
    diagnosticGeneratorSha256: "429ba93e675f2ebf0b42fca4d50f3916021fc5abcb4cc6761be42afb5ec37746",
    recipeSha256: "df3537f308c413fb37ed8771bcc420ae735dfa52ca7f8434f9085a0aab91129c",
    selectionSummarySha256: "d79d37a52edec7d526e2466bfc4af224a1bae567d0215d774afd7871ba3aafad",
    trainManifestSha256: "a0cb9994018b44fa958e312816460b15b5fbab43d489e75b1a9f5647bd70d261",
    selectorManifestSha256: "86b7b9119fc7a118ee8b0a85e1a6b7dce6635154b3eff49cc1197b9d40fca2ff",
    regressionManifestSha256: "a63953148040e1a4223f16fa04ebf4b85c4022da65531ead0b25ce46434eab93",
    runId: RUN_ID,
  },
  commandArguments: COMMAND_ARGUMENTS,
  cacheBytes: 286_698_318,
  markerSha256: "4560e455d813e371856093566651ac6cc96769f761f746d210eb026d6329b916",
  candidateOutputs: CANDIDATE_OUTPUTS,
  publicationOutputs: PUBLICATION_OUTPUTS,
};

const diagnostic = validateM3FailureDiagnostic({
  bytes: diagnosticBytes,
  selectorManifestBytes,
  recipeBytes,
  expected,
});
const receipt = validateM3FailureReceipt({ bytes: receiptBytes, diagnosticBytes, expected });
assert.equal(diagnostic.aggregate.feasibleCandidateCount, 0);
assert.equal(receipt.diagnostic.sha256, digest(diagnosticBytes));

const metadata = parseM3SelectorManifest(selectorManifestBytes);
const feasible = metadata.labels.map((label) => label === 0 ? -10 : 10);
assert.ok(evaluateM3SelectorLogits(feasible, metadata, diagnostic.frozenGates).feasibleThresholds > 0);
const infeasible = metadata.labels.map(() => 0);
assert.equal(evaluateM3SelectorLogits(infeasible, metadata, diagnostic.frozenGates).feasibleThresholds, 0);
assert.deepEqual(completeDecisionThresholds([0, 0, 1]), [0, nextUp(0), nextUp(1)]);
assert.ok(nextUp(1) > 1 && nextUp(-1) > -1);

const badParameter = clone(diagnostic);
badParameter.candidates[0].parameters.weightDecay = 0.2;
assert.throws(() => validateM3FailureDiagnostic({
  bytes: canonical(badParameter), selectorManifestBytes, recipeBytes, expected,
}), /parameter order changed/u);

const truncated = clone(diagnostic);
truncated.candidates[0].selectorLogits.base64 = truncated.candidates[0].selectorLogits.base64.slice(4);
assert.throws(() => validateM3FailureDiagnostic({
  bytes: canonical(truncated), selectorManifestBytes, recipeBytes, expected,
}), /bytes or digest changed/u);

const nonfinite = clone(diagnostic);
const nonfiniteBytes = Buffer.from(nonfinite.candidates[0].selectorLogits.base64, "base64");
nonfiniteBytes.writeFloatLE(Number.NaN, 0);
nonfinite.candidates[0].selectorLogits.base64 = nonfiniteBytes.toString("base64");
nonfinite.candidates[0].selectorLogits.sha256 = digest(nonfiniteBytes);
assert.throws(() => validateM3FailureDiagnostic({
  bytes: canonical(nonfinite), selectorManifestBytes, recipeBytes, expected,
}), /non-finite/u);

const missingGate = clone(diagnostic);
delete missingGate.frozenGates.minimumSyntheticRecallPerFamily;
assert.throws(() => validateM3FailureDiagnostic({
  bytes: canonical(missingGate), selectorManifestBytes, recipeBytes, expected,
}), /gates changed/u);

const wrongRun = clone(receipt);
wrongRun.cacheSnapshot.runId = "0".repeat(32);
assert.throws(() => validateM3FailureReceipt({ bytes: canonical(wrongRun), diagnosticBytes, expected }),
  /cache snapshot changed/u);
const falseH3 = clone(receipt);
falseH3.h3Observation.h3PixelsReadOrScored = true;
assert.throws(() => validateM3FailureReceipt({ bytes: canonical(falseH3), diagnosticBytes, expected }),
  /H3 observation changed/u);
const unknown = clone(receipt);
unknown.acceptanceEligible = true;
assert.throws(() => validateM3FailureReceipt({ bytes: canonical(unknown), diagnosticBytes, expected }), /keys changed/u);
const overbroadAbsence = clone(receipt);
overbroadAbsence.absence.publishedM3EvidencePresent = false;
delete overbroadAbsence.absence.successfulM3PublicationEvidencePresent;
assert.throws(() => validateM3FailureReceipt({ bytes: canonical(overbroadAbsence), diagnosticBytes, expected }),
  /keys changed/u);

const duplicate = Buffer.from(receiptBytes.toString("utf8").replace(
  /^\{/u, '{\n  "schemaVersion": 99,'));
assert.throws(() => parseCanonicalFailureJson(duplicate), /not canonical/u);
const invalidUtf8 = Buffer.from(receiptBytes);
const asciiOffset = invalidUtf8.indexOf(Buffer.from("operator-observed"));
invalidUtf8[asciiOffset] = 0xff;
assert.throws(() => parseCanonicalFailureJson(invalidUtf8), /not valid UTF-8/u);

console.log(JSON.stringify({ cases: 17, candidates: 25, selectorViews: 2_400, policy: "pass" }));
