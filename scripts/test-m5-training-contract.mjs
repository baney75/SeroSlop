import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { validateM5Recipe } from "./m5-training-contract.mjs";

const original = JSON.parse(readFileSync("benchmark/m5/recipe.json", "utf8"));
assert.equal(original.sourceEvidence.initialParityDiagnostic.sha256, "c9c673efa0b1a6e4ea79b195ec16c71ae8ac91f962390a49c4e570b6d8de5c11");
const clone = (value) => JSON.parse(JSON.stringify(value));
assert.equal(validateM5Recipe(clone(original)), true);

const mutations = [
  (value) => { value.training.requiredGpuProduct = "NVIDIA A100"; },
  (value) => { value.deliverable.maximumBytes = 900_000_000; },
  (value) => { value.selection.gates.original.minimumRealRecall = 0.99; },
  (value) => { value.selection.gates.original.minimumBalancedAccuracy = 0.9; },
  (value) => { value.h3Boundary.pixelsMayBeRead = true; },
  (value) => { value.sourceEvidence.trainingManifest.items = 1; },
  (value) => { value.largeSyntheticEvaluation.minimumItems = 99_999; },
  (value) => { value.largeSyntheticEvaluation.source.revision = "0".repeat(40); },
  (value) => { value.training.maximumPaidWallClockSeconds = 86_401; },
  (value) => { value.training.maximumPaidWallClockSeconds = 28_800; },
  (value) => { value.training.providerSignedAttestation = true; },
  (value) => { value.training.provider = "unknown-provider"; },
  (value) => { value.sourceEvidence.initialParityDiagnostic.path = "wrong.json"; },
  (value) => { value.sourceEvidence.initialParityDiagnostic.sha256 = "0".repeat(64); },
  (value) => { value.training.onnxRuntimeProviderPolicy.useTf32 = true; },
  (value) => { value.selection.falsePositiveConfidence.poolAcrossVariants = true; },
  (value) => { value.selection.falsePositiveConfidence.trialsPerVariant = 1_200; },
  (value) => { value.largeSyntheticEvaluation.scoreBlindnessEvidence.privatePriorScoringAbsenceProven = true; },
];

for (const mutate of mutations) {
  const candidate = clone(original);
  mutate(candidate);
  assert.throws(() => validateM5Recipe(candidate));
}

console.log(JSON.stringify({ cases: mutations.length + 1, policy: "pass" }));
