import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { TextDecoder } from "node:util";

import { reconstructM4Candidate, validateM4AdapterModel } from "./m4-candidate-patch.mjs";
import { renderM4PublicDocuments } from "./render-m4-public-docs.mjs";
import {
  M4,
  M4_PUBLICATION_ROWS,
  digest,
  parseCanonicalJson,
  parseCanonicalM4PublicationLock,
  parseManifestMetadata,
  requireCondition,
  validateM4TrainingPacket,
} from "./m4-training-contract.mjs";
import {
  M4_BASE_COMMIT,
  M4_FAILED_PROTOCOL_COMMIT,
  M4_PROTOCOL_RECOVERY_COMMIT,
  M4_LOCK_EXPECTED,
  M4_PUBLICATION_LOCK_PATH,
  M4_SOURCE_EXPECTED,
  matchesM4ProtocolRecoveryLineage,
  matchesExpectedRows,
} from "./m4-stage-policy.mjs";


const FINAL_OUTPUTS = M4_PUBLICATION_ROWS.map(([pathname]) => pathname)
  .filter((pathname) => pathname.startsWith("benchmark/evidence/m4/"));

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }).trim();
}

function commitRows(commit) {
  return git(["diff-tree", "--no-commit-id", "--no-renames", "--name-status", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, pathname] = line.split("\t");
      return [pathname, status];
    });
}

function parents(commit) {
  return git(["show", "-s", "--format=%P", commit]).split(" ").filter(Boolean);
}

function strictJson(value, label) {
  let parsed;
  try {
    parsed = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(value));
  } catch (error) {
    throw new Error(`${label} is not strict JSON`, { cause: error });
  }
  return parsed;
}

function sourceBytes(commit, pathname) {
  return execFileSync("git", ["show", `${commit}:${pathname}`], {
    encoding: null,
    maxBuffer: 128 * 1024 * 1024,
  });
}

function embeddedBytes(lock, name) {
  const text = lock.candidateEvidenceJson[name];
  requireCondition(typeof text === "string", `M4 locked candidate evidence is missing: ${name}`);
  return Buffer.from(text, "utf8");
}

export function validateM4LockedPacket(lockBytes = readFileSync(M4_PUBLICATION_LOCK_PATH)) {
  const lock = parseCanonicalM4PublicationLock(lockBytes);
  const sourceParents = parents(lock.sourceCommit);
  requireCondition(/^[a-f0-9]{40}$/u.test(lock.sourceCommit) && /^[a-f0-9]{40}$/u.test(lock.sourceTree) &&
    sourceParents.length === 1 && git(["rev-parse", `${lock.sourceCommit}^{tree}`]) === lock.sourceTree,
  "M4 lock source commit or tree changed");
  const upstreamBytes = sourceBytes(lock.sourceCommit, "weights/prooflens-cf384.onnx");
  requireCondition(digest(upstreamBytes) === M4.upstreamSha256 &&
    upstreamBytes.length === M4.upstreamBytes &&
    digest(sourceBytes(lock.sourceCommit, "model-lock.json")) === lock.upstreamModelLockSha256,
  "M4 locked upstream model or model lock changed");
  requireCondition(lock.trainerSha256 === digest(sourceBytes(lock.sourceCommit, "benchmark/m4/train_adapter.py")) &&
    lock.recipeSha256 === digest(sourceBytes(lock.sourceCommit, "benchmark/m4/recipe.json")) &&
    lock.sourceLocksSha256 === digest(sourceBytes(lock.sourceCommit, "benchmark/m4/source-locks.json")) &&
    lock.selectionSummarySha256 === digest(sourceBytes(lock.sourceCommit, "benchmark/evidence/m4/selection-summary.json")) &&
    lock.finalizerSha256 === digest(sourceBytes(lock.sourceCommit, "benchmark/m4/finalize.py")) &&
    lock.publicationContractSha256 === digest(sourceBytes(lock.sourceCommit, "benchmark/m4/publication_contract.py")) &&
    lock.fixtureSelectorSha256 === digest(sourceBytes(lock.sourceCommit, "benchmark/m4/select_model_state_fixtures.py")) &&
    lock.documentationRendererSha256 === digest(sourceBytes(lock.sourceCommit, "scripts/render-m4-public-docs.mjs")),
  "M4 publication lock source-tool binding changed");

  const summaryBytes = embeddedBytes(lock, "training-summary.json");
  const calibrationBytes = embeddedBytes(lock, "calibration.json");
  const gridBytes = embeddedBytes(lock, "candidate-grid.json");
  const selectionLockBytes = embeddedBytes(lock, "selection-lock.json");
  const tensorSealBytes = embeddedBytes(lock, "candidate-tensor-seal.json");
  const freshMarkerBytes = embeddedBytes(lock, "fresh-feature-run.json");
  const comparisonBytes = embeddedBytes(lock, "model-comparison.json");
  const fixtureBytes = embeddedBytes(lock, "fixture-manifest.json");
  const summary = parseCanonicalJson(summaryBytes, "M4 locked training summary");
  const calibration = parseCanonicalJson(calibrationBytes, "M4 locked calibration");
  const grid = parseCanonicalJson(gridBytes, "M4 locked candidate grid");
  const selectionLock = parseCanonicalJson(selectionLockBytes, "M4 locked selection lock");
  const tensorSeal = parseCanonicalJson(tensorSealBytes, "M4 locked candidate tensor seal");
  const freshMarker = parseCanonicalJson(freshMarkerBytes, "M4 locked fresh feature marker");
  const comparison = parseCanonicalJson(comparisonBytes, "M4 locked model comparison");
  const fixture = parseCanonicalJson(fixtureBytes, "M4 locked fixture manifest");
  const recipe = strictJson(sourceBytes(lock.sourceCommit, "benchmark/m4/recipe.json"), "M4 recipe");
  const selectionSummary = parseCanonicalJson(
    sourceBytes(lock.sourceCommit, "benchmark/evidence/m4/selection-summary.json"), "M4 source selection summary");
  const selectorBytes = sourceBytes(lock.sourceCommit, "benchmark/evidence/m4/validation-manifest.jsonl");
  const m3Bytes = sourceBytes(lock.sourceCommit, "benchmark/evidence/m3/validation-manifest.jsonl");
  const m2Bytes = sourceBytes(lock.sourceCommit, "benchmark/evidence/m2/validation-manifest.jsonl");
  const selectorMetadata = parseManifestMetadata(selectorBytes, {
    label: "M4 selector", items: 600,
    sources: {
      "british-library-plates": { label: 0, count: 300 },
      "rapidata-dalle-3": { label: 1, count: 75 },
      "rapidata-flux": { label: 1, count: 75 },
      "rapidata-midjourney": { label: 1, count: 75 },
      "rapidata-stable-diffusion": { label: 1, count: 75 },
    },
  });
  const m3Metadata = parseManifestMetadata(m3Bytes, {
    label: "M3 regression", items: 600,
    sources: { "met-open-access": { label: 0, count: 300 }, "flux-1-dev-development": { label: 1, count: 300 } },
  });
  const m2Metadata = parseManifestMetadata(m2Bytes, {
    label: "M2 regression", items: 900,
    sources: {
      "open-images": { label: 0, count: 300 }, "stockimages-cc0": { label: 0, count: 300 },
      "GLM-Image": { label: 1, count: 150 }, "HunyuanImage-3.0": { label: 1, count: 150 },
    },
  });
  const expectedCandidateHashes = {
    model: lock.adapterPatch.candidateSha256,
    summary: digest(summaryBytes),
    calibration: digest(calibrationBytes),
    grid: digest(gridBytes),
    selectionLock: digest(selectionLockBytes),
    tensorSeal: digest(tensorSealBytes),
    freshMarker: digest(freshMarkerBytes),
  };
  requireCondition(JSON.stringify(lock.candidateHashes) === JSON.stringify(expectedCandidateHashes) &&
    freshMarker.schemaVersion === 1 && freshMarker.state === "complete" &&
    freshMarker.runId === summary.freshFeatureRunId && freshMarker.context?.pipelineVersion === 1 &&
    freshMarker.context?.trainerSha256 === lock.trainerSha256 &&
    freshMarker.context?.recipeSha256 === lock.recipeSha256 &&
    freshMarker.context?.modelSha256 === lock.upstreamModelSha256 &&
    freshMarker.context?.trainManifestSha256 === summary.trainManifestSha256 &&
    freshMarker.context?.selectorManifestSha256 === summary.selectorManifestSha256 &&
    freshMarker.context?.m3RegressionManifestSha256 === summary.m3RegressionManifestSha256 &&
    freshMarker.context?.m2RegressionManifestSha256 === summary.m2RegressionManifestSha256 &&
    freshMarker.context?.selectionSummarySha256 === lock.selectionSummarySha256,
  "M4 publication lock candidate hashes changed");
  const winner = validateM4TrainingPacket({
    summary, calibration, grid, recipe, selectionSummary,
    hashes: {
      trainer: lock.trainerSha256,
      selectionSummary: lock.selectionSummarySha256,
      selectorManifest: digest(selectorBytes),
      m3RegressionManifest: digest(m3Bytes),
      m2RegressionManifest: digest(m2Bytes),
      tensorSeal: digest(tensorSealBytes),
      grid: digest(gridBytes),
      selectionLock: digest(selectionLockBytes),
      calibration: digest(calibrationBytes),
      freshMarker: lock.candidateHashes.freshMarker,
      model: lock.adapterPatch.candidateSha256,
      modelBytes: lock.adapterPatch.candidateBytes,
    },
    selectorMetadata,
    regressionMetadata: { m3: m3Metadata, m2: m2Metadata },
    tensorSeal,
    selectionLock,
    freshMarker,
  });
  requireCondition(JSON.stringify(lock.selectionLock) === JSON.stringify(selectionLock) &&
    lock.candidateModelBytes === lock.adapterPatch.candidateBytes &&
    lock.modelComparisonSha256 === digest(comparisonBytes), "M4 lock selection or comparison binding changed");
  const candidateBytes = reconstructM4Candidate({
    baseBytes: upstreamBytes, adapterPatch: lock.adapterPatch,
  });
  const initializerPatch = new Map(lock.adapterPatch.addedInitializers.map((row) => [row.name, row]));
  for (const [name, encoded] of Object.entries(winner.tensorFloat32Base64)) {
    requireCondition(initializerPatch.get(name)?.rawDataBase64 === encoded,
      `M4 selected tensor does not bind the reconstructed model: ${name}`);
  }
  validateM4AdapterModel({
    baseBytes: upstreamBytes, candidateBytes, comparison,
  });
  requireCondition(digest(candidateBytes) === lock.candidateHashes.model && candidateBytes.length === lock.candidateModelBytes,
    "M4 reconstructed model bytes changed");
  const rendered = renderM4PublicDocuments({
    readme: sourceBytes(lock.sourceCommit, "README.md").toString("utf8"),
    modelCard: sourceBytes(lock.sourceCommit, "MODEL_CARD.md").toString("utf8"),
    benchmark: sourceBytes(lock.sourceCommit, "BENCHMARK.md").toString("utf8"),
    summary,
    modelSha256: lock.candidateHashes.model,
    modelBytes: lock.candidateModelBytes,
  });
  for (const [key, value] of Object.entries(rendered)) {
    requireCondition(lock.publicDocumentHashes[`${key}.md`] === digest(Buffer.from(value, "utf8")),
      `M4 locked public-document digest changed: ${key}.md`);
  }
  requireCondition(lock.fixtureManifestSha256 === digest(fixtureBytes) && fixture.schemaVersion === 4 &&
    fixture.adapterModel === true && fixture.assetsUnchangedFromM2 === true &&
    fixture.modelSha256 === lock.candidateHashes.model &&
    fixture.calibrationSha256 === lock.candidateHashes.calibration &&
    fixture.trainingSummarySha256 === lock.candidateHashes.summary &&
    fixture.inferenceProvider === "CPUExecutionProvider" && Array.isArray(fixture.items) && fixture.items.length === 2,
  "M4 locked fixture packet changed");
  requireCondition(winner.candidateId === selectionLock.selectedCandidateId &&
    lock.selectionInfluencedByRegression === false && lock.h3HoldoutScored === false && lock.h3PixelsRead === false,
  "M4 lock selection or H3 boundary changed");
  return { lock, candidateBytes, comparison, fixture, summary, calibration, grid, winner };
}

function validatePinnedStage() {
  requireCondition(git(["status", "--porcelain=v1", "--untracked-files=all"]) === "",
    "M4 pinned verification requires a completely clean repository");
  const head = git(["rev-parse", "HEAD"]);
  const lockParents = parents(head);
  requireCondition(lockParents.length === 1, "The M4 lock commit must have one parent");
  const source = lockParents[0];
  const sourceParents = parents(source);
  requireCondition(sourceParents.length === 1, "The M4 source commit must have one parent");
  const protocol = sourceParents[0];
  const protocolParents = parents(protocol);
  requireCondition(matchesM4ProtocolRecoveryLineage({
    protocolParents,
    protocolRows: commitRows(protocol),
    recoveryProtocolParents: parents(M4_PROTOCOL_RECOVERY_COMMIT),
    recoveryProtocolRows: commitRows(M4_PROTOCOL_RECOVERY_COMMIT),
    recoveryProtocolTree: git(["rev-parse", `${M4_PROTOCOL_RECOVERY_COMMIT}^{tree}`]),
    failedProtocolParents: parents(M4_FAILED_PROTOCOL_COMMIT),
    failedProtocolRows: commitRows(M4_FAILED_PROTOCOL_COMMIT),
    failedProtocolTree: git(["rev-parse", `${M4_FAILED_PROTOCOL_COMMIT}^{tree}`]),
    baseTree: git(["rev-parse", `${M4_BASE_COMMIT}^{tree}`]),
  }), "The M4 pinned lineage changed before the recovered protocol commit");
  requireCondition(matchesExpectedRows(commitRows(source), M4_SOURCE_EXPECTED) &&
    matchesExpectedRows(commitRows(head), M4_LOCK_EXPECTED),
  "The M4 pinned lineage changed outside an exact stage packet");
  const packet = validateM4LockedPacket();
  requireCondition(packet.lock.sourceCommit === source && packet.lock.sourceTree === git(["rev-parse", `${source}^{tree}`]),
    "The M4 publication lock does not bind its exact source commit");
  for (const pathname of FINAL_OUTPUTS) {
    requireCondition(!existsSync(pathname), `M4 pinned stage contains final evidence: ${pathname}`);
  }
  requireCondition(!existsSync("benchmark/evidence/m4/failed-selector-diagnostic-1.json") &&
    !existsSync("benchmark/evidence/m4/failed-training-attempt-1.json"),
  "M4 pinned stage cannot coexist with terminal failure evidence");
  execFileSync("node", ["scripts/check-m4-selection-evidence.mjs"], { stdio: "inherit" });
  console.log(JSON.stringify({
    stage: "m4-pinned", head, source, modelSha256: packet.lock.candidateHashes.model,
    candidateId: packet.winner.candidateId, h3HoldoutScored: false, policy: "pass",
  }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  validatePinnedStage();
}
