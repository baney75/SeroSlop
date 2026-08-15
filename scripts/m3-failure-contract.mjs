import { createHash } from "node:crypto";
import { TextDecoder } from "node:util";

const VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];
const DECAYS = [0.10, 0.03, 0.01, 0.003, 0.001];
const ALPHAS = [0.40, 0.55, 0.70, 0.85, 1.0];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function exactKeys(value, keys, label) {
  requireCondition(value !== null && typeof value === "object" && !Array.isArray(value),
    `${label} must be an object`);
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  requireCondition(JSON.stringify(actual) === JSON.stringify(expected), `${label} keys changed`);
}

function jsonEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function close(left, right, tolerance = 2e-12) {
  return Number.isFinite(left) && Number.isFinite(right) && Math.abs(left - right) <= tolerance;
}

export function parseCanonicalFailureJson(value, label = "M3 failure JSON") {
  const bytes = Buffer.isBuffer(value) ? value : Buffer.from(value);
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch (error) {
    throw new Error(`${label} is not valid UTF-8`, { cause: error });
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} is not valid JSON`, { cause: error });
  }
  const canonical = Buffer.from(`${JSON.stringify(parsed, null, 2)}\n`, "utf8");
  requireCondition(bytes.equals(canonical), `${label} bytes are not canonical`);
  return parsed;
}

export function nextUp(value) {
  requireCondition(Number.isFinite(value), "nextUp requires a finite value");
  if (Object.is(value, -0) || value === 0) return Number.MIN_VALUE;
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  view.setFloat64(0, value, false);
  let bits = view.getBigUint64(0, false);
  bits = value > 0 ? bits + 1n : bits - 1n;
  view.setBigUint64(0, bits, false);
  return view.getFloat64(0, false);
}

export function completeDecisionThresholds(values) {
  requireCondition(Array.isArray(values) && values.length > 0 && values.every(Number.isFinite),
    "Selector logits must be nonempty and finite");
  const unique = [...new Set(values)].sort((left, right) => left - right);
  return [unique[0], ...unique.map(nextUp)];
}

export function parseM3SelectorManifest(value) {
  let text;
  try {
    text = new TextDecoder("utf-8", { fatal: true }).decode(value);
  } catch (error) {
    throw new Error("M3 selector manifest is not valid UTF-8", { cause: error });
  }
  const rows = text.split("\n").filter(Boolean).map((line) => JSON.parse(line));
  requireCondition(rows.length === 600, "M3 selector manifest must contain exactly 600 rows");
  requireCondition(new Set(rows.map((row) => row.id)).size === 600 &&
    new Set(rows.map((row) => row.imageSha256)).size === 600,
  "M3 selector IDs or image hashes are not unique");
  const sourceCounts = new Map();
  const labels = [];
  const sources = [];
  const variants = [];
  for (const [index, row] of rows.entries()) {
    requireCondition(row.rowIndex === index && row.split === "validation", "M3 selector row order changed");
    const expectedLabel = row.source === "met-open-access" ? 0 :
      row.source === "flux-1-dev-development" ? 1 : null;
    requireCondition(row.label === expectedLabel, "M3 selector source or label changed");
    sourceCounts.set(row.source, (sourceCounts.get(row.source) ?? 0) + 1);
    for (let variant = 0; variant < VARIANTS.length; variant += 1) {
      labels.push(row.label);
      sources.push(row.source);
      variants.push(variant);
    }
  }
  requireCondition(sourceCounts.get("met-open-access") === 300 &&
    sourceCounts.get("flux-1-dev-development") === 300 && sourceCounts.size === 2,
  "M3 selector source balance changed");
  return { labels, sources, variants };
}

function variantMetrics(logits, metadata, threshold) {
  const output = {};
  for (let variant = 0; variant < VARIANTS.length; variant += 1) {
    let real = 0;
    let realCorrect = 0;
    let synthetic = 0;
    let syntheticCorrect = 0;
    const syntheticSources = new Map();
    const realSources = new Map();
    for (let index = 0; index < logits.length; index += 1) {
      if (metadata.variants[index] !== variant) continue;
      const label = metadata.labels[index];
      const source = metadata.sources[index];
      if (label === 0) {
        real += 1;
        const passed = logits[index] < threshold ? 1 : 0;
        realCorrect += passed;
        const current = realSources.get(source) ?? [0, 0];
        current[0] += passed;
        current[1] += 1;
        realSources.set(source, current);
      } else {
        synthetic += 1;
        const passed = logits[index] >= threshold ? 1 : 0;
        syntheticCorrect += passed;
        const current = syntheticSources.get(source) ?? [0, 0];
        current[0] += passed;
        current[1] += 1;
        syntheticSources.set(source, current);
      }
    }
    requireCondition(real > 0 && synthetic > 0, "M3 selector variant lost a class");
    const realRecall = realCorrect / real;
    const syntheticRecall = syntheticCorrect / synthetic;
    output[VARIANTS[variant]] = {
      balancedAccuracy: (realRecall + syntheticRecall) / 2,
      realRecall,
      syntheticRecall,
      syntheticRecallBySource: Object.fromEntries([...syntheticSources].map(([source, [correct, total]]) =>
        [source, correct / total])),
      realRecallBySource: Object.fromEntries([...realSources].map(([source, [correct, total]]) =>
        [source, correct / total])),
    };
  }
  return output;
}

function summarizeThreshold(values, gates) {
  const minima = {
    balancedAccuracy: 1,
    realRecall: 1,
    syntheticRecall: 1,
    syntheticRecallBySource: 1,
    metRecall: 1,
  };
  const margins = [];
  for (const row of Object.values(values)) {
    const family = Math.min(...Object.values(row.syntheticRecallBySource));
    const met = row.realRecallBySource["met-open-access"];
    minima.balancedAccuracy = Math.min(minima.balancedAccuracy, row.balancedAccuracy);
    minima.realRecall = Math.min(minima.realRecall, row.realRecall);
    minima.syntheticRecall = Math.min(minima.syntheticRecall, row.syntheticRecall);
    minima.syntheticRecallBySource = Math.min(minima.syntheticRecallBySource, family);
    minima.metRecall = Math.min(minima.metRecall, met);
    margins.push(
      row.balancedAccuracy - gates.minimumBalancedAccuracyPerVariant,
      row.realRecall - gates.minimumRealRecallPerVariant,
      row.syntheticRecall - gates.minimumSyntheticRecallPerVariant,
      family - gates.minimumSyntheticRecallPerFamily,
      met - gates.minimumRealRecallBySource["met-open-access"],
    );
  }
  return { minima, minimumMargin: Math.min(...margins) };
}

function compareKeys(left, right) {
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return left[index] > right[index] ? 1 : -1;
  }
  return 0;
}

export function evaluateM3SelectorLogits(logits, metadata, gates) {
  requireCondition(logits.length === 2_400 && metadata.labels.length === 2_400,
    "M3 selector logits must contain exactly 2,400 values");
  const thresholds = completeDecisionThresholds(logits);
  let feasibleThresholds = 0;
  let closest = null;
  const realMet = [];
  const synthetic = [];
  for (const threshold of thresholds) {
    const values = variantMetrics(logits, metadata, threshold);
    const { minima, minimumMargin } = summarizeThreshold(values, gates);
    if (minimumMargin >= 0) feasibleThresholds += 1;
    const key = [
      minimumMargin,
      minima.balancedAccuracy,
      minima.realRecall,
      minima.syntheticRecall,
      minima.metRecall,
      -Math.abs(threshold),
    ];
    if (closest === null || compareKeys(key, closest.key) > 0) closest = { key, threshold, minima };
    const rows = Object.values(values);
    if (rows.every((row) => row.realRecall >= gates.minimumRealRecallPerVariant &&
      row.realRecallBySource["met-open-access"] >= gates.minimumRealRecallBySource["met-open-access"])) {
      realMet.push({ threshold, values });
    }
    if (rows.every((row) => row.syntheticRecall >= gates.minimumSyntheticRecallPerVariant &&
      Math.min(...Object.values(row.syntheticRecallBySource)) >= gates.minimumSyntheticRecallPerFamily)) {
      synthetic.push({ threshold, values });
    }
  }
  requireCondition(closest !== null && realMet.length > 0 && synthetic.length > 0,
    "M3 selector did not expose both sides of the frozen threshold conflict");
  const minimumThreshold = Math.min(...realMet.map((entry) => entry.threshold));
  const maximumThreshold = Math.max(...synthetic.map((entry) => entry.threshold));
  const bestSynthetic = Math.max(...realMet.map((entry) => Math.min(...Object.values(entry.values)
    .map((row) => row.syntheticRecall))));
  const bestMet = Math.max(...synthetic.map((entry) => Math.min(...Object.values(entry.values)
    .map((row) => row.realRecallBySource["met-open-access"]))));
  return {
    selectorDecisionThresholds: thresholds.length,
    feasibleThresholds,
    thresholdConflict: {
      minimumThresholdForRealAndMetGates: minimumThreshold,
      maximumThresholdForSyntheticAndFamilyGates: maximumThreshold,
      infeasibleGapLogit: minimumThreshold - maximumThreshold,
      bestWorstVariantSyntheticRecallWhileRealAndMetPass: bestSynthetic,
      bestWorstVariantMetRecallWhileSyntheticAndFamilyPass: bestMet,
    },
    closestCompromise: {
      thresholdLogit: closest.threshold,
      minimumGateMargin: closest.key[0],
      worstVariant: closest.minima,
    },
  };
}

function requireNumericObjectEqual(actual, expected, keys, label) {
  exactKeys(actual, keys, label);
  for (const key of keys) requireCondition(close(Number(actual[key]), Number(expected[key])), `${label}.${key} changed`);
}

function decodeLogits(record) {
  exactKeys(record, ["encoding", "count", "bytes", "sha256", "base64"], "M3 selector logits");
  requireCondition(record.encoding === "base64-float32-little-endian" && record.count === 2_400 &&
    record.bytes === 9_600 && typeof record.base64 === "string", "M3 selector logit encoding changed");
  const bytes = Buffer.from(record.base64, "base64");
  requireCondition(bytes.toString("base64") === record.base64 && bytes.length === 9_600 &&
    digest(bytes) === record.sha256, "M3 selector logit bytes or digest changed");
  const values = [];
  for (let offset = 0; offset < bytes.length; offset += 4) values.push(bytes.readFloatLE(offset));
  requireCondition(values.every(Number.isFinite), "M3 selector logits contain a non-finite value");
  return values;
}

export function validateM3FailureDiagnostic({ bytes, selectorManifestBytes, recipeBytes, expected }) {
  const diagnostic = parseCanonicalFailureJson(bytes, "M3 failure diagnostic");
  exactKeys(diagnostic, ["schemaVersion", "profile", "attemptId", "role", "source", "inputBindings",
    "method", "frozenGates", "candidates", "aggregate"], "M3 failure diagnostic");
  requireCondition(diagnostic.schemaVersion === 1 && diagnostic.profile === "m3" &&
    diagnostic.attemptId === "m3-failed-training-attempt-1" &&
    diagnostic.role === "post-failure-selector-replay", "M3 failure diagnostic identity changed");
  exactKeys(diagnostic.source, ["commit", "tree"], "M3 failure diagnostic source");
  requireCondition(diagnostic.source.commit === expected.sourceCommit && diagnostic.source.tree === expected.sourceTree,
    "M3 failure diagnostic source changed");
  exactKeys(diagnostic.inputBindings, ["upstreamModelSha256", "trainerSha256", "diagnosticGeneratorSha256",
    "recipeSha256", "selectionSummarySha256", "trainManifestSha256", "selectorManifestSha256", "runId"],
  "M3 failure diagnostic bindings");
  for (const key of ["upstreamModelSha256", "trainerSha256", "diagnosticGeneratorSha256", "recipeSha256",
    "selectionSummarySha256", "trainManifestSha256", "selectorManifestSha256", "runId"]) {
    requireCondition(diagnostic.inputBindings[key] === expected.inputBindings[key],
      `M3 failure diagnostic ${key} changed`);
  }
  requireCondition(diagnostic.inputBindings.selectorManifestSha256 === digest(selectorManifestBytes) &&
    diagnostic.inputBindings.recipeSha256 === digest(recipeBytes), "M3 failure diagnostic file binding changed");
  exactKeys(diagnostic.method, ["fit", "candidateOrder", "thresholdEnumeration", "selectorViewOrder",
    "logitEncoding", "regressionUsedForSelection", "h3AcceptedAsInput"], "M3 failure diagnostic method");
  requireCondition(diagnostic.method.regressionUsedForSelection === false &&
    diagnostic.method.h3AcceptedAsInput === false && jsonEqual(diagnostic.method.selectorViewOrder, VARIANTS) &&
    diagnostic.method.logitEncoding === "base64-float32-little-endian",
  "M3 failure diagnostic scope changed");
  const recipe = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(recipeBytes));
  requireCondition(jsonEqual(diagnostic.frozenGates, recipe.validationGates), "M3 frozen selector gates changed");
  requireCondition(Array.isArray(diagnostic.candidates) && diagnostic.candidates.length === 25,
    "M3 failure diagnostic must contain 25 candidates");
  const metadata = parseM3SelectorManifest(selectorManifestBytes);
  const recomputed = [];
  let candidateIndex = 0;
  for (const decay of DECAYS) {
    for (const alpha of ALPHAS) {
      const candidate = diagnostic.candidates[candidateIndex];
      exactKeys(candidate, ["parameters", "selectorLogits", "selectorDecisionThresholds", "feasibleThresholds",
        "thresholdConflict", "closestCompromise"], `M3 candidate ${candidateIndex}`);
      requireCondition(jsonEqual(candidate.parameters, { weightDecay: decay, upstreamBlendAlpha: alpha }),
        `M3 candidate ${candidateIndex} parameter order changed`);
      const values = decodeLogits(candidate.selectorLogits);
      const result = evaluateM3SelectorLogits(values, metadata, diagnostic.frozenGates);
      requireCondition(candidate.selectorDecisionThresholds === result.selectorDecisionThresholds &&
        candidate.feasibleThresholds === result.feasibleThresholds,
      `M3 candidate ${candidateIndex} threshold count changed`);
      requireNumericObjectEqual(candidate.thresholdConflict, result.thresholdConflict,
        ["minimumThresholdForRealAndMetGates", "maximumThresholdForSyntheticAndFamilyGates", "infeasibleGapLogit",
          "bestWorstVariantSyntheticRecallWhileRealAndMetPass", "bestWorstVariantMetRecallWhileSyntheticAndFamilyPass"],
        `M3 candidate ${candidateIndex} conflict`);
      exactKeys(candidate.closestCompromise, ["thresholdLogit", "minimumGateMargin", "worstVariant"],
        `M3 candidate ${candidateIndex} closest compromise`);
      requireCondition(close(candidate.closestCompromise.thresholdLogit, result.closestCompromise.thresholdLogit) &&
        close(candidate.closestCompromise.minimumGateMargin, result.closestCompromise.minimumGateMargin),
      `M3 candidate ${candidateIndex} closest threshold changed`);
      requireNumericObjectEqual(candidate.closestCompromise.worstVariant, result.closestCompromise.worstVariant,
        ["balancedAccuracy", "realRecall", "syntheticRecall", "syntheticRecallBySource", "metRecall"],
        `M3 candidate ${candidateIndex} closest metrics`);
      recomputed.push({ candidate, result });
      candidateIndex += 1;
    }
  }
  exactKeys(diagnostic.aggregate, ["candidateCount", "feasibleCandidateCount", "allFrozenSelectorCandidatesInfeasible",
    "minimumThresholdConflictLogit", "maximumThresholdConflictLogit",
    "bestWorstVariantSyntheticRecallWhileRealAndMetPass", "bestWorstVariantMetRecallWhileSyntheticAndFamilyPass",
    "closestCandidateParameters"], "M3 diagnostic aggregate");
  const feasibleCandidateCount = recomputed.filter(({ result }) => result.feasibleThresholds > 0).length;
  requireCondition(diagnostic.aggregate.candidateCount === 25 && feasibleCandidateCount === 0 &&
    diagnostic.aggregate.feasibleCandidateCount === 0 &&
    diagnostic.aggregate.allFrozenSelectorCandidatesInfeasible === true,
  "M3 selector diagnostic did not prove all candidates infeasible");
  const minimumGap = Math.min(...recomputed.map(({ result }) => result.thresholdConflict.infeasibleGapLogit));
  const maximumGap = Math.max(...recomputed.map(({ result }) => result.thresholdConflict.infeasibleGapLogit));
  const bestSynthetic = Math.max(...recomputed.map(({ result }) =>
    result.thresholdConflict.bestWorstVariantSyntheticRecallWhileRealAndMetPass));
  const bestMet = Math.max(...recomputed.map(({ result }) =>
    result.thresholdConflict.bestWorstVariantMetRecallWhileSyntheticAndFamilyPass));
  requireCondition(close(diagnostic.aggregate.minimumThresholdConflictLogit, minimumGap) &&
    close(diagnostic.aggregate.maximumThresholdConflictLogit, maximumGap) &&
    close(diagnostic.aggregate.bestWorstVariantSyntheticRecallWhileRealAndMetPass, bestSynthetic) &&
    close(diagnostic.aggregate.bestWorstVariantMetRecallWhileSyntheticAndFamilyPass, bestMet),
  "M3 failure aggregate metrics changed");
  requireCondition(jsonEqual(diagnostic.aggregate.closestCandidateParameters, { weightDecay: 0.03, upstreamBlendAlpha: 1 }),
    "M3 closest candidate identity changed");
  return diagnostic;
}

function expectedCachePaths() {
  return [
    "benchmark/candidates/prooflens-cf384-m3/feature-model.onnx",
    ...Array.from({ length: 55 }, (_, index) =>
      `benchmark/candidates/prooflens-cf384-m3/features/train-${String(index).padStart(5, "0")}.npz`),
    "benchmark/candidates/prooflens-cf384-m3/features/validation-00000.npz",
    "benchmark/candidates/prooflens-cf384-m3/features/regression-00000.npz",
    "benchmark/candidates/prooflens-cf384-m3/fresh-feature-run.json",
  ].sort();
}

export function validateM3FailureReceipt({ bytes, diagnosticBytes, expected }) {
  const receipt = parseCanonicalFailureJson(bytes, "M3 failure receipt");
  exactKeys(receipt, ["schemaVersion", "profile", "attemptId", "stage", "source", "inputBindings",
    "operatorObservation", "cacheSnapshot", "diagnostic", "absence", "h3Observation", "shippedModel",
    "decision", "limitations"], "M3 failure receipt");
  requireCondition(receipt.schemaVersion === 1 && receipt.profile === "m3" &&
    receipt.attemptId === "m3-failed-training-attempt-1" && receipt.stage === "m3-failed",
  "M3 failure receipt identity changed");
  exactKeys(receipt.source, ["commit", "tree", "parent"], "M3 failure receipt source");
  requireCondition(jsonEqual(receipt.source, {
    commit: expected.sourceCommit, tree: expected.sourceTree, parent: expected.baseCommit,
  }), "M3 failure receipt source changed");
  exactKeys(receipt.inputBindings, ["upstreamModelSha256", "trainerSha256", "diagnosticGeneratorSha256",
    "recipeSha256", "selectionSummarySha256", "selectorManifestSha256", "regressionManifestSha256",
    "commandArguments"], "M3 failure receipt bindings");
  for (const key of ["upstreamModelSha256", "trainerSha256", "diagnosticGeneratorSha256", "recipeSha256",
    "selectionSummarySha256", "selectorManifestSha256", "regressionManifestSha256"]) {
    requireCondition(receipt.inputBindings[key] === expected.inputBindings[key],
      `M3 failure receipt ${key} changed`);
  }
  requireCondition(jsonEqual(receipt.inputBindings.commandArguments, expected.commandArguments) &&
    !receipt.inputBindings.commandArguments.some((value) => String(value).toLowerCase().includes("h3")),
  "M3 failure command changed or accepted H3");
  exactKeys(receipt.operatorObservation, ["evidenceClass", "exitCode", "terminalExceptionType", "terminalMessage",
    "durableStdoutCaptured", "limitation"], "M3 failure operator observation");
  requireCondition(receipt.operatorObservation.evidenceClass === "operator-observed" &&
    receipt.operatorObservation.exitCode === 1 && receipt.operatorObservation.terminalExceptionType === "RuntimeError" &&
    receipt.operatorObservation.terminalMessage === "No trained candidate" &&
    receipt.operatorObservation.durableStdoutCaptured === false,
  "M3 failure operator observation changed");
  exactKeys(receipt.cacheSnapshot, ["evidenceClass", "root", "runId", "marker", "fileCount", "totalBytes",
    "inventory", "persistenceRequired"], "M3 failure cache snapshot");
  requireCondition(receipt.cacheSnapshot.evidenceClass === "captured-local-artifact-inventory" &&
    receipt.cacheSnapshot.root === "benchmark/candidates/prooflens-cf384-m3" &&
    receipt.cacheSnapshot.runId === expected.inputBindings.runId && receipt.cacheSnapshot.fileCount === 59 &&
    receipt.cacheSnapshot.totalBytes === expected.cacheBytes && receipt.cacheSnapshot.persistenceRequired === false,
  "M3 failure cache snapshot changed");
  exactKeys(receipt.cacheSnapshot.marker, ["path", "sha256", "bytes", "state", "payload"],
    "M3 failure cache marker");
  requireCondition(receipt.cacheSnapshot.marker.path ===
    "benchmark/candidates/prooflens-cf384-m3/fresh-feature-run.json" &&
    receipt.cacheSnapshot.marker.sha256 === expected.markerSha256 && receipt.cacheSnapshot.marker.bytes === 1_258 &&
    receipt.cacheSnapshot.marker.state === "extracting" &&
    receipt.cacheSnapshot.marker.payload.runId === expected.inputBindings.runId &&
    receipt.cacheSnapshot.marker.payload.state === "extracting",
  "M3 failure cache marker changed");
  requireCondition(Array.isArray(receipt.cacheSnapshot.inventory) && receipt.cacheSnapshot.inventory.length === 59,
    "M3 failure cache inventory changed");
  const paths = expectedCachePaths();
  let bytesTotal = 0;
  for (const [index, item] of receipt.cacheSnapshot.inventory.entries()) {
    exactKeys(item, ["path", "bytes", "sha256"], `M3 cache inventory ${index}`);
    requireCondition(item.path === paths[index] && Number.isInteger(item.bytes) && item.bytes > 0 &&
      /^[a-f0-9]{64}$/u.test(item.sha256), `M3 cache inventory ${index} changed`);
    bytesTotal += item.bytes;
  }
  requireCondition(bytesTotal === receipt.cacheSnapshot.totalBytes, "M3 cache inventory byte total changed");
  exactKeys(receipt.diagnostic, ["path", "sha256", "method", "candidateCount", "feasibleCandidateCount",
    "allFrozenSelectorCandidatesInfeasible", "originalConsoleReplacement"], "M3 failure diagnostic receipt");
  requireCondition(receipt.diagnostic.path === "benchmark/evidence/m3/failed-selector-diagnostic-1.json" &&
    receipt.diagnostic.sha256 === digest(diagnosticBytes) && receipt.diagnostic.candidateCount === 25 &&
    receipt.diagnostic.feasibleCandidateCount === 0 &&
    receipt.diagnostic.allFrozenSelectorCandidatesInfeasible === true &&
    receipt.diagnostic.originalConsoleReplacement === false,
  "M3 failure diagnostic receipt changed");
  exactKeys(receipt.absence, ["candidateSelectionCompleted", "candidateOutputsAbsent", "publicationOutputsAbsent",
    "successfulCandidateOutputsPresent", "publicationLockPresent", "successfulM3PublicationEvidencePresent", "h3AcceptedAsInput",
    "trackedH3ScoreArtifactsPresent"], "M3 failure absence record");
  requireCondition(receipt.absence.candidateSelectionCompleted === false &&
    jsonEqual(receipt.absence.candidateOutputsAbsent, expected.candidateOutputs) &&
    jsonEqual(receipt.absence.publicationOutputsAbsent, expected.publicationOutputs) &&
    receipt.absence.successfulCandidateOutputsPresent === false && receipt.absence.publicationLockPresent === false &&
    receipt.absence.successfulM3PublicationEvidencePresent === false && receipt.absence.h3AcceptedAsInput === false &&
    receipt.absence.trackedH3ScoreArtifactsPresent === false,
  "M3 failure absence record changed");
  exactKeys(receipt.h3Observation, ["evidenceClass", "h3PixelsReadOrScored", "machineVerifiableBoundary"],
    "M3 H3 observation");
  requireCondition(receipt.h3Observation.evidenceClass === "operator-observed" &&
    receipt.h3Observation.h3PixelsReadOrScored === false,
  "M3 H3 observation changed");
  exactKeys(receipt.shippedModel, ["path", "sha256", "bytes", "retained"], "M3 shipped model");
  requireCondition(jsonEqual(receipt.shippedModel, {
    path: "weights/prooflens-cf384.onnx", sha256: expected.inputBindings.upstreamModelSha256,
    bytes: 87_442_080, retained: true,
  }), "M3 shipped model record changed");
  exactKeys(receipt.decision, ["status", "selectedCandidate", "thresholdChanged", "gatesChanged", "modelPublished",
    "nextAttemptMayReuseThisSelectorForSelection"], "M3 failure decision");
  requireCondition(receipt.decision.status === "terminal-frozen-selector-failure" &&
    receipt.decision.selectedCandidate === null && receipt.decision.thresholdChanged === false &&
    receipt.decision.gatesChanged === false && receipt.decision.modelPublished === false &&
    receipt.decision.nextAttemptMayReuseThisSelectorForSelection === false,
  "M3 failure decision changed");
  requireCondition(Array.isArray(receipt.limitations) && receipt.limitations.length === 4,
    "M3 failure limitations changed");
  return receipt;
}

export const M3_FAILURE_VARIANTS = VARIANTS;
export const M3_FAILURE_PARAMETERS = DECAYS.flatMap((weightDecay) =>
  ALPHAS.map((upstreamBlendAlpha) => ({ weightDecay, upstreamBlendAlpha })));
