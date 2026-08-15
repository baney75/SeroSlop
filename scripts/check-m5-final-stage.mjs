import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import {
  M5_BASE_SOURCE_COMMIT,
  M5_BASE_SOURCE_TREE,
  M5_FAILURE_PATH,
  M5_FINAL_EXPECTED,
  M5_FINAL_RECEIPT_PATH,
  M5_LARGE_EVALUATION_PATH,
  M5_LARGE_SOURCE_EXPECTED,
  M5_LARGE_SOURCE_LOCK_PATH,
  M5_LOCK_EXPECTED,
  M5_SELECTION_LOCK_PATH,
  matchesExpectedRows,
  matchesM5ProtocolCommit,
} from "./m5-stage-policy.mjs";

const hex64 = /^[0-9a-f]{64}$/u;
const digest = (value) => createHash("sha256").update(value).digest("hex");
const fileDigest = (pathname) => digest(readFileSync(pathname));
const json = (pathname) => JSON.parse(readFileSync(pathname, "utf8"));
const equal = (left, right) => JSON.stringify(left) === JSON.stringify(right);

function git(arguments_) {
  return execFileSync("git", arguments_, { encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }).trim();
}

function rows(commit) {
  return git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, pathname, extra] = line.split("\t");
      if (!status || !pathname || extra !== undefined) throw new Error(`Malformed M5 final row: ${line}`);
      return [pathname, status];
    });
}

function requireCondition(value, message) {
  if (!value) throw new Error(message);
}

const head = git(["rev-parse", "HEAD"]);
const finalParents = git(["rev-list", "--parents", "-n", "1", head]).split(" ").slice(1);
requireCondition(finalParents.length === 1 && matchesExpectedRows(rows(head), M5_FINAL_EXPECTED),
  "M5 final publication is not the exact declared transaction");
const sourceLockCommit = finalParents[0];
const sourceLockParents = git(["rev-list", "--parents", "-n", "1", sourceLockCommit]).split(" ").slice(1);
requireCondition(sourceLockParents.length === 1 && matchesExpectedRows(rows(sourceLockCommit), M5_LARGE_SOURCE_EXPECTED),
  "M5 final publication is not the direct child of the exact 100K source lock");
const lockCommit = sourceLockParents[0];
const lockParents = git(["rev-list", "--parents", "-n", "1", lockCommit]).split(" ").slice(1);
requireCondition(lockParents.length === 1 && matchesExpectedRows(rows(lockCommit), M5_LOCK_EXPECTED),
  "M5 final publication is not the direct child of the one-file lock");
const protocol = lockParents[0];
const protocolParents = git(["rev-list", "--parents", "-n", "1", protocol]).split(" ").slice(1);
const baseTree = git(["rev-parse", `${M5_BASE_SOURCE_COMMIT}^{tree}`]);
requireCondition(matchesM5ProtocolCommit({ parents: protocolParents, rows: rows(protocol), parentTree: baseTree }) &&
  baseTree === M5_BASE_SOURCE_TREE, "M5 final publication has the wrong protocol ancestry");
requireCondition(!git(["status", "--porcelain=v1", "--untracked-files=all"]),
  "M5 final verification requires a completely clean repository");
requireCondition(!existsSync(M5_FAILURE_PATH) && !existsSync("docs/COMPETITOR_AUDIT.md"),
  "M5 final publication contains a forbidden failure or competitor audit");

execFileSync("python3", ["-c", [
  "from pathlib import Path",
  "from benchmark.m5.contracts import load_recipe,parse_json_bytes,read_jsonl,validate_regression_state,validate_selection_lock",
  "r=load_recipe(Path('benchmark/m5/recipe.json'))",
  `l=parse_json_bytes(Path('${M5_SELECTION_LOCK_PATH}').read_bytes(),label='selection lock')`,
  "rows=read_jsonl(Path(r['sourceEvidence']['selectorManifest']['path']))",
  "validate_selection_lock(l,r,rows)",
  `assert l['protocolCommit']=='${protocol}'`,
  "regression_path=Path('benchmark/evidence/m5/regression-summary.json')",
  "reg=parse_json_bytes(regression_path.read_bytes(),label='terminal regression summary')",
  `validate_regression_state(reg,r,l,lock_commit='${lockCommit}',selection_lock_sha256=__import__('hashlib').sha256(Path('${M5_SELECTION_LOCK_PATH}').read_bytes()).hexdigest())`,
  "from benchmark.m5.large_synthetic import verify_public_packet",
  "panel=verify_public_packet(r,verify_pixels=False)",
  `source_lock=parse_json_bytes(Path('${M5_LARGE_SOURCE_LOCK_PATH}').read_bytes(),label='large source lock')`,
  "assert source_lock['regressionStateSha256']==__import__('hashlib').sha256(regression_path.read_bytes()).hexdigest()",
  "from benchmark.m5.evaluate_large_synthetic import load_rows,validate_evaluation_receipt",
  `e=parse_json_bytes(Path('${M5_LARGE_EVALUATION_PATH}').read_bytes(),label='large-synthetic evaluation')`,
  `validate_evaluation_receipt(e,r,selection_lock=l,source_lock_commit='${sourceLockCommit}',panel_rows=load_rows(r),regression_state_path=regression_path)`,
  "assert e['status']=='large-synthetic-pass' and e['acceptanceEligible'] is True",
].join(";")], { stdio: "inherit" });

const lock = json(M5_SELECTION_LOCK_PATH);
const regression = json("benchmark/evidence/m5/regression-summary.json");
const largeEvaluation = json(M5_LARGE_EVALUATION_PATH);
const training = readFileSync("benchmark/evidence/m5/training-summary.json");
const calibration = json("benchmark/evidence/m5/calibration.json");
const comparison = readFileSync("benchmark/evidence/m5/model-comparison.json");
const model = readFileSync("weights/prooflens-cf384.onnx");
const modelLock = json("model-lock.json");
const receipt = json(M5_FINAL_RECEIPT_PATH);
requireCondition(regression.schemaVersion === 1 && regression.status === "regression-pass" &&
  regression.lockCommit === lockCommit && regression.selectionLockSha256 === fileDigest(M5_SELECTION_LOCK_PATH) &&
  regression.selectedCandidateId === lock.selectedCandidateId &&
  regression.selectedModelSha256 === lock.selectedModel.sha256 && regression.selectionInfluencedByRegression === false &&
  regression.h3PixelsRead === false && regression.selectorOnnxReplay?.passed === true &&
  Array.isArray(regression.results) && regression.results.length === 2 && regression.results.every((row) => row.passed === true),
  "M5 terminal regression evidence changed or failed");
requireCondition(largeEvaluation.status === "large-synthetic-pass" && largeEvaluation.acceptanceEligible === true &&
  largeEvaluation.sourceLockCommit === sourceLockCommit && largeEvaluation.selectionLockCommit === lockCommit &&
  largeEvaluation.sourceLockSha256 === fileDigest(M5_LARGE_SOURCE_LOCK_PATH) &&
  largeEvaluation.meanBatchRecall > 0.95 && largeEvaluation.medianBatchRecall > 0.95 &&
  largeEvaluation.items === 100000 && largeEvaluation.batches === 1000 && largeEvaluation.batchSize === 100 &&
  largeEvaluation.selectionInfluence === false && largeEvaluation.h3PixelsRead === false,
  "M5 100K synthetic evaluation changed or failed");
requireCondition(digest(training) === lock.trainingSummarySha256 &&
  equal(JSON.parse(training.toString("utf8")), lock.trainingSummary), "M5 public training summary changed");
requireCondition(equal(calibration, lock.calibration), "M5 final calibration changed");
requireCondition(model.byteLength === lock.selectedModel.bytes && digest(model) === lock.selectedModel.sha256,
  "M5 shipped model is not the selected model");
requireCondition(modelLock.artifact === "weights/prooflens-cf384.onnx" && modelLock.bytes === model.byteLength &&
  modelLock.sha256 === lock.selectedModel.sha256 && equal(modelLock.calibration, calibration),
  "M5 model lock does not bind the selected model and calibration");
const receiptKeys = [
  "acceptanceEligible", "calibrationSha256", "h3HoldoutScored", "modelComparisonSha256",
  "protocolCommit", "publicationRows", "publishedSha256", "regressionSummarySha256", "schemaVersion",
  "largeSyntheticEvaluationSha256", "largeSyntheticSourceLockCommit", "largeSyntheticSourceLockSha256",
  "selectedCandidateId", "selectedModelSha256", "selectionLockCommit", "selectionLockSha256", "shippedModel",
  "status", "trainingSummarySha256",
].sort();
requireCondition(equal(Object.keys(receipt).sort(), receiptKeys) && receipt.schemaVersion === 1 &&
  receipt.status === "m5-finalized" && receipt.acceptanceEligible === false && receipt.h3HoldoutScored === false &&
  receipt.protocolCommit === protocol && receipt.selectionLockCommit === lockCommit &&
  receipt.selectionLockSha256 === fileDigest(M5_SELECTION_LOCK_PATH) &&
  receipt.selectedCandidateId === lock.selectedCandidateId && receipt.selectedModelSha256 === lock.selectedModel.sha256 &&
  receipt.trainingSummarySha256 === digest(training) && receipt.regressionSummarySha256 === fileDigest("benchmark/evidence/m5/regression-summary.json") &&
  receipt.largeSyntheticEvaluationSha256 === fileDigest(M5_LARGE_EVALUATION_PATH) &&
  receipt.largeSyntheticSourceLockCommit === sourceLockCommit &&
  receipt.largeSyntheticSourceLockSha256 === fileDigest(M5_LARGE_SOURCE_LOCK_PATH) &&
  receipt.calibrationSha256 === fileDigest("benchmark/evidence/m5/calibration.json") &&
  receipt.modelComparisonSha256 === digest(comparison) && receipt.shippedModel?.path === "weights/prooflens-cf384.onnx" &&
  receipt.shippedModel?.bytes === model.byteLength && receipt.shippedModel?.sha256 === digest(model),
  "M5 finalization receipt changed");
requireCondition(equal(receipt.publicationRows, [...M5_FINAL_EXPECTED.entries()]), "M5 final receipt path map changed");
const expectedPublished = [...M5_FINAL_EXPECTED.keys()].filter((pathname) => pathname !== M5_FINAL_RECEIPT_PATH).sort();
requireCondition(equal(Object.keys(receipt.publishedSha256).sort(), expectedPublished) &&
  expectedPublished.every((pathname) => hex64.test(receipt.publishedSha256[pathname]) &&
    receipt.publishedSha256[pathname] === fileDigest(pathname)), "M5 final published-byte map changed");
console.log(JSON.stringify({ head, protocol, lockCommit, modelSha256: digest(model), policy: "pass" }));
