/* global AbortSignal, fetch */
import { createHash } from "node:crypto";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { digestFileStreaming, digestGitBlob } from "./git-object-digest.mjs";
import { freezeReceiptAdditions, isUnexpectedFreezeReceipt } from "./release-stage-policy.mjs";

const FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze-v2.json";
const LEGACY_FREEZE_PATH = "benchmark/evidence/evaluation/pre-score-freeze.json";
const LEGACY_SOURCE_COMMIT = "0771a9422b552e2023e5150fb6c8b4238b811a74";
const LEGACY_FREEZE_COMMIT = "2bd0c4757f6059c57879414a5dba77629d66460e";
const LEGACY_FREEZE_SHA256 = "400fd2b7a7cd84b063f81799eaf3829f770220c41f58dff53e7caebd1a145c34";
const RECOVERY_REASON = "legacy-verifier-resource-exhaustion";
const PUBLIC_GIT_URL = "https://github.com/baney75/prooflens.git";
const PUBLIC_RAW_BASE_URL = "https://raw.githubusercontent.com/baney75/prooflens";
const PUBLIC_PROOF_METHOD =
  "credential-free HTTPS git ls-remote plus unauthenticated raw byte match";
const CANONICAL_ORIGIN_URLS = new Set([
  "https://github.com/baney75/prooflens",
  "https://github.com/baney75/prooflens.git",
  "git@github.com:baney75/prooflens.git",
  "ssh://git@github.com/baney75/prooflens.git",
]);
const EXPECTED_ALLOWED_PATHS = [
  "artifacts/**",
  "benchmark/evidence/evaluation/**",
  "BENCHMARK.md",
  "MODEL_CARD.md",
  "README.md",
  "docs/ACCEPTANCE.md",
];
const EXPECTED_IMMUTABLE_FILES = [
  "benchmark/bootstrap_ci.py",
  "benchmark/bootstrap_fpr.py",
  "benchmark/evaluate.py",
  "benchmark/evaluation_contract.py",
  "benchmark/large/recipe.json",
  "benchmark/manifests/test.jsonl",
  "benchmark/manifests/validation.jsonl",
  "benchmark/manifests/web-negative.jsonl",
  "benchmark/prediction_contract.py",
  "benchmark/run_release_replay.py",
  "benchmark/verify_evaluation_evidence.py",
  "benchmark/write_pre_score_freeze.py",
  "model-lock.json",
  "scripts/check-benchmark-evidence.mjs",
  "scripts/check-pre-score-freeze.mjs",
  "scripts/git-object-digest.mjs",
  "scripts/release-stage-policy.mjs",
  "src/inference/calibration.ts",
  "src/inference/detector.ts",
  "src/shared/model-spec.ts",
  "weights/prooflens-cf384.onnx",
];
const EXPECTED_RECOVERY_REPAIR_PATHS = [
  "BENCHMARK.md",
  "README.md",
  "benchmark/evaluate.py",
  "benchmark/evaluation_contract.py",
  "benchmark/test_integrity_contracts.py",
  "benchmark/verify_evaluation_evidence.py",
  "benchmark/write_pre_score_freeze.py",
  "package.json",
  "scripts/check-benchmark-evidence.mjs",
  "scripts/check-pre-score-freeze.mjs",
  "scripts/check-pre-score-stage.mjs",
  "scripts/git-object-digest.mjs",
  "scripts/release-stage-policy.mjs",
  "scripts/run-static-verification.mjs",
  "scripts/test-git-object-digest.mjs",
  "scripts/test-release-stage-policy.mjs",
];
const POST_SCORE_PREFIXES = [
  "artifacts/browser-parity",
  "benchmark/evidence/evaluation/confirmatory/",
  "benchmark/evidence/evaluation/web-negative/",
];
const POST_SCORE_FILES = new Set([
  FREEZE_PATH,
  "benchmark/evidence/evaluation/replay-verification.json",
]);

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function git(gitArguments) {
  return execFileSync("git", gitArguments, { encoding: "utf8" }).trim();
}

function gitBytes(gitArguments) {
  return execFileSync("git", gitArguments);
}

function anonymousPublicHead() {
  const environment = { ...process.env };
  for (const key of Object.keys(environment)) {
    if (["GH_TOKEN", "GITHUB_TOKEN", "GIT_ASKPASS", "SSH_ASKPASS", "GIT_CONFIG_COUNT",
      "GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"].includes(key) ||
      key.startsWith("GIT_CONFIG_KEY_") || key.startsWith("GIT_CONFIG_VALUE_")) {
      delete environment[key];
    }
  }
  environment.GIT_CONFIG_GLOBAL = process.platform === "win32" ? "NUL" : "/dev/null";
  environment.GIT_CONFIG_NOSYSTEM = "1";
  environment.GIT_TERMINAL_PROMPT = "0";
  const workingDirectory = mkdtempSync(join(tmpdir(), "prooflens-anonymous-git-"));
  try {
    const output = execFileSync("git", [
      "-c", "credential.helper=",
      "-c", "core.askPass=",
      "-c", "http.extraHeader=",
      "-c", "http.https://github.com/.extraheader=",
      "ls-remote", PUBLIC_GIT_URL, "refs/heads/main",
    ], { cwd: workingDirectory, env: environment, encoding: "utf8", timeout: 30_000 }).trim();
    const rows = output.split(/\s+/u);
    requireCondition(rows.length === 2 && /^[a-f0-9]{40}$/u.test(rows[0]),
      "Could not resolve anonymous public main");
    return rows[0];
  } catch (error) {
    throw new Error("Canonical GitHub repository is not anonymously readable", { cause: error });
  } finally {
    rmSync(workingDirectory, { recursive: true, force: true });
  }
}

async function anonymousRawFile(commit, path) {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/");
  const url = `${PUBLIC_RAW_BASE_URL}/${commit}/${encodedPath}`;
  const response = await fetch(`${url}?prooflens=${commit}`, {
    headers: {
      Accept: "application/octet-stream",
      "Cache-Control": "no-cache",
      "User-Agent": "ProofLens-public-freeze-verifier/1",
    },
    redirect: "error",
    signal: AbortSignal.timeout(30_000),
  });
  requireCondition(response.status === 200,
    `Canonical GitHub file is not anonymously readable: ${path}`);
  return { bytes: Buffer.from(await response.arrayBuffer()), url };
}

function allowedPostScorePath(path) {
  return path.startsWith("artifacts/") || path.startsWith("benchmark/evidence/evaluation/") ||
    ["BENCHMARK.md", "MODEL_CARD.md", "README.md", "docs/ACCEPTANCE.md"].includes(path);
}

const freeze = JSON.parse(await readFile(FREEZE_PATH, "utf8"));
const freezeBytes = await readFile(FREEZE_PATH);
requireCondition(freeze.schemaVersion === 3 && freeze.generation === 2 && freeze.mode ===
  "public recovery pre-score source freeze before any confirmatory or web-negative inference" &&
  freeze.receiptPath === FREEZE_PATH &&
  freeze.repository === "https://github.com/baney75/prooflens" && freeze.branch === "main" &&
  /^[a-f0-9]{40}$/u.test(freeze.sourceCommit) && /^[a-f0-9]{40}$/u.test(freeze.sourceTree) &&
  freeze.remoteObservedHead === freeze.sourceCommit &&
  freeze.publicCommitUrl === `${freeze.repository}/commit/${freeze.sourceCommit}` &&
  Number.isFinite(Date.parse(freeze.remoteVerifiedAt)) &&
  JSON.stringify(freeze.allowedPostScorePaths) === JSON.stringify(EXPECTED_ALLOWED_PATHS),
"Pre-score public freeze metadata changed");
requireCondition(freeze.recovery?.reason === RECOVERY_REASON &&
  freeze.recovery.legacySourceCommit === LEGACY_SOURCE_COMMIT &&
  freeze.recovery.legacyFreezeCommit === LEGACY_FREEZE_COMMIT &&
  freeze.recovery.legacyReceiptPath === LEGACY_FREEZE_PATH &&
  freeze.recovery.legacyReceiptSha256 === LEGACY_FREEZE_SHA256 &&
  JSON.stringify(freeze.recovery.repairPaths) === JSON.stringify(EXPECTED_RECOVERY_REPAIR_PATHS) &&
  typeof freeze.recovery.repositoryEvidenceLimitation === "string" &&
  freeze.recovery.repositoryEvidenceLimitation.includes("cannot prove"),
"Pre-score recovery lineage changed");
requireCondition(freeze.anonymousPublicProof?.method === PUBLIC_PROOF_METHOD &&
  freeze.anonymousPublicProof.head === freeze.sourceCommit &&
  freeze.anonymousPublicProof.fileCommit === freeze.sourceCommit &&
  freeze.anonymousPublicProof.file === "model-lock.json" &&
  freeze.anonymousPublicProof.rawUrl ===
    `${PUBLIC_RAW_BASE_URL}/${freeze.sourceCommit}/model-lock.json` &&
  /^[a-f0-9]{64}$/u.test(freeze.anonymousPublicProof.fileSha256),
"Pre-score anonymous public proof changed");
requireCondition(git(["rev-parse", `${freeze.sourceCommit}^{tree}`]) === freeze.sourceTree,
  "Pre-score source tree does not match its commit");
requireCondition(git(["rev-parse", `${freeze.sourceCommit}^`]) === LEGACY_FREEZE_COMMIT &&
  git(["rev-parse", `${LEGACY_FREEZE_COMMIT}^`]) === LEGACY_SOURCE_COMMIT,
"Recovery source is not the direct child of the failed legacy freeze");
const legacyFreezePaths = git(["diff-tree", "--no-renames", "--no-commit-id", "--name-only", "-r", LEGACY_FREEZE_COMMIT])
  .split("\n").filter(Boolean);
requireCondition(JSON.stringify(legacyFreezePaths) === JSON.stringify([LEGACY_FREEZE_PATH]),
  "Legacy freeze commit shape changed");
const legacyCurrent = await readFile(LEGACY_FREEZE_PATH);
requireCondition(digest(legacyCurrent) === LEGACY_FREEZE_SHA256 &&
  digest(gitBytes(["show", `${LEGACY_FREEZE_COMMIT}:${LEGACY_FREEZE_PATH}`])) === LEGACY_FREEZE_SHA256,
"Legacy freeze receipt bytes changed");
const recoveryPaths = git(["diff", "--no-renames", "--name-only", `${LEGACY_FREEZE_COMMIT}..${freeze.sourceCommit}`])
  .split("\n").filter(Boolean).sort();
requireCondition(JSON.stringify(recoveryPaths) === JSON.stringify([...EXPECTED_RECOVERY_REPAIR_PATHS].sort()),
  "Recovery source changed outside its exact repair set");
const additionCommits = git(["log", "--no-renames", "--diff-filter=A", "--format=%H", "--", FREEZE_PATH])
  .split("\n").filter(Boolean);
requireCondition(additionCommits.length === 1,
  "Pre-score freeze must be added by exactly one immutable commit");
const freezeCommit = additionCommits[0];
const freezeLineage = git(["rev-list", "--parents", "-n", "1", freezeCommit]).split(" ");
requireCondition(freezeLineage.length === 2 && freezeLineage[0] === freezeCommit &&
  freezeLineage[1] === freeze.sourceCommit,
"Pre-score freeze commit must be a freeze-only child of the source commit");
const freezeCommitPaths = git(["diff-tree", "--no-renames", "--no-commit-id", "--name-only", "-r", freezeCommit])
  .split("\n").filter(Boolean);
requireCondition(JSON.stringify(freezeCommitPaths) === JSON.stringify([FREEZE_PATH]),
  "Pre-score freeze commit must add only the freeze receipt");
const addedFreezeReceipts = freezeReceiptAdditions(git([
  "log", "--no-renames", "--diff-filter=A", "--format=", "--name-only", "HEAD",
]).split("\n"));
requireCondition(JSON.stringify(addedFreezeReceipts) ===
  JSON.stringify([LEGACY_FREEZE_PATH, FREEZE_PATH].sort()),
"An alternate pre-score freeze receipt exists in public history");
requireCondition(Buffer.compare(gitBytes(["show", `${freezeCommit}:${FREEZE_PATH}`]), freezeBytes) === 0,
  "Pre-score freeze receipt changed after its public commit");
const ancestor = spawnSync("git", ["merge-base", "--is-ancestor", freezeCommit, "HEAD"]);
requireCondition(ancestor.status === 0, "Pre-score freeze commit is not an ancestor of HEAD");
const frozenPaths = new Set(git(["ls-tree", "-r", "--name-only", freeze.sourceCommit]).split("\n").filter(Boolean));
requireCondition(![...POST_SCORE_FILES].some((path) => frozenPaths.has(path)) &&
  ![...frozenPaths].some((path) => POST_SCORE_PREFIXES.some((prefix) => path.startsWith(prefix))),
"Pre-score source commit already contained post-score evaluation evidence");
const freezeCommitTreePaths = new Set(git(["ls-tree", "-r", "--name-only", freezeCommit])
  .split("\n").filter(Boolean));
requireCondition(![...freezeCommitTreePaths].some((path) =>
  POST_SCORE_PREFIXES.some((prefix) => path.startsWith(prefix))) &&
  !freezeCommitTreePaths.has("benchmark/evidence/evaluation/replay-verification.json"),
"Pre-score freeze commit already contained sealed evaluation output");
const headTreePaths = git(["ls-tree", "-r", "--name-only", "HEAD"]).split("\n").filter(Boolean);
requireCondition(!headTreePaths.some(isUnexpectedFreezeReceipt),
  "An alternate pre-score freeze receipt cannot supersede the public V2 freeze");
requireCondition(JSON.stringify(Object.keys(freeze.immutableFilesSha256).sort()) ===
  JSON.stringify([...EXPECTED_IMMUTABLE_FILES].sort()), "Pre-score immutable-file list changed");
for (const path of EXPECTED_IMMUTABLE_FILES) {
  const frozen = await digestGitBlob(`${freeze.sourceCommit}:${path}`);
  const current = await digestFileStreaming(path);
  requireCondition(frozen.sha256 === freeze.immutableFilesSha256[path],
    `Pre-score immutable hash is false: ${path}`);
  requireCondition(current.sha256 === freeze.immutableFilesSha256[path] && current.bytes === frozen.bytes,
    `Pre-score immutable file changed after scoring: ${path}`);
}
requireCondition(freeze.anonymousPublicProof.fileSha256 ===
  freeze.immutableFilesSha256["model-lock.json"],
"Pre-score anonymous source proof is not bound to model-lock.json");
const changedPaths = git(["diff", "--no-renames", "--name-only", `${freeze.sourceCommit}..HEAD`])
  .split("\n").filter(Boolean);
requireCondition(changedPaths.every(allowedPostScorePath),
  `Post-score commit changed a frozen path: ${changedPaths.filter((path) => !allowedPostScorePath(path)).join(", ")}`);
const worktreeStatus = git(["status", "--porcelain=v1", "--untracked-files=all"]);
requireCondition(worktreeStatus === "",
  "Final verification requires a completely clean index, worktree, and untracked-file set");
const originFetch = git(["remote", "get-url", "origin"]);
const originPush = git(["remote", "get-url", "--push", "origin"]);
requireCondition(CANONICAL_ORIGIN_URLS.has(originFetch) && CANONICAL_ORIGIN_URLS.has(originPush),
  "Public freeze requires the canonical baney75/prooflens GitHub origin");
const remoteRows = git(["ls-remote", "origin", "refs/heads/main"]).split(/\s+/u);
requireCondition(remoteRows.length === 2 && /^[a-f0-9]{40}$/u.test(remoteRows[0]),
  "Could not resolve public origin/main for the pre-score freeze");
const anonymousHead = anonymousPublicHead();
requireCondition(remoteRows[0] === anonymousHead,
  "Authenticated origin/main differs from anonymous public main");
const publicSourceFile = await anonymousRawFile(freeze.sourceCommit, "model-lock.json");
requireCondition(digest(publicSourceFile.bytes) === freeze.anonymousPublicProof.fileSha256 &&
  publicSourceFile.url === freeze.anonymousPublicProof.rawUrl,
"Anonymous source-commit model lock differs from its frozen proof");
const publicFreezeFile = await anonymousRawFile(freezeCommit, FREEZE_PATH);
requireCondition(Buffer.compare(publicFreezeFile.bytes, freezeBytes) === 0,
  "Freeze receipt is not anonymously readable at its immutable public commit");
const publicAncestor = spawnSync("git", ["merge-base", "--is-ancestor", freezeCommit, anonymousHead]);
requireCondition(publicAncestor.status === 0,
  "The freeze-only commit is not an ancestor of anonymous public main");
console.log(JSON.stringify({
  sourceCommit: freeze.sourceCommit,
  sourceTree: freeze.sourceTree,
  freezeCommit,
  legacyFreezeCommit: LEGACY_FREEZE_COMMIT,
  recoveryReason: RECOVERY_REASON,
  anonymousPublicHead: anonymousHead,
  remoteVerifiedAt: freeze.remoteVerifiedAt,
  changedPaths: changedPaths.length,
  policy: "pass",
}));
