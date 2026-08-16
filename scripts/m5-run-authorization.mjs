#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { get } from "node:https";
import { dirname } from "node:path";
import {
  M5_CI_RECOVERY_COMMIT,
  M5_CI_RECOVERY_TREE,
  M5_FAILED_SOURCE_COMMIT,
  M5_FAILED_SOURCE_TREE,
  M5_P2_COMMIT,
  M5_P2_TREE,
  M5_P4_AUTHORIZATION_SHA256,
  M5_P4_COMMIT,
  M5_P4_TREE,
  M5_NUMERIC_AUDIT_AUTHORIZATION_PATH,
  M5_A5_AUTHORIZATION_PATH,
  M5_A5_STATUS,
  M5_A4_COMMIT,
  M5_A4_TREE,
  M5_A4_PATH,
  M5_A4_SHA256,
  M5_R5_EXPECTED,
  M5_RUN_AUTHORIZATION_PATH,
  M5_RUNPOD_ENV_AUTHORIZATION_COMMIT,
  M5_RUNPOD_ENV_AUTHORIZATION_PATH,
  M5_RUNPOD_ENV_AUTHORIZATION_SHA256,
  M5_RUNPOD_ENV_RECOVERY_COMMIT,
  M5_RUNPOD_ENV_RECOVERY_EXPECTED,
  M5_RUNPOD_ENV_RECOVERY_TREE,
  M5_RUNTIME_AUTHORIZATION_COMMIT,
  M5_RUNTIME_AUTHORIZATION_PATH,
  M5_RUNTIME_AUTHORIZATION_SHA256,
  M5_RUNTIME_AUTHORIZATION_TREE,
  M5_RUNTIME_RECOVERY_COMMIT,
  M5_RUNTIME_RECOVERY_EXPECTED,
  M5_RUNTIME_RECOVERY_TREE,
  M5_SOURCE_CI_RECOVERY_EXPECTED,
  M5_SOURCE_RECOVERY_EXPECTED,
  matchesExpectedRows,
} from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git, m5GitBytes } from "./m5-safe-git.mjs";

const WORKFLOW_PATH = ".github/workflows/quality.yml";
const git = (args) => m5Git(args);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const canonical = (value) => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  return value;
};
const rows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
const parents = (commit) => git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
const getJson = (url) => new Promise((resolve, reject) => {
  const request = get(url, { headers: { Accept: "application/vnd.github+json", "User-Agent": "seroslop-m5-authorizer" } }, (response) => {
    const chunks = [];
    response.on("data", (chunk) => chunks.push(chunk));
    response.on("end", () => {
      if (response.statusCode !== 200) return reject(new Error(`Unable to verify public M5 source CI: HTTP ${response.statusCode}`));
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch (error) { reject(error); }
    });
  });
  request.on("error", reject);
});

if (process.argv.length !== 2) throw new Error("M5 run authorization does not accept caller-selected output paths");
if (!existsSync(M5_NUMERIC_AUDIT_AUTHORIZATION_PATH) || existsSync(M5_A5_AUTHORIZATION_PATH)) {
  throw new Error("M5 parity recovery requires the committed A4 receipt and no active A5 authorization");
}
assertM5WorktreeExact();
const head = git(["rev-parse", "HEAD"]);
if (parents(head).length !== 1 || parents(head)[0] !== M5_A4_COMMIT ||
    !matchesExpectedRows(rows(head), M5_R5_EXPECTED) ||
    git(["rev-parse", `${M5_A4_COMMIT}^{tree}`]) !== M5_A4_TREE ||
    sha256(m5GitBytes(["show", `${M5_A4_COMMIT}:${M5_A4_PATH}`])) !== M5_A4_SHA256 ||
    parents(M5_RUNPOD_ENV_AUTHORIZATION_COMMIT).length !== 1 || parents(M5_RUNPOD_ENV_AUTHORIZATION_COMMIT)[0] !== M5_RUNPOD_ENV_RECOVERY_COMMIT ||
    !matchesExpectedRows(rows(M5_RUNPOD_ENV_AUTHORIZATION_COMMIT), new Map([[M5_RUNPOD_ENV_AUTHORIZATION_PATH, "A"]])) ||
    sha256(m5GitBytes(["show", `${M5_RUNPOD_ENV_AUTHORIZATION_COMMIT}:${M5_RUNPOD_ENV_AUTHORIZATION_PATH}`])) !== M5_RUNPOD_ENV_AUTHORIZATION_SHA256 ||
    git(["rev-parse", `${M5_RUNPOD_ENV_RECOVERY_COMMIT}^{tree}`]) !== M5_RUNPOD_ENV_RECOVERY_TREE ||
    parents(M5_RUNPOD_ENV_RECOVERY_COMMIT).length !== 1 || parents(M5_RUNPOD_ENV_RECOVERY_COMMIT)[0] !== M5_RUNTIME_AUTHORIZATION_COMMIT ||
    !matchesExpectedRows(rows(M5_RUNPOD_ENV_RECOVERY_COMMIT), M5_RUNPOD_ENV_RECOVERY_EXPECTED) ||
    git(["rev-parse", `${M5_RUNTIME_AUTHORIZATION_COMMIT}^{tree}`]) !== M5_RUNTIME_AUTHORIZATION_TREE ||
    parents(M5_RUNTIME_AUTHORIZATION_COMMIT).length !== 1 || parents(M5_RUNTIME_AUTHORIZATION_COMMIT)[0] !== M5_RUNTIME_RECOVERY_COMMIT ||
    !matchesExpectedRows(rows(M5_RUNTIME_AUTHORIZATION_COMMIT), new Map([[M5_RUNTIME_AUTHORIZATION_PATH, "A"]])) ||
    sha256(m5GitBytes(["show", `${M5_RUNTIME_AUTHORIZATION_COMMIT}:${M5_RUNTIME_AUTHORIZATION_PATH}`])) !== M5_RUNTIME_AUTHORIZATION_SHA256 ||
    git(["rev-parse", `${M5_RUNTIME_RECOVERY_COMMIT}^{tree}`]) !== M5_RUNTIME_RECOVERY_TREE ||
    parents(M5_RUNTIME_RECOVERY_COMMIT).length !== 1 || parents(M5_RUNTIME_RECOVERY_COMMIT)[0] !== M5_P4_COMMIT ||
    !matchesExpectedRows(rows(M5_RUNTIME_RECOVERY_COMMIT), M5_RUNTIME_RECOVERY_EXPECTED) ||
    git(["rev-parse", `${M5_P4_COMMIT}^{tree}`]) !== M5_P4_TREE ||
    parents(M5_P4_COMMIT).length !== 1 || parents(M5_P4_COMMIT)[0] !== M5_CI_RECOVERY_COMMIT ||
    !matchesExpectedRows(rows(M5_P4_COMMIT), new Map([[M5_RUN_AUTHORIZATION_PATH, "A"]])) ||
    sha256(m5GitBytes(["show", `${M5_P4_COMMIT}:${M5_RUN_AUTHORIZATION_PATH}`])) !== M5_P4_AUTHORIZATION_SHA256 ||
    git(["rev-parse", `${M5_CI_RECOVERY_COMMIT}^{tree}`]) !== M5_CI_RECOVERY_TREE ||
    parents(M5_CI_RECOVERY_COMMIT).length !== 1 || parents(M5_CI_RECOVERY_COMMIT)[0] !== M5_FAILED_SOURCE_COMMIT ||
    !matchesExpectedRows(rows(M5_CI_RECOVERY_COMMIT), M5_SOURCE_CI_RECOVERY_EXPECTED) ||
    git(["rev-parse", `${M5_FAILED_SOURCE_COMMIT}^{tree}`]) !== M5_FAILED_SOURCE_TREE ||
    parents(M5_FAILED_SOURCE_COMMIT).length !== 1 || parents(M5_FAILED_SOURCE_COMMIT)[0] !== M5_P2_COMMIT ||
    git(["rev-parse", `${M5_P2_COMMIT}^{tree}`]) !== M5_P2_TREE ||
    !matchesExpectedRows(rows(M5_FAILED_SOURCE_COMMIT), M5_SOURCE_RECOVERY_EXPECTED)) {
  throw new Error("M5 parity recovery history changed outside its exact authorized surfaces");
}

const publicReference = await getJson("https://api.github.com/repos/baney75/prooflens/git/ref/heads/main");
if (publicReference.object?.sha !== head) throw new Error("M5 parity recovery must be public main before authorization");
const payload = await getJson(`https://api.github.com/repos/baney75/prooflens/actions/runs?event=push&head_sha=${head}&per_page=100`);
const run = payload.workflow_runs?.find((candidate) => candidate.head_sha === head && candidate.event === "push" &&
  candidate.status === "completed" && candidate.conclusion === "success" && candidate.path === WORKFLOW_PATH);
if (!run) throw new Error("M5 parity recovery requires an exact-head successful public quality run");

const sourceTree = git(["rev-parse", `${head}^{tree}`]);
const receipt = canonical({
  schemaVersion: 5,
  status: M5_A5_STATUS,
  protocolCommit: M5_P2_COMMIT,
  protocolTree: M5_P2_TREE,
  priorAuthorizationCommit: M5_A4_COMMIT,
  priorAuthorizationTree: M5_A4_TREE,
  priorAuthorizationPath: M5_A4_PATH,
  priorAuthorizationSha256: M5_A4_SHA256,
  sourceCommit: head,
  sourceTree,
  sourcePathMap: [...M5_R5_EXPECTED.keys()].sort().map((path) => ({ path, sha256: sha256(m5GitBytes(["show", `${head}:${path}`])) })),
  sourcePublicCi: {
    conclusion: run.conclusion,
    event: run.event,
    headSha: run.head_sha,
    runId: run.id,
    status: run.status,
    url: run.html_url,
    workflowPath: run.path,
  },
  parityBoundary: "packaged-m2-reference-preserved-real-input-parity-and-onnx-scoring",
  diagnosticSha256: "c9c673efa0b1a6e4ea79b195ec16c71ae8ac91f962390a49c4e570b6d8de5c11",
  authorizationPath: M5_A5_AUTHORIZATION_PATH,
  scoreBlind: true,
  h3PixelsRead: false,
});
mkdirSync(dirname(M5_A5_AUTHORIZATION_PATH), { recursive: true });
writeFileSync(M5_A5_AUTHORIZATION_PATH, `${JSON.stringify(receipt)}\n`, { flag: "wx" });
console.log(JSON.stringify({ sourceCommit: head, sourceTree, runId: run.id, path: M5_A5_AUTHORIZATION_PATH, policy: "written" }));
