#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { get } from "node:https";
import { dirname } from "node:path";
import { TextDecoder } from "node:util";
import {
  M5_A5_AUTHORIZATION_PATH, M5_A5_COMMIT, M5_A5_SHA256, M5_A5_STATUS, M5_A5_TREE,
  M5_A6_AUTHORIZATION_PATH, M5_A6_STATUS, M5_P2_COMMIT, M5_P2_TREE, M5_R6_EXPECTED,
  matchesExpectedRows,
} from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git, m5GitBytes } from "./m5-safe-git.mjs";

const WORKFLOW_PATH = ".github/workflows/quality.yml";
const git = (arguments_) => m5Git(arguments_);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const canonical = (value) => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
  ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value;
const rows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, pathname, extra] = line.split("\t"); if (!status || !pathname || extra !== undefined) throw new Error("Malformed M5 commit row"); return [pathname, status]; });
const parents = (commit) => git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
const exactKeys = (value, keys) => value && !Array.isArray(value) && typeof value === "object" && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort());
const parseCanonical = (raw, label) => { const value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw)); if (!raw.equals(Buffer.from(`${JSON.stringify(canonical(value))}\n`, "utf8"))) throw new Error(`${label} is not canonical strict UTF-8 JSON`); return value; };
const getJson = (url) => new Promise((resolve, reject) => {
  const request = get(url, { headers: { Accept: "application/vnd.github+json", "User-Agent": "seroslop-m5-authorizer" } }, (response) => {
    const chunks = []; response.on("data", (chunk) => chunks.push(chunk)); response.on("end", () => {
      if (response.statusCode !== 200) return reject(new Error(`Unable to verify public M5 source CI: HTTP ${response.statusCode}`));
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch (error) { reject(error); }
    });
  }); request.setTimeout(30_000, () => request.destroy(new Error("Public M5 authorization verification timed out"))); request.on("error", reject);
});

if (process.argv.length !== 2) throw new Error("M5 authorization accepts no arguments");
if (!existsSync(M5_A5_AUTHORIZATION_PATH) || existsSync(M5_A6_AUTHORIZATION_PATH)) throw new Error("M5 R6 authorization requires inherited A5 and absent A6 receipts");
assertM5WorktreeExact();
const head = git(["rev-parse", "HEAD"]);
if (parents(head).length !== 1 || parents(head)[0] !== M5_A5_COMMIT || !matchesExpectedRows(rows(head), M5_R6_EXPECTED) ||
    git(["rev-parse", `${M5_A5_COMMIT}^{tree}`]) !== M5_A5_TREE ||
    !matchesExpectedRows(rows(M5_A5_COMMIT), new Map([[M5_A5_AUTHORIZATION_PATH, "A"]])) ||
    sha256(m5GitBytes(["show", `${M5_A5_COMMIT}:${M5_A5_AUTHORIZATION_PATH}`])) !== M5_A5_SHA256) {
  throw new Error("M5 R6 authorization lineage changed");
}
const a5Raw = m5GitBytes(["show", `${M5_A5_COMMIT}:${M5_A5_AUTHORIZATION_PATH}`]);
const a5 = parseCanonical(a5Raw, "M5 A5 authorization");
if (!exactKeys(a5, ["authorizationPath","diagnosticSha256","h3PixelsRead","parityBoundary","priorAuthorizationCommit","priorAuthorizationPath","priorAuthorizationSha256","priorAuthorizationTree","protocolCommit","protocolTree","schemaVersion","scoreBlind","sourceCommit","sourcePathMap","sourcePublicCi","sourceTree","status"]) ||
    a5.schemaVersion !== 5 || a5.status !== M5_A5_STATUS || a5.protocolCommit !== M5_P2_COMMIT || a5.protocolTree !== M5_P2_TREE ||
    a5.authorizationPath !== M5_A5_AUTHORIZATION_PATH || a5.scoreBlind !== true || a5.h3PixelsRead !== false) throw new Error("M5 inherited A5 receipt binding changed");

const publicReference = await getJson("https://api.github.com/repos/baney75/prooflens/git/ref/heads/main");
if (publicReference.object?.sha !== head) throw new Error("M5 R6 source must be public main before authorization");
const payload = await getJson(`https://api.github.com/repos/baney75/prooflens/actions/runs?event=push&head_sha=${head}&per_page=100`);
const run = payload.workflow_runs?.find((candidate) => candidate.head_sha === head && candidate.event === "push" && candidate.status === "completed" && candidate.conclusion === "success" && candidate.path === WORKFLOW_PATH);
if (!run) throw new Error("M5 R6 requires exact-head successful public quality CI");
const sourceTree = git(["rev-parse", `${head}^{tree}`]);
const receipt = canonical({
  schemaVersion: 6, status: M5_A6_STATUS,
  protocolCommit: M5_P2_COMMIT, protocolTree: M5_P2_TREE,
  priorAuthorizationCommit: M5_A5_COMMIT, priorAuthorizationTree: M5_A5_TREE,
  priorAuthorizationPath: M5_A5_AUTHORIZATION_PATH, priorAuthorizationSha256: M5_A5_SHA256,
  sourceCommit: head, sourceTree,
  sourcePathMap: [...M5_R6_EXPECTED.keys()].sort().map((pathname) => ({ path: pathname, sha256: sha256(m5GitBytes(["show", `${head}:${pathname}`])) })),
  runtimeBoundary: "trusted-runpod-execution-child-environment-before-torch-import", cublasWorkspaceConfig: ":4096:8",
  sourcePublicCi: { conclusion: run.conclusion, event: run.event, headSha: run.head_sha, runId: run.id, status: run.status, url: run.html_url, workflowPath: run.path },
  authorizationPath: M5_A6_AUTHORIZATION_PATH, scoreBlind: true, h3PixelsRead: false,
});
mkdirSync(dirname(M5_A6_AUTHORIZATION_PATH), { recursive: true });
writeFileSync(M5_A6_AUTHORIZATION_PATH, `${JSON.stringify(receipt)}\n`, { flag: "wx" });
console.log(JSON.stringify({ sourceCommit: head, sourceTree, runId: run.id, path: M5_A6_AUTHORIZATION_PATH, policy: "written" }));
