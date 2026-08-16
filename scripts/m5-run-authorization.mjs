#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { get } from "node:https";
import { dirname } from "node:path";
import {
  M5_P2_COMMIT,
  M5_P2_TREE,
  M5_RUN_AUTHORIZATION_PATH,
  M5_SOURCE_RECOVERY_EXPECTED,
  matchesExpectedRows,
} from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git, m5GitBytes } from "./m5-safe-git.mjs";

const WORKFLOW_PATH = ".github/workflows/quality.yml";
const git = (args) => m5Git(args);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const canonical = (value) => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  }
  return value;
};
const rows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
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
if (existsSync(M5_RUN_AUTHORIZATION_PATH)) throw new Error("M5 run authorization already exists");
assertM5WorktreeExact();
const head = git(["rev-parse", "HEAD"]);
const parents = git(["rev-list", "--parents", "-n", "1", head]).split(" ").slice(1);
if (parents.length !== 1 || parents[0] !== M5_P2_COMMIT || git(["rev-parse", `${M5_P2_COMMIT}^{tree}`]) !== M5_P2_TREE) {
  throw new Error("M5 run authorization requires a direct P3 child of public P2");
}
if (!matchesExpectedRows(rows(head), M5_SOURCE_RECOVERY_EXPECTED)) {
  throw new Error("M5 source recovery changed outside its exact authorized surface");
}
const publicReference = await getJson("https://api.github.com/repos/baney75/prooflens/git/ref/heads/main");
if (publicReference.object?.sha !== head) {
  throw new Error("M5 source recovery must be the public main head before authorization");
}
const payload = await getJson(`https://api.github.com/repos/baney75/prooflens/actions/runs?event=push&head_sha=${head}&per_page=100`);
const run = payload.workflow_runs?.find((candidate) => candidate.head_sha === head && candidate.event === "push" &&
  candidate.status === "completed" && candidate.conclusion === "success" && candidate.path === WORKFLOW_PATH);
if (!run) throw new Error("M5 source recovery requires an exact-head successful public quality run");
const sourceTree = git(["rev-parse", `${head}^{tree}`]);
const sourcePathMap = [...M5_SOURCE_RECOVERY_EXPECTED.keys()].sort().map((path) => ({
  path,
  sha256: sha256(m5GitBytes(["show", `${head}:${path}`])),
}));
const receipt = canonical({
  schemaVersion: 1,
  status: "m5-source-recovery-authorized",
  protocolCommit: M5_P2_COMMIT,
  protocolTree: M5_P2_TREE,
  sourceCommit: head,
  sourceTree,
  sourcePathMap,
  sourcePublicCi: {
    conclusion: run.conclusion,
    event: run.event,
    headSha: run.head_sha,
    runId: run.id,
    status: run.status,
    url: run.html_url,
    workflowPath: run.path,
  },
  authorizationPath: M5_RUN_AUTHORIZATION_PATH,
  scoreBlind: true,
  h3PixelsRead: false,
});
mkdirSync(dirname(M5_RUN_AUTHORIZATION_PATH), { recursive: true });
writeFileSync(M5_RUN_AUTHORIZATION_PATH, `${JSON.stringify(receipt)}\n`, { flag: "wx" });
console.log(JSON.stringify({ sourceCommit: head, sourceTree, runId: run.id, path: M5_RUN_AUTHORIZATION_PATH, policy: "written" }));
