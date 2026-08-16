import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { TextDecoder } from "node:util";
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
  M5_RUN_AUTHORIZATION_PATH,
  M5_RUNPOD_ENV_AUTHORIZATION_PATH,
  M5_RUNPOD_ENV_RECOVERY_EXPECTED,
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
import { m5Git, m5GitBytes } from "./m5-safe-git.mjs";

const git = (args) => m5Git(args);
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const rows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
const parents = (commit) => git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
const canonical = (value) => {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  return value;
};
const exactKeys = (value, expected) => value && typeof value === "object" && !Array.isArray(value) &&
  JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...expected].sort());

export function parseCanonicalM5Authorization(raw) {
  const text = new TextDecoder("utf-8", { fatal: true }).decode(raw);
  const receipt = JSON.parse(text);
  const expectedBytes = Buffer.from(`${JSON.stringify(canonical(receipt))}\n`, "utf8");
  if (!raw.equals(expectedBytes)) throw new Error("M5 authorization is not canonical strict UTF-8 JSON");
  return receipt;
}

export function requireM5AuthorizationSchema(receipt) {
  if (!exactKeys(receipt, [
    "authorizationPath", "environmentBoundary", "h3PixelsRead", "protocolCommit", "protocolTree", "schemaVersion", "scoreBlind",
    "priorAuthorizationCommit", "priorAuthorizationPath", "priorAuthorizationSha256", "priorAuthorizationTree",
    "sourceCommit", "sourcePathMap", "sourcePublicCi", "sourceTree", "status",
  ]) || !exactKeys(receipt.sourcePublicCi, ["conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"])) {
    throw new Error("M5 RunPod environment authorization schema changed");
  }
}

export function validateM5AuthorizedChain() {
  const oldP4Additions = git(["log", "--first-parent", "--no-renames", "--diff-filter=A", "--format=%H", "--", M5_RUN_AUTHORIZATION_PATH])
    .split("\n").filter(Boolean);
  if (oldP4Additions.length !== 1 || oldP4Additions[0] !== M5_P4_COMMIT ||
      !matchesExpectedRows(rows(M5_P4_COMMIT), new Map([[M5_RUN_AUTHORIZATION_PATH, "A"]])) ||
      git(["rev-parse", `${M5_P4_COMMIT}^{tree}`]) !== M5_P4_TREE ||
      digest(m5GitBytes(["show", `${M5_P4_COMMIT}:${M5_RUN_AUTHORIZATION_PATH}`])) !== M5_P4_AUTHORIZATION_SHA256) {
    throw new Error("M5 prior P4 authorization history changed");
  }

  const runtimeAdditions = git(["log", "--first-parent", "--no-renames", "--diff-filter=A", "--format=%H", "--", M5_RUNTIME_AUTHORIZATION_PATH])
    .split("\n").filter(Boolean);
  if (runtimeAdditions.length !== 1 || runtimeAdditions[0] !== M5_RUNTIME_AUTHORIZATION_COMMIT ||
      !matchesExpectedRows(rows(M5_RUNTIME_AUTHORIZATION_COMMIT), new Map([[M5_RUNTIME_AUTHORIZATION_PATH, "A"]])) ||
      git(["rev-parse", `${M5_RUNTIME_AUTHORIZATION_COMMIT}^{tree}`]) !== M5_RUNTIME_AUTHORIZATION_TREE ||
      digest(m5GitBytes(["show", `${M5_RUNTIME_AUTHORIZATION_COMMIT}:${M5_RUNTIME_AUTHORIZATION_PATH}`])) !== M5_RUNTIME_AUTHORIZATION_SHA256) {
    throw new Error("M5 prior runtime authorization history changed");
  }

  const additions = git(["log", "--first-parent", "--no-renames", "--diff-filter=A", "--format=%H", "--", M5_RUNPOD_ENV_AUTHORIZATION_PATH])
    .split("\n").filter(Boolean);
  if (additions.length !== 1) throw new Error("M5 RunPod environment authorization must have exactly one committed addition");
  const authorization = additions[0];
  if (!matchesExpectedRows(rows(authorization), new Map([[M5_RUNPOD_ENV_AUTHORIZATION_PATH, "A"]]))) {
    throw new Error("M5 RunPod environment authorization changed outside its one-file surface");
  }
  const authorizationParents = parents(authorization);
  if (authorizationParents.length !== 1) throw new Error("M5 RunPod environment authorization must have one parent");
  const source = authorizationParents[0];

  if (parents(source).length !== 1 || parents(source)[0] !== M5_RUNTIME_AUTHORIZATION_COMMIT ||
      !matchesExpectedRows(rows(source), M5_RUNPOD_ENV_RECOVERY_EXPECTED) ||
      parents(M5_RUNTIME_AUTHORIZATION_COMMIT).length !== 1 || parents(M5_RUNTIME_AUTHORIZATION_COMMIT)[0] !== M5_RUNTIME_RECOVERY_COMMIT ||
      git(["rev-parse", `${M5_RUNTIME_RECOVERY_COMMIT}^{tree}`]) !== M5_RUNTIME_RECOVERY_TREE ||
      !matchesExpectedRows(rows(M5_RUNTIME_RECOVERY_COMMIT), M5_RUNTIME_RECOVERY_EXPECTED) ||
      parents(M5_RUNTIME_RECOVERY_COMMIT).length !== 1 || parents(M5_RUNTIME_RECOVERY_COMMIT)[0] !== M5_P4_COMMIT ||
      parents(M5_P4_COMMIT).length !== 1 || parents(M5_P4_COMMIT)[0] !== M5_CI_RECOVERY_COMMIT ||
      parents(M5_CI_RECOVERY_COMMIT).length !== 1 || parents(M5_CI_RECOVERY_COMMIT)[0] !== M5_FAILED_SOURCE_COMMIT ||
      git(["rev-parse", `${M5_CI_RECOVERY_COMMIT}^{tree}`]) !== M5_CI_RECOVERY_TREE ||
      !matchesExpectedRows(rows(M5_CI_RECOVERY_COMMIT), M5_SOURCE_CI_RECOVERY_EXPECTED) ||
      git(["rev-parse", `${M5_FAILED_SOURCE_COMMIT}^{tree}`]) !== M5_FAILED_SOURCE_TREE ||
      parents(M5_FAILED_SOURCE_COMMIT).length !== 1 || parents(M5_FAILED_SOURCE_COMMIT)[0] !== M5_P2_COMMIT ||
      git(["rev-parse", `${M5_P2_COMMIT}^{tree}`]) !== M5_P2_TREE ||
      !matchesExpectedRows(rows(M5_FAILED_SOURCE_COMMIT), M5_SOURCE_RECOVERY_EXPECTED)) {
    throw new Error("M5 full append-only authorization chain is invalid");
  }

  const raw = readFileSync(M5_RUNPOD_ENV_AUTHORIZATION_PATH);
  if (!raw.equals(m5GitBytes(["show", `${authorization}:${M5_RUNPOD_ENV_AUTHORIZATION_PATH}`]))) {
    throw new Error("M5 inherited RunPod environment authorization bytes changed");
  }
  const receipt = parseCanonicalM5Authorization(raw);
  requireM5AuthorizationSchema(receipt);
  const sourceTree = git(["rev-parse", `${source}^{tree}`]);
  const expectedMap = [...M5_SOURCE_RECOVERY_EXPECTED.keys()].sort().map((path) => ({
    path,
    sha256: digest(m5GitBytes(["show", `${source}:${path}`])),
  }));
  const ci = receipt.sourcePublicCi;
  if (receipt.schemaVersion !== 3 || receipt.status !== "m5-runpod-environment-recovery-authorized" ||
      receipt.protocolCommit !== M5_P2_COMMIT || receipt.protocolTree !== M5_P2_TREE ||
      receipt.priorAuthorizationCommit !== M5_RUNTIME_AUTHORIZATION_COMMIT || receipt.priorAuthorizationTree !== M5_RUNTIME_AUTHORIZATION_TREE ||
      receipt.priorAuthorizationPath !== M5_RUNTIME_AUTHORIZATION_PATH || receipt.priorAuthorizationSha256 !== M5_RUNTIME_AUTHORIZATION_SHA256 ||
      receipt.sourceCommit !== source || receipt.sourceTree !== sourceTree ||
      JSON.stringify(receipt.sourcePathMap) !== JSON.stringify(expectedMap) ||
      receipt.environmentBoundary !== "validated-single-runpod-pod-id-from-pid1-environ-no-other-record-forwarded" ||
      receipt.authorizationPath !== M5_RUNPOD_ENV_AUTHORIZATION_PATH || receipt.scoreBlind !== true || receipt.h3PixelsRead !== false ||
      ci.conclusion !== "success" || ci.event !== "push" || ci.headSha !== source ||
      !Number.isInteger(ci.runId) || ci.runId <= 0 || ci.status !== "completed" ||
      ci.url !== `https://github.com/baney75/prooflens/actions/runs/${ci.runId}` || ci.workflowPath !== ".github/workflows/quality.yml") {
    throw new Error("M5 RunPod environment authorization binding changed");
  }
  return {
    authorization,
    authorizationReceiptSha256: digest(raw),
    priorAuthorization: M5_RUNTIME_AUTHORIZATION_COMMIT,
    receipt,
    source,
    sourceTree,
  };
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  const result = validateM5AuthorizedChain();
  console.log(JSON.stringify({ authorization: result.authorization, source: result.source, protocol: M5_P2_COMMIT, policy: "pass" }));
}
