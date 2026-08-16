import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { TextDecoder } from "node:util";
import {
  M5_A5_AUTHORIZATION_PATH, M5_A5_COMMIT, M5_A5_SHA256, M5_A5_TREE,
  M5_A6_AUTHORIZATION_PATH, M5_A6_STATUS, M5_P2_COMMIT, M5_P2_TREE,
  M5_R6_EXPECTED,
} from "./m5-stage-policy.mjs";
import { m5Git, m5GitBytes } from "./m5-safe-git.mjs";

const git = (arguments_) => m5Git(arguments_);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const parents = (commit) => git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
const rows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, pathname, extra] = line.split("\t"); if (!status || !pathname || extra !== undefined) throw new Error("Malformed M5 commit row"); return [pathname, status]; });
const canonical = (value) => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object"
  ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value;
const exactKeys = (value, keys) => value && !Array.isArray(value) && typeof value === "object" &&
  JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort());
const sameRows = (actual, expected) => JSON.stringify([...actual].sort()) === JSON.stringify([...expected].sort());
const parseCanonical = (raw, label) => {
  const value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw));
  if (!raw.equals(Buffer.from(`${JSON.stringify(canonical(value))}\n`, "utf8"))) throw new Error(`${label} is not canonical strict UTF-8 JSON`);
  return value;
};

export function validateM5CublasAuthorizedChain() {
  if (!existsSync(M5_A6_AUTHORIZATION_PATH)) throw new Error("M5 A6 authorization receipt is missing");
  if (git(["rev-parse", `${M5_A5_COMMIT}^{tree}`]) !== M5_A5_TREE ||
      sha256(m5GitBytes(["show", `${M5_A5_COMMIT}:${M5_A5_AUTHORIZATION_PATH}`])) !== M5_A5_SHA256 ||
      !sameRows(rows(M5_A5_COMMIT), [[M5_A5_AUTHORIZATION_PATH, "A"]])) {
    throw new Error("M5 immutable A5 authorization binding changed");
  }
  const additions = git(["log", "--first-parent", "--no-renames", "--diff-filter=A", "--format=%H", "--", M5_A6_AUTHORIZATION_PATH]).split("\n").filter(Boolean);
  if (additions.length !== 1) throw new Error("M5 A6 authorization must have one committed addition");
  const authorization = additions[0];
  const authorizationParents = parents(authorization);
  if (authorizationParents.length !== 1 || !sameRows(rows(authorization), [[M5_A6_AUTHORIZATION_PATH, "A"]])) throw new Error("M5 A6 authorization must be receipt-only");
  const source = authorizationParents[0];
  if (parents(source).length !== 1 || parents(source)[0] !== M5_A5_COMMIT || !sameRows(rows(source), M5_R6_EXPECTED)) throw new Error("M5 R6 recovery lineage or surface changed");

  const raw = readFileSync(M5_A6_AUTHORIZATION_PATH);
  if (!raw.equals(m5GitBytes(["show", `${authorization}:${M5_A6_AUTHORIZATION_PATH}`]))) throw new Error("M5 A6 receipt bytes changed");
  const receipt = parseCanonical(raw, "M5 A6 authorization");
  const ci = receipt.sourcePublicCi;
  const expectedMap = [...M5_R6_EXPECTED.keys()].sort().map((pathname) => ({ pathname, row: { path: pathname, sha256: sha256(m5GitBytes(["show", `${source}:${pathname}`])) } })).map(({ row }) => row);
  if (!exactKeys(receipt, ["authorizationPath","cublasWorkspaceConfig","h3PixelsRead","priorAuthorizationCommit","priorAuthorizationPath","priorAuthorizationSha256","priorAuthorizationTree","protocolCommit","protocolTree","runtimeBoundary","schemaVersion","scoreBlind","sourceCommit","sourcePathMap","sourcePublicCi","sourceTree","status"]) ||
      !exactKeys(ci, ["conclusion","event","headSha","runId","status","url","workflowPath"]) ||
      receipt.schemaVersion !== 6 || receipt.status !== M5_A6_STATUS ||
      receipt.protocolCommit !== M5_P2_COMMIT || receipt.protocolTree !== M5_P2_TREE ||
      receipt.priorAuthorizationCommit !== M5_A5_COMMIT || receipt.priorAuthorizationTree !== M5_A5_TREE ||
      receipt.priorAuthorizationPath !== M5_A5_AUTHORIZATION_PATH || receipt.priorAuthorizationSha256 !== M5_A5_SHA256 ||
      receipt.sourceCommit !== source || receipt.sourceTree !== git(["rev-parse", `${source}^{tree}`]) ||
      JSON.stringify(receipt.sourcePathMap) !== JSON.stringify(expectedMap) ||
      receipt.runtimeBoundary !== "trusted-runpod-execution-child-environment-before-torch-import" ||
      receipt.cublasWorkspaceConfig !== ":4096:8" || receipt.authorizationPath !== M5_A6_AUTHORIZATION_PATH ||
      receipt.scoreBlind !== true || receipt.h3PixelsRead !== false ||
      !Number.isInteger(ci.runId) || ci.runId <= 0 || ci.conclusion !== "success" || ci.event !== "push" ||
      ci.headSha !== source || ci.status !== "completed" || ci.url !== `https://github.com/baney75/prooflens/actions/runs/${ci.runId}` ||
      ci.workflowPath !== ".github/workflows/quality.yml") {
    throw new Error("M5 A6 authorization binding changed");
  }
  return { authorization, source, sourceTree: receipt.sourceTree, authorizationReceiptSha256: sha256(raw), receipt };
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  console.log(JSON.stringify({ ...validateM5CublasAuthorizedChain(), policy: "pass" }));
}
