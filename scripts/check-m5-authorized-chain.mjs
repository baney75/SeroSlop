import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { TextDecoder } from "node:util";
import {
  M5_FAILED_SOURCE_COMMIT,
  M5_FAILED_SOURCE_TREE,
  M5_P2_COMMIT,
  M5_P2_TREE,
  M5_RUN_AUTHORIZATION_PATH,
  M5_SOURCE_CI_RECOVERY_EXPECTED,
  M5_SOURCE_RECOVERY_EXPECTED,
  matchesExpectedRows,
} from "./m5-stage-policy.mjs";
import { m5Git, m5GitBytes } from "./m5-safe-git.mjs";

const git = (args) => m5Git(args);
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const rows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, path] = line.split("\t"); return [path, status]; });
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
  if (!raw.equals(expectedBytes)) throw new Error("M5 P4 authorization is not canonical strict UTF-8 JSON");
  return receipt;
}

export function requireM5AuthorizationSchema(receipt) {
  if (!exactKeys(receipt, [
    "authorizationPath", "h3PixelsRead", "protocolCommit", "protocolTree", "schemaVersion", "scoreBlind",
    "sourceCommit", "sourcePathMap", "sourcePublicCi", "sourceTree", "status",
  ]) || !exactKeys(receipt.sourcePublicCi, ["conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"])) {
    throw new Error("M5 P4 authorization schema changed");
  }
}

export function validateM5AuthorizedChain() {
  const additions = git(["log", "--first-parent", "--no-renames", "--diff-filter=A", "--format=%H", "--", M5_RUN_AUTHORIZATION_PATH])
    .split("\n").filter(Boolean);
  if (additions.length !== 1) throw new Error("M5 P4 authorization must have exactly one committed addition");
  const authorization = additions[0];
  if (!matchesExpectedRows(rows(authorization), new Map([[M5_RUN_AUTHORIZATION_PATH, "A"]]))) {
    throw new Error("M5 P4 authorization commit changed outside its one-file surface");
  }
  const authorizationParents = git(["rev-list", "--parents", "-n", "1", authorization]).split(" ").slice(1);
  if (authorizationParents.length !== 1) throw new Error("M5 P4 authorization must have one parent");
  const source = authorizationParents[0];
  const sourceParents = git(["rev-list", "--parents", "-n", "1", source]).split(" ").slice(1);
  const failedSourceParents = git(["rev-list", "--parents", "-n", "1", M5_FAILED_SOURCE_COMMIT]).split(" ").slice(1);
  if (sourceParents.length !== 1 || sourceParents[0] !== M5_FAILED_SOURCE_COMMIT ||
      !matchesExpectedRows(rows(source), M5_SOURCE_CI_RECOVERY_EXPECTED) ||
      git(["rev-parse", `${M5_FAILED_SOURCE_COMMIT}^{tree}`]) !== M5_FAILED_SOURCE_TREE ||
      failedSourceParents.length !== 1 || failedSourceParents[0] !== M5_P2_COMMIT ||
      git(["rev-parse", `${M5_P2_COMMIT}^{tree}`]) !== M5_P2_TREE ||
      !matchesExpectedRows(rows(M5_FAILED_SOURCE_COMMIT), M5_SOURCE_RECOVERY_EXPECTED)) {
    throw new Error("M5 P2->failed-P3->CI-recovery->P4 chain is invalid");
  }
  const raw = readFileSync(M5_RUN_AUTHORIZATION_PATH);
  const receipt = parseCanonicalM5Authorization(raw);
  requireM5AuthorizationSchema(receipt);
  const sourceTree = git(["rev-parse", `${source}^{tree}`]);
  const expectedMap = [...M5_SOURCE_RECOVERY_EXPECTED.keys()].sort().map((path) => ({
    path,
    sha256: digest(m5GitBytes(["show", `${source}:${path}`])),
  }));
  const ci = receipt.sourcePublicCi;
  if (receipt.schemaVersion !== 1 || receipt.status !== "m5-source-recovery-authorized" ||
      receipt.protocolCommit !== M5_P2_COMMIT || receipt.protocolTree !== M5_P2_TREE ||
      receipt.sourceCommit !== source || receipt.sourceTree !== sourceTree ||
      JSON.stringify(receipt.sourcePathMap) !== JSON.stringify(expectedMap) ||
      receipt.authorizationPath !== M5_RUN_AUTHORIZATION_PATH || receipt.scoreBlind !== true || receipt.h3PixelsRead !== false ||
      ci.conclusion !== "success" || ci.event !== "push" || ci.headSha !== source ||
      !Number.isInteger(ci.runId) || ci.runId <= 0 || ci.status !== "completed" ||
      ci.url !== `https://github.com/baney75/prooflens/actions/runs/${ci.runId}` || ci.workflowPath !== ".github/workflows/quality.yml") {
    throw new Error("M5 P4 authorization binding changed");
  }
  return { authorization, authorizationReceiptSha256: digest(raw), receipt, source, sourceTree };
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  const result = validateM5AuthorizedChain();
  console.log(JSON.stringify({ authorization: result.authorization, source: result.source, protocol: M5_P2_COMMIT, policy: "pass" }));
}
