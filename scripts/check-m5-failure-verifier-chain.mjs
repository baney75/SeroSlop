import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { TextDecoder } from "node:util";
import {
  M5_A6_AUTHORIZATION_PATH, M5_A6_COMMIT, M5_A6_STATUS,
  M5_A7_AUTHORIZATION_PATH, M5_A7_STATUS, M5_R7_EXPECTED,
} from "./m5-stage-policy.mjs";
import { validateM5CublasAuthorizedChain } from "./check-m5-cublas-authorized-chain.mjs";
import { m5Git, m5GitBytes } from "./m5-safe-git.mjs";

const git = (args) => m5Git(args);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const parents = (commit) => git(["rev-list", "--parents", "-n", "1", commit]).split(" ").slice(1);
const rows = (commit) => git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
  .split("\n").filter(Boolean).map((line) => { const [status, pathname, extra] = line.split("\t"); if (!status || !pathname || extra !== undefined) throw new Error("Malformed M5 commit row"); return [pathname, status]; });
const sameRows = (actual, expected) => JSON.stringify([...actual].sort()) === JSON.stringify([...expected].sort());
const exact = (value, keys) => value && !Array.isArray(value) && typeof value === "object" && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort());
const canonical = (value) => Array.isArray(value) ? value.map(canonical) : value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])])) : value;
const parseCanonical = (raw) => { const value = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(raw)); if (!raw.equals(Buffer.from(`${JSON.stringify(canonical(value))}\n`))) throw new Error("M5 A7 authorization is not canonical strict UTF-8 JSON"); return value; };

export function validateM5FailureVerifierChain() {
  const a6 = validateM5CublasAuthorizedChain();
  if (a6.authorization !== M5_A6_COMMIT || a6.receipt.status !== M5_A6_STATUS) throw new Error("M5 A6 binding changed");
  if (!existsSync(M5_A7_AUTHORIZATION_PATH)) throw new Error("M5 A7 authorization receipt is missing");
  const additions = git(["log", "--first-parent", "--no-renames", "--diff-filter=A", "--format=%H", "--", M5_A7_AUTHORIZATION_PATH]).split("\n").filter(Boolean);
  if (additions.length !== 1) throw new Error("M5 A7 authorization must have one committed addition");
  const authorization = additions[0];
  if (parents(authorization).length !== 1 || !sameRows(rows(authorization), [[M5_A7_AUTHORIZATION_PATH, "A"]])) throw new Error("M5 A7 authorization must directly follow R7 and be receipt-only");
  const source = parents(authorization)[0];
  if (parents(source).length !== 1 || parents(source)[0] !== M5_A6_COMMIT || !sameRows(rows(source), M5_R7_EXPECTED)) throw new Error("M5 R7 recovery lineage or surface changed");
  const raw = readFileSync(M5_A7_AUTHORIZATION_PATH);
  if (!raw.equals(m5GitBytes(["show", `${authorization}:${M5_A7_AUTHORIZATION_PATH}`]))) throw new Error("M5 A7 receipt bytes changed");
  const receipt = parseCanonical(raw);
  const ci = receipt.sourcePublicCi;
  const expectedMap = [...M5_R7_EXPECTED.keys()].sort().map((path) => ({ path, sha256: sha256(m5GitBytes(["show", `${source}:${path}`])) }));
  if (!exact(receipt, ["authorizationPath", "boundary", "h3PixelsRead", "postTrainingVerifierOnly", "terminalRegressionsRead", "priorAuthorizationCommit", "priorAuthorizationPath", "priorAuthorizationSha256", "priorAuthorizationTree", "protocolCommit", "protocolTree", "schemaVersion", "scoreBlind", "sourceCommit", "sourcePathMap", "sourcePublicCi", "sourceTree", "status", "trainingAuthorizationCommit", "trainingAuthorizationReceiptSha256", "trainingSourceCommit", "trainingSourceTree"]) ||
      !exact(ci, ["conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"]) ||
      receipt.schemaVersion !== 7 || receipt.status !== M5_A7_STATUS || receipt.authorizationPath !== M5_A7_AUTHORIZATION_PATH ||
      receipt.priorAuthorizationCommit !== a6.authorization || receipt.priorAuthorizationPath !== M5_A6_AUTHORIZATION_PATH || receipt.priorAuthorizationSha256 !== a6.authorizationReceiptSha256 || receipt.priorAuthorizationTree !== git(["rev-parse", `${a6.authorization}^{tree}`]) ||
      receipt.protocolCommit !== "1c4ac973785f937fa9023018863941e6d89d8693" || receipt.protocolTree !== "a56caae4291e275029076417fb2111be76b07a41" ||
      receipt.sourceCommit !== source || receipt.sourceTree !== git(["rev-parse", `${source}^{tree}`]) || JSON.stringify(receipt.sourcePathMap) !== JSON.stringify(expectedMap) ||
      receipt.boundary !== "canonical-json-selector-logit-key-set-order-independent-verifier-only" || receipt.scoreBlind !== true || receipt.h3PixelsRead !== false || receipt.postTrainingVerifierOnly !== true || receipt.terminalRegressionsRead !== false ||
      receipt.trainingSourceCommit !== a6.source || receipt.trainingSourceTree !== a6.sourceTree || receipt.trainingAuthorizationCommit !== a6.authorization || receipt.trainingAuthorizationReceiptSha256 !== a6.authorizationReceiptSha256 ||
      !Number.isInteger(ci.runId) || ci.runId <= 0 || ci.conclusion !== "success" || ci.event !== "push" || ci.headSha !== source || ci.status !== "completed" || ci.url !== `https://github.com/baney75/prooflens/actions/runs/${ci.runId}` || ci.workflowPath !== ".github/workflows/quality.yml") throw new Error("M5 A7 authorization binding changed");
  return { authorization, source, sourceTree: receipt.sourceTree, authorizationReceiptSha256: sha256(raw), receipt, trainingSource: a6.source, trainingAuthorization: a6.authorization, trainingAuthorizationReceiptSha256: a6.authorizationReceiptSha256 };
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) console.log(JSON.stringify({ ...validateM5FailureVerifierChain(), policy: "pass" }));
