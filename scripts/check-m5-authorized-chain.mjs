import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { TextDecoder } from "node:util";
import { M5_A4_COMMIT, M5_A4_TREE, M5_A4_PATH, M5_A4_SHA256, M5_A5_AUTHORIZATION_PATH, M5_A5_STATUS, M5_A6_AUTHORIZATION_PATH, M5_R5_EXPECTED } from "./m5-stage-policy.mjs";
import { m5Git, m5GitBytes } from "./m5-safe-git.mjs";
import { validateM5CublasAuthorizedChain } from "./check-m5-cublas-authorized-chain.mjs";
const git = (a) => m5Git(a); const sha = (b) => createHash("sha256").update(b).digest("hex");
const rows = (c) => git(["diff-tree","--root","--no-renames","--name-status","--format=","-r",c]).split("\n").filter(Boolean).map((x) => { const [s,p]=x.split("\t"); return [p,s]; });
const parents = (c) => git(["rev-list","--parents","-n","1",c]).split(" ").slice(1);
const canonical = (v) => Array.isArray(v) ? v.map(canonical) : v && typeof v === "object" ? Object.fromEntries(Object.keys(v).sort().map(k=>[k,canonical(v[k])])) : v;
const exact = (v, ks) => v && !Array.isArray(v) && typeof v === "object" && JSON.stringify(Object.keys(v).sort()) === JSON.stringify([...ks].sort());
export function parseCanonicalM5Authorization(raw) { const r=JSON.parse(new TextDecoder("utf-8",{fatal:true}).decode(raw)); if (!raw.equals(Buffer.from(`${JSON.stringify(canonical(r))}\n`))) throw new Error("M5 authorization is not canonical strict UTF-8 JSON"); return r; }
export function requireM5AuthorizationSchema(r) {
  if (!exact(r,["authorizationPath","parityBoundary","diagnosticSha256","h3PixelsRead","priorAuthorizationCommit","priorAuthorizationPath","priorAuthorizationSha256","priorAuthorizationTree","protocolCommit","protocolTree","schemaVersion","scoreBlind","sourceCommit","sourcePathMap","sourcePublicCi","sourceTree","status"]) || !exact(r.sourcePublicCi,["conclusion","event","headSha","runId","status","url","workflowPath"])) throw new Error("M5 parity authorization schema changed");
}
export function validateM5A5AuthorizedChain() {
  if (!existsSync(M5_A5_AUTHORIZATION_PATH)) throw new Error("M5 A5 authorization receipt is missing");
  const additions=git(["log","--first-parent","--no-renames","--diff-filter=A","--format=%H","--",M5_A5_AUTHORIZATION_PATH]).split("\n").filter(Boolean); if(additions.length!==1) throw new Error("M5 A5 authorization must have one committed addition");
  const authorization=additions[0], ap=parents(authorization); if(ap.length!==1) throw new Error("M5 A5 authorization must have one R5 parent"); const source=ap[0];
  if(source===M5_A4_COMMIT || parents(source).length!==1 || parents(source)[0]!==M5_A4_COMMIT || JSON.stringify(rows(source).sort())!==JSON.stringify([...M5_R5_EXPECTED].sort())) throw new Error("M5 R5 recovery lineage is not exact");
  if(git(["rev-parse",`${M5_A4_COMMIT}^{tree}`])!==M5_A4_TREE || sha(m5GitBytes(["show",`${M5_A4_COMMIT}:${M5_A4_PATH}`]))!==M5_A4_SHA256) throw new Error("M5 A4 binding changed");
  if(JSON.stringify(rows(authorization))!==JSON.stringify([[M5_A5_AUTHORIZATION_PATH,"A"]])) throw new Error("M5 A5 authorization must be receipt-only");
  const raw=readFileSync(M5_A5_AUTHORIZATION_PATH); if(!raw.equals(m5GitBytes(["show",`${authorization}:${M5_A5_AUTHORIZATION_PATH}`]))) throw new Error("M5 A5 receipt bytes changed"); const receipt=parseCanonicalM5Authorization(raw); requireM5AuthorizationSchema(receipt);
  const expected=[...M5_R5_EXPECTED.keys()].sort().map(path=>({path,sha256:sha(m5GitBytes(["show",`${source}:${path}`]))})); const ci=receipt.sourcePublicCi;
  if(receipt.schemaVersion!==5||receipt.status!==M5_A5_STATUS||receipt.protocolCommit!=="1c4ac973785f937fa9023018863941e6d89d8693"||receipt.protocolTree!=="a56caae4291e275029076417fb2111be76b07a41"||receipt.priorAuthorizationCommit!==M5_A4_COMMIT||receipt.priorAuthorizationTree!==M5_A4_TREE||receipt.priorAuthorizationPath!==M5_A4_PATH||receipt.priorAuthorizationSha256!==M5_A4_SHA256||receipt.sourceCommit!==source||receipt.sourceTree!==git(["rev-parse",`${source}^{tree}`])||JSON.stringify(receipt.sourcePathMap)!==JSON.stringify(expected)||receipt.parityBoundary!=="packaged-m2-reference-preserved-real-input-parity-and-onnx-scoring"||receipt.diagnosticSha256!=="c9c673efa0b1a6e4ea79b195ec16c71ae8ac91f962390a49c4e570b6d8de5c11"||receipt.authorizationPath!==M5_A5_AUTHORIZATION_PATH||receipt.scoreBlind!==true||receipt.h3PixelsRead!==false||typeof ci.runId!=="number"||ci.runId<=0||ci.url!==`https://github.com/baney75/prooflens/actions/runs/${ci.runId}`||ci.workflowPath!==".github/workflows/quality.yml"||ci.headSha!==source||ci.event!=="push"||ci.status!=="completed"||ci.conclusion!=="success") throw new Error("M5 A5 authorization binding changed");
  return {authorization,source,sourceTree:receipt.sourceTree,authorizationReceiptSha256:sha(raw),receipt};
}
export function validateM5AuthorizedChain() {
  return existsSync(M5_A6_AUTHORIZATION_PATH) ? validateM5CublasAuthorizedChain() : validateM5A5AuthorizedChain();
}
if(process.argv[1]&&pathToFileURL(process.argv[1]).href===import.meta.url) console.log(JSON.stringify({...validateM5AuthorizedChain(),policy:"pass"}));
