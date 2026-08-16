import { existsSync } from "node:fs";
import { get } from "node:https";
import { validateM5AuthorizedChain } from "./check-m5-authorized-chain.mjs";
import {
  M5_FAILURE_PATH,
  M5_FINAL_RECEIPT_PATH,
  M5_LARGE_SOURCE_LOCK_PATH,
  M5_SELECTION_LOCK_PATH,
} from "./m5-stage-policy.mjs";
import { assertM5WorktreeExact, m5Git } from "./m5-safe-git.mjs";

const git = (args) => m5Git(args);
const getJson = (url) => new Promise((resolve, reject) => {
  const request = get(url, { headers: { Accept: "application/vnd.github+json", "User-Agent": "seroslop-m5-verifier" } }, (response) => {
    const chunks = [];
    response.on("data", (chunk) => chunks.push(chunk));
    response.on("end", () => {
      if (response.statusCode !== 200) return reject(new Error(`Unable to verify public M5 CI: HTTP ${response.statusCode}`));
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch (error) { reject(error); }
    });
  });
  request.on("error", reject);
});
const head = git(["rev-parse", "HEAD"]);
const result = validateM5AuthorizedChain();
if (head !== result.authorization) throw new Error("M5 authorized stage must be the exact RunPod environment receipt commit");
assertM5WorktreeExact();
for (const forbidden of [M5_SELECTION_LOCK_PATH, M5_FAILURE_PATH, M5_LARGE_SOURCE_LOCK_PATH, M5_FINAL_RECEIPT_PATH, "docs/COMPETITOR_AUDIT.md"]) {
  if (existsSync(forbidden)) throw new Error(`M5 RunPod environment authorization contains forbidden later evidence: ${forbidden}`);
}
const publicReference = await getJson("https://api.github.com/repos/baney75/prooflens/git/ref/heads/main");
const publicHead = publicReference.object?.sha;
if (publicHead !== head) throw new Error("M5 RunPod environment authorization must be the anonymous public main head");
const sourceRun = await getJson(`https://api.github.com/repos/baney75/prooflens/actions/runs/${result.receipt.sourcePublicCi.runId}`);
if (sourceRun.head_sha !== result.source || sourceRun.event !== "push" || sourceRun.status !== "completed" ||
    sourceRun.conclusion !== "success" || sourceRun.path !== ".github/workflows/quality.yml" ||
    sourceRun.html_url !== result.receipt.sourcePublicCi.url) {
  throw new Error("M5 runtime source CI proof does not match the public GitHub run");
}
console.log(JSON.stringify({ head, source: result.source, authorizationReceiptSha256: result.authorizationReceiptSha256, policy: "pass" }));
