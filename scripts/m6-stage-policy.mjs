import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

export const M6_BASE_COMMIT = "76d0a807dcf240245830b8510e623d838e43cd4c";
export const M6_BASE_TREE = "5c10012b9520e3936efc86e08ff0a53adecec868";
export const M6_P_COMMIT = "3b29ea2f9e1ad46e4cd78f47c9ccf5fe3a99877e";
export const M6_P_TREE = "dfd29cd86f4f746d403b14994055a575d82f83c4";
export const M6_P2_COMMIT = "0777710c89cd0fa02e2f4bd063ec51664e3fc26a";
export const M6_P2_TREE = "30d8338382033caacd400e1b29c37ed287f9de43";
export const M6_P3_COMMIT = "fa9f002a2f9805b59d7955bf4c4f9992bbfb22ce";
export const M6_P3_TREE = "74f84b12a5a1a126fbb79142f6530be996ce4990";
export const M6_CENSUS_SHA256 = "61f494f09fe256d771bacb809712b5c645e5b25f63cffca52dfc40d0e0ac7adf";
export const M6_P_RECIPE_SHA256 = "56bfe2487760c833c796289e3d4c5e8ef0eb65e62493229f9d62631a573ab613";
export const M6_P2_RECIPE_SHA256 = "42f594fd26ac4949f191eb5c773c977ec8e5bee586f766c9a648afce85bc2984";
export const M6_RECIPE_SHA256 = "a1c1700acbfbed19ef73e3cc4224c994eadef17e81ddb8d6d8040c8a3d5a5e88";
export const M6_SOURCE_SHARDS_PATH = "benchmark/m6/source-shards.json";
export const M6_SOURCE_SHARDS_SHA256 = "a86c7209e76248edddd61537f397379194a7aaa908405e0cede7c8f5a3d7fbfe";
export const M6_VERIFY_REQUIREMENTS_PATH = "benchmark/verify-requirements.txt";
export const M6_P3_VERIFY_REQUIREMENTS_SHA256 = "f34d97a0c10c23d3dba50f8bbcf4df8dad9e5b3cf80510b20c6df14bdc06af75";
export const M6_VERIFY_REQUIREMENTS_SHA256 = "00ea11478746fdb02c445e31c084e496e08f8fc6cc49f313fd229d34d70396ed";
export const M6_RECIPE_PATH = "benchmark/m6/recipe.json";
export const M6_CENSUS_PATH = "benchmark/m6/census-evidence.json";
export const M6_PROTOCOL_PATHS = Object.freeze([
  "benchmark/m6/README.md", "benchmark/m6/THIRD_PARTY_NOTICES.md", "benchmark/m6/__init__.py", "benchmark/m6/census-evidence.json", "benchmark/m6/contracts.py", "benchmark/m6/preflight.py", "benchmark/m6/prepare.py", "benchmark/m6/recipe.json", "benchmark/m6/test_contracts.py",
  "package.json", "scripts/check-m6-protocol-stage.mjs", "scripts/m6-stage-policy.mjs", "scripts/run-static-verification.mjs", "scripts/test-m6-stage-policy.mjs",
]);
export const M6_P7_PARENT = "e6412f90736d39985332ef009a360614865306d2";
export const M6_P7_PARENT_TREE = "b99608de8093fcdd70d7f3520ebfbce3344dfcb5";
export const M6_P7_STATUS = "p7-phase1-taste-unverified";
export const M6_P7_PROTOCOL_SHA256 = "f05deaba1e8d7f056b33ee4ce80c4846259d8425b273dc8a5ccdfd41abeddb95";
export const M6_P7_S_COMMIT = "53b696e09d1e6243349aeece843afb7f7de6f17f";
export const M6_P7_S_TREE = "1793e108eb9bc30c35037fad56da545b4a3c866a";
export const M6_P7_S_ROWS = Object.freeze([["benchmark/m6/README.md","M"],["benchmark/m6/p7-protocol.json","A"],["benchmark/m6/p7_operational.py","A"],["benchmark/m6/test_p7_operational.py","A"],["package.json","M"],["scripts/check-m6-protocol-stage.mjs","M"],["scripts/m6-stage-policy.mjs","M"],["scripts/run-static-verification.mjs","M"],["scripts/test-m6-stage-policy.mjs","M"]]);
export const M6_P7_S_ARTIFACT_SHA256 = Object.freeze({"benchmark/m6/README.md":"25249cb448d8c208aeaab5e8d701bf53109afe5cf8a7feb90732788f8e8caa23","benchmark/m6/p7-protocol.json":"f05deaba1e8d7f056b33ee4ce80c4846259d8425b273dc8a5ccdfd41abeddb95","benchmark/m6/p7_operational.py":"daa6c2f7b1512ef9540e6cba3318ff92d46719a4f59a3ead9b9a72fc44564cd9","benchmark/m6/test_p7_operational.py":"2d1fc6b0f3d98a8f127c183835e26b273a15f138d2d48fd52c26bda2909eca5e","package.json":"707dd55d4b4177504b33881bb9bf5f9b38352e43c76f7c92cbb8e28d42d8f864","scripts/check-m6-protocol-stage.mjs":"1747f424287de357b90e8392c4a81b9110583957e9e48c1cbe83e85dcbc02db7","scripts/m6-stage-policy.mjs":"6d5e95739745b63d88833253e4196817429e18de6288b68ee4bdee8871738723","scripts/run-static-verification.mjs":"ffe4ad530bbd5f9bf056688fa2017ac2e2a37b9cac3e9899c21f2117d79ae3c0","scripts/test-m6-stage-policy.mjs":"1563ac7ec5a8d6a1a33e350694aca81d7aef1336eb0c926d8cd1669792213940"});
export const M6_P7_R_EXPECTED = Object.freeze([["scripts/check-m6-protocol-stage.mjs","M"],["scripts/m6-stage-policy.mjs","M"],["scripts/run-static-verification.mjs","M"],["scripts/test-m6-stage-policy.mjs","M"]]);
export const M6_P7_R_STATUS = "p7-phase1-taste-verifier-ready";
export const M6_P7_A_STATUS = "p7-phase1-taste-authorized";
export const M6_P7_AUTHORIZATION_PATH = "benchmark/evidence/m6/p7-phase1-taste-authorization.json";
export function matchesM6P7RHead({head,parent,rows=[]}={}) { return typeof head==="string" && /^[0-9a-f]{40}$/.test(head) && head!==parent && parent===M6_P7_S_COMMIT && JSON.stringify(normalizedRows(rows))===JSON.stringify(normalizedRows(M6_P7_R_EXPECTED)); }
export function matchesM6P7AuthorizationHead({head,parent,rows=[]}={}) { return typeof head==="string" && /^[0-9a-f]{40}$/.test(head) && head!==parent && JSON.stringify(normalizedRows(rows))===JSON.stringify([[M6_P7_AUTHORIZATION_PATH,"A"]]); }
export function validateM6P7Authorization(bytes,{sourceCommit,sourceTree,sourceParent=M6_P7_PARENT,sourceRows,sourcePathMap,verifierCommit,verifierTree,verifierRows,publicCi}={}) {
  const raw=Buffer.from(bytes), text=raw.toString("utf8"); if(!text.endsWith("\n")||Buffer.from(text).compare(raw)!==0) throw new Error("P7 authorization must be canonical UTF-8 JSON");
  rejectDuplicateKeys(text); const value=JSON.parse(text); if(text!==canonicalM6Json(value)) throw new Error("P7 authorization bytes are not canonical");
  const keys=["acceptanceEligible","authorizationPath","commercialRightsClearanceClaimed","h3PixelsRead","independentOriginProofClaimed","phase1InputsAuthorized","protocolCommit","protocolParent","protocolPathMap","protocolRows","protocolTree","publisherAssertionOnly","schemaVersion","sourceLockAuthorized","status","tasteMaterializationAuthorized","trainingAuthorized","verifierCommit","verifierPublicCi","verifierRows","verifierTree"].sort();
  if(JSON.stringify(Object.keys(value).sort())!==JSON.stringify(keys)) throw new Error("P7 authorization schema changed");
  if(value.schemaVersion!==1||value.status!==M6_P7_A_STATUS||value.authorizationPath!==M6_P7_AUTHORIZATION_PATH||value.protocolCommit!==sourceCommit||value.protocolTree!==sourceTree||value.protocolParent!==sourceParent||value.publisherAssertionOnly!==true||value.independentOriginProofClaimed!==false||value.commercialRightsClearanceClaimed!==false||value.phase1InputsAuthorized!==true||value.tasteMaterializationAuthorized!==true||value.sourceLockAuthorized!==false||value.trainingAuthorized!==false||value.acceptanceEligible!==false||value.h3PixelsRead!==false) throw new Error("P7 authorization boundary changed");
  if(canonicalM6Json(normalizedRows(value.protocolRows))!==canonicalM6Json(normalizedRows(sourceRows))||canonicalM6Json(value.protocolPathMap)!==canonicalM6Json(sourcePathMap)||value.verifierCommit!==verifierCommit||value.verifierTree!==verifierTree||canonicalM6Json(normalizedRows(value.verifierRows))!==canonicalM6Json(normalizedRows(verifierRows))) throw new Error("P7 authorization source binding changed");
  const ci=value.verifierPublicCi, expected=publicCi??{}; const ciKeys=["conclusion","event","headSha","runId","status","url","workflowPath"].sort(); if(!ci||JSON.stringify(Object.keys(ci).sort())!==JSON.stringify(ciKeys)||!Number.isSafeInteger(ci.runId)||ci.runId<=0||ci.status!=="completed"||ci.conclusion!=="success"||ci.event!=="push"||ci.workflowPath!==".github/workflows/quality.yml"||ci.headSha!==verifierCommit||ci.url!==`https://github.com/baney75/prooflens/actions/runs/${ci.runId}`||JSON.stringify(ci)!==JSON.stringify(expected)) throw new Error("P7 authorization CI proof changed");
  return value;
}
export const M6_P7_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"], ["benchmark/m6/p7-protocol.json", "A"], ["benchmark/m6/p7_operational.py", "A"], ["benchmark/m6/test_p7_operational.py", "A"], ["package.json", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"], ["scripts/m6-stage-policy.mjs", "M"], ["scripts/run-static-verification.mjs", "M"], ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export function matchesM6P7Head({ head, parent, rows = [], treePaths = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent && parent === M6_P7_PARENT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_P7_EXPECTED)) && M6_P7_EXPECTED.every(([path]) => treePaths.includes(path));
}
export function validateM6P7Protocol(bytes) {
  const text = Buffer.from(bytes).toString("utf8"); const value = JSON.parse(text);
  if (createHash("sha256").update(Buffer.from(bytes)).digest("hex") !== M6_P7_PROTOCOL_SHA256) throw new Error("P7 protocol bytes changed");
  if (!text.endsWith("\n") || /\n\s*\n/.test(text)) throw new Error("P7 protocol bytes are not canonical");
  if (value.schemaVersion !== 1 || value.status !== "p7-phase1-taste-unverified" || value.parentCommit !== M6_P7_PARENT || value.parentTree !== M6_P7_PARENT_TREE || value.claims?.trainingAuthorized !== false || value.claims?.acceptanceEligible !== false) throw new Error("P7 protocol semantics changed");
  return true;
}
export const M6_P8_PARENT = "6fc4422327a9c6c40963ff1a57e62baa94511f1f";
export const M6_P8_STATUS = "p8-frontier-adapters-unverified";
export const M6_P8_PROTOCOL_PATH = "benchmark/m6/p8-protocol.json";
export const M6_P8_PROTOCOL_SHA256 = "2f95f93232219b9eb3c9afe98b50188c3c889309f0526f0fd05ba14e80b978b7";
export const M6_P8_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md","M"],["benchmark/m6/p7_operational.py","M"],["benchmark/m6/test_p7_operational.py","M"],[M6_P8_PROTOCOL_PATH,"A"],
  ["scripts/check-m6-protocol-stage.mjs","M"],["scripts/m6-stage-policy.mjs","M"],["scripts/run-static-verification.mjs","M"],["scripts/test-m6-stage-policy.mjs","M"]
]);
export function matchesM6P8Head({head,parent,rows=[]}={}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent && parent === M6_P8_PARENT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_P8_EXPECTED));
}
export function validateM6P8Protocol(bytes) {
  const text=Buffer.from(bytes).toString("utf8"); rejectDuplicateKeys(text); const value=JSON.parse(text);
  if (text !== canonicalM6Json(value)) throw new Error("P8 protocol is not canonical");
  if (!text.endsWith("\n") || createHash("sha256").update(Buffer.from(bytes)).digest("hex") !== M6_P8_PROTOCOL_SHA256) throw new Error("P8 protocol bytes changed or noncanonical");
  const claims=value.claims;
  if (value.schemaVersion!==1 || value.status!==M6_P8_STATUS || value.parentCommit!==M6_P8_PARENT || Object.keys(value).length!==5 || Object.keys(claims ?? {}).length!==8 || !Object.values(claims ?? {}).every((value) => value === false)) throw new Error("P8 protocol boundary changed");
  return true;
}
export const M6_P8_S_COMMIT = "d041d54ce757cfb0c1b61240937911177b4ee0c8";
export const M6_P8_S_TREE = "5f9a5d3619aa95263ac340891f8ba499a04981e8";
export const M6_P8_S_ROWS = Object.freeze([["benchmark/m6/README.md","M"],["benchmark/m6/p7_operational.py","M"],["benchmark/m6/p8-protocol.json","A"],["benchmark/m6/test_p7_operational.py","M"],["scripts/check-m6-protocol-stage.mjs","M"],["scripts/m6-stage-policy.mjs","M"],["scripts/run-static-verification.mjs","M"],["scripts/test-m6-stage-policy.mjs","M"]]);
export const M6_P8_S_ARTIFACT_SHA256 = Object.freeze({"benchmark/m6/README.md":"cc5a67d93829599490473210d72753fb0b2db96283a03961d0f8ec7028f3e2c2","benchmark/m6/p7_operational.py":"e0a8866ed59968c604bbe566a2cd488efe94d62e5836e8c52a1e51d655b87b2c","benchmark/m6/test_p7_operational.py":"f233a5d16c4574f2d9b75f10c4db60d93f8f4c69e34f06fb3afb0fc088659dc7","benchmark/m6/p8-protocol.json":"2f95f93232219b9eb3c9afe98b50188c3c889309f0526f0fd05ba14e80b978b7","scripts/check-m6-protocol-stage.mjs":"e4c3e2866060c0c1427e09b291b61076a58c4c3ac74689f2eda4dad4bf1e7ca2","scripts/m6-stage-policy.mjs":"ac59c3b5794de95f7fcbb7ad31db5f346625e705d071dd6ef16c92134a7b230c","scripts/run-static-verification.mjs":"ce36ad844864ea8ea1db9f2f7b9430caca3c9e34f168b64ac05df379aabb67be","scripts/test-m6-stage-policy.mjs":"a6355ea15ab84116461e58e2b81c2d6cb24415476504785e6e522934fd6b7a91"});
export const M6_P8_R_EXPECTED = Object.freeze([["scripts/check-m6-protocol-stage.mjs","M"],["scripts/m6-stage-policy.mjs","M"],["scripts/run-static-verification.mjs","M"],["scripts/test-m6-stage-policy.mjs","M"]]);
export const M6_P8_R_STATUS = "p8-frontier-adapters-verifier-ready";
export const M6_P8_A_STATUS = "p8-frontier-adapters-authorized";
export const M6_P8_AUTHORIZATION_PATH = "benchmark/evidence/m6/p8-frontier-adapters-authorization.json";
export function matchesM6P8RHead({head,parent,rows=[]}={}) { return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent && parent === M6_P8_S_COMMIT && JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_P8_R_EXPECTED)); }
export function matchesM6P8AuthorizationHead({head,parent,rows=[]}={}) { return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent && JSON.stringify(normalizedRows(rows)) === JSON.stringify([[M6_P8_AUTHORIZATION_PATH,"A"]]); }
export function validateM6P8Authorization(bytes,{sourceCommit,sourceTree,sourceParent=M6_P8_PARENT,sourceRows,sourcePathMap,verifierCommit,verifierTree,verifierRows,publicCi}={}) {
  const raw=Buffer.from(bytes), text=raw.toString("utf8"); if(!text.endsWith("\n")||Buffer.from(text).compare(raw)!==0) throw new Error("P8 authorization must be canonical UTF-8 JSON"); rejectDuplicateKeys(text); const value=JSON.parse(text); if(text!==canonicalM6Json(value)) throw new Error("P8 authorization bytes are not canonical");
  const keys=["acceptanceEligible","adapterImplementationVerified","authorizationPath","commercialRightsClearanceClaimed","fullCacheMaterializationAuthorized","h3PixelsRead","independentOriginProofClaimed","protocolCommit","protocolParent","protocolPathMap","protocolRows","protocolTree","publisherAssertionOnly","schemaVersion","sourceLockAuthorized","status","trainingAuthorized","verifierCommit","verifierPublicCi","verifierRows","verifierTree"].sort(); if(JSON.stringify(Object.keys(value).sort())!==JSON.stringify(keys)) throw new Error("P8 authorization schema changed");
  if(value.schemaVersion!==1||value.status!==M6_P8_A_STATUS||value.authorizationPath!==M6_P8_AUTHORIZATION_PATH||value.protocolCommit!==sourceCommit||value.protocolTree!==sourceTree||value.protocolParent!==sourceParent||value.publisherAssertionOnly!==true||value.independentOriginProofClaimed!==false||value.commercialRightsClearanceClaimed!==false||value.adapterImplementationVerified!==true||value.fullCacheMaterializationAuthorized!==true||value.sourceLockAuthorized!==false||value.trainingAuthorized!==false||value.acceptanceEligible!==false||value.h3PixelsRead!==false) throw new Error("P8 authorization boundary changed");
  if(canonicalM6Json(normalizedRows(value.protocolRows))!==canonicalM6Json(normalizedRows(sourceRows))||canonicalM6Json(value.protocolPathMap)!==canonicalM6Json(sourcePathMap)||value.verifierCommit!==verifierCommit||value.verifierTree!==verifierTree||canonicalM6Json(normalizedRows(value.verifierRows))!==canonicalM6Json(normalizedRows(verifierRows))) throw new Error("P8 authorization source binding changed");
  const ci=value.verifierPublicCi, expected=publicCi??{}; const ciKeys=["conclusion","event","headSha","runId","status","url","workflowPath"].sort(); if(!ci||JSON.stringify(Object.keys(ci).sort())!==JSON.stringify(ciKeys)||!Number.isSafeInteger(ci.runId)||ci.runId<=0||ci.status!=="completed"||ci.conclusion!=="success"||ci.event!=="push"||ci.workflowPath!==".github/workflows/quality.yml"||ci.headSha!==verifierCommit||ci.url!==`https://github.com/baney75/prooflens/actions/runs/${ci.runId}`||JSON.stringify(ci)!==JSON.stringify(expected)) throw new Error("P8 authorization CI proof changed"); return value;
}
export const M6_SUBMISSION_PAGES_COMMIT = "896429fdef327106704b01f7ea3b4772db4cea42";
export const M6_SUBMISSION_PAGES_TREE = "555081e194ab15a90cdc6815ebd270f69226ae97";
export const M6_SUBMISSION_PAGES_PARENT = "c4d71abfef2809125d21ad9797d13e9534d54860";
export const M6_P8_R_COMMIT = "c6ec48cfb5fe2c27971d1aa22746e91e11e3c025";
export const M6_P8_R_TREE = "aa2e54f22a114fb731b1489fa67ecfeb20e961f6";
export const M6_P8_A_TREE = "88268ef8003fadd467ae0158ba17222b627609de";
export const M6_P8_A_RECEIPT_SHA256 = "fd1cb09cf9407f58423fc5248bcd38caf336438566d28b919004b04f38dcb09d";
export const M6_SUBMISSION_PAGES_ROWS = Object.freeze([
  [".github/workflows/pages.yml", "A"],
  ["README.md", "M"],
  ["package.json", "M"],
  ["scripts/check-site.mjs", "A"],
  ["site/assets/seroslop.svg", "A"],
  ["site/index.html", "A"],
  ["site/styles.css", "A"],
]);
export const M6_SUBMISSION_PAGES_ARTIFACT_SHA256 = Object.freeze({
  ".github/workflows/pages.yml": "fa0d8d530376747a3b41753f03763b68bc4eda3e1df4e6ba3fa1ac3e2f1fc3f5",
  "README.md": "e1af63fc654ff9c1641370eda8b347619697c89f3ffa5c7624c867de0d41e1cb",
  "package.json": "49b9893e93bc5cf97f3edbc13e040f6265a3873d855efdd7dfd6e6dd8f599f83",
  "scripts/check-site.mjs": "c889861fe457fcd2c2e18e07be7931583be8363dff65ff2551ac645ac4c45081",
  "site/assets/seroslop.svg": "3305b345c480a6ea2f3ed7e7ae907c7ff2ebbffa9dd221342070cbe294ab9c9f",
  "site/index.html": "9f58e66bfa4f3b6c2995314fab69244964fd59a537b5c125b40eed3c61ae552b",
  "site/styles.css": "a0ad46eceb862790e6f1faf6c32f35f5439874cf3ae44703a022545b69fa0fc1",
});
export const M6_SUBMISSION_PROXY_LOCK_STATUS = "m2-bounty-proxy-pre-score-locked";
export const M6_SUBMISSION_PROXY_S_COMMIT = "b5bce594dba3b14bcfd279d5ccb4ad0af2dc9df0";
export const M6_SUBMISSION_PROXY_S_TREE = "cbd512770b2327dfea32ec88f6719a61ca0468cd";
export const M6_SUBMISSION_PROXY_S_ROWS = Object.freeze([
  ["benchmark/bounty_proxy_m2.py", "A"],
  ["benchmark/evidence/bounty-proxy-m2-v1/README.md", "A"],
  ["benchmark/evidence/bounty-proxy-m2-v1/frozen/manifest.jsonl", "A"],
  ["benchmark/evidence/bounty-proxy-m2-v1/frozen/source-lock.json", "A"],
  ["benchmark/test_bounty_proxy_m2.py", "A"],
  ["package.json", "M"],
  ["scripts/bounty-proxy-browser.mjs", "A"],
  ["scripts/test-bounty-proxy-browser.mjs", "A"],
]);
export const M6_SUBMISSION_PROXY_S_ARTIFACT_SHA256 = Object.freeze({
  "benchmark/bounty_proxy_m2.py": "7ff36769220508e5aa1392f7d9f33e55944f09927d431ef080be8b01cbf2077b",
  "benchmark/evidence/bounty-proxy-m2-v1/README.md": "bc4e9ad0d9daaf34f96ee44ed904080d728907fa5bb7c750630868781fb7de2b",
  "benchmark/evidence/bounty-proxy-m2-v1/frozen/manifest.jsonl": "2f713788a8813c185fdea450db46fa4d5243467fd3cd405953a3130bf06b62a5",
  "benchmark/evidence/bounty-proxy-m2-v1/frozen/source-lock.json": "fe49a561efc95618411931188668fa0180797a2b1afac22985475dc6fb24dd48",
  "benchmark/test_bounty_proxy_m2.py": "e212ea7c2f2a86d5bf6b02c81096a6362b976024cd3dcae18c226b75d6fb6be3",
  "package.json": "42e95ad85235ed48b4947733907c3d2eecb6df0e7919cf6330706c81a3981c9d",
  "scripts/bounty-proxy-browser.mjs": "91ec043179b7a7c25141ced0cb39d389684cb4b39570970b4c364c36ef1cc1fa",
  "scripts/test-bounty-proxy-browser.mjs": "d39eeb6182252a8f564ac6f7462dcc12924e2377493cb536e7372612a10439d7",
});
export const M6_SUBMISSION_PROXY_R_STATUS = "m2-bounty-proxy-verifier-ready";
export const M6_SUBMISSION_PROXY_R_COMMIT = "3c0e616a8e47e63f3d6a8e9511bedf1c28f478f3";
export const M6_SUBMISSION_PROXY_R_TREE = "847396c3b010de2ef7a22ff4e2bff7e032f3768a";
export const M6_SUBMISSION_PROXY_R_EXPECTED = Object.freeze([
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export function matchesM6SubmissionProxyRHead({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent &&
    parent === M6_SUBMISSION_PROXY_S_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_SUBMISSION_PROXY_R_EXPECTED));
}
export const M6_SUBMISSION_PROXY_R2_STATUS = "m2-bounty-proxy-ci-recovery-ready";
export const M6_SUBMISSION_PROXY_R2_COMMIT = "f69f185ff774b80a9dc09f23dadfc24e80f5c6b5";
export const M6_SUBMISSION_PROXY_R2_TREE = "904b1775e37b1f46bf1adb10c71820da666889b7";
export const M6_SUBMISSION_PROXY_R2_EXPECTED = Object.freeze([
  ["benchmark/test_bounty_proxy_m2.py", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_SUBMISSION_PROXY_R2_ARTIFACT_SHA256 = Object.freeze({
  "benchmark/test_bounty_proxy_m2.py": "c670552ae86e2de4f2a9646aeb7691aa0c642c35aac2e575378e238722f76f54",
});
export function matchesM6SubmissionProxyR2Head({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent &&
    parent === M6_SUBMISSION_PROXY_R_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_SUBMISSION_PROXY_R2_EXPECTED));
}
export const M6_SUBMISSION_PROXY_R3_STATUS = "m2-bounty-proxy-render-path-recovery-ready";
export const M6_SUBMISSION_PROXY_R3_EXPECTED = Object.freeze([
  ["scripts/bounty-proxy-browser.mjs", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-bounty-proxy-browser.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_SUBMISSION_PROXY_R3_ARTIFACT_SHA256 = Object.freeze({
  "scripts/bounty-proxy-browser.mjs": "31e290cad0100c9a2067dbb7bd993a44fa9892983e1be687d93b66aa41e27a87",
  "scripts/test-bounty-proxy-browser.mjs": "37cb78b8db7372769017b7ff50995831eb1a7cb48fe4460c2f39d341d80a38aa",
});
export function matchesM6SubmissionProxyR3Head({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent &&
    parent === M6_SUBMISSION_PROXY_R2_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_SUBMISSION_PROXY_R3_EXPECTED));
}
export const M6_PROTOCOL_RECOVERY_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"],
  ["benchmark/m6/contracts.py", "M"],
  ["benchmark/m6/prepare.py", "M"],
  ["benchmark/m6/recipe.json", "M"],
  ["benchmark/m6/test_contracts.py", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_MATERIALIZER_RECOVERY_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"],
  ["benchmark/m6/contracts.py", "M"],
  ["benchmark/m6/historical.py", "A"],
  ["benchmark/m6/materialize.py", "A"],
  ["benchmark/m6/prepare.py", "M"],
  ["benchmark/m6/recipe.json", "M"],
  ["benchmark/m6/source-shards.json", "A"],
  ["benchmark/m6/test_contracts.py", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_CI_RECOVERY_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"],
  ["benchmark/verify-requirements.txt", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_STAGES = Object.freeze(["m6-protocol", "m6-source-lock", "m6-preflight", "m6-trained", "m6-evaluated"]);
export const M6_P5_PARENT = "05a131b64fdef5f7fe8a6bdad4dac6d401e8193a";
export const M6_P5_COMMIT = "c878c2dc7ecbb49edb1cac4395aa20649471a330";
export const M6_P5_TREE = "4657121b4ba5c99006b8f0df8f0c4b629c78dc2d";
export const M6_P5_CI_RECOVERY_COMMIT = "57beb67fbfd9a140565b3b83436ba2893fd0fd82";
export const M6_P5_CI_RECOVERY_TREE = "bd75e0203f3b7156ee4cd3e870d22f1787d60b05";
export const M6_SUBMISSION_UI_COMMIT = "d3b712513e91d89c9de2ac7d958b1d24ca844b1a";
export const M6_SUBMISSION_UI_TREE = "25c896193faa5e26520c5ea4334e922e3a3ee74f";
export const M6_NO_SLOP_UI_COMMIT = "d4bc2fb9299534e821f05914d51cab0d41e7a030";
export const M6_NO_SLOP_UI_TREE = "347d36e24363215fd43e72d321ddf5260d007a32";
export const M6_P5_PROTOCOL_PATHS = Object.freeze(["benchmark/m6/DATA_PROVENANCE.md", "benchmark/m6/p5-protocol.json", "benchmark/m6/p5-quota-census.json", "benchmark/m6/p5_protocol.py", "benchmark/m6/p5_transform_fixture.py", "benchmark/m6/test_p5_protocol.py", "benchmark/m6/README.md", "benchmark/m6/THIRD_PARTY_NOTICES.md", "package.json", "scripts/m6-stage-policy.mjs", "scripts/check-m6-protocol-stage.mjs", "scripts/test-m6-stage-policy.mjs", "scripts/run-static-verification.mjs"]);
export const M6_P5_ARTIFACT_SHA256 = Object.freeze({
  "benchmark/m6/DATA_PROVENANCE.md": "858f02abaf94445387f1cbd91f8495b0f0e179bfc3fd63e6f678e32f6fa90523",
  "benchmark/m6/p5-protocol.json": "ebee34dab243eb320679fad9dcd2166f1da4af1bc2ac46986c11d0d024f4582b",
  "benchmark/m6/p5-quota-census.json": "373bb0c6aad9980c1f9860e98e16286d27e8a0bc54962c5d02a9209cfb41d47e",
  "benchmark/m6/p5_protocol.py": "ac3f64057071cdb9a059498f1ef0b6fcbf55bd8f0c27bda026f3d41589a3caa9",
  "benchmark/m6/p5_transform_fixture.py": "17076323643442cf104d23260b19968d07dfd5cb1156aae73ecbfd3d1ccf357b",
  "benchmark/m6/test_p5_protocol.py": "33ebd1400ee9e88daf80682b8e34b1fab342d434dd511ad9312706b878eaaad4",
  "benchmark/m6/README.md": "da12842e6c7b8f48bc54e60f9b5bde901ea640ac105148cd7dac2db180a6acb6",
  "benchmark/m6/THIRD_PARTY_NOTICES.md": "e66693fd6b18d089f127dfd5199bc88fd2e3ddf5b85924aea661bb15db730c27",
  "package.json": "fd0c21754713ad1bda2dd22c520133bd118fc46e35c551832b878e5e61d96f4e",
});
export const M6_P5_CI_RECOVERY_EXPECTED = Object.freeze([
  ["benchmark/m6/README.md", "M"],
  ["benchmark/m6/p5_transform_fixture.py", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_P5_RECOVERY_ARTIFACT_SHA256 = Object.freeze({
  ...M6_P5_ARTIFACT_SHA256,
  "benchmark/m6/p5_transform_fixture.py": "5cfbb8c3df33887aea2740003f8ef7ea39b2c691ca9464214f7eb739b399f73f",
  "benchmark/m6/README.md": "6fd0e35cf55fcfd580d2e634e22626c1910aacf4246e402fc67280bc3ca0e1ac",
});

// P6 is an append-only frontier child of the source-lock authorization head.
export const M6_P6_PARENT = "20154cb1bceda64aa74bbf1020f0a8ae8b2a513f";
export const M6_P6_PARENT_TREE = "8469c2609db99096695bf9d9f4ee88b54cc09d0b";
export const M6_P6_CHECK_STATUS = "m6-p6-metadata-only-unverified-pass";
export const M6_P6_S_COMMIT = "5efcd49f70ee4956770cc05f17a7e16e6f80d15a";
export const M6_P6_S_TREE = "0dacf4d6811138a73fec404de93a5236cc3cbab9";
export const M6_P6_R_EXPECTED = Object.freeze([
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_P6_S_ARTIFACT_SHA256 = Object.freeze({
  "benchmark/m6/p6-frontier-inventory.json": "9b3a2a75b5d6ce1262ca3c39a100dbca801f0a09dcf939be5478f740989bcf74",
  "benchmark/m6/p6_frontier_inventory.py": "5e52a074aac595b6bfae808adc47d1f53adaa64b609091cf50db07a4ae6d74ab",
  "benchmark/m6/test_p6_frontier_inventory.py": "e68bd93b3ce4668ab829330e589994d2fa0d0a916122993a9eceec27a7c180a5",
  "benchmark/m6/source-cards/aigenimages2026.README.md": "53d6bbd13bdeb16c6da0f4d7a780fca4b411422e26debf61450ec758847a6ff0",
  "benchmark/m6/source-cards/x-aigd.README.md": "e0121b9772a42b5953b68d26e99c4b0d6be29b7995156e13494a67b6dde99484",
  "benchmark/m6/source-cards/taste.README.md": "a602cf5bf11e67ab3e89f4fe3853326ae2978a8312f8d1633b1f2a9d3d1c562b",
  "benchmark/m6/source-cards/nano-banana.README.md": "d7f5f97cea4776c0a0c1ed97ba5139afda4f94dc42d5aec5dff2dd2c459f875c",
  "package.json": "d6e064d8c9df1e65e8359e8ca24c5cdf1f03356bb59fdab15af967d4e40d4450",
  "scripts/check-m6-protocol-stage.mjs": "26052bbefe0d6dfee7f6d6eb38c520e8fba0b14cf02b77ea6b2088a2e6b52310",
  "scripts/m6-stage-policy.mjs": "53d547570846b4b54b94b4553a7fd94b5c6738cd03a4effebf9b7fc5ad833225",
  "scripts/run-static-verification.mjs": "be653f9a20665a606c36a39000f88f68d6882fa505de75596f6caf10af800b68",
  "scripts/test-m6-stage-policy.mjs": "0533e414094938c5e7a931b7f8a3f83890f67ad6964f5c156b8f31dc48694fda",
});
export const M6_P6_R_STATUS = "m6-p6-verifier-ready";
export const M6_P6_AUTHORIZATION_PATH = "benchmark/evidence/m6/p6-protocol-authorization.json";
export const M6_P6_A_COMMIT = "285bc3eefcaff35a6ae8a6cc9b23b2d0abdd4f90";
export const M6_P6_R_COMMIT = "9d04c7fb49d79dae572007adbd917510daf26001";
export const M6_P6_A_TREE = "2457bb455d05fcef86aee07fb0c38cccd6ba289e";
export const M6_P6_A_RECEIPT_SHA256 = "c848f6581a73d4420248c55ae1f15cbfb7c1ceb5fcf2cd33171b06531781eede";
export const M6_P6_R2_EXPECTED = Object.freeze(M6_P6_R_EXPECTED);
export const M6_P6_R2_STATUS = "m6-p6-protocol-ci-recovery-ready";
export const M6_P6_EXPECTED = Object.freeze([
  ["benchmark/m6/p6-frontier-inventory.json", "A"],
  ["benchmark/m6/p6_frontier_inventory.py", "A"],
  ["benchmark/m6/test_p6_frontier_inventory.py", "A"],
  ["benchmark/m6/source-cards/aigenimages2026.README.md", "A"],
  ["benchmark/m6/source-cards/x-aigd.README.md", "A"],
  ["benchmark/m6/source-cards/taste.README.md", "A"],
  ["benchmark/m6/source-cards/nano-banana.README.md", "A"],
  ["package.json", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
  ["scripts/run-static-verification.mjs", "M"],
]);
export const M6_P6_ARTIFACT_SHA256 = Object.freeze({
  "benchmark/m6/p6-frontier-inventory.json": "9b3a2a75b5d6ce1262ca3c39a100dbca801f0a09dcf939be5478f740989bcf74",
  "benchmark/m6/p6_frontier_inventory.py": "5e52a074aac595b6bfae808adc47d1f53adaa64b609091cf50db07a4ae6d74ab",
  "benchmark/m6/test_p6_frontier_inventory.py": "e68bd93b3ce4668ab829330e589994d2fa0d0a916122993a9eceec27a7c180a5",
  "benchmark/m6/source-cards/aigenimages2026.README.md": "53d6bbd13bdeb16c6da0f4d7a780fca4b411422e26debf61450ec758847a6ff0",
  "benchmark/m6/source-cards/x-aigd.README.md": "e0121b9772a42b5953b68d26e99c4b0d6be29b7995156e13494a67b6dde99484",
  "benchmark/m6/source-cards/taste.README.md": "a602cf5bf11e67ab3e89f4fe3853326ae2978a8312f8d1633b1f2a9d3d1c562b",
  "benchmark/m6/source-cards/nano-banana.README.md": "d7f5f97cea4776c0a0c1ed97ba5139afda4f94dc42d5aec5dff2dd2c459f875c",
  "package.json": "d6e064d8c9df1e65e8359e8ca24c5cdf1f03356bb59fdab15af967d4e40d4450",
});
export function matchesM6P6Head({ head, parent, rows = [], treePaths = [] } = {}) {
  const expected = normalizedRows(M6_P6_EXPECTED);
  const actual = normalizedRows(rows);
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent && parent === M6_P6_PARENT &&
    JSON.stringify(actual) === JSON.stringify(expected) && M6_P6_EXPECTED.every(([path]) => treePaths.includes(path));
}
export function validateM6P6Artifacts(artifactBytes = {}) {
  const expectedPaths = Object.keys(M6_P6_ARTIFACT_SHA256).sort();
  if (JSON.stringify(Object.keys(artifactBytes).sort()) !== JSON.stringify(expectedPaths)) throw new Error("M6 P6 artifact inventory changed");
  for (const path of expectedPaths) {
    if (createHash("sha256").update(Buffer.from(artifactBytes[path])).digest("hex") !== M6_P6_ARTIFACT_SHA256[path]) throw new Error(`M6 P6 artifact bytes changed: ${path}`);
  }
  return true;
}
export function validateM6P6Inventory(bytes) {
  const raw = Buffer.from(bytes); const text = raw.toString("utf8");
  if (!text.endsWith("\n") || Buffer.from(text).compare(raw) !== 0) throw new Error("M6 P6 inventory must be canonical UTF-8 JSON");
  rejectDuplicateKeys(text); const value = JSON.parse(text);
  if (text !== canonicalM6Json(value)) throw new Error("M6 P6 inventory bytes are not canonical");
  if (["sourceLock", "trainingClaim", "acceptanceClaim", "benchmarkAcceptanceEligible"].some((key) => Object.hasOwn(value, key))) throw new Error("M6 P6 stage claim boundary changed");
  if (value.schemaVersion !== 1 || value.sourceCommit !== "d4bc2fb9299534e821f05914d51cab0d41e7a030" || value.status !== "metadata-only; no acquisition or materialization") throw new Error("M6 P6 inventory lineage changed");
  if (value.claims?.publisherMetadataIsGroundTruthBoundary !== true || value.claims?.commercialRightsClearanceClaimed !== false || value.claims?.h3PixelsRead !== false) throw new Error("M6 P6 claims boundary changed");
  if (value.quotaCorrection?.aigenimages2026TrainAdmittedRows !== 4879 || value.quotaCorrection?.omniSetTrainSynthetic !== 43783 || value.quotaCorrection?.rightsClaimed !== false) throw new Error("M6 P6 quota correction changed");
  const sources = value.sources ?? {};
  for (const source of Object.values(sources)) if (source.publisherAssertionOnly !== true || source.independentOriginProofClaimed !== false || source.labelEvidenceScope !== "publisherAssertionOnly") throw new Error("M6 P6 source boundary changed");
  return value;
}
export function matchesM6P6RHead({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent && parent === M6_P6_S_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_P6_R_EXPECTED));
}
export function matchesM6P6AuthorizationHead({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && parent !== M6_P6_S_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify([[M6_P6_AUTHORIZATION_PATH, "A"]]);
}
export function matchesM6P6R2Head({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent && parent === M6_P6_A_COMMIT && JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_P6_R2_EXPECTED));
}
export function validateM6P6Authorization(bytes, { sourceCommit, sourceTree, sourceRows, sourcePathMap, sourceParent = M6_P6_PARENT, publicCi, verifierCommit, verifierTree, verifierRows } = {}) {
  const raw = Buffer.from(bytes); const text = raw.toString("utf8");
  if (!text.endsWith("\n") || Buffer.from(text).compare(raw) !== 0) throw new Error("P6 authorization must be canonical UTF-8 JSON");
  rejectDuplicateKeys(text); const value = JSON.parse(text);
  if (text !== canonicalM6Json(value)) throw new Error("P6 authorization bytes are not canonical");
  const keys = ["acceptanceEligible", "authorizationPath", "commercialRightsClearanceClaimed", "h3PixelsRead", "independentOriginProofClaimed", "metadataOnly", "publisherAssertionOnly", "schemaVersion", "protocolCommit", "protocolParent", "protocolPathMap", "protocolRows", "protocolTree", "sourceLockAuthorized", "status", "trainingAuthorized", "verifierCommit", "verifierTree", "verifierRows", "verifierPublicCi"].sort();
  if (JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(keys)) throw new Error("P6 authorization schema changed");
  if (value.authorizationPath !== M6_P6_AUTHORIZATION_PATH || value.schemaVersion !== 1 || value.status !== "m6-p6-protocol-verified" || value.protocolCommit !== sourceCommit || value.protocolTree !== sourceTree || value.protocolParent !== sourceParent || value.metadataOnly !== true || value.publisherAssertionOnly !== true || value.independentOriginProofClaimed !== false || value.commercialRightsClearanceClaimed !== false || value.sourceLockAuthorized !== false || value.trainingAuthorized !== false || value.acceptanceEligible !== false || value.h3PixelsRead !== false) throw new Error("P6 authorization boundary changed");
  if (canonicalM6Json(normalizedRows(value.protocolRows)) !== canonicalM6Json(normalizedRows(sourceRows)) || canonicalM6Json(value.protocolPathMap) !== canonicalM6Json(sourcePathMap) || value.verifierCommit !== verifierCommit || value.verifierTree !== verifierTree || canonicalM6Json(normalizedRows(value.verifierRows)) !== canonicalM6Json(normalizedRows(verifierRows))) throw new Error("P6 authorization source binding changed");
  const ci = value.verifierPublicCi; const expectedCi = publicCi ?? {};
  const ciKeys = ["conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"].sort();
  if (!ci || JSON.stringify(Object.keys(ci).sort()) !== JSON.stringify(ciKeys) || !Number.isSafeInteger(ci.runId) || ci.runId <= 0) throw new Error("P6 authorization CI schema changed");
  if (!ci || ci.status !== "completed" || ci.conclusion !== "success" || ci.event !== "push" || ci.workflowPath !== ".github/workflows/quality.yml" || ci.headSha !== value.verifierCommit || ci.url !== `https://github.com/baney75/prooflens/actions/runs/${ci.runId}` || JSON.stringify(ci) !== JSON.stringify(expectedCi)) throw new Error("P6 authorization CI proof changed");
  return value;
}
export const M6_SUBMISSION_UI_EXPECTED = Object.freeze([
  ["scripts/build.mjs", "M"],
  ["scripts/check-benchmark-evidence.mjs", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
  ["src/static/popup.html", "M"],
  ["src/static/seroslop.svg", "A"],
]);
export const M6_SUBMISSION_UI_ARTIFACT_SHA256 = Object.freeze({
  "scripts/build.mjs": "bba20c2e37c0d1c8479c7ee0d3fa1b98956e033c9858871994a72bebadd64b3c",
  "scripts/check-benchmark-evidence.mjs": "5343e28c755c46d6b034d06991c3d0120d75aa77b69d933401d2c01d3ec15a93",
  "scripts/chrome-smoke.mjs": "a8f4c21b36e474a5c5b55ff349df110ef0336824b893a7f12104a8ef090fc33d",
  "src/static/popup.html": "747150fc7fc2e22de4a9cbf2e3afacd410d975380255abb5d84c5909062166bb",
  "src/static/seroslop.svg": "3305b345c480a6ea2f3ed7e7ae907c7ff2ebbffa9dd221342070cbe294ab9c9f",
});
export const M6_NO_SLOP_UI_EXPECTED = Object.freeze([
  ["DESIGN.md", "M"],
  ["scripts/browser-geometry-contract.mjs", "M"],
  ["scripts/build.mjs", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-browser-geometry-contract.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
  ["src/content.ts", "M"],
  ["src/popup.ts", "M"],
  ["src/setup.ts", "M"],
  ["src/static/icons/seroslop-128.png", "A"],
  ["src/static/icons/seroslop-16.png", "A"],
  ["src/static/icons/seroslop-32.png", "A"],
  ["src/static/icons/seroslop-48.png", "A"],
  ["src/static/manifest.json", "M"],
  ["src/static/setup.html", "M"],
  ["tests/manifest.test.ts", "M"],
]);
export const M6_NO_SLOP_UI_ARTIFACT_SHA256 = Object.freeze({
  "DESIGN.md": "432f12c90ece677dd947a8ea0dec370c7f6f476bce091bcaf6018bb4c2b842cb",
  "scripts/browser-geometry-contract.mjs": "d590a587ac18bc46fbc679eafce9eea9cb940ae6b6d86ade10e9f15d4455b031",
  "scripts/build.mjs": "f3c26c1af5d18a511330d8ce09185e3de56d4d7040e23c800f6dd7bfb736979e",
  "scripts/chrome-smoke.mjs": "5db788535afe6142dbee77377b6912aefa67453f614cba0f6c1024bc8cb02de4",
  "scripts/test-browser-geometry-contract.mjs": "cade1710c7b4be98154324c289279e6ff447cd8c8c4be66adb5b6fcf3ff8bc64",
  "src/content.ts": "e38c857774e2ac6962f2f59e8df3cc9769d2128ca5ea101b082e09904f00eb5e",
  "src/popup.ts": "d8f9bde4db68a26f0cce92379f2a4945e7ffafbb1c22265ba442e6fc872f7085",
  "src/setup.ts": "ff2eb523281f12a1f5c518fe74f9631e11960068f8757539954ef5030cfc76fd",
  "src/static/icons/seroslop-128.png": "106dc7077994c7fd19ca50b0397229951b35df32c21eca64865fd36e62b596b2",
  "src/static/icons/seroslop-16.png": "8e2f9c496b9bf1dbd00f902907b7a46759eb0efa4952c09f8d4fc0a5da3bcca2",
  "src/static/icons/seroslop-32.png": "04430af321509b88d3cddb083ea976ad4c6a39067574ed3f875e9c68a2b14ff0",
  "src/static/icons/seroslop-48.png": "9a86cb3c7b2697834dc3802091a5bf6c0ab4726f537fe6275636549e7b16e55d",
  "src/static/manifest.json": "f4189c1eda0b7fb06bad8a9ca537756693a4cbbde413df1b50c30265325aa77a",
  "src/static/setup.html": "bc3ddc345d5e3b1c26c1114856822f091d4034247e43501047457e60f645c00b",
  "tests/manifest.test.ts": "1292cb527a577f1b78ee4ad41019c4368cf4321b1ee19ab60026f011a32823ed",
});
export const M6_BETA1_EXPECTED = Object.freeze([
  [".github/workflows/quality.yml", "M"],
  [".gitignore", "M"],
  ["DESIGN.md", "M"],
  ["README.md", "M"],
  ["contributor/DATA_POLICY.md", "A"],
  ["contributor/PRIVACY.md", "A"],
  ["contributor/README.md", "A"],
  ["contributor/background.js", "A"],
  ["contributor/content.js", "A"],
  ["contributor/manifest.json", "A"],
  ["contributor/popup.css", "A"],
  ["contributor/popup.html", "A"],
  ["contributor/popup.js", "A"],
  ["contributor/test.mjs", "A"],
  ["package.json", "M"],
  ["scripts/build-contributor.mjs", "A"],
  ["scripts/check-benchmark-evidence.mjs", "M"],
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/contributor-chrome-smoke.mjs", "A"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
  ["src/background.ts", "M"],
  ["src/content.ts", "M"],
  ["src/popup.ts", "M"],
  ["src/setup.ts", "M"],
  ["src/shared/contracts.ts", "M"],
  ["src/shared/scan-mode.ts", "A"],
  ["src/static/popup.html", "M"],
  ["src/static/setup.html", "M"],
  ["tests/scan-mode.test.ts", "A"],
]);
export const M6_BETA1_ARTIFACT_SHA256 = Object.freeze({
  ".github/workflows/quality.yml": "3f61aafcc66b581cf60c89304dc1262b7532d138582e1c70378813f7a4e3511e",
  ".gitignore": "06b0704026e84cedc17383d15f8eddbb388ebdba72eb25e64d29d375b8342f87",
  "DESIGN.md": "38553cec88be787f627215c3cac0616b6c2a324db1b193c29bb27b48f02200ce",
  "README.md": "9664aa2401f5bc1dd1c31708ee278fbcaa3f539c4466aafb014cc6cde190e6b9",
  "contributor/DATA_POLICY.md": "0ac90532a1b2b2952cf7d1eb6018bef8a7f6d8a000adb37aa5c5dad0257e0bda",
  "contributor/PRIVACY.md": "bd5ab24c2b1c5a12f1853a2694a3521e55787ea1b6f87472c474b95f3ef940f1",
  "contributor/README.md": "158ca2a6c94e819ccd814fe5e355373f07c9371fec2d1b2457af674f907b28b9",
  "contributor/background.js": "cf32d508fedca15bcec16c6eec089f6b882198fc172dace487854330c39cf26c",
  "contributor/content.js": "a351f818606becd869c95afe22443ab0ff50dca6a7577863a44150c474092282",
  "contributor/manifest.json": "0f36a700497ecb77c7c6b182f671135f64c47002f7faed3786052dd0e0914845",
  "contributor/popup.css": "e475c4d0bd3f99cd7194e9131d0e5a6581a6a1cbe83c40a436c22d889a7bf7e2",
  "contributor/popup.html": "4243ce10222adb20cbd67053c3ac1382dbc4146b8809ffba46573a47352aabab",
  "contributor/popup.js": "c03c20db9d986708ea0f4feb95abbdb2c54d5d980afda3fde90a3615f6d77e0a",
  "contributor/test.mjs": "b3843a95813bc584aca1343be1bf08a9e03ecd15423e4b7fd1f85f40ac16c800",
  "package.json": "b6a15c2b71fc38b97b3c63d6360bf90d95775c8ec644b490766ca60a4d6ae994",
  "scripts/build-contributor.mjs": "deb69e2d03b589bff22d9cfee715c398af79930efe1274064d7e190f80d1b41d",
  "scripts/check-benchmark-evidence.mjs": "3e6f28237dda205b45674a427add9827185c3ecbcfbed2fa2574d5bd05627bbc",
  "scripts/chrome-smoke.mjs": "c84e02f692813b11364830b0268534766eb0c4f72b3e28d3605eec3196cffebe",
  "scripts/contributor-chrome-smoke.mjs": "9db59f642e577862d004783d0509b46427866025ebc23c27690857c9b3b5859a",
  "src/background.ts": "c455ce7387e53dd07928ef70cced113ab2f36604f1620f3c45c020bbd63e3ce4",
  "src/content.ts": "6f057d3dc247317d5808edaf51d54d483a45db5ff90c518cd8edce70e9eeb49b",
  "src/popup.ts": "a80771b3ca847d3ada28f5c11f70d03a12fd603bada78a425a4485506459f7bc",
  "src/setup.ts": "3acad049128a4a8881f74cdc5231cfc36990c0b9769609e128d0a8d49fabf61e",
  "src/shared/contracts.ts": "edc85a3c8bd9fad9113b5c3cc0b8f4bbbd2b3dfc31776cdb3e9ae8d23fd59120",
  "src/shared/scan-mode.ts": "11970df19a508fe8790b142f296d98c47df3083dd2dad4b3703610848c53df5c",
  "src/static/popup.html": "8b704d63de87f739b842aa9dd6c3464fcd835f005d5085e26a45ba0bfedde321",
  "src/static/setup.html": "531317dacd3af7d911ed7097b4a011aa50cbfa87d1eb020e8826822f861cf359",
  "tests/scan-mode.test.ts": "ace8ae5877b4294bb0dc370280783f0d61c8750d6f56e4231b77fecd8af2f865",
});
export const M6_BETA1_AUTHORIZATION_PATH = "benchmark/evidence/m6/beta1-authorization.json";
export const M6_BETA1_AUTHORIZATION_STATUS = "m6-beta1-source-authorized";
export const M6_BETA1_COMMIT = "296631aaaf6c4afd26982488c79f17163e14513f";
export const M6_BETA1_TREE = "68ee80e624a57f8bab21ab0dde96f5c13a789b2c";
export const M6_BETA1_RECOVERY_EXPECTED = Object.freeze([
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_BETA1_RECOVERY_ARTIFACT_SHA256 = Object.freeze({
  ...M6_BETA1_ARTIFACT_SHA256,
  "scripts/chrome-smoke.mjs": "ba70e516ab43cf6370c369833870d9176ee7a0635ce8f47c44c197d2d7590f7b",
});
export const M6_BETA1_RECOVERY_COMMIT = "340acf528e139676797ad7eb4ee7616f64b07102";
export const M6_BETA1_RECOVERY_TREE = "afabb73f9026d213251aea8fd6e5814c4b143296";
export const M6_BETA1_RECOVERY2_EXPECTED = Object.freeze([
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_BETA1_RECOVERY2_ARTIFACT_SHA256 = Object.freeze({
  ...M6_BETA1_ARTIFACT_SHA256,
  "scripts/chrome-smoke.mjs": "4bfacb25d1a77a1d1d0a3c23d12c6a7563686e27b91cec05ba41a7957d877dd9",
});
export const M6_BETA1_RECOVERY2_COMMIT = "8a4fc1ae61e2d27436ed42167c7defdcd1629f53";
export const M6_BETA1_RECOVERY2_TREE = "4310b1987a21370bb9aae58e05d0ad8967c603af";
export const M6_BETA1_RECOVERY3_EXPECTED = Object.freeze([
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_BETA1_RECOVERY3_ARTIFACT_SHA256 = Object.freeze({
  ...M6_BETA1_ARTIFACT_SHA256,
  "scripts/chrome-smoke.mjs": "19dc8cd3135b47b3b4c12874819911fda50ad021a09345ab0031b528f0c1bbcb",
});
export const M6_BETA1_RECOVERY3_COMMIT = "3ebda725a5fa6c7653831b3201e6d9669eb1cc4d";
export const M6_BETA1_RECOVERY3_TREE = "9c8ff707ce150997bb34f29f4e2f0bd3b98992b4";
export const M6_BETA1_RECOVERY4_EXPECTED = Object.freeze([
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
]);
export const M6_BETA1_RECOVERY4_ARTIFACT_SHA256 = Object.freeze({
  ...M6_BETA1_ARTIFACT_SHA256,
  "scripts/chrome-smoke.mjs": "34d456e683cb24bdc8618e2ea16f38bef6ea1b0580cc9ce1246ca6a911e93784",
});
export const M6_BETA1_RECOVERY4_COMMIT = "caecc4a788c9a79704b45900030dde8bc5dd357d";
export const M6_BETA1_RECOVERY4_TREE = "64349f9ea3c4dac3026b21e334f671dcb7a96c33";
export const M6_BETA1_RECOVERY5_EXPECTED = Object.freeze([
  ["scripts/check-m6-protocol-stage.mjs", "M"],
  ["scripts/chrome-smoke.mjs", "M"],
  ["scripts/m6-stage-policy.mjs", "M"],
  ["scripts/test-m6-stage-policy.mjs", "M"],
  ["src/popup.ts", "M"],
]);
export const M6_BETA1_RECOVERY5_ARTIFACT_SHA256 = Object.freeze({
  ...M6_BETA1_ARTIFACT_SHA256,
  "scripts/chrome-smoke.mjs": "0939b28d4c5366d7a847dd4edafd010e51a6594db4498f0d16e8f72f93208fec",
  "src/popup.ts": "fb56ae35b05fa26ffaecac00618a9ca73a183060163dbabc237c50972106d802",
});
export function validateM6P5Artifacts(artifactBytes = {}, expectedDigests = M6_P5_ARTIFACT_SHA256) {
  const expectedPaths = Object.keys(expectedDigests).sort();
  if (JSON.stringify(Object.keys(artifactBytes).sort()) !== JSON.stringify(expectedPaths)) throw new Error("M6 P5 artifact inventory changed");
  for (const path of expectedPaths) {
    const bytes = Buffer.from(artifactBytes[path]);
    if (createHash("sha256").update(bytes).digest("hex") !== expectedDigests[path]) throw new Error(`M6 P5 artifact bytes changed: ${path}`);
  }
  return true;
}
export function matchesProspectiveP5({ head, parent, paths = [], statuses = {} } = {}) {
  const expected = [...M6_P5_PROTOCOL_PATHS].sort(); const actual = [...new Set(paths)].sort();
  const additions = new Set(["benchmark/m6/DATA_PROVENANCE.md", "benchmark/m6/p5-protocol.json", "benchmark/m6/p5-quota-census.json", "benchmark/m6/p5_protocol.py", "benchmark/m6/p5_transform_fixture.py", "benchmark/m6/test_p5_protocol.py"]);
  return /^[0-9a-f]{40}$/.test(head ?? "") && parent === M6_P5_PARENT && JSON.stringify(expected) === JSON.stringify(actual) && M6_P5_PROTOCOL_PATHS.every((p) => statuses[p] === (additions.has(p) ? "A" : "M"));
}
export function matchesM6P5Head({ head, parent, rows = [], treePaths = [] } = {}) {
  const statuses = Object.fromEntries(rows.map(([path, status]) => [path, status]));
  return matchesProspectiveP5({ head, parent, paths: rows.map(([p]) => p), statuses }) && M6_P5_PROTOCOL_PATHS.every((p) => treePaths.includes(p));
}

export function matchesM6P5CiRecovery({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P5_COMMIT && parent === M6_P5_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_P5_CI_RECOVERY_EXPECTED));
}

export function matchesM6SubmissionUiHead({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P5_CI_RECOVERY_COMMIT &&
    parent === M6_P5_CI_RECOVERY_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_SUBMISSION_UI_EXPECTED));
}

export function matchesM6NoSlopUiHead({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_SUBMISSION_UI_COMMIT &&
    parent === M6_SUBMISSION_UI_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_NO_SLOP_UI_EXPECTED));
}

export function matchesM6Beta1Head({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_NO_SLOP_UI_COMMIT &&
    parent === M6_NO_SLOP_UI_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_BETA1_EXPECTED));
}

export function matchesM6Beta1AuthorizationHead({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && typeof parent === "string" && /^[0-9a-f]{40}$/.test(parent) &&
    head !== parent && JSON.stringify(normalizedRows(rows)) === JSON.stringify([[M6_BETA1_AUTHORIZATION_PATH, "A"]]);
}

export function matchesM6Beta1RecoveryHead({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent &&
    parent === M6_BETA1_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_BETA1_RECOVERY_EXPECTED));
}

export function matchesM6Beta1Recovery2Head({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent &&
    parent === M6_BETA1_RECOVERY_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_BETA1_RECOVERY2_EXPECTED));
}

export function matchesM6Beta1Recovery3Head({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent &&
    parent === M6_BETA1_RECOVERY2_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_BETA1_RECOVERY3_EXPECTED));
}

export function matchesM6Beta1Recovery4Head({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent &&
    parent === M6_BETA1_RECOVERY3_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_BETA1_RECOVERY4_EXPECTED));
}

export function matchesM6Beta1Recovery5Head({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== parent &&
    parent === M6_BETA1_RECOVERY4_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_BETA1_RECOVERY5_EXPECTED));
}

function canonicalM6Value(value) {
  if (Array.isArray(value)) return value.map(canonicalM6Value);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonicalM6Value(value[key])]));
  return value;
}

export function canonicalM6Json(value) {
  return `${JSON.stringify(canonicalM6Value(value))}\n`;
}

export function validateM6Beta1Authorization(bytes, { sourceCommit, sourceTree, sourcePathMap } = {}) {
  const valueBytes = Buffer.from(bytes);
  const text = valueBytes.toString("utf8");
  if (Buffer.from(text, "utf8").compare(valueBytes) !== 0 || !text.endsWith("\n")) throw new Error("M6 Beta1 authorization must be canonical UTF-8 JSON");
  rejectDuplicateKeys(text);
  const value = JSON.parse(text);
  if (text !== canonicalM6Json(value)) throw new Error("M6 Beta1 authorization bytes are not canonical");
  const keys = ["authorizationPath", "benchmarkAcceptanceEligible", "contributorUploadEnabled", "h3PixelsRead", "modelSha256", "schemaVersion", "sourceCommit", "sourcePathMap", "sourcePublicCi", "sourceTree", "status"].sort();
  if (JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(keys)) throw new Error("M6 Beta1 authorization schema changed");
  if (value.schemaVersion !== 1 || value.status !== M6_BETA1_AUTHORIZATION_STATUS || value.authorizationPath !== M6_BETA1_AUTHORIZATION_PATH ||
    value.sourceCommit !== sourceCommit || value.sourceTree !== sourceTree || value.modelSha256 !== "a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47" ||
    value.benchmarkAcceptanceEligible !== false || value.contributorUploadEnabled !== false || value.h3PixelsRead !== false) {
    throw new Error("M6 Beta1 authorization boundary changed");
  }
  const expectedPaths = M6_BETA1_EXPECTED.map(([path]) => path).sort();
  if (JSON.stringify(Object.keys(value.sourcePathMap ?? {}).sort()) !== JSON.stringify(expectedPaths) ||
    JSON.stringify(value.sourcePathMap) !== JSON.stringify(sourcePathMap)) throw new Error("M6 Beta1 authorization source map changed");
  if (Object.values(value.sourcePathMap).some((digest) => typeof digest !== "string" || !/^[0-9a-f]{64}$/.test(digest))) throw new Error("M6 Beta1 authorization source digest changed");
  const ci = value.sourcePublicCi;
  const ciKeys = ["conclusion", "event", "headSha", "runId", "status", "url", "workflowPath"].sort();
  if (!ci || JSON.stringify(Object.keys(ci).sort()) !== JSON.stringify(ciKeys) || !Number.isSafeInteger(ci.runId) || ci.runId <= 0 ||
    ci.status !== "completed" || ci.conclusion !== "success" || ci.event !== "push" || ci.headSha !== sourceCommit ||
    ci.workflowPath !== ".github/workflows/quality.yml" || ci.url !== `https://github.com/baney75/prooflens/actions/runs/${ci.runId}`) {
    throw new Error("M6 Beta1 source CI proof changed");
  }
  return value;
}

function rejectDuplicateKeys(text) {
  const stack = []; let inString = false; let escaped = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (inString) { if (escaped) escaped = false; else if (c === "\\") escaped = true; else if (c === '"') inString = false; continue; }
    if (c === '"') {
      let j = i + 1; let e = false;
      for (; j < text.length; j += 1) { if (e) { e = false; continue; } if (text[j] === "\\") { e = true; continue; } if (text[j] === '"') break; }
      let k = j + 1; while (/\s/.test(text[k] ?? "")) k += 1;
      if (text[k] === ":" && stack.length) { const key = JSON.parse(text.slice(i, j + 1)); if (stack.at(-1).has(key)) throw new Error(`duplicate M6 key: ${key}`); stack.at(-1).add(key); }
      i = j; continue;
    }
    if (c === "{") stack.push(new Set()); else if (c === "}") stack.pop();
  }
}

export function parseM6Recipe(bytes = readFileSync(M6_RECIPE_PATH)) {
  const text = Buffer.from(bytes).toString("utf8");
  if (Buffer.from(text, "utf8").compare(Buffer.from(bytes)) !== 0) throw new Error("M6 recipe must be strict UTF-8");
  const value = JSON.parse(text);
  rejectDuplicateKeys(text);
  if (createHash("sha256").update(bytes).digest("hex") !== M6_RECIPE_SHA256) throw new Error("M6 recipe digest changed");
  if (value.baseCommit !== M6_BASE_COMMIT) throw new Error("M6 base commit changed");
  if (value.evaluation?.items !== 100000 || value.evaluation?.batches !== 1000) throw new Error("M6 evaluation precommit changed");
  if (value.selector?.source !== "Omni-Fake-SET:image/validation") throw new Error("M6 selector source changed");
  if (value.selector?.generators?.length !== 34 || value.selector.generators.includes("Flux.1_pro")) throw new Error("M6 selector generator census changed");
  const allowed = new Map([["deliverable", new Set(["format","input","output","maximumBytes","browserExecution","networkAfterInstall"])],["sources", new Set(["omniFakeSet","omniFakeOOD"])],["selector", new Set(["source","baseItems","real","synthetic","syntheticSelection","generators","views","zeroObservedFalsePositive","wilsonConfidence","wilsonUpperBoundAtZero","poolViews","thresholdSearch","gates"])],["evaluation", new Set(["items","batches","batchSize","synthetic","assignedBeforeSelectorScoring","itemDisjoint","selectionInfluence","selectionExcludesSelector","strictMeanRecallGreaterThan","strictMedianBatchRecallGreaterThan","failureConsumesPanel"])]]);
  for (const [name, keys] of allowed) if (value[name] && Object.keys(value[name]).some((key) => !keys.has(key))) throw new Error(`unknown M6 ${name} key`);
  return value;
}

export function validateM6VerifyRequirements(bytes = readFileSync(M6_VERIFY_REQUIREMENTS_PATH)) {
  const value = Buffer.from(bytes);
  if (createHash("sha256").update(value).digest("hex") !== M6_VERIFY_REQUIREMENTS_SHA256) {
    throw new Error("M6 verification requirements bytes changed");
  }
  const text = value.toString("utf8");
  if (Buffer.from(text, "utf8").compare(value) !== 0 || !text.endsWith("\n")) {
    throw new Error("M6 verification requirements must be canonical UTF-8 text");
  }
  const lines = text.trimEnd().split("\n");
  if (lines.filter((line) => line === "pyarrow==20.0.0").length !== 1) {
    throw new Error("M6 verification requirements must add exactly pyarrow 20.0.0");
  }
  return true;
}

export function classifyM6Stage({ protocol = true, sourceLock = false, preflight = false, trained = false, evaluated = false } = {}) {
  if (sourceLock || preflight || trained || evaluated) throw new Error("M6 progressed stages require executable Git/artifact checker; booleans are not trusted");
  if (evaluated && !trained) throw new Error("evaluation requires training");
  if (trained && !preflight) throw new Error("training requires preflight");
  if (preflight && !sourceLock) throw new Error("preflight requires source lock");
  if (sourceLock && !protocol) throw new Error("source lock requires protocol");
  if (evaluated) return "m6-evaluated";
  if (trained) return "m6-trained";
  if (preflight) return "m6-preflight";
  if (sourceLock) return "m6-source-lock";
  return protocol ? "m6-protocol" : null;
}

export function matchesProspectiveP({ head, parents = [], paths = [], statuses = {} } = {}) {
  const required = [...M6_PROTOCOL_PATHS].sort(); const actual = [...new Set(paths)].sort();
  const expectedStatuses = Object.fromEntries(required.map((path) => [path, ["package.json", "scripts/run-static-verification.mjs"].includes(path) ? "M" : "A"]));
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_BASE_COMMIT && parents.length === 1 && parents[0] === M6_BASE_COMMIT && JSON.stringify(actual) === JSON.stringify(required) && JSON.stringify(statuses) === JSON.stringify(expectedStatuses);
}

export function isM6ProtocolHead({ head, parent, treePaths = [] } = {}) {
  return parent === M6_BASE_COMMIT && M6_PROTOCOL_PATHS.every((path) => treePaths.includes(path)) && head !== M6_BASE_COMMIT;
}

function normalizedRows(rows = []) {
  return [...rows].map(([path, status]) => [path, status]).sort((a, b) => a[0].localeCompare(b[0]));
}

export function matchesM6ProtocolRecovery({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P_COMMIT &&
    parent === M6_P_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_PROTOCOL_RECOVERY_EXPECTED));
}

export function matchesM6MaterializerRecovery({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P2_COMMIT &&
    parent === M6_P2_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_MATERIALIZER_RECOVERY_EXPECTED));
}

export function matchesM6CiRecovery({ head, parent, rows = [] } = {}) {
  return typeof head === "string" && /^[0-9a-f]{40}$/.test(head) && head !== M6_P3_COMMIT &&
    parent === M6_P3_COMMIT &&
    JSON.stringify(normalizedRows(rows)) === JSON.stringify(normalizedRows(M6_CI_RECOVERY_EXPECTED));
}

export function isM6ProtocolLineageHead({ head, parent, treePaths = [], rows = [] } = {}) {
  return matchesM6SubmissionProxyR3Head({ head, parent, rows }) || matchesM6SubmissionProxyR2Head({ head, parent, rows }) || matchesM6SubmissionProxyRHead({ head, parent, rows }) || matchesM6P8AuthorizationHead({ head, parent, rows }) || matchesM6P8RHead({ head, parent, rows }) || matchesM6P8Head({ head, parent, rows }) || matchesM6P7AuthorizationHead({ head, parent, rows }) || matchesM6P7RHead({ head, parent, rows }) || matchesM6P7Head({ head, parent, rows, treePaths }) || matchesM6P6R2Head({ head, parent, rows }) || matchesM6P6AuthorizationHead({ head, parent, rows }) || matchesM6P6RHead({ head, parent, rows }) || matchesM6P6Head({ head, parent, rows, treePaths }) ||
    isM6ProtocolHead({ head, parent, treePaths }) ||
    matchesM6P5Head({ head, parent, rows, treePaths }) ||
    matchesM6P5CiRecovery({ head, parent, rows }) ||
    matchesM6SubmissionUiHead({ head, parent, rows }) ||
    matchesM6NoSlopUiHead({ head, parent, rows }) ||
    matchesM6Beta1Head({ head, parent, rows }) ||
    matchesM6Beta1RecoveryHead({ head, parent, rows }) ||
    matchesM6Beta1Recovery2Head({ head, parent, rows }) ||
    matchesM6Beta1Recovery3Head({ head, parent, rows }) ||
    matchesM6Beta1Recovery4Head({ head, parent, rows }) ||
    matchesM6Beta1Recovery5Head({ head, parent, rows }) ||
    matchesM6Beta1AuthorizationHead({ head, parent, rows }) ||
    matchesM6ProtocolRecovery({ head, parent, rows }) ||
    matchesM6MaterializerRecovery({ head, parent, rows }) ||
    matchesM6CiRecovery({ head, parent, rows });
}

export function recipeSha256() { return createHash("sha256").update(readFileSync(M6_RECIPE_PATH)).digest("hex"); }
