import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

const expectedModelHash = "29545a1da0cfe2bf0149448334fd45a21f48074c57296db3b84437dd66f80a43";
const model = await readFile("dist/weights/prooflens-cf384.onnx");
if (createHash("sha256").update(model).digest("hex") !== expectedModelHash) throw new Error("Packaged model hash mismatch");

const requiredNotices = new Map([
  ["dist/LICENSE", "3cfa8ec8f09d7f680e4fa0777f584016dc9f10b69a9fc8808eca2603dfc3f52a"],
  ["dist/THIRD_PARTY_NOTICES.md", "f09d36cb40f016766af31aaa528e61b0fe420d7b1181d1ac87c02fa63a7ed01b"],
  ["dist/LICENSES/APACHE-2.0.txt", "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"],
  ["dist/LICENSES/COMMUNITY_FORENSICS_MODEL_MIT.txt", "69a0eab6ca179df33ed80fa378b9458632e14ba9547374e299249e0a4f8076cb"],
  ["dist/LICENSES/ONNX_RUNTIME_MIT.txt", "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c"],
  ["dist/LICENSES/ONNX_RUNTIME_THIRD_PARTY_NOTICES.txt", "e9e90971a8e75a9a8ac0c6412e29c1202d079998389915aa485f46c816c3b4cc"],
  ["dist/LICENSES/SYNTHCHECK_MIT.txt", "5a8ee7ffa018b7d8e888903acba24b16072ce84e95bf07d7fb6ebdd8a10f9c84"],
]);
for (const [notice, expectedHash] of requiredNotices) {
  const bytes = await readFile(notice);
  if (expectedHash && createHash("sha256").update(bytes).digest("hex") !== expectedHash) {
    throw new Error(`Packaged notice hash mismatch: ${notice}`);
  }
}
const noticeIndex = await readFile("dist/THIRD_PARTY_NOTICES.md", "utf8");
for (const required of ["Community Forensics model", "ONNX Runtime Web 1.22.0", "SynthCheck"]) {
  if (!noticeIndex.includes(required)) throw new Error(`Third-party notice index omits ${required}`);
}

async function filesUnder(directory) {
  const output = [];
  for (const name of await readdir(directory)) {
    const absolute = path.join(directory, name);
    if ((await stat(absolute)).isDirectory()) output.push(...(await filesUnder(absolute)));
    else output.push(absolute);
  }
  return output;
}

const files = await filesUnder("dist");
const textFiles = files.filter((name) => /\.(?:js|html|json|mjs)$/u.test(name));
const combined = (await Promise.all(textFiles.map((name) => readFile(name, "utf8")))).join("\n");
for (const forbidden of [
  /fetch\(\s*(?:source|message\.source|record\.source)/u,
  /https?:\/\/localhost/iu,
  /https?:\/\/127\./iu,
  /segment\.com/iu,
  /google-analytics/iu,
  /sentry\.io/iu,
]) {
  if (forbidden.test(combined)) throw new Error(`Forbidden packaged runtime reference: ${forbidden}`);
}
const manifest = JSON.parse(await readFile("dist/manifest.json", "utf8"));
if (manifest.manifest_version !== 3) throw new Error("Package is not Manifest V3");
if (!String(manifest.content_security_policy?.extension_pages).includes("script-src 'self'")) {
  throw new Error("Package CSP does not restrict scripts to self");
}
if (!files.some((name) => name.endsWith(".wasm"))) throw new Error("Package lacks WASM fallback assets");
if (!combined.includes('redirect: "error"')) throw new Error("Local image reads do not reject redirects");
if (!combined.includes("Only locally rendered image pixels are accepted")) {
  throw new Error("Offscreen inference does not enforce local image pixels");
}
const guardedLocalRead = combined.indexOf("assertLocalImageUrl(value)");
const loadImageMethod = combined.indexOf("async loadImage(value)");
const guardedLoadImage = combined.indexOf("assertLocalImageUrl(value)", loadImageMethod);
const localFetch = combined.indexOf("fetch(value", loadImageMethod);
if (guardedLocalRead < 0 || loadImageMethod < 0 || guardedLoadImage < loadImageMethod || localFetch < guardedLoadImage ||
  localFetch - guardedLoadImage > 256) {
  throw new Error("Local image fetch is not immediately preceded by the data-image policy guard");
}
const archiveEntries = execFileSync("unzip", ["-Z1", "release/prooflens.zip"], { encoding: "utf8" }).trim().split("\n");
for (const notice of requiredNotices.keys()) {
  const relative = notice.replace(/^dist\//u, "");
  if (!archiveEntries.includes(relative)) throw new Error(`Release archive omits ${relative}`);
}
console.log(JSON.stringify({ files: files.length, modelSha256: expectedModelHash, notices: requiredNotices.size, policy: "pass" }));
