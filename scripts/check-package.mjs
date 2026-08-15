import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

const modelLock = JSON.parse(await readFile("dist/model-lock.json", "utf8"));
const expectedModelHash = modelLock.sha256;
const model = await readFile("dist/weights/prooflens-cf384.onnx");
if (model.byteLength !== modelLock.bytes || createHash("sha256").update(model).digest("hex") !== expectedModelHash) {
  throw new Error("Packaged model does not match model-lock.json");
}

const requiredNotices = new Map([
  ["dist/LICENSE", createHash("sha256").update(await readFile("LICENSE")).digest("hex")],
  ["dist/THIRD_PARTY_NOTICES.md", createHash("sha256").update(await readFile("THIRD_PARTY_NOTICES.md")).digest("hex")],
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

const sourceFiles = (await filesUnder("src")).filter((name) => /\.(?:ts|html)$/u.test(name));
const detectorSourcePath = path.normalize("src/inference/detector.ts");
for (const sourceFile of sourceFiles) {
  const source = await readFile(sourceFile, "utf8");
  if (sourceFile !== detectorSourcePath && /\bfetch\s*\(/u.test(source)) {
    throw new Error(`Only the guarded local detector may call fetch: ${sourceFile}`);
  }
  for (const primitive of [/\bXMLHttpRequest\b/u, /\bWebSocket\b/u, /\bEventSource\b/u, /\bsendBeacon\b/u]) {
    if (primitive.test(source)) throw new Error(`Forbidden source network primitive in ${sourceFile}: ${primitive}`);
  }
}
const detectorSource = await readFile(detectorSourcePath, "utf8");
if ([...detectorSource.matchAll(/\bfetch\s*\(/gu)].length !== 2 ||
  !detectorSource.includes("fetch(chrome.runtime.getURL(MODEL_SPEC.bundledWeightsPath)") ||
  !detectorSource.includes("assertLocalImageUrl(value)") || !detectorSource.includes("fetch(value")) {
  throw new Error("Detector fetch surface changed from the two guarded package/data-image reads");
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
if (manifest.name !== "SeroSlop" || manifest.action?.default_title !== "SeroSlop") {
  throw new Error("Packaged extension branding is not SeroSlop");
}
const extensionPolicy = String(manifest.content_security_policy?.extension_pages);
for (const directive of [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "object-src 'none'",
  "connect-src 'self' data: blob:",
  "img-src 'self' data: blob:",
  "style-src 'self' 'unsafe-inline'",
  "worker-src 'self'",
]) {
  if (!extensionPolicy.includes(directive)) {
    throw new Error(`Package CSP is missing its local-only directive: ${directive}`);
  }
}
if (/https?:/u.test(extensionPolicy)) {
  throw new Error("Package CSP permits a remote extension-page origin");
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
