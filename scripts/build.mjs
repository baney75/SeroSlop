import { createHash } from "node:crypto";
import { build } from "esbuild";
import { cp, mkdir, readFile, readdir, rm, stat, utimes, writeFile } from "node:fs/promises";
import path from "node:path";
import { execFileSync } from "node:child_process";

const root = process.cwd();
const outdir = path.join(root, "dist");
const releaseDir = path.join(root, "release");
const modelPath = path.join(root, "weights", "prooflens-cf384.onnx");
const modelLock = JSON.parse(await readFile(path.join(root, "model-lock.json"), "utf8"));
const expectedModelHash = modelLock.sha256;
const expectedModelBytes = modelLock.bytes;
const reproducibleTime = new Date("2000-01-01T00:00:00.000Z");

function digest(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

const modelBytes = await readFile(modelPath);
if (modelBytes.byteLength !== expectedModelBytes || digest(modelBytes) !== expectedModelHash) {
  throw new Error("weights/prooflens-cf384.onnx does not match model-lock.json");
}

await rm(outdir, { recursive: true, force: true });
await rm(releaseDir, { recursive: true, force: true });
await mkdir(outdir, { recursive: true });
await mkdir(releaseDir, { recursive: true });

for (const name of ["manifest.json", "offscreen.html", "popup.html", "setup.html", "seroslop.svg"]) {
  await cp(path.join(root, "src", "static", name), path.join(outdir, name));
}
await cp(path.join(root, "src", "static", "icons"), path.join(outdir, "icons"), { recursive: true });
await cp(path.join(root, "model-lock.json"), path.join(outdir, "model-lock.json"));
await cp(path.join(root, "LICENSE"), path.join(outdir, "LICENSE"));
await cp(path.join(root, "THIRD_PARTY_NOTICES.md"), path.join(outdir, "THIRD_PARTY_NOTICES.md"));
await cp(path.join(root, "LICENSES"), path.join(outdir, "LICENSES"), { recursive: true });

const ortSource = path.join(root, "node_modules", "onnxruntime-web", "dist");
const ortTarget = path.join(outdir, "ort");
await mkdir(ortTarget, { recursive: true });
const runtimeFiles = (await readdir(ortSource)).filter(
  (name) => name.startsWith("ort-wasm") && (name.endsWith(".wasm") || name.endsWith(".mjs")),
);
if (!runtimeFiles.some((name) => name.endsWith(".wasm"))) throw new Error("ONNX Runtime WASM assets are missing");
await Promise.all(runtimeFiles.map((name) => cp(path.join(ortSource, name), path.join(ortTarget, name))));

const weightsTarget = path.join(outdir, "weights");
await mkdir(weightsTarget, { recursive: true });
await cp(modelPath, path.join(weightsTarget, "prooflens-cf384.onnx"));

await build({
  entryPoints: {
    background: "src/background.ts",
    content: "src/content.ts",
    offscreen: "src/offscreen.ts",
    popup: "src/popup.ts",
    setup: "src/setup.ts",
  },
  bundle: true,
  entryNames: "[name]",
  format: "iife",
  outdir,
  platform: "browser",
  target: "chrome121",
  sourcemap: true,
  minify: false,
  logLevel: "info",
});

async function filesUnder(directory, prefix = "") {
  const output = [];
  for (const name of (await readdir(directory)).sort()) {
    const absolute = path.join(directory, name);
    const relative = path.join(prefix, name);
    if ((await stat(absolute)).isDirectory()) output.push(...(await filesUnder(absolute, relative)));
    else output.push(relative);
  }
  return output;
}

const releaseFiles = await filesUnder(outdir);
await Promise.all(releaseFiles.map((name) => utimes(path.join(outdir, name), reproducibleTime, reproducibleTime)));
const archive = path.join(releaseDir, "prooflens.zip");
// Info-ZIP writes DOS timestamps in the process timezone. Pin it so the
// byte-level release is identical on developer machines and CI.
execFileSync("zip", ["-X", "-q", archive, ...releaseFiles], {
  cwd: outdir,
  env: { ...process.env, TZ: "UTC" },
});
const archiveBytes = await readFile(archive);
const archiveHash = digest(archiveBytes);
await writeFile(path.join(releaseDir, "SHA256SUMS.txt"), `${archiveHash}  prooflens.zip\n`);
console.log(JSON.stringify({ archive: "release/prooflens.zip", bytes: archiveBytes.byteLength, sha256: archiveHash }));
