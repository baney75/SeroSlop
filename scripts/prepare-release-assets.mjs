import { createHash } from "node:crypto";
import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

export async function prepareReleaseAssets({ root, channel, commit, refName }) {
  if (channel !== "stable" && channel !== "nightly") throw new Error("release channel must be stable or nightly");
  if (!/^[0-9a-f]{40}$/u.test(commit)) throw new Error("release commit must be a full Git SHA-1");
  const manifest = JSON.parse(await readFile(path.join(root, "src/static/manifest.json"), "utf8"));
  if (!/^\d+\.\d+\.\d+(?:\.\d+)?$/u.test(manifest.version)) throw new Error("manifest version is not a Chrome extension version");
  if (channel === "stable" && refName !== `v${manifest.version}`) {
    throw new Error(`stable release tag must be v${manifest.version}`);
  }
  const source = path.join(root, "release/prooflens.zip");
  const archiveBytes = await readFile(source);
  const digest = sha256(archiveBytes);
  const suffix = channel === "stable" ? manifest.version : `${commit.slice(0, 12)}`;
  const archive = `seroslop-${channel === "stable" ? suffix : `nightly-${suffix}`}.zip`;
  const checksum = `${archive}.sha256`;
  const publishDirectory = path.join(root, "release/publish");
  await rm(publishDirectory, { recursive: true, force: true });
  await mkdir(publishDirectory, { recursive: true });
  await cp(source, path.join(publishDirectory, archive));
  await writeFile(path.join(publishDirectory, checksum), `${digest}  ${archive}\n`);
  const metadata = {
    archive,
    channel,
    checksum,
    commit,
    manifestVersion: manifest.version,
    sha256: digest,
  };
  await writeFile(path.join(publishDirectory, "release-metadata.json"), `${JSON.stringify(metadata, null, 2)}\n`);
  return metadata;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [channel, commit, refName] = process.argv.slice(2);
  const metadata = await prepareReleaseAssets({ root: process.cwd(), channel, commit, refName });
  console.log(JSON.stringify(metadata));
}
