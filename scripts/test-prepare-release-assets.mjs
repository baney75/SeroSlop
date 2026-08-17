import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { prepareReleaseAssets } from "./prepare-release-assets.mjs";

const root = await mkdtemp(path.join(os.tmpdir(), "seroslop-release-test-"));
try {
  await mkdir(path.join(root, "src/static"), { recursive: true });
  await mkdir(path.join(root, "release"), { recursive: true });
  await writeFile(path.join(root, "src/static/manifest.json"), '{"version":"1.2.3"}\n');
  const archiveBytes = Buffer.from("fixed release bytes");
  await writeFile(path.join(root, "release/prooflens.zip"), archiveBytes);
  const commit = "a".repeat(40);
  const expectedDigest = createHash("sha256").update(archiveBytes).digest("hex");

  const stable = await prepareReleaseAssets({ root, channel: "stable", commit, refName: "v1.2.3" });
  assert.equal(stable.archive, "seroslop-1.2.3.zip");
  assert.equal(stable.sha256, expectedDigest);
  assert.equal(await readFile(path.join(root, "release/publish", stable.checksum), "utf8"), `${expectedDigest}  ${stable.archive}\n`);
  assert.deepEqual(await readFile(path.join(root, "release/publish", stable.archive)), archiveBytes);

  const nightly = await prepareReleaseAssets({ root, channel: "nightly", commit, refName: "main" });
  assert.equal(nightly.archive, "seroslop-nightly-aaaaaaaaaaaa.zip");
  assert.equal(nightly.channel, "nightly");
  await assert.rejects(
    prepareReleaseAssets({ root, channel: "stable", commit, refName: "v9.9.9" }),
    /stable release tag must be v1\.2\.3/u,
  );
  await assert.rejects(
    prepareReleaseAssets({ root, channel: "preview", commit, refName: "main" }),
    /stable or nightly/u,
  );
  console.log("release channel test: PASS");
} finally {
  await rm(root, { recursive: true, force: true });
}
