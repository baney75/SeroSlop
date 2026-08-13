import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";

const archivePath = "release/prooflens.zip";
const sha256 = async () => createHash("sha256").update(await readFile(archivePath)).digest("hex");
const buildUnder = (timezone) => {
  execFileSync(process.execPath, ["scripts/build.mjs"], {
    env: { ...process.env, TZ: timezone },
    stdio: "ignore",
  });
};

buildUnder("America/New_York");
const newYorkHash = await sha256();
buildUnder("UTC");
const utcHash = await sha256();
if (newYorkHash !== utcHash) {
  throw new Error(`Release archive is timezone-dependent: ${newYorkHash} != ${utcHash}`);
}
console.log(JSON.stringify({ archive: archivePath, sha256: utcHash, timezones: ["America/New_York", "UTC"] }));
