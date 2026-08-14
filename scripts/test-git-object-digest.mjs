import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { digestFileStreaming, digestGitBlob } from "./git-object-digest.mjs";

const repository = await mkdtemp(path.join(os.tmpdir(), "prooflens-git-digest-"));
try {
  execFileSync("git", ["init", "-b", "main"], { cwd: repository, stdio: "ignore" });
  execFileSync("git", ["config", "user.name", "ProofLens test"], { cwd: repository });
  execFileSync("git", ["config", "user.email", "test@example.invalid"], { cwd: repository });
  const marker = Buffer.from("PROOFLENS_STREAMING_SENTINEL_DO_NOT_LOG");
  const bytes = Buffer.alloc(2 * 1024 * 1024 + marker.length, 0x5a);
  marker.copy(bytes, bytes.length - marker.length);
  const largePath = path.join(repository, "large.bin");
  await writeFile(largePath, bytes);
  execFileSync("git", ["add", "large.bin"], { cwd: repository });
  execFileSync("git", ["commit", "-m", "Add large fixture"], { cwd: repository, stdio: "ignore" });
  const expected = createHash("sha256").update(bytes).digest("hex");
  assert.deepEqual(await digestGitBlob("HEAD:large.bin", { cwd: repository }), {
    sha256: expected,
    bytes: bytes.length,
  });
  assert.deepEqual(await digestFileStreaming(largePath), { sha256: expected, bytes: bytes.length });
  await assert.rejects(() => digestGitBlob("HEAD:missing.bin", { cwd: repository }), (error) => {
    assert.match(error.message, /Git blob HEAD:missing\.bin failed/u);
    assert.equal(error.message.includes(marker.toString("utf8")), false);
    return true;
  });
  if (process.platform !== "win32") {
    const fakeBin = path.join(repository, "fake-bin");
    await mkdir(fakeBin);
    const fakeGit = path.join(fakeBin, "git");
    await writeFile(fakeGit, "#!/bin/sh\nprintf 'INTERRUPTED_SENTINEL'\nkill -TERM $$\n");
    await chmod(fakeGit, 0o755);
    await assert.rejects(
      () => digestGitBlob("HEAD:large.bin", {
        cwd: repository,
        env: { ...process.env, PATH: `${fakeBin}${path.delimiter}${process.env.PATH ?? ""}` },
      }),
      (error) => {
        assert.match(error.message, /signal SIGTERM/u);
        assert.equal(error.message.includes("INTERRUPTED_SENTINEL"), false);
        return true;
      },
    );
  }
  assert.notEqual(expected, "0".repeat(64));
  console.log(JSON.stringify({ cases: process.platform === "win32" ? 4 : 5,
    streamedGitObjectBytes: bytes.length, policy: "pass" }));
} finally {
  await rm(repository, { recursive: true, force: true });
}
