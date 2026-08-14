import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";

const MAX_STDERR_BYTES = 64 * 1024;

function digestStream(stream, label, processResult) {
  return new Promise((resolve, reject) => {
    const hash = createHash("sha256");
    let bytes = 0;
    let settled = false;
    const fail = (error) => {
      if (settled) return;
      settled = true;
      reject(error);
    };
    stream.on("data", (chunk) => {
      hash.update(chunk);
      bytes += chunk.length;
    });
    stream.on("error", (error) => fail(new Error(`${label} stream failed`, { cause: error })));
    processResult((error) => {
      if (error) return fail(error);
      if (settled) return;
      settled = true;
      resolve({ sha256: hash.digest("hex"), bytes });
    });
  });
}

export async function digestGitBlob(revisionPath, { cwd = process.cwd(), env = process.env } = {}) {
  const child = spawn("git", ["cat-file", "blob", revisionPath], {
    cwd,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stderr = "";
  child.stderr.on("data", (chunk) => {
    if (Buffer.byteLength(stderr) >= MAX_STDERR_BYTES) return;
    stderr += chunk.toString("utf8", 0, Math.max(0, MAX_STDERR_BYTES - Buffer.byteLength(stderr)));
  });
  return digestStream(child.stdout, `Git blob ${revisionPath}`, (done) => {
    child.once("error", (error) => done(new Error(`Could not start Git blob reader for ${revisionPath}`, { cause: error })));
    child.once("close", (code, signal) => {
      if (code === 0 && signal === null) return done();
      const detail = stderr.trim() ? `: ${stderr.trim()}` : "";
      done(new Error(`Git blob ${revisionPath} failed (code ${String(code)}, signal ${String(signal)})${detail}`));
    });
  });
}

export async function digestFileStreaming(file) {
  const stream = createReadStream(file);
  return digestStream(stream, `File ${file}`, (done) => stream.once("end", () => done()));
}
