import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { lstatSync, readFileSync, realpathSync } from "node:fs";
import { dirname, resolve, sep } from "node:path";

export const M5_GIT_PATH = "/usr/bin/git";
export const M5_TRUSTED_PATH = "/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";

const FIXED_ARGUMENTS = Object.freeze([
  "-c", "core.fsmonitor=false",
  "-c", "core.hooksPath=/dev/null",
  "-c", "core.pager=cat",
  "-c", "core.attributesFile=/dev/null",
]);

export function m5GitEnvironment(source = process.env) {
  const environment = { ...source };
  for (const name of Object.keys(environment)) {
    if (name.startsWith("GIT_")) delete environment[name];
  }
  return {
    ...environment,
    PATH: M5_TRUSTED_PATH,
    HOME: "/nonexistent/seroslop-m5-git",
    XDG_CONFIG_HOME: "/nonexistent/seroslop-m5-git/xdg",
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_CONFIG_GLOBAL: "/dev/null",
    GIT_CONFIG_SYSTEM: "/dev/null",
    GIT_NO_REPLACE_OBJECTS: "1",
    GIT_OPTIONAL_LOCKS: "0",
    GIT_TERMINAL_PROMPT: "0",
    GIT_ASKPASS: "/bin/false",
    GIT_SSH_COMMAND: "/bin/false",
    GIT_PAGER: "cat",
  };
}

export function m5Git(arguments_, { cwd = process.cwd(), encoding = "utf8" } = {}) {
  const output = execFileSync(M5_GIT_PATH, [...FIXED_ARGUMENTS, ...arguments_], {
    cwd,
    encoding,
    env: m5GitEnvironment(),
    maxBuffer: 128 * 1024 * 1024,
  });
  return Buffer.isBuffer(output) ? output : output.trim();
}

export function m5GitBytes(arguments_, options = {}) {
  return m5Git(arguments_, { ...options, encoding: "buffer" });
}

function records(bytes) {
  return bytes.toString("utf8").split("\0").filter(Boolean);
}

function gitBlobOid(bytes) {
  return createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex");
}

export function assertM5WorktreeExact({
  root = process.cwd(),
  allowedUntracked = [],
  allowedIgnoredPath = undefined,
} = {}) {
  if (m5Git(["rev-parse", "--show-object-format"], { cwd: root }) !== "sha1") {
    throw new Error("M5 exact-worktree verification requires the frozen SHA-1 Git object format");
  }
  const indexFlags = records(m5GitBytes(["ls-files", "-v", "-z"], { cwd: root }));
  const abnormal = indexFlags.filter((record) => !record.startsWith("H "));
  if (abnormal.length) throw new Error(`M5 exact-worktree verification rejects non-normal index flags: ${abnormal.join(", ")}`);

  const index = new Map();
  for (const record of records(m5GitBytes(["ls-files", "--stage", "-z"], { cwd: root }))) {
    const match = /^(100644|100755) ([0-9a-f]{40}) 0\t(.+)$/u.exec(record);
    if (!match || index.has(match[3])) throw new Error(`M5 exact-worktree index row changed: ${record}`);
    index.set(match[3], { mode: match[1], oid: match[2] });
  }
  const committed = new Map();
  for (const record of records(m5GitBytes(["ls-tree", "-r", "-z", "--full-tree", "HEAD"], { cwd: root }))) {
    const match = /^(100644|100755) blob ([0-9a-f]{40})\t(.+)$/u.exec(record);
    if (!match || committed.has(match[3])) throw new Error(`M5 exact-worktree committed row changed: ${record}`);
    committed.set(match[3], { mode: match[1], oid: match[2] });
  }
  if (JSON.stringify([...index]) !== JSON.stringify([...committed])) {
    throw new Error("M5 exact-worktree index differs from the committed HEAD tree");
  }
  const rootPath = resolve(root);
  if (realpathSync(rootPath) !== rootPath) throw new Error("M5 repository root must not traverse a symlink");
  for (const [pathname, expected] of index) {
    const absolute = resolve(rootPath, pathname);
    if (!absolute.startsWith(`${rootPath}${sep}`)) throw new Error(`M5 tracked path escapes the repository: ${pathname}`);
    for (let parent = dirname(absolute); parent !== rootPath; parent = dirname(parent)) {
      if (lstatSync(parent).isSymbolicLink()) throw new Error(`M5 tracked path traverses a symlink: ${pathname}`);
    }
    const stat = lstatSync(absolute);
    if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`M5 tracked file is missing, non-regular, or symlinked: ${pathname}`);
    const executable = Boolean(stat.mode & 0o111);
    if (executable !== (expected.mode === "100755")) throw new Error(`M5 tracked file mode changed: ${pathname}`);
    if (gitBlobOid(readFileSync(absolute)) !== expected.oid) throw new Error(`M5 tracked file bytes changed: ${pathname}`);
  }
  const allowed = new Set(allowedUntracked);
  const untracked = records(m5GitBytes(["ls-files", "--others", "--exclude-standard", "-z"], { cwd: root }));
  const unexpected = untracked.filter((pathname) => !allowed.delete(pathname));
  if (unexpected.length || allowed.size) {
    throw new Error(`M5 exact-worktree untracked surface changed: ${[...unexpected, ...allowed].join(", ")}`);
  }
  if (allowedIgnoredPath) {
    const ignored = records(m5GitBytes(["ls-files", "--others", "--ignored", "--exclude-standard", "--directory", "-z"], { cwd: root }));
    const unsafe = ignored.filter((pathname) => !allowedIgnoredPath(pathname));
    if (unsafe.length) throw new Error(`M5 exact-worktree rejects ignored executable/import surface: ${unsafe.join(", ")}`);
  }
}
