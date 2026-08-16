#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, realpathSync } from "node:fs";
import { get } from "node:https";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { validateM5AuthorizedChain } from "./check-m5-authorized-chain.mjs";
import { M5_TRUSTED_PATH, assertM5WorktreeExact, m5Git, m5GitEnvironment } from "./m5-safe-git.mjs";
import {
  M5_LARGE_EVALUATION_PATH,
  M5_LARGE_SOURCE_EXPECTED,
  M5_LOCK_EXPECTED,
  matchesExpectedRows,
} from "./m5-stage-policy.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const WORKFLOW_PATH = ".github/workflows/quality.yml";
const NODE_VERSION = "v24.18.1";
const NODE_SHA256 = "f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a";
const NODE_PATH = "/workspace/.seroslop/runtime/node-v24.18.1-linux-x64/bin/node";
const PYTHON_PATH = "/opt/conda/bin/python";
const ALLOWED_IGNORED_PREFIXES = [
  ".verify-venv/",
  "benchmark/.venv/",
  "benchmark/candidates/",
  "benchmark/data/",
  "dist/",
  "node_modules/",
  "release/",
];
const MODES = Object.freeze({
  preflight: { script: "benchmark/m5/train_gpu.py", stage: "authorization", requiredFlag: "--preflight-only" },
  train: { script: "benchmark/m5/train_gpu.py", stage: "authorization", forbiddenFlag: "--preflight-only" },
  regress: { script: "benchmark/m5/evaluate_locked.py", stage: "selection", headFlag: "--lock-commit" },
  "lock-large-synthetic": { script: "benchmark/m5/large_synthetic.py", stage: "selection", headFlag: "--lock-commit" },
  "evaluate-large-synthetic": { script: "benchmark/m5/evaluate_large_synthetic.py", stage: "source-lock", headFlag: "--source-lock-commit" },
  finalize: { script: "benchmark/m5/finalize.py", stage: "source-lock", allowedUntracked: [M5_LARGE_EVALUATION_PATH] },
});

function git(arguments_, { cwd = ROOT, encoding = "utf8" } = {}) {
  return m5Git(arguments_, { cwd, encoding });
}

function commitRows(commit) {
  return git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit]).trim()
    .split("\n").filter(Boolean).map((line) => {
      const [status, pathname, extra] = line.split("\t");
      if (!status || !pathname || extra !== undefined) throw new Error(`Malformed M5 commit row: ${line}`);
      return [pathname, status];
    });
}

function parents(commit) {
  return git(["rev-list", "--parents", "-n", "1", commit]).trim().split(" ").slice(1);
}

function allowedIgnoredPath(pathname) {
  return ALLOWED_IGNORED_PREFIXES.some((prefix) => pathname.startsWith(prefix));
}

export function requireCleanM5PythonLaunchSurface(root = ROOT, allowedUntracked = []) {
  assertM5WorktreeExact({ root, allowedUntracked, allowedIgnoredPath });
}

function requireHeadStage(mode, authorized) {
  const head = git(["rev-parse", "HEAD"]).trim();
  if (mode.stage === "authorization") {
    if (head !== authorized.authorization) throw new Error("M5 train/preflight must run at the exact P4 authorization head");
    return head;
  }
  if (mode.stage === "selection") {
    if (parents(head).length !== 1 || parents(head)[0] !== authorized.authorization ||
        !matchesExpectedRows(commitRows(head), M5_LOCK_EXPECTED)) {
      throw new Error("M5 regression/materialization must run at the exact one-file selection-lock head");
    }
    return head;
  }
  const sourceParents = parents(head);
  if (sourceParents.length !== 1 || !matchesExpectedRows(commitRows(head), M5_LARGE_SOURCE_EXPECTED)) {
    throw new Error("M5 evaluation/finalization must run at the exact 100K source-lock head");
  }
  const lockCommit = sourceParents[0];
  if (parents(lockCommit).length !== 1 || parents(lockCommit)[0] !== authorized.authorization ||
      !matchesExpectedRows(commitRows(lockCommit), M5_LOCK_EXPECTED)) {
    throw new Error("M5 100K source lock is not the direct child of the authorized selection lock");
  }
  return head;
}

function bindCurrentHead(arguments_, flag, head) {
  const indexes = arguments_.flatMap((value, index) => value === flag ? [index] : []);
  if (indexes.length !== 1 || arguments_[indexes[0] + 1] !== "@HEAD") {
    throw new Error(`M5 launch requires exactly ${flag} @HEAD; the trusted launcher derives the commit`);
  }
  const bound = [...arguments_];
  bound[indexes[0] + 1] = head;
  return bound;
}

function getJson(url) {
  return new Promise((resolvePromise, reject) => {
    const request = get(url, { headers: { Accept: "application/vnd.github+json", "User-Agent": "seroslop-m5-python-launch" } }, (response) => {
      const chunks = [];
      response.on("data", (chunk) => chunks.push(chunk));
      response.on("end", () => {
        if (response.statusCode !== 200) return reject(new Error(`Unable to verify public M5 launch state: HTTP ${response.statusCode}`));
        try { resolvePromise(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch (error) { reject(error); }
      });
    });
    request.setTimeout(30_000, () => request.destroy(new Error("Public M5 launch verification timed out")));
    request.on("error", reject);
  });
}

async function requirePublicGreenCommit(commit) {
  const payload = await getJson(`https://api.github.com/repos/baney75/prooflens/actions/runs?event=push&head_sha=${commit}&per_page=100`);
  const row = payload.workflow_runs?.find((candidate) => candidate.head_sha === commit && candidate.event === "push" &&
    candidate.status === "completed" && candidate.conclusion === "success" && candidate.path === WORKFLOW_PATH);
  if (!row) throw new Error(`M5 Python launch requires exact-head successful public quality CI for ${commit}`);
  return row;
}

export async function launch(modeName, arguments_) {
  const mode = MODES[modeName];
  if (!mode) throw new Error(`Unknown M5 Python launch mode: ${modeName ?? "<missing>"}`);
  if (mode.requiredFlag && !arguments_.includes(mode.requiredFlag)) throw new Error(`M5 ${modeName} requires ${mode.requiredFlag}`);
  if (mode.forbiddenFlag && arguments_.includes(mode.forbiddenFlag)) throw new Error(`M5 ${modeName} forbids ${mode.forbiddenFlag}`);
  if (process.env.NODE_OPTIONS !== undefined || process.env.NODE_PATH !== undefined) {
    throw new Error("M5 Python launch requires NODE_OPTIONS and NODE_PATH to be removed by the outer RunPod boundary");
  }
  if (process.env.PATH !== M5_TRUSTED_PATH || Object.keys(process.env).some((name) => name.startsWith("GIT_"))) {
    throw new Error("M5 Python launch requires the trusted PATH and a clean Git environment from the outer RunPod boundary");
  }
  const actualNodePath = realpathSync(process.execPath);
  const actualNodeSha256 = createHash("sha256").update(readFileSync(actualNodePath)).digest("hex");
  if (process.platform !== "linux" || process.arch !== "x64" || process.version !== NODE_VERSION ||
      actualNodePath !== NODE_PATH || actualNodeSha256 !== NODE_SHA256) {
    throw new Error("M5 Python launch requires the exact pinned Linux Node runtime");
  }
  requireCleanM5PythonLaunchSurface(ROOT, mode.allowedUntracked ?? []);
  const authorized = validateM5AuthorizedChain();
  const head = requireHeadStage(mode, authorized);
  const boundArguments = mode.headFlag ? bindCurrentHead(arguments_, mode.headFlag, head) : arguments_;
  const publicReference = await getJson("https://api.github.com/repos/baney75/prooflens/git/ref/heads/main");
  const publicHead = publicReference.object?.sha;
  if (publicHead !== head) throw new Error("M5 Python launch requires the exact current head to be public main");
  await requirePublicGreenCommit(authorized.authorization);
  if (head !== authorized.authorization) await requirePublicGreenCommit(head);
  const environment = {
    ...m5GitEnvironment(),
    PYTHONNOUSERSITE: "1",
    PYTHONSAFEPATH: "1",
    PYTHONDONTWRITEBYTECODE: "1",
    SEROSLOP_M5_LAUNCH_NODE_VERSION: NODE_VERSION,
    SEROSLOP_M5_LAUNCH_NODE_SHA256: NODE_SHA256,
  };
  for (const name of ["PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONUSERBASE", "PYTHONBREAKPOINT"]) delete environment[name];
  const result = spawnSync(PYTHON_PATH, ["-I", mode.script, ...boundArguments], { cwd: ROOT, env: environment, stdio: "inherit" });
  if (result.error) throw result.error;
  return result.status ?? 1;
}

if (process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url) {
  const separator = process.argv.indexOf("--", 3);
  if (separator !== 3) throw new Error("M5 Python launcher requires: <mode> -- <fixed arguments>");
  process.exit(await launch(process.argv[2], process.argv.slice(separator + 1)));
}
