import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";
import { parseCanonicalM5Authorization, requireM5AuthorizationSchema } from "./check-m5-authorized-chain.mjs";
import { m5Git } from "./m5-safe-git.mjs";
import { requireCleanM5PythonLaunchSurface } from "./m5-python-launch.mjs";
import {
  M5_BASE_SOURCE_COMMIT,
  M5_BASE_SOURCE_TREE,
  M5_ORIGINAL_PROTOCOL_COMMIT,
  M5_ORIGINAL_PROTOCOL_TREE,
  M5_PROTOCOL_EXPECTED,
  M5_PROTOCOL_RECOVERY_EXPECTED,
  M5_FAILED_SOURCE_COMMIT,
  M5_FAILED_SOURCE_TREE,
  M5_CI_RECOVERY_COMMIT,
  M5_P4_COMMIT,
  M5_P4_TREE,
  M5_P2_COMMIT,
  M5_NUMERIC_AUDIT_AUTHORIZATION_PATH,
  M5_NUMERIC_AUDIT_RECOVERY_EXPECTED,
  M5_RUNPOD_ENV_AUTHORIZATION_COMMIT,
  M5_RUNPOD_ENV_AUTHORIZATION_PATH,
  M5_RUNPOD_ENV_AUTHORIZATION_TREE,
  M5_RUNPOD_ENV_RECOVERY_COMMIT,
  M5_RUNPOD_ENV_RECOVERY_EXPECTED,
  M5_RUNTIME_AUTHORIZATION_COMMIT,
  M5_RUNTIME_AUTHORIZATION_PATH,
  M5_RUNTIME_AUTHORIZATION_TREE,
  M5_RUNTIME_RECOVERY_COMMIT,
  M5_RUNTIME_RECOVERY_EXPECTED,
  M5_SOURCE_CI_RECOVERY_EXPECTED,
  M5_SOURCE_RECOVERY_EXPECTED,
  M5_R5_EXPECTED,
  M5_A4_COMMIT,
  M5_A5_AUTHORIZATION_PATH,
  M5_A6_AUTHORIZATION_PATH,
  M5_R6_EXPECTED,
  M5_RUN_AUTHORIZATION_PATH,
  classifyM5Stage,
  matchesExpectedRows,
  matchesM5ProtocolLineage,
  matchesM5RuntimeRecoveryCommit,
  matchesM5RunpodEnvironmentRecoveryCommit,
  matchesM5AuthorizedChain,
} from "./m5-stage-policy.mjs";

const rows = [...M5_PROTOCOL_EXPECTED.entries()];
const recoveryRows = [...M5_PROTOCOL_RECOVERY_EXPECTED.entries()];
assert.equal(matchesExpectedRows(rows, M5_PROTOCOL_EXPECTED), true);
assert.equal(matchesExpectedRows([...rows.slice(1), rows[1]], M5_PROTOCOL_EXPECTED), false);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: [M5_ORIGINAL_PROTOCOL_COMMIT], recoveryRows,
  originalTree: M5_ORIGINAL_PROTOCOL_TREE,
  originalParents: [M5_BASE_SOURCE_COMMIT], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), true);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: ["0".repeat(40)], recoveryRows,
  originalTree: M5_ORIGINAL_PROTOCOL_TREE,
  originalParents: [M5_BASE_SOURCE_COMMIT], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), false);
const sourceRows = [...M5_SOURCE_RECOVERY_EXPECTED.entries()];
const ciRecoveryRows = [...M5_SOURCE_CI_RECOVERY_EXPECTED.entries()];
const runtimeRows = [...M5_RUNTIME_RECOVERY_EXPECTED.entries()];
const environmentRows = [...M5_RUNPOD_ENV_RECOVERY_EXPECTED.entries()];
const numericRows = [...M5_NUMERIC_AUDIT_RECOVERY_EXPECTED.entries()];
assert.equal(M5_R5_EXPECTED.size, 17);
assert.equal(M5_R5_EXPECTED.has("benchmark/evidence/m5/initial-parity-diagnostic.json"), true);
assert.equal(matchesExpectedRows([...M5_R5_EXPECTED], M5_R5_EXPECTED), true);
assert.equal(matchesExpectedRows([...M5_R5_EXPECTED].slice(1), M5_R5_EXPECTED), false);
assert.equal(matchesExpectedRows([...M5_R5_EXPECTED, ["extra", "M"]], M5_R5_EXPECTED), false);
assert.equal(matchesExpectedRows([...M5_R5_EXPECTED].map(([path, status], index) => [path, index === 0 ? "A" : status]), M5_R5_EXPECTED), false);
assert.equal(M5_A4_COMMIT.length, 40);
assert.equal(M5_A5_AUTHORIZATION_PATH, "benchmark/evidence/m5/parity-recovery-authorization.json");
assert.equal(M5_A6_AUTHORIZATION_PATH, "benchmark/evidence/m5/cublas-recovery-authorization.json");
assert.equal(M5_R6_EXPECTED.size, 17);
assert.equal(M5_R6_EXPECTED.has("package.json"), false);
assert.equal(M5_R6_EXPECTED.get("scripts/m5-run-authorization.mjs"), "M");
assert.equal(M5_R6_EXPECTED.get("scripts/m5-training-contract.mjs"), "M");
assert.equal(M5_R6_EXPECTED.get("scripts/check-m5-cublas-authorized-chain.mjs"), "A");
assert.equal(matchesExpectedRows([...M5_R6_EXPECTED], M5_R6_EXPECTED), true);
assert.equal(matchesExpectedRows([...M5_R6_EXPECTED].slice(1), M5_R6_EXPECTED), false);
assert.equal(matchesExpectedRows([...M5_R6_EXPECTED, ["extra", "M"]], M5_R6_EXPECTED), false);
assert.equal(environmentRows.length, 12);
assert.equal(numericRows.length, 12);
for (const [pathname] of environmentRows) assert.equal(M5_SOURCE_RECOVERY_EXPECTED.has(pathname), true);
const runtimeCommit = M5_RUNTIME_RECOVERY_COMMIT;
const runtimeFixture = {
  runtimeParents: [M5_P4_COMMIT], runtimeRows,
  p4Tree: M5_P4_TREE, p4Parents: [M5_CI_RECOVERY_COMMIT], p4Rows: [[M5_RUN_AUTHORIZATION_PATH, "A"]],
  sourceCommit: M5_CI_RECOVERY_COMMIT, sourceParents: [M5_FAILED_SOURCE_COMMIT], sourceRows: ciRecoveryRows,
  failedSourceTree: M5_FAILED_SOURCE_TREE, failedSourceParents: [M5_P2_COMMIT], failedSourceRows: sourceRows,
};
assert.equal(matchesM5RuntimeRecoveryCommit(runtimeFixture), true);
assert.equal(matchesM5RuntimeRecoveryCommit({ ...runtimeFixture, runtimeRows: [...runtimeRows, ["torch.py", "A"]] }), false);
assert.equal(matchesM5RuntimeRecoveryCommit({ ...runtimeFixture, runtimeParents: [M5_CI_RECOVERY_COMMIT] }), false);
assert.equal(matchesM5RuntimeRecoveryCommit({ ...runtimeFixture, p4Tree: "0".repeat(40) }), false);
const environmentCommit = M5_RUNPOD_ENV_RECOVERY_COMMIT;
const environmentFixture = {
  environmentParents: [M5_RUNTIME_AUTHORIZATION_COMMIT], environmentRows,
  runtimeAuthorizationTree: M5_RUNTIME_AUTHORIZATION_TREE,
  runtimeAuthorizationParents: [runtimeCommit], runtimeAuthorizationRows: [[M5_RUNTIME_AUTHORIZATION_PATH, "A"]],
  runtimeCommit, ...runtimeFixture,
};
assert.equal(matchesM5RunpodEnvironmentRecoveryCommit(environmentFixture), true);
assert.equal(matchesM5RunpodEnvironmentRecoveryCommit({ ...environmentFixture, environmentRows: [...environmentRows, ["extra", "M"]] }), false);
assert.equal(matchesM5RunpodEnvironmentRecoveryCommit({ ...environmentFixture, runtimeAuthorizationTree: "0".repeat(40) }), false);
const numericCommit = "c".repeat(40);
assert.equal(matchesM5AuthorizedChain({
  authorizationCommit: "d".repeat(40), authorizationParents: [numericCommit],
  authorizationRows: [[M5_NUMERIC_AUDIT_AUTHORIZATION_PATH, "A"]], numericCommit,
  numericParents: [M5_RUNPOD_ENV_AUTHORIZATION_COMMIT], numericRows,
  environmentAuthorizationTree: M5_RUNPOD_ENV_AUTHORIZATION_TREE,
  environmentAuthorizationParents: [environmentCommit],
  environmentAuthorizationRows: [[M5_RUNPOD_ENV_AUTHORIZATION_PATH, "A"]],
  environmentCommit, ...environmentFixture,
}), true);
assert.equal(matchesM5AuthorizedChain({
  authorizationCommit: "d".repeat(40), authorizationParents: [numericCommit],
  authorizationRows: [[M5_NUMERIC_AUDIT_AUTHORIZATION_PATH, "A"], ["extra", "A"]], numericCommit,
  numericParents: [M5_RUNPOD_ENV_AUTHORIZATION_COMMIT], numericRows,
  environmentAuthorizationTree: M5_RUNPOD_ENV_AUTHORIZATION_TREE,
  environmentAuthorizationParents: [environmentCommit],
  environmentAuthorizationRows: [[M5_RUNPOD_ENV_AUTHORIZATION_PATH, "A"]],
  environmentCommit, ...environmentFixture,
}), false);
assert.deepEqual(parseCanonicalM5Authorization(Buffer.from('{"a":1}\n')), { a: 1 });
assert.throws(() => parseCanonicalM5Authorization(Buffer.from('{"a":1,"a":2}\n')));
assert.throws(() => parseCanonicalM5Authorization(Buffer.from([0xff])));
assert.throws(() => requireM5AuthorizationSchema({ acceptanceEligible: true }));
assert.equal(matchesM5RuntimeRecoveryCommit({ ...runtimeFixture, sourceParents: ["0".repeat(40)] }), false);
assert.equal(matchesM5RuntimeRecoveryCommit({ ...runtimeFixture, failedSourceTree: "0".repeat(40) }), false);
assert.equal(matchesM5RuntimeRecoveryCommit({ ...runtimeFixture, failedSourceRows: sourceRows.slice(1) }), false);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: [M5_ORIGINAL_PROTOCOL_COMMIT], recoveryRows: recoveryRows.slice(1),
  originalTree: M5_ORIGINAL_PROTOCOL_TREE,
  originalParents: [M5_BASE_SOURCE_COMMIT], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), false);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: [M5_ORIGINAL_PROTOCOL_COMMIT], recoveryRows,
  originalTree: "0".repeat(40),
  originalParents: [M5_BASE_SOURCE_COMMIT], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), false);
assert.equal(matchesM5ProtocolLineage({
  recoveryParents: [M5_ORIGINAL_PROTOCOL_COMMIT], recoveryRows,
  originalTree: M5_ORIGINAL_PROTOCOL_TREE,
  originalParents: ["0".repeat(40)], originalRows: rows, baseTree: M5_BASE_SOURCE_TREE,
}), false);
assert.equal(classifyM5Stage({ protocolExists: false, lockExists: false, failureExists: false, largeSourceLockExists: false, finalExists: false }), null);
assert.equal(classifyM5Stage({ protocolExists: true, lockExists: false, failureExists: false, largeSourceLockExists: false, finalExists: false }), "m5-protocol");
assert.equal(classifyM5Stage({ protocolExists: true, sourceRecoveryExists: true, lockExists: false, failureExists: false, largeSourceLockExists: false, finalExists: false }), "m5-source-recovery");
assert.equal(classifyM5Stage({ protocolExists: true, authorizationExists: true, sourceRecoveryExists: true, lockExists: false, failureExists: false, largeSourceLockExists: false, finalExists: false }), "m5-source-recovery");
assert.equal(classifyM5Stage({ protocolExists: true, authorizationExists: true, sourceRecoveryExists: false, lockExists: false, failureExists: false, largeSourceLockExists: false, finalExists: false }), "m5-authorized");
assert.equal(classifyM5Stage({ protocolExists: true, authorizationExists: true, lockExists: true, failureExists: false, largeSourceLockExists: false, finalExists: false }), "m5-pinned");
assert.equal(classifyM5Stage({ protocolExists: true, authorizationExists: true, lockExists: true, failureExists: false, largeSourceLockExists: true, finalExists: false }), "m5-eval-locked");
assert.equal(classifyM5Stage({ protocolExists: true, authorizationExists: true, lockExists: false, failureExists: true, largeSourceLockExists: false, finalExists: false }), "m5-failed");
assert.equal(classifyM5Stage({ protocolExists: true, authorizationExists: true, lockExists: true, failureExists: false, largeSourceLockExists: true, finalExists: true }), "m5-final");
assert.throws(() => classifyM5Stage({ protocolExists: true, authorizationExists: false, lockExists: true, failureExists: false, largeSourceLockExists: false, finalExists: false }));
assert.throws(() => classifyM5Stage({ protocolExists: true, lockExists: true, failureExists: true, largeSourceLockExists: false, finalExists: false }));
assert.throws(() => classifyM5Stage({ protocolExists: true, lockExists: false, failureExists: false, largeSourceLockExists: true, finalExists: false }));
assert.throws(() => classifyM5Stage({ protocolExists: true, lockExists: true, failureExists: false, largeSourceLockExists: false, finalExists: true }));
const dispatcher = readFileSync("scripts/run-static-verification.mjs", "utf8");
assert.ok(dispatcher.includes("M5_A4_COMMIT"));
assert.ok(dispatcher.includes("M5_NUMERIC_AUDIT_RECOVERY_EXPECTED"));
assert.ok(dispatcher.includes("M5_RUNPOD_ENV_AUTHORIZATION_COMMIT"));
assert.ok(dispatcher.includes("M5_A5_AUTHORIZATION_PATH"));
assert.ok(dispatcher.includes("M5_A6_AUTHORIZATION_PATH"));
assert.ok(dispatcher.includes("M5_R5_EXPECTED"));
assert.ok(dispatcher.includes("M5_R6_EXPECTED"));
const packageJson = JSON.parse(readFileSync("package.json", "utf8"));
assert.equal(packageJson.scripts["check:m5-pipeline"], "node scripts/run-benchmark-python.mjs -m unittest benchmark.m5.test_contracts");
for (const [stage, script] of [
  ["m5-source-recovery", "verify:m5-source-recovery"], ["m5-authorized", "verify:m5-authorized"],
  ["m5-failed", "verify:m5-failed"], ["m5-eval-locked", "verify:m5-eval-locked"], ["m5-final", "verify:m5-final"],
]) {
  assert.ok(dispatcher.includes(`["${stage}", "${script}"]`));
  assert.equal(typeof packageJson.scripts[script], "string");
}
for (const name of ["benchmark:m5:preflight", "benchmark:m5:train", "benchmark:m5:regress",
  "benchmark:m5:lock-large-synthetic", "benchmark:m5:evaluate-large-synthetic", "benchmark:m5:finalize"]) {
  assert.equal(packageJson.scripts[name], undefined, `${name} must not start through preloadable npm/Node`);
}
const launcherSource = readFileSync("scripts/m5-python-launch.mjs", "utf8");
assert.ok(launcherSource.includes('spawnSync(PYTHON_PATH, ["-I", mode.script, ...boundArguments]'));
assert.ok(launcherSource.includes("bindCurrentHead(arguments_, mode.headFlag, head)"));
assert.ok(launcherSource.includes('const PYTHON_PATH = "/opt/conda/bin/python"'));
assert.ok(launcherSource.includes("process.env.PATH !== M5_TRUSTED_PATH"));
assert.ok(launcherSource.includes('process.env.NODE_OPTIONS !== undefined'));
assert.ok(!launcherSource.includes("ls-remote"));
assert.ok(!readFileSync("scripts/m5-run-authorization.mjs", "utf8").includes("ls-remote"));
assert.ok(!readFileSync("scripts/check-m5-run-authorization-stage.mjs", "utf8").includes("ls-remote"));
const safeGitSource = readFileSync("scripts/m5-safe-git.mjs", "utf8");
for (const token of ["GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM", "GIT_NO_REPLACE_OBJECTS", "core.fsmonitor=false"]) {
  assert.ok(safeGitSource.includes(token));
}
const runpodShell = readFileSync("scripts/m5-runpod-launch.sh", "utf8");
assert.ok(runpodShell.includes("unset BASH_ENV ENV NODE_OPTIONS NODE_PATH"));
assert.ok(runpodShell.includes("PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"));
assert.ok(runpodShell.includes("export CUBLAS_WORKSPACE_CONFIG=:4096:8"));
assert.ok(runpodShell.includes("/usr/bin/uname -s"));
assert.ok(runpodShell.includes("/usr/bin/git -c core.fsmonitor=false"));
assert.ok(runpodShell.includes("safe_git rev-parse --show-toplevel"));
assert.ok(runpodShell.includes("/workspace/.seroslop/runtime/node-v24.18.1-linux-x64/bin/node"));
execFileSync("bash", ["-n", "scripts/m5-runpod-launch.sh"]);
const preexecBootstrapPath = "scripts/m5-preexec-bootstrap.py";
const preexecBootstrap = readFileSync(preexecBootstrapPath);
const preexecBootstrapSha256 = createHash("sha256").update(preexecBootstrap).digest("hex");
assert.equal(preexecBootstrapSha256, "b2a187f1d7d81a4644c4667fae35f5826ff2ffb311d3cc499df59cfdb4b8ad3d");
const readmeBootstrapDigests = [...readFileSync("benchmark/m5/README.md", "utf8").matchAll(/scripts\/m5-preexec-bootstrap\.py ([0-9a-f]{64})/gu)].map((match) => match[1]);
assert.deepEqual(readmeBootstrapDigests, [preexecBootstrapSha256, preexecBootstrapSha256]);
const m5Readme = readFileSync("benchmark/m5/README.md", "utf8");
assert.match(m5Readme, /From the\s+clean public-green R6 commit, run:/u);
assert.ok(m5Readme.includes("git add benchmark/evidence/m5/cublas-recovery-authorization.json"));
assert.ok(m5Readme.includes('git commit -m "Evidence: authorize exact M5 deterministic CUDA recovery"'));
assert.ok(!m5Readme.includes("git add benchmark/evidence/m5/parity-recovery-authorization.json"));
const preexecSource = preexecBootstrap.toString("utf8");
assert.ok(preexecSource.indexOf("verify_exact_worktree(allowed_untracked=allowed_untracked)") < preexecSource.indexOf('if mode == "authorize"'));
assert.ok(preexecSource.includes('LOCAL_NODE = Path("/Users/baney/.local/node/bin/node")'));
assert.ok(preexecSource.includes("LOCAL_NODE_SHA256 = \"3200fbd9f7fd4410426dd541e10d1ab829d3472f270d743c7fabd1696c03fe32\""));
assert.ok(preexecSource.includes('environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"'));
assert.ok(!JSON.parse(readFileSync("package.json", "utf8")).scripts["benchmark:m5:authorize"]);
const directPython = spawnSync("python3", ["-I", "benchmark/m5/train_gpu.py", "--help"], {
  encoding: "utf8",
  env: Object.fromEntries(Object.entries(process.env).filter(([key]) => !key.startsWith("PYTHON"))),
});
assert.notEqual(directPython.status, 0);
assert.match(directPython.stderr, /pinned (?:Node parent|RunPod launcher)/u);
for (const pathname of ["benchmark/m5/evaluate_locked.py", "benchmark/m5/evaluate_large_synthetic.py", "benchmark/m5/large_synthetic.py"]) {
  assert.ok(!readFileSync(pathname, "utf8").includes("--untracked-files=no"));
}
const launcherDigest = createHash("sha256").update(readFileSync("scripts/m5-python-launch.mjs")).digest("hex");
for (const pathname of ["benchmark/m5/train_gpu.py", "benchmark/m5/evaluate_locked.py", "benchmark/m5/evaluate_large_synthetic.py", "benchmark/m5/large_synthetic.py", "benchmark/m5/finalize.py"]) {
  const source = readFileSync(pathname, "utf8");
  assert.ok(source.includes("expected.is_symlink()"), `${pathname} does not reject a symlinked parent executable`);
  assert.ok(source.includes("digest.hexdigest()"), `${pathname} does not hash its parent Node executable`);
  assert.ok(source.includes("f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a"));
  assert.ok(source.includes(launcherDigest), `${pathname} does not bind the exact tracked M5 launcher bytes`);
}
const trainExecute = readFileSync("benchmark/m5/train_gpu.py", "utf8").split("def execute(args: argparse.Namespace) -> int:")[1];
assert.ok(trainExecute.indexOf("require_cuda_determinism_environment(recipe)") < trainExecute.indexOf("import torch"));
assert.ok(trainExecute.indexOf("resolve_authorized_run()") < trainExecute.indexOf("import torch"));

const launcherUrl = pathToFileURL(resolve("scripts/m5-python-launch.mjs")).href;
function realTempDirectory(prefix) {
  return realpathSync(mkdtempSync(join(tmpdir(), prefix)));
}
function assertShadowCannotExecute(moduleName, { ignored = false, packageDirectory = false } = {}) {
  const root = realTempDirectory("seroslop-m5-launch-");
  const marker = join(root, "shadow-executed");
  try {
    execFileSync("git", ["init", "-q"], { cwd: root });
    writeFileSync(join(root, "driver.py"), `import ${moduleName}\n`, "utf8");
    if (ignored) writeFileSync(join(root, ".gitignore"), `${moduleName}/\n`, "utf8");
    execFileSync("git", ["add", "."], { cwd: root });
    execFileSync("git", ["-c", "user.name=M5 Test", "-c", "user.email=m5@example.invalid", "commit", "-qm", "fixture"], { cwd: root });
    const payload = `from pathlib import Path\nPath(${JSON.stringify(marker)}).write_text("executed")\n`;
    if (packageDirectory) {
      mkdirSync(join(root, moduleName));
      writeFileSync(join(root, moduleName, "__init__.py"), payload, "utf8");
    } else {
      writeFileSync(join(root, `${moduleName}.py`), payload, "utf8");
    }
    const program = [
      `import { requireCleanM5PythonLaunchSurface } from ${JSON.stringify(launcherUrl)};`,
      `import { spawnSync } from "node:child_process";`,
      `const root=${JSON.stringify(root)};`,
      `requireCleanM5PythonLaunchSurface(root);`,
      `const child=spawnSync("python3",["driver.py"],{cwd:root,stdio:"ignore"});`,
      `process.exit(child.status ?? 1);`,
    ].join("\n");
    const result = spawnSync(process.execPath, ["--input-type=module", "-e", program], { encoding: "utf8" });
    assert.notEqual(result.status, 0, `${moduleName} shadow launch unexpectedly succeeded`);
    assert.equal(existsSync(marker), false, `${moduleName} shadow executed before the launch gate`);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}
assertShadowCannotExecute("torch");
assertShadowCannotExecute("numpy");
assertShadowCannotExecute("PIL", { ignored: true, packageDirectory: true });

const bytecodeRoot = realTempDirectory("seroslop-m5-bytecode-");
const bytecodeMarker = join(bytecodeRoot, "unchecked-pyc-executed");
try {
  const moduleRoot = join(bytecodeRoot, "benchmark", "m5");
  mkdirSync(moduleRoot, { recursive: true });
  execFileSync("git", ["init", "-q"], { cwd: bytecodeRoot });
  writeFileSync(join(bytecodeRoot, ".gitignore"), "benchmark/**/__pycache__/\n", "utf8");
  writeFileSync(join(moduleRoot, "driver.py"), "import victim\n", "utf8");
  writeFileSync(join(moduleRoot, "victim.py"), "# tracked benign source\n", "utf8");
  execFileSync("git", ["add", "."], { cwd: bytecodeRoot });
  execFileSync("git", ["-c", "user.name=M5 Test", "-c", "user.email=m5@example.invalid", "commit", "-qm", "fixture"], { cwd: bytecodeRoot });
  writeFileSync(join(moduleRoot, "victim.py"), `from pathlib import Path\nPath(${JSON.stringify(bytecodeMarker)}).write_text("executed")\n`, "utf8");
  execFileSync("python3", [
    "-c",
    "import py_compile,sys; py_compile.compile(sys.argv[1], doraise=True, invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)",
    join(moduleRoot, "victim.py"),
  ]);
  writeFileSync(join(moduleRoot, "victim.py"), "# tracked benign source\n", "utf8");
  const direct = spawnSync("python3", [join(moduleRoot, "driver.py")], { cwd: bytecodeRoot, stdio: "ignore" });
  assert.equal(direct.status, 0);
  assert.equal(existsSync(bytecodeMarker), true, "unchecked-hash bytecode fixture did not execute without the gate");
  unlinkSync(bytecodeMarker);
  const program = [
    `import { requireCleanM5PythonLaunchSurface } from ${JSON.stringify(launcherUrl)};`,
    `import { spawnSync } from "node:child_process";`,
    `const root=${JSON.stringify(bytecodeRoot)};`,
    `requireCleanM5PythonLaunchSurface(root);`,
    `const child=spawnSync("python3",[${JSON.stringify(join(moduleRoot, "driver.py"))}],{cwd:root,stdio:"ignore"});`,
    `process.exit(child.status ?? 1);`,
  ].join("\n");
  const gated = spawnSync(process.execPath, ["--input-type=module", "-e", program], { encoding: "utf8" });
  assert.notEqual(gated.status, 0, "ignored unchecked-hash bytecode passed the launch gate");
  assert.equal(existsSync(bytecodeMarker), false, "ignored unchecked-hash bytecode executed before the launch gate");
} finally {
  rmSync(bytecodeRoot, { recursive: true, force: true });
}

const preloadRoot = realTempDirectory("seroslop-m5-preload-");
try {
  const marker = join(preloadRoot, "node-options-executed");
  const preload = join(preloadRoot, "preload.mjs");
  writeFileSync(preload, `import { writeFileSync } from "node:fs"; writeFileSync(${JSON.stringify(marker)}, "executed");\n`, "utf8");
  const result = spawnSync("/usr/bin/env", [
    "-u", "BASH_ENV", "-u", "ENV", "-u", "NODE_OPTIONS", "-u", "NODE_PATH",
    "/bin/bash", "--noprofile", "--norc", "-c",
    "unset BASH_ENV ENV NODE_OPTIONS NODE_PATH; exec /usr/bin/env -u NODE_OPTIONS -u NODE_PATH \"$1\" -e 'process.exit(0)'",
    "m5-boundary", process.execPath,
  ], {
    encoding: "utf8",
    env: { ...process.env, NODE_OPTIONS: `--import=${pathToFileURL(preload).href}`, BASH_ENV: preload },
  });
  assert.equal(result.status, 0);
  assert.equal(existsSync(marker), false, "canonical outer boundary allowed NODE_OPTIONS/BASH_ENV execution");
} finally {
  rmSync(preloadRoot, { recursive: true, force: true });
}

const localAuthorizationRoot = realTempDirectory("seroslop-m5-preexec-");
const localAuthorizationAttackRoot = realTempDirectory("seroslop-m5-preexec-attack-");
try {
  const fakeBin = join(localAuthorizationAttackRoot, "bin");
  const fakeNodeMarker = join(localAuthorizationAttackRoot, "fake-node-executed");
  mkdirSync(fakeBin);
  const fakeNode = join(fakeBin, "node");
  writeFileSync(fakeNode, `#!/bin/sh\nprintf executed > ${JSON.stringify(fakeNodeMarker)}\nexit 99\n`, "utf8");
  chmodSync(fakeNode, 0o755);
  mkdirSync(join(localAuthorizationRoot, "scripts"));
  writeFileSync(join(localAuthorizationRoot, preexecBootstrapPath), preexecBootstrap);
  const protectedFiles = new Map([
    ["scripts/m5-run-authorization.mjs", "process.exit(0);\n"],
    ["scripts/m5-runpod-launch.sh", "#!/bin/bash\nexit 0\n"],
    ["scripts/m5-python-launch.mjs", "process.exit(0);\n"],
  ]);
  for (const [pathname, source] of protectedFiles) writeFileSync(join(localAuthorizationRoot, pathname), source, "utf8");
  execFileSync("/usr/bin/git", ["init", "-q"], { cwd: localAuthorizationRoot });
  execFileSync("/usr/bin/git", ["add", "."], { cwd: localAuthorizationRoot });
  execFileSync("/usr/bin/git", ["-c", "user.name=M5 Test", "-c", "user.email=m5@example.invalid", "commit", "-qm", "fixture"], { cwd: localAuthorizationRoot });
  const loader = [
    "import hashlib,pathlib,sys",
    "p=pathlib.Path(sys.argv[1]); raw=p.read_bytes()",
    'if hashlib.sha256(raw).hexdigest()!=sys.argv[2]: raise SystemExit("M5 pre-exec bootstrap bytes changed")',
    "sys.argv=[str(p),*sys.argv[3:]]",
    'exec(compile(raw,str(p),"exec"),{"__name__":"__main__","__file__":str(p)})',
  ].join("\n");
  const runPreexec = (...arguments_) => spawnSync("/usr/bin/python3", [
    "-I", "-c", loader, preexecBootstrapPath, preexecBootstrapSha256, ...arguments_,
  ], { cwd: localAuthorizationRoot, encoding: "utf8", env: { ...process.env, PATH: `${fakeBin}:${process.env.PATH}` } });
  const clean = runPreexec("verify-only");
  assert.equal(clean.status, 0, clean.stderr);
  const authorizationProbe = runPreexec("authorize");
  if (process.platform === "darwin" && process.arch === "arm64") assert.equal(authorizationProbe.status, 0, authorizationProbe.stderr);
  else assert.notEqual(authorizationProbe.status, 0);
  assert.equal(existsSync(fakeNodeMarker), false, "local authorization executed PATH-injected Node");
  for (const [pathname, source] of protectedFiles) {
    const marker = join(localAuthorizationAttackRoot, pathname.split("/").at(-1) + "-executed");
    const mutation = pathname.endsWith(".sh")
      ? `#!/bin/bash\nprintf executed > ${JSON.stringify(marker)}\nexit 99\n`
      : `import { writeFileSync } from "node:fs"; writeFileSync(${JSON.stringify(marker)}, "executed");\n${source}`;
    writeFileSync(join(localAuthorizationRoot, pathname), mutation, "utf8");
    const rejected = runPreexec("verify-only");
    assert.notEqual(rejected.status, 0, `${pathname} mutation passed the external pre-exec gate`);
    assert.equal(existsSync(marker), false, `${pathname} mutation executed before the pre-exec gate`);
    writeFileSync(join(localAuthorizationRoot, pathname), source, "utf8");
  }
  const bootstrapMarker = join(localAuthorizationAttackRoot, "bootstrap-executed");
  const changedBootstrap = Buffer.concat([
    Buffer.from(`from pathlib import Path\nPath(${JSON.stringify(bootstrapMarker)}).write_text("executed")\n`),
    preexecBootstrap,
  ]);
  writeFileSync(join(localAuthorizationRoot, preexecBootstrapPath), changedBootstrap);
  const bootstrapRejected = runPreexec("verify-only");
  assert.notEqual(bootstrapRejected.status, 0, "changed pre-exec bootstrap passed its external byte lock");
  assert.equal(existsSync(bootstrapMarker), false, "changed pre-exec bootstrap executed before its external byte lock");
} finally {
  rmSync(localAuthorizationRoot, { recursive: true, force: true });
  rmSync(localAuthorizationAttackRoot, { recursive: true, force: true });
}

const pathAttackRoot = realTempDirectory("seroslop-m5-path-");
try {
  const fakeBin = join(pathAttackRoot, "bin");
  const fakeGitMarker = join(pathAttackRoot, "fake-git-executed");
  const fakeUnameMarker = join(pathAttackRoot, "fake-uname-executed");
  mkdirSync(fakeBin);
  for (const [name, marker] of [["git", fakeGitMarker], ["uname", fakeUnameMarker]]) {
    const executable = join(fakeBin, name);
    writeFileSync(executable, `#!/bin/sh\nprintf executed > ${JSON.stringify(marker)}\nexit 99\n`, "utf8");
    chmodSync(executable, 0o755);
  }
  const shellResult = spawnSync("/usr/bin/env", [
    "-u", "BASH_ENV", "-u", "ENV", "-u", "NODE_OPTIONS", "-u", "NODE_PATH",
    "/bin/bash", "--noprofile", "--norc", "scripts/m5-runpod-launch.sh", "preflight", "--", "--help",
  ], {
    cwd: process.cwd(),
    encoding: "utf8",
    env: { ...process.env, PATH: `${fakeBin}:${process.env.PATH}`, GIT_DIR: join(pathAttackRoot, "bogus-git-dir") },
  });
  assert.notEqual(shellResult.status, 0, shellResult.stderr);
  assert.equal(existsSync(fakeUnameMarker), false, "canonical outer boundary executed PATH-injected uname");
  assert.equal(existsSync(fakeGitMarker), false, "canonical outer boundary executed PATH-injected git");

  const repository = join(pathAttackRoot, "repository");
  mkdirSync(repository);
  execFileSync("/usr/bin/git", ["init", "-q"], { cwd: repository });
  writeFileSync(join(repository, "flag.txt"), "tracked\n", "utf8");
  execFileSync("/usr/bin/git", ["add", "flag.txt"], { cwd: repository });
  execFileSync("/usr/bin/git", ["-c", "user.name=M5 Test", "-c", "user.email=m5@example.invalid", "commit", "-qm", "baseline"], { cwd: repository });
  const fsmonitorMarker = join(pathAttackRoot, "fsmonitor-executed");
  const fsmonitor = join(pathAttackRoot, "fsmonitor.sh");
  const filterMarker = join(pathAttackRoot, "clean-filter-executed");
  const cleanFilter = join(pathAttackRoot, "clean-filter.sh");
  writeFileSync(fsmonitor, `#!/bin/sh\nprintf executed > ${JSON.stringify(fsmonitorMarker)}\nexit 1\n`, "utf8");
  chmodSync(fsmonitor, 0o755);
  writeFileSync(cleanFilter, `#!/bin/sh\nprintf executed > ${JSON.stringify(filterMarker)}\n/bin/cat\n`, "utf8");
  chmodSync(cleanFilter, 0o755);
  execFileSync("/usr/bin/git", ["config", "core.fsmonitor", fsmonitor], { cwd: repository });
  execFileSync("/usr/bin/git", ["config", "filter.m5attack.clean", cleanFilter], { cwd: repository });
  execFileSync("/usr/bin/git", ["config", "filter.m5attack.smudge", "/bin/cat"], { cwd: repository });
  mkdirSync(join(repository, ".git", "info"), { recursive: true });
  writeFileSync(join(repository, ".git", "info", "attributes"), "*.txt filter=m5attack\n", "utf8");
  writeFileSync(join(pathAttackRoot, ".gitconfig"), [
    "[core]",
    `\tfsmonitor = ${fsmonitor}`,
    '[url "file:///tmp/m5-evil/"]',
    "\tinsteadOf = https://github.com/",
    "",
  ].join("\n"), "utf8");
  const cleanProgram = [
    `import { requireCleanM5PythonLaunchSurface } from ${JSON.stringify(launcherUrl)};`,
    `requireCleanM5PythonLaunchSurface(${JSON.stringify(repository)});`,
  ].join("\n");
  const nodeResult = spawnSync(process.execPath, ["--input-type=module", "-e", cleanProgram], {
    encoding: "utf8",
    env: { ...process.env, HOME: pathAttackRoot, PATH: `${fakeBin}:${process.env.PATH}`, GIT_DIR: join(pathAttackRoot, "bogus-git-dir") },
  });
  assert.equal(nodeResult.status, 0, nodeResult.stderr);
  assert.equal(existsSync(fakeGitMarker), false, "pinned Node launcher executed PATH-injected git");
  assert.equal(existsSync(fsmonitorMarker), false, "pinned Git wrapper executed configured core.fsmonitor");
  assert.equal(existsSync(filterMarker), false, "exact-worktree verification executed a configured clean filter");

  execFileSync("/usr/bin/git", ["update-index", "--assume-unchanged", "flag.txt"], { cwd: repository });
  assert.throws(() => requireCleanM5PythonLaunchSurface(repository), /non-normal index flags/u);
  execFileSync("/usr/bin/git", ["update-index", "--no-assume-unchanged", "flag.txt"], { cwd: repository });
  execFileSync("/usr/bin/git", ["update-index", "--skip-worktree", "flag.txt"], { cwd: repository });
  assert.throws(() => requireCleanM5PythonLaunchSurface(repository), /non-normal index flags/u);
  execFileSync("/usr/bin/git", ["update-index", "--no-skip-worktree", "flag.txt"], { cwd: repository });

  writeFileSync(join(repository, "object.txt"), "original\n", "utf8");
  execFileSync("/usr/bin/git", ["add", "object.txt"], { cwd: repository });
  execFileSync("/usr/bin/git", ["-c", "user.name=M5 Test", "-c", "user.email=m5@example.invalid", "commit", "-qm", "original"], { cwd: repository });
  const original = execFileSync("/usr/bin/git", ["rev-parse", "HEAD"], { cwd: repository, encoding: "utf8" }).trim();
  writeFileSync(join(repository, "object.txt"), "replacement\n", "utf8");
  execFileSync("/usr/bin/git", ["add", "object.txt"], { cwd: repository });
  execFileSync("/usr/bin/git", ["-c", "user.name=M5 Test", "-c", "user.email=m5@example.invalid", "commit", "-qm", "replacement"], { cwd: repository });
  const replacement = execFileSync("/usr/bin/git", ["rev-parse", "HEAD"], { cwd: repository, encoding: "utf8" }).trim();
  execFileSync("/usr/bin/git", ["replace", original, replacement], { cwd: repository });
  assert.equal(m5Git(["show", `${original}:object.txt`], { cwd: repository }), "original");
  execFileSync("/usr/bin/git", ["config", "url.file:///tmp/m5-evil/.insteadOf", "https://github.com/"], { cwd: repository });
  assert.ok(!safeGitSource.includes("ls-remote"), "safe Git wrapper must not provide a configurable public-network proof");
} finally {
  rmSync(pathAttackRoot, { recursive: true, force: true });
}

console.log(JSON.stringify({ cases: 61, policy: "pass" }));
