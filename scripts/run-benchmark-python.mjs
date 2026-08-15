import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

const localPython = process.platform === "win32"
  ? "benchmark/.venv/Scripts/python.exe"
  : "benchmark/.venv/bin/python";
const command = existsSync(localPython)
  ? localPython
  : process.platform === "win32" ? "python" : "python3";
const result = spawnSync(command, process.argv.slice(2), { stdio: "inherit" });
if (result.error) throw result.error;
process.exit(result.status ?? 1);
