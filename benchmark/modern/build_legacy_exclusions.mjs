/* global console */
import { execFileSync } from "node:child_process";
import { writeFile } from "node:fs/promises";
import path from "node:path";

const sourceCommit = "403274c50a53e675d4d12ae7ce08080bf69a6d62";
const splits = ["validation", "test"];
const rows = splits.flatMap((split) => execFileSync(
  "git",
  ["show", `${sourceCommit}:benchmark/manifests/${split}.jsonl`],
  { encoding: "utf8" },
).trim().split("\n").filter(Boolean).map((line) => JSON.parse(line)));

const promptGroups = rows.filter((row) => row.label === 1).map((row) => {
  const match = /^(\d+)_/u.exec(path.basename(row.path));
  if (!match) throw new Error(`Legacy Qwen row has no prompt group: ${row.id}`);
  return match[1];
});
const openImageIds = rows.filter((row) => row.label === 0).map((row) => path.basename(row.path, ".jpg"));
const output = {
  schemaVersion: 1,
  sourceCommit,
  sourceSplits: splits,
  qwenPromptGroups: [...new Set(promptGroups)].sort(),
  openImageIds: [...new Set(openImageIds)].sort(),
  evaluationIds: rows.map((row) => row.id).sort(),
  evaluationImageSha256: rows.map((row) => row.imageSha256).sort(),
};
await writeFile(
  "benchmark/manifests/legacy-evaluation-exclusions.json",
  `${JSON.stringify(output, null, 2)}\n`,
);
console.log(JSON.stringify({
  sourceCommit,
  qwenPromptGroups: output.qwenPromptGroups.length,
  openImageIds: output.openImageIds.length,
  evaluationImageSha256: output.evaluationImageSha256.length,
}));
