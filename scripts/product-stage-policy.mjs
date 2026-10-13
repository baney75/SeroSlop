import { readFileSync } from "node:fs";
import path from "node:path";
import { m5Git } from "./m5-safe-git.mjs";

// The research protocol ends at the submission release commit. Every later commit
// is ordinary product work: it may change the extension, site, docs and tooling,
// but never the model, its runtime contract, or the evidence and checkers the
// README cites. A full SHA pins that commit's own bytes, so only drift since it
// needs checking.
export const PRODUCT_BASE_COMMIT = "3853150f31ea40e63849467f4fe712799fa5737b";
export const PRODUCT_FROZEN_PATHS = Object.freeze([
  "benchmark",
  "weights",
  "model-lock.json",
  "src/shared/model-spec.ts",
  "scripts/check-bounty-proxy-results.mjs",
  ":(glob)scripts/check-*-evidence.mjs",
  ":(glob)scripts/*-contract.mjs",
  ":(glob)scripts/m[2-6]-*.mjs",
  ":(glob)scripts/check-m[2-6]-*.mjs",
]);

export function isProductHead(head = m5Git(["rev-parse", "HEAD"]), { cwd = process.cwd() } = {}) {
  if (head === PRODUCT_BASE_COMMIT) return false;
  try {
    m5Git(["merge-base", "--is-ancestor", PRODUCT_BASE_COMMIT, head], { cwd });
    return true;
  } catch (error) {
    if (error.status === 1) return false;
    throw new Error(`Cannot compare HEAD with product base ${PRODUCT_BASE_COMMIT}; fetch full history`, { cause: error });
  }
}

export function verifyProductLineage({ cwd = process.cwd() } = {}) {
  const head = m5Git(["rev-parse", "HEAD"], { cwd });
  if (!isProductHead(head, { cwd })) throw new Error(`HEAD does not descend from product base ${PRODUCT_BASE_COMMIT}`);
  const changed = m5Git(["diff", "--no-renames", "--name-only", PRODUCT_BASE_COMMIT, head, "--", ...PRODUCT_FROZEN_PATHS], { cwd })
    .split("\n").filter(Boolean);
  if (changed.length > 0) throw new Error(`product commits changed frozen research paths: ${changed.join(", ")}`);
  const dirty = m5Git(["status", "--porcelain", "--untracked-files=no", "--", ...PRODUCT_FROZEN_PATHS], { cwd });
  if (dirty !== "") throw new Error(`worktree changes frozen research paths:\n${dirty}`);
  const lock = JSON.parse(readFileSync(path.join(cwd, "model-lock.json"), "utf8"));
  const contracts = readFileSync(path.join(cwd, "src/shared/contracts.ts"), "utf8");
  const threshold = /^export const AI_THRESHOLD = ([0-9.]+);$/mu.exec(contracts)?.[1];
  if (threshold === undefined || Number(threshold) !== lock.calibration.displayThreshold) {
    throw new Error(`AI_THRESHOLD must equal model-lock.json displayThreshold ${lock.calibration.displayThreshold}`);
  }
  return { base: PRODUCT_BASE_COMMIT, head, threshold: Number(threshold) };
}
