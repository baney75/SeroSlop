import { createHash } from "node:crypto";
import { m5Git, m5GitBytes } from "./m5-safe-git.mjs";
import {
  M6_SUBMISSION_RELEASE_ARTIFACT_SHA256,
  M6_SUBMISSION_RELEASE_PARENT,
  M6_SUBMISSION_RELEASE_PARENT_TREE,
  matchesM6SubmissionReleaseHead,
} from "./m6-stage-policy.mjs";

// The research protocol ends at the submission release commit. Every later commit
// is ordinary product work: it may change the extension, site, docs and tooling,
// but never the frozen model, benchmark code or evidence the README cites.
export const PRODUCT_BASE_COMMIT = "3853150f31ea40e63849467f4fe712799fa5737b";
export const PRODUCT_FROZEN_PATHS = Object.freeze(["benchmark", "weights", "model-lock.json"]);

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

function rowsOf(commit) {
  return m5Git(["diff-tree", "--root", "--no-renames", "--name-status", "--format=", "-r", commit])
    .split("\n").filter(Boolean).map((line) => {
      const [status, path] = line.split("\t");
      return [path, status];
    });
}

export function isProductHead(head = m5Git(["rev-parse", "HEAD"])) {
  if (head === PRODUCT_BASE_COMMIT) return false;
  try {
    m5Git(["merge-base", "--is-ancestor", PRODUCT_BASE_COMMIT, head]);
    return true;
  } catch {
    return false;
  }
}

export function verifyProductLineage(head = m5Git(["rev-parse", "HEAD"])) {
  if (!isProductHead(head)) throw new Error(`HEAD does not descend from product base ${PRODUCT_BASE_COMMIT}`);
  const parents = m5Git(["rev-list", "--parents", "-n", "1", PRODUCT_BASE_COMMIT]).split(" ").slice(1);
  if (parents.length !== 1 || parents[0] !== M6_SUBMISSION_RELEASE_PARENT ||
      m5Git(["rev-parse", `${M6_SUBMISSION_RELEASE_PARENT}^{tree}`]) !== M6_SUBMISSION_RELEASE_PARENT_TREE ||
      !matchesM6SubmissionReleaseHead({ head: PRODUCT_BASE_COMMIT, parent: parents[0], rows: rowsOf(PRODUCT_BASE_COMMIT) })) {
    throw new Error("product base is not the verified submission release commit");
  }
  for (const [path, expected] of Object.entries(M6_SUBMISSION_RELEASE_ARTIFACT_SHA256)) {
    if (sha256(m5GitBytes(["show", `${PRODUCT_BASE_COMMIT}:${path}`])) !== expected) {
      throw new Error(`submission release artifact bytes changed: ${path}`);
    }
  }
  const changed = m5Git(["diff", "--no-renames", "--name-only", PRODUCT_BASE_COMMIT, head, "--", ...PRODUCT_FROZEN_PATHS])
    .split("\n").filter(Boolean);
  if (changed.length > 0) throw new Error(`product commits changed frozen research paths: ${changed.join(", ")}`);
  const dirty = m5Git(["status", "--porcelain", "--untracked-files=no", "--", ...PRODUCT_FROZEN_PATHS]);
  if (dirty !== "") throw new Error(`worktree changes frozen research paths:\n${dirty}`);
  return { base: PRODUCT_BASE_COMMIT, head, frozenPaths: PRODUCT_FROZEN_PATHS };
}
