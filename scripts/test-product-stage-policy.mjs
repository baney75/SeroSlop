import assert from "node:assert/strict";
import { appendFileSync, mkdtempSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { m5Git } from "./m5-safe-git.mjs";
import { M6_SUBMISSION_RELEASE_PARENT } from "./m6-stage-policy.mjs";
import { isProductHead, PRODUCT_BASE_COMMIT, verifyProductLineage } from "./product-stage-policy.mjs";

assert.equal(isProductHead(PRODUCT_BASE_COMMIT), false);
assert.equal(isProductHead(M6_SUBMISSION_RELEASE_PARENT), false);

const scratch = mkdtempSync(path.join(os.tmpdir(), "seroslop-product-stage-"));
const worktree = path.join(scratch, "tree");
const git = (args) => m5Git(["-c", "user.name=SeroSlop test", "-c", "user.email=test@invalid", ...args], { cwd: worktree });
m5Git(["worktree", "add", "--detach", "--quiet", worktree, PRODUCT_BASE_COMMIT]);
try {
  assert.throws(() => verifyProductLineage({ cwd: worktree }), /does not descend/u);

  appendFileSync(path.join(worktree, "README.md"), "\nProduct change.\n");
  git(["commit", "--quiet", "--all", "--message", "product change"]);
  const product = git(["rev-parse", "HEAD"]);
  assert.equal(isProductHead(product, { cwd: worktree }), true);
  assert.equal(verifyProductLineage({ cwd: worktree }).head, product);

  const contractsPath = path.join(worktree, "src/shared/contracts.ts");
  writeFileSync(contractsPath, readFileSync(contractsPath, "utf8").replace("AI_THRESHOLD = 0.65", "AI_THRESHOLD = 0.5"));
  git(["commit", "--quiet", "--all", "--message", "move threshold"]);
  assert.throws(() => verifyProductLineage({ cwd: worktree }), /AI_THRESHOLD/u);
  git(["reset", "--quiet", "--hard", product]);

  appendFileSync(path.join(worktree, "model-lock.json"), "\n");
  assert.throws(() => verifyProductLineage({ cwd: worktree }), /worktree changes frozen/u);
  git(["commit", "--quiet", "--all", "--message", "touch frozen"]);
  assert.throws(() => verifyProductLineage({ cwd: worktree }), /changed frozen research paths: model-lock.json/u);
} finally {
  m5Git(["worktree", "remove", "--force", worktree]);
  rmSync(scratch, { recursive: true, force: true });
}
console.log(JSON.stringify({ cases: 9, policy: "pass" }));
