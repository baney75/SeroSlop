import assert from "node:assert/strict";
import { M6_SUBMISSION_RELEASE_PARENT } from "./m6-stage-policy.mjs";
import { isProductHead, PRODUCT_BASE_COMMIT, verifyProductLineage } from "./product-stage-policy.mjs";

assert.equal(isProductHead(PRODUCT_BASE_COMMIT), false);
assert.equal(isProductHead(M6_SUBMISSION_RELEASE_PARENT), false);
assert.throws(() => verifyProductLineage(PRODUCT_BASE_COMMIT), /does not descend/u);
assert.throws(() => verifyProductLineage(M6_SUBMISSION_RELEASE_PARENT), /does not descend/u);
console.log(JSON.stringify({ cases: 4, policy: "pass" }));
