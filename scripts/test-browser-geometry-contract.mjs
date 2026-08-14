import assert from "node:assert/strict";
import {
  validateBrowserGeometryEvidence,
} from "./browser-geometry-contract.mjs";

const screenshot = { sha256: "a".repeat(64), width: 480, height: 720 };
const target = { left: 100, top: 100, right: 300, bottom: 300 };
const inside = (elementId) => ({
  elementId,
  state: "complete",
  hidden: false,
  display: "block",
  placement: "inside-top-right",
  pointerEvents: "none",
  text: "Likely AI · 99.9/100",
  accessibleName: `Likely AI result for image “${elementId}”; score 99.9 out of 100; not proof`,
  badgeRect: { left: 160, top: 108, right: 292, bottom: 136 },
  targetRect: target,
});
const outside = {
  elementId: "prooflens-small-target",
  state: "complete",
  hidden: false,
  display: "block",
  placement: "outside-right",
  pointerEvents: "none",
  text: "Below flag threshold · 0.1/100",
  accessibleName: "Below flag threshold for image “small target fixture”; score 0.1 out of 100; not proof",
  badgeRect: { left: 306, top: 110, right: 450, bottom: 138 },
  targetRect: target,
};
const hidden = {
  ...inside("collision-hidden"),
  hidden: true,
  display: "none",
  placement: "collision-hidden",
  badgeRect: { left: 0, top: 0, right: 0, bottom: 0 },
  targetRect: { left: -300, top: -300, right: -100, bottom: -100 },
};
const section = (badges) => ({
  viewport: { width: 480, height: 720 },
  screenshot,
  badges,
});
const valid = {
  schemaVersion: 1,
  positionMargin: 6,
  modelState: section([inside("likely-ai-fixture"), { ...inside("below-threshold-fixture"),
    targetRect: { left: 100, top: 340, right: 300, bottom: 540 },
    badgeRect: { left: 160, top: 348, right: 292, bottom: 376 } }, hidden]),
  narrow: { ...section([inside("narrow-target"), hidden]), deviceScaleFactor: 1.5 },
  smallTarget: section([outside]),
};
const actual = { modelState: screenshot, narrow: screenshot, smallTarget: screenshot };

assert.deepEqual(validateBrowserGeometryEvidence(valid, actual), {
  modelState: { visibleBadges: 2, hiddenBadges: 1 },
  narrow: { visibleBadges: 1, hiddenBadges: 1 },
  smallTarget: { visibleBadges: 1, hiddenBadges: 0 },
});
const clone = () => JSON.parse(JSON.stringify(valid));
const staleScreenshot = clone();
staleScreenshot.narrow.screenshot.sha256 = "b".repeat(64);
assert.throws(() => validateBrowserGeometryEvidence(staleScreenshot, actual), /screenshot hash is stale/u);
const detached = clone();
detached.smallTarget.badges[0].badgeRect.left += 20;
detached.smallTarget.badges[0].badgeRect.right += 20;
assert.throws(() => validateBrowserGeometryEvidence(detached, actual), /lost target association/u);
const renderedHidden = clone();
renderedHidden.narrow.badges[1].display = "block";
assert.throws(() => validateBrowserGeometryEvidence(renderedHidden, actual), /hidden badge .* is still rendered/u);
const overlapping = clone();
overlapping.modelState.badges[0].placement = "outside-right";
overlapping.modelState.badges[0].targetRect = { left: 100, top: 100, right: 244, bottom: 300 };
overlapping.modelState.badges[0].badgeRect = { left: 250, top: 110, right: 300, bottom: 138 };
overlapping.modelState.badges[1].placement = "outside-left";
overlapping.modelState.badges[1].targetRect = { left: 306, top: 100, right: 450, bottom: 300 };
overlapping.modelState.badges[1].badgeRect = { left: 250, top: 110, right: 300, bottom: 138 };
assert.throws(() => validateBrowserGeometryEvidence(overlapping, actual), /visible badges overlap/u);
const inaccessible = clone();
inaccessible.narrow.badges[0].accessibleName = inaccessible.narrow.badges[0].text;
assert.throws(() => validateBrowserGeometryEvidence(inaccessible, actual), /accessible name/u);
const insideOtherTarget = clone();
insideOtherTarget.modelState.badges[1].targetRect = { left: 250, top: 100, right: 450, bottom: 300 };
insideOtherTarget.modelState.badges[1].badgeRect = { left: 310, top: 200, right: 442, bottom: 228 };
assert.throws(() => validateBrowserGeometryEvidence(insideOtherTarget, actual),
  /inside label intersects another visible target/u);

console.log(JSON.stringify({ cases: 7, browserGeometryContract: "pass" }));
