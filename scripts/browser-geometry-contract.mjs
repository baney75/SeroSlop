const POSITION_MARGIN = 6;
const TOLERANCE = 1;

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function requireRect(rect, label) {
  requireCondition(rect && typeof rect === "object", `${label} is missing`);
  for (const edge of ["left", "top", "right", "bottom"]) {
    requireCondition(Number.isFinite(rect[edge]), `${label}.${edge} is not finite`);
  }
  requireCondition(rect.right >= rect.left && rect.bottom >= rect.top, `${label} is inverted`);
}

export function rectsIntersect(left, right) {
  return left.left < right.right && left.right > right.left &&
    left.top < right.bottom && left.bottom > right.top;
}

function rectIntersectsViewport(rect, viewport) {
  return rect.right > 0 && rect.bottom > 0 && rect.left < viewport.width && rect.top < viewport.height;
}

function rectInsideViewport(rect, viewport) {
  return rect.left >= 0 && rect.top >= 0 && rect.right <= viewport.width && rect.bottom <= viewport.height;
}

export function badgeAssociationError(badge, visibleTargetRects, positionMargin = POSITION_MARGIN) {
  const badgeRect = badge.badgeRect;
  const targetRect = badge.targetRect;
  const placement = String(badge.placement ?? "");
  const containedByTarget = badgeRect.left >= targetRect.left - TOLERANCE &&
    badgeRect.top >= targetRect.top - TOLERANCE &&
    badgeRect.right <= targetRect.right + TOLERANCE &&
    badgeRect.bottom <= targetRect.bottom + TOLERANCE;
  if (placement === "inside-top-right") {
    if (!containedByTarget) return "inside label is not contained by its target";
    if (visibleTargetRects.some((rect) => rect !== targetRect && rectsIntersect(badgeRect, rect))) {
      return "inside label intersects another visible target";
    }
    return undefined;
  }
  if (!placement.startsWith("outside-")) {
    return `unsupported visible placement ${placement}`;
  }
  if (visibleTargetRects.some((rect) => rectsIntersect(badgeRect, rect))) {
    return "outside label intersects a visible target";
  }
  const verticallyAligned = badgeRect.top < targetRect.bottom && badgeRect.bottom > targetRect.top;
  const horizontallyAligned = badgeRect.left < targetRect.right && badgeRect.right > targetRect.left;
  if (placement === "outside-right" && verticallyAligned &&
    Math.abs(badgeRect.left - targetRect.right - positionMargin) <= TOLERANCE) return undefined;
  if (placement === "outside-left" && verticallyAligned &&
    Math.abs(targetRect.left - badgeRect.right - positionMargin) <= TOLERANCE) return undefined;
  if (placement === "outside-bottom" && horizontallyAligned &&
    Math.abs(badgeRect.top - targetRect.bottom - positionMargin) <= TOLERANCE) return undefined;
  if (placement === "outside-top" && horizontallyAligned &&
    Math.abs(targetRect.top - badgeRect.bottom - positionMargin) <= TOLERANCE) return undefined;
  return "outside label is not immediately associated with its declared target side";
}

export function pngDimensions(bytes) {
  requireCondition(Buffer.isBuffer(bytes) && bytes.length >= 24, "Screenshot is not a complete PNG");
  const signature = "89504e470d0a1a0a";
  requireCondition(bytes.subarray(0, 8).toString("hex") === signature &&
    bytes.subarray(12, 16).toString("ascii") === "IHDR", "Screenshot is not a PNG with an IHDR header");
  return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
}

function validateScreenshot(section, actualScreenshot, label) {
  requireCondition(section.screenshot && typeof section.screenshot === "object", `${label} screenshot record is missing`);
  requireCondition(/^[a-f0-9]{64}$/u.test(section.screenshot.sha256 ?? ""), `${label} screenshot hash is invalid`);
  requireCondition(section.screenshot.sha256 === actualScreenshot.sha256, `${label} screenshot hash is stale`);
  requireCondition(section.screenshot.width === actualScreenshot.width &&
    section.screenshot.height === actualScreenshot.height, `${label} screenshot dimensions are stale`);
  requireCondition(section.viewport.width === actualScreenshot.width &&
    section.viewport.height === actualScreenshot.height, `${label} viewport does not match its screenshot coordinates`);
}

function validateSection(section, actualScreenshot, label) {
  requireCondition(section && typeof section === "object", `${label} geometry record is missing`);
  requireCondition(Number.isInteger(section.viewport?.width) && section.viewport.width > 0 &&
    Number.isInteger(section.viewport?.height) && section.viewport.height > 0, `${label} viewport is invalid`);
  requireCondition(Array.isArray(section.badges) && section.badges.length > 0, `${label} badge records are missing`);
  validateScreenshot(section, actualScreenshot, label);

  for (const [index, badge] of section.badges.entries()) {
    requireRect(badge.badgeRect, `${label}.badges[${index}].badgeRect`);
    requireRect(badge.targetRect, `${label}.badges[${index}].targetRect`);
    requireCondition(typeof badge.elementId === "string" && badge.elementId.length > 0,
      `${label}.badges[${index}] has no target identity`);
    requireCondition(typeof badge.hidden === "boolean" && typeof badge.display === "string",
      `${label}.badges[${index}] visibility state is invalid`);
    if (badge.hidden) {
      requireCondition(badge.display === "none", `${label} hidden badge ${badge.elementId} is still rendered`);
    }
  }

  const visibleTargetRects = section.badges.map((badge) => badge.targetRect)
    .filter((rect) => rectIntersectsViewport(rect, section.viewport));
  const visibleBadges = section.badges.filter((badge) => !badge.hidden && badge.display !== "none" &&
    rectIntersectsViewport(badge.badgeRect, section.viewport));
  requireCondition(visibleBadges.length > 0, `${label} contains no visible badge`);
  for (const badge of visibleBadges) {
    requireCondition(rectInsideViewport(badge.badgeRect, section.viewport),
      `${label} badge ${badge.elementId} leaves the viewport`);
    requireCondition(badge.pointerEvents === "none", `${label} badge ${badge.elementId} can intercept input`);
    requireCondition(typeof badge.text === "string" && typeof badge.accessibleName === "string" &&
      badge.accessibleName.length > badge.text.length,
    `${label} badge ${badge.elementId} has no target-specific accessible name`);
    if (badge.state === "complete") {
      requireCondition(badge.accessibleName.includes("not proof"),
        `${label} badge ${badge.elementId} omits the not-proof caveat`);
    }
    const associationError = badgeAssociationError(badge, visibleTargetRects, POSITION_MARGIN);
    requireCondition(!associationError, `${label} badge ${badge.elementId} lost target association: ${associationError}`);
  }
  for (let leftIndex = 0; leftIndex < visibleBadges.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < visibleBadges.length; rightIndex += 1) {
      requireCondition(!rectsIntersect(visibleBadges[leftIndex].badgeRect, visibleBadges[rightIndex].badgeRect),
        `${label} visible badges overlap`);
    }
  }
  return { visibleBadges: visibleBadges.length, hiddenBadges: section.badges.filter((badge) => badge.hidden).length };
}

export function validateBrowserGeometryEvidence(evidence, actualScreenshots) {
  requireCondition(evidence?.schemaVersion === 1 && evidence.positionMargin === POSITION_MARGIN,
    "Browser geometry evidence schema changed");
  requireCondition(actualScreenshots && typeof actualScreenshots === "object", "Actual screenshots are missing");
  const modelState = validateSection(evidence.modelState, actualScreenshots.modelState, "model-state");
  const narrow = validateSection(evidence.narrow, actualScreenshots.narrow, "narrow");
  const smallTarget = validateSection(evidence.smallTarget, actualScreenshots.smallTarget, "small-target");

  requireCondition(evidence.modelState.badges.some((badge) => badge.elementId === "likely-ai-fixture" &&
    badge.state === "complete" && !badge.hidden && badge.display !== "none"),
  "Model-state evidence omits the visible likely-AI result");
  requireCondition(evidence.modelState.badges.some((badge) => badge.elementId === "below-threshold-fixture" &&
    badge.state === "complete" && !badge.hidden && badge.display !== "none"),
  "Model-state evidence omits the visible below-threshold result");
  requireCondition(evidence.narrow.viewport.width === 480 && evidence.narrow.viewport.height === 720 &&
    evidence.narrow.deviceScaleFactor === 1.5, "Narrow geometry does not bind the 480x720, 1.5x contract");
  requireCondition(evidence.smallTarget.viewport.width === 480 && evidence.smallTarget.viewport.height === 720 &&
    evidence.smallTarget.badges.length === 1, "Small-target geometry is not isolated at 480x720");
  const isolated = evidence.smallTarget.badges[0];
  requireCondition(isolated.elementId === "prooflens-small-target" && isolated.state === "complete" &&
    !isolated.hidden && isolated.display !== "none" && String(isolated.placement).startsWith("outside-") &&
    !rectsIntersect(isolated.badgeRect, isolated.targetRect), "Small-target evidence does not prove non-obstruction");

  return { modelState, narrow, smallTarget };
}

export function browserGeometryBadgeRecord(badge) {
  return {
    elementId: String(badge.elementId ?? ""),
    state: String(badge.state ?? ""),
    hidden: Boolean(badge.hidden),
    display: String(badge.display ?? ""),
    placement: String(badge.placement ?? ""),
    pointerEvents: String(badge.pointerEvents ?? ""),
    text: String(badge.text ?? ""),
    accessibleName: String(badge.accessibleName ?? ""),
    badgeRect: Object.fromEntries(["left", "top", "right", "bottom"].map((edge) => [edge, badge.badgeRect?.[edge]])),
    targetRect: Object.fromEntries(["left", "top", "right", "bottom"].map((edge) => [edge, badge.targetRect?.[edge]])),
  };
}

export const BROWSER_GEOMETRY_POSITION_MARGIN = POSITION_MARGIN;
