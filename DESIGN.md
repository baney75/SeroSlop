# SeroSlop design system

SeroSlop should feel direct, calm, and precise. The interface exists to answer three questions: is the model ready, what is happening on this page, and what score did this image receive? Product UI must not contain marketing panels, privacy slogans, repeated reassurance, or general detector caveats. Documentation carries privacy, security, benchmark, and limitation detail.

## Product principles

1. **Show state before explanation.** Lead with a short status, then one sentence that tells the user what to do or what happens next.
2. **One primary action.** A surface should expose only the next useful action. Completed and automatic states do not keep disabled decorative buttons visible.
3. **Scores stay literal.** Completed labels show a one-decimal score out of 100. The inclusive 65.0 threshold is never described as proof, authenticity, or certainty.
4. **Unavailable is not a score.** Capture or inference failures use `unavailable`; the UI never invents a number.
5. **Technical detail is available, not dominant.** Model hashes and package facts live behind a clearly named disclosure on the setup screen.
6. **Light and dark are equal products.** Both themes use the same hierarchy, spacing, target sizes, focus treatment, and state semantics.

## Visual language

- Use the SeroSlop slashed-zero mark for Chrome toolbar icons and the setup identity. The wide SeroSlop logo may appear in the popup header. Never substitute a letter fallback when a brand asset is available.
- Use system UI type. Headings are compact, semibold or bold, and slightly tightened; body text remains at least 13px in extension surfaces and 15px on full pages.
- Use neutral near-black, white, and gray surfaces. Red is the brand/progress accent, green is reserved for a verified-ready state, and red error text is reserved for a failed action. State meaning must also appear in text.
- Use 8–12px control radii, 18–20px large-card radii, hairline borders, and restrained shadows. Avoid decorative gradients, glass effects, feature cards, and illustration panels.
- All actionable controls are at least 44px tall, have a visible keyboard focus ring, and remain understandable without hover.

## Setup screen

The setup screen is a centered, single-card workflow with a small brand header, one state heading, one supporting sentence, one progress region when applicable, one primary action when applicable, and a collapsed `Model details` disclosure.

| State | Heading | Supporting text | Progress | Action |
|---|---|---|---|---|
| Checking | `Checking model` | `Reading the packaged model status.` | May show the current byte count | None |
| Not verified | `Model not verified` | `Verify the model before scanning pages.` | Hidden | `Verify model` |
| Verifying | `Verifying model` | `Keep this tab open until verification finishes.` | Determinate verified bytes | None |
| Ready | `Offline ready` | `Model verified. You can close this tab.` | Complete | None |
| Error | `Setup failed` | The bounded runtime error or the safe fallback message | Hidden | `Retry verification` |

The progress text uses tabular numbers and reports verified bytes without duplicating the state. The details disclosure contains the display name, packaged size, inclusive threshold, and SHA-256. Checking and verification set the status region busy; a separate polite live region announces phase changes without announcing every byte update. At 480px and below, details use a single column and the primary action fills the available width. Reduced-motion mode disables progress transitions.

SeroSlop adds compact status labels near analyzed images without changing page layout. The extension popup controls one active tab; the setup page verifies the packaged detector.

## Label states

| State | Visible label | Meaning |
|---|---|---|
| Queued | `SeroSlop · queued` | The target is waiting to enter the bounded local queue. |
| Analyzing | `SeroSlop · analyzing` | Local preprocessing or inference is running. |
| Likely AI | `Likely AI · n.n/100` | The model score is at least the inclusive 65.0/100 threshold. |
| Below flag threshold | `Below flag threshold · n.n/100` | The model score is below 65.0/100. |
| Unavailable | `SeroSlop · unavailable` | Pixels could not be captured, decoded, or processed safely. No score is inferred. |

State text must carry the meaning without color. A below-threshold result uses a neutral surface rather than an “authentic” green. Labels use noninteractive `role=status` nodes with live announcements disabled, so a page with many images does not create dead keyboard controls or announcement noise. Each accessible label includes a bounded image alt/accessible name when present, or identifies a CSS background, so the result remains associated with its target. The label root is closed and page-removal is repaired, but an in-page overlay cannot prevent a hostile site from covering it or imitating its appearance elsewhere.

An HTML image receives one label. All CSS background layers rendered on one element form one composite visual target and receive one label; SeroSlop does not imply a separate decision for each layer.

## Scan modes

The first successful model setup requires one explicit choice. There is no automatic default.

| Stored value | User-facing name | Meaning | Primary popup action |
|---|---|---|---|
| `pick` | `Choose an image` | Analyze one user-selected image and nothing else. A successful image context-menu action also selects this mode. | `Choose image` |
| `main` | `Main images` | Analyze supported images inside `main`, `article`, or `[role=main]`, excluding nested header, navigation, aside, and footer regions. It never falls back to every-image scanning. | `Re-scan main images` |
| `all` | `Every image` | Analyze every supported visible image target within the existing document and queue limits. | `Re-scan every image` |

The selected mode is global and persists locally. The setup page presents the three radio choices only after `Offline ready`, followed by one explicit `Save mode` action. The popup shows the active host, current mode, page counts, one `Change mode` action, and one mode-specific primary action. Its inline mode sheet uses the same names and descriptions as setup and does not save merely because a radio receives focus. A failed save restores the previous mode and keeps the sheet open with a short error.

Before the popup saves or relays a page action, it revalidates the active tab ID and origin. A cross-origin navigation permanently disables that popup instance. A confirmed global mode save may remain valid if page delivery fails; the popup then states that the mode will apply after reload. Unsupported pages expose no scanning action. Setup and popup never display marketing slogans, repeated privacy reassurance, or general detector caveats.

## Image picker

`Choose an image` performs no automatic analysis. The picker starts only after the user presses the popup’s `Choose image` button on a supported page. While active it adds one instruction bar and one outline in the extension’s closed overlay root. The outline is a 2px solid SeroSlop red stroke, offset 2px from the exact candidate bounds, with no fill, dimming, glow, or animation. It follows the topmost eligible image or supported background under the pointer. A click chooses that target; `Tab` and `Shift+Tab` move through the bounded candidate list, `Enter` or `Space` chooses, and `Escape` cancels. Selection removes any previous result and queues exactly one descriptor for the chosen element.

The picker is bound to the current document and origin. It cancels on mode change or page exit, rejects a stale popup origin, restores the previous focus when possible, and removes every temporary node and listener on completion. It does not add permanent tab stops or modify page element attributes. If there is no eligible target, the popup reports that directly.

### Image context menu

Right-clicking an eligible top-level page image exposes one image-only command: `Analyze this image with SeroSlop`. Chrome supplies the clicked image URL and frame. The content script also records the trusted context-menu target for at most 15 seconds. Analysis starts only when the current document, exact rendered source, and recorded element still match. A successful action selects `Choose an image`, removes previous page records, and queues exactly that image. A stale, scripted, navigated, disabled, unsupported, or mismatched target does nothing. If the model is not ready, the command opens setup instead of fabricating a result.

## Popup and setup

The popup names the active host and shows current complete, flagged, unavailable, analyzing, and queued counts. It refreshes while open. Re-scan remains disabled on unsupported pages, stays visibly busy for a perceptible interval, and must produce fresh target work on supported pages. The setup page shows model name, packaged size, SHA-256, threshold, determinate byte progress, `Offline ready`, and the required scan-mode choice.

## Responsive and accessibility behavior

Labels remain inside the viewport at its current zoom and stay inside or immediately beside the target they describe. An exterior chip cannot cross another tracked target or drift along the page to avoid a collision. If no unambiguous position exists, the chip is hidden rather than shown beside the wrong image; the popup still reports the page-level count. Text truncation may occur on narrow viewports, while the complete meaning remains in the accessible label. Motion appears only for the analyzing state and is disabled by `prefers-reduced-motion`. SeroSlop does not add a keyboard stop unless an element has an action.

Release evidence must cover setup preparing, ready, explicit mode selection, and mode-save failure; popup supported/unsupported and cross-origin-navigation states; mode persistence after restart; pick mode with zero automatic records, visible target outline, keyboard cancel, and exact-one-target completion; main mode exclusion outside semantic content; every-image coverage; likely-AI, below-flag-threshold, unavailable, closed-root recovery, dynamic images, responsive images, a CSS composite, rejection of an in-flight result after its CSS source changes, CSSOM-only background reconciliation, reduced-motion behavior, target-associated dense labels, an isolated 64-pixel target, and a narrow viewport at a non-default device scale. Popup and setup require visual checks in light and dark themes at 320, 375, 414, and 768 CSS pixels; no horizontal overflow, clipped actions, or sub-44px targets are accepted. The project-owned Chrome test is the executable source of truth for supported targets and offline behavior; it opens the popup document as an extension-page tab and does not claim Chrome toolbar-window lifecycle coverage.

The public install page is checked at 320, 375, 390, 414, and 768 CSS pixels in light and dark themes. Every install step has one counter column and one content column, commands remain at least 160px wide, navigation targets remain at least 44px tall, and horizontal overflow is a release failure.
