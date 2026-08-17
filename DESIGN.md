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

## Popup and setup

The popup names the active host and shows current complete, flagged, unavailable, analyzing, and queued counts. It refreshes while open. “Show labels on this page (temporary)” controls the current document and resets after a reload or navigation; “Analyze this site (saved)” persists by origin. Before any control mutates state, the popup revalidates the active tab ID and origin; a cross-origin navigation permanently disables that popup instance without writing or relaying a stale setting. Both controls remain busy until confirmation or a surfaced terminal failure. A failed saved write restores the persisted state with a short error; if saving succeeds but the current page changes before delivery, the popup retains the saved state and explains that it applies after reload. Re-scan remains disabled on unsupported pages, stays visibly busy for a perceptible interval, and must produce fresh target work on supported pages. The setup page shows model name, packaged size, SHA-256, threshold, determinate byte progress, and the explicit `Offline ready` state.

## Responsive and accessibility behavior

Labels remain inside the viewport at its current zoom and stay inside or immediately beside the target they describe. An exterior chip cannot cross another tracked target or drift along the page to avoid a collision. If no unambiguous position exists, the chip is hidden rather than shown beside the wrong image; the popup still reports the page-level count. Text truncation may occur on narrow viewports, while the complete meaning remains in the accessible label. Motion appears only for the analyzing state and is disabled by `prefers-reduced-motion`. SeroSlop does not add a keyboard stop unless an element has an action.

Release evidence must cover setup preparing and ready states, the production popup document’s supported/unsupported, failure, cross-origin-navigation, and re-scan states, likely-AI, below-flag-threshold, unavailable, closed-root recovery, dynamic images, responsive images, a CSS composite, rejection of an in-flight result after its CSS source changes, CSSOM-only background reconciliation, reduced-motion behavior, target-associated dense labels, an isolated 64-pixel target, and a narrow viewport at a non-default device scale. The project-owned Chrome test is the executable source of truth for supported targets and offline behavior; it opens the popup document as an extension-page tab and does not claim Chrome toolbar-window lifecycle coverage.
