# SeroSlop interface contract

SeroSlop adds compact status labels near analyzed images without changing page layout. The extension popup controls one active tab; the setup page verifies the packaged detector.

## Label states

| State | Visible label | Meaning |
|---|---|---|
| Queued | `SeroSlop · queued` | The target is waiting to enter the bounded local queue. |
| Analyzing | `SeroSlop · analyzing` | Local preprocessing or inference is running. |
| Likely AI | `Likely AI · n.n/100` | The model score is at least the inclusive 65.0/100 threshold. |
| Below flag threshold | `Below flag threshold · n.n/100` | The model score is below 65.0/100; this is not proof of authenticity. |
| Unavailable | `SeroSlop · unavailable` | Pixels could not be captured, decoded, or processed safely. No score is inferred. |

State text must carry the meaning without color. A below-threshold result uses a neutral surface rather than an “authentic” green. Labels use noninteractive `role=status` nodes with live announcements disabled, so a page with many images does not create dead keyboard controls or announcement noise. Each accessible label includes a bounded image alt/accessible name when present, or identifies a CSS background, so the estimate remains associated with its target. The label root is closed and page-removal is repaired, but an in-page overlay cannot prevent a hostile site from covering it or imitating its appearance elsewhere.

An HTML image receives one label. All CSS background layers rendered on one element form one composite visual target and receive one label; SeroSlop does not imply a separate decision for each layer.

## Popup and setup

The popup names the active host and shows current complete, flagged, unavailable, analyzing, and queued counts. It refreshes while open and states that estimates are not proof. “Show labels on this page (temporary)” controls the current document and resets after a reload or navigation; “Analyze this site (saved)” persists by origin. Before any control mutates state, the popup revalidates the active tab ID and origin; a cross-origin navigation permanently disables that popup instance without writing or relaying a stale setting. Both controls remain busy until confirmation or a surfaced terminal failure. A failed saved write restores the persisted state with a short error; if saving succeeds but the current page changes before delivery, the popup retains the saved state and explains that it applies after reload. Re-scan remains disabled on unsupported pages, stays visibly busy for a perceptible interval, and must produce fresh target work on supported pages. The setup page shows model name, packaged size, SHA-256, threshold, determinate byte progress, and the explicit `Offline ready` state.

## Responsive and accessibility behavior

Labels remain inside the viewport at its current zoom and stay inside or immediately beside the target they describe. An exterior chip cannot cross another tracked target or drift along the page to avoid a collision. If no unambiguous position exists, the chip is hidden rather than shown beside the wrong image; the popup still reports the page-level count. Text truncation may occur on narrow viewports, while the complete meaning remains in the accessible label. Motion appears only for the analyzing state and is disabled by `prefers-reduced-motion`. SeroSlop does not add a keyboard stop unless an element has an action.

Release evidence must cover setup preparing and ready states, the production popup document’s supported/unsupported, failure, cross-origin-navigation, and re-scan states, likely-AI, below-flag-threshold, unavailable, closed-root recovery, dynamic images, responsive images, a CSS composite, rejection of an in-flight result after its CSS source changes, CSSOM-only background reconciliation, reduced-motion behavior, target-associated dense labels, an isolated 64-pixel target, and a narrow viewport at a non-default device scale. The project-owned Chrome test is the executable source of truth for supported targets and offline behavior; it opens the popup document as an extension-page tab and does not claim Chrome toolbar-window lifecycle coverage.
