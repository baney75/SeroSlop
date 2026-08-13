# ProofLens interface contract

ProofLens adds compact status labels near analyzed images without changing page layout. The extension popup controls one active tab; the setup page verifies the packaged detector.

## Label states

| State | Visible label | Meaning |
|---|---|---|
| Queued | `ProofLens · queued` | The target is waiting to enter the bounded local queue. |
| Analyzing | `ProofLens · analyzing` | Local preprocessing or inference is running. |
| Likely AI | `Likely AI · n.n%` | The numeric estimate is at least the inclusive 65.0% threshold. |
| Not flagged | `Not flagged · n.n%` | The estimate is below 65.0%; this is not proof of authenticity. |
| Unavailable | `ProofLens · unavailable` | Pixels could not be captured, decoded, or processed safely. No score is inferred. |

State text must carry the meaning without color. Labels use noninteractive `role=status` nodes with live announcements disabled, so a page with many images does not create dead keyboard controls or announcement noise. The label root is closed and page-removal is repaired, but an in-page overlay cannot prevent a hostile site from covering it or imitating its appearance elsewhere.

An HTML image receives one label. All CSS background layers rendered on one element form one composite visual target and receive one label; ProofLens does not imply a separate decision for each layer.

## Popup and setup

The popup shows current complete, flagged, unavailable, analyzing, and queued counts. It refreshes while open. “Show labels in this tab” is a tab-lifetime view control; “Analyze this site” persists by origin. The setup page must show model name, packaged size, SHA-256, threshold, progress, and the explicit `Offline ready` state.

## Responsive and accessibility behavior

Labels remain inside the viewport at its current zoom. Text truncation may occur on narrow viewports; the complete meaning remains in the accessible label. Motion appears only for the analyzing state and is disabled by `prefers-reduced-motion`. ProofLens does not add a keyboard stop unless an element has an action.

Release evidence must cover setup, popup progress, likely-AI, not-flagged, unavailable, closed-root recovery, dynamic images, responsive images, a CSS composite, rejection of an in-flight result after its CSS source changes, CSSOM-only background reconciliation, reduced-motion behavior, and a narrow viewport. The project-owned Chrome test is the executable source of truth for supported targets and offline behavior.
