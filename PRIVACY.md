# Privacy and permissions

SeroSlop has no analytics, telemetry, ads, user account, detector API, cloud inference, localhost backend, or remote model loader. Extension-page Content Security Policy also blocks remote connections and image beacons, so this boundary does not rely only on current call sites.

## Data flow

- The content script discovers image targets and renders labels.
- When page pixels are canvas-readable, it reduces images above a 1,024-pixel long edge, encodes the bounded pixels locally as lossless PNG, and sends that data URL through extension messaging.
- Otherwise, the service worker locally captures the already-rendered active viewport while hiding SeroSlop labels. Capture is bound to the exact content-script document ID and origin before and after Chrome’s capture call; a navigation fails closed. The offscreen document crops the target region before inference. The transient full-viewport capture is never stored, logged, or transmitted.
- Extension code never fetches a page-controlled image URL. Redirects, DNS rebinding, cookies, referrers, localhost, and private-network destinations are therefore absent from image acquisition.
- Decoded pixels, tensors, scores, and content hashes remain in the browser process.
- The model is bundled in the extension package. Setup copies it to extension-owned IndexedDB only after verifying its size and SHA-256. Inference never downloads or replaces it.

## Permissions

| Permission | Reason |
|---|---|
| `offscreen` | Chrome service workers have no DOM/canvas; an offscreen document decodes images and runs ONNX Runtime Web. |
| `storage` | Saves per-origin enable/disable choices. |
| `unlimitedStorage` | Holds the verified 87.4 MB model in extension-owned IndexedDB. |
| `<all_urls>` host access | Injects automatic labels on ordinary webpages and permits local capture of rendered pixels in the active tab when canvas pixels are unavailable. |

An image is marked unavailable when neither a local canvas snapshot nor a safe active-viewport crop can be produced. Local image payloads, decoded dimensions, per-page targets, and queued work all have hard limits.

The popup can disable analysis for an origin or hide labels. Uninstalling the extension removes its local model and settings through Chrome’s normal extension-storage lifecycle.
