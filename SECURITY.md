# Security model

ProofLens processes untrusted webpage pixels and DOM state with broad host access, so its release contract is deliberately narrow.

## Trust boundaries

- Webpage DOM, CSS, URLs, image headers, rendered pixels, and mutations are untrusted.
- The extension package, model lock, bundled ONNX Runtime, and extension-owned IndexedDB are trusted only after explicit integrity checks.
- Page scripts cannot call privileged extension handlers directly; content-script messages cross Chrome’s isolated-world boundary.
- No external service is trusted with image content or decisions.

## Controls

- Manifest V3 and extension CSP restrict executable code to the package, block remote connections and image beacons, and explicitly allow only the local WASM/worker/style requirements.
- The model is verified by exact byte count and SHA-256 before storage and re-hashed from persisted bytes once per offscreen-document lifetime.
- Runtime results are cached by local-pixel content SHA-256, not mutable URL.
- Extension inference accepts only `data:image/*` payloads created from already-rendered pixels. It never fetches a page-controlled URL.
- Local image payloads are capped before messaging and again while streaming. Supported raster headers are inspected before decode; 8,192-pixel edges, more than 25 megapixels, and aspect ratios above 16:1 fail closed.
- A document retains at most 512 targets and 32 pending records. Only one content analysis runs at a time; service-worker and offscreen admission are independently capped at eight. Active-tab captures are serialized and start at least 600 milliseconds apart to stay within Chrome's capture quota during rapid CSS changes. Each capture requires the original sender document ID and origin before and after Chrome's capture call; page lifecycle changes invalidate in-flight content results.
- Full-DOM discovery examines at most 5,000 elements per pass and is throttled to once per second.
- Content results require the matching request ID and unchanged source; late results are discarded.
- Badges live in a closed styling boundary, repair host removal/tampering, and use text plus color. Failures never become confidence scores.
- Page summaries are partitioned by document navigation; revisions prevent late messages from regressing state within one document, and tab navigation clears retained state.
- The package audit rejects localhost, common telemetry endpoints, a missing WASM fallback, a model-hash mismatch, and an unsafe CSP.

## Residual risks

- ONNX Runtime, browser image decoders, GPU drivers, and Chrome itself remain a dependency attack surface.
- `<all_urls>` is powerful. It is needed to inject automatic labels and locally capture rendered pixels on ordinary active tabs; users can disable analysis per origin.
- A hostile page can visually cover the overlay, imitate ProofLens styling elsewhere on the page, or continuously mutate the DOM. The closed root, self-repair, throttling, and hard admission limits contain direct tampering and resource use but cannot make an in-page UI equivalent to trusted browser chrome.
- Active-viewport capture is transient and local, but it can contain pixels outside the target before the offscreen crop. The full capture is not persisted, logged, or transmitted.

Report vulnerabilities privately to the repository owner. Do not include secrets, private images, or exploit payloads in a public issue.
