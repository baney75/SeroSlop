# ProofLens

ProofLens is a native Manifest V3 Chrome extension that shows a local AI-image model score beside images on ordinary webpages. It runs one packaged ONNX model inside the browser, without a detector API, localhost process, hash lookup, telemetry, or post-install model download.

The fixed decision rule is inclusive: **AI score ≥ 65.0/100 is flagged**. Every completed analysis shows a numeric model score; failures say **unavailable** instead of inventing one. The score is not a calibrated probability, and a badge is not proof of origin or authenticity.

## Release evidence

The shipped classifier keeps the Community Forensics ViT-S/16 backbone frozen and replaces only its 384-to-1 head. The upstream backbone was trained on 5.4 million real/synthetic examples spanning 4,803 generators. ProofLens trained the replacement head on **103,600 public images**: 51,200 synthetic and 52,400 non-AI. This produced 114,400 feature views, and fresh extraction covered every selected image and view. Independent ONNX comparison found exactly two changed initializers, `classifier.weight` and `classifier.bias`; the other 198 initializers and the graph contract are unchanged.

Validation balanced accuracy was 95.00% on originals, 96.33% on screenshots, and 94.50% on both JPEG stress views. These are model-selection results, not an untouched generalization estimate.

The original 600-image Kling v2.1/Library of Congress holdout was run once and is consumed. Its stored predictions failed the frozen numeric contract before bootstrap: 2,231 of 2,400 probabilities did not equal the verifier’s binary64 sigmoid of their recorded logits within `2e-12`. The failure packet is permanently marked `acceptanceEligible: false`; its point estimates are diagnostic only. The model, calibration, threshold, preprocessing, acceptance gates, and extension runtime did not change in response.

The score-blind replacement packet was scored once from the public V3 freeze and is now consumed. Its 600-image confirmation passed every frozen accuracy gate: balanced accuracy was 89.67%–97.67%, and the lowest 95% balanced-accuracy bound was 87.17%. The separate 319-photo StockImages slice failed the stricter false-positive gate. Original photos had an 8.78% false-positive rate with a 12.39% Wilson upper bound; JPEG-75 photos had a 13.79% rate with an 18.01% upper bound. Both exceed the frozen 10% overall upper-bound limit, so the packet is permanently `acceptanceEligible: false`. It is development evidence only, not a release result. [BENCHMARK.md](BENCHMARK.md) records the complete packet, append-only lineage, and unchanged gates.

Public repository evidence does not establish the bounty maintainer’s private score, acceptance decision, or payment.

## Install from source

Requirements: Node.js 20.9 or newer, Python 3.10 or newer, `npm`, `zip`, and Chrome 121 or newer.

```bash
git clone https://github.com/baney75/prooflens.git && cd prooflens
python3 -m venv .verify-venv
source .verify-venv/bin/activate
python -m pip install -r benchmark/verify-requirements.txt
npm ci && npm run verify:static
```

Then:

1. Open `chrome://extensions`, then enable **Developer mode** using the toggle in its upper-right corner.
2. Select **Load unpacked** and choose the generated `dist/` directory.
3. Keep the setup tab open until it says **Offline ready**. The model is already in the package; setup verifies its SHA-256 and prepares local storage.
4. Visit an ordinary webpage. ProofLens automatically queues visible `<img>`, `picture`/`srcset`, dynamic images, and CSS `background-image` targets.

The reproducible archive is generated at `release/prooflens.zip`; its digest is recorded in `release/SHA256SUMS.txt`.

## Verify the release

```bash
python3 -m venv .verify-venv
source .verify-venv/bin/activate
python -m pip install -r benchmark/verify-requirements.txt
npm ci
npm run verify:static      # stage-aware public source/freeze/final pixel-free checks
npm run verify:release     # mandatory local pixel + shipped-ONNX byte-identical replay
npm run browser:install    # one-time project browser install
npm run test:chrome        # fresh profile, restart, forced WASM, offline/no-localhost E2E
npm run test:chrome:webgpu # same contract through WebGPU
```

Keep `.verify-venv` active for both verification commands. Before replacement scoring, `verify:static` accepts only the exact clean A4 numeric-recovery source or its receipt-only V3 child and proves that replacement confirmatory, web-negative, replay, parity, and browser artifacts are absent. Neither earlier receipt nor the consumed v1 packet can authorize the replacement evaluator. On a descendant of V3, the same command requires the complete final packet. `verify:release` additionally needs the local replacement pixels and pinned benchmark environment at `benchmark/.venv`; it replays only replacement-v2 outputs and distinguishes stored predictions from observed inference. GitHub Actions runs the stage-aware pixel-free contract and portable forced-WASM path. WebGPU remains a separate fixed-head local gate because hosted-runner GPU availability is not stable; its source-, model-, and archive-bound receipt is checked by the final static stage.

The browser test closes its fixture server and puts the browser offline before analysis. It verifies measurable setup progress, model persistence across a browser restart, zero post-cutoff network requests, confirmed control states (including injected failures), fresh re-scan work, dynamic and responsive images, one result for a rendered CSS composite, rejection of an in-flight stale CSS result, CSSOM-only background reconciliation, numeric labels, an explicit unavailable state, a closed/recoverable label boundary, bounded hostile-page work, cap-replacement recovery, and target-associated labels at a 480 px viewport with a 1.5× device scale. A chip that cannot be placed without covering another target is hidden instead of being detached from the image it describes; the popup still reports the result count. Release verification also rebuilds under New York and UTC time zones and requires byte-identical archives.

## Runtime design

```text
content script: discovers targets and creates local rendered-pixel or viewport-crop requests
  |
  v
MV3 service worker: rate-limits active-viewport capture when canvas pixels are unavailable,
bounds work, and restores the offscreen document
  |
  v
offscreen document: verifies model bytes, decodes and preprocesses the image,
then runs WebGPU or its clean WASM fallback and applies frozen calibration
  |
  v
content script: accepts only the matching request/source and renders a result
```

When canvas access permits, ProofLens reduces images above a 1,024-pixel long edge and encodes the bounded pixels locally as lossless PNG. Otherwise, it captures the already-rendered active viewport locally and crops the target inside the offscreen document. Viewport capture is confirmed against the exact sending document ID and origin before and after Chrome's capture call. A page-controlled URL is never fetched by extension code, and extension-page CSP blocks remote connection/image beacons, so redirects, DNS rebinding, localhost, and private-network destinations are outside the acquisition path. Oversized sources, excessive decoded dimensions, corrupt model bytes, unsupported protocols, navigated/inactive-tab crops, and decode failures fail closed.

## Model lock

- Artifact: `weights/prooflens-cf384.onnx`
- Bytes: `87,442,080`
- SHA-256: `941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c`
- Input/output: `pixel_values [N,3,384,384]` to `logits [N,1]`
- Upstream corrected revision: `ac6ee457bea904a373065754107451793b56db00`
- License: MIT

The complete machine-readable contract is [model-lock.json](model-lock.json). Training replaces only the frozen backbone’s 384-to-1 classifier head; it does not add a second model or heuristic score.

## Documentation

- [BENCHMARK.md](BENCHMARK.md): evaluation protocol and evidence
- [MODEL_CARD.md](MODEL_CARD.md): provenance, preprocessing, calibration, and limitations
- [PRIVACY.md](PRIVACY.md): data flow and permission rationale
- [SECURITY.md](SECURITY.md): threat model and fail-closed controls
- [DESIGN.md](DESIGN.md): visible states, controls, accessibility, and responsive contract
- [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md): bounty criteria mapped to executable proof
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): bundled runtime/model notices and training-data provenance

ProofLens is MIT licensed. Dataset pixels are not included in this repository.
