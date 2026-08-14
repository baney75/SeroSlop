# Accepted brief and bounty evidence

## Brief

- Outcome: a public, MIT-licensed Manifest V3 Chrome extension that automatically scores ordinary webpage images with one browser-local model and is published only if it clears POIDH bounty 323’s fixed-threshold accuracy bar.
- Audience: the bounty maintainer using a clean Chrome profile and reviewers rebuilding the public repository.
- Deliverable: source, packaged ONNX, reproducible build, install guide, frozen training/evaluation evidence, clean-profile browser evidence, and privacy/security documentation.
- Non-goals: on-chain claim submission, wallet signing, cloud inference, a localhost detector, provenance proof, generator attribution, or a claim that the maintainer’s private benchmark has already passed.
- Authorized changed surface: this repository, its public non-protected GitHub branch, and local temporary training/browser artifacts. No wallet, DNS, protected branch, release, or bounty transaction is authorized.
- Stop condition: stop and remove the unsubmitted repository only if an accepted bounty claim appears before publication. Otherwise publish only after every gate below and four independent fixed-head reviews pass.

## Criteria matrix

| Bounty criterion | Implementation | Required proof before publication |
|---|---|---|
| MIT source | Root license plus verbatim model/runtime/derivative notices in source and ZIP | package audit and archive inspection |
| Native Manifest V3 | service worker, offscreen document, content script | manifest tests and built manifest |
| Browser-local inference | one packaged FP32 ONNX; WebGPU with WASM fallback | both clean-profile E2E paths |
| Offline after setup | no model downloader/API; model is hash-verified from the package | restart, server shutdown, browser-offline mode, zero post-cutoff requests |
| Automatic ordinary-page analysis | image, responsive source, dynamic DOM, CSS composite, and bounded viewport-crop paths | Chrome target/race/reconciliation assertions |
| Per-image score | one-decimal 0–100 model score; failure is `unavailable` | unit and browser-state assertions |
| Fixed threshold | inclusive score `>= 65.0/100`; one validation-frozen intercept | boundary tests and final calibration lock |
| Large realistic local model training | frozen 5.4M-example upstream backbone; replacement head trained on 103,600 unique public images / 114,400 views | corpus packet, trainer summary, grid, coverage, classifier-only ONNX proof |
| Accuracy | every confirmatory view must have lower 95% BA, real recall, and synthetic recall at least 75%; Kling recall at least 60% | one-time frozen predictions and recomputed bootstrap |
| False-positive robustness | every web-negative view must have Wilson upper 95% FPR at most 10% overall and 20% per source | frozen 319-row challenge and recomputed intervals |
| Reproducible delivery | lockfile, model lock, fixed UTC ZIP timestamps, stage-aware pre-score/final CI | with `benchmark/verify-requirements.txt` installed in the active verification venv: `npm ci && npm run verify:static`, cross-time-zone byte equality, checksum |
| Predictions came from the shipped model | frozen local pixels and packaged ONNX | `npm run verify:release` performs byte-identical inference replay before pixel cleanup |
| Easy and safe controls | saved site toggle, page-temporary labels, perceptible re-scan, no pointer interception, target-associated chips with honest collision hiding | interaction/geometry E2E and inspected screenshots |

## Revision conditions

Rework is mandatory for any model/hash mismatch, missing corpus attribution, split leakage, incomplete threshold search, failed confidence gate, material class/source collapse, network activity after cutoff, fabricated score, stale-result race, inaccessible control, misleading probability language, non-reproducible archive, or fixed-head reviewer finding. Passing checks is evidence, not proof of the maintainer’s private result.
