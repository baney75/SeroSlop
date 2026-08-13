# Accepted brief and bounty evidence

## Brief

- Outcome: a public, MIT-licensed Chrome extension that meets POIDH bounty 323’s observable implementation contract and materially exceeds its 75% balanced-accuracy floor on a frozen public holdout.
- Audience: the bounty maintainer evaluating a clean Chrome profile, plus reviewers reproducing the source build.
- Deliverable: source, exact model, build/install instructions, aggregate evaluation evidence, project-owned browser tests, privacy/security documentation, and a public GitHub repository.
- Non-goals: on-chain claim submission, wallet signing, cloud inference, a local server, provenance proof, generator attribution, or a claim that the private maintainer benchmark has already passed.
- Changed surface: this repository and one temporary GPU pod. No user browser profile, wallet, DNS, production service, or private repository is changed.
- Stop condition: if the bounty becomes completed/accepted before publication, stop and remove the unsubmitted repository. Otherwise publish only after all gates below pass.

## Criteria matrix

| Bounty criterion | Implementation | Executable evidence |
|---|---|---|
| MIT source | Root `LICENSE`; verbatim model/runtime/derivative notices shipped in the ZIP | `npm run check:package` and repository audit |
| Native Manifest V3 extension | Service worker + offscreen document + content script | `tests/manifest.test.ts`, built `dist/manifest.json` |
| Browser-local inference | Packaged ONNX; WebGPU primary; WASM fallback | `npm run test:chrome:webgpu`, `npm run test:chrome` |
| Offline after model setup | No model downloader; packaged model copied after SHA check | fresh-profile restart/offline E2E; zero post-cutoff requests |
| No cloud/API/backend/hash lookup | No detector endpoint, localhost path, or page-controlled runtime fetch | `npm run check:package`, URL policy tests, `SECURITY.md` |
| Automatic ordinary-page analysis | `<img>`, `currentSrc`, `picture/srcset`, dynamic DOM, CSS composites; local viewport crop when canvas is blocked | Chrome E2E target, CSSOM, and in-flight stale-response assertions |
| Per-image confidence | Stable one-decimal numeric badge for completed targets | unit format test and Chrome E2E |
| Fixed 65% threshold | Single inclusive policy and frozen intercept | boundary tests at 0.649999 / 0.65 / 0.650001 |
| Reproducible build/install | Lockfile, model lock, fixed UTC archive timestamps, documented commands | `npm ci && npm run verify`; byte-identical New York/UTC archive check |
| Hostile-page safety | Closed/recoverable labels; hard target/queue/body/pixel limits; no privileged URL fetch | policy tests and both Chrome E2E paths |
| ≥75% balanced accuracy | 93.83% original sealed holdout; lower 95% CI 91.83% | committed manifests/predictions, `npm run check:benchmark`, and exact reevaluation commands |

## Revision conditions

Rework is required for any model/hash mismatch, score below 75%, class-recall collapse, network activity after cutoff, fabricated score on failure, missing fallback runtime, stale-result race, inaccessible install instructions, or material independent-review finding. Passing project checks is evidence, not proof of the maintainer’s private evaluation.
