# Public submission audit

Snapshot: 2026-08-13 22:49 UTC. The [bounty page](https://poidh.xyz/arbitrum/bounty/323) was open, not canceled, not in voting, and had no accepted claim when this audit was frozen. The page listed 13 active claims. Repository heads and claim text can change; the listed scores are each project’s public claims, not a shared independent benchmark.

| Claim | Inspected repository/head | Public evidence | Material evaluation gap |
|---:|---|---|---|
| 1028 | [pixilated730/local-ai-image-detector](https://github.com/pixilated730/local-ai-image-detector) `dddb57b` | Bundled 87 MB model; broad browser target coverage; reports 89.2% pooled balanced accuracy across 2,350 images and 19 current generator families | The shipped `+2.29` calibration offset was selected using the same real and synthetic holdout pools used for the reported score; evaluation rows and executable Chrome evidence are not committed, and there is no CI |
| 1027 | [Phineas1500/sieve-ai-image-detector](https://github.com/Phineas1500/sieve-ai-image-detector) `0ec11e5` | 43.8 MB model release; reports 91.3% clean, 87.3% web, and 84.6% hard balanced accuracy over 36,384 images | Scores exist only as README prose, without frozen manifests, predictions, or result artifacts; metadata byte strings can force a 99% score without authenticated provenance; no CI or release gate |
| 1026 | [ebreen/pixelwitness](https://github.com/ebreen/pixelwitness) `89e45d7` | Bundled three-model WASM ensemble; reports 82.5% balanced accuracy on an 80-image public proxy | Small holdout with no committed prediction rows; no reproducible release archive, CI, or fixed-head browser report; project documentation leaves the all-MIT eligibility of one Apache-2.0 model unresolved |
| 1025 | [maddiedreese/blur](https://github.com/maddiedreese/blur) `49b411d` | Reports 95.6% balanced accuracy, 98.8% real recall, and 92.4% AI recall across 1,052 frozen full-resolution bases | Its own frozen cross-view gate rejects release: JPEG-75 thumbnail AI recall is 39.7% and conservative cross-view balanced accuracy is 66.2%; a clean-clone `npm run verify` cannot build because the exported ONNX model is absent; raw evaluation rows and product CI are absent |
| 1023 | [dusy4/local-ai-image-detector-bounty](https://github.com/dusy4/local-ai-image-detector-bounty) `8d82a74` | Three-model hybrid; reports 81.3% OpenFake, 94.6% AIGIBench, 84.4% transformed | Cross-dataset scores are not one sealed evaluation; roughly 187 MB of learned artifacts; no observed GitHub CI run |
| 1022 | [anudit/aidetect](https://github.com/anudit/aidetect) `301adbc` | Hand-written WebGPU Community Forensics ViT; reports 89.13% | WebGPU-only path, no lockfile/CI/repository benchmark report, and no CPU/WASM fallback proof |
| 1021 | [RajeshRk18/ai-image-detector](https://github.com/RajeshRk18/ai-image-detector) `67585ea` | Two ONNX models; reports 92.6% fit and 91.2% leave-one-source-out CV | Cross-validation is not an untouched holdout; build-time model downloads depend on upstream availability |
| 1020 | [Dyno-man/Dino-ImageGen-Ext](https://github.com/Dyno-man/Dino-ImageGen-Ext) `ef986ac` | 23 MB Q8 model, green CI; reports 94.6% sealed original and 91.4% web-degraded | Strong public evidence, but final evaluation leans heavily on older Tiny-GenImage generators, so its score is not directly comparable to current-family tests |
| 1019 | [choir94/RealGuard-ai-image-detector](https://github.com/choir94/RealGuard-ai-image-detector) `cb8b987` | Community Forensics ViT, 27 tests, green CI; reports 88.33% | Calibration/evaluation contains 110 images; one-time network install needs a clean offline-restart proof |
| 1018 | [thedudeb/synthcheck](https://github.com/thedudeb/synthcheck) `5883d1c` | 23.4 MB Q8 modern re-head; reports 92.33% original, 94.0% screenshot, 90.0% JPEG-75, 87.33% heavy | Strongest comparable modern protocol; no observed GitHub CI, and validation/test share generator families |
| 1016 | [the-gadget-lab/Six-Fingers](https://github.com/the-gadget-lab/Six-Fingers) `5737d89` | Fine-tuned int8 Community Forensics model, green CI; reports 85.3% extension E2E | Model is downloaded rather than stored in the repository; reported holdout is a subsample |
| 1015 | Intended repository appears to be [CaravelaLabs/local-ai-detector](https://github.com/CaravelaLabs/local-ai-detector) `9a49375` | Six-crop TTA; reports 93.8% | Claim contains no parseable repository URL; score is based on 42 images; no observed CI |
| 1014 | [takhir-iota/locallens-ai-detector](https://github.com/takhir-iota/locallens-ai-detector) `2e11ac8` | ViT-B/16, green CI; reports 83.33% | Evaluation contains 31 images; bounty comments report false positives on real photos |

## Resulting design bar

ProofLens adopts the strongest common requirements and closes the most consequential public gaps:

- exact checked-in FP32 artifact and model lock, rather than a fragile runtime download;
- corrected 384-pixel Community Forensics preprocessing and current generator families;
- fixed displayed 65% cutoff chosen before a one-time, sample-disjoint test;
- separate real and synthetic recall, per-family results, transformations, and bootstrap intervals;
- fresh-profile WebGPU and forced-WASM runs after model preparation, browser restart, fixture-server shutdown, and offline transition;
- automatic `img`, responsive, dynamic, and CSS-background coverage;
- zero page-controlled extension fetches, local rendered-pixel acquisition, content-hash caching, model revalidation, and stale-result guards;
- strict TypeScript, unit/policy checks, deterministic packaging, and GitHub CI.

For the four newest claims, repository-native checks were run after their scripts were inspected. Blur passed typecheck, lint, and 44 unit tests but failed its required clean-clone build because the exported model was absent. Sieve passed six metadata tests and its extension build. PixelWitness and Local AI Image Detector had no installed dependency tree or CI; their JavaScript and Python sources passed syntax parsing, and their benchmark paths were reviewed directly. These checks establish implementation state, not private-benchmark performance.

Different public datasets prevent a fair numerical ranking across every claim. This audit therefore uses competitor evidence to set engineering and evaluation gates; it does not declare another repository incorrect or claim that a public score predicts the maintainer’s private benchmark.
