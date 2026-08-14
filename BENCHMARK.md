# Benchmark protocol and evidence

This file freezes the evaluation design before the confirmatory images are scored. The shipped model, validation-selected threshold, preprocessing, corpus, acceptance gates, runtime, and output paths become immutable before the confirmatory run. Any post-score runtime defect requires a new untouched holdout and a new public freeze before source changes; the observed confirmatory set may never become a tuning set.

## Model-selection boundary

The Community Forensics ViT-S/16 backbone is frozen. Training changes only its `384 → 1` classifier weight and bias. The upstream backbone was trained on 5.4 million total images arranged as 2.7 million real/synthetic pairs and spanning 4,803 generators. ProofLens trains the replacement head on 103,600 unique public images:

| Training source | Class | Images | Views used for head training |
|---|---|---:|---:|
| DiffusionDB, pinned Stable Diffusion corpus | synthetic | 50,000 | 50,000 |
| Qwen Image Bench, 15 current generator versions | synthetic | 1,200 | 4,800 |
| Open Images V7 train | non-AI | 50,000 | 50,000 |
| Open Images V7 modern split | non-AI | 1,200 | 4,800 |
| DOCCI train | non-AI | 1,200 | 4,800 |
| **Total** | **51,200 synthetic / 52,400 non-AI** | **103,600** | **114,400** |

The four modern views are original, social screenshot, JPEG-75 resize, and heavy double-JPEG. Source-balanced loss gives each class half of the objective and each named source an equal share within its class. A 25-candidate grid varies only weight decay and upstream-head blending. Validation selects the lexicographically best worst-case result and searches every distinct logit decision boundary; the test is never accepted by the trainer.

The pixel-free corpus packet is under `benchmark/evidence/large/`. It pins 85 DiffusionDB archives, 55,000 Open Images candidates, all selected IDs and byte hashes, 50,000 Open Images attributions, and EXIF-oriented dHash evidence. Training has zero ID or byte overlap with validation, test, or the 19-image Chartography exclusion. Every dHash pair at Hamming distance 8 or lower is either rejected or recorded as visually distinct.

Fresh extraction covered all 103,600 images and 114,400 configured feature views. All 25 candidates passed the validation gates; the selected head uses weight decay `0.1` and upstream blend `0.85`. The finalized ONNX SHA-256 is `941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c`. Independent comparison against the pinned upstream bytes found only `classifier.weight` and `classifier.bias` changed; 198 other initializers and the graph contract are unchanged.

## Frozen splits

| Split | Synthetic | Non-AI | Purpose | SHA-256 |
|---|---|---|---|---|
| Training manifest | 1,200 current-generator images | 2,400 Open Images/DOCCI images | modern stratum included in the 103,600-image head corpus | `03b88b3804244018fbdf532b2b7d451db91dad7c3229c9013ee4ede9fa798015` |
| Validation | 150 GLM-Image + 150 HunyuanImage-3.0 | 300 Open Images | candidate and threshold selection only | `41be10ef876ecef0635744ed29677a1888a7759cc8060dc7a392f76f83ab263b` |
| Confirmatory test | 300 Kling v2.1 | 300 Library of Congress FSA/OWI photographs | one-time final accuracy estimate | `28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae` |
| Web-negative challenge | none | the same 300 Library of Congress rows + 19 expert-created Chartography rows | false-positive stress check | `ad8b3f30a37feb3b6b046683db2d4071e236e6878612c7d8733869699d7f7824` |

Generator versions, real sources, synthetic prompt groups, IDs, and image bytes are disjoint across training, validation, and confirmatory test. The web-negative challenge deliberately reuses all 300 confirmatory real rows. It is therefore a targeted extension of the real-side analysis, not an independent 319-image estimate.

## Frozen decision and acceptance rules

Validation emits one intercept that aligns its selected raw-logit threshold to the extension’s fixed display threshold of **65.0/100**. The inclusive decision is `display score >= 65.0`. This is a model score, not a calibrated probability.

All evaluation uses the exact shipped FP32 ONNX and all four predeclared views. The confirmatory test passes only if every view satisfies:

- lower 95% class-stratified bootstrap bound for balanced accuracy at least 75%;
- lower 95% bound for non-AI recall at least 75%;
- lower 95% bound for synthetic recall at least 75%;
- Kling v2.1 point recall at least 60%.

The bootstrap uses 20,000 one-image-cluster replicates, seed `20260813`, and a variant-and-class-derived RNG. Verification recomputes every interval from the committed predictions.

The web-negative challenge passes only if every view has a Wilson 95% upper bound on false-positive rate of at most 10% overall and at most 20% for each named source.

The exact gates and post-score policy are machine-readable in `benchmark/large/recipe.json`. `benchmark/evaluate.py` validates the model, manifest, calibration, source/class allocation, every image hash, provider, output absence, and all four variants before creating an inference session. It stages all files and publishes a completion marker last. Existing canonical evidence cannot be overwritten.

## Predeclared execution order

1. Complete head training and validation selection.
2. Run `benchmark/finalize_training_evidence.py`; independently prove that only `classifier.weight` and `classifier.bias` changed.
3. Evaluate validation and select two high-margin Chrome UI fixtures from validation only.
4. Commit and push the clean model, calibration, corpus, evaluator, gates, runtime, and validation evidence to public `main` as source commit A. Its stage-aware `npm run verify:static` run must pass with every confirmatory, web-negative, replay, and browser-parity artifact absent.
5. Run `benchmark/write_pre_score_freeze.py`, commit **only** its receipt as commit B, and push B. Its pre-score CI run must pass before inference. Require public `origin/main` to equal B, require B to be the sole child step from A, and require the receipt bytes to remain identical to the blob first committed in B.
6. Run the confirmatory protocol once, then its deterministic bootstrap.
7. Run the web-negative protocol once, then its Wilson intervals.
8. Run `npm run verify:release` to replay the immutable evaluator and require byte-identical outputs. This is a reproducibility check, never a tuning step.
9. Run browser parity and clean-profile WebGPU/WASM E2E without changing the frozen model contract.

`benchmark/verify_evaluation_evidence.py` performs the local pixel-and-ONNX replay before corpus cleanup. Its committed receipt records exact commands and file hashes for reproducibility; it is not a cryptographic attestation that those commands ran. `npm run verify:static` is stage-aware: at A and B it requires all post-score outputs to be absent, and on every descendant of B it requires the complete final pixel-free packet. `npm run verify:release` additionally observes the full local replay and fails when the frozen pixels are absent. Public source commit A anchors every model-behavior and protocol file; freeze-only commit B publicly anchors the exact receipt before confirmatory inference. The final checker requires both commits, immutable receipt bytes, a clean worktree, and only evaluation evidence, result documentation, and browser artifacts after A.

The canonical evaluator shape is:

```bash
benchmark/.venv/bin/python benchmark/evaluate.py \
  --model weights/prooflens-cf384.onnx \
  --expected-model-sha256 941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c \
  --data-root benchmark/data \
  --manifest benchmark/manifests/test.jsonl \
  --expected-manifest-sha256 28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae \
  --output-dir benchmark/evidence/evaluation/confirmatory \
  --protocol confirmatory \
  --batch-size 16 \
  --execution-provider cpu \
  --calibration benchmark/evidence/large/calibration.json \
  --expected-calibration-sha256 607ec2d8a4428f97cd51ae020f3168bf451201a19b117372033d7becd5a5559c
```

Validation uses `--protocol validation`, its frozen manifest and `benchmark/data/modern-head`; web-negative uses `--protocol web-negative`, its frozen manifest and `benchmark/data/web-negative`.

## Results status

Head training, finalization, and canonical validation are complete. Validation balanced accuracy is 95.00% on originals, 96.33% on screenshots, and 94.50% on both JPEG stress views. The weakest class recall across those views is 91.67%; the weakest named synthetic-family recall is 84.00%. These are validation-selected results, not the final estimate.

No confirmatory or web-negative score is reported in this pre-score protocol revision. Their canonical output directories are intentionally absent. Results will be added only after public source commit A and freeze-only commit B pass their gates.

Public results cannot establish the bounty maintainer’s private benchmark score. They establish a reproducible local evidence bar and expose class-specific failure instead of hiding it inside a single average.

## Browser evidence

The post-score browser diagnostic uses a deterministic 30-real/30-synthetic confirmatory subset selected before scoring. Its pixels and reference probabilities are materialized only after the canonical confirmatory completion and bootstrap exist; `artifacts/browser-parity.json` is forbidden from both public pre-score commits. The diagnostic must run the packaged extension in a clean profile while offline, agree with the frozen reference decisions at least 95%, achieve at least 75% balanced accuracy on that diagnostic subset, and stay within the frozen probability-difference limits. It is diagnostic evidence only: any runtime defect blocks this release and requires a new untouched holdout plus a new public freeze before source changes.

The full Chrome E2E separately proves model persistence across restart, determinate setup progress, no post-cutoff HTTP(S) requests, WebGPU and forced-WASM inference, actual high/low model states from validation fixtures, confirmed saved and temporary control behavior (including injected delivery failures), a completed re-scan, explicit unavailable output, responsive/CSS targets, stale-result rejection, hostile-page bounds, and target-associated narrow/1.5×-scale label geometry. Exterior labels cannot cross another tracked target or drift away from their declared side; a chip with no honest placement is not rendered. GitHub Actions gates the portable forced-WASM run; WebGPU is a fixed-head local platform gate whose source-, model-, and archive-bound receipt is committed and checked statically because hosted-runner GPU availability is not stable. Automation exercises the production popup document in an extension-page tab; it does not claim to test Chrome’s toolbar-popup window lifecycle.

## Reproduction limits

Dataset pixels are excluded from Git. Public IDs, revisions, byte hashes, licenses, attribution, selection code, predictions, and aggregate evidence are committed. Full reconstruction requires roughly 106 GB for the large corpus plus the source archives. DiffusionDB adds breadth but is dominated by an older Stable Diffusion generation era; the smaller current-generator stratum does not represent every future model. Library of Congress photographs are an unusually well-proven real source, not a complete sample of today’s web. The 19 Chartography rows broaden false-positive coverage but are too small to support a standalone population claim.
