# Benchmark protocol and evidence

## Evaluation status

The original Kling v2.1/Library of Congress holdout was scored once and is consumed. Its predictions failed the frozen numeric contract before bootstrap: the evaluator applied `np.exp` to float32 logits, while the independent contract recomputed binary64 sigmoid. The immutable failure record reports 2,231 violations across 2,400 rows and marks the packet `acceptanceEligible: false`. No v1 bootstrap was published and v1 web-negative inference never started. Those point estimates are diagnostic only and may not guide the model, threshold, preprocessing, gates, or replacement selection.

The score-blind replacement packet under `benchmark/recovery_v3/` was scored once from the public V3 freeze and is consumed. It uses 300 Coxy7 Infinity images and 300 KoalaAI StockImages-CC0 photographs for confirmation, plus 319 different StockImages rows for false-positive testing. Confirmation passed every frozen accuracy gate. The StockImages slice failed the 10% overall Wilson upper-bound limit on original images (`12.39%`) and JPEG-75 images (`18.01%`). Its complete packet is preserved with `acceptanceEligible: false` and may inform development only.

The A4 repair converts logits to binary64 before sigmoid, gives the replacement protocols new canonical names and output paths, and strengthens the publication contract. It does not change model bytes, calibration, threshold, preprocessing, statistical gates, training data, or extension runtime. Replacement inference remains unauthorized until A4 and its receipt-only V3 child are both public and green.

## Model-selection boundary

The Community Forensics ViT-S/16 backbone is frozen. Training changes only its `384 → 1` classifier weight and bias. The upstream backbone was trained on 5.4 million paired real/synthetic images spanning 4,803 generators. ProofLens trained the replacement head on 103,600 unique public images:

| Training source | Class | Images | Feature views |
|---|---|---:|---:|
| DiffusionDB, pinned Stable Diffusion corpus | synthetic | 50,000 | 50,000 |
| Qwen Image Bench, 15 current generator versions | synthetic | 1,200 | 4,800 |
| Open Images V7 train | non-AI | 50,000 | 50,000 |
| Open Images V7 modern split | non-AI | 1,200 | 4,800 |
| DOCCI train | non-AI | 1,200 | 4,800 |
| **Total** | **51,200 synthetic / 52,400 non-AI** | **103,600** | **114,400** |

The modern images use original, screenshot, JPEG-75 resize, and heavy double-JPEG views. Source-balanced loss gives each class half of the objective and each named source an equal share inside its class. A 25-candidate grid varies only weight decay and upstream-head blending. Validation selects the lexicographically best worst-case result and searches every distinct logit decision partition; training code cannot read a confirmatory manifest.

Fresh extraction covered all 103,600 images and 114,400 configured views. All 25 candidates passed the validation gates; the selected head uses weight decay `0.1` and upstream blend `0.85`. The finalized ONNX SHA-256 is `941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c`. Independent comparison against the pinned upstream artifact found only `classifier.weight` and `classifier.bias` changed; 198 other initializers and the graph contract match.

The pixel-free training packet under `benchmark/evidence/large/` pins source locks, 103,600 IDs and byte hashes, 50,000 Open Images attribution records, and EXIF-oriented dHash review. Training has zero ID or byte overlap with the frozen evaluation exclusions. Every training/evaluation dHash pair at Hamming distance 8 or lower was rejected or reviewed as visually distinct.

## Frozen splits

| Split | Synthetic | Non-AI | Role | SHA-256 |
|---|---|---|---|---|
| Training manifest | 1,200 current-generator images | 2,400 Open Images/DOCCI images | modern stratum inside the 103,600-image corpus | `03b88b3804244018fbdf532b2b7d451db91dad7c3229c9013ee4ede9fa798015` |
| Validation | 150 GLM-Image + 150 HunyuanImage-3.0 | 300 Open Images | candidate and threshold selection only | `41be10ef876ecef0635744ed29677a1888a7759cc8060dc7a392f76f83ab263b` |
| Historical consumed v1 confirmation | 300 Kling v2.1 | 300 Library of Congress FSA/OWI photographs | failed numeric-contract diagnostic; never acceptance evidence | `28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae` |
| Historical unrun v1 web-negative | none | 300 Library of Congress + 19 Chartography | frozen but never inferred | `ad8b3f30a37feb3b6b046683db2d4071e236e6878612c7d8733869699d7f7824` |
| Replacement-v2 confirmation | 300 Coxy7 Infinity | 300 KoalaAI StockImages-CC0 | consumed; accuracy gates passed, paired release packet failed | `773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8` |
| Replacement-v2 web-negative | none | 319 different StockImages-CC0 rows | consumed; overall false-positive gate failed | `6a1287bae6826811c81cbebab79a1bc6abb475fde70c9aa1529c390ed97014c9` |

The replacement selection excludes all 106,019 historical training/evaluation IDs, bytes, source groups, and dHashes. It also rejects cross-protocol overlaps. The tracked packet contains 248 deterministic rejects and no retained dHash match at Hamming distance 8 or lower. Fixed dataset cards report CC BY 4.0 for Coxy7 and CC0-1.0 for StockImages; these are source-reported license statements, not independent rights clearance. Pixels remain outside Git.

## Decision and acceptance rules

Validation emits one intercept that aligns its selected raw-logit boundary to the fixed display threshold of **65.0/100**. The inclusive decision is `display score >= 65.0`. The display value is a model score, not a calibrated probability.

All release evaluation uses the shipped FP32 ONNX and four views: original, screenshot, JPEG-75 resize, and heavy double-JPEG. Replacement confirmation passes only if every view has:

- lower 95% class-stratified bootstrap bound for balanced accuracy at least 75%;
- lower 95% bound for non-AI recall at least 75%;
- lower 95% bound for synthetic recall at least 75%;
- Infinity point recall at least 60%.

The bootstrap uses 20,000 one-image-cluster replicates, seed `20260813`, and variant/class-derived random streams. Static verification reruns the interval calculation from the committed prediction rows.

Replacement web-negative passes only if every view has a Wilson 95% upper false-positive bound of at most 10% overall and at most 20% for each named source. These gate values remain byte-bound to `benchmark/large/recipe.json`; the replacement packet changes the held-out population, not the acceptance bar.

## Append-only public lineage

| Stage | Commit | Public result | Meaning |
|---|---|---|---|
| A | `0771a9422b552e2023e5150fb6c8b4238b811a74` | [passed](https://github.com/baney75/prooflens/actions/runs/31843811845) | original source freeze |
| B | `2bd0c4757f6059c57879414a5dba77629d66460e` | [failed](https://github.com/baney75/prooflens/actions/runs/31844088383) | receipt-only; Node verifier hit `ENOBUFS` before inference |
| A2 | `99861df575854511c685d7b8f90acdc7ed4e5923` | [failed](https://github.com/baney75/prooflens/actions/runs/31846361076) | copied evaluator imported absent inference dependencies before its guard |
| A3 | `17124df0bf390c2c2c27583ae81f06b65ead2e3f` | [passed](https://github.com/baney75/prooflens/actions/runs/31847256279) | dependency loading moved behind authorization |
| B2 | `2757a4ff267d580a7dd8ad4918885441fa887f1b` | [passed](https://github.com/baney75/prooflens/actions/runs/31847694896) | V2 receipt-only public boundary |
| F1 | `45400803a19b967c8cae0bbf4817fe984aea349a` | [failed](https://github.com/baney75/prooflens/actions/runs/31848762781) | consumed v1 packet disclosed as numeric-contract failure |
| P3 | `baaf3eb0b7a22f635d2ec6a3cb2496b9e76313b8` | [failed](https://github.com/baney75/prooflens/actions/runs/31853385690) | score-blind replacement packet; old stage policy correctly rejected its new paths |
| A4 | `1323c10a151bdd0b96640962b447607371607b90` | [passed](https://github.com/baney75/prooflens/actions/runs/31855698629) | binary64 evaluator and replacement-v2 protocol freeze |
| B3 | `4fdf6b7dcc53371f00aff5f6b449f4299c2988cb` | [passed](https://github.com/baney75/prooflens/actions/runs/31855954386) | receipt-only public replacement-v2 pre-score boundary |

A4 is P3’s direct child and changes exactly the 23 numeric/protocol/test/documentation paths declared in `benchmark/evaluation_contract.py`. B3 is A4’s receipt-only direct child. The evaluator independently verified both exact public `quality` runs before importing ONNX Runtime or Pillow or reading model/pixel inputs.

Git history and public CI establish the recorded boundary. They cannot prove that nobody viewed or processed pixels outside this repository workflow.

## Canonical replacement execution

After A4 and V3 are public and green, run replacement confirmation exactly once:

```bash
benchmark/.venv/bin/python benchmark/evaluate.py \
  --model weights/prooflens-cf384.onnx \
  --expected-model-sha256 941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c \
  --data-root benchmark/data/replacement-v2 \
  --manifest benchmark/manifests/test-v2.jsonl \
  --expected-manifest-sha256 773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8 \
  --output-dir benchmark/evidence/evaluation/confirmatory-v2 \
  --protocol confirmatory-v2 \
  --batch-size 16 \
  --execution-provider cpu \
  --calibration benchmark/evidence/large/calibration.json \
  --expected-calibration-sha256 607ec2d8a4428f97cd51ae020f3168bf451201a19b117372033d7becd5a5559c
```

Then run `benchmark/bootstrap_ci.py` with all four canonical prediction files. Run `web-negative-v2` once against `benchmark/manifests/web-negative-v2.jsonl`, then `benchmark/bootstrap_fpr.py`. Existing or partial canonical output fails before dependency or input loading.

`npm run verify:release` replays only replacement-v2 confirmation, web-negative, bootstrap, and Wilson evidence byte-for-byte. It does not replay legacy validation or the consumed v1 packet because those stored probabilities preserve the disclosed float32 representation. The replay receipt records exact commands and hashes; it is a reproducibility record, not a cryptographic attestation that commands ran.

## Current results

Head training, finalization, and validation are complete. Validation balanced accuracy is 95.00% on originals, 96.33% on screenshots, and 94.50% on both JPEG stress views. The weakest class recall is 91.67%; the weakest named synthetic-family recall is 84.00%. These are selection results.

There is no current acceptance result. V1 is consumed and acceptance-ineligible. Replacement-v2 passed its accuracy gates but failed the paired false-positive gate, so it is also consumed and acceptance-ineligible. Public repository evidence does not establish the bounty maintainer’s private score, acceptance decision, or payment.

## Browser evidence

`benchmark/manifests/parity-ids-v2.json` fixes a prediction-blind 30-real/30-Infinity subset before replacement scoring. After the canonical completion and bootstrap exist, `prepare_parity.py` materializes those exact local pixels and reference scores. The packaged extension must run them in a clean offline profile with at least 95% decision agreement, at least 75% diagnostic balanced accuracy, and the frozen probability-difference limits.

The full Chrome E2E separately proves setup progress, model persistence after restart, zero post-cutoff HTTP(S) requests, WebGPU and forced-WASM inference, actual high/low validation fixtures, honest saved/temporary control failures, completed re-scan work, unavailable output, responsive/CSS targets, stale-result rejection, hostile-page bounds, and target-associated geometry at a narrow 1.5× scale. GitHub Actions gates forced WASM. WebGPU is a fixed-head local gate because hosted-runner GPU availability is not stable. Automation opens the production popup document in an extension-page tab; it does not claim Chrome toolbar-window lifecycle coverage.

## Reproduction limits

Dataset pixels are excluded from Git. Public IDs, revisions, byte hashes, source-reported licenses, attribution, selection code, predictions, and aggregate evidence are committed. Full reconstruction requires about 106 GB plus source archives. DiffusionDB supplies scale but overrepresents an older Stable Diffusion era. Validation covers two modern generator families, and replacement confirmation covers one unseen family; neither represents every future generator. StockImages is one real-photo corpus, not the full ordinary web. Illustrations, CGI, charts, memes, screenshots, scans, and edited photographs remain important false-positive risks.
