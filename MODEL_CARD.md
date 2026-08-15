# ProofLens model card

## Model

ProofLens packages one FP32 ONNX ViT-S/16 image classifier with a 384-dimensional hidden representation and one synthetic-image logit. Inference happens inside the Chrome extension. There is no remote model call, local server, second detector, metadata heuristic, or hash lookup.

The frozen backbone comes from the corrected CommunityForensics DeepfakeDet-ViT revision `ac6ee457bea904a373065754107451793b56db00`. Its exact upstream ONNX SHA-256 is `a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1`; the model license is MIT. The Community Forensics training reported 2.7 million generated images from 4,803 generators paired with 2.7 million real images, or 5.4 million examples total.

ProofLens freezes that backbone and trains only `classifier.weight [1,384]` and `classifier.bias [1]`.

<!-- PROOFLENS_CURRENT_M2_START -->
## Current M2 head-training data

The shipped artifact is 87,442,080 bytes with SHA-256 `a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47`. Independent ONNX comparison found changes in exactly those two classifier initializers; the other 198 initializers, graph nodes, inputs, outputs, and opsets match the pinned upstream structure.

The M2 head was trained on 105,978 unique public images and 123,912 feature views:

- 50,000 DiffusionDB synthetic images, revision `fb620fbe49fa4420e0734bd9c0df11f51176b61f`, CC0-1.0;
- 1,200 Qwen Image Bench synthetic images across 15 current generator versions, revision `d2493deb153b020cf169c7e3f57d15e4dd697038`, Apache-2.0;
- 50,000 Open Images V7 train images, CC BY 2.0;
- 1,200 Open Images V7 modern-split images, CC BY 2.0;
- 1,200 DOCCI train images, revision `a0a43eaf34676ffd008fb6565dd8c2ba00d09100`, CC BY 4.0;
- 2,378 StockImages-CC0 non-AI photographs, revision `206f3575579f1187548c6f47042ae9174c0a51fc`, source card reports CC0-1.0.

The two 50,000-image strata use one original view. Each modern and StockImages item uses original, screenshot, JPEG-75 resize, and heavy double-JPEG views. Source-balanced loss gives each class half the objective and equal weight to each named source within that class.

The source freeze rejects exact ID, byte, and source-group overlap with historical and consumed v1/v2 data. Every new StockImages dHash near match at Hamming distance 8 or lower was rejected. The retained M1 base corpus carries forward 100 pre-existing training/evaluation dHash pairs previously reviewed as visually distinct; no pair remains unreviewed. Training freshly extracted every selected view under one completed 54-shard run. Validation selected weight decay `0.1` and upstream-head blend `0.85`; 24 of 25 candidate configurations passed all frozen gates. `npm run check:m2-training-evidence` binds the recipe, selection evidence, training summary, grid, calibration, model bytes, and classifier-only comparison. See `benchmark/m2/README.md`.
<!-- PROOFLENS_CURRENT_M2_END -->

<!-- PROOFLENS_HISTORICAL_M1_START -->
## Historical M1 and evaluation evidence

The prior M1 head used model SHA-256 `941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c`, 103,600 training images, and 114,400 feature views. Its 25 candidate configurations were valid. That training packet remains immutable under `benchmark/evidence/large/`.

The original v1 and replacement-v2 are consumed and acceptance-ineligible. V1 failed its numeric contract before bootstrap; replacement-v2 passed accuracy gates but failed its paired StockImages false-positive gate. Their immutable records use `acceptanceEligible: false`. Neither result is evidence for the current M2 model.
<!-- PROOFLENS_HISTORICAL_M1_END -->

## Input contract

Apply EXIF orientation and convert to RGB. Preserve aspect ratio while resizing the shortest edge to 440 pixels with bicubic resampling, take the 384×384 center crop, scale RGB to `[0,1]`, normalize with mean `[0.48145466, 0.4578275, 0.40821073]` and standard deviation `[0.26862954, 0.26130258, 0.27577711]`, then transpose to NCHW float32. The ONNX interface is `pixel_values [N,3,384,384] → logits [N,1]`.

## Output and calibration

Validation selects a single global raw-logit boundary over every distinct decision partition. The extension adds one frozen intercept so that boundary displays as **65.0/100**. Scores at least 65.0 are labeled `Likely AI`; lower scores are `Below flag threshold`.

This alignment is not probability calibration. A score of 65/100 does not mean a 65% chance that an image is synthetic. No site, source, file type, generator, or transformation receives a different threshold.

## Evaluation boundary

Development validation contains 300 Open Images photographs, 300 StockImages-CC0 photographs, and 300 synthetic images from GLM-Image and HunyuanImage-3.0. It is the only split used for M2 candidate and threshold selection.

| Validation view | Balanced accuracy | Non-AI recall | Synthetic recall | Worst synthetic-family recall | StockImages recall |
|---|---:|---:|---:|---:|---:|
| Original | 94.08% | 95.83% | 92.33% | 85.33% | 96.00% |
| Screenshot | 95.83% | 99.67% | 92.00% | 84.67% | 100.00% |
| JPEG 75 | 93.67% | 94.33% | 93.00% | 88.00% | 94.00% |
| Heavy double-JPEG | 94.25% | 97.50% | 91.00% | 84.00% | 98.00% |

These figures come from the canonical validation evaluator and are selection results, not an untouched estimate of generalization.

The consumed v1 and replacement-v2 results are historical M1 evidence, summarized in the bounded section above. Dataset cards at the pinned replacement revisions report CC BY 4.0 and CC0-1.0. Those are source-reported license statements, not independent rights clearance; `benchmark/manifests/replacement-v2-attribution.json` preserves authors, links, and the rights caveat.

No current acceptance result exists. See `BENCHMARK.md` for the immutable v1 failure, replacement hashes, and public authorization boundary.

## Intended use

ProofLens is a local screening hint for ordinary webpage images. It can help a person decide what deserves closer inspection. It does not prove origin or authenticity, identify a generator or author, or justify moderation, employment, legal, financial, medical, or safety decisions without independent human evidence.

## Known limitations

- New generators and transformations can shift performance.
- DiffusionDB adds scale but overrepresents an older Stable Diffusion era.
- Open Images, DOCCI, Library of Congress, and StockImages photographs do not span every real-web visual type.
- Illustrations, CGI, memes, charts, scans, UI screenshots, heavily edited photos, and unusual camera pipelines can produce false positives.
- A fixed center crop can discard useful evidence.
- Browser canvas resampling is not bit-identical to Pillow; the post-score parity test measures this gap.
- `Unavailable` means the pixels could not be acquired or decoded. It is not a low AI score.
