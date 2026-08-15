# ProofLens model card

## Model

ProofLens packages one FP32 ONNX ViT-S/16 image classifier with a 384-dimensional hidden representation and one synthetic-image logit. Inference happens inside the Chrome extension. There is no remote model call, local server, second detector, metadata heuristic, or hash lookup.

The frozen backbone comes from the corrected CommunityForensics DeepfakeDet-ViT revision `ac6ee457bea904a373065754107451793b56db00`. Its exact upstream ONNX SHA-256 is `a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1`; the model license is MIT. The Community Forensics training reported 2.7 million generated images from 4,803 generators paired with 2.7 million real images, or 5.4 million examples total.

ProofLens freezes that backbone and trains only `classifier.weight [1,384]` and `classifier.bias [1]`. The shipped artifact is 87,442,080 bytes with SHA-256 `941e3914c075a735db5795e897b71c1d8b2f6b7c2cf2cb7777d0a6999aa02e6c`. Independent ONNX comparison found changes in exactly those two initializers; the other 198 initializer digests, graph nodes, inputs, outputs, and opsets match the pinned upstream structure.

## ProofLens head-training data

The replacement head was trained on 103,600 unique public images and 114,400 feature views:

- 50,000 DiffusionDB synthetic images, revision `fb620fbe49fa4420e0734bd9c0df11f51176b61f`, CC0-1.0;
- 1,200 Qwen Image Bench synthetic images across 15 current generator versions, revision `d2493deb153b020cf169c7e3f57d15e4dd697038`, Apache-2.0;
- 50,000 Open Images V7 train images, CC BY 2.0;
- 1,200 Open Images V7 modern-split images, CC BY 2.0;
- 1,200 DOCCI train images, revision `a0a43eaf34676ffd008fb6565dd8c2ba00d09100`, CC BY 4.0.

The two 50,000-image strata are configured for one original view. Each modern image is configured for original, screenshot, JPEG-resize, and heavy double-JPEG views. Source-balanced loss gives each class half the objective and equal weight to each named source within that class.

The pixel-free corpus packet preserves source revisions, selection rules, 103,600 IDs and byte hashes, 50,000 Open Images attribution records, exact train/evaluation exclusions, and reviewed perceptual-near-match evidence without redistributing pixels. Training freshly extracted and covered all 103,600 images and all 114,400 configured views. All 25 candidate configurations were valid; validation selected weight decay `0.1` and upstream-head blend `0.85`. The training summary, candidate grid, calibration, and classifier-only comparison are checked by `npm run check:large-training-evidence`. See `benchmark/large/DATASET_CARD.md`.

## Input contract

Apply EXIF orientation and convert to RGB. Preserve aspect ratio while resizing the shortest edge to 440 pixels with bicubic resampling, take the 384×384 center crop, scale RGB to `[0,1]`, normalize with mean `[0.48145466, 0.4578275, 0.40821073]` and standard deviation `[0.26862954, 0.26130258, 0.27577711]`, then transpose to NCHW float32. The ONNX interface is `pixel_values [N,3,384,384] → logits [N,1]`.

## Output and calibration

Validation selects a single global raw-logit boundary over every distinct decision partition. The extension adds one frozen intercept so that boundary displays as **65.0/100**. Scores at least 65.0 are labeled `Likely AI`; lower scores are `Below flag threshold`.

This alignment is not probability calibration. A score of 65/100 does not mean a 65% chance that an image is synthetic. No site, source, file type, generator, or transformation receives a different threshold.

## Evaluation boundary

Validation contains 300 Open Images photographs and 300 synthetic images from GLM-Image and HunyuanImage-3.0. It is the only split used for candidate and threshold selection.

| Validation view | Balanced accuracy | Non-AI recall | Synthetic recall | Worst synthetic-family recall |
|---|---:|---:|---:|---:|
| Original | 95.00% | 95.67% | 94.33% | 88.67% |
| Screenshot | 96.33% | 99.33% | 93.33% | 86.67% |
| JPEG 75 | 94.50% | 94.67% | 94.33% | 88.67% |
| Heavy double-JPEG | 94.50% | 97.33% | 91.67% | 84.00% |

These figures come from the canonical validation evaluator and are selection results, not an untouched estimate of generalization.

The original Kling v2.1/Library of Congress confirmatory set was scored once and is consumed. Its stored predictions failed the binary64 numeric contract before bootstrap and are permanently marked `acceptanceEligible: false`; they are not release evidence.

The unscored score-blind replacement contains 300 Coxy7 Infinity images and 300 KoalaAI StockImages-CC0 photographs. Its 319-row false-positive slice is row-, byte-, and dHash-disjoint from the real confirmation slice but shares the StockImages source corpus, so it is not an independent source-population estimate. Dataset cards at the pinned revisions report CC BY 4.0 and CC0-1.0. Those are source-reported license statements, not independent rights clearance; `benchmark/manifests/replacement-v2-attribution.json` preserves authors, links, and the rights caveat.

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
