# ProofLens large-head corpus

## Purpose

This corpus trains only the replacement 384-to-1 classifier on top of the frozen Community Forensics ViT-S/16 backbone. It is designed to improve current-generator coverage and ordinary-web false-positive behavior without redistributing source pixels.

## Composition

| Stratum | Label | Images | Role |
|---|---|---:|---|
| DiffusionDB 2M | synthetic | 50,000 | Broad Stable Diffusion subjects, styles, aspect ratios, and prompt distributions |
| Qwen Image Bench | synthetic | 1,200 | Fifteen current generator versions, 80 prompt groups each |
| Open Images V7 train | non-AI | 50,000 | Large ordinary-web stratum including photographs, products, scans, illustrations, and other pre-generative imagery |
| Open Images V7 validation | non-AI | 1,200 | Independently selected modern-training stratum |
| DOCCI train | non-AI | 1,200 | High-resolution, text-rich natural photographs |
| **Total** |  | **103,600** | 52,400 non-AI / 51,200 synthetic |

The two 50,000-image strata use only the original view. Each modern image uses original, screenshot, JPEG-resize, and heavy double-JPEG views, producing 114,400 frozen feature views. Source-balanced loss assigns half of the objective to each class and equal mass to each named source within that class; the large strata therefore add coverage without drowning out the current generators.

## Source and license locks

- DiffusionDB: `poloclub/diffusiondb`, revision `fb620fbe49fa4420e0734bd9c0df11f51176b61f`, CC0-1.0.
- Qwen Image Bench: `Qwen/Qwen-Image-Bench`, revision `d2493deb153b020cf169c7e3f57d15e4dd697038`, Apache-2.0.
- Open Images V7 train: pinned 2018-04 metadata, CC BY 2.0. All 50,000 selected rows retain author, title, landing page, original URL, and byte hash in the compressed evidence packet.
- Open Images V7 validation: pinned CVDF mirror metadata, CC BY 2.0.
- DOCCI: `google/docci`, revision `a0a43eaf34676ffd008fb6565dd8c2ba00d09100`, CC BY 4.0.

`recipe.json` pins the source revisions and metadata hashes. `selection-plan.json.gz` pins all 85 DiffusionDB archives and 55,000 Open Images reserve candidates. `npm run check:large-training-evidence` verifies these locks.

## Isolation and quality controls

- All 103,600 IDs, byte hashes, paths, and 64-bit perceptual hashes are unique where the contract requires uniqueness.
- Training has zero ID or byte overlap with validation, the confirmatory test, or the Chartography challenge slice.
- Synthetic generator versions and prompt groups are disjoint across training, validation, and confirmatory test.
- A threshold-complete dHash search at Hamming distance 8 identifies possible cross-split visual matches. Every retained low-frequency candidate has an immutable side-by-side `visually-distinct` review; unreviewed matches fail materialization.
- Open Images responses are size-bounded, decoded, orientation-normalized, dimension-checked, content-hashed, and selected through a deterministic reserve.
- Evaluation images never enter candidate training or threshold selection.

## Limitations

DiffusionDB is large but is dominated by an older Stable Diffusion generation era. The smaller Qwen stratum supplies current proprietary and open generator versions but cannot represent every future model. “Non-AI” means the source predates the modern generative pipeline or has documented historical provenance; it is not a claim that every image is a camera-original photograph. Dataset licenses and attribution support reproducibility, but they do not substitute for legal advice about every downstream use.

The confirmatory test and web-negative challenge remain separate from this corpus and are evaluated only after model and calibration freeze. Public results do not predict the bounty maintainer’s private score.
