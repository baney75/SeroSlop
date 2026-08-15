# M4 residual-adapter development protocol

M4 is a bounded development experiment after M3 failed its frozen selector. It does not reinterpret or relax that failure. The shipped M2 model remains the upstream artifact until a new candidate passes a fresh selector, both consumed regression packets, ONNX/browser parity, and the later untouched H3 evaluation.

## Accepted brief

- **Outcome:** test whether a zero-initialized 384→64→384 residual adapter can improve cultural-heritage and modern-generator coverage without sacrificing the shipped M2 boundary.
- **Audience:** ProofLens maintainers, reviewers, and people reproducing the local model packet.
- **Deliverables:** a public protocol freeze, a score-blind pixel-free source packet, one finite candidate grid, and either a terminal failure packet or a separately locked publication packet.
- **Non-goals:** no H3 pixel access, no private-bounty claim, no probability-calibration claim, no gate relaxation, no regression-driven reselection, and no model publication merely because training finishes.
- **Changed surface:** M4-only benchmark, evidence, stage-policy, and eventual model-publication files. Historical M1/M2/M3 evidence remains immutable.
- **Proofs:** fixed source revisions and file hashes; book- and prompt-group isolation; ID, byte, group, and dHash exclusion; fresh feature extraction; exhaustive binary64 threshold enumeration; terminal M3 and M2 regressions; exact ONNX graph/tensor comparison; forced-WASM and WebGPU browser parity.
- **Stop conditions:** insufficient lawful/source-pinned capacity, any H3 read, zero-adapter parity failure, no selector-valid candidate, either regression failure, model/runtime parity failure, or a non-exact stage boundary.

## Score-blind partitions

M4 adds 2,400 British Library plate images from distinct books and 1,920 Rapidata images from 120 distinct prompt groups to the unchanged 108,378-image M3 training manifest. The fresh selector contains 300 other British Library books plus 300 images from 75 other Rapidata prompt groups: 75 each labeled by the publisher as DALL-E 3, Flux, MidJourney, and Stable Diffusion. All images from a selector prompt group are excluded from training even though only one image per family is selected. The 43 remaining complete prompt groups are a fail-closed capacity reserve for source or overlap rejection; they are not a second search budget. The 32 locked British Library shards contain 19,060 raw rows: 18,451 rows fall in the frozen 1800–1890 date strata and 609 are excluded before pixel decoding (`Unknown`: 595, `1754`: 1, `1777`: 13). The public source index exhaustively accounts for every raw shard row and retains the rejected row positions without publishing pixels.

The British Library mirror is pinned to revision `c288990ce59b055e7bf9411f663d0f672ae16102`, config `plates`. Its card says `plates` is an algorithmic page-layout category, not a curated art taxonomy. The machine-readable card reports CC0-1.0, while the prose describes the underlying British Library Flickr Commons release as Public Domain Mark / no known copyright restrictions. ProofLens records that distinction and does not independently clear every depicted or third-party right.

The synthetic source is `Rapidata/700k_Human_Preference_Dataset_FLUX_SD3_MJ_DALLE3` at revision `96a4db1d70fbf08f1054dff771f465dccab94535`. Its card and associated publisher paper provide prompts, paired image paths, model-family labels, and methodology. The card reports CDLA-Permissive-2.0. These are publisher-authored provenance and license statements, not independent rights clearance or exact generator-revision receipts, so M4 uses the corpus only for development selection/training—not final acceptance provenance.

The score-blind Rapidata selector keeps one image per family in each of 75 prompt groups. Training uses 120 different prompt groups. Within a training group, every output that passes the frozen ID, byte, source-group, and dHash checks is retained only when at least one clean image remains from all four families; rejected outputs stay unassigned. The frozen allocation is 1,784 training images: 456 DALL-E 3, 443 FLUX, 457 Midjourney, and 428 Stable Diffusion. This preserves all four families in every admitted training group without weakening the overlap threshold.

M3's 600-image selector and M2's 900-image development packet are consumed post-selection regressions only. Candidate fitting and ranking cannot read them. H3 is represented solely by the committed pixel-free manifest hash used for exclusion; no M4 command accepts an H3 data root.

## Frozen model experiment

The ViT backbone and M2 classifier stay byte-identical. M4 inserts a zero-initialized residual adapter before the existing classifier:

```text
u = (z - mean) / std
h = ReLU(Gemm(u, 64x384, 64))
r = Gemm(h, 384x64, 384)
z_m4 = z + std * r
logits = existing_M2_classifier(z_m4)
```

Only the four adapter tensors are trained. The classifier remains frozen. The 12 candidates are the Cartesian product of adapter weight decay `[0.003, 0.01, 0.03]` and M2-logit anchor coefficient `[0.01, 0.03, 0.10, 0.30]`, with one fixed width, learning rate, initialization, epoch count, and seed. The anchor applies to all carried-forward M3 sources except `met-open-access`; new British Library and Rapidata rows are not anchored.

The candidate and threshold are selected solely on the fresh M4 selector. The fixed 65/100 display threshold remains a threshold-label alignment and is not a calibrated probability.

## Intended sequence

1. Commit and publicly verify this protocol freeze with no selected source pixels, manifests, features, candidates, or scores. The append-only date-eligibility and Rapidata-capacity recoveries were made after the immutable source archives were downloaded but before any selected manifest, materialized output, feature extraction, candidate fitting, or model score existed. The first score-blind materialization then produced the final 112,562/600 packet, but its independent public replay exposed a verifier-only group-key mismatch: the producer keyed Rapidata groups by prompt hash while the public verifier keyed the same rows by source-group ID. The append-only replay recovery changes only that verifier and its contracts, freezes the already-generated packet hashes, and records that no model output or H3 pixel was read.
2. Materialize the pinned source files locally, publish only pixel-free source evidence, and publicly verify that exact source packet.
3. Run one fresh CPU feature-extraction/training attempt from the documented command emitted by the source packet.
4. Freeze the selector winner before evaluating M3, then M2, as terminal regressions. A failure records one append-only failure packet; it cannot trigger another candidate or threshold.
5. Only a candidate that passes both regressions may proceed to model locking, browser parity, and untouched H3.

No M4 result, H3 result, bounty acceptance, or payment is claimed by this protocol freeze.

## Canonical commands

Run these from the repository root. Source acquisition is the only networked step; it downloads the exact files listed in `source-locks.json`. Training uses CPU ONNX feature extraction and resumes only caches carrying the same fresh-run ID and frozen configuration.

```bash
npm run benchmark:m4:materialize
npm run benchmark:m4:verify-public
npm run benchmark:m4:train
```

If training ends with no selector-valid candidate or a failed terminal regression, publish only the failure packet:

```bash
npm run benchmark:m4:failure
```

If the selector and both regressions pass, create the one-file output lock, commit and publicly verify that lock, then publish the exact locked model/evidence transaction:

```bash
npm run benchmark:m4:lock
npm run benchmark:m4:finalize
```

The public failure verifier recomputes selector and regression decisions from canonical recorded logits and validates the embedded raw candidate tensors. The local failure finalizer additionally reopens the ignored feature caches, source pixels, candidate files, and chained regression state files. Public Git evidence cannot independently prove those ignored local files after cleanup; the receipt records their exact hashes and states that boundary rather than treating it as acceptance evidence.
