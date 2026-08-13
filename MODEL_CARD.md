# Model card

## Model

ProofLens uses a frozen Community Forensics ViT-S/16 backbone with a replacement linear classifier trained for current generator families and common web transformations.

- Packaged artifact: `weights/prooflens-cf384.onnx`
- Format: ONNX FP32
- Parameters changed from upstream: classifier weight `[1,384]` and bias only
- Packaged SHA-256: `29545a1da0cfe2bf0149448334fd45a21f48074c57296db3b84437dd66f80a43`
- Upstream: [CommunityForensics DeepfakeDet-ViT](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)
- Corrected upstream revision: `ac6ee457bea904a373065754107451793b56db00`
- Upstream artifact SHA-256: `a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1`
- License: MIT

The underlying [Community Forensics paper](https://arxiv.org/abs/2411.04125) reports training across millions of outputs and thousands of generators. ProofLens uses the corrected 2026 model configuration rather than older cached or legacy exports.

## Input and preprocessing

Decode the displayed RGB image with EXIF orientation applied by the browser, then resize it without changing aspect ratio until its shortest edge is 440 pixels. Take the 384×384 center crop and scale its RGB bytes to `[0,1]`. Normalize the channels with mean `[0.48145466, 0.4578275, 0.40821073]` and standard deviation `[0.26862954, 0.26130258, 0.27577711]`, transpose to planar NCHW float32, and run `pixel_values` → `logits`.

## Output and calibration

The graph returns one synthetic-image logit. ProofLens displays:

```text
sigmoid(raw_logit + 0.30374610239790173)
```

The fixed classification is **likely AI-generated when displayed probability ≥ 0.65**. No site, source, file type, metadata, generator name, or transformation receives a different threshold.

## Intended use

ProofLens provides a local screening hint for ordinary webpage images. It may help a person decide which images deserve closer inspection.

It is not intended to:

- prove that an image is authentic or synthetic;
- identify a generator, author, or copyright owner;
- make moderation, employment, legal, financial, or safety decisions without human review;
- classify medical or scientific imagery outside the evaluated distribution.

## Known limitations

- New generators and post-processing can shift performance.
- Real illustrations, unusual camera pipelines, screenshots, or heavily edited photographs may be false positives.
- Cropping can remove useful evidence; center crop is fixed for reproducibility and latency.
- The public test is sample-disjoint but shares its six synthetic families and real source with validation.
- The browser’s canvas resampling is not bit-identical to Pillow bicubic. A 60-image WebGPU diagnostic showed 98.33% decision agreement with the reference evaluator at the frozen cutoff.
- An unavailable badge means the pixels could not be acquired or decoded. It is not a low AI score.

See [BENCHMARK.md](BENCHMARK.md) for measured class recalls, transformation results, and confidence intervals.
