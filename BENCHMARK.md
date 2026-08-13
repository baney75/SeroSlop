# Benchmark and browser evidence

## Decision contract

The browser displays `sigmoid(raw_logit + 0.30374610239790173)`. A displayed score **greater than or equal to 0.65** is classified as likely AI-generated. This maps to raw logit `0.3152931060083219` (raw sigmoid probability `0.5781767196773971`). The mapping and artifact were frozen before the sealed test ran.

## Data and split design

The deterministic preparation code is under `benchmark/modern/`.

- Synthetic source: [Qwen Image Bench](https://huggingface.co/datasets/Qwen/Qwen-Image-Bench), pinned to revision `d2493deb153b020cf169c7e3f57d15e4dd697038` (Apache-2.0).
- Real sources: [Open Images V7](https://storage.googleapis.com/openimages/web/index.html) validation images (CC BY 2.0) and the official [DOCCI](https://google.github.io/docci/) train split (CC BY 4.0).
- Training: 1,200 synthetic images balanced over 12 generator families plus 2,400 real images from two source pipelines. Each image produces original, screenshot, JPEG-75-range, and heavy double-JPEG training views.
- Validation: 300 synthetic images from six generator families excluded from training, plus 300 disjoint Open Images photographs. Validation selected one of 25 predeclared regularization/upstream-blend candidates and one global threshold alignment.
- Test: 300 further synthetic images from those six held-out families plus 300 further Open Images photographs. It is sample-disjoint and was evaluated once after the artifact and calibration were frozen. It is not generator-family-disjoint from validation; that limitation is reported rather than hidden.

Exact duplicate hashes and exposed audit names/hashes are excluded by the preparation scripts. Originals and every transformed derivative remain in the same split.

## Frozen artifacts

| Item | SHA-256 |
|---|---|
| Upstream corrected CF384 ONNX | `a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1` |
| Trained ProofLens FP32 ONNX | `29545a1da0cfe2bf0149448334fd45a21f48074c57296db3b84437dd66f80a43` |
| Training manifest | `8207d373a6abec32e4a687361199de43568fe982a5ff2e045987d3a029b85626` |
| Validation manifest | `651704a6d3fb86b9324f13bea721c40854535925ce33b49b8749978ee2bed915` |
| Sealed test manifest | `cd4d09fbb59d695ebc0cc4dc96f0dd17caea9e0e8865d7658c6100ef723e977f` |

## Sealed results

All rows contain 300 real and 300 synthetic images. Confidence intervals use 20,000 deterministic, class-stratified nonparametric bootstrap replicates.

| View | Balanced accuracy | Real recall | Synthetic recall | 95% BA interval |
|---|---:|---:|---:|---:|
| Original | 93.83% | 92.67% | 95.00% | 91.83–95.67% |
| Social screenshot | 96.00% | 97.67% | 94.33% | 94.33–97.50% |
| Social JPEG 75 | 92.67% | 92.67% | 92.67% | 90.50–94.67% |
| Social heavy | 92.17% | 95.67% | 88.67% | 90.00–94.17% |

Synthetic recall by held-out family:

| Family | Original | Screenshot | JPEG 75 | Heavy |
|---|---:|---:|---:|---:|
| FLUX.2 max | 90% | 80% | 84% | 84% |
| Imagen 4 Ultra | 98% | 98% | 94% | 94% |
| Qwen Image 2 Pro | 98% | 98% | 94% | 90% |
| Seedream 5 | 96% | 98% | 96% | 94% |
| GPT Image 2 | 92% | 94% | 92% | 82% |
| Nano Banana 2 | 96% | 98% | 96% | 88% |

Machine-readable aggregates are in `benchmark/results/`. Dataset pixels and per-image sealed predictions are intentionally not committed.

## Browser parity

A deterministic diagnostic subset contains 30 real images and 30 synthetic images, with five samples from each held-out generator family. Chrome 139 ran the exact packaged model and JavaScript preprocessing through WebGPU while offline:

- Browser balanced accuracy: 90.00%
- Reference evaluator balanced accuracy on the same subset: 88.33%
- Decision agreement: 98.33% (59/60)
- Mean absolute raw-probability difference: 0.0231
- Maximum absolute raw-probability difference: 0.1891
- Network requests during inference: 0

This subset is a preprocessing/runtime parity diagnostic, not a second accuracy estimate. Its full report is `artifacts/browser-parity.json`.

## Clean-profile browser contract

The project-owned Chrome E2E runs twice: WebGPU and forced WASM. Each run:

1. launches a fresh profile with only ProofLens loaded;
2. verifies and prepares the bundled model;
3. closes and restarts the browser against the same profile;
4. loads a fixture page while analysis is disabled;
5. stops the fixture server, enables browser-offline mode, and records all later HTTP(S) requests;
6. enables automatic analysis;
7. requires five numeric results and one explicit unavailable result across normal, duplicate, responsive `picture/srcset`, dynamic, and CSS-background targets;
8. requires zero post-cutoff network requests.

The reports are `artifacts/chrome-e2e-webgpu.json` and `artifacts/chrome-e2e-wasm.json`.

## Reproduction

Extension and browser checks:

```bash
npm ci
npm run verify
npm run browser:install
npm run test:chrome
npm run test:chrome:webgpu
```

Model/data reproduction requires substantial disk and GPU time:

```bash
python3 -m venv benchmark/.venv
benchmark/.venv/bin/pip install -r benchmark/requirements.txt
npm run benchmark:modern:prepare
npm run benchmark:modern:prepare:docci
benchmark/.venv/bin/python benchmark/download_upstream.py
benchmark/.venv/bin/python benchmark/modern/train_rehead.py \
  --model benchmark/candidates/upstream-cf384.onnx \
  --expected-model-sha256 a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1 \
  --data-root benchmark/data/modern-head \
  --output-dir benchmark/candidates/prooflens-cf384
```

The benchmark is designed to resist simple sample memorization, but no public holdout predicts every future generator, real-image source, or the maintainer’s private benchmark. Results should not be interpreted as provenance proof.
