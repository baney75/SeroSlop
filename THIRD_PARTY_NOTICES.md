# Third-party notices

## Community Forensics

The packaged detector is derived from the MIT-licensed Community Forensics model and code:

- Jeongsoo Park and colleagues, *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*, CVPR 2025.
- Repository: https://github.com/JeongsooP/Community-Forensics
- Corrected model: https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT
- Upstream revision: `ac6ee457bea904a373065754107451793b56db00`

## ONNX Runtime Web

The built extension bundles ONNX Runtime Web 1.22.0 and its WASM runtime files. ONNX Runtime is licensed under the MIT License. Its package includes Microsoft third-party notices in `node_modules/onnxruntime-web` after `npm ci`.

## SynthCheck

ProofLens’s TypeScript MV3 service-worker/offscreen structure, setup flow, and initial test/build design are derived from the MIT-licensed [SynthCheck](https://github.com/thedudeb/synthcheck) project by TheDudeb, inspected at commit `5883d1c23895d407ee1ec50fccfdccca165cc072`.

ProofLens materially changes the model artifact and preprocessing, classifier head, threshold calibration, WebGPU/WASM fallback, content-hash cache, local-only image acquisition, CSS-background scanning, stale-result handling, persisted-model revalidation, tests, and benchmark evidence.

## Evaluation datasets

Dataset pixels are not redistributed in this repository.

- Qwen Image Bench: Apache-2.0
- Open Images V7 selected images: CC BY 2.0; attribution is generated beside the local dataset
- DOCCI selected training images: CC BY 4.0; attribution is generated beside the local dataset

The source MIT license for ProofLens is [LICENSE](LICENSE).
