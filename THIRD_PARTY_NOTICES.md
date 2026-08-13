# Third-party notices

ProofLens distributes the following third-party model and runtime components. Verbatim license and notice files are shipped under `LICENSES/` in both the source tree and release archive.

## Community Forensics model

The packaged detector is derived from the corrected CommunityForensics DeepfakeDet-ViT artifact at revision `ac6ee457bea904a373065754107451793b56db00`.

- Work: Jeongsoo Park and colleagues, *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*, CVPR 2025
- Model source: https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/tree/ac6ee457bea904a373065754107451793b56db00
- Verbatim license: `LICENSES/COMMUNITY_FORENSICS_MODEL_MIT.txt`
- License SHA-256: `69a0eab6ca179df33ed80fa378b9458632e14ba9547374e299249e0a4f8076cb`

## ONNX Runtime Web 1.22.0

The extension bundles ONNX Runtime Web 1.22.0 code and WASM runtime assets from tag `rel-1.22.0`.

- Source: https://github.com/microsoft/onnxruntime/tree/rel-1.22.0
- MIT license: `LICENSES/ONNX_RUNTIME_MIT.txt`
- MIT license SHA-256: `2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c`
- Complete upstream notices: `LICENSES/ONNX_RUNTIME_THIRD_PARTY_NOTICES.txt`
- Notice-file SHA-256: `e9e90971a8e75a9a8ac0c6412e29c1202d079998389915aa485f46c816c3b4cc`
- Apache License 2.0 copy for retained Google notices: `LICENSES/APACHE-2.0.txt`

The generated ONNX Runtime Web bundle retains its upstream copyright comments. The complete tag-level notice file is included conservatively because compiled WASM attribution cannot be reduced reliably from the generated package alone.

## SynthCheck

ProofLens’s initial Manifest V3 service-worker/offscreen structure, setup flow, and test/build design were derived from SynthCheck commit `5883d1c23895d407ee1ec50fccfdccca165cc072`.

- Source: https://github.com/thedudeb/synthcheck/tree/5883d1c23895d407ee1ec50fccfdccca165cc072
- Verbatim license: `LICENSES/SYNTHCHECK_MIT.txt`
- License SHA-256: `5a8ee7ffa018b7d8e888903acba24b16072ce84e95bf07d7fb6ebdd8a10f9c84`

ProofLens changes the model and preprocessing, classifier head, calibration, WebGPU/WASM fallback, content-hash cache, local-only pixel acquisition, CSS composite scanning, stale-result handling, persisted-model revalidation, hostile-page bounds, tests, and benchmark evidence.

## Evaluation datasets

Dataset pixels are not redistributed in this repository or release archive.

- Qwen Image Bench: Apache-2.0
- Open Images V7 selected images: CC BY 2.0; attribution is generated beside a local reconstruction
- DOCCI selected training images: CC BY 4.0; selected public example IDs are committed in `benchmark/manifests/docci-attribution.json`

ProofLens itself is licensed under the root `LICENSE`.
