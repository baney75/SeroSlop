# Replacement holdout selection

This directory defines the score-blind replacement for the consumed Kling/Library of Congress holdout. The first confirmatory packet remains under `benchmark/evidence/evaluation/confirmatory/` with `acceptanceEligible: false`; its numeric-contract failure is not a reason to tune the model or choose easier data.

The replacement confirmation set contains 300 Infinity images from pinned Coxy7 shards whose dataset card at the fixed revision reports CC BY 4.0, and 300 photographs from the pinned KoalaAI StockImages-CC0 revision whose card reports CC0-1.0. Infinity is absent from the linear-head training and validation generator families. StockImages-CC0 is absent from training and every earlier evaluation source. The separate 319-image web-negative set uses different StockImages rows and bytes. It shares the source dataset with the real confirmation slice, so it is a second false-positive sample, not an independent source-population estimate. These uploader-provided license statements are provenance evidence, not independent verification of contributor ownership or other third-party rights. The full source-reported license, author, fixed-revision, asset-locality, and no-endorsement notice is in `benchmark/manifests/replacement-v2-attribution.json`.

Run selection without model inference:

```bash
benchmark/.venv/bin/python benchmark/recovery_v3/test_prepare.py
benchmark/.venv/bin/python benchmark/recovery_v3/build_historical_index.py
benchmark/.venv/bin/python benchmark/recovery_v3/prepare.py --offline
benchmark/.venv/bin/python benchmark/recovery_v3/verify.py
```

`build_historical_index.py` acquires the official DOCCI metadata and recovers the 1,200 legacy evaluation pixels from their original Open Images and pinned Qwen Image Bench locators. Every source must match its fixed byte count and SHA-256 before parsing or dHash computation; `--offline` fails closed if any pinned byte is absent or changed. It merges those rows with existing training and evaluation evidence into one 106,019-item exclusion index. Official DOCCI `cluster_id` metadata replaces per-image IDs for the 1,200 historical DOCCI training rows. The tracked gzip uses a fixed header and deterministic raw DEFLATE encoding, so a clean rebuild must be byte-identical across supported Python runtimes.

`prepare.py` consumes three pinned Coxy7 and two pinned StockImages parquet shards, then writes selected pixels only below ignored `benchmark/data/replacement-v2/`. It rejects every historical or cross-protocol ID, byte, source-group, or dHash match. Replacement v2 permits no perceptual-overlap exception. `verify.py` recomputes the legacy index from locked pixels and reruns the complete selector in an isolated directory before comparing every tracked artifact and selected byte.

The tracked packet is:

- `benchmark/manifests/test-v2.jsonl`: 600 replacement confirmation rows;
- `benchmark/manifests/web-negative-v2.jsonl`: 319 row-, byte-, and dHash-disjoint false-positive rows; not an independent source-population estimate;
- `benchmark/manifests/historical-perceptual-exclusions-v1.json.gz`: one byte-bound dHash and source group for all 106,019 historical exclusions;
- `benchmark/manifests/replacement-v2-selection.json`: source locks, counts, digests, and every rejected candidate;
- `benchmark/manifests/replacement-v2-perceptual-review.json`: the zero-exception dHash policy;
- `benchmark/manifests/replacement-v2-attribution.json`: dataset revisions, licenses, and citations.

No ProofLens model, threshold, or prediction is an input to this selection. The committed V2 receipt records the boundary before the failed v1 run. Replacement scoring remains unauthorized until the A4 binary64 recovery source and its receipt-only V3 child both pass locally, on public GitHub Actions, and through anonymous exact-head checks.
