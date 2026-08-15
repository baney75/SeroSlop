# M2 hard-negative development

M2 is a new classifier head over the same frozen Community Forensics ViT-S/16 backbone. It responds to the consumed replacement-v2 result without changing the extension preprocessing contract, the fixed 65/100 display threshold, or any external acceptance gate.

The only new training source is `KoalaAI/StockImages-CC0` at revision `206f3575579f1187548c6f47042ae9174c0a51fc`. Its dataset card reports CC0-1.0; that uploader-provided statement is not independent rights clearance. The two source parquet shards are already pinned by byte count and SHA-256.

Selection is score-independent. It excludes every historical item plus every v1/v2 item by ID, exact bytes, source group, and EXIF-oriented 64-bit dHash at Hamming distance 8 or lower. Of 2,699 rows that clear those prior-data checks, 300 enter a development-only real-photo slice, 2,378 enter training, and 21 are rejected as new cross-partition or within-training near duplicates. The consumed v2 rows never enter gradients or M2 development metrics.

The 103,600-image M1 corpus remains intact. M2 adds 2,378 four-view StockImages rows, yielding 105,978 unique training images and 123,912 feature views. The 900-image development set combines the original 600 validation rows with 300 new four-view StockImages rows. Candidate selection uses the same 25-entry linear-head grid and source-balanced BCE. Every candidate must retain the original validation gates and reach at least 93% StockImages real recall on each development view.

Pixels, source parquet shards, feature caches, and uncompressed manifests stay under ignored `benchmark/data/` or `benchmark/candidates/`. The source-freeze evidence packet contains hashes, manifests, rejects, attribution, and selection receipts, not source pixels. Training outputs and a classifier-only model comparison are added only after the frozen training run completes.

## Reproduce the packet and train

From the repository root, using the pinned environment under `benchmark/.venv`:

```bash
benchmark/.venv/bin/python benchmark/m2/prepare.py --materialize --publish
benchmark/.venv/bin/python benchmark/m2/verify.py --verify-pixels
```

The first command securely downloads the two ignored StockImages parquet shards when they are missing. The fixed dataset revision, expected byte counts, and SHA-256 values are in `recipe.json`; a shard is used only after all three checks pass. Add `--offline` to `prepare.py` when the pinned shards must already be cached. The independent verifier is always offline and fails closed if either shard is absent or changed.

The canonical fresh-feature training command is:

```bash
benchmark/.venv/bin/python benchmark/modern/train_rehead.py \
  --model benchmark/candidates/upstream-cf384.onnx \
  --expected-model-sha256 a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1 \
  --data-root benchmark/data/m2-head \
  --train-manifest benchmark/data/m2-head/train-manifest.jsonl \
  --validation-data-root benchmark/data/m2-head \
  --validation-manifest benchmark/evidence/m2/validation-manifest.jsonl \
  --recipe benchmark/m2/recipe.json \
  --selection-summary benchmark/data/m2-head/selection-summary.json \
  --single-view-source diffusiondb-stable-diffusion \
  --single-view-source open-images-train \
  --execution-provider cpu \
  --batch-size 24 \
  --feature-shard-images 2000 \
  --reextract-cached-features \
  --output-dir benchmark/candidates/prooflens-cf384-m2
```

Training must start from an absent M2 candidate directory. The fresh-run marker makes an interrupted invocation resumable only with shards created by that exact run context. The trainer verifies every manifest-bound pixel before fresh extraction and records the full command, trainer hash, runtime versions, feature-shard hashes, candidate grid, calibration, and exported model parity.
