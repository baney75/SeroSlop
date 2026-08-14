# Large-head training corpus

This directory defines the source-pinned, 103,600-image training expansion used
for the ProofLens classifier head. Dataset pixels and generated feature caches
remain outside Git.

The corpus combines:

- 50,000 CC0 DiffusionDB 2M images selected from deterministic archive parts;
- 50,000 Open Images V7 training photos with per-image CC-BY 2.0 attribution;
- the existing 3,600-image modern training split (Qwen Image Bench, Open
  Images V7 validation, and DOCCI).

Selection, content hashing, rejection, attribution, and exact ID/byte overlap
checks against both frozen evaluation manifests are implemented by `prepare.py`.
The exact recipe is frozen in `recipe.json`. A deterministic reserve absorbs
ordinary missing or corrupt inputs without weakening the target counts.

The plan currently locks 85 DiffusionDB archives totaling about 53.8 GB and
55,000 Open Images candidates. Materialization processes at most four archives
and 24 individual photographs at once, caps each Open Images response at 25 MB, and requires 140 GB of free
space before network work begins. Cached archives and verified images make an
interrupted run resumable. Replanning is a deliberate operation; an ordinary
restart reuses and structurally validates the existing plan.

```bash
python3 -m venv benchmark/.venv
benchmark/.venv/bin/pip install -r benchmark/requirements.txt
benchmark/.venv/bin/python benchmark/large/prepare.py --phase plan --replan
benchmark/.venv/bin/python benchmark/large/prepare.py --phase materialize --workers 24
```

The canonical head-training run uses CPU deterministic algorithms. Every one
of the 114,400 extracted feature views participates in every training epoch;
source-balanced loss prevents the two 50,000-image sources from drowning out
the modern generator families.

```bash
benchmark/.venv/bin/python benchmark/modern/train_rehead.py \
  --model benchmark/candidates/upstream-cf384.onnx \
  --expected-model-sha256 a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1 \
  --data-root benchmark/data/large-head \
  --train-manifest benchmark/data/large-head/train-manifest.jsonl \
  --validation-data-root benchmark/data/modern-head \
  --validation-manifest benchmark/manifests/validation.jsonl \
  --recipe benchmark/large/recipe.json \
  --selection-summary benchmark/data/large-head/selection-summary.json \
  --single-view-source diffusiondb-stable-diffusion \
  --single-view-source open-images-train \
  --execution-provider cpu \
  --batch-size 24 \
  --feature-shard-images 2000 \
  --reextract-cached-features \
  --output-dir benchmark/candidates/prooflens-cf384-large
```

`--reextract-cached-features` rejects every shard from an earlier extraction run. It creates `fresh-feature-run.json` in the candidate directory and may resume after interruption only from source-verified shards carrying that exact run ID and the current model, manifest, provider, Pillow, and preprocessing contract.

No dataset pixels are redistributed by ProofLens. See `BENCHMARK.md` and
`MODEL_CARD.md` for evaluation boundaries and model provenance.
