# M3 cultural-heritage hard-negative head

M3 is a development-only repair for one measured weakness in M2: false positives on public-domain museum images. A consumed 100-image Met probe produced false-positive rates of 25% on originals, 10% on screenshot views, 29% on JPEG-Q75 views, and 18% on the heavy social-media transform. Those observations may justify a repair, but they are not acceptance evidence.

The score-blind source packet fixes four separate roles before training:

- 2,400 Met Open Access public-domain images are appended to the unchanged M2 training corpus;
- 300 different Met images and 300 FLUX.1-dev images form the fresh M3 selector used for candidate and threshold selection;
- the existing 900-image M2 development set is used only as a post-selection regression gate;
- 600 additional Met images are reserved for the later H3 web-negative evaluation and are not read by training or selection.

The resulting training manifest contains 108,378 unique images and 133,512 feature views. The fresh selector contains 600 images and 2,400 views. The frozen M2 regression packet contains 900 images and 3,600 views. Every new cross-pool ID, image SHA-256, normalized source group, and dHash match at Hamming distance 8 or lower is rejected. The unchanged M2 prefix carries forward 100 earlier dHash candidates that were reviewed as visually distinct; there are no unreviewed carried-forward pairs.

## Source locks and limits

Met eligibility is joined from two byte-locked sources: the official `MetObjects.csv` at commit `e901de145e60258542243571098245826a01fe47`, and the fixed `metmuseum/openaccess` metadata mirror at revision `c65f8d6041aea7b3bc767a54d93772c3c6a365f6`. Rows must be marked public domain, have an object end date no later than 1980, belong to an allowlisted visual or physical collection department, and expose a canonical `images.metmuseum.org` image URL. The Met and mirror license labels are source-reported; they are provenance evidence, not an independent clearance of every third-party right.

The synthetic selector uses `LukasT9/Flux-1-Dev-Images-1k` at revision `ade8147d5f10bac016d691fd2b564e877e6315f7`. Its card reports Apache-2.0 and says the images were generated with FLUX.1-dev at 50 steps. It does not identify a model revision, seed, or the remaining generation parameters. This set is therefore suitable only as development evidence, not as source-native generation proof or final acceptance evidence.

Source pixels, the 317.7 MB Met CSV, the 28.2 MB fixed metadata file, and feature caches remain under ignored `benchmark/data` or `benchmark/candidates` paths. The repository publishes pixel-free manifests, source locks, attribution, rejection evidence, and overlap evidence.

## Reconstruct and verify

From the repository root, using the benchmark environment documented in [`../README.md`](../README.md):

```bash
benchmark/.venv/bin/python benchmark/m3/prepare.py --materialize --write-local --publish
benchmark/.venv/bin/python benchmark/m3/verify.py --verify-pixels
```

`--materialize` downloads only missing byte-locked source inputs and selected pixels. `--offline` disables all downloads and fails if a pinned input is missing. The verifier independently reruns selection offline, requires byte-identical public evidence and canonical gzip bytes, rechecks the exact M2 manifest prefix, and proves the partition boundaries. `--verify-pixels` additionally hashes and decodes every local training, selector, probe, and reserved-H3 image.

## Frozen CPU training command

Training must not begin until the complete source packet and trainer contract are committed on a clean public head. The canonical local command is:

```bash
benchmark/.venv/bin/python benchmark/modern/train_rehead.py \
  --model weights/prooflens-cf384.onnx \
  --expected-model-sha256 a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47 \
  --data-root benchmark/data/m3-head \
  --train-manifest benchmark/data/m3-head/train-manifest.jsonl \
  --validation-data-root benchmark/data/m3-head \
  --validation-manifest benchmark/evidence/m3/validation-manifest.jsonl \
  --regression-data-root benchmark/data/m2-head \
  --regression-manifest benchmark/evidence/m2/validation-manifest.jsonl \
  --recipe benchmark/m3/recipe.json \
  --selection-summary benchmark/evidence/m3/selection-summary.json \
  --single-view-source diffusiondb-stable-diffusion \
  --single-view-source open-images-train \
  --execution-provider cpu \
  --batch-size 24 \
  --feature-shard-images 2000 \
  --reextract-cached-features \
  --output-dir benchmark/candidates/prooflens-cf384-m3
```

The 25 fixed head candidates and their thresholds are selected only from the fresh 600-image selector. After the winning candidate and threshold are frozen, that one candidate is evaluated on the M2 regression packet. A regression failure is terminal; it cannot select another candidate or change the threshold. The reserved H3 rows are not an argument accepted by this command.

After training succeeds, validate the two fixed browser QA assets without changing their bytes, then create the one-file output lock:

```bash
benchmark/.venv/bin/python benchmark/m3/select_model_state_fixtures.py
npm run benchmark:m3:lock
```

Commit and publicly verify only `benchmark/evidence/m3/publication-lock.json`. The lock contains the exact JSON evidence bytes, the two classifier raw-tensor replacements needed to reconstruct the candidate from the shipped M2 model, and hashes of the deterministic documents and fixture. A clean checkout can therefore validate the candidate and derived packet without the ignored training cache. From that clean lock commit, publish the complete bound packet:

```bash
npm run benchmark:m3:finalize
```

The final publication commit is restricted to the model, model lock, weights README, five M3 training-evidence files, three public documents, and the fixture manifest. All 12 bytesets are derived before mutation, bound by the output lock, and owned by one rollback transaction. The finalizer writes its receipt last and restores every target it touched if publication raises an exception.

No M3 model result is claimed by this source packet. Passing the M3 selector and M2 regression would remain development evidence, not proof of final H3 generalization, the bounty maintainer's private result, acceptance, or payment.
