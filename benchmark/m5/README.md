# M5 RunPod GPU fine-tuning

## Accepted brief

- Outcome: produce one fully local FP32 ViT-S/384 ONNX detector that is no larger than 90,000,000 bytes and runs through the extension's existing WASM and WebGPU paths.
- Audience: Chrome users and public reviewers who need an auditable local detector, not a cloud service.
- Deliverables: a six-candidate RunPod training packet, a one-file public selection lock, terminal M3/M2 regression evidence, the final packaged model, Chrome/offline evidence, and an untouched final H3 evaluation.
- Non-goals: universal perfection, a calibrated probability claim, H3-guided selection, selector rows in gradients, or a larger teacher model in the extension.
- Authority: one bounded paid RunPod L40S job is authorized. Training may write only ignored `benchmark/candidates/prooflens-cf384-m5`; publication is a separate fixed-head Git transaction.
- Proofs: exact source/model hashes, RunPod L40S/CUDA receipt, complete candidate grid, embedded selector logits, exhaustive thresholds, zero observed selector false positives, public pre-regression lock, ordered terminal regressions, a separate 100,000-image synthetic-recall panel, ONNX parity, Chrome offline E2E, and four final critical reviews.
- Stop conditions: no selector-feasible candidate; a terminal regression failure; any H3 read; a changed source/model/runtime; model size above 90 MB; non-L40S or non-RunPod training; or eight paid GPU hours without a completed train-select packet.

The zero-false-positive gate means exactly zero observed false positives among the 300 fresh British Library real images in each of the four declared views. It is intentionally strict, but it does not prove that errors are impossible on all future images. The final evidence must disclose the sample count and confidence bound.

After training and terminal regressions, a separate public source lock must freeze at least 100,000 never-trained and never-scored synthetic images before their first inference. Deterministic SHA-256 ordering divides the panel into 1,000 batches of 100. At the already locked model and raw threshold, both the mean and median per-batch synthetic recall must be strictly above 95/100. This panel cannot select a candidate, threshold, gate, or training change. If the model misses either aggregate, those 100,000 rows become consumed development evidence; any later revision needs new training data and a different never-scored 100,000-image panel.

## Frozen model and data

M5 starts from the official `buildborderless/CommunityForensics-DeepfakeDet-ViT` PyTorch checkpoint at revision `ac6ee457bea904a373065754107451793b56db00`, then replaces its classifier with the exact shipped M2 classifier before fine-tuning. A 16-image preflight must keep PyTorch versus packaged-M2 ONNX logits within `2e-4`.

Training uses the public M4 packet only:

- 112,562 training images: 59,578 real and 52,984 synthetic;
- 600 fresh selector images: 300 British Library real and 300 Rapidata synthetic across four generator-family labels;
- four frozen views: original, screenshot, JPEG-q75, and heavy social recompression;
- no selector, regression, or H3 row in gradients;
- no H3 data root or argument accepted by the training command.

The source-balanced BCE gives the two classes equal total mass and each named source equal mass inside its class. It is never renormalized per minibatch. A frozen-M2 same-view logit anchor protects inherited M3 sources; new M4 sources are not anchored.

## Candidate grid

One L40S job trains two predeclared branches from the same initial model:

| Branch | Trainable backbone | Backbone LR | Head LR | Anchor | Candidate epochs |
|---|---|---:|---:|---:|---|
| `last4` | encoder blocks 8–11, final norm, classifier | 3e-5 | 3e-4 | 0.10 | 4, 6, 8 |
| `full` | embeddings, all 12 blocks, final norm, classifier | 1e-5 with 0.8 layer decay | 2e-4 | 0.15 | 4, 6, 8 |

Both run eight epochs with eager attention, AdamW, cosine decay, 5% warmup, bfloat16 autocast, batch 64 × accumulation 2, global source weights, deterministic image ordering, deterministic stateless view selection, and gradient clipping at 1.0.

Only fresh-selector evidence ranks candidates. All variants require 300/300 real images below the chosen raw threshold. The original view additionally requires at least 97% balanced accuracy and 94% synthetic recall; stress views require at least 95% balanced accuracy and 90% synthetic recall. Family gates are also fixed in `recipe.json`.

## RunPod execution

Use one RunPod Secure Cloud L40S 48 GB Pod. The recommended image is:

```text
pytorch/pytorch@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385
```

The repository must be cloned at the exact public M5 protocol commit. Transfer only `benchmark/data/m4-head` for train-select. Do not transfer any H3 root. The training root is about 51 GB. The later fixed Omni-Fake source contains 49.75 GB of Parquet shards and also needs room for 100,000 extracted source images. Use a 250 GB one-shot Pod volume for the repository, both datasets, Hugging Face cache, checkpoints/models, and logs. A persistent network volume is unnecessary for this one job.

Before transfer, the authenticated local RunPod operator writes `benchmark/candidates/prooflens-cf384-m5/runpod-provisioning-receipt.json`. It records the control-plane-observed Pod ID hash, `SECURE` cloud type, exact L40S product, creation time, and the authorized eight-hour workload-stop time. RunPod Pods do not expose a provider-enforced TTL or auto-stop field. The executable controls are the trainer's absolute deadline plus an authenticated operator stop after evidence retrieval or at the deadline. The receipt is an operator-recorded control-plane observation, not cryptographic provider attestation. The Pod process must independently match its hashed `RUNPOD_POD_ID` and GPU to that receipt; a caller-set environment variable alone cannot satisfy the gate.

Inside the Pod:

```bash
cd /workspace/prooflens
python -m pip install --disable-pip-version-check --require-hashes -r benchmark/m5/runpod-requirements.txt
export M5_PROTOCOL_COMMIT="$(git rev-parse HEAD)"
npm run benchmark:m5:preflight
```

The preflight verifies one complete 64-image training batch, runs CUDA/bfloat16 forward and backward passes, clips gradients, performs one optimizer step, exports an FP32 ONNX model, and checks PyTorch/ONNX parity. It does not open the selector, either regression split, or H3. Only after its canonical receipt says `preflight-pass`, run:

```bash
npm run benchmark:m5:train
```

The two package commands expand to the frozen arguments below; the full command is:

```bash
python benchmark/m5/train_gpu.py \
  --protocol-commit "$M5_PROTOCOL_COMMIT" \
  --data-root benchmark/data/m4-head \
  --train-manifest benchmark/evidence/m4/train-manifest.jsonl.gz \
  --selector-manifest benchmark/evidence/m4/validation-manifest.jsonl \
  --output-dir benchmark/candidates/prooflens-cf384-m5
```

The command fails before training unless the tracked tree is clean, the canonical provisioning receipt and successful one-batch preflight both exist, every source and pixel hash matches, CUDA exposes exactly one NVIDIA L40S with at least 45 GB, the pinned Hugging Face files match, the packaged M2 model matches its lock, and initial PyTorch/ONNX parity passes.

The run is resumable only inside the same exact Pod/environment: `run-marker.json` binds the protocol, manifests, initial model, provisioning receipt, absolute paid deadline, and hashed Pod identity. Model/optimizer state and candidate checkpoints are written through fsync/replace, followed by a digest seal written last. Resume requires a contiguous sealed history; one complete unsealed next epoch may be reconstructed and sealed, while gaps, extras, digest changes, and partial files fail closed. The trainer's absolute workload-stop time is checked before every training or selection batch and cumulative paid time is persisted across processes. The authenticated operator remains responsible for stopping the Pod because RunPod exposes no provider-side TTL.

Retrieve the complete candidate directory after train-select. If no candidate passes, publish only the failure receipt. If a candidate passes, publish `selection-lock.json` as the sole child change and wait for exact-head public CI before terminal regression evaluation.

Terminal regressions run in order and cannot select another candidate or threshold:

```bash
export M5_LOCK_COMMIT="$(git rev-parse HEAD)"
python benchmark/m5/evaluate_locked.py \
  --lock-commit "$M5_LOCK_COMMIT" \
  --selection-lock benchmark/evidence/m5/selection-lock.json \
  --m3-data-root benchmark/data/m3-head \
  --m3-manifest benchmark/evidence/m3/validation-manifest.jsonl \
  --m2-data-root benchmark/data/m2-head \
  --m2-manifest benchmark/evidence/m2/validation-manifest.jsonl \
  --output-dir benchmark/candidates/prooflens-cf384-m5
```

The evaluator writes terminal state after M3 before it opens M2. A failure stops immediately. Neither command accepts an H3 path.

After both regressions pass, materialize the fixed Omni-Fake source without opening the selected model:

```bash
export M5_LOCK_COMMIT="$(git rev-parse HEAD)"
npm run benchmark:m5:lock-large-synthetic
```

This verifies the pinned `JamalLee/Omni-Fake-SET` revision, all 71 Image-config Parquet LFS hashes (49,751,776,056 bytes), keeps only `full_synthetic`, excludes generator families already represented directly in M4 when named, and rejects historical/training/selector/regression/H3-metadata overlap by ID, encoded-byte SHA-256, and dHash distance at most 8. It deterministically chooses exactly 100,000 generator-stratified rows and publishes only the compressed pixel-free manifest, 1,000×100 batch assignment, attribution, and source lock. Source pixels remain ignored under `benchmark/data/m5-large-synthetic`.

Commit and push that exact four-file packet as the direct child of the selection lock. Wait for its exact-head public gate, then score it once:

```bash
export M5_LARGE_SOURCE_LOCK_COMMIT="$(git rev-parse HEAD)"
npm run benchmark:m5:evaluate-large-synthetic
```

The evaluator uses the already locked ONNX bytes and raw threshold, recomputes each batch decision from the stored float32 logits, reports all 1,000 batch recalls plus overall/Wilson/per-generator results, and passes only when both the unrounded mean and median batch recall are strictly greater than `0.95`. A failed panel is consumed and cannot be moved into training or selection.

If and only if the selector, both terminal regressions, and the 100,000-image panel pass, publish the exact local model and evidence transaction:

```bash
npm run benchmark:m5:finalize
```

The finalizer independently recomputes both regression packets from their stored float32 logits and frozen manifests, validates the 100,000-image receipt, rechecks the selected ONNX bytes and browser fixtures, stages exactly the declared final paths, and writes the finalization receipt last. H3 remains untouched and is not claimed by this transaction.

## Cost and cleanup

The paid wall-clock stop is eight L40S hours. Transfer, dependency, one-batch, and parity failures are fail-fast and do not authorize a different GPU or an altered training recipe. After candidate/evidence retrieval and local hash verification, terminate the Pod rather than leaving it stopped with billable storage.

The shipped extension remains local and offline regardless of the training host. RunPod is used only to fit the fixed model; it is never a runtime dependency.
