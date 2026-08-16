# M5 RunPod GPU fine-tuning

## Source-recovery and authorization sequence

The append-only public history retains every failed launch boundary: the
platform-specific CI assertion, incomplete Python lineage map, scrubbed
`RUNPOD_POD_ID`, and Python 3.11 audit accumulation. Each failed packet stopped
before training, selector scoring, terminal regression scoring, or any H3 read.
The public A4 receipt at
`f3d86077cf5e7a124d09b593d69e9a1769d7e295` preserves the completed fixes.

The first A4 RunPod preflight then passed source, identity, CUDA, L40S, data,
model-load, and runtime gates but stopped at the frozen `2e-4` initial-model
parity gate. A score-blind diagnostic on the same 16 training images proved
that ONNX Runtime CUDA's default TF32 policy caused the mismatch: the maximum
PyTorch-CPU versus ORT-CUDA error was `0.028517723083496094`, while the same
ORT session with `use_tf32=0` was `0.0000247955322265625`. CPU ONNX parity also
passed. The tolerance is not relaxed.

Publish the exact 17-path R5 source-recovery child of A4. R5 freezes the
diagnostic bytes, requires ORT CUDA with TF32 disabled for every model score,
checks real-image export parity, and ranks candidates from each exported ONNX
model instead of a bfloat16 PyTorch proxy. It does not change the training
data, optimizer, loss, candidate grid, selector gates, terminal regressions,
or H3 boundary. Wait for exact-head green CI. From the clean public-green R5
commit, run:

```bash
/usr/bin/env -i PATH=/usr/bin:/bin /usr/bin/python3 -I -c '
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1]); raw = p.read_bytes()
if hashlib.sha256(raw).hexdigest() != sys.argv[2]: raise SystemExit("M5 pre-exec bootstrap bytes changed")
sys.argv = [str(p), *sys.argv[3:]]
exec(compile(raw, str(p), "exec"), {"__name__": "__main__", "__file__": str(p)})
' scripts/m5-preexec-bootstrap.py 54d94c8e696b9accb7bae4de6427922c1c72975b105b0a35ce0f74e741dead6d authorize
git add benchmark/evidence/m5/parity-recovery-authorization.json
git commit -m "Evidence: authorize exact M5 parity recovery"
```

The inline system-Python loader verifies the stdlib-only pre-exec bootstrap before
any repository source runs. That bootstrap verifies the complete HEAD/index/worktree,
then pins the operator Mac's absolute Node executable, version, and SHA-256 and
starts it with a minimal environment. Do not use `npm`, an inherited `PATH`, or a
different Node binary for this one-file writer.

The schema-5 A5 authorization binds A4, the exact R5 commit/tree/17-path source
map and public CI, the immutable diagnostic SHA-256, the packaged-M2 reference
boundary, ONNX scoring, score blindness, and no H3 read. A5 must add only that
receipt. Push it and wait for its own exact-head green CI before retrying paid
preflight. Later selection, failure, 100K, and final stages must retain the
complete public history through R5 and A5.

## Accepted brief

- Outcome: produce one fully local FP32 ViT-S/384 ONNX detector that is no larger than 90,000,000 bytes and runs through the extension's existing WASM and WebGPU paths.
- Audience: Chrome users and public reviewers who need an auditable local detector, not a cloud service.
- Deliverables: a six-candidate RunPod training packet, a one-file public selection lock, terminal M3/M2 regression evidence, the final packaged model, Chrome/offline evidence, and an untouched final H3 evaluation.
- Non-goals: universal perfection, a calibrated probability claim, H3-guided selection, selector rows in gradients, or a larger teacher model in the extension.
- Authority: one bounded paid RunPod L40S job is authorized. Training may write only ignored `benchmark/candidates/prooflens-cf384-m5`; publication is a separate fixed-head Git transaction.
- Proofs: exact source/model hashes, RunPod L40S/CUDA receipt, complete candidate grid, embedded selector logits, exhaustive thresholds, zero observed selector false positives, public pre-regression lock, ordered terminal regressions, a separate 100,000-image synthetic-recall panel, ONNX parity, Chrome offline E2E, and four final critical reviews.
- Stop conditions: no selector-feasible candidate; a terminal regression failure; any H3 read; a changed source/model/runtime; model size above 90 MB; non-L40S or non-RunPod training; or 24 paid GPU hours without a completed train-select packet.

The zero-false-positive gate means exactly zero observed false positives among the same 300 fresh British Library base images in each of the four declared views. A two-sided 95% Wilson interval is reported separately for every view; at 0/300 its upper bound is 1.2643%. The four transformed views are correlated and are never pooled as 0/1,200. This is an observed-sample gate and binomial-model reference interval, not proof that errors are impossible on future images.

After training and terminal regressions, a separate public source lock freezes at least 100,000 synthetic images. The panel has no detected ID, SHA-256, or dHash-distance-at-most-8 overlap with the fixed SeroSlop training, selector, regression, H3-metadata, or historical exclusion packets, and its rows are prohibited from M2-M5 gradients and candidate selection. The public lock contains no model scores and precedes the repository evaluation receipt; the operator attests that first inference occurs after that lock, while Git history cannot prove the absence of private prior scoring or upstream training exposure. Deterministic SHA-256 ordering divides the panel into 1,000 batches of 100. At the already locked model and raw threshold, both the mean and median per-batch synthetic recall must be strictly above 95/100. If the model misses either aggregate, those rows become consumed development evidence and a later revision needs new training data and a different fixed panel.

## Frozen model and data

M5 starts from the official `buildborderless/CommunityForensics-DeepfakeDet-ViT` PyTorch checkpoint at revision `ac6ee457bea904a373065754107451793b56db00`, then replaces its classifier with the exact shipped M2 classifier before fine-tuning. A 16-image preflight must keep PyTorch versus packaged-M2 ONNX logits within `2e-4`, with ORT CUDA TF32 explicitly disabled. Each candidate is exported first, checked against a nonzero image, and scored for selection through that exact ONNX artifact and provider policy.

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

The repository must be cloned at the exact public M5 A5 parity authorization commit. The launcher derives and verifies the full append-only lineage, rejects unexpected tracked, untracked, or ignored executable surfaces before Python starts, requires public-green exact-head CI, and starts Python in isolated mode; no caller can nominate a protocol commit or Python entry point. The pre-exec bootstrap reads the bounded PID-1 environment only in `runpod` mode, validates exactly one `RUNPOD_POD_ID`, and forwards only that provider variable into the otherwise fixed child environment. It never forwards `RUNPOD_API_KEY` or any neighboring environment record. Transfer only `benchmark/data/m4-head` for train-select. Do not transfer any H3 root. The training root is about 51 GB. The later fixed Omni-Fake source contains 49.75 GB of Parquet shards and also needs room for 100,000 extracted source images. Use a temporary 300 GB network volume for the repository, datasets, Hugging Face cache, checkpoints/models, and logs; delete it after all evidence is retrieved and verified.

The pre-exec gate detects source drift present when a command starts. It is not
a sandbox against a hostile same-user process changing files after verification;
the dedicated operator-controlled Pod must have no concurrent repository writer.

Before transfer, the authenticated local RunPod operator writes `benchmark/candidates/prooflens-cf384-m5/runpod-provisioning-receipt.json`. It records the control-plane-observed Pod ID hash, `SECURE` cloud type, exact L40S product, creation time, and the authorized 24-hour workload-stop time. RunPod Pods do not expose a provider-enforced TTL or auto-stop field. The executable controls are the trainer's absolute deadline plus an authenticated operator stop after evidence retrieval or at the deadline. The authenticated operator attests that the job runs on a RunPod Secure Cloud L40S. The runtime Pod-ID hash and locally observed L40S/CUDA facts must match that operator-authored record. This is consistency evidence, not provider-signed proof of RunPod identity.

Inside the Pod, use the fixed Conda interpreter from the pinned image. The first
RunPod launch downloads Node.js `v24.18.1` only from the official Node release
URL, requires the 31,525,884-byte archive SHA-256
`d6c664df3f3f61458e8c277585571328522d705166723a7c7823a9253a4d15a0`,
verifies the extracted Node executable SHA-256
`f3432a45b03b2da0d270095fdd8813dc34cbea73f5fc8b18c7a384b7cf9b333a`,
and then uses that exact binary for every M5 command. The outer command clears
shell and Node preload variables before Node starts; the inner launcher also
requires `/opt/conda/bin/python` and starts it with `-I`.

```bash
cd /workspace/prooflens
m5_preexec() {
  /opt/conda/bin/python -I -c '
import hashlib, pathlib, sys
p = pathlib.Path(sys.argv[1]); raw = p.read_bytes()
if hashlib.sha256(raw).hexdigest() != sys.argv[2]: raise SystemExit("M5 pre-exec bootstrap bytes changed")
sys.argv = [str(p), *sys.argv[3:]]
exec(compile(raw, str(p), "exec"), {"__name__": "__main__", "__file__": str(p)})
' scripts/m5-preexec-bootstrap.py 54d94c8e696b9accb7bae4de6427922c1c72975b105b0a35ce0f74e741dead6d "$@"
}
m5_preexec runpod-install
m5_preexec runpod preflight -- \
  --data-root benchmark/data/m4-head \
  --train-manifest benchmark/evidence/m4/train-manifest.jsonl.gz \
  --selector-manifest benchmark/evidence/m4/validation-manifest.jsonl \
  --output-dir benchmark/candidates/prooflens-cf384-m5 \
  --preflight-only
```

The preflight verifies one complete 64-image training batch, runs CUDA/bfloat16 forward and backward passes, clips gradients, performs one optimizer step, exports an FP32 ONNX model, and checks PyTorch/ONNX parity. It does not open the selector, either regression split, or H3. Only after its canonical receipt says `preflight-pass`, run:

```bash
m5_preexec runpod train -- \
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
m5_preexec runpod regress -- \
  --lock-commit @HEAD \
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
m5_preexec runpod lock-large-synthetic -- \
  --lock-commit @HEAD --allow-download
```

This verifies the pinned `JamalLee/Omni-Fake-SET` revision, all 71 Image-config Parquet LFS hashes (49,751,776,056 bytes), keeps only `full_synthetic`, excludes generator families already represented directly in M4 when named, and rejects historical/training/selector/regression/H3-metadata overlap by ID, encoded-byte SHA-256, and dHash distance at most 8. It deterministically chooses exactly 100,000 generator-stratified rows and publishes only the compressed pixel-free manifest, 1,000×100 batch assignment, attribution, and source lock. Source pixels remain ignored under `benchmark/data/m5-large-synthetic`.

Commit and push that exact four-file packet as the direct child of the selection lock. Wait for its exact-head public gate, then score it once:

```bash
m5_preexec runpod evaluate-large-synthetic -- \
  --source-lock-commit @HEAD
```

The evaluator uses the already locked ONNX bytes and raw threshold, recomputes each batch decision from the stored float32 logits, reports all 1,000 batch recalls plus overall/Wilson/per-generator results, and passes only when both the unrounded mean and median batch recall are strictly greater than `0.95`. A failed panel is consumed and cannot be moved into training or selection.

If and only if the selector, both terminal regressions, and the 100,000-image panel pass, publish the exact local model and evidence transaction:

```bash
m5_preexec runpod finalize --
```

The finalizer independently recomputes both regression packets from their stored float32 logits and frozen manifests, validates the 100,000-image receipt, rechecks the selected ONNX bytes and browser fixtures, stages exactly the declared final paths, and writes the finalization receipt last. H3 remains untouched and is not claimed by this transaction.

## Recovery and authorization lineage

The public numeric receipt is A4. A source-only R5 recovery must be its exact
single child with the frozen 17-row surface, and may not contain A5. Only the
receipt-only `benchmark/evidence/m5/parity-recovery-authorization.json` A5
child is an active authorization. The A5 receipt binds protocol P2, the exact
R5 source map, successful public CI, the packaged-M2 parity boundary, and the
initial diagnostic SHA; training remains score-blind and H3-free.

## Cost and cleanup

The paid wall-clock stop is 24 L40S hours, with a five-minute trainer safety margin. Transfer, dependency, one-batch, and parity failures are fail-fast and do not authorize a different GPU or an altered training recipe. After candidate/evidence retrieval and local hash verification, terminate the Pod and delete the temporary network volume rather than leaving billable resources behind.

The shipped extension remains local and offline regardless of the training host. RunPod is used only to fit the fixed model; it is never a runtime dependency.
