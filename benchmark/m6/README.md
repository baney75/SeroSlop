# M6 score-blind protocol

## P7 Phase 1 + TASTE verification

P7 currently provides Phase 1 input verification and a real TASTE byte-rederiving adapter. AIGen, X-AIGD, Nano Banana, source-lock, training, and acceptance remain disabled. It does not assert independent origin or rights clearance. Run with a verified local cache:

```bash
PYTHONNOUSERSITE=1 benchmark/.venv/bin/python -m benchmark.m6.p7_operational --phase materialize-frontier --source taste --cache-root benchmark/data/m6-frontier-cache/taste --output /tmp/p7-taste-receipts
```

M6 is a fresh RunPod-only experiment, starting from terminal M5 HEAD
`76d0a807dcf240245830b8510e623d838e43cd4c`. It preserves the local 384px
FP32 ONNX contract and does not alter M5 artifacts.

The immutable public P commit is
`3b29ea2f9e1ad46e4cd78f47c9ccf5fe3a99877e`. This direct append-only recovery
child corrects one P wording error and supplies the deterministic metadata
source-lock core. It does not rewrite P history and is not S authorization.

Public materializer recovery P3 is
`fa9f002a2f9805b59d7955bf4c4f9992bbfb22ce`. Its first quality run was red
only because the generic GitHub verification environment did not install
`pyarrow`, so four Parquet fixture tests could not import their declared test
dependency. The direct append-only CI recovery adds `pyarrow==20.0.0` to that
verification environment and changes no recipe, source inventory,
materializer, model, data, threshold, or evaluation behavior. P3 remains
immutable history; paid or pixel work still requires the recovery child itself
to be exact-head public-green.

P records the source-reported
SET shard census (train 19 shards/10,447,263,801 bytes; validation 47
shards/37,645,535,846 bytes) and OOD test census (19 shards/16,521,804,450
bytes). This recovery still performs no selector inference, training, scoring,
network download, pixel materialization, or RunPod provisioning. The source-lock
builder accepts only already-materialized metadata facts, requires complete
H3/M2-M5 comparator receipts, constructs the fixed train/selector/evaluation
assignments, and writes a non-overwriting atomic candidate packet. A later
public S checker must independently re-open the source and historical evidence
before that packet is authoritative.

The corrected overlap boundary is explicit: canonical identity and encoded
image bytes SHA-256 apply to every fresh and historical row. Decoded
EXIF-oriented RGB SHA-256 is available and compared only for fresh rows.
EXIF-oriented RGB dHash distance at most eight is applied wherever the source
metadata recorded a dHash; 600 M2 validation rows legitimately have no recorded
dHash and remain protected by identity and encoded-byte comparisons. Historical
decoded hashes are unavailable, and neither missing historical field is
reconstructed from pixels. H3 pixels remain unread. The original P phrase
`decoded image SHA256`, and P2's overbroad historical dHash wording, are
superseded only by the direct append-only recovery lineage.

The pinned fresh-source inventory is `source-shards.json` (85 image Parquet
shards, 64,614,604,097 bytes; SHA-256
`a86c7209e76248edddd61537f397379194a7aaa908405e0cede7c8f5a3d7fbfe`).
`materialize.py` verifies every shard's LFS SHA-256 before decoding, records
encoded bytes SHA-256, and defines decoded RGB SHA-256 as SHA-256 over the
namespace `seroslop-m6-decoded-rgb-v1\0`, big-endian 32-bit width and height,
and the EXIF-transposed RGB bytes. Its dHash matches the M4 algorithm exactly.
The controlled download phase requests only the 85 pinned dataset paths at the
exact revisions with `token=False`, then re-verifies a regular non-symlink file,
its byte count, and LFS SHA-256 before publishing canonical download receipts.
Materialization decodes from the same verified file descriptor, derives row IDs
from dataset, revision, shard path, and shard-local ordinal, and publishes each
fragment plus receipt as one fsync-bound atomic directory. Resume re-derives the
complete fragment from the pinned Parquet bytes; a matching sidecar alone is not
trusted. It reads fresh Omni-Fake pixels only; H3 and M2-M5 pixels are never
opened.

`historical.py` independently re-opens the fixed committed metadata artifacts
and normalizes 442,780 comparator rows: H3 600, M2 106,878, M3 108,978, M4
113,162, and M5 113,162. M5's rows deliberately bind the M4 manifests reused
by M5 plus the terminal M5 selector-failure receipt. The authoritative
source-lock CLI no longer accepts a caller-authored history bundle; it derives
history from those fixed repository artifacts and fresh facts from the verified
materialization cache.

P3 remains protocol recovery until its exact public commit is green. Only then,
from the physical repository root and the isolated benchmark environment, run:

```bash
PYTHONNOUSERSITE=1 benchmark/.venv/bin/python -m benchmark.m6.materialize \
  --phase download --cache-root benchmark/data/m6-shards
PYTHONNOUSERSITE=1 benchmark/.venv/bin/python -m benchmark.m6.materialize \
  --phase materialize --cache-root benchmark/data/m6-shards \
  --output-root benchmark/data/m6-materialized --workers 8
PYTHONNOUSERSITE=1 benchmark/.venv/bin/python -m benchmark.m6.prepare \
  --phase source-lock --cache-root benchmark/data/m6-shards \
  --materialized-root benchmark/data/m6-materialized \
  --output benchmark/evidence/m6/source-lock-candidate
```

The download phase reads no image pixels. The materialize and source-lock
commands are forbidden before public-green P3, and S is not authoritative until
its later exact-path public checker and authorization both pass.

Training uses only clean Omni-Fake-SET train rows (real/full_synthetic), two
predeclared last-six branches, and four snapshots. Omni-Fake-OOD is
evaluation-only. Selector rows are from SET validation and are excluded from
the 100K evaluation. Tampered rows, H3 pixels, legacy pixels, and teacher
anchors are forbidden.

Dataset cards report CC-BY-4.0, but this protocol makes no independent
constituent-rights warranty. Dataset pixels are not shipped and no endorsement
or comparative/best-in-class claim is made. Freshness claims are item-level.

## P5 public-green recovery packet

P5 freezes admission, identity, source-round-robin allocation, seven-view
transform seeds, Platt calibration, the score-65-inclusive decision, ranking,
acceptance, provenance, and paid-receipt contracts in `p5_protocol.py` and
`p5-protocol.json`. It is metadata-only: no candidate payload is downloaded or
materialized, and selector, inference, training, and provisioning remain
forbidden. Every production allocation, scoring, training, acceptance, and
paid-receipt entry point deliberately hard-stops in P5. P6 must pin its exact
public commit/tree and independently validated admission ledger, historical
receipt, overlap manifest, allocation receipt, and source-lock receipt before
wrapping the deterministic P5 cores. The X-AIGD labeled-test member is pinned to
`f86630ae51ef1103de204c879ad74d70bacaeca258489f2c32102851344a5c75`.
The quota census in `p5-quota-census.json` supersedes the provisional panel:
SET validation synthetic 58,228 and OOD synthetic 28,693, yielding exactly
100,000 synthetic items after overlap quarantine.

P5 commit `c878c2dc7ecbb49edb1cac4395aa20649471a330` is retained as
immutable public-red history. Its protocol logic passed locally, but the first
public run exposed that the tiny transform fixture hashed PNG container bytes;
the same Pillow version can produce different compressed PNG bytes across
platform builds even when the transformed RGB pixels are identical. The direct
CI recovery changes no transform, view, data, model, threshold, quota, or paid
boundary. It hashes a canonical domain-tagged width/height/RGB payload instead,
then updates only the lineage checker and this explanation. P6 remains blocked
until that exact recovery child is public-green.
## P8 frontier adapters

P8 adds fail-closed, local-cache adapters for AIGenImages2026, X-AIGD, and
Nano-Banana. The public adapter interface accepts only `--cache-root`,
`--output`, and a source selector; it does not accept caller metadata, rows,
or image bytes. Each adapter reopens the pinned physical container itself,
binds decoded image facts to source/card/container evidence, atomically writes
canonical receipts, and strictly reopens those receipts before returning.

P8 is still publisher-assertion-only and unverified. No source lock, training,
rights, commercial-use, or acceptance authority is implied. AIGen is streamed
without extraction; X-AIGD and Nano-Banana read embedded Parquet image bytes.
The large source runs are intentionally not part of this repository’s tests.
