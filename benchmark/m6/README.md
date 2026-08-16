# M6 score-blind protocol

M6 is a fresh RunPod-only experiment, starting from terminal M5 HEAD
`76d0a807dcf240245830b8510e623d838e43cd4c`. It preserves the local 384px
FP32 ONNX contract and does not alter M5 artifacts.

The immutable public P commit is
`3b29ea2f9e1ad46e4cd78f47c9ccf5fe3a99877e`. This direct append-only recovery
child corrects one P wording error and supplies the deterministic metadata
source-lock core. It does not rewrite P history and is not S authorization.

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

The corrected overlap boundary is explicit: canonical identity, encoded image
bytes SHA-256, and EXIF-oriented RGB dHash distance at most eight apply to both
fresh and historical metadata. Decoded EXIF-oriented RGB SHA-256 is available
and compared only for fresh rows. Historical decoded hashes are unavailable;
H3 pixels remain unread. The original P phrase `decoded image SHA256` did not
truthfully describe historical metadata and is superseded only by this direct
recovery lineage.

Training uses only clean Omni-Fake-SET train rows (real/full_synthetic), two
predeclared last-six branches, and four snapshots. Omni-Fake-OOD is
evaluation-only. Selector rows are from SET validation and are excluded from
the 100K evaluation. Tampered rows, H3 pixels, legacy pixels, and teacher
anchors are forbidden.

Dataset cards report CC-BY-4.0, but this protocol makes no independent
constituent-rights warranty. Dataset pixels are not shipped and no endorsement
or comparative/best-in-class claim is made. Freshness claims are item-level.
