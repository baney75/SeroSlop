# M6 score-blind protocol

M6 is a fresh RunPod-only experiment, starting from terminal M5 HEAD
`76d0a807dcf240245830b8510e623d838e43cd4c`. It preserves the local 384px
FP32 ONNX contract and does not alter M5 artifacts.

This packet is P-only metadata protocol work. It records the source-reported
SET shard census (train 19 shards/10,447,263,801 bytes; validation 47
shards/37,645,535,846 bytes) and OOD test census (19 shards/16,521,804,450
bytes). No S materialization, selector inference, training, scoring, network
download, or RunPod provisioning surface is present. `benchmark/m6/prepare.py
--phase census` only validates this local metadata contract.

Training uses only clean Omni-Fake-SET train rows (real/full_synthetic), two
predeclared last-six branches, and four snapshots. Omni-Fake-OOD is
evaluation-only. Selector rows are from SET validation and are excluded from
the 100K evaluation. Tampered rows, H3 pixels, legacy pixels, and teacher
anchors are forbidden.

Dataset cards report CC-BY-4.0, but this protocol makes no independent
constituent-rights warranty. Dataset pixels are not shipped and no endorsement
or comparative/best-in-class claim is made. Freshness claims are item-level.
