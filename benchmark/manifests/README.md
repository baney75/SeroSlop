# Frozen benchmark manifests

These pixel-free JSONL files record public dataset IDs and revisions, reconstruction paths, source-byte SHA-256 values, class labels, generator or real-source names, and prompt groups. Pixels remain outside Git.

| File | Rows | Role | SHA-256 |
|---|---:|---|---|
| `train.jsonl` | 3,600 | modern stratum inside the 103,600-image head-training corpus | `03b88b3804244018fbdf532b2b7d451db91dad7c3229c9013ee4ede9fa798015` |
| `validation.jsonl` | 600 | model and threshold selection only | `41be10ef876ecef0635744ed29677a1888a7759cc8060dc7a392f76f83ab263b` |
| `test.jsonl` | 600 | consumed v1 confirmation; numeric-contract failure, never acceptance evidence | `28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae` |
| `web-negative.jsonl` | 319 | frozen v1 challenge; inference never started | `ad8b3f30a37feb3b6b046683db2d4071e236e6878612c7d8733869699d7f7824` |
| `test-v2.jsonl` | 600 | unscored replacement confirmation: 300 StockImages + 300 Infinity | `773128e53fc3d82ca802cc1571809975e96d4583e1ed66d9a98767f8d1a43da8` |
| `web-negative-v2.jsonl` | 319 | unscored StockImages false-positive slice; separate rows, same real source corpus | `6a1287bae6826811c81cbebab79a1bc6abb475fde70c9aa1529c390ed97014c9` |

Training, validation, and the historical v1 test are generator-version, real-source, prompt-group, ID, and byte disjoint. Historical `web-negative.jsonl` intentionally shares v1’s 300 Library of Congress rows and adds 19 Chartography rows.

Replacement-v2 excludes all 106,019 historical training/evaluation IDs, bytes, source groups, and dHashes. Its confirmation and false-positive manifests share no row or byte. Both real slices come from StockImages-CC0, so the false-positive slice is not an independent source-population estimate.

`selection.json` freezes the historical modern split. The replacement selection, attribution, perceptual review, and complete historical dHash index use the `replacement-v2-*` and `historical-perceptual-*` files. `parity-ids-v2.json` pins the prediction-blind 30-StockImages/30-Infinity browser diagnostic before replacement scoring; the earlier `parity-ids.json` remains historical.

Run `npm run check:benchmark` for the zero-download integrity audit. See `BENCHMARK.md` for the predeclared scoring order, confidence methods, gates, and reconstruction limits.
