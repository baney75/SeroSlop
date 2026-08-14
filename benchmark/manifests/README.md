# Frozen benchmark manifests

These pixel-free JSONL files record public dataset IDs and revisions, reconstruction paths, source-byte SHA-256 values, class labels, generator or real-source names, and prompt groups. Pixels remain outside Git.

| File | Rows | Role | SHA-256 |
|---|---:|---|---|
| `train.jsonl` | 3,600 | modern stratum inside the 103,600-image head-training corpus | `03b88b3804244018fbdf532b2b7d451db91dad7c3229c9013ee4ede9fa798015` |
| `validation.jsonl` | 600 | model and threshold selection only | `41be10ef876ecef0635744ed29677a1888a7759cc8060dc7a392f76f83ab263b` |
| `test.jsonl` | 600 | one-time confirmatory evaluation | `28e9d70698c1ec2f7692241fc29f961f32d01551c4a18ffa56f22c2188bfa5ae` |
| `web-negative.jsonl` | 319 | real-image false-positive challenge | `ad8b3f30a37feb3b6b046683db2d4071e236e6878612c7d8733869699d7f7824` |

Training, validation, and test are generator-version, real-source, prompt-group, ID, and byte disjoint. `web-negative.jsonl` intentionally shares the test’s 300 Library of Congress rows and adds 19 expert-created Chartography rows; it is not independent of the confirmatory real-side sample.

`selection.json` freezes modern-source revisions and selection counts. Attribution and review files preserve Open Images, DOCCI, Library of Congress, Chartography, legacy-exclusion, and perceptual-overlap evidence. `parity-ids.json` pins the deterministic 30-real/30-synthetic post-score Chrome diagnostic.

Run `npm run check:benchmark` for the zero-download integrity audit. See `BENCHMARK.md` for the predeclared scoring order, confidence methods, gates, and reconstruction limits.
