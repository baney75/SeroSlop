# Frozen benchmark selection

These files preserve the exact pixel-free selection used to train, calibrate, and test ProofLens. Each JSONL row records a public dataset ID and revision, split, relative reconstruction path, SHA-256 of the source image bytes, class label, and source family. No image pixels are committed.

| File | Rows | SHA-256 |
|---|---:|---|
| `train.jsonl` | 3,600 | `8207d373a6abec32e4a687361199de43568fe982a5ff2e045987d3a029b85626` |
| `validation.jsonl` | 600 | `651704a6d3fb86b9324f13bea721c40854535925ce33b49b8749978ee2bed915` |
| `test.jsonl` | 600 | `cd4d09fbb59d695ebc0cc4dc96f0dd17caea9e0e8865d7658c6100ef723e977f` |

`selection.json` freezes source revisions and selection counts. `docci-attribution.json` preserves the 1,200 selected DOCCI train example IDs. `parity-ids.json` pins the 30-real/30-synthetic browser diagnostic subset.

Run `npm run check:benchmark` for the zero-download integrity and metric audit. See the root `BENCHMARK.md` for full pixel reconstruction, sealed evaluation, bootstrap, and browser-parity commands.
