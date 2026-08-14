# Large-head evidence packet

This directory contains the pixel-free evidence for the finalized 103,600-image classifier-head training run. Dataset pixels are intentionally excluded from Git.

The five compressed source artifacts are deterministic `gzip -n -9` copies of the ignored materialization outputs. `selection-summary.json` binds their expanded SHA-256 values to the checked-in recipe, source metadata, archive locks, attribution, rejects, evaluation exclusions, and reviewed perceptual-match decisions.

The directory also contains the unchanged training summary, calibration, 25-candidate grid, and classifier-only ONNX comparison. Fresh extraction used a run-bound marker: caches from earlier runs were rejected, while an interrupted release run could resume only atomically saved shards carrying the same marker and current model/manifest/preprocessing hashes. The packaged model at `weights/prooflens-cf384.onnx` is published only after this packet passes.

Run the complete fail-closed audit with:

```bash
npm run check:large-training-evidence
```

That command expands and hashes every compressed artifact, validates all 103,600 manifest rows and 50,000 Open Images attributions, recomputes the cross-split near-duplicate pair set, checks training coverage and validation gates, and requires the packaged ONNX to differ from its pinned upstream artifact only in `classifier.weight` and `classifier.bias`.
