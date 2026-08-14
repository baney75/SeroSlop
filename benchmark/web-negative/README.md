# Web-negative challenge

This frozen, evaluation-only challenge measures false positives on real visual
content that differs from the detector's photo training sources:

- 300 public-domain Library of Congress FSA/OWI color photographs created from
  1939–1944 transparencies, frozen with exact catalog IDs, image hashes,
  photographer/rights provenance, and row-level visual review;
- all 19 `Expert-created` Chartography charts, whose pinned dataset card grants
  CC BY 4.0. Third-party `Sourced online` Chartography rows are excluded.

No challenge pixel or result participates in training, model selection,
threshold selection, or calibration. Pixels stay outside Git. The pixel-free
plan and human review are committed before scoring; the exact manifest,
predictions, and aggregate false-positive result are committed after the final
model is frozen. The 300 historical photographs also form the real half of the
source-disjoint confirmatory test; the synthetic half is 300 prompt-disjoint
images from a test-only generator family.

```bash
benchmark/.venv/bin/python benchmark/web-negative/prepare.py --phase plan --replan
# Review every planned row, then mark only verified rows include in:
# benchmark/manifests/web-negative-review.json
benchmark/.venv/bin/python benchmark/web-negative/prepare.py --phase materialize
```
