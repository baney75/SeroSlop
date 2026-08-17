---
language:
- en
license: mit
pretty_name: TASTE — Human Preferences for Design-Quality Image Comparison
size_categories:
- 10K<n<100K
task_categories:
- image-classification
- image-to-text
tags:
- preference-learning
- vision-language
- design
- aesthetics
- reward-model
configs:
- config_name: rankings_with_images
  default: true
  data_files:
  - split: train
    path: rankings_with_images.parquet
- config_name: hallucinations_with_images
  data_files:
  - split: train
    path: hallucinations_with_images.parquet
- config_name: prompts
  data_files:
  - split: train
    path: prompts.parquet
- config_name: assets
  data_files:
  - split: train
    path: assets.parquet
- config_name: rankings
  data_files:
  - split: train
    path: rankings.parquet
- config_name: hallucinations
  data_files:
  - split: train
    path: hallucinations.parquet
- config_name: evaluators
  data_files:
  - split: train
    path: evaluators.parquet
---

# TASTE: Human Preferences for Design-Quality Image Comparison

This dataset is the human-evaluation corpus released alongside the
**TASTE** preference model.  It contains panel rankings of generated
images across multiple quality dimensions — both *aesthetic* (does the
image look good?) and *description-faithfulness* (does the image match
what the prompt describes?) — plus a per-image hallucination
judgement.

## Quick stats

| Table                         | Rows  | Notes                                       |
|-------------------------------|-------|---------------------------------------------|
| `prompts.parquet`             |  ~200 | one row per unique prompt                   |
| `assets.parquet`              |   ~1k | one row per generated image                 |
| `rankings.parquet`            |  ~50k | long-format: (track, dimension, evaluator)  |
| `hallucinations.parquet`      |   ~3k | per-evaluator binary judgements             |
| `evaluators.parquet`          |  ~10  | one row per anonymised evaluator            |

## Files

```
data_hf/
├── README.md                              ← this file (HF dataset card)
├── LICENSE
│
├── ── canonical (normalised) tables ────────────────────────────────
├── prompts.parquet                        ← prompt_id, track, dimension, prompt_id_src, prompt_text
├── assets.parquet                         ← asset_id, model, image_url, image_path, track
├── rankings.parquet                       ← long-format ranking votes (joinable via ids)
├── hallucinations.parquet                 ← per-image binary hallucination votes
├── evaluators.parquet                     ← anonymised evaluator metadata
│
├── ── browseable (denormalised) views ──────────────────────────────
├── rankings_with_images.parquet           ← one row per vote, joined with prompt text + image
├── hallucinations_with_images.parquet     ← one row per halluc judgement, joined likewise
│
├── images/                                ← one file per asset, named <asset_id>.<ext>
│   ├── 0000001.png
│   ├── 0000002.png
│   └── ...
```

## Browsing on the Hub

The HF Hub dataset viewer treats each parquet file as a separate config (we
declare them explicitly in the YAML frontmatter).  Use the config dropdown
in HF Studio to switch between:

- **`rankings_with_images`** (default) — one row per (evaluator, dimension,
  ranked image), with `prompt_text` and the rendered `image_path` inline.
  This is the right view for *seeing the data*.
- **`hallucinations_with_images`** — same idea for the binary hallucination
  judgements, joined with a representative prompt per scene.
- **`prompts`** / **`assets`** / **`rankings`** / **`hallucinations`** /
  **`evaluators`** — the canonical normalised tables.  Use these for joins
  in code; the views above are derived from them.

The two `*_with_images` parquets are convenience views built by the build
script — they're not the source of truth, just an HF-Studio-friendly
projection.  If you're loading from Python you almost always want the
canonical tables.

## Tracks and dimensions

The corpus is split into two **tracks** by which side of image quality
the evaluator was asked to judge.  Each prompt belongs to exactly one
track (prompt-id ranges are disjoint).

| Track          | Prompt content                                       | What evaluators judge                          |
|----------------|------------------------------------------------------|------------------------------------------------|
| `aesthetics`   | `User Intent` + an `Aesthetics` guidance section.    | Whether the image *looks good* on this axis.   |
| `descriptions` | `User Intent` + a literal `Description` of the scene.| Whether the image *faithfully realises* the description. |

The naming `ui+a` / `ui+d` in the raw filenames comes from this:
**ui** = user intent, **+a** = aesthetics guidance appended,
**+d** = description appended.

Each track is annotated on its own subset of ranking **dimensions**:

| Dimension              | aesthetics | descriptions | What it measures                              |
|------------------------|:----------:|:------------:|-----------------------------------------------|
| `preference`           | yes        | yes          | Overall preference (HPS-style head).          |
| `typography`           | yes        | yes          | Quality and legibility of typesetting.        |
| `color_harmony`        | yes        |  —           | Whether colors work together.                |
| `mood_and_color_tone`  | yes        |  —           | Atmosphere / palette match.                   |
| `visual_hierarchy`     | yes        |  —           | Whether the eye is guided correctly.          |
| `color_accuracy`       |  —         | yes          | Faithfulness to colors specified in prompt.  |
| `spatial_accuracy`     |  —         | yes          | Layout / placement correctness.               |

`preference` and `typography` are evaluated under both prompt regimes,
so you can directly study how added aesthetic vs descriptive guidance
shifts the same axis.  The remaining five dimensions are exclusive to
their track.

A separate hallucination judgement is collected per (asset, evaluator)
in both tracks — see `hallucinations.parquet`.

## Schemas

### Identifier strategy

`prompt_id` and `asset_id` in this release are **content-derived** —
sequential integers assigned over the unique
`(track, dimension, prompt_text)` and `image_url` values in the source
CSVs.  We do *not* reuse the upstream numeric `prompt_id` as a key,
because the raw data has at least one collision (`prompt_id 613` in
`ui+a_preference.csv` references two completely different scenes with
two distinct image URLs).  Both id columns are guaranteed unique under
the new scheme; the original numeric id is preserved as `prompt_id_src`
on `prompts.parquet` (and as `prompt_id_src` on `hallucinations.parquet`)
for provenance and cross-reference with the upstream system.

### `prompts.parquet`

| column         | type    | notes                                              |
|----------------|---------|----------------------------------------------------|
| `prompt_id`    | int64   | Derived; 1 per unique (track, dimension, prompt_text). |
| `track`        | enum    | `aesthetics` or `descriptions`.                    |
| `dimension`    | enum    | which criterion the evaluator was being asked to judge under |
| `prompt_id_src`| int64   | Provenance: the upstream CSV's numeric prompt_id (not unique). |
| `prompt_text`  | string  | Full multi-line evaluator-facing prompt.           |

Each row is the evaluator-facing prompt for that (track, dimension)
combination: `User Intent` plus a criterion-specific guidance section
(e.g. "Aesthetics: focus on color harmony…" for `color_harmony` vs
"Aesthetics: focus on typography…" for `typography`).  The User Intent
half is generally the same for a given scene, but the criterion-specific
half differs by `dimension`.

### `assets.parquet`

| column       | type    | notes                                              |
|--------------|---------|----------------------------------------------------|
| `asset_id`   | int64   | Derived; 1 per unique `image_url`.  Filename stem in `images/`. |
| `model`      | string  | Generator model name, verbatim.                    |
| `image_url`  | string  | Original (Cloudinary) URL — kept for provenance.   |
| `image_path` | string  | Relative to dataset root, e.g. `images/0000001.png`. |
| `track`      | enum    | `aesthetics` or `descriptions`.                    |


### `rankings.parquet` (long format)

| column                | type    | notes                                       |
|-----------------------|---------|---------------------------------------------|
| `eval_round_stage_id` | int64   | Passthrough; identifies an evaluation session. |
| `dimension`           | enum    | one of the dimensions above                 |
| `track`               | enum    | `aesthetics` / `descriptions`               |
| `prompt_id`           | int64   | FK                                          |
| `asset_id`            | int64   | FK                                          |
| `evaluator_id`        | string  | anonymised, e.g. `eval_037`                 |
| `rank`                | int8    | 1 (best) … 5 (worst) within the session     |

### `hallucinations.parquet` (long format)

| column                | type    | notes                                       |
|-----------------------|---------|---------------------------------------------|
| `track`               | enum    | `aesthetics` / `descriptions`               |
| `prompt_id_src`       | int64   | Upstream CSV's numeric prompt_id (provenance, not unique). |
| `asset_id`            | int64?  | FK → `assets.asset_id`; null when `(track, prompt_id_src, model)` is ambiguous (multiple distinct images) or unseen in any ranking CSV. |
| `evaluator_id`        | string  | anonymised                                  |
| `hallucination_value` | int8    | 0 / 1                                       |
| `hallucination_flag`  | string  | e.g. `"No Hallucination"`, `"Minor"`, etc.  |

### `evaluators.parquet`

| column           | type           | notes                              |
|------------------|----------------|------------------------------------|
| `evaluator_id`   | string         | `eval_001`, `eval_002`, …          |
| `tracks`         | list<string>   | tracks the evaluator labelled in   |
| `n_ranking_rows` | int64          | total ranking votes by this person |
| `n_halluc_rows`  | int64          | total hallucination votes          |

### `rankings_with_images.parquet` (browseable view)

A pre-joined denormalisation of `rankings` × `prompts` × `assets`,
intended for the HF Studio viewer.  Each row is one evaluator's
ranking vote, fully self-contained.

| column                | type    | notes                                       |
|-----------------------|---------|---------------------------------------------|
| `track`               | enum    | `aesthetics` / `descriptions`               |
| `dimension`           | enum    | the criterion this vote was scored under    |
| `prompt_text`         | string  | the full evaluator-facing prompt            |
| `model`               | string  | generator that produced the image           |
| `image_path`          | string  | rendered inline by HF Studio                |
| `evaluator_id`        | string  | anonymised                                  |
| `rank`                | int8    | 1 (best) … 5 (worst)                        |
| `eval_round_stage_id` | int64   | evaluation session id                       |
| `prompt_id`           | int64   | FK → `prompts.prompt_id` (round-trip)       |
| `asset_id`            | int64   | FK → `assets.asset_id` (round-trip)         |

### `hallucinations_with_images.parquet` (browseable view)

Same idea for the hallucination judgements, but **no `dimension`
column**: a halluc vote is collected once per `(evaluator, asset)` and
the regime under which it was made is identified by `track` alone
(`aesthetics` or `descriptions`), not by a sub-criterion.

`prompt_text` is a representative prompt for the scene (chosen
deterministically; the User Intent half is shared across criteria, only
the criterion-specific guidance differs).

| column                | type    | notes                                       |
|-----------------------|---------|---------------------------------------------|
| `track`               | enum    | `aesthetics` / `descriptions`               |
| `prompt_text`         | string  | representative prompt for the scene         |
| `model`               | string  |                                             |
| `image_path`          | string  | rendered inline by HF Studio                |
| `evaluator_id`        | string  | anonymised                                  |
| `hallucination_value` | int8    | 0 / 1                                       |
| `hallucination_flag`  | string  | `"No Hallucination"`, etc.                  |
| `prompt_id_src`       | int64   | upstream scene id                           |
| `asset_id`            | int64?  | FK → `assets.asset_id`; null when ambiguous |

## How to load (`datasets` library)

For browsing or quick training, the pre-joined view is one call:

```python
from datasets import load_dataset, Image

ds = (
    load_dataset("path/to/data_hf", name="rankings_with_images", split="train")
    .cast_column("image_path", Image())   # auto-loads PIL images
)
ds[0]   # {'track': 'aesthetics', 'dimension': 'color_harmony', 'prompt_text': …,
        #  'model': 'GPT Image 1.5', 'image_path': <PIL.Image>, 'evaluator_id': …,
        #  'rank': 1, …}
```

For analysis with joins, load the canonical tables:

```python
prompts = load_dataset(
    "parquet",
    data_files="data_hf/prompts.parquet",
    split="train",
)
assets = (
    load_dataset(
        "parquet",
        data_files="data_hf/assets.parquet",
        split="train",
    )
    .cast_column("image_path", Image())   # auto-loads the PIL image
)
rankings = load_dataset(
    "parquet",
    data_files="data_hf/rankings.parquet",
    split="train",
)

# example: get all aesthetics-track color_harmony votes
ch_aest = rankings.filter(
    lambda r: r["dimension"] == "color_harmony"
              and r["track"] == "aesthetics"
)
```

## How to derive pairwise battles

Each (evaluator, prompt_id, dimension, track) group contains five
ranked assets (rank 1..5).  To produce Bradley-Terry battles, pick any
two ranks (e.g. rank 1 vs rank 5 for clear pairs, or every pair for full
information):

```python
import pandas as pd

r = pd.read_parquet("data_hf/rankings.parquet")
pairs = (
    r.merge(r, on=["eval_round_stage_id", "evaluator_id", "prompt_id",
                   "dimension", "track"], suffixes=("_a", "_b"))
     .query("rank_a < rank_b")
)
# pairs[asset_id_a] is preferred over pairs[asset_id_b]
```

## Anonymisation

Evaluator real names have been replaced with opaque `eval_NNN` ids
(`eval_001`, `eval_002`, …).
If you have a legitimate need to re-identify, contact the authors.


## Licensing

The full release — annotations *and* images — is published under the
[**MIT License**](LICENSE).

Note on generated images: each upstream generator (GPT Image, FLUX,
Nano Banana, Seedream, …) has its own terms of service governing the
redistribution of model outputs.  By aggregating them under the MIT
license we are asserting our own permission to redistribute these
specific generations; downstream users should still consult the
relevant generator's TOS if they intend a use that goes beyond what
those terms allow.  If any generator's TOS prevents redistribution of
its outputs, drop those rows before re-publishing — `assets.parquet`
has a `model` column that makes this a one-line filter.

## Citation

```bibtex
SOON
```

## Contact

SOON
