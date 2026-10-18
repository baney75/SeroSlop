# SeroSlop

SeroSlop is a Manifest V3 Chrome extension that shows a local AI-image score beside images on webpages, using one packaged ONNX model; people load it unpacked in desktop Chrome 121+, and a static site in `site/` ships on GitHub Pages.

## Run and check

- Install: `npm ci`, plus a Python 3.12 venv at `benchmark/.venv` from `benchmark/verify-requirements.txt` (the pinned `onnxruntime` has no wheel for newer Python).
- Check: `npm run check` (runs `npm run verify:static`: lint, typecheck, vitest, contributor, benchmark-evidence and stage-policy checks, ONNX inspection, reproducible build, package, site and store-asset checks). Run it with the venv's `bin/` first on `PATH`. CI runs `npm run verify:static` the same way.
- Live proof: for a change to scoring badges, the popup or setup tab, run `npm run build`, load `dist/` unpacked in Chrome, finish setup (**Offline ready**), pick a mode and confirm a score appears on a test page. `npm run test:chrome` automates this smoke test. If you can't run them, report "untested live" and give these steps.
- Deploy: merging a `site/**` change to `main` runs `.github/workflows/pages.yml`; a `main` build also publishes a nightly prerelease via `.github/workflows/release.yml`. Don't run either by hand.

## Brand

- SeroSlop's own design rules are in `DESIGN.md`: neutral near-black, white and gray; red is the accent; use the slashed-zero mark in `src/static/seroslop.svg`. Site styles are in `site/styles.css`. It is not a Tyto, Rome or Martyr project.

## Layout

- `src/`: extension code (`background.ts`, `content.ts`, `popup.ts`, `offscreen.ts`, `inference/`, `shared/`, `static/` for manifest, HTML and icons).
- `tests/`: vitest unit tests and `tests/fixtures/`.
- `scripts/`: build, release and `check-*` verification scripts; `scripts/run-static-verification.mjs` picks the stage checks.
- `benchmark/`: training, evaluation and frozen evidence. Evidence under `benchmark/evidence/` is append-only.
- `contributor/`: the separate contributor extension and its tests.
- `site/`: the public Pages site.
- `store/`: Chrome Web Store assets.
- `weights/`: the packaged ONNX model, pinned by `model-lock.json`.
- `docs/`: acceptance notes and README images.

## Hard limits

- Consumed evaluation packets and frozen evidence in `benchmark/evidence/` stay unchanged. The stage-policy checks guard the append-only lineage.
- No detector API, localhost process, telemetry or post-install model download (`PRIVACY.md`).
- Historical `prooflens` names stay in paths and evidence identifiers so receipts reproduce.

## References

- `README.md`, `DESIGN.md`, `BENCHMARK.md`, `MODEL_CARD.md`, `SECURITY.md`.
