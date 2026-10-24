# XRPL AML detector — Decode & Detect: The XRPL AI Hack

Scores every XRPL payment with a 0–1 money-laundering suspicion score.
Metric: 100 × average precision (transaction level).

## Layout
```
starter/payments.py   organiser helper (load_payments) — imported, not rewritten
src/features.py       per-transaction graph + transfer-leg features (~190) + flow features
src/flows.py          label-free flow reconstruction (time-bounded shared-wallet components, 60 features)
src/train.py          GroupKFold(5) CV: starter, v1 (phase-1 LightGBM), flow models and the v2 blend; per-fold, top-1%, per-family/difficulty AP
src/predict.py        v2: LightGBM (5 seeds) + HistGradientBoosting + logistic regression, equal rank average;
                      writes + validates outputs/predictions_v2.csv and outputs/predictions.csv
src/common.py         CLI args, default paths, LightGBM/HGB params, rank helper, optional feature cache (features_v3_*.parquet)
outputs/              cv_results.md, predictions.csv (= predictions_v2.csv, current), predictions_v1.csv (phase-1 submission, never overwritten)
experiments/          dropped phase-2 experiments (hub relay features, stage-2 stacking, CV-grouping check); not used by the pipeline
knowledge/approach.md study guide (XRPL, AML typologies, features, metric) with sources
```

## Setup (Python 3.13)
```bash
pip install -r requirements.txt
```

## Run
Default data paths (relative to the repo root):
`../decode-and-detect-training-data/train` and `../decode-and-detect-test-data`.

```bash
# cross-validation report -> outputs/cv_results.md
python src/train.py
# final v2 predictions -> outputs/predictions_v2.csv + outputs/predictions.csv (validated: one row per test tx, scores in [0,1], <12 MiB)
python src/predict.py
```

Custom paths / feature cache (keep the cache outside the repo):
```bash
python src/train.py   --train-dir /path/to/train --cache-dir /tmp/xrpl-cache
python src/predict.py --train-dir /path/to/train --test-dir /path/to/test --cache-dir /tmp/xrpl-cache --out-dir outputs
```
Cold run (no cache), 8 cores: train.py ~93 s (incl. train features), predict.py ~57 s (incl. test features); full pipeline ~2.5 min (measured 2026-10-24, v2).

CV (scenario-grouped 5-fold AP x100): v1 53.27 ± 6.97 → v2 59.46 ± 5.82, better in 5/5 folds; see `outputs/cv_results.md`.

Optional experiments (documented in `outputs/cv_results.md` and `knowledge/approach.md` §7):
```bash
python experiments/hub_cv.py          --train-dir /path/to/train --cache-dir /tmp/xrpl-cache   # ~1 min
python experiments/stacking_cv.py     --train-dir /path/to/train --cache-dir /tmp/xrpl-cache   # ~2.5 min
python experiments/group_robustness.py --train-dir /path/to/train --cache-dir /tmp/xrpl-cache  # ~1 min
```

## Notes
- Labels' `family`, `difficulty`, `scenario_id`, `cohort_id` are never used as features;
  `scenario_id` (NaN → unique id) is only the CV grouping key.
- Seeds fixed (42); the final LightGBM is an average of seeds 42–46, and HGB/logistic regression use random_state 42.
- Old caches (`features_v2_*`) are ignored; flow features are in `features_v3_*`.

## Licence and attribution
MIT, see `LICENSE`. Third-party components, libraries and AI assistance are listed in `ATTRIBUTION.md`.
