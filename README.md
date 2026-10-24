# XRPL AML detector — Decode & Detect: The XRPL AI Hack

Scores every XRPL payment with a 0–1 money-laundering suspicion score.
Metric: 100 × average precision (transaction level).

## Layout
```
starter/payments.py   organiser helper (load_payments) — imported, not rewritten
src/features.py       per-transaction graph + transfer-leg features (~190 features)
src/train.py          GroupKFold(5) CV: starter logistic regression vs LightGBM, per-family/difficulty AP
src/predict.py        train on all train data, score test, write + validate outputs/predictions.csv
src/common.py         CLI args, default paths, LightGBM params, optional feature cache
outputs/              cv_results.md, predictions.csv
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
# final predictions -> outputs/predictions.csv (validated: one row per test tx, scores in [0,1])
python src/predict.py
```

Custom paths / feature cache (keep the cache outside the repo):
```bash
python src/train.py   --train-dir /path/to/train --cache-dir /tmp/xrpl-cache
python src/predict.py --train-dir /path/to/train --test-dir /path/to/test --cache-dir /tmp/xrpl-cache --out-dir outputs
```
Cold run (no cache), 8 cores: train.py ~73 s (incl. train features), predict.py ~42 s (incl. test features); full pipeline ~2 min.

## Notes
- Labels' `family`, `difficulty`, `scenario_id`, `cohort_id` are never used as features;
  `scenario_id` (NaN → unique id) is only the CV grouping key.
- Seeds fixed (42); LightGBM final model is an average of seeds 42–46.

## Licence and attribution
MIT, see `LICENSE`. Third-party components, libraries and AI assistance are listed in `ATTRIBUTION.md`.
