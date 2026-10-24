# XRPL AML detector — Decode & Detect: The XRPL AI Hack

Scores every XRPL payment with a 0–1 money-laundering suspicion score.
Metric: 100 × average precision (transaction level).

## Layout
```
starter/payments.py   organiser helper (load_payments) — imported, not rewritten
src/features.py       per-transaction graph + transfer-leg features (175) + flow features; v3 uses 247 columns
src/flows.py          label-free flow reconstruction (time-bounded shared-wallet components, 60 features + 12 gap-regularity
                      features) and the component ids used for post-hoc smoothing
src/train.py          GroupKFold(5) CV: starter, v1, the v2 blend and the v3 pipeline; per-fold, top-1%, AP beyond top 1%,
                      per-family/difficulty AP
src/predict.py        v3: weighted LightGBM (5 seeds) + HGB + logistic regression, rank blend 2:1:1, flow max-score smoothing,
                      tie-breaker; writes + validates outputs/predictions_v3.csv and outputs/predictions.csv (no ties)
src/explain.py        worked alerts: flow summary + top-3 LightGBM pred_contrib reasons -> outputs/example_alerts.md
src/common.py         CLI args, default paths, LightGBM/HGB params, rank helper, training weights, smoothing,
                      optional feature cache (features_v5_*.parquet)
outputs/              cv_results.md, predictions.csv (= predictions_v3.csv, current), predictions_v2.csv and predictions_v1.csv
                      (earlier submissions, never overwritten), example_alerts.md (3 worked alerts)
make_zip.py           builds the submission ZIP (git HEAD + predictions + CV results + SUBMISSION.md with commit hash)
tests/                smoke tests on synthetic payments (no competition data needed)
experiments/          dropped phase-2 experiments (hub relay features, stage-2 stacking, CV-grouping check); not used by the pipeline
knowledge/approach.md study guide (XRPL, AML typologies, features, metric) with sources
knowledge/pitch-brief.md   product / impact / security brief and 2-min pitch script (current numbers)
knowledge/training-pack.md background training pack (XRPL, AML, metric), corrected for this data
docs/method_report.md|pdf  4-page methodology report
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
# final v3 predictions -> outputs/predictions_v3.csv + outputs/predictions.csv (validated: one row per test tx, scores in [0,1], <12 MiB)
python src/predict.py
# worked alerts with explanations -> outputs/example_alerts.md (needs outputs/predictions.csv; ~5 s with a warm cache)
python src/explain.py
# smoke tests (no data needed)
python -m unittest discover -s tests
# submission ZIP -> ../follow-the-drops-submission.zip (commit first: it packs git HEAD)
python make_zip.py
```

Leaderboard probes (all off by default; plain `python src/predict.py` reproduces the submitted v3 run bit for bit).
A probe writes `outputs/predictions_<tag>.csv` + `outputs/run_<tag>.json` and never touches `predictions.csv`:
```bash
python src/predict.py --cache-dir /tmp/xrpl-cache --lam 0.1                       # smoothing weight
python src/predict.py --cache-dir /tmp/xrpl-cache --smooth-sets 600u,3600f,120u   # extra flow set for smoothing
python src/predict.py --cache-dir /tmp/xrpl-cache --weights 1,1,0                 # blend LGB,HGB,LR
python src/predict.py --cache-dir /tmp/xrpl-cache --rank-features                 # per-split percentile features
python src/predict.py --cache-dir /tmp/xrpl-cache --pseudo 400                    # pseudo-label top 400 / bottom 50% of test
python src/predict.py --cache-dir /tmp/xrpl-cache --drop time,legs               # drop feature groups (common.FEATURE_GROUPS)
python src/predict.py --cache-dir /tmp/xrpl-cache --smooth-agg mean --lam 1       # flow mean instead of max
# make the best probe the final submission (writes predictions.csv + outputs/final_settings.json for make_zip.py)
python src/predict.py --cache-dir /tmp/xrpl-cache <best switches> --as-final
```
These switches were tuned on the public leaderboard, not in grouped CV (`src/train.py` covers the default only).

Custom paths / feature cache (keep the cache outside the repo):
```bash
python src/train.py   --train-dir /path/to/train --cache-dir /tmp/xrpl-cache
python src/predict.py --train-dir /path/to/train --test-dir /path/to/test --cache-dir /tmp/xrpl-cache --out-dir outputs
python src/explain.py --train-dir /path/to/train --test-dir /path/to/test --cache-dir /tmp/xrpl-cache --out-dir outputs
```
Cold run (no cache), 8 cores: train.py ~186 s (incl. train features; also re-runs v1/v2 for comparison), predict.py ~63 s (incl. test features); full pipeline ~4.2 min (measured 2026-10-24, v3).

CV (scenario-grouped 5-fold AP x100): v1 53.27 ± 6.97 → v2 59.46 ± 5.82 → v3 62.22 ± 4.83 (v3 better than v2 in 5/5 folds); see `outputs/cv_results.md`.

Optional experiments (documented in `outputs/cv_results.md` and `knowledge/approach.md` §7):
```bash
python experiments/diagnose.py     --train-dir /path/to/train --test-dir /path/to/test --cache-dir /tmp/xrpl-cache  # ~5 min: where AP is lost -> outputs/diagnose.md
python experiments/hub_cv.py          --train-dir /path/to/train --cache-dir /tmp/xrpl-cache   # ~1 min
python experiments/stacking_cv.py     --train-dir /path/to/train --cache-dir /tmp/xrpl-cache   # ~2.5 min
python experiments/group_robustness.py --train-dir /path/to/train --cache-dir /tmp/xrpl-cache  # ~1 min
```

## Notes
- Labels' `family`, `difficulty`, `scenario_id`, `cohort_id` are never used as features;
  `scenario_id` (NaN → unique id) is only the CV grouping key.
- Final smoothing weight is λ = 0.25 (`smooth_scores` in `src/common.py`; public LB 64.5 vs 64.4 at λ 0.5).
- Seeds fixed (42); the final LightGBM is an average of seeds 42–46, and HGB/logistic regression use random_state 42.
- Old caches (`features_v2_*` to `features_v4_*`) are ignored; the current cache is `features_v5_*` (v4 + a `deliv_asset_code`
  that means the same currency in every split; v4 numbered currencies per split with `pd.factorize`).
- v3 training gives sample_weight 0 to positives of family aggregate_relay, distributed_relay or ambiguous; `family` is used
  only for this weight, never as a feature. Smoothing in CV uses same-fold scores only (no leakage across folds).

## Team
Follow the Drops: Francesco Coccia (captain, sole member).

## Licence and attribution
MIT, see `LICENSE`. Third-party components, libraries and AI assistance are listed in `ATTRIBUTION.md`.
