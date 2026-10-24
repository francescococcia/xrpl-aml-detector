# Decisions
- 10:55 Project folder `xrpl-aml-detector` created; data kept outside repo.
- 11:22 Adopted LightGBM on ~190 graph features (CV AP 53.3 vs 28.3 starter baseline). Relays still at random level.
- v2 Flow reconstruction (60 features) + LGB/HGB/LR rank blend: CV 59.5, LB 63.5.
- v3 Relay/ambiguous weight 0, 12 gap features, 2:1:1 blend, flow max-score smoothing, tie-breaker: CV 62.2, LB 64.4.
- 13:30 Audit fixes: `deliv_asset_code` is now the same code for the same currency in every split (pd.factorize numbered
  currencies per split, so e.g. EUR/USD codes could swap between train and test); cache bumped to features_v5; LightGBM
  n_jobs = cpu count; ATTRIBUTION rewritten with licences + AI disclosure; make_zip.py; smoke tests. Needs a re-run of
  train.py + predict.py to refresh CV and predictions (expected change: small, feature not in top 25).

## Leaderboard log (24 Oct)
- v1 57.4 | v2 63.5 | v3 64.4 | v3_smooth025 64.5 (best, 13:27 BST)
- 13:29 top 3: coldsoup 72.2, Model Phi 71.3, LedgerLine 71.1 (same top-1% profile as ours: ~94% P, 30% R)

