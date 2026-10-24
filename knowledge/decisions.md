# Decisions
- 10:55 Project folder `xrpl-aml-detector` created; data kept outside repo.
- 11:22 Adopted LightGBM on ~190 graph features (CV AP 53.3 vs 28.3 starter baseline). Relays still at random level.

## Leaderboard log (24 Oct)
- v1 57.4 | v2 63.5 | v3 64.4 | v3_smooth025 64.5 (best, 13:27 BST)
- 13:29 top 3: coldsoup 72.2, Model Phi 71.3, LedgerLine 71.1 (same top-1% profile as ours: ~94% P, 30% R)

