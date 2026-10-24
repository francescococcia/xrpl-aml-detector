# Grouped 5-fold CV (GroupKFold by scenario_id, NaN -> unique; seed 42)

| model | mean AP x100 | std | pooled OOF AP | precision@top1% | recall@top1% |
|---|---|---|---|---|---|
| logreg_starter | 28.25 | 8.75 | 27.86 | 0.538 | 0.174 |
| logreg_all | 55.96 | 5.48 | 55.18 | 0.860 | 0.293 |
| lightgbm_v1 | 53.27 | 6.97 | 53.26 | 0.781 | 0.264 |
| lightgbm_flow | 58.10 | 5.35 | 57.74 | 0.852 | 0.293 |
| hgb_flow | 58.88 | 5.51 | 58.46 | 0.871 | 0.299 |
| blend_v2 | 59.46 | 5.82 | 58.00 | 0.875 | 0.301 |

Per-fold AP x100 (same folds for every model):

| model          |   fold0 |   fold1 |   fold2 |   fold3 |   fold4 |
|:---------------|--------:|--------:|--------:|--------:|--------:|
| logreg_starter |   26.55 |   19.21 |   41.56 |   34.64 |   19.3  |
| logreg_all     |   53.97 |   59.43 |   62.22 |   57.8  |   46.4  |
| lightgbm_v1    |   52.26 |   53.65 |   62.31 |   56.95 |   41.16 |
| lightgbm_flow  |   56.41 |   61.93 |   65.22 |   57.45 |   49.48 |
| hgb_flow       |   56.89 |   63.09 |   66.1  |   58.25 |   50.07 |
| blend_v2       |   56.79 |   64.71 |   66.68 |   58.75 |   50.39 |

Top-1% metrics are computed per validation fold (top 1% of that fold's txs) and averaged.

Adopted: **blend_v2** (v1 = lightgbm_v1 = phase-1 submission)

## Breakdown (OOF scores: adopted v2 vs v1 vs starter baseline)

| slice                         | name              |   n_pos |    AP |   AP_v1 |   AP_baseline |
|:------------------------------|:------------------|--------:|------:|--------:|--------------:|
| family (within-family AP)     | aggregate_relay   |     102 |  1.02 |    0.87 |          1.56 |
| family (within-family AP)     | bipartite         |     134 | 63.98 |   63.25 |         69.15 |
| family (within-family AP)     | chain             |      73 | 73.46 |   82.8  |         60    |
| family (within-family AP)     | cycle             |      78 | 85.45 |   82.45 |         72.79 |
| family (within-family AP)     | distributed_relay |      98 |  1.11 |    1.1  |          1.07 |
| family (within-family AP)     | fan_in            |     101 | 89.47 |   72.38 |         85.63 |
| family (within-family AP)     | fan_out           |      92 | 82.53 |   68.76 |         77.64 |
| family (within-family AP)     | fan_out_in        |     178 | 83.21 |   92.92 |         85.35 |
| family (within-family AP)     | multi_stage       |     173 | 86.45 |   82.17 |         79.17 |
| family (within-family AP)     | random            |      80 | 66.67 |   54.28 |         54.02 |
| family (within-family AP)     | stack             |     116 | 80.51 |   71.97 |         74.44 |
| difficulty (vs all negatives) | easy              |     800 | 78.29 |   71.7  |         36.23 |
| difficulty (vs all negatives) | medium            |     225 |  6.27 |    3.78 |          0.92 |
| difficulty (vs all negatives) | hard              |     225 |  0.75 |    0.65 |          0.97 |

## Top 25 features (lightgbm_flow gain %, full train)

|                      |   gain_pct |
|:---------------------|-----------:|
| fl120_size           |      11.04 |
| fl120_density        |       7.09 |
| pair_total           |       5.77 |
| fl600_rate           |       3.43 |
| fl600_v_std          |       3.1  |
| fl600_density        |       3.01 |
| pair_share_of_sender |       2.51 |
| fl120_nw             |       2.26 |
| fl600_dur            |       1.38 |
| fl120_dur            |       1.36 |
| flf600_rate          |       1.1  |
| s_w_median_hold      |       1.02 |
| s_ep_lr_mean         |       0.98 |
| s_in_best_sim_1h     |       0.98 |
| fl600_size           |       0.97 |
| r_fwd_best_sim_1h    |       0.96 |
| fl600_ns             |       0.91 |
| r_clustering         |       0.86 |
| s_out_cnt_6h         |       0.85 |
| s_ep_lr_std          |       0.83 |
| s_ep_lag_med         |       0.83 |
| r_w_in_uniq          |       0.81 |
| r_in_sum_1h          |       0.78 |
| r_in_cnt_6h          |       0.76 |
| s_in_before_sum_6h   |       0.76 |

Runtime: 92s, 235 features (175 phase-1 + 60 flow)

<!-- hand-written notes below are kept on regeneration -->

## Phase 2: relay hub features and stage-2 stacking (same GroupKFold folds)

| model | mean AP x100 | std | verdict |
|---|---|---|---|
| lightgbm_all (stage-1, submitted) | **53.27** | 6.97 | kept |
| + 47 hub relay features | 50.78 | 6.66 | dropped |
| stage-2, neighbour aggregates only (strict nested OOF) | 44.25 | 7.10 | dropped |
| stage-2, base + neighbour aggregates (strict nested OOF) | 51.95 | 7.60 | dropped |
| rank blend 0.7 stage-1 + 0.3 stage-2 | 53.79 | 7.06 | dropped (inside noise, 2x runtime) |

Within-family AP (stage-1 / +hub / stage-2 base+agg / stage-2 agg-only):

| family | stage-1 | +hub | s2 base+agg | s2 agg-only |
|---|---|---|---|---|
| aggregate_relay | 0.87 | 0.92 | 0.83 | 0.95 |
| bipartite | 63.25 | 60.51 | 57.64 | 61.71 |
| chain | 82.80 | 80.80 | 81.73 | 78.33 |
| cycle | 82.45 | 81.86 | 82.23 | 77.71 |
| distributed_relay | 1.10 | 0.98 | 1.35 | 1.27 |
| fan_in | 72.38 | 70.74 | 71.11 | 63.98 |
| fan_out | 68.76 | 63.62 | 63.77 | 66.21 |
| fan_out_in | 92.92 | 94.24 | 91.95 | 91.95 |
| multi_stage | 82.17 | 79.70 | 82.36 | 77.17 |
| random | 54.28 | 53.40 | 49.46 | 48.59 |
| stack | 71.97 | 70.94 | 69.28 | 74.27 |

Difficulty (vs all negatives): stage-1 easy 71.70 / medium 3.78 / hard 0.65; +hub 68.47 / 3.12 / 0.64; stage-2 base+agg 69.89 / 4.02 / 0.63; stage-2 agg-only 59.28 / 2.37 / 0.71.

Robustness: StratifiedGroupKFold(5) grouped by sender gives 69.12 ± 3.68 (folds 68.09, 64.46, 73.49, 66.24, 73.33). Scenario grouping (53.27) is the stricter estimate. The wallet graph is a single connected component, so component grouping is impossible.

Reproduce: `python experiments/hub_cv.py`, `python experiments/stacking_cv.py` (~2.5 min), `python experiments/group_robustness.py` (all take `--train-dir/--cache-dir`).


## Phase 3 (v2): flow reconstruction + 3-model rank blend

Same folds as above (GroupKFold(5, shuffle, seed 42) by scenario_id). The table at the top is regenerated by `src/train.py`.

| | mean AP | std | per-fold AP | P@1% | R@1% |
|---|---|---|---|---|---|
| before: lightgbm_v1 | 53.27 | 6.97 | 52.26, 53.65, 62.31, 56.95, 41.16 | 0.781 | 0.264 |
| after: blend_v2 (adopted) | **59.46** | 5.82 | 56.79, 64.71, 66.68, 58.75, 50.39 | **0.875** | **0.301** |

The gain is +6.19 AP, better in 5/5 folds (+4.5 / +11.1 / +4.4 / +1.8 / +9.2).

Adopted:
- 60 label-free flow features (`src/flows.py`): connected components of payments linked by consecutive activity on a shared wallet. Undirected windows are 30/120/600 s; forward (inflow → next outflow) windows are 60/600/3600 s.
- Equal-weight rank average of lightgbm_flow (58.10), hgb_flow (58.88) and logreg_all (55.96).

Tried and not adopted (scratch harness, same folds; float32, so the base is 52.83 rather than 53.27):

| idea | AP | verdict |
|---|---|---|
| LGB tuning, no flows (leaves 7/15/63, min_child 50/100, colsample 0.3, lr 0.01 x 1800, is_unbalance) | 50.59–53.86 | noise; is_unbalance −1.5 |
| flows: forward windows only | 53.11 | undirected windows needed |
| flows: + 15 s / 60 s / 1200 s windows, pass-through & activity-share stats | 56.88–58.58 | no gain over the adopted set (58.25 in harness) |
| post-hoc smoothing with the flow's mean/max score | +0.3–0.5 | too small |
| flow-group mean/max of the top base features | 57.44–58.54 | no gain |
| LGB + HGB only | 58.74 | below the 3-model blend in 4/5 folds |
| LGB + HGB + 0.5·LR | 59.50 | same as equal weights; kept the simpler one |
| transductive train+test wallet features | — | train/test share 0 of 700 wallets each |
| pseudo-labelling | — | not tried (time) |

Weak spots for v2 vs v1: chain (82.8 → 73.5) and fan_out_in (92.9 → 83.2). Aggregate and distributed relays are still near-random (~1 AP), as in phase 2.
