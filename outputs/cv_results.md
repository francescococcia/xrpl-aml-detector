# Grouped 5-fold CV (GroupKFold by scenario_id, NaN -> unique; seed 42)

| model | mean AP x100 | std | pooled OOF AP |
|---|---|---|---|
| logreg_starter | 28.25 | 8.75 | 27.86 |
| logreg_all | 51.71 | 8.90 | 52.20 |
| lightgbm_all | 53.27 | 6.97 | 53.26 |

Best: **lightgbm_all**

## Breakdown (OOF scores, best vs starter baseline)

| slice                         | name              |   n_pos |    AP |   AP_baseline |
|:------------------------------|:------------------|--------:|------:|--------------:|
| family (within-family AP)     | aggregate_relay   |     102 |  0.87 |          1.56 |
| family (within-family AP)     | bipartite         |     134 | 63.25 |         69.15 |
| family (within-family AP)     | chain             |      73 | 82.8  |         60    |
| family (within-family AP)     | cycle             |      78 | 82.45 |         72.79 |
| family (within-family AP)     | distributed_relay |      98 |  1.1  |          1.07 |
| family (within-family AP)     | fan_in            |     101 | 72.38 |         85.63 |
| family (within-family AP)     | fan_out           |      92 | 68.76 |         77.64 |
| family (within-family AP)     | fan_out_in        |     178 | 92.92 |         85.35 |
| family (within-family AP)     | multi_stage       |     173 | 82.17 |         79.17 |
| family (within-family AP)     | random            |      80 | 54.28 |         54.02 |
| family (within-family AP)     | stack             |     116 | 71.97 |         74.44 |
| difficulty (vs all negatives) | easy              |     800 | 71.7  |         36.23 |
| difficulty (vs all negatives) | medium            |     225 |  3.78 |          0.92 |
| difficulty (vs all negatives) | hard              |     225 |  0.65 |          0.97 |

## Top 25 features (LightGBM gain %, full train)

|                      |   gain_pct |
|:---------------------|-----------:|
| pair_total           |       8.65 |
| s_out_cnt_10m        |       4.15 |
| fwd_ratio_10m        |       4.09 |
| r_in_cnt_10m         |       3.91 |
| pair_share_of_sender |       2.78 |
| s_in_best_sim_1h     |       2.34 |
| r_fwd_best_sim_1h    |       1.94 |
| r_burst_10m          |       1.86 |
| s_hold_before        |       1.74 |
| fund_ratio_10m       |       1.69 |
| s_out_sum_10m        |       1.53 |
| s_ep_lag_med         |       1.44 |
| s_w_median_hold      |       1.26 |
| s_burst_10m          |       1.25 |
| r_w_in_uniq          |       1.23 |
| r_min_to_forward     |       1.1  |
| s_ep_lr_std          |       1.08 |
| s_out_uniq_6h        |       1.08 |
| s_in_before_sum_6h   |       1.07 |
| s_out_cnt_6h         |       1.05 |
| r_clustering         |       1.02 |
| hour                 |       1    |
| s_clustering         |       0.98 |
| r_in_cnt_6h          |       0.98 |
| r_in_cnt_1h          |       0.94 |

Runtime: 78s, 175 features

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
