# Grouped 5-fold CV (GroupKFold by scenario_id, NaN -> unique; seed 42)

| model | mean AP x100 | std | pooled OOF AP | precision@top1% | recall@top1% | AP beyond top1% |
|---|---|---|---|---|---|---|
| logreg_starter | 28.25 | 8.75 | 27.86 | 0.538 | 0.174 | 16.44 |
| lightgbm_v1 | 53.27 | 6.97 | 53.26 | 0.781 | 0.264 | 31.26 |
| lightgbm_flow | 58.10 | 5.35 | 57.74 | 0.852 | 0.293 | 36.48 |
| hgb_flow | 58.88 | 5.51 | 58.46 | 0.871 | 0.299 | 37.43 |
| logreg_all | 55.96 | 5.48 | 55.18 | 0.860 | 0.293 | 35.01 |
| lightgbm_v3 | 61.08 | 5.08 | 60.56 | 0.852 | 0.295 | 42.65 |
| hgb_v3 | 61.62 | 4.61 | 61.43 | 0.880 | 0.305 | 39.40 |
| logreg_v3 | 55.45 | 5.84 | 54.94 | 0.837 | 0.286 | 35.55 |
| blend_v2 | 59.46 | 5.82 | 58.00 | 0.875 | 0.301 | 37.26 |
| blend_v3_unsmoothed | 61.12 | 5.42 | 59.36 | 0.865 | 0.300 | 40.53 |
| v3 | 62.22 | 4.83 | 60.81 | 0.875 | 0.305 | 41.08 |

Per-fold AP x100 (same folds for every model):

| model               |   fold0 |   fold1 |   fold2 |   fold3 |   fold4 |
|:--------------------|--------:|--------:|--------:|--------:|--------:|
| logreg_starter      |   26.55 |   19.21 |   41.56 |   34.64 |   19.3  |
| lightgbm_v1         |   52.26 |   53.65 |   62.31 |   56.95 |   41.16 |
| lightgbm_flow       |   56.41 |   61.93 |   65.22 |   57.45 |   49.48 |
| hgb_flow            |   56.89 |   63.09 |   66.1  |   58.25 |   50.07 |
| logreg_all          |   53.97 |   59.43 |   62.22 |   57.8  |   46.4  |
| lightgbm_v3         |   59.56 |   66.89 |   66.37 |   59.4  |   53.19 |
| hgb_v3              |   60.06 |   66.74 |   66    |   61.31 |   53.98 |
| logreg_v3           |   54.13 |   59.46 |   61.3  |   57.58 |   44.77 |
| blend_v2            |   56.79 |   64.71 |   66.68 |   58.75 |   50.39 |
| blend_v3_unsmoothed |   59.24 |   67.15 |   66.84 |   59.74 |   52.62 |
| v3                  |   61.53 |   67.16 |   67.1  |   61.25 |   54.04 |

Top-1% metrics are computed per validation fold (top 1% of that fold's txs) and averaged.

Adopted: **v3** (v1 = lightgbm_v1, v2 = blend_v2 = phase-3 submission; v3 = 2:1:1 blend of the *_v3 models + same-fold flow max-score smoothing + tie-breaker)

## Breakdown (OOF scores: adopted v3 vs v2 vs v1 vs starter baseline)

| slice                         | name              |   n_pos |    AP |   AP_v2 |   AP_v1 |   AP_baseline |
|:------------------------------|:------------------|--------:|------:|--------:|--------:|--------------:|
| family (within-family AP)     | aggregate_relay   |     102 |  1.1  |    1.02 |    0.87 |          1.56 |
| family (within-family AP)     | bipartite         |     134 | 58.51 |   63.98 |   63.25 |         69.15 |
| family (within-family AP)     | chain             |      73 | 79.41 |   73.46 |   82.8  |         60    |
| family (within-family AP)     | cycle             |      78 | 86.33 |   85.45 |   82.45 |         72.79 |
| family (within-family AP)     | distributed_relay |      98 |  1.04 |    1.11 |    1.1  |          1.07 |
| family (within-family AP)     | fan_in            |     101 | 90.22 |   89.47 |   72.38 |         85.63 |
| family (within-family AP)     | fan_out           |      92 | 84.53 |   82.53 |   68.76 |         77.64 |
| family (within-family AP)     | fan_out_in        |     178 | 86.37 |   83.21 |   92.92 |         85.35 |
| family (within-family AP)     | multi_stage       |     173 | 89.39 |   86.45 |   82.17 |         79.17 |
| family (within-family AP)     | random            |      80 | 74.74 |   66.67 |   54.28 |         54.02 |
| family (within-family AP)     | stack             |     116 | 81.55 |   80.51 |   71.97 |         74.44 |
| difficulty (vs all negatives) | easy              |     800 | 81.68 |   78.29 |   71.7  |         36.23 |
| difficulty (vs all negatives) | medium            |     225 |  9.79 |    6.27 |    3.78 |          0.92 |
| difficulty (vs all negatives) | hard              |     225 |  0.69 |    0.75 |    0.65 |          0.97 |

## Top 25 features (lightgbm_v3 gain %, full train, weighted)

|                      |   gain_pct |
|:---------------------|-----------:|
| fl120_size           |      11.85 |
| pair_total           |       8.79 |
| fl120_nw             |       7.55 |
| fl120_density        |       6.39 |
| fl600_density        |       3.75 |
| fl600_rate           |       3.43 |
| fl600_dur            |       3.32 |
| fl600_gapcv          |       3.11 |
| pair_share_of_sender |       2.06 |
| pair_rank            |       1.85 |
| fl600_v_std          |       1.47 |
| fl120_dur            |       1.28 |
| s_in_best_sim_1h     |       1.21 |
| fl120_gapcv          |       1.12 |
| fl600_gapmed         |       1.08 |
| flf600_rate          |       1.03 |
| r_fwd_best_sim_1h    |       1.03 |
| s_out_cnt_6h         |       0.95 |
| s_in_before_sum_6h   |       0.91 |
| r_in_cnt_6h          |       0.83 |
| r_out_after_sum_6h   |       0.78 |
| fl600_size           |       0.71 |
| r_in_cnt_10m         |       0.6  |
| s_ep_lr_std          |       0.59 |
| s_out_cnt_10m        |       0.57 |

Runtime: 185s, 247 features (175 phase-1 + 72 flow incl. gap stats)

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


## Phase 4 (v3): training weights, gap features, flow smoothing, 2:1:1 blend

Same folds as above. Inputs: the parent's brief and an internal independent audit (not a published source), whose four
recommendations were each re-measured here.

| | mean AP | std | per-fold AP | P@1% | R@1% | AP beyond top 1% |
|---|---|---|---|---|---|---|
| before: blend_v2 | 59.46 | 5.82 | 56.79, 64.71, 66.68, 58.75, 50.39 | 0.875 | 0.301 | 37.26 |
| after: v3 (adopted) | **62.22** | **4.83** | 61.53, 67.16, 67.10, 61.25, 54.04 | 0.875 | **0.305** | **41.08** |

The gain is +2.76 AP, better in 5/5 folds (+4.7 / +2.5 / +0.4 / +2.5 / +3.7). Top-1% precision is unchanged; the gain is in
ranking below the top 1% (AP beyond top 1% +3.8), which is where the leaderboard gap was.

Adopted (all in `src/`):
1. Training `sample_weight` 0 for positives of family aggregate_relay, distributed_relay or ambiguous (family is used only for
   the weight, never as a feature; these rows are still scored). Applied to LGB, HGB and LR.
2. 12 gap-regularity features: log1p(median inter-payment gap) and gap CV in each of the 6 flow windows.
3. Rank blend LGB:HGB:LR = 2:1:1 (lightgbm_v3 61.08, hgb_v3 61.62, logreg_v3 55.45; blend_v3_unsmoothed 61.12).
4. Post-hoc smoothing: rank(score) + 0.5·rank(max score in the 600 s undirected component) + 0.5·rank(max score in the
   3600 s forward component). In CV the component max uses only same-fold OOF scores (key = component × fold), so no
   score comes from a model that saw the row's label. +1.10 AP over the unsmoothed blend.
5. Tie-breaker 1e-6·rank(LGB): the v2 CSV had 5,085 tied scores, the v3 CSV has 0 (written at 15 decimals; checked by `predict.py`).

Ablations (scratch harness, same folds; d = vs v2 59.46):

| variant | AP | note |
|---|---|---|
| weights + gaps, 1:1:1 | 60.97 | |
| … + smoothing | 62.21 | |
| weights + gaps, 2:1:1 | 61.12 | 5/5 folds |
| **2:1:1 + smoothing λ 0.5 (adopted)** | **62.22** | +2.76, 5/5 folds |
| λ 1.0 | 62.30 | noise |
| LGB+HGB only + smoothing | 62.41 | noise |
| 1:2:1 + smoothing | 62.44 | noise |
| adopted + 0.5·v1 member | 62.45 | noise |
| logreg with weights vs without | 55.45 vs 55.96 | weights slightly hurt LR alone; kept for consistency (blend still gains) |

Tried and not adopted:

| idea | AP | verdict |
|---|---|---|
| v1 LightGBM as 4th member of v2 (1.0 / 0.5 weight) | 59.40 / 59.62 | noise |
| chain-depth features (24 cols: back/forward depth, value-matched depth, predecessors, gap, value ratio, hop, 600/3600 s) | 59.47 | no gain |
| value-gated long flows (link only if value ratio ≈ 1) | AUC 0.60–0.65 alone | too weak, dropped |
| stage-2 on neighbour-score aggregates, cross-fold | +3.07 | **leaky**: neighbours' OOF scores came from models that saw this row's label |
| stage-2 on neighbour-score aggregates, fold-local | 56.74 alone | hurts the blend, dropped |
| medium-tier analysis | — | medium scenarios last ~1–8 h with median gap 1,660 s vs 2,108 s for normal wallet activity; not separable with our features |

Family AP (within-family, v2 → v3): chain 73.5 → 79.4, fan_out_in 83.2 → 86.4, random 66.7 → 74.7, multi_stage 86.5 → 89.4,
fan_out 82.5 → 84.5, cycle 85.5 → 86.3, stack 80.5 → 81.6, fan_in 89.5 → 90.2, bipartite 64.0 → 58.5 (only loss).
Relays stay at ~1. Difficulty: easy 78.3 → 81.7, medium 6.3 → 9.8, hard 0.75 → 0.69.

Why smoothing works now but did not in phase 3 (+0.3–0.5 then): phase 3 tested one component set with the mean or max;
here we use the max over two complementary sets (short undirected and long forward) and add it on the rank scale with λ 0.5,
on top of better base models *(our data)*.
