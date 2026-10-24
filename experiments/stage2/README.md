# Stage 2: learned neighbourhood smoother (public LB 65.7)

Added after the main pipeline (LB 64.9). For each payment, features are the max, count and mean of the
base model's rank among payments sharing a sender or receiver wallet within +-600 s, 1800 s and 7200 s,
plus its own rank. A small LightGBM (7 leaves, 400 trees, lr 0.02, lambda 5, 3 seeds) is trained on
out-of-fold base ranks (GroupKFold 5 by scenario_id) and applied to test. Grouped CV gain about +1.8 to +2.1 AP.

Run order (scripts were written on a Linux box with paths under /workspace; edit the paths at the top of each):
1. `python load.py` - one row per payment (sender, receiver, time) for train/test.
2. `python base.py` - out-of-fold base scores on train; `python predict_base.py` - base scores on test (test_base.parquet).
3. `python stage2d.py` - neighbour features on train OOF (nb_train.parquet) and stage-2 CV.
4. `python predict_s2.py fold_all 0.4 stage2_nb` - stage-2 test scores (s2_test_stage2_nb.npy).
5. `python blend.py outputs/predictions.csv s2_test_stage2_nb.npy test_base.parquet outputs/predictions_s2.csv 0.4`
   where outputs/predictions.csv is the default `python src/predict.py` run.

Written with AI assistance (Grok); see ATTRIBUTION.md.
