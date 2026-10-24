import argparse
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = (REPO / ".." / "decode-and-detect-training-data" / "train").resolve()
DEFAULT_TEST = (REPO / ".." / "decode-and-detect-test-data").resolve()
SEED = 42


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--train-dir", default=str(DEFAULT_TRAIN))
    p.add_argument("--test-dir", default=str(DEFAULT_TEST))
    p.add_argument("--out-dir", default=str(REPO / "outputs"))
    p.add_argument("--cache-dir", default=None, help="optional dir to cache feature parquet (outside repo)")
    return p


def cached_features(data_dir, cache_dir, tag):
    from features import build_features
    if cache_dir:
        path = Path(cache_dir) / f"features_v5_{tag}.parquet"  # v5 = v4 + split-stable deliv_asset_code
        if path.exists():
            return pd.read_parquet(path)
        f = build_features(data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        f.to_parquet(path)
        return f
    return build_features(data_dir)


def lgb_params():
    return dict(objective="binary", learning_rate=0.03, n_estimators=600, num_leaves=31,
                min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                reg_lambda=1.0, random_state=SEED, n_jobs=os.cpu_count() or 1, verbose=-1)


def hgb_model(early_stopping="auto"):
    """early_stopping='auto' (sklearn default, used by v3): with > 10,000 rows HGB holds out a random, row-level 10% of
    the training data and stops on it. False: train on every row for all 500 iterations (predict.py --hgb-full)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=SEED, early_stopping=early_stopping)


def rank01(x):
    """Rank-normalise scores to (0, 1]; used to average models on a common scale."""
    from scipy.stats import rankdata
    return rankdata(x) / len(x)


RELAY_FAMILIES = ("aggregate_relay", "distributed_relay", "ambiguous")


def train_weights(labels_df):
    """Phase 4: weight 0 in TRAINING for relay/ambiguous positives (indistinguishable from their decoys, our data).
    Uses the training-only family label; never a feature, and every row is still scored in CV."""
    import numpy as np
    w = np.ones(len(labels_df))
    w[(labels_df.family.isin(RELAY_FAMILIES) & (labels_df.is_suspicious == 1)).values] = 0.0
    return w


# feature groups that can be dropped in leaderboard probes / diagnostics (see experiments/diagnose.py)
FEATURE_GROUPS = {
    # absolute clock time: train, validation and test weeks start on different weekdays
    "time": ["hour", "dow", "hour_sin", "hour_cos"],
    # asset / conversion choices: the organiser README says they "do not inspect suspicious labels"
    "legs": ["n_legs", "n_path_legs", "n_assets_legs", "asset_change", "has_sendmax", "has_paths", "conv_spread",
             "sendmax_slack", "deliv_is_xrp", "deliv_asset_code", "round_100", "round_10", "round_int", "n_decimals"],
    # fixed USD thresholds and roundness
    "usd": ["just_under_1000", "just_under_3000", "just_under_5000", "just_under_10000", "usd_round_100"],
}


def drop_groups(cols, groups):
    """Remove the named FEATURE_GROUPS (comma-separated string or list) from a column list."""
    if isinstance(groups, str):
        groups = [g.strip() for g in groups.split(",") if g.strip()]
    gone = {c for g in groups for c in FEATURE_GROUPS[g]}
    return [c for c in cols if c not in gone]


def smooth_scores(score, comp_labels, lam=0.25, agg="max"):
    """Phase 4: rank(score) + lam * rank(max score within each label-free flow component), summed over component sets.
    lam 0.25 is the final setting (public LB 64.5 vs 64.4 for lam 0.5, run v3_smooth025); predict.py and train.py both use it.
    Returns values in (0, 1 + lam*len(comp_labels)]."""
    out = rank01(score)
    for lab in comp_labels:
        cmax = pd.Series(score).groupby(lab).transform(agg).values
        out = out + lam * rank01(cmax)
    return out
