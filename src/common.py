import argparse
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
        path = Path(cache_dir) / f"features_{tag}.parquet"
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
                reg_lambda=1.0, random_state=SEED, n_jobs=8, verbose=-1)
