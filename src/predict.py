"""v2: train LightGBM (5 seeds) + HistGradientBoosting + logistic regression on phase-1 + flow features using
all of train, rank-average their test scores (equal weights), write and validate outputs/predictions_v2.csv and outputs/predictions.csv.
outputs/predictions_v1.csv (phase 1) is never touched."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, cached_features, hgb_model, lgb_params, parser, rank01  # noqa: E402
from features import feature_columns  # noqa: E402

import lightgbm as lgb  # noqa: E402


def main():
    args = parser(__doc__).parse_args()
    t0 = time.time()
    tr = cached_features(args.train_dir, args.cache_dir, "train")
    te = cached_features(args.test_dir, args.cache_dir, "test")
    labels = pd.read_csv(Path(args.train_dir) / "labels.csv")
    tr = tr.merge(labels[["TRANSACTION_HASH", "is_suspicious"]], on="TRANSACTION_HASH", how="left")
    cols = feature_columns(te)
    # average 5 seeds for stability
    Xtr, ytr, Xte = tr[cols].values, tr.is_suspicious.values, te[cols].values
    s_lgb = np.zeros(len(te))
    for seed in range(5):
        p = lgb_params()
        p["random_state"] = 42 + seed
        s_lgb += lgb.LGBMClassifier(**p).fit(Xtr, ytr).predict_proba(Xte)[:, 1] / 5
    s_hgb = hgb_model().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    s_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, random_state=SEED)) \
        .fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    score = (rank01(s_lgb) + rank01(s_hgb) + rank01(s_lr)) / 3
    out = pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH.values, "score": np.clip(score, 0, 1)})
    # validation
    expected = set(pq.read_table(Path(args.test_dir) / "transactions.parquet", columns=["TRANSACTION_HASH"])
                   .column(0).to_pylist())
    assert len(out) == len(expected), (len(out), len(expected))
    assert out.TRANSACTION_HASH.is_unique and set(out.TRANSACTION_HASH) == expected
    assert out.score.notna().all() and out.score.between(0, 1).all()
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for name in ("predictions_v2.csv", "predictions.csv"):
        out.to_csv(od / name, index=False)
        assert (od / name).stat().st_size < 12 * 2**20
    print(f"wrote {od / 'predictions_v2.csv'} (+ predictions.csv): {len(out)} rows, {out.TRANSACTION_HASH.nunique()} unique, "
          f"score range [{out.score.min():.5f}, {out.score.max():.5f}], mean {out.score.mean():.4f}; "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
