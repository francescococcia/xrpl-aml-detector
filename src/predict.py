"""Train LightGBM on all of train, score test, write and validate outputs/predictions.csv."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_features, lgb_params, parser  # noqa: E402
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
    score = np.zeros(len(te))
    for seed in range(5):
        p = lgb_params()
        p["random_state"] = 42 + seed
        m = lgb.LGBMClassifier(**p).fit(tr[cols].values, tr.is_suspicious.values)
        score += m.predict_proba(te[cols].values)[:, 1] / 5
    out = pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH.values, "score": np.clip(score, 0, 1)})
    # validation
    expected = set(pq.read_table(Path(args.test_dir) / "transactions.parquet", columns=["TRANSACTION_HASH"])
                   .column(0).to_pylist())
    assert len(out) == len(expected), (len(out), len(expected))
    assert out.TRANSACTION_HASH.is_unique and set(out.TRANSACTION_HASH) == expected
    assert out.score.notna().all() and out.score.between(0, 1).all()
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "predictions.csv", index=False)
    print(f"wrote {od / 'predictions.csv'}: {len(out)} rows, {out.TRANSACTION_HASH.nunique()} unique, "
          f"score range [{out.score.min():.5f}, {out.score.max():.5f}], mean {out.score.mean():.4f}; "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
