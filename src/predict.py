"""v3: LightGBM (5 seeds) + HistGradientBoosting + logistic regression on phase-1 + flow features (incl. gap stats),
trained on all of train with relay/ambiguous positives at weight 0; rank blend 2:1:1; post-hoc max-score smoothing
over label-free flow components; tie-breaker. Writes and validates outputs/predictions_v3.csv and
outputs/predictions.csv. outputs/predictions_v1.csv and predictions_v2.csv are never touched."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cached_features, hgb_model, lgb_params, parser, rank01, smooth_scores, train_weights  # noqa: E402
from features import feature_columns  # noqa: E402
from train import _lr, fit_predict  # noqa: E402

import lightgbm as lgb  # noqa: E402


def main():
    args = parser(__doc__).parse_args()
    t0 = time.time()
    tr = cached_features(args.train_dir, args.cache_dir, "train")
    te = cached_features(args.test_dir, args.cache_dir, "test")
    labels = pd.read_csv(Path(args.train_dir) / "labels.csv")
    tr = tr.merge(labels[["TRANSACTION_HASH", "is_suspicious", "family"]], on="TRANSACTION_HASH", how="left")
    w = train_weights(tr)
    cols = feature_columns(te)
    Xtr, ytr, Xte = tr[cols].values, tr.is_suspicious.values, te[cols].values
    s_lgb = np.zeros(len(te))
    for seed in range(5):  # average 5 seeds for stability
        p = lgb_params()
        p["random_state"] = 42 + seed
        s_lgb += lgb.LGBMClassifier(**p).fit(Xtr, ytr, sample_weight=w).predict_proba(Xte)[:, 1] / 5
    s_hgb = fit_predict(hgb_model, Xtr, ytr, w, Xte)
    s_lr = fit_predict(_lr, Xtr, ytr, w, Xte)
    blend = (2 * rank01(s_lgb) + rank01(s_hgb) + rank01(s_lr)) / 4
    sm = smooth_scores(blend, [te["_lab600u"].values, te["_lab3600f"].values])
    score = (sm + 1e-6 * rank01(s_lgb)) / (2 + 1e-6)
    out = pd.DataFrame({"TRANSACTION_HASH": te.TRANSACTION_HASH.values, "score": np.clip(score, 0, 1)})
    # validation
    expected = set(pq.read_table(Path(args.test_dir) / "transactions.parquet", columns=["TRANSACTION_HASH"])
                   .column(0).to_pylist())
    assert len(out) == len(expected), (len(out), len(expected))
    assert out.TRANSACTION_HASH.is_unique and set(out.TRANSACTION_HASH) == expected
    assert out.score.notna().all() and out.score.between(0, 1).all()
    ties = len(out) - out.score.nunique()
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    for name in ("predictions_v3.csv", "predictions.csv"):
        out.to_csv(od / name, index=False, float_format="%.15f")
        assert (od / name).stat().st_size < 12 * 2**20
    chk = pd.read_csv(od / "predictions_v3.csv")
    assert ties == 0 and chk.score.nunique() == len(chk), "ties in final CSV"
    print(f"wrote {od / 'predictions_v3.csv'} (+ predictions.csv): {len(out)} rows, {out.TRANSACTION_HASH.nunique()} unique, "
          f"score range [{out.score.min():.5f}, {out.score.max():.5f}], ties in memory {ties}, "
          f"ties in written CSV {len(chk) - chk.score.nunique()}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
