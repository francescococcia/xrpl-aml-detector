"""v3: LightGBM (5 seeds) + HistGradientBoosting + logistic regression on phase-1 + flow features (incl. gap stats),
trained on all of train with relay/ambiguous positives at weight 0; rank blend 2:1:1; post-hoc max-score smoothing
over label-free flow components; tie-breaker. Writes and validates outputs/predictions_v3.csv and
outputs/predictions.csv. outputs/predictions_v1.csv and predictions_v2.csv are never touched.

Leaderboard-probe switches (all off by default; the defaults reproduce the submitted v3 run exactly):
  --lam L             smoothing weight (default 0.25)
  --smooth-sets S     flow component sets for smoothing, e.g. 600u,3600f,120u (u = undirected, f = forward)
  --weights a,b,c     rank-blend weights LightGBM,HGB,LR (0 = model not trained)
  --rank-features     replace every feature by its percentile rank within its own split (train / test separately)
  --drop G1,G2        drop feature groups from common.FEATURE_GROUPS: time, legs, usd, shift (default: shift)
  --smooth-agg A      max (default) or mean score per flow component for smoothing
  --hgb-full          HGB without sklearn's automatic early stopping (trains on all rows, 500 iterations)
  --pseudo K          pseudo-labelling: add the top-K test rows as positives and the bottom --pseudo-neg share as
                      negatives to training, retrain every member once, rescore
A non-default run writes outputs/predictions_<tag>.csv + outputs/run_<tag>.json and leaves predictions.csv alone,
unless --as-final is given (then it also writes predictions.csv and outputs/final_settings.json)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from common import cached_features, drop_groups, hgb_model, lgb_params, parser, rank01, smooth_scores, train_weights  # noqa: E402
from features import feature_columns  # noqa: E402
from train import _lr, fit_predict  # noqa: E402

import lightgbm as lgb  # noqa: E402

DEFAULTS = dict(lam=0.25, smooth_sets="600u,3600f", weights="2,1,1", rank_features=False, pseudo=0, pseudo_neg=0.5,
                drop="shift", smooth_agg="max", hgb_full=False)


def component_labels(te, sets, test_dir):
    """Component ids per smoothing set; 600u / 3600f come from the feature cache, others are rebuilt from payments."""
    labs, missing = [], [k for k in sets if f"_lab{k}" not in te.columns]
    if missing:
        from flows import _components
        from payments import load_payments
        pay = pd.DataFrame(load_payments(test_dir))
        assert (pay.TRANSACTION_HASH.values == te.TRANSACTION_HASH.values).all()
        codes = pd.factorize(np.r_[pay.FROM_ADDRESS.values, pay.TO_ADDRESS.values])[0]
        s, r, t = codes[:len(pay)], codes[len(pay):], pay.TIMESTAMP.astype(np.int64).values
    for k in sets:
        if f"_lab{k}" in te.columns:
            labs.append(te[f"_lab{k}"].values)
        else:
            assert k[-1] in "uf" and k[:-1].isdigit(), f"bad smoothing set {k!r} (use e.g. 120u or 600f)"
            labs.append(_components(s, r, t, int(k[:-1]), forward_only=k[-1] == "f"))
    return labs


def score_test(Xtr, ytr, w, Xte, weights, labs, lam, agg="max", hgb_es="auto"):
    """Train the blend members on (Xtr, ytr, w), score Xte, rank-blend, smooth, tie-break; returns final scores."""
    wl, wh, wr = weights
    s_lgb = np.zeros(len(Xte))
    for seed in range(5):  # average 5 seeds for stability (always trained: it is also the tie-breaker)
        p = lgb_params()
        p["random_state"] = 42 + seed
        s_lgb += lgb.LGBMClassifier(**p).fit(Xtr, ytr, sample_weight=w).predict_proba(Xte)[:, 1] / 5
    blend = wl * rank01(s_lgb)
    if wh:
        blend = blend + wh * rank01(fit_predict(lambda: hgb_model(hgb_es), Xtr, ytr, w, Xte))
    if wr:
        blend = blend + wr * rank01(fit_predict(_lr, Xtr, ytr, w, Xte))
    blend = blend / (wl + wh + wr)
    sm = smooth_scores(blend, labs, lam=lam, agg=agg)
    # divisor 2 for the default (1 + 0.25 * 2 = 1.5 <= 2), so the default run is bit-identical to the submitted v3
    return (sm + 1e-6 * rank01(s_lgb)) / (max(2.0, 1 + lam * len(labs)) + 1e-6)


def rank_within(X):
    return np.column_stack([rank01(X[:, j]) for j in range(X.shape[1])])


def main():
    p = parser(__doc__)
    p.add_argument("--lam", type=float, default=DEFAULTS["lam"])
    p.add_argument("--smooth-sets", default=DEFAULTS["smooth_sets"])
    p.add_argument("--weights", default=DEFAULTS["weights"])
    p.add_argument("--rank-features", action="store_true")
    p.add_argument("--drop", default=DEFAULTS["drop"], help="feature groups to drop, e.g. time,legs")
    p.add_argument("--smooth-agg", default=DEFAULTS["smooth_agg"], choices=["max", "mean"])
    p.add_argument("--hgb-full", action="store_true")
    p.add_argument("--pseudo", type=int, default=DEFAULTS["pseudo"])
    p.add_argument("--pseudo-neg", type=float, default=DEFAULTS["pseudo_neg"], help="bottom share of test used as negatives")
    p.add_argument("--tag", default=None, help="name for a non-default run (default: built from the switches)")
    p.add_argument("--as-final", action="store_true", help="also write predictions.csv + final_settings.json")
    args = p.parse_args()
    t0 = time.time()
    settings = dict(lam=args.lam, smooth_sets=args.smooth_sets, weights=args.weights, rank_features=args.rank_features,
                    pseudo=args.pseudo, pseudo_neg=args.pseudo_neg, drop=args.drop, smooth_agg=args.smooth_agg,
                    hgb_full=args.hgb_full)
    default = settings == DEFAULTS
    weights = [float(x) for x in args.weights.split(",")]
    assert len(weights) == 3 and weights[0] > 0, "--weights needs 3 values and LightGBM > 0"
    sets = [k.strip() for k in args.smooth_sets.split(",") if k.strip()]

    tr = cached_features(args.train_dir, args.cache_dir, "train")
    te = cached_features(args.test_dir, args.cache_dir, "test")
    labels = pd.read_csv(Path(args.train_dir) / "labels.csv")
    tr = tr.merge(labels[["TRANSACTION_HASH", "is_suspicious", "family"]], on="TRANSACTION_HASH", how="left")
    w = train_weights(tr)
    cols = drop_groups(feature_columns(te), args.drop)
    Xtr, ytr, Xte = tr[cols].values, tr.is_suspicious.values, te[cols].values
    if args.rank_features:
        Xtr, Xte = rank_within(Xtr), rank_within(Xte)
    labs = component_labels(te, sets, args.test_dir)
    hgb_es = False if args.hgb_full else "auto"
    score = score_test(Xtr, ytr, w, Xte, weights, labs, args.lam, args.smooth_agg, hgb_es)
    if args.pseudo:
        order = np.argsort(-score)
        pos, neg = order[:args.pseudo], order[len(order) - int(args.pseudo_neg * len(order)):]
        idx = np.r_[pos, neg]
        print(f"pseudo-labels: {len(pos)} positives (top), {len(neg)} negatives (bottom {args.pseudo_neg:.0%})", flush=True)
        score = score_test(np.vstack([Xtr, Xte[idx]]), np.r_[ytr, np.ones(len(pos)), np.zeros(len(neg))],
                           np.r_[w, np.ones(len(idx))], Xte, weights, labs, args.lam, args.smooth_agg, hgb_es)
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
    tag = args.tag or ("v3" if default else "_".join(
        [f"lam{args.lam:g}", args.smooth_sets.replace(",", "-"), "w" + args.weights.replace(",", "-")]
        + (["rank"] if args.rank_features else []) + ([f"pseudo{args.pseudo}"] if args.pseudo else [])
        + ([f"drop-{args.drop.replace(',', '-')}"] if args.drop else []) + (["mean"] if args.smooth_agg == "mean" else []) + (["hgbfull"] if args.hgb_full else [])))
    names = [f"predictions_{tag}.csv"] + (["predictions.csv"] if default or args.as_final else [])
    for name in names:
        out.to_csv(od / name, index=False, float_format="%.15f")
        assert (od / name).stat().st_size < 12 * 2**20
    chk = pd.read_csv(od / names[0])
    assert ties == 0 and chk.score.nunique() == len(chk), "ties in final CSV"
    command = "python src/predict.py " + " ".join(sys.argv[1:])
    info = dict(tag=tag, settings=settings, command=command, rows=len(out))
    (od / f"run_{tag}.json").write_text(json.dumps(info, indent=2) + "\n")
    if default or args.as_final:
        (od / "final_settings.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"wrote {', '.join(str(od / n) for n in names)}: {len(out)} rows, {out.TRANSACTION_HASH.nunique()} unique, "
          f"score range [{out.score.min():.5f}, {out.score.max():.5f}], ties in memory {ties}, "
          f"ties in written CSV {len(chk) - chk.score.nunique()}; settings {settings}; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
