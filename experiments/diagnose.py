"""Where are the AP points lost? Five checks on train (grouped CV, same folds as src/train.py) and one train-vs-test check.

A. Grouping headroom: OOF LightGBM AP raw, with flow smoothing (max / mean, several lam), and with ORACLE smoothing
   by the true scenario_id (analysis only, never available at test time). Oracle >> flow means the grouping is the
   bottleneck; oracle ~ flow means the per-payment model is.
B. Flow purity: for each label-free component set, how much of a scenario lands in one component (coverage) and how
   much of that component is the scenario (purity). Scenarios are all-positive or all-decoy, so pure groups are what
   smoothing needs.
C. Scheme-level separation: per family, AUC of the scenario mean score, positive scenarios vs decoy scenarios; the
   worst positive and decoy scenarios.
D. Feature-group ablation in CV (common.FEATURE_GROUPS: time, legs, usd).
E. Adversarial validation: can a model tell train rows from test rows? AUC and the most shifted features.

Run: python experiments/diagnose.py --train-dir TRAIN --test-dir TEST --cache-dir /tmp/xrpl-cache  (~5 min)
Writes outputs/diagnose.md."""
import time
from pathlib import Path

from _setup import load_train  # noqa: F401  (sets sys.path)
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import FEATURE_GROUPS, SEED, cached_features, drop_groups, lgb_params, rank01, smooth_scores, train_weights
from features import feature_columns
from flows import _components
from payments import load_payments

EXTRA_SETS = ("30u", "120u", "1800u", "3600u", "600f", "7200f")


def oof_lgb(X, y, w, folds):
    pred = np.zeros(len(y))
    for tr, va in folds:
        pred[va] = lgb.LGBMClassifier(**lgb_params()).fit(X[tr], y[tr], sample_weight=w[tr]).predict_proba(X[va])[:, 1]
    return pred


def fold_ap(y, p, folds):
    a = [100 * AP(y[va], p[va]) for _, va in folds]
    return np.mean(a), np.std(a)


def smoothed(p, labs, folds, lam, agg):
    out = np.zeros(len(p))
    for _, va in folds:  # same-fold only, as in train.py
        out[va] = smooth_scores(p[va], [lab[va] for lab in labs], lam=lam, agg=agg)
    return out


def main():
    a, feats, df, y, groups = load_train(__doc__)
    t0 = time.time()
    lines = ["# Diagnostics (train, GroupKFold(5) by scenario_id, LightGBM v3 member, seed 42)", ""]
    w = train_weights(df)
    cols = feature_columns(feats)
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, groups))
    p = oof_lgb(df[cols].values, y, w, folds)
    print(f"OOF done {time.time() - t0:.0f}s", flush=True)

    # component labels: cached sets + rebuilt extra sets
    pay = pd.DataFrame(load_payments(a.train_dir))
    assert (pay.TRANSACTION_HASH.values == df.TRANSACTION_HASH.values).all()
    codes = pd.factorize(np.r_[pay.FROM_ADDRESS.values, pay.TO_ADDRESS.values])[0]
    s, r, t = codes[:len(pay)], codes[len(pay):], pay.TIMESTAMP.astype(np.int64).values
    comp = {"600u": df["_lab600u"].values, "3600f": df["_lab3600f"].values}
    for k in EXTRA_SETS:
        comp[k] = _components(s, r, t, int(k[:-1]), forward_only=k[-1] == "f")

    # A. grouping headroom
    lines += ["## A. Grouping headroom (mean ± std AP over folds)", "", "| variant | AP |", "|---|---|"]
    m, sd = fold_ap(y, p, folds)
    lines.append(f"| raw LightGBM OOF | {m:.2f} ± {sd:.2f} |")
    base = [comp["600u"], comp["3600f"]]
    for agg in ("max", "mean"):
        for lam in (0.25, 0.5, 1.0, 2.0):
            m, sd = fold_ap(y, smoothed(p, base, folds, lam, agg), folds)
            lines.append(f"| flow smoothing 600u+3600f, {agg}, lam {lam} | {m:.2f} ± {sd:.2f} |")
    for k in EXTRA_SETS:
        m, sd = fold_ap(y, smoothed(p, base + [comp[k]], folds, 0.25, "max"), folds)
        lines.append(f"| + {k}, max, lam 0.25 | {m:.2f} ± {sd:.2f} |")
    scen = df.scenario_id.astype(object).where(df.scenario_id.notna(), "u_" + df.index.astype(str)).values
    for agg in ("mean", "max"):
        o = pd.Series(p).groupby(scen).transform(agg).values
        m, sd = fold_ap(y, o, folds)
        lines.append(f"| ORACLE: score = scenario {agg} (true scenario_id, analysis only) | {m:.2f} ± {sd:.2f} |")
    print(f"A done {time.time() - t0:.0f}s", flush=True)

    # B. flow purity
    lines += ["", "## B. Flow purity vs true scenarios (non-background scenarios, averaged per scenario)", "",
              "| set | coverage (share of scenario in its main component) | purity (share of that component that is the scenario) | median component size |",
              "|---|---|---|---|"]
    has = df.scenario_id.notna().values
    for k, lab in comp.items():
        d = pd.DataFrame({"sc": df.scenario_id.values, "c": lab})
        main_c = d[has].groupby("sc").c.agg(lambda x: x.value_counts().index[0])
        cov = d[has].groupby("sc").c.apply(lambda x: x.value_counts().iloc[0] / len(x))
        csize = d.groupby("c").size()
        inside = d[has].groupby(["sc", "c"]).size()
        pur = np.array([inside[(sc, c)] / csize[c] for sc, c in main_c.items()])
        lines.append(f"| {k} | {cov.mean():.2f} | {pur.mean():.2f} | {int(csize.median())} |")

    # C. scheme-level separation
    lines += ["", "## C. Scheme-level separation (scenario mean OOF score)", "",
              "| family | pos scenarios | decoy scenarios | AUC pos vs decoy |", "|---|---|---|---|"]
    d = df[has].assign(p=p[has]).groupby("scenario_id").agg(fam=("family", "first"), diff=("difficulty", "first"),
                                                             y=("is_suspicious", "max"), n=("p", "size"), p=("p", "mean"))
    for fam, g in d.groupby("fam"):
        if g.y.nunique() == 2:
            lines.append(f"| {fam} | {int(g.y.sum())} | {int((g.y == 0).sum())} | {AUC(g.y, g.p):.3f} |")
    nr = d[~d.fam.isin(["aggregate_relay", "distributed_relay", "ambiguous"])]
    lines += ["", "Lowest-scored positive scenarios (non-relay):", "", nr[nr.y == 1].nsmallest(12, "p").round(4).to_markdown(),
              "", "Highest-scored decoy scenarios (non-relay):", "", nr[nr.y == 0].nlargest(12, "p").round(4).to_markdown()]
    print(f"C done {time.time() - t0:.0f}s", flush=True)

    # D. feature-group ablation
    lines += ["", "## D. Feature-group ablation (raw LightGBM OOF)", "", "| dropped | n features | AP |", "|---|---|---|"]
    for g in [None] + list(FEATURE_GROUPS) + ["time,legs,usd"]:
        c = cols if g is None else drop_groups(cols, g)
        m, sd = fold_ap(y, p if g is None else oof_lgb(df[c].values, y, w, folds), folds)
        lines.append(f"| {g or '(none)'} | {len(c)} | {m:.2f} ± {sd:.2f} |")
    print(f"D done {time.time() - t0:.0f}s", flush=True)

    # E. adversarial validation
    te = cached_features(a.test_dir, a.cache_dir, "test")
    X = np.vstack([df[cols].values, te[cols].values])
    z = np.r_[np.zeros(len(df)), np.ones(len(te))]
    padv, gain = np.zeros(len(z)), np.zeros(len(cols))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, z):
        m_ = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, colsample_bytree=0.6,
                                random_state=SEED, verbose=-1).fit(X[tr], z[tr])
        padv[va] = m_.predict_proba(X[va])[:, 1]
        gain += m_.booster_.feature_importance("gain")
    top = pd.Series(gain / gain.sum() * 100, index=cols).sort_values(ascending=False).head(15).round(2)
    lines += ["", "## E. Adversarial validation (train vs test rows)", "",
              f"AUC = {AUC(z, padv):.3f} (0.5 = no shift). Most shifted features (gain %):", "", top.to_frame("gain_pct").to_markdown(),
              "", "Reading: drop or rank-normalise (`predict.py --rank-features`) features that dominate here if they also "
              "rank high in the model; a large AUC means the test batch differs and train-shape thresholds may not transfer."]

    out = Path(a.out_dir if hasattr(a, "out_dir") else "outputs") / "diagnose.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + f"\n\nRuntime {time.time() - t0:.0f}s\n")
    print("\n".join(lines))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
