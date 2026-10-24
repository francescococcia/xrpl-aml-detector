"""Grouped 5-fold CV: starter logistic regression, phase-1 LightGBM (v1), phase-3 blend (v2) and phase-4 blend (v3)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, cached_features, hgb_model, lgb_params, parser, rank01, smooth_scores, train_weights  # noqa: E402
from features import STARTER, feature_columns  # noqa: E402

import lightgbm as lgb  # noqa: E402


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, random_state=SEED))


# name -> (model factory, feature set, weighted training)
MODELS = {
    "logreg_starter": (lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
                       "starter", False),
    "lightgbm_v1": (lambda: lgb.LGBMClassifier(**lgb_params()), "v1", False),   # phase-1 features only
    # v2 members (phase 3): phase-1 + 60 flow features, unweighted
    "lightgbm_flow": (lambda: lgb.LGBMClassifier(**lgb_params()), "v2", False),
    "hgb_flow": (hgb_model, "v2", False),
    "logreg_all": (_lr, "v2", False),
    # v3 members (phase 4): + 12 flow gap stats, relay/ambiguous positives weight 0 in training
    "lightgbm_v3": (lambda: lgb.LGBMClassifier(**lgb_params()), "v3", True),
    "hgb_v3": (hgb_model, "v3", True),
    "logreg_v3": (_lr, "v3", True),
}
BLEND_V2 = {"lightgbm_flow": 1, "hgb_flow": 1, "logreg_all": 1}
BLEND_V3 = {"lightgbm_v3": 2, "hgb_v3": 1, "logreg_v3": 1}
ADOPTED = "v3"


def fit_predict(mk, Xtr, ytr, wtr, Xte):
    m = mk()
    if wtr is None:
        m.fit(Xtr, ytr)
    elif hasattr(m, "steps"):  # sklearn Pipeline
        m.fit(Xtr, ytr, **{f"{m.steps[-1][0]}__sample_weight": wtr})
    else:
        m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict_proba(Xte)[:, 1]


def ap_beyond_top1(y, p):
    """AP of the ranking after removing the top 1% (how well positives below the top 1% are ordered)."""
    k = max(1, int(round(0.01 * len(y))))
    o = np.argsort(-p)[k:]
    return 100 * AP(y[o], p[o])


def top1(y, p):
    k = max(1, int(round(0.01 * len(y))))
    hit = y[np.argsort(-p)[:k]].sum()
    return hit / k, hit / max(y.sum(), 1)


def breakdown(df, score):
    rows = []
    neg = df.is_suspicious == 0
    for fam, g in df.groupby("family"):
        if g.is_suspicious.nunique() == 2:
            rows.append(("family (within-family AP)", fam, int(g.is_suspicious.sum()), 100 * AP(g.is_suspicious, score[g.index])))
    for d in ("easy", "medium", "hard"):
        m = neg | (df.difficulty == d)
        rows.append(("difficulty (vs all negatives)", d, int((df.difficulty == d).sum()), 100 * AP(df.is_suspicious[m], score[m])))
    return pd.DataFrame(rows, columns=["slice", "name", "n_pos", "AP"])


def main():
    args = parser(__doc__).parse_args()
    t0 = time.time()
    feats = cached_features(args.train_dir, args.cache_dir, "train")
    labels = pd.read_csv(Path(args.train_dir) / "labels.csv")
    df = feats.merge(labels, on="TRANSACTION_HASH", how="left").reset_index(drop=True)
    y = df.is_suspicious.values
    groups = df.scenario_id.astype(object).where(df.scenario_id.notna(), "u_" + df.index.astype(str)).values
    w = train_weights(df)
    sets = {"starter": STARTER, "v1": feature_columns(feats, flows=False), "v2": feature_columns(feats, gaps=False),
            "v3": feature_columns(feats)}
    cols, cols_v1 = sets["v3"], sets["v1"]
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, groups))
    oof = {}
    for name, (mk, fs, weighted) in MODELS.items():
        X = df[sets[fs]].values
        pred = np.zeros(len(df))
        for tr, va in folds:
            pred[va] = fit_predict(mk, X[tr], y[tr], w[tr] if weighted else None, X[va])
        oof[name] = pred
        print(f"{name:16s} done ({time.time() - t0:.0f}s)", flush=True)
    labs = [df[c].values for c in ("_lab600u", "_lab3600f")]
    for bname, members in (("blend_v2", BLEND_V2), ("blend_v3_unsmoothed", BLEND_V3)):
        b = np.zeros(len(df))
        for _, va in folds:  # rank-average inside each validation fold, as on the test set
            b[va] = sum(wt * rank01(oof[k][va]) for k, wt in members.items()) / sum(members.values())
        oof[bname] = b
    v3 = np.zeros(len(df))
    for _, va in folds:  # smoothing only over same-fold rows (a neighbour's OOF score never saw this row's label)
        sm = smooth_scores(oof["blend_v3_unsmoothed"][va], [lab[va] for lab in labs])
        v3[va] = (sm + 1e-6 * rank01(oof["lightgbm_v3"][va])) / (2 + 1e-6)  # tie-breaker
    oof["v3"] = v3
    results, fold_rows = {}, []
    for name, pred in oof.items():
        aps = [100 * AP(y[va], pred[va]) for _, va in folds]
        tp = [top1(y[va], pred[va]) for _, va in folds]
        bey = np.mean([ap_beyond_top1(y[va], pred[va]) for _, va in folds])
        results[name] = (np.mean(aps), np.std(aps), 100 * AP(y, pred), np.mean([a for a, _ in tp]), np.mean([b for _, b in tp]), bey)
        fold_rows.append([name] + [round(a, 2) for a in aps])
        print(f"{name:16s} AP = {np.mean(aps):.2f} +- {np.std(aps):.2f}  folds {np.round(aps, 2).tolist()}  "
              f"P@1% {results[name][3]:.3f} R@1% {results[name][4]:.3f} AP>1% {bey:.2f}", flush=True)
    best = ADOPTED  # breakdown is reported for the adopted model, not the argmax
    bd = breakdown(df, pd.Series(oof[best]))
    bd["AP_v2"] = breakdown(df, pd.Series(oof["blend_v2"])).AP.values
    bd["AP_v1"] = breakdown(df, pd.Series(oof["lightgbm_v1"])).AP.values
    bd["AP_baseline"] = breakdown(df, pd.Series(oof["logreg_starter"])).AP.values
    # feature importance (gain) on full data
    imp = None
    if True:
        m = lgb.LGBMClassifier(**lgb_params()).fit(df[cols].values, y, sample_weight=w)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        imp = (100 * imp / imp.sum()).round(2)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = "<!-- hand-written notes below are kept on regeneration -->"
    old = (out / "cv_results.md").read_text() if (out / "cv_results.md").exists() else ""
    notes = old[old.index(marker):] if marker in old else marker + "\n"
    with open(out / "cv_results.md", "w") as fh:
        fh.write("# Grouped 5-fold CV (GroupKFold by scenario_id, NaN -> unique; seed 42)\n\n")
        fh.write("| model | mean AP x100 | std | pooled OOF AP | precision@top1% | recall@top1% | AP beyond top1% |\n|---|---|---|---|---|---|---|\n")
        for k, (a, sd, p, pr, rc, bey) in results.items():
            fh.write(f"| {k} | {a:.2f} | {sd:.2f} | {p:.2f} | {pr:.3f} | {rc:.3f} | {bey:.2f} |\n")
        fh.write("\nPer-fold AP x100 (same folds for every model):\n\n")
        fh.write(pd.DataFrame(fold_rows, columns=["model"] + [f"fold{i}" for i in range(5)]).to_markdown(index=False))
        fh.write("\n\nTop-1% metrics are computed per validation fold (top 1% of that fold's txs) and averaged.\n")
        fh.write(f"\nAdopted: **{best}** (v1 = lightgbm_v1, v2 = blend_v2 = phase-3 submission; v3 = 2:1:1 blend of the *_v3 models + same-fold flow max-score smoothing + tie-breaker)\n\n## Breakdown (OOF scores: adopted v3 vs v2 vs v1 vs starter baseline)\n\n")
        fh.write(bd.round(2).to_markdown(index=False))
        if imp is not None:
            fh.write("\n\n## Top 25 features (lightgbm_v3 gain %, full train, weighted)\n\n")
            fh.write(imp.head(25).to_frame("gain_pct").to_markdown())
        fh.write(f"\n\nRuntime: {time.time() - t0:.0f}s, {len(cols)} features ({len(cols_v1)} phase-1 + {len(cols) - len(cols_v1)} flow incl. gap stats)\n\n")
        fh.write(notes)
    print(bd.round(2).to_string(index=False))
    if imp is not None:
        print(imp.head(25).to_string())
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
