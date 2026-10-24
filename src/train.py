"""Grouped 5-fold CV: starter-style logistic regression vs LightGBM."""
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
from common import SEED, cached_features, lgb_params, parser  # noqa: E402
from features import STARTER, feature_columns  # noqa: E402

import lightgbm as lgb  # noqa: E402


def make_models():
    return {
        "logreg_starter": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        "logreg_all": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, random_state=SEED)),
        "lightgbm_all": lambda: lgb.LGBMClassifier(**lgb_params()),
    }


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
    cols = feature_columns(feats)
    gkf = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, oof = {}, {}
    for name, mk in make_models().items():
        X = df[STARTER].values if name == "logreg_starter" else df[cols].values
        pred = np.zeros(len(df))
        aps = []
        for tr, va in gkf.split(X, y, groups):
            m = mk().fit(X[tr], y[tr])
            pred[va] = m.predict_proba(X[va])[:, 1]
            aps.append(100 * AP(y[va], pred[va]))
        results[name] = (np.mean(aps), np.std(aps), 100 * AP(y, pred))
        oof[name] = pred
        print(f"{name:16s} AP = {np.mean(aps):.2f} +- {np.std(aps):.2f}  (pooled OOF {100 * AP(y, pred):.2f})", flush=True)
    best = max(results, key=lambda k: results[k][0])
    bd = breakdown(df, pd.Series(oof[best]))
    bd_base = breakdown(df, pd.Series(oof["logreg_starter"]))
    bd["AP_baseline"] = bd_base.AP.values
    # feature importance (gain) on full data
    imp = None
    if best.startswith("lightgbm"):
        m = lgb.LGBMClassifier(**lgb_params()).fit(df[cols].values, y)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        imp = (100 * imp / imp.sum()).round(2)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cv_results.md", "w") as fh:
        fh.write("# Grouped 5-fold CV (GroupKFold by scenario_id, NaN -> unique; seed 42)\n\n")
        fh.write("| model | mean AP x100 | std | pooled OOF AP |\n|---|---|---|---|\n")
        for k, (a, sd, p) in results.items():
            fh.write(f"| {k} | {a:.2f} | {sd:.2f} | {p:.2f} |\n")
        fh.write(f"\nBest: **{best}**\n\n## Breakdown (OOF scores, best vs starter baseline)\n\n")
        fh.write(bd.round(2).to_markdown(index=False))
        if imp is not None:
            fh.write("\n\n## Top 25 features (LightGBM gain %, full train)\n\n")
            fh.write(imp.head(25).to_frame("gain_pct").to_markdown())
        fh.write(f"\n\nRuntime: {time.time() - t0:.0f}s, {len(cols)} features\n")
    print(bd.round(2).to_string(index=False))
    if imp is not None:
        print(imp.head(25).to_string())
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
