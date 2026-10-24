"""Grouped 5-fold CV: starter logistic regression, phase-1 LightGBM (v1) and phase-3 flow models (v2)."""
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
from common import SEED, cached_features, hgb_model, lgb_params, parser, rank01  # noqa: E402
from features import STARTER, feature_columns  # noqa: E402

import lightgbm as lgb  # noqa: E402


def make_models():
    return {
        "logreg_starter": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)),
        "logreg_all": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1, random_state=SEED)),
        "lightgbm_v1": lambda: lgb.LGBMClassifier(**lgb_params()),    # phase-1 features only
        "lightgbm_flow": lambda: lgb.LGBMClassifier(**lgb_params()),  # + flow features
        "hgb_flow": hgb_model,
    }


BLEND = ("lightgbm_flow", "hgb_flow", "logreg_all")  # v2 = equal-weight rank average of these three
ADOPTED = "blend_v2"


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
    cols = feature_columns(feats)
    cols_v1 = feature_columns(feats, flows=False)
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(df, y, groups))
    oof = {}
    for name, mk in make_models().items():
        X = df[STARTER if name == "logreg_starter" else cols_v1 if name == "lightgbm_v1" else cols].values
        pred = np.zeros(len(df))
        for tr, va in folds:
            pred[va] = mk().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        oof[name] = pred
        print(f"{name:16s} done ({time.time() - t0:.0f}s)", flush=True)
    blend = np.zeros(len(df))
    for _, va in folds:  # rank-average inside each validation fold, as on the test set
        blend[va] = np.mean([rank01(oof[k][va]) for k in BLEND], axis=0)
    oof[ADOPTED] = blend
    results, fold_rows = {}, []
    for name, pred in oof.items():
        aps = [100 * AP(y[va], pred[va]) for _, va in folds]
        tp = [top1(y[va], pred[va]) for _, va in folds]
        results[name] = (np.mean(aps), np.std(aps), 100 * AP(y, pred), np.mean([a for a, _ in tp]), np.mean([b for _, b in tp]))
        fold_rows.append([name] + [round(a, 2) for a in aps])
        print(f"{name:16s} AP = {np.mean(aps):.2f} +- {np.std(aps):.2f}  folds {np.round(aps, 2).tolist()}  "
              f"P@1% {results[name][3]:.3f} R@1% {results[name][4]:.3f}", flush=True)
    best = ADOPTED  # breakdown is reported for the adopted v2 model, not the argmax
    bd = breakdown(df, pd.Series(oof[best]))
    bd["AP_v1"] = breakdown(df, pd.Series(oof["lightgbm_v1"])).AP.values
    bd["AP_baseline"] = breakdown(df, pd.Series(oof["logreg_starter"])).AP.values
    # feature importance (gain) on full data
    imp = None
    if best.startswith(("lightgbm", "blend")):
        m = lgb.LGBMClassifier(**lgb_params()).fit(df[cols].values, y)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        imp = (100 * imp / imp.sum()).round(2)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = "<!-- hand-written notes below are kept on regeneration -->"
    old = (out / "cv_results.md").read_text() if (out / "cv_results.md").exists() else ""
    notes = old[old.index(marker):] if marker in old else marker + "\n"
    with open(out / "cv_results.md", "w") as fh:
        fh.write("# Grouped 5-fold CV (GroupKFold by scenario_id, NaN -> unique; seed 42)\n\n")
        fh.write("| model | mean AP x100 | std | pooled OOF AP | precision@top1% | recall@top1% |\n|---|---|---|---|---|---|\n")
        for k, (a, sd, p, pr, rc) in results.items():
            fh.write(f"| {k} | {a:.2f} | {sd:.2f} | {p:.2f} | {pr:.3f} | {rc:.3f} |\n")
        fh.write("\nPer-fold AP x100 (same folds for every model):\n\n")
        fh.write(pd.DataFrame(fold_rows, columns=["model"] + [f"fold{i}" for i in range(5)]).to_markdown(index=False))
        fh.write("\n\nTop-1% metrics are computed per validation fold (top 1% of that fold's txs) and averaged.\n")
        fh.write(f"\nAdopted: **{best}** (v1 = lightgbm_v1 = phase-1 submission)\n\n## Breakdown (OOF scores: adopted v2 vs v1 vs starter baseline)\n\n")
        fh.write(bd.round(2).to_markdown(index=False))
        if imp is not None:
            fh.write("\n\n## Top 25 features (lightgbm_flow gain %, full train)\n\n")
            fh.write(imp.head(25).to_frame("gain_pct").to_markdown())
        fh.write(f"\n\nRuntime: {time.time() - t0:.0f}s, {len(cols)} features ({len(cols_v1)} phase-1 + {len(cols) - len(cols_v1)} flow)\n\n")
        fh.write(notes)
    print(bd.round(2).to_string(index=False))
    if imp is not None:
        print(imp.head(25).to_string())
    print(f"runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
