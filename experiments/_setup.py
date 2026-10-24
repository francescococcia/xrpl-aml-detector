"""Shared setup for the dropped-experiment scripts (not part of the submission pipeline)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "starter"), str(REPO / "experiments")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import cached_features, parser  # noqa: E402


def load_train(desc):
    a = parser(desc).parse_args()
    f = cached_features(a.train_dir, a.cache_dir, "train")
    lab = pd.read_csv(Path(a.train_dir) / "labels.csv")
    df = f.merge(lab, on="TRANSACTION_HASH", how="left")
    groups = df.scenario_id.astype(object).where(df.scenario_id.notna(), "u_" + df.index.astype(str)).values
    return a, f, df, df.is_suspicious.values, groups


def report(df, y, pred, name):
    from sklearn.metrics import average_precision_score as AP
    fam = {k: round(100 * AP(g.is_suspicious, pred[g.index]), 2) for k, g in df.groupby("family")
           if g.is_suspicious.nunique() == 2}
    neg = y == 0
    dif = {d: round(100 * AP(y[neg | (df.difficulty == d).values], pred[neg | (df.difficulty == d).values]), 2)
           for d in ["easy", "medium", "hard"]}
    print(name, "family", fam, "\n", name, "difficulty", dif, flush=True)
