"""Improvement #1 (DROPPED): grouped CV of base features vs base + hub relay features."""
import time
import lightgbm as lgb
from sklearn.metrics import average_precision_score as AP
from sklearn.model_selection import GroupKFold
from _setup import load_train, report, np
from common import lgb_params
from features import feature_columns
from hub import hub_features

a, f, df, y, groups = load_train(__doc__)
h = hub_features(a.train_dir)
df = df.merge(h, on="TRANSACTION_HASH", how="left")
base = feature_columns(f); hubc = [c for c in h.columns if c != "TRANSACTION_HASH"]
for name, cols in [("base", base), ("base+hub", base + hubc)]:
    t = time.time(); X = df[cols].values; pred = np.zeros(len(df)); aps = []
    for tr, va in GroupKFold(5, shuffle=True, random_state=42).split(X, y, groups):
        pred[va] = lgb.LGBMClassifier(**lgb_params()).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        aps.append(100 * AP(y[va], pred[va]))
    print(name, round(np.mean(aps), 2), round(np.std(aps), 2), f"{time.time()-t:.0f}s")
    report(df, y, pred, name)
