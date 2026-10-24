"""Improvement #2 (DROPPED): strict nested-OOF stage-2 on neighbourhood / flow-group aggregates.

Outer GroupKFold(5, seed 42). Per outer fold: inner GroupKFold(5, seed 43) gives OOF stage-1 scores
on the train part; stage-1 fit on the whole train part scores the validation fold. Neighbour
aggregates are built from that score vector, stage-2 is fit on the train part only.
"""
import time
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from sklearn.model_selection import GroupKFold
from _setup import load_train, report, np
from common import lgb_params
from features import feature_columns
from payments import load_payments
from stack import neighbour_features

a, f, df, y, groups = load_train(__doc__)
pay = pd.DataFrame(load_payments(a.train_dir))
assert (pay.TRANSACTION_HASH.values == f.TRANSACTION_HASH.values).all()
tt = pay.TIMESTAMP.astype(np.int64).values; tt = tt - tt.min()
X = df[feature_columns(f)].values
fit = lambda A, b: lgb.LGBMClassifier(**lgb_params()).fit(A, b)  # noqa: E731
T0 = time.time(); p1 = np.zeros(len(df)); p2 = {k: np.zeros(len(df)) for k in ["agg_only", "base_agg"]}
aps = {k: [] for k in ["stage1", *p2]}
for k, (tr, va) in enumerate(GroupKFold(5, shuffle=True, random_state=42).split(X, y, groups)):
    sc = np.zeros(len(df))
    for itr, iva in GroupKFold(5, shuffle=True, random_state=43).split(X[tr], y[tr], groups[tr]):
        sc[tr[iva]] = fit(X[tr[itr]], y[tr[itr]]).predict_proba(X[tr[iva]])[:, 1]
    sc[va] = fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    p1[va] = sc[va]; aps["stage1"].append(100 * AP(y[va], sc[va]))
    nb = neighbour_features(f._sender.values, f._receiver.values, tt, sc).values
    for name in p2:
        Z = nb if name == "agg_only" else np.hstack([X, nb])
        p2[name][va] = fit(Z[tr], y[tr]).predict_proba(Z[va])[:, 1]; aps[name].append(100 * AP(y[va], p2[name][va]))
    print("fold", k, {n: round(v[-1], 2) for n, v in aps.items()}, f"{time.time()-T0:.0f}s", flush=True)
for n, v in aps.items():
    print(n, round(np.mean(v), 2), round(np.std(v), 2))
for n, p in [("stage1", p1), *p2.items()]:
    report(df, y, p, n)
