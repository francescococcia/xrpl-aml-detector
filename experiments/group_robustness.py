"""Robustness check: stage-1 LightGBM under StratifiedGroupKFold grouped by sender wallet
and by wallet connected component (skipped if one component holds >30% of txs)."""
import lightgbm as lgb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import average_precision_score as AP
from sklearn.model_selection import StratifiedGroupKFold
from _setup import load_train, np, pd
from common import lgb_params
from features import feature_columns

a, f, df, y, groups = load_train(__doc__)
w = pd.factorize(np.r_[f._sender.values, f._receiver.values])[0]; n = len(f); s, r = w[:n], w[n:]
nc, lab = connected_components(coo_matrix((np.ones(n), (s, r)), shape=(w.max() + 1,) * 2), directed=False)
comp = lab[s]; share = np.bincount(comp).max() / n
print("components", nc, "largest component tx share", round(share, 3))
X = df[feature_columns(f)].values
for name, g in [("sender", s), ("component", comp)]:
    if name == "component" and share > 0.3:
        print("component grouping skipped (giant component)"); continue
    aps = [100 * AP(y[va], lgb.LGBMClassifier(**lgb_params()).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1])
           for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g)]
    print(name, np.round(aps, 2), round(np.mean(aps), 2), round(np.std(aps), 2), flush=True)
