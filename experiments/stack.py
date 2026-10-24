"""Second stage (improvement #2): propagate first-stage scores over the wallet graph.

Laundering labels cover whole scenarios (many linked payments), so a payment whose wallets and
neighbours carry high first-stage scores is more likely suspicious. Aggregates use only scores
(never labels) and are rank-normalised per dataset so they are comparable between train and test.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _wallet_codes(sender, receiver):
    wallets, codes = np.unique(np.concatenate([sender, receiver]), return_inverse=True)
    n = len(sender)
    return codes[:n], codes[n:], len(wallets)


def neighbour_features(sender, receiver, t, p):
    """sender/receiver: address arrays, t: seconds, p: first-stage scores (same order)."""
    s, r, nw = _wallet_codes(np.asarray(sender), np.asarray(receiver))
    n = len(p)
    q = pd.Series(p).rank(pct=True).values  # rank-normalised score in [0, 1]
    w_all = np.concatenate([s, r])
    q2 = np.concatenate([q, q])
    cnt = np.bincount(w_all, minlength=nw).astype(float)
    sm = np.bincount(w_all, weights=q2, minlength=nw)
    sq = np.bincount(w_all, weights=q2 ** 2, minlength=nw)
    mx = np.zeros(nw)
    np.maximum.at(mx, w_all, q2)
    # top-share: fraction of a wallet's payments in the global top 5% of scores
    hi = (q2 > 0.95).astype(float)
    hic = np.bincount(w_all, weights=hi, minlength=nw)
    F = {"stage1_rank": q}
    for pre, w in (("s", s), ("r", r)):
        c = cnt[w]
        F[f"{pre}_nb_mean_excl"] = np.where(c > 1, (sm[w] - q) / np.maximum(c - 1, 1), 0)
        F[f"{pre}_nb_max"] = mx[w]
        F[f"{pre}_nb_std"] = np.sqrt(np.maximum(sq[w] / c - (sm[w] / c) ** 2, 0))
        F[f"{pre}_nb_hi_frac"] = (hic[w] - (q > 0.95)) / np.maximum(c - 1, 1)
    # two-hop: mean over counterparties of their wallet mean score
    wmean = sm / np.maximum(cnt, 1)
    other = np.concatenate([r, s])
    nb2 = np.bincount(w_all, weights=wmean[other], minlength=nw) / np.maximum(cnt, 1)
    F["s_nb2_mean"], F["r_nb2_mean"] = nb2[s], nb2[r]
    # sender activity within +-24h: mean score of the sender's/receiver's payments nearby in time
    for pre, w in (("s", s), ("r", r)):
        ww = np.concatenate([s, r])
        tt = np.concatenate([t, t])
        key = ww.astype(np.int64) * 10 ** 8 + tt
        o = np.argsort(key, kind="stable")
        key, qq = key[o], q2[o]
        cq = np.concatenate([[0.0], np.cumsum(qq)])
        base = w.astype(np.int64) * 10 ** 8 + t
        a = np.searchsorted(key, base - 86400, "left")
        b = np.searchsorted(key, base + 86400, "right")
        k = b - a
        F[f"{pre}_nb24_mean_excl"] = np.where(k > 1, (cq[b] - cq[a] - q) / np.maximum(k - 1, 1), 0)
    # flow groups: connected components over high-scoring edges (top 10% of this dataset)
    for thr, nm in ((0.90, "g90"), (0.97, "g97")):
        m = q > thr
        A = coo_matrix((np.ones(m.sum()), (s[m], r[m])), shape=(nw, nw))
        _, lab = connected_components(A, directed=False)
        gsize = np.bincount(lab[s[m]], minlength=nw)  # edges per component
        gsum = np.bincount(lab[s[m]], weights=q[m], minlength=nw)
        comp = lab[s]
        F[f"{nm}_edges"] = np.where(m, np.log1p(gsize[comp]), 0)
        F[f"{nm}_mean"] = np.where(m, gsum[comp] / np.maximum(gsize[comp], 1), 0)
        # endpoint touches a high-score group even if this edge is below threshold
        touch = np.zeros(nw)
        np.maximum.at(touch, s[m], 1)
        np.maximum.at(touch, r[m], 1)
        F[f"{nm}_touch"] = np.maximum(touch[s], touch[r])
    return pd.DataFrame(F)
