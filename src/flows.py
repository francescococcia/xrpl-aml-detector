"""Label-free flow reconstruction (phase 3).

A "flow" is a connected component of payments in which two payments are linked if they share a
wallet (as sender or receiver) and are consecutive for that wallet within W seconds. Easy/medium
laundering scenarios are tight bursts of linked payments, while background traffic on the same
wallets is sparser. Group-level aggregates are attached back to every payment in the group.
No labels are used, and the graph is built per dataset (train and test share no wallets).
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

UNDIRECTED_WINDOWS = (30, 120, 600)
FORWARD_WINDOWS = (60, 600, 3600)


def _components(w_s, w_r, t, win, forward_only=False):
    n = len(t)
    ev = pd.DataFrame({"w": np.r_[w_s, w_r], "t": np.r_[t, t], "i": np.r_[np.arange(n), np.arange(n)],
                       "role": np.r_[np.zeros(n, int), np.ones(n, int)]}).sort_values(["w", "t", "i"], kind="stable")
    w, tt, ii, role = ev.w.values, ev.t.values, ev.i.values, ev.role.values
    link = (w[1:] == w[:-1]) & (tt[1:] - tt[:-1] <= win) & (ii[1:] != ii[:-1])
    if forward_only:  # an inflow followed by an outflow of the same wallet
        link &= (role[:-1] == 1) & (role[1:] == 0)
    a, b = ii[:-1][link], ii[1:][link]
    return connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)[1]


def _group_stats(lab, s, r, t, v, win, tag, wallet_cnt):
    d = pd.DataFrame({"g": lab, "s": s, "r": r, "t": t, "v": v})
    g = d.groupby("g")
    size = g.t.transform("size").values
    dur = (g.t.transform("max") - g.t.transform("min")).values
    wal = pd.concat([d[["g", "s"]].rename(columns={"s": "w"}), d[["g", "r"]].rename(columns={"r": "w"})])
    nw = wal.groupby("g").w.nunique().reindex(lab).values
    ns = g.s.transform("nunique").values
    nr = g.r.transform("nunique").values
    # wallets acting as both sender and receiver inside the flow (pass-through)
    both = pd.Series(list(set(zip(d.g, d.s)) & set(zip(d.g, d.r))), dtype=object)
    nboth = pd.Series([x[0] for x in both], dtype="int64").value_counts().reindex(lab).fillna(0).values
    o = {}
    o[f"{tag}_size"] = np.log1p(size)
    o[f"{tag}_dur"] = np.log1p(dur)
    o[f"{tag}_rate"] = size / (dur + win)
    o[f"{tag}_ns"] = ns
    o[f"{tag}_nr"] = nr
    o[f"{tag}_nw"] = nw
    o[f"{tag}_density"] = size / np.maximum(nw, 1)
    o[f"{tag}_pass"] = nboth / np.maximum(nw, 1)
    o[f"{tag}_v_std"] = g.v.transform("std").fillna(0).values
    o[f"{tag}_v_rel"] = v - g.v.transform("mean").values
    o[f"{tag}_pos"] = (g.t.rank(method="first").values - 1) / np.maximum(size - 1, 1)
    # how much of the sender's / receiver's whole activity this flow represents (burst saliency)
    o[f"{tag}_s_share"] = d.groupby(["g", "s"]).t.transform("size").values / wallet_cnt[s]
    o[f"{tag}_r_share"] = d.groupby(["g", "r"]).t.transform("size").values / wallet_cnt[r]
    return o


def flow_features(sender, receiver, t, value, und=UNDIRECTED_WINDOWS, fwdw=FORWARD_WINDOWS, extra=False):
    codes = pd.factorize(np.r_[sender, receiver])[0]
    n = len(t)
    s, r = codes[:n], codes[n:]
    t = np.asarray(t, dtype=np.int64)
    v = np.log1p(np.asarray(value, dtype=float))
    wallet_cnt = np.bincount(codes, minlength=codes.max() + 1).astype(float)
    out = {}
    for fwd, wins in ((False, und), (True, fwdw)):
        for win in wins:
            lab = _components(s, r, t, win, fwd)
            o = _group_stats(lab, s, r, t, v, win, f"fl{'f' if fwd else ''}{win}", wallet_cnt)
            if not extra:  # pass-through / activity-share stats did not help in CV (our data)
                o = {k: x for k, x in o.items() if not k.endswith(("_pass", "_s_share", "_r_share"))}
            out.update(o)
    return pd.DataFrame(out)
