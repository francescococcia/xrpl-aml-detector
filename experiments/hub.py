"""Label-free hub-level relay features (improvement #1).

A relay hub repeatedly receives value and passes most of it on after a fairly regular delay,
over several days, often to rotating counterparties. Everything here is scale-free
(ratios, log-lag spreads, fractions), so no constant from the training batch (e.g. a 4.5h lag or a
0.93 ratio) is hard-coded. Computed per dataset from payments only (no labels).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "starter"))
from payments import load_payments  # noqa: E402

DAY = 86400


def _wallet_stats(s, r, t, v, nw):
    out = pd.DataFrame({"w": s, "t": t, "v": v, "o": r})
    inn = pd.DataFrame({"w": r, "t": t, "v": v, "o": s})
    res = {}

    def put(series, fill):
        x = np.full(nw, float(fill))
        series = series.replace([np.inf, -np.inf], np.nan).dropna()
        x[series.index.values.astype(int)] = series.values
        return x

    # --- daily conservation: per wallet-day out/in value ratio
    out["d"], inn["d"] = out.t // DAY, inn.t // DAY
    di = inn.groupby(["w", "d"]).v.sum().rename("i")
    do = out.groupby(["w", "d"]).v.sum().rename("o")
    dd = pd.concat([di, do], axis=1).fillna(0).reset_index()
    both = (dd.i > 0) & (dd.o > 0)
    dd["lr"] = np.where(both, np.log(dd.o.clip(lower=1e-9) / dd.i.clip(lower=1e-9)), np.nan)
    g = dd.groupby("w")
    res["hub_days_active"] = put(g.size(), 0)
    res["hub_days_both"] = put(both.groupby(dd.w).sum(), 0)
    res["hub_days_both_frac"] = put(both.groupby(dd.w).mean(), 0)
    res["hub_day_lr_med"] = put(dd.lr.groupby(dd.w).median(), 0)
    res["hub_day_lr_std"] = put(dd.lr.groupby(dd.w).std(), 3)
    # days whose out/in ratio is close to the wallet's own typical ratio (repeatable pattern)
    dd["dev"] = (dd.lr - dd.w.map(dd.lr.groupby(dd.w).median())).abs()
    res["hub_days_consistent"] = put((dd.dev < 0.1).groupby(dd.w).sum(), 0)
    # cross-day correlation of daily in and out (relay: out tracks in)
    c = dd.groupby("w")[["i", "o"]].corr().unstack().iloc[:, 1] if len(dd) else pd.Series(dtype=float)
    res["hub_day_io_corr"] = put(c, 0)
    # lagged conservation: out(day d+1) vs in(day d)
    dd = dd.sort_values(["w", "d"])
    nxt = dd.groupby("w").o.shift(-1)
    lr1 = np.log((dd.o + nxt.fillna(0)).clip(lower=1e-9) / dd.i.clip(lower=1e-9)).where(dd.i > 0)
    res["hub_lr_2day_med"] = put(lr1.groupby(dd.w).median(), -5)

    # --- lag: for each outflow, time since the wallet's most recent inflow
    ik = inn.w.values.astype(np.int64) * 10 ** 8 + inn.t.values
    order = np.argsort(ik)
    ik = ik[order]
    ok = out.w.values.astype(np.int64) * 10 ** 8 + out.t.values
    pos = np.searchsorted(ik, ok, side="right") - 1
    valid = (pos >= 0) & ((ik[np.maximum(pos, 0)] // 10 ** 8) == out.w.values)
    lag = np.where(valid, ok - ik[np.maximum(pos, 0)], np.nan)
    out["llag"] = np.log1p(lag)
    gl = out.groupby("w").llag
    q75, q25 = gl.quantile(0.75), gl.quantile(0.25)
    res["hub_lag_med"] = put(gl.median(), 0)
    res["hub_lag_iqr"] = put(q75 - q25, 5)
    res["hub_lag_cv"] = put(gl.std() / gl.mean().clip(lower=1e-6), 5)
    res["hub_out_funded_frac"] = put(out.llag.notna().groupby(out.w).mean(), 0)
    # --- time-of-day regularity (circular concentration of activity times)
    for nm, df in (("in", inn), ("out", out)):
        ang = 2 * np.pi * (df.t % DAY) / DAY
        cs = pd.DataFrame({"w": df.w, "c": np.cos(ang), "s": np.sin(ang)}).groupby("w")[["c", "s"]].mean()
        res[f"hub_{nm}_tod_conc"] = put(np.sqrt(cs.c ** 2 + cs.s ** 2), 0)
    # --- counterparty rotation / novelty
    for nm, df in (("in", inn), ("out", out)):
        gg = df.groupby("w")
        res[f"hub_{nm}_uniq_ratio"] = put(gg.o.nunique() / gg.size(), 0)
        first_day = df.groupby(["w", "o"]).d.transform("min")
        res[f"hub_{nm}_new_cp_frac"] = put((first_day == df.d).groupby(df.w).mean(), 0)
        res[f"hub_{nm}_cnt"] = put(np.log1p(gg.size()), 0)
    # in/out counterparties overlap (relay: distinct sources and sinks)
    pairs_in = set(zip(inn.w, inn.o))
    ov = pd.Series([(w, o) in pairs_in for w, o in zip(out.w, out.o)], index=out.index)
    res["hub_cp_overlap"] = put(ov.groupby(out.w).mean(), 0)
    # total throughput conservation
    ti, to = inn.groupby("w").v.sum(), out.groupby("w").v.sum()
    res["hub_total_lr"] = put(np.log(to.reindex(ti.index).fillna(0).clip(lower=1e-9) / ti.clip(lower=1e-9)), -5)
    # composite: conserving + regular + repeating
    res["hub_relay_score"] = (np.exp(-np.abs(res["hub_day_lr_med"])) * np.exp(-res["hub_lag_iqr"])
                              * np.log1p(res["hub_days_both"]))
    return res


def hub_features(data_dir):
    pay = pd.DataFrame(load_payments(Path(data_dir)))
    t = pay.TIMESTAMP.astype(np.int64).values
    t = t - t.min()
    v = pay.VALUE.astype(float).values
    wallets, codes = np.unique(np.concatenate([pay.FROM_ADDRESS.values, pay.TO_ADDRESS.values]), return_inverse=True)
    n = len(pay)
    s, r = codes[:n], codes[n:]
    W = _wallet_stats(s, r, t, v, len(wallets))
    F = {}
    for k, arr in W.items():
        F[f"s_{k}"] = arr[s]
        F[f"r_{k}"] = arr[r]
    # propagate hub score to each tx: max over its two endpoints, and over the receiver's next hop
    F["hub_relay_max"] = np.maximum(W["hub_relay_score"][s], W["hub_relay_score"][r])
    out = pd.DataFrame(F)
    out.insert(0, "TRANSACTION_HASH", pay.TRANSACTION_HASH.values)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0)
