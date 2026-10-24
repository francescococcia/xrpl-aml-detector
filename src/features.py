"""Per-transaction AML features from the XRPL payment graph + transfer legs.

Wallet = node, payment = timed directed edge (FROM -> TO, VALUE in reference USD).
All window queries are vectorised with a global (wallet, time) sort + searchsorted.
Offline investigation: windows may look both backwards and forwards in the 7-day data.
"""
import json
import sys
import zlib
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "starter"))
from payments import load_payments, load_rates, XRP, SCALE  # noqa: E402
from flows import flow_features  # noqa: E402

WINDOWS = {"10m": 600, "1h": 3600, "6h": 6 * 3600, "24h": 86400, "7d": 7 * 86400}
BIG = 10 ** 8  # > time span in seconds


class WalletIndex:
    """Edges sorted by (wallet, time) for one side (out = by sender, in = by receiver)."""

    def __init__(self, wallet_codes, t, v, other_codes, edge_ids):
        key = wallet_codes.astype(np.int64) * BIG + t
        order = np.argsort(key, kind="stable")
        self.key = key[order]
        self.t = t[order]
        self.v = v[order]
        self.other = other_codes[order]
        self.eid = edge_ids[order]
        self.cv = np.concatenate([[0.0], np.cumsum(self.v)])

    def range(self, wallet_codes, lo, hi, lo_incl=True, hi_incl=True):
        base = wallet_codes.astype(np.int64) * BIG
        a = np.searchsorted(self.key, base + lo, side="left" if lo_incl else "right")
        b = np.searchsorted(self.key, base + hi, side="right" if hi_incl else "left")
        return a, np.maximum(a, b)

    def count_sum(self, wallet_codes, lo, hi, **kw):
        a, b = self.range(wallet_codes, lo, hi, **kw)
        return (b - a).astype(float), self.cv[b] - self.cv[a]

    def wsum(self, weights_by_eid, a, b):
        c = np.concatenate([[0.0], np.cumsum(weights_by_eid[self.eid])])
        return c[b] - c[a]


def _decode(wire):
    if isinstance(wire, str):
        return "XRP", int(wire) / SCALE
    return f"{wire['issuer']}:{wire['currency']}", float(wire["value"])


def leg_features(data_dir, tx, pay_value):
    """Currency-conversion features from transfer legs and SendMax."""
    rates = load_rates(data_dir)
    legs = pd.read_parquet(Path(data_dir) / "transfers.parquet",
                           columns=["TRANSACTION_HASH", "TRANSFER_TYPE", "TRANSFER_INDEX", "TOKEN_ADDRESS",
                                    "SYMBOL", "VALUE", "TIMESTAMP"])
    legs["TRANSFER_INDEX"] = legs["TRANSFER_INDEX"].astype(int)
    legs["asset"] = np.where(legs.TOKEN_ADDRESS == XRP, "XRP", legs.TOKEN_ADDRESS + ":" + legs.SYMBOL)
    hr = legs.TIMESTAMP.astype(np.int64) // 3600 * 3600
    legs["usd"] = [v * rates[(h, a)] / SCALE for v, h, a in zip(legs.VALUE, hr, legs.asset)]
    legs = legs.sort_values(["TRANSACTION_HASH", "TRANSFER_INDEX"])
    g = legs.groupby("TRANSACTION_HASH")
    out = pd.DataFrame({
        "n_legs": g.size(),
        "n_path_legs": g.TRANSFER_TYPE.apply(lambda s: (s == "path_payment").sum()),
        "n_assets_legs": g.asset.nunique(),
        "first_leg_asset": g.asset.first(),
        "cost_usd": g.usd.first(),
    })
    out = out.reindex(tx.TRANSACTION_HASH.values)
    extra = [json.loads(x) for x in tx.EXTRA_TRANSACTION_FIELDS]
    deliv = [_decode(json.loads(x)) for x in tx.DELIVERED_AMOUNT]
    deliv_asset = np.array([d[0] for d in deliv])
    deliv_units = np.array([d[1] for d in deliv])
    f = pd.DataFrame(index=tx.index)
    f["n_legs"] = out.n_legs.fillna(1).values
    f["n_path_legs"] = out.n_path_legs.fillna(0).values
    f["n_assets_legs"] = out.n_assets_legs.fillna(1).values
    f["asset_change"] = (out.first_leg_asset.fillna("").values != deliv_asset).astype(float) * (f.n_legs > 1)
    f["has_sendmax"] = [float("SendMax" in e) for e in extra]
    f["has_paths"] = [float("Paths" in e) for e in extra]
    cost = out.cost_usd.values
    f["conv_spread"] = np.where(f.n_legs > 1, cost / np.maximum(pay_value, 1e-9) - 1, 0.0)
    smax = []
    for e, ts in zip(extra, tx.TIMESTAMP.astype(np.int64)):
        if "SendMax" in e:
            a, u = _decode(e["SendMax"])
            smax.append(u * rates[(ts // 3600 * 3600, a)] / SCALE)
        else:
            smax.append(np.nan)
    smax = np.array(smax)
    f["sendmax_slack"] = np.where(np.isnan(smax), 0.0, smax / np.maximum(np.nan_to_num(cost, nan=1), 1e-9) - 1)
    f["deliv_is_xrp"] = (deliv_asset == "XRP").astype(float)
    # roundness in the delivered (native) currency
    f["round_100"] = (np.abs(deliv_units / 100 - np.round(deliv_units / 100)) < 1e-6).astype(float)
    f["round_10"] = (np.abs(deliv_units / 10 - np.round(deliv_units / 10)) < 1e-6).astype(float)
    f["round_int"] = (np.abs(deliv_units - np.round(deliv_units)) < 1e-6).astype(float)
    f["n_decimals"] = [len(str(x).split(".")[1].rstrip("0")) if "." in str(x) else 0 for x in deliv_units]
    # currency only; crc32 gives the same code for the same currency in every split (pd.factorize did not:
    # it numbers currencies in order of first appearance, so codes meant different things in train and test)
    f["deliv_asset_code"] = [zlib.crc32(c.encode()) % 100_003 for c in pd.Series(deliv_asset).str.split(":").str[-1]]
    return f


def cycle_features(s, r, t, v, outidx, horizon=86400, max_hops=3, cap=64):
    """Does value leaving s via r come back to s within `horizon` in <= max_hops hops?"""
    n = len(s)
    ret = np.zeros(n)
    ret_hops = np.full(n, 9.0)
    reach = np.zeros(n)
    for i in range(n):
        src, t0 = s[i], t[i]
        frontier = {r[i]: t0}
        seen = set(frontier)
        found = 0
        for hop in range(1, max_hops + 1):
            nxt = {}
            for w, tw in frontier.items():
                a, b = outidx.range(np.array([w]), tw, t0 + horizon, lo_incl=False)
                a, b = a[0], b[0]
                if b - a > cap:
                    b = a + cap
                for j in range(a, b):
                    o = outidx.other[j]
                    if o == src:
                        found = hop
                        break
                    if o not in seen and (o not in nxt or outidx.t[j] < nxt[o]):
                        nxt[o] = outidx.t[j]
                if found:
                    break
            if found:
                break
            seen.update(nxt)
            frontier = nxt
            if not frontier:
                break
        ret[i] = float(found > 0)
        ret_hops[i] = found if found else 9.0
        reach[i] = len(seen)
    return ret, ret_hops, reach


def episode_features(s, r, t, v, nw, gap=6 * 3600):
    """Split each wallet's activity into episodes (gaps > `gap`) and measure in/out conservation."""
    ev = pd.DataFrame({"w": np.concatenate([r, s]), "t": np.concatenate([t, t]),
                       "inv": np.concatenate([v, np.zeros_like(v)]), "outv": np.concatenate([np.zeros_like(v), v])})
    ev = ev.sort_values(["w", "t"]).reset_index(drop=True)
    newep = (ev.w.diff() != 0) | (ev.t.diff() > gap)
    ev["ep"] = newep.cumsum()
    e = ev.groupby(["w", "ep"]).agg(i=("inv", "sum"), o=("outv", "sum"), n=("t", "size")).reset_index()
    both = (e.i > 0) & (e.o > 0)
    e["lr"] = np.where(both, np.log(np.maximum(e.o, 1e-9) / np.maximum(e.i, 1e-9)), np.nan)
    g = e.groupby("w")
    res = {}
    def put(series, fill=0.0):
        x = np.full(nw, fill)
        x[series.index.values] = series.fillna(fill).values
        return x
    res["ep_n"] = put(g.size())
    res["ep_both_frac"] = put(both.groupby(e.w).mean())
    res["ep_lr_absmean"] = put(e.lr.abs().groupby(e.w).mean(), 5.0)
    res["ep_lr_mean"] = put(e.lr.groupby(e.w).mean(), 0.0)
    res["ep_lr_std"] = put(e.lr.groupby(e.w).std(), 5.0)
    # running balance: layering wallets hold little relative to throughput
    ev["bal"] = (ev.inv - ev.outv).groupby(ev.w).cumsum()
    gb = ev.groupby("w")
    tot_in = gb.inv.sum().clip(lower=1e-9)
    res["bal_max_rel"] = put(gb.bal.max() / tot_in, 0.0)
    res["bal_min_rel"] = put(gb.bal.min() / tot_in, 0.0)
    res["bal_end_rel"] = put(gb.bal.last() / tot_in, 0.0)
    # in->out lag: median time from an episode's first inflow to its last outflow
    ev["tin"] = np.where(ev.inv > 0, ev.t, np.nan)
    ev["tout"] = np.where(ev.outv > 0, ev.t, np.nan)
    ee = ev.groupby(["w", "ep"]).agg(a=("tin", "min"), b=("tout", "max")).reset_index()
    ee["lag"] = np.log1p((ee.b - ee.a).clip(lower=0))
    res["ep_lag_med"] = put(ee.lag.groupby(ee.w).median(), 0.0)
    return res


def build_features(data_dir):
    data_dir = Path(data_dir)
    pay = pd.DataFrame(load_payments(data_dir))
    tx = pd.read_parquet(data_dir / "transactions.parquet",
                         columns=["TRANSACTION_HASH", "TIMESTAMP", "EXTRA_TRANSACTION_FIELDS", "DELIVERED_AMOUNT"])
    tx = tx.set_index("TRANSACTION_HASH").loc[pay.TRANSACTION_HASH].reset_index()
    t_abs = pay.TIMESTAMP.astype(np.int64).values
    t = t_abs - t_abs.min()
    v = pay.VALUE.astype(float).values
    wallets, codes = np.unique(np.concatenate([pay.FROM_ADDRESS.values, pay.TO_ADDRESS.values]), return_inverse=True)
    n = len(pay)
    s, r = codes[:n], codes[n:]
    eid = np.arange(n)
    OUT = WalletIndex(s, t, v, r, eid)
    IN = WalletIndex(r, t, v, s, eid)
    F = {}
    lv = np.log1p(v)
    F["log_value"] = lv
    F["hour"] = (t_abs // 3600) % 24
    F["dow"] = (t_abs // 86400) % 7
    F["hour_sin"] = np.sin(2 * np.pi * F["hour"] / 24)
    F["hour_cos"] = np.cos(2 * np.pi * F["hour"] / 24)
    for thr in (1000, 3000, 5000, 10000):
        F[f"just_under_{thr}"] = ((v >= 0.9 * thr) & (v < thr)).astype(float)
    F["usd_round_100"] = (np.abs(v / 100 - np.round(v / 100)) < 0.01).astype(float)

    # --- windowed activity counts / sums (sender & receiver, both directions)
    for name, w in WINDOWS.items():
        c, sm = OUT.count_sum(s, t - w, t + w)
        F[f"s_out_cnt_{name}"], F[f"s_out_sum_{name}"] = c, np.log1p(sm)
        c, sm = IN.count_sum(s, t - w, t, hi_incl=False)
        F[f"s_in_before_cnt_{name}"], F[f"s_in_before_sum_{name}"] = c, np.log1p(sm)
        c, sm = IN.count_sum(r, t - w, t + w)
        F[f"r_in_cnt_{name}"], F[f"r_in_sum_{name}"] = c, np.log1p(sm)
        c, sm = OUT.count_sum(r, t, t + w, lo_incl=False)
        F[f"r_out_after_cnt_{name}"], F[f"r_out_after_sum_{name}"] = c, np.log1p(sm)
        # flow ratios (pass-through around this payment)
        F[f"fwd_ratio_{name}"] = np.minimum(np.expm1(F[f"r_out_after_sum_{name}"]) / np.maximum(v, 1e-9), 5)
        F[f"fund_ratio_{name}"] = np.minimum(np.expm1(F[f"s_in_before_sum_{name}"]) / np.maximum(v, 1e-9), 5)

    # --- unique counterparties in windows (fan-in / fan-out breadth)
    for name in ("1h", "6h", "24h"):
        w = WINDOWS[name]
        for side, idx, who, lo, hi in (("s_out", OUT, s, -w, w), ("r_in", IN, r, -w, w),
                                       ("s_in", IN, s, -w, 0), ("r_out", OUT, r, 0, w)):
            a, b = idx.range(who, t + lo, t + hi)
            F[f"{side}_uniq_{name}"] = np.array([len(set(idx.other[x:y])) for x, y in zip(a, b)], float)

    # --- wallet-level 7-day aggregates
    g_out = pd.DataFrame({"w": s, "v": v, "o": r}).groupby("w")
    g_in = pd.DataFrame({"w": r, "v": v, "o": s}).groupby("w")
    nw = len(wallets)

    def agg(g, col, fn):
        x = np.zeros(nw)
        a = getattr(g[col], fn)()
        x[a.index.values] = a.values
        return x

    out_cnt, out_sum, out_uniq = agg(g_out, "v", "count"), agg(g_out, "v", "sum"), agg(g_out, "o", "nunique")
    in_cnt, in_sum, in_uniq = agg(g_in, "v", "count"), agg(g_in, "v", "sum"), agg(g_in, "o", "nunique")
    passthru = np.minimum(out_sum / np.maximum(in_sum, 1e-9), 10)
    for pre, who in (("s", s), ("r", r)):
        F[f"{pre}_w_out_cnt"] = out_cnt[who]
        F[f"{pre}_w_in_cnt"] = in_cnt[who]
        F[f"{pre}_w_out_uniq"] = out_uniq[who]
        F[f"{pre}_w_in_uniq"] = in_uniq[who]
        F[f"{pre}_w_passthru"] = passthru[who]
        F[f"{pre}_w_in_sum"] = np.log1p(in_sum[who])
        F[f"{pre}_w_out_sum"] = np.log1p(out_sum[who])
        F[f"{pre}_w_uniq_ratio"] = (out_uniq[who] + 1) / (in_uniq[who] + 1)

    # --- holding times
    a, b = IN.range(s, -BIG // 2, t, hi_incl=False)
    last_in = np.where(b > a, IN.t[np.maximum(b - 1, 0)], np.nan)
    F["s_hold_before"] = np.log1p(np.nan_to_num(t - last_in, nan=7 * 86400))
    a, b = OUT.range(r, t, BIG // 2, lo_incl=False)
    next_out = np.where(b > a, OUT.t[np.minimum(a, len(OUT.t) - 1)], np.nan)
    F["r_min_to_forward"] = np.log1p(np.nan_to_num(next_out - t, nan=7 * 86400))
    # wallet median holding time: for each outgoing payment, time since wallet's last incoming
    a, b = IN.range(s, -BIG // 2, t, hi_incl=False)
    hold = pd.Series(F["s_hold_before"]).groupby(s).median()
    hw = np.full(nw, np.log1p(7 * 86400))
    hw[hold.index.values] = hold.values
    F["s_w_median_hold"] = hw[s]
    F["r_w_median_hold"] = hw[r]

    # --- amount similarity to neighbouring flows
    for name in ("1h", "6h"):
        w = WINDOWS[name]
        a, b = OUT.range(r, t, t + w, lo_incl=False)
        F[f"r_fwd_best_sim_{name}"] = np.array(
            [np.min(np.abs(OUT.v[x:y] / vi - 1)) if y > x else 5.0 for x, y, vi in zip(a, b, np.maximum(v, 1e-9))])
        F[f"r_fwd_sim10_cnt_{name}"] = np.array(
            [np.sum(np.abs(OUT.v[x:y] / vi - 1) < 0.1) for x, y, vi in zip(a, b, np.maximum(v, 1e-9))], float)
        a, b = IN.range(s, t - w, t, hi_incl=False)
        F[f"s_in_best_sim_{name}"] = np.array(
            [np.min(np.abs(IN.v[x:y] / vi - 1)) if y > x else 5.0 for x, y, vi in zip(a, b, np.maximum(v, 1e-9))])

    # --- sender history anomaly (leave-one-out z-score of log value)
    df = pd.DataFrame({"s": s, "lv": lv, "r": r})
    gs = df.groupby("s").lv
    cnt, sm, sq = gs.transform("count").values, gs.transform("sum").values, (df.lv ** 2).groupby(df.s).transform("sum").values
    m = (sm - lv) / np.maximum(cnt - 1, 1)
    var = np.maximum((sq - lv ** 2) / np.maximum(cnt - 1, 1) - m ** 2, 1e-4)
    F["s_value_z"] = np.where(cnt > 2, (lv - m) / np.sqrt(var), 0)
    gr = pd.DataFrame({"r": r, "lv": lv}).groupby("r").lv
    F["r_value_z"] = ((lv - gr.transform("mean")) / gr.transform("std").fillna(1).clip(lower=0.01)).fillna(0).values
    F["value_cv_r_in"] = (gr.transform("std").fillna(0) / gr.transform("mean").clip(lower=1e-6)).values
    # --- counterparty novelty
    order = np.lexsort((t, r, s))
    pair_rank = np.empty(n)
    pair_rank[order] = pd.DataFrame({"s": s[order], "r": r[order]}).groupby(["s", "r"]).cumcount().values
    F["pair_rank"] = pair_rank
    F["pair_first"] = (pair_rank == 0).astype(float)
    F["pair_total"] = df.groupby(["s", "r"]).lv.transform("count").values
    F["pair_share_of_sender"] = F["pair_total"] / cnt
    rev = set(zip(r.tolist(), s.tolist()))
    F["reverse_edge_exists"] = np.array([(a_, b_) in rev for a_, b_ in zip(s.tolist(), r.tolist())], float)

    # --- multi-hop downstream / upstream flow (vectorised via cumsum over next-edge ranges)
    for name in ("1h", "6h", "24h"):
        w = WINDOWS[name]
        a, b = OUT.range(r, t, t + w, lo_incl=False)
        d1 = np.minimum((OUT.cv[b] - OUT.cv[a]) / np.maximum(v, 1e-9), 1.0)
        d2 = np.minimum(OUT.wsum(v * d1, a, b) / np.maximum(v, 1e-9), 1.0)
        d3 = np.minimum(OUT.wsum(v * d2, a, b) / np.maximum(v, 1e-9), 1.0)
        F[f"down1_{name}"], F[f"down2_{name}"], F[f"down3_{name}"] = d1, d2, d3
        a, b = IN.range(s, t - w, t, hi_incl=False)
        u1 = np.minimum((IN.cv[b] - IN.cv[a]) / np.maximum(v, 1e-9), 1.0)
        u2 = np.minimum(IN.wsum(v * u1, a, b) / np.maximum(v, 1e-9), 1.0)
        u3 = np.minimum(IN.wsum(v * u2, a, b) / np.maximum(v, 1e-9), 1.0)
        F[f"up1_{name}"], F[f"up2_{name}"], F[f"up3_{name}"] = u1, u2, u3
        F[f"chain_score_{name}"] = d1 + d2 + d3 + u1 + u2 + u3
    # decayed downstream (tau = 1h) for hop 1
    a, b = OUT.range(r, t, t + 6 * 3600, lo_incl=False)
    F["down1_decay"] = np.array([np.sum(OUT.v[x:y] * np.exp(-(OUT.t[x:y] - ti) / 3600)) / max(vi, 1e-9)
                                 for x, y, ti, vi in zip(a, b, t, v)])

    # --- 10-minute bursts
    c, _ = OUT.count_sum(s, t - 600, t + 600)
    F["s_burst_10m"] = c
    c, _ = IN.count_sum(r, t - 600, t + 600)
    F["r_burst_10m"] = c

    # --- cycles / return to origin
    for hz, nm in ((86400, "24h"), (7 * 86400, "7d")):
        ret, hops, reach = cycle_features(s, r, t, v, OUT, horizon=hz)
        F[f"return_origin_{nm}"], F[f"return_hops_{nm}"], F[f"reach_{nm}"] = ret, hops, np.log1p(reach)

    # --- static graph stats
    G = nx.Graph()
    G.add_edges_from(zip(s.tolist(), r.tolist()))
    clus = nx.clustering(G)
    comp = np.zeros(nw)
    for cc in nx.connected_components(G):
        for x in cc:
            comp[x] = len(cc)
    cl = np.array([clus.get(i, 0) for i in range(nw)])
    deg = np.array([G.degree(i) if i in G else 0 for i in range(nw)], float)
    for pre, who in (("s", s), ("r", r)):
        F[f"{pre}_clustering"] = cl[who]
        F[f"{pre}_comp_size"] = np.log1p(comp[who])
        F[f"{pre}_degree"] = deg[who]
    # two-hop neighbourhood size of receiver
    nb_deg = np.array([sum(deg[j] for j in G.neighbors(i)) if i in G else 0 for i in range(nw)])
    F["r_nb_deg_sum"] = np.log1p(nb_deg[r])

    # --- episode-level flow conservation (relays: value in ~= value out, repeatedly, across days)
    ep = episode_features(s, r, t, v, nw)
    for k, arr in ep.items():
        F[f"s_{k}"] = arr[s]
        F[f"r_{k}"] = arr[r]

    feats = pd.DataFrame(F)
    legf = leg_features(data_dir, tx, v)
    feats = pd.concat([feats, legf.reset_index(drop=True)], axis=1)
    # --- phase 3: label-free flow (burst) reconstruction, group-level aggregates (src/flows.py)
    feats = pd.concat([feats, flow_features(pay.FROM_ADDRESS.values, pay.TO_ADDRESS.values, t_abs, v)], axis=1)
    feats.insert(0, "TRANSACTION_HASH", pay.TRANSACTION_HASH.values)
    feats["_sender"] = pay.FROM_ADDRESS.values
    feats["_receiver"] = pay.TO_ADDRESS.values
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feats


STARTER = ["log_value", "s_out_cnt_7d", "r_in_cnt_7d", "r_out_after_cnt_1h", "s_in_before_cnt_1h",
           "s_burst_10m", "r_burst_10m", "r_fwd_sim10_cnt_1h", "r_min_to_forward", "s_w_passthru",
           "r_w_passthru", "fwd_ratio_1h"]


def feature_columns(df, flows=True, gaps=True):
    """flows=False: phase-1 (v1) columns; gaps=False: v2 columns (no phase-4 flow gap stats)."""
    cols = [c for c in df.columns if c not in ("TRANSACTION_HASH",) and not c.startswith("_")]
    if not flows:
        return [c for c in cols if not c.startswith(FLOW_PREFIXES)]
    return cols if gaps else [c for c in cols if not c.endswith(("_gapmed", "_gapcv"))]


FLOW_PREFIXES = ("fl30_", "fl120_", "fl600_", "flf60_", "flf600_", "flf3600_")


if __name__ == "__main__":
    import time
    t0 = time.time()
    f = build_features(sys.argv[1])
    print(f.shape, f"{time.time() - t0:.1f}s")
    print(f.describe().T.to_string())
