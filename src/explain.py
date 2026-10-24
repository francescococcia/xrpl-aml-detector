"""Worked alerts for the analyst: takes the top-scored test payments in outputs/predictions.csv (v3), rebuilds
each one's label-free flow (600 s undirected component), and explains the score with LightGBM's built-in per-feature
contributions (pred_contrib=True, log-odds units; no extra dependency). Writes outputs/example_alerts.md.

The explaining model is the v3 LightGBM member with seed 42 (same features, weights and parameters as in predict.py),
retrained here on all of train. The alert order comes from the final v3 score, which also blends HGB and logistic
regression and adds flow smoothing, so contributions explain the main member, not every part of the final score."""
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))
from common import cached_features, lgb_params, parser, train_weights  # noqa: E402
from features import feature_columns  # noqa: E402
from payments import load_payments  # noqa: E402

WIN = {"10m": "10 min", "1h": "1 h", "6h": "6 h", "24h": "24 h", "7d": "7 days"}
FLOW = {"fl30": "30 s flow", "fl120": "2 min flow", "fl600": "10 min flow",
        "flf60": "1 min forward flow", "flf600": "10 min forward flow", "flf3600": "1 h forward flow"}
FLOW_STAT = {"size": "number of payments (log)", "dur": "duration (log s)", "rate": "payments per second",
             "ns": "distinct senders", "nr": "distinct receivers", "nw": "distinct wallets",
             "density": "payments per wallet", "v_std": "spread of values (USD)", "v_rel": "this value minus flow mean (USD)",
             "pos": "position in the flow (0 first, 1 last)", "gapmed": "median gap between payments (log s)",
             "gapcv": "irregularity of gaps (std/mean; -1 = single gap)"}
FIXED = {
    "pair_total": "payments between this sender and receiver in the week",
    "pair_rank": "order of this payment among the pair's payments",
    "pair_first": "first payment between this pair",
    "pair_share_of_sender": "share of the sender's payments that go to this receiver",
    "log_value": "payment value (log USD)",
    "r_min_to_forward": "seconds until the receiver forwards money",
    "s_hold_before": "seconds the sender held funds before paying",
    "r_fwd_best_sim_1h": "how closely the receiver's next outflow (1 h) matches this value",
    "s_in_best_sim_1h": "how closely a prior inflow of the sender (1 h) matches this value",
    "r_fwd_best_sim_6h": "how closely the receiver's next outflow (6 h) matches this value",
    "s_in_best_sim_6h": "how closely a prior inflow of the sender (6 h) matches this value",
    "s_ep_lr_std": "variability of the sender's in/out balance per episode",
    "s_w_passthru": "sender's weekly out/in ratio (pass-through)",
    "r_w_passthru": "receiver's weekly out/in ratio (pass-through)",
}


def describe(name):
    if name in FIXED:
        return FIXED[name]
    m = re.match(r"(flf?\d+)_(\w+)$", name)
    if m and m.group(1) in FLOW and m.group(2) in FLOW_STAT:
        return f"{FLOW[m.group(1)]}: {FLOW_STAT[m.group(2)]}"
    m = re.match(r"([sr])_(out|in_before|in|out_after)_(cnt|sum|uniq)_(\w+)$", name)
    if m:
        who = "sender" if m.group(1) == "s" else "receiver"
        what = {"cnt": "number of", "sum": "value (log USD) of", "uniq": "distinct counterparties in"}[m.group(3)]
        d = {"out": "outgoing payments", "in": "incoming payments", "in_before": "incoming payments before this one",
             "out_after": "outgoing payments after this one"}[m.group(2)]
        return f"{who}: {what} {d} within {WIN.get(m.group(4), m.group(4))}"
    return name.replace("_", " ")


def flow_summary(idx, pay, lab):
    """Plain facts about the reconstructed flow (all payments in the same 600 s undirected component)."""
    members = np.where(lab == lab[idx])[0]
    f = pay.iloc[members].sort_values("TIMESTAMP")
    senders, receivers = set(f.FROM_ADDRESS), set(f.TO_ADDRESS)
    wallets = senders | receivers
    inter = senders & receivers  # wallets that both receive and send inside the flow
    v_in = f[f.TO_ADDRESS.isin(inter)].groupby("TO_ADDRESS").VALUE.sum()
    v_out = f[f.FROM_ADDRESS.isin(inter)].groupby("FROM_ADDRESS").VALUE.sum()
    fwd = float(np.minimum(v_in, v_out.reindex(v_in.index).fillna(0)).sum() / v_in.sum()) if len(v_in) else 0.0
    return dict(n=len(f), wallets=len(wallets), senders=len(senders), receivers=len(receivers),
                duration=int(f.TIMESTAMP.max() - f.TIMESTAMP.min()), inter=len(inter),
                wallet_pass=len(inter) / len(wallets), value_pass=fwd, total=f.VALUE.sum(), rows=f)


def short(a):
    return a[:6] + "…" + a[-4:]


def main():
    p = parser(__doc__)
    p.add_argument("--n-alerts", type=int, default=3)
    args = p.parse_args()
    t0 = time.time()
    tr = cached_features(args.train_dir, args.cache_dir, "train")
    te = cached_features(args.test_dir, args.cache_dir, "test")
    labels = pd.read_csv(Path(args.train_dir) / "labels.csv")
    tr = tr.merge(labels[["TRANSACTION_HASH", "is_suspicious", "family"]], on="TRANSACTION_HASH", how="left")
    cols = feature_columns(te)
    model = lgb.LGBMClassifier(**lgb_params()).fit(tr[cols].values, tr.is_suspicious.values, sample_weight=train_weights(tr))

    pred = pd.read_csv(Path(args.out_dir) / "predictions.csv")
    pay = pd.DataFrame(load_payments(args.test_dir))
    assert (pay.TRANSACTION_HASH.values == te.TRANSACTION_HASH.values).all()
    pos = pd.Series(np.arange(len(te)), index=te.TRANSACTION_HASH.values)
    lab = te["_lab600u"].values
    order = pred.sort_values("score", ascending=False).TRANSACTION_HASH.map(pos).values
    rank_of = {i: k + 1 for k, i in enumerate(order)}
    chosen, seen = [], set()
    for i in order:  # top alerts, one per reconstructed flow
        if lab[i] not in seen:
            seen.add(lab[i]); chosen.append(i)
        if len(chosen) == args.n_alerts:
            break
    X = te[cols].values[chosen]
    contrib = model.booster_.predict(X, pred_contrib=True)  # last column = bias (expected log-odds)
    p_lgb = model.predict_proba(X)[:, 1]
    score = pred.set_index("TRANSACTION_HASH").score

    out = ["# Example alerts (v3, test set)", "",
           "Generated by `python src/explain.py` from `outputs/predictions.csv`. Each alert is the highest-ranked payment of a",
           "different reconstructed flow (payments linked by consecutive activity on a shared wallet within 600 s).",
           "Contributions are LightGBM `pred_contrib` values in log-odds (positive = pushes towards suspicious) from the v3",
           "LightGBM member (seed 42). They explain that member, not every part of the blended and smoothed final score.",
           "These are model outputs on synthetic data, not allegations.", ""]
    for k, (i, c) in enumerate(zip(chosen, contrib), 1):
        h = te.TRANSACTION_HASH.values[i]
        fs = flow_summary(i, pay, lab)
        top = np.argsort(-c[:-1])[:3]
        out += [f"## Alert {k}: rank {rank_of[i]} of {len(te):,}, score {score[h]:.6f}", "",
                f"- Transaction: `{h}`",
                f"- Payment: {short(pay.FROM_ADDRESS[i])} → {short(pay.TO_ADDRESS[i])}, {pay.VALUE[i]:,.2f} USD (reference value)",
                f"- Reconstructed flow: **{fs['n']} payments, {fs['wallets']} wallets** ({fs['senders']} senders, "
                f"{fs['receivers']} receivers) in **{fs['duration']} s**, total {fs['total']:,.0f} USD",
                f"- Pass-through: {fs['inter']} of {fs['wallets']} wallets ({fs['wallet_pass']:.0%}) both receive and send "
                f"inside the flow; they forward {fs['value_pass']:.0%} of the value they receive",
                f"- LightGBM member: p = {p_lgb[k - 1]:.3f} (base log-odds {c[-1]:+.2f}, total {c.sum():+.2f})", "",
                "Top 3 reasons:", ""]
        for j, f in enumerate(top, 1):
            out.append(f"{j}. **{describe(cols[f])}** = {te[cols[f]].values[i]:.3g} → {c[f]:+.2f} log-odds (`{cols[f]}`)")
        rows = fs["rows"]
        t_min = rows.TIMESTAMP.min()
        out += ["", "<details><summary>Flow payments (first 12)</summary>", "",
                "| t (s) | from | to | USD |", "|---|---|---|---|"]
        for _, r in rows.head(12).iterrows():
            mark = " **(alert)**" if r.TRANSACTION_HASH == h else ""
            out.append(f"| {int(r.TIMESTAMP - t_min)} | {short(r.FROM_ADDRESS)} | {short(r.TO_ADDRESS)} | {r.VALUE:,.2f}{mark} |")
        out += ["", "</details>", ""]
    dest = Path(args.out_dir) / "example_alerts.md"
    dest.write_text("\n".join(out) + "\n")
    print(f"wrote {dest} ({len(chosen)} alerts) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
