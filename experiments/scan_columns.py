"""Scan raw transaction columns we never used as features: do they separate laundering schemes from their decoys?

Every scenario is all-positive or all-decoy and decoys copy family, shape and size, so a useful signal must differ
between positive and decoy SCENARIOS of the same family. For each candidate (raw numeric columns, EXTRA_TRANSACTION_FIELDS
keys, and derived ones such as account age from Sequence) we report, over non-relay scenarios:
  scen_AUC  AUC of the scenario mean, positive vs decoy scenarios (0.5 = useless; far from 0.5 either way = useful)
  fam_AUC   the same inside each family, averaged (controls for family mix)
  bg_mean / pos_mean / decoy_mean  for orientation
Run: python experiments/scan_columns.py [--train-dir TRAIN]   (~30 s; prints a table, nothing written)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import parser  # noqa: E402

RELAY = ("aggregate_relay", "distributed_relay", "ambiguous")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def main():
    a = parser(__doc__).parse_args()
    d = Path(a.train_dir)
    tx = pd.read_parquet(d / "transactions.parquet")
    lab = pd.read_csv(d / "labels.csv")
    tx = tx.merge(lab, on="TRANSACTION_HASH", how="left")
    print("transaction columns:", list(tx.columns), "\n")
    C = pd.DataFrame(index=tx.index)
    # 1. every raw column that is numeric (or numeric text) and not constant
    for c in tx.columns:
        if c in lab.columns:
            continue
        x = num(tx[c]) if tx[c].dtype == object or str(tx[c].dtype) in ("str", "string") else tx[c]
        if pd.api.types.is_numeric_dtype(x) and x.notna().mean() > 0.5 and x.nunique() > 1:
            C[f"raw:{c}"] = x.astype(float)
    # 2. keys inside EXTRA_TRANSACTION_FIELDS: presence + numeric value
    if "EXTRA_TRANSACTION_FIELDS" in tx:
        ex = [json.loads(e) if isinstance(e, str) else {} for e in tx.EXTRA_TRANSACTION_FIELDS]
        keys = sorted({k for e in ex for k in e})
        print("EXTRA_TRANSACTION_FIELDS keys:", keys, "\n")
        for k in keys:
            C[f"has:{k}"] = [float(k in e) for e in ex]
            v = num(pd.Series([e.get(k) if not isinstance(e.get(k), (dict, list)) else None for e in ex]))
            if v.notna().mean() > 0.05 and v.nunique() > 1:
                C[f"extra:{k}"] = v.values
    # 3. derived: account age / history from Sequence (XRPL: a new account's Sequence = its creation ledger index)
    seq = next((c for c in tx.columns if c.upper() in ("SEQUENCE", "ACCOUNT_SEQUENCE")), None)
    led = next((c for c in tx.columns if c.upper() == "LEDGER_INDEX"), None)
    acct = next((c for c in tx.columns if c.upper() == "ACCOUNT"), None)
    ts = next((c for c in tx.columns if c.upper() == "TIMESTAMP"), None)
    if seq and led:
        C["age:ledger_minus_seq"] = np.log1p((num(tx[led]) - num(tx[seq])).clip(lower=0))
    if seq and acct and ts:
        o = tx.assign(_s=num(tx[seq]), _t=num(tx[ts])).sort_values([acct, "_t"])
        gap = o.groupby(acct)._s.diff()  # > 1: the account sent transactions that are not in the data
        C.loc[o.index, "seq:gap_to_prev"] = np.log1p(gap.clip(lower=0))
        first = o.groupby(acct)._s.transform("min")
        C.loc[o.index, "seq:sender_min_seq_rank"] = first.rank(pct=True)
        C.loc[o.index, "seq:skipped_share"] = o.groupby(acct)._s.transform(lambda s: (s.diff() > 1).mean())
    # 4. delivered amount precision (number of decimals in native units)
    if "DELIVERED_AMOUNT" in tx:
        def dec(w):
            w = json.loads(w)
            v = w["value"] if isinstance(w, dict) else str(int(w) / 1e6)
            return len(v.split(".")[1].rstrip("0")) if "." in v else 0
        C["amt:decimals"] = [dec(w) for w in tx.DELIVERED_AMOUNT]

    sc = tx.scenario_id.notna() & ~tx.family.isin(RELAY)
    rows = []
    for c in C.columns:
        g = pd.DataFrame({"x": C[c], "y": tx.is_suspicious, "s": tx.scenario_id, "f": tx.family})[sc].dropna()
        if g.x.nunique() < 2:
            continue
        s = g.groupby("s").agg(x=("x", "mean"), y=("y", "max"), f=("f", "first"))
        if s.y.nunique() < 2:
            continue
        auc = roc_auc_score(s.y, s.x)
        fa = [roc_auc_score(h.y, h.x) for _, h in s.groupby("f") if h.y.nunique() == 2 and h.x.nunique() > 1]
        bg = C[c][tx.family == "background"].mean()
        rows.append((c, auc, np.mean([max(v, 1 - v) for v in fa]) if fa else np.nan,
                     bg, s.x[s.y == 1].mean(), s.x[s.y == 0].mean()))
    out = pd.DataFrame(rows, columns=["candidate", "scen_AUC", "fam_AUC(|.5|-folded)", "bg_mean", "pos_mean", "decoy_mean"])
    out["strength"] = (out.scen_AUC - 0.5).abs()
    print(out.sort_values("strength", ascending=False).round(3).to_string(index=False))
    npos, nneg = int((s_all := tx[sc].groupby("scenario_id").is_suspicious.max()).sum()), int((s_all == 0).sum())
    band = 2 * np.sqrt((npos + nneg + 1) / (12 * npos * nneg))  # ~2 std of AUC under no signal
    print(f"\n{npos} positive vs {nneg} decoy scenarios: chance band for scen_AUC is about 0.5 +- {band:.2f}. "
          "Outside it (and ideally >= 0.65 or <= 0.35) is worth a feature in src/features.py.")


if __name__ == "__main__":
    main()
