"""Final blend used for the 65.7 public-LB run (shiftdrop_plus_s2.csv).
final = 0.6 * rank(base: src/predict.py default run, LB 64.9) + 0.4 * rank(stage-2 neighbour model, predict_s2.py fold_all 0.4)
Usage: python blend.py BASE_CSV S2_NPY TEST_BASE_PARQUET OUT_CSV [ALPHA]"""
import sys, numpy as np, pandas as pd
base_csv, s2_npy, tb_path, out_csv = sys.argv[1:5]
a = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
tb = pd.read_parquet(tb_path)
d = pd.DataFrame({"TRANSACTION_HASH": tb.TRANSACTION_HASH.values, "s2": np.load(s2_npy)})
d = d.merge(pd.read_csv(base_csv), on="TRANSACTION_HASH")
assert len(d) == 40101
r = lambda x: x.rank(method="average") / len(x)
f = (1 - a) * r(d.score) + a * r(d.s2) + 1e-9 * r(d.score)
d["score"] = (f.rank(method="first") - 1) / (len(f) - 1)
assert d.score.nunique() == len(d)
d[["TRANSACTION_HASH", "score"]].to_csv(out_csv, index=False)
