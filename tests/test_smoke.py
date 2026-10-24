"""Smoke tests on tiny synthetic payments; no competition data needed. Run: python -m unittest discover -s tests"""
import sys
import unittest
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import rank01, smooth_scores, train_weights  # noqa: E402
from flows import flow_features  # noqa: E402


def toy():
    # a 4-payment burst A->B->C->D within 30 s, plus two isolated payments hours apart
    s = np.array(["A", "B", "C", "X", "Y", "D"])
    r = np.array(["B", "C", "D", "Z", "W", "E"])
    t = np.array([0, 10, 20, 9000, 20000, 25], dtype=np.int64)
    v = np.array([100.0, 99.0, 98.0, 5.0, 7.0, 97.0])
    return s, r, t, v


class FlowTests(unittest.TestCase):
    def test_shape_and_burst(self):
        s, r, t, v = toy()
        f = flow_features(s, r, t, v)
        self.assertEqual(len(f), len(t))
        self.assertFalse(f.isna().any().any())
        lab = f["_lab600u"].values
        self.assertEqual(len({lab[0], lab[1], lab[2], lab[5]}), 1)  # the burst is one flow
        self.assertNotEqual(lab[3], lab[0])  # isolated payments are not
        self.assertGreater(f["fl30_size"].values[0], f["fl30_size"].values[3])

    def test_feature_columns_count(self):
        s, r, t, v = toy()
        f = flow_features(s, r, t, v)
        feats = [c for c in f.columns if not c.startswith("_")]
        self.assertEqual(len(feats), 72)  # 60 phase-3 + 12 gap stats, as documented


class CommonTests(unittest.TestCase):
    def test_smoothing_range_and_order(self):
        score = np.array([0.9, 0.1, 0.2, 0.3])
        comp = np.array([0, 0, 1, 1])
        sm = smooth_scores(score, [comp, comp], lam=0.25)
        self.assertTrue((sm > 0).all() and (sm <= 1.5).all())  # (0, 1 + lam * n_sets]
        boost = sm - rank01(score)
        self.assertGreater(boost[1], boost[2])  # a payment in a high-score flow gets the bigger lift
        self.assertEqual(smooth_scores.__defaults__, (0.25,))  # final setting (LB 64.5)
        self.assertAlmostEqual(rank01(score).max(), 1.0)

    def test_train_weights_zero_relays(self):
        lab = pd.DataFrame({"family": ["aggregate_relay", "aggregate_relay", "fan_in", "ambiguous"],
                            "is_suspicious": [1, 0, 1, 1]})
        self.assertEqual(train_weights(lab).tolist(), [0.0, 1.0, 1.0, 0.0])

    def test_asset_code_is_split_stable(self):
        # the code must depend on the currency only, not on order of appearance in a split
        code = lambda c: zlib.crc32(c.encode()) % 100_003  # noqa: E731  (same expression as features.leg_features)
        self.assertEqual(code("USD"), code("USD"))
        src = (Path(__file__).resolve().parents[1] / "src" / "features.py").read_text()
        self.assertIn("zlib.crc32(c.encode()) % 100_003", src)
        self.assertNotIn('f["deliv_asset_code"] = pd.factorize', src)


if __name__ == "__main__":
    unittest.main()
