"""Check Figure 2 statistics against every archived graph, without subsampling."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figures.reproduce_fig2 import (
    DATA_DIR, PLOTTED_SIZES, aggregate_ensemble, load_ensemble, k4_observed_nc,
)


class Figure2EnsembleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool = pd.concat(
            [pd.read_csv(DATA_DIR / name, float_precision="round_trip")
             for name in ("sampling_pool.csv.gz", "supplemental_n10000.csv")],
            ignore_index=True,
        )
        supplemented = {100, 200, 316, 500, 1000, 1200, 1300, 1500, 1600, 1700,
                        1800, 2000, 3162, 5000, 10000}
        baseline = {300, 700, 1400, 1900, 3000, 7000}
        cls.counts = {n: 500 for n in supplemented} | {n: 100 for n in baseline}

    def test_full_ensemble_statistics_and_stored_views(self):
        ba, rr, mechanism = load_ensemble()
        self.assertEqual(mechanism.set_index("N").sample_count.to_dict(), self.counts)
        self.assertEqual(int(mechanism.sample_count.sum()), 8100)
        self.assertTrue(mechanism.set_index("N").loc[list(PLOTTED_SIZES), "sample_count"].eq(500).all())
        pd.testing.assert_frame_equal(ba.rename(columns={"n": "N"}), mechanism)
        for n, group in self.pool.groupby("N"):
            row = mechanism.set_index("N").loc[n]
            for source, mean, stem in [
                ("threshold", "threshold_mean", "threshold"),
                ("epsilon", "cov_over_Etau_mean", "cov_over_Etau"),
                ("epsilon_crit", "epsilon_crit", "epsilon_crit"),
            ]:
                values = group[source].to_numpy()
                np.testing.assert_allclose(row[mean], np.mean(values), rtol=1e-12)
                np.testing.assert_allclose(row[stem + "_std"], np.std(values, ddof=1), rtol=1e-12)
                np.testing.assert_allclose(row[stem + "_sem"], np.std(values, ddof=1) / np.sqrt(len(values)), rtol=1e-12)
        for name, actual in [("threshold_ba.csv", ba), ("threshold_regular.csv", rr), ("mechanism.csv", mechanism)]:
            stored = pd.read_csv(DATA_DIR / name, float_precision="round_trip")
            pd.testing.assert_frame_equal(actual, stored, rtol=1e-12, atol=1e-15)
        self.assertEqual(k4_observed_nc(), 1600)

    def test_missing_network_rejected(self):
        with self.assertRaisesRegex(ValueError, "counts differ"):
            aggregate_ensemble(self.pool.iloc[1:], self.counts)

    def test_duplicate_seed_rejected_even_with_correct_counts(self):
        data = self.pool.copy()
        indices = data.index[data.N.eq(100)][:2]
        data.loc[indices[1], "seed"] = data.loc[indices[0], "seed"]
        with self.assertRaisesRegex(ValueError, "repeated graph identities"):
            aggregate_ensemble(data, self.counts)

    def test_nonfinite_value_rejected(self):
        for column in ["threshold", "epsilon", "epsilon_crit"]:
            for value in [np.nan, np.inf]:
                with self.subTest(column=column, value=value):
                    data = self.pool.copy()
                    data.loc[0, column] = value
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        aggregate_ensemble(data, self.counts)


if __name__ == "__main__":
    unittest.main()
