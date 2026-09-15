"""Check the approximation against the analytic regular-network benchmark."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from algorithms.approximate import analyze


class ApproximationChecks(unittest.TestCase):
    def test_regular_graph_benchmark_and_no_node_files(self):
        # Identity closure fixes N_eff=N on a cycle; p_i=1/2 is known analytically.
        with tempfile.TemporaryDirectory() as tmp:
            edge = Path(tmp) / "cycle.edgelist"
            edge.write_text("\n".join(f"{i} {(i + 1) % 12}" for i in range(12)))
            row = analyze(edge, samples=1000, seed=42)
            self.assertEqual(row["status"], "ok")
            self.assertAlmostEqual(row["N_eff"], 12, places=10)
            self.assertAlmostEqual(row["bc_star"], (12 - 2) / (12 / 2 - 2), places=10)
            self.assertAlmostEqual(row["epsilon"], 0, places=12)
            self.assertAlmostEqual(row["remeeting_identity_residual"], 0, places=12)
            self.assertEqual(list(Path(tmp).iterdir()), [edge])

    def test_invalid_sample_settings_fail_before_computation(self):
        edge = ROOT / "examples/pa_n100_k4_seed42.edgelist"
        for kwargs in ({"samples": 3}, {"max_samples": 10}, {"chunk_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyze(edge, **kwargs)


if __name__ == "__main__":
    unittest.main()
