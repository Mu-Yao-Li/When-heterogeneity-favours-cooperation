"""Independent checks of manuscript equations and packaging conventions."""
from pathlib import Path
import math
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from analyze_network import analyze
from run_adaptive_closed_tail_pipeline import load_edge_list
from single_layer_threshold import build_pa_degree_power_exact
from sparse_threshold import (solve_eta_single_layer_linear_operator,
                              compute_threshold_single_layer_matrix_form)


def adjacency(graph):
    return sparse.csr_matrix(nx.to_scipy_sparse_array(graph, format='csr', dtype=float))


def reference_meeting_times(a):
    """Directly assemble Eq. 5 with RHS=1, independently of production matvec."""
    n = a.shape[0]
    dense = a.toarray()
    p = dense / dense.sum(axis=1)[:, None]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    index = {pair: idx for idx, pair in enumerate(pairs)}
    operator = np.eye(len(pairs))
    for row, (i, j) in enumerate(pairs):
        for k in range(n):
            if k != j:
                operator[row, index[tuple(sorted((k, j)))]] -= p[i, k] / 2
            if k != i:
                operator[row, index[tuple(sorted((i, k)))]] -= p[j, k] / 2
    values = np.linalg.solve(operator, np.ones(len(pairs)))
    result = np.zeros((n, n))
    for (i, j), value in zip(pairs, values):
        result[i, j] = result[j, i] = value
    return result


class NumericalCoreTests(unittest.TestCase):
    def test_meeting_equation_and_factor_two(self):
        graph = nx.barbell_graph(4, 2)
        a = adjacency(graph)
        eta, stats = solve_eta_single_layer_linear_operator(a, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(2 * eta, reference_meeting_times(a), rtol=1e-9, atol=1e-9)
        self.assertLess(stats['final_relative_residual'], 1e-9)

    def test_regular_benchmarks(self):
        for graph, k in [(nx.cycle_graph(12), 2), (nx.watts_strogatz_graph(20, 4, 0), 4)]:
            row = analyze(adjacency(graph))
            n = len(graph)
            self.assertAlmostEqual(row['N_eff'], n, places=7)
            self.assertAlmostEqual(row['epsilon'], 0, places=9)
            self.assertAlmostEqual(row['threshold'], (n - 2) / (n / k - 2), places=7)

    def test_heterogeneous_identities_and_decomposition(self):
        a = build_pa_degree_power_exact(40, 4, 1, 42)
        row = analyze(a)
        eta, _ = solve_eta_single_layer_linear_operator(a, rtol=1e-11, atol=1e-13)
        legacy, _ = compute_threshold_single_layer_matrix_form(a, eta)
        self.assertLess(abs(row['remeeting_identity_residual']), 1e-8)
        self.assertAlmostEqual(row['threshold_signed'], legacy, places=7)
        decomposed = (row['N_eff'] - 2) / (row['N_eff'] * (1 / row['avg_degree'] + row['epsilon']) - 2)
        self.assertAlmostEqual(row['threshold_signed'], decomposed, places=8)

    def test_negative_threshold_not_promoting(self):
        row = analyze(adjacency(nx.complete_graph(8)))
        self.assertLess(row['threshold_signed'], 0)
        self.assertTrue(math.isinf(row['threshold']))
        self.assertFalse(row['heterogeneity_advantage'])

    def test_generator_reproducibility_and_exact_degree(self):
        a = build_pa_degree_power_exact(100, 4, 1, 42)
        b = build_pa_degree_power_exact(100, 4, 1, 42)
        self.assertEqual((a != b).nnz, 0)
        self.assertEqual(a.nnz // 2, 200)

    def test_edge_labels_and_disconnected_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'edges.csv'
            path.write_text('source,target\na,b\nb,c\nc,a\na,b\na,a\n', encoding='utf-8')
            a, labels = load_edge_list(path)
            self.assertEqual(a.nnz, 6)
            self.assertEqual(set(labels), {'a', 'b', 'c'})
            path.write_text('a b\nc d\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_edge_list(path)


if __name__ == '__main__':
    unittest.main()
