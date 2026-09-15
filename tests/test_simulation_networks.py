"""Small transition-kernel checks and network-generator invariants."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import networkx as nx
import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "algorithms"))
from generate_network import generate  # noqa: E402
from simulate import _choose_source, simulate  # noqa: E402


class SimulationChecks(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (3, 4)])
        self.a = sparse.csr_matrix(nx.to_scipy_sparse_array(self.graph, nodelist=range(5), dtype=float))

    def test_all_update_kernels_match_independent_flip_probabilities(self):
        state = np.array([1, 0, 1, 0, 0])
        neighbours = [np.flatnonzero(self.a.toarray()[i]) for i in range(5)]
        degree = np.array([len(row) for row in neighbours])
        n_c = np.asarray(self.a @ state).astype(int)
        b, c, delta, mu = 4.3, 1.0, 0.4, 0.05
        trials = 15_000
        for payoff in ("average", "accumulated"):
            f = b * n_c - c * degree * state
            if payoff == "average":
                f = f / degree
            for rule in ("DB", "PC", "IM"):
                with self.subTest(rule=rule, payoff=payoff):
                    expected = np.zeros(5)
                    for i in range(5):
                        if rule == "PC":
                            adoption = sum((state[j] != state[i]) /
                                           (1 + math.exp(delta * float(f[i] - f[j])))
                                           for j in neighbours[i]) / degree[i]
                        else:
                            candidates = [i, *neighbours[i]] if rule == "IM" else neighbours[i]
                            fitness = np.exp(delta * f[candidates])
                            adoption = float(sum(w for j, w in zip(candidates, fitness)
                                                 if state[j] != state[i]) / fitness.sum())
                        expected[i] = (mu / 2 + (1 - mu) * adoption) / 5
                    rng = np.random.default_rng(73041)
                    observed = np.zeros(5)
                    for _ in range(trials):
                        i = int(rng.integers(5))
                        j = _choose_source(i, state, n_c, neighbours, degree,
                                           b, c, delta, rule, payoff, rng)
                        new = int(rng.integers(2)) if rng.random() < mu else state[j]
                        observed[i] += new != state[i]
                    standard_error = np.sqrt(expected * (1 - expected) / trials)
                    self.assertTrue(np.all(abs(observed / trials - expected) < 7 * standard_error + 1 / trials))

    def test_incremental_counts_and_seed_reproducibility(self):
        for payoff in ("average", "accumulated"):
            for rule in ("DB", "PC", "IM"):
                with self.subTest(rule=rule, payoff=payoff):
                    settings = dict(rule=rule, payoff=payoff, steps=500, seed=382,
                                    mutation_rate=0.1, delta=0.3)
                    first = simulate(self.a, **settings)
                    second = simulate(self.a, **settings)
                    np.testing.assert_array_equal(first["final_state"], second["final_state"])
                    self.assertEqual(first["cooperation_sum"], second["cooperation_sum"])
                    np.testing.assert_array_equal(first["cooperative_neighbours"], self.a @ first["final_state"])

    def test_absorbing_states_count_all_holds_and_partial_trace_block(self):
        for rule in ("DB", "PC", "IM"):
            trace = []
            result = simulate(self.a, rule=rule, steps=23, burn_in=7, mutation_rate=0,
                              initial_state=np.ones(5), record_every=10,
                              trace_hook=lambda *row: trace.append(row))
            self.assertEqual(result["cooperation_sum"], 5 * 23)
            self.assertEqual(result["total_updates"], 30)
            self.assertEqual(result["mean_cooperation"], 1)
            self.assertEqual(trace, [(10, 1, 1), (20, 1, 1), (23, 1, 1)])


class NetworkChecks(unittest.TestCase):
    def test_all_families_are_connected_with_requested_edges(self):
        for model in ("PA", "RR", "ER", "SW", "BA", "HK", "KE", "Shifted", "FF",
                      "IslandBA", "IslandER", "Core-periphery"):
            with self.subTest(model=model):
                result = generate(model, n=100, k=4, seed=42)
                self.assertEqual(result.graph.number_of_nodes(), 100)
                self.assertEqual(result.graph.number_of_edges(), 200)
                self.assertEqual(nx.number_of_selfloops(result.graph), 0)
                self.assertTrue(nx.is_connected(result.graph))

    def test_pa_seed_and_legacy_ba_are_explicit_distinct_conventions(self):
        first = generate("PA", n=100, k=4, seed=42, gamma=1)
        second = generate("PA", n=100, k=4, seed=42, gamma=1)
        legacy = generate("BA", n=100, k=4, seed=42)
        self.assertEqual(set(first.graph.edges()), set(second.graph.edges()))
        self.assertNotEqual(set(first.graph.edges()), set(legacy.graph.edges()))
        self.assertEqual(first.metadata["initial_complete_graph_size"], 3)

    def test_impossible_density_rejected_before_generator(self):
        for k in (0, 1, 100, float("nan")):
            with self.assertRaises(ValueError):
                generate("ER", n=100, k=k)


if __name__ == "__main__":
    unittest.main()
