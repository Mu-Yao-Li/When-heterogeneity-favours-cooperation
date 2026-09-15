"""Independent full-state Markov and limiting-case checks of all six cases."""
import unittest
import networkx as nx
import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import splu
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from algorithms.exact import analyze, RULE_NAMES
from algorithms.exact_support.thresholds import rr_threshold




def exact_fixation_gradients(graph, rule, payoff):
    """Differentiate the full absorbing chain; no coalescent formula used."""
    w = nx.to_numpy_array(graph, nodelist=range(len(graph)))
    n = len(w)
    degree = w.sum(axis=1)
    p = w / degree[:, None]
    h = p if payoff == "average" else w
    s = h.sum(axis=1)
    full = (1 << n) - 1
    size = full - 1
    matrices = [sp.lil_matrix((size, size)) for _ in range(3)]
    absorbing = [np.zeros(size) for _ in range(3)]
    for mask in range(1, full):
        row = mask - 1
        x = ((mask >> np.arange(n)) & 1).astype(float)
        fb, fc = h @ x, -s * x
        stay = np.array([1.0, 0.0, 0.0])
        for focal in range(n):
            neigh = np.flatnonzero(w[focal])
            if rule == "pairwise_comparison":
                opposite = neigh[x[neigh] != x[focal]]
                rates = np.array([len(opposite)/(2*n*degree[focal]),
                    (fb[opposite]-fb[focal]).sum()/(4*n*degree[focal]),
                    (fc[opposite]-fc[focal]).sum()/(4*n*degree[focal])])
            else:
                candidates = np.append(neigh, focal) if rule == "imitation" else neigh
                opposite = candidates[x[candidates] != x[focal]]
                rates = np.array([len(opposite),
                    (fb[opposite]-fb[candidates].mean()).sum(),
                    (fc[opposite]-fc[candidates].mean()).sum()])/(n*len(candidates))
            target = mask ^ (1 << focal)
            stay -= rates
            for j in range(3):
                if target == full:
                    absorbing[j][row] += rates[j]
                elif target != 0:
                    matrices[j][row, target - 1] += rates[j]
        for j in range(3):
            matrices[j][row, row] += stay[j]
    q0, qb, qc = [m.tocsr() for m in matrices]
    factor = splu(sp.eye(size, format="csc") - q0.tocsc())
    rho = factor.solve(absorbing[0])
    rb = factor.solve(qb @ rho + absorbing[1])
    rc = factor.solve(qc @ rho + absorbing[2])
    ci = np.array([(1 << i)-1 for i in range(n)])
    di = np.array([(full ^ (1 << i))-1 for i in range(n)])
    assert abs(rho[ci].mean()+rho[di].mean()-1) < 1e-10
    return rb[ci].mean()+rb[di].mean(), rc[ci].mean()+rc[di].mean()


class ExactRuleTests(unittest.TestCase):
    def test_six_cases_against_full_state_absorbing_chain(self):
        # Independent first derivatives of the full 2^N-state chain,
        # including both benefit and cost (not only their ratio).
        graphs = [nx.path_graph(7), nx.star_graph(6), nx.cycle_graph(7),
                  nx.complete_graph(7), nx.barabasi_albert_graph(8, 2, seed=19)]
        for graph in graphs:
            for rule in RULE_NAMES:
                for payoff in ("average", "accumulated"):
                    with self.subTest(degrees=sorted(dict(graph.degree()).values()),
                                      rule=rule, payoff=payoff):
                        row = analyze(graph, update_rule=rule, payoff=payoff,
                                      rtol=1e-12, atol=1e-13)
                        benefit, cost = exact_fixation_gradients(
                            graph, RULE_NAMES[rule], payoff)
                        np.testing.assert_allclose(
                            [row["benefit_coefficient"], row["cost_coefficient"]],
                            [benefit, cost], rtol=1e-8, atol=1e-10)
                        if abs(benefit) < 1e-10:
                            self.assertEqual(row["threshold_status"], "no_benefit_effect")
                            self.assertIsNone(row["bc_star_signed"])
                        else:
                            np.testing.assert_allclose(row["bc_star_signed"], -cost / benefit,
                                                       rtol=1e-8, atol=1e-8)
                        if row["threshold_status"] != "finite_positive":
                            self.assertIsNone(row["bc_star"])

    def test_regular_graph_closed_forms_all_rules_and_payoffs(self):
        for n in (12, 30):
            graph = nx.random_regular_graph(4, n, seed=n)
            for rule in RULE_NAMES:
                for payoff in ("average", "accumulated"):
                    row = analyze(graph, update_rule=rule, payoff=payoff)
                    expected = rr_threshold(n, 4, RULE_NAMES[rule])
                    self.assertAlmostEqual(row["bc_star_signed"], expected, places=6)

    def test_default_matches_db_remeeting_formula(self):
        row = analyze(nx.barabasi_albert_graph(30, 2, seed=9))
        self.assertEqual(row["update_rule"], "DB")
        self.assertEqual(row["payoff"], "average")
        self.assertEqual(row["mutation_rate"], 0.0)
        self.assertAlmostEqual(row["remeeting_identity_residual"], 0.0, places=8)
        self.assertAlmostEqual(row["bc_star_signed"],
            (row["N_eff"] - 2) / (row["R_N"] - 2), places=8)
        self.assertAlmostEqual(row["epsilon"],
            row["R_N"] / row["N_eff"] - 1 / row["avg_degree"], places=10)

    def test_rejects_graphs_outside_model(self):
        bad = [nx.DiGraph([(0, 1), (1, 0)]), nx.MultiGraph([(0, 1), (0, 1)]),
               nx.disjoint_union(nx.path_graph(3), nx.path_graph(3))]
        loop = nx.cycle_graph(4)
        loop.add_edge(0, 0)
        bad.append(loop)
        weighted = nx.path_graph(3)
        weighted[0][1]["weight"] = 2
        bad.append(weighted)
        for graph in bad:
            with self.assertRaises(ValueError):
                analyze(graph)


if __name__ == "__main__":
    unittest.main()

