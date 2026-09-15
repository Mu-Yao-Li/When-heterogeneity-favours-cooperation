"""Exact rare-mutation weak-selection donation-game thresholds.

Connected simple undirected graphs; uniform focal updating and symmetric
offspring mutation. The criterion is rho_C > rho_D, with uniformly located
single mutants, equivalently stationary abundance > 1/2 as u tends to zero.
DB and IM select a parent proportional to 1 + delta * payoff, from the
neighbors and closed neighborhood respectively. PC uses F_j/(F_i+F_j).

Let H=P (average) or W (accumulated), s=H*1, and
T(M)=sum_ij pi_i M_ij tau_ij. For DB/IM the ancestry walk is Q=P or
(D+I)^(-1)(W+I), tau=1+(Q*tau+tau*Q.T)/2 off diagonal, tau_ii=0.
The cost is T(Q^2 diag(s)); the benefit is T(Q^2 H)-T(H).
For PC use Q=P and replace Q^2 by P in the contractions. Lazy PC ancestry
is accounted for in the time scale. In all cases the derivative of
rho_C-rho_D is (b*benefit-c*cost)/N, with this tau convention.

The formulas follow the neutral occupation/coalescence representation in
McAvoy & Allen (2021), DOI 10.1007/s00285-021-01568-4. Sparse walk propagation
avoids explicitly forming Q^2 H, which can become dense around large hubs.
Adapted from the local research extension sigma/rare_mutation_thresholds.py;
see PROVENANCE.md for provenance.
"""

from __future__ import annotations

import gc
import time
import numpy as np
from scipy import sparse as sp

from .coalescence import _as_unweighted_adjacency, solve_im_coalescence_times

RULES = ("death_birth", "imitation", "pairwise_comparison")
PAYOFFS = ("average", "accumulated")
METHOD = "exact_rare_mutation_coalescence_v1"


def rr_threshold(n: int, k: int, rule: str) -> float:
    if rule == "death_birth":
        numerator, denominator = k * (n - 2), n - 2 * k
    elif rule == "imitation":
        numerator = n * (k + 2) - 2 * (k + 1)
        denominator = n - 2 * (k + 1)
    elif rule == "pairwise_comparison":
        return -float(n - 1)
    else:
        raise ValueError(rule)
    return float(numerator / denominator) if denominator else float("nan")


def _contract(w, p, pi, tau, propagated, rule, payoff, stats):
    start = time.perf_counter()
    h = p if payoff == "average" else w
    s = np.asarray(h.sum(axis=1)).ravel()
    # Reversibility moves K from T(K H) to K*tau; no matrix powers needed.
    hcoo = h.tocoo()
    benefit = float(np.dot(pi[hcoo.row] * hcoo.data,
        propagated[hcoo.row, hcoo.col] - tau[hcoo.row, hcoo.col]))
    cost = float(np.dot(pi * s, np.diag(propagated)))
    tolerance = 1e-10 * max(abs(benefit), abs(cost), 1.0)
    if abs(benefit) <= tolerance:
        threshold, status = float("nan"), "no_benefit_effect"
    else:
        threshold = cost / benefit
        if benefit > tolerance and cost > tolerance:
            status = "finite_positive"
        elif benefit < -tolerance and cost > tolerance:
            status = "negative_zero_crossing_no_positive_threshold"
        else:
            status = "nonstandard_signs"
    n = w.shape[0]
    result = {
        "method": METHOD, "mutation_regime": "rare_limit_u_to_zero",
        "mutation_rate": 0.0, "mutant_initialization": "uniform_single_mutant",
        "selection_criterion": "rho_C_greater_than_rho_D",
        "update_rule": rule, "payoff_aggregation": payoff,
        "benefit_coefficient": benefit / n, "cost_coefficient": -cost / n,
        "benefit_coalescence": benefit, "cost_coalescence": cost,
        "bc_star": threshold, "threshold": threshold, "threshold_status": status,
        "linear_iterations": stats["iterations"],
        "relative_residual": stats["relative_residual"],
        "solve_time_sec": stats["solve_time_sec"],
        "identity_iterations": stats["iterations"],
        "identity_relative_residual": stats["relative_residual"],
        "identity_solve_time_sec": stats["solve_time_sec"],
        "contraction_time_sec": time.perf_counter() - start,
    }
    if rule == "death_birth" and payoff == "average":
        degree = np.asarray(w.sum(axis=1)).ravel()
        mean_degree = float(degree.mean())
        remeeting = 1.0 + np.asarray(p.multiply(tau).sum(axis=1)).ravel()
        return_probability = np.asarray(p.multiply(p.T).sum(axis=1)).ravel()
        effective_size = float(pi @ remeeting)
        weighted_return = float(pi @ (remeeting * return_probability))
        epsilon = weighted_return / effective_size - 1.0 / mean_degree
        delta = (2.0 * (mean_degree - 1.0) * (n - effective_size)
            / (mean_degree * effective_size * (n - 2.0))) if n > 2 else float("nan")
        rr_denominator = n / mean_degree - 2.0
        rr = (n - 2.0) / rr_denominator if rr_denominator > 1e-10 else float("nan")
        result.update({
            "N_eff": effective_size, "R_N": weighted_return,
            "epsilon": epsilon, "Delta": delta, "margin": epsilon - delta,
            "threshold_RR": rr,
            "benchmark_defined": bool(np.isfinite(rr) and rr > 0),
            "heterogeneity_advantage": bool(status == "finite_positive"
                and np.isfinite(rr) and rr > 0 and threshold < rr),
            "remeeting_identity_residual": float((pi * pi) @ remeeting - 1.0),
        })
    return result


def rare_mutation_thresholds(structure, *, rules=RULES,
        payoff_aggregations=PAYOFFS, rtol=1e-10, atol=1e-12,
        maxiter=6000, solver="bicgstab", verbose=False):
    """Return six (or a selected subset of) result dictionaries.

    Does not approximate the limit using a small nonzero mutation rate.
    One solve is shared by DB and PC. Average and accumulated payoffs share
    each neutral solve. Node-level statistics are not returned or persisted.
    """
    rules, payoffs = tuple(rules), tuple(payoff_aggregations)
    if not rules or any(r not in RULES for r in rules):
        raise ValueError("Only DB, IM and PC are supported")
    if not payoffs or any(p not in PAYOFFS for p in payoffs):
        raise ValueError("Unknown payoff aggregation")
    w = _as_unweighted_adjacency(structure)
    n = w.shape[0]
    degree = np.asarray(w.sum(axis=1)).ravel()
    p = (sp.diags(1 / degree) @ w).tocsr()
    results = []
    for group in ("simple", "imitation"):
        selected = [r for r in rules if (r == "imitation") == (group == "imitation")]
        if not selected:
            continue
        start = time.perf_counter()
        if group == "simple":
            q, pi = p, degree / degree.sum()
        else:
            q = (sp.diags(1 / (degree + 1)) @ (w + sp.eye(n))).tocsr()
            pi = (degree + 1) / (degree.sum() + n)
        tau, stats = solve_im_coalescence_times(q, solver=solver, rtol=rtol, atol=atol,
            maxiter=maxiter, verbose=verbose)
        if not np.isfinite(stats["relative_residual"]) or stats["relative_residual"] > 1e-8:
            raise RuntimeError(f"Coalescence residual fails QC: {stats}")
        if not np.isfinite(tau).all() or np.min(tau) < -1e-8:
            raise RuntimeError("Non-finite or negative coalescence times")
        once = q @ tau
        group_results = []
        if "pairwise_comparison" in selected:
            group_results += [_contract(w, p, pi, tau, once,
                "pairwise_comparison", payoff, stats) for payoff in payoffs]
        if any(r != "pairwise_comparison" for r in selected):
            twice = q @ once
            del once
            for rule in selected:
                if rule != "pairwise_comparison":
                    group_results += [_contract(w, p, pi, tau, twice,
                        rule, payoff, stats) for payoff in payoffs]
            del twice
        else:
            del once
        for result in group_results:
            result["shared_neutral_time_sec"] = stats["solve_time_sec"]
            result["rule_group_wall_time_sec"] = time.perf_counter() - start
        results.extend(group_results)
        del tau
        gc.collect()
    return sorted(results, key=lambda r: (rules.index(r["update_rule"]),
        payoffs.index(r["payoff_aggregation"])))
