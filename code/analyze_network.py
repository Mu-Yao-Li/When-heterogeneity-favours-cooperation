"""Portable exact analysis for manuscript Eqs. 5-10; no node-level files."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

from run_adaptive_closed_tail_pipeline import load_edge_list
from run_pa_gamma_metric_grid import compute_remeeting_metrics
from sparse_threshold import solve_eta_single_layer_linear_operator


def analyze(adjacency: sparse.spmatrix, *, rtol: float = 1e-10) -> dict:
    a = sparse.csr_matrix(adjacency, dtype=float)
    a.eliminate_zeros()
    n = a.shape[0]
    if a.shape != (n, n) or n < 3:
        raise ValueError('A square adjacency matrix with at least 3 nodes is required.')
    if (a != a.T).nnz or np.any(a.diagonal()) or np.any(a.data != 1):
        raise ValueError('Expected a symmetric, binary adjacency matrix without self-loops.')
    if csgraph.connected_components(a, directed=False, return_labels=False) != 1:
        raise ValueError('The graph must be connected.')
    degree = np.asarray(a.sum(axis=1)).ravel()
    k = float(degree.mean())
    eta, stats = solve_eta_single_layer_linear_operator(
        a, rtol=rtol, atol=1e-12, maxiter=3000, verbose=False
    )
    metrics = compute_remeeting_metrics(a, eta, k)
    transition = sparse.diags(1.0 / degree) @ a
    pi = degree / degree.sum()
    reach = 1.0 + 2.0 * np.asarray(transition.multiply(eta).sum(axis=1)).ravel()
    numerator, denominator = metrics['S_N'] - 2.0, metrics['R_N'] - 2.0
    tol = 1e-10 * max(1.0, abs(metrics['R_N']))
    signed = numerator / denominator if abs(denominator) > tol else math.inf
    threshold = signed if denominator > tol and signed > 0 else math.inf
    rr_denominator = n / k - 2.0
    benchmark = (n - 2.0) / rr_denominator if rr_denominator > 1e-10 else math.inf
    return {
        'N': n, 'edges': a.nnz // 2, 'avg_degree': k,
        'degree_cv': float(degree.std() / k),
        'N_eff': metrics['S_N'], 'epsilon': metrics['epsilon_N'],
        'Delta': metrics['epsilon_crit'], 'margin': metrics['margin_M'],
        'threshold_signed': signed, 'threshold': threshold,
        'threshold_denominator': denominator, 'threshold_RR': benchmark,
        'benchmark_defined': math.isfinite(benchmark),
        'heterogeneity_advantage': math.isfinite(threshold) and math.isfinite(benchmark)
                                   and threshold < benchmark,
        'remeeting_identity_residual': float((pi * pi) @ reach - 1.0),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rtol', type=float, default=1e-10)
    args = parser.parse_args()
    adjacency, _ = load_edge_list(args.edge_file)
    row = analyze(adjacency, rtol=args.rtol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f'N={row["N"]}, N_eff={row["N_eff"]:.8g}, threshold={row["threshold"]:.8g}')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
