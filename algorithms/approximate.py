"""Spectral-cutoff, common-tail approximation: DB, average payoff, rare mutation.

The implementation reuses the production random-walk and closure functions.
Defaults reproduce the fixed R=1000 method; optional targeted supplementation
is enabled only when --max-samples exceeds --samples. Writes one network row.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import platform
from statistics import NormalDist
import sys
import time

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run_adaptive_closed_tail_pipeline import (
    choose_supplement_nodes, closed_tail_estimate, combine_moments,
    estimate_spectral_cutoff, load_edge_list, local_return_two_step,
    simulate_remeeting_moments, stable_seed, tail_uncertainty_diagnostics,
)


def analyze(edge_file: Path, *, samples: int = 1000, max_samples: int | None = None,
            seed: int = 20260823, network_id: str | None = None,
            chunk_size: int = 128) -> dict:
    """Estimate network-level quantities without saving node-level arrays.

    Reproducibility requires the same file ordering, ID, seed, sample settings,
    chunk size and dependency versions. Diagnostic uncertainty concerns the
    estimated tail denominator, not a confidence interval for the threshold.
    """
    if samples < 2 or samples % 2:
        raise ValueError("samples must be an even integer of at least 2")
    max_samples = samples if max_samples is None else max_samples
    if max_samples < samples or chunk_size < 1 or seed < 0:
        raise ValueError("Require max_samples >= samples, chunk_size >= 1 and seed >= 0")
    started = time.perf_counter()
    edge_file = Path(edge_file)
    network_id = network_id or edge_file.stem
    adjacency, _ = load_edge_list(edge_file)
    n = adjacency.shape[0]
    if n < 3:
        raise ValueError("At least three nodes are required")
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    pi = degree / degree.sum()
    p = local_return_two_step(adjacency, degree)
    spectrum = estimate_spectral_cutoff(
        adjacency, tolerance=1e-7, maxiter=None, maximum_l=None,
        seed=stable_seed(seed, network_id, "spectral"),
    )
    half_counts = np.full(n, samples // 2, dtype=np.int64)
    parts = [simulate_remeeting_moments(
        adjacency, sample_counts=half_counts, max_steps=spectrum.adaptive_l,
        chunk_size=chunk_size, seed=stable_seed(seed, network_id, label),
    ) for label in ("initial-first", "initial-second")]
    moments = combine_moments(*parts)
    counts = np.full(n, samples, dtype=np.int64)

    def diagnostics():
        return tail_uncertainty_diagnostics(
            pi=pi, moments=moments, sample_counts=counts,
            z_value=NormalDist().inv_cdf(0.975), variance_inflation=1.25, smoothing=0.5,
        )

    uncertainty, arrays = diagnostics()
    selected = np.zeros(n, dtype=bool)
    if max_samples > samples and uncertainty["tail_relative_ci_halfwidth"] > 0.05:
        selected, _ = choose_supplement_nodes(
            arrays["variance_contribution"], target_share=0.99, maximum_nodes=50,
        )
        extra_counts = selected.astype(np.int64) * (max_samples - samples)
        extra = simulate_remeeting_moments(
            adjacency, sample_counts=extra_counts, max_steps=spectrum.adaptive_l,
            chunk_size=chunk_size, seed=stable_seed(seed, network_id, "targeted"),
        )
        moments = combine_moments(moments, extra)
        counts += extra_counts
        uncertainty, _ = diagnostics()
    estimate, _ = closed_tail_estimate(pi, p, moments, counts)
    neff = float(estimate["effective_size_hat"])
    weighted_return = float(estimate["weighted_return_hat"])
    numerator, denominator = neff - 2.0, weighted_return - 2.0
    tolerance = 1e-12 * max(1.0, abs(weighted_return))
    signed = numerator / denominator if abs(denominator) > tolerance else math.inf
    valid = not estimate["negative_tail"] and denominator > tolerance and signed > 0
    k = float(degree.mean())
    epsilon = weighted_return / neff - 1.0 / k if neff != 0 else math.nan
    delta = 2 * (k - 1) * (n - neff) / (k * neff * (n - 2)) if neff != 0 else math.nan
    rr = (n - 2) / (n / k - 2) if n / k - 2 > tolerance else math.inf
    return {
        "network_id": network_id, "edge_file": edge_file.name,
        "edge_file_sha256": hashlib.sha256(edge_file.read_bytes()).hexdigest(),
        "status": "invalid_negative_tail" if estimate["negative_tail"] else "ok",
        "method": "spectral_cutoff_common_tail", "update_rule": "DB",
        "payoff": "average", "mutation_regime": "rare_mutation",
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "N": n, "edges": adjacency.nnz // 2, "avg_degree": k,
        "seed": seed, "samples": samples, "max_samples": max_samples,
        "chunk_size": chunk_size, "selected_node_count": int(selected.sum()),
        "total_trajectories": int(counts.sum()), "L": spectrum.adaptive_l,
        "spectral_gap": spectrum.spectral_gap,
        "spectral_residual": max(spectrum.residual_lambda_1, spectrum.residual_lambda_2),
        "N_eff": neff, "epsilon": epsilon, "Delta": delta, "margin": epsilon - delta,
        "bc_star_signed": signed, "bc_star": signed if valid else None,
        "threshold_status": "finite_positive" if valid else "no_valid_positive_threshold",
        "threshold_RR": rr,
        "heterogeneity_advantage": valid and math.isfinite(rr) and signed < rr,
        "tail_time": estimate["tail_time"], "negative_tail": estimate["negative_tail"],
        "remeeting_identity_residual": estimate["remeeting_identity_residual"],
        "tail_relative_halfwidth_target": 0.05, "tail_diagnostic_confidence": 0.95,
        "tail_variance_inflation": 1.25, "tail_smoothing": 0.5,
        "supplement_variance_share_target": 0.99, "maximum_selected_nodes": 50,
        "tail_target_met_final": uncertainty["tail_relative_ci_halfwidth"] <= 0.05,
        **uncertainty, "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="One-row CSV output")
    parser.add_argument("--network-id", help="Part of the random-seed derivation")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, help="Default: same as --samples")
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    row = analyze(args.edge_file, samples=args.samples, max_samples=args.max_samples,
                  seed=args.seed, network_id=args.network_id, chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    threshold_text = f"{row['bc_star']:.8g}" if row["bc_star"] is not None else "undefined"
    print(f"{row['status']}: N={row['N']}, L={row['L']}, (b/c)*={threshold_text}")
    print(f"Wrote {args.output}")
    if row["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
