from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh


METHOD = "adaptive_closed_tail_tail_ess"


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int((base_seed + int.from_bytes(digest[:4], "little")) % (2**32 - 1))


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "network"


def iter_edge_tokens(path: Path) -> Iterable[tuple[str, str]]:
    header_words = {"source", "target", "from", "to", "node1", "node2", "u", "v"}
    first_data_line = True
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("%"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if first_data_line and u.lower() in header_words and v.lower() in header_words:
                first_data_line = False
                continue
            first_data_line = False
            if u != v:
                yield u, v


def load_edge_list(path: Path) -> tuple[sparse.csr_matrix, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    labels: dict[str, int] = {}
    original_labels: list[str] = []
    row: list[int] = []
    col: list[int] = []

    def index_for(label: str) -> int:
        index = labels.get(label)
        if index is None:
            index = len(original_labels)
            labels[label] = index
            original_labels.append(label)
        return index

    for u_label, v_label in iter_edge_tokens(path):
        u = index_for(u_label)
        v = index_for(v_label)
        row.extend((u, v))
        col.extend((v, u))
    if not row:
        raise ValueError(f"no valid non-self edges found in {path}")

    n = len(original_labels)
    adjacency = sparse.csr_matrix(
        (np.ones(len(row), dtype=np.float64), (row, col)), shape=(n, n)
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    adjacency.eliminate_zeros()
    degree = np.diff(adjacency.indptr)
    if np.any(degree == 0):
        raise ValueError("isolated nodes are not supported")
    components = int(
        csgraph.connected_components(adjacency, directed=False, return_labels=False)
    )
    if components != 1:
        raise ValueError(f"graph must be connected; found {components} components")
    return adjacency, np.asarray(original_labels, dtype=object)


class CountedNormalizedAdjacency(LinearOperator):
    def __init__(self, adjacency: sparse.csr_matrix):
        self.adjacency = adjacency
        degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
        self.inv_sqrt_degree = 1.0 / np.sqrt(degree)
        self.matvec_count = 0
        super().__init__(dtype=np.dtype(np.float64), shape=adjacency.shape)

    def _matvec(self, vector: np.ndarray) -> np.ndarray:
        self.matvec_count += 1
        scaled = self.inv_sqrt_degree * np.asarray(vector).ravel()
        return self.inv_sqrt_degree * (self.adjacency @ scaled)


@dataclass(frozen=True)
class SpectralResult:
    lambda_1: float
    lambda_2: float
    spectral_gap: float
    adaptive_l: int
    adaptive_l_uncapped: int
    l_was_capped: bool
    matvecs: int
    residual_lambda_1: float
    residual_lambda_2: float
    elapsed_sec: float


def eigen_residual(
    operator: LinearOperator, eigenvalue: float, eigenvector: np.ndarray
) -> float:
    vector = np.asarray(eigenvector, dtype=np.float64)
    denominator = max(float(np.linalg.norm(vector)), np.finfo(float).tiny)
    return float(np.linalg.norm(operator @ vector - eigenvalue * vector) / denominator)


def estimate_spectral_cutoff(
    adjacency: sparse.csr_matrix,
    *,
    tolerance: float,
    maxiter: int | None,
    seed: int,
    maximum_l: int | None,
) -> SpectralResult:
    n = adjacency.shape[0]
    operator = CountedNormalizedAdjacency(adjacency)
    start = time.perf_counter()
    if n <= 4:
        degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
        inv_sqrt = 1.0 / np.sqrt(degree)
        normalized = adjacency.toarray() * inv_sqrt[:, None] * inv_sqrt[None, :]
        values, vectors = np.linalg.eigh(normalized)
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        lambda_1 = float(values[0])
        lambda_2 = float(values[1])
        vector_1 = vectors[:, 0]
        vector_2 = vectors[:, 1]
    else:
        rng = np.random.default_rng(seed)
        try:
            values, vectors = eigsh(
                operator,
                k=2,
                which="LA",
                tol=tolerance,
                maxiter=maxiter,
                v0=rng.standard_normal(n),
            )
        except ArpackNoConvergence as error:
            raise RuntimeError(
                f"ARPACK did not converge; returned {len(error.eigenvalues)} eigenvalues"
            ) from error
        order = np.argsort(values)[::-1]
        values = values[order]
        vectors = vectors[:, order]
        lambda_1 = float(values[0])
        lambda_2 = float(values[1])
        vector_1 = vectors[:, 0]
        vector_2 = vectors[:, 1]

    spectral_gap = float(1.0 - lambda_2)
    if not np.isfinite(spectral_gap) or spectral_gap <= 0:
        raise ValueError(f"non-positive spectral gap: {spectral_gap}")
    uncapped_l = int(math.ceil(2.0 / spectral_gap))
    adaptive_l = uncapped_l if maximum_l is None else min(uncapped_l, maximum_l)
    return SpectralResult(
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        spectral_gap=spectral_gap,
        adaptive_l=adaptive_l,
        adaptive_l_uncapped=uncapped_l,
        l_was_capped=adaptive_l != uncapped_l,
        matvecs=int(operator.matvec_count),
        residual_lambda_1=eigen_residual(operator, lambda_1, vector_1),
        residual_lambda_2=eigen_residual(operator, lambda_2, vector_2),
        elapsed_sec=float(time.perf_counter() - start),
    )


def local_return_two_step(
    adjacency: sparse.csr_matrix, degree: np.ndarray
) -> np.ndarray:
    inverse_degree = 1.0 / degree
    return np.asarray(adjacency @ inverse_degree).ravel() / degree


def simulate_remeeting_moments(
    adjacency: sparse.csr_matrix,
    *,
    sample_counts: np.ndarray,
    max_steps: int,
    chunk_size: int,
    seed: int,
) -> dict[str, np.ndarray | float]:
    counts = np.asarray(sample_counts, dtype=np.int64)
    n = adjacency.shape[0]
    if counts.shape != (n,) or np.any(counts < 0):
        raise ValueError("sample_counts must contain one non-negative value per node")
    if max_steps < 1 or chunk_size < 1:
        raise ValueError("max_steps and chunk_size must be positive")

    indptr = adjacency.indptr.astype(np.int64, copy=False)
    indices = adjacency.indices.astype(np.int64, copy=False)
    degree = np.diff(indptr).astype(np.int64, copy=False)
    sum_y = np.zeros(n, dtype=np.float64)
    sum_z = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(seed)
    order = np.argsort(counts, kind="stable")
    order = order[counts[order] > 0]
    simulated = 0
    padded = 0
    start = time.perf_counter()

    for lo in range(0, len(order), chunk_size):
        nodes = order[lo : lo + chunk_size]
        node_counts = counts[nodes]
        width = len(nodes)
        max_samples = int(node_counts.max())
        active = np.arange(max_samples, dtype=np.int64)[None, :] < node_counts[:, None]
        sources = nodes[:, None]
        a = np.broadcast_to(sources, (width, max_samples)).copy()
        offsets = (
            rng.random((width, max_samples)) * degree[nodes, None]
        ).astype(np.int64)
        b = indices[indptr[nodes, None] + offsets].astype(np.int64)
        alive = active.copy()
        short_value = np.zeros((width, max_samples), dtype=np.int32)
        short_value[active] = 2

        for step in range(1, max_steps + 1):
            move_a = (rng.random((width, max_samples)) < 0.5) & alive
            move_b = (~move_a) & alive
            if move_a.any():
                current = a[move_a]
                offsets = (rng.random(current.size) * degree[current]).astype(np.int64)
                a[move_a] = indices[indptr[current] + offsets]
            if move_b.any():
                current = b[move_b]
                offsets = (rng.random(current.size) * degree[current]).astype(np.int64)
                b[move_b] = indices[indptr[current] + offsets]
            alive &= a != b
            if step < max_steps:
                short_value += alive

        sum_y[nodes] = short_value.astype(np.float64, copy=False).sum(axis=1)
        sum_z[nodes] = alive.astype(np.float64, copy=False).sum(axis=1)
        simulated += int(node_counts.sum())
        padded += int(width * max_samples)

    return {
        "sum_y": sum_y,
        "sum_z": sum_z,
        "elapsed_sec": float(time.perf_counter() - start),
        "simulated_trajectories": float(simulated),
        "padded_trajectories": float(padded),
    }


def combine_moments(
    *parts: dict[str, np.ndarray | float]
) -> dict[str, np.ndarray]:
    return {
        name: np.sum(
            [np.asarray(part[name], dtype=np.float64) for part in parts], axis=0
        )
        for name in ("sum_y", "sum_z")
    }


def closed_tail_estimate(
    pi: np.ndarray,
    p_i: np.ndarray,
    moments: dict[str, np.ndarray],
    sample_counts: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    counts = sample_counts.astype(np.float64)
    short_part = np.asarray(moments["sum_y"], dtype=np.float64) / counts
    survival_l = np.asarray(moments["sum_z"], dtype=np.float64) / counts
    pi2 = pi * pi
    denominator = float(pi2 @ survival_l)
    if denominator <= 0:
        raise ValueError("pi-squared weighted survival at L is zero")
    tail_time = (1.0 - float(pi2 @ short_part)) / denominator
    tau_hat = short_part + survival_l * tail_time
    effective_size = float(pi @ tau_hat)
    weighted_return = float(pi @ (tau_hat * p_i))
    threshold_denominator = weighted_return - 2.0
    tolerance = 1e-12 * max(1.0, abs(weighted_return))
    threshold = (
        math.inf
        if threshold_denominator <= tolerance
        else float((effective_size - 2.0) / threshold_denominator)
    )
    estimate = {
        "tail_time": tail_time,
        "tail_denominator": denominator,
        "negative_tail": tail_time < 0,
        "effective_size_hat": effective_size,
        "weighted_return_hat": weighted_return,
        "threshold_hat": threshold,
        "threshold_denominator": threshold_denominator,
        "remeeting_identity_residual": float(pi2 @ tau_hat - 1.0),
    }
    arrays = {
        "short_part": short_part,
        "survival_l": survival_l,
        "tau_hat": tau_hat,
    }
    return estimate, arrays


def tail_uncertainty_diagnostics(
    *,
    pi: np.ndarray,
    moments: dict[str, np.ndarray],
    sample_counts: np.ndarray,
    z_value: float,
    variance_inflation: float,
    smoothing: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    counts = sample_counts.astype(np.float64)
    survivors = np.asarray(moments["sum_z"], dtype=np.float64)
    survival_hat = survivors / counts
    survival_smooth = (survivors + smoothing) / (counts + 2.0 * smoothing)
    pi2 = pi * pi
    denominator = float(pi2 @ survival_hat)
    denominator_smooth = float(pi2 @ survival_smooth)
    node_variance = (
        variance_inflation
        * pi2**2
        * survival_smooth
        * (1.0 - survival_smooth)
    )
    variance_contribution = node_variance / counts
    variance = float(variance_contribution.sum())
    standard_error = math.sqrt(max(0.0, variance))
    relative_halfwidth = (
        z_value * standard_error / denominator_smooth
        if denominator_smooth > 0
        else math.inf
    )
    concentration_denominator = float((pi2 * survival_smooth) @ (pi2 * survival_smooth))
    concentration_ess = (
        denominator_smooth**2 / concentration_denominator
        if concentration_denominator > 0
        else 0.0
    )
    mc_ess = denominator_smooth**2 / variance if variance > 0 else math.inf
    diagnostics = {
        "tail_denominator": denominator,
        "tail_denominator_smoothed": denominator_smooth,
        "tail_denominator_se": standard_error,
        "tail_relative_ci_halfwidth": relative_halfwidth,
        "tail_mc_effective_sample_size": mc_ess,
        "tail_node_concentration_ess": concentration_ess,
        "tail_survivor_count": float(survivors.sum()),
        "tail_zero_survivor_node_fraction": float(np.mean(survivors == 0)),
        "tail_max_node_variance_share": (
            float(variance_contribution.max() / variance) if variance > 0 else 0.0
        ),
    }
    arrays = {
        "survivors": survivors,
        "survival_hat": survival_hat,
        "survival_smooth": survival_smooth,
        "node_variance": node_variance,
        "variance_contribution": variance_contribution,
    }
    return diagnostics, arrays


def choose_supplement_nodes(
    variance_contribution: np.ndarray,
    *,
    target_share: float,
    maximum_nodes: int,
) -> tuple[np.ndarray, float]:
    total = float(variance_contribution.sum())
    selected = np.zeros(len(variance_contribution), dtype=bool)
    if total <= 0:
        return selected, 0.0
    order = np.argsort(variance_contribution, kind="stable")[::-1]
    cumulative = np.cumsum(variance_contribution[order]) / total
    count = min(maximum_nodes, int(np.searchsorted(cumulative, target_share) + 1))
    selected[order[:count]] = True
    return selected, float(variance_contribution[selected].sum() / total)


def scalar_metadata(record: dict[str, object]) -> dict[str, object]:
    excluded = {"network_id", "case_id", "edge_file"}
    out: dict[str, object] = {}
    for key, value in record.items():
        if key in excluded or key.startswith("Unnamed"):
            continue
        if pd.isna(value):
            continue
        out[key] = value
    return out


def exact_value(record: dict[str, object], *names: str) -> float | None:
    for name in names:
        value = record.get(name)
        if value is not None and not pd.isna(value):
            return float(value)
    return None


def run_network(record: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    start = time.perf_counter()
    network_id = str(record["network_id"])
    filename = safe_name(network_id)
    output_dir = Path(str(config["output_dir"]))
    summary_path = output_dir / "case_summary" / f"{filename}_summary.csv"
    node_path = output_dir / "node_level" / f"{filename}_nodes.csv.gz"
    selected_path = output_dir / "selected_nodes" / f"{filename}_selected.csv.gz"
    error_path = output_dir / "errors" / f"{filename}.traceback.txt"

    try:
        edge_path = Path(str(record["edge_path"]))
        adjacency, original_labels = load_edge_list(edge_path)
        n = adjacency.shape[0]
        degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
        pi = degree / degree.sum()
        p_i = local_return_two_step(adjacency, degree)
        spectral = estimate_spectral_cutoff(
            adjacency,
            tolerance=float(config["spectral_tolerance"]),
            maxiter=(
                None
                if int(config["spectral_maxiter"]) <= 0
                else int(config["spectral_maxiter"])
            ),
            seed=stable_seed(int(config["seed"]), network_id, "spectral"),
            maximum_l=(
                None if int(config["maximum_l"]) <= 0 else int(config["maximum_l"])
            ),
        )

        initial_samples = int(config["initial_samples"])
        maximum_samples = int(config["maximum_samples"])
        first_counts = np.full(n, initial_samples // 2, dtype=np.int64)
        second_counts = np.full(n, initial_samples - initial_samples // 2, dtype=np.int64)
        sample_counts = first_counts + second_counts
        first_run = simulate_remeeting_moments(
            adjacency,
            sample_counts=first_counts,
            max_steps=spectral.adaptive_l,
            chunk_size=int(config["chunk_size"]),
            seed=stable_seed(int(config["seed"]), network_id, "initial-first"),
        )
        second_run = simulate_remeeting_moments(
            adjacency,
            sample_counts=second_counts,
            max_steps=spectral.adaptive_l,
            chunk_size=int(config["chunk_size"]),
            seed=stable_seed(int(config["seed"]), network_id, "initial-second"),
        )
        moments = combine_moments(first_run, second_run)
        initial_moments = {name: values.copy() for name, values in moments.items()}
        initial_estimate, _ = closed_tail_estimate(pi, p_i, moments, sample_counts)
        initial_diagnostics, initial_arrays = tail_uncertainty_diagnostics(
            pi=pi,
            moments=moments,
            sample_counts=sample_counts,
            z_value=float(config["z_value"]),
            variance_inflation=float(config["variance_inflation"]),
            smoothing=float(config["smoothing"]),
        )

        triggered = (
            initial_diagnostics["tail_relative_ci_halfwidth"]
            > float(config["target_tail_relative_halfwidth"])
        )
        selected = np.zeros(n, dtype=bool)
        selected_variance_share = 0.0
        extra_run: dict[str, np.ndarray | float] | None = None
        if triggered:
            selected, selected_variance_share = choose_supplement_nodes(
                initial_arrays["variance_contribution"],
                target_share=float(config["target_tail_variance_share"]),
                maximum_nodes=int(config["maximum_selected_nodes"]),
            )
            extra_counts = np.zeros(n, dtype=np.int64)
            extra_counts[selected] = maximum_samples - sample_counts[selected]
            if np.any(extra_counts > 0):
                extra_run = simulate_remeeting_moments(
                    adjacency,
                    sample_counts=extra_counts,
                    max_steps=spectral.adaptive_l,
                    chunk_size=int(config["chunk_size"]),
                    seed=stable_seed(int(config["seed"]), network_id, "targeted"),
                )
                moments = combine_moments(moments, extra_run)
                sample_counts += extra_counts

        final_estimate, final_arrays = closed_tail_estimate(
            pi, p_i, moments, sample_counts
        )
        final_diagnostics, final_diagnostic_arrays = tail_uncertainty_diagnostics(
            pi=pi,
            moments=moments,
            sample_counts=sample_counts,
            z_value=float(config["z_value"]),
            variance_inflation=float(config["variance_inflation"]),
            smoothing=float(config["smoothing"]),
        )
        exact_threshold = exact_value(record, "threshold_exact", "exact_threshold")
        exact_effective_size = exact_value(
            record, "exact_effective_size", "S_exact", "N_eff_exact"
        )
        mc_initial_time = float(first_run["elapsed_sec"]) + float(second_run["elapsed_sec"])
        mc_extra_time = 0.0 if extra_run is None else float(extra_run["elapsed_sec"])
        initial_trajectories = n * initial_samples
        final_trajectories = int(sample_counts.sum())
        collision_size = float(1.0 / np.sum(pi * pi))

        result: dict[str, object] = {
            "network_id": network_id,
            "edge_file": str(record["edge_file"]),
            "status": "ok",
            "method": METHOD,
            "N": n,
            "edges": int(adjacency.nnz // 2),
            "average_degree": float(degree.mean()),
            "collision_size": collision_size,
            "lambda_1": spectral.lambda_1,
            "lambda_2": spectral.lambda_2,
            "spectral_gap": spectral.spectral_gap,
            "pair_relaxation_time": 2.0 / spectral.spectral_gap,
            "adaptive_L": spectral.adaptive_l,
            "adaptive_L_uncapped": spectral.adaptive_l_uncapped,
            "L_was_capped": spectral.l_was_capped,
            "spectral_matvecs": spectral.matvecs,
            "spectral_residual_lambda_1": spectral.residual_lambda_1,
            "spectral_residual_lambda_2": spectral.residual_lambda_2,
            "spectral_time_sec": spectral.elapsed_sec,
            "initial_samples_per_node": initial_samples,
            "maximum_samples_per_selected_node": maximum_samples,
            "supplement_triggered": triggered,
            "selected_node_count": int(selected.sum()),
            "selected_initial_tail_variance_share": selected_variance_share,
            "R_final_min": int(sample_counts.min()),
            "R_final_median": float(np.median(sample_counts)),
            "R_final_max": int(sample_counts.max()),
            "mean_samples_final": float(sample_counts.mean()),
            "initial_trajectories": initial_trajectories,
            "extra_trajectories": final_trajectories - initial_trajectories,
            "final_trajectories": final_trajectories,
            "trajectory_ratio_to_fixed_R": final_trajectories / initial_trajectories,
            "tail_time_initial": float(initial_estimate["tail_time"]),
            "tail_time_final": float(final_estimate["tail_time"]),
            "negative_tail": bool(final_estimate["negative_tail"]),
            "effective_size_hat": float(final_estimate["effective_size_hat"]),
            "weighted_return_hat": float(final_estimate["weighted_return_hat"]),
            "threshold_hat": float(final_estimate["threshold_hat"]),
            "remeeting_identity_residual": float(
                final_estimate["remeeting_identity_residual"]
            ),
            "target_tail_relative_ci_halfwidth_pct": 100.0
            * float(config["target_tail_relative_halfwidth"]),
            "tail_target_met_final": final_diagnostics[
                "tail_relative_ci_halfwidth"
            ]
            <= float(config["target_tail_relative_halfwidth"]),
            "tail_denominator_initial": initial_diagnostics["tail_denominator"],
            "tail_denominator_final": final_diagnostics["tail_denominator"],
            "initial_tail_relative_ci_halfwidth_pct": 100.0
            * initial_diagnostics["tail_relative_ci_halfwidth"],
            "final_tail_relative_ci_halfwidth_pct": 100.0
            * final_diagnostics["tail_relative_ci_halfwidth"],
            "initial_tail_mc_ess": initial_diagnostics[
                "tail_mc_effective_sample_size"
            ],
            "final_tail_mc_ess": final_diagnostics[
                "tail_mc_effective_sample_size"
            ],
            "initial_tail_node_concentration_ess": initial_diagnostics[
                "tail_node_concentration_ess"
            ],
            "final_tail_node_concentration_ess": final_diagnostics[
                "tail_node_concentration_ess"
            ],
            "final_tail_zero_survivor_node_fraction": final_diagnostics[
                "tail_zero_survivor_node_fraction"
            ],
            "final_tail_max_node_variance_share": final_diagnostics[
                "tail_max_node_variance_share"
            ],
            "mc_initial_time_sec": mc_initial_time,
            "mc_extra_time_sec": mc_extra_time,
            "mc_total_time_sec": mc_initial_time + mc_extra_time,
            "total_time_sec": float(time.perf_counter() - start),
            "node_level_file": "" if bool(config["no_node_output"]) else str(node_path),
            "selected_node_file": str(selected_path) if np.any(selected) else "",
            "error": "",
        }
        for key, value in scalar_metadata(record).items():
            result.setdefault(key, value)
        if exact_threshold is not None:
            result["threshold_exact"] = exact_threshold
            result["threshold_relative_error_pct"] = 100.0 * (
                float(final_estimate["threshold_hat"]) - exact_threshold
            ) / exact_threshold
        if exact_effective_size is not None:
            result["exact_effective_size"] = exact_effective_size
            result["effective_size_relative_error_pct"] = 100.0 * (
                float(final_estimate["effective_size_hat"]) - exact_effective_size
            ) / exact_effective_size

        if not bool(config["no_node_output"]):
            pd.DataFrame(
                {
                    "network_id": network_id,
                    "node": np.arange(n, dtype=np.int64),
                    "original_node_id": original_labels,
                    "degree": degree,
                    "pi": pi,
                    "p_i": p_i,
                    "R_i": sample_counts,
                    "short_part": final_arrays["short_part"],
                    "survival_L": final_arrays["survival_l"],
                    "tau_hat": final_arrays["tau_hat"],
                    "tail_variance_contribution": final_diagnostic_arrays[
                        "variance_contribution"
                    ],
                }
            ).to_csv(node_path, index=False, compression="gzip")

        if np.any(selected):
            selected_indices = np.flatnonzero(selected)
            pd.DataFrame(
                {
                    "network_id": network_id,
                    "node": selected_indices,
                    "original_node_id": original_labels[selected_indices],
                    "degree": degree[selected_indices],
                    "pi": pi[selected_indices],
                    "p_i": p_i[selected_indices],
                    "R_initial": initial_samples,
                    "R_final": sample_counts[selected_indices],
                    "tail_survivors_initial": np.asarray(
                        initial_moments["sum_z"]
                    )[selected_indices],
                    "tail_survivors_final": np.asarray(moments["sum_z"])[
                        selected_indices
                    ],
                    "initial_tail_variance_share": initial_arrays[
                        "variance_contribution"
                    ][selected_indices]
                    / max(
                        float(initial_arrays["variance_contribution"].sum()),
                        np.finfo(float).tiny,
                    ),
                }
            ).to_csv(selected_path, index=False, compression="gzip")

        pd.DataFrame([result]).to_csv(summary_path, index=False)
        if error_path.exists():
            error_path.unlink()
        return result
    except Exception as error:
        result = {
            "network_id": network_id,
            "edge_file": str(record.get("edge_file", "")),
            "status": "error",
            "method": METHOD,
            "total_time_sec": float(time.perf_counter() - start),
            "error": repr(error),
        }
        pd.DataFrame([result]).to_csv(summary_path, index=False)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        return result


def prepare_records(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.edge_file is not None:
        network_id = args.network_id or args.edge_file.stem
        return [
            {
                "network_id": network_id,
                "edge_file": str(args.edge_file),
                "edge_path": str(args.edge_file.resolve()),
            }
        ]

    manifest = pd.read_csv(args.manifest, low_memory=False)
    id_column = "network_id" if "network_id" in manifest.columns else "case_id"
    if id_column not in manifest.columns or "edge_file" not in manifest.columns:
        raise ValueError("manifest requires network_id (or case_id) and edge_file columns")
    if manifest[id_column].astype(str).duplicated().any():
        raise ValueError("manifest network IDs must be unique")
    records: list[dict[str, object]] = []
    for row in manifest.to_dict("records"):
        row["network_id"] = str(row[id_column])
        edge_file = Path(str(row["edge_file"]).replace("\\", "/"))
        edge_path = edge_file if edge_file.is_absolute() else args.edge_root / edge_file
        row["edge_path"] = str(edge_path.resolve())
        records.append(row)
    filenames = [safe_name(str(record["network_id"])) for record in records]
    if len(filenames) != len(set(filenames)):
        raise ValueError("network IDs collide after filename sanitization")
    return records


def existing_success(summary_path: Path) -> dict[str, object] | None:
    if not summary_path.exists():
        return None
    try:
        frame = pd.read_csv(summary_path, low_memory=False)
    except Exception:
        return None
    if len(frame) == 1 and str(frame.iloc[0].get("status", "")) == "ok":
        return frame.iloc[0].to_dict()
    return None


def collect_output_directory(output_dir: Path, expected_networks: int) -> None:
    summary_files = sorted(output_dir.rglob("case_summary/*_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"no per-network summaries found under {output_dir}")
    combined = pd.concat(
        [pd.read_csv(path, low_memory=False) for path in summary_files],
        ignore_index=True,
        sort=False,
    )
    if "network_id" not in combined:
        raise ValueError("per-network summaries do not contain network_id")
    duplicate = combined["network_id"].astype(str).duplicated(keep=False)
    if duplicate.any():
        values = sorted(combined.loc[duplicate, "network_id"].astype(str).unique())
        raise ValueError(f"duplicate network IDs across summaries: {values[:10]}")
    if expected_networks > 0 and len(combined) != expected_networks:
        raise ValueError(
            f"expected {expected_networks} networks but collected {len(combined)}"
        )
    combined = combined.sort_values("network_id").reset_index(drop=True)
    combined_path = output_dir / "network_summary_all.csv"
    combined.to_csv(combined_path, index=False)

    node_files = sorted(output_dir.rglob("node_level/*_nodes.csv.gz"))
    selected_files = sorted(output_dir.rglob("selected_nodes/*_selected.csv.gz"))
    pd.DataFrame({"node_level_file": [str(path.resolve()) for path in node_files]}).to_csv(
        output_dir / "node_level_file_index.csv", index=False
    )
    pd.DataFrame(
        {"selected_node_file": [str(path.resolve()) for path in selected_files]}
    ).to_csv(output_dir / "selected_node_file_index.csv", index=False)
    failed = int(combined["status"].ne("ok").sum())
    print(
        f"collected={len(combined)} failed={failed} node_files={len(node_files)} "
        f"selected_files={len(selected_files)} output={combined_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Adaptive spectral-cutoff, closed-tail threshold estimator with "
            "targeted tail-effective-sample supplementation"
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--edge-file", type=Path)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--collect-only", action="store_true")
    parser.add_argument("--network-id")
    parser.add_argument("--edge-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-samples", type=int, default=1000)
    parser.add_argument("--maximum-samples", type=int, default=5000)
    parser.add_argument("--target-tail-relative-halfwidth", type=float, default=0.05)
    parser.add_argument("--target-tail-variance-share", type=float, default=0.99)
    parser.add_argument("--maximum-selected-nodes", type=int, default=50)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--variance-inflation", type=float, default=1.25)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--spectral-tolerance", type=float, default=1e-7)
    parser.add_argument("--spectral-maxiter", type=int, default=0)
    parser.add_argument("--maximum-l", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-networks", type=int, default=0)
    parser.add_argument("--expected-networks", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-node-output", action="store_true")
    args = parser.parse_args()

    if args.initial_samples < 2 or args.initial_samples % 2:
        raise ValueError("initial-samples must be an even integer of at least two")
    if args.maximum_samples < args.initial_samples:
        raise ValueError("maximum-samples must be at least initial-samples")
    if not 0 < args.target_tail_relative_halfwidth < 1:
        raise ValueError("target-tail-relative-halfwidth must lie in (0, 1)")
    if not 0 < args.target_tail_variance_share <= 1:
        raise ValueError("target-tail-variance-share must lie in (0, 1]")
    if args.maximum_selected_nodes < 1:
        raise ValueError("maximum-selected-nodes must be positive")
    if not 0 < args.confidence_level < 1:
        raise ValueError("confidence-level must lie in (0, 1)")
    if args.variance_inflation < 1 or args.smoothing <= 0:
        raise ValueError("invalid variance-inflation or smoothing")
    if args.workers < 1 or args.shard_count < 1:
        raise ValueError("workers and shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must lie in [0, shard-count)")

    for name in ("case_summary", "node_level", "selected_nodes", "errors"):
        (args.output_dir / name).mkdir(parents=True, exist_ok=True)
    if args.collect_only:
        collect_output_directory(args.output_dir, args.expected_networks)
        return
    records = prepare_records(args)
    records = [
        record
        for record in records
        if stable_seed(0, str(record["network_id"]), "shard") % args.shard_count
        == args.shard_index
    ]
    if args.max_networks > 0:
        records = records[: args.max_networks]

    completed: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    for record in records:
        summary_path = (
            args.output_dir
            / "case_summary"
            / f"{safe_name(str(record['network_id']))}_summary.csv"
        )
        existing = None if args.overwrite else existing_success(summary_path)
        if existing is None:
            pending.append(record)
        else:
            completed.append(existing)

    z_value = NormalDist().inv_cdf(0.5 + args.confidence_level / 2.0)
    config: dict[str, object] = {
        "output_dir": str(args.output_dir.resolve()),
        "initial_samples": args.initial_samples,
        "maximum_samples": args.maximum_samples,
        "target_tail_relative_halfwidth": args.target_tail_relative_halfwidth,
        "target_tail_variance_share": args.target_tail_variance_share,
        "maximum_selected_nodes": args.maximum_selected_nodes,
        "z_value": z_value,
        "variance_inflation": args.variance_inflation,
        "smoothing": args.smoothing,
        "chunk_size": args.chunk_size,
        "spectral_tolerance": args.spectral_tolerance,
        "spectral_maxiter": args.spectral_maxiter,
        "maximum_l": args.maximum_l,
        "seed": args.seed,
        "no_node_output": args.no_node_output,
    }
    print(
        f"method={METHOD} shard={args.shard_index}/{args.shard_count} "
        f"networks={len(records)} pending={len(pending)} workers={args.workers} "
        f"R={args.initial_samples}->{args.maximum_samples}",
        flush=True,
    )

    results = completed.copy()
    if args.workers == 1:
        for index, record in enumerate(pending, start=1):
            result = run_network(record, config)
            results.append(result)
            print(
                f"[{index}/{len(pending)}] {result['network_id']} "
                f"status={result['status']} L={result.get('adaptive_L', 0)} "
                f"selected={result.get('selected_node_count', 0)} "
                f"threshold={result.get('threshold_hat', math.nan):.8g} "
                f"elapsed={result.get('total_time_sec', math.nan):.2f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_network, record, config): str(record["network_id"])
                for record in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[{index}/{len(futures)}] {futures[future]} "
                    f"status={result['status']} L={result.get('adaptive_L', 0)} "
                    f"selected={result.get('selected_node_count', 0)} "
                    f"threshold={result.get('threshold_hat', math.nan):.8g} "
                    f"elapsed={result.get('total_time_sec', math.nan):.2f}s",
                    flush=True,
                )

    combined = pd.DataFrame(results)
    if not combined.empty and "network_id" in combined:
        combined = combined.sort_values("network_id").reset_index(drop=True)
    combined_path = args.output_dir / f"network_summary_shard{args.shard_index}.csv"
    combined.to_csv(combined_path, index=False)
    failed = 0 if combined.empty else int(combined["status"].ne("ok").sum())
    print(f"output={combined_path} rows={len(combined)} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
