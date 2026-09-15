from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres, spsolve


@dataclass
class InterdependentSystem:
    adjacency: sparse.csr_matrix
    layer_labels: np.ndarray
    alpha: float
    q: np.ndarray
    same: sparse.csr_matrix
    cross: sparse.csr_matrix
    weighted: sparse.csr_matrix
    transition: sparse.csr_matrix
    same_prob: sparse.csr_matrix
    cross_prob: sparse.csr_matrix
    reproductive_weight: np.ndarray

    @property
    def n(self) -> int:
        return self.adjacency.shape[0]

    @property
    def pair_count(self) -> int:
        n = self.n
        return n * (n - 1) // 2


def _as_csr(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float64)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))


def _row_normalize(matrix: sparse.csr_matrix, row_sum: np.ndarray) -> sparse.csr_matrix:
    inv = np.zeros_like(row_sum, dtype=np.float64)
    mask = row_sum > 0
    inv[mask] = 1.0 / row_sum[mask]
    return sparse.diags(inv) @ matrix


def build_system(
    adjacency: sparse.spmatrix | np.ndarray,
    layer_labels: np.ndarray,
    alpha: float,
    q: np.ndarray,
) -> InterdependentSystem:
    adjacency = _as_csr(adjacency)
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency = adjacency.tolil()
    adjacency.setdiag(0.0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()

    layer_labels = np.asarray(layer_labels, dtype=np.int8)
    q = np.asarray(q, dtype=np.float64)
    n = adjacency.shape[0]
    if layer_labels.shape != (n,):
        raise ValueError("layer_labels must have shape (n,)")
    if q.shape != (n,):
        raise ValueError("q must have shape (n,)")

    coo = adjacency.tocoo()
    same_mask = layer_labels[coo.row] == layer_labels[coo.col]
    same = sparse.csr_matrix(
        (coo.data[same_mask], (coo.row[same_mask], coo.col[same_mask])),
        shape=adjacency.shape,
    )
    cross = sparse.csr_matrix(
        (coo.data[~same_mask], (coo.row[~same_mask], coo.col[~same_mask])),
        shape=adjacency.shape,
    )

    weighted = (1.0 - alpha) * same + alpha * cross

    weighted_sum = np.asarray(weighted.sum(axis=1)).ravel()
    same_sum = np.asarray(same.sum(axis=1)).ravel()
    cross_sum = np.asarray(cross.sum(axis=1)).ravel()

    transition = _row_normalize(weighted, weighted_sum).tocsr()
    same_prob = _row_normalize(same, same_sum).tocsr()
    cross_prob = _row_normalize(cross, cross_sum).tocsr()

    reproductive_weight = weighted_sum / np.sum(weighted_sum / q)

    return InterdependentSystem(
        adjacency=adjacency,
        layer_labels=layer_labels,
        alpha=alpha,
        q=q,
        same=same,
        cross=cross,
        weighted=weighted.tocsr(),
        transition=transition,
        same_prob=same_prob,
        cross_prob=cross_prob,
        reproductive_weight=reproductive_weight,
    )


def build_symmetric_interdependent_ba(
    layer_size: int,
    m: int,
    alpha: float,
    beta: float,
    seed: Optional[int] = None,
    identical_layers: bool = True,
    coupled: bool = True,
) -> InterdependentSystem:
    if layer_size < 2:
        raise ValueError("layer_size must be at least 2")
    if not 1 <= m < layer_size:
        raise ValueError("Barabasi-Albert parameter m must satisfy 1 <= m < layer_size")

    rng = np.random.default_rng(seed)
    graph_1 = nx.barabasi_albert_graph(layer_size, m, seed=int(rng.integers(2**31 - 1)))
    if identical_layers:
        graph_2 = graph_1.copy()
    else:
        graph_2 = nx.barabasi_albert_graph(layer_size, m, seed=int(rng.integers(2**31 - 1)))

    layer_1 = nx.to_scipy_sparse_array(graph_1, dtype=np.float64, format="csr")
    layer_2 = nx.to_scipy_sparse_array(graph_2, dtype=np.float64, format="csr")
    if coupled:
        coupling = sparse.identity(layer_size, dtype=np.float64, format="csr")
    else:
        coupling = sparse.csr_matrix((layer_size, layer_size), dtype=np.float64)
    adjacency = sparse.bmat(
        [[layer_1, coupling], [coupling, layer_2]],
        format="csr",
        dtype=np.float64,
    )

    layer_labels = np.concatenate(
        [np.ones(layer_size, dtype=np.int8), -np.ones(layer_size, dtype=np.int8)]
    )
    q1 = np.full(layer_size, beta / layer_size, dtype=np.float64)
    q2 = np.full(layer_size, (1.0 - beta) / layer_size, dtype=np.float64)
    q = np.concatenate([q1, q2])

    return build_system(adjacency=adjacency, layer_labels=layer_labels, alpha=alpha, q=q)


def build_two_layer_system(
    layer_1: sparse.spmatrix | np.ndarray,
    layer_2: sparse.spmatrix | np.ndarray,
    coupling: sparse.spmatrix | np.ndarray,
    alpha: float,
    beta: float,
) -> InterdependentSystem:
    layer_1 = _as_csr(layer_1)
    layer_2 = _as_csr(layer_2)
    coupling = _as_csr(coupling)
    n1 = layer_1.shape[0]
    n2 = layer_2.shape[0]
    if layer_1.shape != (n1, n1):
        raise ValueError("layer_1 must be square")
    if layer_2.shape != (n2, n2):
        raise ValueError("layer_2 must be square")
    if coupling.shape != (n1, n2):
        raise ValueError("coupling must have shape (n1, n2)")

    adjacency = sparse.bmat(
        [[layer_1, coupling], [coupling.T, layer_2]],
        format="csr",
        dtype=np.float64,
    )
    layer_labels = np.concatenate(
        [np.ones(n1, dtype=np.int8), -np.ones(n2, dtype=np.int8)]
    )
    q1 = np.full(n1, beta / n1, dtype=np.float64)
    q2 = np.full(n2, (1.0 - beta) / n2, dtype=np.float64)
    q = np.concatenate([q1, q2])
    return build_system(adjacency=adjacency, layer_labels=layer_labels, alpha=alpha, q=q)


def _upper_triangle_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    tri_i, tri_j = np.triu_indices(n, k=1)
    return tri_i.astype(np.int32), tri_j.astype(np.int32)


def _vector_to_symmetric_matrix(
    values: np.ndarray,
    n: int,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[tri_i, tri_j] = values
    matrix[tri_j, tri_i] = values
    return matrix


def solve_eta_linear_operator(
    system: InterdependentSystem,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    restart: int = 50,
    maxiter: int = 300,
    solver: str = "bicgstab",
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n = system.n
    q = system.q
    transition = system.transition
    tri_i, tri_j = _upper_triangle_indices(n)
    pair_q_sum = q[tri_i] + q[tri_j]
    rhs = (1.0 / n) / pair_q_sum
    left_weight = q[tri_i] / pair_q_sum
    right_weight = q[tri_j] / pair_q_sum
    iteration_count = 0
    last_gmres_residual = 0.0

    def matvec(values: np.ndarray) -> np.ndarray:
        eta = _vector_to_symmetric_matrix(values, n, tri_i, tri_j)
        moved = transition @ eta
        return values - left_weight * moved[tri_i, tri_j] - right_weight * moved[tri_j, tri_i]

    def gmres_callback(value: float) -> None:
        nonlocal iteration_count, last_gmres_residual
        iteration_count += 1
        last_gmres_residual = float(value)

    def krylov_callback(_: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    operator = LinearOperator(
        shape=(system.pair_count, system.pair_count),
        matvec=matvec,
        dtype=np.float64,
    )

    start = time.perf_counter()
    if solver == "gmres":
        try:
            solution, info = gmres(
                operator,
                rhs,
                rtol=rtol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                callback=gmres_callback,
                callback_type="pr_norm",
            )
        except TypeError:
            solution, info = gmres(
                operator,
                rhs,
                tol=rtol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                callback=gmres_callback,
                callback_type="pr_norm",
            )
    elif solver == "bicgstab":
        try:
            solution, info = bicgstab(
                operator,
                rhs,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                callback=krylov_callback,
            )
        except TypeError:
            solution, info = bicgstab(
                operator,
                rhs,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
                callback=krylov_callback,
            )
    else:
        raise ValueError(f"Unsupported solver '{solver}'")
    elapsed = time.perf_counter() - start

    if info != 0:
        raise RuntimeError(f"{solver} did not converge, info={info}")

    eta = _vector_to_symmetric_matrix(solution, n, tri_i, tri_j)
    residual_vector = operator.matvec(solution) - rhs
    rhs_norm = np.linalg.norm(rhs)
    final_residual = np.linalg.norm(residual_vector)
    stats = {
        "solver": solver,
        "solve_time_sec": elapsed,
        "linear_iterations": float(iteration_count),
        "final_residual_norm": float(final_residual),
        "final_relative_residual": float(final_residual / rhs_norm) if rhs_norm > 0 else 0.0,
    }
    if solver == "gmres":
        stats["final_preconditioned_residual"] = float(last_gmres_residual)
    if verbose:
        print(
            "eta solved",
            f"solver={solver}",
            f"time={elapsed:.3f}s",
            f"iters={int(stats['linear_iterations'])}",
            f"residual={stats['final_relative_residual']:.3e}",
        )
    return eta, stats


def _rowwise_inner_sum(dense_matrix: np.ndarray, sparse_matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(sparse_matrix.multiply(dense_matrix).sum(axis=1)).ravel()


def compute_threshold_matrix_form(
    system: InterdependentSystem,
    eta: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    c = system.reproductive_weight
    alpha = system.alpha
    p = system.transition
    s = system.same_prob
    cprob = system.cross_prob

    pc = (p @ cprob).tocsr()
    ps = (p @ s).tocsr()
    pcs = (pc @ s).tocsr()
    cs = (cprob @ s).tocsr()

    moved = p @ eta

    v0 = alpha * float(c @ _rowwise_inner_sum(eta, cprob))
    v2 = (1.0 - alpha) * float(c @ _rowwise_inner_sum(moved, p))
    v2 += alpha * float(c @ _rowwise_inner_sum(moved, pc))

    u0 = (1.0 - alpha) * float(c @ _rowwise_inner_sum(eta, s))
    u0 += alpha * float(c @ _rowwise_inner_sum(eta, cs))

    u2 = (1.0 - alpha) * float(c @ _rowwise_inner_sum(moved, ps))
    u2 += alpha * float(c @ _rowwise_inner_sum(moved, pcs))

    threshold = (v0 - v2) / (u0 - u2)
    terms = {
        "v0": v0,
        "v2": v2,
        "u0": u0,
        "u2": u2,
        "threshold": threshold,
    }
    return threshold, terms


def build_single_layer_ba(
    layer_size: int,
    m: int,
    seed: Optional[int] = None,
) -> sparse.csr_matrix:
    if layer_size < 2:
        raise ValueError("layer_size must be at least 2")
    if not 1 <= m < layer_size:
        raise ValueError("Barabasi-Albert parameter m must satisfy 1 <= m < layer_size")
    graph = nx.barabasi_albert_graph(layer_size, m, seed=seed)
    adjacency = nx.to_scipy_sparse_array(graph, dtype=np.float64, format="csr")
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency = adjacency.tolil()
    adjacency.setdiag(0.0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    return adjacency


def solve_eta_single_layer_linear_operator(
    adjacency: sparse.spmatrix | np.ndarray,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    restart: int = 50,
    maxiter: int = 300,
    solver: str = "bicgstab",
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    adjacency = _as_csr(adjacency)
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency = adjacency.tolil()
    adjacency.setdiag(0.0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()

    n = adjacency.shape[0]
    weight_vector = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = _row_normalize(adjacency, weight_vector).tocsr()
    tri_i, tri_j = _upper_triangle_indices(n)
    rhs = np.full(tri_i.shape[0], 0.5, dtype=np.float64)
    iteration_count = 0
    last_gmres_residual = 0.0

    def matvec(values: np.ndarray) -> np.ndarray:
        eta = _vector_to_symmetric_matrix(values, n, tri_i, tri_j)
        moved = transition @ eta
        return values - 0.5 * moved[tri_i, tri_j] - 0.5 * moved[tri_j, tri_i]

    def gmres_callback(value: float) -> None:
        nonlocal iteration_count, last_gmres_residual
        iteration_count += 1
        last_gmres_residual = float(value)

    def krylov_callback(_: np.ndarray) -> None:
        nonlocal iteration_count
        iteration_count += 1

    operator = LinearOperator(shape=(rhs.size, rhs.size), matvec=matvec, dtype=np.float64)

    start = time.perf_counter()
    if solver == "gmres":
        try:
            solution, info = gmres(
                operator,
                rhs,
                rtol=rtol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                callback=gmres_callback,
                callback_type="pr_norm",
            )
        except TypeError:
            solution, info = gmres(
                operator,
                rhs,
                tol=rtol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
                callback=gmres_callback,
                callback_type="pr_norm",
            )
    elif solver == "bicgstab":
        try:
            solution, info = bicgstab(
                operator,
                rhs,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                callback=krylov_callback,
            )
        except TypeError:
            solution, info = bicgstab(
                operator,
                rhs,
                tol=rtol,
                atol=atol,
                maxiter=maxiter,
                callback=krylov_callback,
            )
    else:
        raise ValueError(f"Unsupported solver '{solver}'")
    elapsed = time.perf_counter() - start

    if info != 0:
        raise RuntimeError(f"{solver} did not converge, info={info}")

    eta = _vector_to_symmetric_matrix(solution, n, tri_i, tri_j)
    residual_vector = operator.matvec(solution) - rhs
    rhs_norm = np.linalg.norm(rhs)
    final_residual = np.linalg.norm(residual_vector)
    stats = {
        "solver": solver,
        "solve_time_sec": elapsed,
        "linear_iterations": float(iteration_count),
        "final_residual_norm": float(final_residual),
        "final_relative_residual": float(final_residual / rhs_norm) if rhs_norm > 0 else 0.0,
    }
    if solver == "gmres":
        stats["final_preconditioned_residual"] = float(last_gmres_residual)
    if verbose:
        print(
            "single eta solved",
            f"solver={solver}",
            f"time={elapsed:.3f}s",
            f"iters={int(stats['linear_iterations'])}",
            f"residual={stats['final_relative_residual']:.3e}",
        )
    return eta, stats


def compute_threshold_single_layer_matrix_form(
    adjacency: sparse.spmatrix | np.ndarray,
    eta: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    adjacency = _as_csr(adjacency)
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency = adjacency.tolil()
    adjacency.setdiag(0.0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()

    n = adjacency.shape[0]
    weight_vector = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = _row_normalize(adjacency, weight_vector).tocsr()
    reproductive_value = weight_vector / np.sum(weight_vector)
    moved = transition @ eta
    transition_sq = (transition @ transition).tocsr()

    v2 = float(reproductive_value @ _rowwise_inner_sum(moved, transition)) / n
    u0 = float(reproductive_value @ _rowwise_inner_sum(eta, transition)) / n
    u2 = float(reproductive_value @ _rowwise_inner_sum(moved, transition_sq)) / n
    threshold = -v2 / (u0 - u2)
    terms = {
        "v0": 0.0,
        "v2": v2,
        "u0": u0,
        "u2": u2,
        "threshold": threshold,
    }
    return threshold, terms


def run_single_layer_case(
    adjacency: sparse.spmatrix | np.ndarray,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    solver: str,
) -> Dict[str, float]:
    adjacency = _as_csr(adjacency)
    eta, solve_stats = solve_eta_single_layer_linear_operator(
        adjacency,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solver=solver,
        verbose=True,
    )
    threshold, terms = compute_threshold_single_layer_matrix_form(adjacency, eta)
    return {
        "layer_size": float(adjacency.shape[0]),
        "alpha": 0.0,
        "threshold": float(threshold),
        **solve_stats,
        **terms,
    }


def run_single_ba_case(
    layer_size: int,
    m: int,
    seed: Optional[int],
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    solver: str,
) -> Dict[str, float]:
    adjacency = build_single_layer_ba(layer_size=layer_size, m=m, seed=seed)
    return run_single_layer_case(
        adjacency=adjacency,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solver=solver,
    )


def run_general_two_layer_case(
    layer_1: sparse.spmatrix | np.ndarray,
    layer_2: sparse.spmatrix | np.ndarray,
    coupling: sparse.spmatrix | np.ndarray,
    alpha: float,
    beta: float,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    solver: str,
) -> Dict[str, float]:
    system = build_two_layer_system(
        layer_1=layer_1,
        layer_2=layer_2,
        coupling=coupling,
        alpha=alpha,
        beta=beta,
    )
    eta, solve_stats = solve_eta_linear_operator(
        system,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solver=solver,
        verbose=True,
    )
    threshold, terms = compute_threshold_matrix_form(system, eta)
    return {
        "layer_1_size": float(layer_1.shape[0]),
        "layer_2_size": float(layer_2.shape[0]),
        "total_nodes": float(system.n),
        "alpha": alpha,
        "beta": beta,
        "threshold": float(threshold),
        **solve_stats,
        **terms,
    }


def compute_threshold_dense_reference(
    system: InterdependentSystem,
    eta: np.ndarray,
) -> float:
    n = system.n
    alpha = system.alpha
    weighted = system.weighted.toarray()
    weighted_sum = np.asarray(system.weighted.sum(axis=1)).ravel()
    same_indicator = (
        system.layer_labels[:, None] == system.layer_labels[None, :]
    ).astype(np.float64)
    cross_weighted_sum = np.asarray((system.alpha * system.cross).sum(axis=1)).ravel()
    same_weighted_sum = np.asarray(((1.0 - system.alpha) * system.same).sum(axis=1)).ravel()
    pi = weighted_sum / system.q
    pi = pi / np.sum(pi)
    q = system.q

    v0 = 0.0
    v2 = 0.0
    u0 = 0.0
    u2 = 0.0

    for i in range(n):
        for a in range(n):
            if cross_weighted_sum[i] > 0:
                v0 += (
                    pi[i]
                    * q[i]
                    * eta[i, a]
                    * alpha
                    * (1.0 - same_indicator[i, a])
                    * weighted[i, a]
                    / cross_weighted_sum[i]
                )
            if same_weighted_sum[i] > 0:
                u0 += (
                    pi[i]
                    * q[i]
                    * eta[i, a]
                    * (1.0 - alpha)
                    * same_indicator[i, a]
                    * weighted[i, a]
                    / same_weighted_sum[i]
                )
            for k in range(n):
                if cross_weighted_sum[i] > 0 and same_weighted_sum[a] > 0:
                    u0 += (
                        pi[i]
                        * q[i]
                        * eta[i, k]
                        * alpha
                        * same_indicator[k, a]
                        * (1.0 - same_indicator[i, a])
                        * weighted[k, a]
                        * weighted[i, a]
                        / same_weighted_sum[a]
                        / cross_weighted_sum[i]
                    )

    for i in range(n):
        for j in range(n):
            for k in range(n):
                if weighted_sum[i] > 0:
                    v2 += (
                        pi[i]
                        * q[i]
                        * eta[j, k]
                        * (1.0 - alpha)
                        * weighted[i, j]
                        * weighted[i, k]
                        / weighted_sum[i]
                        / weighted_sum[i]
                    )
                for t in range(n):
                    if weighted_sum[i] > 0 and cross_weighted_sum[t] > 0:
                        v2 += (
                            pi[i]
                            * q[i]
                            * eta[j, k]
                            * alpha
                            * (1.0 - same_indicator[k, t])
                            * weighted[i, j]
                            * weighted[i, t]
                            * weighted[k, t]
                            / weighted_sum[i]
                            / weighted_sum[i]
                            / cross_weighted_sum[t]
                        )
                    if weighted_sum[i] > 0 and same_weighted_sum[k] > 0:
                        u2 += (
                            pi[i]
                            * q[i]
                            * eta[j, t]
                            * (1.0 - alpha)
                            * same_indicator[k, t]
                            * weighted[i, j]
                            * weighted[i, k]
                            * weighted[k, t]
                            / weighted_sum[i]
                            / weighted_sum[i]
                            / same_weighted_sum[k]
                        )
                    for a in range(n):
                        if (
                            weighted_sum[i] > 0
                            and same_weighted_sum[a] > 0
                            and cross_weighted_sum[k] > 0
                        ):
                            u2 += (
                                pi[i]
                                * q[i]
                                * eta[j, t]
                                * alpha
                                * same_indicator[t, a]
                                * (1.0 - same_indicator[k, a])
                                * weighted[i, j]
                                * weighted[i, k]
                                * weighted[a, k]
                                * weighted[a, t]
                                / weighted_sum[i]
                                / weighted_sum[i]
                                / same_weighted_sum[a]
                                / cross_weighted_sum[k]
                            )
    return (v0 - v2) / (u0 - u2)


def build_pair_matrix_reference(system: InterdependentSystem) -> sparse.csr_matrix:
    n = system.n
    transition = system.transition
    q = system.q
    pair_count = system.pair_count
    rows = []
    cols = []
    data = []
    tri_i, tri_j = _upper_triangle_indices(n)
    offsets = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        offsets[i] = offsets[i - 1] + (n - i)

    def pair_index(i: int, j: int) -> int:
        if i > j:
            i, j = j, i
        return offsets[i] + (j - i - 1)

    for row, (i, j) in enumerate(zip(tri_i, tri_j)):
        rows.append(row)
        cols.append(row)
        data.append(1.0)
        denom = q[i] + q[j]

        row_j = transition.getrow(j)
        for k, value in zip(row_j.indices, row_j.data):
            if i == k:
                continue
            rows.append(row)
            cols.append(pair_index(i, int(k)))
            data.append(-q[j] * value / denom)

        row_i = transition.getrow(i)
        for k, value in zip(row_i.indices, row_i.data):
            if j == k:
                continue
            rows.append(row)
            cols.append(pair_index(j, int(k)))
            data.append(-q[i] * value / denom)

    return sparse.csr_matrix((data, (rows, cols)), shape=(pair_count, pair_count))


def solve_eta_reference(system: InterdependentSystem) -> np.ndarray:
    n = system.n
    tri_i, tri_j = _upper_triangle_indices(n)
    pair_q_sum = system.q[tri_i] + system.q[tri_j]
    rhs = (1.0 / n) / pair_q_sum
    matrix = build_pair_matrix_reference(system)
    solution = spsolve(matrix, rhs)
    return _vector_to_symmetric_matrix(solution, n, tri_i, tri_j)


def run_small_validation() -> Dict[str, float]:
    system = build_symmetric_interdependent_ba(
        layer_size=6,
        m=2,
        alpha=0.1,
        beta=0.5,
        seed=7,
        identical_layers=True,
    )
    eta_reference = solve_eta_reference(system)
    threshold_reference = compute_threshold_dense_reference(system, eta_reference)

    eta_fast, stats = solve_eta_linear_operator(system, rtol=1e-10, atol=1e-12)
    threshold_fast, _ = compute_threshold_matrix_form(system, eta_fast)

    return {
        "eta_max_abs_diff": float(np.max(np.abs(eta_reference - eta_fast))),
        "threshold_reference": float(threshold_reference),
        "threshold_fast": float(threshold_fast),
        "threshold_abs_diff": float(abs(threshold_reference - threshold_fast)),
        **stats,
    }


def run_ba_case(
    layer_size: int,
    m: int,
    alpha: float,
    beta: float,
    seed: Optional[int],
    identical_layers: bool,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    solver: str,
    coupled: bool = True,
) -> Dict[str, float]:
    system = build_symmetric_interdependent_ba(
        layer_size=layer_size,
        m=m,
        alpha=alpha,
        beta=beta,
        seed=seed,
        identical_layers=identical_layers,
        coupled=coupled,
    )
    eta, solve_stats = solve_eta_linear_operator(
        system,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solver=solver,
        verbose=True,
    )
    threshold, terms = compute_threshold_matrix_form(system, eta)
    return {
        "layer_size": float(layer_size),
        "total_nodes": float(system.n),
        "alpha": alpha,
        "beta": beta,
        "threshold": float(threshold),
        **solve_stats,
        **terms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse Python rewrite of the interdependent-network threshold calculation."
    )
    parser.add_argument("--validate-small", action="store_true", help="Run a small internal self-check.")
    parser.add_argument("--layer-size", type=int, default=50, help="Nodes in each BA layer.")
    parser.add_argument("--m", type=int, default=3, help="BA attachment parameter.")
    parser.add_argument("--alpha", type=float, default=0.1, help="Coupling intensity.")
    parser.add_argument("--beta", type=float, default=0.5, help="System-1 update probability.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--independent-layers", action="store_true", help="Use two independently drawn BA layers.")
    parser.add_argument("--rtol", type=float, default=1e-7, help="GMRES relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-9, help="GMRES absolute tolerance.")
    parser.add_argument("--restart", type=int, default=40, help="GMRES restart parameter.")
    parser.add_argument("--maxiter", type=int, default=200, help="Maximum iterations for the linear solver.")
    parser.add_argument(
        "--solver",
        choices=["bicgstab", "gmres"],
        default="bicgstab",
        help="Iterative solver used for eta.",
    )
    args = parser.parse_args()

    if args.validate_small:
        stats = run_small_validation()
        print("small_validation")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    stats = run_ba_case(
        layer_size=args.layer_size,
        m=args.m,
        alpha=args.alpha,
        beta=args.beta,
        seed=args.seed,
        identical_layers=not args.independent_layers,
        rtol=args.rtol,
        atol=args.atol,
        restart=args.restart,
        maxiter=args.maxiter,
        solver=args.solver,
    )
    print("ba_case")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
