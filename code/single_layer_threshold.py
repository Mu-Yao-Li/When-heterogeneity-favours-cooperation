from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres


def as_clean_csr(matrix: sparse.spmatrix | np.ndarray) -> sparse.csr_matrix:
    adjacency = matrix.tocsr().astype(np.float64) if sparse.issparse(matrix) else sparse.csr_matrix(matrix, dtype=np.float64)
    adjacency = adjacency.maximum(adjacency.T).tolil()
    adjacency.setdiag(0.0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degree <= 0):
        raise ValueError("The single-layer graph must not contain isolated nodes.")
    return adjacency


def row_normalize(matrix: sparse.csr_matrix, row_sum: np.ndarray) -> sparse.csr_matrix:
    inv = np.zeros_like(row_sum, dtype=np.float64)
    mask = row_sum > 0
    inv[mask] = 1.0 / row_sum[mask]
    return sparse.diags(inv) @ matrix


def upper_triangle_indices(n: int) -> Tuple[np.ndarray, np.ndarray]:
    tri_i, tri_j = np.triu_indices(n, k=1)
    return tri_i.astype(np.int32), tri_j.astype(np.int32)


def vector_to_symmetric_matrix(values: np.ndarray, n: int, tri_i: np.ndarray, tri_j: np.ndarray) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[tri_i, tri_j] = values
    matrix[tri_j, tri_i] = values
    return matrix


def rowwise_inner_sum(dense_matrix: np.ndarray, sparse_matrix: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(sparse_matrix.multiply(dense_matrix).sum(axis=1)).ravel()


def solve_eta_single_layer(
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    restart: int = 40,
    maxiter: int = 1200,
    solver: str = "bicgstab",
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, float]]:
    adjacency = as_clean_csr(adjacency)
    n = adjacency.shape[0]
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = row_normalize(adjacency, degree).tocsr()
    tri_i, tri_j = upper_triangle_indices(n)
    rhs = np.full(tri_i.shape[0], 0.5, dtype=np.float64)
    iteration_count = 0
    last_gmres_residual = 0.0

    def matvec(values: np.ndarray) -> np.ndarray:
        eta = vector_to_symmetric_matrix(values, n, tri_i, tri_j)
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
        raise ValueError(f"Unsupported solver: {solver}")
    elapsed = time.perf_counter() - start
    if info != 0:
        raise RuntimeError(f"{solver} did not converge, info={info}")

    eta = vector_to_symmetric_matrix(solution, n, tri_i, tri_j)
    residual = operator.matvec(solution) - rhs
    rhs_norm = np.linalg.norm(rhs)
    stats = {
        "solver": solver,
        "solve_time_sec": float(elapsed),
        "linear_iterations": float(iteration_count),
        "final_residual_norm": float(np.linalg.norm(residual)),
        "final_relative_residual": float(np.linalg.norm(residual) / rhs_norm) if rhs_norm > 0 else 0.0,
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
            flush=True,
        )
    return eta, stats


def compute_single_layer_threshold(
    adjacency: sparse.spmatrix | np.ndarray,
    eta: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    adjacency = as_clean_csr(adjacency)
    n = adjacency.shape[0]
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = row_normalize(adjacency, degree).tocsr()
    reproductive_value = degree / degree.sum()
    moved = transition @ eta
    transition_sq = (transition @ transition).tocsr()

    v2 = float(reproductive_value @ rowwise_inner_sum(moved, transition)) / n
    u0 = float(reproductive_value @ rowwise_inner_sum(eta, transition)) / n
    u2 = float(reproductive_value @ rowwise_inner_sum(moved, transition_sq)) / n
    threshold = -v2 / (u0 - u2)
    terms = {
        "v0": 0.0,
        "v2": float(v2),
        "u0": float(u0),
        "u2": float(u2),
        "threshold": float(threshold),
    }
    return float(threshold), terms


def run_single_layer_threshold(
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    restart: int = 40,
    maxiter: int = 1200,
    solver: str = "bicgstab",
    verbose: bool = True,
) -> Dict[str, float]:
    adjacency = as_clean_csr(adjacency)
    eta, solve_stats = solve_eta_single_layer(
        adjacency,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solver=solver,
        verbose=verbose,
    )
    threshold, terms = compute_single_layer_threshold(adjacency, eta)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    return {
        "n": float(adjacency.shape[0]),
        "edges": float(adjacency.nnz // 2),
        "avg_degree": float(degree.mean()),
        "min_degree": float(degree.min()),
        "max_degree": float(degree.max()),
        "degree_std": float(degree.std(ddof=0)),
        "degree_cv": float(degree.std(ddof=0) / degree.mean()),
        "threshold": threshold,
        **solve_stats,
        **terms,
    }


def target_edge_count(n: int, avg_degree: float) -> int:
    edge_count = n * avg_degree / 2.0
    rounded = int(round(edge_count))
    if not np.isclose(edge_count, rounded):
        raise ValueError("n * avg_degree must be even for exact average degree.")
    return rounded


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return (v, u) if u > v else (u, v)


def edges_to_csr(n: int, edges: Iterable[Tuple[int, int]]) -> sparse.csr_matrix:
    row: List[int] = []
    col: List[int] = []
    data: List[float] = []
    for u, v in edges:
        row.extend([u, v])
        col.extend([v, u])
        data.extend([1.0, 1.0])
    return as_clean_csr(sparse.csr_matrix((data, (row, col)), shape=(n, n), dtype=np.float64))


def build_ba_exact_mean_degree(n: int, avg_degree: float, seed: int) -> sparse.csr_matrix:
    target_edges = target_edge_count(n, avg_degree)
    m = max(1, min(n - 1, int(np.floor(avg_degree / 2.0))))
    rng = np.random.default_rng(seed)
    graph = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(2**31 - 1)))
    current_edges = graph.number_of_edges()
    if current_edges > target_edges:
        raise ValueError("BA backbone already exceeds the target edge count.")
    degree = np.array([deg for _, deg in graph.degree()], dtype=np.float64)
    while current_edges < target_edges:
        probabilities = degree / degree.sum()
        u = int(rng.choice(n, p=probabilities))
        v = int(rng.choice(n, p=probabilities))
        if u == v or graph.has_edge(u, v):
            continue
        graph.add_edge(u, v)
        degree[u] += 1.0
        degree[v] += 1.0
        current_edges += 1
    return as_clean_csr(nx.to_scipy_sparse_array(graph, dtype=np.float64, format="csr"))


def build_connected_er_exact_mean_degree(n: int, avg_degree: float, seed: int) -> sparse.csr_matrix:
    target_edges = target_edge_count(n, avg_degree)
    if target_edges < n - 1:
        raise ValueError("Target edge count is too small to guarantee connectivity.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    edges = set()
    for idx in range(n - 1):
        edges.add(edge_key(int(order[idx]), int(order[idx + 1])))
    while len(edges) < target_edges:
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n - 1))
        if v >= u:
            v += 1
        edges.add(edge_key(u, v))
    return edges_to_csr(n, edges)


def build_periodic_lattice(rows: int, cols: int) -> sparse.csr_matrix:
    n = rows * cols
    edges = set()

    def node(row: int, col: int) -> int:
        return row * cols + col

    for row in range(rows):
        for col in range(cols):
            source = node(row, col)
            edges.add(edge_key(source, node(row, (col + 1) % cols)))
            edges.add(edge_key(source, node((row + 1) % rows, col)))
    return edges_to_csr(n, edges)


def build_pa_degree_power_exact(n: int, avg_degree: float, gamma: float, seed: int) -> sparse.csr_matrix:
    target_edges = target_edge_count(n, avg_degree)
    m = max(1, int(np.floor(avg_degree / 2.0)))
    initial = max(m + 1, 3)
    rng = np.random.default_rng(seed)
    edges = set()
    neighbors = [set() for _ in range(n)]
    degree = np.zeros(n, dtype=np.int32)

    def add_edge(u: int, v: int) -> None:
        key = edge_key(u, v)
        edges.add(key)
        neighbors[u].add(v)
        neighbors[v].add(u)
        degree[u] += 1
        degree[v] += 1

    for u in range(initial):
        for v in range(u + 1, initial):
            add_edge(u, v)

    for new_node in range(initial, n):
        blocked: set[int] = set()
        for _ in range(min(m, new_node)):
            weights = np.ones(new_node, dtype=np.float64) if gamma == 0 else degree[:new_node].astype(np.float64) ** gamma
            if blocked:
                weights[list(blocked)] = 0.0
            probabilities = weights / weights.sum()
            target = int(rng.choice(new_node, p=probabilities))
            add_edge(new_node, target)
            blocked.add(target)

    while len(edges) < target_edges:
        weights = np.ones(n, dtype=np.float64) if gamma == 0 else degree.astype(np.float64) ** gamma
        u = int(rng.choice(n, p=weights / weights.sum()))
        candidate_weights = weights.copy()
        candidate_weights[u] = 0.0
        if neighbors[u]:
            candidate_weights[list(neighbors[u])] = 0.0
        total = candidate_weights.sum()
        if total <= 0:
            continue
        v = int(rng.choice(n, p=candidate_weights / total))
        add_edge(u, v)
    return edges_to_csr(n, edges)


def load_edge_list(path: Path, delimiter: str | None = None) -> sparse.csr_matrix:
    edges: List[Tuple[int, int]] = []
    max_node = -1
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(delimiter)
            if len(parts) < 2:
                raise ValueError(f"Invalid edge-list line: {line!r}")
            u, v = int(parts[0]), int(parts[1])
            edges.append(edge_key(u, v))
            max_node = max(max_node, u, v)
    if max_node < 0:
        raise ValueError("Edge-list file is empty.")
    return edges_to_csr(max_node + 1, edges)


def write_stats_csv(path: Path, stats: Dict[str, float], metadata: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**metadata, **{key: f"{value:.15g}" if isinstance(value, float) else str(value) for key, value in stats.items()}}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def build_network_from_args(args: argparse.Namespace) -> Tuple[sparse.csr_matrix, Dict[str, str]]:
    if args.network == "ba":
        return build_ba_exact_mean_degree(args.n, args.avg_degree, args.seed), {"network": "ba"}
    if args.network == "er":
        return build_connected_er_exact_mean_degree(args.n, args.avg_degree, args.seed), {"network": "er"}
    if args.network == "lattice":
        if args.rows * args.cols != args.n:
            raise ValueError("For lattice, rows * cols must equal n.")
        return build_periodic_lattice(args.rows, args.cols), {"network": "lattice", "rows": str(args.rows), "cols": str(args.cols)}
    if args.network == "pa-gamma":
        return build_pa_degree_power_exact(args.n, args.avg_degree, args.gamma, args.seed), {
            "network": "pa-gamma",
            "gamma": f"{args.gamma:.6g}",
            "kernel": "k^gamma",
        }
    if args.network == "edge-list":
        if args.edge_list is None:
            raise ValueError("--edge-list is required for network=edge-list.")
        return load_edge_list(Path(args.edge_list)), {"network": "edge-list", "edge_list": str(args.edge_list)}
    raise ValueError(f"Unsupported network type: {args.network}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute the single-layer critical threshold (b/c)*.")
    parser.add_argument("--network", choices=["ba", "er", "lattice", "pa-gamma", "edge-list"], required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--avg-degree", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--rows", type=int, default=100)
    parser.add_argument("--cols", type=int, default=100)
    parser.add_argument("--edge-list", type=str, default=None)
    parser.add_argument("--solver", choices=["bicgstab", "gmres"], default="bicgstab")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--restart", type=int, default=40)
    parser.add_argument("--maxiter", type=int, default=1200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    adjacency, metadata = build_network_from_args(args)
    stats = run_single_layer_threshold(
        adjacency,
        rtol=args.rtol,
        atol=args.atol,
        restart=args.restart,
        maxiter=args.maxiter,
        solver=args.solver,
        verbose=True,
    )
    print(f"threshold={stats['threshold']:.15g}")
    print(f"n={int(stats['n'])} edges={int(stats['edges'])} avg_degree={stats['avg_degree']:.15g}")
    print(f"degree_cv={stats['degree_cv']:.15g}")
    print(f"solve_time_sec={stats['solve_time_sec']:.3f}")
    print(f"final_relative_residual={stats['final_relative_residual']:.3e}")

    if args.output:
        write_stats_csv(Path(args.output), stats, metadata)
        print(f"saved={args.output}")


if __name__ == "__main__":
    main()
