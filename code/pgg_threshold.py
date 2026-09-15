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

from single_layer_threshold import (
    as_clean_csr,
    build_ba_exact_mean_degree,
    build_connected_er_exact_mean_degree,
    build_periodic_lattice,
    edge_key,
    edges_to_csr,
    load_edge_list,
    row_normalize,
    solve_eta_single_layer,
    upper_triangle_indices,
    vector_to_symmetric_matrix,
)


def parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def graph_to_csr(graph: nx.Graph) -> sparse.csr_matrix:
    graph = nx.convert_node_labels_to_integers(graph)
    return edges_to_csr(graph.number_of_nodes(), graph.edges())


def build_network(args: argparse.Namespace, n: int, seed: int) -> Tuple[sparse.csr_matrix, Dict[str, str]]:
    if args.network == "star":
        return graph_to_csr(nx.star_graph(n - 1)), {"network": "star"}
    if args.network == "cycle":
        return graph_to_csr(nx.cycle_graph(n)), {"network": "cycle"}
    if args.network == "path":
        return graph_to_csr(nx.path_graph(n)), {"network": "path"}
    if args.network == "complete":
        return graph_to_csr(nx.complete_graph(n)), {"network": "complete"}
    if args.network == "rr":
        graph = nx.random_regular_graph(int(args.avg_degree), n, seed=seed)
        return graph_to_csr(graph), {"network": "rr"}
    if args.network == "ba":
        return build_ba_exact_mean_degree(n, args.avg_degree, seed), {"network": "ba"}
    if args.network == "er":
        return build_connected_er_exact_mean_degree(n, args.avg_degree, seed), {"network": "er"}
    if args.network == "lattice":
        if args.rows * args.cols != n:
            raise ValueError("For lattice, rows * cols must equal n.")
        return build_periodic_lattice(args.rows, args.cols), {
            "network": "lattice",
            "rows": str(args.rows),
            "cols": str(args.cols),
        }
    if args.network == "edge-list":
        if args.edge_list is None:
            raise ValueError("--edge-list is required for network=edge-list.")
        return load_edge_list(Path(args.edge_list)), {"network": "edge-list", "edge_list": str(args.edge_list)}
    raise ValueError(f"Unsupported network: {args.network}")


def directed_edges(adjacency: sparse.csr_matrix) -> int:
    return int(adjacency.nnz)


def solve_tau_pc_db(
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    solver: str,
    verbose: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    eta, stats = solve_eta_single_layer(
        adjacency,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        solver=solver,
        verbose=verbose,
    )
    tau = 2.0 * eta
    np.fill_diagonal(tau, 0.0)
    return tau, {f"tau_{key}": value for key, value in stats.items()}


def solve_tau_bd(
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    solver: str,
    verbose: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    adjacency = as_clean_csr(adjacency)
    n = adjacency.shape[0]
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = row_normalize(adjacency, degree).tocsr()
    col_sum = np.asarray(transition.sum(axis=0)).ravel()
    tri_i, tri_j = upper_triangle_indices(n)
    denom = col_sum[tri_i] + col_sum[tri_j]
    if np.any(denom <= 0):
        raise ValueError("BD coalescence equation has a zero denominator.")
    rhs = 1.0 / denom
    iteration_count = 0
    last_gmres_residual = 0.0

    def matvec(values: np.ndarray) -> np.ndarray:
        tau = vector_to_symmetric_matrix(values, n, tri_i, tri_j)
        moved = tau @ transition
        return values - (moved[tri_j, tri_i] + moved[tri_i, tri_j]) / denom

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
        raise RuntimeError(f"{solver} did not converge for BD tau, info={info}")

    tau = vector_to_symmetric_matrix(solution, n, tri_i, tri_j)
    residual = operator.matvec(solution) - rhs
    rhs_norm = np.linalg.norm(rhs)
    stats = {
        "tau_solver": solver,
        "tau_solve_time_sec": float(elapsed),
        "tau_linear_iterations": float(iteration_count),
        "tau_final_residual_norm": float(np.linalg.norm(residual)),
        "tau_final_relative_residual": float(np.linalg.norm(residual) / rhs_norm) if rhs_norm > 0 else 0.0,
    }
    if solver == "gmres":
        stats["tau_final_preconditioned_residual"] = float(last_gmres_residual)
    if verbose:
        print(
            "BD tau solved",
            f"solver={solver}",
            f"time={elapsed:.3f}s",
            f"iters={iteration_count}",
            f"residual={stats['tau_final_relative_residual']:.3e}",
            flush=True,
        )
    return tau, stats


def compute_pgg_upsilon(adjacency: sparse.spmatrix | np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Compute Upsilon_ij from Wang and Su, Eq. 5 / Supplementary Eq. S32.

    The implementation uses the unweighted neighbor-sum expression in matrix
    form. It is algebraically identical to the transition-probability form in
    Supplementary Eq. S32 because k_i p_il = A_il.
    """
    adjacency = as_clean_csr(adjacency)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    group_size = degree + 1.0
    neighbor_tau_sum = adjacency @ tau
    direct = tau + neighbor_tau_sum - np.diag(neighbor_tau_sum)[:, None]
    organizer_terms = (tau + neighbor_tau_sum) / group_size[:, None]
    neighbor_organizer_sum = adjacency @ organizer_terms
    second = neighbor_organizer_sum - np.diag(neighbor_organizer_sum)[:, None]
    upsilon = direct / (group_size[:, None] ** 2) + second / group_size[:, None]
    np.fill_diagonal(upsilon, 0.0)
    return upsilon


def weighted_sum(weight: sparse.csr_matrix, dense: np.ndarray) -> float:
    return float(weight.multiply(dense).sum())


def compute_pc_or_db_threshold(
    adjacency: sparse.spmatrix | np.ndarray,
    tau: np.ndarray,
    upsilon: np.ndarray,
    *,
    update_rule: str,
    payoff: str,
) -> Dict[str, float]:
    adjacency = as_clean_csr(adjacency)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = row_normalize(adjacency, degree).tocsr()
    if update_rule == "pc":
        walk = transition
        walk_steps = 1
    elif update_rule == "db":
        walk = (transition @ transition).tocsr()
        walk_steps = 2
    else:
        raise ValueError(f"Expected pc or db, got {update_rule}.")

    row_weight = degree.copy()
    if payoff == "accumulated":
        row_weight *= degree + 1.0
    elif payoff != "average":
        raise ValueError(f"Unsupported payoff: {payoff}")
    weight = (sparse.diags(row_weight) @ walk).tocsr()
    numerator = weighted_sum(weight, tau)
    denominator = weighted_sum(weight, upsilon)
    threshold = float(numerator / denominator) if denominator > 0 else float("inf")
    return {
        "threshold": threshold,
        "numerator_tau": float(numerator),
        "denominator_upsilon": float(denominator),
        "walk_steps": float(walk_steps),
    }


def compute_bd_threshold(
    adjacency: sparse.spmatrix | np.ndarray,
    tau_tilde: np.ndarray,
    upsilon_tilde: np.ndarray,
    *,
    payoff: str,
) -> Dict[str, float]:
    adjacency = as_clean_csr(adjacency)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    row, col = adjacency.nonzero()
    data = 1.0 / (degree[row] * degree[col])
    if payoff == "accumulated":
        data = data * (degree[row] + 1.0)
    elif payoff != "average":
        raise ValueError(f"Unsupported payoff: {payoff}")
    weight = sparse.csr_matrix((data, (row, col)), shape=adjacency.shape)
    numerator = weighted_sum(weight, tau_tilde)
    denominator = weighted_sum(weight, upsilon_tilde)
    threshold = float(numerator / denominator) if denominator > 0 else float("inf")
    return {
        "threshold": threshold,
        "numerator_tau": float(numerator),
        "denominator_upsilon": float(denominator),
        "walk_steps": 1.0,
    }


def compute_pgg_thresholds(
    adjacency: sparse.spmatrix | np.ndarray,
    *,
    update_rules: Iterable[str],
    payoffs: Iterable[str],
    rtol: float,
    atol: float,
    restart: int,
    maxiter: int,
    solver: str,
    verbose: bool,
) -> List[Dict[str, float | str]]:
    adjacency = as_clean_csr(adjacency)
    update_rules = list(update_rules)
    payoffs = list(payoffs)
    rows: List[Dict[str, float | str]] = []
    tau_pcdb = None
    upsilon_pcdb = None
    pcdb_stats: Dict[str, float] = {}
    if any(rule in {"pc", "db"} for rule in update_rules):
        tau_pcdb, pcdb_stats = solve_tau_pc_db(
            adjacency,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solver=solver,
            verbose=verbose,
        )
        upsilon_pcdb = compute_pgg_upsilon(adjacency, tau_pcdb)

    tau_bd = None
    upsilon_bd = None
    bd_stats: Dict[str, float] = {}
    if "bd" in update_rules:
        tau_bd, bd_stats = solve_tau_bd(
            adjacency,
            rtol=rtol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            solver=solver,
            verbose=verbose,
        )
        upsilon_bd = compute_pgg_upsilon(adjacency, tau_bd)

    for update_rule in update_rules:
        for payoff in payoffs:
            if update_rule in {"pc", "db"}:
                assert tau_pcdb is not None and upsilon_pcdb is not None
                result = compute_pc_or_db_threshold(
                    adjacency,
                    tau_pcdb,
                    upsilon_pcdb,
                    update_rule=update_rule,
                    payoff=payoff,
                )
                stats = pcdb_stats
            elif update_rule == "bd":
                assert tau_bd is not None and upsilon_bd is not None
                result = compute_bd_threshold(adjacency, tau_bd, upsilon_bd, payoff=payoff)
                stats = bd_stats
            else:
                raise ValueError(f"Unsupported update rule: {update_rule}")
            rows.append(
                {
                    "update_rule": update_rule,
                    "payoff": payoff,
                    **result,
                    **stats,
                }
            )
    return rows


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def expanded_choices(value: str, all_values: List[str]) -> List[str]:
    if value == "all" or value == "both":
        return all_values
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce weak-selection public-goods-game thresholds on arbitrary graphs."
    )
    parser.add_argument("--network", choices=["star", "cycle", "path", "complete", "rr", "ba", "er", "lattice", "edge-list"], required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--n-values", type=str, default=None, help="Comma-separated N values; overrides --n.")
    parser.add_argument("--avg-degree", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--edge-list", type=str, default=None)
    parser.add_argument("--update-rule", choices=["pc", "db", "bd", "all"], default="pc")
    parser.add_argument("--payoff", choices=["average", "accumulated", "both"], default="average")
    parser.add_argument("--solver", choices=["bicgstab", "gmres"], default="bicgstab")
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--restart", type=int, default=40)
    parser.add_argument("--maxiter", type=int, default=1200)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    n_values = parse_int_list(args.n_values) if args.n_values else [args.n]
    update_rules = expanded_choices(args.update_rule, ["pc", "db", "bd"])
    payoffs = expanded_choices(args.payoff, ["average", "accumulated"])
    all_rows: List[Dict[str, object]] = []
    for n in n_values:
        for sample in range(args.samples):
            seed = int(args.seed + sample + 1000003 * n)
            adjacency, metadata = build_network(args, n, seed)
            degree = np.asarray(adjacency.sum(axis=1)).ravel()
            rows = compute_pgg_thresholds(
                adjacency,
                update_rules=update_rules,
                payoffs=payoffs,
                rtol=args.rtol,
                atol=args.atol,
                restart=args.restart,
                maxiter=args.maxiter,
                solver=args.solver,
                verbose=not args.quiet,
            )
            for row in rows:
                full_row: Dict[str, object] = {
                    **metadata,
                    "N": int(adjacency.shape[0]),
                    "edges": int(adjacency.nnz // 2),
                    "avg_degree_target": float(args.avg_degree),
                    "avg_degree": float(degree.mean()),
                    "min_degree": float(degree.min()),
                    "max_degree": float(degree.max()),
                    "degree_cv": float(degree.std(ddof=0) / degree.mean()),
                    "sample": sample,
                    "seed": seed,
                    **row,
                }
                all_rows.append(full_row)
                print(
                    " ".join(
                        [
                            f"network={full_row['network']}",
                            f"N={full_row['N']}",
                            f"sample={sample}",
                            f"rule={row['update_rule']}",
                            f"payoff={row['payoff']}",
                            f"r_star={float(row['threshold']):.12g}",
                        ]
                    ),
                    flush=True,
                )
    if args.output:
        write_rows(Path(args.output), all_rows)
        print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
