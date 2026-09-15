from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse

from single_layer_threshold import (
    as_clean_csr,
    build_connected_er_exact_mean_degree,
    row_normalize,
    rowwise_inner_sum,
)
from sparse_threshold import (
    compute_threshold_single_layer_matrix_form,
    solve_eta_single_layer_linear_operator,
)


SAMPLE_FIELDS = [
    "status",
    "network_type",
    "N",
    "avg_degree_target",
    "sample",
    "seed",
    "edges",
    "avg_degree",
    "is_connected",
    "component_count",
    "isolated_count",
    "mean_degree_sq",
    "mean_degree_cubed",
    "degree_moment_ratio_k2_over_k",
    "excess_degree_mean",
    "degree_var",
    "degree_std",
    "degree_cv",
    "degree_gini",
    "min_degree",
    "max_degree",
    "kmax_over_N",
    "kmax_over_sqrtN",
    "log_kmax_over_logN",
    "assortativity",
    "average_clustering",
    "transitivity",
    "rich_club_top_1pct",
    "rich_club_top_5pct",
    "rich_club_top_10pct",
    "degree_share_top_1pct",
    "degree_share_top_5pct",
    "degree_share_top_10pct",
    "max_core_number",
    "max_core_size",
    "max_core_fraction",
    "threshold",
    "threshold_RR",
    "delta_threshold",
    "S_N",
    "R_N",
    "pbar_tau",
    "epsilon_N",
    "epsilon_crit",
    "margin_M",
    "cov_tauplus_p",
    "S_over_N",
    "p_mean",
    "SN_logN_over_N",
    "solve_time_sec",
    "total_time_sec",
    "linear_iterations",
    "relative_residual",
    "error",
]


def parse_range(spec: str, *, cast=float) -> list:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) == 3:
        start, stop, step = map(float, parts)
        count = int(round((stop - start) / step))
        values = [start + i * step for i in range(count + 1)]
    else:
        values = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if cast is int:
        return [int(round(value)) for value in values]
    return [round(float(value), 10) for value in values]


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.15g}"
    return str(value)


def seed_for(n: int, k: int, sample: int, base_seed: int) -> int:
    return int(base_seed + n * 17 + k * 1009 + sample * 1_000_003)


def regular_threshold(n: int, k: float) -> float:
    denominator = n / k - 2.0
    return math.inf if denominator <= 0 else (n - 2.0) / denominator


def degree_gini(degree: np.ndarray) -> float:
    total = float(degree.sum())
    if total <= 0.0:
        return math.nan
    x = np.sort(degree.astype(float))
    index = np.arange(1, x.size + 1, dtype=float)
    return float((2.0 * np.sum(index * x) / (x.size * total)) - (x.size + 1.0) / x.size)


def rich_club_top_fraction(graph: nx.Graph, degree: np.ndarray, fraction: float) -> float:
    n = degree.size
    count = max(2, int(math.ceil(n * fraction)))
    if count >= n:
        return math.nan
    order = np.argsort(degree)[::-1][:count]
    sub = graph.subgraph([int(i) for i in order])
    possible = count * (count - 1) / 2.0
    return float(sub.number_of_edges() / possible) if possible > 0 else math.nan


def degree_share_top_fraction(degree: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(degree.size * fraction)))
    total = float(degree.sum())
    if total <= 0.0:
        return math.nan
    return float(np.sort(degree)[::-1][:count].sum() / total)


def graph_to_csr(graph: nx.Graph) -> sparse.csr_matrix:
    adjacency = nx.to_scipy_sparse_array(graph, dtype=np.float64, format="csr")
    adjacency = adjacency.maximum(adjacency.T).tolil()
    adjacency.setdiag(0.0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    return adjacency


def compute_structural_metrics(adjacency: sparse.csr_matrix) -> dict[str, float]:
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(float)
    n = adjacency.shape[0]
    graph = nx.from_scipy_sparse_array(adjacency)
    mean_degree = float(degree.mean()) if n else math.nan
    mean_degree_sq = float(np.mean(degree * degree)) if n else math.nan
    mean_degree_cubed = float(np.mean(degree * degree * degree)) if n else math.nan
    try:
        assortativity = float(nx.degree_assortativity_coefficient(graph))
    except Exception:
        assortativity = math.nan
    try:
        core = nx.core_number(graph)
    except Exception:
        core = {int(node): 0 for node in graph.nodes()}
    core_values = np.array([core[node] for node in graph.nodes()], dtype=int)
    max_core = int(core_values.max()) if core_values.size else 0
    max_core_size = int(np.sum(core_values == max_core)) if core_values.size else 0
    kmax = float(degree.max()) if degree.size else 0.0
    return {
        "edges": int(graph.number_of_edges()),
        "avg_degree": mean_degree,
        "mean_degree_sq": mean_degree_sq,
        "mean_degree_cubed": mean_degree_cubed,
        "degree_moment_ratio_k2_over_k": float(mean_degree_sq / mean_degree) if mean_degree > 0 else math.nan,
        "excess_degree_mean": float(mean_degree_sq / mean_degree - 1.0) if mean_degree > 0 else math.nan,
        "degree_var": float(degree.var(ddof=0)) if degree.size else math.nan,
        "degree_std": float(degree.std(ddof=0)) if degree.size else math.nan,
        "degree_cv": float(degree.std(ddof=0) / mean_degree) if mean_degree > 0 else math.nan,
        "degree_gini": degree_gini(degree),
        "min_degree": float(degree.min()) if degree.size else math.nan,
        "max_degree": kmax,
        "kmax_over_N": float(kmax / n),
        "kmax_over_sqrtN": float(kmax / math.sqrt(n)),
        "log_kmax_over_logN": float(math.log(max(kmax, 1.0)) / math.log(n)),
        "assortativity": assortativity,
        "average_clustering": float(nx.average_clustering(graph)),
        "transitivity": float(nx.transitivity(graph)),
        "rich_club_top_1pct": rich_club_top_fraction(graph, degree, 0.01),
        "rich_club_top_5pct": rich_club_top_fraction(graph, degree, 0.05),
        "rich_club_top_10pct": rich_club_top_fraction(graph, degree, 0.10),
        "degree_share_top_1pct": degree_share_top_fraction(degree, 0.01),
        "degree_share_top_5pct": degree_share_top_fraction(degree, 0.05),
        "degree_share_top_10pct": degree_share_top_fraction(degree, 0.10),
        "max_core_number": float(max_core),
        "max_core_size": float(max_core_size),
        "max_core_fraction": float(max_core_size / n),
    }


def compute_remeeting_metrics(
    adjacency: sparse.csr_matrix,
    eta: np.ndarray,
    benchmark_degree: float,
) -> dict[str, float]:
    n = adjacency.shape[0]
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = row_normalize(adjacency, degree).tocsr()
    pi = degree / degree.sum()
    tau_plus = 1.0 + 2.0 * rowwise_inner_sum(eta, transition)
    local_return_2 = np.asarray(transition.multiply(transition.T).sum(axis=1)).ravel()
    s_n = float(pi @ tau_plus)
    r_n = float(pi @ (tau_plus * local_return_2))
    p_mean = float(pi @ local_return_2)
    pbar_tau = float(r_n / s_n)
    epsilon = pbar_tau - 1.0 / benchmark_degree
    epsilon_crit = (
        2.0
        * (benchmark_degree - 1.0)
        * (n - s_n)
        / (benchmark_degree * s_n * (n - 2.0))
    )
    return {
        "S_N": s_n,
        "R_N": r_n,
        "pbar_tau": pbar_tau,
        "epsilon_N": float(epsilon),
        "epsilon_crit": float(epsilon_crit),
        "margin_M": float(epsilon - epsilon_crit),
        "cov_tauplus_p": float(r_n - s_n * p_mean),
        "S_over_N": float(s_n / n),
        "p_mean": p_mean,
        "SN_logN_over_N": float(s_n * math.log(n) / n),
    }


def task_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row["N"], row["avg_degree_target"], row["sample"])


def load_done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            task_key(row)
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def append_row(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def run_one_task(task: tuple[int, int, int, argparse.Namespace]) -> dict[str, object]:
    n, k, sample, args = task
    seed = seed_for(n, k, sample, args.base_seed)
    row: dict[str, object] = {
        "status": "running",
        "network_type": "connected-er-exact-mean-degree",
        "N": n,
        "avg_degree_target": k,
        "sample": sample,
        "seed": seed,
        "error": "",
    }
    start = time.perf_counter()
    try:
        adjacency = as_clean_csr(build_connected_er_exact_mean_degree(n, float(k), seed))
        graph = nx.from_scipy_sparse_array(adjacency)
        component_count = nx.number_connected_components(graph)
        isolated_count = sum(1 for _, degree in graph.degree() if degree == 0)
        structural = compute_structural_metrics(adjacency)
        row.update(
            {
                "is_connected": int(component_count == 1),
                "component_count": component_count,
                "isolated_count": isolated_count,
            }
        )
        row.update(structural)
        eta, solve_stats = solve_eta_single_layer_linear_operator(
            adjacency,
            rtol=args.rtol,
            atol=args.atol,
            maxiter=args.maxiter,
            restart=args.restart,
            solver=args.solver,
            verbose=False,
        )
        threshold, _terms = compute_threshold_single_layer_matrix_form(adjacency, eta)
        remeeting = compute_remeeting_metrics(adjacency, eta, float(k))
        row.update(
            {
                "status": "ok",
                "threshold": float(threshold),
                "threshold_RR": regular_threshold(n, float(k)),
                "delta_threshold": float(threshold - regular_threshold(n, float(k))),
                "solve_time_sec": float(solve_stats["solve_time_sec"]),
                "total_time_sec": float(time.perf_counter() - start),
                "linear_iterations": int(solve_stats["linear_iterations"]),
                "relative_residual": float(solve_stats["final_relative_residual"]),
            }
        )
        row.update(remeeting)
    except Exception as exc:
        row["status"] = "error"
        row["total_time_sec"] = float(time.perf_counter() - start)
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    return row


def build_tasks(args: argparse.Namespace) -> list[tuple[int, int, int, argparse.Namespace]]:
    ns = parse_range(args.n_values, cast=int)
    ks = parse_range(args.k_values, cast=int)
    samples = list(range(args.sample_start, args.sample_stop))
    tasks = []
    task_index = 0
    for n in ns:
        for k in ks:
            for sample in samples:
                if args.partition_count > 1 and task_index % args.partition_count != args.partition_index:
                    task_index += 1
                    continue
                tasks.append((n, k, sample, args))
                task_index += 1
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch connected ER metric and threshold sweep.")
    parser.add_argument("--n-values", default="100:10000:50")
    parser.add_argument("--k-values", default="2:50:2")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-stop", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260430)
    parser.add_argument("--solver", choices=["bicgstab", "gmres"], default="bicgstab")
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--restart", type=int, default=40)
    parser.add_argument("--maxiter", type=int, default=1200)
    parser.add_argument("--manifest", default="")
    args = parser.parse_args()

    output = Path(args.output)
    tasks = build_tasks(args)
    done = load_done_keys(output)
    tasks = [task for task in tasks if (str(task[0]), str(task[1]), str(task[2])) not in done]
    if args.max_tasks > 0:
        tasks = tasks[: args.max_tasks]

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "output": str(output),
        "task_count_to_run": len(tasks),
        "args": vars(args),
        "fields": SAMPLE_FIELDS,
        "notes": "Uses build_connected_er_exact_mean_degree: a connected ER-like graph with exact average degree, matching the existing ER correction code.",
    }
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)

    if args.workers <= 1:
        for idx, task in enumerate(tasks, start=1):
            row = run_one_task(task)
            append_row(output, row, SAMPLE_FIELDS)
            print(
                f"task_done {idx}/{len(tasks)} status={row['status']} "
                f"N={task[0]} k={task[1]} sample={task[2]} time={row.get('total_time_sec')}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_one_task, task) for task in tasks]
            future_to_task = dict(zip(futures, tasks))
            for idx, future in enumerate(as_completed(futures), start=1):
                task = future_to_task[future]
                row = future.result()
                append_row(output, row, SAMPLE_FIELDS)
                print(
                    f"task_done {idx}/{len(tasks)} status={row['status']} "
                    f"N={task[0]} k={task[1]} sample={task[2]} time={row.get('total_time_sec')}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
