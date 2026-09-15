"""Donation-game simulations with DB (default), PC, or IM updates.

Average payoff is b * cooperative_neighbours / degree - c * own_strategy;
accumulated payoff is b * cooperative_neighbours - c * degree * own_strategy.
Fitness is exp(delta * payoff). At each focal update mutation resets the
strategy to uniform C/D with probability mu, including PC self-copy events.
Every post-update state is counted, including holds. No burn-in is discarded
by default. Python is a readable reference; --backend julia runs the compiled
production adaptation. Seeds are reproducible within a backend and runtime,
not identical between Python and Julia. A short run is not a convergence test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run_adaptive_closed_tail_pipeline import load_edge_list  # noqa: E402

RULES = ("DB", "PC", "IM")


def validate_adjacency(adjacency) -> sparse.csr_matrix:
    a = sparse.csr_matrix(adjacency, dtype=float)
    a.eliminate_zeros()
    if a.shape[0] < 2 or a.shape[0] != a.shape[1] or (a != a.T).nnz:
        raise ValueError("Require a symmetric graph with at least two nodes.")
    if np.any(a.diagonal()) or np.any(a.data != 1):
        raise ValueError("Require a binary, unweighted graph without self-loops.")
    if connected_components(a, directed=False, return_labels=False) != 1:
        raise ValueError("The graph must be connected.")
    a.sort_indices()
    return a


def validate_parameters(*, benefit, cost, delta, mutation_rate, steps, burn_in,
                        seed, rule, payoff, initial_cooperation, record_every):
    if rule not in RULES or payoff not in ("average", "accumulated"):
        raise ValueError("Use rule DB/PC/IM and payoff average/accumulated.")
    if any(not math.isfinite(x) or x < 0 for x in (benefit, cost, delta)):
        raise ValueError("Benefit, cost, and delta must be finite and nonnegative.")
    if not 0 <= mutation_rate <= 1 or not 0 <= initial_cooperation <= 1:
        raise ValueError("Mutation and initial cooperation probabilities must be in [0, 1].")
    if steps < 1 or burn_in < 0 or record_every < 1 or not 0 <= seed < 2**63:
        raise ValueError("Require steps > 0, burn-in >= 0, record-every > 0, 0 <= seed < 2**63.")


def _logistic(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _choose_source(focal, state, cooperative_neighbours, neighbours, degree,
                   benefit, cost, delta, rule, payoff, rng):
    candidates = neighbours[focal]
    if rule == "PC":
        source = int(candidates[rng.integers(len(candidates))])
        pair = np.array([focal, source])
        raw = benefit * cooperative_neighbours[pair] - cost * degree[pair] * state[pair]
        values = raw / degree[pair] if payoff == "average" else raw
        return source if rng.random() < _logistic(delta * float(values[1] - values[0])) else focal
    if rule == "IM":
        candidates = np.concatenate(([focal], candidates))
    raw = benefit * cooperative_neighbours[candidates] - cost * degree[candidates] * state[candidates]
    values = raw / degree[candidates] if payoff == "average" else raw
    scores = delta * values
    weights = np.exp(scores - scores.max())
    threshold = rng.random() * float(weights.sum())
    index = int(np.searchsorted(np.cumsum(weights), threshold, side="right"))
    return int(candidates[min(index, len(candidates) - 1)])


def simulate(adjacency, *, benefit=4.0, cost=1.0, delta=0.01,
             mutation_rate=1e-4, steps=100_000, burn_in=0, seed=42,
             rule="DB", payoff="average", initial_cooperation=0.5,
             record_every=10_000, trace_hook=None, initial_state=None):
    """Run one fixed-length trajectory; return its time mean and final state.

    ``steps`` counts measured updates, in addition to ``burn_in`` updates.
    A trace hook receives (measured_steps, block_mean, cumulative_mean).
    Incremental integer neighbour counts prevent payoff-update rounding drift.
    """
    validate_parameters(benefit=benefit, cost=cost, delta=delta, mutation_rate=mutation_rate,
                        steps=steps, burn_in=burn_in, seed=seed, rule=rule, payoff=payoff,
                        initial_cooperation=initial_cooperation, record_every=record_every)
    a = validate_adjacency(adjacency)
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    if initial_state is None:
        state = (rng.random(n) < initial_cooperation).astype(np.int64)
    else:
        supplied = np.asarray(initial_state)
        if supplied.shape != (n,) or not np.isin(supplied, (0, 1)).all():
            raise ValueError("initial_state must have N entries, each 0 or 1.")
        state = supplied.astype(np.int64, copy=True)
    neighbours = [a.indices[a.indptr[i]:a.indptr[i + 1]] for i in range(n)]
    degree = np.diff(a.indptr).astype(np.int64)
    cooperative_neighbours = np.asarray(a @ state).astype(np.int64)
    count = int(state.sum())
    total = block_total = block_steps = 0
    started = time.perf_counter()
    for update in range(1, burn_in + steps + 1):
        focal = int(rng.integers(n))
        source = _choose_source(focal, state, cooperative_neighbours, neighbours, degree,
                                benefit, cost, delta, rule, payoff, rng)
        new = int(rng.integers(2)) if rng.random() < mutation_rate else int(state[source])
        change = new - int(state[focal])
        if change:
            state[focal] = new
            cooperative_neighbours[neighbours[focal]] += change
            count += change
        if update > burn_in:
            measured = update - burn_in
            total += count
            block_total += count
            block_steps += 1
            if measured % record_every == 0 or measured == steps:
                if trace_hook is not None:
                    trace_hook(measured, block_total / (n * block_steps), total / (n * measured))
                block_total = block_steps = 0
    return {"cooperation_sum": total, "mean_cooperation": total / (n * steps),
            "final_cooperation": count / n, "measured_steps": steps,
            "total_updates": burn_in + steps, "seconds": time.perf_counter() - started,
            "final_state": state, "cooperative_neighbours": cooperative_neighbours}


def run_julia(adjacency, *, executable, trace_path=None, **parameters):
    """Use the included Julia production adaptation on the same normalized graph."""
    engine = Path(__file__).resolve().parent / "simulation_support/production_julia/simulate.jl"
    a = validate_adjacency(adjacency)
    upper = sparse.triu(a, k=1).tocoo()
    with tempfile.TemporaryDirectory(prefix="cooperation_sim_") as directory:
        temporary = Path(directory)
        edges = temporary / "edges.txt"
        edges.write_text("".join(f"{i} {j}\n" for i, j in zip(upper.row, upper.col)), encoding="ascii")
        output = temporary / "result.csv"
        args = [executable, "--startup-file=no", str(engine), str(edges), str(a.shape[0]),
                str(output), str(trace_path) if trace_path else "-", parameters["rule"],
                parameters["payoff"], str(parameters["benefit"]), str(parameters["cost"]),
                str(parameters["delta"]), str(parameters["mutation_rate"]), str(parameters["steps"]),
                str(parameters["burn_in"]), str(parameters["seed"]),
                str(parameters["initial_cooperation"]), str(parameters["record_every"])]
        subprocess.run(args, check=True)
        with output.open(encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))
    integers = ("cooperation_sum", "measured_steps", "total_updates")
    return {key: int(value) if key in integers else float(value) for key, value in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Replicate means CSV; also writes .summary.json.")
    parser.add_argument("--rule", "--update-rule", dest="rule", choices=RULES, default="DB", type=str.upper)
    parser.add_argument("--payoff", choices=("average", "accumulated"), default="average")
    parser.add_argument("--benefit", type=float, default=4.0)
    parser.add_argument("--cost", type=float, default=1.0)
    parser.add_argument("--delta", "--selection-strength", dest="delta", type=float, default=0.01)
    parser.add_argument("--mutation-rate", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=100_000, help="Measured updates per replicate, after burn-in.")
    parser.add_argument("--burn-in", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42, help="SeedSequence master seed; actual replicate seeds are recorded.")
    parser.add_argument("--initial-cooperation", type=float, default=0.5)
    parser.add_argument("--record-every", type=int, default=10_000)
    parser.add_argument("--trace-dir", type=Path, help="Optional block and cumulative trajectory means.")
    parser.add_argument("--backend", choices=("python", "julia"), default="python")
    parser.add_argument("--julia-executable", default="julia")
    args = parser.parse_args()
    parameters = {key: getattr(args, key) for key in (
        "benefit", "cost", "delta", "mutation_rate", "steps", "burn_in", "seed", "rule",
        "payoff", "initial_cooperation", "record_every")}
    validate_parameters(**parameters)
    if args.replicates < 1:
        parser.error("replicates must be positive.")
    executable = shutil.which(args.julia_executable) if args.backend == "julia" else None
    if args.backend == "julia" and not executable:
        parser.error("Julia was not found; install Julia or set --julia-executable to its executable.")
    a, _ = load_edge_list(args.edge_file)
    if args.trace_dir:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.generate_state(1, dtype=np.uint64)[0]) % (2**63)
             for s in np.random.SeedSequence(args.seed).spawn(args.replicates)]
    rows = []
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = None
        for replicate, seed in enumerate(seeds, start=1):
            trace_path = args.trace_dir / f"replicate_{replicate:03d}.csv" if args.trace_dir else None
            parameters["seed"] = seed
            if args.backend == "julia":
                result = run_julia(a, executable=executable, trace_path=trace_path, **parameters)
            elif trace_path:
                with trace_path.open("w", encoding="utf-8", newline="") as trace:
                    trace_writer = csv.writer(trace)
                    trace_writer.writerow(("measured_steps", "block_mean", "cumulative_mean"))
                    def record(*row):
                        trace_writer.writerow(row)
                        trace.flush()
                    result = simulate(a, trace_hook=record, **parameters)
            else:
                result = simulate(a, **parameters)
            result.pop("final_state", None)
            result.pop("cooperative_neighbours", None)
            row = {"replicate": replicate, "seed": seed, "backend": args.backend,
                   "rule": args.rule, "payoff": args.payoff, "N": a.shape[0],
                   **{key: parameters[key] for key in ("benefit", "cost", "delta", "mutation_rate", "burn_in")},
                   **result}
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(f'Replicate {replicate}/{args.replicates}: mean={row["mean_cooperation"]:.8g}, '
                  f'seconds={row["seconds"]:.3f}', flush=True)
    means = np.asarray([row["mean_cooperation"] for row in rows])
    summary = {
        "configuration": {key: value for key, value in vars(args).items() if key not in
                          ("edge_file", "output", "trace_dir", "julia_executable")},
        "edge_file": args.edge_file.name,
        "edge_file_sha256": hashlib.sha256(args.edge_file.read_bytes()).hexdigest(),
        "N": a.shape[0], "edges": a.nnz // 2, "replicate_seeds": seeds,
        "mean_cooperation": float(means.mean()),
        "replicate_standard_deviation_ddof1": float(means.std(ddof=1)) if len(means) > 1 else None,
        "standard_error_of_replicate_means": float(means.std(ddof=1) / math.sqrt(len(means))) if len(means) > 1 else None,
        "sampling": "Every state after a measured update, including holds; burn-in separately recorded.",
        "fitness": "exp(delta * payoff)",
        "mutation": "With probability mu, set focal strategy to uniform C/D; flip probability mu/2.",
        "convergence_assessed": False, "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    if executable:
        summary["julia_version"] = subprocess.check_output([executable, "--version"], text=True).strip()
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} and {summary_path}")


if __name__ == "__main__":
    main()
