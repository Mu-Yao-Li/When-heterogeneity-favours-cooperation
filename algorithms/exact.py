"""Exact weak-selection, rare-mutation donation-game thresholds.

Default: death-birth (DB) updating and average payoffs. Also supports
pairwise comparison (PC), imitation (IM), and accumulated payoffs.
Run ``python algorithms/exact.py --help`` for the command-line interface.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy

if __package__:
    from .exact_support.coalescence import _as_unweighted_adjacency
    from .exact_support.thresholds import rare_mutation_thresholds
else:
    from exact_support.coalescence import _as_unweighted_adjacency
    from exact_support.thresholds import rare_mutation_thresholds

RULE_NAMES = {"DB": "death_birth", "PC": "pairwise_comparison", "IM": "imitation"}


def analyze(structure, *, update_rule="DB", payoff="average", rtol=1e-10,
            atol=1e-12, maxiter=6000, solver="bicgstab") -> dict:
    """Return the exact first-order criterion for a connected unweighted graph.

    ``structure`` is a NetworkX graph, SciPy sparse adjacency, or NumPy
    adjacency. Selection favours cooperation when
    ``b * benefit_coefficient + c * cost_coefficient > 0``.
    The criterion compares uniformly initialized single-mutant fixation
    probabilities, equivalently abundance in the rare-mutation limit.

    ``bc_star_signed`` is the formal zero crossing. ``bc_star`` and
    ``threshold`` are null when no ordinary positive lower threshold exists.
    A negative zero crossing must not be interpreted as an advantageous
    positive benefit-to-cost ratio. All non-finite results are returned as
    null, so the result can be written as standards-compliant JSON.

    DB chooses a neighbour of a uniform focal site in proportion to fitness;
    IM includes the focal site in that candidate set. PC compares with one
    uniform neighbour and copies with probability F_j/(F_i+F_j).
    Average payoffs are b*(P x)_i-c*x_i; accumulated payoffs are
    b*(W x)_i-c*k_i*x_i. exp(delta*f) and 1+delta*f have the same first-order
    expansion used here. The solver stores O(N^2) coalescence times.
    """
    if update_rule not in RULE_NAMES:
        raise ValueError("update_rule must be DB, PC, or IM")
    if payoff not in ("average", "accumulated"):
        raise ValueError("payoff must be average or accumulated")
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be positive and finite")
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("atol must be nonnegative and finite")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    started = time.perf_counter()
    adjacency = _as_unweighted_adjacency(structure)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    raw = rare_mutation_thresholds(adjacency,
        rules=(RULE_NAMES[update_rule],), payoff_aggregations=(payoff,),
        rtol=rtol, atol=atol, maxiter=maxiter, solver=solver)[0]
    signed = raw["bc_star"]
    threshold = signed if raw["threshold_status"] == "finite_positive" else None
    result = {
        "N": int(adjacency.shape[0]), "edges": int(adjacency.nnz // 2),
        "avg_degree": float(degree.mean()),
        "degree_cv": float(degree.std() / degree.mean()),
        **raw,
        "update_rule": update_rule, "update_rule_name": RULE_NAMES[update_rule],
        "payoff": payoff, "bc_star_signed": signed,
        "threshold_signed": signed, "bc_star": threshold, "threshold": threshold,
        "selection_condition": "b * benefit_coefficient + c * cost_coefficient > 0",
        "solver": solver, "rtol": rtol, "atol": atol, "maxiter": maxiter,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "total_time_sec": time.perf_counter() - started,
    }
    return {key: (None if isinstance(value, (float, np.floating))
                 and not math.isfinite(value) else value)
            for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-file", type=Path, required=True,
                        help="two-column unweighted edge list")
    parser.add_argument("--update-rule", choices=tuple(RULE_NAMES), default="DB")
    parser.add_argument("--payoff", choices=("average", "accumulated"), default="average")
    parser.add_argument("--output", type=Path,
                        help="output .json or .csv; omitted: write JSON to stdout")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--maxiter", type=int, default=6000)
    parser.add_argument("--solver", choices=("bicgstab", "gmres"), default="bicgstab")
    args = parser.parse_args()
    if args.output and args.output.suffix.lower() not in (".json", ".csv"):
        parser.error("--output must end in .json or .csv")
    # Share the repository's edge normalization with approximate/simulation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
    from run_adaptive_closed_tail_pipeline import load_edge_list
    try:
        adjacency, _ = load_edge_list(args.edge_file)
        row = analyze(adjacency, update_rule=args.update_rule, payoff=args.payoff,
            rtol=args.rtol, atol=args.atol, maxiter=args.maxiter, solver=args.solver)
        row.update(edge_file=args.edge_file.name,
                   edge_file_sha256=hashlib.sha256(args.edge_file.read_bytes()).hexdigest())
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Exact calculation failed: {error}\n")
    if args.output is None:
        print(json.dumps(row, indent=2, allow_nan=False))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    else:
        args.output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    print(f"{args.update_rule}, {args.payoff}: {row['threshold_status']}; "
          f"signed b/c = {row['bc_star_signed']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
