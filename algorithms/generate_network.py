"""Generate the manuscript network families as portable, unweighted edge lists.

The implementation is delegated to the preserved generators in ``code/``.
PA uses the degree-power attachment convention; BA retains the separate legacy
BA convention used in the network-family comparison. They need not give the
same graph for gamma=1 and the same seed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from generative_network_models import (  # noqa: E402
    GeneratedNetwork, MODEL_ORDER, generate_network as generate_family,
    normalize_model_name, serializable, target_edge_count,
)
from single_layer_threshold import build_pa_degree_power_exact  # noqa: E402


def generate(model: str, n: int = 100, k: float = 4, seed: int = 42,
             *, gamma: float = 1.0, **options) -> GeneratedNetwork:
    """Return a connected graph with exactly N*k/2 edges, or reject the request."""
    if n < 3 or not math.isfinite(k) or not 0 <= seed < 2**63:
        raise ValueError("Require N >= 3, finite k, and a seed in [0, 2**63).")
    target = target_edge_count(n, k)
    if not n - 1 <= target <= n * (n - 1) // 2:
        raise ValueError("The requested edge count is impossible for a connected simple graph.")
    compact = model.lower().replace("_", "").replace("-", "").replace(" ", "")
    model = {"islandba": "Island BA", "islander": "Island ER"}.get(compact, model)
    if compact == "pa":
        if not math.isfinite(gamma):
            raise ValueError("gamma must be finite.")
        if k < 2 or k != int(k) or int(k) % 2 or n <= max(int(k) // 2 + 1, 3):
            raise ValueError("PA requires an even integer k >= 2 and N > max(k/2 + 1, 3).")
        matrix = build_pa_degree_power_exact(n, k, gamma, seed)
        graph = nx.from_scipy_sparse_array(matrix)
        result = GeneratedNetwork(graph, metadata={
            "model": "PA", "N": n, "k": float(k), "seed": seed, "gamma": gamma,
            "m": int(k) // 2, "initial_complete_graph_size": max(int(k) // 2 + 1, 3),
            "attachment": "degree**gamma; weighted missing edges added to N*k/2",
            "generator": "code/single_layer_threshold.py:build_pa_degree_power_exact",
        })
    else:
        result = generate_family(normalize_model_name(model), n, k, seed, **options)
        result.metadata["generator"] = "code/generative_network_models.py:generate_network"
    if not nx.is_connected(result.graph):
        raise ValueError("This seed produced a disconnected graph; choose another seed.")
    if result.graph.number_of_edges() != target:
        raise RuntimeError("The generated edge count does not equal N*k/2.")
    result.metadata.update({
        "edges": result.graph.number_of_edges(), "connected": True,
        "numpy_version": np.__version__, "networkx_version": nx.__version__,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="PA", help="PA, " + ", ".join(MODEL_ORDER))
    parser.add_argument("--n", "--nodes", dest="n", type=int, default=100)
    parser.add_argument("--k", "--degree", dest="k", type=float, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=1.0, help="Degree exponent for PA.")
    parser.add_argument("--output", type=Path, required=True, help="Output .edgelist file.")
    parser.add_argument("--sw-beta", type=float, default=0.1)
    parser.add_argument("--hk-triangle-probability", type=float, default=0.5)
    parser.add_argument("--ke-mu", type=float, default=0.1)
    parser.add_argument("--shifted-alpha", type=float, default=1.0)
    parser.add_argument("--ff-burn-probability", type=float)
    parser.add_argument("--ff-max-addition-fraction", type=float)
    parser.add_argument("--ff-max-attempts", type=int, default=100)
    parser.add_argument("--island-count", type=int, default=5)
    parser.add_argument("--island-mixing-fraction", type=float, default=0.05)
    parser.add_argument("--core-fraction", type=float, default=0.10)
    parser.add_argument("--core-pair-weights", type=float, nargs=3, default=(12.0, 2.0, 0.35))
    args = parser.parse_args()
    options = {key: value for key, value in vars(args).items()
               if key not in {"model", "n", "k", "seed", "gamma", "output"}}
    for key in ("sw_beta", "hk_triangle_probability", "ke_mu", "island_mixing_fraction"):
        if not 0 <= options[key] <= 1:
            parser.error(f"{key} must be in [0, 1].")
    if not 0 < args.core_fraction < 1 or args.island_count < 2:
        parser.error("Require 0 < core-fraction < 1 and island-count >= 2.")
    if any(not math.isfinite(x) or x < 0 for x in args.core_pair_weights) or not any(args.core_pair_weights):
        parser.error("core-pair-weights must be finite, nonnegative, and not all zero.")
    result = generate(args.model, args.n, args.k, args.seed, gamma=args.gamma, **options)
    edges = sorted(tuple(sorted((int(u), int(v)))) for u, v in result.graph.edges())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    edge_bytes = "".join(f"{u} {v}\n" for u, v in edges).encode("ascii")
    args.output.write_bytes(edge_bytes)
    result.metadata["edge_file_sha256"] = hashlib.sha256(edge_bytes).hexdigest()
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(result.metadata, indent=2, default=serializable) + "\n", encoding="utf-8")
    if result.labels is not None:
        label_path = args.output.with_suffix(".communities.csv")
        label_path.write_text("node,community\n" + "".join(
            f"{node},{int(label)}\n" for node, label in enumerate(result.labels)), encoding="ascii")
    print(f'{result.metadata["model"]}: N={args.n}, edges={len(edges)}, k={args.k:g}, seed={args.seed}')
    print(f"Wrote {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()
