"""Standalone implementations of all network families used in the comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx
import numpy as np
from scipy import sparse


MODEL_ORDER = (
    "RR",
    "ER",
    "SW",
    "BA",
    "HK",
    "KE",
    "Shifted",
    "FF",
    "Island BA",
    "Island ER",
    "Core-periphery",
)


@dataclass
class GeneratedNetwork:
    graph: nx.Graph
    labels: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_csr(self) -> sparse.csr_matrix:
        array = nx.to_scipy_sparse_array(
            self.graph,
            nodelist=range(self.graph.number_of_nodes()),
            dtype=np.float64,
            format="csr",
        )
        return sparse.csr_matrix(array)


def target_edge_count(n: int, k: float) -> int:
    value = n * float(k) / 2.0
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("N*k must be even so that the target edge count is an integer.")
    return rounded


def even_integer_degree(k: float, model: str) -> int:
    value = int(round(k))
    if not math.isclose(float(value), float(k), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{model} requires an integer target degree; got {k}.")
    if value < 2 or value % 2:
        raise ValueError(f"{model} requires a positive even target degree; got {k}.")
    return value


def edge_key(u: int, v: int) -> tuple[int, int]:
    return (v, u) if u > v else (u, v)


def choose_two_from_array(
    nodes: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int]:
    size = int(nodes.size)
    if size < 2:
        raise ValueError("At least two nodes are required.")
    first = int(rng.integers(size))
    second = int(rng.integers(size - 1))
    if second >= first:
        second += 1
    return int(nodes[first]), int(nodes[second])


def seed_for(model: str, n: int, sample: int, base_seed: int) -> int:
    model_part = sum((index + 1) * ord(char) for index, char in enumerate(model))
    return int((base_seed + 1_000_003 * sample + 1009 * n + 97 * model_part) % (2**31 - 1))


def normalize_model_name(model: str) -> str:
    key = " ".join(model.strip().lower().replace("_", " ").replace("-", " ").split())
    aliases = {
        " ".join(name.lower().replace("-", " ").split()): name
        for name in MODEL_ORDER
    }
    aliases.update(
        {
            "random regular": "RR",
            "erdos renyi": "ER",
            "erdos--renyi": "ER",
            "watts strogatz": "SW",
            "barabasi albert": "BA",
            "holme kim": "HK",
            "klemm eguiluz": "KE",
            "forest fire": "FF",
            "cp": "Core-periphery",
            "core periphery": "Core-periphery",
        }
    )
    if key not in aliases:
        raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(MODEL_ORDER)}")
    return aliases[key]


def add_uniform_missing_edges(
    graph: nx.Graph,
    target_edges: int,
    rng: np.random.Generator,
) -> None:
    n = graph.number_of_nodes()
    while graph.number_of_edges() < target_edges:
        u = int(rng.integers(n))
        v = int(rng.integers(n - 1))
        if v >= u:
            v += 1
        graph.add_edge(u, v)


def weighted_targets(
    nodes: np.ndarray,
    weights: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> list[int]:
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities /= probabilities.sum()
    count = min(int(count), int(nodes.size))
    return [
        int(value)
        for value in rng.choice(nodes, size=count, replace=False, p=probabilities)
    ]


def add_weighted_missing_edges(
    graph: nx.Graph,
    target_edges: int,
    degree: np.ndarray,
    offset: float,
    rng: np.random.Generator,
) -> None:
    nodes = np.arange(graph.number_of_nodes(), dtype=np.int64)
    while graph.number_of_edges() < target_edges:
        u, v = weighted_targets(nodes, degree + offset, 2, rng)
        if graph.has_edge(u, v):
            continue
        graph.add_edge(u, v)
        degree[u] += 1.0
        degree[v] += 1.0


def generate_rr(n: int, k: float, seed: int) -> GeneratedNetwork:
    degree = even_integer_degree(k, "RR")
    if n <= degree:
        raise ValueError("RR requires N > k.")
    raw_graph = nx.random_regular_graph(degree, n, seed=seed)
    graph = nx.convert_node_labels_to_integers(raw_graph, ordering="default")
    return GeneratedNetwork(graph, metadata={"degree": degree})


def generate_er(n: int, k: float, seed: int) -> GeneratedNetwork:
    target = target_edge_count(n, k)
    if target < n - 1:
        raise ValueError("The ER target edge count is too small to guarantee connectivity.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(
        (int(order[index]), int(order[index + 1])) for index in range(n - 1)
    )
    add_uniform_missing_edges(graph, target, rng)
    return GeneratedNetwork(graph, metadata={"connected_backbone": "random path"})


def generate_sw(
    n: int,
    k: float,
    seed: int,
    *,
    rewiring_probability: float = 0.1,
) -> GeneratedNetwork:
    degree = even_integer_degree(k, "SW")
    if n <= degree:
        raise ValueError("SW requires N > k.")
    graph = nx.connected_watts_strogatz_graph(
        n,
        k=degree,
        p=float(rewiring_probability),
        tries=500,
        seed=seed,
    )
    return GeneratedNetwork(
        graph,
        metadata={"degree": degree, "rewiring_probability": rewiring_probability},
    )


def generate_ba(n: int, k: float, seed: int) -> GeneratedNetwork:
    degree_target = even_integer_degree(k, "BA")
    m = degree_target // 2
    if n <= m:
        raise ValueError("BA requires N > m.")
    target = target_edge_count(n, degree_target)
    rng = np.random.default_rng(seed)
    graph = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(2**31 - 1)))
    degree = np.fromiter((value for _, value in graph.degree()), dtype=np.float64, count=n)
    while graph.number_of_edges() < target:
        probabilities = degree / degree.sum()
        u = int(rng.choice(n, p=probabilities))
        v = int(rng.choice(n, p=probabilities))
        if u == v or graph.has_edge(u, v):
            continue
        graph.add_edge(u, v)
        degree[u] += 1.0
        degree[v] += 1.0
    return GeneratedNetwork(graph, metadata={"m": m, "attachment": "degree"})


def generate_hk(
    n: int,
    k: float,
    seed: int,
    *,
    triangle_probability: float = 0.5,
) -> GeneratedNetwork:
    degree_target = even_integer_degree(k, "HK")
    m = degree_target // 2
    if n <= m:
        raise ValueError("HK requires N > m.")
    rng = np.random.default_rng(seed)
    graph = nx.powerlaw_cluster_graph(
        n,
        m=m,
        p=float(triangle_probability),
        seed=int(rng.integers(2**31 - 1)),
    )
    add_uniform_missing_edges(graph, target_edge_count(n, degree_target), rng)
    return GeneratedNetwork(
        graph,
        metadata={"m": m, "triangle_probability": triangle_probability},
    )


def generate_ke(
    n: int,
    k: float,
    seed: int,
    *,
    mixing_probability: float = 0.1,
) -> GeneratedNetwork:
    degree_target = even_integer_degree(k, "KE")
    m = degree_target // 2
    mu = float(mixing_probability)
    if m < 2 or n <= m:
        raise ValueError("KE requires m >= 2 and N > m.")
    if not 0.0 <= mu <= 1.0:
        raise ValueError("KE mixing_probability must lie in [0, 1].")

    rng = np.random.default_rng(seed)
    graph = nx.complete_graph(m)
    graph.add_nodes_from(range(m, n))
    degree = np.zeros(n, dtype=np.int64)
    degree[:m] = m - 1
    active = list(range(m))
    repeated_nodes = [node for node in range(m) for _ in range(m - 1)]

    for new_node in range(m, n):
        active_array = np.asarray(active, dtype=np.int64)
        use_global = rng.random(m) < mu
        targets = {
            int(active_array[index])
            for index in range(m)
            if not use_global[index]
        }
        while len(targets) < m:
            for _ in range(10_000):
                candidate = int(repeated_nodes[int(rng.integers(len(repeated_nodes)))])
                if candidate not in targets:
                    targets.add(candidate)
                    break
            else:
                candidates = np.setdiff1d(
                    np.arange(new_node, dtype=np.int64),
                    np.fromiter(targets, dtype=np.int64),
                )
                weights = degree[candidates].astype(np.float64)
                targets.add(int(rng.choice(candidates, p=weights / weights.sum())))

        ordered_targets = sorted(targets)
        for target in ordered_targets:
            graph.add_edge(new_node, target)
            degree[target] += 1
        degree[new_node] = m
        repeated_nodes.extend(ordered_targets)
        repeated_nodes.extend([new_node] * m)

        active.append(new_node)
        active_degree = degree[np.asarray(active, dtype=np.int64)].astype(np.float64)
        deactivate = int(rng.choice(len(active), p=(1.0 / active_degree) / (1.0 / active_degree).sum()))
        active.pop(deactivate)

    add_uniform_missing_edges(graph, target_edge_count(n, degree_target), rng)
    return GeneratedNetwork(graph, metadata={"m": m, "mixing_probability": mu})


def generate_shifted(
    n: int,
    k: float,
    seed: int,
    *,
    shifted_alpha: float = 1.0,
) -> GeneratedNetwork:
    degree_target = even_integer_degree(k, "Shifted")
    m = degree_target // 2
    alpha = float(shifted_alpha)
    if alpha <= -1.0:
        raise ValueError("Shifted requires shifted_alpha > -1.")
    initial = max(m + 1, 3)
    if n <= initial:
        raise ValueError("Shifted requires N larger than its initial clique.")

    rng = np.random.default_rng(seed)
    attractiveness = alpha * m
    graph = nx.complete_graph(initial)
    degree = np.zeros(n, dtype=np.float64)
    for node, value in graph.degree():
        degree[int(node)] = float(value)

    for new_node in range(initial, n):
        nodes = np.arange(new_node, dtype=np.int64)
        targets = weighted_targets(nodes, degree[:new_node] + attractiveness, m, rng)
        graph.add_node(new_node)
        for target in targets:
            graph.add_edge(new_node, target)
            degree[target] += 1.0
            degree[new_node] += 1.0

    add_weighted_missing_edges(
        graph,
        target_edge_count(n, degree_target),
        degree,
        attractiveness,
        rng,
    )
    return GeneratedNetwork(
        graph,
        metadata={"m": m, "shifted_alpha": alpha, "attractiveness": attractiveness},
    )


def forest_fire_probability_theory(m: int) -> float:
    if m < 1:
        raise ValueError("FF requires m >= 1.")
    if m == 1:
        return 0.0
    return float((m - 1) / (3 * m - 1))


def _generate_ff_raw(
    n: int,
    probability: float,
    seed: int,
) -> tuple[nx.Graph, set[tuple[int, int]], np.random.Generator]:
    rng = np.random.default_rng(seed)
    directed = nx.DiGraph()
    directed.add_node(0)
    ambassador_edges: set[tuple[int, int]] = set()

    for new_node in range(1, n):
        directed.add_node(new_node)
        ambassador = int(rng.integers(new_node))
        directed.add_edge(new_node, ambassador)
        ambassador_edges.add(edge_key(new_node, ambassador))
        visited = {ambassador}
        frontier: deque[int] = deque([ambassador])

        while frontier:
            current = frontier.popleft()
            if probability > 0.0:
                forward_count = int(rng.geometric(1.0 - probability) - 1)
                backward_count = int(rng.geometric(1.0 - probability) - 1)
            else:
                forward_count = backward_count = 0
            forward = [
                int(candidate)
                for candidate in directed.successors(current)
                if candidate != new_node and candidate not in visited
            ]
            backward = [
                int(candidate)
                for candidate in directed.predecessors(current)
                if candidate != new_node and candidate not in visited
            ]
            rng.shuffle(forward)
            rng.shuffle(backward)
            for candidate in forward[:forward_count] + backward[:backward_count]:
                if candidate in visited:
                    continue
                visited.add(candidate)
                frontier.append(candidate)
                directed.add_edge(new_node, candidate)

    return directed.to_undirected(), ambassador_edges, rng


def generate_ff(
    n: int,
    k: float,
    seed: int,
    *,
    burn_probability: float | None = None,
    adjust_edges: bool = True,
    max_addition_fraction: float | None = None,
    max_attempts: int = 100,
) -> GeneratedNetwork:
    degree_target = even_integer_degree(k, "FF")
    m = degree_target // 2
    probability = (
        forest_fire_probability_theory(m)
        if burn_probability is None
        else float(burn_probability)
    )
    if not 0.0 <= probability < 1.0:
        raise ValueError("FF burn_probability must lie in [0, 1).")
    target = target_edge_count(n, degree_target)
    if max_addition_fraction is not None and not 0.0 <= max_addition_fraction <= 1.0:
        raise ValueError("max_addition_fraction must lie in [0, 1].")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive.")

    graph: nx.Graph | None = None
    ambassador_edges: set[tuple[int, int]] = set()
    accepted_rng: np.random.Generator | None = None
    generation_seed = int(seed)
    raw_edges = 0
    adjustment = 0
    for attempt in range(max_attempts):
        generation_seed = (
            int(seed)
            if attempt == 0
            else int(np.random.SeedSequence([int(seed), attempt]).generate_state(1)[0])
        )
        candidate, candidate_ambassadors, candidate_rng = _generate_ff_raw(
            n,
            probability,
            generation_seed,
        )
        candidate_edges = candidate.number_of_edges()
        candidate_adjustment = target - candidate_edges
        addition_fraction = max(0, candidate_adjustment) / target
        if max_addition_fraction is None or addition_fraction <= max_addition_fraction:
            graph = candidate
            ambassador_edges = candidate_ambassadors
            accepted_rng = candidate_rng
            raw_edges = candidate_edges
            adjustment = candidate_adjustment
            attempts_used = attempt + 1
            break
    else:
        raise RuntimeError(
            "FF could not satisfy the random-addition cap after "
            f"{max_attempts} attempts."
        )

    assert graph is not None
    assert accepted_rng is not None
    rng = accepted_rng
    if adjust_edges and adjustment > 0:
        add_uniform_missing_edges(graph, target, rng)
    elif adjust_edges and adjustment < 0:
        removable = [
            edge_key(int(u), int(v))
            for u, v in graph.edges()
            if edge_key(int(u), int(v)) not in ambassador_edges
        ]
        if -adjustment > len(removable):
            raise RuntimeError("FF correction would have to remove an ambassador edge.")
        rng.shuffle(removable)
        graph.remove_edges_from(removable[: -adjustment])

    return GeneratedNetwork(
        graph,
        metadata={
            "m": m,
            "forward_probability": probability,
            "backward_probability": probability,
            "raw_edges": raw_edges,
            "edge_adjustment": adjustment,
            "edge_adjustment_fraction": abs(adjustment) / target,
            "random_addition_fraction": max(0, adjustment) / target,
            "generation_seed": generation_seed,
            "generation_attempts": attempts_used,
            "max_addition_fraction": max_addition_fraction,
            "probability_source": "theory" if burn_probability is None else "provided_or_calibrated",
        },
    )


def mean_ff_raw_degree(
    n: int,
    k: int,
    probability: float,
    *,
    samples: int,
    base_seed: int,
) -> tuple[float, float]:
    values: list[float] = []
    for sample in range(samples):
        result = generate_ff(
            n,
            k,
            seed_for(f"FF calibration k={k}", n, sample, base_seed),
            burn_probability=probability,
            adjust_edges=False,
        )
        values.append(2.0 * result.graph.number_of_edges() / n)
    return float(np.mean(values)), float(np.std(values, ddof=1)) if samples > 1 else 0.0


def calibrate_ff_probability(
    n: int,
    k: int,
    *,
    samples: int = 4,
    iterations: int = 8,
    base_seed: int = 20260811,
    initial_upper_cap: float = 0.36,
) -> dict[str, float | int]:
    m = even_integer_degree(k, "FF") // 2
    lower = forest_fire_probability_theory(m)
    upper = max(lower + 0.01, min(float(initial_upper_cap), lower + 0.03))
    upper_mean, _ = mean_ff_raw_degree(
        n, k, upper, samples=samples, base_seed=base_seed
    )
    while upper_mean < k and upper < 0.45:
        upper = min(0.45, upper + 0.01)
        upper_mean, _ = mean_ff_raw_degree(
            n, k, upper, samples=samples, base_seed=base_seed
        )
    if upper_mean < k:
        raise RuntimeError(f"Could not bracket FF target k={k} at N={n}.")

    raw_mean = math.nan
    for _ in range(iterations):
        midpoint = 0.5 * (lower + upper)
        raw_mean, _ = mean_ff_raw_degree(
            n, k, midpoint, samples=samples, base_seed=base_seed
        )
        if raw_mean < k:
            lower = midpoint
        else:
            upper = midpoint
    probability = 0.5 * (lower + upper)
    validation_mean, validation_sd = mean_ff_raw_degree(
        n,
        k,
        probability,
        samples=samples,
        base_seed=base_seed + 10_000_000,
    )
    return {
        "N": n,
        "k": k,
        "m": m,
        "burn_probability": probability,
        "p_theory": forest_fire_probability_theory(m),
        "raw_degree_mean": validation_mean,
        "raw_degree_sd": validation_sd,
        "signed_adjustment_fraction": (k - validation_mean) / k,
        "samples": samples,
        "iterations": iterations,
    }


def isotonic_nonincreasing(values: Iterable[float]) -> np.ndarray:
    sequence = [float(value) for value in values]
    blocks: list[dict[str, float | int]] = []
    for index, value in enumerate(sequence):
        blocks.append({"start": index, "end": index, "weight": 1.0, "value": float(value)})
        while len(blocks) >= 2 and float(blocks[-2]["value"]) < float(blocks[-1]["value"]):
            right = blocks.pop()
            left = blocks.pop()
            weight = float(left["weight"]) + float(right["weight"])
            pooled = (
                float(left["value"]) * float(left["weight"])
                + float(right["value"]) * float(right["weight"])
            ) / weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": weight,
                    "value": pooled,
                }
            )
    fitted = np.empty(len(sequence), dtype=np.float64)
    for block in blocks:
        fitted[int(block["start"]) : int(block["end"]) + 1] = float(block["value"])
    return fitted


def calibrate_ff_probability_grid(
    n_values: Iterable[int],
    k: int,
    *,
    anchor_count: int = 15,
    pilot_samples: int = 4,
    small_n_max: int = 500,
    small_n_samples: int = 100,
    iterations: int = 8,
    base_seed: int = 20260811,
) -> tuple[dict[int, float], list[dict[str, float | int]]]:
    values = np.asarray(sorted({int(value) for value in n_values}), dtype=np.int64)
    anchor_indices = np.unique(
        np.rint(np.linspace(0, len(values) - 1, min(anchor_count, len(values)))).astype(int)
    )
    anchors = values[anchor_indices]
    diagnostics: list[dict[str, float | int]] = []
    raw_probabilities: list[float] = []
    for n in anchors:
        samples = small_n_samples if n <= small_n_max else pilot_samples
        diagnostic = calibrate_ff_probability(
            int(n),
            k,
            samples=samples,
            iterations=iterations,
            base_seed=base_seed,
        )
        diagnostics.append(diagnostic)
        raw_probabilities.append(float(diagnostic["burn_probability"]))
    monotone = isotonic_nonincreasing(raw_probabilities)
    interpolated = np.interp(
        np.log(values.astype(np.float64)),
        np.log(anchors.astype(np.float64)),
        monotone,
    )
    return {int(n): float(p) for n, p in zip(values, interpolated)}, diagnostics


def split_sizes(n: int, groups: int) -> list[int]:
    base = n // groups
    remainder = n % groups
    return [base + (1 if index < remainder else 0) for index in range(groups)]


def block_graph(parts: list[nx.Graph]) -> nx.Graph:
    arrays = [
        nx.to_scipy_sparse_array(part, dtype=np.float64, format="csr")
        for part in parts
    ]
    adjacency = sparse.block_diag(arrays, format="csr", dtype=np.float64)
    return nx.from_scipy_sparse_array(adjacency)


def rewire_islands_degree_preserving(
    graph: nx.Graph,
    labels: np.ndarray,
    *,
    island_count: int,
    mixing_fraction: float,
    target_edges: int,
    rng: np.random.Generator,
) -> int:
    degree_before = np.fromiter(
        (graph.degree(node) for node in range(graph.number_of_nodes())),
        dtype=np.int64,
        count=graph.number_of_nodes(),
    )
    removable: dict[int, list[tuple[int, int]]] = {}
    for island in range(island_count):
        nodes = np.flatnonzero(labels == island)
        subgraph = graph.subgraph(nodes)
        protected = {
            edge_key(int(u), int(v))
            for u, v in nx.dfs_edges(subgraph, source=int(nodes[0]))
        }
        pool = [
            edge_key(int(u), int(v))
            for u, v in subgraph.edges()
            if edge_key(int(u), int(v)) not in protected
        ]
        rng.shuffle(pool)
        removable[island] = pool

    target_cross_edges = 2 * int(math.floor(mixing_fraction * target_edges / 2.0 + 0.5))
    target_cross_edges = max(2 * (island_count - 1), target_cross_edges)

    def swap_between(left: int, right: int) -> None:
        attempts = 0
        while removable[left] and removable[right]:
            attempts += 1
            first_internal = removable[left].pop()
            second_internal = removable[right].pop()
            u, v = first_internal
            x, y = second_internal
            proposals = [((u, x), (v, y)), ((u, y), (v, x))]
            if rng.random() < 0.5:
                proposals.reverse()
            for first_cross, second_cross in proposals:
                if graph.has_edge(*first_cross) or graph.has_edge(*second_cross):
                    continue
                graph.remove_edge(*first_internal)
                graph.remove_edge(*second_internal)
                graph.add_edge(*first_cross)
                graph.add_edge(*second_cross)
                return
            if attempts > 20_000:
                break
        raise RuntimeError(f"Could not rewire islands {left} and {right}.")

    order = [int(value) for value in rng.permutation(island_count)]
    for left, right in zip(order[:-1], order[1:]):
        swap_between(left, right)
    for _ in range(target_cross_edges // 2 - (island_count - 1)):
        available = [island for island, pool in removable.items() if pool]
        left, right = [int(value) for value in rng.choice(available, size=2, replace=False)]
        swap_between(left, right)

    degree_after = np.fromiter(
        (graph.degree(node) for node in range(graph.number_of_nodes())),
        dtype=np.int64,
        count=graph.number_of_nodes(),
    )
    cross_edges = sum(labels[int(u)] != labels[int(v)] for u, v in graph.edges())
    if not np.array_equal(degree_before, degree_after):
        raise RuntimeError("Island rewiring changed the degree sequence.")
    if graph.number_of_edges() != target_edges or cross_edges != target_cross_edges:
        raise RuntimeError("Island rewiring changed an edge-count invariant.")
    if not nx.is_connected(graph):
        raise RuntimeError("The rewired island graph is disconnected.")
    return int(cross_edges)


def generate_island(
    family: str,
    n: int,
    k: float,
    seed: int,
    *,
    island_count: int = 5,
    mixing_fraction: float = 0.05,
) -> GeneratedNetwork:
    model = normalize_model_name(family)
    if model not in {"Island BA", "Island ER"}:
        raise ValueError("family must be Island BA or Island ER.")
    degree_target = even_integer_degree(k, model)
    sizes = split_sizes(n, island_count)
    if min(sizes) <= degree_target:
        raise ValueError("Every island must contain more nodes than k.")

    rng = np.random.default_rng(seed)
    labels = np.empty(n, dtype=np.int32)
    parts: list[nx.Graph] = []
    offset = 0
    for island, size in enumerate(sizes):
        local_seed = int(rng.integers(2**31 - 1))
        part = (
            generate_ba(size, degree_target, local_seed).graph
            if model == "Island BA"
            else generate_er(size, degree_target, local_seed).graph
        )
        parts.append(part)
        labels[offset : offset + size] = island
        offset += size

    graph = block_graph(parts)
    target = target_edge_count(n, degree_target)
    if graph.number_of_edges() != target:
        raise RuntimeError("Independent islands do not match the target edge count.")
    cross_edges = rewire_islands_degree_preserving(
        graph,
        labels,
        island_count=island_count,
        mixing_fraction=mixing_fraction,
        target_edges=target,
        rng=rng,
    )
    return GeneratedNetwork(
        graph,
        labels=labels,
        metadata={
            "island_count": island_count,
            "mixing_fraction": mixing_fraction,
            "cross_edges": cross_edges,
            "island_sizes": sizes,
        },
    )


def generate_core_periphery(
    n: int,
    k: float,
    seed: int,
    *,
    core_fraction: float = 0.10,
    pair_weights: tuple[float, float, float] = (12.0, 2.0, 0.35),
) -> GeneratedNetwork:
    target = target_edge_count(n, k)
    rng = np.random.default_rng(seed)
    core_size = min(max(4, int(round(core_fraction * n))), n - 1)
    order = rng.permutation(n)
    core = np.asarray(order[:core_size], dtype=np.int64)
    periphery = np.asarray(order[core_size:], dtype=np.int64)
    labels = np.ones(n, dtype=np.int32)
    labels[core] = 0
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((int(u), int(v)) for u, v in zip(core[:-1], core[1:]))
    for node in periphery:
        graph.add_edge(int(node), int(core[int(rng.integers(core_size))]))

    periphery_size = n - core_size
    remaining = np.array(
        [
            core_size * (core_size - 1) // 2 - (core_size - 1),
            core_size * periphery_size - periphery_size,
            periphery_size * (periphery_size - 1) // 2,
        ],
        dtype=np.int64,
    )
    weights = np.asarray(pair_weights, dtype=np.float64)
    misses = 0
    while graph.number_of_edges() < target:
        capacities = weights * np.maximum(remaining, 0)
        if capacities.sum() <= 0:
            raise ValueError("No free core-periphery edge candidates remain.")
        category = int(rng.choice(3, p=capacities / capacities.sum()))
        if category == 0:
            u, v = choose_two_from_array(core, rng)
        elif category == 1:
            u = int(core[int(rng.integers(core_size))])
            v = int(periphery[int(rng.integers(periphery_size))])
        else:
            u, v = choose_two_from_array(periphery, rng)
        if graph.has_edge(u, v):
            misses += 1
            if misses > 20_000:
                add_uniform_missing_edges(graph, target, rng)
                break
            continue
        graph.add_edge(u, v)
        remaining[category] -= 1
        misses = 0

    return GeneratedNetwork(
        graph,
        labels=labels,
        metadata={
            "core_fraction": core_fraction,
            "core_size": core_size,
            "pair_weights": pair_weights,
        },
    )


def validate_network(
    result: GeneratedNetwork,
    *,
    model: str,
    n: int,
    k: float,
) -> None:
    graph = result.graph
    if set(graph.nodes()) != set(range(n)):
        raise RuntimeError(f"{model}: nodes are not labelled 0,...,N-1.")
    if graph.number_of_edges() != target_edge_count(n, k):
        raise RuntimeError(f"{model}: incorrect final edge count.")
    if nx.number_of_selfloops(graph):
        raise RuntimeError(f"{model}: self-loops are present.")
    if model != "RR" and not nx.is_connected(graph):
        raise RuntimeError(f"{model}: graph is disconnected.")
    if result.labels is not None and result.labels.shape != (n,):
        raise RuntimeError(f"{model}: invalid community-label array.")


def generate_network(
    model: str,
    n: int,
    k: float,
    seed: int,
    *,
    sw_beta: float = 0.1,
    hk_triangle_probability: float = 0.5,
    ke_mu: float = 0.1,
    shifted_alpha: float = 1.0,
    ff_burn_probability: float | None = None,
    ff_max_addition_fraction: float | None = None,
    ff_max_attempts: int = 100,
    island_count: int = 5,
    island_mixing_fraction: float = 0.05,
    core_fraction: float = 0.10,
    core_pair_weights: tuple[float, float, float] = (12.0, 2.0, 0.35),
) -> GeneratedNetwork:
    canonical = normalize_model_name(model)
    dispatch = {
        "RR": lambda: generate_rr(n, k, seed),
        "ER": lambda: generate_er(n, k, seed),
        "SW": lambda: generate_sw(n, k, seed, rewiring_probability=sw_beta),
        "BA": lambda: generate_ba(n, k, seed),
        "HK": lambda: generate_hk(n, k, seed, triangle_probability=hk_triangle_probability),
        "KE": lambda: generate_ke(n, k, seed, mixing_probability=ke_mu),
        "Shifted": lambda: generate_shifted(n, k, seed, shifted_alpha=shifted_alpha),
        "FF": lambda: generate_ff(
            n,
            k,
            seed,
            burn_probability=ff_burn_probability,
            max_addition_fraction=ff_max_addition_fraction,
            max_attempts=ff_max_attempts,
        ),
        "Island BA": lambda: generate_island(
            "Island BA",
            n,
            k,
            seed,
            island_count=island_count,
            mixing_fraction=island_mixing_fraction,
        ),
        "Island ER": lambda: generate_island(
            "Island ER",
            n,
            k,
            seed,
            island_count=island_count,
            mixing_fraction=island_mixing_fraction,
        ),
        "Core-periphery": lambda: generate_core_periphery(
            n,
            k,
            seed,
            core_fraction=core_fraction,
            pair_weights=core_pair_weights,
        ),
    }
    result = dispatch[canonical]()
    result.metadata.update(
        {
            "model": canonical,
            "N": n,
            "k": float(k),
            "seed": int(seed),
            "edges": result.graph.number_of_edges(),
            "connected": nx.is_connected(result.graph),
        }
    )
    validate_network(result, model=canonical, n=n, k=k)
    return result


def serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    return value


def export_network(result: GeneratedNetwork, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / f"{stem}.edgelist"
    with edge_path.open("w", encoding="utf-8") as handle:
        for u, v in sorted(edge_key(int(u), int(v)) for u, v in result.graph.edges()):
            handle.write(f"{u} {v}\n")
    if result.labels is not None:
        with (output_dir / f"{stem}_labels.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["node", "community"])
            writer.writerows((node, int(label)) for node, label in enumerate(result.labels))
    with (output_dir / f"{stem}_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {key: serializable(value) for key, value in result.metadata.items()},
            handle,
            indent=2,
            sort_keys=True,
        )


def run_self_test() -> None:
    n = 120
    for k in (4, 6, 8, 10):
        ff_probability = min(0.42, forest_fire_probability_theory(k // 2) + 0.08)
        for index, model in enumerate(MODEL_ORDER):
            result = generate_network(
                model,
                n,
                k,
                seed=10_000 + 100 * k + index,
                ff_burn_probability=ff_probability,
            )
            print(
                f"ok model={model:15s} N={n} k={k} "
                f"E={result.graph.number_of_edges()} connected={nx.is_connected(result.graph)}"
            )
    diagnostic = calibrate_ff_probability(100, 4, samples=4, iterations=4)
    assert 0.0 < float(diagnostic["burn_probability"]) < 0.45
    print("ok FF calibration smoke test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone generators for all comparison networks.")
    parser.add_argument("--model", default="all")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--shifted-alpha", type=float, default=1.0)
    parser.add_argument("--ke-mu", type=float, default=0.1)
    parser.add_argument("--ff-burn-probability", type=float, default=None)
    parser.add_argument("--calibrate-ff", action="store_true")
    parser.add_argument("--ff-calibration-samples", type=int, default=4)
    parser.add_argument("--ff-calibration-iterations", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    models = MODEL_ORDER if args.model.lower() in {"all", "*"} else (normalize_model_name(args.model),)
    ff_probability = args.ff_burn_probability
    if "FF" in models and ff_probability is None and args.calibrate_ff:
        diagnostic = calibrate_ff_probability(
            args.n,
            args.k,
            samples=args.ff_calibration_samples,
            iterations=args.ff_calibration_iterations,
            base_seed=args.seed,
        )
        ff_probability = float(diagnostic["burn_probability"])
        print(json.dumps({key: serializable(value) for key, value in diagnostic.items()}, indent=2))

    for index, model in enumerate(models):
        model_seed = seed_for(model, args.n, index, args.seed)
        result = generate_network(
            model,
            args.n,
            args.k,
            model_seed,
            shifted_alpha=args.shifted_alpha,
            ke_mu=args.ke_mu,
            ff_burn_probability=ff_probability,
        )
        print(json.dumps({key: serializable(value) for key, value in result.metadata.items()}, sort_keys=True))
        if args.output_dir is not None:
            stem = model.lower().replace(" ", "_").replace("-", "_")
            export_network(result, args.output_dir, stem)


if __name__ == "__main__":
    main()
