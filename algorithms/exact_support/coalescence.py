"""Neutral two-lineage solver for DB, IM and PC rare-mutation calculations.

Adapted from the local research extension sigma/im_rare_mutation.py.
See PROVENANCE.md for source and mathematical attribution.
"""

from __future__ import annotations

import time
from typing import Any

import networkx as nx
import numpy as np
from scipy import sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres


def _as_unweighted_adjacency(
    structure: nx.Graph | sp.spmatrix | np.ndarray,
    nodelist: list[Any] | None = None,
) -> sp.csr_matrix:
    """Return a validated symmetric 0/1 adjacency matrix."""
    if isinstance(structure, nx.Graph):
        if structure.is_directed():
            raise ValueError("The graph must be undirected.")
        if structure.is_multigraph():
            raise ValueError("The graph must be simple, not a multigraph.")
        if any(data.get("weight", 1) != 1 for _, _, data in structure.edges(data=True)):
            raise ValueError("This implementation requires an unweighted graph.")
        if nodelist is None:
            nodelist = list(structure.nodes())
        adjacency = nx.to_scipy_sparse_array(
            structure,
            nodelist=nodelist,
            dtype=float,
            format="csr",
            weight=None,
        )
        adjacency = sp.csr_matrix(adjacency)
    elif sp.issparse(structure):
        adjacency = sp.csr_matrix(structure, dtype=float)
    else:
        adjacency = sp.csr_matrix(np.asarray(structure, dtype=float))

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("The adjacency matrix must be square.")
    if adjacency.shape[0] < 2:
        raise ValueError("The graph must contain at least two nodes.")
    adjacency.eliminate_zeros()
    if adjacency.diagonal().any():
        raise ValueError("The original graph must not contain self-loops.")
    if adjacency.nnz and not np.allclose(adjacency.data, 1.0):
        raise ValueError("This implementation requires an unweighted graph.")
    difference = adjacency - adjacency.transpose()
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError("The adjacency matrix must be symmetric.")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degrees <= 0):
        raise ValueError("The graph must not contain isolated nodes.")
    component_count, _ = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    if component_count != 1:
        raise ValueError("The graph must be connected.")
    return adjacency


def _upper_triangle_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    tri_i, tri_j = np.triu_indices(n, k=1)
    index_type = np.int32 if n <= np.iinfo(np.int32).max else np.int64
    return tri_i.astype(index_type), tri_j.astype(index_type)


def _vector_to_symmetric_matrix(
    values: np.ndarray,
    n: int,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    matrix[tri_i, tri_j] = values
    matrix[tri_j, tri_i] = values
    return matrix


def solve_im_coalescence_times(
    q: sp.spmatrix | np.ndarray,
    *,
    solver: str = "bicgstab",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    maxiter: int = 4000,
    restart: int = 50,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Solve asynchronous two-lineage coalescence times for ancestry walk Q.

    For i != j, the solved recurrence is

        tau_ij = 1 + 1/2 sum_l (Q_il tau_lj + Q_jl tau_il),

    with tau_ii = 0. Only the upper triangle is used as Krylov unknowns.
    """
    q = sp.csr_matrix(q, dtype=float)
    n = q.shape[0]
    if q.shape != (n, n):
        raise ValueError("Q must be square.")
    row_sums = np.asarray(q.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Q must be row-stochastic.")

    tri_i, tri_j = _upper_triangle_indices(n)
    rhs = np.ones(tri_i.size, dtype=float)
    iterations = 0
    last_gmres_residual = np.nan

    def matvec(values: np.ndarray) -> np.ndarray:
        tau = _vector_to_symmetric_matrix(values, n, tri_i, tri_j)
        moved = q @ tau
        return values - 0.5 * (
            moved[tri_i, tri_j] + moved[tri_j, tri_i]
        )

    def krylov_callback(_: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1

    def gmres_callback(value: float) -> None:
        nonlocal iterations, last_gmres_residual
        iterations += 1
        last_gmres_residual = float(value)

    operator = LinearOperator(
        shape=(rhs.size, rhs.size),
        matvec=matvec,
        dtype=float,
    )
    started = time.perf_counter()
    if solver == "bicgstab":
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
    elif solver == "gmres":
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
    else:
        raise ValueError("solver must be 'bicgstab' or 'gmres'.")
    elapsed = time.perf_counter() - started
    if info != 0:
        raise RuntimeError(f"{solver} did not converge, info={info}.")

    residual = operator.matvec(solution) - rhs
    rhs_norm = np.linalg.norm(rhs)
    relative_residual = (
        float(np.linalg.norm(residual) / rhs_norm) if rhs_norm else 0.0
    )
    tau = _vector_to_symmetric_matrix(solution, n, tri_i, tri_j)
    stats: dict[str, float | int | str] = {
        "solver": solver,
        "pair_count": int(rhs.size),
        "iterations": int(iterations),
        "solve_time_sec": float(elapsed),
        "relative_residual": relative_residual,
    }
    if solver == "gmres":
        stats["last_preconditioned_residual"] = last_gmres_residual
    if verbose:
        print(
            "IM coalescence solved",
            f"n={n}",
            f"pairs={rhs.size}",
            f"solver={solver}",
            f"iterations={iterations}",
            f"residual={relative_residual:.3e}",
            f"time={elapsed:.3f}s",
            flush=True,
        )
    return tau, stats

