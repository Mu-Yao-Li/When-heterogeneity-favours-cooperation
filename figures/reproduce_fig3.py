"""Reproduce manuscript Figure 3 from the bundled, frozen source data.

Run from any working directory; see figures/README.md for provenance and scope.
The plotting functions preserve the manuscript plotting workflow.
"""
from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm as TwoSlopeNorm
from pathlib import Path
import argparse
import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "fig3"
AXIS_LABEL_SIZE = 21
TICK_LABEL_SIZE = 16.8
TITLE_SIZE = 21
PANEL_LABEL_SIZE = 24
LEGEND_SIZE = 21
ANNOTATION_SIZE = 21
SCHEMATIC_LABEL_SIZE = 21
SCHEMATIC_TICK_SIZE = 16.8
NETWORK_LABEL_SIZE = 21
COLORBAR_TITLE_SIZE = 21
COLORBAR_TICK_SIZE = 21
FIGSIZE = (15.1, 7.8)
COLOR_HETEROGENEITY = "#6E6E6E"
COLOR_HETEROGENEITY_FILL = "#B8B8B8"
HETEROGENEITY_GAMMA_MIN = 0.5
NETWORK_PANEL_SAMPLES = {0.5: 52, 1.0: 26, 1.5: 58}
COLOR_AXIS = "#333333"
NETWORK_CMAP = LinearSegmentedColormap(
    "network_degree",
    {
        "red": np.array(
            [
                [0.0, 0.48235294, 0.48235294],
                [0.25, 0.85098039, 0.85098039],
                [0.5, 0.90588235, 0.90588235],
                [0.75, 0.83529412, 0.83529412],
                [1.0, 0.49803922, 0.49803922],
            ]
        ),
        "green": np.array(
            [
                [0.0, 0.65490196, 0.65490196],
                [0.25, 0.8627451, 0.8627451],
                [0.5, 0.71372549, 0.71372549],
                [0.75, 0.36862745, 0.36862745],
                [1.0, 0.15294118, 0.15294118],
            ]
        ),
        "blue": np.array(
            [
                [0.0, 0.70196078, 0.70196078],
                [0.25, 0.77254902, 0.77254902],
                [0.5, 0.35294118, 0.35294118],
                [0.75, 0.0, 0.0],
                [1.0, 0.01568627, 0.01568627],
            ]
        ),
        "alpha": np.array(
            [[0.0, 1.0, 1.0], [0.25, 1.0, 1.0], [0.5, 1.0, 1.0], [0.75, 1.0, 1.0], [1.0, 1.0, 1.0]]
        ),
    },
    N=256,
    gamma=1.0,
)
NETWORK_DEGREE_COLOR_REF = 58.0
NETWORK_HUB_DEGREE = 18.0
NETWORK_PANEL_N = 200
K_FIXED = 4.0
N_FIXED = 2000
COLOR_ZERO_CONTOUR = "#000000"
PHASE_CMAP = LinearSegmentedColormap(
    "pnas_negative_neutral_positive_blue",
    {
        "red": np.array(
            [
                [0.0, 0.78823529, 0.78823529],
                [0.5, 0.96862745, 0.96862745],
                [1.0, 0.43529412, 0.43529412],
            ]
        ),
        "green": np.array(
            [
                [0.0, 0.74509804, 0.74509804],
                [0.5, 0.96470588, 0.96470588],
                [1.0, 0.65490196, 0.65490196],
            ]
        ),
        "blue": np.array(
            [
                [0.0, 0.70588235, 0.70588235],
                [0.5, 0.94901961, 0.94901961],
                [1.0, 0.78431373, 0.78431373],
            ]
        ),
        "alpha": np.array([[0.0, 1.0, 1.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    },
    N=256,
    gamma=1.0,
)
PHASE_GAMMAS = [
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.71,
    0.72,
    0.73,
    0.74,
    0.75,
    0.76,
    0.77,
    0.78,
    0.79,
    0.8,
    0.81,
    0.82,
    0.83,
    0.84,
    0.85,
    0.86,
    0.87,
    0.88,
    0.89,
    0.9,
    0.91,
    0.92,
    0.93,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    1.0,
    1.01,
    1.02,
    1.03,
    1.04,
    1.05,
    1.06,
    1.07,
    1.08,
    1.09,
    1.1,
    1.11,
    1.12,
    1.13,
    1.14,
    1.15,
    1.16,
    1.17,
    1.18,
    1.19,
    1.2,
    1.21,
    1.22,
    1.23,
    1.24,
    1.25,
    1.3,
    1.35,
    1.4,
    1.45,
    1.5,
]
COLOR_EPSILON = "#8172B3"
COLOR_EPSILON_CRIT = "#DD8452"
COLOR_REFERENCE = "#999999"
GAMMA_WINDOW_COLOR = "#3B6FB0"
BACKGROUND = "#E0E0E0"
WINDOW = "#8DB3D7"


def apply_compact_style() -> None:
    AXIS_LABEL_SIZE = 21
    TICK_LABEL_SIZE = 16.8
    TITLE_SIZE = 21
    PANEL_LABEL_SIZE = 24
    LEGEND_SIZE = 21
    ANNOTATION_SIZE = 21
    SCHEMATIC_LABEL_SIZE = 21
    SCHEMATIC_TICK_SIZE = 16.8
    NETWORK_LABEL_SIZE = 21
    COLORBAR_TITLE_SIZE = 21
    COLORBAR_TICK_SIZE = 21
    plt.rcParams.update(
        {"font.size": 16.8, "axes.labelsize": 21, "xtick.labelsize": 16.8, "ytick.labelsize": 16.8}
    )


def apply_requested_font_style() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "mathtext.fontset": "cm"})


def add_axes_inch(fig: plt.Figure, *, x: float, y: float, w: float, h: float) -> plt.Axes:
    (fig_w, fig_h) = FIGSIZE
    return fig.add_axes([x / fig_w, y / fig_h, w / fig_w, h / fig_h])


def apply_cpt_axis(ax: plt.Axes, *, tick_size: int = TICK_LABEL_SIZE) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor(COLOR_AXIS)
    ax.tick_params(
        width=0.8,
        length=6,
        labelsize=tick_size,
        direction="out",
        color=COLOR_AXIS,
        labelcolor="black",
    )


def complete_mask(data: pd.DataFrame) -> pd.Series:
    if "complete20" in data.columns:
        return data["complete20"].astype(bool)
    if "complete10" in data.columns:
        return data["complete10"].astype(bool)
    return pd.Series(True, index=data.index)


def presentation_layout(graph, seed, gamma):
    nodes = pd.read_csv(DATA_DIR / f"network_gamma{gamma:g}_nodes.csv")
    return {int(row.node): np.array([row.x, row.y]) for row in nodes.itertuples(index=False)}


def representative_graph(gamma, sample):
    nodes = pd.read_csv(DATA_DIR / f"network_gamma{gamma:g}_nodes.csv")
    edges = pd.read_csv(DATA_DIR / f"network_gamma{gamma:g}_edges.csv")
    graph = nx.Graph()
    graph.add_nodes_from(nodes["node"].astype(int))
    graph.add_edges_from(edges.itertuples(index=False, name=None))
    return graph


def server_seed(gamma: float, sample: int, n: int = N_FIXED) -> int:
    gamma_part = int(round(gamma * 10000))
    return (
        20260428
        + sample * 1000003
        + n * 17
        + int(round(K_FIXED * 1000)) * 31
        + gamma_part * 101
        + 11
    )


def draw_network_snapshot(
    ax: plt.Axes, *, gamma: float, sample: int, label: str, size_scale: float = 0.46
) -> None:
    graph = representative_graph(gamma, sample)
    pos = presentation_layout(graph, server_seed(gamma, sample, NETWORK_PANEL_N), gamma)
    nodes = np.array(list(graph.nodes()))
    degree = np.array([graph.degree(int(node)) for node in nodes], dtype=float)
    degree_by_node = {int(node): float(deg) for (node, deg) in zip(nodes, degree)}
    node_xy = np.array([pos[int(node)] for node in nodes])
    edges = list(graph.edges())
    edge_segments = np.array([[pos[int(u)], pos[int(v)]] for (u, v) in edges])
    max_degree = max(float(degree.max()), 1.0)
    degree_scale = np.clip(degree / NETWORK_DEGREE_COLOR_REF, 0.0, 1.0)
    draw_order = np.argsort(degree)
    rank_order = np.argsort(-degree)
    top_ratio = degree[rank_order[0]] / max(degree[rank_order[1]], 1.0) if len(nodes) > 1 else 1.0
    if max_degree < NETWORK_HUB_DEGREE:
        hub_count = 0
    elif top_ratio >= 3.2:
        hub_count = 1
    else:
        hub_count = int(np.count_nonzero(degree[rank_order[:6]] >= NETWORK_HUB_DEGREE))
    hub_nodes = set((int(nodes[i]) for i in rank_order[:hub_count]))
    hub_rank_weight = np.zeros(len(nodes), dtype=float)
    if hub_count:
        hub_rank_weight[rank_order[:hub_count]] = np.linspace(1.0, 0.36, hub_count)
    color_scale = 0.05 + 0.91 * degree_scale**0.64
    node_sizes = (
        8.5
        + 8.6 * np.sqrt(np.maximum(degree, 0.0))
        + 0.019 * degree**2
        + 58.0 * hub_rank_weight**2
    ) * size_scale
    edge_strength = np.array(
        [
            min(max(degree_by_node[int(u)], degree_by_node[int(v)]) / NETWORK_DEGREE_COLOR_REF, 1.0)
            for (u, v) in edges
        ],
        dtype=float,
    )
    hub_edge_mask = np.array(
        [int(u) in hub_nodes or int(v) in hub_nodes for (u, v) in edges], dtype=bool
    )
    ax.set_facecolor((1, 1, 1, 0))
    ax.patch.set_alpha(0.0)
    ax.add_collection(
        LineCollection(edge_segments, colors=(0.2, 0.24, 0.25, 0.06), linewidths=0.22, zorder=1)
    )
    if hub_edge_mask.any():
        hub_edge_colors = NETWORK_CMAP(0.18 + 0.82 * edge_strength[hub_edge_mask])
        hub_edge_colors[:, 3] = 0.1 + 0.26 * edge_strength[hub_edge_mask] ** 0.75
        ax.add_collection(
            LineCollection(
                edge_segments[hub_edge_mask],
                colors=hub_edge_colors,
                linewidths=0.25 + 0.6 * edge_strength[hub_edge_mask] ** 0.85,
                zorder=2,
            )
        )
    ax.scatter(
        node_xy[draw_order, 0],
        node_xy[draw_order, 1],
        s=node_sizes[draw_order],
        c=color_scale[draw_order],
        cmap=NETWORK_CMAP,
        vmin=0,
        vmax=1,
        linewidths=0.12,
        edgecolors=(1.0, 1.0, 1.0, 0.68),
        alpha=0.95,
        zorder=3,
    )
    if hub_count:
        hub_idx = rank_order[:hub_count]
        ax.scatter(
            node_xy[hub_idx, 0],
            node_xy[hub_idx, 1],
            s=node_sizes[hub_idx] * 1.34,
            facecolors="white",
            edgecolors="none",
            alpha=0.92,
            zorder=4,
        )
        ax.scatter(
            node_xy[hub_idx, 0],
            node_xy[hub_idx, 1],
            s=node_sizes[hub_idx] * 1.08,
            c=color_scale[hub_idx],
            cmap=NETWORK_CMAP,
            vmin=0,
            vmax=1,
            edgecolors="#6F2A00",
            linewidths=0.68,
            alpha=0.98,
            zorder=5,
        )
    ax.text(
        0.52,
        0.96,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=NETWORK_LABEL_SIZE,
    )
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.06, 1.06)
    ax.set_aspect("equal")
    ax.set_axis_off()


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(width=1.5, length=7, labelsize=20)


def draw_combined_ab_panel(ax: plt.Axes, data: pd.DataFrame) -> None:
    mask = complete_mask(data)
    complete = data[mask].copy()
    complete = complete[complete["gamma"].between(HETEROGENEITY_GAMMA_MIN, 1.5)].copy()
    gamma = complete["gamma"].to_numpy(dtype=float)
    cv_mean = complete["degree_cv_mean"].to_numpy(dtype=float)
    cv_std = complete["degree_cv_std"].fillna(0.0).to_numpy(dtype=float)
    positive_floor = max(float(cv_mean[cv_mean > 0].min()) * 0.65, 1e-06)
    cv_lower = np.maximum(cv_mean - cv_std, positive_floor)
    cv_upper = cv_mean + cv_std
    ax.fill_betweenx(
        gamma, cv_lower, cv_upper, color=COLOR_HETEROGENEITY_FILL, alpha=0.42, linewidth=0, zorder=1
    )
    ax.plot(cv_mean, gamma, color=COLOR_HETEROGENEITY, linewidth=3.0, zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(0.65, 14.0)
    ax.set_ylim(1.5, 0.5)
    ax.set_xticks([1, 3, 10])
    ax.set_xticklabels(["1", "3", "10"])
    ax.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_yticklabels(["0.5", "", "1.0", "", "1.5"])
    ax.set_xlabel("Degree CV, $\\sigma(k_i)/k$", fontsize=SCHEMATIC_LABEL_SIZE, labelpad=4)
    ax.set_ylabel("Attachment nonlinearity, $\\gamma$", fontsize=SCHEMATIC_LABEL_SIZE, labelpad=12)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    style_axes(ax)
    apply_cpt_axis(ax, tick_size=SCHEMATIC_TICK_SIZE)
    ax.tick_params(pad=2)
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
        tick.set_ha("right")
        tick.set_va("center")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    snapshots = [
        (0.5, [0.14, 0.69, 0.82, 0.285]),
        (1.0, [0.14, 0.36, 0.82, 0.285]),
        (1.5, [0.14, 0.035, 0.82, 0.285]),
    ]
    for gamma_value, rect in snapshots:
        sample = NETWORK_PANEL_SAMPLES[gamma_value]
        inset = ax.inset_axes(rect)
        draw_network_snapshot(
            inset,
            gamma=gamma_value,
            sample=sample,
            label=f"$\\gamma={gamma_value:.1f}$",
            size_scale=0.58,
        )


def centered_edges(values: np.ndarray, *, log: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        width = values[0] * 0.08 if log else 0.05
        return np.array([values[0] - width, values[0] + width])
    work = np.log(values) if log else values
    mids = (work[:-1] + work[1:]) / 2.0
    first = work[0] - (mids[0] - work[0])
    last = work[-1] + (work[-1] - mids[-1])
    edges = np.r_[first, mids, last]
    return np.exp(edges) if log else edges


def interp_limited(
    x_known: np.ndarray, y_known: np.ndarray, x_new: np.ndarray, *, max_gap: float
) -> np.ndarray:
    known = np.isfinite(x_known) & np.isfinite(y_known)
    x = np.asarray(x_known[known], dtype=float)
    y = np.asarray(y_known[known], dtype=float)
    out = np.full_like(x_new, np.nan, dtype=float)
    if x.size < 2:
        return out
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    (unique_x, inverse) = np.unique(x, return_inverse=True)
    if unique_x.size != x.size:
        unique_y = np.array([y[inverse == idx].mean() for idx in range(unique_x.size)])
        x = unique_x
        y = unique_y
    for idx in range(x.size - 1):
        x0 = x[idx]
        x1 = x[idx + 1]
        gap = x1 - x0
        if gap <= 0 or gap > max_gap:
            continue
        mask = (x_new >= x0) & (x_new <= x1)
        out[mask] = y[idx] + (y[idx + 1] - y[idx]) * (x_new[mask] - x0) / gap
    return out


def gaussian_kernel1d(sigma: float, *, radius: int | None = None) -> np.ndarray:
    radius = radius or int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def smooth_2d(values: np.ndarray, *, sigma_y: float = 2.0, sigma_x: float = 2.0) -> np.ndarray:
    work = np.asarray(values, dtype=float)
    finite = np.isfinite(work)
    if not finite.all():
        filled = np.where(finite, work, 0.0)
        weights = finite.astype(float)
        smooth_values = smooth_2d(filled, sigma_y=sigma_y, sigma_x=sigma_x)
        smooth_weights = smooth_2d(weights, sigma_y=sigma_y, sigma_x=sigma_x)
        out = np.full_like(smooth_values, np.nan, dtype=float)
        valid = smooth_weights > 0.18
        out[valid] = smooth_values[valid] / smooth_weights[valid]
        return out
    kernel_x = gaussian_kernel1d(sigma_x)
    kernel_y = gaussian_kernel1d(sigma_y)
    pad_x = len(kernel_x) // 2
    pad_y = len(kernel_y) // 2
    x_smooth = np.apply_along_axis(
        lambda row: np.convolve(np.pad(row, pad_x, mode="edge"), kernel_x, mode="valid"),
        axis=1,
        arr=work,
    )
    return np.apply_along_axis(
        lambda col: np.convolve(np.pad(col, pad_y, mode="edge"), kernel_y, mode="valid"),
        axis=0,
        arr=x_smooth,
    )


def densify_phase_surface(
    gammas: np.ndarray,
    n_values: np.ndarray,
    z: np.ndarray,
    *,
    gamma_count: int = 260,
    n_count: int = 420,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_n = np.log10(n_values)
    gamma_dense = np.linspace(float(gammas.min()), float(gammas.max()), gamma_count)
    log_n_dense = np.linspace(float(log_n.min()), float(log_n.max()), n_count)
    by_gamma = np.empty((gamma_count, len(n_values)), dtype=float)
    for col_idx in range(len(n_values)):
        column = z[:, col_idx]
        by_gamma[:, col_idx] = interp_limited(gammas, column, gamma_dense, max_gap=0.061)
    dense = np.empty((gamma_count, n_count), dtype=float)
    for row_idx in range(gamma_count):
        dense[row_idx, :] = interp_limited(log_n, by_gamma[row_idx, :], log_n_dense, max_gap=0.2)
    return (gamma_dense, np.power(10.0, log_n_dense), smooth_2d(dense, sigma_y=2.2, sigma_x=2.8))


def smooth_branch(n_values: np.ndarray, gamma_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(n_values)
    log_n = np.log10(n_values[order])
    gamma = gamma_values[order]
    dense_log_n = np.linspace(float(log_n.min()), float(log_n.max()), 320)
    dense_gamma = np.interp(dense_log_n, log_n, gamma)
    kernel = gaussian_kernel1d(5.0)
    pad = len(kernel) // 2
    dense_gamma = np.convolve(np.pad(dense_gamma, pad, mode="edge"), kernel, mode="valid")
    dense_gamma[0] = gamma[0]
    dense_gamma[-1] = gamma[-1]
    return (np.power(10.0, dense_log_n), dense_gamma)


def phase_zero_branches(
    gammas: np.ndarray, n_values: np.ndarray, z: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    lower_n = []
    lower_gamma = []
    upper_n = []
    upper_gamma = []
    previous_n = None
    previous_max = None
    left_cusp = None
    for col_idx, n_value in enumerate(n_values):
        values = z[:, col_idx]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        column_max = float(np.nanmax(values))
        crossings = []
        for idx in range(len(gammas) - 1):
            y0 = values[idx]
            y1 = values[idx + 1]
            if not np.isfinite(y0) or not np.isfinite(y1):
                continue
            if y0 == 0:
                crossings.append(float(gammas[idx]))
            elif y0 * y1 < 0:
                frac = abs(y0) / (abs(y0) + abs(y1))
                crossings.append(float(gammas[idx] + frac * (gammas[idx + 1] - gammas[idx])))
        if len(crossings) >= 2:
            if left_cusp is None:
                cusp_n = float(n_value)
                if (
                    previous_n is not None
                    and previous_max is not None
                    and (previous_max < 0.0 < column_max)
                ):
                    weight = (0.0 - previous_max) / (column_max - previous_max)
                    log_cusp_n = np.log10(previous_n) + weight * (
                        np.log10(n_value) - np.log10(previous_n)
                    )
                    cusp_n = float(10.0**log_cusp_n)
                left_cusp = (cusp_n, 0.5 * (crossings[0] + crossings[-1]))
            lower_n.append(float(n_value))
            lower_gamma.append(crossings[0])
            upper_n.append(float(n_value))
            upper_gamma.append(crossings[-1])
        else:
            previous_n = float(n_value)
            previous_max = column_max
    branches = []
    if len(lower_n) >= 4:
        branch_n = lower_n
        branch_gamma = lower_gamma
        if left_cusp is not None and left_cusp[0] < lower_n[0]:
            branch_n = [left_cusp[0], *lower_n]
            branch_gamma = [left_cusp[1], *lower_gamma]
        branches.append(smooth_branch(np.asarray(branch_n), np.asarray(branch_gamma)))
    if len(upper_n) >= 4:
        branch_n = upper_n
        branch_gamma = upper_gamma
        if left_cusp is not None and left_cusp[0] < upper_n[0]:
            branch_n = [left_cusp[0], *upper_n]
            branch_gamma = [left_cusp[1], *upper_gamma]
        branches.append(smooth_branch(np.asarray(branch_n), np.asarray(branch_gamma)))
    return branches


def draw_phase_panel_N_gamma(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    cbar_ax: plt.Axes | None = None,
    show_colorbar: bool = True,
    colorbar_inside: bool = False,
) -> None:
    used = data[data["used_for_phase"]].copy()
    gammas = np.array(PHASE_GAMMAS, dtype=float)
    n_values = np.array(sorted(used["N"].unique()), dtype=float)
    pivot = used.pivot(index="gamma", columns="N", values="rr_advantage_mean").reindex(
        index=gammas, columns=n_values
    )
    z_raw = np.asarray(pivot.to_numpy(dtype=float))
    (gamma_plot, n_plot, z_plot) = densify_phase_surface(
        gammas, n_values, z_raw, gamma_count=320, n_count=520
    )
    z = np.ma.masked_invalid(z_plot)
    n_edges = centered_edges(n_plot, log=True)
    gamma_edges = centered_edges(gamma_plot)
    norm = TwoSlopeNorm(vmin=-0.08, vcenter=0.0, vmax=0.04)
    mesh = ax.pcolormesh(
        n_edges,
        gamma_edges,
        z,
        cmap=PHASE_CMAP,
        norm=norm,
        shading="auto",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
    )
    mesh.set_edgecolor("face")
    for branch_n, branch_gamma in phase_zero_branches(gammas, n_values, z_raw):
        ax.plot(
            branch_n,
            branch_gamma,
            color=COLOR_ZERO_CONTOUR,
            linewidth=3.0,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=6,
        )
    ax.set_xscale("log")
    ax.set_xlim(100, 10000)
    ax.set_ylim(1.5, 0.5)
    x_ticks = [100, 1000, 10000]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["$10^2$", "$10^3$", "$10^4$"])
    ax.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_yticklabels(["0.5", "", "1.0", "", "1.5"])
    ax.set_xlabel("Population size, $N$", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Attachment nonlinearity, $\\gamma$", fontsize=AXIS_LABEL_SIZE, labelpad=10)
    ax.yaxis.set_label_coords(-0.09, 0.5)
    style_axes(ax)
    apply_cpt_axis(ax)
    if show_colorbar:
        if colorbar_inside and cbar_ax is None:
            cbar_ax = ax.inset_axes([0.55, 0.8, 0.38, 0.06])
            cbar_orientation = "horizontal"
            cbar_fraction = None
            cbar_pad = None
        else:
            cbar_orientation = "vertical"
            cbar_fraction = 0.032
            cbar_pad = 0.024
        cbar = ax.figure.colorbar(
            mesh,
            ax=ax if cbar_ax is None and (not colorbar_inside) else None,
            cax=cbar_ax,
            fraction=cbar_fraction,
            pad=cbar_pad,
            extend="both",
            orientation=cbar_orientation,
        )
        if colorbar_inside:
            cbar.set_ticks([-0.08, 0.0, 0.04])
            cbar.ax.set_title("$(b/c)^*_{r}-(b/c)^*$", fontsize=COLORBAR_TITLE_SIZE, pad=17)
            cbar.ax.tick_params(
                labelsize=COLORBAR_TICK_SIZE,
                width=0.8,
                length=5.0,
                pad=2.5,
                direction="out",
                color=COLOR_AXIS,
            )
        else:
            cbar.ax.set_title(
                "$(b/c)^*_{r}$\n$-(b/c)^*$", fontsize=COLORBAR_TITLE_SIZE, pad=6, linespacing=0.92
            )
            cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, width=0.8, length=4, direction="out")
        if cbar.solids is not None:
            cbar.solids.set_edgecolor("face")
            cbar.solids.set_linewidth(0.0)
            cbar.solids.set_antialiased(False)
            cbar.solids.set_rasterized(True)
        for patch in cbar.ax.patches:
            patch.set_edgecolor(patch.get_facecolor())
            patch.set_linewidth(0.0)
            patch.set_antialiased(False)
        cbar.outline.set_linewidth(0.8)
        cbar.outline.set_edgecolor(COLOR_AXIS)


def retune_phase_panel_b(ax: plt.Axes) -> None:
    for line in ax.lines:
        line.set_color("black")
        line.set_linewidth(1.6)
        line.set_solid_capstyle("round")
        line.set_solid_joinstyle("round")
    ax.tick_params(axis="x", labelbottom=True, labeltop=False, pad=4, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontsize(TICK_LABEL_SIZE)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position("bottom")


def _p0_apply_nature_small_axis(ax: plt.Axes, *, tick_size: int = TICK_LABEL_SIZE) -> None:
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_edgecolor(COLOR_AXIS)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(
        axis="both",
        which="major",
        width=1.1,
        length=5.0,
        pad=3.0,
        labelsize=tick_size,
        direction="out",
        color=COLOR_AXIS,
        labelcolor="black",
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=0.8,
        length=3.0,
        direction="out",
        color=COLOR_AXIS,
        top=False,
        right=False,
    )


def sign_crossings(x: np.ndarray, y: np.ndarray, *, level: float = 0.0) -> list[float]:
    values = y - level
    crossings: list[float] = []
    for idx in range(len(x) - 1):
        y0 = values[idx]
        y1 = values[idx + 1]
        if not np.isfinite(y0) or not np.isfinite(y1):
            continue
        if y0 == 0:
            crossings.append(float(x[idx]))
        elif y0 * y1 < 0:
            frac = abs(y0) / (abs(y0) + abs(y1))
            crossings.append(float(x[idx] + frac * (x[idx + 1] - x[idx])))
    if len(x) and values[-1] == 0:
        crossings.append(float(x[-1]))
    return crossings


def signed_log(values: np.ndarray, *, linthresh: float = 0.00025) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log10(1.0 + np.abs(values) / linthresh)


def draw_mechanism_explanation_panel(ax: plt.Axes, data) -> None:
    data = data.sort_values("gamma").copy()
    gamma = data["gamma"].to_numpy(dtype=float)
    epsilon = data["epsilon_mean"].to_numpy(dtype=float)
    delta = data["epsilon_crit_mean"].to_numpy(dtype=float)
    margin = epsilon - delta
    epsilon_plot = signed_log(epsilon)
    delta_plot = signed_log(delta)
    ax.axhline(0.0, color=COLOR_REFERENCE, linewidth=0.9, linestyle=(0, (2.0, 2.0)), zorder=1)
    ax.fill_between(
        gamma,
        epsilon_plot,
        delta_plot,
        where=margin >= 0.0,
        interpolate=True,
        color=GAMMA_WINDOW_COLOR,
        alpha=0.2,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        gamma, epsilon_plot, color=COLOR_EPSILON, linewidth=2.2, zorder=3, label="$\\varepsilon$"
    )
    ax.plot(gamma, delta_plot, color=COLOR_EPSILON_CRIT, linewidth=2.2, zorder=3, label="$\\Delta$")
    for crossing_gamma in sign_crossings(gamma, margin):
        ax.axvline(
            crossing_gamma,
            color=COLOR_REFERENCE,
            linewidth=0.8,
            linestyle=(0, (2.0, 2.0)),
            alpha=0.7,
            zorder=2,
        )
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(-1.15, max(float(np.nanmax(delta_plot)), float(np.nanmax(epsilon_plot))) * 1.1)
    ax.set_xticks([0.5, 1.0, 1.5])
    ytick_values = np.array([-0.001, 0.0, 0.001, 0.01, 0.1])
    ax.set_yticks(signed_log(ytick_values))
    ax.set_yticklabels(["$-10^{-3}$", "$0$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax.set_xlabel("Attachment nonlinearity, $\\gamma$", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_ylabel("Mechanistic balance", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    _p0_apply_nature_small_axis(ax, tick_size=TICK_LABEL_SIZE)
    ax.yaxis.set_label_coords(-0.28, 0.5)
    ax.legend(
        frameon=False, fontsize=LEGEND_SIZE, loc="upper left", handlelength=1.8, borderaxespad=0.2
    )


def _p1_apply_nature_small_axis(ax: plt.Axes, *, tick_size: int = TICK_LABEL_SIZE) -> None:
    ax.grid(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_edgecolor(COLOR_AXIS)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(
        axis="both",
        which="major",
        width=1.1,
        length=5.0,
        pad=3.0,
        labelsize=tick_size,
        direction="out",
        color=COLOR_AXIS,
        labelcolor="black",
        top=False,
        right=False,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        width=0.8,
        length=3.0,
        direction="out",
        color=COLOR_AXIS,
        top=False,
        right=False,
    )


def draw_n1e7_window_panel(ax: plt.Axes, windows: pd.DataFrame, k_values: list[int]) -> None:
    selected = windows.set_index("k").loc[k_values].reset_index()
    y_positions = np.arange(len(selected), dtype=float)
    for y, row in zip(y_positions, selected.itertuples(index=False)):
        ax.plot(
            [0.5, 1.5], [y, y], color=BACKGROUND, linewidth=13.0, solid_capstyle="butt", zorder=1
        )
        ax.plot(
            [row.gamma_left, row.gamma_right],
            [y, y],
            color=WINDOW,
            linewidth=13.0,
            solid_capstyle="butt",
            zorder=2,
        )
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(-0.65, len(selected) - 0.35)
    ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_xticklabels(["0.5", "", "1.0", "", "1.5"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(k) for k in selected["k"]])
    ax.set_xlabel("Attachment nonlinearity, $\\gamma$", fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_ylabel("Mean degree, $k$", fontsize=AXIS_LABEL_SIZE, labelpad=5)
    style_axes(ax)
    _p1_apply_nature_small_axis(ax, tick_size=TICK_LABEL_SIZE)


def panel_label(
    fig: plt.Figure,
    ax: plt.Axes,
    label: str,
    *,
    x: float | None = None,
    y: float | None = None,
    dx: float = 0.024,
    dy: float = 0.02,
) -> None:
    bbox = ax.get_position()
    fig.text(
        bbox.x0 - dx if x is None else x,
        bbox.y1 + dy if y is None else y,
        label,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce manuscript Figure 3 from frozen source data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_compact_style()
    apply_requested_font_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    n2000 = pd.read_csv(DATA_DIR / "heterogeneity.csv")
    n10000 = pd.read_csv(DATA_DIR / "mechanism.csv")
    phase = pd.read_csv(DATA_DIR / "phase.csv.gz", float_precision="round_trip")
    windows = pd.read_csv(DATA_DIR / "windows.csv")
    fig = plt.figure(figsize=(15.1, 7.8))
    ax_a = add_axes_inch(fig, x=0.78, y=0.82, w=2.65, h=6.43)
    ax_b = add_axes_inch(fig, x=4.38, y=0.82, w=6.43, h=6.43)
    ax_c = add_axes_inch(fig, x=11.96, y=4.53, w=2.72, h=2.72)
    ax_d = add_axes_inch(fig, x=11.96, y=0.82, w=2.72, h=2.72)
    draw_combined_ab_panel(ax_a, n2000)
    draw_phase_panel_N_gamma(ax_b, phase, show_colorbar=True, colorbar_inside=True)
    retune_phase_panel_b(ax_b)
    draw_mechanism_explanation_panel(ax_c, n10000)
    draw_n1e7_window_panel(ax_d, windows, [4, 6, 8, 10, 20, 30])
    for ax, label, dx, dy in [
        (ax_a, "a", 0.038, 0.06),
        (ax_b, "b", 0.044, 0.06),
        (ax_c, "c", 0.04, 0.06),
        (ax_d, "d", 0.04, 0.018),
    ]:
        panel_label(fig, ax, label, dx=dx, dy=dy)
    stem = args.output_dir / "fig3"
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
