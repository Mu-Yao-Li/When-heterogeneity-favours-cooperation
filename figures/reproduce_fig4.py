"""Reproduce manuscript Figure 4 from the bundled, frozen source data.

Run from any working directory; see figures/README.md for provenance and scope.
The plotting functions preserve the manuscript plotting workflow.
"""
from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as _p0_LineCollection
from matplotlib.collections import LineCollection as _p1_LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle as Rectangle
from matplotlib.transforms import Bbox
from pathlib import Path
from scipy import sparse
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.stats as stats

DATA_DIR = Path(__file__).resolve().parent / "data" / "fig4"
BASELINE_THRESHOLD_BEFORE = 4.031390985209702
BASELINE_THRESHOLD_RR = 4.024193548387097
_p1_DARK = "#20262E"
_p1_FIG4_FONT_PT = 10.0
FIG_W_MM = 183.0
MODULE_N = 1000
PANEL_A_CONTENT_SHIFT_MM = -8.0
PANEL_A_ENHANCED_DEGREE_DRAW = True
PANEL_A_HORIZONTAL_TEXT_BELOW = True
PANEL_A_LABEL_X = 0.0
PANEL_A_LABEL_Y = 1.08
PANEL_A_LAYOUT_MODE = "horizontal"
PANEL_A_LOWER_NETWORK_SHIFT_MM = 0.0
PANEL_A_NETWORK_SIZE_Y = 0.255
PANEL_A_NETWORK_VERTICAL_SHIFT_MM = 4.0
PANEL_A_NETWORK_ZOOM = 2.5056
PANEL_A_NODE_EDGECOLOR = "none"
PANEL_A_NODE_EDGEWIDTH = 0.0
PANEL_A_NODE_SIZE_EXPONENT = 1.0
PANEL_A_NODE_SIZE_MIN = 2.8
PANEL_A_NODE_SIZE_SCALE = 62.0
PANEL_A_SHOW_NETWORK_SIZE = False
PANEL_A_TEXT_X = 0.005
PANEL_A_THRESHOLD_BEFORE_Y = 0.2
PANEL_A_THRESHOLD_BEST_Y = 0.115
PANEL_A_THRESHOLD_LABEL_X = 0.13
PANEL_A_THRESHOLD_VALUE_X = 0.4
PANEL_A_UPPER_NETWORK_SHIFT_MM = 3.5
_p1_PANEL_SHELL_COLORS = {
    0: "#C8788A",
    1: "#D8BC70",
    2: "#6FB6A5",
    3: "#7899BC",
    4: "#7B68A6",
    5: "#6A4C93",
    6: "#7A8B95",
}
SEED_1 = 20261478
SEED_2 = 20261478
SHELL_LEVELS = [0, 1, 2, 3]
_p0_DARK = "#20262E"
_p0_FIG4_FONT_PT = 10.0
_p0_PANEL_SHELL_COLORS = {
    0: "#C8788A",
    1: "#D8BC70",
    2: "#6FB6A5",
    3: "#7899BC",
    4: "#7B68A6",
    5: "#6A4C93",
    6: "#7A8B95",
}
BEST_STAR_MARKER_AREA = 49.0
PANEL_B_LABEL_X = -0.16
PANEL_B_LABEL_Y = 1.07
RED = "#C0392B"
TEAL = "#0B7285"
UNIFIED_BC_BOXPLOT_STYLE = True
UNIFIED_BC_MEAN_MARKER_AREA = 18.0
UNIFIED_BC_MEAN_MARKER_COLOR = "#3F444A"
UNIFIED_BC_MEAN_MARKER_EDGE_VISIBLE = False
UNIFIED_BC_MEDIAN_LINEWIDTH = 0.7
UNIFIED_BC_WHISKER_LINEWIDTH = 0.7
GRAY = "#5C6670"
PANEL_C_SHOW_BRIDGE_LEGEND = False
PERIPHERAL_COLOR = "#7899BC"
PERIPHERAL_PAIR_LABEL = "shell3-shell3"
PERIPHERAL_PAIR_VALUE = "3-3"
PERIPHERAL_SHELL = 3
SHELL0_COLOR = "#C8788A"
UNIFIED_BC_BOX_EDGE_VISIBLE = False
PANEL_C_LABEL_X = -0.3125
SAME_SHELL_ZORDER_BASE = 3.0
DENSITY_BACKGROUND_ALPHA = 0.5
DENSITY_BACKGROUND_ZORDER = 0.5
DENSITY_CMAP = LinearSegmentedColormap(
    "bridge_density",
    {
        "red": np.array(
            [
                [0.0, 0.95686275, 0.95686275],
                [0.5, 0.78823529, 0.78823529],
                [1.0, 0.45490196, 0.45490196],
            ]
        ),
        "green": np.array(
            [
                [0.0, 0.96078431, 0.96078431],
                [0.5, 0.80784314, 0.80784314],
                [1.0, 0.49019608, 0.49019608],
            ]
        ),
        "blue": np.array(
            [
                [0.0, 0.96470588, 0.96470588],
                [0.5, 0.82745098, 0.82745098],
                [1.0, 0.5254902, 0.5254902],
            ]
        ),
        "alpha": np.array([[0.0, 1.0, 1.0], [0.5, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    },
    N=256,
    gamma=1.0,
)
THRESHOLD_TICKS = [3.998, 4.002, 4.006, 4.01]


def rotate_to_angle(xy: np.ndarray, node: int, target_angle: float) -> np.ndarray:
    current = float(np.arctan2(xy[node, 1], xy[node, 0]))
    theta = target_angle - current
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=float)
    out = xy @ rot.T
    out -= out.mean(axis=0, keepdims=True)
    return out


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.055,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=_p0_FIG4_FONT_PT,
        fontweight="bold",
        color=_p0_DARK,
        clip_on=False,
    )


def _p0_edge_segments(adjacency: sparse.csr_matrix, xy: np.ndarray) -> np.ndarray:
    coo = sparse.triu(adjacency, k=1).tocoo()
    return np.stack([xy[coo.row], xy[coo.col]], axis=1)


def draw_module(
    ax: plt.Axes,
    adjacency: sparse.csr_matrix,
    xy: np.ndarray,
    degree: np.ndarray,
    hubs: np.ndarray,
    shell: np.ndarray,
    label: str,
    show_label: bool = True,
) -> None:
    ax.add_collection(
        _p0_LineCollection(
            _p0_edge_segments(adjacency, xy),
            colors=(0.14, 0.16, 0.18, 0.055),
            linewidths=0.13,
            zorder=1,
        )
    )
    sizes = 4.5 + 34.0 * np.sqrt(degree / max(float(degree.max()), 1.0))
    for shell_level in sorted((int(value) for value in np.unique(shell))):
        nodes = np.flatnonzero(shell == shell_level)
        ax.scatter(
            xy[nodes, 0],
            xy[nodes, 1],
            s=sizes[nodes],
            c=_p0_PANEL_SHELL_COLORS.get(shell_level, "#B8B8B8"),
            alpha=0.86 if shell_level == 0 else 0.68,
            linewidths=0,
            zorder=2,
        )
    if show_label:
        ax.text(
            float(np.mean(xy[:, 0])),
            float(np.min(xy[:, 1]) - 0.12),
            label,
            ha="center",
            va="top",
            fontsize=10.0,
            color=_p0_DARK,
        )


def _p1_edge_segments(adjacency: sparse.csr_matrix, xy: np.ndarray) -> np.ndarray:
    coo = sparse.triu(adjacency, k=1).tocoo()
    return np.stack([xy[coo.row], xy[coo.col]], axis=1)


def module_geometry(seed, endpoint, target_angle, center_x):
    nodes = pd.read_csv(DATA_DIR / "module_nodes.csv")
    edges = pd.read_csv(DATA_DIR / "module_edges.csv")
    (u, v) = (edges["source"].to_numpy(int), edges["target"].to_numpy(int))
    adjacency = sparse.csr_matrix(
        (np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(len(nodes), len(nodes))
    )
    degree = nodes["degree"].to_numpy(float)
    hubs = nodes.loc[nodes["is_hub"].eq(1), "node"].to_numpy(int)
    shell = nodes["shell"].to_numpy(int)
    xy = rotate_to_angle(nodes[["x", "y"]].to_numpy(float), endpoint, target_angle)
    xy /= max(float(np.ptp(xy, axis=0).max()), 1e-12)
    xy *= 1.65
    xy[:, 0] += center_x
    return (adjacency, degree, hubs, shell, xy)


def draw_panel_a_slanted(ax: plt.Axes, best: pd.Series) -> None:
    if PANEL_A_LAYOUT_MODE not in {"slanted", "horizontal"}:
        raise ValueError(f"Unsupported panel-a layout: {PANEL_A_LAYOUT_MODE}")
    horizontal = PANEL_A_LAYOUT_MODE == "horizontal"
    if horizontal:
        ax.text(
            PANEL_A_LABEL_X,
            PANEL_A_LABEL_Y,
            "a",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_p1_FIG4_FONT_PT,
            fontweight="bold",
            color=_p1_DARK,
            clip_on=False,
        )
    else:
        add_panel_label(ax, "a")
    ax.set_axis_off()
    ax.set_aspect("equal")
    u = int(best["local_u"])
    v = int(best["local_v"])
    orient = 0.0 if horizontal else np.pi / 4.0
    zoom = PANEL_A_NETWORK_ZOOM
    (adj1, deg1, hubs1, shell1, xy1) = module_geometry(SEED_1, u, target_angle=orient, center_x=0.0)
    (adj2, deg2, hubs2, shell2, xy2) = module_geometry(
        SEED_2, v, target_angle=np.pi if horizontal else np.pi + orient, center_x=0.0
    )
    xy1 = xy1.copy()
    xy2 = xy2.copy()
    xy1 *= zoom
    xy2 *= zoom
    bridge_direction = np.array([1.0, 0.0]) if horizontal else np.array([1.0, 1.0]) / np.sqrt(2.0)
    bridge_half_gap = 0.34 if horizontal else 0.3
    xy1 += -bridge_half_gap * bridge_direction - xy1[u]
    xy2 += bridge_half_gap * bridge_direction - xy2[v]
    unshifted_all_xy = np.vstack([xy1, xy2]).copy()
    axes_width_mm = max(ax.get_position().width * FIG_W_MM, 1e-12)
    content_shift_axes = PANEL_A_CONTENT_SHIFT_MM / axes_width_mm
    content_shift_data = 0.0
    if PANEL_A_CONTENT_SHIFT_MM:
        unshifted_width = float(np.ptp(unshifted_all_xy[:, 0]) + 0.24)
        content_shift_data = PANEL_A_CONTENT_SHIFT_MM * unshifted_width / axes_width_mm
        xy1[:, 0] += content_shift_data
        xy2[:, 0] += content_shift_data
    if PANEL_A_LOWER_NETWORK_SHIFT_MM or PANEL_A_UPPER_NETWORK_SHIFT_MM:
        unshifted_width = float(np.ptp(unshifted_all_xy[:, 0]) + 0.24)
        shift_scale = unshifted_width / axes_width_mm
        xy1[:, 0] -= PANEL_A_LOWER_NETWORK_SHIFT_MM * shift_scale
        xy2[:, 0] += PANEL_A_UPPER_NETWORK_SHIFT_MM * shift_scale
    if PANEL_A_NETWORK_VERTICAL_SHIFT_MM:
        unshifted_width = float(np.ptp(unshifted_all_xy[:, 0]) + 0.24)
        shift_scale = unshifted_width / axes_width_mm
        vertical_shift_data = PANEL_A_NETWORK_VERTICAL_SHIFT_MM * shift_scale
        xy1[:, 1] += vertical_shift_data
        xy2[:, 1] += vertical_shift_data
    if PANEL_A_ENHANCED_DEGREE_DRAW:
        for adjacency, xy, degree, hubs, shell in [
            (adj1, xy1, deg1, hubs1, shell1),
            (adj2, xy2, deg2, hubs2, shell2),
        ]:
            ax.add_collection(
                _p1_LineCollection(
                    _p1_edge_segments(adjacency, xy),
                    colors=(0.2, 0.24, 0.27, 0.06),
                    linewidths=0.11,
                    zorder=1,
                )
            )
            degree_fraction = degree / max(float(degree.max()), 1.0)
            sizes = PANEL_A_NODE_SIZE_MIN + PANEL_A_NODE_SIZE_SCALE * np.power(
                degree_fraction, PANEL_A_NODE_SIZE_EXPONENT
            )
            for shell_level in sorted((int(value) for value in np.unique(shell))):
                nodes = np.flatnonzero(shell == shell_level)
                nodes = nodes[np.argsort(degree[nodes], kind="stable")]
                ax.scatter(
                    xy[nodes, 0],
                    xy[nodes, 1],
                    s=sizes[nodes],
                    c=_p1_PANEL_SHELL_COLORS.get(shell_level, "#B8B8B8"),
                    alpha=0.84 if shell_level == 0 else 0.74,
                    edgecolors=PANEL_A_NODE_EDGECOLOR,
                    linewidths=PANEL_A_NODE_EDGEWIDTH,
                    zorder=2,
                )
    else:
        draw_module(ax, adj1, xy1, deg1, hubs1, shell1, "BA1", show_label=False)
        draw_module(ax, adj2, xy2, deg2, hubs2, shell2, "BA2", show_label=False)
    p1 = xy1[u]
    p2 = xy2[v]
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="white",
        linewidth=5.8,
        solid_capstyle="round",
        zorder=7,
    )
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="#111111",
        linewidth=2.5,
        solid_capstyle="round",
        zorder=8,
    )
    ax.scatter(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        s=31,
        facecolors="none",
        edgecolors="#111111",
        linewidths=0.85,
        zorder=9,
    )
    if (
        PANEL_A_CONTENT_SHIFT_MM
        or PANEL_A_LOWER_NETWORK_SHIFT_MM
        or PANEL_A_UPPER_NETWORK_SHIFT_MM
        or PANEL_A_NETWORK_VERTICAL_SHIFT_MM
    ):
        for artist in [*ax.collections, *ax.lines]:
            artist.set_clip_on(False)
    mid = 0.5 * (p1 + p2)
    best_text_offset = np.array([0.0, 0.1]) if horizontal else np.array([0.18, -0.1])
    ax.text(
        float(mid[0] + best_text_offset[0]),
        float(mid[1] + best_text_offset[1]),
        "Best",
        ha="center",
        va="bottom" if horizontal else "top",
        fontsize=_p1_FIG4_FONT_PT * 0.9,
        color=_p1_DARK,
    )
    if horizontal:
        horizontal_text_y = 0.145 if PANEL_A_HORIZONTAL_TEXT_BELOW else 0.975
        horizontal_text_va = "bottom" if PANEL_A_HORIZONTAL_TEXT_BELOW else "top"
        horizontal_text_x = max(PANEL_A_TEXT_X + content_shift_axes, 0.0)
        horizontal_second_text_offset = 0.52 if PANEL_A_HORIZONTAL_TEXT_BELOW else 0.31
        if PANEL_A_HORIZONTAL_TEXT_BELOW:
            if PANEL_A_SHOW_NETWORK_SIZE:
                ax.text(
                    0.5,
                    PANEL_A_NETWORK_SIZE_Y,
                    f"Two BA networks, $N={MODULE_N:,}$ each",
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=_p1_FIG4_FONT_PT,
                    color=_p1_DARK,
                )
            threshold_rows = [
                (
                    PANEL_A_THRESHOLD_BEFORE_Y,
                    "Before bridge:",
                    f"$(b/c)^*={BASELINE_THRESHOLD_BEFORE:.3f} > (b/c)^*_r={BASELINE_THRESHOLD_RR:.3f}$",
                ),
                (
                    PANEL_A_THRESHOLD_BEST_Y,
                    "Best bridge:",
                    f"$(b/c)^*={float(best['exact_threshold']):.3f} < (b/c)^*_r={float(best['threshold_RR']):.3f}$",
                ),
            ]
            for row_y, row_label, row_value in threshold_rows:
                ax.text(
                    PANEL_A_THRESHOLD_LABEL_X,
                    row_y,
                    row_label,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=_p1_FIG4_FONT_PT,
                    color=_p1_DARK,
                )
                ax.text(
                    PANEL_A_THRESHOLD_VALUE_X,
                    row_y,
                    row_value,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=_p1_FIG4_FONT_PT,
                    color=_p1_DARK,
                )
        else:
            ax.text(
                horizontal_text_x,
                horizontal_text_y,
                f"Before bridge\n$(b/c)^*={BASELINE_THRESHOLD_BEFORE:.3f}$\n$(b/c)^*_r={BASELINE_THRESHOLD_RR:.3f}$",
                transform=ax.transAxes,
                ha="left",
                va=horizontal_text_va,
                fontsize=_p1_FIG4_FONT_PT,
                linespacing=1.08,
                color=_p1_DARK,
            )
            ax.text(
                horizontal_text_x + horizontal_second_text_offset,
                horizontal_text_y,
                f"Best bridge\n$(b/c)^*={float(best['exact_threshold']):.3f}$\n$(b/c)^*_r={float(best['threshold_RR']):.3f}$",
                transform=ax.transAxes,
                ha="left",
                va=horizontal_text_va,
                fontsize=_p1_FIG4_FONT_PT,
                linespacing=1.08,
                color=_p1_DARK,
            )
    else:
        ax.text(
            PANEL_A_TEXT_X,
            0.915,
            f"Before bridge\n$(b/c)^*={BASELINE_THRESHOLD_BEFORE:.3f}$\n$(b/c)^*_r={BASELINE_THRESHOLD_RR:.3f}$\n\nBest bridge\n$(b/c)^*={float(best['exact_threshold']):.3f}$\n$(b/c)^*_r={float(best['threshold_RR']):.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=_p1_FIG4_FONT_PT,
            linespacing=1.08,
            color=_p1_DARK,
        )
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.0,
            markerfacecolor=_p1_PANEL_SHELL_COLORS[level],
            markeredgecolor="none",
            label=f"Shell {level}",
        )
        for level in SHELL_LEVELS
    ]
    if horizontal:
        ax.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(max(0.52 + content_shift_axes, 0.46), 0.005),
            ncol=4,
            frameon=False,
            fontsize=_p1_FIG4_FONT_PT,
            handletextpad=0.3,
            columnspacing=0.75,
            borderaxespad=0.0,
        )
    else:
        ax.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(0.98 - 2.0 / (FIG_W_MM * 0.485), 0.05),
            ncol=1,
            frameon=False,
            fontsize=_p1_FIG4_FONT_PT,
            handletextpad=0.5,
            labelspacing=0.4,
            borderaxespad=0.0,
        )
    (xmin, ymin) = unshifted_all_xy.min(axis=0)
    (xmax, ymax) = unshifted_all_xy.max(axis=0)
    if horizontal:
        ax.set_xlim(
            xmin + min(content_shift_data, 0.0) - 0.08, xmax + max(content_shift_data, 0.0) + 0.12
        )
        if PANEL_A_HORIZONTAL_TEXT_BELOW:
            ax.set_ylim(ymin - 1.35, ymax + 0.35)
        else:
            ax.set_ylim(ymin - 0.55, ymax + 1.15)
    else:
        ax.set_xlim(xmin - 0.1, xmax + 0.14)
        ax.set_ylim(ymin - 0.18, ymax + 0.18)


def ORIGINAL_DRAW_METRIC_BOX_PANEL(
    ax: plt.Axes,
    frame: pd.DataFrame,
    grouped: pd.DataFrame,
    *,
    metric: str,
    metric_label: str,
    title: str,
    panel_label: str,
    show_ylabels: bool,
    global_best: pd.Series | None = None,
    mark_group_best: bool = False,
) -> None:
    ax.text(
        PANEL_B_LABEL_X,
        PANEL_B_LABEL_Y,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=_p1_FIG4_FONT_PT,
        fontweight="bold",
        color=_p1_DARK,
        clip_on=False,
    )
    order = grouped["top10_shell_pair_unordered"].tolist()
    values = [
        frame.loc[frame["top10_shell_pair_unordered"].eq(pair), metric].to_numpy() for pair in order
    ]
    pos = np.arange(1, len(order) + 1)
    box = ax.boxplot(
        values,
        positions=pos,
        orientation="horizontal",
        widths=0.54,
        patch_artist=True,
        showfliers=False,
    )
    if metric == "drop_vs_rr":
        mean_col = "drop_mean"
        color_col = "drop_mean"
    elif metric == "margin_M":
        mean_col = "margin_mean"
        color_col = "margin_mean"
    elif metric == "exact_threshold":
        mean_col = "threshold_mean"
        color_col = "margin_mean"
    else:
        mean_col = "sg_mean"
        color_col = "sg_mean"
    means = grouped[mean_col].to_numpy()
    color_values = grouped[color_col].to_numpy(dtype=float)
    (lo, hi) = (float(np.nanmin(color_values)), float(np.nanmax(color_values)))
    cmap = plt.get_cmap("YlOrRd")
    for patch, color_value in zip(box["boxes"], color_values):
        t = 0.5 if hi == lo else (float(color_value) - lo) / (hi - lo)
        patch.set_facecolor(cmap(t))
        patch.set_alpha(0.84)
        patch.set_edgecolor("#454545")
        patch.set_linewidth(0.8)
    for item in box["medians"]:
        item.set_color("#111111")
        item.set_linewidth(UNIFIED_BC_MEDIAN_LINEWIDTH if UNIFIED_BC_BOXPLOT_STYLE else 1.05)
    for item in box["whiskers"] + box["caps"]:
        item.set_color("#4A4A4A")
        item.set_linewidth(UNIFIED_BC_WHISKER_LINEWIDTH if UNIFIED_BC_BOXPLOT_STYLE else 0.75)
    rng = np.random.default_rng(20260611 + (0 if metric == "drop_vs_rr" else 37))
    for y, pair, vals in zip(pos, order, values):
        take = np.arange(len(vals))
        if len(vals) > 85:
            take = rng.choice(len(vals), 85, replace=False)
        ax.scatter(
            vals[take],
            np.full(len(take), y) + rng.normal(0, 0.055, len(take)),
            s=5.2,
            color="#202020",
            alpha=0.18,
            linewidths=0,
            zorder=2,
        )
        row = grouped.loc[grouped["top10_shell_pair_unordered"].eq(pair)].iloc[0]
        mean_value = float(getattr(row, mean_col))
        ax.scatter(
            mean_value,
            y,
            marker="D",
            s=UNIFIED_BC_MEAN_MARKER_AREA if UNIFIED_BC_BOXPLOT_STYLE else 24.0,
            color=UNIFIED_BC_MEAN_MARKER_COLOR if UNIFIED_BC_BOXPLOT_STYLE else TEAL,
            edgecolor="white"
            if not UNIFIED_BC_BOXPLOT_STYLE or UNIFIED_BC_MEAN_MARKER_EDGE_VISIBLE
            else "none",
            linewidth=0.4
            if not UNIFIED_BC_BOXPLOT_STYLE or UNIFIED_BC_MEAN_MARKER_EDGE_VISIBLE
            else 0.0,
            zorder=4,
        )
        is_global_pair = global_best is not None and pair == str(
            global_best["top10_shell_pair_unordered"]
        )
        if mark_group_best and (not is_global_pair):
            group_best = (
                float(np.nanmin(vals)) if metric == "exact_threshold" else float(np.nanmax(vals))
            )
            ax.scatter(
                group_best,
                y,
                marker="*",
                s=BEST_STAR_MARKER_AREA if UNIFIED_BC_BOXPLOT_STYLE else 47,
                color=UNIFIED_BC_MEAN_MARKER_COLOR if UNIFIED_BC_BOXPLOT_STYLE else RED,
                edgecolor="none" if UNIFIED_BC_BOXPLOT_STYLE else "white",
                linewidth=0.0 if UNIFIED_BC_BOXPLOT_STYLE else 0.35,
                zorder=5,
            )
    if global_best is not None:
        best_pair = str(global_best["top10_shell_pair_unordered"])
        if best_pair in order:
            best_y = float(pos[order.index(best_pair)])
            best_x = float(global_best[metric])
            ax.scatter(
                best_x,
                best_y,
                marker="*",
                s=BEST_STAR_MARKER_AREA if UNIFIED_BC_BOXPLOT_STYLE else 47,
                color=UNIFIED_BC_MEAN_MARKER_COLOR if UNIFIED_BC_BOXPLOT_STYLE else RED,
                edgecolor="none" if UNIFIED_BC_BOXPLOT_STYLE else "white",
                linewidth=0.0 if UNIFIED_BC_BOXPLOT_STYLE else 0.35,
                zorder=6,
            )
    labels = [str(row.top10_shell_pair_unordered) for row in grouped.itertuples(index=False)]
    ax.set_yticks(pos)
    if show_ylabels:
        ax.set_yticklabels(labels, fontsize=_p1_FIG4_FONT_PT)
        ax.set_ylabel("Shell pair class", fontsize=_p1_FIG4_FONT_PT)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel(metric_label, fontsize=_p1_FIG4_FONT_PT)
    ax.tick_params(axis="x", labelsize=_p1_FIG4_FONT_PT, length=2.4, width=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_shell_colored_box_panel(ax: plt.Axes, frame, grouped, **kwargs) -> None:
    ORIGINAL_DRAW_METRIC_BOX_PANEL(ax, frame, grouped, **kwargs)
    rr_threshold = float(np.nanmedian(frame["threshold_RR"].to_numpy(dtype=float)))
    ax.axvline(
        rr_threshold, color=GRAY, linewidth=0.8, linestyle=(0, (3.0, 2.2)), alpha=0.88, zorder=0.8
    )
    ax.set_xlim(3.9965, 4.0145)
    threshold_ticks = [3.998, 4.002, 4.006, 4.01, 4.014]
    ax.set_xticks(threshold_ticks)
    ax.set_xticklabels([f"{tick:.3f}" for tick in threshold_ticks])
    ax.text(
        rr_threshold - 0.00012,
        0.975,
        "$(b/c)^*_r$",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=_p1_FIG4_FONT_PT * 0.88,
        color=GRAY,
    )
    for collection in list(ax.collections):
        alpha = collection.get_alpha()
        if alpha is not None and np.isclose(alpha, 0.18):
            collection.remove()
    order = grouped["top10_shell_pair_unordered"].tolist()
    values_by_group = [
        frame.loc[frame["top10_shell_pair_unordered"].eq(pair), kwargs["metric"]].to_numpy(
            dtype=float
        )
        for pair in order
    ]
    box_patches = ax.patches[-len(order) :]
    for patch in box_patches:
        patch.set_facecolor("none")
        patch.set_edgecolor("none")
        patch.set_linewidth(0.0)
        patch.set_alpha(1.0)
    for y, pair, values in zip(np.arange(1, len(order) + 1), order, values_by_group):
        (q1, q3) = np.nanpercentile(values, [25.0, 75.0])
        (shell_a, shell_b) = (int(value) for value in pair.split("-"))
        pair_color = _p1_PANEL_SHELL_COLORS[shell_a] if shell_a == shell_b else "#A3A9AE"
        width = float(q3 - q1)
        ax.add_patch(
            Rectangle(
                (float(q1), y - 0.27),
                width,
                0.54,
                facecolor=pair_color,
                edgecolor="none",
                alpha=0.55,
                zorder=1.5,
            )
        )


def _plot_tau_shell_change_content(
    ax: plt.Axes, nodes: pd.DataFrame, *, show_bridge_endpoints: bool = True
) -> None:
    cases = [
        ("shell0-shell0", SHELL0_COLOR, -0.13),
        (PERIPHERAL_PAIR_LABEL, PERIPHERAL_COLOR, 0.13),
    ]
    shells = SHELL_LEVELS
    for label, color, offset in cases:
        line_color = "#4A4A4A" if UNIFIED_BC_BOXPLOT_STYLE else color
        mean_color = UNIFIED_BC_MEAN_MARKER_COLOR if UNIFIED_BC_BOXPLOT_STYLE else color
        line_width = UNIFIED_BC_WHISKER_LINEWIDTH if UNIFIED_BC_BOXPLOT_STYLE else 0.65
        line_alpha = 0.92 if UNIFIED_BC_BOXPLOT_STYLE else 0.78
        values = [
            nodes.loc[
                nodes["case_label"].eq(label) & nodes["shell"].eq(shell), "delta_tau"
            ].to_numpy(dtype=float)
            for shell in shells
        ]
        positions = [shell + offset for shell in shells]
        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
            manage_ticks=False,
            medianprops=dict(
                color=_p1_DARK,
                linewidth=UNIFIED_BC_MEDIAN_LINEWIDTH if UNIFIED_BC_BOXPLOT_STYLE else 0.65,
            ),
            whiskerprops=dict(color=line_color, linewidth=line_width, alpha=line_alpha),
            capprops=dict(color=line_color, linewidth=line_width, alpha=line_alpha),
            meanprops=dict(
                marker="D",
                markerfacecolor=mean_color,
                markeredgecolor="white"
                if not UNIFIED_BC_BOXPLOT_STYLE or UNIFIED_BC_MEAN_MARKER_EDGE_VISIBLE
                else "none",
                markeredgewidth=0.4
                if UNIFIED_BC_BOXPLOT_STYLE and UNIFIED_BC_MEAN_MARKER_EDGE_VISIBLE
                else 0.35
                if not UNIFIED_BC_BOXPLOT_STYLE
                else 0.0,
                markersize=3.4,
            ),
        )
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55 if UNIFIED_BC_BOXPLOT_STYLE else 0.3)
            if UNIFIED_BC_BOXPLOT_STYLE and (not UNIFIED_BC_BOX_EDGE_VISIBLE):
                patch.set_edgecolor("none")
                patch.set_linewidth(0.0)
            else:
                patch.set_edgecolor("#4A4A4A" if UNIFIED_BC_BOXPLOT_STYLE else color)
                patch.set_linewidth(0.8 if UNIFIED_BC_BOXPLOT_STYLE else 0.75)
        if show_bridge_endpoints:
            endpoints = nodes.loc[
                nodes["case_label"].eq(label) & nodes["is_bridge_endpoint"].eq(1)
            ].copy()
            endpoint_jitter = (
                np.linspace(-0.022, 0.022, len(endpoints))
                if len(endpoints) > 1
                else np.array([0.0])
            )
            endpoint_x = endpoints["shell"].to_numpy(dtype=float) + offset + endpoint_jitter
            endpoint_y = endpoints["delta_tau"].to_numpy(dtype=float)
            ax.scatter(
                endpoint_x,
                endpoint_y,
                marker="^",
                s=36,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )


def add_bottom_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        PANEL_C_LABEL_X if label == "c" else -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=_p1_FIG4_FONT_PT,
        fontweight="bold",
        color=_p1_DARK,
        clip_on=False,
    )


def load_tau_node_changes():
    return pd.read_csv(DATA_DIR / "reach_changes.csv")


def draw_tau_shell_change_panel(
    fig: plt.Figure, panel_rect: tuple[float, float, float, float]
) -> None:
    (left, bottom, width, height) = panel_rect
    ax = fig.add_axes([left, bottom, width, height])
    nodes = load_tau_node_changes()
    _plot_tau_shell_change_content(ax, nodes, show_bridge_endpoints=False)
    ax.set_ylim(300, 700)
    ax.set_yticks([300, 500, 700])
    ax.set_xticks(SHELL_LEVELS)
    ax.set_xlabel("Shell", fontsize=_p1_FIG4_FONT_PT)
    ax.set_ylabel(
        "Reach change, $\\tau_i^{\\prime}-\\tau_i$", fontsize=_p1_FIG4_FONT_PT, labelpad=1.0
    )
    ax.yaxis.set_label_coords(-0.18, 0.5)
    ax.tick_params(labelsize=_p1_FIG4_FONT_PT, length=2.2, width=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_bottom_panel_label(ax, "c")
    if PANEL_C_SHOW_BRIDGE_LEGEND:
        bridge_legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=5.3,
                markerfacecolor=SHELL0_COLOR,
                markeredgecolor="none",
                alpha=0.55,
                label="0-0 bridge",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="",
                markersize=5.3,
                markerfacecolor=PERIPHERAL_COLOR,
                markeredgecolor="none",
                alpha=0.55,
                label=f"{PERIPHERAL_SHELL}-{PERIPHERAL_SHELL} bridge",
            ),
        ]
        ax.legend(
            handles=bridge_legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            frameon=False,
            fontsize=_p1_FIG4_FONT_PT * 0.82,
            handletextpad=0.28,
            labelspacing=0.24,
            borderaxespad=0.0,
        )
    inset = ax.inset_axes([0.49, 0.61, 0.48, 0.34])
    inset.set_facecolor((1.0, 1.0, 1.0, 0.97))
    endpoint_specs = [
        ("shell0-shell0", SHELL0_COLOR, 0),
        (PERIPHERAL_PAIR_LABEL, PERIPHERAL_COLOR, 1),
    ]
    for label, color, x0 in endpoint_specs:
        endpoints = nodes.loc[nodes["case_label"].eq(label) & nodes["is_bridge_endpoint"].eq(1)]
        endpoint_y = endpoints["delta_tau"].to_numpy(dtype=float)
        endpoint_x = np.full(len(endpoint_y), x0, dtype=float)
        if len(endpoint_y) > 1:
            endpoint_x += np.linspace(-0.045, 0.045, len(endpoint_y))
        inset.scatter(
            endpoint_x,
            endpoint_y,
            marker="^",
            s=30,
            color=color,
            edgecolor="none",
            linewidth=0.0,
            zorder=3,
        )
    inset.set_xlim(-0.45, 1.45)
    inset.set_ylim(0, 3600)
    inset.set_xticks([0, 1])
    inset.set_xticklabels(["0-0", PERIPHERAL_PAIR_VALUE], fontsize=_p1_FIG4_FONT_PT * 0.76)
    inset.set_yticks([0, 1500, 3000])
    inset.set_yticklabels(["0", "1500", "3000"], fontsize=_p1_FIG4_FONT_PT * 0.72)
    inset.set_title("Bridge nodes", fontsize=_p1_FIG4_FONT_PT * 0.84, pad=1.0)
    inset.tick_params(length=1.7, width=0.5, pad=1.0)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#A8AFB5")
        spine.set_linewidth(0.55)


def draw_same_shell_best_bridges(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, frame: pd.DataFrame
) -> None:
    pair_values = frame["top10_shell_pair_unordered"]
    for level in SHELL_LEVELS:
        positions = np.flatnonzero(pair_values.eq(f"{level}-{level}").to_numpy())
        if positions.size == 0:
            continue
        best_position = positions[np.nanargmin(y[positions])]
        ax.scatter(
            [x[best_position]],
            [y[best_position]],
            marker="*",
            s=BEST_STAR_MARKER_AREA,
            color=_p1_PANEL_SHELL_COLORS[level],
            edgecolor="none",
            linewidth=0.0,
            zorder=6,
        )


def draw_density_background(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, mask: np.ndarray, *, gridsize: int
) -> None:
    density = ax.hexbin(
        x[mask],
        y[mask],
        gridsize=gridsize,
        mincnt=1,
        bins="log",
        cmap=DENSITY_CMAP,
        linewidths=0,
        alpha=DENSITY_BACKGROUND_ALPHA,
        rasterized=True,
        zorder=DENSITY_BACKGROUND_ZORDER,
    )
    density.set_rasterized(True)


def sampled_mask(mask: np.ndarray, *, max_points: int, seed: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) > max_points:
        indices = np.random.default_rng(seed).choice(indices, max_points, replace=False)
    sampled = np.zeros(mask.size, dtype=bool)
    sampled[indices] = True
    return sampled


def draw_same_shell_samples(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, frame: pd.DataFrame, *, gridsize: int, seed: int
) -> None:
    pair_values = frame["top10_shell_pair_unordered"]
    same_shell_masks = {
        level: pair_values.eq(f"{level}-{level}").to_numpy() for level in SHELL_LEVELS
    }
    any_same_shell = np.logical_or.reduce(list(same_shell_masks.values()))
    draw_density_background(ax, x, y, ~any_same_shell, gridsize=gridsize)
    draw_order = [level for level in SHELL_LEVELS if level != 0] + [0]
    for draw_rank, level in enumerate(draw_order):
        mask = same_shell_masks[level]
        sampled = sampled_mask(mask, max_points=1200, seed=seed + level)
        ax.scatter(
            x[sampled],
            y[sampled],
            s=7.5,
            color=_p1_PANEL_SHELL_COLORS[level],
            alpha=0.52 if level != 2 else 0.62,
            linewidths=0,
            rasterized=True,
            zorder=SAME_SHELL_ZORDER_BASE + 0.1 * draw_rank,
        )


def format_threshold_y_axis(ax: plt.Axes) -> None:
    ax.set_ylim(3.9965, 4.0102)
    ax.set_yticks(THRESHOLD_TICKS)
    ax.set_yticklabels([f"{tick:.3f}" for tick in THRESHOLD_TICKS])


def draw_p_product_panel(
    ax: plt.Axes, frame: pd.DataFrame, best: pd.Series, show_legend: bool = True
) -> dict[str, float]:
    add_bottom_panel_label(ax, "d")
    x = frame["p_before_product"].to_numpy(dtype=float)
    y = frame["exact_threshold"].to_numpy(dtype=float)
    (pearson_r, pearson_p) = stats.pearsonr(x, y)
    (spearman_rho, spearman_p) = stats.spearmanr(x, y)
    draw_same_shell_samples(ax, x, y, frame, gridsize=54, seed=20260710)
    draw_same_shell_best_bridges(ax, x, y, frame)
    ax.set_xlabel("Reciprocity product, $p_i p_j$", fontsize=_p1_FIG4_FONT_PT)
    ax.set_xlim(-0.005, 0.18)
    ax.set_xticks([0.0, 0.05, 0.1, 0.15])
    ax.set_ylabel(
        "Critical benefit-to-cost ratio, $(b/c)^*$", fontsize=_p1_FIG4_FONT_PT, labelpad=1.5
    )
    ax.yaxis.set_label_coords(-0.27, 0.5)
    format_threshold_y_axis(ax)
    ax.tick_params(labelsize=_p1_FIG4_FONT_PT, length=2.2, width=0.6)
    ax.text(
        0.97,
        0.985,
        f"Spearman: {spearman_rho:.2f}".replace("-", "−"),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=_p1_FIG4_FONT_PT,
        color=_p1_DARK,
    )
    if show_legend:
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=3.2,
                markerfacecolor=SHELL0_COLOR,
                markeredgecolor="none",
                alpha=0.58,
                label="shell0-shell0",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=3.2,
                markerfacecolor=PERIPHERAL_COLOR,
                markeredgecolor="none",
                alpha=0.62,
                label=PERIPHERAL_PAIR_LABEL,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=3.2,
                markerfacecolor="#7F8790",
                markeredgecolor="none",
                alpha=0.34,
                label="others",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            fontsize=_p1_FIG4_FONT_PT,
            handletextpad=0.18,
            columnspacing=0.42,
            borderaxespad=0.05,
            ncol=3,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return {
        "p_before_product_pearson_r": pearson_r,
        "p_before_product_pearson_p": pearson_p,
        "p_before_product_spearman_rho": spearman_rho,
        "p_before_product_spearman_p": spearman_p,
    }


def draw_normalized_reach_product_panel(
    ax: plt.Axes, frame: pd.DataFrame, best: pd.Series, show_legend: bool = True
) -> dict[str, float]:
    add_bottom_panel_label(ax, "e")
    x = frame["normalized_reach_product_before"].to_numpy(dtype=float)
    y = frame["exact_threshold"].to_numpy(dtype=float)
    (pearson_r, pearson_p) = stats.pearsonr(x, y)
    (spearman_rho, spearman_p) = stats.spearmanr(x, y)
    draw_same_shell_samples(ax, x, y, frame, gridsize=50, seed=20260720)
    draw_same_shell_best_bridges(ax, x, y, frame)
    ax.set_xlabel(
        "Scaled reach product, $\\tau_i\\tau_j/N_{\\mathrm{eff}}^2$", fontsize=_p1_FIG4_FONT_PT
    )
    ax.set_xlim(0.35, 1.5)
    ax.set_xticks([0.4, 0.8, 1.2])
    format_threshold_y_axis(ax)
    ax.tick_params(labelleft=False, labelsize=_p1_FIG4_FONT_PT, length=2.1, width=0.58)
    ax.text(
        0.04,
        0.985,
        f"Spearman: {spearman_rho:.2f}".replace("-", "−"),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=_p1_FIG4_FONT_PT,
        color=_p1_DARK,
    )
    if show_legend:
        legend_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=3.2,
                markerfacecolor=SHELL0_COLOR,
                markeredgecolor="none",
                alpha=0.58,
                label="shell0-shell0",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=3.2,
                markerfacecolor=PERIPHERAL_COLOR,
                markeredgecolor="none",
                alpha=0.62,
                label=PERIPHERAL_PAIR_LABEL,
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=3.2,
                markerfacecolor="#7F8790",
                markeredgecolor="none",
                alpha=0.34,
                label="others",
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            fontsize=_p1_FIG4_FONT_PT,
            handletextpad=0.18,
            columnspacing=0.42,
            borderaxespad=0.05,
            ncol=3,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return {
        "normalized_reach_product_before_pearson_r": pearson_r,
        "normalized_reach_product_before_pearson_p": pearson_p,
        "normalized_reach_product_before_spearman_rho": spearman_rho,
        "normalized_reach_product_before_spearman_p": spearman_p,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce manuscript Figure 4 from frozen source data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_DIR / "bridges.csv.gz")
    best = pd.read_csv(DATA_DIR / "best_bridge.csv").iloc[0]
    grouped = pd.read_csv(DATA_DIR / "shell_groups.csv", float_precision="round_trip")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(
        figsize=(7.204724409448819, 5.708661417322835), dpi=300, constrained_layout=False
    )
    ax_a = fig.add_axes([0.005, 0.495, 0.52, 0.47])
    ax_b = fig.add_axes([0.565, 0.61, 0.385, 0.33])
    bottom_y = 0.16948275862068968
    (bottom_w, bottom_h) = (0.24, 0.30289655172413793)
    ax_d = fig.add_axes([0.43, bottom_y, bottom_w, bottom_h])
    ax_e = fig.add_axes([0.73, bottom_y, bottom_w, bottom_h])
    draw_panel_a_slanted(ax_a, best)
    draw_shell_colored_box_panel(
        ax_b,
        frame,
        grouped,
        metric="exact_threshold",
        metric_label="Critical benefit-to-cost ratio, $(b/c)^*$",
        title="Bridge effects by endpoint shell",
        panel_label="b",
        show_ylabels=True,
        global_best=best,
        mark_group_best=True,
    )
    draw_tau_shell_change_panel(fig, (0.08, bottom_y, bottom_w, bottom_h))
    draw_p_product_panel(ax_d, frame, best, show_legend=False)
    draw_normalized_reach_product_panel(ax_e, frame, best, show_legend=False)
    colors = {
        0: "#C8788A",
        1: "#D8BC70",
        2: "#6FB6A5",
        3: "#7899BC",
        4: "#7B68A6",
        5: "#6A4C93",
        6: "#7A8B95",
    }
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4.0,
            markerfacecolor=colors[level],
            markeredgecolor="none",
            alpha=0.68,
            label=f"{level}-{level}",
        )
        for level in range(4)
    ]
    handles += [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4.0,
            markerfacecolor="#7F8790",
            markeredgecolor="none",
            alpha=0.34,
            label="other",
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.525, 0.05448275862068966),
        fontsize=10,
        handletextpad=0.22,
        columnspacing=0.55,
        borderaxespad=0.0,
        ncol=5,
    )
    export_bbox = Bbox.from_bounds(
        0, 0.35433070866141736, fig.get_figwidth(), fig.get_figheight() - 0.42519685039370086
    )
    stem = args.output_dir / "fig4"
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=args.dpi,
        facecolor="white",
        bbox_inches=export_bbox,
        pad_inches=0,
    )
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", bbox_inches=export_bbox, pad_inches=0)
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
