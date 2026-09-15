"""Reproduce manuscript Figure 2 from the bundled, frozen source data.

Run from any working directory; see figures/README.md for provenance and scope.
The plotting functions preserve the manuscript plotting workflow.
"""
from __future__ import annotations
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator as LogLocator
from matplotlib.ticker import NullFormatter as NullFormatter
from pathlib import Path
import argparse
import math as math
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "fig2"
COLOR_BA = "#3B6FB0"
COLOR_LIGHT_GUIDE = "#999999"
COLOR_NC = "#999999"
COLOR_RR = "#C44E52"
ERROR_BAR_ALPHA = 0.65
ERROR_BAR_CAPSIZE = 1.5
ERROR_BAR_WIDTH = 0.55
GUIDE_WIDTH = 0.65
INSET_MARKER_SIZE = 2.1
INSET_TICK_FONT_SIZE = 6.5
LABEL_FONT_SIZE = 10.0
LEGEND_FONT_SIZE = 8.0
LINE_WIDTH = 1.15
MAIN_N_XLIM = (90.0, 11200.0)
MARKER_SIZE = 3.4
TITLE_FONT_SIZE = 10.0
ERROR_BAR_KIND = "sd"
AXES_WIDTH = 0.8
TICK_FONT_SIZE = 8.0
THEORY_LINE_WIDTH = 0.75
COLOR_CRIT = "#DD8452"
COLOR_EPS = "#8172B3"
MECHANISM_FORMULA_SIZE = 10.0
COLOR_BA_BETTER_BG = "#DDEBF7"
COLOR_BA_BETTER_TEXT = "#2F5F8F"
COLOR_BA_WORSE_BG = "#F6DDDD"
COLOR_BA_WORSE_TEXT = "#8A3D3D"
PANEL_LABEL_SIZE = 12.0


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "mathtext.fontset": "cm",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.right": True,
            "axes.spines.top": True,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.55,
            "ytick.minor.width": 0.55,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "legend.frameon": False,
        }
    )


def log_axis_fraction(value: float, xlim: tuple[float, float]) -> float:
    (lo, hi) = np.log10(xlim)
    return float((math.log10(value) - lo) / (hi - lo))


def aligned_log_inset(
    ax: plt.Axes,
    *,
    anchor_value: float,
    main_xlim: tuple[float, float],
    inset_xlim: tuple[float, float],
    width: float,
    height: float,
    bottom: float,
) -> plt.Axes:
    main_fraction = log_axis_fraction(anchor_value, main_xlim)
    inset_fraction = log_axis_fraction(anchor_value, inset_xlim)
    left = main_fraction - width * inset_fraction
    left = min(max(left, 0.03), 0.97 - width)
    return ax.inset_axes([left, bottom, width, height])


def ensemble_error(data: pd.DataFrame, stem: str) -> np.ndarray:
    suffix = "std" if ERROR_BAR_KIND == "sd" else "sem"
    return data[f"{stem}_{suffix}"].fillna(0.0).to_numpy(dtype=float)


def k4_observed_nc():
    """First sampled N with positive mean alignment-minus-penalty thereafter.

    This is the observed crossing on the sampled population-size grid. Panel d
    retains the separately archived critical-size predictions and validation.
    """
    _, _, ensemble = load_selected_ensemble()
    favorable = ensemble["margin"].to_numpy(float) > 0
    persistent = np.logical_and.accumulate(favorable[::-1])[::-1]
    candidates = ensemble.loc[persistent, "N"]
    if candidates.empty:
        raise ValueError("The selected ensemble has no persistent favorable size.")
    return float(candidates.iloc[0])


def set_log_x_ticks(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())


def style_axes(ax: plt.Axes) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(AXES_WIDTH)
    ax.tick_params(labelsize=TICK_FONT_SIZE, width=AXES_WIDTH, top=False, right=False)


def draw_panel_a(ax: plt.Axes, ba: pd.DataFrame, rr: pd.DataFrame) -> None:
    display_n = {100, 200, 316, 500, 1000, 1600, 2000, 3162, 5000, 10000}
    ba_plot = ba[ba["n"].isin(display_n)].copy()
    rr_plot = rr[rr["n"].isin(display_n)].copy()
    nc = k4_observed_nc()
    main_xlim = MAIN_N_XLIM
    rr_handle = ax.plot(
        rr_plot["n"],
        rr_plot["threshold_mean"],
        color=COLOR_RR,
        marker="s",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        label="Regular",
        clip_on=False,
        zorder=5,
    )[0]
    ba_handle = ax.errorbar(
        ba_plot["n"],
        ba_plot["threshold_mean"],
        yerr=ensemble_error(ba_plot, "threshold"),
        color=COLOR_BA,
        marker="o",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        elinewidth=ERROR_BAR_WIDTH,
        capsize=ERROR_BAR_CAPSIZE,
        capthick=ERROR_BAR_WIDTH,
        ecolor=COLOR_BA,
        alpha=ERROR_BAR_ALPHA,
        label="BA",
        clip_on=False,
        zorder=6,
    )[0]
    ba_handle.set_alpha(1.0)
    ax.axhline(4.0, color=COLOR_LIGHT_GUIDE, linestyle="--", linewidth=GUIDE_WIDTH, zorder=0)
    ax.axvline(nc, color=COLOR_NC, linestyle="--", linewidth=GUIDE_WIDTH)
    ax.set_xscale("log")
    ax.set_xlim(*main_xlim)
    set_log_x_ticks(ax)
    ba_error = ensemble_error(ba_plot, "threshold")
    y_low = min(
        float(np.min(ba_plot["threshold_mean"].to_numpy(dtype=float) - ba_error)),
        float(rr_plot["threshold_mean"].min()),
    )
    y_high = max(
        float(np.max(ba_plot["threshold_mean"].to_numpy(dtype=float) + ba_error)),
        float(rr_plot["threshold_mean"].max()),
    )
    y_pad = 0.11 * (y_high - y_low)
    ax.set_ylim(y_low - y_pad, y_high + y_pad)
    ax.set_xlabel("Population size, $N$", fontsize=LABEL_FONT_SIZE, labelpad=1.5)
    ax.set_ylabel(
        "Critical benefit-to-cost ratio, $(b/c)^*$", fontsize=LABEL_FONT_SIZE, labelpad=1.5
    )
    ax.legend(
        [ba_handle, rr_handle],
        ["BA", "Regular"],
        fontsize=LEGEND_FONT_SIZE,
        loc="upper right",
        handlelength=1.6,
        borderaxespad=0.2,
        labelspacing=0.35,
    )
    style_axes(ax)
    inset_xlim = (900.0, 11200.0)
    inset = aligned_log_inset(
        ax,
        anchor_value=nc,
        main_xlim=main_xlim,
        inset_xlim=inset_xlim,
        width=0.43,
        height=0.43,
        bottom=0.34,
    )
    inset_n = {1000, 1200, 1300, 1500, 1600, 1700, 1800, 2000, 3162, 5000, 10000}
    ba_in = ba[ba["n"].isin(inset_n)].sort_values("n")
    rr_in = rr[rr["n"].isin(inset_n)].sort_values("n")
    inset.plot(
        rr_in["n"],
        rr_in["threshold_mean"],
        color=COLOR_RR,
        marker="s",
        markersize=INSET_MARKER_SIZE,
        linewidth=0.75,
        clip_on=False,
        zorder=5,
    )
    inset_ba_errorbar = inset.errorbar(
        ba_in["n"],
        ba_in["threshold_mean"],
        yerr=ensemble_error(ba_in, "threshold"),
        color=COLOR_BA,
        marker="o",
        markersize=INSET_MARKER_SIZE,
        linewidth=0.75,
        elinewidth=0.4,
        capsize=1.0,
        capthick=0.4,
        ecolor=COLOR_BA,
        clip_on=False,
        zorder=6,
    )
    for capline in inset_ba_errorbar[1]:
        capline.set_alpha(ERROR_BAR_ALPHA)
    for barline in inset_ba_errorbar[2]:
        barline.set_alpha(ERROR_BAR_ALPHA)
    inset.axhline(4.0, color=COLOR_LIGHT_GUIDE, linestyle="--", linewidth=0.45, zorder=0)
    inset.axvline(nc, color=COLOR_NC, linestyle="--", linewidth=0.55)
    inset.text(
        nc * 1.08, 4.032, "$N_c$", color=COLOR_NC, fontsize=TITLE_FONT_SIZE, ha="left", va="center"
    )
    inset.set_xscale("log")
    inset.set_xlim(*inset_xlim)
    inset.set_ylim(3.97, 4.04)
    set_log_x_ticks(inset)
    inset.tick_params(
        labelsize=INSET_TICK_FONT_SIZE, width=0.55, length=1.8, top=False, right=False
    )
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)


def simulation_error(data: pd.DataFrame) -> np.ndarray:
    column = "sd_x" if ERROR_BAR_KIND == "sd" else "sem_x"
    return data[column].fillna(0.0).to_numpy(dtype=float)


def draw_panel_b_delta_comparison(
    axes: tuple[plt.Axes, plt.Axes], plot_data: pd.DataFrame, ba: pd.DataFrame, rr: pd.DataFrame
) -> None:
    del ba, rr
    markers = {"BA": "o", "RR": "s"}
    colors = {"BA": COLOR_BA, "RR": COLOR_RR}
    axis_specs = {
        1000: {
            "axis": axes[0],
            "xlim": (3.999, 4.07),
            "xticks": [4.0, 4.02, 4.04, 4.06],
            "title": "$N=1000$",
        },
        2000: {
            "axis": axes[1],
            "xlim": (3.985, 4.035),
            "xticks": [3.99, 4.01, 4.03],
            "title": "$N=2000$",
        },
    }
    theory = plot_data[
        ["network", "N", "theory_bc_star", "sim_b50", "sim_minus_theory", "abs_error", "source"]
    ].drop_duplicates(["network", "N"])
    for n_value, spec in axis_specs.items():
        ax = spec["axis"]
        for network in ["RR", "BA"]:
            subset = plot_data[
                plot_data["network"].eq(network) & plot_data["N"].eq(n_value)
            ].sort_values("b")
            if subset.empty:
                continue
            color = colors[network]
            marker = markers[network]
            errorbar = ax.errorbar(
                subset["b"].to_numpy(dtype=float),
                subset["mean_x"].to_numpy(dtype=float),
                yerr=simulation_error(subset),
                color=color,
                marker=marker,
                markersize=MARKER_SIZE,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=0.75,
                elinewidth=ERROR_BAR_WIDTH,
                capsize=ERROR_BAR_CAPSIZE,
                capthick=ERROR_BAR_WIDTH,
                ecolor=color,
                linewidth=0,
                linestyle="none",
                label=network,
                zorder=4 if network == "BA" else 3,
            )
            for capline in errorbar[1]:
                capline.set_alpha(ERROR_BAR_ALPHA)
            for barline in errorbar[2]:
                barline.set_alpha(ERROR_BAR_ALPHA)
            threshold = float(subset["theory_bc_star"].iloc[0])
            ax.axvline(
                threshold,
                color=color,
                linewidth=THEORY_LINE_WIDTH,
                linestyle="-",
                alpha=0.45,
                zorder=1,
            )
        ax.axhline(0.5, color="#666666", linewidth=GUIDE_WIDTH, linestyle="--", zorder=1)
        ax.set_xlim(*spec["xlim"])
        ax.set_xticks(spec["xticks"])
        ax.set_ylim(0.476, 0.525)
        ax.set_yticks([0.48, 0.5, 0.52])
        ax.set_title(spec["title"], fontsize=TITLE_FONT_SIZE, pad=2.0)
        style_axes(ax)
    axes[0].set_ylabel("Cooperation level, $x$", fontsize=LABEL_FONT_SIZE, labelpad=1.5)
    axes[1].tick_params(labelleft=False)
    axes[0].legend(
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        ncol=2,
        handlelength=0.95,
        columnspacing=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
    )


def draw_panel_c_mechanism(ax: plt.Axes, panel_b: pd.DataFrame) -> None:
    nc = k4_observed_nc()
    display_n = {100, 200, 316, 500, 1000, 1600, 2000, 3162, 5000, 10000}
    data = panel_b[panel_b["N"].astype(int).isin(display_n)].sort_values("N").copy()
    eps_handle = ax.errorbar(
        data["N"],
        data["cov_over_Etau_mean"],
        yerr=ensemble_error(data, "cov_over_Etau"),
        color=COLOR_EPS,
        marker="D",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        elinewidth=ERROR_BAR_WIDTH,
        capsize=ERROR_BAR_CAPSIZE,
        capthick=ERROR_BAR_WIDTH,
        ecolor=COLOR_EPS,
        alpha=ERROR_BAR_ALPHA,
        label="$\\varepsilon$",
        clip_on=False,
        zorder=5,
    )[0]
    eps_handle.set_alpha(1.0)
    crit_handle = ax.errorbar(
        data["N"],
        data["epsilon_crit"],
        yerr=ensemble_error(data, "epsilon_crit"),
        color=COLOR_CRIT,
        marker="^",
        markersize=MARKER_SIZE,
        linewidth=LINE_WIDTH,
        elinewidth=ERROR_BAR_WIDTH,
        capsize=ERROR_BAR_CAPSIZE,
        capthick=ERROR_BAR_WIDTH,
        ecolor=COLOR_CRIT,
        alpha=ERROR_BAR_ALPHA,
        label="$\\Delta$",
        clip_on=False,
        zorder=4,
    )[0]
    crit_handle.set_alpha(1.0)
    ax.axvline(nc, color=COLOR_NC, linestyle="--", linewidth=GUIDE_WIDTH)
    ax.set_xscale("log")
    ax.set_xlim(*MAIN_N_XLIM)
    set_log_x_ticks(ax)
    visible = data[data["N"].between(100, 10000)]
    eps_upper = visible["cov_over_Etau_mean"].to_numpy(dtype=float) + ensemble_error(
        visible, "cov_over_Etau"
    )
    crit_upper = visible["epsilon_crit"].to_numpy(dtype=float) + ensemble_error(
        visible, "epsilon_crit"
    )
    y_upper = max(float(np.max(eps_upper)), float(np.max(crit_upper)))
    ax.set_ylim(0.0, y_upper * 1.1)
    ax.set_yticks([0.0, 0.004, 0.008, 0.012, 0.016, 0.02])
    ax.set_yticklabels(["0", "4", "8", "12", "16", "20"])
    ax.text(
        0.0,
        1.025,
        "$\\times 10^{-3}$",
        transform=ax.transAxes,
        fontsize=TICK_FONT_SIZE,
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Population size, $N$", fontsize=LABEL_FONT_SIZE, labelpad=1.5)
    ax.set_ylabel("Mechanistic terms for BA networks", fontsize=LABEL_FONT_SIZE, labelpad=1.5)
    ax.text(
        0.62,
        0.5,
        "$\\varepsilon\\approx 2.1\\times10^{-3}$\n$\\Delta\\approx \\frac{0.65\\ln N-1.50}{N-2}$",
        transform=ax.transAxes,
        color="black",
        fontsize=MECHANISM_FORMULA_SIZE,
        ha="left",
        va="top",
        linespacing=1.35,
    )
    ax.legend(
        [eps_handle, crit_handle],
        ["Alignment, $\\varepsilon$", "Penalty, $\\Delta$"],
        fontsize=LEGEND_FONT_SIZE,
        loc="upper right",
        handlelength=1.35,
        borderaxespad=0.2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
    )
    style_axes(ax)


def draw_panel_d_nc(ax: plt.Axes, nc_data: pd.DataFrame) -> None:
    nc_data = nc_data.sort_values("k").copy()
    y = nc_data["k"].to_numpy(dtype=float)
    pred = nc_data["predicted_Nc"].to_numpy(dtype=float)
    (x_min, x_max) = (80, 300000)
    (y_min, y_max) = (1.7, 20.7)
    y_bg = np.r_[y_min, y, y_max]
    pred_bg = 10 ** np.interp(
        y_bg, y, np.log10(pred), left=np.log10(pred[0]), right=np.log10(pred[-1])
    )
    ax.fill_betweenx(y_bg, x_min, pred_bg, color=COLOR_BA_WORSE_BG, linewidth=0, zorder=0)
    ax.fill_betweenx(y_bg, pred_bg, x_max, color=COLOR_BA_BETTER_BG, linewidth=0, zorder=0)
    ax.text(
        10**3.3,
        10.4,
        "$(b/c)^* > (b/c)^*_{r}$",
        color=COLOR_BA_WORSE_TEXT,
        fontsize=TITLE_FONT_SIZE,
        ha="center",
        va="center",
    )
    ax.text(
        45000,
        6.0,
        "$(b/c)^* < (b/c)^*_{r}$",
        color=COLOR_BA_BETTER_TEXT,
        fontsize=TITLE_FONT_SIZE,
        ha="center",
        va="center",
    )
    ax.plot(
        pred,
        y,
        color="black",
        linestyle="--",
        linewidth=LINE_WIDTH,
        label="Predicted $N_c$",
        clip_on=False,
        zorder=3,
    )
    observed = nc_data[nc_data["observed_Nc"].notna()].copy()
    observed_method = observed.get("observed_method", pd.Series("", index=observed.index)).astype(
        str
    )
    exact = observed[
        ~observed["observed_is_lower_bound"].astype(bool) & ~observed_method.eq("short_proxy")
    ]
    proxy = observed[
        ~observed["observed_is_lower_bound"].astype(bool) & observed_method.eq("short_proxy")
    ]
    if not exact.empty:
        ax.plot(
            exact["observed_Nc"],
            exact["k"],
            color="black",
            marker="o",
            linestyle="none",
            markersize=MARKER_SIZE,
            label="Observed $N_c$",
            clip_on=False,
            zorder=4,
        )
    if not proxy.empty:
        ax.plot(
            proxy["observed_Nc"],
            proxy["k"],
            color="black",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.75,
            linestyle="none",
            markersize=MARKER_SIZE,
            label="Approximated $N_c$",
            clip_on=False,
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([2, 6, 10, 14, 18])
    ax.set_xlabel("Population size, $N$", fontsize=LABEL_FONT_SIZE, labelpad=1.5)
    ax.set_ylabel("Mean degree, $k$", fontsize=LABEL_FONT_SIZE, labelpad=1.5)
    ax.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        handlelength=1.45,
        borderaxespad=0.2,
        labelspacing=0.25,
    )
    style_axes(ax)


def add_panel_label(fig: plt.Figure, ax: plt.Axes, label: str, x: float) -> None:
    pos = ax.get_position()
    fig.text(
        x,
        pos.y1 + 0.08 * pos.height,
        label,
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def load_selected_ensemble():
    """Replay the fixed sample selection and aggregate the same graphs for a/c.

    The pool contains existing valid exact calculations, not new simulations.
    Standard deviations use ddof=1 across 20 independent graph seeds per N.
    """
    pool = pd.read_csv(DATA_DIR / "sampling_pool.csv.gz", float_precision="round_trip")
    selected = pd.read_csv(DATA_DIR / "selected_networks.csv", float_precision="round_trip")
    pool = pool.sort_values(["N", "seed", "sample"]).reset_index(drop=True)
    if pool.duplicated(["N", "seed"]).any():
        raise ValueError("The sampling pool contains repeated graph identities.")
    rng = np.random.default_rng(20260915)
    replay = []
    for n, group in pool.groupby("N", sort=True):
        if len(group) < 20:
            raise ValueError(f"Fewer than 20 valid independent networks at N={n}.")
        replay.append(
            group.iloc[rng.choice(len(group), 20, replace=False)].sort_values(["seed", "sample"])
        )
    replay = pd.concat(replay, ignore_index=True)
    pd.testing.assert_frame_equal(selected, replay, check_exact=True)
    summary = (
        selected.groupby("N", as_index=False)
        .agg(
            sample_count=("threshold", "count"),
            threshold_mean=("threshold", "mean"),
            threshold_std=("threshold", "std"),
            threshold_RR=("threshold_RR", "mean"),
            threshold_delta_mean=("delta_threshold", "mean"),
            threshold_delta_std=("delta_threshold", "std"),
            cov_over_Etau_mean=("epsilon", "mean"),
            cov_over_Etau_std=("epsilon", "std"),
            epsilon_crit=("epsilon_crit", "mean"),
            epsilon_crit_std=("epsilon_crit", "std"),
            margin=("margin_M", "mean"),
            margin_std=("margin_M", "std"),
            S_N_mean=("S_N", "mean"),
            S_N_std=("S_N", "std"),
        )
        .sort_values("N")
        .reset_index(drop=True)
    )
    if not summary["sample_count"].eq(20).all():
        raise ValueError("Every population size must have exactly 20 networks.")
    for stem in ["threshold", "cov_over_Etau", "epsilon_crit"]:
        summary[f"{stem}_sem"] = summary[f"{stem}_std"] / np.sqrt(20)
    ba = summary.rename(columns={"N": "n"}).copy()
    rr = pd.DataFrame({"n": ba["n"], "threshold_mean": 4 * (ba["n"] - 2) / (ba["n"] - 8)})
    return (ba, rr, summary)


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce manuscript Figure 2 from frozen source data."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "figures",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    (ba, rr, mechanism) = load_selected_ensemble()
    sim = pd.read_csv(DATA_DIR / "simulation.csv")
    nc = pd.read_csv(DATA_DIR / "critical_sizes.csv")
    ba.to_csv(args.output_dir / "fig2_panel_a_summary.csv", index=False)
    mechanism.to_csv(args.output_dir / "fig2_panel_c_summary.csv", index=False)
    (fig, axes) = plt.subplots(2, 2, figsize=(7.204724409448819, 4.921259842519685))
    (ax_a, ax_b, ax_c, ax_d) = axes.ravel()
    panel_b_spec = ax_b.get_subplotspec()
    ax_b.remove()
    panel_b_grid = panel_b_spec.subgridspec(1, 2, wspace=0.12)
    ax_b_left = fig.add_subplot(panel_b_grid[0, 0])
    ax_b_right = fig.add_subplot(panel_b_grid[0, 1], sharey=ax_b_left)
    draw_panel_a(ax_a, ba, rr)
    draw_panel_b_delta_comparison((ax_b_left, ax_b_right), sim, ba, rr)
    draw_panel_c_mechanism(ax_c, mechanism)
    draw_panel_d_nc(ax_d, nc)
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.085, top=0.94, wspace=0.25, hspace=0.43)
    (left, right) = (ax_b_left.get_position(), ax_b_right.get_position())
    fig.text(
        (left.x0 + right.x1) / 2,
        min(left.y0, right.y0) - 0.048,
        "Benefit-to-cost ratio, $b/c$",
        fontsize=10,
        ha="center",
        va="top",
    )
    (left, right) = (ax_a.get_position(), ax_d.get_position())
    label_x = [left.x0 - 0.19 * left.width, right.x0 - 0.17 * right.width]
    for ax, label, x in zip([ax_a, ax_b_left, ax_c, ax_d], "abcd", label_x * 2):
        add_panel_label(fig, ax, label, x)
    stem = args.output_dir / "fig2"
    fig.savefig(stem.with_suffix(".png"), dpi=args.dpi)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
