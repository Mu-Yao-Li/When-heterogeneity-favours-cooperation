# Reproduce Figures 2–4

Each figure has one independent Python entry point and its own bundled source
data. Install the repository requirements, then run:

```bash
python figures/reproduce_fig2.py
python figures/reproduce_fig3.py
python figures/reproduce_fig4.py
```

Each command writes `figN.png` and `figN.pdf` to `outputs/figures/`. Use
`--output-dir PATH` to change the output directory and `--dpi 600` for a
higher-resolution PNG. Input paths are relative to the script, so each command
also works from another working directory.

The commands redraw archived numerical results. They do not rerun the large
network ensembles, exhaustive bridge search, or long stochastic simulations.
Keep the accompanying `figures/data/figN/` directory with each script.

## Figure 2: full archived ensembles

Panels a and c use **500 independent networks at every plotted N**, including
N=10000 and the panel-a inset, to improve statistical stability. The source data
combine the unchanged 7,700-row `sampling_pool.csv.gz` with 400 additional
N=10000 networks in `supplemental_n10000.csv`. The complete 8,100-row archive
also retains six unplotted diagnostic sizes with 100 networks each. The exact
schedule is recorded in `sampling_manifest.json` and
[the provenance table](../docs/FIGURE_DATA_PROVENANCE.md). No random subsampling
is applied.

The script checks per-N counts, finite values and unique graph seeds, then
requires 500 networks at every plotted size, then computes means and sample SD
(`ddof=1`) from the same graphs for a/c. It exports
summary CSVs including `sample_count`; SEM uses SD divided by the square root of
the actual count. Panel b retains its ten-replicate simulation summaries; panel d
retains the separate critical-size analysis. The annotated scaling formula
retains the existing fitted theory.

## Figure 3

The four panels include all six mean degrees `k=4,6,8,10,20,30` in panel d.
Network coordinates are frozen to avoid layout changes across NetworkX versions.
The phase data use exact floating-point round-trip parsing because exponents are
used as grid keys.

## Figure 4

The figure has five panels, a–e. Its source includes all **500,500 unordered
bridge endpoint pairs** for two identical 1,000-node modules. Boxplots, minima,
and correlations use the full data; the display retains the original
deterministic scatter subsampling and density background.

See [data provenance](../docs/FIGURE_DATA_PROVENANCE.md) for statistics, workflow
versions, and verification. `data_manifest.json` records input hashes and row
counts. Arial is preferred, with DejaVu Sans as the portable fallback; small
text-metric differences across operating systems are expected.
