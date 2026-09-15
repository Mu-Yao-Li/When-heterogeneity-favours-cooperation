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

## Figure 2: updated 20-network redraw

The author requested a random selection of 20 existing independent networks at
each population size. Panels a and c use the same selection, with fixed random
seed **20260915**. The script replays the selection from `sampling_pool.csv.gz`,
checks it against all 420 rows of `selected_networks.csv`, and computes means and
sample SD (`ddof=1`). It also exports a/c summary CSVs.

This redraw supersedes panels a/c in the supplied manuscript image, whose
plotting loader used larger available ensembles although the caption stated 20.
Panel b retains its ten-replicate simulation summaries; panel d retains the
separate critical-size analysis. The annotated scaling formula retains the
existing fitted theory.

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
