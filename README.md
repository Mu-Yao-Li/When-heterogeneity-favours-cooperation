# When heterogeneity favours cooperation

Research code accompanying the manuscript by Muyao Li, Juyi Li, Benjamin Allen,
and Qi Su (manuscript dated 15 September 2026).

**Preparation status:** this initial repository version contains the numerical
core and small reproducible examples. It is not yet a complete reproduction
archive for every manuscript figure. See [release status](docs/RELEASE_STATUS.md).

Repository: [When-heterogeneity-favours-cooperation](https://github.com/johnston0603-stack/When-heterogeneity-favours-cooperation)

## What this code computes

For connected, undirected, unweighted networks, the main analysis considers
death-birth updating, averaged donation-game payoffs, and weak selection.
The code solves pairwise coalescence times and computes effective size
`N_eff`, reach-reciprocity alignment `epsilon`, the finite-size penalty `Delta`,
and the critical benefit-to-cost ratio `(b/c)*`.

The large-network approximation combines short random-walk simulations with
a common tail fixed by the remeeting identity. The public-goods implementation
and additional network generators are included as supporting code, with their
validation status listed separately.

## Install

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records the versions from the validation environment.
`requirements.txt` gives the supported dependency ranges for a fresh resolution.

## Quick start

```bash
python -m unittest discover -s tests -v
python examples/quickstart.py
```

The example compares a fixed 100-node preferential-attachment graph with a
100-node, degree-4 ring lattice. It writes network-level results to
`outputs/quickstart.csv`. It does not reproduce the paper's full sample ensembles.
The included edge list fixes the example network independently of future changes
to random-number generators.

To analyse your own connected graph:

```bash
python code/analyze_network.py --edge-file examples/pa_n100_k4_seed42.edgelist --output outputs/exact.csv
```

The edge-list reader accepts integer/string labels, whitespace or comma separators,
optional `source target` headers, and `#`/`%` comments. Duplicate edges are merged
and self-loops ignored. Disconnected graphs are rejected. This command does not
silently select a largest component; empirical preprocessing must be recorded.

## Large-network approximation

```bash
python code/run_adaptive_closed_tail_pipeline.py --edge-file examples/pa_n100_k4_seed42.edgelist --network-id pa-demo --output-dir outputs/approximation --initial-samples 1000 --maximum-samples 1000 --no-node-output
python tests/check_approximation_output.py outputs/approximation
```

This command uses the manuscript's fixed `R=1000` setting. The implementation
also supports targeted additional sampling: `--maximum-samples 5000` enables
the existing diagnostic-driven supplementation. The short simulation length
is `ceil(2/spectral_gap)`; this is an approximation and can be expensive on
slowly mixing graphs. Inspect status, uncertainty, and negative-tail diagnostics.
Identity closure alone is not evidence of approximation accuracy.

## Small parameter grids

```bash
python code/run_pa_gamma_metric_grid.py --n-values 100 --k-values 4 --gamma-values 0.5,1,1.5 --sample-start 0 --sample-stop 2 --workers 1 --output outputs/pa_grid.csv
python code/run_er_metric_grid.py --n-values 100 --k-values 4 --sample-start 0 --sample-stop 2 --workers 1 --output outputs/er_grid.csv
```

The grid programs have large defaults: supply all ranges explicitly. Exact solves
store a dense pair-state result and require quadratic memory; start small.
For multi-process runs, limit BLAS threads (`OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) to avoid oversubscription.
Inspect every CSV row's `status`, `error`, and `relative_residual` before analysis.

## Repository contents

| Path | Purpose |
| --- | --- |
| `code/analyze_network.py` | Portable entry point for exact analysis |
| `code/sparse_threshold.py` | Existing matrix-free coalescence solver |
| `code/single_layer_threshold.py` | Single-layer calculations and graph generators |
| `code/run_pa_gamma_metric_grid.py` | PA parameter grids and structural metrics |
| `code/run_er_metric_grid.py` | Connected ER-like parameter grids |
| `code/run_adaptive_closed_tail_pipeline.py` | Spectral cutoff, Monte Carlo and closed-tail approximation |
| `code/generative_network_models.py` | Additional synthetic network families |
| `code/pgg_threshold.py` | Supporting public-goods threshold implementation |
| `examples/` | Fixed small graph and a runnable example |
| `tests/` | Independent coalescence reference, analytic and invariant checks |
| `docs/` | Equation mapping, provenance, figure map and release status |

The copied scientific implementations are unchanged; hashes are recorded in
[source_manifest.json](docs/source_manifest.json). The new wrapper and tests
are publication-preparation additions.

## Data, attribution and licence

This draft includes one generated demonstration network. Empirical network files,
full production results, and the manuscript PDF are not bundled. Original sources
and the remaining empirical reproduction work are recorded in
[DATA_SOURCES.md](docs/DATA_SOURCES.md).

`CITATION.cff` currently lists the manuscript authors; code authorship and
third-party attribution must be reviewed before release. A licence has not yet
been selected by the authors. Do not interpret this draft as a licence grant.
