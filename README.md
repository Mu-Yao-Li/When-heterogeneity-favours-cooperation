# When heterogeneity favours cooperation

Research code accompanying the manuscript by Muyao Li, Juyi Li, Benjamin Allen,
and Qi Su (15 September 2026).

Repository: [Mu-Yao-Li/When-heterogeneity-favours-cooperation](https://github.com/Mu-Yao-Li/When-heterogeneity-favours-cooperation)

The repository provides exact thresholds, a large-network approximation,
evolutionary simulations, network generators, and a separate reproduction script
for each of Figures 2, 3 and 4.

## Start here

| Task | Entry point | Default / scope |
| --- | --- | --- |
| Exact threshold | `algorithms/exact.py` | DB, average payoff; also PC, IM and accumulated payoff |
| Approximate threshold | `algorithms/approximate.py` | DB, average payoff; spectral cutoff and common-tail closure |
| Evolutionary simulation | `algorithms/simulate.py` | DB, average payoff; also PC, IM and accumulated payoff |
| Generate a network | `algorithms/generate_network.py` | Seeded synthetic network families and PA exponent sweeps |
| Reproduce Figure 2 | `figures/reproduce_fig2.py` | Reads the bundled figure source data |
| Reproduce Figure 3 | `figures/reproduce_fig3.py` | Reads the bundled figure source data |
| Reproduce Figure 4 | `figures/reproduce_fig4.py` | Reads the bundled figure source data |

The exact and approximate thresholds concern **weak selection and the rare-mutation
limit**. Simulations use explicit, finite selection strength and mutation rate.
The graph model is connected, undirected and unweighted, without self-loops.

## Install

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records the validation environment. `requirements.txt`
allows compatible dependency versions. Julia production simulations have separate
instructions in `algorithms/simulation_support/production_julia/`.

## 1. Exact algorithm

```bash
# Default: DB, average payoff
python algorithms/exact.py --edge-file examples/pa_n100_k4_seed42.edgelist --output outputs/exact_db_average.csv

# Change the updating rule and payoff aggregation independently
python algorithms/exact.py --edge-file examples/pa_n100_k4_seed42.edgelist --update-rule PC --payoff average --output outputs/exact_pc_average.csv
python algorithms/exact.py --edge-file examples/pa_n100_k4_seed42.edgelist --update-rule IM --payoff accumulated --output outputs/exact_im_accumulated.csv
```

All six combinations of `DB|PC|IM` and `average|accumulated` are supported.
Average payoff is `f_i = -c*x_i + b*sum_j A_ij*x_j/k_i`; accumulated payoff is
`f_i = -c*k_i*x_i + b*sum_j A_ij*x_j`.

The result preserves the signed algebraic root and the direction of the selection
inequality. A negative root or a rule that never favours cooperation at positive
`b/c` must not be interpreted as a cooperation-promoting threshold. The DB/average
case additionally reports the manuscript's effective size and structural metrics.
See [methods](docs/METHODS.md) for definitions and solver conventions.

Exact calculations require quadratic memory in the number of nodes. Start with
the included 100-node graph before attempting larger networks.

All graph-input entry points share an edge-list reader: whitespace or comma
separators, optional column headers, and `#`/`%` comments are accepted. Duplicate
edges are merged and self-loops ignored. Disconnected graphs are rejected;
largest-component selection must be performed and documented separately.

## 2. Approximation algorithm

```bash
python algorithms/approximate.py --edge-file examples/pa_n100_k4_seed42.edgelist --network-id pa-demo --output outputs/approximate.csv
```

The default uses `R=1000` trajectories per node and `L=ceil(2/spectral_gap)`.
For optional targeted supplementation, add `--max-samples 5000`. The implementation
records the seed, file checksum, trajectory count, spectral residual and tail
diagnostics. It writes one network-level CSV row.

This approximation applies to **DB with average payoff**. Its tail diagnostic is
not a confidence interval for the final threshold. A negative-tail result is
marked invalid, and identity closure alone does not establish accuracy.

## 3. Evolutionary simulation

```bash
python algorithms/simulate.py --edge-file examples/pa_n100_k4_seed42.edgelist --benefit 5 --cost 1 --steps 10000 --replicates 2 --seed 42 --output outputs/simulation.csv
```

Use `--update-rule DB|PC|IM` and `--payoff average|accumulated` to select the model.
DB chooses an updater uniformly and a neighbour in proportion to fitness.
IM includes the updater itself among the competing candidates. PC chooses a
neighbour and copies with probability `F_j/(F_i+F_j)`, where `F_i=exp(delta*f_i)`.
With mutation probability `mu`, the updater instead adopts a uniformly random
cooperator/defector strategy. The default has no burn-in.

The Python entry point is a readable reference for small experiments. The
production Julia implementation is included for long simulations. The manuscript's
Figure 2 simulation used `delta=0.01`, `mu=0.0001`, `10^12` updates per replicate
and ten replicates; the short command above demonstrates execution and does not
rerun those production trajectories. Frozen plot data provide quick figure
reproduction.

## 4. Network generation

```bash
python algorithms/generate_network.py --model PA --nodes 100 --degree 4 --gamma 1 --seed 42 --output outputs/pa.edgelist
python algorithms/generate_network.py --model RR --nodes 100 --degree 4 --seed 42 --output outputs/rr.edgelist
```

The generator also exposes ER, SW, BA, HK, KE, Shifted, FF, IslandBA, IslandER
and Core-periphery families. It writes an edge list and parameter metadata,
including the realised node count and mean degree. See `--help` for family-specific
parameters. PA uses the manuscript's degree-power attachment convention; the
separate BA family retains the historical generator used in network comparisons.

## 5. Reproduce Figures 2, 3 and 4

```bash
python figures/reproduce_fig2.py --output-dir outputs/figures
python figures/reproduce_fig3.py --output-dir outputs/figures
python figures/reproduce_fig4.py --output-dir outputs/figures
```

Each command reads its own bundled source data and writes PNG/PDF files. It does
not require access to the authors' original project directory. Source data,
panel mapping and manuscript-version details are documented in
[FIGURE_MAP.md](docs/FIGURE_MAP.md) and
[FIGURE_DATA_PROVENANCE.md](docs/FIGURE_DATA_PROVENANCE.md).

Figure 2 panels a/c use the authors' requested random selection of 20 networks
per size (seed 20260915), with the selected records saved for audit. This revised
Figure 2 should replace the earlier manuscript figure. The other panels retain
their separately documented simulation and critical-size data.

## Validation and supporting files

```bash
python -m unittest discover -s tests -v
python examples/quickstart.py
```

Small exact-state checks cover the updating rules and payoff conventions;
coalescence tests check independent equations, regular-network benchmarks and
invariants. [VALIDATION.md](docs/VALIDATION.md) records the checks actually run.

`code/` contains shared numerical implementations and optional batch-grid tools.
`algorithms/` provides the primary user-facing entry points. The fixed example
network is in `examples/`; temporary calculation and plotting outputs go to
`outputs/`, which is ignored by Git.

## Data, citation and licence

Figure source data and one small demonstration network are bundled. The full
empirical network collection, manuscript PDF, Figure 1 and Extended Data
reproduction pipelines are outside this package's current scope; see
[DATA_SOURCES.md](docs/DATA_SOURCES.md) and [release status](docs/RELEASE_STATUS.md).

Please cite the associated manuscript using `CITATION.cff`. No publication DOI
has been assigned in this metadata. The authors have not yet selected a software
licence; repository visibility alone does not grant an open-source licence.
