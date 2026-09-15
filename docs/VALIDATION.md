# Validation

Validated locally on Windows on 15 September 2026, using Python 3.12.14 and
the packages pinned in `requirements-lock.txt`.

## Numerical algorithms and interfaces

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | 18 test methods passed |
| Exact rule/payoff reference | 30 small-graph combinations match independently assembled full-state absorbing Markov-chain derivatives |
| Regular-network and DB identities | Analytic thresholds, meeting-time normalization, remeeting identity and decomposition passed |
| Network generators | All 12 exposed models generated connected N=100, k=4 graphs with 200 edges |
| Python simulation | Six rule/payoff kernels match independent one-step transition probabilities; incremental counts and holds checked |
| README generation, exact, approximation and Python simulation commands | Completed successfully |
| Julia 1.13.0 backend self-test | Both groups passed: 180,030 and 6,027 assertions |
| Python-to-Julia command | N=100, DB/average, 1,000 updates x 2 replicates; CSV, summary and partial trace block checked |

The exact tests cover DB, PC and IM independently with both payoff aggregations,
including finite positive thresholds, negative formal roots and invalid graph
inputs. The update-kernel tests compare probabilities, not convergence of a
short simulated trajectory to its stationary distribution.

## Fixed demonstration network

| Calculation | Result |
| --- | ---: |
| Exact DB, average payoff: N_eff | 49.80984996 |
| Exact DB, average payoff: epsilon | 0.00237606 |
| Exact DB, average payoff: (b/c)* | 4.52281635 |
| Approximation, R=1000, seed=20260823, network ID pa-demo | 4.5271265 |
| Approximation relative difference from exact | about 0.0953% |
| Exact PC, accumulated payoff: (b/c)* | 14.66422978 |
| Exact IM, average payoff: (b/c)* | 7.98832660 |

The approximation example had cutoff L=10. The observed error is one packaging
example, not a general accuracy bound.

The earlier batch-grid checks remain applicable to the unchanged shared modules:
six PA rows (three exponents, two seeds) and two ER-like rows completed with
successful status and small solver residuals. Original copied-module hashes are
recorded in `source_manifest.json`.

## Figure reproduction

All three standalone figure commands completed successfully, producing PNG/PDF
files in the repository output directory. Visual inspection checked the layouts,
labels and plotted summaries. `python tests/check_figure_inputs.py` verified all
24 bundled input files against their SHA256 digests and CSV row/column counts.

Each figure has an independent script and bundled inputs. Figure 2's sampling
pool and selected records allow its 20-network selection to be replayed.
Panel means and standard deviations are recomputed from those records.
Figure-specific version selection and visual checks are recorded in
`FIGURE_DATA_PROVENANCE.md`.

## Limits

The full production ensembles and 10^12-update simulation trajectories were
not rerun. The full empirical network archive, Figure 1, Extended Data and
Supplementary Information are not validated by these checks. A short smoke
test establishes that a program executes, not statistical convergence.

Local validation and hosted GitHub Actions are separate. The workflow runs the
Python test suite, four primary numerical entry points, and all three figure
scripts. Read the repository's Actions page for the outcome of a particular
published commit.
