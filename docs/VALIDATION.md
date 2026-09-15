# Numerical-core validation

Validated locally on Windows with Python 3.12.14 on 15 September 2026, using a
new virtual environment and the versions recorded in `requirements-lock.txt`.

## Checks actually run

| Check | Result |
| --- | --- |
| `python -m unittest discover -s tests -v` | 6 tests passed |
| `python examples/quickstart.py` | Both fixed networks completed |
| Exact edge-list CLI in README | Completed, matching quick-start PA result |
| Approximation command in README, R=1000 | status=ok, L=10, no supplemental nodes |
| README PA grid, 3 gamma values x 2 seeds | 6 successful rows |
| README ER-like grid, 2 seeds | 2 successful rows |
| Clean clone of the local Git commit | All 7 source hashes matched; all 6 tests and quick start passed |

The tests independently assemble the pairwise-coalescence linear system, compare
regular-network results with analytic thresholds, check the remeeting identity and
the threshold decomposition, reject disconnected inputs, and ensure negative
signed thresholds are not misclassified as cooperation advantages.

## Example output

| Network | N_eff | epsilon | Exact (b/c)* |
| --- | ---: | ---: | ---: |
| Fixed PA, N=100, k=4 | 49.80984996 | 0.00237606 | 4.52281635 |
| Degree-4 ring, N=100 | 100.00000000 | 0.00000000 | 4.26086957 |

The approximation on the same PA example gave `(b/c)* = 4.5271265`, a relative
difference of about 0.0953%. This is one small packaging example, not a general
accuracy bound or a replacement for the manuscript's approximation benchmarks.

Additional network-family generation, public-goods production grids, long
agent-based simulations, empirical preprocessing and final plotting pipelines
have not been independently validated in this draft. Hosted CI status should be
read from the repository's Actions page; local checks do not establish a hosted run.
