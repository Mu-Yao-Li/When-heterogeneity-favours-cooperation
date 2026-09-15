# Exact-algorithm provenance

The exact entry point uses the project's local rare-mutation implementation,
separately from its finite-mutation identity-by-state code. These source files
were inspected on 15 September 2026:

| Local research source | Published adaptation | Original SHA-256 |
| --- | --- | --- |
| `sigma/sigma/im_rare_mutation.py` | `coalescence.py` (validation and neutral solver only) | `0eb0401ecd73a372557d7f2d565bb697ec5f73a0901cb8ef2941e73153839248` |
| `sigma/sigma/rare_mutation_thresholds.py` | `thresholds.py` | `8207e40dfde79b664a95d418ed7ec8688281ed0c81b938dd56c3e92bd16c4f91` |
| `sigma/tests/test_rare_mutation_thresholds.py` | `tests/test_exact_rules.py` (independent full-state chain) | `3c1ae8167e4977e70a1525ec67acce3287812f306719394d52aa1bf1f69bbb03` |

All three were untracked local additions in the research checkout whose remote
is [Alex McAvoy's sigma repository](https://github.com/alexmcavoy/sigma). That
checkout's README attributes its original structure-coefficient package to
Alex McAvoy and John Wakeley. The three local additions are not identified as
upstream files. The checkout's `LICENSE.md` was empty; no MIT, GPL, or other
license grant is inferred from it. This provenance note makes no new license
grant or claim about individual code authorship.

Packaging changes include portable imports and a command-line interface,
rejection of weighted NetworkX inputs, DB average-payoff summary statistics,
explicit signed-zero-crossing versus positive-threshold fields, and strict JSON
serialization of undefined values as `null`.

## Mathematical source and conventions

The neutral occupation/coalescence representation follows Alex McAvoy and
Benjamin Allen, *Fixation probabilities in evolutionary dynamics under weak
selection*, Journal of Mathematical Biology 82, 14 (2021),
[doi:10.1007/s00285-021-01568-4](https://doi.org/10.1007/s00285-021-01568-4);
[author manuscript](https://arxiv.org/abs/1908.03827).

The implemented criterion is the derivative of `rho_C - rho_D` for a uniformly
placed single mutant in a finite connected simple unweighted undirected graph.
It is evaluated at zero selection intensity and in the rare-mutation limit.
It does not approximate that limit with a small positive mutation rate.

For average payoffs, `H=P` and `s=1`. For accumulated payoffs, `H=W` and `s=k`.
Define `T(M)=sum_ij pi_i M_ij tau_ij`. DB uses ancestry `Q=P`; IM uses
`Q=(D+I)^(-1)(W+I)`. Their cost and benefit contractions are
`T(Q^2 diag(s))` and `T(Q^2 H)-T(H)`. PC uses `Q=P` and replaces `Q^2`
by `P` in these contractions. Its half-lazy comparison step is incorporated
in the fixation-derivative scale. The corresponding coefficients are divided
by population size and tested against an independent absorbing Markov chain
for all six combinations.

The solver uses `tau_ij = 1 + (sum_l Q_il tau_lj + sum_l Q_jl tau_il)/2`
for distinct nodes, and `tau_ii=0`. Thus its `tau` is twice the legacy DB
solver's `eta`. It retains an `N` by `N` matrix in memory and is intended for
graphs small enough for this exact calculation; the separate approximation
entry point serves large DB average-payoff networks.
