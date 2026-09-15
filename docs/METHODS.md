# Mapping between manuscript notation and code

Reference: supplied 32-page manuscript dated 15 September 2026, title
"When heterogeneity favours cooperation". Supplementary Information was not
included in that file.

| Manuscript | Implementation |
| --- | --- |
| Eq. 5, pairwise meeting time `tau_ij` | `2 * eta_ij` from `solve_eta_single_layer_linear_operator` |
| Reach `tau_i` | `1 + 2 * sum_j p_ij eta_ij` |
| Reciprocity `p_i` | `sum_j p_ij p_ji` |
| Eq. 7, `N_eff` | `S_N` in grid outputs |
| Eq. 8, alignment | `epsilon_N` |
| Eq. 10, finite-size penalty | `epsilon_crit` |
| `epsilon - Delta` | `margin_M` |
| Eq. 6 threshold | `(S_N - 2) / (R_N - 2)` |
| Eq. 15-17 approximation | `run_adaptive_closed_tail_pipeline.py` |

The factor of two in `eta` is deliberate: the existing solver uses a right-hand
side of 0.5, while manuscript Eq. 5 uses 1. The independent test constructs the
Eq. 5 linear system directly to verify this convention.

The new `algorithms/exact.py` entry point solves directly for `tau_ij` with
right-hand side 1. It reports `N_eff`, `epsilon` and `Delta` under those names.
`algorithms/approximate.py` reuses the production short-walk/closure functions
and exposes the same three network-level quantities as estimates.

## Updating rules and payoff aggregation

All three rules choose a focal updater uniformly. DB samples one of its neighbours
in proportion to fitness; IM samples from the neighbours plus the focal individual.
PC chooses one uniform neighbour and copies with probability `F_j/(F_i+F_j)`.
Average and accumulated payoffs are respectively `b*P*x-c*x` and `b*A*x-c*k*x`.
Simulation fitness is `exp(delta*f)`; its derivative at zero selection is the
same as the linear fitness expansion used by the exact method.

The exact selection condition is
`b*benefit_coefficient + c*cost_coefficient > 0`, the first-order criterion for
`rho_C > rho_D` with uniformly located single mutants. It directly evaluates the
rare-mutation limit, without substituting a small positive mutation rate. DB
and PC share the neutral walk `P`; IM uses `(D+I)^(-1)*(A+I)`. Full definitions,
adaptation history and mathematical attribution are in
`algorithms/exact_support/PROVENANCE.md`.

The approximation currently supports DB/average only. It draws `R=1000`
trajectories per node by default, with `L=ceil(2/(1-lambda_2))`, and determines
the common unresolved tail using `sum_i pi_i^2*tau_i=1`. Optional supplementation
uses the original diagnostic constants and records them in each output row.
It saves no node-level arrays. Small tail-denominator uncertainty and exact
identity closure do not establish a confidence bound for the final threshold.

Simulation samples every post-update state, including events with no strategy
change. Mutation is uniform C/D replacement with probability `mu`, giving a
strategy-flip probability of `mu/2`. Replicate uncertainty is calculated from
replicate means; no automatic claim of equilibration or mixing is made.

## Threshold interpretation

The signed algebraic threshold can be negative when the denominator is negative.
The primary exact entry point preserves it as `bc_star_signed` and reports
`bc_star=null` in JSON (an empty field in CSV) when an ordinary positive lower
threshold is undefined. The approximation also leaves an invalid positive
threshold blank in CSV. The older `code/analyze_network.py` wrapper uses `inf`
instead. Check `threshold_status` and coefficient signs; a negative formal root
does not by itself indicate cooperation promotion.

The regular benchmark is `(N-2)/(N/k-2)` and is only used as a positive-threshold
comparison when `N/k-2 > 0`. The wrapper reports `benchmark_defined` separately.
`heterogeneity_advantage` requires both thresholds to be positive and finite.

## Graph construction details to preserve

The PA grid uses `build_pa_degree_power_exact`: a complete seed, distinct
attachments weighted by `degree**gamma`, and supplemental edges to reach exactly
`N*k/2`. Supplemental endpoints are also degree-power weighted. The manuscript
says these edges are added "at random"; its Supplementary Information should
specify that weighting if this is the final production generator.

`single_layer_threshold.py --network ba` uses a different legacy BA generator.
Use `--network pa-gamma --gamma 1` or the PA grid for the manuscript's PA convention.
The ER grid uses a random Hamiltonian-path backbone plus uniformly sampled missing
edges to enforce connectivity and exact mean degree. It is an ER-like connected
construction, not an unconditional draw from `G(N,p)` or `G(N,m)`.
The additional generator module has its own family definitions; select production
conventions after checking the corresponding Supplementary Information.

## Validation scope

The small tests check the linear-system equation independently, analytic regular
benchmarks, the remeeting identity, the threshold decomposition, negative-threshold
handling, and deterministic graph generation. The quick start and a small grid
exercise end-to-end execution. These checks validate packaging and small numerical
cases; they do not establish every scientific claim or recreate production runs.
