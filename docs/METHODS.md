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

## Threshold interpretation

The signed algebraic threshold can be negative when the denominator is negative.
The wrapper preserves it as `threshold_signed` and reports `threshold=inf` if
the weak-selection inequality has no positive critical ratio (up to numerical
tolerance). Such cases must not be labelled cooperation-promoting simply because
their signed threshold is less than a positive regular benchmark.

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
