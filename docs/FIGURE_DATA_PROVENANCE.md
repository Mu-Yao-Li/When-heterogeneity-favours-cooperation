# Figure source data and verification

## Scope and figure contracts

The backend is Python/Matplotlib throughout. Each script is self-contained apart
from scientific Python packages and its bundled data. PDF text uses embedded
TrueType fonts; dense scatter/density layers are rasterized. Graph drawings use
archived coordinates and node/edge tables, not embedded screenshots.

| Figure | Scientific claim and panel logic | Composition/export |
| --- | --- | --- |
| 2 | Size can reverse the effect of BA heterogeneity: thresholds (a), simulation check (b), alignment/penalty mechanism (c), critical-size dependence on degree (d). | Quantitative grid; 183 × 125 mm canvas; PNG/PDF. |
| 3 | A bounded heterogeneity window favors cooperation: degree CV and networks (a), size/exponent phase map (b), mechanism (c), degree-dependent windows (d). | Mixed graph/quantitative panels; original 15.1 × 7.8 inch canvas retained and reduced when placed in the manuscript; PNG/PDF. |
| 4 | Peripheral bridges yield the largest improvement between two BA modules: best bridge (a), exhaustive shell comparison (b), reach changes (c), associations with reciprocity/reach products (d,e). | Graph plus quantitative panels; 183 × 145 mm canvas, cropped 9 mm below and 1.8 mm above as in the manuscript; PNG/PDF. |

## Figure 2: full archived network ensembles

Panels a/c use **all 7,700 independently seeded networks** at 21 population
sizes. For statistical stability, sampling was increased to **500 networks** at
selected sizes, including smaller N and the transition region. The remaining
archived sizes use **100 networks**. This is a per-size sampling schedule, not a
uniform cutoff in N:

| Networks per N | Population sizes N |
| --- | --- |
| 500 | 100, 200, 316, 500, 1000, 1200, 1300, 1500, 1600, 1700, 1800, 2000, 3162, 5000 |
| 100 | 300, 700, 1400, 1900, 3000, 7000, 10000 |

The sampling rationale concerns stability of ensemble estimates. It does not
change the solver or establish statistical convergence. The script uses the
complete archived pool without random subsampling, sorts rows by N/seed/sample,
and checks exact per-N counts, finite values and unique `(N, seed)` identities.
The original source validity filter requires `status=ok` and finite retained
numeric values.

`sampling_manifest.json` records the source SHA-256 digest, sampling rationale
and exact counts. `source_row` is the zero-based row index in the original CSV,
excluding the header. Private server/path fields were excluded.

| Input in `figures/data/fig2/` | Panel/use | Statistical meaning |
| --- | --- | --- |
| `sampling_pool.csv.gz` | a,c | All 7,700 valid networks. Both panels use the same graphs at each N; arithmetic means and sample SD (`ddof=1`). `S_N` is effective size, `epsilon` is alignment, `epsilon_crit` is penalty. |
| `sampling_manifest.json` | Ensemble audit | Exact population-size/sample-count schedule and original source digest. |
| `threshold_ba.csv`, `threshold_regular.csv`, `mechanism.csv` | Source-data views | 21 rows each. The script recomputes a/c from the complete pool; the regular threshold is analytic. `sample_count` records the actual ensemble size; SEM is SD/sqrt(sample_count). |
| `simulation.csv` | b | 20 parameter points, each summarizing ten stochastic replicates; `mean_x` and `sd_x` are plotted. Selection intensity 0.01 and mutation probability 0.0001. |
| `critical_sizes.csv` | d | Ten degrees; independently archived predictions and observed/approximated crossings. Filled/open symbols retain their method distinctions. |

The a/c vertical marker is computed from the full ensemble: the first sampled
N with positive mean alignment-minus-penalty that remains positive at all larger
sampled sizes. It is **N=1600**. This is a grid-based
observation, not an interpolated continuous root. Panel d retains its separate
critical-size analysis. The asymptotic formula in panel c retains its original
fit parameters.

**Caption wording for panels a/c:** Points show means and error bars show sample
standard deviations across independently generated networks. To improve
statistical stability, selected population sizes use 500 network realizations;
the remaining sizes use 100, as detailed in the source-data sampling schedule.
The same realizations are used for panels a and c.

The simulation workbook was converted to CSV, so Excel is not needed. The
manuscript reports 10^12 updates per replicate for panel b. This packaging pass
does not rerun or independently certify completion of those trajectories.

## Figure 3

| Input in `figures/data/fig3/` | Use |
| --- | --- |
| `heterogeneity.csv` | 21 exponent values at N=2000, with actual sample counts, means, and SD; panel a. |
| `network_gamma*_nodes.csv`, `network_gamma*_edges.csv` | Three illustrative 200-node, 400-edge graphs and fixed coordinates at exponents 0.5, 1, 1.5. |
| `phase.csv.gz` | 13,715 aggregate N/exponent rows, sample counts, threshold advantage, eligibility flags, and residual diagnostics; panel b. |
| `mechanism.csv` | 21 exponent values at N=10000, ten networks per value; panel c. |
| `windows.csv` | Six approximate favorable intervals at N=10^7 for k=4,6,8,10,20,30; original 0.001 exponent refinement retained. |

Panel a plots means and SD bands. Panel b preserves the `used_for_phase` filter,
interpolation limits, Gaussian smoothing, and zero boundary estimated from raw
sign crossings. These are the original presentation operations. Reading CSV
floats with `float_precision="round_trip"` is essential: it preserves exact
exponent keys during pivot/reindex and prevents dropped rows and artificial gaps.
Panel c retains the original signed-log axis treatment. Panel d uses all six
intervals, including k=8; approximate intervals are not new exact calculations.

## Figure 4

| Input in `figures/data/fig4/` | Use |
| --- | --- |
| `bridges.csv.gz` | All 500,500 pairs `0 <= u <= v < 1000`, with threshold, regular benchmark, shell class, reciprocity product, and normalized reach product. |
| `best_bridge.csv` | Minimum-threshold endpoints: local node 925 in each module. |
| `shell_groups.csv` | Ten shell-pair summaries from the full enumeration; original ordering and mean markers. |
| `reach_changes.csv` | 4,000 rows: 2,000 nodes for each of two bridge cases; panel c. |
| `module_nodes.csv`, `module_edges.csv` | 1,000 nodes and 2,000 edges; shell/hub assignments and unrotated coordinates. Module 2 is an identical copy. |

Boxes show the interquartile range and median, with the original 1.5-IQR
whiskers; outliers are not drawn separately. Diamonds show means, stars minima,
and inset triangles bridge nodes. Correlations use all candidates. Displayed
same-shell scatter points are deterministically subsampled to at most 1,200 per
class; other classes use a density background. This display operation does not
affect boxplots, minima, or correlations.

## Workflow versions and integrity

Confirmed source entry points, relative to the working project:

- Figure 2: `plot_figure2_abc_threshold_simulation_nature.py`; final four-panel
  order retained, a/c aggregated from the complete archived network ensembles.
- Figure 3: `plot_figure3_compact_abcd_pnas_b_equalheight_d_N1e7.py`, **all6**
  variant. The recommended5 variant omits k=8 and differs from the manuscript.
- Figure 4: `plot_figure4_panelb_box_softened_horizontal_trial.py`; its horizontal
  configuration differs from the underlying unconfigured slanted layout.

Only necessary plotting functions were retained. Directory-scanning and absolute
path loaders were replaced by bundled source data. `figures/data_manifest.json`
records the byte size, row count, and SHA-256 digest of every public input file.

## Validation

- All three scripts produced PNG/PDF in the clean Python 3.12 scientific
  environment, without imports or files from the original working project.
- Figure 2 uses all 7,700 rows, with 500 unique seeds at 14 sizes and 100 at
  seven sizes. Panels a/c share these rows; source-data means, sample SD and
  count-aware SEM agree with independent NumPy calculations.
- Figure 3's layout, six interval rows, illustrative graphs, and phase boundary
  were visually compared with the manuscript. Exact CSV float parsing fixed an
  artificial boundary gap without changing scientific values.
- Figure 4's horizontal layout, five panels, shell colors, best bridge, and
  displayed Spearman coefficients (-0.64 and 0.43) match the manuscript visually.
- The revised Figure 2 was checked for intact error bars, text, inset placement,
  and the full-ensemble aggregation behavior.
- Scripts and input tables were checked for private absolute paths and server
  addresses. Font fallback may produce minor text-metric changes on other systems.

The large production calculations were not rerun. This validation establishes
portable plotting and reproducible aggregation of archived results, not
independent replication of every original experiment.
