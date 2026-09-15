# Manuscript figure map

This map follows the supplied 32-page PDF dated 15 September 2026. Source names
below refer to the author's working project; they are **candidate matches**, not
included or verified figure-reproduction commands. A filename match alone does
not establish that a plot is the version embedded in the PDF.

| Figure | Manuscript content | Candidate source / remaining work |
| --- | --- | --- |
| Main 1 | Model and effective-size/alignment mechanism | `redraw_heterogeneity_figure1.py`; compare with final composed artwork |
| Main 2 | Threshold vs size; simulations; alignment/penalty; critical-size boundary | `plot_figure2_abc_threshold_simulation_nature.py` (modified 14 Sep 2026); replace absolute spreadsheet path and freeze all imported data loaders |
| Main 3 | PA degree CV, size/gamma phase map, mechanism, favourable windows at N=10^7 | `plot_figure3_compact_abcd_pnas_b_equalheight_d_N1e7.py`; confirm composed output and complete imported input tables |
| Main 4 | Peripheral bridges between BA modules | `plot_two_ba_bridge_abcd_nature.py`; freeze exact bridge enumeration and reach-change tables |
| Extended Data 1 | BA/ER node-level alignment | `plot_extended_local_epsilon_ba_er.py`; compare node samples and source graph seeds |
| Extended Data 2 | Large BA threshold advantage | PA grid and finite-size scaling analysis; exact plotting entry point remains unconfirmed |
| Extended Data 3 | Approximation accuracy and runtime | `plot_appendix_s1_adaptive_relaxation_validation.py`; verify which benchmark snapshot generated the supplied panels |
| Extended Data 4 | Selection-intensity robustness | Simulation code/configuration in the separate `sigma` work area; final program and input tables remain unconfirmed |
| Extended Data 5 | Accumulated-payoff robustness | Simulation code/configuration in `sigma`; exact update rule, mutation and averaging settings must be matched |
| Extended Data 6 | Ten synthetic families, scaling and regime counts | `plot_supp_generated_10model_*` family; identify final composite and exact/approximate data boundary |
| Extended Data 7 | Empirical examples, 335,341 positive-threshold networks, null models | `plot_extended_empirical_collection_audit.py` and later figure composition; verify filtering against 910 realized networks and null-model tables |
| Extended Data 8 | Empirical bridges, 12,000 module pairs | `plot_regime_pair_interlink_exhaustive_summary.py` plus empirical module drawings; verify all 9,296,432 candidate bridges and layout |
| Extended Data 9 | Internal peripheral-shell deletions | `plot_extended_fig9_ba_shell_deletion.py`; freeze deletion and threshold data |
| Extended Data 10 | Public-goods thresholds and simulations | `pgg_threshold.py`, `plot_pgg_mutation_steady_delta0p1_bc_theory_3x4.py`; verify theory/sample conventions against final figure |

## Scope of the currently included numerical core

The exact solver and PA grid support the theory behind Main 2-3. The approximation
pipeline implements the method described on manuscript pages 16-17. Additional
generators and public-goods code are supplied, but do not by themselves reproduce
all ensemble/simulation panels. Original simulation code may include third-party
components that need their existing licence and attribution retained when bundled.

## Known reproduction differences to resolve

- The June Figure 2 handoff uses a different panel order from the supplied PDF.
  The September script is the better candidate and still requires data freezing.
- Figure 2's simulation source uses an absolute local Excel path. Its manuscript
  settings are `delta=0.01`, `mu=1e-4`, `10^12` updates and ten replicates.
- The Methods describe other simulations with `mu=1e-3`, `10^10` updates and no
  burn-in. Audit actual production configurations before claiming those settings.
- Supplementary Information is referenced by the PDF but is not part of it.
- Each final figure needs source-table hashes, graph seeds/identifiers, a portable
  command and an output comparison to the submitted figure.
