# Release scope

This package is organized around exact calculations, an approximation, simulations,
network generation, and independent plotting entry points for Figures 2, 3 and 4.

## Included

- Exact rare-mutation weak-selection thresholds for DB, PC and IM, with average
  and accumulated payoffs. DB/average is the default.
- The DB/average spectral-cutoff approximation, with fixed R=1000 by default.
- A readable Python simulation and an extracted Julia production backend,
  using the same six rule/payoff definitions.
- Seeded network generators, including the manuscript PA convention and the
  historical synthetic network-family implementations.
- Frozen plotting inputs and one script per requested main figure.
- A fixed demonstration network, dependency lock, numerical tests and CI.

Figure 2 panels a/c were recompiled during repository preparation using 20
randomly sampled valid networks for each of 21 sizes, as instructed by the
authors. The seed and selection records are included. This is a **revised Figure 2**
and should replace the earlier figure in the manuscript. Its simulation and
critical-size panels use their separately documented archived data.

See [VALIDATION.md](VALIDATION.md) for checks actually performed and
[FIGURE_DATA_PROVENANCE.md](FIGURE_DATA_PROVENANCE.md) for panel-level sources.
The short validation commands do not rerun the manuscript's long production
simulations or large parameter ensembles.

## Outside the current requested package

- Figure 1, the ten Extended Data figures, and Supplementary Information
  reproduction pipelines.
- The full empirical network acquisition/preprocessing archive and associated
  network-level manifest (see [DATA_SOURCES.md](DATA_SOURCES.md)).
- A tagged submission release or persistent archive DOI; neither has been
  created or assigned in this package.
- A software licence selected by the authors. The code attribution notes
  record the sources inspected without inventing a licence grant.

The original research project is preserved. The manuscript PDF is not included.
