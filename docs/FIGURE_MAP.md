# Manuscript figure map

The supplied manuscript has four main figures and ten Extended Data figures.
This release provides independent entry points for the requested main Figures
2, 3, and 4.

| Figure | Entry point | Included evidence |
| --- | --- | --- |
| 2 | `figures/reproduce_fig2.py` | Shared 20-network selection for threshold/mechanism panels; simulation summaries; separate critical-size analysis. |
| 3 | `figures/reproduce_fig3.py` | Degree heterogeneity, network drawings, size/exponent phase map, mechanism, and six favorable degree-dependent windows. |
| 4 | `figures/reproduce_fig4.py` | Module drawing, all 500,500 bridge candidates, shell boxplots, reach changes, and two correlation panels. |

**Figure 2 is an author-requested 20-network redraw.** Panels a/c supersede the
supplied manuscript image. Selection is deterministic and recorded at the
individual-network level; the larger-ensemble averages were not relabeled as 20.
Panels b/d keep their independent archived data.

Figure 3 uses the six-row panel-d configuration, including k=8. Figure 4 uses the
horizontal module layout and shell colors in the manuscript; it contains a–e,
despite historical source names containing `abcd`.

These commands redraw bundled numerical results. They do not rerun original
long simulations or exhaustive bridge enumeration. See [figure commands](../figures/README.md)
and [data provenance](FIGURE_DATA_PROVENANCE.md).

## Outside the requested plotting scope

Main Figure 1, Extended Data 1–10, and the separate Supplementary Information do
not yet have independent plotting entry points in this release. Numerical
methods elsewhere in the repository can support those analyses; this is not a
claim of complete manuscript reproduction.
