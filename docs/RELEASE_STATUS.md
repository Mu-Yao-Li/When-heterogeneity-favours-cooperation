# Release status

This initial version provides a validated numerical core; a full paper-reproduction release is still being prepared.

## Prepared

- Seven existing scientific source modules, copied unchanged with SHA-256 hashes.
- Portable exact-analysis entry point and fixed small-network example.
- Small mathematical checks and end-to-end validation commands.
- Dependency specification, citation metadata draft, Git ignore rules and CI configuration.
- Main/Extended Data figure source mapping with explicit confidence limits.

See `VALIDATION.md` for commands actually run and results. A CI configuration is
not evidence of a successful hosted CI run.

## Required before a complete paper-code release

1. Match Supplementary Information, including simulations and robustness rules,
   to the corresponding scripts and configurations. The supplied PDF refers to
   Supplementary Figures S1-S22 and beyond but does not contain those pages.
2. Freeze final plotting programs and exact input tables for all 4 main and
   10 Extended Data figures. Current local source candidates are listed in
   `FIGURE_MAP.md`; older handoffs disagree with the submitted layout.
3. Include audited empirical preprocessing and network-level statistics promised
   in the paper's code/data statement.
4. Confirm code authorship, third-party attribution and the intended licence.
5. Repository selected: `johnston0603-stack/When-heterogeneity-favours-cooperation`
   (public). The URL is in `CITATION.cff`. Add it to the manuscript after verifying
   the published contents and completing the promised code/data coverage.
6. Tag the exact submission version and, if desired, archive that version with a
   persistent identifier. Do not invent a DOI or claim a release exists.

The user's original project has not been modified by this preparation.
