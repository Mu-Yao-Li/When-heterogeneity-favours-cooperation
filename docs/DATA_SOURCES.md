# Empirical data and production outputs

The manuscript names TUDataset, SNAP, Network Repository, KONECT, networkdata,
and SocioPatterns. Its Methods report 347,391 retained human-interaction networks,
including 12,050 excluded for non-positive thresholds and 335,341 classified
networks. These are manuscript-reported counts, not independently verified counts
from this draft.

Before a full reproduction release, freeze a network-level manifest containing:

- original dataset identifier, source URL, and applicable terms;
- directed/weighted to undirected/unweighted conversion;
- loop and duplicate handling, largest-component selection and resulting `N`, `E`;
- inclusion/exclusion reason and graph checksum;
- exact solver settings, status, residual and network statistics;
- production source file, seed where relevant, and figure/table membership.

The local project contains candidate acquisition, manifest and classification
scripts such as `download_fourdb_real_networks.py`,
`build_fourdb_manifest_from_processed.py`, and
`classify_empirical_network_instances.py`. Their final combination and dataset
snapshot remain to be matched to Supplementary Section 6. They are not included
as a verified pipeline in this draft.

Raw third-party networks are not bundled. The manuscript's promised preprocessing
scripts and network statistics still need to be added before claiming that the
code/data availability statement has been fulfilled.
