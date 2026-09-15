"""Small fixed-network example; outputs are not manuscript source data."""
from pathlib import Path
import csv
import sys

import networkx as nx
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from analyze_network import analyze
from run_adaptive_closed_tail_pipeline import load_edge_list


def main():
    pa, _ = load_edge_list(ROOT / 'examples/pa_n100_k4_seed42.edgelist')
    ring = nx.watts_strogatz_graph(100, 4, 0, seed=42)
    rr = sparse.csr_matrix(nx.to_scipy_sparse_array(ring, format='csr', dtype=float))
    rows = [{'network': label, **analyze(a)} for label, a in [('PA', pa), ('regular_ring', rr)]]
    output = ROOT / 'outputs/quickstart.csv'
    output.parent.mkdir(exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f'{row["network"]}: N_eff={row["N_eff"]:.8f}, epsilon={row["epsilon"]:.8f}, threshold={row["threshold"]:.8f}')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
