"""Fail CI on approximation row errors, which a batch CLI can otherwise log."""
from pathlib import Path
import sys
import pandas as pd

directory = Path(sys.argv[1])
paths = list(directory.rglob('*summary.csv'))
assert paths, f'No summary output in {directory}'
rows = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
assert 'status' in rows, rows.columns.tolist()
assert rows['status'].eq('ok').all(), rows.to_string()
print(f'Checked {len(rows)} successful summary rows.')
