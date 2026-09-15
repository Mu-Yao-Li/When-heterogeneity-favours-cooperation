"""Verify bundled plotting inputs against their frozen data manifest."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "figures/data_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Size or checksum mismatch: {entry['path']}")
        if entry.get("rows") is not None:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                count = sum(1 for _ in reader)
            if count != entry["rows"] or header != entry["columns"]:
                raise ValueError(f"CSV shape mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} figure inputs (checksums and CSV shapes).")


if __name__ == "__main__":
    main()
