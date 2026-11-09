#!/usr/bin/env python3
"""Write the sparse .json export beside every .npz in tensorDecomposition/.

The .npz files are the authoritative data; the .json files are a derived,
dependency-free exchange format for readers who do not want numpy. Rerunning
this script reproduces them byte for byte, and verify_decompositions.py
checks the two representations against each other.

Format
------
    {"dimensions": [m, k, n], "rank": r, "scale": 1, "layout": "row-major",
     "u": [...], "v": [...], "w": [...]}

Each of u, v, w is a list of r rows; a row is a list of [index, value] pairs
with zero-based flattened indices, and any index not listed is zero.

Usage: python3 export_json.py [tensorDecomposition]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

NAME_RE = re.compile(r"^(\d+)x(\d+)x(\d+)_rank(\d+)\.npz$")


def sparse_rows(arr):
    rows = []
    for t in range(arr.shape[0]):
        idx = np.flatnonzero(arr[t])
        rows.append([[int(c), int(arr[t, c])] for c in idx])
    return rows


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "tensorDecomposition").resolve()
    for path in sorted(root.glob("*.npz")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        m, k, n, rank = (int(g) for g in match.groups())
        with np.load(path, allow_pickle=False) as data:
            u, v, w = (data[key].astype(np.int64) for key in ("u", "v", "w"))
            scale = int(data["scale"]) if "scale" in data.files else 1
        doc = {
            "dimensions": [m, k, n],
            "rank": rank,
            "scale": scale,
            "layout": "row-major",
            "u": sparse_rows(u),
            "v": sparse_rows(v),
            "w": sparse_rows(w),
        }
        out = path.with_suffix(".json")
        existed = out.exists()
        out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
        print(f"{'rewrote' if existed else 'wrote  '} {out.name}")


if __name__ == "__main__":
    main()
