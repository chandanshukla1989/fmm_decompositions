#!/usr/bin/env python3
"""Exact verification of every tensor decomposition in tensorDecomposition/.

For each file the checker establishes, over the integers and with no
floating-point step anywhere:

  * shapes, dtype and the integer scale factor;
  * an a priori overflow bound, proved before any contraction is computed;
  * that no row of u, v or w is zero, so every one of the r products is used;
  * that no pair (u_t, v_t) of input forms repeats, so the r products are
    genuinely distinct and the count is not inflated;
  * the FULL matrix-multiplication tensor identity, i.e. every one of the
    (m*k) * (k*n) * (m*n) coefficients, including the zero ones;
  * random matrix products recomputed in arbitrary-precision Python integer
    arithmetic, independent of the int64 contraction above;
  * agreement between the .npz and the accompanying sparse .json, when both
    are present.

The full tensor identity, not the random tests, is the certificate. The
verified quantity is an upper bound on the multiplication count; nothing here
claims minimality.

Conventions
-----------
For A of shape m x k and B of shape k x n, flatten A, B and C = A B in
row-major order.  u, v, w have shapes (r, m*k), (r, k*n), (r, m*n) and

    p[t] = (sum_a u[t,a] * A_flat[a]) * (sum_b v[t,b] * B_flat[b])
    C_flat[c] = (sum_t w[t,c] * p[t]) / scale

All files in this repository have scale = 1 and integer coefficients, and the
formulas preserve the A-before-B product order.

Usage
-----
    python3 verify_decompositions.py [tensorDecomposition]

Requires numpy only.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

NAME_RE = re.compile(r"^(\d+)x(\d+)x(\d+)_rank(\d+)\.npz$")


class Failure(Exception):
    pass


def need(ok: bool, message: str) -> None:
    if not ok:
        raise Failure(message)


def expected_slice(a: int, m: int, k: int, n: int, scale: int) -> np.ndarray:
    """The tensor slice for a fixed flattened A-index a = i*k + j.

    T[a, b, c] is nonzero exactly when b = j*n + l and c = i*n + l.
    """
    i, j = divmod(a, k)
    out = np.zeros((k * n, m * n), dtype=np.int64)
    l = np.arange(n)
    out[j * n + l, i * n + l] = scale
    return out


def check_json(path: Path, u, v, w, m, k, n, rank) -> dict | str:
    if not path.exists():
        return "absent"
    doc = json.loads(path.read_text())
    need(list(doc["dimensions"]) == [m, k, n], "json dimensions disagree with filename")
    need(int(doc["rank"]) == rank, "json rank disagrees with filename")
    need(doc.get("layout", "row-major") == "row-major", "unexpected json layout")
    report = {}
    for name, arr in (("u", u), ("v", v), ("w", w)):
        rows = doc[name]
        need(len(rows) == rank, f"json {name} has {len(rows)} rows, expected {rank}")
        rebuilt = np.zeros(arr.shape, dtype=np.int64)
        for t, entries in enumerate(rows):
            for col, value in entries:
                rebuilt[t, col] = value
        need(np.array_equal(rebuilt, arr), f"json {name} disagrees with npz")
        report[name] = True
    return report


def verify(path: Path) -> dict:
    match = NAME_RE.match(path.name)
    need(match is not None, f"cannot read dimensions from filename {path.name}")
    m, k, n, rank = (int(g) for g in match.groups())

    with np.load(path, allow_pickle=False) as data:
        keys = sorted(data.files)
        u, v, w = data["u"], data["v"], data["w"]
        scale = int(data["scale"]) if "scale" in data.files else 1

    # Files may store the coefficients in any signed integer width; int8 is
    # enough for every scheme here. Widen to int64 for the contractions and
    # record the stored width, leaving the archived bytes untouched.
    stored_dtypes = [str(x.dtype) for x in (u, v, w)]
    need(all(x.dtype.kind == "i" for x in (u, v, w)),
         f"coefficients are not signed integers (got {stored_dtypes})")
    u, v, w = (x.astype(np.int64) for x in (u, v, w))

    need(u.shape == (rank, m * k), f"u has shape {u.shape}, expected {(rank, m * k)}")
    need(v.shape == (rank, k * n), f"v has shape {v.shape}, expected {(rank, k * n)}")
    need(w.shape == (rank, m * n), f"w has shape {w.shape}, expected {(rank, m * n)}")
    need(scale == 1, f"scale is {scale}, expected 1")

    maxima = [int(np.abs(x).max()) for x in (u, v, w)]
    bound = rank * maxima[0] * maxima[1] * maxima[2]
    need(bound < 2 ** 62, "int64 overflow is possible for this instance")

    need(bool(np.all(np.any(u != 0, axis=1))), "a row of u is zero")
    need(bool(np.all(np.any(v != 0, axis=1))), "a row of v is zero")
    need(bool(np.all(np.any(w != 0, axis=1))), "a row of w is zero")

    pairs = {(u[t].tobytes(), v[t].tobytes()) for t in range(rank)}
    need(len(pairs) == rank, f"only {len(pairs)} distinct (u,v) input-form pairs for {rank} products")

    # full tensor identity, one A-index at a time
    checked = 0
    for a in range(m * k):
        rows = np.flatnonzero(u[:, a])
        got = np.einsum("t,tb,tc->bc", u[rows, a], v[rows], w[rows],
                        dtype=np.int64, optimize=True)
        need(np.array_equal(got, expected_slice(a, m, k, n, scale)),
             f"tensor identity fails at flattened A-index {a}")
        checked += (k * n) * (m * n)
    need(checked == m * k * k * n * m * n, "internal accounting error")

    # independent recomputation in exact Python integer arithmetic
    rng = np.random.default_rng(20261109)
    ul, vl, wl = u.tolist(), v.tolist(), w.tolist()
    trials = 8
    for trial in range(trials):
        hi = 60 if trial % 2 else 6
        A = rng.integers(-hi, hi + 1, size=(m, k)).tolist()
        B = rng.integers(-hi, hi + 1, size=(k, n)).tolist()
        af = [x for row in A for x in row]
        bf = [x for row in B for x in row]
        C = [0] * (m * n)
        for t in range(rank):
            ut, vt, wt = ul[t], vl[t], wl[t]
            p = (sum(ut[x] * af[x] for x in range(m * k))
                 * sum(vt[y] * bf[y] for y in range(k * n)))
            if p:
                for c in range(m * n):
                    if wt[c]:
                        C[c] += wt[c] * p
        ref = [sum(A[i][j] * B[j][l] for j in range(k))
               for i in range(m) for l in range(n)]
        need(C == ref, f"exact-integer matrix product {trial} disagrees")

    json_report = check_json(path.with_suffix(".json"), u, v, w, m, k, n, rank)

    return {
        "file": path.name,
        "format": [m, k, n],
        "multiplications": rank,
        "naive_multiplications": m * k * n,
        "npz_keys": keys,
        "stored_dtypes": stored_dtypes,
        "scale": scale,
        "max_abs_coefficient": maxima,
        "nonzeros": [int(np.count_nonzero(x)) for x in (u, v, w)],
        "distinct_input_form_pairs": len(pairs),
        "a_priori_coefficient_bound": bound,
        "tensor_coefficients_checked": checked,
        "exact_tensor_identity": True,
        "exact_integer_matrix_products": trials,
        "json_matches_npz": json_report,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "verdict": "PASS",
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "tensorDecomposition").resolve()
    files = sorted(p for p in root.glob("*.npz"))
    if not files:
        print(f"no .npz files under {root}", file=sys.stderr)
        return 2

    results, failed = [], False
    for path in files:
        try:
            results.append(verify(path))
        except (Failure, KeyError, ValueError) as exc:
            failed = True
            results.append({"file": path.name, "verdict": "FAIL", "reason": str(exc)})

    print(json.dumps(results, indent=2))
    print()
    for r in results:
        if r["verdict"] == "PASS":
            print(f"PASS  {str(r['format']):14s} {r['multiplications']:>5d} multiplications "
                  f"(naive {r['naive_multiplications']}), "
                  f"{r['tensor_coefficients_checked']} tensor coefficients checked")
        else:
            print(f"FAIL  {r['file']}: {r['reason']}")
    print("\nALL CHECKS PASSED" if not failed else "\nVERIFICATION FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
