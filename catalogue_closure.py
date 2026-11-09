#!/usr/bin/env python3
"""Check each scheme in this repository against the closure of fmm_sota.json.

A smaller multiplication count only matters if the published catalogue does
not already reach it by one of the operations the catalogue is itself built
from. This script closes the catalogue under those operations and then reports,
for every format in this repository, whether the scheme here is still an
improvement.

Operations used
---------------
base        r<a,b,1> = a*b and permutations;  r<a,b,c> <= a*b*c
direct sum  r<a,b,c1+c2> <= r<a,b,c1> + r<a,b,c2>, applied to any one mode
Kronecker   r<a1*a2, b1*b2, c1*c2> <= r<a1,b1,c1> * r<a2,b2,c2>
projection  r<a,b,c-1> <= r<a,b,c> - max(a,b)

The projection step is the lemma stated at
https://fmm.univ-lille.fr/projalongmode.html, which gives three rules:

    <n x m x (p+1) : r>  ->  <n x m x p : r - m>
    <n x (m+1) x p : r>  ->  <n x m x p : r - n>
    <(n+1) x m x p : r>  ->  <n x m x p : r - p>

In each rule the mode that shrinks and the mode that is subtracted are
different modes. Matrix-multiplication tensor rank is invariant under
permuting the three modes, so for a format {a,b,c} any mode may be shrunk and
either of the two remaining modes subtracted; subtracting the larger one gives
the stronger bound, which is what the code does.

Formats are stored as unordered triples sorted ascending, matching the
convention of fmm_sota.json.

Usage
-----
    curl -O https://raw.githubusercontent.com/sedoglavic/fmm_digest/master/fmm_sota.json
    python3 catalogue_closure.py fmm_sota.json

Requires no third-party packages.
"""
import json
import math
import re
import sys
from collections import deque
from pathlib import Path

NAME_RE = re.compile(r"^(\d+)x(\d+)x(\d+)_rank(\d+)\.npz$")


def key(a, b, c):
    return tuple(sorted((a, b, c)))


def local_schemes(root="tensorDecomposition"):
    """The formats present in this repository, best rank per unordered format."""
    found = {}
    for path in sorted(Path(root).glob("*.npz")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        m, k, n, rank = (int(g) for g in match.groups())
        f = key(m, k, n)
        if f not in found or rank < found[f]:
            found[f] = rank
    return found


def close(seed, maxdim):
    """Relax `seed` under base, direct sum, Kronecker and projection."""
    best = {}

    def put(f, r):
        if max(f) > maxdim or r < 1:
            return False
        if f not in best or r < best[f]:
            best[f] = r
            return True
        return False

    for a in range(1, maxdim + 1):
        for b in range(a, maxdim + 1):
            for c in range(b, maxdim + 1):
                put((a, b, c), a * b * c)
    for a in range(1, maxdim + 1):
        for b in range(1, maxdim + 1):
            put(key(a, b, 1), a * b)
    for f, r in seed.items():
        put(f, r)

    queue = deque(best)
    while queue:
        f = queue.popleft()
        r = best[f]
        a, b, c = f

        for i in range(3):
            d = list(f)
            if d[i] - 1 < 1:
                continue
            others = [d[j] for j in range(3) if j != i]
            d[i] -= 1
            g = key(*d)
            if put(g, r - max(others)):
                queue.append(g)

        for i in range(3):
            for add in range(1, maxdim + 1):
                partner = list(f)
                partner[i] = add
                pk = key(*partner)
                if pk not in best:
                    continue
                d = list(f)
                d[i] += add
                if d[i] > maxdim:
                    break
                g = key(*d)
                if put(g, r + best[pk]):
                    queue.append(g)

        for g, rg in list(best.items()):
            for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                t = key(a * g[perm[0]], b * g[perm[1]], c * g[perm[2]])
                if max(t) > maxdim:
                    continue
                if put(t, r * rg):
                    queue.append(t)
    return best


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    doc = json.load(open(sys.argv[1]))
    published = {}
    for e in doc["entries"]:
        f = tuple(sorted(e["format"]))
        if f not in published or e["rank"] < published[f]:
            published[f] = e["rank"]

    mine = local_schemes(sys.argv[2] if len(sys.argv) > 2 else "tensorDecomposition")
    if not mine:
        print("no schemes found in tensorDecomposition/", file=sys.stderr)
        return 2

    maxdim = max(max(f) for f in mine)
    print(f"catalogue:  {len(published)} entries, createdAt {doc.get('createdAt')}")
    print(f"closing under base + direct sum + Kronecker + projection, modes up to {maxdim}")
    closed = close(published, maxdim)
    print(f"closed table: {len(closed)} formats\n")

    width = max(len(str(list(f))) for f in mine)
    improved = 0
    for f, r in sorted(mine.items()):
        direct = published.get(f)
        bound = closed.get(f)
        exponent = 3 * math.log(r) / math.log(f[0] * f[1] * f[2])
        if bound is None:
            verdict = "not reachable by the closure"
        elif r < bound:
            verdict = f"improves the closure by {bound - r}"
            improved += 1
        elif r == bound:
            verdict = "matches the closure"
        else:
            verdict = f"the closure is better by {r - bound}"
        print(f"  {str(list(f)):{width}s}  rank {r:5d}  exponent {exponent:.6f}  |  "
              f"catalogue entry {str(direct):>6s}  closure {str(bound):>6s}  ->  {verdict}")

    print(f"\n{improved} of {len(mine)} formats improve on the closed catalogue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
