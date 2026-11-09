# Verified bilinear decompositions for three small matrix formats

Explicit integer bilinear algorithms for three small matrix-multiplication
formats, with the full coefficient matrices, a checker that verifies the
complete tensor identity exactly, and a script that compares each result
against the published state of the art.

Two of the three lower the multiplication count currently recorded in
[`fmm_sota.json`](https://github.com/sedoglavic/fmm_digest/blob/master/fmm_sota.json);
the third lies past the `32 x 32 x 32` limit of that table.

Chandan Shukla, independent researcher · <https://orcid.org/0009-0008-0897-4332>

## Results

| Format | Multiplications | Naive | Exponent | `fmm_sota.json` | Closed catalogue | Improvement |
| --- | --- | --- | --- | --- | --- | --- |
| 14 × 18 × 19 | **2829** | 4788 | 2.813713 | 2890 | 2890 | −61 |
| 10 × 19 × 24 | **2723** | 4560 | 2.816409 | 2754 | 2754 | −31 |
| 10 × 13 × 36 | **2786** | 4680 | 2.815872 | not listed (36 > 32) | 2902 | −116 |

`tensorDecomposition/14x19x18_rank2829.npz` is the same scheme as the
14 × 18 × 19 entry in the orientation in which it was originally found; it is
included because the two orientations index their coefficients differently and
both are occasionally convenient. It is one result, not two.

The exponent column is `3·ln(r)/ln(m·k·n)`, the convention used in
`fmm_sota.json`.

The *closed catalogue* column is the key one. A lower count is only interesting
if the published table does not already reach it, so `catalogue_closure.py`
closes `fmm_sota.json` under the operations the table is itself built from —
base cases, direct sums, Kronecker products — together with the
projection-along-mode lemma from
<https://fmm.univ-lille.fr/projalongmode.html>, and only then compares. All
three formats survive that closure. For 10 × 13 × 36 the best route the closure
finds is the direct sum 36 = 30 + 6, giving 2382 + 520 = 2902.

```
$ curl -O https://raw.githubusercontent.com/sedoglavic/fmm_digest/master/fmm_sota.json
$ python3 catalogue_closure.py fmm_sota.json

catalogue:  5456 entries, createdAt 2026-11-06T20:58:44
closing under base + direct sum + Kronecker + projection, modes up to 36
closed table: 8436 formats

  [10, 13, 36]  rank  2786  exponent 2.815872  |  catalogue entry   None  closure   2902  ->  improves the closure by 116
  [10, 19, 24]  rank  2723  exponent 2.816409  |  catalogue entry   2754  closure   2754  ->  improves the closure by 31
  [14, 18, 19]  rank  2829  exponent 2.813713  |  catalogue entry   2890  closure   2890  ->  improves the closure by 61

3 of 3 formats improve on the closed catalogue.
```

## What is claimed, and what is not

These are **upper bounds** on the multiplication count. Nothing here is a
claim about tensor rank, minimality, or a new multiplication technique: every
scheme is composed from component schemes that are already published, and the
attribution is recorded below. The contribution is that these particular
formats are reachable at these counts, that the coefficients are written out
explicitly, and that the identity is verified rather than reported.

## Verification

```
$ python3 -m pip install -r requirements.txt
$ python3 verify_decompositions.py

PASS  [10, 13, 36]    2786 multiplications (naive 4680), 21902400 tensor coefficients checked
PASS  [10, 19, 24]    2723 multiplications (naive 4560), 20793600 tensor coefficients checked
PASS  [14, 18, 19]    2829 multiplications (naive 4788), 22924944 tensor coefficients checked
PASS  [14, 19, 18]    2829 multiplications (naive 4788), 22924944 tensor coefficients checked

ALL CHECKS PASSED
```

Everything is done over the integers, with no floating-point step anywhere.
For each file the checker establishes:

- shapes, integer dtype and the scale factor, which is 1 in every file here;
- an a priori overflow bound, proved before any contraction is computed;
- that no row of `u`, `v` or `w` is zero, so all `r` products are used;
- that no pair `(u_t, v_t)` of input forms repeats, so the `r` products are
  genuinely distinct and the count is not inflated by duplicates;
- the **full tensor identity** — every one of the `(mk)(kn)(mn)` coefficients
  of the matrix-multiplication tensor, including the zero ones;
- random matrix products recomputed in arbitrary-precision Python integer
  arithmetic, independent of the `int64` contraction above;
- agreement between each `.npz` and its sparse `.json` export.

The full tensor identity, not the random tests, is the certificate. The random
products are there because an independent recomputation in a different
arithmetic is cheap insurance against a mistake in the contraction itself.

[`verification_results.json`](verification_results.json) is the output of a run
on 9 November 2026, including the SHA-256 digest of each `.npz`.

## Coefficient convention

For `A` of shape `m × k` and `B` of shape `k × n`, flatten `A`, `B` and
`C = AB` in row-major order. Then `u`, `v`, `w` have shapes `(r, mk)`,
`(r, kn)`, `(r, mn)` and

```
p[t]      = (sum_a u[t,a] · A_flat[a]) · (sum_b v[t,b] · B_flat[b])      t = 0 … r-1
C_flat[c] = (sum_t w[t,c] · p[t]) / scale
```

So `u` is indexed by the `(i,j)` of `A` as `i·k + j`, `v` by the `(j,l)` of `B`
as `j·n + l`, and `w` by the `(i,l)` of `C` as `i·n + l`. Every file here has
`scale = 1`, integer coefficients bounded by 2 in absolute value, and preserves
the `A`-before-`B` product order, so the schemes remain valid over any
commutative ring and can be applied recursively.

## Files

```
tensorDecomposition/
  10x13x36_rank2786.npz      coefficient matrices u, v, w and the integer scale
  10x19x24_rank2723.npz
  14x18x19_rank2829.npz
  14x19x18_rank2829.npz      the 14x18x19 scheme in its original orientation
  *.json                     the same data as a dependency-free sparse export
verify_decompositions.py     exact verification of every file above
catalogue_closure.py         comparison against the closed published catalogue
export_json.py               regenerates the .json exports from the .npz files
verification_results.json    output of a verification run, with SHA-256 digests
fmm_sota_additions.json      the three entries in the fmm_digest table format
tensor_decomposition_schema.json   a draft JSON schema for decompositions
SCHEMA_NOTES.md              notes on the existing fmm_digest schema
```

The `.npz` files are the authoritative data; the `.json` files are a derived
export for readers who would rather not install numpy. `export_json.py`
reproduces them byte for byte, and the checker cross-checks the two
representations against each other.

## Provenance and attribution

Every component scheme used here is already published; none of them is mine.
The counts decompose as follows.

| Format | Count | Component dimension classes |
| --- | --- | --- |
| 14 × 18 × 19 | 2829 = 123 + 11·150 + 4·264 | 4×6×7, 5×6×7, 6×7×9 |
| 10 × 19 × 24 | 2723 = 63 + 144 + 10·170 + 4·204 | 2×5×8, 5×5×8, 5×6×8, 5×7×8 |
| 10 × 13 × 36 | 2786 = 94 + 136 + 10·174 + 4·204 | 5×2×12, 5×3×12, 5×4×12, 5×5×12 |

For the first two, the components are schemes of **Andrey Perminov**, obtained
through [`solven-eu/matmulcatalog`](https://github.com/solven-eu/matmulcatalog)
at pinned commit `71920f6b205872ddaf9d0d15ce8f4ddc4a475f04`.

For 10 × 13 × 36 the component ranks ⟨5,2,12⟩ = 94, ⟨5,3,12⟩ = 136,
⟨5,4,12⟩ = 174 and ⟨5,5,12⟩ = 204, and the outer ⟨2,3,3⟩ = 15 and
⟨2,4,3⟩ = 20, are the catalogue values for those formats; credit for each
individual component belongs with whoever the corresponding catalogue entry
attributes it to, and I would rather defer to the catalogue on that than guess
here.

For 10 × 13 × 36 the construction is written out in full: the outer scheme is
⟨2,13,3⟩ at rank 65, taken as a block sum along `k` with 13 = 3 + 3 + 3 + 4, so
three copies of ⟨2,3,3⟩ at rank 15 together with ⟨2,4,3⟩ at rank 20, and the
inner scheme is ⟨5,1,12⟩. The 65 outer terms fall into 16 classes sharing a
common `w`-vector, and a class of size `s` can be done as a single ⟨5,s,12⟩
product instead of `s` separate outer products. The class sizes are one 2, one
3, ten 4s and four 5s, which sum to 65, giving the count above. A plain
Kronecker product would give 65 × 60 = 3900.

## Acknowledgements

To **Alexandre Sedoglavic** for maintaining <https://fmm.univ-lille.fr/> and
the `fmm_digest` catalogue, which was my main reference throughout, and for
suggesting this repository and pointing me to the projection-along-mode rules;
and to **Andrey Perminov**, whose schemes the components come from.

## Reuse

The coefficient data is offered for free reuse, including incorporation into
<https://fmm.univ-lille.fr/> and any other catalogue, with attribution.
