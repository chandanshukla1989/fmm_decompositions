# Notes on `json_schema.json` in `fmm_digest`

Two small things I ran into while writing `fmm_sota_additions.json` against
[`json_schema.json`](https://github.com/sedoglavic/fmm_digest/blob/master/json_schema.json).
Both are easy fixes and I mention them only because the schema is explicitly
offered as a first step to be completed.

## 1. The schema cannot be satisfied as written

```json
"required": ["entries", "contributor", "CreatedAt"],
"additionalProperties": false
```

`CreatedAt` is required, but the declared property is `createdAt` with a lower
case `c`, and `additionalProperties: false` forbids adding the capitalised
spelling. So no document can validate: supplying `CreatedAt` is rejected as an
additional property, and omitting it fails the `required` check.

The current `fmm_sota.json` uses `createdAt`, so the intended fix is presumably
to write `createdAt` in `required` as well.

## 2. `previous_hash` does not match its own pattern

The property is documented as a SHA-256 digest and constrained to 64 hex
characters:

```json
"previous_hash": { "pattern": "^[0-9a-fA-F]{64}$" }
```

The value in the current `fmm_sota.json` is

```
2f5bb3adf18c809b671e16526655512ff795d146
```

which is 40 hex characters, the length of a SHA-1 digest, so it fails the
pattern. Either the digests are SHA-1 and the description and pattern should
say so, or the value should be recomputed as SHA-256.

Both points are what a validator reports. Checking the current `fmm_sota.json`
against its own schema with `jsonschema` 4.26 (Draft 2020-12) gives exactly
these two errors and nothing else:

```
'2f5bb3adf18c809b671e16526655512ff795d146' does not match '^[0-9a-fA-F]{64}$'
'CreatedAt' is a required property
```

With `required` corrected to `createdAt`, only the digest error remains.
`fmm_sota_additions.json` in this repository validates cleanly once that one
change is made.

## 3. Everything else checks out

For the record, the rest of `fmm_sota.json` is consistent with the schema:
all 5456 entries have distinct formats, every format is sorted ascending as
the convention requires, and no entry has a mode below the declared minimum
of 2.

## 4. A tensor-decomposition schema

`json_schema.json` describes the format-rank table. A schema for the
decompositions themselves is still missing, and
[`tensor_decomposition_schema.json`](tensor_decomposition_schema.json) in this
repository is a first attempt at one. It describes the files under
`tensorDecomposition/` and adds two optional blocks that seem worth having:

- `provenance`, so that credit for component schemes survives composition,
  with the rank written out as the arithmetic of the component ranks;
- `verification`, so that a reader can tell an entry whose full tensor
  identity was checked exactly from one that was only reported, and in what
  arithmetic.

It is meant as something to argue with rather than a finished proposal. It is
itself a valid Draft 2020-12 schema, all four files under
`tensorDecomposition/` validate against it, and it rejects the obvious
corruptions I tried: a zero rank, a coefficient entry with three elements
instead of two, an unknown `layout`, and a missing `scale`.
