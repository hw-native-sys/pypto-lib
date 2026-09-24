# Compensated Cube score contraction: numerical contract

This document describes the compensated DSpark CSA decode Indexer selected by
`DSPARK_INDEXER_SCORE_IMPL=cube_compensated`. The original `vector` path remains
the default. The compensated device path has no runtime coefficient-range guard
and no automatic fallback. The bounds and measured validation below describe
its numerical scope; they do not establish arbitrary-input score equivalence.

## Original expression and precision contract

For a query, 64 heads, and a candidate key `j`, the original expression is:

```text
Z[j,h] = sum_{d=0..127} INT8_Key[j,d] * INT8_Query[h,d]  # exact INT32
c[h]   = FP32(query_scale[h] * head_weight[h])
score[j] = key_scale[j] * sum_h (FP32(max(Z[j,h], 0)) * c[h])
```

The multiplication and reduction in the last line use FP32. Coefficients
can have either sign, so large positive and negative contributions can cancel.
The score test remains `abs(actual - reference) <= 1e-4 + abs(reference)/128`,
with the existing maximum outlier ratio `0.001`. The dedicated one-score
cancellation test is checked separately, so other scores cannot hide its failure.
Selected-index equality is an additional check, separate from score tolerance.

Algebraically this is `ReLU(Z) @ c`. The implementation below requires six
FP16-input, FP32-accumulating Cube GEMVs to retain coefficient precision.
It also retains the original INT8 QK operation and uses three additional
Cube accumulator corrections to construct the two R limbs. **It is not one
hardware Matmul.** The large head tensor remains on the Cube side; only one
FP32 score per candidate crosses to Vector. Vector still multiplies the
one-dimensional score by the original FP32 key scale and performs Top-K.

## Exact, nonnegative R decomposition

Use `A = 16384 = 2^14` and round-to-nearest, ties-to-even FP16 conversion:

```text
R_hi = half(max((Z - 1024)/A, 0))
R_lo = half(max(Z/A - float(R_hi), 0))
```

For all signed INT8 inputs, each absolute product is at most `128^2`.
Thus a conservative bound for a 128-wide dot is `|Z| <= 2^21`.
The negative endpoint of this symmetric bound need not be attainable; testing
it is stronger than testing only attainable dots. Every such integer and every
intermediate integer offset is exactly representable in FP32.

The identity is exact over the entire bound:

```text
float(R_hi) + float(R_lo) = max(Z, 0) / A.
```

Proof:

1. If `Z <= 0`, both limbs are zero.
2. If `0 < Z <= 1024`, `R_hi = 0` and `Z/A` is on the `2^-14` grid in
   `(0, 1/16]`. All these values are exactly representable in FP16.
3. If `Z > 1024`, let `x = (Z - 1024)/A` and `e = half(x) - x`.
   Here `0 < x <= 127.9375`; the largest FP16 spacing is `1/16`, so
   `|e| <= 1/32`. Consequently
   `Z/A - R_hi = 1/16 - e` lies in `[1/32, 3/32]`.
   Both `x` and its half rounding lie on the `2^-14` grid. Their residual
   therefore lies on that grid too. FP16 spacing throughout `[1/32, 3/32]`
   is at most `2^-14`, so the residual is represented exactly by `R_lo`.

Every nonzero R limb is at least `2^-14`, the minimum normal FP16 value.
R therefore does not depend on half-subnormal support. Its low limb is bounded
by `3/32`; ordinary `half(ReLU(Z)/A)` alone does not preserve the integer dot.

The hardware-oriented construction uses the INT32 bias `0x44400000 - 1024`.
Reinterpreting the biased accumulator bits as FP32 gives exactly
`768 + (Z - 1024)/A`: the entire range remains in the FP32 binade `[512,1024)`
whose spacing is `2^-14`. Subtracting 768 recovers `x`; later subtracting
`R_hi` and adding `1/16` recovers the nonnegative residual above. The CPU test
checks both this bit-alias identity and the independent ReLU reconstruction.

## Three coefficient limbs and their error bound

The prepared coefficient is the already-rounded original FP32 `c`. Scale by
the exact power of two, then split it into three nearest-half residuals:

```text
B  = FP32(A*c)
W0 = half(B)
t1 = FP32(B - float(W0))
W1 = half(t1)
t2 = FP32(t1 - float(W1))
W2 = half(t2)
```

The encoding bound assumes finite `B` with `|B| <= 65504`, equivalently
`|c| <= 65504/16384 = 3.998046875`. Nonfinite or out-of-range values are outside
this encoding contract. The device path does not test this precondition or
switch back to Vector automatically; the caller must select the original path
when needed. The CPU test helper rejects unsupported encodings with
`ValueError`, but that check is only in the test helper, not in the device
implementation. Even within the finite range, FP32 reduction order prevents
a universal original-score equivalence claim for arbitrary signed coefficients.

Under IEEE nearest rounding **with half subnormals preserved**,

```text
|B - (float(W0) + float(W1) + float(W2))| <= 2^-25
|c - (float(W0) + float(W1) + float(W2))/A| <= 2^-39.
```

Why this bound is small: each residual subtraction is exact in FP32 (nearby
rounded operands satisfy the usual exact-subtraction condition; a zero rounded
operand is trivial). If the first two half roundings are normal, let `e` be the
original FP32 exponent. The first residual has magnitude at most `2^(e-11)`;
the second has magnitude at most `2^(e-22)` and remains on the original FP32
`2^(e-23)` grid. It has very few significant bits, so its third half conversion
is exact whenever its value is representable on the half lattice. If a rounding
enters the half-subnormal range, that lattice has spacing `2^-24`, with maximum
nearest-rounding error `2^-25`. This is the only remaining encoding error.

Half subnormal support is essential for W, unlike R. The target cancellation
example below has a nonzero third limb of `2^-16`. Flushing it to zero restores
the old failure. The observed device's separate FP16-subnormal probe preserves
`2^-24` through Cube multiplication; this is a device/toolchain property to
retain in validation, not a portable guarantee made by these CPU tests.

The coefficient-encoding contribution to score error satisfies

```text
E_encoding[j] <= |key_scale[j]| * 2^21 * 64 * 2^-39
              = |key_scale[j]| / 4096.
```

This is a bound on encoding error in real arithmetic. It does not include
FP32 products, accumulation/reassociation, final scaling, or Top-K ordering.
For the measured target's maximum visible scale `0.008735235780477524`, the
bound is `2.1326259e-6`.

## Score accumulation, cancellation, and limits

The six terms are the products of the two R limbs and three W limbs:

```text
score = key_scale * (
    R_hi @ W0 + R_hi @ W1 + R_hi @ W2
  + R_lo @ W0 + R_lo @ W1 + R_lo @ W2)
```

The actual implementation accumulates these products in FP32. This changes
rounding order relative to FP32 elementwise multiplication followed by a
head reduction. Exact R and nearly exact W therefore do not mean bitwise
identical scores, or a universal tolerance guarantee for arbitrary inputs.
The CPU matvec reduction order is also not a bit-level model of the Cube.

A concrete target-amplitude regression uses all 128 INT8 components equal to
127, so each head dot is `2064512`; key scale is `0.00784325785934925`.
The first 32 coefficients are `-0.010233103297650814`; the other 32 are
`+0.010233104228973389`.

| CPU expression | Score | Original tolerance |
|---|---:|---:|
| Original FP32 expression | 0.0004902036162 | 0.0001038297123 |
| Two coefficient limbs | 0 | fails |
| Three coefficient limbs, six GEMVs | 0.0004825741053 | passes |
| Three limbs with subnormals flushed | 0 | fails |

A separate finite example with coefficients around one reconstructs W exactly
but still exceeds the original score tolerance through FP32 reassociation.
It is retained as a limitation test, not counted as a target-amplitude pass.
It rules out claims that finite operands within the encoding range, or even
exact operands, establish arbitrary-input or bitwise score equivalence.

## Measured validation of the retained implementation

The rebase comparison against main `d4b05f9` validates the Cube source at
`92c3e39` with unchanged thresholds. All eight A-B-B-A runs pass. Across all
four Vector/Cube cross-pairs, the 128 Top-512 sets are identical; 80 returned
index positions differ only in ordering. Ranked scores and all 65,536 scores
matched by candidate ID have zero outliers, with maximum absolute difference
3.0517578125e-5. Complete CSA outputs are bitwise equal across 2,097,152
elements. Repeated runs within each implementation are also bitwise stable.
This new-baseline validation supersedes the older boundary-selection
observation below; it does not imply arbitrary-input equivalence or a
performance benefit. See [README.md](README.md) for the latency regression
against the new Vector implementation.

The original 2026-09-17 validation of the retained six-GEMV implementation
(snapshot `v8r`) passed the unchanged device precision checks for `TP=1`,
runtime `B=16`, eight query positions per request, and every start position
equal to 131072. The original
compile-time capacity remains 64 requests; the fixture has 128 active queries.

| Device check | Observed result |
|---|---|
| Target Indexer stock precision gate | Passed without changing thresholds |
| Target Indexer Top-512 sets | All 128 query sets match the golden reference |
| Shuffled-page, empty-query, tail and cancellation smoke | 9,799 compared scores, zero outliers, no missing/duplicate/invalid indices, canaries preserved |
| Actual INT8 boundary dots `-2080768` and `2097152` | Passed; negative score exactly zero, positive score equals the original-expression golden |
| Dedicated signed-cancellation probe | Passed its separate one-score tolerance |
| Full target CSA output vs captured actual Vector output | All 2,097,152 elements bitwise equal |

The earlier 2026-09-19 comparison (`eacafcf` versus `af71111`)
also passes the original Indexer and CSA gates. All four A-B-B-A cross-pairs
have zero score outliers against Vector, with maximum absolute difference
2.2888184e-5. Strict selected-set equality has one boundary difference:
query 80 selects candidate 32712 on Vector and 28162 on Cube. Both selected
boundary scores are 22.83275604248047; the frozen Torch golden selects 28162,
and all compensated query sets match that golden. This does not prove that
both candidates have equal scores within either implementation. The 65,535
common selected IDs also have zero score outliers. Complete CSA outputs
remain bitwise identical across all 2,097,152 elements, and repeated runs of
each implementation are bitwise stable. These fresh device checks are
separate from the historical fixed-ID CPU study below.

The boundary smoke uses a query of all `-128` components and keys of all `127`
and all `-128` components, respectively. It reaches the real positive INT8 dot
endpoint, not merely the quantizer's usual `127` limit. The measured positive
score is `1807.6390380859375`; the negative-dot score is zero. The separate
cancellation probe produces `0.00048257410526275635` on the device against the
original-expression golden `0.0004902036162093282`, passing the unchanged
`0.00010382971231592819` tolerance.

These are measurements on the stated fixtures and device/toolchain. Bitwise
agreement of the full CSA output does not imply bitwise Indexer scores or an
arbitrary-input FP32 reassociation guarantee. Full CSA consumes the selected
candidate indices, so its output can match even when small score differences
remain. See [validation_results.json](validation_results.json) for the recorded
validation and [README.md](README.md) for execution and benchmark context.

A separate CPU study used all 8192 captured coefficients and the 65,536 actual
Vector-selected IDs. Its six-GEMV model had zero score outliers and maximum
absolute difference `2.2888184e-5`. This fixed-ID CPU evidence is distinct from
the complete device gates above. Earlier approximate variants and causal
experiments are not part of this implementation.

## Reproducing the bounded CPU checks

Only PyTorch is required. No NPU, model fixture, PyPTO compiler, or network is
used by the test. From the repository root, with PyTorch available:

```bash
python models/deepseek_v4_flash_dspark/kernels/indexer_cube_score/test_compensated_numerics.py
```

The test prints a JSON report; `--report <path>` optionally saves it.

The six default tests:

- Exhaust all 4,194,305 integers in `[-2^21,2^21]` in chunks of 65,536,
  comparing both half limbs in FP64 against `max(Z,0)/A` and checking the
  biased bit-alias identity.
- Check coefficient encodings at every finite-half value, neighboring FP32
  values, half midpoints and their neighbors, and exponent-wide random FP32
  samples, with both signs: 446,456 inputs in the recorded run.
- Verify that the CPU encoding helper rejects invalid or out-of-range inputs;
  this does not test or provide an automatic device fallback.
- Compare five small random cases with signed target-amplitude coefficients
  against an independent exact-INT32-QK + original-FP32-expression golden;
  separately check selected-index sets.
- Require the old two-limb and FTZ cancellation examples to fail while the
  three-limb version passes the unchanged one-element score tolerance.
- Preserve the distinction between exact coefficient encoding and bitwise
  FP32 reduction for the larger-coefficient limitation example.

The recorded run used PyTorch 2.6.0+cpu, four threads, and took about 0.10 s
inside the tests (interpreter/import startup excluded). All six tests passed.
This duration is a local observation, not a performance requirement.
