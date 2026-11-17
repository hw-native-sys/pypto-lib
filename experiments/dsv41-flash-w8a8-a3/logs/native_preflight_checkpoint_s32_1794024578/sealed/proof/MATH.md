# Exact CPU integer-dot evaluation through FP64

This concerns CPU reference evaluation only. Original model/kernel/golden source
and the current validation process were not modified. There is no tolerance,
rounding-boundary or reference-scope change.

For a signed INT8-domain dot, a_i,b_i are integers in[-128,127]. Each product is
an integer with absolute value at most16384. For K terms, every partial sum in
any ordinary multiply-add/reduction ordering has absolute value at most16384*K.
The checked evaluator accepts only K<=floor((2^31-1)/16384)=131071. Therefore:

- Original signed-INT32 multiplication/accumulation cannot overflow on any
  accepted product or intermediate partial sum. The mathematical result is
  the exact signed-INT32 result.
- Binary64 exactly represents all input integers, products and all these
  integer partial sums: their magnitudes are below2^31, far below2^53.
  FP64 multiply/add/FMA reduction is exact for this bounded integer problem.
- The FP64 result is an exactly represented in-range INT32 integer. Conversion
  to INT32 loses no information and equals the original S32 accumulator
  BITWISE. This proof is independent of M,N, cancellation, packing/view strides
  and reduction tiling. It is not a claim about arbitrary floating operands.
- Keeping the original subsequent acc.float(), FP32 multiplication order,
  per-token/per-channel scales and BF16 casts gives identical downstream
  inputs and results. No dequantization operation is reassociated.

The guard refuses non-CPU, non-INT8/INT32, incompatible matrices, INT32 values
outside the INT8 domain, and K beyond131071. For a future evaluator, any rejected
case must use the ORIGINAL S32 reference or fail explicitly. Do not convert an
overflowing FP64 integer to INT32 and assume it emulates S32 wrap semantics.

## Actual four frozen consumers and K bounds

| Consumer | S32 source | Actual model K |
| --- | --- | --- |
| moe_experts._gemm_golden | moe_experts.py114 |5120 for gate/up;2304 for down |
| golden_w8a8_gemm | gemm_w8a8.py90 |5120 in its standalone contract; shared callers use the bounds listed here |
| nested gemm in _golden_mla_proj | mla_proj.py147 |5120 for input projections;1280 for query expansion |
| query projection in golden_index_scores | indexer.py450 |1280 |

Frozen config hidden_size=5120, moe_intermediate_size=2304, q_lora_rank=1280.
The largest actual absolute bound is5120*16384=83,886,080, below2^31. The exact
helper still checks actual operand shape/domain, so it cannot silently extend
this proof to a wider future contract. Generic standalone shapes beyond this
model require the same checks.

## Executed evidence

CPU task115146_414060215361 PASS:

-18 matrix cases across K1280/2304/5120: nonconstant random, all minimum,
  all maximum, mixed extremes, cancellation, strided input plus actual frozen
  NZ pack/unpack. FP64-to-INT32 equals original S32 and independent INT64
  accumulation bitwise.
- An odd result83,885,825 above2^24 proves FP32 is not an integer-exact substitute;
  its computed FP32 integer differs. The accepted FP64 path matches S32.
- Oversized-K, out-of-domain, wrong-dtype and mismatched-shape controls reject.
- EXACT frozen function/expression ASTs from all four consumers were executed
  independently with only the integer matmul callee replaced. Original INT32
  argument casts and all FP32 scales/BF16 output casts remained. Four output
  comparisons are bitwise equal. The complete model was not run.

CPU task115357_417597622890 additionally checked BOTH operands strided and
measured the actual proposed AST interface, including original to(INT32) casts,
range scans, FP64 conversions, matmul and output INT32 conversion. No cached
full weights or global module monkeypatch was used.
