# Intermediate Examples

The intermediate examples add reduction loops, normalization, and
transformer-oriented elementwise operations. Each script is a runnable
single-device Golden Harness case.

All five scripts declare `a2a3`, `a2a3sim`, `a5`, and `a5sim`. Repository CI
exercises `a2a3`, `a2a3sim`, and `a5sim`.

## GEMM

[View the source](../../examples/intermediate/gemm.py).

This example extends the beginner matmul by tiling K. The first K tile creates
the accumulator with `pl.matmul`; later tiles update it with
`pl.matmul_acc`.

```bash
python examples/intermediate/gemm.py -p a2a3sim
```

Key topics: M/N/K blocking, sequential reduction, and FP32 accumulation.

## LayerNorm

[View the source](../../examples/intermediate/layer_norm.py).

The hidden dimension fits in one tile. Each row tile computes the mean and
variance, normalizes its input, then applies gamma and beta with broadcast
operations.

```bash
python examples/intermediate/layer_norm.py -p a2a3sim
```

Key topics: `row_sum`, row and column broadcasts, reshaping reduction results,
and numerical tolerances.

## RMSNorm

[View the source](../../examples/intermediate/rms_norm.py).

RMSNorm demonstrates a reduction whose hidden dimension is larger than one
tile. One pass accumulates the sum of squares across hidden chunks; a second
pass normalizes each chunk and applies gamma.

```bash
python examples/intermediate/rms_norm.py -p a2a3sim
```

Key topics: chunked reductions, persistent accumulators, `rsqrt`, and a
two-pass kernel structure.

## RoPE

[View the source](../../examples/intermediate/rope.py).

The RoPE example splits each head into two halves and applies the rotary
position transform with column broadcasts.

```bash
python examples/intermediate/rope.py -p a2a3sim
```

Key topics: transformer tensor layout, half-vector slicing, broadcast
multiplication, and assembling an output from slices.

## Softmax

[View the source](../../examples/intermediate/softmax.py).

This is a numerically stable row-wise softmax: subtract the row maximum,
exponentiate, reduce the denominator, and broadcast the division.

```bash
python examples/intermediate/softmax.py -p a2a3sim
```

Key topics: stable reductions, `row_max`, `row_sum`, and row broadcast.

## Suggested reading order

Use this order when learning the DSL:

1. GEMM for a loop-carried tile accumulator.
2. Softmax for a compact reduction-and-broadcast pipeline.
3. LayerNorm for multiple dependent reductions.
4. RMSNorm for a reduction split across hidden chunks.
5. RoPE for transformer-specific slicing and layout.
