# Advanced and Distributed Examples

These examples focus on fusion, reusable inline functions, specialized tile
instructions, and distributed L3 execution. The first three are single-device
cases. All-reduce requires two ranks.

All four scripts declare `a2a3`, `a2a3sim`, `a5`, and `a5sim`. Repository CI
exercises `a2a3`, `a2a3sim`, and `a5sim`; A5 device execution is not currently
covered by CI.

## GEMM plus elementwise

[View the source](../../examples/advanced/gemm_eltwise.py).

This kernel performs a BF16 matrix multiplication with an FP32 accumulator,
then adds the residual before leaving the same core-group scope.

```bash
python examples/advanced/gemm_eltwise.py -p a2a3sim
```

Key topics: operation fusion, on-chip value reuse, mixed input and output
dtypes, and a tiled K reduction.

## Multi-projection

[View the source](../../examples/advanced/multi_proj.py).

`qkv_proj` calls one shared `@pl.jit.inline` projection body three times for
Q, K, and V. It shows how to reuse a kernel fragment without creating a
separate runtime dispatch for each helper call.

```bash
python examples/advanced/multi_proj.py -p a2a3sim
```

Key topics: inline JIT functions, multiple outputs, and repeated tiled
projections.

## Top-k

[View the source](../../examples/advanced/topk.py).

The top-k example builds a row-wise selection from `sort32` and chained
`mrgsort` instructions. It extracts values and indices from interleaved pairs
and uses a paired comparator so tied values may select any valid tied index.

```bash
python examples/advanced/topk.py -p a2a3sim
```

Key topics: specialized sort instructions, mask-based gather, tie-aware golden
validation, and sequential row handling within a core group.

## Two-rank all-reduce

[View the source](../../examples/advanced/allreduce.py).

All-reduce is the only multi-device example. It uses distributed window
buffers, remote tile loads, and notify/wait synchronization. The program is
statically fixed at two ranks, and the CLI requires exactly two device IDs.
[Distributed Programming](../pypto-coding/distributed-programming.md) is the
reference for every construct it uses.

Run it on two A2/A3 devices:

```bash
python examples/advanced/allreduce.py -p a2a3 -d 0,1
```

To check compilation without executing the distributed program:

```bash
python examples/advanced/allreduce.py -p a2a3 --compile-only -d 0,1
```

The `# ci: devices=2` marker is executable test metadata: the A2/A3 CI job
borrows two cards for this case. It does not imply that larger world sizes are
supported.
