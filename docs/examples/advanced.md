# Advanced and Distributed Examples

These examples focus on fusion, reusable inline functions, specialized tile
instructions, and distributed L3 execution. The first three are single-device
cases. The last two are multi-device: all-reduce is statically fixed at two
ranks, and the L3 all-gather + GEMM example runs any rank count (at least
two).

All five scripts declare `a2a3`, `a2a3sim`, `a5`, and `a5sim`. Repository CI
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

## N-rank L3 all-gather + GEMM

[View the source](../../examples/advanced/l3_allgather_gemm.py).

This example generalizes the all-reduce pattern to an arbitrary rank count.
Each rank splits its local `A` shard into row chunks and publishes them to
every peer with explicit `pld.tensor.put`, then notifies a per-chunk signal
cell. A single merged AIC kernel runs on multiple cores: some tasks compute
the local chunk GEMMs right away, while others wait on the exact
`[src_rank, chunk_idx]` signal before computing the matching remote chunk.
`comm_local_shard` and `gemm_from_gathered` are launched with independent
SPMD core counts via `--comm-cores` and `--gemm-cores`.

Run it on two or more A2/A3 devices:

```bash
python examples/advanced/l3_allgather_gemm.py -p a2a3 -d 0,1
```

The rank count is inferred from the length of `-d`; three or more comma-separated
device IDs run a larger allgather. Add `--benchmark` for device/host timing
summaries.

The `# ci: devices=2` marker mirrors all-reduce: CI runs the default two-rank
case, not the general N-rank path.
