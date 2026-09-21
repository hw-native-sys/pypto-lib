# Examples

`examples/` holds small, self-contained kernels arranged as a learning path.
Every example has a command-line entry point, builds synthetic inputs, computes
a Torch golden result, runs through the Golden Harness, and exits nonzero when
validation fails. They demonstrate kernel construction and validation; none of
them loads a model checkpoint.

They all share one shape: declare a `@pl.jit` kernel, split the problem into
`pl.parallel` tiles, do the work inside a `pl.at` core-group scope, describe
inputs and outputs with Golden Harness specs, and compare the result against a
Torch reference. Read them in listed order — each one adds a single idea to the
one above it.

## The catalog

| Level | Example | What it adds |
| --- | --- | --- |
| Beginner | [Hello World](../../examples/beginner/hello_world.py) | The whole path from a tensor signature to a validated result: `pl.parallel` row tiles, `pl.range` column tiles, one `pl.add` |
| Beginner | [Matmul](../../examples/beginner/matmul.py) | M/N tiling with K in a single tile — the first cube example, before a reduction loop |
| Intermediate | [GEMM](../../examples/intermediate/gemm.py) | K tiling: the first tile creates the accumulator with `pl.matmul`, later tiles update it with `pl.matmul_acc` |
| Intermediate | [Softmax](../../examples/intermediate/softmax.py) | A stable reduce-and-broadcast pipeline — `row_max`, `row_sum`, row broadcast |
| Intermediate | [LayerNorm](../../examples/intermediate/layer_norm.py) | Two dependent reductions in one tile, then gamma and beta by broadcast |
| Intermediate | [RMSNorm](../../examples/intermediate/rms_norm.py) | A reduction wider than one tile: chunked accumulation, then a second normalizing pass |
| Intermediate | [RoPE](../../examples/intermediate/rope.py) | Transformer layout — half-head slicing, column broadcast, assembling an output from slices |
| Advanced | [GEMM + elementwise](../../examples/advanced/gemm_eltwise.py) | Fusion: a BF16 matmul with an FP32 accumulator adds its residual without leaving the scope |
| Advanced | [Multi-projection](../../examples/advanced/multi_proj.py) | One `@pl.jit.inline` body reused for Q, K and V without a dispatch per call |
| Advanced | [Top-k](../../examples/advanced/topk.py) | Specialized instructions — `sort32` and chained `mrgsort`, mask gather, tie-aware validation |
| Advanced | [All-reduce](../../examples/advanced/allreduce.py) | L3 distributed execution: window buffers, remote tile loads, notify/wait |

All 11 declare `a2a3`, `a2a3sim`, `a5`, and `a5sim` in their `--platform`
choices. An accepted `-p a5` is not by itself evidence that the example has been
validated on an A5 device.

## Running an example

Use a simulator for the quickest functional check, or a real device by ID:

```bash
python examples/beginner/hello_world.py -p a2a3sim
python examples/intermediate/softmax.py -p a2a3 -d 0
```

The single-device examples also accept `--enable-chip-swimlane` for a timeline
capture; it takes a level `0`-`4`, and a bare flag means level 1.

All-reduce is the only multi-device case. It is written for exactly two ranks
and its CLI requires two device IDs —
[L3 Programming](../pypto-coding/l3-programming.md) is the
reference for every construct it uses:

```bash
python examples/advanced/allreduce.py -p a2a3 -d 0,1
python examples/advanced/allreduce.py -p a2a3 -d 0,1 --compile-only
```
