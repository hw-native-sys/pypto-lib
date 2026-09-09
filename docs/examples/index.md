# Examples

The `examples/` tree contains small, self-contained kernels arranged as a
learning path. Every example has a command-line entry point, builds synthetic
inputs, computes a Torch golden result, runs through the Golden Harness, and
exits nonzero when validation fails.

Start with [Beginner](beginner.md), continue with
[Intermediate](intermediate.md), and use
[Advanced and distributed](advanced.md) for composition, specialized
instructions, and multi-device execution.

## Platform status

**Declared** means the script accepts the platform in its `--platform` choices.
All 11 tracked examples declare `a2a3`, `a2a3sim`, `a5`, and `a5sim`. An
accepted `-p a5` argument is not by itself evidence that the example has been
validated on an A5 device.

| Level | Example | Main topic | Devices | Declared |
| --- | --- | --- | ---: | --- |
| Beginner | [Hello World](../../examples/beginner/hello_world.py) | Tiled scalar add | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Beginner | [Matmul](../../examples/beginner/matmul.py) | M/N tiled matrix multiply | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Intermediate | [GEMM](../../examples/intermediate/gemm.py) | M/N/K tiling and accumulation | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Intermediate | [LayerNorm](../../examples/intermediate/layer_norm.py) | Row reduction and broadcast | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Intermediate | [RMSNorm](../../examples/intermediate/rms_norm.py) | Chunked reduction | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Intermediate | [RoPE](../../examples/intermediate/rope.py) | Rotary embedding | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Intermediate | [Softmax](../../examples/intermediate/softmax.py) | Stable row softmax | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Advanced | [GEMM + elementwise](../../examples/advanced/gemm_eltwise.py) | Fused residual add | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Advanced | [Multi-projection](../../examples/advanced/multi_proj.py) | Reusable inline kernels | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Advanced | [Top-k](../../examples/advanced/topk.py) | Sort and merge instructions | 1 | A2/A3, A2/A3 sim, A5, A5 sim |
| Advanced | [All-reduce](../../examples/advanced/allreduce.py) | L3 distributed execution | 2 | A2/A3, A2/A3 sim, A5, A5 sim |

## Running an example

Use a simulator for the quickest functional check:

```bash
python examples/beginner/hello_world.py -p a2a3sim
```

Use a real A2/A3 device by selecting its device ID:

```bash
python examples/intermediate/softmax.py -p a2a3 -d 0
```

The single-device examples also accept `--enable-chip-swimlane` for a chip
swimlane timeline capture; it takes a level `0`-`4`, and a bare flag means
level 1. The distributed all-reduce instead takes a comma-separated device list
and requires exactly two ranks:

```bash
python examples/advanced/allreduce.py -p a2a3 -d 0,1
```

The examples use synthetic fixtures. They demonstrate kernel construction and
validation; they do not load model checkpoints.
