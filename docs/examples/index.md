# Examples

The `examples/` tree contains small, self-contained kernels arranged as a
learning path. Every example has a command-line entry point, builds synthetic
inputs, computes a Torch golden result, runs through the Golden Harness, and
exits nonzero when validation fails.

Start with [Beginner](beginner.md), continue with
[Intermediate](intermediate.md), and use
[Advanced and distributed](advanced.md) for composition, specialized
instructions, and multi-device execution.

## Platform and CI status

Two kinds of support are reported here:

- **Declared** means the script accepts the platform in its `--platform`
  choices.
- **CI coverage** means the repository workflow invokes the script on that
  platform. It describes the configured test target; consult the latest
  workflow run for the result of a particular revision.

All 11 tracked examples declare `a2a3`, `a2a3sim`, `a5`, and `a5sim`. The
regular CI workflow runs them on `a2a3`, `a2a3sim`, and `a5sim`. It does not
currently run examples on an `a5` device, so an accepted `-p a5` argument is
not the same as CI-verified A5 execution.

| Level | Example | Main topic | Devices | Declared | CI coverage |
| --- | --- | --- | ---: | --- | --- |
| Beginner | [Hello World](../../examples/beginner/hello_world.py) | Tiled scalar add | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Beginner | [Matmul](../../examples/beginner/matmul.py) | M/N tiled matrix multiply | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Intermediate | [GEMM](../../examples/intermediate/gemm.py) | M/N/K tiling and accumulation | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Intermediate | [LayerNorm](../../examples/intermediate/layer_norm.py) | Row reduction and broadcast | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Intermediate | [RMSNorm](../../examples/intermediate/rms_norm.py) | Chunked reduction | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Intermediate | [RoPE](../../examples/intermediate/rope.py) | Rotary embedding | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Intermediate | [Softmax](../../examples/intermediate/softmax.py) | Stable row softmax | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Advanced | [GEMM + elementwise](../../examples/advanced/gemm_eltwise.py) | Fused residual add | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Advanced | [Multi-projection](../../examples/advanced/multi_proj.py) | Reusable inline kernels | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Advanced | [Top-k](../../examples/advanced/topk.py) | Sort and merge instructions | 1 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |
| Advanced | [All-reduce](../../examples/advanced/allreduce.py) | L3 distributed execution | 2 | A2/A3, A2/A3 sim, A5, A5 sim | A2/A3, A2/A3 sim, A5 sim |

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
swimlane timeline capture. The distributed all-reduce instead takes a
comma-separated device list and requires exactly two ranks:

```bash
python examples/advanced/allreduce.py -p a2a3 -d 0,1
```

The examples use synthetic fixtures. They demonstrate kernel construction and
validation; they do not load model checkpoints.
