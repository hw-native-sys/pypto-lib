# Run Your First Kernel

The smallest end-to-end example is
[`examples/beginner/hello_world.py`](../../examples/beginner/hello_world.py).
It adds a scalar to every element of an FP32 matrix:

```text
y[r, c] = x[r, c] + a
```

The file contains four parts:

1. a module-level `@pl.jit` kernel;
2. `TensorSpec` and `ScalarSpec` inputs for the Golden Harness;
3. a PyTorch golden function;
4. a CLI that calls `golden.run` and exits non-zero on failure.

## Run on the A2/A3 simulator

Activate the environment from
[Installation and Environment](installation.md), change to the repository
root, and run:

```bash
PYTHONPATH="$PWD" \
  python examples/beginner/hello_world.py -p a2a3sim
```

The harness reports these stages:

```text
[RUN] compile ...
[RUN] generate inputs ...
[RUN] compute golden ...
[RUN] runtime ...
[RUN]   'y' PASS ...
[RUN] PASS (...)
```

The simulator path does not require CANN or an NPU in this repository's CI.
It still requires the PyPTO, simpler, PTOAS, PTO ISA, and compiler setup
described on the installation page.

## Run on an A2/A3 device

In a shell with the CANN environment loaded and a visible device:

```bash
source "$CANN_ROOT/set_env.sh"
npu-smi info
PYTHONPATH="$PWD" \
  python examples/beginner/hello_world.py -p a2a3 -d 0
```

`-d 0` selects device 0. Do not assume that device is available on a shared
host; use the allocation mechanism provided by that host.

See [Platforms and Devices](platforms.md) before changing `-p`, and check the
target script's `--help` because distributed and model entry points may use a
different device argument shape.

## Read the kernel

The kernel divides the matrix into row and column tiles:

```python
for r in pl.parallel(0, ROWS, ROW_TILE):
    for c in pl.range(0, COLS, COL_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="add_scalar"):
            tile_x = x[r : r + ROW_TILE, c : c + COL_TILE]
            y[r : r + ROW_TILE, c : c + COL_TILE] = pl.add(tile_x, a)
```

- `pl.parallel` distributes row tiles across core groups.
- `pl.range` walks the column tiles assigned within that structure.
- `pl.at` defines an InCore region.
- The slice load, `pl.add`, and slice store operate on one tile.

Read the [kernel coding style](../pypto-coding/pypto-coding-style.md) before
editing this or another kernel.

## Understand the validation

`build_specs()` explicitly initializes `x` with `torch.randn`, sets scalar
`a` to `1.0`, and marks `y` as an output. The golden function fills its output
with the equivalent PyTorch expression:

```python
def golden_hello_world(values):
    values["y"][:] = values["x"] + values["a"]
```

The harness compares the device or simulator result with this reference at
`rtol=1e-5` and `atol=1e-5`. A mismatch produces a failed `RunResult`, and the
CLI exits with status 1.

Generated files are written to the run's work directory under
`build_output/`. Continue with the
[Golden Harness](../run-and-validate/golden-harness.md) or the detailed
[compile and runtime workflow](../run-and-validate/compile-runtime-workflow.md).
