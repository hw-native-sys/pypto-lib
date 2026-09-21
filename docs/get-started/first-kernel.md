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

The simulator path requires neither CANN nor an NPU. It still requires the
PyPTO, simpler, PTOAS, PTO ISA, and compiler setup described on the
installation page.

## Run on an A2/A3 device

In a shell with the CANN environment loaded and a visible device:

```bash
source "$CANN_ROOT/set_env.sh"
npu-smi info
PYTHONPATH="$PWD" \
  python examples/beginner/hello_world.py -p a2a3 -d 0
```

`-d 0` selects device 0. Do not assume it is free on a shared host — use that
host's allocator rather than probing for an idle card and racing another user.

## Platforms and devices

`-p` selects both the PyPTO backend and the simpler runtime target:

| CLI value | PyPTO backend | Execution target | Device argument |
|---|---|---|---|
| `a2a3sim` | `Ascend910B` | A2/A3 simulator | none |
| `a2a3` | `Ascend910B` | Ascend 910B/C NPU | usually one integer ID |
| `a5sim` | `Ascend950` | A5 simulator | none |
| `a5` | `Ascend950` | Ascend 950 NPU | usually one integer ID |

The mapping is enforced by
[`golden.runner._backend_for_platform`](../../golden/runner.py) — an unknown
platform fails instead of silently choosing a default. The host's own CPU
architecture (`uname -m`) only picks the PTOAS release asset at install time; it
does not decide whether a run uses a simulator or a real NPU.

Two caveats hold everywhere beyond the beginner examples. **Declared is not
validated**: a CLI choice means the script accepts that target, not that every
path in it was verified there. And **the device argument is per-script** — a
distributed entry takes a comma-separated set plus a world-size argument, and
some large programs are device-only, or take a compile-only path on a simulator.
Read the target's `--help` first.

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

Read [L2 Programming](../pypto-coding/l2-programming.md) and
[Operations](../pypto-coding/operations.md) before editing this or another
kernel.

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
