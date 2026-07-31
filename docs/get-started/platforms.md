# Platforms and Devices

The `platform` passed to the Golden Harness selects both the PyPTO backend and
the simpler runtime target.

| CLI value | PyPTO backend | Execution target | Device argument |
|---|---|---|---|
| `a2a3sim` | `Ascend910B` | A2/A3 simulator | No physical NPU is selected |
| `a2a3` | `Ascend910B` | Ascend 910B/C NPU | Usually one integer device ID |
| `a5sim` | `Ascend950` | A5 simulator | No physical NPU is selected |
| `a5` | `Ascend950` | Ascend 950 NPU | Usually one integer device ID |

The mapping is enforced by
[`golden.runner._backend_for_platform`](../../golden/runner.py). An unknown
platform fails instead of silently choosing a default backend.

## Architecture is not the platform

`uname -m` is used during installation to select a PTOAS release asset for the
host CPU architecture. It does not determine whether a script uses a
simulator or a real NPU.

Choose execution explicitly:

```bash
PYTHONPATH="$PWD" \
  python examples/beginner/hello_world.py -p a2a3sim

PYTHONPATH="$PWD" \
  python examples/beginner/hello_world.py -p a2a3 -d 0
```

The current CI makes the same distinction: simulator jobs do not source CANN,
while real-device jobs source CANN and verify the NPU before execution.

## Inspect each entry point

The beginner examples accept:

```text
-p {a2a3,a2a3sim,a5,a5sim}
-d <integer>
```

Do not generalize that device shape to every model. A distributed program can
require a comma-separated device set and an explicit world-size argument.
Some large programs are marked `# ci: no-sim`, or take a compile-only path on
a simulator.

Always inspect the selected file:

```bash
PYTHONPATH="$PWD" python path/to/kernel.py --help
rg -n '^#\s*ci:' path/to/kernel.py
```

## What CI coverage means

For selected runnable changes, pull-request CI exercises:

- `a2a3sim` and `a5sim` in the simulator matrix, unless the file opts out;
- `a2a3` on a real NPU, including declared multi-card allocations.

Additional scheduled jobs are configured for broader model and A5-device
cases. A CLI choice means an entry point accepts the target; it does not by
itself prove that every path in that program is validated on the target.
Consult the [Examples](../examples/index.md) and
[Models](../models/index.md) pages for the stated coverage of a particular
entry point.

## Shared device hosts

On a shared machine, do not choose an apparently idle device by probing and
then racing another user. Use the site's device allocator or scheduler and
pass the allocated ID or ID set to the kernel.

For a direct device shell:

```bash
source "$CANN_ROOT/set_env.sh"
npu-smi info
```

The repository CI uses `task-submit` to allocate devices, but that service is
infrastructure-specific and is not required by PyPTO-Lib itself.
