# Save and Replay Golden Data

The Golden Harness can persist one run's inputs and expected outputs, then
load that snapshot during later runs. This is useful when repeated
performance or profiling iterations preserve the same numerical contract.

Replay does not weaken validation: the new runtime output is still compared
with the saved expected output.

## When to use replay

Good uses include:

- tile-size and buffer-size sweeps;
- loop or scheduling changes intended to preserve results;
- repeated runtime profiling;
- deterministic reproduction of one failing input.

Do not freeze a golden while changing:

- the intended mathematics or golden function;
- tensor names, order, shapes, or dtypes;
- scalar values or input initialization;
- quantization, rounding, or precision fixtures;
- an RNG seed when the purpose is to explore different inputs.

For precision debugging, generate fresh data unless reproducing one known
failure is the explicit goal.

## Harness API

`run` takes two keyword arguments for this:

```python
result = run(
    fn=kernel,
    specs=specs,
    golden_fn=golden_fn,
    golden_data=args.golden_data,
    save_data=args.save_data,
    config=config,
)
```

- `save_data=True` writes generated inputs and computed golden outputs under
  the run's work directory.
- `golden_data=<directory>` reads `in/` and `out/` from that directory.
- `golden_data` takes precedence over `golden_fn`.
- `save_data` defaults to `False`; ordinary validation remains in memory and
  does not create a snapshot.

If the kernel CLI should expose this behavior, wire both options directly:

```python
parser.add_argument(
    "--save-data",
    action="store_true",
    default=False,
    help="persist inputs and golden outputs for replay",
)
parser.add_argument(
    "--golden-data",
    type=str,
    default=None,
    help="directory containing cached in/ and out/ tensors",
)
```

Then forward `args.save_data` and `args.golden_data` to `run`.
Do not assume every existing entry point already exposes both flags; check its
`--help` and call site.

## Capture a snapshot

Run once with saving enabled:

```bash
PYTHONPATH="$PWD" \
  python path/to/kernel.py -p a2a3 -d 0 --save-data
```

Only keep a snapshot from a passing run. The `RunResult.work_dir` is the
authoritative build directory, and the snapshot is:

```text
<work_dir>/data/
├── in/
│   └── <input-or-scalar-name>.pt
└── out/
    └── <output-name>.pt
```

To find recent snapshots created beneath the default output root:

```bash
find build_output -type d -name data -print
```

The output location is relative to the directory from which the script was
launched unless the compile configuration overrides it.

## Replay the snapshot

Point the next run at the `data/` directory, not its `in/` or `out/`
subdirectory:

```bash
export GOLDEN_SNAPSHOT="build_output/<program-and-timestamp>/data"

PYTHONPATH="$PWD" \
  python path/to/kernel.py -p a2a3 -d 0 \
    --golden-data "$GOLDEN_SNAPSHOT"
```

A replay reports both cache hits:

```text
[RUN] generate inputs ...
[RUN]   cache hit: .../data/in
[RUN] compute golden ...
[RUN]   cache hit: .../data/out
```

The harness checks that every required file exists before runtime:

| Spec kind | Required cache files |
|---|---|
| `ScalarSpec` | `in/<name>.pt` |
| pure tensor input | `in/<name>.pt` |
| pure tensor output | `out/<name>.pt` |
| initialized output / inout tensor | both `in/<name>.pt` and `out/<name>.pt` |

An incomplete snapshot returns a failed `RunResult` with the missing paths.

## Invalidation rules

Delete or archive the old snapshot and capture a new one whenever its specs,
inputs, or reference computation are no longer the intended test. File
presence alone cannot prove that cached tensor data still matches revised
kernel semantics.

Snapshots can be large, especially for model weights. Keep them in
`build_output/` or another untracked location; do not commit generated `.pt`
files.

## Relationship to runtime_dir

`golden_data` and `runtime_dir` are independent:

- `golden_data` skips input generation and golden computation;
- `runtime_dir` reuses a precompiled work directory and skips PyPTO compile;
- validation still runs in either case;
- the `golden_data` cache is read-only during replay.

Use `runtime_dir` only while the compiled kernel logic remains compatible with
that build. See [Compile and Runtime Workflow](compile-runtime-workflow.md) and
[Debugging](../debug-and-tune/debugging.md) for compile reuse details.

The behavior above is implemented in
[`golden/runner.py`](../../golden/runner.py) and covered by
[`tests/golden/test_runner.py`](../../tests/golden/test_runner.py).
