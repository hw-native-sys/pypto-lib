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
- With `golden_fn=None`, `save_data=True` writes only `in/`, and a later
  `golden_data` replay of that directory reuses the inputs and skips
  validation — see [Inputs-only snapshots](#inputs-only-snapshots).
- `save_data` defaults to `False`; ordinary validation remains in memory and
  does not create a snapshot.
- `golden_only=True` stops the run right after the golden is computed and
  forces `save_data=True`; see [Capture without a
  device](#capture-without-a-device).

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

## Capture without a device

Only the runtime phase needs a device. Compile, input generation and the Torch
golden are CPU work, and on a shared host the golden is usually the longest of
the four — so a plain `--save-data` capture holds a die for minutes while
computing something the die never touches.

`--golden-only` stops the run immediately after the golden and persists it:

```bash
# No device: compile + generate inputs + compute golden.
PYTHONPATH="$PWD" \
  python path/to/kernel.py -p a2a3 --golden-only
```

```text
[RUN] compile ...
[RUN] generate inputs ...
[RUN] compute golden ...
[RUN] PASS (412.11s, golden saved to build_output/<program-and-timestamp>/data)
```

Hand that directory to a second invocation, which is the only one that needs a
card:

```bash
PYTHONPATH="$PWD" \
  python path/to/kernel.py -p a2a3 -d 0 \
    --golden-data "build_output/<program-and-timestamp>/data"
```

The split changes nothing about validation: the device outputs are still
compared against the same golden. It only moves where the golden is computed.

`golden_only` forces `save_data=True` — a golden that is not persisted would be
wasted work, so entries that keep `--save-data` off because their fixtures are
large will write that snapshot here regardless. Nothing then deletes it for you:
a manual run leaves both the snapshot and the build around it under
`build_output/` until you remove them, and a full-model fixture reaches ~1GB
(`models/qwen3_14b/decode_fwd.py`). CI is the exception — its driver
(`.github/scripts/run_device_cases.py`) drops each producing build as soon as
the device run has replayed it, which is also what bounds how many exist at
once.

`golden_only` requires `golden_fn`, and the harness rejects it alongside
`golden_data` (the golden already exists, so there is nothing to produce),
`compile_only` (which stops one phase earlier) and `runtime_dir`.

Wire it next to the other two flags:

```python
parser.add_argument(
    "--golden-only",
    action="store_true",
    default=False,
    help="compute and persist the golden, then stop before the device run",
)
```

The platform still matters: it is what the compile phase targets, so pass the
same `-p` the device run will use. `-d` is irrelevant here and can be omitted.

A `-p a2a3 --golden-only` run opens no per-die device node: the only
`/dev/davinci*` traffic is the driver-library probe that importing `golden`
(torch_npu / CANN) performs anyway, and the run passes on a host where that
probe is denied. So the producing half needs no lease at all, not merely a
shorter one.

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

### Inputs-only snapshots

An entry that passes `golden_fn=None` (typically a performance-only model
entry whose torch reference is too expensive or absent) still benefits from a
snapshot: input generation is skipped on replay. Capturing with `save_data=True`
writes only `in/`, since there is no golden to save.

Replaying such a directory requires only the `in/` rows of the table above.
The harness detects the case — `golden_fn` is `None` and the directory has no
`out/` — and says so:

```text
[RUN]   golden_data has no out/: reusing inputs only
[RUN] generate inputs ...
[RUN]   cache hit: .../data/in
...
[RUN] PASS (…s, validation skipped: golden_data has no out/)
```

The run passes without comparing outputs. If `out/` is expected but was lost
in a copy, this is the line that reveals it. A directory without `out/` still
fails when a `golden_fn` is passed, and a directory that has `out/` is always
validated against it.

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
- `golden_only` produces what `golden_data` later consumes, and is rejected
  together with `runtime_dir`;
- `runtime_dir` reuses a precompiled work directory and skips PyPTO compile;
- validation still runs in either case;
- the `golden_data` cache is read-only during replay.

Use `runtime_dir` only while the compiled kernel logic remains compatible with
that build. See [Compile and Runtime Workflow](compile-runtime-workflow.md) and
[Debugging](../debug-and-tune/debugging.md) for compile reuse details.

The behavior above is implemented in
[`golden/runner.py`](../../golden/runner.py) and covered by
[`tests/golden/test_runner.py`](../../tests/golden/test_runner.py).
