---
name: test-with-golden
description: Speed up iterative kernel testing by generating the golden reference ONCE (save_data=True) and replaying it on every later run via golden_data, skipping input generation and the torch golden recompute each iteration. Adds a --save-data flag to the kernel when it lacks one. Use for performance/timing tuning where the kernel's numerical result is unchanged between runs; NOT recommended for precision debugging.
---

# Test with a frozen golden (`test-with-golden`)

Iterating on a kernel usually means running the same golden test dozens of
times. Every run regenerates random inputs and recomputes the torch golden —
work that is pure overhead when you are only changing *how fast* the kernel
runs, not *what* it computes. This skill freezes the golden once and replays
it, so each later run is just **compile + device + validate**.

The harness already supports this through two `run` / `run_jit` kwargs:

- `save_data=True` — persist generated inputs to `{work_dir}/data/in/` and the
  golden outputs to `{work_dir}/data/out/`.
- `golden_data=<dir>` — load inputs from `<dir>/in/` and expected outputs from
  `<dir>/out/` instead of generating fresh data and recomputing the golden.
  `golden_data` takes precedence over `golden_fn`.

Since `save_data` now defaults to **False**, the first run must explicitly opt
in to produce the snapshot; every run after that reuses it.

## When to use

**Use it for performance / timing work** — tile-size sweeps, scope tuning,
loop-construct changes, buffer sizing, swimlane / PMU profiling. The kernel's
numerical output is expected to be identical across iterations, so a single
frozen golden stays valid and you save the per-run generation + golden cost.

**Do NOT use it for precision debugging.** Precision work changes the numerics
(quant schemes, dtype/rounding, fixtures) and often varies the RNG seed to
probe input-dependent error. A frozen golden pins one input sample, hiding
that variation, and any fixture/quant change silently invalidates the cached
`data/out/`. Regenerate the golden every run there instead.

## Workflow

### 1. First run — generate and persist the golden

If the kernel's `__main__` exposes a `--save-data` flag, use it:

```bash
python models/deepseek/v4/decode_attention_swa.py -p a2a3 -d 0 --save-data
```

**If the kernel has no `--save-data` flag, add it as part of this workflow.**
`save_data` defaults to False, so a kernel without the flag never persists a
snapshot and there is nothing to replay. Grep the kernel's `__main__` for
`--save-data`; when it is missing, wire the flag in with two small edits:

1. Add the argument next to the existing `--golden-data` (or the other flags):

   ```python
   parser.add_argument("--save-data", action="store_true", default=False,
                       help="persist inputs + golden to data/ for later --golden-data replay")
   ```

2. Forward it into the `run` / `run_jit` call:

   ```python
   save_data=args.save_data,
   ```

If the kernel also lacks `--golden-data`, add that too
(`parser.add_argument("--golden-data", type=str, default=None)` +
`golden_data=args.golden_data`) so step 3 has a knob to point at. These edits
are local to the kernel's CLI; keep them only as long as you are iterating,
and revert them when done unless the kernel should carry the flags upstream.

### 2. Locate the persisted data directory

The snapshot lands under the compiled output dir, `{work_dir}/data`, which is
`build_output/<program>/data` **relative to the directory you launched the
kernel from** (not necessarily next to the kernel file):

```bash
find build_output -type d -name data
# -> build_output/_jit_<program>_<timestamp>/data   (contains in/ and out/)
```

You can also read `result.work_dir` from the `RunResult` — the data dir is
`<work_dir>/data`.

### 3. Subsequent runs — replay the frozen golden

Point every later run at that directory. Input generation and the torch golden
are both skipped (the run logs a `cache hit` for each); validation still runs
against the cached `out/`:

```bash
python models/deepseek/v4/decode_attention_swa.py -p a2a3 -d 0 \
    --golden-data build_output/_jit_<program>_<timestamp>/data
```

A successful replay prints, before the runtime stage:

```
[RUN] generate inputs ...
[RUN]   cache hit: build_output/_jit_<program>_<timestamp>/data/in
[RUN] compute golden ...
[RUN]   cache hit: build_output/_jit_<program>_<timestamp>/data/out
```

For kernels without a `--golden-data` flag, add one (step 1) or pass
`golden_data="<dir>"` at the `run` / `run_jit` call site instead.

## Requirements and caveats

- **Specs must be unchanged.** Replay loads `data/in/*.pt` / `data/out/*.pt`
  by spec name; if you change tensor shapes, dtypes, the spec set, or the RNG
  seed, the cached data no longer matches. Delete the `data/` dir and redo
  step 1.
- **Golden logic changes invalidate the cache.** If you edit `golden_fn` (the
  reference computation), the cached `out/` is stale — regenerate it.
- **Compile still runs.** `golden_data` only skips input/golden generation, not
  codegen. To also skip recompilation between iterations, combine with
  `runtime_dir=<build_output/...>` (valid only while the kernel source is
  unchanged; cpp edits are picked up, kernel-logic edits are not).
- **Validation is unaffected.** The device output is always compared against
  the (cached or freshly computed) golden; replay does not weaken the check.
