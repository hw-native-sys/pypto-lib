# Debugging

A practical guide to debugging pypto-lib kernels — from compile errors
through runtime hangs to precision mismatches. It pairs with
[compile-runtime-workflow.md](compile-runtime-workflow.md) (what each phase
does) and [performance-tuning.md](performance-tuning.md) (perf). To locate
*which pypto commit* introduced a precision regression, use the
`bisect-precision` skill instead.

The harness exposes most of these as both a `run` / `run_jit` kwarg and a
CLI flag; a typical model `__main__` wires them up like:

```python
parser.add_argument("--runtime-dir", type=str, default=None)
parser.add_argument("--dump-tensor", action="store_true")
parser.add_argument("--enable-dep-gen", action="store_true")
...
result = run_jit(
    fn=indexer_test,
    specs=build_tensor_specs(...),
    golden_fn=golden_indexer,
    runtime_dir=args.runtime_dir,                     # reuse a compile (§3)
    runtime_cfg=dict(
        platform=args.platform, device_id=args.device,
        log_level="v5",                               # runtime log level; raise to v0 for hangs (§4)
        enable_dump_tensor=args.dump_tensor,          # precision dump (§5)
        enable_dep_gen=args.enable_dep_gen,           # dependency graph (§6)
    ),
    rtol=1e-3, atol=1e-3,
)
```

---

## 1. Read the pypto / ptoas error first

pypto's IR verifier and ptoas's assembler emit **direct, actionable**
errors — they name the offending op, the bad shape / layout, and a source
location. Most compile failures are fixed at the cited site without any
further tooling; read the message before reaching for the heavier
mechanisms below.

- **Compile failure** — the IR after every pass is under
  `build_output/<...>/passes_dump/` (written by default, `dump_passes=True`).
  Diff the last clean pass against the first failing one to see which pass
  rejected the IR. `report/` holds scheduling diagnostics.
- **ptoas failure** — the error quotes the `.pto` op. `skip_ptoas=True`
  (a `compile_cfg` knob) keeps the raw `.pto` MLIR and stops before the C++
  wrapper, isolating whether the regression is in pypto's IR→MLIR or in
  ptoas.
- **Runtime crash** — rerun on the matching simulator (`-p a2a3sim` /
  `a5sim`); it gives more diagnostic output than the device backend and
  reproduces most lowering bugs.

---

## 2. Replay failing data with `golden_data`

Every run snapshots its inputs to `data/in/<name>.pt` and its golden
outputs to `data/out/<name>.pt` inside the build directory (unless the run
used `save_data=False` — e.g. the `--save-data`-off full-model kernels — in
which case nothing was saved and there is nothing to replay). To reproduce a
failure on the **exact same tensors** instead of re-rolling random data,
point a re-run at that directory:

```bash
python models/deepseek/v4/decode_attention_csa.py -p a2a3 -d 0 \
    --golden-data build_output/_jit_attention_csa_test_20260602_020256/data
```

`golden_data="<dir>"` loads `<dir>/in/*.pt` as inputs and `<dir>/out/*.pt`
as the reference, skipping both input generation and `golden_fn` — it wins
over `golden_fn`. This makes a mismatch deterministic and is the starting
point for every precision investigation.

---

## 3. Reuse a compile with `runtime_dir`; edit `.cpp` / `.pto` and retest

`runtime_dir="<build_output dir>"` (CLI `--runtime-dir`) **skips compile and
codegen** and runs the existing artifacts straight through the
runtime/simpler. This is the tight loop for hand-editing generated code —
a generated kernel or the orchestration — and re-testing in seconds:

1. Edit any `kernels/aic/*.cpp` / `kernels/aiv/*.cpp` or
   `orchestration/*.cpp`. You can also edit the raw `ptoas/*.pto` MLIR — the
   harness splices `.pto` edits back into the owning `.cpp`
   (`rebuild_kernel_cpp_from_pto`) and bumps its mtime.
2. Re-run with `--runtime-dir` pointing at the same directory (note: the
   build directory itself, **not** its `data/` subdir):

   ```bash
   python models/deepseek/v4/decode_attention_csa.py -p a2a3 -d 0 \
       --runtime-dir build_output/_jit_attention_csa_test_20260602_020256
   ```

   The harness flags every `.cpp` whose `.so`/`.o` is **missing or older
   than the `.cpp`**, drops the cached binaries for the build, and the
   runtime rebuilds them. You do **not** need to `rm` the `.o`/`.so`
   yourself — editing the `.cpp` (mtime bump) is the signal.

The log states which path it took:

```
[cpp->.so] cpp edits or missing binaries detected (2 file(s)): kernels/aiv/foo.cpp, ...; rebuilding
[cpp->.so] no cpp edits since last build; reusing cached binaries
```

> A single stale `.cpp` invalidates the cached binaries for the whole build
> directory, so **batch all your edits before one run** rather than
> re-running per file.

---

## 4. Runtime hang / deadlock — device log via `log_level` + `ASCEND_PROCESS_LOG_PATH`

When a run **hangs** (no progress, then AICPU 2-second sync timeouts) rather
than raising a clean error, the Python side has nothing to show — the stall
is on the device. Raise the runtime log verbosity and read the device log:

1. Set `log_level` in `runtime_cfg`:

   ```python
   runtime_cfg=dict(platform=..., device_id=..., log_level="v0")
   ```

   `log_level` is a harness-only key — it is consumed up front
   (`configure_log`) and not forwarded to the runtime call. Accepted values:
   `debug`, `v0`..`v9`, `info`, `warn`, `error`, `null`. The runtime default
   is `v5` (= INFO); lower is more verbose, so `v0` (or `debug`) raises the
   detail above the default to surface the most runtime tracing.

2. Point the CANN / simpler runtime at a device-log directory before
   running:

   ```bash
   export ASCEND_PROCESS_LOG_PATH=/device_log
   python models/deepseek/v4/moe.py -p a2a3 -d 0
   ```

3. Read the logs under `/device_log` to find the **last task that
   dispatched** and which core stalled — that pins the kernel or dependency
   the schedule is waiting on.

`ASCEND_PROCESS_LOG_PATH` is a runtime environment variable, not a harness
kwarg, so it is set in the shell. Hangs under high host concurrency are
often false timeouts — run the suspect test serially before deep-diving.

---

## 5. Localize a precision mismatch with dump-tensor

`enable_dump_tensor` writes
`dfx_outputs/tensor_dump/{tensor_dump.json,tensor_dump.bin}` — the
intermediate tensor values captured at kernel-task boundaries. Use it to turn a
"the whole kernel is wrong" mismatch into "this one op is wrong".

**Dump levels** (`runtime_cfg["enable_dump_tensor"]`): `0` off · `1` partial —
only tensors you mark · `2` full — every task's inputs/outputs (heavy; can
saturate the host collector / trip AICPU timeouts on large workloads).

**The usual flow — tag the tensor, dump at level 1.** Mark the tensor of
interest with `pl.dump_tag(t)` right where it's produced, then run with level
`1`:

```python
h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8)
pl.dump_tag(h_tile_i8)          # capture this one tensor under partial dump
```

```python
run_jit(..., runtime_cfg=dict(platform=..., enable_dump_tensor=1))
```

`pl.dump_tag` works on plain function args and on internal
`pl.create_tensor` GM tensors (incl. inside `@pl.jit.inline`); it returns the
tensor unchanged and is a no-op when dump is off. Equivalent per-scope form:
`pl.at(..., dumps=[t])` / `pl.submit(..., dumps=[t])`. Prefer level `1` + tags
over level `2` — it keeps the dump small and avoids the full-dump timeouts.

Inspect the result with the viewer — with no filters it lists every captured
tensor (task_id / stage / role / dtype / shape); add filters (`--task`,
`--stage before|after`, `--role`, `--arg`, `-i N`) + `--export` to decode the
chosen tensors to `tensor_dump/txt/` for element-wise comparison against torch:

```bash
python -m simpler_setup.tools.dump_viewer <build_output/.../dfx_outputs/tensor_dump>
```

Pass the dump dir **explicitly** — with no argument the viewer looks under
`./outputs/*/tensor_dump`, but `run_jit` writes to
`build_output/<...>/dfx_outputs/tensor_dump`.

To localize a precision bug:

1. Reproduce on fixed data (§2) with the suspect tensors tagged, level `1`.
2. Compute the matching torch intermediates stage by stage.
3. Walk the dumped tensors in dependency order and find the **first** one
   that diverges from its torch reference — the producing kernel is the
   culprit.

This narrows a multi-stage kernel (norm → matmul → dequant → activation) to
the single stage where the error first appears.

---

## 6. Find missing dependencies with gen-deps

`enable_dep_gen=True` (CLI `--enable-dep-gen`) writes
`dfx_outputs/deps.json`, rendered as `deps_graph.html` — the task-graph
dependency edges the orchestration emitted. Open it when results are
**non-deterministic** (values shift run to run, or shift with location) or a
GM write→read looks raced: a dropped edge — a consumer that does not wait on
its producer — shows up as a missing arrow. That points straight at the
orchestration dependency that was lost (a classic cause is `add_output`
instead of `add_inout` on a write-then-read GM round-trip, which drops the
read-dep and lets the downstream task race).

---

## Quick reference

| Symptom | Tool | Kwarg / flag |
|---------|------|--------------|
| Compile / ptoas error | `passes_dump/`, `skip_ptoas` | `compile_cfg=dict(skip_ptoas=True)` |
| Need to reproduce on the same inputs | golden-data replay (§2) | `golden_data=` / `--golden-data` |
| Iterating on generated `.cpp` / `.pto` | runtime-dir reuse (§3) | `runtime_dir=` / `--runtime-dir` |
| Run hangs / deadlocks (§4) | device log | `runtime_cfg["log_level"]="v0"` + `ASCEND_PROCESS_LOG_PATH` |
| Precision mismatch, unknown stage (§5) | tensor dump | `enable_dump_tensor=` / `--dump-tensor` |
| Non-deterministic / raced result (§6) | dependency graph | `enable_dep_gen=` / `--enable-dep-gen` |
| Regression vs. a known-good pypto commit | `bisect-precision` skill | — |
