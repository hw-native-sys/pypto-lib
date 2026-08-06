# Compile and Runtime Workflow

What usually happens when you run `python <kernel>.py -p <platform>`.
Most examples and model harnesses use `golden.run` for `@pl.program` kernels
or `golden.run_jit` for module-level `@pl.jit` kernels. A few specialized
smoke, artifact-regeneration, and external-runtime drivers call PyPTO's
compile/runtime APIs directly; their `__main__` blocks are the authority for
what a command actually validates.

## CLI shape

A typical model `__main__` block parses three flags and dispatches into
the harness:

```python
parser.add_argument("-p", "--platform", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
parser.add_argument("-d", "--device", type=int, default=0)
parser.add_argument("--enable-l2-swimlane", action="store_true")
args = parser.parse_args()

result = run(
    program=build_qwen3_decode_program(...),  # @pl.program class
    specs=build_tensor_specs(...),            # ordered TensorSpec / ScalarSpec list
    golden_fn=golden_qwen3_decode,            # PyTorch reference
    compile_cfg=dict(dump_passes=True),
    runtime_cfg=dict(platform=args.platform, device_id=args.device,
                     enable_l2_swimlane=args.enable_l2_swimlane),
    rtol=3e-3, atol=3e-3,
)
```

A kernel written as a module-level `@pl.jit` function calls **`run_jit`**
instead, passing `fn=<jit_function>` in place of `program=`. Both entry points
share tensor specs, golden computation, runtime dispatch, and validation, but
their `compile_cfg` fields and defaults differ; see
[Compile configuration](#compile-configuration).

| Flag | Purpose |
|------|---------|
| `-p` / `--platform` | Target backend. `a2a3` is Ascend 910B/C; `a5` is Ascend 950 — both run on real NPU. `a2a3sim` / `a5sim` are the matching simulators. |
| `-d` / `--device` | Device ID for multi-card hosts. |
| `--enable-l2-swimlane` | Forwarded to the runtime; collects per-task L2 perf records into the build_output (see [Runtime DFX flags](#runtime-dfx-flags)). |

`a2a3*` maps to `BackendType.Ascend910B`; `a5*` maps to
`BackendType.Ascend950`.

### Multi-card kernels in CI

Most kernels take a single `-d <id>`. A kernel that needs several NPUs
(e.g. an EP/TP program parsing `-d` as a comma-separated list) declares its
card count with a marker comment near the top of the file:

```python
# ci: devices=2
```

The real-NPU CI job greps for `# ci: devices=N`; when `N > 1` it borrows
that many cards from the host device queue with
`task-submit --device "$DEVICE_ID" --device-num N`. `$DEVICE_ID` is `auto`
(borrow any free cards) or a fixed id set by the CI backend, and the lent
set comes back as `$TASK_DEVICE`, passed straight to `-d`. Files without the
marker default to one card. Runs use each program's **default** world size,
commonly EP2 for distributed DeepSeek entries. The current workflow contains
only a commented EP4 command example; it does not provide active EP4 or EP8
per-file coverage. See the `a2a3` job in
[.github/workflows/ci.yml](../../.github/workflows/ci.yml).

Multi-card kernels use HCCL, which silent-crashes inside docker. For this
reason the real-NPU job runs **on the host (no container)**. The shared
`setup-ci-job` action writes an activation script that enters the Python
environment and sources CANN's `set_env.sh`; each `task-submit` child sources
that script. The workflow also preserves its `PTO2_RING_*` settings into the
child. Running a multi-card kernel locally needs the same kind of real-device
shell, e.g.
`python models/deepseek_v4_flash_mtp/moe.py -p a2a3 --ep 2 -d 0,1`.

## Phases inside the Golden Harness

The harness prints `[RUN] <stage> ...` / `[RUN] <stage> done (Xs)` around
each phase, so the console log is the authoritative trace of what ran:

### 1. Compile (pypto)

Driven by the **pypto** repo. `run` calls
`pypto.ir.compile(program, backend_type=..., **compile_cfg)` directly.
`run_jit` builds a `pypto.runtime.RunConfig` from `compile_cfg` and calls
`fn.compile(..., config=...)`, which specializes the JIT function before
entering the same IR compiler. Both paths run a **pass pipeline** followed by
a **codegen pipeline** and normally write
`build_output/<ProgramName>_<timestamp>/`.

#### 1a. Pass pipeline

`PassManager.get_strategy(strategy).run_passes(program, ...)` runs an
ordered sequence of passes that progressively rewrites the IR. The exact
pass list changes often — consult the pypto repo for the current pipeline,
and look at `passes_dump/` when `dump_passes` is enabled. Direct `run`
compilation inherits `ir.compile`'s enabled default; `run_jit` inherits
`RunConfig`'s disabled default.

The end state, regardless of which passes ran, is the same:

- exactly **one orchestration function** (`FunctionType.Orchestration`),
- plus **one InCore function per outlined `pl.at` / `pl.spmd` region**.

A `pl.at` region that mixes cube and vector ops is split into **two**
InCore functions during outlining: one cube-only kernel (matmul,
matmul_acc, …) and one vector-only kernel (cast, add, row_sum, …). The
orchestration function calls them in dependency order.

The InCore / Orchestration boundary the frontend left implicit becomes
explicit at this stage.

#### 1b. Codegen pipeline

`pypto.backend.pto_backend.generate(...)` walks the transformed program
and emits files in three streams:

- **InCore kernels → `.pto` → C++ wrapper.** Each kernel function (or
  group thereof) goes through `PTOCodegen` to produce an MLIR text file
  (`.pto`) under `ptoas/`. Then `ptoas` (the external assembler/optimizer
  toolchain) compiles each `.pto` to a C++ kernel wrapper under
  `kernels/aic/` (cube) or
  `kernels/aiv/` (vector). The ptoas invocations run in a thread pool
  since each is an independent subprocess. `skip_ptoas=True` keeps the
  raw `.pto` files and skips the C++ wrapper step (useful for inspecting
  pure MLIR output or for isolating whether a regression came from
  pypto's IR→MLIR or from ptoas).
- **Orchestration → C++.** `generate_orchestration` emits one
  `orchestration/<orch_name>.cpp` that drives the kernels through the
  PTO2 runtime API (task graph build, scheduling, dependencies).
- **Config → `kernel_config.py`.** Records each kernel's name, runtime
  ID, and core type (cube / vector) for the runtime to load.

When `PTOAS_ROOT` is set, PyPTO searches only
`$PTOAS_ROOT/ptoas` and then `$PTOAS_ROOT/bin/ptoas`; it deliberately does
not fall back to a potentially mismatched binary on `PATH`. When
`PTOAS_ROOT` is unset, PyPTO searches `PATH`.

#### Output directory layout

```
build_output/<ProgramName>_<ts>/
├── passes_dump/    # IR after each pass, when dump_passes is enabled
├── ptoas/          # raw .pto MLIR + ptoas intermediates
├── kernels/
│   ├── aic/        # cube kernel C++ wrappers from ptoas
│   └── aiv/        # vector kernel C++ wrappers from ptoas
├── orchestration/  # generated AICPU orchestration C++ (compiled into .so)
├── kernel_config.py
├── report/         # memory allocation + scheduling reports
├── data/           # populated by later phases (in/, out/)
└── dfx_outputs/    # runtime DFX artefacts (any --enable-* flag)
```

#### Compile configuration

For **`run`**, `compile_cfg` is forwarded to `ir.compile`. Common fields are:

| `compile_cfg` field | Purpose |
|---|---|
| `output_dir` | Override `build_output/<name>_<timestamp>/`. |
| `strategy` | Select the optimization strategy. |
| `dump_passes` | Write pass IR under `passes_dump/`; defaults to `True` on this direct compiler path. |
| `skip_ptoas` | Stop after `.pto` generation without producing kernel C++ wrappers. |
| `profiling` | Write compile-stage timing reports under `report/`. |
| `verification_level`, `diagnostic_phase`, `disabled_diagnostics` | Configure compiler verification and diagnostics. |
| `distributed_config`, `analyze_auto_scopes_for_deps`, `memory_planner` | Configure distributed lowering, AUTO-scope dependency analysis, and memory planning. |

The harness derives `backend_type` and `platform` from
`runtime_cfg["platform"]` unless the direct compile configuration already
supplies them.

For **`run_jit`**, `compile_cfg` must instead contain fields accepted by
`pypto.runtime.RunConfig`. The JIT layer maps its compile-side fields into
`ir.compile`:

| `compile_cfg` field | Compiler mapping |
|---|---|
| `dump_passes` | Same pass dumps, but the `RunConfig` default is `False`. |
| `save_kernels_dir` | Maps to `ir.compile(output_dir=...)`. |
| `compile_profiling` | Maps to `ir.compile(profiling=...)`. |
| `strategy`, `diagnostic_phase`, `disabled_diagnostics` | Forwarded to the corresponding compiler fields. |
| `distributed_config`, `analyze_auto_scopes_for_deps`, `memory_planner` | Forwarded when set. |

`output_dir`, `profiling`, `skip_ptoas`, and `verification_level` are not
`RunConfig` field names and therefore cannot be copied unchanged from a
`run` call into `run_jit`. Unknown fields raise while constructing
`RunConfig`; unknown direct-compiler fields raise in `ir.compile`.

To stop after compile without touching the device, see `compile_only` under
[Skipping phases](#skipping-phases).

### 2. Generate inputs

Each entry of `specs` is a `TensorSpec` (named tensor, shape, dtype,
direction) or a `ScalarSpec` (named scalar, dtype, value); see
`golden/spec.py`. The list is ordered to match the parameter order of the
top opaque function. For each entry, allocate a torch tensor:

- Pure inputs and inout initial values are filled via `spec.create_tensor()`
  (`init_value=None` creates zeros; random data requires an explicit factory
  such as `torch.randn`).
- Pure outputs are zero-initialised.
- Scalars become 0-D tensors carrying the spec value.

See [Golden Harness](golden-harness.md) for the complete
`TensorSpec` initialization contract.

When `save_data=True`, the input snapshot is written to `data/in/<name>.pt`
so the same inputs can be replayed later; this is off by default, so no
snapshot is written unless you opt in. If `golden_data=<dir>` is passed
instead, the harness loads `<dir>/in/*.pt` rather than generating fresh
data — useful for deterministic regression checks.

### 3. Compute golden

The golden runs **before** device execution: it depends only on the input
snapshot, not on the runtime, so the reference is ready for validation. With
`save_data=True`, a later runtime crash still leaves the persisted
`data/out/` snapshot.

If `golden_fn` is provided, `run` builds a `scratch` dict with cloned
inputs and zero-init outputs, calls `golden_fn(scratch)` (which fills the
output entries in place), and — when `save_data=True` — writes the result
to `data/out/<name>.pt`.

If `golden_data=<dir>` is set, the harness loads `<dir>/out/*.pt` instead
of recomputing — `golden_data` always wins over `golden_fn`.

If neither is provided, validation is skipped and the run reports
`PASS (validation skipped)`.

Golden PyTorch operations use 16 intra-op CPU threads by default. Set
`PYPTO_GOLDEN_NUM_THREADS` to a positive integer to override the repository
default for a run:

```bash
PYPTO_GOLDEN_NUM_THREADS=8 \
  python models/deepseek_v4_flash_mtp/decode_csa.py -p a2a3 -d 0
```

### 4. Runtime (simpler)

Driven by the **simpler** repo (PTO2 runtime). For a single-chip build, the
harness orders the arguments according to `specs` and calls
`pypto.runtime.execute_compiled`. For an L3
`DistributedCompiledProgram`, it instead dispatches the compiled object with a
`pypto.runtime.RunConfig`; resident-weight L3 programs use the prepared-worker
path. Tensors are mutated in place, so outputs land in the same Python tensors
after dispatch.

`runtime_cfg` is therefore not forwarded verbatim in every case:

- `log_level` is consumed by the harness to configure PyPTO's runtime logger;
- the five DFX fields below are bundled into the runtime's DFX options on the
  single-chip path;
- remaining single-chip fields are passed to `execute_compiled`, which rejects
  unknown names;
- L3 dispatch retains fields supported by `RunConfig`.

#### Runtime DFX flags

PyPTO surfaces simpler's five runtime DFX (Design For X) sub-features as
independent toggles on `runtime_cfg`. They share the same output
directory and can be enabled in any combination. CLI spellings are
entry-specific; the table lists the common spelling when a script exposes it.

| Kwarg | CLI flag | Artefact under `dfx_outputs/` |
|-------|----------|-------------------------------|
| `enable_l2_swimlane=True` (or a supported level) | `--enable-l2-swimlane [N]` | `l2_swimlane_records.json`; onboard runs also attempt `merged_swimlane_*.json` |
| `enable_dump_args=<N>` (int, `0`=off) | `--dump-args [N]` (bare = `1`) | `args_dump/{args_dump.json,args.bin}` |
| `enable_pmu=<N>` (int, `0`=off) | `--enable-pmu [N]` (bare = `2`) | `pmu.csv` |
| `enable_dep_gen=True` | `--enable-dep-gen` | `deps.json` |
| `enable_scope_stats=True` | `--enable-scope-stats` | `scope_stats/scope_stats.jsonl` |

Args-dump level `1` captures only arguments selected with `pl.dump_tag` or a
`dumps=` list; level `2` captures every task's tensor payloads and scalar
values. Level `3` captures the same argument metadata without writing tensor
payloads or `args.bin`.

For an onboard L2 swimlane run, PyPTO first attempts a dependency-graph
capture and then a clean timing capture so the converter can add dependency
arrows without perturbing the timing pass. Open the generated
`merged_swimlane_*.json` at
[ui.perfetto.dev](https://ui.perfetto.dev/) to visualize per-task
execution on each AICPU / AIC / AIV lane and inspect kernel duration,
gaps, and dependency stalls. Simulator runs retain
`l2_swimlane_records.json` but do not generate the merged trace because the
required task metadata is unavailable.

The raw dependency and scope-stat files can also be rendered offline:

```bash
python -m simpler_setup.tools.deps_viewer \
  build_output/<name>/dfx_outputs/deps.json --format html --engine sfdp
python -m simpler_setup.tools.scope_stats_plot \
  build_output/<name>/dfx_outputs/scope_stats/scope_stats.jsonl
```

For kernel-internal swimlanes and MindStudio Insight traces, see
[In-Core Simulator Profiling](../debug-and-tune/incore-simulator-profiling.md).
The repository workflow can reuse an existing build or drive a case
end-to-end. It writes the export root below
`build_output/<ProgramName>_<ts>/kernel_insight_all_funcs_<ts>/`.

See pypto's `docs/en/dev/03-runtime-dfx.md` and the simpler reference at
`runtime/docs/dfx/{l2-swimlane-profiling,args-dump,pmu-profiling,dep_gen,scope-stats}.md`
for full per-flag details. There is no `runtime_profiling` /
`--runtime-profiling` compatibility alias in the current Python API; use
`enable_l2_swimlane` or the entry's `--enable-l2-swimlane` flag.

### 4b. Benchmark (opt-in, before validation)

With `PYPTO_BENCH=1` in the environment, the harness re-dispatches the
compiled program in a timed loop right after the correctness dispatch and
before validation, and prints
`[RUN]   effective_us (N rounds) min=… median=… mean=… max=…`. It is
env-gated only — no model file needs a flag. `PYPTO_BENCH_ROUNDS` /
`PYPTO_BENCH_WARMUP` (default 100 / 5) size the loop and `PYPTO_BENCH_RAW`
dumps the per-dispatch samples. See
[performance-tuning.md](../debug-and-tune/performance-tuning.md#measuring-the-benchmark-loop-pypto_bench)
for the output format, the multi-card breakdown, and what the number means.

### 5. Validate

`golden.validation.validate_golden` compares each device output against
the golden using `torch.allclose(rtol, atol)` by default. Override
per-output with the `compare_fn={"out_name": custom_callable}` argument.
`golden.validation` ships three ready-made comparators:

| Comparator | Use case |
|------------|----------|
| `topk_pair_compare(vals_name)` | Top-k index outputs whose ordering is implementation-dependent — checks the paired value tensor matches after sort, tolerating legal tie-break swaps. |
| `ratio_allclose(atol, rtol, max_error_ratio=0.005)` | Quantized kernels where a small outlier fraction may exceed per-point `atol + rtol·|expected|`. NaN/Inf always fail. |
| `ratio_reldiff(diff_thd, pct_thd, max_diff_hd=inf)` | cann-recipes-infer-style relative-diff check: per-point `rdiff > diff_thd` bad-point ratio capped by `pct_thd`, with optional single-point `max_diff_hd` cap. |

The harness returns `RunResult(passed=True)` on success. Validation mismatches
and a small set of harness-level setup errors return
`RunResult(passed=False, error=...)`. Compiler errors, golden-function errors,
and ordinary runtime exceptions generally propagate instead of being converted
to `RunResult`. A CLI should check a returned result and exit nonzero when it is
false; an uncaught compile/runtime exception is already a nonzero failure.

## Skipping phases

`run` / `run_jit` knobs that short-circuit the pipeline:

| Knob | Effect |
|------|--------|
| `compile_only=True` | Stops after the compile phase. Useful in CI smoke tests that just check the program lowers cleanly. |
| `runtime_dir="<path>"` | Skips compile and reuses an existing `build_output/<...>` directory. Useful when iterating on `golden_fn` or validation logic without recompiling. |
| `golden_data="<path>"` | Loads inputs from `<path>/in/` and goldens from `<path>/out/` instead of generating them. `golden_data` overrides `golden_fn`. Useful for deterministic regressions: a previous run leaves these files in its `data/` dir, so passing that dir reproduces the exact failing inputs. |
| `save_data=True` (default `False`) | Writes the `data/in/` + `data/out/` snapshot so the exact inputs/goldens can be replayed later via `golden_data`. Off by default: runs skip the snapshot and validate against the in-memory golden only. Opt in when you need replay; full-model kernels like `models/qwen3_14b/{prefill_fwd,decode_fwd}.py` expose it as `--save-data`. |

For diagnosing compile errors, runtime hangs, and precision mismatches, see
[debugging.md](../debug-and-tune/debugging.md).
