# Compile and Runtime Workflow

What happens when you run `python <kernel>.py -p <platform>`. All
pypto-lib examples and model kernels follow the same flow, driven by
`golden.run` — or `golden.run_jit` for kernels written as a module-level
`@pl.jit` function instead of a `@pl.program` class.

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
instead, passing `fn=<jit_function>` in place of `program=`. Everything
else — `specs`, `golden_fn`, the tolerances, and the phases below — is
identical.

| Flag | Purpose |
|------|---------|
| `-p` / `--platform` | Target backend. `a2a3` is Ascend 910B/C; `a5` is Ascend 950 — both run on real NPU. `a2a3sim` / `a5sim` are the matching simulators. |
| `-d` / `--device` | Device ID for multi-card hosts. |
| `--enable-l2-swimlane` | Forwarded to the runtime; collects per-task L2 perf records into the build_output (see [Runtime DFX flags](#runtime-dfx-flags)). |
| `--export-kernel-insight` | Qwen3-14B decode helper: after a successful run, invokes `tools/export_all_kernel_insight.py` for the generated kernels and writes Insight exports under the same `build_output/<ProgramName>_<ts>/`. |

`a2a3*` maps to `BackendType.Ascend910B`; `a5*` maps to
`BackendType.Ascend950`.

### Multi-card kernels in CI

Most kernels take a single `-d <id>`. A kernel that needs several NPUs
(e.g. a 2-rank EP program parsing `-d` as a comma-separated list) must
declare its card count with a marker comment near the top of the file:

```python
# ci: devices=2
```

The real-NPU CI job greps for `# ci: devices=N`; when `N > 1` it runs the
file with `$DEVICE_RANGE` (a comma-separated id list such as `0,1`)
instead of the single `$DEVICE_ID`. Files without the marker default to
one device. See the `a2a3` job in
[.github/workflows/ci.yml](../.github/workflows/ci.yml).

Multi-card kernels use HCCL, which silent-crashes inside docker and breaks
when `PTO2_RING_*` are set. For this reason the real-NPU job runs **on the
host (no container)** — set up via conda + `set_env.sh`, mirroring pypto's
`dist-system-tests` — and unsets `PTO2_RING_*` for multi-card files while
keeping the large ring sizes for single-card ones. Running a multi-card
kernel locally needs the same: a real `set_env.sh`-sourced shell with
`PTO2_RING_*` unset, e.g.
`python models/deepseek/v4/moe.py -p a2a3 -d 0,1`.

## Phases inside `golden.run`

The harness prints `[RUN] <stage> ...` / `[RUN] <stage> done (Xs)` around
each phase, so the console log is the authoritative trace of what ran:

### 1. Compile (pypto)

Driven by the **pypto** repo. `pypto.ir.compile(program, backend_type=..., **compile_cfg)`
runs in two sub-stages: a **pass pipeline** that transforms the IR, then a
**codegen pipeline** that emits files. Output goes under
`build_output/<ProgramName>_<timestamp>/`.

#### 1a. Pass pipeline

`PassManager.get_strategy(strategy).run_passes(program, ...)` runs an
ordered sequence of passes that progressively rewrites the IR. The exact
pass list changes often — consult the pypto repo for the current pipeline,
and look at `passes_dump/` (written when `dump_passes=True`, the default)
to see the IR after each pass for any given run.

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
  toolchain — located via `$PTOAS_ROOT/ptoas` or `PATH`) compiles each
  `.pto` to a C++ kernel wrapper under `kernels/aic/` (cube) or
  `kernels/aiv/` (vector). The ptoas invocations run in a thread pool
  since each is an independent subprocess. `skip_ptoas=True` keeps the
  raw `.pto` files and skips the C++ wrapper step (useful for inspecting
  pure MLIR output or for bisecting whether a regression came from
  pypto's IR→MLIR or from ptoas).
- **Orchestration → C++.** `generate_orchestration` emits one
  `orchestration/<orch_name>.cpp` that drives the kernels through the
  PTO2 runtime API (task graph build, scheduling, dependencies).
- **Config → `kernel_config.py`.** Records each kernel's name, runtime
  ID, and core type (cube / vector) for the runtime to load.

#### Output directory layout

```
build_output/<ProgramName>_<ts>/
├── passes_dump/    # IR after each pass (dump_passes=True)
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

#### Compile knobs

Forwarded from `compile_cfg` to `ir.compile`:

| Kwarg | Purpose |
|-------|---------|
| `backend_type` | Auto-set from `runtime.platform` (`a2a3*` → `Ascend910B`, `a5*` → `Ascend950`). |
| `output_dir` | Override the default `build_output/<name>_<timestamp>/`. |
| `strategy` | `OptimizationStrategy.Default` (full pipeline) or `DebugTileOptimization` (skips tensor-only passes — for tile-pass debugging). |
| `dump_passes` | Default `True`; writes IR after every pass to `passes_dump/`. |
| `skip_ptoas` | Stop after `.pto` generation (no kernel C++ wrappers). |
| `profiling` | Record per-stage compile timings under `report/`. |
| `verification_level`, `diagnostic_phase`, `disabled_diagnostics` | Tune the pass-time verifier and diagnostic gates. |

To stop after compile without touching the device, see `compile_only` under
[Skipping phases](#skipping-phases).

### 2. Generate inputs

Each entry of `specs` is a `TensorSpec` (named tensor, shape, dtype,
direction) or a `ScalarSpec` (named scalar, dtype, value); see
`golden/spec.py`. The list is ordered to match the parameter order of the
top opaque function. For each entry, allocate a torch tensor:

- Pure inputs and inout initial values are filled via `spec.create_tensor()`
  (random by default, or constant when `init_value` is set).
- Pure outputs are zero-initialised.
- Scalars become 0-D tensors carrying the spec value.

The input snapshot is written to `data/in/<name>.pt` so the same inputs can
be replayed later (skipped when `save_data=False`). If `golden_data=<dir>` is
passed instead, the harness loads `<dir>/in/*.pt` rather than generating fresh
data — useful for deterministic regression checks.

### 3. Compute golden

The golden runs **before** device execution: it depends only on the input
snapshot, not on the runtime, so the reference is ready for validation and
a later runtime crash still leaves a usable `data/out/`.

If `golden_fn` is provided, `run` builds a `scratch` dict with cloned
inputs and zero-init outputs, calls `golden_fn(scratch)` (which fills the
output entries in place), and writes the result to `data/out/<name>.pt`
(skipped when `save_data=False`).

If `golden_data=<dir>` is set, the harness loads `<dir>/out/*.pt` instead
of recomputing — `golden_data` always wins over `golden_fn`.

If neither is provided, validation is skipped and the run reports
`PASS (validation skipped)`.

### 4. Runtime (simpler)

Driven by the **simpler** repo (PTO2 runtime).
`pypto.runtime.execute_compiled(work_dir, ordered_args, **runtime_cfg)`
loads the compiled artifacts onto the target platform and runs them.
Tensors passed by reference are mutated in place: outputs land back into
the same Python tensors after the call returns.

`runtime_cfg` is forwarded verbatim — `platform`, `device_id`, the
runtime DFX flags below, and any other runtime knobs. Refer to the
simpler repo for the full set of runtime options and platform-specific
behavior.

#### Runtime DFX flags

PyPTO surfaces simpler's four runtime DFX (Design For X) sub-features as
independent toggles on `runtime_cfg`. They share the same output
directory but can be enabled in any combination:

| Kwarg | CLI flag | Artefact under `dfx_outputs/` |
|-------|----------|-------------------------------|
| `enable_l2_swimlane=True` | `--enable-l2-swimlane` | `l2_perf_records.json` → `merged_swimlane_*.json` |
| `enable_dump_tensor=True` | `--dump-tensor` | `tensor_dump/{tensor_dump.json,bin}` |
| `enable_pmu=<N>` (int, `0`=off) | `--enable-pmu [N]` (bare = `2`) | `pmu.csv` |
| `enable_dep_gen=True` | `--enable-dep-gen` | `deps.json` → `deps_graph.html` |

Enabling any flag auto-forces `save_kernels=True` so
`build_output/<ProgramName>_<ts>/dfx_outputs/` survives the run.

For L2 swimlane: open the generated `merged_swimlane_*.json` at
[ui.perfetto.dev](https://ui.perfetto.dev/) to visualize per-task
execution on each AICPU / AIC / AIV lane and inspect kernel duration,
gaps, and dependency stalls.

For kernel-internal swimlane / MindStudio Insight traces, use the repo tool
directly on an existing build:

```bash
python tools/export_all_kernel_insight.py --build-dir build_output/<ProgramName>_<ts>
```

or, for `models/qwen3/14b/qwen3_14b_decode.py`, append
`--export-kernel-insight` to the normal run. The export root is written under
`build_output/<ProgramName>_<ts>/kernel_insight_all_funcs_<ts>/`, and the build
directory also gets `latest_all_funcs_kernel_insight_export_root.txt` pointing
at the latest export.

See pypto's `docs/en/dev/03-runtime-dfx.md` and the simpler reference at
`runtime/docs/dfx/{l2-swimlane,tensor-dump,pmu-profiling,dep_gen}.md` for
full per-flag details.

> The old single boolean `runtime_profiling` / `--runtime-profiling` is
> a deprecated alias for `enable_l2_swimlane` / `--enable-l2-swimlane`.
> It still works but emits a `DeprecationWarning` and will be removed.

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

The harness exits with `RunResult(passed=True)` on success. On any
failure (compile error, runtime crash, validation mismatch) it returns
`passed=False` with the error message; the model's `__main__` then
`raise SystemExit(1)`.

## Skipping phases

`run` / `run_jit` knobs that short-circuit the pipeline:

| Knob | Effect |
|------|--------|
| `compile_only=True` | Stops after the compile phase. Useful in CI smoke tests that just check the program lowers cleanly. |
| `runtime_dir="<path>"` (kwarg to `run`) | Skips compile and reuses an existing `build_output/<...>` directory. Useful when iterating on `golden_fn` or validation logic without recompiling. |
| `golden_data="<path>"` (kwarg to `run`) | Loads inputs from `<path>/in/` and goldens from `<path>/out/` instead of generating them. `golden_data` overrides `golden_fn`. Useful for deterministic regressions: a previous run leaves these files in its `data/` dir, so passing that dir reproduces the exact failing inputs. |
| `save_data=False` (kwarg to `run`; default `True`) | Skips writing the `data/in/` + `data/out/` snapshot. Validation still runs against the in-memory golden. Use for large fixtures (full-model weights / KV cache) where the snapshot is huge and replay is not needed — full-model kernels like `models/qwen3/14b/{prefill_fwd,decode_layer}.py` expose it as `--save-data` (off by default). |

For diagnosing compile errors, runtime hangs, and precision mismatches, see
[debugging.md](debugging.md).
