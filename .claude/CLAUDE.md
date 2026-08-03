# PyPTO-Lib Developer Guidelines

## Project Overview

PyPTO-Lib hosts tensor-level kernels and end-to-end LLM model
implementations built on the **pypto** programming framework, targeting
Ascend NPUs (910B/C, 950). It also ships a golden-validation test harness
(`golden/`).

## Repository Layout

- `examples/{beginner,intermediate,advanced}/` — self-contained kernels for learning the DSL
- `models/{qwen3,deepseek}/` — end-to-end LLM kernels by family
- `golden/` — test harness: compile, run on device, validate against torch
- `tests/` — lint checks and golden-fn unit tests
- `docs/` — canonical public setup, validation, examples, models, debugging, and tuning guidance
- `build_output/` — generated compilation artifacts (gitignored)

Files ending in `_draft.py` are works-in-progress and excluded from CI.

## Key Documentation

- `README.md` — project intro, quick start, dependencies
- `docs/get-started/installation.md` — environment setup and the PyPTO-owned toolchain pin chain
- `docs/get-started/platforms.md` — simulator/device targets and CI coverage semantics
- `docs/run-and-validate/golden-harness.md` — Golden Harness specs, execution, validation, and results
- `docs/run-and-validate/save-and-replay.md` — frozen input/golden capture and replay
- `docs/pypto-coding/pypto-coding-style.md` — **canonical** coding style: the two kernel forms (`@pl.jit` / `@pl.jit.inline` and `@pl.program` / `@pl.function`), `pl.at` scopes, four loop constructs (`pl.range`/`pl.parallel`/`pl.pipeline`/`pl.spmd`), vector / cube / mte ops, dynamic B/S shapes
- `docs/run-and-validate/compile-runtime-workflow.md` — what `python <kernel>.py -p <platform>` does end-to-end (compile passes/codegen → input gen → golden → runtime → validate)
- `docs/debug-and-tune/debugging.md` — debugging playbook: pypto/ptoas errors, `golden_data` replay, `runtime_dir` reuse, runtime-hang device logs, args-dump / dep-gen
- `docs/debug-and-tune/performance-tuning.md` — L2 (inter-kernel) and L1/L0 (intra-kernel) tuning: swimlanes, PMU, buffer-occupancy / perf-hint reports
- `docs/debug-and-tune/cce-incore-profiling.md` — on-device multi-core phase timestamps for fused CCE extern kernels: per-core capture, barrier diagnostics, and exact L2-reconciled partitions
- `docs/debug-and-tune/precision-tuning.md` — keeping a kernel numerically faithful: `pl.cast` rounding modes vs torch, dtype alignment, fp32 intermediates / no double-cast, quant schemes, the `error_distribution` threshold sweep, and real-weight testing
- `docs/pypto-coding/cce-extern-kernel.md` — writing hand-written mixed (cube+vector) CCE kernels behind `pl.jit.extern`: the persistent-kernel runtime model, the tensors-first/scalars-last arg-packing trap, UB/`TPipe`, `SyncAll<false>` cross-core barriers, GM scalar coherency, and the on-device narrowing methodology

## External Dependencies

| Repo | Role |
|------|------|
| **pypto** | Tile-based programming framework — multi-level IR + codegen |
| **simpler** | PTO runtime — task graph build/execute on AICPU + AICore (submodule of pypto) |
| **ptoas** | LLVM/MLIR PTO Bytecode assembler/optimizer |
| **pto-isa** | PTO Tile Library — virtual tile-ISA implementations |

The selected PyPTO checkout is the version source of truth: its `runtime/`
gitlink pins simpler, `toolchain/versions.env` pins PTOAS, and
`runtime/pto_isa.pin` pins PTO ISA. See
[`docs/get-started/installation.md`](../docs/get-started/installation.md).

## Environment Setup

Read [`docs/get-started/installation.md`](../docs/get-started/installation.md)
for the public procedure. Use the `/setup-env` skill to inspect and execute
that procedure on the current machine.

## Common Commands

```bash
# Run an example on the simulator
python examples/beginner/hello_world.py -p a2a3sim

# Run a model on real NPU device 0
python models/qwen3/14b/decode_fwd.py -p a2a3 -d 0
```

Platform and device arguments are entry-specific. Inspect the selected
script's `--help` and `docs/get-started/platforms.md`.

## Important Rules

1. **Read `docs/pypto-coding/pypto-coding-style.md` first** before writing or modifying any kernel — it is the authoritative coding-style reference.
2. **`docs/run-and-validate/compile-runtime-workflow.md`** explains the harness flow end-to-end; **`docs/debug-and-tune/debugging.md`** is the debugging playbook (compile/runtime/validation failures, hangs, precision) and **`docs/debug-and-tune/performance-tuning.md`** the tuning guide.
3. **Treat `docs/` as the technical source of truth.** Consult
   `.claude/skills/` for task-specific execution, safety, and reporting, and
   keep skills linked to the canonical public guide rather than copying it.
4. **No private information** (usernames, absolute paths with usernames, etc.) in code or docs.
5. **All code comments and documentation in English** unless the user explicitly requests otherwise.
