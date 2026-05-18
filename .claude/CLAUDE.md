# PyPTO-Lib Developer Guidelines

## Project Overview

PyPTO-Lib hosts tensor-level kernels and end-to-end LLM model
implementations built on the **pypto** programming framework, targeting
Ascend NPUs (910B/C, 950). It also ships a golden-validation test harness
(`golden/`).

## Repository Layout

- `examples/{beginner,intermediate}/` — self-contained kernels for learning the DSL
- `models/{qwen3,deepseek,kimi,milm}/` — end-to-end LLM kernels by family
- `golden/` — test harness: compile, run on device, validate against torch
- `tests/` — lint checks and golden-fn unit tests
- `docs/` — coding-style and workflow reference
- `build_output/` — generated compilation artifacts (gitignored)

Files ending in `_draft.py` are works-in-progress and excluded from CI.

## Key Documentation

- `README.md` — project intro, quick start, dependencies
- `docs/pypto-coding-style.md` — **canonical** coding style: `pl.at` scopes, four loop constructs (`pl.range`/`pl.parallel`/`pl.pipeline`/`pl.spmd`), vector / cube / mte ops
- `docs/compile-runtime-workflow.md` — what `python <kernel>.py -p <platform>` does end-to-end (compile passes/codegen → runtime → golden → validate)

## External Dependencies

| Repo | Role |
|------|------|
| **pypto** | Tile-based programming framework — multi-level IR + codegen |
| **simpler** | PTO runtime — task graph build/execute on AICPU + AICore (submodule of pypto) |
| **ptoas** | LLVM/MLIR PTO Bytecode assembler/optimizer |
| **pto-isa** | PTO Tile Library — virtual tile-ISA implementations |

Pinned versions live in [.github/workflows/ci.yml](../.github/workflows/ci.yml).

## Environment Setup

Use the `/setup_env` skill, or refer to `.claude/skills/setup_env/SKILL.md`.

## Common Commands

```bash
# Run an example on the simulator
python examples/beginner/hello_world.py -p a2a3sim

# Run a model on real NPU device 0
python models/qwen3/14b/qwen3_14b_decode.py -p a2a3 -d 0
```

Every script accepts `-p {a2a3, a2a3sim, a5, a5sim}` and `-d <device_id>`.

## Important Rules

1. **Read `docs/pypto-coding-style.md` first** before writing or modifying any kernel — it is the authoritative coding-style reference.
2. **`docs/compile-runtime-workflow.md`** explains the harness flow; consult it when debugging compile/runtime/validation failures.
3. **Consult `.claude/skills/`** for task-specific workflows (e.g. `setup_env/`, `bisect-precision/`).
4. **No private information** (usernames, absolute paths with usernames, etc.) in code or docs.
5. **All code comments and documentation in English** unless the user explicitly requests otherwise.
