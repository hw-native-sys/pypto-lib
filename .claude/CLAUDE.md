# PyPTO-Lib Developer Guidelines

## Project Overview

PyPTO-Lib is a **primitive tensor function library** built on the pypto programming framework. It defines tensor-level operations (analogous to PyTorch ATen) that the compiler tiles and lowers to PTO-ISA instructions. The library does not fix incore/orchestration boundaries — that is decided by the backend and runtime.

## Directory Structure

- `examples/` — Examples organized by difficulty level
  - `beginner/` — Basic operations (hello_world, matmul)
  - `intermediate/` — Multi-step algorithms (softmax, rms_norm, layer_norm, rope)
  - `models/` — Full LLM model kernels (Qwen3-32B, DeepSeek V3.2)
- `docs/` — Reference documentation (coding style, runtime design, parallel loops)
- `tests/` — Lint checks and tests
- `build_output/` — Generated compilation artifacts (gitignored)

## Key Documentation

- `README.md` — Library architecture: tensor vs tile, tiling, fusion, incore scope, primitive set
- `docs/pypto-frontend-coding-style.md` — PyPTO frontend syntax and coding conventions
- `docs/pto2_rt.md` — PTO2 runtime (simpler) design principles
- `docs/para_for.md` — Parallel loop constructs

## External Dependencies

| Dependency | Purpose | Repository |
|------------|---------|------------|
| **pypto** | Compiler framework (IR, codegen, passes) | `hw-native-sys/pypto` |
| **ptoas** | PTO assembler & optimizer toolchain | `zhangstevenunity/PTOAS` |
| **simpler** | Runtime (pto-rt2) | `ChaoWao/simpler` |

## Environment Setup

Use the `/setup_env` skill to set up the development environment, or refer to `.claude/skills/setup_env/SKILL.md` for manual steps.

## Common Commands

### Run an example (requires pypto + ptoas installed)
```bash
python examples/beginner/hello_world.py
```

### Check codegen output
```bash
ls build_output/
```

## Coding Conventions

1. **Follow `docs/pypto-frontend-coding-style.md`** for all pypto frontend code
2. Use `import pypto.language as pl` as the standard module prefix
3. Define tensor functions as **opaque functions** (no incore/orchestration boundary specified)
4. Use `pl.Tensor`, `pl.Tile`, `pl.Scalar` types with explicit shapes and dtypes
5. Use `pl.Out[type]` / `pl.InOut[type]` for output/inout parameters

## Important Rules

1. **Read relevant documentation first** — consult `README.md` and `docs/pypto-frontend-coding-style.md` before writing new tensor functions or models
2. **Consult `.claude/rules/`** for coding conventions (when available)
3. **Consult `.claude/skills/`** for task-specific workflows (e.g., `setup_env/` for environment setup)
4. **Avoid including private information** such as usernames, absolute paths with usernames, or other personally identifiable information in documentation or code
5. **All code comments and documentation should be in English** unless the user explicitly requests otherwise
