# PyPTO-Lib

Tensor-level kernels and model implementations built on the **pypto**
programming framework, targeting Ascend NPUs (910B/C, 950).

```
examples/        Self-contained kernels for learning the DSL
  beginner/        hello_world, matmul，etc.
  intermediate/    softmax, rms_norm, etc.
models/          End-to-end LLM kernels organized by family
  qwen3/14b/       Qwen3-14B prefill + decode
  qwen3/32b/       Qwen3-32B decode
  deepseek/v3_2/   DeepSeek V3.2-EXP
  deepseek/v4/     DeepSeek V4
golden/          Test harness — compile, run on device, validate against torch
tests/           Lint checks and golden-fn unit tests
docs/            Coding-style and workflow reference
```

Files ending in `_draft.py` are works-in-progress and excluded from CI.

## Quick start

Install pypto + simpler + ptoas (see [.claude/skills/setup_env/SKILL.md](.claude/skills/setup_env/SKILL.md)
or use the `/setup_env` skill), then run any example:

```bash
python examples/beginner/hello_world.py -p a2a3sim   # simulator
python models/qwen3/14b/qwen3_14b_decode.py -p a2a3 -d 0   # real NPU, device 0
```

Every example accepts `-p {a2a3, a2a3sim, a5, a5sim}` and `-d <device_id>`,
and exits non-zero on validation mismatch. See
[docs/compile-runtime-workflow.md](docs/compile-runtime-workflow.md) for the
full flow (compile → input gen → runtime → golden → validate).

## Writing a kernel

Read [docs/pypto-coding-style.md](docs/pypto-coding-style.md) — it covers
program structure (`@pl.program` + `@pl.function` + `pl.at`
scopes), the four loop constructs (`pl.range`, `pl.parallel`,
`pl.pipeline`, `pl.spmd`), and the vector / cube / mte op set.

Existing kernels under `examples/intermediate/` are the best reference for
single-stage patterns; `models/qwen3/14b/qwen3_14b_decode.py` shows a
full-model fused kernel.

## Dependencies

| Repo | Role |
|------|------|
| **pypto** | Tile-based programming framework — lowers Tensor → Tile → Block → Execution graphs through multi-level IR and codegen |
| **simpler** | PTO runtime — builds and executes task dependency graphs across AICPU + AICore on Ascend devices (submodule of pypto) |
| **ptoas** | LLVM/MLIR-based assembler/optimizer for PTO Bytecode — parses `.pto`, runs Da Vinci-specific passes, lowers to C++ |
| **pto-isa** | PTO Tile Library — virtual tile-ISA implementations and headers shared across Ascend generations |

Pinned versions live in [.github/workflows/ci.yml](.github/workflows/ci.yml).
