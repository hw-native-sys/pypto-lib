# PyPTO-Lib

Tensor-level kernels and model implementations built on the **pypto**
programming framework, targeting Ascend NPUs (910B/C, 950).

**Documentation:** [www.pypto.ai/pypto-lib](https://www.pypto.ai/pypto-lib/)

```
examples/        Self-contained kernels for learning the DSL
  beginner/        hello_world, matmul, etc.
  intermediate/    softmax, rms_norm, rope, etc.
  advanced/        Multi-stage fused + instruction-combo kernels (gemm_eltwise, multi_proj, topk)
models/          End-to-end LLM kernels, one flat directory per model build
  qwen3_14b/                   Qwen3-14B prefill + decode, BF16, serving contract
  deepseek_v4_flash_mtp/       DeepSeek V4-Flash, INT8 W8A8, MTP=1, serving contract
  deepseek_v4_pro/             DeepSeek V4-Pro with an optional Flash preset, A5 variant
  (other directories are kernel harnesses — see the model pages)
golden/          Test harness — compile, run on device, validate against torch
tests/           Lint checks and golden-fn unit tests
docs/            Coding-style and workflow reference
```

Files ending in `_draft.py` are works-in-progress and excluded from CI. The
[model pages](docs/models/index.md) list every model directory, whether it is
wired to `pypto-serving`, and — for the full model trees — its deployment
configuration and how its files compose.

## Quick start

Follow the
[installation and environment guide](docs/get-started/installation.md), then
run a beginner example:

```bash
python examples/beginner/hello_world.py -p a2a3sim   # simulator
python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0   # real NPU, device 0
```

The learning examples accept `-p {a2a3,a2a3sim,a5,a5sim}` and exit non-zero
on validation mismatch. Model and distributed entry points have
script-specific platform and device arguments; inspect `--help` and the
[platform guide](docs/get-started/platforms.md). See the
[compile and runtime workflow](docs/run-and-validate/compile-runtime-workflow.md) for the full
flow (compile → input generation → golden → runtime → validation).

## Writing a kernel

Read [docs/pypto-coding/pypto-coding-style.md](docs/pypto-coding/pypto-coding-style.md) — it covers
the two kernel forms (`@pl.jit` / `@pl.jit.inline` and `@pl.program` /
`@pl.function`), `pl.at` scopes, the five loop constructs (`pl.range`,
`pl.unroll`, `pl.parallel`, `pl.pipeline`, `pl.spmd`), runtime scopes
(`pl.scope`), scalar access (`pl.read` / `pl.write`), and the vector / cube /
mte op set. For a multi-card kernel, add
[docs/pypto-coding/distributed-programming.md](docs/pypto-coding/distributed-programming.md)
— window buffers, cross-rank data movement, and notify / wait.

Existing kernels under `examples/intermediate/` are the best reference for
single-stage patterns; `models/qwen3_14b/decode_fwd.py` shows a
full-model fused kernel.

## Debugging

See [docs/debug-and-tune/debugging.md](docs/debug-and-tune/debugging.md) for the debugging workflow —
reading pypto/ptoas errors, replaying failing data with `golden_data`,
reusing a compile with `runtime_dir`, device logs for runtime hangs, and
the args-dump / dep-gen DFX flags.

## Performance tuning

See [docs/debug-and-tune/performance-tuning.md](docs/debug-and-tune/performance-tuning.md) for the L2
(inter-kernel) and L1/L0 (intra-kernel) tuning workflow — chip swimlane in
Perfetto, PMU counters, and the per-kernel insight swimlane.

## Precision tuning

See [docs/debug-and-tune/precision-tuning.md](docs/debug-and-tune/precision-tuning.md) for keeping a kernel
numerically faithful to its torch reference — `pl.cast` rounding modes vs
torch, kernel/golden parity, dtype alignment, quantization schemes, the
`error_distribution` threshold sweep, and real-weight testing.

## Dependencies

| Repo | Role |
|------|------|
| [**pypto**](https://github.com/hw-native-sys/pypto) | Tile-based programming framework — lowers Tensor → Tile → Block → Execution graphs through multi-level IR and codegen |
| [**simpler**](https://github.com/hw-native-sys/simpler) | PTO runtime — builds and executes task dependency graphs across AICPU + AICore on Ascend devices (submodule of pypto) |
| [**ptoas**](https://github.com/hw-native-sys/PTOAS) | LLVM/MLIR-based assembler/optimizer for PTO Bytecode — parses `.pto`, runs Da Vinci-specific passes, lowers to C++ |
| [**pto-isa**](https://github.com/hw-native-sys/pto-isa) | PTO Tile Library — virtual tile-ISA implementations and headers shared across Ascend generations |

The selected PyPTO revision owns the compatible simpler submodule, PTOAS
release, and PTO ISA commit. See
[Installation and Environment](docs/get-started/installation.md) for the
pinning chain.
