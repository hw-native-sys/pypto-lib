# PyPTO-Lib

**PyPTO** kernels and end-to-end LLM models for Ascend NPUs.

**Documentation:** [www.pypto.ai/pypto-lib](https://www.pypto.ai/pypto-lib/)

```
examples/        Self-contained kernels for learning the DSL
  beginner/        hello_world, matmul, etc.
  intermediate/    softmax, rms_norm, rope, etc.
  advanced/        Multi-stage fused + instruction-combo kernels (gemm_eltwise, multi_proj, topk)
models/          Model operators, decoder blocks, and full prefill/decode programs
golden/          Test harness — compile, run on device, validate against torch
tests/           Lint checks and golden-fn unit tests
docs/            Coding-style and workflow reference
```

Files ending in `_draft.py` are works-in-progress and excluded from CI.

## Models

| Model | Implementation |
| --- | --- |
| [Qwen3-14B](docs/models/qwen3_14b/index.md) | Single-card BF16 prefill and decode with grouped-query attention and a paged KV cache. The directory also includes A8W8 projections, TurboQuant variants, and sampling kernels. |
| [DeepSeek V4-Flash MTP](docs/models/deepseek_v4_flash_mtp/index.md) | W8A8 prefill and decode with expert-parallel MoE. Each rank handles its own decode attention batch, and the multi-token prediction (MTP) layer proposes one draft token per request. |
| [DeepSeek V4-Flash DSpark](docs/models/deepseek_v4_flash_dspark/index.md) | A wide-batch configuration of the same checkpoint. A three-layer drafter proposes seven tokens per request, and the target splits the verification rows across the attention group. |
| [DeepSeek V4-Flash low-latency](docs/models/deepseek_v4_flash_lowlat/index.md) | Experimental batch-one decode and chunked prefill. Prefill partitions attention queries across ranks and hands its resident KV cache to decode. |
| [DeepSeek V4.1 Flash](docs/models/deepseek_v4_1_flash/index.md) | Ascend A5 prefill and decode with MXFP8 projections, MXFP4 routed experts, and Engram embedding lookup. The directory also includes DSpark speculative decoding and a separately callable vision encoder. |
| [GLM-5.3-Flash](docs/models/glm5_3_flash/index.md) | W8A8 prefill and decode on Ascend A2/A3, combining Kimi Delta Attention (KDA), NoPE-MLA attention, and MoE. The model also includes MTP and a vision encoder for images and video. |

See the [model index](docs/models/index.md) for deployment configurations and
the `pypto-serving` accuracy jobs run on relevant PRs.

## Quick start

Follow the
[installation and environment guide](docs/get-started/installation.md), then
run a beginner example:

```bash
python examples/beginner/hello_world.py -p a2a3 --sim   # simulator
python models/qwen3_14b/decode_fwd.py -p a2a3 -d 0   # real NPU, device 0
```

The learning examples accept `-p {a2a3,a5} [--sim]` and exit non-zero
on validation mismatch. Model and distributed entry points have
script-specific platform and device arguments; inspect `--help` and the
[platform table](docs/get-started/first-kernel.md#platforms-and-devices). See the
[compile and runtime workflow](docs/run-and-validate/compile-runtime-workflow.md) for the full
flow (compile → input generation → golden → runtime → validation).

## Writing a kernel

Read [docs/pypto-coding/](docs/pypto-coding/index.md) — the coding style split
across kernel structure (the two authoring forms, `pl.at` regions, dynamic
shapes), operations (vector / cube / mte / scalar), the five loop constructs
(`pl.range`, `pl.unroll`, `pl.parallel`, `pl.pipeline`, `pl.spmd`), mixed
cube + vector regions, and naming. For a multi-card kernel, add
[docs/pypto-coding/l3-programming.md](docs/pypto-coding/l3-programming.md)
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
