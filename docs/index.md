# PyPTO-Lib

PyPTO-Lib is a collection of tensor-level kernels, model implementations, and
validation workflows built with [PyPTO](https://www.pypto.ai/pypto/) for
Ascend NPUs.

Use this documentation to move from a first simulator run to validated model
kernels and systematic precision or performance tuning.

## Choose a path

### Run a kernel

Start with [installation and environment setup](get-started/installation.md),
then [run your first kernel](get-started/first-kernel.md).

### Write a kernel

Use the [PyPTO Coding](pypto-coding/index.md) chapter for the canonical kernel
style and hand-written CCE extern-kernel conventions.

### Examples

Use the [example catalog](examples/index.md) for focused, self-contained
kernels organized by learning level.

### Models

Use the [model pages](models/index.md) for end-to-end and component-level model
implementations.

### Run and Validate

Read the
[compile and runtime workflow](run-and-validate/compile-runtime-workflow.md)
and the
[Golden Harness overview](run-and-validate/golden-harness.md) to understand
how a script compiles, executes, and checks its result.

### Diagnose and optimize

Begin with the [debugging playbook](debug-and-tune/debugging.md), then choose
the [precision](debug-and-tune/precision-tuning.md) or
[performance](debug-and-tune/performance-tuning.md) workflow for the problem
at hand.
[DeepSeek V4 decode optimization](debug-and-tune/deepseek-v4-decode-optimization.md)
follows one model's decode path end to end and records which levers paid.
[Qwen3-14B optimization](debug-and-tune/qwen3-14b-optimization.md) records how a
single-card dense model's kernels were tuned, in the order the work happened.

## Ecosystem

- [PyPTO documentation](https://www.pypto.ai/pypto/) covers the programming
  model, language semantics, and compiler.
- [simpler documentation](https://www.pypto.ai/simpler/) covers the runtime.
- [PyPTO-Lib on GitHub](https://github.com/hw-native-sys/pypto-lib) contains
  the source code and issue tracker.
