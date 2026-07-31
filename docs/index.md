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

### Understand validation

Read the [compile and runtime workflow](compile-runtime-workflow.md) and the
[Golden Harness overview](run-and-validate/golden-harness.md) to understand
how a script compiles, executes, and checks its result.

### Learn from working implementations

Use the [example catalog](examples/index.md) for focused kernels or the
[model support matrix](models/index.md) for end-to-end and component-level
model implementations.

### Diagnose and optimize

Begin with the [debugging playbook](debugging.md), then choose the
[precision](precision-tuning.md) or
[performance](performance-tuning.md) workflow for the problem at hand.

## Ecosystem

- [PyPTO documentation](https://www.pypto.ai/pypto/) covers the programming
  model, language semantics, and compiler.
- [simpler documentation](https://www.pypto.ai/simpler/) covers the runtime.
- [PyPTO-Lib on GitHub](https://github.com/hw-native-sys/pypto-lib) contains
  the source code and issue tracker.
