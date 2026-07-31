# Run and Validate

Runnable PyPTO-Lib scripts use the local Golden Harness to turn a kernel
definition into a correctness result.

The common flow is:

```text
PyPTO compile
    ↓
input generation
    ↓
PyTorch golden computation
    ↓
simpler runtime execution
    ↓
output validation
```

Use the following references:

- [Compile and Runtime Workflow](../compile-runtime-workflow.md) explains the
  generated artifacts and each stage in detail.
- [Golden Harness](golden-harness.md) introduces `TensorSpec`, `ScalarSpec`,
  `run`, `run_jit`, validation, and `RunResult`.
- [Save and Replay Golden Data](save-and-replay.md) freezes inputs and expected
  outputs for repeated performance-oriented runs.
- [CCE Extern Kernels](../cce-extern-kernel-guide.md) covers hand-written CCE
  kernels called through `pl.jit.extern`.

For a first end-to-end command, start with
[Run Your First Kernel](../get-started/first-kernel.md).
