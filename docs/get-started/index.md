# Get Started

PyPTO-Lib holds runnable [PyPTO](https://www.pypto.ai/pypto/) kernels and
end-to-end LLM models, plus the Golden Harness that compiles them, executes
them with [simpler](https://www.pypto.ai/simpler/), and compares their outputs
with PyTorch references.

Follow these pages in order:

1. [Install the development environment](installation.md). PyPTO owns the
   compatible runtime, PTOAS, and PTO ISA revisions, so begin with a selected
   PyPTO checkout.
2. [Run the first kernel](first-kernel.md) on a simulator or an available NPU,
   and pick the platform for later examples and model kernels.
3. Read the [PyPTO Coding](../pypto-coding/index.md) chapter before
   modifying a kernel.

After the first successful run, continue with
[Run and Validate](../run-and-validate/index.md) to understand the harness and
its saved-data replay workflow.

## What belongs in this repository

- `examples/` contains focused programs for learning and reference.
- `models/` contains model-family kernels and runnable validation entry points.
- `golden/` contains the compile, runtime, and validation harness.
- `docs/` contains the public workflows and technical guidance.
- `build_output/` contains generated artifacts from local runs and is not
  source code.

Files ending in `_draft.py` are works in progress and are excluded from the
normal runnable set.
