# Get Started

PyPTO-Lib contains executable kernels and model implementations built with
[PyPTO](https://www.pypto.ai/pypto/). It also contains the Golden Harness used
to compile those programs, execute them with
[simpler](https://www.pypto.ai/simpler/), and compare their outputs with
PyTorch references.

Follow these pages in order:

1. [Install the development environment](installation.md). PyPTO owns the
   compatible runtime, PTOAS, and PTO ISA revisions, so begin with a selected
   PyPTO checkout.
2. [Run the first kernel](first-kernel.md) on a simulator or an available NPU.
3. [Choose a platform and device](platforms.md) for later examples and model
   kernels.
4. Read the
   [kernel coding style](../pypto-coding/pypto-coding-style.md) before
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
normal runnable set in CI.
