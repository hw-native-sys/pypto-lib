---
name: setup-env
description: Set up the pypto-lib development environment, including pypto, ptoas, torch, CANN/device checks, and runtime variables. Use when preparing to run examples, tests, codegen, or CI-equivalent local validation.
---

# Setup Environment

Read the canonical public setup guides before changing the environment:

- [Installation and Environment](../../../docs/get-started/installation.md)
- [Platforms and Devices](../../../docs/get-started/platforms.md)

Use `.github/actions/setup-ci-job/action.yml` as the executable reference for a
CI-equivalent build. Do not duplicate its current versions or host paths in
this skill.

## Workflow

1. Inspect before mutating:
   - identify the requested simulator or device target;
   - record `uname -m`, the active Python version and environment, compiler
     versions, existing PyPTO/PTO ISA/PTOAS checkouts, and relevant exported
     roots;
   - for a device target, check whether CANN is configured and whether the
     user has an allocated device.
2. Do not infer the target from CPU architecture. Architecture selects the
   PTOAS asset; `-p` selects simulator versus device.
3. Reuse a user-selected PyPTO checkout when one is provided. Otherwise place
   a new checkout in a scoped sibling directory after obtaining any required
   network approval. Never overwrite or delete an existing checkout.
4. Follow the installation guide and derive every dependency from the selected
   PyPTO revision:
   - initialize PyPTO submodules;
   - derive the PTOAS version/checksum and PTO ISA commit from that exact PyPTO
     revision;
   - check out PTO ISA at the pin;
   - install PyPTO;
   - verify the PTOAS checksum before extraction.
5. For a device environment, source the selected CANN `set_env.sh` and run
   `npu-smi info` before installing simpler. Simpler detects `ccec` and the
   cross-compiler during installation and only prebuilds the platforms whose
   toolchains are active.
6. Install PyPTO's pinned `runtime/` submodule after the requested simulator or
   device toolchain is active. Confirm that the requested platform binaries
   were built.
7. Keep installs inside the active environment and user-writable directories.
   Do not use `sudo`, modify CANN, or install into a system directory unless
   the user explicitly requests and authorizes it.
8. For a device environment, use only a device allocated to the user.
9. Verify:
   - `import pypto`, `import torch`, and imports from `golden`;
   - one supported PTOAS executable (`ptoas` or `bin/ptoas`) is a regular
     executable file;
   - the simpler submodule revision belongs to the selected PyPTO checkout;
   - PTO ISA `HEAD` matches `runtime/pto_isa.pin`;
   - `PTOAS_ROOT`, `PTO_ISA_ROOT`, and the repository `PYTHONPATH` are usable in
     the shell that will run the case.
10. When the requested platform is available, run
   `examples/beginner/hello_world.py` as the final smoke test. A device smoke
   requires an allocation; do not substitute an arbitrary visible device.

## Reporting

Report the selected platform, Python environment, PyPTO revision, simpler
submodule revision, PTOAS version and architecture, PTO ISA commit, exported
roots, and smoke-test result. Call out any step that was inspected but not
performed, especially network installs, CANN activation, or device execution.
