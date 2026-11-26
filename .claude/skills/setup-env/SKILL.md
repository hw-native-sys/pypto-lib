---
name: setup-env
description: Set up the pypto-lib development environment, including pypto, ptoas, torch, CANN/device checks, and runtime variables. Use when preparing to run examples, tests, codegen, or CI-equivalent local validation.
---

# Setup Environment

Read the canonical public setup guides before changing the environment:

- [Installation and Environment](../../../docs/get-started/installation.md)

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
2. Select the PTOAS asset for the host CPU architecture. On the smoke-test CLI,
   `-p a2a3` or `-p a5` selects the chip family. Add `--sim` for simulation
   or `-d <allocated-device>` for device execution.
3. Reuse a user-selected PyPTO checkout when one is provided. Otherwise place
   a new checkout in a scoped sibling directory after obtaining any required
   network approval. Never overwrite or delete an existing checkout.
4. Follow the installation guide and derive every dependency from the selected
   PyPTO revision:
   - initialize PyPTO submodules;
   - take `PTOAS_VERSION` from that revision's `toolchain/versions.env`, never
     from a log or a previous session;
   - install PyPTO, then point `PTOAS_ROOT` at that release's unpacked
     standalone distribution or a dedicated venv containing its matching wheel;
   - leave PTO ISA alone — simpler clones and pins the one managed checkout
     on first use, and `PTO_ISA_ROOT` cannot redirect it.
5. For a device environment, source the selected CANN `set_env.sh` and run
   `npu-smi info` before installing simpler.
6. Install PyPTO's pinned `runtime/` submodule with the installation guide's
   `build_package_${RUNTIME_PLATFORM}` target. Select `a2a3sim` or `a5sim`
   for simulation and `a2a3` or `a5` for a device. Verify the selected
   platform binaries were built.
7. Keep installs inside the active environment and user-writable directories.
   Do not use `sudo`, modify CANN, or install into a system directory unless
   the user explicitly requests and authorizes it.
8. For a device environment, use only a device allocated to the user.
9. Verify:
   - `import pypto`, `import torch`, and imports from `golden`;
   - `<root>/ptoas`, `<root>/ptoas.sh`, or `<root>/bin/ptoas` under `PTOAS_ROOT`
     answers `--version` — probe in that order, as pypto's
     `backend/_ptoas_locate.py` does. The standalone bundle can use
     `ptoas.sh` to select its interpreter. A wheel venv provides `bin/ptoas`;
   - the simpler submodule revision belongs to the selected PyPTO checkout;
   - the commit at the checkout `pypto.runtime.ensure_pto_isa_root()` returns
     matches the full hash in `runtime/pto_isa.pin`;
   - `PTOAS_ROOT` and the repository `PYTHONPATH` are usable in the shell that
     will run the case.
10. When the requested platform is available, run
   `examples/beginner/hello_world.py` as the final smoke test. A device smoke
   requires an allocation; do not substitute an arbitrary visible device.

## Reporting

Report the selected platform, Python environment, PyPTO revision, simpler
submodule revision, PTOAS version and architecture, PTO ISA commit, exported
roots, and smoke-test result. Call out any step that was inspected but not
performed, especially network installs, CANN activation, or device execution.
