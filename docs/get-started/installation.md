# Installation and Environment

PyPTO-Lib is a source repository rather than an installable Python package.
Run its scripts from the repository root with that root on `PYTHONPATH`.

Four external components sit under it, and the PyPTO checkout you select pins
every one of them:

| Component | How it is installed | Version source |
|---|---|---|
| PyPTO | `pip install` the checkout | the checkout you clone |
| simpler | `pip install` PyPTO's `runtime/` | PyPTO's `runtime/` submodule |
| PTOAS | binary release under `PTOAS_ROOT` | PyPTO's `toolchain/versions.env` |
| PTO ISA | cloned automatically on first use | PyPTO's `runtime/pto_isa.pin` |

Install what the selected PyPTO revision points at. Do not copy a PTOAS version
or a PTO ISA commit from an old CI log or hard-code one in a setup script.

## Prerequisites

- Python 3.10 or newer, Git, CMake, Ninja, and a C++17 compiler — `pip install`
  builds PyPTO's C++ core.
- Simulator platforms compile C++23: `gcc-15` and `g++-15` must resolve to GCC
  15 or newer (`gcc-15 -dumpversion`).
- Real-device runs additionally need CANN and an Ascend device matching the
  platform passed to `-p`. Simulator runs need neither.

## Python environment

From the PyPTO-Lib repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install scikit-build-core nanobind cmake ninja
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Install torch **before** PyPTO, and from the CPU index: PyPTO resolves
`torch>=2.0.0` to the default wheel otherwise, which drags in the whole CUDA
stack that nothing here uses. `PYTHONPATH` is the only variable the repository
itself needs — entry scripts import the `golden/` harness from the repository
root while running from their own directory.

## CANN (real devices only)

Skip this section for a simulator-only environment.

Driver and firmware belong to the machine, need root, and are normally already
there — they are what provides `npu-smi`. Check before installing anything; if
this fails, the host is missing the Ascend HDK packages and that is an
administrator task:

```bash
npu-smi info
```

The toolkit is what PyPTO-Lib builds against, and a plain user can install it.
Download `Ascend-cann-toolkit_<version>_linux-<arch>.run` from the
[community release page](https://www.hiascend.com/developer/download/community/result?module=cann),
matching the driver above (`cat /usr/local/Ascend/driver/version.info`); device
runs in this repository currently use CANN 9.0.0.

```bash
# Packages the .run installer checks for
python -m pip install attrs numpy decorator sympy cffi pyyaml pathlib2 psutil protobuf scipy requests absl-py

chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
./Ascend-cann-toolkit_<version>_linux-<arch>.run --install

# A root install lands in /usr/local/Ascend/ascend-toolkit/latest
export CANN_ROOT="$HOME/Ascend/ascend-toolkit/latest"
source "$CANN_ROOT/set_env.sh"
```

`set_env.sh` exports `ASCEND_HOME_PATH`, where simpler looks for the two
compilers it builds the onboard runtime with. Both must be present, or simpler
builds the simulator platforms only and every device platform is rejected
later:

```bash
test -x "$ASCEND_HOME_PATH/bin/ccec"                                  # AICore
test -f "$ASCEND_HOME_PATH/tools/hcc/bin/aarch64-target-linux-gnu-g++"  # AICPU
```

The toolkit is the only CANN package needed: PyPTO-Lib generates and assembles
its own kernels, and the runtime links only ACL and HCCL, so
`Ascend-cann-kernels-*` and NNAL are not required.

## PyPTO and the runtime

```bash
git clone --recurse-submodules https://github.com/hw-native-sys/pypto.git ../pypto
export PYPTO_ROOT="$(cd ../pypto && pwd)"

python -m pip install --no-build-isolation "$PYPTO_ROOT"
python -m pip install --no-build-isolation "$PYPTO_ROOT/runtime"
```

`runtime/` is the simpler submodule this PyPTO revision pins; installing from
that path keeps build and runtime at the same revision.

**Source CANN before this step.** The simpler installer builds the onboard
binaries with the compilers it finds at install time; sourcing `set_env.sh`
afterwards does not add them, and the fix is to source it and reinstall
`runtime/`.

## PTOAS

Take the pinned version from the same PyPTO checkout and unpack that release:

```bash
version=$(grep '^PTOAS_VERSION=' "$PYPTO_ROOT/toolchain/versions.env" | cut -d= -f2)
echo "$version"   # e.g. v0.57

curl -fL -O "https://github.com/hw-native-sys/PTOAS/releases/download/$version/ptoas-bin-$(uname -m).tar.gz"
mkdir -p ../ptoas-bin
tar -xzf "ptoas-bin-$(uname -m).tar.gz" -C ../ptoas-bin
export PTOAS_ROOT="$(cd ../ptoas-bin && pwd)"
```

The bundle carries its own CPython, so it need not match the venv's Python; the
release also ships `cp310`–`cp312` wheels, which work when installed into a
dedicated venv that `PTOAS_ROOT` then points at. While `PTOAS_ROOT` is set only
that directory is searched, so a `ptoas` earlier on `PATH` cannot shadow the
pinned one.

## PTO ISA

Nothing to clone by hand — PyPTO and simpler each manage their own checkout at
the commit in `$PYPTO_ROOT/runtime/pto_isa.pin`:

| Consumer | Managed checkout | Honours `PTO_ISA_ROOT` |
|---|---|---|
| PyPTO | `build_output/_deps/pto-isa` under the working directory | yes — set it and no clone happens |
| simpler | `build/pto-isa` under the installed `simpler_setup` package | no — it always uses its own copy |

Both clone on first use, so the first device build in a fresh environment
pauses on a `git clone` — on a slow or blocked network that looks like a hang.
Seeding either path with a symlink to an existing pto-isa at the pinned commit
avoids the wait.

## Verify

```bash
python -c "import pypto, torch; from golden import run, run_jit; print(torch.__version__)"
python examples/beginner/hello_world.py -p a2a3sim
```

The example ends in a `[RUN] PASS` line, having exercised the whole chain —
front end, PTOAS, runtime, simulator — so it fails loudly if any piece above is
missing. On a device, in a shell carrying `PYTHONPATH`, `PTOAS_ROOT`, and the
same CANN environment simpler was built against:

```bash
source "$CANN_ROOT/set_env.sh"
python examples/beginner/hello_world.py -p a2a3 -d 0
```

Then proceed to [Run your first kernel](first-kernel.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| `ModuleNotFoundError: golden` | Run from the repository root and export that root in `PYTHONPATH`. |
| `ModuleNotFoundError: pypto` | Activate the intended environment and reinstall the selected PyPTO checkout. |
| PTOAS cannot be found | `PTOAS_ROOT` must be exported and hold `ptoas.sh` (bundle) or `bin/ptoas` (wheel venv). |
| The first device build stalls on a `git clone` | It is fetching pto-isa. On a blocked network, seed the checkout the error names, or symlink an existing clone. |
| Simulator compilation cannot find `g++-15` | Install GCC 15, or provide `gcc-15` / `g++-15` wrappers for the active compiler. |
| A device platform is rejected, or the onboard runtime will not initialize, while `a2a3sim` works | simpler was installed without CANN. Source `set_env.sh`, confirm the two compilers above, then reinstall `$PYPTO_ROOT/runtime`. |
