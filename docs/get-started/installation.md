# Installation and Environment

PyPTO-Lib is a source repository rather than an installable Python package.
Run its scripts from the repository root with that root on `PYTHONPATH`.

The environment has four external components:

| Component | Purpose | Version source |
|---|---|---|
| PyPTO | Language, IR, compiler, and Python runtime API | The selected PyPTO checkout |
| simpler | Runtime implementation | PyPTO's `runtime/` submodule |
| PTOAS | PTO bytecode assembler and optimizer | PyPTO's `toolchain/versions.env` |
| PTO ISA | Tile-ISA implementation and headers | PyPTO's `runtime/pto_isa.pin` |

The selected PyPTO revision is the single source of truth. Do not copy a PTOAS
version or PTO ISA commit from an old CI log or hard-code one in a local setup
script.

## Prerequisites

The repository CI uses Python 3.10. A source build also needs Git, `curl`,
`tar`, `sha256sum`, CMake, Ninja, and a C/C++ compiler.

Simulator builds invoke `gcc-15` and `g++-15` and compile C++23 code. Ensure
those commands resolve to GCC 15 or newer:

```bash
gcc-15 -dumpversion
g++-15 -dumpversion
```

For a real NPU run, also provide:

- an installed CANN toolkit with a usable `set_env.sh`;
- `npu-smi` on `PATH`;
- an Ascend device matching the selected runtime platform.

CANN is not required by the simulator jobs in this repository's CI.

## Create a Python environment

From the PyPTO-Lib repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install scikit-build-core nanobind cmake ninja torch

export PYPTO_LIB_ROOT="$PWD"
export PYTHONPATH="$PYPTO_LIB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYPTO_WORKSPACE="$(dirname "$PYPTO_LIB_ROOT")"
```

The remaining commands assume that `pypto`, `pto-isa`, and `ptoas-bin` do not
already exist in `PYPTO_WORKSPACE`. If they do, inspect and update those
checkouts deliberately instead of overwriting them.

## Clone PyPTO and its runtime

```bash
git clone --recurse-submodules \
  https://github.com/hw-native-sys/pypto.git \
  "$PYPTO_WORKSPACE/pypto"

export PYPTO_ROOT="$PYPTO_WORKSPACE/pypto"
git -C "$PYPTO_ROOT" submodule update --init --recursive
```

The `runtime/` directory is the simpler submodule pinned by this PyPTO
revision.

## Resolve the toolchain pins

Read both pins from the same PyPTO checkout:

```bash
export PTOAS_ARCH="$(uname -m)"
export PTOAS_VERSION="$(
  sed -n 's/^PTOAS_VERSION=//p' "$PYPTO_ROOT/toolchain/versions.env" |
    tr -d '[:space:]'
)"
export PTOAS_SHA256="$(
  sed -n \
    "s/^PTOAS_SHA256_$(printf '%s' "$PTOAS_ARCH" | tr '[:lower:]' '[:upper:]')=//p" \
    "$PYPTO_ROOT/toolchain/versions.env" |
    tr -d '[:space:]'
)"
export PTO_ISA_COMMIT="$(
  tr -d '[:space:]' < "$PYPTO_ROOT/runtime/pto_isa.pin"
)"

test -n "$PTOAS_VERSION"
test -n "$PTOAS_SHA256"
test -n "$PTO_ISA_COMMIT"
```

`uname -m` selects the matching PTOAS release asset and checksum. It does
**not** choose between a simulator and a real device; the script's `-p`
argument makes that choice.

## Install PTO ISA, PyPTO, and simpler

```bash
git clone https://github.com/hw-native-sys/pto-isa.git \
  "$PYPTO_WORKSPACE/pto-isa"
git -C "$PYPTO_WORKSPACE/pto-isa" checkout "$PTO_ISA_COMMIT"
export PTO_ISA_ROOT="$PYPTO_WORKSPACE/pto-isa"

python -m pip install --no-build-isolation "$PYPTO_ROOT"
python -m pip install --no-build-isolation "$PYPTO_ROOT/runtime"
```

Installing simpler from `runtime/` keeps the build and runtime at the revision
selected by PyPTO.

## Install the pinned PTOAS release

```bash
export PTOAS_ROOT="$PYPTO_WORKSPACE/ptoas-bin"
export PTOAS_ARCHIVE="/tmp/ptoas-bin-${PTOAS_ARCH}-${PTOAS_VERSION}.tar.gz"

curl --fail --location --retry 3 --retry-all-errors \
  "https://github.com/hw-native-sys/PTOAS/releases/download/${PTOAS_VERSION}/ptoas-bin-${PTOAS_ARCH}.tar.gz" \
  -o "$PTOAS_ARCHIVE"
printf '%s  %s\n' "$PTOAS_SHA256" "$PTOAS_ARCHIVE" | sha256sum -c -

mkdir -p "$PTOAS_ROOT"
tar -xzf "$PTOAS_ARCHIVE" -C "$PTOAS_ROOT"
chmod +x "$PTOAS_ROOT/ptoas" "$PTOAS_ROOT/bin/ptoas"
```

Do not skip the checksum. A missing checksum for the current architecture
means the selected PyPTO revision does not declare a compatible PTOAS asset.

## Configure a real-device shell

Before an NPU run, source the CANN environment selected for the machine:

```bash
export CANN_ROOT=/path/to/Ascend/cann
source "$CANN_ROOT/set_env.sh"
npu-smi info
```

Keep the environment variables from the earlier steps in the same shell:

```bash
export PYTHONPATH="$PYPTO_LIB_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PTOAS_ROOT="$PYPTO_WORKSPACE/ptoas-bin"
export PTO_ISA_ROOT="$PYPTO_WORKSPACE/pto-isa"
```

## Verify the installation

```bash
python -c "import pypto, torch; from golden import run, run_jit; print(torch.__version__)"
test -x "$PTOAS_ROOT/ptoas"
test -x "$PTOAS_ROOT/bin/ptoas"
git -C "$PYPTO_ROOT" submodule status runtime
git -C "$PTO_ISA_ROOT" rev-parse HEAD
```

Then proceed to [Run your first kernel](first-kernel.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| `ModuleNotFoundError: golden` | Run from the repository root and export that root in `PYTHONPATH`. |
| `ModuleNotFoundError: pypto` | Activate the intended Python environment and reinstall the selected PyPTO checkout. |
| PTOAS cannot be found | Verify both PTOAS executables and keep `PTOAS_ROOT` exported. |
| PTO ISA headers cannot be found | Verify `PTO_ISA_ROOT` and that its `HEAD` equals `runtime/pto_isa.pin`. |
| Simulator compilation cannot find `g++-15` | Install GCC 15 or provide `gcc-15` and `g++-15` wrappers for the active compiler environment. |
| A device run cannot initialize the runtime | Source the intended CANN `set_env.sh`, then check `npu-smi info`. |

The CI-equivalent setup is implemented in
[`.github/actions/setup-ci-job/action.yml`](../../.github/actions/setup-ci-job/action.yml).
Use it as the executable source of truth when this page and automation appear
to disagree.
