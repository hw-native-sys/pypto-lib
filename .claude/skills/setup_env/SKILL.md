---
name: setup_env
description: Set up the pypto-lib development environment, including pypto, ptoas, torch, CANN/device checks, and runtime variables. Use when preparing to run examples, tests, codegen, or CI-equivalent local validation.
---

# Setup Environment

Automated environment setup for pypto-lib development. Follows the same steps
as `.github/workflows/ci.yml`.

## Prerequisites

- Git, Python 3.10+, `python3 -m pip`
- Network access to GitHub

## Step 1: Detect Environment

```bash
uname -m   # x86_64 → sim environment, aarch64 → device (a2a3) environment
```

| Architecture | Environment | torch install | ptoas asset |
|-------------|-------------|---------------|-------------|
| `x86_64` | sim | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | `ptoas-bin-x86_64.tar.gz` |
| `aarch64` | device (a2a3) | `pip install torch` | `ptoas-bin-aarch64.tar.gz` |

Device environment additionally requires `ASCEND_HOME_PATH` (e.g. `/usr/local/Ascend/cann-9.0.0`).

## Step 2: Install Python Dependencies

```bash
python -m pip install --upgrade pip
pip install nanobind
# Choose based on architecture (see table above)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # sim
pip install torch                                                     # device
```

## Step 3: Install pypto (single source of truth for the toolchain)

pypto is the **single source of truth**: the pypto revision transitively pins
everything downstream — the runtime (its `runtime/` submodule), ptoas (via
`toolchain/versions.env`), and pto-isa (via `runtime/pto_isa.pin`). Clone pypto
**first**, then derive the ptoas and pto-isa pins from it (Steps 4–5). Never read
these pins from `pypto-lib`'s `.github/workflows/ci.yml` — that file no longer
declares them inline; CI itself reads them out of the pypto checkout.

```bash
WORKSPACE_DIR="$(cd .. && pwd)"
git clone --recurse-submodules --depth=1 https://github.com/hw-native-sys/pypto.git "$WORKSPACE_DIR/pypto"
pip install -v "$WORKSPACE_DIR/pypto"
export PYPTO_ROOT="$WORKSPACE_DIR/pypto"
```

If pypto is already installed and up to date, skip the install but still set
`PYPTO_ROOT` to the checkout so Steps 4–5 can read its pins.

## Step 4: Install ptoas

The pinned version + per-arch sha256 are read from the pypto checkout's
`toolchain/versions.env` (keys `PTOAS_VERSION`, `PTOAS_SHA256_AARCH64`,
`PTOAS_SHA256_X86_64`). Always derive them from pypto so this skill stays in
sync with whatever pypto revision you cloned in Step 3.

```bash
ARCH=$(uname -m)   # x86_64 or aarch64
source "$PYPTO_ROOT/toolchain/versions.env"   # KEY=value, shell-sourceable
if [ "$ARCH" = "aarch64" ]; then PTOAS_SHA256="$PTOAS_SHA256_AARCH64"; else PTOAS_SHA256="$PTOAS_SHA256_X86_64"; fi

curl --fail --location --retry 3 --retry-all-errors \
  -o /tmp/ptoas-bin-${ARCH}.tar.gz \
  https://github.com/hw-native-sys/PTOAS/releases/download/${PTOAS_VERSION}/ptoas-bin-${ARCH}.tar.gz
echo "${PTOAS_SHA256}  /tmp/ptoas-bin-${ARCH}.tar.gz" | sha256sum -c -   # must print OK
mkdir -p "$WORKSPACE_DIR/ptoas-bin"
tar -xzf /tmp/ptoas-bin-${ARCH}.tar.gz -C "$WORKSPACE_DIR/ptoas-bin"
chmod +x "$WORKSPACE_DIR/ptoas-bin/ptoas" "$WORKSPACE_DIR/ptoas-bin/bin/ptoas"
export PTOAS_ROOT="$WORKSPACE_DIR/ptoas-bin"
```

**Slow download?** If < 50 KB/s or hangs > 2 minutes, ask user to manually
download from GitHub releases to `~/Downloads`, then extract from there.

## Step 5: Clone pto-isa

pto-isa is **not** declared in `versions.env`. The runtime is its source of truth
(build == run), so the pinned commit lives in the pypto checkout's
`runtime/pto_isa.pin`. Bumping the pypto `runtime/` submodule moves pto-isa in
lockstep.

```bash
PTO_ISA_COMMIT=$(tr -d '[:space:]' < "$PYPTO_ROOT/runtime/pto_isa.pin")
git clone https://github.com/hw-native-sys/pto-isa.git "$WORKSPACE_DIR/pto-isa"
git -C "$WORKSPACE_DIR/pto-isa" checkout "$PTO_ISA_COMMIT"
export PTO_ISA_ROOT="$WORKSPACE_DIR/pto-isa"
```

## Step 6: Install simpler / runtime (bundled in pypto submodule)

simpler is the pypto runtime, a git submodule of pypto at `runtime/`. After
cloning pypto in Step 3, install it directly:

```bash
pip install "$PYPTO_ROOT/runtime"
```

## Environment Variables

After setup, these must be set:

| Variable | Points to |
|----------|-----------|
| `PTOAS_ROOT` | `../ptoas-bin` |
| `PTO_ISA_ROOT` | `../pto-isa` |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/cann-9.0.0` (device only) |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: pypto` | Re-run Step 3 or `pip install -v ../pypto` |
| `ptoas: command not found` | Check `PTOAS_ROOT` is exported and `chmod +x` was applied |
| ptoas download very slow | Download manually from GitHub releases to `~/Downloads` |
| Git clone permission denied | Configure SSH keys or use HTTPS URLs |
