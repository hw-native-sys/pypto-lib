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

Device environment additionally requires `ASCEND_HOME_PATH` (e.g. `/usr/local/Ascend/cann-8.5.0`).

## Step 2: Install Python Dependencies

```bash
python -m pip install --upgrade pip
pip install nanobind
# Choose based on architecture (see table above)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # sim
pip install torch                                                     # device
```

## Step 3: Install pypto

```bash
WORKSPACE_DIR="$(cd .. && pwd)"
git clone --recurse-submodules --depth=1 https://github.com/hw-native-sys/pypto.git "$WORKSPACE_DIR/pypto"
pip install -v "$WORKSPACE_DIR/pypto"
```

If pypto is already installed and up to date, skip this step.

## Step 4: Install ptoas

The pinned version is read from `.github/workflows/ci.yml` (`PTOAS_VERSION`).
Do not hardcode a version here — always derive it from the CI config so this
skill stays in sync when CI is bumped.

```bash
PTOAS_VERSION=$(grep -m1 'PTOAS_VERSION:' .github/workflows/ci.yml | awk '{print $2}')
ARCH=$(uname -m)   # x86_64 or aarch64
curl --fail --location --retry 3 --retry-all-errors \
  -o /tmp/ptoas-bin-${ARCH}.tar.gz \
  https://github.com/hw-native-sys/PTOAS/releases/download/${PTOAS_VERSION}/ptoas-bin-${ARCH}.tar.gz
mkdir -p "$WORKSPACE_DIR/ptoas-bin"
tar -xzf /tmp/ptoas-bin-${ARCH}.tar.gz -C "$WORKSPACE_DIR/ptoas-bin"
chmod +x "$WORKSPACE_DIR/ptoas-bin/ptoas" "$WORKSPACE_DIR/ptoas-bin/bin/ptoas"
export PTOAS_ROOT="$WORKSPACE_DIR/ptoas-bin"
```

**Slow download?** If < 50 KB/s or hangs > 2 minutes, ask user to manually
download from GitHub releases to `~/Downloads`, then extract from there.

## Step 5: Clone pto-isa

The pinned commit is read from `.github/workflows/ci.yml` (`PTO_ISA_COMMIT`).

```bash
PTO_ISA_COMMIT=$(grep -m1 'PTO_ISA_COMMIT:' .github/workflows/ci.yml | awk '{print $2}')
git clone https://github.com/hw-native-sys/pto-isa.git "$WORKSPACE_DIR/pto-isa"
git -C "$WORKSPACE_DIR/pto-isa" checkout "$PTO_ISA_COMMIT"
export PTO_ISA_ROOT="$WORKSPACE_DIR/pto-isa"
```

## Step 6: Install simpler (bundled in pypto submodule)

simpler is now a git submodule of pypto at `runtime/`. After cloning pypto
in Step 3, install it directly:

```bash
pip install "$WORKSPACE_DIR/pypto/runtime"
```

## Environment Variables

After setup, these must be set:

| Variable | Points to |
|----------|-----------|
| `PTOAS_ROOT` | `../ptoas-bin` |
| `PTO_ISA_ROOT` | `../pto-isa` |
| `ASCEND_HOME_PATH` | `/usr/local/Ascend/cann-8.5.0` (device only) |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: pypto` | Re-run Step 3 or `pip install -v ../pypto` |
| `ptoas: command not found` | Check `PTOAS_ROOT` is exported and `chmod +x` was applied |
| ptoas download very slow | Download manually from GitHub releases to `~/Downloads` |
| Git clone permission denied | Configure SSH keys or use HTTPS URLs |
