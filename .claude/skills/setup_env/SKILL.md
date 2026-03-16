# Setup Environment

Automated environment setup for pypto-lib development. Detects platform, clones/updates
dependency repos (pypto, PTOAS, simpler), installs Python packages, and validates the
environment by running an example.

## Prerequisites

- Git
- Python 3.10+
- `python3 -m pip` (do NOT use bare `pip3` — it may point to a different Python)
- Network access to GitHub

## Workflow

Execute each step sequentially. Skip steps whose preconditions are already satisfied.

### Step 1: Detect Platform

Run the following and record the results for later use:

```bash
uname -s          # Darwin → macos, Linux → linux
uname -m          # arm64 / aarch64 / x86_64
python3 --version # e.g. Python 3.11.x → PY_VER=cp311
```

Derive platform tags:
- macOS arm64  → wheel tag `macosx_*_arm64`
- macOS x86_64 → wheel tag `macosx_*_x86_64`
- Linux aarch64 → tar.gz asset `ptoas-bin-aarch64.tar.gz`
- Linux x86_64  → tar.gz asset `ptoas-bin-x86_64.tar.gz`

### Step 2: Clone / verify pypto

```bash
WORKSPACE_DIR="$(cd "$(dirname "$PWD")" && pwd)"   # parent of current repo

if [ ! -d "$WORKSPACE_DIR/pypto" ]; then
    git clone git@github.com:hw-native-sys/pypto.git "$WORKSPACE_DIR/pypto"
fi
```

If the directory exists, verify the origin remote points to the correct URL:

```bash
cd "$WORKSPACE_DIR/pypto"
git remote get-url origin   # expect git@github.com:hw-native-sys/pypto.git
```

### Step 3: Check pypto installation & decide whether to update

```bash
python3 -m pip show pypto 2>/dev/null
```

- If pypto is **already installed** and its Location matches the active Python, **ask the
  user** whether they want to pull the latest code and reinstall. If the user says no,
  **skip Step 3a and Step 3b** and proceed to Step 4.
- If pypto is **not installed** (or Location points to a different Python), proceed with
  Step 3a and Step 3b.

### Step 3a: Pull latest pypto main

```bash
cd "$WORKSPACE_DIR/pypto"
git fetch origin
git checkout main
git pull origin main
```

### Step 3b: Install pypto

**Important:** This step requires full permissions (`required_permissions: ["all"]`) because
the build involves CMake/Ninja compilation. Also clean the `build/` directory first to avoid
stale CMake cache conflicts:

```bash
cd "$WORKSPACE_DIR/pypto"
rm -rf build/
python3 -m pip install -e .
```

### Step 4: Clone / verify PTOAS

```bash
if [ ! -d "$WORKSPACE_DIR/PTOAS" ]; then
    git clone git@github.com:zhangstevenunity/PTOAS.git "$WORKSPACE_DIR/PTOAS"
fi
```

### Step 5: Install ptoas from release

The installation method depends on the platform detected in Step 1.

#### Step 5a: Linux — download pre-built binary tarball

On Linux, ptoas is **not** a Python package. Download the pinned version `v0.6` `tar.gz` from
`https://github.com/zhangstevenunity/PTOAS/releases` and extract it next to pypto-lib.

Use the helper script for automated download:

```bash
bash .claude/skills/setup_env/scripts/setup_env.sh install-ptoas
```

Or manually:

1. Pick the tarball matching the architecture from Step 1:
   - `aarch64` → `ptoas-bin-aarch64.tar.gz`
   - `x86_64`  → `ptoas-bin-x86_64.tar.gz`

2. Download the tarball (pinned to `v0.6`):
   ```bash
   PTOAS_VERSION=v0.6
   curl --fail --location --retry 3 --retry-all-errors \
     -o /tmp/ptoas-bin-<arch>.tar.gz \
     https://github.com/zhangstevenunity/PTOAS/releases/download/${PTOAS_VERSION}/ptoas-bin-<arch>.tar.gz
   ```

3. Verify the checksum:
   - For **aarch64**:
     ```bash
     echo "b2082034faf20a2eae9e0355c9c3587353f4a20b39dc68d59bd02a8eb341ca33  /tmp/ptoas-bin-aarch64.tar.gz" | sha256sum -c -
     ```
   - For **x86_64**:
     ```bash
     echo "696782b1c9c0ec24279aaaed83bbb9db97c950584f2c29d229650508824e0851  /tmp/ptoas-bin-x86_64.tar.gz" | sha256sum -c -
     ```

4. Create a target directory and extract into it (the tarball contains `ptoas` and
   `bin/ptoas` at the top level, so extract into a dedicated directory):
   ```bash
   mkdir -p "$WORKSPACE_DIR/ptoas-bin"
   tar -xzf /tmp/ptoas-bin-<arch>.tar.gz -C "$WORKSPACE_DIR/ptoas-bin"
   ```

5. Add execute permissions and set `PTOAS_ROOT`:
   ```bash
   chmod +x "$WORKSPACE_DIR/ptoas-bin/ptoas" "$WORKSPACE_DIR/ptoas-bin/bin/ptoas"
   export PTOAS_ROOT="$WORKSPACE_DIR/ptoas-bin"
   ```

6. Verify:
   ```bash
   "$PTOAS_ROOT/ptoas" --version   # or "$PTOAS_ROOT/bin/ptoas" --version
   ```

**Slow download?** The tarball is ~40–50 MB. If the download speed is very slow (< 50 KB/s)
or the command hangs for more than 2 minutes, **stop the download and ask the user to
manually download the tarball** from `https://github.com/zhangstevenunity/PTOAS/releases/tag/v0.6`
to their `~/Downloads` folder. Then extract from there:

```bash
mkdir -p "$WORKSPACE_DIR/ptoas-bin"
# Verify checksum before extracting (use the appropriate hash for your arch)
echo "<expected_sha256>  ~/Downloads/ptoas-bin-<arch>.tar.gz" | sha256sum -c -
tar -xzf ~/Downloads/ptoas-bin-<arch>.tar.gz -C "$WORKSPACE_DIR/ptoas-bin"
chmod +x "$WORKSPACE_DIR/ptoas-bin/ptoas" "$WORKSPACE_DIR/ptoas-bin/bin/ptoas"
export PTOAS_ROOT="$WORKSPACE_DIR/ptoas-bin"
```

#### Step 5b: macOS — install via Python wheel

On macOS, ptoas is distributed as a Python wheel. Download and install the matching wheel
from `https://github.com/zhangstevenunity/PTOAS/releases/tag/v0.6` (pinned version).

**Important:** This step requires full permissions (`required_permissions: ["all"]`) for
both the download and the `pip install`.

Use the helper script for automated download:

```bash
bash .claude/skills/setup_env/scripts/setup_env.sh install-ptoas
```

Or manually:

1. Visit the [v0.6 release page](https://github.com/zhangstevenunity/PTOAS/releases/tag/v0.6)
   and find the wheel matching your platform: `ptoas-0.1.1-{PY_TAG}-*-{OS_TAG}_*_{ARCH_TAG}.whl`
   (e.g. `ptoas-0.1.1-cp311-cp311-macosx_26_0_arm64.whl`).
2. Download and install:
   ```bash
   PTOAS_VERSION=v0.6
   curl --fail --location --retry 3 --retry-all-errors \
     -o /tmp/<matched_wheel_name> \
     https://github.com/zhangstevenunity/PTOAS/releases/download/${PTOAS_VERSION}/<matched_wheel_name>
   python3 -m pip install /tmp/<matched_wheel_name>
   ```

**Slow download?** If the download speed is very slow (< 50 KB/s)
or the command hangs for more than 2 minutes, **stop the download and ask the user to
manually download the wheel** from `https://github.com/zhangstevenunity/PTOAS/releases/tag/v0.6`
to their `~/Downloads` folder. Then install from there:

```bash
python3 -m pip install ~/Downloads/<matched_wheel_name>
```

### Step 6: Clone / verify simpler

```bash
if [ ! -d "$WORKSPACE_DIR/simpler" ]; then
    git clone git@github.com:ChaoWao/simpler.git "$WORKSPACE_DIR/simpler"
fi
```

### Step 7: Checkout simpler to stable branch

```bash
cd "$WORKSPACE_DIR/simpler"
git fetch origin
git checkout stable
git pull origin stable
export SIMPLER_ROOT="$WORKSPACE_DIR/simpler"
```

### Step 8: Validate environment

Run the example from the pypto-lib working directory:

```bash
cd "$WORKSPACE_DIR/pypto-lib"   # or the current workspace root
python3 examples/paged_attention_example.py
```

Expected output includes:
- `[1] IR Preview:` followed by IR text
- `[2] Compiling...`
- `Output: <path>`
- `[3] Generated files:` followed by a file listing

If the script completes without errors, the environment is correctly configured.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pypto'` | Re-run Step 3b — make sure `python3 -m pip` matches the active `python3` |
| `ModuleNotFoundError: No module named 'ptoas'` | macOS only — re-run Step 5b and check that the wheel matches your platform |
| `ptoas: command not found` or `Permission denied` | Linux — re-run Step 5a, ensure `chmod +x` was applied and `PTOAS_ROOT` is exported |
| `pip3 show pypto` shows wrong Location (e.g. Python 3.9 path) | Uninstall first (`python3 -m pip uninstall pypto`) then reinstall |
| `gh: command not found` | Install GitHub CLI or use `curl` with the GitHub API instead |
| Git clone fails with permission denied | Check SSH keys are configured correctly for GitHub |
| ptoas download is extremely slow | Ask user to download manually from GitHub releases to `~/Downloads` |
| `Wheel ... is invalid` after download | Incomplete download — delete the file and re-download |
