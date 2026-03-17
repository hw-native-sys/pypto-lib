#!/usr/bin/env bash
# Environment setup helper for pypto-lib.
# Usage:
#   bash setup_env.sh            # Run full setup
#   bash setup_env.sh install-ptoas  # Only install ptoas (binary on Linux, wheel on macOS)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
WORKSPACE_DIR="$(cd "$REPO_ROOT/.." && pwd)"

# ---------------------------------------------------------------------------
# Pinned PTOAS version — keep in sync with pypto CI
# Override via environment variable, e.g. PTOAS_VERSION=v0.8 bash setup_env.sh
# ---------------------------------------------------------------------------
PTOAS_VERSION="${PTOAS_VERSION:-v0.8}"
PTOAS_SHA256_AARCH64="7c73ba35accca6f0b1a05e09bbb1966ff1d390462c2193fa09ccf181a6af9982"
PTOAS_SHA256_X86_64="0434fb472978bd7f19a9bf03634e25b970193f8527fd18e6a38b4b6ee932413f"

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
DetectPlatform() {
    OS_NAME="$(uname -s)"
    ARCH="$(uname -m)"
    PY_FULL="$(python3 --version 2>&1 | awk '{print $2}')"
    PY_MAJOR="${PY_FULL%%.*}"
    PY_MINOR="${PY_FULL#*.}"
    PY_MINOR="${PY_MINOR%%.*}"
    PY_TAG="cp${PY_MAJOR}${PY_MINOR}"

    case "$OS_NAME" in
        Darwin) OS_TAG="macosx" ;;
        Linux)  OS_TAG="manylinux" ;;
        *)      echo "ERROR: Unsupported OS: $OS_NAME"; exit 1 ;;
    esac

    case "$ARCH" in
        arm64|aarch64) ARCH_TAG="$ARCH" ;;
        x86_64)        ARCH_TAG="x86_64" ;;
        *)             echo "ERROR: Unsupported arch: $ARCH"; exit 1 ;;
    esac

    echo "Platform: $OS_NAME $ARCH | Python $PY_FULL ($PY_TAG) | wheel tag: ${OS_TAG}_*_${ARCH_TAG}"
}

# ---------------------------------------------------------------------------
# Clone a repo if missing
# ---------------------------------------------------------------------------
EnsureRepo() {
    local dir="$1" url="$2"
    if [ ! -d "$dir" ]; then
        echo "Cloning $url -> $dir"
        git clone "$url" "$dir"
    else
        echo "Repo exists: $dir"
    fi
}

# ---------------------------------------------------------------------------
# pypto: clone + pull + install
# ---------------------------------------------------------------------------
SetupPypto() {
    EnsureRepo "$WORKSPACE_DIR/pypto" "https://github.com/hw-native-sys/pypto.git"

    echo "Pulling latest pypto main..."
    (cd "$WORKSPACE_DIR/pypto" && git fetch origin && git checkout main && git pull origin main)

    if ! python3 -c "import pypto" 2>/dev/null; then
        echo "Installing pypto in editable mode..."
        (cd "$WORKSPACE_DIR/pypto" && python3 -m pip install -e .)
    else
        echo "pypto already importable."
    fi
}

# ---------------------------------------------------------------------------
# PTOAS repo: clone only (source reference, not for building)
# ---------------------------------------------------------------------------
SetupPtoasRepo() {
    EnsureRepo "$WORKSPACE_DIR/PTOAS" "https://github.com/zhangstevenunity/PTOAS.git"
}

# ---------------------------------------------------------------------------
# ptoas: platform-dependent installation
#   Linux  → download pre-built tar.gz, extract, chmod, set PTOAS_ROOT
#   macOS  → download wheel and pip install
# ---------------------------------------------------------------------------
InstallPtoas() {
    DetectPlatform

    if [ "$OS_NAME" = "Linux" ]; then
        InstallPtoasBinary
    elif [ "$OS_NAME" = "Darwin" ]; then
        InstallPtoasWheel
    else
        echo "ERROR: Unsupported OS for ptoas installation: $OS_NAME"
        exit 1
    fi
}

InstallPtoasBinary() {
    local ptoas_dir="$WORKSPACE_DIR/ptoas-bin"
    if [ -x "$ptoas_dir/ptoas" ] || [ -x "$ptoas_dir/bin/ptoas" ]; then
        echo "ptoas binary already present at $ptoas_dir"
        export PTOAS_ROOT="$ptoas_dir"
        echo "PTOAS_ROOT=$PTOAS_ROOT"
        return 0
    fi

    local tarball="ptoas-bin-${ARCH_TAG}.tar.gz"
    local dl_url="https://github.com/zhangstevenunity/PTOAS/releases/download/${PTOAS_VERSION}/${tarball}"

    # Select checksum by architecture
    local expected_sha256=""
    case "$ARCH_TAG" in
        aarch64) expected_sha256="$PTOAS_SHA256_AARCH64" ;;
        x86_64)  expected_sha256="$PTOAS_SHA256_X86_64" ;;
        *)       echo "ERROR: No pinned SHA256 for arch $ARCH_TAG. Refusing unverified binary install."; exit 1 ;;
    esac

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' RETURN
    echo "Downloading ptoas ${PTOAS_VERSION} (${tarball})..."
    curl --fail --location --retry 3 --retry-all-errors -o "$tmp_dir/$tarball" "$dl_url"

    echo "Verifying SHA256 checksum..."
    echo "${expected_sha256}  $tmp_dir/$tarball" | sha256sum -c -

    echo "Extracting to $ptoas_dir..."
    mkdir -p "$ptoas_dir"
    tar -xzf "$tmp_dir/$tarball" -C "$ptoas_dir"

    chmod +x "$ptoas_dir/ptoas" 2>/dev/null || true
    chmod +x "$ptoas_dir/bin/ptoas" 2>/dev/null || true

    export PTOAS_ROOT="$ptoas_dir"
    echo "PTOAS_ROOT=$PTOAS_ROOT"
    echo "ptoas binary installed successfully (${PTOAS_VERSION})."
}

InstallPtoasWheel() {
    if python3 -m pip show ptoas >/dev/null 2>&1; then
        echo "ptoas already installed."
        python3 -m pip show ptoas | head -3
        return 0
    fi

    echo "Fetching release assets for PTOAS ${PTOAS_VERSION}..."
    local assets
    assets="$(curl --http1.1 --fail --location --retry 3 --retry-all-errors -sS \
              "https://api.github.com/repos/zhangstevenunity/PTOAS/releases/tags/${PTOAS_VERSION}" \
              | python3 -c "import sys,json; [print(a['name']) for a in json.load(sys.stdin).get('assets',[])]")"

    if [ -z "$assets" ]; then
        echo "ERROR: Could not fetch release assets for ${PTOAS_VERSION}."
        exit 1
    fi

    local match=""
    while IFS= read -r name; do
        case "$name" in
            *.whl)
                if echo "$name" | grep -q "${PY_TAG}" && echo "$name" | grep -qi "${OS_TAG}" && echo "$name" | grep -q "${ARCH_TAG}"; then
                    match="$name"
                    break
                fi
                ;;
        esac
    done <<< "$assets"

    if [ -z "$match" ]; then
        echo "ERROR: No matching wheel for ${PY_TAG} / ${OS_TAG} / ${ARCH_TAG} in ${PTOAS_VERSION}."
        echo "Available wheels:"
        echo "$assets" | grep '\.whl$' || true
        exit 1
    fi

    echo "Matched wheel: $match"
    local tmp_dir
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' RETURN

    local dl_url="https://github.com/zhangstevenunity/PTOAS/releases/download/${PTOAS_VERSION}/${match}"
    echo "Downloading $match (${PTOAS_VERSION})..."
    curl --fail --location --retry 3 --retry-all-errors -o "$tmp_dir/$match" "$dl_url"

    echo "Installing $match..."
    python3 -m pip install "$tmp_dir/$match"
    echo "ptoas installed successfully (${PTOAS_VERSION})."
}

# ---------------------------------------------------------------------------
# simpler: clone + checkout stable branch
# ---------------------------------------------------------------------------
SetupSimpler() {
    EnsureRepo "$WORKSPACE_DIR/simpler" "https://github.com/ChaoWao/simpler.git"

    echo "Checking out simpler stable branch..."
    (cd "$WORKSPACE_DIR/simpler" && git fetch origin && git checkout stable && git pull origin stable)
}

# ---------------------------------------------------------------------------
# Validate: run example
# ---------------------------------------------------------------------------
Validate() {
    echo "Running validation example..."
    (cd "$REPO_ROOT" && python3 examples/paged_attention_example.py)
    echo "Environment setup validated successfully."
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
Main() {
    local cmd="${1:-all}"
    case "$cmd" in
        install-ptoas)
            InstallPtoas
            ;;
        all)
            DetectPlatform
            SetupPypto
            SetupPtoasRepo
            InstallPtoas
            SetupSimpler
            Validate
            ;;
        *)
            echo "Usage: $0 [all|install-ptoas]"
            exit 1
            ;;
    esac
}

Main "$@"
