#!/usr/bin/env bash
# CSA per-run SHA256 manifest for audit traceability.
#
# Records SHA256 of the 7 CSA source files, git dirty status, PyPTO/PTOAS
# version tags, the full command, and artifact paths.  Run before and after
# each main-length test case to freeze the source baseline.
#
# Usage:
#   scripts/sha_manifest.sh <case_name> [output_dir]
#
# Example:
#   scripts/sha_manifest.sh indexer_16k_b1 /tmp/csa_artifacts
#
# Output: a manifest file named <case_name>_<timestamp>.manifest in the
# output directory (default: current directory).

set -euo pipefail

CASE_NAME="${1:?usage: sha_manifest.sh <case_name> [output_dir]}"
OUTPUT_DIR="${2:-.}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")

MANIFEST_FILE="${OUTPUT_DIR}/${CASE_NAME}_${TIMESTAMP}.manifest"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/models/deepseek_v4_flash_dspark"

CSA_FILES=(
    "${MODEL_DIR}/decode_csa.py"
    "${MODEL_DIR}/decode_indexer.py"
    "${MODEL_DIR}/decode_indexer_topk.py"
    "${MODEL_DIR}/decode_metadata.py"
    "${MODEL_DIR}/decode_sparse_attn_csa.py"
    "${MODEL_DIR}/decode_compressor_ratio4.py"
    "${MODEL_DIR}/decode_indexer_compressor.py"
)

mkdir -p "${OUTPUT_DIR}"
{
    echo "# CSA SHA256 Manifest"
    echo "# case: ${CASE_NAME}"
    echo "# timestamp: ${TIMESTAMP}"
    echo "# host: $(hostname)"
    echo "# user: $(whoami)"
    echo "#"
    echo "# == Source SHA256 =="
    for f in "${CSA_FILES[@]}"; do
        sha256=$(shasum -a 256 "$f" | cut -d' ' -f1)
        basename=$(basename "$f")
        echo "# ${sha256}  ${basename}"
    done
    echo "#"
    echo "# == Git Status =="
    if command -v git &>/dev/null && [ -d "${REPO_ROOT}/.git" ]; then
        echo "# branch: $(cd "${REPO_ROOT}" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'detached')"
        echo "# commit: $(cd "${REPO_ROOT}" && git rev-parse HEAD 2>/dev/null || echo 'unknown')"
        echo "# dirty: $(cd "${REPO_ROOT}" && git diff --quiet HEAD 2>/dev/null && echo 'no' || echo 'yes')"
    else
        echo "# (not a git repository or git unavailable)"
    fi
    echo "#"
    echo "# == Toolchain Versions =="
    if command -v python3 &>/dev/null; then
        echo "# python: $(python3 --version 2>&1)"
    fi
    if command -v pip3 &>/dev/null; then
        pypto_ver=$(pip3 show pypto 2>/dev/null | grep -i '^version:' | cut -d' ' -f2 || echo 'not-installed')
        echo "# pypto: ${pypto_ver}"
    else
        echo "# pypto: (pip unavailable)"
    fi
    echo "#"
    echo "# == Run Command =="
    echo "# ${RUN_CMD:-<not-set>}"
    echo "#"
    echo "# == Artifacts =="
    echo "# output_dir: ${OUTPUT_DIR}"
    echo "# manifest: ${MANIFEST_FILE}"
} > "${MANIFEST_FILE}"

cat "${MANIFEST_FILE}"
echo "" >&2
echo "Manifest written to: ${MANIFEST_FILE}" >&2
