# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Select the reviewed W2 Cube body while retaining generated Vector arithmetic."""

from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import re

NATIVE_SOURCE = Path(__file__).resolve().parent / "native/experts_w2_prefetch.inc"
# The reviewed PTO input and shared ABI are the same for these PTOAS 0.65 and
# 0.72 outputs; their generated Cube expression scheduling differs.
GENERATED_CUBE_DIGESTS = frozenset({
    "bcefd5b7f4a590885e9e9c0753f3ffb76afe7db16410913964e5826178b0ad7f",
    "c6d08d3890d1fc83c7c9276608dba45d708e52a3e68b1e69235db60dbceee581",
})
CUBE_REGION = re.compile(r"^  #if defined\(__DAV_CUBE__\).*?^  #endif // __DAV_CUBE__", re.M | re.S)
VECTOR_REGION = re.compile(r"^  #if defined\(__DAV_VEC__\).*?^  #endif // __DAV_VEC__", re.M | re.S)
KERNEL_PATTERN = re.compile(
    r"\b(?:l2_s6_cycles_only_|l2_full_cycles_)?exp_routed_balanced"
    r"(?:_[0-9]+)?(?P<core>_aic|_aiv)?\b"
)


def fingerprint():
    """Key JIT artifacts by both the native fragment and its selector."""
    content = NATIVE_SOURCE.read_bytes() + b"\0" + Path(__file__).read_bytes()
    return hashlib.sha256(content).hexdigest()


def normalized_cpp(text):
    """Retain operators, types, addresses and constants; normalize only names."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    text = KERNEL_PATTERN.sub(lambda m: "dspark_experts" + (m["core"] or ""), text)
    names = {}

    def rename(match):
        name = match[0]
        if re.search(r"(__ssa_v\d|_inline\d|__rv_v\d|__idx_v\d|_tensor$|_conv$)", name):
            return names.setdefault(name, "local_" + str(len(names)))
        return name

    return re.sub(r"\s+", "", re.sub(r"\b[A-Za-z_]\w*\b", rename, text))


def compiled_cube_cpp(text):
    """Retain the complete Cube, shared declarations, signatures and exports."""
    if len(CUBE_REGION.findall(text)) != 1 or len(VECTOR_REGION.findall(text)) != 1:
        raise ValueError("Expected one generated Cube and Vector region")
    return normalized_cpp(VECTOR_REGION.sub("", text))


def replace_cube_source(text, expected_fingerprint):
    """Replace only the reviewed Cube region of one complete generated source."""
    if fingerprint() != expected_fingerprint:
        raise ValueError("Native W2 source differs from the requested build fingerprint")
    digest = hashlib.sha256(compiled_cube_cpp(text).encode()).hexdigest()
    if digest not in GENERATED_CUBE_DIGESTS:
        raise ValueError(f"Unrecognized generated W2 Cube, ABI or shared declarations: {digest}")
    regions = CUBE_REGION.findall(NATIVE_SOURCE.read_text())
    if len(regions) != 1:
        raise ValueError("Expected one reviewed native Cube body")
    return CUBE_REGION.sub(lambda _: regions[0], text)


def select_sources(files, expected_fingerprint):
    """Preserve generated Vector code and original sources for publication audit."""
    if fingerprint() != expected_fingerprint:
        raise ValueError("Native W2 source differs from the requested build fingerprint")
    result = dict(files)
    selected = []
    for name, text in files.items():
        path = Path(name)
        if path.suffix != ".cpp" or path.parent.name != "aic":
            continue
        if not KERNEL_PATTERN.fullmatch(path.stem) or not path.stem.endswith("_aic"):
            continue
        selected_source = replace_cube_source(text, expected_fingerprint)
        digest = hashlib.sha256(compiled_cube_cpp(text).encode()).hexdigest()
        result[str(Path("report/original-native-source") / path)] = text
        result[name] = selected_source
        selected.append(
            dict(
                source=name,
                original_compiled_cube_sha256=digest,
                selected_source_sha256=hashlib.sha256(selected_source.encode()).hexdigest(),
            )
        )
    if len(selected) != 10:
        raise ValueError(f"Expected five W2 sites in each of two roles, found {len(selected)}")
    result["report/native-w2-source-selection.json"] = (
        json.dumps(
            dict(
                implementation="tracked_w2_prefetch",
                source_sha256=expected_fingerprint,
                fragment_sha256=hashlib.sha256(NATIVE_SOURCE.read_bytes()).hexdigest(),
                selected=selected,
                old_binaries_loaded=False,
                native_build="REQUIRED",
            ),
            indent=2,
        )
        + "\n"
    )
    return result


@contextmanager
def use_native_w2(expected_fingerprint):
    """Scope source selection to one JIT generation; restore the backend on exit."""
    module = importlib.import_module("pypto.ir.compile")
    original = module.generate
    calls = []

    def generate(*args, **kwargs):
        files = original(*args, **kwargs)
        result = select_sources(files, expected_fingerprint)
        calls.append(True)
        return result

    module.generate = generate
    try:
        yield calls
    finally:
        module.generate = original
