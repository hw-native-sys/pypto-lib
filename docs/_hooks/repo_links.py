# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Rewrite links that leave ``docs/`` to repository permalinks."""

from __future__ import annotations

import logging
import os
import posixpath
import re
from pathlib import Path

REPO_URL = "https://github.com/hw-native-sys/pypto-lib"
REPO_ROOT = Path(__file__).resolve().parents[2]
REF = os.environ.get("DOCS_REF", "main")

_FENCE = re.compile(r"(^```.*?^```[ \t]*$|^~~~.*?^~~~[ \t]*$)", re.MULTILINE | re.DOTALL)
_INLINE_LINK = re.compile(
    r"(?<!!)(?P<label>\[[^\]\n]*\])\("
    r"(?!https?://|mailto:|tel:|#)(?P<target>[^)\s]+?)(?P<anchor>#[^)]*)?\)"
)
_REFERENCE_LINK = re.compile(
    r"^(?P<prefix>[ \t]{0,3}\[[^\]]+\]:[ \t]*)(?!https?://|mailto:|tel:|#)"
    r"(?P<target>\S+?)(?P<anchor>#[^\s]+)?(?P<suffix>[ \t]*(?:[\"'(].*)?)$",
    re.MULTILINE,
)

_LOG = logging.getLogger("mkdocs")


def _resolve(page_uri: str, target: str) -> tuple[str, bool] | None:
    """Return a repository path and directory flag for a docs-external link."""
    page_dir = posixpath.dirname(page_uri)
    resolved = posixpath.normpath(posixpath.join(page_dir, target))
    if not resolved.startswith("../"):
        return None

    resolved = resolved[3:]
    if resolved.startswith("../") or not resolved or resolved == ".":
        _LOG.warning("Documentation link escapes the repository: %s", target)
        return None

    local_target = (REPO_ROOT / resolved).resolve()
    try:
        local_target.relative_to(REPO_ROOT)
    except ValueError:
        _LOG.warning("Documentation link escapes the repository: %s", target)
        return None

    if not local_target.exists():
        _LOG.warning("Documentation link target does not exist: %s", resolved)

    is_directory = target.endswith("/") or local_target.is_dir()
    return resolved.rstrip("/"), is_directory


def _repo_url(path: str, is_directory: bool, anchor: str) -> str:
    kind = "tree" if is_directory else "blob"
    return f"{REPO_URL}/{kind}/{REF}/{path}{anchor}"


def _rewrite_segment(markdown: str, page_uri: str) -> str:
    def replace_inline(match: re.Match[str]) -> str:
        resolved = _resolve(page_uri, match.group("target"))
        if resolved is None:
            return match.group(0)
        path, is_directory = resolved
        url = _repo_url(path, is_directory, match.group("anchor") or "")
        return f"{match.group('label')}({url})"

    def replace_reference(match: re.Match[str]) -> str:
        resolved = _resolve(page_uri, match.group("target"))
        if resolved is None:
            return match.group(0)
        path, is_directory = resolved
        url = _repo_url(path, is_directory, match.group("anchor") or "")
        return f"{match.group('prefix')}{url}{match.group('suffix')}"

    markdown = _INLINE_LINK.sub(replace_inline, markdown)
    return _REFERENCE_LINK.sub(replace_reference, markdown)


def on_page_markdown(markdown: str, page, config, files) -> str:  # noqa: ARG001
    """Rewrite source links while leaving fenced code and images unchanged."""
    parts = _FENCE.split(markdown)
    return "".join(
        part if index % 2 else _rewrite_segment(part, page.file.src_uri)
        for index, part in enumerate(parts)
    )
