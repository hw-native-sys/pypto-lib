# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for documentation links that target repository source files."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = ROOT / "docs" / "_hooks" / "repo_links.py"
SPEC = importlib.util.spec_from_file_location("repo_links", HOOK_PATH)
repo_links = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(repo_links)
repo_links.REF = "test-sha"


def rewrite(markdown: str, source_uri: str = "debugging.md") -> str:
    page = SimpleNamespace(file=SimpleNamespace(src_uri=source_uri))
    return repo_links.on_page_markdown(markdown, page, None, None)


class RepoLinksTest(unittest.TestCase):
    def test_internal_documentation_link_is_unchanged(self):
        markdown = "[Precision](precision-tuning.md#casts)"
        self.assertEqual(rewrite(markdown), markdown)

    def test_repository_file_becomes_blob_link(self):
        rewritten = rewrite("[CI](../.github/workflows/ci.yml#sim)")
        self.assertEqual(
            rewritten,
            "[CI](https://github.com/hw-native-sys/pypto-lib/blob/test-sha/"
            ".github/workflows/ci.yml#sim)",
        )

    def test_nested_page_can_link_to_repository_file(self):
        rewritten = rewrite(
            "[Hello](../../examples/beginner/hello_world.py)",
            "get-started/first-kernel.md",
        )
        self.assertIn("/blob/test-sha/examples/beginner/hello_world.py", rewritten)

    def test_repository_directory_becomes_tree_link(self):
        rewritten = rewrite("[Examples](../../examples/)", "examples/index.md")
        self.assertEqual(
            rewritten,
            "[Examples](https://github.com/hw-native-sys/pypto-lib/tree/test-sha/examples)",
        )

    def test_external_anchor_and_image_links_are_unchanged(self):
        markdown = (
            "[PyPTO](https://www.pypto.ai/pypto/)\n"
            "[Section](#section)\n"
            "![Diagram](../../examples/diagram.png)"
        )
        self.assertEqual(rewrite(markdown, "examples/index.md"), markdown)

    def test_fenced_code_is_unchanged(self):
        markdown = "```markdown\n[CI](../.github/workflows/ci.yml)\n```"
        self.assertEqual(rewrite(markdown), markdown)

    def test_reference_definition_is_rewritten(self):
        rewritten = rewrite("[ci]: ../.github/workflows/ci.yml#sim")
        self.assertEqual(
            rewritten,
            "[ci]: https://github.com/hw-native-sys/pypto-lib/blob/test-sha/"
            ".github/workflows/ci.yml#sim",
        )

    def test_missing_target_warns_and_is_still_rewritten(self):
        with self.assertLogs("mkdocs", level="WARNING") as captured:
            rewritten = rewrite("[Missing](../missing-file.md)")
        self.assertIn("does not exist", captured.output[0])
        self.assertIn("/blob/test-sha/missing-file.md", rewritten)

    def test_link_outside_repository_warns_and_is_unchanged(self):
        markdown = "[Outside](../../../outside.md)"
        with self.assertLogs("mkdocs", level="WARNING") as captured:
            rewritten = rewrite(markdown)
        self.assertIn("escapes the repository", captured.output[0])
        self.assertEqual(rewritten, markdown)


if __name__ == "__main__":
    unittest.main()
