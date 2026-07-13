# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Source-level checks for Qwen3 prefill cache layout and runtime scopes.

The prefill kernels run under ``@pl.jit(auto_scope=False)`` /
``@pl.jit.inline(auto_scope=False)``, so runtime-scope *depth* is exactly the
lexical nesting of ``with pl.scope()`` blocks. The simpler runtime sizes
``_RING_DEPTH == 4`` HeapRing levels independently, and reclaims each nesting
level on its own -- but only for scopes that are actually *nested*. Two
``with pl.scope()`` blocks placed side-by-side sit at the same depth and share
one ring, so they do not isolate memory from each other.

These tests pin the intended 4-ring nesting so a future refactor cannot silently
flatten it (as happened before it was restored):

    implicit top-level ring                         depth 0
    for layer_idx: with pl.scope():                 depth 1   (prefill_fwd)
      for p0_idx:  with pl.scope():                 depth 2   (prefill_layer)
        for final_ti0: with pl.scope():             depth 3   (RoPE + attention)
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFILL_SOURCE = ROOT / "models" / "qwen3" / "14b" / "prefill_fwd.py"


def _module() -> ast.Module:
    return ast.parse(PREFILL_SOURCE.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _loop(scope: ast.AST, target: str) -> ast.For:
    return next(
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == target
    )


def _is_runtime_scope(node: ast.stmt) -> bool:
    if not isinstance(node, ast.With) or len(node.items) != 1:
        return False
    context = node.items[0].context_expr
    return (
        isinstance(context, ast.Call)
        and isinstance(context.func, ast.Attribute)
        and isinstance(context.func.value, ast.Name)
        and context.func.value.id == "pl"
        and context.func.attr == "scope"
    )


def _assert_scope_wrapped(loop: ast.For) -> ast.With:
    """Assert the loop body is exactly one ``with pl.scope()`` and return it."""
    assert len(loop.body) == 1, f"expected loop body to be a single scope, got {len(loop.body)} statements"
    assert _is_runtime_scope(loop.body[0]), "loop body is not a `with pl.scope()` block"
    return loop.body[0]


def _called_names(node: ast.AST) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def _reshape_targets(function: ast.FunctionDef) -> dict[str, str]:
    targets = {}
    for node in ast.walk(function):
        if (
            not isinstance(node, ast.Assign)
            or len(node.targets) != 1
            or not isinstance(node.targets[0], ast.Name)
        ):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "pl"
            and call.func.attr == "reshape"
        ):
            targets[node.targets[0].id] = ast.unparse(call)
    return targets


def test_prefill_scopes_are_nested_four_deep() -> None:
    tree = _module()

    # depth 1 -- one scope per transformer layer.
    layer_scope = _assert_scope_wrapped(_loop(_function(tree, "prefill_fwd"), "layer_idx"))
    assert "prefill_layer" in _called_names(layer_scope)

    # depth 2 -- one scope per token block.
    p0_scope = _assert_scope_wrapped(_loop(_function(tree, "prefill_layer"), "p0_idx"))

    # depth 3 -- one scope per finalize group, nested inside the token-block scope.
    final_scope = _assert_scope_wrapped(_loop(p0_scope, "final_ti0"))

    # The depth-3 scope must hold BOTH RoPE/KV-cache work and attention, so they
    # share the same nesting level rather than living in two side-by-side scopes.
    assert _loop(final_scope, "rope_core"), "RoPE loop not found inside the finalize scope"
    attention_calls = {"_attention_phase_window", "_attention_phase_window_full_single_block"}
    assert attention_calls & _called_names(final_scope), "attention calls not found inside the finalize scope"


def test_finalize_scope_is_the_only_scope_under_the_token_block() -> None:
    """The finalize scope is nested (depth 3), not a sibling of another scope at depth 2."""
    tree = _module()
    p0_scope = _assert_scope_wrapped(_loop(_function(tree, "prefill_layer"), "p0_idx"))

    nested_scopes = [node for stmt in p0_scope.body for node in ast.walk(stmt) if _is_runtime_scope(node)]
    assert len(nested_scopes) == 1, (
        f"expected exactly one nested scope under the token block, found {len(nested_scopes)} "
        "(side-by-side scopes share a ring and do not isolate memory)"
    )


def test_prefill_cache_uses_zero_copy_bsnd_views() -> None:
    tree = _module()

    for function_name in (
        "_attention_phase_window",
        "_attention_phase_window_full_single_block",
        "prefill_layer",
    ):
        targets = _reshape_targets(_function(tree, function_name))
        assert targets["k_cache_bsnd"] == "pl.reshape(k_cache, [cache_token_rows, KV_HIDDEN])"
        assert targets["v_cache_bsnd"] == "pl.reshape(v_cache, [cache_token_rows, KV_HIDDEN])"

    source = PREFILL_SOURCE.read_text(encoding="utf-8")
    assert "(pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE" not in source
    assert "(cache_slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE" not in source
    assert "k_cache_bsnd = k_cache.reshape(-1, kv_hidden)" in source
    assert "v_cache_bsnd = v_cache.reshape(-1, kv_hidden)" in source
