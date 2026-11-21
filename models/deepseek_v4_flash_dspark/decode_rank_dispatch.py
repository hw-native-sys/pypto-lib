# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Reuse bound DSpark decode arguments and submit the chip ranks as one group."""

import ast
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import weakref


_FRAME_CACHE = "_dspark_rank_args"
_HOST_HELPER = '''
def _dspark_binding_key(value):
    if isinstance(value, torch.Tensor):
        return id(value), value.data_ptr(), tuple(value.shape), value.stride(), value.dtype
    if type(value) is float:
        return float, value.hex()
    if type(value) is int or type(value) is bool:
        return type(value), value
    return id(value)
'''


def _is_tensor_index(node):
    return isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "tensors"


def rewrite_decode_host(source):
    """Transform the generated one-L2 packing loop, preserving its task ABI."""
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    entry = functions["l3_decode_fwd_dspark"]
    scope = entry.body[-1]
    if not isinstance(scope, ast.With) or len(scope.items) != 1 or len(scope.body) != 2:
        raise ValueError("DSpark decode must submit within one communication domain")
    domain = scope.items[0].optional_vars
    world, loop = scope.body
    worker_loop = isinstance(loop, ast.For) and isinstance(loop.target, ast.Name) and not loop.orelse
    if not isinstance(domain, ast.Name) or not worker_loop:
        raise ValueError("DSpark decode must end with one rank loop")
    if isinstance(world, ast.Assign) and len(world.targets) == 1:
        rank_count = ast.unparse(world.targets[0])
        if ast.unparse(world.value) != "world_size" or ast.unparse(loop.iter) != f"range(0, {rank_count}, 1)":
            raise ValueError("DSpark decode rank loop must visit all workers in order")
    else:
        raise ValueError("DSpark decode rank loop requires a declared worker count")

    rank = loop.target.id
    keep, submit = loop.body[-2:]
    expected = f"_submit_chip(orch, callables['l2_decode_fwd_dspark'], _ta_0, config, {rank})"
    if ast.unparse(keep) != "_keep.append(_ta_0)" or ast.unparse(submit) != expected:
        raise ValueError("DSpark decode rank loop no longer ends with one chip submit")
    loop.body[-2:] = ast.parse("_rank_args.append(_ta_0)").body

    # The generator creates local shape values, tensor views and TaskArgs. No
    # other runtime operation in the rank loop can be skipped on a cache hit.
    for node in ast.walk(loop):
        if isinstance(node, ast.Call):
            call = node.func
            if isinstance(call, ast.Name) and call.id in {"range", "TaskArgs", "add_chip_tensor_arg", "make_tensor_arg"}:
                continue
            if isinstance(call, ast.Attribute) and call.attr in {"add_scalar", "add_tensor", "tensor", "append"}:
                continue
            raise ValueError("DSpark decode rank loop contains an unsupported operation")

    reads, writes = set(), set()
    for node in ast.walk(entry):
        if _is_tensor_index(node):
            if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, str):
                raise ValueError("DSpark decode requires literal names for rank inputs")
            name = node.slice.value
            if isinstance(node.ctx, ast.Store):
                if not name.startswith("t__tmp_v"):
                    raise ValueError("DSpark decode rank loop can only write temporary tensor views")
                writes.add(name)
            else:
                reads.add(name)
    names = tuple(sorted(reads - writes))
    if not names or any(not isinstance(statement, ast.Assign) for statement in entry.body[:-1]):
        raise ValueError("DSpark decode requires a fixed set of rank inputs")

    alloc = functions["_alloc_intermediates"]
    alloc.body.extend(ast.parse(f"tensors[{_FRAME_CACHE!r}] = {{}}").body)
    body = ast.parse(f'''
_bindings = tuple(tensors[name] for name in {names!r})
_key = (id({domain.id}), world_size, tuple(map(_dspark_binding_key, _bindings)))
_cache = tensors[{_FRAME_CACHE!r}]
_cached = _cache.get(_key)
if _cached is None:
    _rank_args = []
    if _domain_provider is not None:
        if len(_cache) == 4:
            del _cache[next(iter(_cache))]
        _cache[_key] = ({domain.id}, _bindings, _rank_args)
else:
    _rank_args = _cached[2]
_keep.extend(_rank_args)
if config.output_prefix:
    for {rank}, _ta_0 in enumerate(_rank_args):
        {expected}
else:
    orch.submit_next_level_group(callables['l2_decode_fwd_dspark'], _rank_args, config, workers=list(range(world_size)))
''').body
    body[4].body.insert(1, loop)
    scope.body[-1:] = body
    tree.body.extend(ast.parse(_HOST_HELPER).body)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def load_decode_rank_dispatch(compiled):
    """Load a derived Host program without changing the original JIT artifact."""
    source = compiled.output_dir
    host_name = Path("orchestration/host_orch.py")
    rewritten = rewrite_decode_host((source / host_name).read_text())
    directory = TemporaryDirectory(prefix="dspark-rank-dispatch-")
    target = Path(directory.name) / "program"
    shutil.copytree(source, target, copy_function=shutil.copyfile)
    (target / host_name).write_text(rewritten)
    result = type(compiled).from_dir(
        target,
        backend_type=compiled.backend_type,
        execution_mode=compiled.execution_mode,
        distributed_config=compiled._distributed_config,
    )
    weakref.finalize(result, directory.cleanup)
    return result
