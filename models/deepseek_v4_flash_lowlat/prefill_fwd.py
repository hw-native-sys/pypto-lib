# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose batch-one prefill over the low-latency model's TP8 weights."""

from contextlib import contextmanager
import importlib
from math import prod
from pathlib import Path
import sys
import threading


_IMPORT_LOCK = threading.RLock()
_RANKS = 8
_LAYERS = 43
_CSA_LAYERS = 21
_HCA_LAYERS = 20


@contextmanager
def _prefill_import(*, query_pipeline=False):
    """Isolate the private prefill operators' import-time TP8 configuration."""
    source = Path(__file__).resolve().parent / "prefill_ops"
    names = tuple(path.stem for path in source.glob("*.py"))
    missing = object()
    with _IMPORT_LOCK:
        old_modules = {name: sys.modules.get(name, missing) for name in names}
        old_path, old_argv = list(sys.path), list(sys.argv)
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(source))
        sys.argv = ["lowlat-prefill", "--tp", "8", "--ep", "8", "--dp", "1", "--lm-head-tp", "8"]
        sys.argv += ["--prefill-query-pipeline", "1" if query_pipeline else "0", "--prefill-resident-weights"]
        try:
            yield
        finally:
            sys.argv = old_argv
            sys.path[:] = old_path
            for name, module in old_modules.items():
                if module is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module


def load_prefill_programs(*, mode="chunk"):
    """Return the TP8 attention, MoE, tap, and head entries for compilation.

    Load in a single-threaded setup process before creating Workers. ``chunk``
    composes all layers in one device program; ``serial`` loads the component
    baseline for the same TP8 weights and arithmetic.
    """
    if mode not in ("chunk", "serial"):
        raise ValueError("Prefill mode must be chunk or serial")
    with _prefill_import(query_pipeline=mode == "chunk"):
        qkv = importlib.import_module("qkv_proj_rope")
        ordered_qkv = importlib.import_module("models.deepseek_v4_flash_lowlat.drafter.qkv_proj_rope")
        geometry = (
            "D", "HEAD_DIM", "ROPE_DIM", "NOPE_DIM", "EPS", "PREFILL_DENSE_TILE", "MATMUL_T_TILE",
            "KV_TILE", "KV_M_TILE", "KV_DENSE_M_TILE", "KV_N_TILE", "KV_K_TILE", "KV_OK", "KV_OM",
            "KV_SPLIT_K_TILE", "KV_RMS_T_TILE",
        )
        mismatched = [name for name in geometry if getattr(qkv, name) != getattr(ordered_qkv, name)]
        if mismatched:
            raise RuntimeError("Prefill KV projection geometry differs from lowlat: " + ", ".join(mismatched))
        qkv.kv_proj_rope = ordered_qkv.kv_proj_rope
        attention = {name: importlib.import_module("prefill_" + name) for name in ("swa", "csa", "hca")}
        head = importlib.import_module("models.deepseek_v4_flash_lowlat.prefill_head")
        moe = importlib.import_module("models.deepseek_v4_flash_lowlat.prefill_moe")
        if any(module.TP_SIZE != _RANKS or module.O_PROJ_LOCAL_GROUPS != 1 for module in attention.values()):
            raise RuntimeError("Unexpected TP8 prefill attention geometry")
        if head.TP_SIZE != _RANKS or (moe.N_RANKS, moe.N_BANK, moe.MOE_INTER) != (8, 257, 256):
            raise RuntimeError("Unexpected TP8 prefill head or MoE weight geometry")
        if mode == "chunk":
            device = importlib.import_module("models.deepseek_v4_flash_lowlat.prefill_device")
            return {"chunk": device.l3_prefill_tp8_fwd}
        programs = {name: getattr(module, "l3_prefill_attention_" + name + "_cp")
                    for name, module in attention.items()}
        programs.update(moe=moe.l3_prefill_moe_tp, tap=head.l3_prefill_tap, head=head.l3_prefill_head)
        return programs


def prefill_arguments(metadata, bindings):
    """Order actual generated prefill arguments by their metadata names."""
    rows = metadata.get("params")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Actual generated prefill parameter metadata is required")
    names = [row["name"].split("__ssa", 1)[0] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Generated prefill parameter names are not unique")
    missing = [name for name in names if name not in bindings]
    if missing:
        raise ValueError(f"Missing prefill arguments: {', '.join(missing)}")
    return tuple(bindings[name] for name in names)


def _checked_arguments(metadata, bindings):
    """Check all component arguments before the first device dispatch."""
    import torch
    from pypto.runtime import StackedDeviceTensor

    rows = metadata["params"]
    arguments = prefill_arguments(metadata, bindings)
    aliases = {"fp32": "float32", "bf16": "bfloat16", "i8": "int8", "i16": "int16",
               "i32": "int32", "i64": "int64"}
    for row, value in zip(rows, arguments, strict=True):
        name = row["name"].split("__ssa", 1)[0]
        expected = row.get("shape")
        dtype = aliases.get(row["dtype"], row["dtype"])
        if expected is None:
            if type(value) is not int or dtype != "int32" or not -(2**31) <= value < 2**31:
                raise ValueError(f"{name} must be a scalar INT32")
            continue
        if not isinstance(value, (torch.Tensor, StackedDeviceTensor)):
            raise TypeError(f"{name} must be a rank-stacked tensor")
        if (len(value.shape) != len(expected)
                or any(want != -1 and want != got for got, want in zip(value.shape, expected, strict=True))
                or str(value.dtype).removeprefix("torch.") != dtype):
            raise ValueError(f"{name} differs from the generated component shape or dtype")
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or not value.is_contiguous() or not value.is_shared():
                raise ValueError(f"{name} must use contiguous pre-fork shared CPU storage")
        elif value.worker_ids != tuple(range(_RANKS)):
            raise ValueError(f"{name} must follow the TP8 worker order")
    return arguments


def prepare_prefill_bindings(worker, caches, weights, inputs, scratch):
    """Bind caller-owned TP8 weights, shared caches, input metadata and scratch."""
    from .prefill_cache import prefill_cache_views

    bindings = {}
    for group in (prefill_cache_views(caches, worker=worker), weights, inputs, scratch):
        overlap = bindings.keys() & group.keys()
        if overlap:
            raise ValueError(f"Duplicate prefill bindings: {', '.join(sorted(overlap))}")
        bindings.update(group)
    return bindings


def _metadata_rows(metadata):
    rows = metadata.get("params")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Actual generated component metadata is required")
    result = {row["name"].split("__ssa", 1)[0]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Generated component parameter names are not unique")
    return result


def _layer_view(worker, root, layer, layers, shape):
    """Borrow a contiguous per-layer view of each rank's owning allocation."""
    import torch
    from pypto.runtime import StackedDeviceTensor

    if not isinstance(root, StackedDeviceTensor) or root.worker_ids != tuple(range(_RANKS)):
        raise TypeError("Layer weights and caches must be rank-ordered device tensors")
    if not 0 <= layer < layers or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("Invalid layer view")
    elements = prod(shape)
    nbytes = elements * torch.empty((), dtype=root.dtype).element_size()
    if any(shard.nbytes != layers * nbytes for shard in root.shards):
        raise ValueError(f"Layer storage does not match {layers} copies of {shape}")
    shards = [worker.view_tensor(shard, shape, byte_offset=layer * nbytes, worker_id=rank)
              for rank, shard in enumerate(root.shards)]
    return StackedDeviceTensor(shards, (_RANKS, *shape), root.worker_ids)


def _layer_weight(worker, bindings, rows, param, source, index, count):
    shape = rows[param].get("shape")
    if not isinstance(shape, list) or shape[0] != _RANKS or any(size <= 0 for size in shape[1:]):
        raise ValueError(f"Actual metadata lacks a static per-layer shape for {param}")
    return _layer_view(worker, bindings[source], index, count, tuple(shape[1:]))


def _layer_cache(worker, bindings, name, index, count):
    root = bindings[name]
    if root.shape[1] % count:
        raise ValueError(f"Cache {name} does not contain {count} equal layer pools")
    shape = (root.shape[1] // count, *root.shape[2:])
    return _layer_view(worker, root, index, count, shape)


def _attention_bindings(worker, metadata, bindings, layer, x_hc):
    kind = "swa" if layer < 2 else ("csa" if layer % 2 == 0 else "hca")
    rows = _metadata_rows(metadata[kind])
    result = {"x_hc_full": x_hc, "x_out_full": bindings["attn_stage"]}
    for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w", "wq_a", "wq_b",
                 "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv", "attn_sink", "wo_b_scale"):
        result[name] = _layer_weight(worker, bindings, rows, name, name, layer, _LAYERS)
    for name, source in (("wo_a", "wo_a_shard"), ("wo_b", "wo_b_shard")):
        result[name] = _layer_weight(worker, bindings, rows, name, source, layer, _LAYERS)
    result["kv_cache"] = _layer_cache(worker, bindings, "kv_cache", layer, _LAYERS)
    for name in ("ori_slot_mapping_full", "position_ids_local", "position_ids_full", "local_request_ids"):
        result[name] = bindings[name]
    result["ori_block_table" if kind != "swa" else "block_table"] = bindings["ori_block_table"]
    prefix = "swa_freqs" if kind == "swa" else "compressed_freqs"
    result["freqs_cos"] = bindings[prefix + "_cos"]
    result["freqs_sin"] = bindings[prefix + "_sin"]
    if kind != "swa":
        csa = kind == "csa"
        order = (layer - 2) // 2 if csa else (layer - 3) // 2
        count = _CSA_LAYERS if csa else _HCA_LAYERS
        family = "csa" if csa else "hca"
        for name in ("wkv", "wgate", "ape", "norm_w"):
            param = "cmp_" + name
            result[param] = _layer_weight(worker, bindings, rows, param, family + "_cmp_" + name, order, count)
        result["compress_state"] = _layer_cache(worker, bindings, family + "_compress_state", order, count)
        result["compress_state_block_table"] = bindings[family + "_compress_state_block_table"]
        result["cmp_kv"] = _layer_cache(worker, bindings, family + "_cmp_kv", order, count)
        result["cmp_block_table"] = bindings[family + "_cmp_block_table"]
        result["cmp_slot_mapping_full"] = bindings[family + "_cmp_slot_mapping_full"]
        result["state_slot_mapping_full"] = bindings[family + "_state_slot_mapping_full"]
        result["cmp_freqs_cos"] = bindings[family + "_cmp_freqs_cos"]
        result["cmp_freqs_sin"] = bindings[family + "_cmp_freqs_sin"]
        result["query_start_loc"] = bindings["query_start_loc"]
        result["state_checkpoint_ends"] = bindings["state_checkpoint_ends"]
        if csa:
            for param, source in (("hadamard_idx", "csa_hadamard_idx"), ("idx_wq_b", "csa_idx_wq_b"),
                                  ("idx_wq_b_scale", "csa_idx_wq_b_scale"),
                                  ("idx_weights_proj", "csa_weights_proj"), ("inner_wkv", "csa_inner_wkv"),
                                  ("inner_wgate", "csa_inner_wgate"), ("inner_ape", "csa_inner_ape"),
                                  ("inner_norm_w", "csa_inner_norm_w")):
                result[param] = _layer_weight(worker, bindings, rows, param, source, order, count)
            for param, source in (("inner_compress_state", "csa_inner_compress_state"),
                                  ("idx_kv_cache", "idx_kv_cache"), ("idx_kv_scale", "idx_kv_scale")):
                result[param] = _layer_cache(worker, bindings, source, order, count)
            result["inner_compress_state_block_table"] = bindings["csa_inner_compress_state_block_table"]
            result["idx_block_table"] = bindings["idx_block_table"]
            result["idx_slot_mapping_full"] = bindings["csa_idx_slot_mapping_full"]
            result["inner_state_slot_mapping_full"] = bindings["csa_inner_state_slot_mapping_full"]
    return kind, result


def _moe_bindings(worker, metadata, bindings, layer, group_tokens):
    rows = _metadata_rows(metadata["moe"])
    result = {"x": bindings["attn_stage"], "out": bindings["x_hc_next"],
              "ids": bindings["input_ids_full"], "layer_id": layer, "group_tokens": group_tokens}
    for param, source in (("hc_fn", "hc_ffn_fn"), ("hc_scale", "hc_ffn_scale"), ("hc_base", "hc_ffn_base"),
                          ("norm_w", "norm_w"), ("gate_w", "gate_w"), ("gate_bias", "gate_bias"),
                          ("tid2eid", "tid2eid"), ("w1", "routed_w1"), ("s1", "routed_w1_scale"),
                          ("w3", "routed_w3"), ("s3", "routed_w3_scale"),
                          ("w2", "routed_w2"), ("s2", "routed_w2_scale")):
        result[param] = _layer_weight(worker, bindings, rows, param, source, layer, _LAYERS)
    return result


def _submit_chunk(worker, program, metadata, bindings, group_tokens, terminal, config):
    """Bind full resident banks and submit one ordered device chunk."""
    aliases = {"wo_a": "wo_a_shard", "wo_b": "wo_b_shard",
               "input_ids": "input_ids_full", "dspark_target_hidden": "target_hidden"}
    rows = _metadata_rows(metadata)
    actual = {}
    for name, row in rows.items():
        if name == "terminal":
            actual[name] = int(terminal)
            continue
        value = bindings[aliases.get(name, name)]
        expected = row.get("shape")
        if expected is not None and list(value.shape) != expected:
            if all(size > 0 for size in expected) and expected[0] == _RANKS:
                value = _layer_view(worker, value, 0, 1, tuple(expected[1:]))
        actual[name] = value
    arguments = _checked_arguments(metadata, actual)
    worker.submit(program, *arguments, config=config).result()
    return {"x_hc_next": bindings["x_hc"], **{name: bindings[name] for name in (
        "target_hidden", "x_out", "logits", "sampled_ids")}}


def submit_prefill(worker, programs, metadata, bindings, *, group_tokens, terminal, config=None):
    """Run 43 TP8 layers using already registered programs on the same Worker.

    The chunk path waits once after all layers. The serial baseline waits at
    component boundaries. The caller retains every bank, cache and scratch
    allocation until completion and submits successive chunks in order.
    """
    from pypto.runtime import StackedDeviceTensor

    required = {"swa", "csa", "hca", "moe", "tap", "head"}
    if set(programs) != set(metadata) or set(programs) not in (required, {"chunk"}):
        raise ValueError("Actual TP8 chunk or six-component programs and metadata are required")
    if type(group_tokens) is not int or not 1 <= group_tokens <= 8192 or type(terminal) is not bool:
        raise ValueError("Invalid prefill token count or terminal flag")
    x_hc = bindings["x_hc"]
    padded = x_hc.shape[1]
    if padded % _RANKS or not group_tokens <= padded < group_tokens + _RANKS:
        raise ValueError("The physical token extent must be padded to TP8")
    if tuple(x_hc.shape) != (_RANKS, padded, 4, 4096):
        raise ValueError("x_hc must contain the full TP8 prompt on each rank")
    scratch_names = ("attn_stage",) if "chunk" in programs else ("attn_stage", "x_hc_next")
    for name in scratch_names:
        if tuple(bindings[name].shape) != tuple(x_hc.shape):
            raise ValueError(f"{name} must match the padded x_hc shape")
    attn = bindings["attn_stage"]
    next_hc = x_hc if "chunk" in programs else bindings["x_hc_next"]
    if not isinstance(attn, StackedDeviceTensor) or not isinstance(next_hc, StackedDeviceTensor):
        raise TypeError("Attention and MoE scratch must be Worker-owned device tensors")
    for left, right in zip(attn.shards, next_hc.shards, strict=True):
        if left.data_ptr < right.data_ptr + right.nbytes and right.data_ptr < left.data_ptr + left.nbytes:
            raise ValueError("Attention and MoE output scratch must not alias")
    expected = {
        "input_ids_full": (_RANKS, padded),
        "position_ids_full": (_RANKS, padded),
        "position_ids_local": (_RANKS, padded // _RANKS),
        "local_request_ids": (_RANKS, padded // _RANKS),
        "target_hidden": (_RANKS, padded // _RANKS, 3 * 4096),
    }
    for name, shape in expected.items():
        if tuple(bindings[name].shape) != shape:
            raise ValueError(f"{name} does not match the TP8 token layout")
    for name in ("ori_slot_mapping_full", "hca_cmp_slot_mapping_full", "hca_state_slot_mapping_full",
                 "csa_cmp_slot_mapping_full", "csa_idx_slot_mapping_full", "csa_state_slot_mapping_full",
                 "csa_inner_state_slot_mapping_full"):
        if tuple(bindings[name].shape) != (_RANKS, padded):
            raise ValueError(f"{name} does not match the padded prompt")
    if "chunk" in programs:
        return _submit_chunk(worker, programs["chunk"], metadata["chunk"], bindings, group_tokens, terminal, config)
    plan = []
    for layer in range(_LAYERS):
        kind, attention = _attention_bindings(worker, metadata, bindings, layer, x_hc)
        plan.append((kind, _checked_arguments(metadata[kind], attention)))
        plan.append(("moe", _checked_arguments(metadata["moe"], _moe_bindings(worker, metadata, bindings, layer, group_tokens))))
        x_hc = bindings["x_hc_next"]
        if layer >= 40:
            tap = {"x_hc": x_hc, "target_hidden": bindings["target_hidden"],
                   "tap_index": layer - 40, "group_tokens": group_tokens}
            plan.append(("tap", _checked_arguments(metadata["tap"], tap)))
    if terminal:
        head = {"x_hc": x_hc, **{name: bindings[name] for name in (
            "hc_head_fn", "hc_head_scale", "hc_head_base", "final_norm_w", "lm_head_weight",
            "logit_row_indices", "grammar_mask", "x_out", "logits", "sampled_ids")}}
        plan.append(("head", _checked_arguments(metadata["head"], head)))
    for name, arguments in plan:
        worker.submit(programs[name], *arguments, config=config).result()
    return {name: bindings[name] for name in ("x_hc_next", "target_hidden", "x_out", "logits", "sampled_ids")}
