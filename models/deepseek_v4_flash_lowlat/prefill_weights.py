# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Convert batch-one TP weights to the DSpark prefill weight layout."""

import torch


TP_RANKS = 8
PREFILL_CP = 4

SHARED_WEIGHT_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w", "wq_b_scale", "wkv",
    "gamma_cq", "gamma_ckv", "attn_sink", "wo_b_scale",
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w", "csa_hadamard_idx",
    "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_inner_wkv", "csa_inner_wgate",
    "csa_inner_ape", "csa_inner_norm_w", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    "norm_w", "gate_w", "gate_bias", "tid2eid", "hc_head_fn", "hc_head_scale",
    "hc_head_base", "final_norm_w",
)


def shared_prefill_weights(metadata, decode_weights):
    """Reuse unchanged resident weights after checking the generated prefill ABI."""
    rows = {row["name"].split("__ssa", 1)[0]: row for row in metadata["params"]}
    aliases = {"fp32": "float32", "f32": "float32", "bf16": "bfloat16", "i8": "int8", "i32": "int32"}
    result = {}
    for name in SHARED_WEIGHT_NAMES:
        row, root = rows[name], decode_weights["s_" + name]
        dtype = str(root.dtype).removeprefix("torch.")
        expected_dtype = aliases.get(row["dtype"].lower(), row["dtype"].lower())
        if (row["direction"] != "In" or tuple(root.shape) != tuple(row["shape"]) or dtype != expected_dtype
                or tuple(root.worker_ids) != tuple(range(TP_RANKS))):
            raise ValueError(f"{name} is not compatible with the generated prefill input")
        result[name] = root
    return result


def prompt_embeddings(tokens, shards):
    """Look up prompt embeddings from the existing TP8 vocabulary shards."""
    if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1 or tokens.dtype != torch.int64:
        raise ValueError("tokens must be a one-dimensional INT64 tensor")
    if tokens.device.type != "cpu" or len(shards) != TP_RANKS:
        raise ValueError("CPU tokens and eight CPU embedding shards are required")
    first = shards[0]
    if first.ndim != 2 or first.dtype != torch.bfloat16 or first.device.type != "cpu":
        raise ValueError("Embedding shards must be CPU BF16 matrices")
    if any(shard.shape != first.shape or shard.dtype != first.dtype or shard.device.type != "cpu" for shard in shards):
        raise ValueError("Embedding shards must have matching shapes and dtypes")
    vocab_per_rank, hidden = first.shape
    if bool(((tokens < 0) | (tokens >= vocab_per_rank * TP_RANKS)).any()):
        raise ValueError("A token is outside the sharded embedding vocabulary")
    output = torch.empty((tokens.numel(), hidden), dtype=first.dtype)
    for rank, shard in enumerate(shards):
        mask = tokens // vocab_per_rank == rank
        if bool(mask.any()):
            output[mask] = shard.index_select(0, tokens[mask] - rank * vocab_per_rank)
    return output


def pack_nz(value):
    """Pack the last two dimensions into the prefill kernels' FRACTAL_NZ layout."""
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise TypeError("NZ packing requires a tensor with at least two dimensions")
    rows, columns = value.shape[-2:]
    c0 = 32 // value.element_size()
    if rows % 16 or columns % c0:
        raise ValueError("NZ packing requires complete 16-row and 32-byte tiles")
    blocked = value.reshape(-1, rows // 16, 16, columns // c0, c0)
    return blocked.permute(0, 3, 1, 2, 4).contiguous().reshape(value.shape)


def attention_layer_weights(*, wq_a, wq_b, wo_a, wo_b, rank, csa_weights_proj=None):
    """Pack one layer's replicated attention weights for a CP4 prefill rank."""
    if type(rank) is not int or not 0 <= rank < TP_RANKS:
        raise ValueError("rank must be in [0, 8)")
    if wq_a.dtype != torch.bfloat16 or wq_b.dtype != torch.int8:
        raise TypeError("Q projections require BF16 and INT8 weights")
    if wo_a.dtype != torch.bfloat16 or wo_b.dtype != torch.int8:
        raise TypeError("Output projections require BF16 and INT8 weights")
    if wq_a.ndim != 2 or wq_b.ndim != 2 or wq_a.shape[1] != wq_b.shape[0]:
        raise ValueError("Q projection shapes do not agree")
    if wo_a.ndim != 3 or wo_b.ndim != 2 or wo_a.shape[0] != 8:
        raise ValueError("Output projections must use the eight-group decode layout")
    groups, lora, _ = wo_a.shape
    if wo_b.shape[1] != groups * lora:
        raise ValueError("wo_b columns must contain all output-projection groups")
    local_groups = groups // PREFILL_CP
    first = rank % PREFILL_CP * local_groups
    wo_b_groups = wo_b.reshape(wo_b.shape[0], groups, lora).permute(1, 0, 2)
    result = {
        "wq_a": pack_nz(wq_a),
        "wq_b": pack_nz(wq_b),
        "wo_a": wo_a[first : first + local_groups].contiguous(),
        "wo_b": wo_b_groups[first : first + local_groups].contiguous(),
    }
    if csa_weights_proj is not None:
        if csa_weights_proj.dtype != torch.bfloat16 or csa_weights_proj.ndim != 2:
            raise TypeError("CSA projection requires a BF16 matrix")
        result["csa_weights_proj"] = pack_nz(csa_weights_proj)
    return result


def lm_head_weight(shards, rank):
    """Merge adjacent TP8 vocab shards into the corresponding CP4 head shard."""
    if len(shards) != TP_RANKS or type(rank) is not int or not 0 <= rank < TP_RANKS:
        raise ValueError("Eight vocab shards and a valid prefill rank are required")
    first = (rank % PREFILL_CP) * 2
    left, right = shards[first : first + 2]
    if left.dtype != torch.bfloat16 or right.dtype != left.dtype or left.shape != right.shape:
        raise ValueError("Vocabulary shards must have matching BF16 shapes")
    return pack_nz(torch.cat((left, right), dim=0))


def moe_layer_weights(shards, rank):
    """Convert one layer from all-expert TP8 to contiguous EP8 experts.

    Each shard contains the 256 routed experts followed by the shared expert.
    The caller can convert and upload one layer/rank at a time.
    """
    if len(shards) != TP_RANKS or type(rank) is not int or not 0 <= rank < TP_RANKS:
        raise ValueError("Eight TP shards and a valid prefill rank are required")
    keys = ("routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2", "routed_w2_scale")
    if any(any(key not in shard for key in keys) for shard in shards):
        raise ValueError("Every TP shard must contain all six expert tensors")
    first = shards[0]
    if first["routed_w1"].ndim != 3:
        raise ValueError("Expert tensors require [expert, intermediate, hidden] geometry")
    bank, intermediate, hidden = first["routed_w1"].shape
    experts = bank - 1
    if experts <= 0 or experts % TP_RANKS or intermediate <= 0 or hidden <= 0:
        raise ValueError("Expert bank must contain equally sharded routed experts plus one shared expert")
    expected = {
        "routed_w1": ((bank, intermediate, hidden), torch.int8),
        "routed_w1_scale": ((bank, intermediate), torch.float32),
        "routed_w3": ((bank, intermediate, hidden), torch.int8),
        "routed_w3_scale": ((bank, intermediate), torch.float32),
        "routed_w2": ((bank, hidden, intermediate), torch.int8),
        "routed_w2_scale": ((bank, hidden), torch.float32),
    }
    for shard in shards:
        for name, (shape, dtype) in expected.items():
            if tuple(shard[name].shape) != shape or shard[name].dtype != dtype:
                raise ValueError(f"Incompatible TP shard for {name}")
    per_rank = experts // TP_RANKS
    selected = slice(rank * per_rank, (rank + 1) * per_rank)
    scale = first["routed_w2_scale"]
    for shard in shards[1:]:
        if not torch.equal(shard["routed_w2_scale"][selected], scale[selected]):
            raise ValueError("W2 output scales differ across TP shards")
        if not torch.equal(shard["routed_w2_scale"][-1], scale[-1]):
            raise ValueError("Shared-expert W2 output scales differ across TP shards")
    result = {}
    for suffix in ("w1", "w3"):
        weights = [shard["routed_" + suffix] for shard in shards]
        scales = [shard["routed_" + suffix + "_scale"] for shard in shards]
        result["routed_" + suffix] = pack_nz(torch.cat([weight[selected] for weight in weights], dim=1))
        result["routed_" + suffix + "_scale"] = torch.cat([item[selected] for item in scales], dim=1)
        result["shared_" + suffix] = pack_nz(torch.cat([weight[-1] for weight in weights], dim=0))
        result["shared_" + suffix + "_scale"] = torch.cat([item[-1] for item in scales], dim=0)
    weights = [shard["routed_w2"] for shard in shards]
    result["routed_w2"] = pack_nz(torch.cat([weight[selected] for weight in weights], dim=2))
    result["shared_w2"] = pack_nz(torch.cat([weight[-1] for weight in weights], dim=1))
    result["routed_w2_scale"] = scale[selected].contiguous()
    result["shared_w2_scale"] = scale[-1].contiguous()
    return result


def upload_layer_weights(worker, destinations, rank, layer, layer_weights, *, layers=43):
    """Upload converted per-layer weights into their owning prefill banks."""
    if type(rank) is not int or not 0 <= rank < TP_RANKS:
        raise ValueError("rank must be in [0, 8)")
    if type(layer) is not int or type(layers) is not int or not 0 <= layer < layers:
        raise ValueError("layer must be inside the destination bank")
    transfers = []
    for name, value in layer_weights.items():
        root = destinations[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not value.is_contiguous()
                or tuple(root.worker_ids) != tuple(range(TP_RANKS)) or root.dtype != value.dtype):
            raise ValueError(f"{name} requires a contiguous CPU tensor and a matching device bank")
        shard = root.shards[rank]
        size = value.numel() * value.element_size()
        if shard.nbytes != layers * size:
            raise ValueError(f"{name} per-layer extent does not match its actual device bank")
        transfers.append((shard, value, size))
    for shard, value, size in transfers:
        worker.copy_to(shard.data_ptr, value.data_ptr(), size, dst_offset=layer * size, worker_id=rank)
