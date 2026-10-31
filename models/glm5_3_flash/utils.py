# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash checkpoint loading and shared host tensor preparation."""

import dataclasses
import json
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

import models.glm5_3_flash.config as C
import models.glm5_3_flash.config as V
from models.glm5_3_flash.config import (
    BLOCK_SIZE, DENSE_INTER, FLASH, INDEX_DIM, INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE, KV_LORA, N_EXPERTS,
    N_LOCAL_EXPERTS, QK_DIM, SUPPORTED_EP_SIZES, SUPPORTED_TP_SIZES, TP_SIZE, V_DIM, D, Glm53FlashConfig,
)

_FF_GATE = ("gate_proj.weight", "gate_proj")
_FF_UP = ("up_proj.weight", "up_proj")
_FF_DOWN = ("down_proj.weight", "down_proj")
_FF_FUSED = ("w_gate_up", "w_gate_up.weight")
_SCALE_SUFFIXES = (".weight_scale", "_scale")
_OFFSET_SUFFIXES = (".weight_offset", "_offset")
_FLOAT32_MIN = torch.finfo(torch.float32).min

_NZ_ROWS = 16
_NZ_INT8_C0 = 32

CHECKPOINT__NZ_ROWS = 16
CHECKPOINT__NZ_INT8_C0 = 32
ROLES: dict[str, tuple[str, ...]] = {
    "routed_gate": ("experts.{e}.gate_proj.weight", "experts.{e}.gate_proj"),
    "routed_up": ("experts.{e}.up_proj.weight", "experts.{e}.up_proj"),
    "routed_down": ("experts.{e}.down_proj.weight", "experts.{e}.down_proj"),
    "routed_fused": ("experts.{e}.w_gate_up", "experts.{e}.w_gate_up.weight"),
    "routed_fused_down": ("experts.{e}.w_down", "experts.{e}.w_down.weight"),
    "shared_gate": (
        "shared_experts.gate_proj.weight",
        "shared_experts.gate_proj",
        "shared_experts.w1.weight",
    ),
    "shared_up": ("shared_experts.up_proj.weight", "shared_experts.up_proj", "shared_experts.w3.weight"),
    "shared_down": (
        "shared_experts.down_proj.weight",
        "shared_experts.down_proj",
        "shared_experts.w2.weight",
    ),
    "shared_fused": ("shared_experts.w_gate_up",),
    "shared_fused_down": ("shared_experts.w_down",),
    "dense_gate": ("gate_proj.weight", "gate_proj"),
    "dense_up": ("up_proj.weight", "up_proj"),
    "dense_down": ("down_proj.weight", "down_proj"),
    "dense_fused": ("w_gate_up",),
    "dense_fused_down": ("w_down",),
    "router_weight": ("gate.weight",),
    "router_bias": ("gate.e_score_correction_bias", "e_score_correction_bias"),
}
SCALE_SUFFIXES = (".weight_scale", "_scale")
OFFSET_SUFFIXES = (".weight_offset", "_offset")
TEXT_LAYER_ROOTS = ("model.layers.", "model.language_model.layers.")


def _prefill_model():
    overrides = json.loads(os.environ.get("GLM53_PREFILL_MODEL", "{}"))
    return dataclasses.replace(C.FLASH, **overrides) if overrides else C.FLASH


def _pad_to(tensor: torch.Tensor, rows: int, value: int) -> torch.Tensor:
    """Right-pad a 1-D int tensor to ``rows`` entries."""
    if tensor.numel() > rows:
        raise ValueError(f"{tensor.numel()} rows exceed the static capacity {rows}")
    if tensor.numel() == rows:
        return tensor
    pad = torch.full((rows - tensor.numel(),), value, dtype=tensor.dtype)
    return torch.cat([tensor, pad])


@dataclass(frozen=True)
class ForwardMetadata:
    """Canonical token-major metadata consumed by every layer kernel.

    Attributes:
        query_start_loc: ``[requests + 1]`` cumulative packed query offsets.
        query_lens: ``[requests]`` tokens contributed by each request.
        request_ids: ``[tokens]`` request id per packed row.
        position_ids: ``[tokens]`` absolute position of each packed row.
        logit_row_indices: rows whose logits the sampler needs.
        moe_token_owners: ``[tokens]`` TP rank that owns each replicated row
            before EP dispatch, so an attention row replicated across TP is not
            dispatched ``TP_SIZE`` times.
        kv_seq_lens: ``[requests]`` cache length before this dispatch.
        new_kv_seq_lens: ``[requests]`` cache length after this dispatch.
        mla_slots: ``[tokens]`` flattened physical row in the 512-latent pool.
        index_slots: ``[tokens]`` flattened physical row in the per-token indexer
            state pool, paged at :data:`BLOCK_SIZE` like the latent cache.
        pool_slots: ``[tokens]`` flattened physical row of the **pooled** state this
            token belongs to, one row per ``index_kpool`` tokens. Tokens inside the
            same pool share a row, and the row is only complete once the pool closes.
        kda_state_rows: seed/working rows and two checkpoint offset/row pairs per request.
        pool_count: ``[requests]`` number of complete indexer pools.
        tail_start: ``[requests]`` first raw index of the incomplete tail pool.
        tail_count: ``[requests]`` tokens in that tail (``0 .. index_kpool - 1``).
    """

    query_start_loc: torch.Tensor
    query_lens: torch.Tensor
    request_ids: torch.Tensor
    position_ids: torch.Tensor
    logit_row_indices: torch.Tensor
    moe_token_owners: torch.Tensor
    kv_seq_lens: torch.Tensor
    new_kv_seq_lens: torch.Tensor
    mla_slots: torch.Tensor
    index_slots: torch.Tensor
    pool_slots: torch.Tensor
    kda_state_rows: torch.Tensor
    pool_count: torch.Tensor
    tail_start: torch.Tensor
    tail_count: torch.Tensor


class CacheTables:
    """Lower scheduler-owned attention pages into GLM cache tables."""

    def __init__(self, requests: int, max_context: int):
        if min(requests, max_context) <= 0:
            raise ValueError("cache table geometry must be positive")
        self.max_context = max_context
        self.table = torch.full((requests, (max_context + BLOCK_SIZE - 1) // BLOCK_SIZE), -1, dtype=torch.int32)

    def pool_table(self) -> torch.Tensor:
        """Map each logical four-pool page onto its raw page's pooled-key storage."""
        pools_per_raw_page = BLOCK_SIZE // INDEX_KPOOL
        pages_per_raw_page = pools_per_raw_page // INDEX_STATE_BLOCK_SIZE
        logical = torch.arange(
            (self.max_context + INDEX_KPOOL * INDEX_STATE_BLOCK_SIZE - 1) // (INDEX_KPOOL * INDEX_STATE_BLOCK_SIZE),
        )
        raw = self.table[:, logical // pages_per_raw_page]
        physical = raw * pages_per_raw_page + logical.remainder(pages_per_raw_page)
        return torch.where(raw >= 0, physical, -1).int()

    def pool_offsets(self) -> torch.Tensor:
        """Pack score segments in request order, retaining page-aligned private visibility."""
        spans = (self.table >= 0).sum(dim=1) * (BLOCK_SIZE // INDEX_KPOOL)
        return torch.cat((torch.zeros(1, dtype=torch.int64), spans.cumsum(0)[:-1])).int()


def tp_token_owners(
    num_tokens: int,
    tp_size: int = TP_SIZE,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Assign each replicated TP token row to exactly one rank before EP dispatch."""
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    return torch.arange(num_tokens, device=device, dtype=torch.int32).remainder(tp_size)


def paged_slots(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Map logical token positions to flattened physical cache rows."""
    logical_block = torch.div(positions, storage_block_size, rounding_mode="floor")
    offset = positions.remainder(storage_block_size)
    physical = block_table[request_ids.to(torch.long), logical_block.to(torch.long)]
    return physical.to(torch.int64) * storage_block_size + offset.to(torch.int64)


def kpool_metadata(
    kv_seq_lens: torch.Tensor,
    index_kpool: int = INDEX_KPOOL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive the indexer's pool count and incomplete-tail window per request.

    The reference pools from the first *valid* key rather than raw slot zero, which
    matters only for a left-padded dense batch. A paged cache has no left padding —
    request-local position zero is always a real token — so ``first_key`` is zero
    here and the pools are simply ``[4p, 4p+3]``. A kernel that ever sees a padded
    dense layout must add ``first_key`` back.

    Returns:
        ``pool_count`` (complete pools available to the last query of the request),
        ``tail_start`` (first raw index of the incomplete tail) and ``tail_count``
        (how many raw indices that tail holds, ``0 .. index_kpool - 1``).
    """
    visible = kv_seq_lens.to(torch.int64)
    tail_count = visible.remainder(index_kpool)
    tail_start = visible - tail_count
    pool_count = torch.div(visible, index_kpool, rounding_mode="floor")
    return pool_count.to(torch.int32), tail_start.to(torch.int32), tail_count.to(torch.int32)


def build_forward_metadata(
    query_start_loc: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    mla_block_table: torch.Tensor,
    index_block_table: torch.Tensor,
    pool_block_table: torch.Tensor,
    kda_state_rows: torch.Tensor,
    tp_size: int = TP_SIZE,
) -> ForwardMetadata:
    """Lower a scheduler batch into the canonical token-major metadata."""
    device = query_start_loc.device
    query_lengths = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    query_lens = query_lengths.to(torch.int32)
    request_rows = torch.arange(query_lengths.numel(), device=device)
    request_ids = torch.repeat_interleave(request_rows, query_lengths).to(torch.int32)
    num_tokens = int(request_ids.numel())

    local_positions = torch.arange(num_tokens, device=device, dtype=torch.int64) - query_start_loc[
        request_ids.to(torch.long)
    ].to(torch.int64)
    position_ids = (local_positions + kv_seq_lens[request_ids.to(torch.long)].to(torch.int64)).to(torch.int32)
    new_kv_seq_lens = kv_seq_lens.to(torch.int32) + query_lens
    logit_row_indices = (query_start_loc[1:].to(torch.int64) - 1).to(torch.int32)

    pool_count, tail_start, tail_count = kpool_metadata(new_kv_seq_lens)
    return ForwardMetadata(
        query_start_loc=query_start_loc.to(torch.int32), query_lens=query_lens, request_ids=request_ids,
        position_ids=position_ids, logit_row_indices=logit_row_indices,
        moe_token_owners=tp_token_owners(num_tokens, tp_size, device),
        kv_seq_lens=kv_seq_lens.to(torch.int32), new_kv_seq_lens=new_kv_seq_lens,
        mla_slots=paged_slots(position_ids, request_ids, mla_block_table, BLOCK_SIZE).to(torch.int32),
        index_slots=paged_slots(position_ids, request_ids, index_block_table, BLOCK_SIZE).to(torch.int32),
        pool_slots=paged_slots(torch.div(position_ids, INDEX_KPOOL, rounding_mode="floor"), request_ids, pool_block_table, INDEX_STATE_BLOCK_SIZE).to(torch.int32),
        kda_state_rows=kda_state_rows.to(torch.int32),
        pool_count=pool_count, tail_start=tail_start, tail_count=tail_count,
    )


def build_indexer_chunk_metadata(
    metadata: ForwardMetadata,
    index_block_table: torch.Tensor,
    pool_offsets: torch.Tensor,
    *,
    pools_cap: int,
    pool_rows: int,
    pool_block_table: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Lower one chunk against persistent raw and pooled indexer caches.

    ``pool_offsets`` gives each batch request's persistent pooled-row segment.
    Only pools completed by this chunk produce write events; their four raw
    members may include rows cached by an earlier chunk. Query positions and
    expanded attention indices remain absolute positions within each request.
    Empty requests produce no events, and padded events target the scratch row.
    """
    if pools_cap <= 0 or (pool_block_table is None and pool_rows < pools_cap * INDEX_STATE_BLOCK_SIZE):
        raise ValueError("indexer pool storage must cover the identity pool-block gather")
    device = metadata.query_start_loc.device
    old_lengths = metadata.kv_seq_lens.to(device=device, dtype=torch.int64)
    new_lengths = metadata.new_kv_seq_lens.to(device=device, dtype=torch.int64)
    offsets = pool_offsets.to(device=device, dtype=torch.int64)
    requests = old_lengths.numel()
    if offsets.ndim != 1 or offsets.numel() != requests or new_lengths.shape != old_lengths.shape:
        raise ValueError("indexer chunk metadata needs one length and pool offset per request")
    if index_block_table.ndim != 2 or index_block_table.shape[0] != requests:
        raise ValueError("indexer chunk metadata needs one raw block-table row per request")
    if bool(((old_lengths < 0) | (new_lengths < old_lengths)).any()):
        raise ValueError("indexer chunk lengths must extend a nonnegative cached prefix")

    prior_pools = torch.div(old_lengths, INDEX_KPOOL, rounding_mode="floor")
    new_pools = torch.div(new_lengths, INDEX_KPOOL, rounding_mode="floor")
    event_counts = new_pools - prior_pools
    if int(event_counts.sum().item()) > pools_cap:
        raise ValueError("indexer pool-close events exceed the compiled event capacity")
    active_requests = new_lengths > old_lengths
    invalid_windows = (offsets < 0) | (offsets + new_pools > pools_cap)
    if bool((active_requests & invalid_windows).any()):
        raise ValueError("indexer request pool segment exceeds the compiled score width")

    members = torch.full((pools_cap, INDEX_KPOOL), -1, dtype=torch.int32, device=device)
    destinations = torch.full((pools_cap,), pool_rows - 1, dtype=torch.int32, device=device)
    raw_table = index_block_table.to(device=device, dtype=torch.int64)
    lanes = torch.arange(INDEX_KPOOL, dtype=torch.int64, device=device)
    event = 0
    for request in range(requests):
        count = int(event_counts[request].item())
        if not count:
            continue
        pools = torch.arange(prior_pools[request], new_pools[request], device=device)
        positions = pools.unsqueeze(1) * INDEX_KPOOL + lanes.unsqueeze(0)
        logical_blocks = torch.div(positions, BLOCK_SIZE, rounding_mode="floor")
        if int(logical_blocks.max().item()) >= raw_table.shape[1]:
            raise ValueError("indexer pool members exceed the request's raw block table")
        pages = raw_table[request, logical_blocks]
        if bool((pages < 0).any()):
            raise ValueError("indexer pool-close event references an unallocated raw page")
        rows = pages * BLOCK_SIZE + positions.remainder(BLOCK_SIZE)
        members[event : event + count] = rows.to(torch.int32)
        if pool_block_table is None:
            destinations[event : event + count] = (offsets[request] + pools).to(torch.int32)
        else:
            page = pool_block_table[request, pools // INDEX_STATE_BLOCK_SIZE].long()
            if (page < 0).any():
                raise ValueError("pool-close event references an unallocated pooled page")
            destinations[event:event + count] = (page * INDEX_STATE_BLOCK_SIZE + pools.remainder(INDEX_STATE_BLOCK_SIZE)).int(
            )
        event += count

    request_ids = metadata.request_ids.to(device=device, dtype=torch.int64)
    positions = metadata.position_ids.to(device=device, dtype=torch.int64)
    if request_ids.shape != positions.shape:
        raise ValueError("indexer chunk token positions must match token request ids")
    valid = request_ids >= 0
    if bool((request_ids[valid] >= requests).any()):
        raise ValueError("indexer chunk token references an unknown batch request")
    safe_requests = request_ids.clamp(min=0)
    length_through = torch.where(valid, positions + 1, torch.zeros_like(positions))
    tails = length_through.remainder(INDEX_KPOOL)
    visible = torch.div(length_through, INDEX_KPOOL, rounding_mode="floor")
    visible = torch.where(valid, torch.minimum(visible, new_pools[safe_requests]), torch.zeros_like(visible))
    segments = torch.where(valid, offsets[safe_requests], torch.zeros_like(positions))
    pool_blocks = torch.arange(pools_cap, dtype=torch.int32, device=device) * INDEX_STATE_BLOCK_SIZE
    if pool_block_table is not None:
        scratch_base = pool_rows - INDEX_STATE_BLOCK_SIZE
        pool_blocks.fill_(scratch_base)
        for request in range(requests):
            pages = (int(new_pools[request]) + INDEX_STATE_BLOCK_SIZE - 1) // INDEX_STATE_BLOCK_SIZE
            begin = int(offsets[request]) // INDEX_STATE_BLOCK_SIZE
            sources = pool_block_table[request, :pages].int() * INDEX_STATE_BLOCK_SIZE
            if (sources < 0).any() or (sources + INDEX_STATE_BLOCK_SIZE > scratch_base).any():
                raise ValueError("pooled page lies outside shared cache storage")
            pool_blocks[begin : begin + pages] = sources
    return {
        "seg_start": segments.to(torch.int32),
        "pool_count_tok": visible.to(torch.int32),
        "tail_start": (length_through - tails).to(torch.int32),
        "tail_count": tails.to(torch.int32),
        "kv_len_tok": length_through.to(torch.int32),
        "pool_token_slots": members,
        "pool_slots": destinations,
        "pool_blocks": pool_blocks,
    }


def scaled_model_from_config(root: str | Path, base: Glm53FlashConfig = FLASH) -> Glm53FlashConfig:
    """Derive a scaled :class:`Glm53FlashConfig` from a checkpoint's ``config.json``.

    Mini checkpoints (e.g. ``layer_ckpt_l0l3``: layers 0-3 with
    ``num_hidden_layers=4`` / ``first_k_dense_replace=3``) keep the deployment
    format but shrink the schedule. Every ``text_config`` key that names a field
    of the base config overrides it; untouched fields keep the deployment value,
    and the frozen-dataclass validators re-run on construction.
    """
    config = json.loads((Path(root) / "config.json").read_text())
    text = config.get("text_config", config)
    field_names = {f.name for f in fields(base)}
    overrides = {key: value for key, value in text.items() if key in field_names}
    return replace(base, **overrides)


def _as_batch_seqs(token_ids: torch.Tensor) -> torch.Tensor:
    """Normalize token ids to an ``[B, S]`` int64 tensor."""
    ids = torch.as_tensor(token_ids, dtype=torch.int64)
    if ids.dim() == 1:
        ids = ids[None]
    if ids.dim() != 2 or ids.numel() == 0:
        raise ValueError(f"token ids must be [S] or [B, S], got {tuple(ids.shape)}")
    return ids


def _randn(shape: tuple[int, ...], generator: torch.Generator, std: float = 0.02) -> torch.Tensor:
    return torch.randn(shape, generator=generator) * std


def _assert_close(
    actual: torch.Tensor, expected: torch.Tensor, what: str, atol: float = 1e-4, rtol: float = 1e-4
) -> float:
    """Assert against the golden comparator; return the max difference."""
    from golden import ratio_allclose  # noqa: PLC0415 - matches the model-local golden imports

    (passed, message) = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=0.0)(
        actual, expected, actual_outputs={}, expected_outputs={}, inputs={}, atol=atol, rtol=rtol,
    )
    if not passed:
        raise AssertionError(f"{what}:{message}")
    return float((actual - expected).abs().max())


def component_options(description, *, tokens=17, layers=None, argv=None):
    """Parse the common standalone component and checkpoint validation options."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hidden-input", type=Path)
    parser.add_argument("--tokens", type=int, default=tokens)
    parser.add_argument("--compile-only", action="store_true")
    if layers is not None:
        parser.add_argument("--layer", type=int, default=0)
    options = parser.parse_args(argv)
    if options.tokens < 1:
        parser.error("--tokens must be positive")
    if layers is not None and not 0 <= options.layer < layers:
        parser.error(f"--layer must be in [0, {layers})")
    return options


def component_hidden(options, width, generator):
    """Load captured component input or construct deterministic finite BF16 rows."""
    if options.hidden_input is None:
        return torch.randn(options.tokens, width, generator=generator).bfloat16()
    value = torch.load(options.hidden_input, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (options.tokens, width):
        raise ValueError("--hidden-input must have shape [tokens, input_width]")
    if not torch.isfinite(value).all():
        raise ValueError("--hidden-input must contain finite values")
    return value.bfloat16().contiguous()


def golden_split_kv_b(w_kv_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into its key and value halves.

    Args:
        w_kv_b: ``[H * (QK_DIM + V_DIM), KV_LORA]`` as stored in the checkpoint.

    Returns:
        ``w_k`` of ``[H, QK_DIM, KV_LORA]`` and ``w_v`` of ``[H, V_DIM, KV_LORA]``.
    """
    heads = w_kv_b.shape[0] // (QK_DIM + V_DIM)
    reshaped = w_kv_b.reshape(heads, QK_DIM + V_DIM, KV_LORA)
    return reshaped[:, :QK_DIM], reshaped[:, QK_DIM:]


def golden_absorb_output(w_v: torch.Tensor, w_o: torch.Tensor) -> torch.Tensor:
    """Fold the value half into ``o_proj``; returns ``[D, H * KV_LORA]``.

    Decode attends in latent space, so its context rows are ``[H, KV_LORA]`` and the
    value expansion never has to happen per token: ``o_proj @ (ctx @ w_v^T)`` equals
    ``(o_proj folded with w_v) @ ctx``. This is a weight-time transform, and both
    operands have been prepared as BF16 by the loader, so nothing is requantized.

    Args:
        w_v: ``[H, V_DIM, KV_LORA]`` value half from :func:`golden_split_kv_b`.
        w_o: ``[D, H * V_DIM]`` head-space output projection.

    There is no kernel for this: it runs once per layer when the weights are staged,
    so it belongs to the weight loader rather than to any per-token scope. Only the
    query half (:func:`absorb_query`) is per-token.

    Returns:
        ``[D, H * KV_LORA]`` absorbed output projection, in ``w_o``'s dtype.
    """
    heads, v_dim, kv_lora = w_v.shape
    per_head = w_o.float().unflatten(-1, (heads, v_dim))
    absorbed = torch.einsum("dhv,hvk->dhk", per_head, w_v.float())
    return absorbed.reshape(w_o.shape[0], heads * kv_lora).to(w_o.dtype)


def pack_int8_nz(logical: torch.Tensor) -> torch.Tensor:
    """Pack a CPU INT8 tensor's trailing matrix into NZ order, preserving its shape."""
    if not isinstance(logical, torch.Tensor) or logical.dtype != torch.int8 or logical.device.type != "cpu":
        raise TypeError("NZ shared weights must be CPU INT8 tensors before packing")
    if logical.ndim < 2 or any(extent <= 0 for extent in logical.shape):
        raise ValueError("NZ weights require positive dimensions and a trailing matrix")
    rows, cols = logical.shape[-2:]
    if rows % _NZ_ROWS or cols % _NZ_INT8_C0:
        raise ValueError("INT8 NZ weights require 16-row fractals and 32-element C0 lines")
    blocks = logical.reshape(-1, rows // _NZ_ROWS, _NZ_ROWS, cols // _NZ_INT8_C0, _NZ_INT8_C0)
    return blocks.permute(0, 3, 1, 2, 4).contiguous().reshape(logical.shape)


def unpack_int8_nz(packed: torch.Tensor) -> torch.Tensor:
    """Restore a CPU INT8 NZ tensor's trailing matrix to logical row-major order."""
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.int8 or packed.device.type != "cpu":
        raise TypeError("NZ shared weights must be CPU INT8 tensors before unpacking")
    if packed.ndim < 2 or any(extent <= 0 for extent in packed.shape):
        raise ValueError("NZ weights require positive dimensions and a trailing matrix")
    rows, cols = packed.shape[-2:]
    if rows % _NZ_ROWS or cols % _NZ_INT8_C0:
        raise ValueError("INT8 NZ weights require 16-row fractals and 32-element C0 lines")
    blocks = packed.reshape(-1, cols // _NZ_INT8_C0, rows // _NZ_ROWS, _NZ_ROWS, _NZ_INT8_C0)
    return blocks.permute(0, 2, 3, 1, 4).contiguous().reshape(packed.shape)


def sylvester_hadamard(head_dim: int = INDEX_DIM) -> torch.Tensor:
    """Build the scaled Sylvester Hadamard for indexer projection and pooling."""
    matrix = torch.ones((1, 1))
    while matrix.shape[0] < head_dim:
        matrix = torch.cat([torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0)
    return (matrix * (head_dim**-0.5)).to(torch.bfloat16)


def gen_routed_weight(shape, dequant_std, chunk_experts=16):
    """Generate [rank, expert, out, in] INT8 weights with per-channel scales.

    Each bounded rank/expert chunk is independently rescaled to ``dequant_std``.
    ``expert_routed.gen_routed_weight`` instead rescales the complete tensor.
    """
    n_lead = shape[0] * shape[1]
    out_features, in_features = shape[-2], shape[-1]
    w_i8 = torch.empty(shape, dtype=torch.int8)
    scale = torch.empty(*shape[:-1], dtype=torch.float32)
    w_flat = w_i8.reshape(n_lead, out_features, in_features)
    s_flat = scale.reshape(n_lead, out_features)
    for i0 in range(0, n_lead, chunk_experts):
        i1 = min(i0 + chunk_experts, n_lead)
        w = torch.randn(i1 - i0, out_features, in_features)
        amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
        s = amax / 127.0
        w_flat[i0:i1] = torch.round(w / s).clamp_(-127.0, 127.0).to(torch.int8)
        del w
        s_ch = s.squeeze(-1)
        w_deq = w_flat[i0:i1].float() * s_ch.unsqueeze(-1)
        s_flat[i0:i1] = s_ch * (dequant_std / w_deq.std())
        del w_deq
    return w_i8, scale


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _report(name: str) -> None:
    print(f"[GOLDEN] PASS {name}")


def _sparse_attention_fixture(seed: int):
    """Build one small paged selection: cache, block table, indices and query."""
    torch.manual_seed(seed)
    tokens, heads = 5, 2
    qk, kv_lora, v_dim = FLASH.qk_head_dim, 24, 16
    block_table = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32)
    latent_cache = torch.randn(4 * BLOCK_SIZE, kv_lora, dtype=torch.bfloat16)
    request_ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    # Logical positions inside the two pages each request owns, with a padded tail.
    topk_indices = torch.tensor(
        [[0, 1, 5, 130, -1, -1], [0, 3, 129, 200, 7, -1], [2, -1, -1, -1, -1, -1], [0, 1, 2, 3, 4, 5], [131, 6, 0, -1, -1, -1]],
        dtype=torch.int32,
    )
    query = torch.randn(tokens, heads, qk, dtype=torch.bfloat16)
    w_k = torch.randn(heads, qk, kv_lora, dtype=torch.bfloat16) * 0.05
    w_v = torch.randn(heads, v_dim, kv_lora, dtype=torch.bfloat16) * 0.05
    return latent_cache, block_table, request_ids, topk_indices, query, w_k, w_v


def _selected_rows(topk_row, request_id, block_table):
    """Resolve one row's valid logical positions into physical cache rows."""
    rows = []
    for position in topk_row.tolist():
        if position < 0:
            continue
        page = block_table[request_id, position // BLOCK_SIZE].item()
        rows.append(int(page) * BLOCK_SIZE + position % BLOCK_SIZE)
    return rows


def _check_sparse_attention_impl(golden_impl, prefill: bool) -> None:
    """Check paged uniform attention and the BF16 exponential boundary."""
    generator = torch.Generator().manual_seed(93)
    counts = [0, 1, 127, 128, 129, 511, 512, 513, 2051] + [1] * 8
    tokens, heads, pages = len(counts), 2, 17
    qk, kv_lora, v_dim = FLASH.qk_head_dim, FLASH.kv_lora_rank, FLASH.v_head_dim
    block_table = torch.randperm(2 * pages, generator=generator).reshape(2, pages).int()
    request_ids = torch.arange(tokens, dtype=torch.int32) % 2
    cache = torch.randint(-16, 17, (2 * pages * BLOCK_SIZE, kv_lora), generator=generator)
    cache = (cache.float() / 32).bfloat16()
    indices = torch.full((tokens, max(counts)), -1, dtype=torch.int32)
    context = torch.zeros(tokens, heads, kv_lora, dtype=torch.bfloat16)
    for token, count in enumerate(counts):
        if count:
            chosen = torch.randperm(pages * BLOCK_SIZE - 1, generator=generator)[: count - 1]
            indices[token, :count] = torch.cat((chosen, torch.tensor([pages * BLOCK_SIZE - 1])))
            rows = _selected_rows(indices[token], int(request_ids[token]), block_table)
            context[token] = (cache[rows].float().sum(dim=0) / count).bfloat16()
    query_width = qk if prefill else kv_lora
    query = torch.zeros(tokens, heads, query_width, dtype=torch.bfloat16)
    w_k = torch.zeros(heads, qk, kv_lora, dtype=torch.bfloat16)
    w_v = torch.zeros(heads, v_dim, kv_lora, dtype=torch.bfloat16)
    eye = torch.eye(v_dim, dtype=torch.bfloat16)
    for head in range(heads):
        w_v[head, :, :v_dim] = eye
        w_v[head, :, v_dim : 2 * v_dim] = eye * (head + 1) / 2

    def evaluate(selection):
        if prefill:
            return golden_impl(query, cache, w_k, w_v, selection, block_table, request_ids)
        return golden_impl(query, cache, selection, block_table, request_ids)

    expected = context
    if prefill:
        coefficients = torch.arange(1, heads + 1).reshape(1, heads, 1).float() / 2
        expanded = context[..., :v_dim].float() + context[..., v_dim : 2 * v_dim].float() * coefficients
        expected = expanded.bfloat16()
    actual = evaluate(indices)
    _check(actual.dtype is torch.bfloat16, "implementation output must be BF16")
    _check(torch.equal(actual, expected), "implementation changed uniform paged attention")
    trimmed = indices[:, :129].contiguous()
    padded = torch.nn.functional.pad(trimmed, (0, 7), value=-1)
    _check(torch.equal(evaluate(trimmed), evaluate(padded)), "implementation included padded slots")

    cache = torch.zeros(2 * BLOCK_SIZE, kv_lora, dtype=torch.bfloat16)
    cache[BLOCK_SIZE, 1] = 1
    cache[BLOCK_SIZE + 1, 0] = 1
    block_table = torch.tensor([[1, 0]], dtype=torch.int32)
    request_ids = torch.zeros(1, dtype=torch.int32)
    indices = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    query = torch.zeros(1, 1, query_width, dtype=torch.bfloat16)
    query[0, 0, 0] = qk**0.5
    w_k = torch.zeros(1, qk, kv_lora, dtype=torch.bfloat16)
    w_k[0, :, :qk] = torch.eye(qk, dtype=torch.bfloat16)
    w_v = torch.zeros(1, v_dim, kv_lora, dtype=torch.bfloat16)
    w_v[0, :, :v_dim] = eye
    exponentials = torch.exp(torch.tensor([-1.0, 0.0]))
    denominator = exponentials.sum()
    context = torch.zeros(1, 1, kv_lora, dtype=torch.bfloat16)
    context[0, 0, 0] = 1 / denominator
    context[0, 0, 1] = exponentials[0].bfloat16().float() / denominator
    unrounded = (exponentials[0] / denominator).bfloat16()
    rounded_denominator = exponentials.bfloat16().float().sum()
    rounded_ratio = (exponentials[0].bfloat16().float() / rounded_denominator).bfloat16()
    _check(context[0, 0, 1] != unrounded, "boundary fixture did not expose the BF16 exponential")
    _check(context[0, 0, 1] != rounded_ratio, "boundary fixture did not expose the FP32 denominator")
    expected = context[..., :v_dim] if prefill else context
    _check(torch.equal(evaluate(indices), expected), "implementation changed exponential rounding")


def vision_weight_layouts() -> dict[str, tuple[int, ...]]:
    """Return the complete released BF16 vision-tower tensor layout."""
    depth, hidden, intermediate = V.VISION_DEPTH, V.VISION_HIDDEN, V.VISION_INTER
    return {
        "patch_weight": (hidden, V.VISION_PATCH_IN),
        "patch_bias": (hidden,),
        "norm1_weight": (depth, hidden),
        "qkv_weight": (depth, 3 * hidden, hidden),
        "qkv_bias": (depth, 3 * hidden),
        "q_norm_weight": (depth, V.VISION_HEAD_DIM),
        "k_norm_weight": (depth, V.VISION_HEAD_DIM),
        "out_weight": (depth, hidden, hidden),
        "out_bias": (depth, hidden),
        "norm2_weight": (depth, hidden),
        "gate_weight": (depth, intermediate, hidden),
        "gate_bias": (depth, intermediate),
        "up_weight": (depth, intermediate, hidden),
        "up_bias": (depth, intermediate),
        "down_weight": (depth, hidden, intermediate),
        "down_bias": (depth, hidden),
        "post_norm_weight": (hidden,),
        "downsample_weight": (D, V.VISION_MERGE**2 * hidden),
        "downsample_bias": (D,),
        "merger_proj_weight": (D, D),
        "merger_norm_weight": (D,),
        "merger_norm_bias": (D,),
        "merger_gate_weight": (V.VISION_PROJ_INTER, D),
        "merger_up_weight": (V.VISION_PROJ_INTER, D),
        "merger_down_weight": (D, V.VISION_PROJ_INTER),
    }


def layer_prefix(layer_id: int) -> str:
    """Canonical layer prefix inside the text backbone."""
    return f"{TEXT_LAYER_ROOTS[0]}{layer_id}.mlp."


def layer_prefixes(layer_id: int) -> tuple[str, ...]:
    """Every accepted layer prefix, canonical spelling first."""
    return tuple(f"{root}{layer_id}.mlp." for root in TEXT_LAYER_ROOTS)


def _prefixed(prefixes: Sequence[str], templates: Sequence[str], **fmt: object) -> list[str]:
    """All ``prefix + template`` lookups, template priority inside a prefix."""
    return [prefix + template.format(**fmt) for prefix in prefixes for template in templates]


@dataclass(frozen=True)
class QuantDescription:
    """The deployment checkpoint's ``quant_model_description.json``, in part."""

    model_quant_type: str
    group_size: int
    is_rot_used: bool = False

    @classmethod
    def parse(cls, description: Mapping[str, object]) -> "QuantDescription":
        return cls(
            model_quant_type=str(description["model_quant_type"]), group_size=int(description["group_size"]),
            is_rot_used=bool(description.get("is_rot_used", False)),
        )

    def require_dynamic_symmetric(self) -> None:
        """The only scheme this loader implements, and the only one shipped."""
        if self.model_quant_type != "W8A8_DYNAMIC":
            raise ValueError(f"expected W8A8_DYNAMIC, got {self.model_quant_type!r}")
        if self.group_size != 0:
            raise ValueError(f"expected group_size 0 (per-channel), got {self.group_size}")


@dataclass(frozen=True)
class FfnShard:
    """One FFN's INT8 weights and dequant scales, full or TP-sliced."""

    gate_up: torch.Tensor  # [2 * inter, D] INT8, gate rows then up rows
    gate_up_scale: torch.Tensor  # [2 * inter] FP32
    down: torch.Tensor  # [D, inter] INT8
    down_scale: torch.Tensor  # [D] FP32


@dataclass(frozen=True)
class RoutedShard:
    """This EP rank's routed experts, one FFN per expert."""

    w_gate_up: torch.Tensor  # [n_local, 2 * MOE_INTER, D] INT8
    w_gate_up_scale: torch.Tensor  # [n_local, 2 * MOE_INTER] FP32
    w_down: torch.Tensor  # [n_local, D, MOE_INTER] INT8
    w_down_scale: torch.Tensor  # [n_local, D] FP32


@dataclass(frozen=True)
class RouterTensors:
    """The router's BF16 weight and FP32 correction bias (replicated, not sharded)."""

    weight: torch.Tensor
    correction_bias: torch.Tensor


def load_tensors(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a checkpoint file or directory into one name -> tensor mapping.

    ``*.safetensors`` is used when the package is available; otherwise a
    directory of ``*.pt`` shards, or a single ``.pt`` mapping, is read. The
    layout contract above is about names and dtypes, so any reader that produces
    the same mapping works.
    """
    root = Path(path)
    if root.is_dir():
        files = sorted(root.glob("*.safetensors")) or sorted(root.glob("*.pt"))
    else:
        files = [root]
    tensors: dict[str, torch.Tensor] = {}
    for file in files:
        if file.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - exercised only with real shards
                raise ImportError(
                    "reading .safetensors needs the safetensors package; convert to .pt or install it"
                ) from exc
            tensors.update(load_file(str(file)))
        else:
            payload = torch.load(str(file), map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError(f"{file} is not a tensor mapping")
            tensors.update(payload)
    return tensors


def _find(tensors: Mapping[str, torch.Tensor], candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in tensors:
            return name
    raise KeyError(f"none of {list(candidates)} is in the checkpoint")


def _scale_of(tensors: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    for suffix in SCALE_SUFFIXES:
        key = name + suffix
        if key in tensors:
            scale = tensors[key]
            if not scale.is_floating_point() or not torch.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError(f"{key} must contain finite positive floating-point scales")
            scale = scale.to(torch.float32)
            # [out, 1] -> [out]; already-1-D stays put.
            return scale.reshape(-1).contiguous()
    raise KeyError(f"{name} has no weight_scale ({list(SCALE_SUFFIXES)})")


def _weight_shape(tensors: Mapping[str, torch.Tensor], name: str) -> tuple[int, ...]:
    return tensors.shape(name) if isinstance(tensors, Checkpoint) else tuple(tensors[name].shape)


def _read_quant(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    rows: slice = slice(None),
    cols: slice = slice(None),
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(tensors, Checkpoint):
        return tensors.quant(name, rows, cols)
    weight = tensors[name]
    scale = _scale_of(tensors, name)
    if weight.dtype != torch.int8 or weight.ndim != 2 or scale.numel() != weight.shape[0]:
        raise ValueError(f"{name} must be an INT8 matrix with one scale per output row")
    for suffix in OFFSET_SUFFIXES:
        offset = tensors.get(name + suffix)
        if offset is not None and bool(offset.any()):
            raise ValueError(f"{name}{suffix} must be zero for symmetric W8A8")
    return weight[rows, cols].contiguous(), scale[rows].contiguous()


def fuse_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    gate_scale: torch.Tensor,
    up_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse separate gate/up projections into the kernels' layout.

    The kernels read ``w_gate_up`` with gate rows first and up rows second, and
    one scale vector in the same order.
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shapes differ: {tuple(gate.shape)} vs {tuple(up.shape)}")
    return (
        torch.cat([gate, up], dim=0).contiguous(),
        torch.cat([gate_scale.reshape(-1), up_scale.reshape(-1)]).contiguous(),
    )


def _try_find(tensors: Mapping[str, torch.Tensor], candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in tensors:
            return name
    return None


def _fused_ffn(
    tensors: Mapping[str, torch.Tensor],
    prefixes: Sequence[str],
    roles: tuple[str, str, str, str, str],
    *,
    expert: int | None = None,
    intermediate: int | None = None,
    channels: slice = slice(None),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read one FFN's gate/up halves, down projection and scales for selected channels.

    ``roles`` is ``(fused_gate_up, fused_down, gate, up, down)``: a checkpoint
    that stores the fused pair names its down projection ``w_down``, a separate
    one names it ``down_proj``, and either spelling is accepted. ``prefixes``
    carries every accepted layer-root spelling.
    """
    fmt = {"e": expert} if expert is not None else {}
    fused_role, fused_down_role, gate_role, up_role, down_role = roles
    fused_name = _try_find(tensors, _prefixed(prefixes, ROLES[fused_role], **fmt))
    down_name = _find(tensors, _prefixed(prefixes, ROLES[fused_down_role] + ROLES[down_role], **fmt))
    if fused_name is not None:
        shape = _weight_shape(tensors, fused_name)
        if len(shape) != 2:
            raise ValueError(f"{fused_name} must be a matrix, got shape {shape}")
        intermediate = intermediate if intermediate is not None else shape[0] // 2
        if shape != (2 * intermediate, D):
            raise ValueError(f"{fused_name} must have shape {(2 * intermediate, D)}, got {shape}")
        if channels == slice(None):
            gate_up, gate_up_scale = _read_quant(tensors, fused_name)
        else:
            gate, gate_scale = _read_quant(tensors, fused_name, channels)
            up_rows = slice(intermediate + channels.start, intermediate + channels.stop)
            up, up_scale = _read_quant(tensors, fused_name, up_rows)
            gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    else:
        gate_name = _find(tensors, _prefixed(prefixes, ROLES[gate_role], **fmt))
        up_name = _find(tensors, _prefixed(prefixes, ROLES[up_role], **fmt))
        shape = _weight_shape(tensors, gate_name)
        if len(shape) != 2:
            raise ValueError(f"{gate_name} must be a matrix, got shape {shape}")
        intermediate = intermediate if intermediate is not None else shape[0]
        if shape != (intermediate, D) or _weight_shape(tensors, up_name) != shape:
            raise ValueError(f"{gate_name} and {up_name} must have shape {(intermediate, D)}")
        gate, gate_scale = _read_quant(tensors, gate_name, channels)
        up, up_scale = _read_quant(tensors, up_name, channels)
        gate_up, gate_up_scale = fuse_gate_up(gate, up, gate_scale, up_scale)
    if _weight_shape(tensors, down_name) != (D, intermediate):
        raise ValueError(f"{down_name} must have shape {(D, intermediate)}")
    down, down_scale = _read_quant(tensors, down_name, cols=channels)
    return gate_up, gate_up_scale, down, down_scale


def shard_routed_experts(
    tensors: Mapping[str, torch.Tensor],
    layer_id: int,
    ep_rank: int,
    ep_size: int,
    *,
    local_experts: int = N_LOCAL_EXPERTS,
    global_experts: int = N_EXPERTS,
) -> RoutedShard:
    """Stack this EP rank's experts into the batched kernel tensors."""
    if ep_size not in SUPPORTED_EP_SIZES:
        raise ValueError(f"ep_size must be one of {SUPPORTED_EP_SIZES}, got {ep_size}")
    if global_experts <= 0 or local_experts <= 0:
        raise ValueError("global_experts and local_experts must be positive")
    if global_experts % ep_size:
        raise ValueError(f"{global_experts} experts cannot be split across EP{ep_size}")
    if local_experts != global_experts // ep_size:
        raise ValueError(f"local_experts {local_experts} != {global_experts} / {ep_size}")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}")

    prefixes = layer_prefixes(layer_id)
    first = ep_rank * local_experts
    stacked = []
    for local, expert in enumerate(range(first, first + local_experts)):
        values = _fused_ffn(
            tensors, prefixes,
            ("routed_fused", "routed_fused_down", "routed_gate", "routed_up", "routed_down"), expert=expert,
            intermediate=FLASH.moe_intermediate_size,
        )
        if local == 0:
            stacked = [torch.empty((local_experts, *value.shape), dtype=value.dtype) for value in values]
        for target, value in zip(stacked, values):
            target[local].copy_(value)
    return RoutedShard(*stacked)


def load_shared_expert(
    tensors: Mapping[str, torch.Tensor],
    layer_id: int,
) -> FfnShard:
    """Load the complete shared expert replicated on every EP rank."""
    prefixes = layer_prefixes(layer_id)
    intermediate = FLASH.moe_intermediate_size
    (gate_up, gate_up_scale, down, down_scale) = _fused_ffn(
        tensors, prefixes, ("shared_fused", "shared_fused_down", "shared_gate", "shared_up", "shared_down"),
        intermediate=intermediate,
    )
    return FfnShard(gate_up, gate_up_scale, down, down_scale)


def shard_dense_mlp(
    tensors: Mapping[str, torch.Tensor],
    layer_id: int,
    tp_rank: int,
    tp_size: int,
    *,
    intermediate: int = DENSE_INTER,
    local_inter: int | None = None,
) -> FfnShard:
    """TP-slice one dense layer's MLP for one rank (layers 0-2)."""
    if tp_size not in SUPPORTED_TP_SIZES:
        raise ValueError(f"tp_size must be one of {SUPPORTED_TP_SIZES}, got {tp_size}")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    if intermediate % tp_size:
        raise ValueError(f"dense intermediate {intermediate} does not divide TP{tp_size}")
    local_inter = local_inter if local_inter is not None else intermediate // tp_size
    if local_inter != intermediate // tp_size:
        raise ValueError(f"local_inter must be {intermediate // tp_size}, got {local_inter}")
    channels = slice(tp_rank * local_inter, (tp_rank + 1) * local_inter)
    prefixes = layer_prefixes(layer_id)
    (gate_up, gate_up_scale, down, down_scale) = _fused_ffn(
        tensors, prefixes, ("dense_fused", "dense_fused_down", "dense_gate", "dense_up", "dense_down"),
        intermediate=intermediate, channels=channels,
    )
    return FfnShard(gate_up, gate_up_scale, down, down_scale)


def layer_router(tensors: Mapping[str, torch.Tensor], layer_id: int) -> RouterTensors:
    """Read the layer's router: BF16 weights, FP32 correction bias."""
    prefixes = layer_prefixes(layer_id)
    weight_name = _find(tensors, _prefixed(prefixes, ROLES["router_weight"]))
    bias_name = _find(tensors, _prefixed(prefixes, ROLES["router_bias"]))
    if isinstance(tensors, Checkpoint):
        tensors._require_float_metadata(weight_name)
        tensors._require_float_metadata(bias_name)
    weight, bias = tensors[weight_name], tensors[bias_name]
    if weight.dtype not in (torch.bfloat16, torch.float32) or bias.dtype not in (
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError("router weights and correction bias must be BF16 or FP32")
    if weight.shape != (FLASH.n_routed_experts, D) or bias.shape != (FLASH.n_routed_experts,):
        raise ValueError(f"router must cover all {FLASH.n_routed_experts} experts")
    return RouterTensors(
        weight=weight.to(torch.bfloat16).contiguous(), correction_bias=bias.to(torch.float32).contiguous(),
    )


def check_ffn_scales(tensors: Mapping[str, torch.Tensor], names: Sequence[str]) -> None:
    """Every quantized FFN tensor must carry a scale.

    The checkpoint's ``weight_offset`` is ignored (nothing consumes it), but a
    non-zero one would invalidate the symmetric-quantizer assumption, so it is
    checked rather than silently dropped.
    """
    for name in names:
        _scale_of(tensors, name)
        for suffix in OFFSET_SUFFIXES:
            offset = tensors.get(name + suffix)
            if offset is not None and bool(offset.any()):
                raise ValueError(f"{name}{suffix} is not all zero; symmetric dequant assumes it is")


class Checkpoint(Mapping[str, torch.Tensor]):
    """Read indexed W8A8 tensors and prepare the 45-layer text backbone on CPU.

    ``layer`` stages one rank's layer. ``iter_layers`` leaves layer placement and
    lifetime to the caller; ``model`` retains every prepared layer in host memory.
    Shared experts are complete replicas; routed experts use EP and dense FFNs use
    TP. Attention and vocabulary sharding use the explicitly supplied TP geometry.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.root}")
        config = self._json("config.json")
        self.config = config.get("text_config", config)
        self.description = self._json("quant_model_description.json")
        QuantDescription.parse(self.description).require_dynamic_symmetric()
        self._validate_config()
        index_name = "quant_model_weights.safetensors.index.json"
        self.index = self._json(index_name).get("weight_map")
        if not isinstance(self.index, dict) or not self.index:
            raise ValueError(f"{index_name} must contain a nonempty weight_map")
        for key, filename in self.index.items():
            if not isinstance(key, str) or not isinstance(filename, str):
                raise ValueError("checkpoint weight_map must map tensor names to shard filenames")
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".safetensors":
                raise ValueError(f"invalid checkpoint shard path for {key}: {filename}")
        self.used: set[str] = set()
        self.dequantized: set[str] = set()
        self._shapes: dict[str, tuple[int, ...]] = {}

    def _json(self, name: str) -> dict:
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint metadata is missing: {path}")
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
        if not isinstance(value, dict):
            raise ValueError(f"checkpoint metadata must be a JSON object: {path}")
        return value

    def _validate_config(self) -> None:
        fields = (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "first_k_dense_replace",
            "intermediate_size",
            "rms_norm_eps",
            "swiglu_limit",
            "tie_word_embeddings",
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "index_kpool",
            "index_kpool_compress",
            "index_kpool_always_select_tail",
            "n_routed_experts",
            "n_shared_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "scoring_func",
            "topk_method",
            "norm_topk_prob",
            "routed_scaling_factor",
            "n_group",
            "topk_group",
            "mhc",
            "hc_mult",
            "hc_sinkhorn_iters",
            "hc_eps",
        )
        for name in fields:
            expected = getattr(FLASH, name)
            if self.config.get(name) != expected:
                raise ValueError(f"checkpoint {name} must be {expected!r}, got {self.config.get(name)!r}")
        schedule = [FLASH.layer_config(i).attention_kind.value for i in range(FLASH.num_hidden_layers)]
        if self.config.get("layer_types") != schedule:
            raise ValueError("checkpoint layer_types must match the GLM-5.3-Flash KDA/MLA schedule")
        mlp_schedule = [FLASH.layer_config(i).mlp_kind.value for i in range(FLASH.num_hidden_layers)]
        if "mlp_layer_types" in self.config and self.config["mlp_layer_types"] != mlp_schedule:
            raise ValueError("checkpoint mlp_layer_types must match the GLM-5.3-Flash dense/MoE schedule")
        linear = self.config.get("linear_attn_config", {})
        linear_fields = {
            "num_heads": FLASH.linear_num_heads,
            "head_dim": FLASH.linear_head_dim,
            "short_conv_kernel_size": FLASH.linear_conv_kernel_dim,
            "gate_lower_bound": FLASH.linear_lower_bound,
        }
        for name, expected in linear_fields.items():
            if linear.get(name) != expected:
                raise ValueError(f"checkpoint linear_attn_config.{name} must be {expected!r}")

    def __contains__(self, key: object) -> bool:
        return key in self.index

    def __iter__(self) -> Iterator[str]:
        return iter(self.index)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.read(key)

    def _open(self, key: str):
        if key not in self.index:
            raise KeyError(f"tensor is absent from checkpoint index: {key}")
        path = (self.root / self.index[key]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard escapes its directory: {self.index[key]}")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint shard for {key} is missing: {path}")
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("checkpoint loading requires the safetensors package") from exc
        try:
            return safe_open(str(path), framework="pt", device="cpu")
        except OSError:
            # Surface native permission errors hidden by the safetensors reader.
            with path.open("rb"):
                pass
            raise

    def shape(self, key: str) -> tuple[int, ...]:
        """Read a tensor's dimensions from its shard header without loading its payload."""
        if key not in self._shapes:
            with self._open(key) as source:
                self._shapes[key] = tuple(source.get_slice(key).get_shape())
        return self._shapes[key]

    def read(self, key: str, selection=None) -> torch.Tensor:
        """Read only the requested tensor or slice; retain names rather than tensor copies."""
        with self._open(key) as source:
            value = source.get_tensor(key) if selection is None else source.get_slice(key)[selection]
        self.used.add(key)
        return value.contiguous()

    def vision_tensor(self, suffix: str, shape: Sequence[int]) -> torch.Tensor:
        """Load one unquantized vision tensor from either supported checkpoint root."""
        candidates = [root + suffix for root in ("model.visual.", "visual.")]
        names = [name for name in candidates if name in self]
        if len(names) != 1:
            raise ValueError(f"vision tensor must resolve exactly once: {suffix}")
        value = self.read(names[0])
        if tuple(value.shape) != tuple(shape) or value.dtype != torch.bfloat16:
            raise ValueError(f"vision {suffix} must have BF16 shape {tuple(shape)}")
        if not torch.isfinite(value).all():
            raise ValueError(f"vision tensor contains nonfinite values: {suffix}")
        return value

    def vision_weights(self) -> dict[str, torch.Tensor]:
        """Stage all 24 vision blocks and the spatial merger without quantization."""
        config = self._json("config.json").get("vision_config", {})
        required = {
            "depth": V.VISION_DEPTH,
            "hidden_size": V.VISION_HIDDEN,
            "num_heads": V.VISION_HEADS,
            "intermediate_size": V.VISION_INTER,
            "out_hidden_size": D,
            "patch_size": V.VISION_PATCH,
            "temporal_patch_size": V.VISION_TEMPORAL_PATCH,
            "spatial_merge_size": V.VISION_MERGE,
            "projection_intermediate_size": V.VISION_PROJ_INTER,
            "rms_norm_eps": V.VISION_RMS_NORM_EPS,
            "swiglu_limit": V.VISION_SWIGLU_LIMIT,
            "attention_bias": True,
        }
        for name, expected in required.items():
            if config.get(name) != expected:
                raise ValueError(f"vision_config.{name} must be {expected!r}")
        blocks = {
            "norm1_weight": "norm1.weight",
            "qkv_weight": "attn.qkv.weight",
            "qkv_bias": "attn.qkv.bias",
            "q_norm_weight": "attn.q_norm.weight",
            "k_norm_weight": "attn.k_norm.weight",
            "out_weight": "attn.proj.weight",
            "out_bias": "attn.proj.bias",
            "norm2_weight": "norm2.weight",
            "gate_weight": "mlp.gate_proj.weight",
            "gate_bias": "mlp.gate_proj.bias",
            "up_weight": "mlp.up_proj.weight",
            "up_bias": "mlp.up_proj.bias",
            "down_weight": "mlp.down_proj.weight",
            "down_bias": "mlp.down_proj.bias",
        }
        singles = {
            "patch_weight": "patch_embed.proj.weight",
            "patch_bias": "patch_embed.proj.bias",
            "post_norm_weight": "post_layernorm.weight",
            "downsample_weight": "downsample.weight",
            "downsample_bias": "downsample.bias",
            "merger_proj_weight": "merger.proj.weight",
            "merger_norm_weight": "merger.post_projection_norm.weight",
            "merger_norm_bias": "merger.post_projection_norm.bias",
            "merger_gate_weight": "merger.gate_proj.weight",
            "merger_up_weight": "merger.up_proj.weight",
            "merger_down_weight": "merger.down_proj.weight",
        }
        values = {}
        for name, shape in vision_weight_layouts().items():
            if name in blocks:
                values[name] = torch.stack(
                    [self.vision_tensor(f"blocks.{layer}.{blocks[name]}", shape[1:]) for layer in range(V.VISION_DEPTH)],
                )
            else:
                source_shape = shape
                if name == "patch_weight":
                    source_shape = (
                        V.VISION_HIDDEN,
                        V.VISION_IN_CHANNELS,
                        V.VISION_TEMPORAL_PATCH,
                        V.VISION_PATCH,
                        V.VISION_PATCH,
                    )
                elif name == "downsample_weight":
                    source_shape = (D, V.VISION_HIDDEN, V.VISION_MERGE, V.VISION_MERGE)
                values[name] = self.vision_tensor(singles[name], source_shape).reshape(shape)
        return values

    def layer_key(self, layer: int, suffix: str) -> str:
        """Resolve a tensor name in a text-backbone layer, excluding the MTP layer."""
        if not 0 <= layer < FLASH.num_hidden_layers:
            raise ValueError(f"layer must be in [0, {FLASH.num_hidden_layers}), got {layer}")
        candidates = [f"{root}{layer}.{suffix}" for root in TEXT_LAYER_ROOTS]
        if suffix.endswith("."):
            for root in TEXT_LAYER_ROOTS:
                if f"{root}{layer}.input_layernorm.weight" in self.index:
                    return f"{root}{layer}.{suffix}"
        return _find(self, candidates)

    def _require_shape(self, key: str, expected: tuple[int, ...]) -> None:
        actual = self.shape(key)
        if actual != expected:
            raise ValueError(f"{key} must have shape {expected}, got {actual}")

    def _require_float_metadata(self, key: str) -> None:
        kind = self.description.get(key)
        if kind != "FLOAT":
            raise ValueError(f"{key} must be declared FLOAT, got {kind!r}")

    def _float(
        self,
        key: str,
        shape: tuple[int, ...],
        selection=None,
        *,
        dtype: torch.dtype = torch.bfloat16,
        allow_fp32: bool = False,
    ) -> torch.Tensor:
        self._require_float_metadata(key)
        self._require_shape(key, shape)
        value = self.read(key, selection)
        allowed = (torch.bfloat16, torch.float32) if allow_fp32 else (torch.bfloat16,)
        if value.dtype not in allowed:
            raise ValueError(f"{key} must have dtype {allowed}, got {value.dtype}")
        return value.to(dtype).contiguous()

    def quant(
        self,
        key: str,
        rows: slice = slice(None),
        cols: slice = slice(None),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep INT8 values and FP32 per-output scales; reject asymmetric offsets."""
        if self.description.get(key) != "W8A8_DYNAMIC":
            raise ValueError(f"{key} is not declared W8A8_DYNAMIC in quant_model_description.json")
        shape = self.shape(key)
        if len(shape) != 2:
            raise ValueError(f"quantized weight must be a matrix: {key}, shape {shape}")
        scale_name = _find(self, [key + suffix for suffix in SCALE_SUFFIXES])
        offset_name = _find(self, [key + suffix for suffix in OFFSET_SUFFIXES])
        scale = self.read(scale_name)
        offset = self.read(offset_name)
        allowed_shapes = ((shape[0],), (shape[0], 1))
        if tuple(scale.shape) not in allowed_shapes or tuple(offset.shape) not in allowed_shapes:
            raise ValueError(f"{key} needs one scale and offset per output row")
        if not scale.is_floating_point() or not offset.is_floating_point():
            raise ValueError(f"{key} scales and offsets must be floating point")
        if offset.any() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError(f"{key} has nonzero offsets or nonpositive/nonfinite scales")
        weight = self.read(key, (rows, cols))
        scale = scale.reshape(-1)[rows].float().contiguous()
        if weight.dtype != torch.int8 or weight.ndim != 2 or scale.numel() != weight.shape[0]:
            raise ValueError(f"{key} must be INT8 with one scale per output row")
        return weight, scale

    def matrix(
        self,
        key: str,
        rows: slice = slice(None),
        cols: slice = slice(None),
        *,
        mode: str = "native",
    ) -> dict[str, torch.Tensor]:
        """Preserve native INT8/scales or explicitly dequantize a matrix into BF16."""
        if mode not in ("native", "bf16"):
            raise ValueError("matrix mode must be native or bf16")
        kind = self.description.get(key)
        if kind == "W8A8_DYNAMIC":
            weight, scale = self.quant(key, rows, cols)
            if mode == "native":
                return {"weight": weight, "scale": scale}
            self.dequantized.add(key)
            return {"weight": (weight.float() * scale[:, None]).bfloat16()}
        if kind != "FLOAT":
            raise ValueError(f"unsupported or missing quantization description for {key}: {kind!r}")
        weight = self.read(key, (rows, cols))
        if weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError(f"{key} must be a BF16 matrix, got {weight.dtype} {tuple(weight.shape)}")
        return {"weight": weight}

    @staticmethod
    def _partition(size: int, rank: int, world: int) -> slice:
        if world not in SUPPORTED_TP_SIZES or not 0 <= rank < world or size % world:
            raise ValueError(f"cannot partition {size} over rank {rank}/{world}")
        width = size // world
        return slice(rank * width, (rank + 1) * width)

    @staticmethod
    def _ffn_dict(shard: FfnShard) -> dict[str, torch.Tensor]:
        return {
            "w_gate_up": shard.gate_up,
            "w_gate_up_scale": shard.gate_up_scale,
            "w_down": shard.down,
            "w_down_scale": shard.down_scale,
        }

    def ffn(
        self,
        layer: int,
        *,
        tp_rank: int = 0,
        tp: int = 16,
        ep_rank: int = 0,
        ep: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Load dense TP channels or complete EP experts with a replicated shared FFN."""
        norm = self._float(self.layer_key(layer, "post_attention_layernorm.weight"), (D,))
        if layer < FLASH.first_k_dense_replace:
            return {"norm_weight": norm, **self._ffn_dict(shard_dense_mlp(self, layer, tp_rank, tp))}
        experts = self._partition(FLASH.n_routed_experts, ep_rank, ep)
        routed = shard_routed_experts(
            self, layer, ep_rank, ep, local_experts=experts.stop - experts.start,
            global_experts=FLASH.n_routed_experts,
        )
        shared = self._ffn_dict(load_shared_expert(self, layer))
        router = layer_router(self, layer)
        return {
            "norm_weight": norm,
            "gate_weight": router.weight,
            "correction_bias": router.correction_bias,
            "w_gate_up": routed.w_gate_up,
            "w_gate_up_scale": routed.w_gate_up_scale,
            "w_down": routed.w_down,
            "w_down_scale": routed.w_down_scale,
            **{"shared_" + name: value for name, value in shared.items()},
        }

    def mhc(self, layer: int, side: str) -> dict[str, torch.Tensor]:
        """Load the attention or FFN hyper-connection function, scales and base."""
        if side not in ("attn", "ffn"):
            raise ValueError("side must be attn or ffn")
        layout = (
            ("function", "fn", (FLASH.mix_hc, FLASH.hc_dim), torch.bfloat16),
            ("scale", "scale", (3,), torch.float32),
            ("base", "base", (FLASH.mix_hc,), torch.float32),
        )
        out = {}
        for name, suffix, shape, dtype in layout:
            key = self.layer_key(layer, f"{side}_hc.{suffix}")
            out[name] = self._float(key, shape, dtype=dtype, allow_fp32=True)
        return out

    def attention(
        self,
        layer: int,
        *,
        tp_rank: int = 0,
        tp: int = 16,
        mla_mode: str = "bf16",
    ) -> dict[str, torch.Tensor]:
        """Load KDA weights or MLA weights with explicit BF16/native quantization mode."""
        if mla_mode not in ("native", "bf16"):
            raise ValueError("mla_mode must be native or bf16")
        prefix = self.layer_key(layer, "self_attn.")
        if layer in FLASH.kda_layer_ids:
            heads = self._partition(FLASH.linear_num_heads, tp_rank, tp)
            width = FLASH.linear_head_dim
            full_width = FLASH.linear_num_heads * width
            rows = slice(heads.start * width, heads.stop * width)
            out = {}
            for name in ("q", "k", "v"):
                key = prefix + name + "_proj.weight"
                out["w_" + name] = self._float(key, (full_width, D), (rows, slice(None)))
            layout = (
                ("w_f_a", "forget_gate.f_a_proj.weight", (width, D), None, torch.bfloat16),
                (
                    "w_f_b",
                    "forget_gate.f_b_proj.weight",
                    (full_width, width),
                    (rows, slice(None)),
                    torch.bfloat16,
                ),
                ("dt_bias", "forget_gate.dt_bias", (full_width,), rows, torch.float32),
                ("a_log", "forget_gate.A_log", (FLASH.linear_num_heads,), heads, torch.float32),
                ("w_b", "b_proj.weight", (FLASH.linear_num_heads, D), (heads, slice(None)), torch.bfloat16),
                ("w_g_a", "g_a_proj.weight", (width, D), None, torch.bfloat16),
                ("w_g_b", "g_b_proj.weight", (full_width, width), (rows, slice(None)), torch.bfloat16),
                ("norm_weight", "o_norm.weight", (width,), None, torch.bfloat16),
                ("w_o", "o_proj.weight", (D, full_width), (slice(None), rows), torch.bfloat16),
            )
            for target, name, shape, selection, dtype in layout:
                allow_fp32 = dtype == torch.float32
                out[target] = self._float(prefix + name, shape, selection, dtype=dtype, allow_fp32=allow_fp32)
            conv_shape = (full_width, 1, FLASH.linear_conv_kernel_dim)
            conv = [self._float(prefix + q + "_conv1d.weight", conv_shape, (rows, 0)) for q in ("q", "k", "v")]
            out["weight"] = torch.cat(conv).T.contiguous()
            return out
        heads = self._partition(FLASH.num_attention_heads, tp_rank, tp)
        qdim, vdim, latent = FLASH.qk_nope_head_dim, FLASH.v_head_dim, FLASH.kv_lora_rank
        q_rows = slice(heads.start * qdim, heads.stop * qdim)
        o_cols = slice(heads.start * vdim, heads.stop * vdim)
        projections = (
            ("w_q_a", "q_a_proj", (FLASH.q_lora_rank, D), slice(None), slice(None)),
            ("w_q_b", "q_b_proj", (FLASH.num_attention_heads * qdim, FLASH.q_lora_rank), q_rows, slice(None)),
            ("w_kv_a", "kv_a_proj_with_mqa", (latent, D), slice(None), slice(None)),
            ("w_o", "o_proj", (D, FLASH.num_attention_heads * vdim), slice(None), o_cols),
        )
        out = {}
        for target, source, shape, rows, cols in projections:
            key = prefix + source + ".weight"
            self._require_shape(key, shape)
            pair = self.matrix(key, rows, cols, mode=mla_mode)
            out[target] = pair["weight"]
            if "scale" in pair:
                out[target + "_scale"] = pair["scale"]
        kv_rows = slice(heads.start * (qdim + vdim), heads.stop * (qdim + vdim))
        kv_shape = (FLASH.num_attention_heads * (qdim + vdim), latent)
        kv = self._float(prefix + "kv_b_proj.weight", kv_shape, (kv_rows, slice(None)))
        kv = kv.reshape(heads.stop - heads.start, qdim + vdim, latent)
        out.update(w_k=kv[:, :qdim].contiguous(), w_v=kv[:, qdim:].contiguous())
        out["q_a_norm_weight"] = self._float(prefix + "q_a_layernorm.weight", (FLASH.q_lora_rank,))
        out["kv_a_norm_weight"] = self._float(prefix + "kv_a_layernorm.weight", (latent,))
        if mla_mode == "bf16":
            out["w_o_absorbed"] = golden_absorb_output(out["w_v"], out["w_o"])
        return out

    def indexer(self, layer: int) -> dict[str, torch.Tensor]:
        """Load replicated Indexer projections, normalization and pooling parameters."""
        prefix = self.layer_key(layer, "self_attn.indexer.")
        if layer not in FLASH.dsa_layer_ids:
            raise ValueError(f"layer {layer} is a KDA layer and has no Indexer")
        ih, width = FLASH.index_n_heads, FLASH.index_head_dim
        projections = (
            ("w_q_b", "wq_b.weight", (ih * width, FLASH.q_lora_rank)),
            ("w_k", "wk.weight", (width, D)),
            ("w_weights", "weights_proj.weight", (ih, D)),
            ("w_compress_gate", "index_kpool_compress_gate", (width, D)),
        )
        out = {name: self._float(prefix + key, shape).T.contiguous() for name, key, shape in projections}
        vectors = (
            ("k_norm_weight", "k_norm.weight", (width,)),
            ("k_norm_bias", "k_norm.bias", (width,)),
            ("compress_ape", "index_kpool_compress_ape", (FLASH.index_kpool, width)),
        )
        out.update({name: self._float(prefix + key, shape) for name, key, shape in vectors})
        out["hadamard"] = sylvester_hadamard(width)
        return out

    def layer(
        self,
        layer: int,
        *,
        tp_rank: int = 0,
        tp: int = 16,
        ep_rank: int = 0,
        ep: int = 16,
        mla_mode: str = "bf16",
    ) -> dict:
        """Prepare one rank's complete backbone layer for prefill and decode."""
        self._partition(FLASH.num_attention_heads, tp_rank, tp)
        self._partition(FLASH.n_routed_experts, ep_rank, ep)
        if ep % tp:
            raise ValueError(f"complete-layer EP size {ep} must be divisible by TP size {tp}")
        out = {
            "input_norm": self._float(self.layer_key(layer, "input_layernorm.weight"), (D,)),
            "attn_hc": self.mhc(layer, "attn"),
            "ffn_hc": self.mhc(layer, "ffn"),
            "attention": self.attention(layer, tp_rank=tp_rank, tp=tp, mla_mode=mla_mode),
            "ffn": self.ffn(layer, tp_rank=tp_rank, tp=tp, ep_rank=ep_rank, ep=ep),
        }
        if layer in FLASH.dsa_layer_ids:
            out["indexer"] = self.indexer(layer)
        return out

    def _vocab_key(self, suffix: str) -> str:
        return _find(self, [f"{root.removesuffix('layers.')}{suffix}" for root in TEXT_LAYER_ROOTS])

    def embedding(self, token_ids) -> torch.Tensor:
        """Look up selected token rows without materializing the complete embedding table."""
        ids = torch.as_tensor(token_ids)
        if ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("token ids must be integers")
        ids = ids.flatten().tolist()
        if not ids or min(ids) < 0 or max(ids) >= FLASH.vocab_size:
            raise ValueError("token ids must be nonempty and within the vocabulary")
        key = self._vocab_key("embed_tokens.weight")
        shape = (FLASH.vocab_size, D)
        unique = {i: self._float(key, shape, (slice(i, i + 1), slice(None))) for i in set(ids)}
        return torch.cat([unique[i] for i in ids])

    def vocabulary(self, tp_rank: int = 0, tp: int = 16) -> dict:
        """Prepare vocabulary TP shards and the complete final RMSNorm vector."""
        rows = self._partition(FLASH.vocab_size, tp_rank, tp)
        shape = (FLASH.vocab_size, D)
        return {
            "vocab_start": rows.start,
            "vocab_stop": rows.stop,
            "embedding": self._float(self._vocab_key("embed_tokens.weight"), shape, (rows, slice(None))),
            "lm_head": self._float("lm_head.weight", shape, (rows, slice(None))),
            "final_norm": self._float(self._vocab_key("norm.weight"), (D,)),
        }

    def rotation(self) -> torch.Tensor:
        """Expose the saved QuaRot artifact without applying an additional rotation."""
        return self.read("rot.weight")

    def iter_layers(
        self,
        *,
        tp_rank: int,
        tp: int,
        ep_rank: int,
        ep: int,
        mla_mode: str = "bf16",
    ) -> Iterator[tuple[int, dict]]:
        """Yield backbone layers in order; the caller owns placement and tensor lifetime."""
        for layer_id in range(FLASH.num_hidden_layers):
            yield (
                layer_id,
                self.layer(layer_id, tp_rank=tp_rank, tp=tp, ep_rank=ep_rank, ep=ep, mla_mode=mla_mode),
            )

    def model(self, *, tp_rank: int, tp: int, ep_rank: int, ep: int, mla_mode: str = "bf16") -> dict:
        """Materialize all 45 layers and vocabulary weights for one rank in CPU memory."""
        layers = dict(self.iter_layers(tp_rank=tp_rank, tp=tp, ep_rank=ep_rank, ep=ep, mla_mode=mla_mode))
        return {"layers": layers, **self.vocabulary(tp_rank, tp)}


def _resolve_name(checkpoint: Checkpoint, layer: int, sub_prefix: str, suffixes: Sequence[str]) -> str:
    """First present checkpoint spelling of one tensor (both layer roots)."""
    for root in TEXT_LAYER_ROOTS:
        for suffix in suffixes:
            name = f"{root}{layer}.{sub_prefix}{suffix}"
            if name in checkpoint:
                return name
    wanted = [f"{root}{layer}.{sub_prefix}{suffixes[0]}" for root in TEXT_LAYER_ROOTS]
    raise KeyError(
        f"layer {layer} tensor {sub_prefix}{suffixes[0]!r} is missing; tried {wanted}. "
        "Absent shards mean the checkpoint download is incomplete."
    )


def _resolve_top_name(checkpoint: Checkpoint, suffixes: Sequence[str]) -> str:
    """First present spelling of a top-level (non-layer) tensor, bare name included."""
    for root in ("model.language_model.", "model.", ""):
        for suffix in suffixes:
            name = f"{root}{suffix}"
            if name in checkpoint:
                return name
    raise KeyError(f"top-level tensor {suffixes[0]!r} is missing; tried {suffixes}")


def _quant_sidecar(checkpoint: Checkpoint, name: str, suffixes: Sequence[str]) -> str | None:
    """Resolve a quant sidecar (scale/offset) for a tensor, both stem spellings."""
    stems = [name]
    if name.endswith(".weight"):
        stems.append(name[: -len(".weight")])
    for stem in stems:
        for suffix in suffixes:
            sidecar = stem + suffix
            if sidecar in checkpoint:
                return sidecar
    return None


def read_reference_float(checkpoint: Checkpoint, name: str) -> torch.Tensor:
    """Read one tensor as FP32: FLOAT casts, W8A8_DYNAMIC dequantizes.

    Dequant is ``int8 * per-output-channel scale`` with the all-zero offset
    verified at the checkpoint reader's quantization boundary.
    """
    label = checkpoint.description.get(name)
    raw = checkpoint.read(name)
    if label == "FLOAT":
        return raw.to(torch.float32)
    if label == "W8A8_DYNAMIC":
        scale_name = _quant_sidecar(checkpoint, name, _SCALE_SUFFIXES)
        if scale_name is None:
            raise KeyError(f"{name} is W8A8_DYNAMIC but has no weight scale")
        scale = checkpoint.read(scale_name).to(torch.float32).reshape(-1)
        weight = raw.to(torch.float32)
        if scale.numel() != weight.shape[0]:
            raise ValueError(f"{name} scale has {scale.numel()} channels for {weight.shape[0]} rows")
        offset_name = _quant_sidecar(checkpoint, name, _OFFSET_SUFFIXES)
        if offset_name is not None:
            offset = checkpoint.read(offset_name)
            if bool(offset.any()):
                raise ValueError(f"{name} offset is not all zero; symmetric dequant is invalid")
        return weight * scale.reshape(-1, *([1] * (weight.dim() - 1)))
    raise ValueError(f"{name} has unsupported quantization label {label!r}")


def split_kv_b(
    w_kv_b: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    *,
    qk_dim: int = QK_DIM,
    v_dim: int = V_DIM,
    kv_lora: int = KV_LORA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split checkpoint KV projection rows into key/value weights for one TP rank."""
    heads = w_kv_b.shape[0] // (qk_dim + v_dim) if w_kv_b.shape[0] % (qk_dim + v_dim) == 0 else -1
    if w_kv_b.dim() != 2 or heads <= 0:
        raise ValueError(f"kv_b_proj must be 2-D with rows a multiple of QK+V, got {tuple(w_kv_b.shape)}")
    local_h = heads // tp_size
    reshaped = w_kv_b.reshape(heads, qk_dim + v_dim, kv_lora)
    key, value = reshaped[:, :qk_dim], reshaped[:, qk_dim:]
    start = tp_rank * local_h
    return key[start : start + local_h].contiguous(), value[start : start + local_h].contiguous()


def gather_rows(checkpoint, name: str, rows: torch.Tensor) -> torch.Tensor:
    """Read selected rows of one tensor through the checkpoint read protocol.

    ``Checkpoint.read`` selects through the shard's slice view, which takes
    slices rather than an index tensor, so the rows are read one at a time and
    concatenated (deduplicated, in ``token_ids`` order). Fixture stand-ins may
    gather directly.
    """
    ids = torch.as_tensor(rows, device="cpu")
    flat = ids.reshape(-1)
    gather = getattr(checkpoint, "gather", None)
    if gather is not None:
        gathered = gather(name, flat)
    else:
        unique = {
            int(row): checkpoint.read(name, slice(row, row + 1))
            for row in sorted({int(r) for r in flat.tolist()})
        }
        gathered = torch.cat([unique[int(row)] for row in flat.tolist()], dim=0)
    return gathered.reshape(*ids.shape, *gathered.shape[1:])
