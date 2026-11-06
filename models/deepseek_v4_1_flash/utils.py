# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Metadata lowering and shared validation helpers for prefill and decode."""

from collections.abc import Callable

import torch


import config as C
from config import BLOCK_SIZE, FLASH as M, TP_SIZE
from golden import ScalarSpec


def parse_decode_startpos(value):
    """Parse one absolute starting position per request."""
    import argparse

    try:
        positions = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("start-pos must be a comma-separated list of integers") from error
    limit = min(C.MAX_BATCH_PER_DP, C.DECODE_MAX_TOKENS)
    if not 1 <= len(positions) <= limit:
        raise argparse.ArgumentTypeError(f"start-pos must contain between 1 and {limit} positions")
    max_start = M.max_position_embeddings - 1
    if any(position < 0 or position > max_start for position in positions):
        raise argparse.ArgumentTypeError(f"positions must be in [0, {max_start}]")
    return positions


def parse_decode_query_lens(value):
    """Parse positive query lengths in request order."""
    import argparse

    try:
        lengths = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("query-lens must be a comma-separated list of integers") from error
    if any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("query-lens must contain positive lengths")
    return lengths


def configure_decode_workload(args, parser):
    """Resolve query lengths, packed token count, and absolute position bounds."""
    if args.startpos is None:
        if args.query_lens is not None:
            parser.error("--query-lens requires --start-pos")
        return
    args.requests = len(args.startpos)
    if args.query_lens is None:
        args.query_lens = [C.DECODE_ROWS_PER_REQUEST] * args.requests
    if len(args.query_lens) != args.requests:
        parser.error("--query-lens must contain one length per --start-pos value")
    args.tokens = sum(args.query_lens)
    if args.tokens > C.DECODE_MAX_TOKENS:
        parser.error(f"total query tokens must not exceed {C.DECODE_MAX_TOKENS}")
    ends = [pos + length for pos, length in zip(args.startpos, args.query_lens)]
    if max(ends) > M.max_position_embeddings:
        parser.error("a query block exceeds max_position_embeddings")


def packed_query_lengths(tokens, requests, query_lens=None):
    """Use explicit request lengths or a balanced packed token split."""
    if query_lens is None:
        return [tokens // requests + (index < tokens % requests) for index in range(requests)]
    if len(query_lens) != requests or sum(query_lens) != tokens or any(length <= 0 for length in query_lens):
        raise ValueError("query_lens must contain one positive length per request and sum to tokens")
    return list(query_lens)


def validate_request_id_tensors(host_token_to_req_indices, device_token_to_req_indices, num_tokens):
    """Check CPU snapshots of the active Host/Device request mapping in debug or golden validation."""
    for name, value in (("host_token_to_req_indices", host_token_to_req_indices), ("device_token_to_req_indices", device_token_to_req_indices)):
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise ValueError(f"{name} validation requires a CPU snapshot; device readback must be explicit")
        if value.dtype != torch.int32 or value.ndim != 1:
            raise ValueError(f"{name} must be a rank-1 INT32 tensor")
    if host_token_to_req_indices.shape != device_token_to_req_indices.shape:
        raise ValueError("Host/Device request mapping capacities must match")
    if not 0 <= num_tokens <= host_token_to_req_indices.numel():
        raise ValueError("num_tokens must fit the request mapping capacity")
    if not torch.equal(host_token_to_req_indices[:num_tokens], device_token_to_req_indices[:num_tokens]):
        raise ValueError("Host/Device request mappings disagree in the active token range")


def paged_slots(
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int = BLOCK_SIZE,
    logical_divisor: int = 1,
    publish_only_complete: bool = False,
) -> torch.Tensor:
    """Map logical token positions to flattened physical cache rows."""
    if positions.ndim != 1 or token_to_req_indices.shape != positions.shape or block_table.ndim != 2:
        raise ValueError("positions/token_to_req_indices must be matching vectors and block_table must be a matrix")
    if storage_block_size <= 0 or logical_divisor <= 0:
        raise ValueError("storage_block_size and logical_divisor must be positive")
    if any(value.dtype not in (torch.int32, torch.int64) for value in (positions, token_to_req_indices, block_table)):
        raise ValueError("positions, token_to_req_indices and block_table must contain integer indices")
    publish = torch.ones_like(positions, dtype=torch.bool)
    if publish_only_complete and logical_divisor > 1:
        publish = (positions + 1).remainder(logical_divisor) == 0
    slots = torch.full_like(positions, -1, dtype=torch.int64)
    logical = torch.div(positions[publish], logical_divisor, rounding_mode="floor")
    logical_block = torch.div(logical, storage_block_size, rounding_mode="floor")
    offset = logical.remainder(storage_block_size)
    active_requests = token_to_req_indices[publish]
    if bool(((active_requests < 0) | (active_requests >= block_table.shape[0])).any()):
        raise ValueError("published request id is outside the block table")
    if bool(((logical_block < 0) | (logical_block >= block_table.shape[1])).any()):
        raise ValueError("block table does not cover a required logical page")
    physical = block_table[token_to_req_indices[publish].to(torch.long), logical_block.to(torch.long)]
    if bool((physical < 0).any()):
        raise ValueError("required logical page has no physical allocation")
    slots[publish] = physical.to(torch.int64) * storage_block_size + offset.to(torch.int64)
    return slots


def window_metadata(
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    window: int = M.sliding_window,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged write slots plus causal indices for the visible sliding window."""
    slots = paged_slots(positions, token_to_req_indices, block_table)
    lens = torch.minimum(positions + 1, torch.full_like(positions, window)).to(torch.int32)
    offsets = torch.arange(window, device=positions.device)
    starts = positions - lens.to(positions.dtype) + 1
    visible = starts.unsqueeze(-1) + offsets
    valid = offsets.unsqueeze(0) < lens.unsqueeze(-1)
    indices = torch.full_like(visible, -1, dtype=torch.int64)
    visible_requests = token_to_req_indices.unsqueeze(-1).expand_as(visible)
    indices[valid] = paged_slots(visible[valid], visible_requests[valid], block_table)
    return slots, indices.to(torch.int32), lens


def upload_forward_inputs(runtime, specs, parameters, ranks, host_inputs, upload_buffer):
    """Upload rank-local storage through a shared buffer registered before runtime preparation."""
    from pypto.ir import ParamDirection
    from pypto.runtime import StackedDeviceTensor

    chunk_bytes = upload_buffer.numel()
    arguments = []
    for spec, parameter in zip(specs, parameters, strict=True):
        if isinstance(spec, ScalarSpec):
            arguments.append(spec.to_python())
            continue
        if spec.name in host_inputs:
            arguments.append(host_inputs[spec.name])
            continue
        shape = spec.shape[1:]
        if spec.shape[0] != ranks:
            raise ValueError(f"{spec.name} needs a leading rank axis of size {ranks}")
        shards = []
        for rank in range(ranks):
            handle = runtime.alloc_tensor(shape, spec.dtype, worker_id=rank)
            shards.append(handle)
            if parameter.direction == ParamDirection.Out:
                continue
            tensor_value = isinstance(spec.init_value, torch.Tensor)
            if tensor_value:
                if list(spec.init_value.shape) != spec.shape:
                    raise ValueError(f"{spec.name} initialization must match its tensor spec")
                value = spec.init_value[rank].contiguous().view(torch.uint8).reshape(-1)
            else:
                fill = 0 if spec.init_value is None else spec.init_value
                active_buffer = upload_buffer[:min(chunk_bytes, handle.nbytes)]
                if spec.dtype == torch.float8_e8m0fnu:
                    code = torch.tensor(float(fill)).to(spec.dtype).view(torch.uint8).item()
                    active_buffer.fill_(code)
                else:
                    active_buffer.view(spec.dtype).fill_(fill)
            for offset in range(0, handle.nbytes, chunk_bytes):
                size = min(chunk_bytes, handle.nbytes - offset)
                if tensor_value:
                    upload_buffer[:size].copy_(value[offset:offset + size])
                runtime.copy_to(handle.data_ptr, upload_buffer.data_ptr(), size, dst_offset=offset, worker_id=rank)
        arguments.append(StackedDeviceTensor(shards, tuple(spec.shape), tuple(range(ranks))))
    return arguments


def token_shards(value, local_tokens):
    """Pad scheduled rows and split them in contiguous TP rank order."""
    padded = value.new_zeros(TP_SIZE * local_tokens, *value.shape[1:])
    padded[:value.shape[0]].copy_(value)
    return padded.reshape(TP_SIZE, local_tokens, *value.shape[1:])


def _copy_cache_rows(destination: torch.Tensor, source: torch.Tensor, rows: torch.Tensor) -> None:
    destination_rows = destination.flatten(0, 1).view(torch.uint8)
    source_rows = source.contiguous().view(torch.uint8).reshape_as(destination_rows[rows])
    destination_rows[rows] = source_rows


def _publish_quantized_cache(
    payload: torch.Tensor,
    scale: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and replace only mapped rows in a packed cache pair."""
    updated_payload = payload.clone()
    updated_scale = scale.clone()
    valid = slots >= 0
    rows = slots[valid].to(torch.int64)
    if rows.numel():
        row_payload, row_scale = quantize(values[valid])
        _copy_cache_rows(updated_payload, row_payload, rows)
        _copy_cache_rows(updated_scale, row_scale, rows)
    return updated_payload, updated_scale


DECODE_CACHE_NAMES = {
    "window_cache_pool": "window_cache", "window_cache_scale_pool": "window_scales",
    "compressed_cache_pool": "compressed_cache", "compressed_cache_scale_pool": "compressed_scales",
    "index_cache_pool": "index_cache", "index_cache_scale_pool": "index_scales",
    "state_cache_pool": "state_cache",
}


def decode_cache_views(cache_state):
    """Alias contiguous prefill storage as decode pools by flattening source/page axes."""
    return {target: cache_state[source].flatten(0, 1) for target, source in DECODE_CACHE_NAMES.items()}


def prepare_decode_inputs(values):
    """Lower one request's current prefill metadata to the TP1 decode input contract."""
    embedding = values["embedding"].float()
    result = {
        "x_hc": embedding[:, None, :].expand(-1, M.hc_mult, -1).contiguous(),
        "pre_mix": values["initial_pre"], "engram_lookup": values["engram_lookup"],
        "logit_row_indices": values["logit_rows"],
    }
    # Prefill metadata has encoder/decoder stages; decode starts from encoder rows.
    for name in ("window_slots", "window_indices", "token_to_req_indices", "position_ids", "query_start_loc"):
        result[name] = values[name][0].contiguous()
    for name in ("freqs_cos", "freqs_sin", "c2a_compressed_slots", "c1a_compressed_slots",
                 "c2a_index_block_table", "c1a_index_block_table", "state_block_table"):
        result[name] = values[name]
    return result
