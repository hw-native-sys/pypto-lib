# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Real-weight vLLM fixture helpers adapted from the validated Ascend reproducer."""

import copy
import tempfile
from contextlib import contextmanager


@contextmanager
def native_session(config, device_index: int):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_ascend.ascend_config import init_ascend_config

    torch.npu.set_device(device_index)
    with tempfile.TemporaryDirectory(prefix="csa_tp1_") as temporary, set_current_vllm_config(config):
        init_ascend_config(config)
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=device_index,
                distributed_init_method=f"file://{temporary}/distributed",
                backend="hccl",
            )
            initialize_model_parallel(backend="hccl")
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def make_cache_groups(config, device, attention):
    """Use the Native owner, specification and builder for each cache group."""
    from vllm_ascend.attention.dsa_v1 import AscendDSAMetadataBuilder

    owners = {
        "swa": attention.dsa_attn.swa_cache_layer,
        "compressed": attention.dsa_attn.dsa_attn,
        "state": attention.compressor.state_cache,
    }
    if attention.compress_ratio == 4:
        owners.update(
            indexer=attention.indexer.k_cache, indexer_state=attention.indexer.compressor.state_cache
        )
    groups = {}
    for kind, owner in owners.items():
        prefix = owner.layer_name if kind == "compressed" else owner.prefix
        spec = owner.get_kv_cache_spec(config)
        builder = AscendDSAMetadataBuilder(spec, [prefix], config, device)
        groups[kind] = {"owner": owner, "prefix": prefix, "spec": spec, "builder": builder}
    return groups


def allocate_native_cache(group, pages, device):
    """Build Native runner page views over a guarded allocation."""
    import torch

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    spec = group["spec"]
    backend = group["owner"].get_attn_backend()
    page_size = spec.block_size
    shape = backend.get_kv_cache_shape(pages, page_size, spec.num_kv_heads, spec.head_size)
    shapes, dtypes = [shape], [spec.dtype]
    if getattr(spec, "scale_dim", 0):
        shapes.append(backend.get_kv_cache_shape(pages, page_size, spec.num_kv_heads, spec.scale_dim))
        dtypes.append(spec.scale_dtype)
    guard_bytes = 128
    allocation = torch.full(
        (pages * spec.page_size_bytes + 2 * guard_bytes,), 37, dtype=torch.int8, device=device
    )
    raw = allocation[guard_bytes:-guard_bytes]
    views = NPUModelRunner._adjust_kv_layout(None, raw, shapes, dtypes, spec.page_size_bytes)
    group.update(allocation=allocation, raw=raw, views=views, page_size=page_size, pages=pages)
    return {
        "spec": repr(spec),
        "logical_block_size": spec.block_size,
        "storage_block_size": page_size,
        "page_bytes": spec.page_size_bytes,
        "raw_offset": raw.storage_offset(),
        "views": [
            {
                "shape": list(view.shape),
                "dtype": str(view.dtype),
                "stride": list(view.stride()),
                "storage_offset_elements": view.storage_offset(),
                "storage_offset_bytes": view.storage_offset() * view.element_size(),
                "contiguous": view.is_contiguous(),
                "storage_ptr": view.untyped_storage().data_ptr(),
            }
            for view in views
        ],
    }


def writable_bytes(allocation, views, slots):
    """Permit writes only to Native slots; preserve page padding and allocation guards."""
    import torch

    allowed = torch.zeros(allocation.numel(), dtype=torch.bool)
    for page, row in slots.detach().cpu().tolist():
        if page < 0 or row < 0:
            continue
        for view in views:
            if page >= view.shape[0] or row >= view.shape[1]:
                raise ValueError(f"Slot outside Native page view: {page}, {row}, {view.shape}")
            begin = (
                view.storage_offset() + page * view.stride(0) + row * view.stride(1)
            ) * view.element_size()
            end = begin + view.shape[-1] * view.element_size()
            if end > allowed.numel():
                raise ValueError("Slot write exceeds the guarded allocation")
            allowed[begin:end] = True
    return allowed


def make_fixture(
    config,
    attention,
    batch,
    history,
    seed,
    device,
    table_history=None,
    reverse_pages=False,
    *,
    query_tokens=6,
    indexer_page_pattern="original",
    query_lengths=None,
    history_lengths=None,
    token_capacity=None,
    request_capacity=None,
    page_owners=None,
    physical_request_capacity=None,
    allow_unassigned_tail=False,
):
    """Construct Native decode metadata from actual per-request lengths.

    DSpark keeps five configured speculative tokens while individual requests
    may contain fewer actual tokens. Page permutations are applied before the
    Native builder, compact slots and read-only snapshots are constructed.
    """
    import torch

    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
    from vllm_ascend.worker.block_table import BlockTable

    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    if type(query_tokens) is not int or not 1 <= query_tokens <= 6:
        raise ValueError("This fixture supports 1..6 actual query tokens with DSpark=5")
    queries = [query_tokens] * batch if query_lengths is None else list(query_lengths)
    histories = [history] * batch if history_lengths is None else list(history_lengths)
    if len(queries) != batch or any(type(value) is not int or not 1 <= value <= 6 for value in queries):
        raise ValueError("query_lengths must provide 1..6 query tokens for each request")
    if len(histories) != batch or any(type(value) is not int or value < 0 for value in histories):
        raise ValueError("history_lengths must provide a nonnegative history for each request")
    if batch <= 0:
        raise ValueError("This fixture requires nonempty requests; empty calls use serving fallback tests")
    query_tokens = max(queries)
    max_history = max(histories)
    if indexer_page_pattern not in ("original", "ascending_history", "fragmented_history"):
        raise ValueError("Unknown Indexer history page pattern")
    actual_tokens = sum(queries)
    tokens = actual_tokens if token_capacity is None else token_capacity
    requests = batch if request_capacity is None else request_capacity
    physical_requests = requests if physical_request_capacity is None else physical_request_capacity
    if type(tokens) is not int or tokens < actual_tokens or type(requests) is not int or requests < batch:
        raise ValueError("TND fixture capacity cannot be smaller than active token/request counts")
    if type(physical_requests) is not int or physical_requests < requests:
        raise ValueError("Physical request capacity must cover metadata request capacity")
    if page_owners is not None and (
        len(page_owners) != batch
        or len(set(page_owners)) != batch
        or any(type(owner) is not int or not 0 <= owner < physical_requests for owner in page_owners)
    ):
        raise ValueError("page_owners must contain valid, distinct physical request slots")
    padded_queries = queries + [0] * (requests - batch)
    remaining = tokens - actual_tokens
    for request in range(batch, requests):
        padded_queries[request] = min(6, remaining)
        remaining -= padded_queries[request]
    if remaining and not allow_unassigned_tail:
        raise ValueError("Decode padding needs enough inactive request slots, with at most six rows each")
    bounds_cpu = torch.zeros(requests + 1, dtype=torch.int32)
    bounds_cpu[1:] = torch.tensor(padded_queries, dtype=torch.int32).cumsum(0)
    lengths_cpu = torch.tensor(
        [h + q for h, q in zip(histories, queries)] + [0] * (requests - batch),
        dtype=torch.int32,
    )
    bounds, lengths = bounds_cpu.to(device), lengths_cpu.to(device)
    active_positions = torch.cat([h + torch.arange(q) for h, q in zip(histories, queries)])
    positions = torch.zeros(tokens, device=device, dtype=torch.int64)
    positions[:actual_tokens].copy_(active_positions)
    hidden = torch.randn((tokens, 4, 4096), device=device, dtype=torch.bfloat16)
    groups = make_cache_groups(config, device, attention)
    metadata, common_cache, prefill_cache, decode_cache = {}, {}, {}, {}
    for name, group in groups.items():
        spec = group["spec"]
        ratio = getattr(spec, "compress_ratio", 1)
        # A fixed table_history preserves page-table shape across metadata updates.
        width_history = max_history if table_history is None else table_history
        if width_history < max_history:
            raise ValueError("table_history cannot be smaller than the maximum actual history")
        width_query = 6 if table_history is not None else query_tokens
        columns = (width_history + width_query + spec.block_size * ratio - 1) // (spec.block_size * ratio) + 1
        # Keep full logical tables but ring-map recent SWA/state rows to owned pages.
        per_request = columns if name in ("compressed", "indexer") else 9
        if name == "state" and attention.compress_ratio == 128:
            per_request = 18
        pages = physical_requests * per_request + 1
        group["layout"] = allocate_native_cache(group, pages, device)
        for i, view in enumerate(group["views"]):
            if view.dtype == torch.int8:
                view.random_(-64, 64)
            elif name == "indexer" and i == 1:
                view.fill_(0.01)
            else:
                view.normal_(0, 0.5)
        group["owner"].kv_cache = [group["views"]] if name == "indexer" else group["views"]
        # Preserve the original minimum capacities, expanding for larger cases.
        table = BlockTable(
            spec.block_size,
            max(40, requests),
            columns,
            max(256, tokens),
            True,
            device,
            num_speculative_tokens=5,
        )
        for row in range(requests):
            if row >= batch:
                table.add_row([0] * columns, row)
                continue
            owner = page_owners[row] if page_owners is not None else batch - 1 - row if reverse_pages else row
            page_ids = [
                1 + owner * per_request + (per_request - 1 - col) % per_request for col in range(columns)
            ]
            if name == "indexer" and indexer_page_pattern != "original":
                full_history_pages = histories[row] // (spec.block_size * ratio)
                if full_history_pages < 4:
                    raise ValueError("Page permutation cases need at least four complete historical pages")
                current_suffix = page_ids[full_history_pages:]
                if indexer_page_pattern == "ascending_history":
                    page_ids[:full_history_pages] = reversed(page_ids[:full_history_pages])
                else:
                    page_ids[1], page_ids[2] = page_ids[2], page_ids[1]
                assert page_ids[full_history_pages:] == current_suffix
            table.add_row(page_ids, row)
        table.commit_block_table(requests)
        table.compute_slot_mapping(requests, bounds, positions // ratio)
        if tokens > actual_tokens:
            table.slot_mapping.gpu[actual_tokens:tokens].fill_(-1)
        common = AscendCommonAttentionMetadata(
            query_start_loc=bounds,
            query_start_loc_cpu=bounds_cpu,
            seq_lens=lengths,
            _seq_lens_cpu=lengths_cpu,
            seq_lens_cpu=lengths_cpu,
            num_reqs=requests,
            num_actual_tokens=actual_tokens,
            num_input_tokens=tokens,
            max_query_len=max(padded_queries),
            max_seq_len=int(lengths_cpu.max()),
            block_table_tensor=table.block_table.gpu[:requests],
            slot_mapping=table.slot_mapping.gpu[:tokens],
            positions=positions,
            attn_state=AscendAttentionState.SpecDecoding,
            causal=True,
        )
        metadata[group["prefix"]] = group["builder"].build(
            0,
            common,
            num_reqs_actual=batch,
            block_size=spec.block_size,
            common_ratio_to_sas_metadata=common_cache,
            prefill_ratio_to_sas_metadata=prefill_cache,
            decode_ratio_to_sas_metadata=decode_cache,
        )
        group.update(table=table, common=common)
    compact = {
        name: attention.dsa_attn.dsa_attn.impl._compute_compressor_metadata(
            metadata[groups[name]["prefix"]].decode
        )
        for name in ("compressed", "indexer")
        if name in groups
    }
    for name, group in groups.items():
        slots = compact[name][2] if name in compact else metadata[group["prefix"]].decode.slot_mapping
        group["allowed"] = writable_bytes(group["allocation"], group["views"], slots)
        group["initial"] = group["allocation"].cpu()
    readonly = {"positions": positions}
    for name, group in groups.items():
        req = metadata[group["prefix"]].decode
        for field in ("query_start_loc", "seq_lens", "block_table", "slot_mapping", "start_pos"):
            value = getattr(req, field)
            if value is not None:
                readonly[f"{name}.{field}"] = value
    readonly.update({f"{name}.compact_slots": value[2] for name, value in compact.items()})
    readonly = {name: (value, value.cpu()) for name, value in readonly.items()}
    return {
        "groups": groups,
        "metadata": metadata,
        "compact": compact,
        "positions": positions,
        "readonly": readonly,
        "hidden": hidden,
        "tokens": tokens,
        "actual_tokens": actual_tokens,
        "request_capacity": requests,
        "query_tokens": query_tokens,
        "query_lengths": queries,
        "history_lengths": histories,
        "table_history": table_history,
    }


def collect_state(fixture, output, topk):
    actual_tokens = fixture.get("actual_tokens", fixture["tokens"])
    state = {
        "x_out": output[:actual_tokens].detach().cpu(),
        "idx_topk": topk.detach().cpu().reshape(-1, 512)[:actual_tokens],
    }
    for name, group in fixture["groups"].items():
        for index, view in enumerate(group["views"]):
            state[f"{name}.{index}"] = view.detach().cpu()
    return state


def restore(fixture):
    for group in fixture["groups"].values():
        group["allocation"].copy_(group["initial"])


def runtime_module_name(name):
    if name == "head":
        return "lm_head"
    if name.startswith("layers."):
        name = "model." + name
    if name == "embed":
        return "model.embed_tokens"
    # Match DeepseekV4ForCausalLM.load_weights, including module-end names.
    parts = name.split(".")
    replacements = {
        "attn": "self_attn",
        "ffn": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "attn_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
    }
    return ".".join(replacements.get(part, part) for part in parts)


class DeepseekCompressedOverrides:
    """Picklable transform, also propagated to the DSpark draft ModelConfig."""

    def __init__(self, target_architecture=None):
        self.target_architecture = target_architecture

    def __call__(self, config):
        # vLLM first calls overrides on a minimal dummy config to probe model_type.
        if str(config.model_type).startswith("dummy_"):
            return config
        quant = copy.deepcopy(getattr(config, "quantization_config", None))
        if not quant or quant.get("quant_method") != "compressed-tensors":
            raise ValueError("This local override requires compressed-tensors")
        architectures = getattr(config, "architectures", ())
        v4_mtp = config.model_type == "deepseek_mtp" and "DeepSeekV4MTPModel" in architectures
        if not str(config.model_type).startswith("deepseek_v4") and not v4_mtp:
            raise ValueError(
                f"This local override is limited to DeepSeek V4: {config.model_type}/{architectures}"
            )
        for group in quant["config_groups"].values():
            if group.get("targets") != ["Linear"]:
                raise ValueError("Only the audited Linear-target checkpoint is supported")
        quant["ignore"] = [runtime_module_name(name) for name in quant.get("ignore", [])]
        config.quantization_config = quant
        config.sliding_window = 128
        draft = any("dspark" in name.lower() or "mtp" in name.lower() for name in architectures)
        if self.target_architecture and not draft:
            config.architectures = [self.target_architecture]
        return config
