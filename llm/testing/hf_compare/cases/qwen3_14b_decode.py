# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comparison case: Qwen3-14B decode (single layer) vs Hugging Face Qwen3DecoderLayer.

Port of ``examples/models/qwen3/14b/compare_with_hf.py`` onto the
``llm.testing.hf_compare`` framework. Run via:

    python -m llm.testing.hf_compare run qwen3_14b.decode \\
        -k hf_model_path=/data/linyifan/models/Qwen3-14B \\
        -k platform=a2a3 -k batch=16
    # CPU-only (use the script's golden_qwen3_decode as target):
    python -m llm.testing.hf_compare run qwen3_14b.decode -k cpu_only=true

    # For a shorter context length (alias: ctx_len):
    python -m llm.testing.hf_compare run qwen3_14b.decode -k cpu_only=true -k seq_len=128
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..base import ComparisonCase, InputSpec, OutputSelector, TensorSpec, Tolerance, register_case
from ..input_sampler import build_rope_tables, int_fill, uniform
from ..paged_kv import (
    compute_decode_slot_mapping,
    init_block_table_identity,
    paged_to_dense_history,
    select_kv_at_decode_slot,
)
from ..reference import CallableReference
from ..target import CallableTarget, PyPTOKernelTarget
from ._qwen3_14b_common import (
    DECODE_SPEC_ORDER as SPEC_ORDER,
    DEFAULT_MODEL_PATH,
    HEAD_DIM,
    HIDDEN,
    NUM_KV_HEADS,
    coerce_bool,
    load_kernel_module,
    single_layer_adapter,
)


# ---------------------------------------------------------------------------
# HF reference forward
# ---------------------------------------------------------------------------
def _hf_decode_forward(
    inputs: Mapping[str, torch.Tensor],
    hf_state: Mapping[str, torch.Tensor],
    *,
    model_path: str,
    max_seq: int,
    block_size: int,
    hf_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Run Qwen3DecoderLayer for layer 0 with the given inputs / KV history."""
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    cfg = AutoConfig.from_pretrained(model_path)
    cfg._attn_implementation = "eager"
    layer = Qwen3DecoderLayer(cfg, layer_idx=0).to(hf_dtype).eval()

    prefix = "model.layers.0."
    sd = {k[len(prefix):]: v.to(hf_dtype) for k, v in hf_state.items() if k.startswith(prefix)}
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"[qwen3.decode] missing HF keys: {missing}")
    if unexpected:
        raise RuntimeError(f"[qwen3.decode] unexpected HF keys: {unexpected}")

    batch = inputs["hidden_states"].shape[0]
    seq_len = int(inputs["seq_lens"][0].item())
    pos = seq_len - 1

    block_table = inputs["block_table"]
    max_blocks_per_seq = (max_seq + block_size - 1) // block_size
    k_hist, v_hist = paged_to_dense_history(
        inputs["k_cache"], inputs["v_cache"], block_table,
        batch=batch, ctx_len=pos,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
        block_size=block_size, max_blocks_per_seq=max_blocks_per_seq,
        hf_dtype=hf_dtype,
    )

    cache = DynamicCache()
    cache.update(k_hist, v_hist, layer_idx=0)

    cos = inputs["rope_cos"][pos:pos + 1].to(hf_dtype).unsqueeze(0).expand(batch, -1, -1)
    sin = inputs["rope_sin"][pos:pos + 1].to(hf_dtype).unsqueeze(0).expand(batch, -1, -1)
    hs_in = inputs["hidden_states"].to(hf_dtype).unsqueeze(1)
    pos_ids = torch.full((batch, 1), pos, dtype=torch.long)

    with torch.no_grad():
        out = layer(
            hidden_states=hs_in,
            attention_mask=None,
            position_ids=pos_ids,
            past_key_values=cache,
            position_embeddings=(cos, sin),
        )
    if isinstance(out, tuple):
        out = out[0]
    out_hs = out.squeeze(1)

    new_k = cache.layers[0].keys[:, :, pos:pos + 1, :].squeeze(2)  # [B, nkv, head_dim]
    new_v = cache.layers[0].values[:, :, pos:pos + 1, :].squeeze(2)

    return {"layer_out": out_hs, "k_pos": new_k, "v_pos": new_v}


# ---------------------------------------------------------------------------
# Case factory
# ---------------------------------------------------------------------------
@register_case("qwen3_14b.decode")
def build(
    hf_model_path: str = DEFAULT_MODEL_PATH,
    cpu_only: Any = False,
    platform: str = "a2a3",
    device_id: int | str | None = None,
    batch: int = 1,
    seed: int = 0,
    seq_len: int | None = None,
    max_seq: int | None = None,
    ctx_len: int | None = None,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    hf_dtype: str = "fp32",
) -> ComparisonCase:
    cpu_only = coerce_bool(cpu_only)
    device_id = int(device_id) if device_id is not None else None
    batch = int(batch)
    seed = int(seed)
    seq_len = int(seq_len) if seq_len is not None else None
    max_seq = int(max_seq) if max_seq is not None else None
    ctx_len = int(ctx_len) if ctx_len is not None else None
    atol = float(atol)
    rtol = float(rtol)
    hf_dtype_t = torch.float32 if hf_dtype == "fp32" else torch.bfloat16

    dec = load_kernel_module(
        "models/qwen3/14b/qwen3_14b_decode.py",
        "_qwen3_14b_decode_kernel",
    )

    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    # Decode is a single-token step that assumes position = (seq_len - 1).
    # `seq_len` is the context length for this step; `max_seq` controls the
    # cache capacity / RoPE table size (defaults to the kernel's MAX_SEQ).
    # Back-compat: `ctx_len` is an alias for `seq_len` if `seq_len` is unset.
    context_len = seq_len if seq_len is not None else (ctx_len or dec.MAX_SEQ)
    max_seq_eff = max_seq or dec.MAX_SEQ
    if context_len > max_seq_eff:
        raise ValueError(f"seq_len={context_len} > max_seq={max_seq_eff}")

    max_blocks_per_seq = (max_seq_eff + dec.BLOCK_SIZE - 1) // dec.BLOCK_SIZE
    num_blocks = batch * max_blocks_per_seq
    cache_rows = num_blocks * NUM_KV_HEADS * dec.BLOCK_SIZE

    # ---- inputs ------------------------------------------------------------
    cos_tab, sin_tab = build_rope_tables(max_seq_eff, HEAD_DIM, base=1_000_000.0)
    input_spec = InputSpec(
        seed=seed,
        tensors={
            "hidden_states": TensorSpec((batch, HIDDEN), torch.bfloat16, sampler=uniform()),
            "seq_lens": TensorSpec((batch,), torch.int32, sampler=int_fill(context_len)),
            "block_table": TensorSpec((batch * max_blocks_per_seq,), torch.int32),
            "slot_mapping": TensorSpec((batch,), torch.int32),
            "k_cache": TensorSpec((cache_rows, HEAD_DIM), torch.bfloat16, sampler=uniform()),
            "v_cache": TensorSpec((cache_rows, HEAD_DIM), torch.bfloat16, sampler=uniform()),
            "out": TensorSpec((batch, HIDDEN), torch.bfloat16),
            # RoPE tables are deterministic; inject via constant samplers.
            "rope_cos": TensorSpec(tuple(cos_tab.shape), torch.float32,
                                   sampler=lambda s, d, g: cos_tab.clone().to(d)),
            "rope_sin": TensorSpec(tuple(sin_tab.shape), torch.float32,
                                   sampler=lambda s, d, g: sin_tab.clone().to(d)),
        },
    )

    # ---- weight adapter ----------------------------------------------------
    adapter = single_layer_adapter(layer_idx=0)

    # ---- reference ---------------------------------------------------------
    reference = CallableReference(
        name="hf.Qwen3DecoderLayer",
        fn=lambda inp, st: _hf_decode_forward(
            inp, st, model_path=hf_model_path, max_seq=max_seq_eff,
            block_size=int(dec.BLOCK_SIZE), hf_dtype=hf_dtype_t,
        ),
    )

    # ---- target ------------------------------------------------------------
    def _post_run(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        kv_kwargs = dict(
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            block_size=int(dec.BLOCK_SIZE),
        )
        return {
            "layer_out": tensors["out"].clone(),
            "k_pos": select_kv_at_decode_slot(
                tensors["k_cache"], tensors["slot_mapping"], **kv_kwargs,
            ),
            "v_pos": select_kv_at_decode_slot(
                tensors["v_cache"], tensors["slot_mapping"], **kv_kwargs,
            ),
        }

    if cpu_only:
        # Use the script's PyTorch golden as target (HF vs golden_qwen3_decode).
        def _golden(inp: Mapping[str, torch.Tensor], _w: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            ref_inputs = {**_w, **inp}
            dec.golden_qwen3_decode(ref_inputs)
            kv_kwargs = dict(
                num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                block_size=int(dec.BLOCK_SIZE),
            )
            return {
                "layer_out": ref_inputs["out"].clone(),
                "k_pos": select_kv_at_decode_slot(
                    ref_inputs["k_cache"], ref_inputs["slot_mapping"], **kv_kwargs,
                ),
                "v_pos": select_kv_at_decode_slot(
                    ref_inputs["v_cache"], ref_inputs["slot_mapping"], **kv_kwargs,
                ),
            }
        target = CallableTarget(name="pytorch.golden_qwen3_decode", fn=_golden)
        # Note: the script's golden does not write back to the input dict, so
        # k_pos / v_pos comparisons are degenerate. Restrict to layer_out.
        selectors = [
            OutputSelector(name="layer_out", ref_key="layer_out", tgt_key="layer_out"),
        ]
    else:
        target = PyPTOKernelTarget(
            name=f"pypto.qwen3_14b_decode[{platform}]",
            build_program=lambda: dec.build_qwen3_decode_program(batch=batch, max_seq=max_seq_eff),
            spec_order=SPEC_ORDER,
            platform=platform,
            device_id=device_id,
            post_run=_post_run,
        )
        selectors = [
            OutputSelector(name="layer_out", ref_key="layer_out", tgt_key="layer_out"),
            OutputSelector(name="k_cache[pos]", ref_key="k_pos", tgt_key="k_pos"),
            OutputSelector(name="v_cache[pos]", ref_key="v_pos", tgt_key="v_pos"),
        ]

    return ComparisonCase(
        name="qwen3_14b.decode",
        reference=reference,
        target=target,
        input_spec=input_spec,
        weight_adapter=adapter,
        selectors=selectors,
        tolerance=Tolerance(atol=atol, rtol=rtol),
        hf_weights=hf_model_path,  # path -> auto-loaded by runner
        on_inputs=lambda t: _init_paged_attention_decode_inputs(
            t,
            batch=batch,
            block_size=int(dec.BLOCK_SIZE),
            max_blocks_per_seq=max_blocks_per_seq,
        ),
    )


def _init_paged_attention_decode_inputs(
    tensors: dict[str, torch.Tensor],
    *,
    batch: int,
    block_size: int,
    max_blocks_per_seq: int,
) -> None:
    init_block_table_identity(
        tensors["block_table"], batch=batch, max_blocks_per_seq=max_blocks_per_seq,
    )
    compute_decode_slot_mapping(
        tensors["slot_mapping"], tensors["seq_lens"],
        batch=batch, block_size=block_size, max_blocks_per_seq=max_blocks_per_seq,
    )
