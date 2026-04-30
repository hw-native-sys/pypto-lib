# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comparison case: Qwen3-14B prefill (single layer) vs Hugging Face Qwen3DecoderLayer.

Port of ``examples/models/qwen3/14b/compare_prefill_with_hf.py``. Run via:

    python -m llm.testing.hf_compare run qwen3_14b.prefill \\
        -k hf_model_path=/data/linyifan/models/Qwen3-14B \\
        -k seq_len=128 -k platform=a2a3 -k batch=16
    # CPU-only:
    python -m llm.testing.hf_compare run qwen3_14b.prefill -k cpu_only=true -k seq_len=128
"""
from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..base import ComparisonCase, InputSpec, OutputSelector, TensorSpec, Tolerance, register_case
from ..input_sampler import build_rope_tables, int_fill, uniform
from ..reference import CallableReference
from ..target import CallableTarget, PyPTOKernelTarget
from ..weight_adapter import Cast, Contiguous, DictAdapter, Map, Transpose, View


DEFAULT_MODEL_PATH = "/data/linyifan/models/Qwen3-14B"

# Qwen3-14B layer constants (mirrors qwen3_14b_prefill.py).
HIDDEN = 5120
HEAD_DIM = 128
NUM_HEADS = 40
NUM_KV_HEADS = 8

# Spec order must match qwen3_14b_prefill.qwen3_14b_prefill signature.
SPEC_ORDER = [
    "hidden_states",
    "seq_lens",
    "input_rms_weight",
    "wq",
    "wk",
    "wv",
    "q_norm_weight",
    "k_norm_weight",
    "rope_cos",
    "rope_sin",
    "block_table",
    "slot_mapping",
    "k_cache",
    "v_cache",
    "wo",
    "post_rms_weight",
    "w_gate",
    "w_up",
    "w_down",
    "out",
]


# ---------------------------------------------------------------------------
# HF reference forward (full-sequence prefill)
# ---------------------------------------------------------------------------
def _hf_prefill_forward(
    inputs: Mapping[str, torch.Tensor],
    hf_state: Mapping[str, torch.Tensor],
    *,
    model_path: str,
    hf_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Run Qwen3DecoderLayer for layer 0 on a full prefill sequence."""
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
        raise RuntimeError(f"[qwen3.prefill] missing HF keys: {missing}")
    if unexpected:
        raise RuntimeError(f"[qwen3.prefill] unexpected HF keys: {unexpected}")

    batch = inputs["hidden_states"].shape[0]
    seq_len = int(inputs["seq_lens"][0].item())

    cos = inputs["rope_cos"][:seq_len].to(hf_dtype).unsqueeze(0).expand(batch, -1, -1)
    sin = inputs["rope_sin"][:seq_len].to(hf_dtype).unsqueeze(0).expand(batch, -1, -1)
    hs_in = inputs["hidden_states"][:, :seq_len, :].to(hf_dtype)
    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch, -1)

    causal_mask = torch.full((seq_len, seq_len), float("-inf"), dtype=hf_dtype)
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)

    cache = DynamicCache()
    with torch.no_grad():
        out = layer(
            hidden_states=hs_in,
            attention_mask=causal_mask,
            position_ids=pos_ids,
            past_key_values=cache,
            position_embeddings=(cos, sin),
        )
    if isinstance(out, tuple):
        out = out[0]
    return {
        "layer_out": out,                     # [B, S, H]
        "k_cache": cache.layers[0].keys,      # [B, nkv, S, head_dim]
        "v_cache": cache.layers[0].values,    # [B, nkv, S, head_dim]
    }


def _select_npu_layer_out(
    flat: torch.Tensor, *, batch: int, max_seq: int, seq_len: int,
) -> torch.Tensor:
    return flat.view(batch, max_seq, HIDDEN)[:, :seq_len, :].clone()


def _select_npu_kv(
    flat: torch.Tensor,
    *,
    slot_mapping: torch.Tensor,
    block_size: int,
    batch: int,
    max_seq: int,
    seq_len: int,
) -> torch.Tensor:
    # The kernel writes KV into a paged cache using slot_mapping, not a simple
    # [B, nkv, S, D] contiguous layout. Reconstruct the logical [B, nkv, S, D]
    # view by gathering the written rows.
    #
    # cache_row = (slot_block * nkv + kvh) * BLOCK_SIZE + slot_offset
    #
    # slot = slot_mapping[b * max_seq + pos]
    # slot_block = slot // BLOCK_SIZE
    # slot_offset = slot % BLOCK_SIZE
    out = torch.empty((batch, NUM_KV_HEADS, seq_len, HEAD_DIM), dtype=flat.dtype, device=flat.device)
    for b in range(batch):
        for pos in range(seq_len):
            slot = int(slot_mapping[b * max_seq + pos].item())
            slot_block = slot // block_size
            slot_off = slot - slot_block * block_size
            base = slot_block * NUM_KV_HEADS * block_size + slot_off
            for kvh in range(NUM_KV_HEADS):
                out[b, kvh, pos, :] = flat[base + kvh * block_size, :]
    return out.clone()


# ---------------------------------------------------------------------------
# Case factory
# ---------------------------------------------------------------------------
def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "on"}


@register_case("qwen3_14b.prefill")
def build(
    hf_model_path: str = DEFAULT_MODEL_PATH,
    cpu_only: Any = False,
    platform: str = "a2a3",
    device_id: int | str | None = None,
    batch: int = 1,
    seed: int = 0,
    seq_len: int = 128,
    max_seq: int | None = None,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    hf_dtype: str = "fp32",
) -> ComparisonCase:
    cpu_only = _coerce_bool(cpu_only)
    device_id = int(device_id) if device_id is not None else None
    batch = int(batch)
    seed = int(seed)
    seq_len = int(seq_len)
    max_seq_arg = int(max_seq) if max_seq is not None else None
    atol = float(atol)
    rtol = float(rtol)
    hf_dtype_t = torch.float32 if hf_dtype == "fp32" else torch.bfloat16

    repo_root = Path(__file__).resolve().parents[4]
    # Load the kernel directly from examples (bypass llm.model shim).
    _KERNEL_PATH = repo_root / "examples" / "models" / "qwen3" / "14b" / "qwen3_14b_prefill.py"
    _SPEC = importlib.util.spec_from_file_location("_qwen3_14b_prefill_kernel", _KERNEL_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Cannot load Qwen3-14B prefill kernel from {_KERNEL_PATH}")
    prefill = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(prefill)

    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    max_seq_eff = max_seq_arg or prefill.MAX_SEQ
    if seq_len > max_seq_eff:
        raise ValueError(f"seq_len={seq_len} > max_seq={max_seq_eff}")
    max_blocks_per_seq = (max_seq_eff + prefill.BLOCK_SIZE - 1) // prefill.BLOCK_SIZE
    num_blocks = batch * max_blocks_per_seq
    cache_rows = num_blocks * NUM_KV_HEADS * prefill.BLOCK_SIZE

    # Capture block_size for the KV selector closure.
    _block_size = int(prefill.BLOCK_SIZE)

    # ---- inputs ------------------------------------------------------------
    cos_tab, sin_tab = build_rope_tables(max_seq_eff, HEAD_DIM, base=1_000_000.0)
    input_spec = InputSpec(
        seed=seed,
        tensors={
            "hidden_states": TensorSpec(
                (batch, max_seq_eff, HIDDEN), torch.bfloat16, sampler=uniform()
            ),
            "seq_lens": TensorSpec((batch,), torch.int32, sampler=int_fill(seq_len)),
            "block_table": TensorSpec((batch * max_blocks_per_seq,), torch.int32),
            "slot_mapping": TensorSpec((batch * max_seq_eff,), torch.int32),
            "k_cache": TensorSpec((cache_rows, HEAD_DIM), torch.bfloat16),
            "v_cache": TensorSpec((cache_rows, HEAD_DIM), torch.bfloat16),
            "out": TensorSpec((batch, max_seq_eff, HIDDEN), torch.bfloat16),
            "rope_cos": TensorSpec(
                tuple(cos_tab.shape), torch.float32,
                sampler=lambda s, d, g: cos_tab.clone().to(d),
            ),
            "rope_sin": TensorSpec(
                tuple(sin_tab.shape), torch.float32,
                sampler=lambda s, d, g: sin_tab.clone().to(d),
            ),
        },
    )

    # ---- weight adapter (identical layout to decode case) ------------------
    adapter = DictAdapter(
        prefix="model.layers.0.",
        mapping={
            "input_rms_weight": Map("input_layernorm.weight",
                                    ops=[View([1, HIDDEN]), Cast(torch.float32)]),
            "wq": Map("self_attn.q_proj.weight",
                      ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "wk": Map("self_attn.k_proj.weight",
                      ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "wv": Map("self_attn.v_proj.weight",
                      ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "q_norm_weight": Map("self_attn.q_norm.weight",
                                 ops=[View([1, HEAD_DIM]), Cast(torch.float32)]),
            "k_norm_weight": Map("self_attn.k_norm.weight",
                                 ops=[View([1, HEAD_DIM]), Cast(torch.float32)]),
            "wo": Map("self_attn.o_proj.weight",
                      ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "post_rms_weight": Map("post_attention_layernorm.weight",
                                   ops=[View([1, HIDDEN]), Cast(torch.float32)]),
            "w_gate": Map("mlp.gate_proj.weight",
                          ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "w_up": Map("mlp.up_proj.weight",
                        ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "w_down": Map("mlp.down_proj.weight",
                          ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
        },
    )

    # ---- reference ---------------------------------------------------------
    reference = CallableReference(
        name="hf.Qwen3DecoderLayer[prefill]",
        fn=lambda inp, st: _hf_prefill_forward(
            inp, st, model_path=hf_model_path, hf_dtype=hf_dtype_t,
        ),
    )

    # ---- target ------------------------------------------------------------
    def _post_run(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        slot_mapping = tensors["slot_mapping"]
        return {
            "layer_out": _select_npu_layer_out(
                tensors["out"], batch=batch, max_seq=max_seq_eff, seq_len=seq_len,
            ),
            "k_cache_written": _select_npu_kv(
                tensors["k_cache"], slot_mapping=slot_mapping, block_size=_block_size,
                batch=batch, max_seq=max_seq_eff, seq_len=seq_len,
            ),
            "v_cache_written": _select_npu_kv(
                tensors["v_cache"], slot_mapping=slot_mapping, block_size=_block_size,
                batch=batch, max_seq=max_seq_eff, seq_len=seq_len,
            ),
        }

    if cpu_only:
        def _golden(inp: Mapping[str, torch.Tensor], _w: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            ref_inputs = {**_w, **inp}
            prefill.golden_qwen3_14b_prefill(ref_inputs)
            return {
                "layer_out": _select_npu_layer_out(
                    ref_inputs["out"], batch=batch, max_seq=max_seq_eff, seq_len=seq_len,
                ),
            }
        target = CallableTarget(name="pytorch.golden_qwen3_14b_prefill", fn=_golden)
        # Golden does not write KV back; only layer_out is comparable.
        selectors = [
            OutputSelector(name="layer_out", ref_key="layer_out", tgt_key="layer_out"),
        ]
    else:
        target = PyPTOKernelTarget(
            name=f"pypto.qwen3_14b_prefill[{platform}]",
            build_program=lambda: prefill.build_qwen3_14b_prefill_program(
                batch=batch, max_seq=max_seq_eff,
            ),
            spec_order=SPEC_ORDER,
            platform=platform,
            device_id=device_id,
            post_run=_post_run,
        )
        selectors = [
            OutputSelector(name="layer_out", ref_key="layer_out", tgt_key="layer_out"),
            OutputSelector(name="k_cache", ref_key="k_cache", tgt_key="k_cache_written"),
            OutputSelector(name="v_cache", ref_key="v_cache", tgt_key="v_cache_written"),
        ]

    return ComparisonCase(
        name="qwen3_14b.prefill",
        reference=reference,
        target=target,
        input_spec=input_spec,
        weight_adapter=adapter,
        selectors=selectors,
        tolerance=Tolerance(atol=atol, rtol=rtol),
        hf_weights=hf_model_path,
        on_inputs=lambda t: _init_paged_attention_inputs(
            t, batch=batch, max_seq=max_seq_eff, block_size=prefill.BLOCK_SIZE, max_blocks_per_seq=max_blocks_per_seq,
        ),
    )


# ---------------------------------------------------------------------------
# Paged-attention metadata initializers (block_table / slot_mapping)
# ---------------------------------------------------------------------------


def _init_paged_attention_inputs(
    tensors: dict[str, torch.Tensor],
    *,
    batch: int,
    max_seq: int,
    block_size: int,
    max_blocks_per_seq: int,
) -> None:
    # Identity mapping: logical blocks map to physical blocks in order.
    tensors["block_table"][:] = torch.arange(batch * max_blocks_per_seq, dtype=torch.int32)

    # slot_mapping maps each (b, pos) -> (phys_block * block_size + offset).
    slot = tensors["slot_mapping"]
    for b in range(batch):
        base_block = b * max_blocks_per_seq
        for pos in range(max_seq):
            logical_block = pos // block_size
            page_off = pos - logical_block * block_size
            phys_block = base_block + logical_block
            slot[b * max_seq + pos] = phys_block * block_size + page_off
