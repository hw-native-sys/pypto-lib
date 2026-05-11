"""Comparison case: Qwen3-14B fused multi-layer decode vs HF.

Stacks weights across `num_layers` and compares the fused pypto kernel
output against `num_layers` HF Qwen3DecoderLayer instances in sequence.

Run via:

    python -m llm.testing.hf_compare run qwen3_14b.decode_full \\
        -k hf_model_path=/data/linyifan/models/Qwen3-14B \\
        -k platform=a2a3 -k num_layers=4 -k seq_len=128
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..base import ComparisonCase, InputSpec, OutputSelector, TensorSpec, Tolerance, register_case
from ..base import WeightAdapter
from ..input_sampler import build_rope_tables, int_fill, uniform
from ..paged_kv import (
    compute_decode_slot_mapping,
    init_block_table_identity,
    paged_to_dense_history,
    select_kv_at_decode_slot,
)
from ..reference import CallableReference
from ..target import PyPTOKernelTarget
from ._qwen3_14b_common import (
    DECODE_SPEC_ORDER as SPEC_ORDER,
    DEFAULT_MODEL_PATH,
    HEAD_DIM,
    HIDDEN,
    INTER,
    NUM_HEADS,
    NUM_KV_HEADS,
    load_kernel_module,
)


# ---------------------------------------------------------------------------
# HF reference: run num_layers Qwen3DecoderLayer in sequence.
# ---------------------------------------------------------------------------
def _hf_decode_full_forward(
    inputs: Mapping[str, torch.Tensor],
    hf_state: Mapping[str, torch.Tensor],
    *,
    model_path: str,
    max_seq: int,
    num_layers: int,
    block_size: int,
    hf_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    cfg = AutoConfig.from_pretrained(model_path)
    cfg._attn_implementation = "eager"

    layers = []
    for li in range(num_layers):
        layer = Qwen3DecoderLayer(cfg, layer_idx=li).to(hf_dtype).eval()
        prefix = f"model.layers.{li}."
        sd = {k[len(prefix):]: v.to(hf_dtype) for k, v in hf_state.items() if k.startswith(prefix)}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        if missing:
            raise RuntimeError(f"[decode_full] L{li} missing HF keys: {missing}")
        if unexpected:
            raise RuntimeError(f"[decode_full] L{li} unexpected HF keys: {unexpected}")
        layers.append(layer)

    batch = inputs["hidden_states"].shape[0]
    seq_len = int(inputs["seq_lens"][0].item())
    pos = seq_len - 1

    block_table = inputs["block_table"]
    max_blocks_per_seq = (max_seq + block_size - 1) // block_size
    layer_cache_rows = batch * max_blocks_per_seq * NUM_KV_HEADS * block_size

    cache = DynamicCache()
    for li in range(num_layers):
        k_hist, v_hist = paged_to_dense_history(
            inputs["k_cache"], inputs["v_cache"], block_table,
            batch=batch, ctx_len=pos,
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
            block_size=block_size, max_blocks_per_seq=max_blocks_per_seq,
            hf_dtype=hf_dtype,
            layer_offset_rows=li * layer_cache_rows,
        )
        cache.update(k_hist, v_hist, layer_idx=li)

    cos = inputs["rope_cos"][pos:pos + 1].to(hf_dtype).unsqueeze(0).expand(batch, -1, -1)
    sin = inputs["rope_sin"][pos:pos + 1].to(hf_dtype).unsqueeze(0).expand(batch, -1, -1)
    pos_ids = torch.full((batch, 1), pos, dtype=torch.long)

    hs = inputs["hidden_states"].to(hf_dtype).unsqueeze(1)
    per_layer_out: list[torch.Tensor] = []
    per_layer_k: list[torch.Tensor] = []
    per_layer_v: list[torch.Tensor] = []
    with torch.no_grad():
        for li in range(num_layers):
            out = layers[li](
                hidden_states=hs,
                attention_mask=None,
                position_ids=pos_ids,
                past_key_values=cache,
                position_embeddings=(cos, sin),
            )
            if isinstance(out, tuple):
                out = out[0]
            hs = out
            per_layer_out.append(hs.squeeze(1).clone())
            per_layer_k.append(cache.layers[li].keys[:, :, pos:pos + 1, :].squeeze(2).clone())
            per_layer_v.append(cache.layers[li].values[:, :, pos:pos + 1, :].squeeze(2).clone())

    result: dict[str, torch.Tensor] = {
        "layer_out": hs.squeeze(1),
    }
    for li in range(num_layers):
        result[f"layer{li:02d}_hidden"] = per_layer_out[li]
        result[f"layer{li:02d}_k_pos"] = per_layer_k[li]
        result[f"layer{li:02d}_v_pos"] = per_layer_v[li]
    return result


# ---------------------------------------------------------------------------
# Stacked weight adapter.
# ---------------------------------------------------------------------------
class StackedAdapter(WeightAdapter):
    """Stacks per-layer HF weights into the fused kernel layout."""

    def __init__(self, num_layers: int, hidden: int, head_dim: int):
        self.num_layers = num_layers
        self.hidden = hidden
        self.head_dim = head_dim

    def adapt(self, hf_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        L = self.num_layers
        H = self.hidden
        D = self.head_dim

        def get(key_template: str, li: int) -> torch.Tensor:
            return hf_state[f"model.layers.{li}.{key_template}"]

        def stack_norm(name: str, dim_per_layer: int) -> torch.Tensor:
            rows = [get(name, li).view(1, dim_per_layer).float() for li in range(L)]
            return torch.cat(rows, dim=0).contiguous()

        def stack_proj(name: str, out_dim: int) -> torch.Tensor:
            # HF stores weights as [out, in]; kernel expects [in, out] (transposed).
            rows = [get(name, li).t().contiguous().to(torch.bfloat16) for li in range(L)]
            return torch.cat(rows, dim=0).contiguous()

        return {
            "input_rms_weight": stack_norm("input_layernorm.weight", H),
            "wq": stack_proj("self_attn.q_proj.weight", NUM_HEADS * D),
            "wk": stack_proj("self_attn.k_proj.weight", NUM_KV_HEADS * D),
            "wv": stack_proj("self_attn.v_proj.weight", NUM_KV_HEADS * D),
            "q_norm_weight": stack_norm("self_attn.q_norm.weight", D),
            "k_norm_weight": stack_norm("self_attn.k_norm.weight", D),
            "wo": stack_proj("self_attn.o_proj.weight", H),
            "post_rms_weight": stack_norm("post_attention_layernorm.weight", H),
            "w_gate": stack_proj("mlp.gate_proj.weight", INTER),
            "w_up": stack_proj("mlp.up_proj.weight", INTER),
            "w_down": stack_proj("mlp.down_proj.weight", H),
        }


# ---------------------------------------------------------------------------
# Case factory
# ---------------------------------------------------------------------------
@register_case("qwen3_14b.decode_full")
def build(
    hf_model_path: str = DEFAULT_MODEL_PATH,
    platform: str = "a2a3",
    device_id: int | str | None = None,
    batch: int = 1,
    seed: int = 0,
    seq_len: int | None = None,
    max_seq: int | None = None,
    num_layers: int = 4,
    atol: float = 1.5e-2,
    rtol: float = 1.5e-2,
    pass_rate: float = 0.99,
    hf_dtype: str = "fp32",
) -> ComparisonCase:
    # Tolerance defaults are sized for the BF16-faithful NPU path vs FP32 HF
    # reference. K-cache (after K-norm + RoPE) sees ~3e-3 mean / ~0.2 max
    # absolute error from intermediate BF16 round-trips on UB boundaries; a
    # ~2x BF16 ulp budget (1.5e-2) absorbs that, and 0.99 pass_rate accepts
    # the BF16 long-tail outliers. Override via -k atol/-k rtol/-k pass_rate.
    device_id = int(device_id) if device_id is not None else None
    batch = int(batch)
    seed = int(seed)
    num_layers = int(num_layers)
    seq_len = int(seq_len) if seq_len is not None else None
    max_seq = int(max_seq) if max_seq is not None else None
    atol = float(atol)
    rtol = float(rtol)
    pass_rate = float(pass_rate)
    hf_dtype_t = torch.float32 if hf_dtype == "fp32" else torch.bfloat16

    dec = load_kernel_module(
        "models/qwen3/14b/qwen3_14b_decode_full.py",
        "_qwen3_14b_decode_full_kernel",
    )

    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    context_len = seq_len if seq_len is not None else dec.MAX_SEQ
    max_seq_eff = max_seq or dec.MAX_SEQ
    if context_len > max_seq_eff:
        raise ValueError(f"seq_len={context_len} > max_seq={max_seq_eff}")

    block_size = int(dec.BLOCK_SIZE)
    max_blocks_per_seq = (max_seq_eff + block_size - 1) // block_size
    layer_cache_rows = batch * max_blocks_per_seq * NUM_KV_HEADS * block_size
    cache_rows = num_layers * layer_cache_rows

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
            "rope_cos": TensorSpec(tuple(cos_tab.shape), torch.float32,
                                   sampler=lambda s, d, g: cos_tab.clone().to(d)),
            "rope_sin": TensorSpec(tuple(sin_tab.shape), torch.float32,
                                   sampler=lambda s, d, g: sin_tab.clone().to(d)),
        },
    )

    adapter = StackedAdapter(num_layers=num_layers, hidden=HIDDEN, head_dim=HEAD_DIM)

    reference = CallableReference(
        name="hf.Qwen3DecoderLayer_x{}".format(num_layers),
        fn=lambda inp, st: _hf_decode_full_forward(
            inp, st,
            model_path=hf_model_path, max_seq=max_seq_eff,
            num_layers=num_layers, block_size=block_size, hf_dtype=hf_dtype_t,
        ),
    )

    def _post_run(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {"layer_out": tensors["out"].clone()}
        kv_kwargs = dict(
            num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, block_size=block_size,
        )
        for li in range(num_layers):
            offset = li * layer_cache_rows
            result[f"layer{li:02d}_k_pos"] = select_kv_at_decode_slot(
                tensors["k_cache"], tensors["slot_mapping"],
                layer_offset_rows=offset, **kv_kwargs,
            )
            result[f"layer{li:02d}_v_pos"] = select_kv_at_decode_slot(
                tensors["v_cache"], tensors["slot_mapping"],
                layer_offset_rows=offset, **kv_kwargs,
            )
        return result

    target = PyPTOKernelTarget(
        name=f"pypto.qwen3_14b_decode_full[{platform}]",
        build_program=lambda: dec.build_qwen3_decode_program(
            batch=batch, max_seq=max_seq_eff, num_layers=num_layers,
        ),
        spec_order=SPEC_ORDER,
        platform=platform,
        device_id=device_id,
        post_run=_post_run,
    )

    selectors: list[OutputSelector] = [
        OutputSelector(name="layer_out", ref_key="layer_out", tgt_key="layer_out"),
    ]
    for li in range(num_layers):
        selectors.append(OutputSelector(
            name=f"layer{li:02d}.k_pos",
            ref_key=f"layer{li:02d}_k_pos", tgt_key=f"layer{li:02d}_k_pos",
        ))
        selectors.append(OutputSelector(
            name=f"layer{li:02d}.v_pos",
            ref_key=f"layer{li:02d}_v_pos", tgt_key=f"layer{li:02d}_v_pos",
        ))

    return ComparisonCase(
        name="qwen3_14b.decode_full",
        reference=reference,
        target=target,
        input_spec=input_spec,
        weight_adapter=adapter,
        selectors=selectors,
        tolerance=Tolerance(atol=atol, rtol=rtol, pass_rate_threshold=pass_rate),
        hf_weights=hf_model_path,
        on_inputs=lambda t: _init_paged_attention_inputs(
            t, batch=batch, block_size=block_size, max_blocks_per_seq=max_blocks_per_seq,
        ),
    )


def _init_paged_attention_inputs(
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
