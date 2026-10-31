"""Layer-0 assembly references and the PyPTO attention subgraph.

The first assembly boundary is deliberately explicit.  It covers the part
that is resident on every A2/A3 rank: normalized input, QKV, paged SWA,
grouped output projection, and the shared/routed MoE result.  HC state and
Engram state are separate boundaries and are added by the L1/L2 harnesses.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import pypto.language as pl

from .decode_swa import swa_attention_reference
from .gemm import make_fp32_dequantize, make_int8_gemm, make_pack_weight
from .moe import routed_expert_reference, shared_expert_reference
from .norm_rope import make_rope, make_rmsnorm
from .output_proj import output_projection_reference
from .qkv import qkv_reference, qkv_shape
from .swa_cache import gather_swa_cache_reference, write_swa_cache_reference
from .quant import make_dynamic_int8_quantize


def route_reference(
    value: torch.Tensor,
    gate_weight: torch.Tensor,
    text_bias: torch.Tensor,
    route_weights: Mapping[str, torch.Tensor] | Callable[[int], Mapping[str, torch.Tensor]],
    *,
    top_k: int = 6,
    routed_scaling_factor: float = 1.5,
    input_ids: torch.Tensor | None = None,
    vision_bias: torch.Tensor | None = None,
    image_sentinel_base_id: int = 129257,
) -> dict[str, torch.Tensor]:
    """Run the V4.1 router and selected routed experts on CPU.

    ``route_weights`` may be a dense mapping with an expert dimension or a
    callable returning one expert's six tensors.  The lazy form avoids loading
    all 384 experts into host memory for a small layer validation case.
    """

    logits = torch.matmul(value.float(), gate_weight.float().transpose(0, 1))
    positive = logits.clamp_min(0.0)
    score = torch.sqrt(positive + torch.log1p(torch.exp(-logits.abs())))
    bias = text_bias.float().reshape(1, -1)
    if input_ids is not None and vision_bias is not None:
        token_ids = input_ids.reshape(-1)
        vision_mask = (token_ids >= image_sentinel_base_id) & (
            token_ids < image_sentinel_base_id + 5
        )
        if vision_mask.any():
            bias = bias.expand(value.shape[0], -1).clone()
            bias[vision_mask] = vision_bias.float().reshape(1, -1)
    biased = score + bias
    selected_score, selected_ids = torch.topk(biased, top_k, dim=-1)
    del selected_score
    raw = score.gather(-1, selected_ids)
    weights = raw / raw.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    weights = weights * routed_scaling_factor
    if callable(route_weights):
        # Keep the reference memory bounded by evaluating one selected expert
        # at a time.  ``routed_expert_reference`` is still used for the
        # numerically sensitive W8A8/SwiGLU/down path.
        result = torch.zeros_like(value, dtype=torch.float32)

        def reference_weight(
            tensors: Mapping[str, torch.Tensor], name: str, input_width: int, output_width: int
        ):
            weight = tensors[name]
            # Prepared safetensors keep canonical [K, N] tensors for PyPTO;
            # the Torch golden consumes output-major [N, K].  Existing dense
            # fixtures already use the latter, so detect the layout from K.
            if tuple(weight.shape) == (output_width, input_width):
                return weight
            if tuple(weight.shape) == (input_width, output_width):
                return weight.transpose(0, 1).contiguous()
            raise ValueError(
                f"{name} has no [N,K]/[K,N] layout for ({output_width},{input_width}): {tuple(weight.shape)}"
            )

        for token in range(value.shape[0]):
            for route in range(top_k):
                expert = int(selected_ids[token, route])
                tensors = route_weights(expert)
                one = routed_expert_reference(
                    value[token:token + 1],
                    torch.zeros((1, 1), dtype=torch.int64),
                    weights[token:token + 1, route:route + 1],
                    reference_weight(tensors, "w1", value.shape[-1], tensors["w1_scale"].numel()).unsqueeze(0), tensors["w1_scale"].unsqueeze(0),
                    reference_weight(tensors, "w2", tensors["w1_scale"].numel(), value.shape[-1]).unsqueeze(0), tensors["w2_scale"].unsqueeze(0),
                    reference_weight(tensors, "w3", value.shape[-1], tensors["w3_scale"].numel()).unsqueeze(0), tensors["w3_scale"].unsqueeze(0),
                )
                result[token] += one[0].float()
        result = result.to(torch.bfloat16)
    else:
        result = routed_expert_reference(
            value,
            selected_ids,
            weights,
            route_weights["w1"],
            route_weights["w1_scale"],
            route_weights["w2"],
            route_weights["w2_scale"],
            route_weights["w3"],
            route_weights["w3_scale"],
        )
    return {"logits": logits, "route_ids": selected_ids.to(torch.int32), "route_weights": weights, "output": result}


def layer0_reference(
    value: torch.Tensor,
    qkv_weights: Mapping[str, torch.Tensor],
    output_weights: Mapping[str, torch.Tensor],
    shared_weights: Mapping[str, torch.Tensor],
    *,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    slots: torch.Tensor,
    indices: torch.Tensor,
    cache: torch.Tensor,
    sink: torch.Tensor,
    gate_weight: torch.Tensor | None = None,
    gate_bias: torch.Tensor | None = None,
    routed_weights: Mapping[str, torch.Tensor] | None = None,
    input_ids: torch.Tensor | None = None,
    active: int | None = None,
    config: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """CPU golden for the assembled L0 attention + MoE boundaries."""

    shape = qkv_shape(config)
    active = value.shape[0] if active is None else int(active)
    qkv = qkv_reference(value, qkv_weights, cosine, sine, config)
    cache_after = write_swa_cache_reference(cache, qkv["window_kv"], slots)
    selected = gather_swa_cache_reference(cache_after, indices)
    attended = swa_attention_reference(
        qkv["query"], selected, indices, sink, shape.head_dim, valid_tokens=active
    )
    projected = output_projection_reference(
        attended,
        output_weights["wo_a"],
        output_weights["wo_b"],
        output_weights["wo_b_scale"],
        cosine,
        sine,
        shape.num_attention_heads,
        shape.head_dim,
        shape.qk_rope_head_dim,
    )
    shared = shared_expert_reference(
        value,
        shared_weights["gate_up_weight"].transpose(0, 1).contiguous(),
        shared_weights["gate_up_scale"],
        shared_weights["down_weight"].transpose(0, 1).contiguous(),
        shared_weights["down_scale"],
    )
    routed = torch.zeros_like(shared)
    router = {}
    if gate_weight is not None and gate_bias is not None and routed_weights is not None:
        router = route_reference(value, gate_weight, gate_bias, routed_weights, input_ids=input_ids)
        routed = router["output"]
    combined = shared + routed
    return {
        **qkv,
        "cache_after": cache_after,
        "selected": selected,
        "attended": attended,
        "rotated": projected["rotated"],
        "grouped": projected["grouped"],
        "attention_output": projected["output"],
        "shared_output": shared,
        "routed_output": routed,
        "combined_output": combined,
        **router,
    }


def make_l0_attention(capacity: int, config: Mapping[str, object]):
    """Build QKV + SWA attention + grouped output projection for L0.

    The MoE router and expert dispatch remain a host/EP boundary.  This
    function therefore returns the attention subgraph and its intermediate
    cache/attention tensors, which are the stable inputs to the routed MoE
    verifier.
    """

    shape = qkv_shape(config)
    hidden = shape.hidden_size
    latent = shape.q_lora_rank
    heads = shape.num_attention_heads
    head_dim = shape.head_dim
    width = shape.query_width
    pairs = shape.rope_pairs
    rope_dim = shape.qk_rope_head_dim
    quant_x = make_dynamic_int8_quantize(capacity, hidden)
    quant_q = make_dynamic_int8_quantize(capacity, latent)
    pack_qa = make_pack_weight(hidden, latent)
    pack_qb = make_pack_weight(latent, width)
    pack_kv = make_pack_weight(hidden, head_dim)
    dot_qa = make_int8_gemm(capacity, hidden, latent)
    dot_qb = make_int8_gemm(capacity, latent, width)
    dot_kv = make_int8_gemm(capacity, hidden, head_dim)
    dequant_qa = make_fp32_dequantize(capacity, latent)
    dequant_qb = make_fp32_dequantize(capacity, width)
    dequant_kv = make_fp32_dequantize(capacity, head_dim)
    norm_q = make_rmsnorm(capacity, latent, shape.rms_norm_eps)
    norm_kv = make_rmsnorm(capacity, head_dim, shape.rms_norm_eps)
    rope_q = make_rope(capacity, heads, head_dim, rope_dim)
    rope_kv = make_rope(capacity, 1, head_dim, rope_dim)
    from .decode_swa import make_swa_attention
    from .output_proj import make_output_projection

    attend = make_swa_attention(capacity, shape.num_attention_heads, shape.head_dim, int(config["sliding_window"]))
    project = make_output_projection(
        capacity,
        shape.num_attention_heads,
        shape.head_dim,
        shape.qk_rope_head_dim,
        int(config["o_groups"]),
        int(config["o_lora_rank"]),
    )
    pairs = shape.rope_pairs
    hidden = shape.hidden_size
    width = shape.query_width
    window = int(config["sliding_window"])
    groups = int(config["o_groups"])
    group_input = width // groups
    rank = int(config["o_lora_rank"])

    @pl.jit
    def layer0_attention(
        value: pl.Tensor[[capacity, hidden], pl.BF16],
        wq_a: pl.Tensor[[hidden, shape.q_lora_rank], pl.INT8],
        wq_a_scale: pl.Tensor[[shape.q_lora_rank], pl.FP32],
        q_norm_weight: pl.Tensor[[shape.q_lora_rank], pl.BF16],
        wq_b: pl.Tensor[[shape.q_lora_rank, width], pl.INT8],
        wq_b_scale: pl.Tensor[[width], pl.FP32],
        wkv: pl.Tensor[[hidden, shape.head_dim], pl.INT8],
        wkv_scale: pl.Tensor[[shape.head_dim], pl.FP32],
        kv_norm_weight: pl.Tensor[[shape.head_dim], pl.BF16],
        wo_a: pl.Tensor[[groups, group_input, rank], pl.BF16],
        wo_b: pl.Tensor[[groups * rank, hidden], pl.INT8],
        wo_b_scale: pl.Tensor[[hidden], pl.FP32],
        cosine: pl.Tensor[[capacity, pairs], pl.FP32],
        sine: pl.Tensor[[capacity, pairs], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        sink: pl.Tensor[[shape.num_attention_heads], pl.FP32],
        positions: pl.InOut[pl.Tensor[[capacity], pl.INT64]],
        query_latent: pl.InOut[pl.Tensor[[capacity, shape.q_lora_rank], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, shape.head_dim], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[2, 128, 1, shape.head_dim], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        attended: pl.InOut[pl.Tensor[[capacity, width], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, hidden], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        wq_a_output = pl.create_tensor([capacity, latent], dtype=pl.BF16)
        wq_b_output = pl.create_tensor([capacity, width], dtype=pl.BF16)
        wkv_output = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        wkv_normed_output = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        x_i8 = pl.create_tensor([capacity, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_x(value, x_i8, x_scale, active)
        qa_packed = pl.create_tensor([hidden * (latent // 64), 64], dtype=pl.INT8)
        pack_qa(wq_a, qa_packed)
        qa_i32 = pl.create_tensor([capacity, latent], dtype=pl.INT32)
        dot_qa(x_i8, qa_packed, qa_i32, active)
        dequant_qa(qa_i32, x_scale, wq_a_scale, wq_a_output, active)
        norm_q(wq_a_output, q_norm_weight, query_latent, active)
        q_i8 = pl.create_tensor([capacity, latent], dtype=pl.INT8)
        q_scale = pl.create_tensor([capacity, 1], dtype=pl.FP32)
        quant_q(query_latent, q_i8, q_scale, active)
        qb_packed = pl.create_tensor([latent * (width // 64), 64], dtype=pl.INT8)
        pack_qb(wq_b, qb_packed)
        qb_i32 = pl.create_tensor([capacity, width], dtype=pl.INT32)
        dot_qb(q_i8, qb_packed, qb_i32, active)
        dequant_qb(qb_i32, q_scale, wq_b_scale, wq_b_output, active)
        rope_q(wq_b_output, cosine, sine, query, active)
        kv_packed = pl.create_tensor([hidden * (head_dim // 64), 64], dtype=pl.INT8)
        pack_kv(wkv, kv_packed)
        kv_i32 = pl.create_tensor([capacity, head_dim], dtype=pl.INT32)
        dot_kv(x_i8, kv_packed, kv_i32, active)
        dequant_kv(kv_i32, x_scale, wkv_scale, wkv_output, active)
        norm_kv(wkv_output, kv_norm_weight, wkv_normed_output, active)
        rope_kv(wkv_normed_output, cosine, sine, window_kv, active)
        attend(query, window_kv, slots, indices, sink, cache, attended, active)
        project(attended, wo_a, wo_b, wo_b_scale, cosine, sine, output, active)
        return positions, query_latent, query, window_kv, cache, cache_lens, attended, output

    return layer0_attention


__all__ = ["layer0_reference", "make_l0_attention", "route_reference"]
