"""DeepSeek V4.1 A2/A3 decoder stages for layers 14 and 15.

Layer 14 is the ratio-2 source layer.  It publishes the BF16 compressed KV
row and the INT8/FP16 index row, and owns the second Engram invocation.  Layer
15 is a ratio-2 consumer: it reuses layer-14 cache and prepared QLI indices.

The functions in this module are deliberately stage shaped.  They can be
used with tensors captured from vLLM-Ascend, without requiring a complete
checkpoint in this repository, and the PyPTO factories expose the same dtype
boundaries for a later device run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import pypto.language as pl

from .operator_config import A2A3_DTYPE_CONTRACT, CONTRACT
from .engram import HASH_COLS, HEAD_DIM, HC, HIDDEN, LOOKUP_WIDTH, engram_reference
from .quant import w8a8_linear
from .sparse_attention import make_sparse_attention, sparse_attention_reference
from .index_cache import quantize_index_cache
from .indexer_prepare import (
    make_indexer_prepare,
    prepare_indexer_indices_reference,
)
from .indexer_query import make_l3_indexer_query
from .indexer_select import (
    indexer_select_reference,
    make_indexer_select,
)
from .norm_rope import make_rope, make_rmsnorm, rms_norm_reference, rope_reference
from .output_proj import make_output_projection


@dataclass(frozen=True)
class Layer14A2A3Contract:
    """Static topology and cache ownership for the L14 source layer."""

    layer_id: int = 14
    compress_ratio: int = 2
    kv_source_layer_id: int = 14
    index_source_layer_id: int = 14
    source_mode: str = "source"
    has_engram: bool = True
    reuse_source_layer_id: int | None = None
    compressed_cache_dtype: str = A2A3_DTYPE_CONTRACT["compressed_kv_cache"]
    index_cache_dtype: str = A2A3_DTYPE_CONTRACT["index_cache"]
    index_scale_dtype: str = A2A3_DTYPE_CONTRACT["index_cache_scale"]
    compressor_state_dtype: str = "float32"


@dataclass(frozen=True)
class Layer15A2A3Contract:
    """Static topology and cache ownership for the L15 reuse layer."""

    layer_id: int = 15
    compress_ratio: int = 2
    kv_source_layer_id: int = 14
    index_source_layer_id: int = 14
    source_mode: str = "reuse"
    has_engram: bool = False
    reuse_source_layer_id: int = 14
    compressed_cache_dtype: str = A2A3_DTYPE_CONTRACT["compressed_kv_cache"]
    index_cache_dtype: str = A2A3_DTYPE_CONTRACT["index_cache"]
    index_scale_dtype: str = A2A3_DTYPE_CONTRACT["index_cache_scale"]
    compressor_state_dtype: str = "float32"


LAYER14 = Layer14A2A3Contract()
LAYER15 = Layer15A2A3Contract()


def validate_layer14_15_contract() -> tuple[Layer14A2A3Contract, Layer15A2A3Contract]:
    """Assert the model topology that the two decoder entry points require."""

    if LAYER14.source_mode != "source" or not LAYER14.has_engram:
        raise AssertionError("layer 14 must be a ratio-2 source layer with Engram")
    if LAYER15.source_mode != "reuse" or LAYER15.reuse_source_layer_id != 14:
        raise AssertionError("layer 15 must reuse layer-14 compressed/index state")
    for contract in (LAYER14, LAYER15):
        if contract.compress_ratio != 2:
            raise AssertionError(f"layer {contract.layer_id} is not a ratio-2 layer")
        if contract.compressed_cache_dtype != "bfloat16":
            raise AssertionError("A2/A3 compressed cache must be BF16")
        if contract.index_cache_dtype != "int8" or contract.index_scale_dtype != "float16":
            raise AssertionError("A2/A3 index cache must be INT8 with FP16 scales")
        if contract.compressor_state_dtype != "float32":
            raise AssertionError("ratio-2 compressor state must be FP32")
    return LAYER14, LAYER15


def _check_rows(*values: torch.Tensor, width: int, name: str) -> None:
    if not values:
        return
    rows = values[0].shape[0]
    if any(value.ndim != 2 or value.shape != (rows, width) for value in values):
        raise ValueError(f"{name} tensors must all have shape [T, {width}]")


def ratio2_pool_reference(
    kv_proj: torch.Tensor,
    score_proj: torch.Tensor,
    previous_kv: torch.Tensor,
    previous_score: torch.Tensor,
    norm_weight: torch.Tensor,
    position_ids: torch.Tensor,
    previous_valid: torch.Tensor | None = None,
    eps: float = 1.0e-20,
) -> dict[str, torch.Tensor]:
    """Pool ratio-2 compressor partners and publish BF16 latent rows.

    ``kv_proj`` and ``score_proj`` are FP32 compressor projections.  The
    weighted pair is narrowed to BF16 before RMSNorm, matching the A2/A3
    compressor boundary.  ``previous_*`` is the row loaded from the FP32
    request-owned state cache; a false ``previous_valid`` suppresses a row.
    """

    if kv_proj.dtype != torch.float32 or score_proj.dtype != torch.float32:
        raise TypeError("compressor projections must be FP32")
    if kv_proj.shape != score_proj.shape or previous_kv.shape != kv_proj.shape or previous_score.shape != kv_proj.shape:
        raise ValueError("ratio-2 projections and partner rows must have the same [T, D] shape")
    if position_ids.ndim != 1 or position_ids.shape[0] != kv_proj.shape[0]:
        raise ValueError("position_ids must have shape [T]")
    if norm_weight.shape != (kv_proj.shape[-1],):
        raise ValueError("norm_weight must match compressor head dimension")
    if previous_valid is None:
        previous_valid = torch.ones(kv_proj.shape[0], dtype=torch.bool, device=kv_proj.device)
    if previous_valid.shape != position_ids.shape:
        raise ValueError("previous_valid must have shape [T]")

    tokens, head_dim = kv_proj.shape
    latent = torch.zeros(tokens, head_dim, dtype=torch.bfloat16, device=kv_proj.device)
    publish = torch.zeros(tokens, dtype=torch.bool, device=kv_proj.device)
    for token in range(tokens):
        # A compressed row is emitted for the second token of each pair.
        if int(position_ids[token]) < 0 or int(position_ids[token]) % 2 != 1:
            continue
        if not bool(previous_valid[token]):
            continue
        current_score = score_proj[token].float()
        partner_score = previous_score[token].float()
        logits = torch.stack((partner_score, current_score), dim=0)
        weights = torch.softmax(logits, dim=0)
        pooled = weights[0] * previous_kv[token].float() + weights[1] * kv_proj[token].float()
        pooled_bf16 = pooled.to(torch.bfloat16)
        latent[token] = rms_norm_reference(pooled_bf16.unsqueeze(0), norm_weight, eps=eps)[0]
        publish[token] = True
    return {"latent": latent, "publish": publish}


def update_ratio2_state_reference(
    state_cache: torch.Tensor,
    kv_proj: torch.Tensor,
    score_proj: torch.Tensor,
    position_ids: torch.Tensor,
    block: int = 0,
) -> torch.Tensor:
    """Write FP32 compressor state rows into a request-owned ring buffer."""

    if state_cache.dtype != torch.float32 or state_cache.ndim != 3:
        raise TypeError("state_cache must be FP32 [blocks, capacity, 2 * head_dim]")
    head_dim = kv_proj.shape[-1]
    if state_cache.shape[-1] != 2 * head_dim:
        raise ValueError("state_cache width must be 2 * compressor head dimension")
    if not 0 <= block < state_cache.shape[0]:
        raise ValueError("state block is out of range")
    result = state_cache.clone()
    for token, position in enumerate(position_ids.tolist()):
        if position < 0:
            continue
        slot = int(position) % result.shape[1]
        result[block, slot, :head_dim] = kv_proj[token]
        result[block, slot, head_dim:] = score_proj[token]
    return result


def index_key_reference(
    compressed: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    eps: float = 1.0e-20,
    *,
    cosine: torch.Tensor | None = None,
    sine: torch.Tensor | None = None,
    rope_dim: int = 64,
) -> dict[str, torch.Tensor]:
    """Project a BF16 compressed row into the INT8/FP16 QLI key cache."""

    if compressed.ndim != 2 or compressed.dtype != torch.bfloat16:
        raise TypeError("compressed rows must be a BF16 [T, D] tensor")
    if index_wk.ndim != 2 or index_wk.shape[0] != compressed.shape[-1] or index_wk.shape[-1] != CONTRACT.index_head_dim:
        raise ValueError("index_wk must be [D, 128]")
    if index_norm_weight.shape != (CONTRACT.index_head_dim,):
        raise ValueError("index_norm_weight must have shape [128]")
    projected = torch.matmul(compressed.float(), index_wk.float()).to(torch.bfloat16)
    normalized = rms_norm_reference(projected, index_norm_weight, eps=eps)
    if (cosine is None) != (sine is None):
        raise ValueError("index-key cosine and sine tables must be supplied together")
    if cosine is not None:
        if cosine.shape != sine.shape or cosine.shape != (compressed.shape[0], rope_dim // 2):
            raise ValueError("index-key RoPE tables must have shape [T, rope_dim // 2]")
        normalized = rope_reference(
            normalized, cosine, sine, heads=1, head_dim=CONTRACT.index_head_dim, rope_dim=rope_dim
        )
    payload, scale = quantize_index_cache(normalized.unsqueeze(1))
    return {
        "projected": projected,
        "normalized": normalized,
        "payload": payload[:, 0],
        "scale": scale[:, 0],
    }


def layer14_source_reference(
    *,
    kv_proj: torch.Tensor,
    score_proj: torch.Tensor,
    previous_kv: torch.Tensor,
    previous_score: torch.Tensor,
    norm_weight: torch.Tensor,
    position_ids: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    qli_query: torch.Tensor,
    qli_query_scale: torch.Tensor,
    qli_weights: torch.Tensor,
    topk: int = 512,
    previous_valid: torch.Tensor | None = None,
    compressed_cosine: torch.Tensor | None = None,
    compressed_sine: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run the L14 source handoff through compression, QLI and preparation.

    C2A publishes only completed pair rows.  Keep the token-shaped
    ``compressed`` result for existing callers and expose compact published
    rows for the cache and QLI key plane.
    """

    source = ratio2_pool_reference(
        kv_proj, score_proj, previous_kv, previous_score, norm_weight, position_ids,
        previous_valid=previous_valid,
    )
    if (compressed_cosine is None) != (compressed_sine is None):
        raise ValueError("compressed-key cosine and sine tables must be supplied together")
    published = source["publish"]
    latent = source["latent"]
    if compressed_cosine is not None:
        if compressed_cosine.shape != compressed_sine.shape or compressed_cosine.shape != (
            latent.shape[0], 64 // 2
        ):
            raise ValueError("compressed-key RoPE tables must have shape [T, rope_dim // 2]")
        compressed_full = rope_reference(
            latent,
            compressed_cosine,
            compressed_sine,
            heads=1,
            head_dim=latent.shape[-1],
            rope_dim=64,
        )
        published_cosine = compressed_cosine[published]
        published_sine = compressed_sine[published]
    else:
        compressed_full = latent
        published_cosine = published_sine = None
    published_latent = latent[published]
    key = index_key_reference(
        published_latent,
        index_wk,
        index_norm_weight,
        cosine=published_cosine,
        sine=published_sine,
    )
    key_payload = key["payload"]
    key_scale = key["scale"]
    key_capacity = max(topk, key_payload.shape[0])
    if key_capacity > key_payload.shape[0]:
        padding = key_capacity - key_payload.shape[0]
        key_payload = torch.cat(
            (key_payload, torch.zeros(padding, key_payload.shape[-1], dtype=key_payload.dtype, device=key_payload.device)),
            dim=0,
        )
        key_scale = torch.cat(
            (key_scale, torch.ones(padding, dtype=key_scale.dtype, device=key_scale.device)),
            dim=0,
        )
    qli = indexer_select_reference(
        qli_query,
        qli_query_scale,
        key_payload,
        key_scale,
        qli_weights,
        position_ids,
        topk=topk,
        compress_ratio=2,
    )
    prepared = prepare_indexer_indices_reference(qli["indices"], position_ids, compress_ratio=2)
    return {
        "compressed": compressed_full,
        "published_compressed": compressed_full[published],
        "publish": source["publish"],
        "index_key": key["payload"],
        "index_key_scale": key["scale"],
        "qli_scores": qli["scores"],
        "qli_indices": qli["indices"],
        "prepared_indices": prepared["indices"],
        "prepared_lengths": prepared["lengths"],
    }


def layer15_reuse_reference(
    query: torch.Tensor,
    window_rows: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_rows: torch.Tensor,
    compressed_indices: torch.Tensor,
    sink: torch.Tensor,
    *,
    active: int | None = None,
) -> torch.Tensor:
    """Run the L15 consumer using L14's already prepared compressed indices."""

    if compressed_indices.dtype != torch.int32:
        raise TypeError("L15 compressed indices must be INT32")
    return sparse_attention_reference(
        query,
        window_rows,
        window_indices,
        compressed_rows,
        compressed_indices,
        sink,
        active=active,
    )


def apply_layer14_engram_reference(**kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Keep L14 Engram as a named BF16 stage; L15 must not call this helper."""

    if not LAYER14.has_engram:
        raise AssertionError("Engram is only valid on layer 14")
    return engram_reference(**kwargs)


def quantize_engram_rows_reference(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Match vLLM-Ascend A2/A3 group-32 Engram row quantization."""

    if rows.ndim < 2 or rows.shape[-1] % 32:
        raise ValueError("Engram rows must have a trailing width divisible by 32")
    grouped = rows.float().reshape(*rows.shape[:-1], -1, 32)
    maximum = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)
    exponent = torch.ceil(torch.log2(scale))
    scale = torch.where(torch.isfinite(exponent), torch.ldexp(torch.ones_like(scale), exponent.to(torch.int32)), scale)
    codes = torch.round(grouped / scale).clamp(-127, 127).to(torch.int8).reshape_as(rows)
    return codes, scale.squeeze(-1)


def dequantize_engram_rows_reference(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Decode group-32 INT8 Engram rows into the BF16 lookup ABI."""

    if codes.dtype != torch.int8 or codes.shape[-1] % 32:
        raise TypeError("Engram row payload must be INT8 with width divisible by 32")
    groups = codes.shape[-1] // 32
    if scale.shape != (*codes.shape[:-1], groups):
        raise ValueError("Engram row scales must have one FP32 value per group of 32")
    return (codes.float().reshape(*codes.shape[:-1], groups, 32) * scale.float().unsqueeze(-1)).reshape_as(codes).to(torch.bfloat16)


def engram_a2a3_reference(
    hash_ids: torch.Tensor,
    table_codes: torch.Tensor,
    table_scale: torch.Tensor,
    wkv_weight: torch.Tensor,
    wkv_scale: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    hidden: torch.Tensor,
    token_mask: torch.Tensor,
    eps: float = 1.0e-20,
) -> dict[str, torch.Tensor]:
    """A2/A3 Engram lookup plus W8A8 WKV and BF16 gate.

    ``table_codes`` uses the vLLM group-32 INT8 row format and
    ``wkv_weight`` is canonical ``[K, N]`` INT8 with FP32 per-output scales.
    """

    if hash_ids.ndim != 2 or hash_ids.shape[-1] != HASH_COLS:
        raise ValueError(f"hash_ids must have shape [T, {HASH_COLS}]")
    if table_codes.dtype != torch.int8 or table_codes.shape[-1] != HEAD_DIM:
        raise TypeError(f"table_codes must be INT8 [rows, {HEAD_DIM}]")
    if table_scale.dtype != torch.float32 or table_scale.shape != (table_codes.shape[0], HEAD_DIM // 32):
        raise TypeError(f"table_scale must be FP32 [rows, {HEAD_DIM // 32}]")
    if wkv_weight.dtype != torch.int8 or tuple(wkv_weight.shape) != (LOOKUP_WIDTH, (HC + 1) * HIDDEN):
        raise ValueError("wkv_weight must be canonical INT8 [6144, 25600]")
    if wkv_scale.shape != ((HC + 1) * HIDDEN,):
        raise ValueError("wkv_scale must have one FP32 value per WKV output channel")
    if hidden.dtype != torch.bfloat16:
        raise TypeError("Engram hidden state must be BF16")
    lookup = dequantize_engram_rows_reference(
        table_codes[hash_ids.long()], table_scale[hash_ids.long()]
    ).reshape(hash_ids.shape[0], LOOKUP_WIDTH)
    kv = w8a8_linear(lookup, wkv_weight.transpose(0, 1).contiguous(), wkv_scale)
    key, value = kv[:, :HC * HIDDEN].reshape(kv.shape[0], HC, HIDDEN), kv[:, HC * HIDDEN:]
    source = hidden.float()
    channel_weight = q_weight.float() * k_weight.float()
    source_rstd = torch.rsqrt(source.square().mean(-1, keepdim=True) + eps)
    key_rstd = torch.rsqrt(key.float().square().mean(-1, keepdim=True) + eps)
    dot = (source.unsqueeze(1) * channel_weight.unsqueeze(0) * key.float()).sum(-1, keepdim=True)
    dot = dot * source_rstd.unsqueeze(1) * key_rstd * (HIDDEN ** -0.5)
    signed_root = torch.copysign(dot.abs().clamp_min(1.0e-6).sqrt(), dot)
    gate = torch.sigmoid(signed_root) * token_mask.reshape(-1, 1, 1).float()
    output = (source.unsqueeze(1) + gate * value.float().unsqueeze(1)).to(torch.bfloat16)
    return {"lookup": lookup, "kv": kv, "output": output}


def _make_fp32_linear_inline(capacity: int, width: int, output_width: int):
    """Composable FP32-input accumulation leaf for source projections."""

    n_tile = 64 if output_width % 64 == 0 else 32
    blocks = output_width // n_tile

    @pl.jit.inline
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP32],
        output: pl.Tensor[[capacity, output_width], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_l14_fp32_projection"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * n_tile
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, width, 256, stage=2):
                lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                # A2/A3 cube matmul requires matching operand dtypes.  The
                # source weight remains FP32 in the ABI; this tile is narrowed
                # for the cube and the FP32 accumulator is retained.
                rhs = pl.cast(weight[k0:k0 + 256, n0:n0 + n_tile], pl.BF16, mode="rint")
                acc = pl.matmul_acc(acc, lhs, rhs, init_cond=(k0 == 0))
            output[t0:t0 + 16, n0:n0 + n_tile] = pl.set_validshape(acc, rows, n_tile)
        return output

    return linear


def _make_bf16_linear_inline(capacity: int, width: int, output_width: int):
    n_tile = 64 if output_width % 64 == 0 else 32
    blocks = output_width // n_tile

    @pl.jit.inline
    def linear(
        value: pl.Tensor[[capacity, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[capacity, output_width], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_l14_bf16_projection"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * n_tile
            rows = pl.min(16, active - t0)
            acc = pl.create_tensor([16, n_tile], dtype=pl.FP32)
            for k0 in pl.pipeline(0, width, 256, stage=2):
                lhs = pl.slice(value, [16, 256], [t0, k0], valid_shape=[rows, 256])
                lhs = pl.set_validshape(pl.fillpad(lhs, pad_value=pl.PadValue.zero), 16, 256)
                acc = pl.matmul_acc(acc, lhs, weight[k0:k0 + 256, n0:n0 + n_tile], init_cond=(k0 == 0))
            output[t0:t0 + 16, n0:n0 + n_tile] = pl.set_validshape(
                pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
            )
        return output

    return linear


def make_ratio2_projection(capacity: int, hidden_size: int = 5120, head_dim: int = 512):
    """PyPTO FP32 compressor projection pair used by L14.

    The projection factory keeps the compressor state in FP32.  Pooling and
    normalization are a separate boundary so the state cache never receives
    a packed FP8/FP4 payload.
    """

    if capacity <= 0 or capacity % 16 or hidden_size % 256 or head_dim % 32:
        raise ValueError("A2/A3 compressor dimensions are not tile aligned")
    project = _make_fp32_linear_inline(capacity, hidden_size, head_dim)

    @pl.jit
    def kernel(
        value: pl.Tensor[[capacity, hidden_size], pl.BF16],
        wkv: pl.Tensor[[hidden_size, head_dim], pl.FP32],
        wgate: pl.Tensor[[hidden_size, head_dim], pl.FP32],
        kv_output: pl.Out[pl.Tensor[[capacity, head_dim], pl.FP32]],
        score_output: pl.Out[pl.Tensor[[capacity, head_dim], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        # The cube projection publishes a BF16 temporary; this explicit cast
        # is the FP32 compressor-state handoff consumed by pair pooling.
        kv_bf16 = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        score_bf16 = pl.create_tensor([capacity, head_dim], dtype=pl.BF16)
        project(value, wkv, kv_bf16, active)
        project(value, wgate, score_bf16, active)
        blocks = head_dim // 64
        for block in pl.spmd(((active + 15) // 16) * blocks, name_hint="dsv41_a2a3_l14_fp32_publish"):
            t0 = (block // blocks) * 16
            n0 = (block % blocks) * 64
            rows = pl.min(16, active - t0)
            kv_output[t0:t0 + 16, n0:n0 + 64] = pl.set_validshape(
                pl.cast(kv_bf16[t0:t0 + 16, n0:n0 + 64], pl.FP32), rows, 64
            )
            score_output[t0:t0 + 16, n0:n0 + 64] = pl.set_validshape(
                pl.cast(score_bf16[t0:t0 + 16, n0:n0 + 64], pl.FP32), rows, 64
            )
        return kv_output, score_output

    return kernel


def make_index_key_publish(capacity: int, compressed_dim: int = 512):
    """PyPTO BF16 index-key projection followed by INT8/FP16 publication."""

    if compressed_dim % 256:
        raise ValueError("compressed dimension must be divisible by 256")
    index_dim = CONTRACT.index_head_dim
    project = _make_bf16_linear_inline(capacity, compressed_dim, index_dim)
    norm = make_rmsnorm(capacity, index_dim, 1.0e-20)

    @pl.jit.inline
    def quantize_key(
        value: pl.Tensor[[capacity, index_dim], pl.BF16],
        payload: pl.Tensor[[capacity, index_dim], pl.INT8],
        scales: pl.Tensor[[capacity], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        scale_rows = pl.reshape(scales, [capacity, 1])
        for block in pl.spmd((active + 15) // 16, name_hint="dsv41_a2a3_index_key_quant"):
            row = block * 16
            valid_rows = pl.min(16, active - row)
            tile = pl.slice(value, [16, index_dim], [row, 0], valid_shape=[valid_rows, index_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, index_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            raw_scale = pl.maximum(pl.div(pl.row_max(pl.abs(value_fp32)), 127.0), 1.0e-30)
            stored_scale = pl.maximum(pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32), 2.0 ** -24)
            normalized = pl.row_expand_div(value_fp32, raw_scale)
            positive = pl.maximum(normalized, 0.0)
            negative = pl.minimum(normalized, 0.0)
            epsilon = pl.mul(pl.maximum(pl.abs(normalized), 1.0), 2.0 ** -20)
            positive = pl.cast(pl.sub(positive, epsilon), pl.INT32, mode="rint")
            negative = pl.cast(negative, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(pl.add(positive, negative), -127), 127)
            payload[row:row + 16, :] = pl.set_validshape(pl.cast(bounded, pl.INT8, mode="trunc"), valid_rows, index_dim)
            scale_rows[row:row + 16, 0:1] = pl.set_validshape(
                pl.reshape(pl.cast(stored_scale, pl.FP16), [16, 1]), valid_rows, 1
            )
        return payload, scales

    @pl.jit
    def kernel(
        compressed: pl.Tensor[[capacity, compressed_dim], pl.BF16],
        index_wk: pl.Tensor[[compressed_dim, index_dim], pl.BF16],
        index_norm_weight: pl.Tensor[[index_dim], pl.BF16],
        projected: pl.Out[pl.Tensor[[capacity, index_dim], pl.BF16]],
        payload: pl.Out[pl.Tensor[[capacity, index_dim], pl.INT8]],
        scale: pl.Out[pl.Tensor[[capacity], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        project(compressed, index_wk, projected, active)
        normalized = pl.create_tensor([capacity, index_dim], dtype=pl.BF16)
        norm(projected, index_norm_weight, normalized, active)
        # Publish one FP16 scale per compressed token, matching the vLLM
        # index-cache key ABI rather than the 32-head query ABI.
        quantize_key(normalized, payload, scale, active)
        return projected, payload, scale

    return kernel


def make_index_key_publish_rope(
    capacity: int,
    compressed_dim: int = 512,
    rope_dim: int = 64,
):
    """Publish an index key after the A2/A3 partial-RoPE transform.

    L20/C1A computes the index key from the normalized compressed row and
    rotates its last ``rope_dim`` values before INT8 publication.  The older
    ``make_index_key_publish`` entry point remains unchanged for existing
    source-layer callers.
    """

    if compressed_dim % 256:
        raise ValueError("compressed dimension must be divisible by 256")
    if rope_dim <= 0 or rope_dim % 2 or rope_dim > CONTRACT.index_head_dim:
        raise ValueError("index-key rope dimension must be positive, even, and at most 128")
    if capacity <= 0 or capacity % 16:
        raise ValueError("capacity must be a positive multiple of 16")
    index_dim = CONTRACT.index_head_dim
    project = _make_bf16_linear_inline(capacity, compressed_dim, index_dim)
    norm = make_rmsnorm(capacity, index_dim, 1.0e-20)
    rotate = make_rope(capacity, 1, index_dim, rope_dim)

    @pl.jit.inline
    def quantize_key(
        value: pl.Tensor[[capacity, index_dim], pl.BF16],
        payload: pl.Tensor[[capacity, index_dim], pl.INT8],
        scales: pl.Tensor[[capacity], pl.FP16],
        active: pl.Scalar[pl.INT32],
    ):
        scale_rows = pl.reshape(scales, [capacity, 1])
        for block in pl.spmd((active + 15) // 16, name_hint="dsv41_a2a3_index_key_rope_quant"):
            row = block * 16
            valid_rows = pl.min(16, active - row)
            tile = pl.slice(value, [16, index_dim], [row, 0], valid_shape=[valid_rows, index_dim])
            tile = pl.set_validshape(pl.fillpad(tile, pad_value=pl.PadValue.zero), 16, index_dim)
            value_fp32 = pl.cast(tile, pl.FP32)
            raw_scale = pl.maximum(pl.div(pl.row_max(pl.abs(value_fp32)), 127.0), 1.0e-30)
            stored_scale = pl.maximum(
                pl.cast(pl.cast(raw_scale, pl.FP16, mode="rint"), pl.FP32), 2.0 ** -24
            )
            normalized = pl.row_expand_div(value_fp32, raw_scale)
            positive = pl.maximum(normalized, 0.0)
            negative = pl.minimum(normalized, 0.0)
            epsilon = pl.mul(pl.maximum(pl.abs(normalized), 1.0), 2.0 ** -20)
            positive = pl.cast(pl.sub(positive, epsilon), pl.INT32, mode="rint")
            negative = pl.cast(negative, pl.INT32, mode="rint")
            bounded = pl.minimum(pl.maximum(pl.add(positive, negative), -127), 127)
            payload[row:row + 16, :] = pl.set_validshape(
                pl.cast(bounded, pl.INT8, mode="trunc"), valid_rows, index_dim
            )
            scale_rows[row:row + 16, 0:1] = pl.set_validshape(
                pl.reshape(pl.cast(stored_scale, pl.FP16), [16, 1]), valid_rows, 1
            )
        return payload, scales

    @pl.jit
    def kernel(
        compressed: pl.Tensor[[capacity, compressed_dim], pl.BF16],
        index_wk: pl.Tensor[[compressed_dim, index_dim], pl.BF16],
        index_norm_weight: pl.Tensor[[index_dim], pl.BF16],
        cosine: pl.Tensor[[capacity, rope_dim // 2], pl.FP32],
        sine: pl.Tensor[[capacity, rope_dim // 2], pl.FP32],
        projected: pl.Out[pl.Tensor[[capacity, index_dim], pl.BF16]],
        rotated: pl.Out[pl.Tensor[[capacity, index_dim], pl.BF16]],
        payload: pl.Out[pl.Tensor[[capacity, index_dim], pl.INT8]],
        scale: pl.Out[pl.Tensor[[capacity], pl.FP16]],
        active: pl.Scalar[pl.INT32],
    ):
        scale_rows = pl.reshape(scale, [capacity, 1])
        with pl.spmd(capacity // 16, name_hint="dsv41_a2a3_index_key_rope_init"):
            row = pl.tile.get_block_idx() * 16
            projected_zero = pl.tile.full([16, index_dim], dtype=pl.BF16, value=0.0)
            rotated_zero = pl.tile.full([16, index_dim], dtype=pl.BF16, value=0.0)
            payload_zero = pl.cast(
                pl.tile.full([16, index_dim], dtype=pl.INT32, value=0), pl.INT8, mode="trunc"
            )
            scale_zero = pl.tile.full([16, 16], dtype=pl.FP16, value=0.0)
            pl.store(projected_zero, [row, 0], projected)
            pl.store(rotated_zero, [row, 0], rotated)
            pl.store(payload_zero, [row, 0], payload)
            pl.store(pl.set_validshape(scale_zero, 16, 1), [row, 0], scale_rows)
        project(compressed, index_wk, projected, active)
        normalized = pl.create_tensor([capacity, index_dim], dtype=pl.BF16)
        norm(projected, index_norm_weight, normalized, active)
        rotate(normalized, cosine, sine, rotated, active)
        quantize_key(rotated, payload, scale, active)
        return projected, rotated, payload, scale

    return kernel


def make_layer14_stages(
    capacity: int,
    *,
    hidden_size: int = 5120,
    compressed_dim: int = 512,
    output_groups: int = 8,
    output_rank: int = 1024,
    output_hidden: int = 5120,
    index_key_capacity: int = 512,
    index_topk: int = 512,
) -> dict[str, Any]:
    """Return the independently compilable L14 source stage factories."""

    validate_layer14_15_contract()
    return {
        "compressor_projection": make_ratio2_projection(capacity, hidden_size, compressed_dim),
        "index_key": make_index_key_publish(capacity, compressed_dim),
        "index_query": make_l3_indexer_query(capacity),
        "index_select": make_indexer_select(capacity, index_key_capacity, topk=index_topk),
        "index_prepare": make_indexer_prepare(capacity, topk=index_topk),
        "sparse_attention": make_sparse_attention(capacity, 64, 512),
        "output_projection": make_output_projection(
            capacity, 64, 512, 64, output_groups, output_rank
        ),
        "engram": "use apply_layer14_engram_reference or engram.make_lookup/make_gate",
        "dtype_contract": A2A3_DTYPE_CONTRACT,
        "output_hidden": output_hidden,
    }


def make_layer15_stages(
    capacity: int,
    *,
    output_groups: int = 8,
    output_rank: int = 1024,
) -> dict[str, Any]:
    """Return the L15 reuse stage factories.

    There is intentionally no compressor or QLI selection factory here: both
    cache planes and prepared indices are owned by L14.
    """

    validate_layer14_15_contract()
    return {
        "index_prepare": make_indexer_prepare(capacity),
        "sparse_attention": make_sparse_attention(capacity, 64, 512),
        "output_projection": make_output_projection(capacity, 64, 512, 64, output_groups, output_rank),
        "dtype_contract": A2A3_DTYPE_CONTRACT,
        "reuse_source_layer_id": LAYER15.reuse_source_layer_id,
    }


def load_a2a3_layer14_15_bundle(path: str | Path) -> dict[str, torch.Tensor]:
    """Load an injected converted bundle and reject A5 packed dtypes.

    L14/L15 converted weights are not present in the checked-in checkpoint
    directory.  This helper provides a strict handoff for externally prepared
    bundles and makes accidental FP8/FP4 reinterpretation fail early.
    """

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()):
        raise TypeError("layer14/15 bundle must be a mapping of named tensors")
    forbidden = {
        dtype for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
        ) if dtype is not None
    }
    for name, value in payload.items():
        if value.dtype in forbidden:
            raise TypeError(f"{name} is FP8; decode and quantize it before A2/A3 use")
        if value.dtype == torch.uint8 and ("weight" in name or "cache" in name):
            raise TypeError(f"{name} is a packed byte payload; A2/A3 expects BF16 or INT8")
    return dict(payload)


__all__ = [
    "LAYER14",
    "LAYER15",
    "Layer14A2A3Contract",
    "Layer15A2A3Contract",
    "apply_layer14_engram_reference",
    "dequantize_engram_rows_reference",
    "engram_a2a3_reference",
    "index_key_reference",
    "layer14_source_reference",
    "layer15_reuse_reference",
    "load_a2a3_layer14_15_bundle",
    "make_index_key_publish",
    "make_index_key_publish_rope",
    "make_layer14_stages",
    "make_layer15_stages",
    "make_ratio2_projection",
    "quantize_engram_rows_reference",
    "ratio2_pool_reference",
    "update_ratio2_state_reference",
    "validate_layer14_15_contract",
]
