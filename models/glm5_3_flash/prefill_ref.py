# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU reference driver for the GLM-5.3-Flash prefill backbone (host-only).

Purpose
-------
``prefill_ref.py`` runs the *mathematical* prefill forward of GLM-5.3-Flash on the
host CPU in FP32, driven by the same W8A8 deployment checkpoint the device kernels
consume. It exists for two consumers:

* **single-layer diff** (t8/t9): dump any layer's input/output activations —
  including the mHC four-stream tensor ``x_hc`` and every sub-layer boundary —
  and compare them against the device ``prefill_fwd`` tensors;
* **true-weight token check** (first light): run the full 45-layer backbone on
  prompt token ids and compute top-1 logits with the host LM head.

The local W8A8 weights fold the QuaRot rotation, so this reference must run on
the same W8A8 tensors (dequantized), never on the FP8 source weights. The
Hadamard-128 the device indexer applies to its INT8 query/key basis is
orthogonal and applied to both sides, so it cancels analytically and does not
appear here.

Numerics: all weights are upcast/dequantized to FP32 and every activation is
FP32. The device path computes in BF16, so diffs against this reference measure
the device's own rounding error.

The layer math is ported from the HuggingFace reference implementation
``transformers/src/transformers/models/glm5_next/modeling_glm5_next.py``
(Apache License 2.0, generated from ``modular_glm5_next.py``; local read-only
copy at ``/data/g00957430/glm53-refs/modeling_glm5_next.py``). The KDA chunk and
recurrent cores, the k-pool indexer, and the mHC Sinkhorn mapping follow that
file statement by statement; adaptations are commented at each site.

Checkpoint access goes through :class:`models.glm5_3_flash.weights.Checkpoint`
(t6): lazy per-tensor safetensors header+range reads, the authority-driven
``FLOAT`` / ``W8A8_DYNAMIC`` boundary from ``quant_model_description.json``, and
row gathers for the embedding table. Memory contract for the streaming driver:
one layer's non-expert weights (~0.5-1.2 GB FP32) plus one expert's three
matrices (~100 MB FP32) at a time; the 331 GB checkpoint is never resident.

Usage
-----
Full backbone with dumps (real weights)::

    driver = PrefillReferenceDriver(
        "/data/models/GLM-5.3-Flash/quantized/vllm-ascend-a3-w8a8-published"
    )
    result = driver.run(
        token_ids,                       # [S] or [B, S] int ids
        dump_dir=Path("/tmp/ref_dumps"), # one layer_XX.pt per dumped layer
        dump_layers={0, 3},              # default: every executed layer
    )
    result.final_hidden                  # [B, S, 4096] FP32, post final-norm
    values, ids = driver.lm_head_top1(result.final_hidden[:, -1, :])

Single-layer diff (real or synthetic weights)::

    weights = driver.load_layer(3)                       # DSA/MLA + MoE layer
    streams_out, boundaries = reference_layer_forward(   # streams = x_hc [B,S,4,4096]
        x_hc_streams, weights, attention_mask
    )

Scaled mini checkpoints (layer-0..3 first-light diff source)::

    model = scaled_model_from_config("/…/layer_ckpt_l0l3")   # num_hidden_layers=4 etc.
    driver = PrefillReferenceDriver("/…/layer_ckpt_l0l3", model=model)
    result = driver.run(token_ids, dump_dir=…)               # layers (0, 4), final norm

Weight baselines (do not cross-compare): the mini checkpoint is a **bare
dequant** (``is_rot_used=false``: no QuaRot/Flex-SmoothQuant), while the
published W8A8 snapshot folds those rotations. This driver never reads
``rot.weight`` and applies no rotation, so ref-vs-device pairs are
self-consistent **only when both sides read the same checkpoint**; never diff
numbers produced from different baselines against each other.

Self-test (no real weights; synthetic tensors at the true ``config.py``
geometry)::

    python models/glm5_3_flash/prefill_ref.py
    python models/glm5_3_flash/prefill_ref.py \
        --checkpoint /data/models/GLM-5.3-Flash/quantized/vllm-ascend-a3-w8a8-published
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import (
    FLASH,
    AttentionKind,
    Glm53FlashConfig,
    Glm53LayerConfig,
    MlpKind,
)
from models.glm5_3_flash.weights import TEXT_LAYER_ROOTS, Checkpoint, gather_rows

# --------------------------------------------------------------------------------------
# Checkpoint spelling tables. Mirrors weights.py's family tables (canonical spelling
# first); kept local so this file stays the single new artefact of the task.
# --------------------------------------------------------------------------------------

# (attribute, sub-prefix inside the layer root, candidate suffixes).
_KDA_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("w_q", "self_attn.", ("q_proj.weight",)),
    ("w_k", "self_attn.", ("k_proj.weight",)),
    ("w_v", "self_attn.", ("v_proj.weight",)),
    ("conv_q", "self_attn.", ("q_conv1d.weight",)),
    ("conv_k", "self_attn.", ("k_conv1d.weight",)),
    ("conv_v", "self_attn.", ("v_conv1d.weight",)),
    ("w_f_a", "self_attn.", ("forget_gate.f_a_proj.weight", "f_a_proj.weight")),
    ("w_f_b", "self_attn.", ("forget_gate.f_b_proj.weight", "f_b_proj.weight")),
    ("dt_bias", "self_attn.", ("forget_gate.dt_bias", "dt_bias")),
    ("a_log", "self_attn.", ("forget_gate.A_log", "A_log")),
    ("w_g_a", "self_attn.", ("g_a_proj.weight",)),
    ("w_g_b", "self_attn.", ("g_b_proj.weight",)),
    ("w_b", "self_attn.", ("b_proj.weight",)),
    ("o_norm", "self_attn.", ("o_norm.weight",)),
    ("w_o", "self_attn.", ("o_proj.weight",)),
)

_MLA_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("w_q_a", "self_attn.", ("q_a_proj.weight",)),
    ("q_a_norm", "self_attn.", ("q_a_layernorm.weight",)),
    ("w_q_b", "self_attn.", ("q_b_proj.weight",)),
    ("w_kv_a", "self_attn.", ("kv_a_proj_with_mqa.weight",)),
    ("kv_a_norm", "self_attn.", ("kv_a_layernorm.weight",)),
    ("kv_b", "self_attn.", ("kv_b_proj.weight",)),
    ("w_o", "self_attn.", ("o_proj.weight",)),
)

_INDEXER_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("wq_b", "self_attn.indexer.", ("wq_b.weight",)),
    ("wk", "self_attn.indexer.", ("wk.weight",)),
    ("k_norm_w", "self_attn.indexer.", ("k_norm.weight",)),
    ("k_norm_b", "self_attn.indexer.", ("k_norm.bias",)),
    ("w_weights", "self_attn.indexer.", ("weights_proj.weight",)),
    ("w_compress_gate", "self_attn.indexer.", ("index_kpool_compress_gate",)),
    ("ape", "self_attn.indexer.", ("index_kpool_compress_ape",)),
)

_MHC_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("attn_fn", "", ("attn_hc.fn", "hc_attn_fn")),
    ("attn_scale", "", ("attn_hc.scale", "hc_attn_scale")),
    ("attn_base", "", ("attn_hc.base", "hc_attn_base")),
    ("ffn_fn", "", ("ffn_hc.fn", "hc_ffn_fn")),
    ("ffn_scale", "", ("ffn_hc.scale", "hc_ffn_scale")),
    ("ffn_base", "", ("ffn_hc.base", "hc_ffn_base")),
)

_NORM_TENSORS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("input_norm", "", ("input_layernorm.weight",)),
    ("post_norm", "", ("post_attention_layernorm.weight",)),
)

_FF_GATE = ("gate_proj.weight", "gate_proj")
_FF_UP = ("up_proj.weight", "up_proj")
_FF_DOWN = ("down_proj.weight", "down_proj")
_FF_FUSED = ("w_gate_up", "w_gate_up.weight")

_SCALE_SUFFIXES = (".weight_scale", "_scale")
_OFFSET_SUFFIXES = (".weight_offset", "_offset")

# Boundary keys written to every layer dump (metadata aside). The *_internals keys
# carry the sub-layer internals used for tighter diff localization.
DEFAULT_DUMP_KEYS = (
    "x_hc_in",      # mHC four-stream tensor entering the layer  [B, S, hc_mult, D]
    "attn_norm_in", # collapsed stream after input_layernorm     [B, S, D]
    "attn_out",     # attention sub-layer output                 [B, S, D]
    "x_hc_mid",     # four streams after the attention combine   [B, S, hc_mult, D]
    "ffn_norm_in",  # collapsed stream after post_attention_layernorm [B, S, D]
    "ffn_out",      # MLP sub-layer output                       [B, S, D]
    "x_hc_out",     # four streams leaving the layer             [B, S, hc_mult, D]
)

INTERNAL_DUMP_KEYS = (
    "attn_hc_pre", "attn_hc_post", "attn_hc_comb",
    "ffn_hc_pre", "ffn_hc_post", "ffn_hc_comb",
    "kda_q", "kda_k", "kda_v", "kda_g", "kda_beta", "kda_core", "kda_gate", "kda_o_normed",
    "mla_q_resid", "mla_k_latent", "mla_topk_indices",
    "router_topk_weights", "router_topk_indices", "moe_routed", "moe_shared",
)

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


_FLOAT32_MIN = torch.finfo(torch.float32).min


def _as_batch_seqs(token_ids: torch.Tensor) -> torch.Tensor:
    """Normalize token ids to an ``[B, S]`` int64 tensor."""
    ids = torch.as_tensor(token_ids, dtype=torch.int64)
    if ids.dim() == 1:
        ids = ids[None]
    if ids.dim() != 2 or ids.numel() == 0:
        raise ValueError(f"token ids must be [S] or [B, S], got {tuple(ids.shape)}")
    return ids


# --------------------------------------------------------------------------------------
# Authority-driven checkpoint reads (FLOAT cast / W8A8_DYNAMIC dequant), FP32 out.
# --------------------------------------------------------------------------------------


def _resolve_name(checkpoint: Checkpoint, layer: int, sub_prefix: str,
                  suffixes: Sequence[str]) -> str:
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


def _quant_sidecar(checkpoint: Checkpoint, name: str,
                   suffixes: Sequence[str]) -> str | None:
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
    verified (not assumed), exactly the boundary ``weights.py`` pins.
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


def _read_ffn(checkpoint: Checkpoint, layer: int, mlp_base: str
              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One FFN as FP32 ``(gate, up, down)``, fused or separate gate/up spelling."""
    def names(suffixes: Sequence[str]) -> list[str]:
        return [f"{root}{layer}.{mlp_base}{s}" for root in TEXT_LAYER_ROOTS for s in suffixes]

    fused = next((n for n in names(_FF_FUSED) if n in checkpoint), None)
    if fused is not None:
        gate_up = read_reference_float(checkpoint, fused)
        gate, up = gate_up.chunk(2, dim=0)
    else:
        gate = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_GATE))
        up = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_UP))
    down = read_reference_float(checkpoint, _resolve_name(checkpoint, layer, mlp_base, _FF_DOWN))
    return gate.contiguous(), up.contiguous(), down.contiguous()


# --------------------------------------------------------------------------------------
# Weight containers (FP32) and expert providers.
# --------------------------------------------------------------------------------------


@dataclass
class MhcTensors:
    """One hyper-connection site's mapping parameters (``attn_hc`` or ``ffn_hc``)."""

    fn: torch.Tensor     # [mix_hc, hc_mult * D]
    base: torch.Tensor   # [mix_hc] FP32
    scale: torch.Tensor  # [3] FP32


@dataclass
class KdaTensors:
    """Kimi-delta linear-attention weights of one KDA layer (FP32)."""

    w_q: torch.Tensor        # [qkv, D]
    w_k: torch.Tensor        # [qkv, D]
    w_v: torch.Tensor        # [qkv, D]
    conv_q: torch.Tensor     # [qkv, 1, conv_k]
    conv_k: torch.Tensor     # [qkv, 1, conv_k]
    conv_v: torch.Tensor     # [qkv, 1, conv_k]
    w_f_a: torch.Tensor      # [head_dim, D]
    w_f_b: torch.Tensor      # [qkv, head_dim]
    dt_bias: torch.Tensor    # [qkv] (FP32 in the checkpoint)
    a_log: torch.Tensor      # [H] (FP32 in the checkpoint)
    w_g_a: torch.Tensor      # [head_dim, D]
    w_g_b: torch.Tensor      # [qkv, head_dim]
    w_b: torch.Tensor        # [H, D]
    o_norm: torch.Tensor     # [head_dim]
    w_o: torch.Tensor        # [D, qkv]


@dataclass
class MlaTensors:
    """NoPE-MLA weights of one DSA layer (FP32). ``kv_b`` stays unsplit."""

    w_q_a: torch.Tensor    # [q_lora, D]
    q_a_norm: torch.Tensor  # [q_lora]
    w_q_b: torch.Tensor     # [H * qk_dim, q_lora]
    w_kv_a: torch.Tensor    # [kv_lora, D] (NoPE: no rope columns)
    kv_a_norm: torch.Tensor  # [kv_lora]
    kv_b: torch.Tensor      # [H * (qk_dim + v_dim), kv_lora]
    w_o: torch.Tensor       # [D, H * v_dim]


@dataclass
class IndexerTensors:
    """k-pool indexer weights of one DSA layer (FP32, checkpoint orientation)."""

    wq_b: torch.Tensor          # [n_heads * head_dim, q_lora]
    wk: torch.Tensor            # [head_dim, D]
    k_norm_w: torch.Tensor      # [head_dim]
    k_norm_b: torch.Tensor      # [head_dim]
    w_weights: torch.Tensor     # [n_heads, D]
    w_compress_gate: torch.Tensor  # [head_dim, D]
    ape: torch.Tensor           # [kpool, head_dim]


@dataclass
class DenseTensors:
    """One FFN as separate FP32 gate/up/down (dense MLP or shared expert)."""

    gate: torch.Tensor  # [inter, D]
    up: torch.Tensor    # [inter, D]
    down: torch.Tensor  # [D, inter]


@dataclass
class RouterTensors:
    weight: torch.Tensor  # [n_experts, D]
    bias: torch.Tensor    # [n_experts] FP32


class ExpertProvider:
    """Routed-expert weight source: one expert's FP32 matrices at a time.

    The streaming contract that bounds host memory: the provider materializes a
    single expert per call and the caller releases it before the next. The
    synthetic provider keeps one shared base and re-scales it per expert, so
    every expert is distinct without 15 GB of random tensors.
    """

    def gate_up(self, expert: int) -> torch.Tensor:
        """``[2 * moe_inter, D]`` with gate rows first, then up rows."""
        raise NotImplementedError

    def down(self, expert: int) -> torch.Tensor:
        """``[D, moe_inter]``."""
        raise NotImplementedError


class CheckpointExpertProvider(ExpertProvider):
    """Lazy per-expert dequant reads straight from the checkpoint."""

    def __init__(self, checkpoint: Checkpoint, layer: int):
        self._checkpoint = checkpoint
        self._layer = layer
        self._names: dict[int, tuple[str, str | None, str]] = {}

    def _resolve(self, expert: int) -> tuple[str, str | None, str]:
        if expert not in self._names:
            base = f"mlp.experts.{expert}."
            # Fused spelling first: _resolve_name raises when a spelling is
            # absent, so a fused-only checkpoint would fail on gate_proj
            # before the fused branch is ever considered.
            fused = next(
                (
                    f"{root}{self._layer}.{base}{s}"
                    for root in TEXT_LAYER_ROOTS
                    for s in _FF_FUSED
                    if f"{root}{self._layer}.{base}{s}" in self._checkpoint
                ),
                None,
            )
            if fused is None:
                gate = _resolve_name(self._checkpoint, self._layer, base, _FF_GATE)
                up = _resolve_name(self._checkpoint, self._layer, base, _FF_UP)
            else:
                gate, up = fused, None
            down = _resolve_name(self._checkpoint, self._layer, base, _FF_DOWN)
            self._names[expert] = (gate, up, down)
        return self._names[expert]

    def gate_up(self, expert: int) -> torch.Tensor:
        gate_or_fused, up, _ = self._resolve(expert)
        if up is None:
            return read_reference_float(self._checkpoint, gate_or_fused).contiguous()
        gate = read_reference_float(self._checkpoint, gate_or_fused)
        up_w = read_reference_float(self._checkpoint, up)
        return torch.cat([gate, up_w], dim=0)

    def down(self, expert: int) -> torch.Tensor:
        return read_reference_float(self._checkpoint, self._resolve(expert)[2]).contiguous()


class SyntheticExpertProvider(ExpertProvider):
    """Deterministic synthetic experts: shared base scaled per expert id."""

    def __init__(self, model: Glm53FlashConfig, seed: int):
        generator = torch.Generator().manual_seed(seed)
        inter, hidden = model.moe_intermediate_size, model.hidden_size
        self._base_gate_up = torch.randn(2 * inter, hidden, generator=generator) * 0.02
        self._base_down = torch.randn(hidden, inter, generator=generator) * 0.02
        self._gate_up_buf = torch.empty_like(self._base_gate_up)
        self._down_buf = torch.empty_like(self._base_down)

    def gate_up(self, expert: int) -> torch.Tensor:
        torch.mul(self._base_gate_up, 1.0 + expert * 1e-3, out=self._gate_up_buf)
        return self._gate_up_buf

    def down(self, expert: int) -> torch.Tensor:
        torch.mul(self._base_down, 1.0 + expert * 1e-3, out=self._down_buf)
        return self._down_buf


@dataclass
class LayerRefWeights:
    """Everything one backbone layer's reference forward needs (FP32)."""

    layer_id: int
    config: Glm53LayerConfig
    input_norm: torch.Tensor   # [D]
    post_norm: torch.Tensor    # [D]
    # None exactly when config.has_hyper_connections is False (the MTP layer),
    # which carries no hc_* tensors and uses a plain residual wiring instead.
    attn_hc: MhcTensors | None
    ffn_hc: MhcTensors | None
    model: Glm53FlashConfig | None = None  # geometry the weights belong to (default FLASH)
    kda: KdaTensors | None = None
    mla: MlaTensors | None = None
    indexer: IndexerTensors | None = None
    dense: DenseTensors | None = None            # dense-MLP layers (0..first_k_dense)
    router: RouterTensors | None = None          # MoE layers
    experts: ExpertProvider | None = None        # MoE layers, streamed
    shared: DenseTensors | None = None           # MoE layers


# --------------------------------------------------------------------------------------
# Core math. Ported from modeling_glm5_next.py (Apache-2.0); FP32 throughout.
# --------------------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Glm5NextTextRMSNorm: FP32 variance, weight applied after the inverse cast."""
    variance = x.pow(2).mean(-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return weight * normalized


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Glm5NextTextUnweightedRMSNorm (mHC stream-norm), no learned weight."""
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def mhc_mapping(streams: torch.Tensor, weights: MhcTensors, model: Glm53FlashConfig
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """mHC mapping: ``pre``, ``post``, Sinkhorn-``comb``, and the collapsed stream.

    Port of ``Glm5NextTextHyperConnection.forward``. ``comb`` gets one initial
    column normalization then ``iters - 1`` row/column passes; ``pre`` collapses
    the ``hc_mult`` streams into the sub-layer input.
    """
    hc = model.hc_mult
    flat = unweighted_rms_norm(streams.flatten(start_dim=2), model.rms_norm_eps)
    logits = flat @ weights.fn.t()
    pre_w, post_w, comb_w = logits.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = weights.base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = weights.scale.unbind(0)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + model.hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + model.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + model.hc_eps)
    for _ in range(model.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + model.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + model.hc_eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2)
    return pre, post, comb, collapsed


def combine_streams(post: torch.Tensor, comb: torch.Tensor, sublayer_out: torch.Tensor,
                    residual: torch.Tensor) -> torch.Tensor:
    """Decoder-layer combine: ``post * sublayer_out + comb^T @ residual`` (HF eq.)."""
    return post.unsqueeze(-1) * sublayer_out.unsqueeze(-2) + comb.transpose(-1, -2) @ residual


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-style l2norm: ``eps`` added inside the sqrt (intentionally not max(...))."""
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def causal_conv1d_silu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Grouped causal conv + SiLU; ``x`` is ``[B, C, S]``, ``weight`` ``[C, 1, K]``.

    Port of ``causal_conv1d_fn`` with ``activation="silu"`` (hidden_act): pad
    ``K - 1`` on the left, conv1d with ``groups=C``, keep the first ``S`` outputs.
    """
    seq_len = x.shape[-1]
    out = F.conv1d(x, weight, bias=None, padding=weight.shape[-1] - 1, groups=x.shape[1])
    return F.silu(out[:, :, :seq_len])


def kda_chunk_core(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                   g: torch.Tensor, beta: torch.Tensor, *, chunk_size: int = 64,
                   l2norm_eps: float = 1e-6) -> torch.Tensor:
    """Chunked Kimi-delta attention core (port of ``chunk_kimi_delta_attention``).

    Inputs ``[B, S, H, Dk]`` (q/k/v/g) and ``[B, S, H]`` (beta); output
    ``[B, S, H, Dv]``. The caller guarantees FP32. Follows the HF reference:
    in-kernel l2norm on q/k, ``1/sqrt(Dk)`` scale, log-space intra-chunk decay,
    inter-chunk state reuse, and the delta update rule.
    """
    query, key, value, beta, g = [t.transpose(1, 2).contiguous() for t in (query, key, value, beta, g)]
    query = _l2norm(query, dim=-1, eps=l2norm_eps)
    key = _l2norm(key, dim=-1, eps=l2norm_eps)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    total_sequence_length = sequence_length + pad_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        t.reshape(t.shape[0], t.shape[1], -1, chunk_size, t.shape[-1])
        for t in (query, key, value, g, k_beta, v_beta)
    ]
    beta = beta.reshape(beta.shape[0], beta.shape[1], -1, chunk_size)

    # Intra-chunk attention in log space (per-head decay, unlike GDN's broadcast).
    g = g.cumsum(dim=-2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    strict_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).masked_fill(strict_mask[..., None], float("-inf"))
    decay_mask = decay_mask.exp()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim)
    core_attn_out = torch.zeros_like(value)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=1)
    for i in range(total_sequence_length // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1)
        attn_intra = attn_intra.masked_fill(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    return core_attn_out[:, :, :sequence_length].transpose(1, 2).contiguous()


def kda_recurrent_core(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       g: torch.Tensor, beta: torch.Tensor, *, l2norm_eps: float = 1e-6
                       ) -> torch.Tensor:
    """Step-by-step Kimi-delta recurrence (port of the HF recurrent fallback).

    O(S) loop over the reference recurrence itself — the ground truth the chunk
    core is checked against in the self-test. Same input contract as
    :func:`kda_chunk_core`.
    """
    query, key, value, g, beta = [t.transpose(1, 2).contiguous() for t in (query, key, value, g, beta)]
    query = _l2norm(query, dim=-1, eps=l2norm_eps)
    key = _l2norm(key, dim=-1, eps=l2norm_eps)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    state = torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim)
    out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim)
    for i in range(sequence_length):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i][..., None].exp()
        b_i = beta[:, :, i][..., None]

        state = state * g_i
        kv_mem = (state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * b_i
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, i] = (state * q_i.unsqueeze(-1)).sum(dim=-2)
    return out.contiguous()  # [B, S, H, V], the layout HF's recurrent returns


def kda_forward(x: torch.Tensor, weights: KdaTensors, model: Glm53FlashConfig,
                attention_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One KDA sub-layer: projections -> causal SiLU conv -> gates -> core -> o_proj.

    Port of ``Glm5NextTextLinearAttention.forward`` (full-prefill branch, no
    cache). Returns ``(output [B, S, D], internals)``.
    """
    if attention_mask is not None:
        # Tune out padding tokens before every projection (mamba-style).
        x = x * attention_mask[..., None].to(x.dtype)
    batch_size, seq_len, _ = x.shape
    heads, head_dim = model.linear_num_heads, model.linear_head_dim
    qkv = heads * head_dim
    hidden_shape = (batch_size, seq_len, heads, head_dim)

    mixed_qkv = torch.cat([x @ weights.w_q.t(), x @ weights.w_k.t(), x @ weights.w_v.t()], dim=-1)
    mixed_qkv = mixed_qkv.transpose(1, 2)
    conv_weight = torch.cat([weights.conv_q, weights.conv_k, weights.conv_v], dim=0)
    mixed_qkv = causal_conv1d_silu(mixed_qkv, conv_weight)
    mixed_qkv = mixed_qkv[:, :, -seq_len:]
    query, key, value = torch.split(mixed_qkv.transpose(1, 2), [qkv] * 3, dim=-1)
    query = query.reshape(hidden_shape)
    key = key.reshape(hidden_shape)
    value = value.reshape(hidden_shape)

    # Forget gate (safe lower-bound branch) and input gate.
    forget = (x @ weights.w_f_a.t()) @ weights.w_f_b.t() + weights.dt_bias
    g = model.linear_lower_bound * torch.sigmoid(
        weights.a_log.exp().view(1, 1, heads, 1) * forget.reshape(hidden_shape)
    )
    beta = torch.sigmoid(x @ weights.w_b.t())

    core = kda_chunk_core(query, key, value, g, beta)

    gate = ((x @ weights.w_g_a.t()) @ weights.w_g_b.t()).reshape(hidden_shape)
    # RMSNormGated with sigmoid activation, strict FP32 (weight applied in FP32).
    variance = core.pow(2).mean(-1, keepdim=True)
    normed = weights.o_norm * (core * torch.rsqrt(variance + model.rms_norm_eps))
    normed = normed * torch.sigmoid(gate)
    output = normed.reshape(batch_size, seq_len, qkv) @ weights.w_o.t()

    internals = {
        "kda_q": query, "kda_k": key, "kda_v": value,
        "kda_g": g, "kda_beta": beta, "kda_core": core,
        "kda_gate": gate, "kda_o_normed": normed.reshape(batch_size, seq_len, qkv),
    }
    return output, internals


def pooled_states(packed_states: torch.Tensor, ape: torch.Tensor, kpool: int
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compressed k-pool candidates (port of ``Glm5NextTextIndexer.get_pooled_states``).

    Pooling starts at each row's first valid token; a pool is selectable only
    when all its ``kpool`` tokens are valid. Returns ``(pool_keys [B, P, hd],
    pool_indices [B, P, kpool] with -1 holes, pool_valid [B, P])``.
    """
    head_dim = packed_states.shape[-1] // 2  # packed layout is [key | gate | valid]
    keys, gate_scores, valid_keys = torch.split(packed_states, [head_dim, head_dim, 1], dim=-1)
    valid_keys = valid_keys.bool().squeeze(-1)

    batch_size, seq_len = keys.shape[:2]
    number_of_pools = (seq_len + kpool - 1) // kpool

    first_key = torch.where(
        valid_keys.any(-1),
        valid_keys.long().argmax(-1),
        torch.full((batch_size,), seq_len, dtype=torch.long),
    )
    pool_offsets = torch.arange(number_of_pools * kpool)
    pool_offsets = pool_offsets.view(1, number_of_pools, kpool)
    pool_indices = first_key[:, None, None] + pool_offsets

    batch_idx = torch.arange(batch_size)[:, None, None]
    safe_indices = pool_indices.clamp(0, seq_len - 1)

    grouped_keys = keys[batch_idx, safe_indices]
    grouped_gate_scores = gate_scores[batch_idx, safe_indices]
    grouped_valid_keys = valid_keys[batch_idx, safe_indices]

    grouped_valid_keys = grouped_valid_keys & (pool_indices < seq_len)
    pool_valid = grouped_valid_keys.all(-1)
    pool_indices = pool_indices.masked_fill(~grouped_valid_keys, -1)

    # Learned weighted average over the tokens of each complete pool.
    logits = grouped_gate_scores + ape[None, None]
    logits = logits.masked_fill(~grouped_valid_keys[..., None], float("-inf"))
    probabilities = torch.nan_to_num(logits.softmax(dim=2))
    pool_keys = (probabilities * grouped_keys).sum(dim=2)

    keep = pool_valid.any(0)  # keep pools valid in any batch row
    return pool_keys[:, keep], pool_indices[:, keep], pool_valid[:, keep]


def append_visible_tail(topk_indices: torch.Tensor, token_visible: torch.Tensor,
                        key_valid: torch.Tensor, kpool: int) -> torch.Tensor:
    """Append the incomplete tail pool as raw indices (HF ``append_visible_tail``)."""
    max_tail_width = kpool - 1
    if max_tail_width == 0:
        return topk_indices
    batch_size, _, kv_length = token_visible.shape

    first_key = torch.where(
        key_valid.any(-1),
        key_valid.long().argmax(-1),
        torch.full((batch_size,), kv_length, dtype=torch.long),
    )
    visible_count = token_visible.long().sum(-1)
    tail_count = visible_count.remainder(kpool)
    tail_offsets = torch.arange(max_tail_width)

    tail_start = first_key[:, None] + visible_count - tail_count
    tail_indices = tail_start[..., None] + tail_offsets

    tail_valid = (tail_offsets[None, None, :] < tail_count[..., None]) & tail_indices.lt(kv_length)
    kv_idx = tail_indices.clamp(0, kv_length - 1)
    tail_visible = token_visible.gather(dim=-1, index=kv_idx)
    tail_indices = tail_indices.masked_fill(~(tail_valid & tail_visible), -1)
    return torch.cat([topk_indices, tail_indices], dim=-1)


def indexer_topk(hidden: torch.Tensor, q_resid: torch.Tensor, weights: IndexerTensors,
                 attention_mask: torch.Tensor, model: Glm53FlashConfig
                 ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """k-pool DSA indexer (port of ``Glm5NextTextIndexer.forward``, no cache).

    Returns ``(topk_indices int32 [B, S, index_topk + kpool - 1], internals)``;
    ``-1`` marks invalid slots. Valid values are raw key positions.
    """
    batch_size, seq_len = hidden.shape[:2]
    head_dim = model.index_head_dim
    kv_len = seq_len  # full prefill: no past cache

    q = (q_resid @ weights.wq_b.t()).view(batch_size, seq_len, -1, head_dim)
    k = F.layer_norm(hidden @ weights.wk.t(), (head_dim,), weights.k_norm_w, weights.k_norm_b, 1e-6)
    gate_scores = hidden @ weights.w_compress_gate.t()
    valid_channel = attention_mask.to(k.dtype)[..., None]
    packed_states = torch.cat([k, gate_scores, valid_channel], dim=-1)

    valid_keys = packed_states[..., -1].bool()
    kv_positions = torch.arange(kv_len)
    q_positions = torch.arange(seq_len)
    visible_tokens = (kv_positions[None, None, :] <= q_positions[None, :, None]) & valid_keys[:, None, :]

    pool_keys, pool_indices, pool_valid = pooled_states(packed_states, weights.ape, model.index_kpool)
    scores = torch.matmul(q, pool_keys.transpose(-1, -2).unsqueeze(1))
    scores = F.relu(scores * head_dim ** -0.5)

    # Per-head weights, summed over heads: [B, S, 1, H] @ [B, S, H, P] -> [B, S, P].
    head_weights = (hidden @ weights.w_weights.t()) * (model.index_n_heads ** -0.5)
    index_scores = torch.matmul(head_weights.unsqueeze(-2), scores).squeeze(-2)

    # A pool is selectable only if its final token is visible to the query.
    pool_end = pool_indices[..., -1].clamp(0, kv_len - 1)
    pool_visible = visible_tokens.gather(
        dim=-1, index=pool_end[:, None, :].expand(batch_size, seq_len, -1)
    )
    valid_candidates = pool_visible & pool_valid[:, None]
    index_scores = index_scores.masked_fill(~valid_candidates, _FLOAT32_MIN)

    select_k = min(model.index_topk // model.index_kpool, index_scores.shape[-1])
    selected = index_scores.topk(select_k, dim=-1).indices
    batch_idx = torch.arange(batch_size)[:, None, None]
    selected_valid = valid_candidates.gather(-1, selected)
    selected_indices = pool_indices[batch_idx, selected]

    topk_indices = selected_indices.flatten(-2)
    invalid = ~selected_valid[..., None].expand_as(selected_indices).flatten(-2)
    topk_indices = topk_indices.masked_fill(invalid, -1)

    output_width = model.index_topk
    if model.index_kpool_always_select_tail:
        topk_indices = append_visible_tail(topk_indices, visible_tokens, valid_keys, model.index_kpool)
        output_width += model.index_kpool - 1
    topk_indices = F.pad(topk_indices, (0, output_width - topk_indices.shape[-1]), value=-1)
    topk_indices = topk_indices[..., :output_width]
    topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)

    internals = {
        "pool_keys": pool_keys, "pool_indices": pool_indices,
        "pool_valid": pool_valid, "index_scores": index_scores,
    }
    return topk_indices.to(torch.int32), internals


def topk_attention_mask(topk_indices: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Additive ``[B, 1, S, kv_len]`` FP32 mask from the indexer output (eager path).

    Port of ``build_attention_mask_from_topk``: scatter the selected positions
    into a per-query count, then map ``0 -> finfo.min``.
    """
    batch_size, q_len, _ = topk_indices.shape
    topk_valid = topk_indices.ge(0) & topk_indices.lt(kv_len)
    safe_indices = topk_indices.clamp(0, kv_len - 1).to(torch.int64)
    selected_counts = torch.zeros(batch_size, q_len, kv_len, dtype=torch.int32)
    selected_counts.scatter_add_(-1, safe_indices, topk_valid.to(torch.int32))
    mask = selected_counts.ne(0)[:, None]
    return torch.where(mask, 0.0, _FLOAT32_MIN)


def mla_forward(x: torch.Tensor, weights: MlaTensors, indexer: IndexerTensors,
                attention_mask: torch.Tensor, model: Glm53FlashConfig
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One NoPE-MLA DSA sub-layer (port of ``Glm5NextTextAttention.forward``).

    Returns ``(output [B, S, D], internals)``.
    """
    batch_size, seq_len, _ = x.shape
    heads = model.num_attention_heads
    qk_dim, v_dim = model.qk_head_dim, model.v_head_dim

    q_resid = rms_norm(x @ weights.w_q_a.t(), weights.q_a_norm, model.rms_norm_eps)
    query = (q_resid @ weights.w_q_b.t()).view(batch_size, seq_len, heads, qk_dim).transpose(1, 2)

    compressed = x @ weights.w_kv_a.t()  # NoPE: no rope columns
    k_latent = rms_norm(compressed, weights.kv_a_norm, model.rms_norm_eps)

    # kv_b per head: [H, qk + v, kv_lora] split into key/value halves.
    kv_b = weights.kv_b.view(heads, qk_dim + v_dim, model.kv_lora_rank)
    k_nope = torch.einsum("bsl,hdl->bhsd", k_latent, kv_b[:, :qk_dim, :])
    value = torch.einsum("bsl,hdl->bhsd", k_latent, kv_b[:, qk_dim:, :])

    topk_indices, indexer_internals = indexer_topk(x, q_resid, indexer, attention_mask, model)
    mask = topk_attention_mask(topk_indices, kv_len=seq_len)

    scores = query @ k_nope.transpose(-1, -2) * (qk_dim ** -0.5)
    probs = torch.softmax(scores + mask, dim=-1, dtype=torch.float32)
    attn = probs @ value
    output = attn.transpose(1, 2).reshape(batch_size, seq_len, heads * v_dim) @ weights.w_o.t()

    internals = {
        "mla_q_resid": q_resid, "mla_k_latent": k_latent, "mla_topk_indices": topk_indices,
        **indexer_internals,
    }
    return output, internals


def router_forward(flat: torch.Tensor, weights: RouterTensors, model: Glm53FlashConfig
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """noaux-tc sigmoid router (port of ``Glm5NextTextTopkRouter.forward``).

    Returns ``(topk_weights [T, k], topk_indices [T, k])``; weights are the
    bias-free sigmoid scores, top-k normalized and scaled by
    ``routed_scaling_factor``.
    """
    num_experts = model.n_routed_experts
    router_logits = flat @ weights.weight.t()
    scores = router_logits.sigmoid()
    scores_for_choice = scores + weights.bias
    group_scores = (
        scores_for_choice.view(-1, model.n_group, num_experts // model.n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=model.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, model.n_group, num_experts // model.n_group)
        .reshape(-1, num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=model.num_experts_per_tok, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if model.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * model.routed_scaling_factor
    return topk_weights, topk_indices


def swiglu_limited(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """Clamped SwiGLU: clamp(gate, max=limit), clamp(up, ±limit), silu(gate)*up."""
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


def dense_mlp_forward(x: torch.Tensor, weights: DenseTensors, model: Glm53FlashConfig
                      ) -> torch.Tensor:
    """Dense MLP (port of ``Glm5NextTextMLP.forward``) with the SwiGLU clamp."""
    gate = x @ weights.gate.t()
    up = x @ weights.up.t()
    return swiglu_limited(gate, up, model.swiglu_limit) @ weights.down.t()


def moe_forward(x: torch.Tensor, router: RouterTensors, experts: ExpertProvider,
                shared: DenseTensors, model: Glm53FlashConfig
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sparse MoE (ports of ``Glm5NextTextMoE``/``TextExperts``) with expert streaming.

    Only experts the router actually hit are read from the provider, one at a
    time; shared experts run on the un-routed input. Returns
    ``(output [B, S, D], internals)``.
    """
    orig_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])
    topk_weights, topk_indices = router_forward(flat, router, model)

    routed = torch.zeros_like(flat)
    hit = torch.unique(topk_indices)
    for expert in hit.tolist():
        rows, positions = torch.where(topk_indices == expert)
        gate_up = experts.gate_up(expert)
        gate, up = (flat[rows] @ gate_up.t()).chunk(2, dim=-1)
        hidden_act = swiglu_limited(gate, up, model.swiglu_limit)
        del gate_up
        down = experts.down(expert)
        contribution = (hidden_act @ down.t()) * topk_weights[rows, positions][:, None]
        del down
        routed.index_add_(0, rows, contribution)

    shared_out = dense_mlp_forward(x, shared, model)
    output = routed.view(*orig_shape) + shared_out
    internals = {
        "router_topk_weights": topk_weights, "router_topk_indices": topk_indices,
        "moe_routed": routed.view(*orig_shape), "moe_shared": shared_out,
    }
    return output, internals


def reference_layer_forward(streams: torch.Tensor, weights: LayerRefWeights,
                            attention_mask: torch.Tensor | None = None
                            ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One decoder layer on the mHC four-stream tensor (HF decoder-layer order).

    ``streams`` is ``[B, S, hc_mult, D]`` FP32; ``attention_mask`` is ``[B, S]``
    bool (defaults to all-visible). Returns the new streams and a dict of every
    sub-layer boundary (the default dump set plus internals). Geometry comes
    from ``weights.model`` so scaled mini-checkpoint models behave identically.
    """
    model_cfg = weights.model if weights.model is not None else FLASH
    boundaries: dict[str, torch.Tensor] = {"x_hc_in": streams}
    if attention_mask is None:
        attention_mask = torch.ones(streams.shape[:2], dtype=torch.bool)
    if weights.attn_hc is None or weights.ffn_hc is None:
        raise NotImplementedError(
            f"layer {weights.layer_id} has no hyper-connections (MTP/plain residual): its "
            "wiring (eh_proj/enorm/hnorm) is outside the prefill backbone. Load it with "
            "load_layer() and drive mla_forward()/moe_forward() on the sub-layer boundaries."
        )

    residual = streams
    pre, post, comb, collapsed = mhc_mapping(streams, weights.attn_hc, model_cfg)
    x = rms_norm(collapsed, weights.input_norm, model_cfg.rms_norm_eps)
    boundaries.update(
        attn_hc_pre=pre, attn_hc_post=post, attn_hc_comb=comb, attn_norm_in=x
    )
    if weights.config.attention_kind is AttentionKind.LINEAR:
        attn_out, attn_internals = kda_forward(x, weights.kda, model_cfg, attention_mask)
    else:
        attn_out, attn_internals = mla_forward(x, weights.mla, weights.indexer,
                                               attention_mask, model_cfg)
    boundaries["attn_out"] = attn_out
    boundaries.update(attn_internals)
    streams = combine_streams(post, comb, attn_out, residual)
    boundaries["x_hc_mid"] = streams

    residual = streams
    pre, post, comb, collapsed = mhc_mapping(streams, weights.ffn_hc, model_cfg)
    x = rms_norm(collapsed, weights.post_norm, model_cfg.rms_norm_eps)
    boundaries.update(
        ffn_hc_pre=pre, ffn_hc_post=post, ffn_hc_comb=comb, ffn_norm_in=x
    )
    if weights.config.mlp_kind is MlpKind.DENSE:
        ffn_out = dense_mlp_forward(x, weights.dense, model_cfg)
        ffn_internals: dict[str, torch.Tensor] = {}
    else:
        ffn_out, ffn_internals = moe_forward(x, weights.router, weights.experts,
                                             weights.shared, model_cfg)
    boundaries["ffn_out"] = ffn_out
    boundaries.update(ffn_internals)
    streams = combine_streams(post, comb, ffn_out, residual)
    boundaries["x_hc_out"] = streams
    return streams, boundaries


# --------------------------------------------------------------------------------------
# Synthetic weights (self-test mode): true config.py geometry, no checkpoint.
# --------------------------------------------------------------------------------------


def _randn(shape: tuple[int, ...], generator: torch.Generator, std: float = 0.02
           ) -> torch.Tensor:
    return torch.randn(shape, generator=generator) * std


def synthetic_mhc(model: Glm53FlashConfig, generator: torch.Generator) -> MhcTensors:
    """mHC mapping parameters at the HF init distribution (fn~N(0,0.02), base 0, scale 1)."""
    return MhcTensors(
        fn=_randn((model.mix_hc, model.hc_dim), generator),
        base=torch.zeros(model.mix_hc),
        scale=torch.ones(3),
    )


def synthetic_kda(model: Glm53FlashConfig, generator: torch.Generator) -> KdaTensors:
    qkv = model.linear_num_heads * model.linear_head_dim
    return KdaTensors(
        w_q=_randn((qkv, model.hidden_size), generator),
        w_k=_randn((qkv, model.hidden_size), generator),
        w_v=_randn((qkv, model.hidden_size), generator),
        conv_q=torch.randn((qkv, 1, model.linear_conv_kernel_dim), generator=generator) * 0.2,
        conv_k=torch.randn((qkv, 1, model.linear_conv_kernel_dim), generator=generator) * 0.2,
        conv_v=torch.randn((qkv, 1, model.linear_conv_kernel_dim), generator=generator) * 0.2,
        w_f_a=_randn((model.linear_head_dim, model.hidden_size), generator),
        w_f_b=_randn((qkv, model.linear_head_dim), generator),
        dt_bias=torch.randn(qkv, generator=generator) - 4.0,
        a_log=torch.zeros(model.linear_num_heads),
        w_g_a=_randn((model.linear_head_dim, model.hidden_size), generator),
        w_g_b=_randn((qkv, model.linear_head_dim), generator),
        w_b=_randn((model.linear_num_heads, model.hidden_size), generator),
        o_norm=1.0 + torch.randn(model.linear_head_dim, generator=generator) * 0.05,
        w_o=_randn((model.hidden_size, qkv), generator),
    )


def synthetic_mla(model: Glm53FlashConfig, generator: torch.Generator) -> MlaTensors:
    heads, qk, v = model.num_attention_heads, model.qk_head_dim, model.v_head_dim
    return MlaTensors(
        w_q_a=_randn((model.q_lora_rank, model.hidden_size), generator),
        q_a_norm=1.0 + torch.randn(model.q_lora_rank, generator=generator) * 0.05,
        w_q_b=_randn((heads * qk, model.q_lora_rank), generator),
        w_kv_a=_randn((model.kv_lora_rank, model.hidden_size), generator),
        kv_a_norm=1.0 + torch.randn(model.kv_lora_rank, generator=generator) * 0.05,
        kv_b=_randn((heads * (qk + v), model.kv_lora_rank), generator),
        w_o=_randn((model.hidden_size, heads * v), generator),
    )


def synthetic_indexer(model: Glm53FlashConfig, generator: torch.Generator) -> IndexerTensors:
    return IndexerTensors(
        wq_b=_randn((model.index_n_heads * model.index_head_dim, model.q_lora_rank), generator),
        wk=_randn((model.index_head_dim, model.hidden_size), generator),
        k_norm_w=1.0 + torch.randn(model.index_head_dim, generator=generator) * 0.05,
        k_norm_b=torch.randn(model.index_head_dim, generator=generator) * 0.05,
        w_weights=_randn((model.index_n_heads, model.hidden_size), generator),
        w_compress_gate=_randn((model.index_head_dim, model.hidden_size), generator),
        ape=torch.zeros(model.index_kpool, model.index_head_dim),
    )


def synthetic_dense(model: Glm53FlashConfig, generator: torch.Generator,
                    intermediate: int) -> DenseTensors:
    return DenseTensors(
        gate=_randn((intermediate, model.hidden_size), generator),
        up=_randn((intermediate, model.hidden_size), generator),
        down=_randn((model.hidden_size, intermediate), generator),
    )


def synthetic_layer(model: Glm53FlashConfig, layer_id: int, seed: int) -> LayerRefWeights:
    """Synthetic layer weights at the true deployment geometry."""
    generator = torch.Generator().manual_seed(seed + 7919 * layer_id)
    config = model.layer_config(layer_id)
    layer = LayerRefWeights(
        layer_id=layer_id,
        config=config,
        model=model,
        input_norm=1.0 + torch.randn(model.hidden_size, generator=generator) * 0.05,
        post_norm=1.0 + torch.randn(model.hidden_size, generator=generator) * 0.05,
        attn_hc=synthetic_mhc(model, generator) if config.has_hyper_connections else None,
        ffn_hc=synthetic_mhc(model, generator) if config.has_hyper_connections else None,
    )
    if config.attention_kind is AttentionKind.LINEAR:
        layer.kda = synthetic_kda(model, generator)
    else:
        layer.mla = synthetic_mla(model, generator)
        layer.indexer = synthetic_indexer(model, generator)
    if config.mlp_kind is MlpKind.DENSE:
        layer.dense = synthetic_dense(model, generator, model.intermediate_size)
    else:
        layer.router = RouterTensors(
            weight=_randn((model.n_routed_experts, model.hidden_size), generator),
            bias=torch.randn(model.n_routed_experts, generator=generator) * 0.01,
        )
        layer.experts = SyntheticExpertProvider(model, seed + 31 * layer_id)
        layer.shared = synthetic_dense(model, generator, model.moe_intermediate_size)
    return layer


# --------------------------------------------------------------------------------------
# The driver: streaming 45-layer reference with per-layer activation dumps.
# --------------------------------------------------------------------------------------


@dataclass
class RefRunResult:
    """What ``PrefillReferenceDriver.run`` returns."""

    final_hidden: torch.Tensor | None      # [B, S, D] post final-norm (full range only)
    streams_out: torch.Tensor              # [B, S, hc_mult, D] leaving the last layer
    layer_summaries: list[dict[str, float]] = field(default_factory=list)
    dump_paths: dict[int, Path] = field(default_factory=dict)
    final_dump_path: Path | None = None


class PrefillReferenceDriver:
    """Host-side reference prefill driver over real or synthetic weights.

    Real mode (``checkpoint_root`` given) wraps the deployment W8A8 checkpoint
    through :class:`Checkpoint` and streams one layer at a time; the
    routed experts of a MoE layer stream one expert at a time inside the layer.
    Synthetic mode (``checkpoint_root=None``) serves deterministic synthetic
    weights at the true ``config.py`` geometry for the no-checkpoint self-test.
    """

    def __init__(self, checkpoint_root: str | Path | None = None,
                 model: Glm53FlashConfig = FLASH):
        if checkpoint_root is not None and model is not FLASH:
            raise ValueError("Checkpoint validates the deployment geometry; scaled models run checkpoint-less")
        self.model = model
        self.checkpoint_root = Path(checkpoint_root) if checkpoint_root is not None else None
        self.checkpoint = (
            Checkpoint(self.checkpoint_root)
            if self.checkpoint_root is not None
            else None
        )
        self._lm_head_cache: torch.Tensor | None = None

    # -- weight loading -----------------------------------------------------------------

    def load_layer(self, layer_id: int, *, seed: int = 0) -> LayerRefWeights:
        """Load (and dequantize) one layer, or synthesize it in checkpoint-less mode.

        Backbone layers are ``[0, num_hidden_layers)``; the MTP layer
        (``model.mtp_layer_id``) also loads when present — it has no mHC
        tensors (``has_hyper_connections=False``), so its ``attn_hc``/``ffn_hc``
        are ``None`` and its sub-layer weights feed :func:`mla_forward` /
        :func:`moe_forward` directly for the MTP diff harness.
        """
        model = self.model
        if not 0 <= layer_id <= model.mtp_layer_id:
            raise ValueError(
                f"layer_id must be in [0, {model.mtp_layer_id}] (backbone layers plus the "
                f"MTP layer), got {layer_id}"
            )
        if self.checkpoint is None:
            return synthetic_layer(model, layer_id, seed)
        checkpoint = self.checkpoint
        config = model.layer_config(layer_id)

        def read(sub_prefix: str, suffixes: Sequence[str]) -> torch.Tensor:
            return read_reference_float(checkpoint, _resolve_name(checkpoint, layer_id, sub_prefix, suffixes))

        norms = {member: read(prefix, suffixes) for member, prefix, suffixes in _NORM_TENSORS}

        def read_mhc(prefix_name: str) -> MhcTensors:
            picked = {
                member: read(prefix, suffixes)
                for member, prefix, suffixes in _MHC_TENSORS
                if member.startswith(prefix_name)
            }
            return MhcTensors(
                fn=picked[f"{prefix_name}fn"],
                base=picked[f"{prefix_name}base"],
                scale=picked[f"{prefix_name}scale"],
            )

        layer = LayerRefWeights(
            layer_id=layer_id,
            config=config,
            model=model,
            input_norm=norms["input_norm"],
            post_norm=norms["post_norm"],
            # The MTP layer carries no hc_* tensors: mHC reads are gated on the
            # layer's has_hyper_connections flag, never unconditional.
            attn_hc=read_mhc("attn_") if config.has_hyper_connections else None,
            ffn_hc=read_mhc("ffn_") if config.has_hyper_connections else None,
        )
        if config.attention_kind is AttentionKind.LINEAR:
            kda = {member: read(prefix, suffixes) for member, prefix, suffixes in _KDA_TENSORS}
            layer.kda = KdaTensors(**kda)
        else:
            mla = {member: read(prefix, suffixes) for member, prefix, suffixes in _MLA_TENSORS}
            layer.mla = MlaTensors(**mla)
            indexer = {member: read(prefix, suffixes) for member, prefix, suffixes in _INDEXER_TENSORS}
            layer.indexer = IndexerTensors(**indexer)
        if config.mlp_kind is MlpKind.DENSE:
            gate, up, down = _read_ffn(checkpoint, layer_id, "mlp.")
            layer.dense = DenseTensors(gate, up, down)
        else:
            router_weight = read("mlp.", ("gate.weight",))
            bias_name = _resolve_name(checkpoint, layer_id, "mlp.", ("gate.e_score_correction_bias",))
            layer.router = RouterTensors(router_weight, read_reference_float(checkpoint, bias_name))
            layer.experts = CheckpointExpertProvider(checkpoint, layer_id)
            gate, up, down = _read_ffn(checkpoint, layer_id, "mlp.shared_experts.")
            layer.shared = DenseTensors(gate, up, down)
        return layer

    def missing_layer_tensors(self, layer_id: int) -> list[str]:
        """Checkpoint tensors this layer still needs but no present shard carries.

        A non-empty answer means the download is incomplete and ``load_layer``
        will raise; empty means the layer can run against true weights today.
        Names are symbolic (canonical root spelling): a tensor counts as present
        when any accepted spelling resolves, mirroring :func:`_resolve_name`.
        mHC tensors are requested only when the layer has hyper-connections —
        the MTP layer carries none (``has_hyper_connections=False``).
        """
        if self.checkpoint is None:
            return []
        if not 0 <= layer_id <= self.model.mtp_layer_id:
            raise ValueError(f"layer_id must be in [0, {self.model.mtp_layer_id}], got {layer_id}")
        model = self.model
        config = model.layer_config(layer_id)
        checkpoint = self.checkpoint

        def present(sub_prefix: str, suffixes: Sequence[str]) -> bool:
            return any(
                f"{root}{layer_id}.{sub_prefix}{suffix}" in checkpoint
                for root in TEXT_LAYER_ROOTS
                for suffix in suffixes
            )

        def gap(sub_prefix: str, suffixes: Sequence[str]) -> list[str]:
            if present(sub_prefix, suffixes):
                return []
            return [f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{sub_prefix}{suffixes[0]}"]

        def ffn_gaps(base: str) -> list[str]:
            gate_up_ok = present(base, _FF_FUSED) or (present(base, _FF_GATE) and present(base, _FF_UP))
            gaps = [] if gate_up_ok else [f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{base}{_FF_FUSED[0]}"]
            if not present(base, _FF_DOWN):
                gaps.append(f"{TEXT_LAYER_ROOTS[0]}{layer_id}.{base}{_FF_DOWN[0]}")
            return gaps

        gaps: list[str] = []
        for _, prefix, suffixes in _NORM_TENSORS:
            gaps += gap(prefix, suffixes)
        if config.has_hyper_connections:
            for _, prefix, suffixes in _MHC_TENSORS:
                gaps += gap(prefix, suffixes)
        family = (
            _KDA_TENSORS
            if config.attention_kind is AttentionKind.LINEAR
            else _MLA_TENSORS + _INDEXER_TENSORS
        )
        for _, prefix, suffixes in family:
            gaps += gap(prefix, suffixes)
        if config.mlp_kind is MlpKind.DENSE:
            gaps += ffn_gaps("mlp.")
        else:
            gaps += gap("mlp.", ("gate.weight",))
            gaps += gap("mlp.", ("gate.e_score_correction_bias",))
            for expert in range(model.n_routed_experts):
                gaps += ffn_gaps(f"mlp.experts.{expert}.")
            gaps += ffn_gaps("mlp.shared_experts.")
        return gaps

    # -- host tables --------------------------------------------------------------------

    def embed_rows(self, token_ids: torch.Tensor, *, generator: torch.Generator | None = None
                   ) -> torch.Tensor:
        """Embedding rows ``[B, S, D]`` FP32: checkpoint row-gather or synthetic."""
        ids = _as_batch_seqs(token_ids)
        if self.checkpoint is not None:
            name = _resolve_top_name(self.checkpoint, ("embed_tokens.weight",))
            return gather_rows(self.checkpoint, name, ids).to(torch.float32)
        gen = generator if generator is not None else torch.Generator().manual_seed(1234)
        rows = torch.randn(ids.numel(), self.model.hidden_size, generator=gen) * 0.02
        return rows.view(*ids.shape, self.model.hidden_size)

    def final_norm_weight(self) -> torch.Tensor:
        """The final RMSNorm weight ``[D]`` (ones in synthetic mode)."""
        if self.checkpoint is not None:
            name = _resolve_top_name(self.checkpoint, ("norm.weight",))
            return read_reference_float(self.checkpoint, name)
        return torch.ones(self.model.hidden_size)

    def _lm_head(self) -> torch.Tensor:
        if self._lm_head_cache is None:
            if self.checkpoint is None:
                raise RuntimeError("lm_head top-1 needs a real checkpoint")
            name = _resolve_top_name(self.checkpoint, ("lm_head.weight",))
            self._lm_head_cache = self.checkpoint.read(name).to(torch.bfloat16).contiguous()
        return self._lm_head_cache

    def lm_head_top1(self, hidden: torch.Tensor, k: int = 1, *, chunk: int = 8192
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Host LM-head top-k: chunked FP32 matmul over the full ``[vocab, D]`` table.

        Returns ``(values, indices)``, each ``[N, k]``, rows of ``hidden`` ``[N, D]``.
        """
        head = self._lm_head()
        rows = hidden.reshape(-1, hidden.shape[-1]).to(torch.float32)
        best_values = torch.full((rows.shape[0], k), -torch.inf)
        best_indices = torch.zeros((rows.shape[0], k), dtype=torch.int64)
        for start in range(0, head.shape[0], chunk):
            logits = (head[start : start + chunk].to(torch.float32)) @ rows.t()  # [chunk, N]
            candidates = torch.cat([best_values, logits.t()], dim=1)
            indices = torch.cat(
                [best_indices, torch.arange(start, start + logits.shape[0])[None].expand(rows.shape[0], -1)],
                dim=1,
            )
            order = candidates.argsort(dim=1, descending=True)[:, :k]
            best_values = candidates.gather(1, order)
            best_indices = indices.gather(1, order)
        return best_values.contiguous(), best_indices.contiguous()

    # -- the streaming run ----------------------------------------------------------------

    def run(
        self,
        token_ids: torch.Tensor,
        *,
        layers: tuple[int, int] | None = None,
        attention_mask: torch.Tensor | None = None,
        initial_streams: torch.Tensor | None = None,
        dump_dir: Path | str | None = None,
        dump_layers: set[int] | None = None,
        dump_keys: Sequence[str] | None = None,
        dump_internals: bool = False,
        dump_dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> RefRunResult:
        """Stream the requested backbone layers on a prompt.

        ``layers`` is an inclusive-exclusive ``(start, stop)`` over
        ``[0, num_hidden_layers)``; default runs the whole backbone. Entering
        streams come from the embedding (all ``hc_mult`` streams equal) unless
        ``initial_streams [B, S, hc_mult, D]`` is given. When ``stop`` reaches
        the backbone depth the final collapse (mean over streams) and final
        RMSNorm run, and ``final_hidden`` is set. Layer ``L`` is dumped to
        ``dump_dir/layer_{L:02d}.pt`` when ``dump_layers`` selects it (default:
        every executed layer); ``dump_keys`` filters the payload (default:
        :data:`DEFAULT_DUMP_KEYS`, plus internals with ``dump_internals``).
        """
        model = self.model
        ids = _as_batch_seqs(token_ids)
        batch_size, seq_len = ids.shape
        start, stop = layers if layers is not None else (0, model.num_hidden_layers)
        if not 0 <= start < stop <= model.num_hidden_layers:
            raise ValueError(f"layers must satisfy 0 <= start < stop <= {model.num_hidden_layers}")

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask = attention_mask.to(torch.bool)

        if initial_streams is None:
            embedded = self.embed_rows(ids, generator=torch.Generator().manual_seed(seed))
            streams = embedded.unsqueeze(2).expand(-1, -1, model.hc_mult, -1).contiguous()
        else:
            streams = initial_streams.to(torch.float32).contiguous()

        keys = tuple(dump_keys) if dump_keys is not None else DEFAULT_DUMP_KEYS
        if dump_internals:
            keys = tuple(dict.fromkeys((*keys, *INTERNAL_DUMP_KEYS)))
        dump_dir_path = Path(dump_dir) if dump_dir is not None else None
        if dump_dir_path is not None:
            dump_dir_path.mkdir(parents=True, exist_ok=True)

        result = RefRunResult(final_hidden=None, streams_out=streams)
        for layer_id in range(start, stop):
            weights = self.load_layer(layer_id, seed=seed)
            streams, boundaries = reference_layer_forward(streams, weights, attention_mask)
            del weights
            result.layer_summaries.append(
                {
                    "layer": layer_id,
                    **{
                        key: float(boundaries[key].abs().max())
                        for key in ("x_hc_in", "attn_out", "x_hc_mid", "ffn_out", "x_hc_out")
                    },
                }
            )
            if dump_dir_path is not None and (dump_layers is None or layer_id in dump_layers):
                payload: dict[str, object] = {
                    "layer_id": layer_id,
                    "attention_kind": _attention_kind_name(layer_id, model),
                    "mlp_kind": _mlp_kind_name(layer_id, model),
                }
                payload.update(
                    {key: boundaries[key].to(dump_dtype) for key in keys if key in boundaries}
                )
                path = dump_dir_path / f"layer_{layer_id:02d}.pt"
                torch.save(payload, path)
                result.dump_paths[layer_id] = path
            del boundaries

        result.streams_out = streams
        if stop == model.num_hidden_layers:
            collapsed = streams.mean(dim=2)
            result.final_hidden = rms_norm(collapsed, self.final_norm_weight(), model.rms_norm_eps)
            if dump_dir_path is not None:
                result.final_dump_path = dump_dir_path / "final.pt"
                torch.save(
                    {
                        "final_hidden": result.final_hidden.to(dump_dtype),
                        "streams_mean_pre_norm": collapsed.to(dump_dtype),
                        "token_ids": ids,
                        "layer_range": (start, stop),
                        "final_norm_eps": model.rms_norm_eps,
                    },
                    result.final_dump_path,
                )
        return result


def _attention_kind_name(layer_id: int, model: Glm53FlashConfig) -> str:
    return model.layer_config(layer_id).attention_kind.value


def _mlp_kind_name(layer_id: int, model: Glm53FlashConfig) -> str:
    return model.layer_config(layer_id).mlp_kind.value


# --------------------------------------------------------------------------------------
# Self-test: synthetic weights at the true config.py geometry, CPU only.
# --------------------------------------------------------------------------------------


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str,
                  atol: float = 1e-4, rtol: float = 1e-4) -> float:
    """Assert against the golden comparator; return the max difference."""
    from golden import ratio_allclose  # noqa: PLC0415 - matches the model-local golden imports

    passed, message = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=0.0)(
        actual,
        expected,
        actual_outputs={},
        expected_outputs={},
        inputs={},
        atol=atol,
        rtol=rtol,
    )
    if not passed:
        raise AssertionError(f"{what}:{message}")
    return float((actual - expected).abs().max())


def _test_mhc(model: Glm53FlashConfig) -> None:
    torch.manual_seed(11)
    hc, hidden = model.hc_mult, model.hidden_size
    streams = torch.randn(2, 5, hc, hidden) * 0.5
    weights = synthetic_mhc(model, torch.Generator().manual_seed(7))
    pre, post, comb, collapsed = mhc_mapping(streams, weights, model)

    assert pre.shape == (2, 5, hc) and post.shape == (2, 5, hc)
    assert comb.shape == (2, 5, hc, hc) and collapsed.shape == (2, 5, hidden)
    assert float(pre.min()) >= model.hc_eps and float(pre.max()) <= 1.0 + model.hc_eps + 1e-5
    assert 0.0 <= float(post.min()) and float(post.max()) <= 2.0 + 1e-5
    # Sinkhorn: columns are exact by construction (final op normalizes them);
    # rows only converge linearly, and the shipped 20 iterations on a 4x4 leave
    # O(1e-2) row deviation — verified against the verbatim HF snippet.
    row_error = (comb.sum(-1) - 1).abs().max()
    col_error = (comb.sum(-2) - 1).abs().max()
    assert float(row_error) < 0.05 and float(col_error) < 1e-3, (float(row_error), float(col_error))
    _assert_close(collapsed, torch.einsum("bsh,bshd->bsd", pre, streams), "mHC collapse")
    # Combine against a per-stream loop: out_h = post_h * sublayer + sum_k comb[k,h] * res_k.
    sublayer = torch.randn(2, 5, hidden)
    manual = torch.stack(
        [
            post[..., h, None] * sublayer
            + sum(comb[..., k, h, None] * streams[..., k, :] for k in range(hc))
            for h in range(hc)
        ],
        dim=2,
    )
    _assert_close(combine_streams(post, comb, sublayer, streams), manual, "mHC combine formula")
    print(f"  mHC mapping: shapes, ranges, sinkhorn rows {float(row_error):.2e}")


def _test_kda_cores(model: Glm53FlashConfig) -> None:
    torch.manual_seed(13)
    heads, head_dim = model.linear_num_heads, model.linear_head_dim
    b, s = 2, 37  # deliberately not chunk-aligned
    q = torch.randn(b, s, heads, head_dim)
    k = torch.randn(b, s, heads, head_dim) * 2
    v = torch.randn(b, s, heads, head_dim)
    g = -5.0 * torch.sigmoid(torch.randn(b, s, heads, head_dim))
    beta = torch.sigmoid(torch.randn(b, s, heads))

    chunked = kda_chunk_core(q, k, v, g, beta)
    recurrent = kda_recurrent_core(q, k, v, g, beta)
    assert chunked.shape == (b, s, heads, head_dim)
    core_diff = _assert_close(chunked, recurrent, "KDA chunk vs recurrent", atol=1e-3, rtol=1e-3)

    # Causal conv against an unfold-based reference (cross-correlation, left pad).
    channels = heads * head_dim
    kernel = model.linear_conv_kernel_dim
    x = torch.randn(1, channels, s)
    weight = torch.randn(channels, 1, kernel) * 0.2
    out = causal_conv1d_silu(x, weight)
    padded = F.pad(x, (kernel - 1, 0))
    expected = F.silu(torch.einsum("ctk,ck->ct", padded[0].unfold(-1, kernel, 1), weight[:, 0, :]))
    conv_diff = _assert_close(out[0], expected, "causal conv1d")
    print(f"  KDA cores: chunk==recurrent max|d|={core_diff:.2e} (S={s}); conv max|d|={conv_diff:.2e}")


def _test_indexer(model: Glm53FlashConfig) -> None:
    torch.manual_seed(17)
    b, s = 2, 18  # 4 full pools + a 2-token tail per row
    hidden = torch.randn(b, s, model.hidden_size) * 0.5
    q_resid = torch.randn(b, s, model.q_lora_rank) * 0.5
    weights = synthetic_indexer(model, torch.Generator().manual_seed(5))
    mask = torch.ones(b, s, dtype=torch.bool)
    mask[1, 15:] = False  # second row carries right padding

    topk, internals = indexer_topk(hidden, q_resid, weights, mask, model)
    width = model.index_topk + model.index_kpool - 1
    assert topk.dtype == torch.int32 and topk.shape == (b, s, width)

    valid = topk.ge(0)
    assert int(topk.min()) >= -1 and int(topk.max()) < s
    # Causality: every selected key position is <= its query position.
    positions = torch.arange(s)
    assert bool((topk <= positions[None, :, None]).all()), "indexer selected a future key"
    # Padded query rows must be fully masked.
    assert not bool(valid[1, 15:].any()), "padded query rows must be -1"
    # Pooling starts at slot 0; with S=18 exactly 4 complete pools exist.
    assert int(internals["pool_valid"].any(0).sum()) == s // model.index_kpool
    # Query 17 of row 0 selects every pool: leading select_k*kpool slots hold the
    # pool quartets [4j, 4j+1, 4j+2, 4j+3]; the appended tail covers the last
    # S % kpool = 2 tokens (16, 17), then padding.
    row = topk[0, 17]
    select_k = min(model.index_topk // model.index_kpool, internals["pool_keys"].shape[1])
    pool_part = row[: select_k * model.index_kpool]
    assert bool(pool_part.ge(0).all())
    quartets = pool_part.reshape(-1, model.index_kpool)
    assert bool((quartets[:, 0] % model.index_kpool == 0).all())
    assert bool((quartets[:, 1:] - quartets[:, :-1] == 1).all()), f"quartets broken: {quartets}"
    tail = row[select_k * model.index_kpool : select_k * model.index_kpool + model.index_kpool - 1]
    assert tail.tolist() == [s - 2, s - 1, -1], f"tail is {tail.tolist()}, want [16, 17, -1]"
    print(f"  indexer: width {width}, causal, padding-masked, {select_k} quartet pools, tail "
          f"{tail.tolist()}")


def _test_mla(model: Glm53FlashConfig) -> None:
    torch.manual_seed(19)
    b, s = 2, 18
    x = torch.randn(b, s, model.hidden_size) * 0.5
    weights = synthetic_mla(model, torch.Generator().manual_seed(3))
    indexer = synthetic_indexer(model, torch.Generator().manual_seed(4))
    mask = torch.ones(b, s, dtype=torch.bool)
    out, internals = mla_forward(x, weights, indexer, mask, model)
    assert out.shape == (b, s, model.hidden_size) and torch.isfinite(out).all()
    assert internals["mla_q_resid"].shape == (b, s, model.q_lora_rank)
    assert internals["mla_topk_indices"].shape[-1] == model.index_topk + model.index_kpool - 1
    print("  MLA: shapes finite, NoPE latent path, topk width")


def _test_router_and_moe(model: Glm53FlashConfig) -> None:
    torch.manual_seed(23)
    tokens = 5
    router = RouterTensors(
        weight=torch.randn(model.n_routed_experts, model.hidden_size) * 0.02,
        bias=torch.randn(model.n_routed_experts) * 0.01,
    )
    flat = torch.randn(tokens, model.hidden_size) * 0.5
    topk_weights, topk_indices = router_forward(flat, router, model)
    assert topk_indices.shape == (tokens, model.num_experts_per_tok)
    assert bool((topk_indices >= 0).all()) and bool((topk_indices < model.n_routed_experts).all())
    for row in range(tokens):
        assert len(set(topk_indices[row].tolist())) == model.num_experts_per_tok
        _assert_close(
            topk_weights[row].sum(), torch.tensor(model.routed_scaling_factor),
            "router weights must renormalize to routed_scaling_factor", atol=1e-4,
        )
    # Manual noaux-tc recompute: bias steers choice, weights stay bias-free.
    scores = (flat @ router.weight.t()).sigmoid()
    biased = scores + router.bias
    expected_idx = biased.topk(model.num_experts_per_tok, dim=-1, sorted=False).indices
    assert set(expected_idx[0].tolist()) == set(topk_indices[0].tolist())
    expected_w = scores[0].gather(0, topk_indices[0])
    expected_w = expected_w / (expected_w.sum() + 1e-20) * model.routed_scaling_factor
    _assert_close(topk_weights[0], expected_w, "router weight formula")

    # MoE output vs a manual per-token recompute (expert streaming correctness).
    shared = synthetic_dense(model, torch.Generator().manual_seed(9), model.moe_intermediate_size)
    provider = SyntheticExpertProvider(model, 77)
    x = torch.randn(2, 6, model.hidden_size) * 0.5
    out, internals = moe_forward(x, router, provider, shared, model)
    topk_weights, topk_indices = internals["router_topk_weights"], internals["router_topk_indices"]
    flat_x = x.reshape(-1, model.hidden_size)
    for token in (0, 7):
        manual = dense_mlp_forward(x.reshape(-1, model.hidden_size)[token], shared, model)
        for position, expert in enumerate(topk_indices[token].tolist()):
            gate_up = provider.gate_up(expert)
            gate, up = gate_up.chunk(2, dim=0)
            hidden_act = swiglu_limited(
                flat_x[token] @ gate.t(), flat_x[token] @ up.t(), model.swiglu_limit
            )
            manual = manual + (hidden_act @ provider.down(expert).t()) * topk_weights[token, position]
        _assert_close(out.reshape(-1, model.hidden_size)[token], manual, "MoE manual recompute",
                      atol=1e-4, rtol=1e-4)
    print("  router+MoE: noaux-tc formula, renormalized weights, expert streaming")


def _test_dense_mlp(model: Glm53FlashConfig) -> None:
    torch.manual_seed(29)
    weights = synthetic_dense(model, torch.Generator().manual_seed(2), model.intermediate_size)
    x = torch.randn(2, 9, model.hidden_size) * 0.5
    out = dense_mlp_forward(x, weights, model)
    gate = (x @ weights.gate.t()).clamp(max=model.swiglu_limit)
    up = (x @ weights.up.t()).clamp(-model.swiglu_limit, model.swiglu_limit)
    _assert_close(out, (F.silu(gate) * up) @ weights.down.t(), "dense MLP with clamps")
    over = torch.full((1, 1, model.intermediate_size), 100.0)
    clamped = swiglu_limited(over, over, model.swiglu_limit)
    assert float(clamped.abs().max()) < 100.0
    print("  dense MLP: manual equivalence, SwiGLU clamp")


def _test_layers_end_to_end(model: Glm53FlashConfig, tmp_root: Path) -> None:
    torch.manual_seed(31)
    b, s = 2, 17
    driver = PrefillReferenceDriver()  # synthetic mode
    mask = torch.ones(b, s, dtype=torch.bool)

    for layer_id, attention, mlp in ((0, "linear_attention", "dense"), (4, "linear_attention", "sparse"),
                                     (3, "deepseek_sparse_attention", "sparse")):
        weights = driver.load_layer(layer_id)
        assert weights.config.attention_kind.value == attention
        assert weights.config.mlp_kind.value == mlp
        embedded = torch.randn(b, s, model.hc_mult, model.hidden_size) * 0.5
        streams_a, bounds_a = reference_layer_forward(embedded, weights, mask)
        streams_b, _ = reference_layer_forward(embedded, weights, mask)
        assert streams_a.shape == (b, s, model.hc_mult, model.hidden_size)
        assert torch.equal(streams_a, streams_b), f"layer {layer_id} is nondeterministic"
        assert torch.isfinite(streams_a).all()
        for key in DEFAULT_DUMP_KEYS:
            assert key in bounds_a, f"layer {layer_id} dump key {key} missing"
        # Wiring: recompute the final combine from the dumped boundaries.
        recomputed = combine_streams(
            bounds_a["ffn_hc_post"], bounds_a["ffn_hc_comb"], bounds_a["ffn_out"], bounds_a["x_hc_mid"]
        )
        _assert_close(streams_a, recomputed, f"layer {layer_id} combine wiring", atol=1e-4, rtol=1e-4)
        print(f"  layer {layer_id:2d} ({attention}/{mlp}): deterministic, wired, finite")

    # Driver run over layers 0..4 with internals: dump-chain continuity and shapes.
    with tempfile.TemporaryDirectory(dir=str(tmp_root)) as tmp:
        token_ids = torch.randint(0, 1000, (b, s))
        result = driver.run(
            token_ids, layers=(0, 5), dump_dir=tmp, dump_internals=True,
        )
        assert result.final_hidden is None, "mid-range stop must not produce final hidden"
        assert set(result.dump_paths) == {0, 1, 2, 3, 4}
        payloads = {L: torch.load(path, weights_only=False) for L, path in result.dump_paths.items()}
        for layer_id, payload in payloads.items():
            assert payload["layer_id"] == layer_id
            assert payload["x_hc_in"].shape == (b, s, model.hc_mult, model.hidden_size)
            assert payload["x_hc_out"].shape == (b, s, model.hc_mult, model.hidden_size)
        # Stream continuity: each layer's dumped input is the previous output.
        for layer_id in range(4):
            assert torch.equal(payloads[layer_id]["x_hc_out"], payloads[layer_id + 1]["x_hc_in"])
        assert payloads[3]["mla_topk_indices"].shape[-1] == model.index_topk + model.index_kpool - 1
        assert payloads[0]["kda_core"].shape == (
            b, s, model.linear_num_heads, model.linear_head_dim,
        )
        assert payloads[4]["router_topk_weights"].shape == (b * s, model.num_experts_per_tok)
        assert "router_topk_weights" not in payloads[1]  # layer 1 is a dense-MLP layer

    # Whole-backbone synthetic run (final-norm path + final.pt roundtrip).
    with tempfile.TemporaryDirectory(dir=str(tmp_root)) as tmp:
        result = driver.run(torch.randint(0, 1000, (1, 8)), dump_dir=tmp, dump_layers=set())
        assert result.final_hidden.shape == (1, 8, model.hidden_size)
        assert torch.isfinite(result.final_hidden).all()
        final_dump = torch.load(result.final_dump_path, weights_only=False)
        _assert_close(
            final_dump["final_hidden"].to(torch.float32), result.final_hidden, "final dump roundtrip",
            atol=1e-4,
        )
        manual = rms_norm(
            result.streams_out.mean(dim=2), driver.final_norm_weight(), model.rms_norm_eps
        )
        _assert_close(result.final_hidden, manual, "final norm recompute", atol=1e-4)
    print("  driver.run: dump-chain continuity, internal keys, final norm + final.pt")


def _test_scaled_model(model: Glm53FlashConfig) -> None:
    """The mini-checkpoint consumption path: a scaled 4-layer schedule.

    ``layer_ckpt_l0l3`` (layers 0-3, num_hidden_layers=4, first_k_dense_replace=3)
    is the first real single-layer-diff weight source; this exercises the same
    model-override mechanics without needing its files on disk yet.
    """
    scaled = replace(model, num_hidden_layers=4, first_k_dense_replace=3)
    assert scaled.kda_layer_ids == (0, 1, 2) and scaled.dsa_layer_ids == (3,)
    assert scaled.layer_config(3).mlp_kind is MlpKind.SPARSE

    driver = PrefillReferenceDriver(model=scaled)  # synthetic weights, scaled schedule
    with tempfile.TemporaryDirectory() as tmp:
        result = driver.run(torch.randint(0, 1000, (1, 8)), dump_dir=tmp)
        assert len(result.layer_summaries) == 4
        assert result.final_hidden.shape == (1, 8, scaled.hidden_size)
        assert result.dump_paths[3].name == "layer_03.pt"
        payload = torch.load(result.dump_paths[3], weights_only=False)
        assert payload["attention_kind"] == "deepseek_sparse_attention"
        assert payload["mlp_kind"] == "sparse"
    # Layers past the MTP ordinal stay rejected under the scaled model.
    try:
        driver.load_layer(scaled.mtp_layer_id + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("scaled model must reject layers past the MTP ordinal")
    # ...and scaled_model_from_config round-trips a deployment-style config.json.
    with tempfile.TemporaryDirectory() as tmp:
        text = {
            "num_hidden_layers": 4,
            "first_k_dense_replace": 3,
            "hidden_size": model.hidden_size,
            "linear_attn_config": {"num_heads": model.linear_num_heads},
        }
        (Path(tmp) / "config.json").write_text(json.dumps({"text_config": text}))
        derived = scaled_model_from_config(tmp)
        assert derived.num_hidden_layers == 4 and derived.first_k_dense_replace == 3
        assert derived.hidden_size == model.hidden_size and derived.vocab_size == model.vocab_size

    # The MTP layer (layer 4 of the scaled schedule) loads without mHC tensors
    # and refuses the backbone wiring instead of demanding absent hc_* names.
    mtp = driver.load_layer(scaled.mtp_layer_id)
    assert mtp.attn_hc is None and mtp.ffn_hc is None
    assert mtp.config.has_hyper_connections is False
    assert driver.missing_layer_tensors(scaled.mtp_layer_id) == []
    try:
        reference_layer_forward(torch.randn(1, 4, scaled.hc_mult, scaled.hidden_size), mtp)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("MTP layer must reject the hyper-connection wiring")
    print("  scaled model: 4-layer schedule, DSA/MoE layer 3, config.json override, MTP gating")


def _test_lm_head_top1(model: Glm53FlashConfig) -> None:
    """Chunked top-k merge against one direct matmul (fake 20000-row head)."""
    torch.manual_seed(41)
    driver = PrefillReferenceDriver()
    driver._lm_head_cache = torch.randn(20000, model.hidden_size, dtype=torch.bfloat16) * 0.02
    hidden = torch.randn(7, model.hidden_size)
    values, indices = driver.lm_head_top1(hidden, k=3)  # 20000 rows -> 3 merge chunks
    reference = driver._lm_head().to(torch.float32) @ hidden.t()
    top = reference.topk(3, dim=0)
    assert indices.shape == (7, 3)
    assert torch.equal(indices, top.indices.t()), "top-k indices disagree with direct matmul"
    _assert_close(values, top.values.t(), "lm_head top-k merge values", atol=1e-3, rtol=1e-4)
    print("  lm_head top1: chunked merge == direct matmul (k=3, 3 chunks)")


def _real_checkpoint_smoke(root: str) -> None:
    """Against the real (possibly still downloading) W8A8 snapshot."""
    driver = PrefillReferenceDriver(root)
    print(f"  checkpoint opened: {len(driver.checkpoint)} indexed tensors")
    # Dequant one real W8A8 tensor through this module's reader and verify the
    # scale multiply against the raw INT8 payload directly.
    name = "model.language_model.layers.0.mlp.down_proj.weight"
    if name in driver.checkpoint:
        dequant = read_reference_float(driver.checkpoint, name)
        raw = driver.checkpoint.read(name).to(torch.float32)
        scale = (
            driver.checkpoint.read(name[: -len(".weight")] + ".weight_scale").float().reshape(-1)
        )
        _assert_close(dequant, raw * scale[:, None], "real W8A8 dequant", atol=0, rtol=0)
        print(f"  real dequant exact: {name} {tuple(dequant.shape)}")
    for layer_id in (0, 3):
        missing = driver.missing_layer_tensors(layer_id)
        verdict = "loadable" if not missing else f"{len(missing)} tensors pending"
        print(f"  layer {layer_id} real-weight coverage: {verdict}")


def run_self_tests(model: Glm53FlashConfig = FLASH, checkpoint_root: str | None = None) -> None:
    """CPU self-test battery (synthetic weights at the true geometry)."""
    print(f"prefill_ref self-test: hidden={model.hidden_size} layers={model.num_hidden_layers} "
          f"heads={model.num_attention_heads}/{model.linear_num_heads} experts={model.n_routed_experts}")
    _test_mhc(model)
    _test_kda_cores(model)
    _test_indexer(model)
    _test_mla(model)
    _test_router_and_moe(model)
    _test_dense_mlp(model)
    _test_lm_head_top1(model)
    _test_layers_end_to_end(model, Path("."))
    _test_scaled_model(model)
    if checkpoint_root is not None:
        print(f"real-checkpoint smoke: {checkpoint_root}")
        _real_checkpoint_smoke(checkpoint_root)
    print("prefill_ref self-test: ALL PASS")


def main() -> None:
    """Run the CPU self-test and optionally probe a real W8A8 checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="optional real W8A8 checkpoint root for a read/dequant/coverage smoke",
    )
    args = parser.parse_args()
    # Tiny-tensor CPU math parallelizes badly past a few dozen threads (this
    # host defaults to 320); cap the self-test only, library imports stay pure.
    torch.set_num_threads(min(32, torch.get_num_threads()))
    run_self_tests(checkpoint_root=args.checkpoint)


if __name__ == "__main__":
    main()
