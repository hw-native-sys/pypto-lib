# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared helpers for the DeepSeek-V4-Pro drivers.

Previously ``utils/golden_fwd.py`` and ``utils/weights_flash.py``. Flattened
into one module so this directory matches the other model dirs
(``deepseek_v4_flash_mtp/utils.py``, ``deepseek_v4_flash_dspark/utils.py``),
and so the daily sweeps skip it. Those sweeps decide what is a case by
grepping each file for a script-entry guard, and the converter's guard made it
one; the converter now runs through :func:`main` instead, reading ``sys.argv``
unchanged. That grep is a plain substring match over the whole file, so do not
spell the guard's dunder name anywhere in this module, comments included.

**Import order matters.** This module sets ``config.MOE_TOKENS`` and imports
``moe`` plus the prefill attention modules at import time. A decode driver
must import ``moe`` before it imports anything from here — every current call
site does, because they all import from this module inside a function body
rather than at module level. Keep it that way.

Full-network torch golden (packed-prefill forward)
--------------------------------------------------

``golden_prefill_fwd(tensors)`` replays ``prefill_fwd.l3_prefill_fwd`` on host
by chaining the leaf goldens exactly the way the device program composes the
leaf kernels:

- ``input_pack.golden_pack_x_hc`` builds the layer-0 hidden state per rank.
- Per layer, the preset's attention kind runs per rank
  (``golden_prefill_attention_{swa,hca,csa}``) followed by the cross-rank
  ``moe.golden_moe`` whose ``x_next`` becomes the next layer's ``x_hc``.
- The trailing layer's MoE output is written straight into
  ``pre_hc_hidden_out``; ``hc_head`` + the final ``rms_norm`` produce
  ``hidden_out``, and a distributed-LM-head replay fills ``logits`` and
  ``sampled_ids``.

The *tensors* dict is the golden-harness scratch dict for
``prefill_fwd.build_tensor_specs()``: every tensor is ``[N_RANKS, ...]``
rank-stacked, per-FWD-layer weights and caches are packed along dim 1 by
FWD-layer id, and the CSA/HCA-compact stacks are packed by per-kind order
(ascending layer id of that kind). All cache slices handed to the leaf goldens
are torch views of the stacked tensors, so their in-place slot updates land in
the output cache tensors that validation reads back.

Real DeepSeek-V4-Flash checkpoint loader
----------------------------------------

Converts the HuggingFace-style hybrid MXFP4-MXFP8 checkpoint (43 layers,
256 routed experts, ``expert_dtype=fp4`` + block-FP8 attention linears) into
the exact host-tensor ABI ``prefill_fwd.py`` / ``decode_fwd.py`` consume:

- FP8 e4m3 weights (128x128-block UE8M0 scales) are dequantized, then either
  kept BF16 (``wq_a``, ``wkv``, ``wo_a``) or re-quantized to the kernels'
  W8A8 form: symmetric INT8 with a per-output-channel FP32 scale (amax/127,
  the same round -> clamp +-127 -> fp16 -> int8 chain as the fixtures).
- FP4 e2m1 routed-expert weights (packed two-per-byte, per-32-group UE8M0
  scales along the input dim) are unpacked, dequantized and re-quantized to
  the same INT8 + per-output-channel FP32 scale form.
- Per-layer tensors are stacked along dim 1 exactly like
  ``_make_stacked_spec`` (FWD stacks by model layer id 0..42; CSA/HCA stacks
  by kind order = ascending layer id of that compress-ratio kind), sharded
  per EP rank for the routed experts (rank ``r`` owns global experts
  ``[r*N_LOCAL, (r+1)*N_LOCAL)``) and per TP rank for ``lm_head_weight``
  (rank ``r`` reads vocab shard ``r % TP``), and replicated across ranks
  otherwise.

Synthesized inputs (RoPE tables, ``csa_hadamard_idx``, caches, per-step
metadata) keep their fixture initializers and are not touched here.

Usage — one-time offline conversion (recommended; the routed experts alone
re-quantize ~280 GB), then run the drivers against the cache::

    PYTHONPATH=.:models/deepseek_v4_pro python -c 'import utils; utils.main()' \\
        --variant flash --ep 8 --tp 2 \\
        --ckpt /path/to/DeepSeek-V4-Flash --out build_output/flash_weights_ep8_tp2
    python models/deepseek_v4_pro/prefill_fwd.py --variant flash --ep 8 --tp 2 \\
        -p a5 -d 0,1,2,3,4,5,6,7 --weights build_output/flash_weights_ep8_tp2

``--weights`` also accepts the raw checkpoint directory directly (detected by
``model.safetensors.index.json``); every weight is then converted on the fly
while the harness builds its inputs.
"""

# The prefill path runs PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe, which freezes recv shapes and derives RECV_MAX at import time (mirrors
# prefill_fwd.py; a no-op when prefill_fwd already imported moe).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS

import argparse
import json
import mmap
import struct
import warnings
from pathlib import Path
from typing import Callable

import torch

# Import moe first: it applies the EP/active-preset override before the
# attention modules bake config-derived shapes (matches prefill_fwd's order).
# `D` comes from moe (moe.py defines it as M.hidden_size), so the converter
# half no longer redefines it.
from moe import D, HC_MULT, N_RANKS, T, golden_moe
from config import (
    ACTIVE as M,
    ACTIVE as MODEL_CONFIG,
    ACTIVE_BASE,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
)
from prefill_attention_swa import golden_prefill_attention_swa
from prefill_attention_hca import HCA_STATE_BLOCK_SIZE, golden_prefill_attention_hca
from prefill_attention_csa import (
    CSA_STATE_BLOCK_SIZE,
    INNER_STATE_BLOCK_SIZE,
    golden_prefill_attention_csa,
)
from input_pack import golden_pack_x_hc
from hc_head import golden_hc_head_rows
from lm_head import golden_lm_head_all_ranks
from rmsnorm import golden_rms_norm

from golden import ratio_allclose

# ===========================================================================
# Real DeepSeek-V4-Flash checkpoint loader
# ===========================================================================
# ---------------------------------------------------------------------------
# Model-layer geometry (mirrors prefill_fwd/decode_fwd stacking rules).
# ---------------------------------------------------------------------------
NUM_LAYERS = M.num_hidden_layers
FWD_RATIOS = M.compress_ratios[:NUM_LAYERS]
# Kind-order slot k of the CSA/HCA stacks maps to the k-th layer of that
# compress-ratio kind in ascending layer-id order (lead layers first, loop
# pairs next, trailing CSA last — ascending id gives exactly that order).
CSA_LAYERS = [i for i, r in enumerate(FWD_RATIOS) if r == 4]
HCA_LAYERS = [i for i, r in enumerate(FWD_RATIOS) if r == 128]

O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
MOE_INTER = M.moe_intermediate_size
# The FULL deployment expert count. Read from the immutable ACTIVE_BASE, never
# from ACTIVE: moe.py shrinks config.ACTIVE.n_routed_experts to 32*EP at import,
# so in driver context ACTIVE already carries the reduced routing space.
N_EXPERTS_FULL = ACTIVE_BASE.n_routed_experts
VOCAB = M.vocab_size

FP8_BLOCK = 128   # weight_block_size for e4m3 weights
FP4_GROUP = 32    # scale group along the input dim for e2m1 experts

# fp4 e2m1 value table for nibble indices 0..15 (bit 3 = sign).
_FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.uint8,  # decoded via _e8m0_to_fp32
    "I8": torch.int8,
    "I64": torch.int64,
}


class FlashCheckpoint:
    """Zero-copy reader for the sharded DeepSeek-V4-Flash safetensors checkpoint."""

    def __init__(self, ckpt_dir: str | Path) -> None:
        self.dir = Path(ckpt_dir)
        index_path = self.dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"not a checkpoint dir (missing {index_path})")
        with open(index_path, encoding="utf-8") as f:
            self._shard_of = json.load(f)["weight_map"]
        # shard name -> (mmap, {tensor: (dtype_str, shape, start, end)})
        self._shards: dict[str, tuple[mmap.mmap, dict]] = {}

    def _shard(self, shard_name: str) -> tuple[mmap.mmap, dict]:
        cached = self._shards.get(shard_name)
        if cached is not None:
            return cached
        with open(self.dir / shard_name, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
            header.pop("__metadata__", None)
            base = 8 + header_len
            entries = {
                name: (info["dtype"], info["shape"], base + info["data_offsets"][0], base + info["data_offsets"][1])
                for name, info in header.items()
            }
            # The mapping stays valid after the fd is closed.
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._shards[shard_name] = (mm, entries)
        return mm, entries

    def __contains__(self, name: str) -> bool:
        return name in self._shard_of

    def get(self, name: str) -> torch.Tensor:
        """Return tensor ``name`` as a read-only view into the shard mmap."""
        shard_name = self._shard_of.get(name)
        if shard_name is None:
            raise KeyError(f"tensor {name!r} not in checkpoint index")
        mm, entries = self._shard(shard_name)
        dtype_str, shape, start, end = entries[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # non-writable mmap buffer
            flat = torch.frombuffer(memoryview(mm)[start:end], dtype=torch.uint8)
        return flat.view(_SAFETENSORS_DTYPES[dtype_str]).reshape(shape)


# ---------------------------------------------------------------------------
# Dequantization (checkpoint grids) and requantization (kernel W8A8 ABI).
# ---------------------------------------------------------------------------
def _e8m0_to_fp32(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 bytes (unsigned power-of-two exponents) to fp32: 2^(x-127)."""
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def dequant_fp8_block(weight: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize an e4m3 ``[out, in]`` weight with a 128x128-block UE8M0 scale."""
    out_dim, in_dim = weight.shape
    scale = _e8m0_to_fp32(scale_u8)
    scale = scale.repeat_interleave(FP8_BLOCK, dim=0)[:out_dim]
    scale = scale.repeat_interleave(FP8_BLOCK, dim=1)[:, :in_dim]
    return weight.to(torch.float32) * scale


def dequant_fp4(weight_packed: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """Dequantize a packed e2m1 ``[..., out, in/2]`` weight to fp32 ``[..., out, in]``.

    Two fp4 values per byte along the input dim, low nibble first; one UE8M0
    scale per output row per group of 32 unpacked input elements.
    """
    bytes_u8 = weight_packed.view(torch.uint8)
    low = bytes_u8 & 0x0F
    high = (bytes_u8 >> 4) & 0x0F
    nibbles = torch.stack([low, high], dim=-1).reshape(*bytes_u8.shape[:-1], -1)
    values = _FP4_TABLE[nibbles.to(torch.int64)]
    scale = _e8m0_to_fp32(scale_u8).repeat_interleave(FP4_GROUP, dim=-1)
    return values * scale


def quant_int8_per_out_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8 quant of an ``[..., out, in]`` weight.

    Identical chain to the fixture helpers (``quant_w_per_output_channel`` /
    ``quant_w_per_row``): amax over the input dim clamped to INT8_AMAX_EPS,
    round -> int32 -> clamp +-127 -> fp16 -> int8, FP32 dequant scale amax/127.
    """
    w = w.to(torch.float32)
    amax = w.abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    w_i32 = torch.round(w * scale_quant.unsqueeze(-1)).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    return w_i32.to(torch.float16).to(torch.int8), (1.0 / scale_quant).float()


# ---------------------------------------------------------------------------
# Spec-name converters: checkpoint tensors -> per-rank-stacked host tensors.
# ---------------------------------------------------------------------------
def _replicate(x: torch.Tensor, n_ranks: int) -> torch.Tensor:
    return x.unsqueeze(0).expand(n_ranks, *x.shape).contiguous()


class FlashWeightConverter:
    """Converts spec-named host tensors from a :class:`FlashCheckpoint`.

    Weight/scale pairs (``wq_b``/``wq_b_scale``, routed and shared experts,
    ...) are produced by one dequant+requant pass: converting the weight
    stashes its scale, so requesting the scale right after (the spec order of
    the drivers and of :data:`REAL_WEIGHT_NAMES`) is free.
    """

    def __init__(self, ckpt: FlashCheckpoint, *, ep: int, tp: int) -> None:
        if VOCAB % tp != 0:
            raise ValueError(f"TP {tp} does not divide vocab_size {VOCAB}")
        self.ckpt = ckpt
        self.ep = ep
        self.tp = tp
        # Mirror moe.py: the kernel programs keep 32 local experts per rank and
        # shrink the GLOBAL routing space to 32*EP, so only EP8 deploys the full
        # 256-expert model. For EP<8 this loader takes the FIRST 32*EP checkpoint
        # experts and reduces the router tables to match — a reduced-expert
        # smoke configuration, not the true model output.
        self.n_experts = N_EXPERTS_FULL // 8 * ep
        self.n_local = self.n_experts // ep
        self._stash: dict[str, torch.Tensor] = {}

    # ---- generic helpers -------------------------------------------------
    def _fwd_stack(self, per_layer: Callable[[int], torch.Tensor]) -> torch.Tensor:
        return torch.cat([per_layer(layer) for layer in range(NUM_LAYERS)], dim=0)

    def _kind_stack(self, layers: list[int], per_layer: Callable[[int], torch.Tensor]) -> torch.Tensor:
        return torch.cat([per_layer(layer) for layer in layers], dim=0)

    def _deq_fp8(self, prefix: str) -> torch.Tensor:
        return dequant_fp8_block(self.ckpt.get(f"{prefix}.weight"), self.ckpt.get(f"{prefix}.scale"))

    def _raw(self, name: str) -> torch.Tensor:
        return self.ckpt.get(name).clone()

    def _stacked_pair(
        self, scale_name: str, layers: list[int],
        pair_of_layer: Callable[[int], tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Convert a per-layer (weight, scale) pair; stash the stacked scale."""
        weights, scales = zip(*[pair_of_layer(layer) for layer in layers])
        self._stash[scale_name] = _replicate(torch.cat(scales, dim=0), self.ep)
        return _replicate(torch.cat(weights, dim=0), self.ep)

    # ---- attention linears ----------------------------------------------
    def _wq_b(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        w_i8, scale = quant_int8_per_out_channel(self._deq_fp8(f"layers.{layer}.attn.wq_b"))
        return w_i8.t().contiguous(), scale  # kernel layout [Q_LORA, H*HEAD_DIM]

    def _wo_b(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return quant_int8_per_out_channel(self._deq_fp8(f"layers.{layer}.attn.wo_b"))  # [D, O_GROUPS*O_LORA]

    def _idx_wq_b(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self._deq_fp8(f"layers.{layer}.attn.indexer.wq_b")
        w_i8, scale = quant_int8_per_out_channel(w)
        return w_i8.t().contiguous(), scale  # kernel layout [Q_LORA, IDX_N_HEADS*IDX_HEAD_DIM]

    def _shared(self, layer: int, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        return quant_int8_per_out_channel(self._deq_fp8(f"layers.{layer}.ffn.shared_experts.{which}"))

    # ---- routed experts (EP-sharded) ------------------------------------
    def _routed_layer(self, layer: int, which: str) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer's EP-sharded INT8 experts: ``([ep, n_local, out, in], [ep, n_local, out])``."""
        weights, scales = [], []
        for rank in range(self.ep):
            experts = range(rank * self.n_local, (rank + 1) * self.n_local)
            packed = torch.stack(
                [self.ckpt.get(f"layers.{layer}.ffn.experts.{e}.{which}.weight") for e in experts]
            )
            scales_u8 = torch.stack(
                [self.ckpt.get(f"layers.{layer}.ffn.experts.{e}.{which}.scale") for e in experts]
            )
            w_i8, w_scale = quant_int8_per_out_channel(dequant_fp4(packed, scales_u8))
            weights.append(w_i8)
            scales.append(w_scale)
        return torch.stack(weights), torch.stack(scales)

    def _routed(self, scale_name: str, which: str) -> torch.Tensor:
        out_dim, in_dim = (MOE_INTER, D) if which in ("w1", "w3") else (D, MOE_INTER)
        weight = torch.empty([self.ep, NUM_LAYERS * self.n_local, out_dim, in_dim], dtype=torch.int8)
        scale = torch.empty([self.ep, NUM_LAYERS * self.n_local, out_dim], dtype=torch.float32)
        for layer in range(NUM_LAYERS):
            block = slice(layer * self.n_local, (layer + 1) * self.n_local)
            w_i8, w_scale = self._routed_layer(layer, which)
            weight[:, block] = w_i8
            scale[:, block] = w_scale
        self._stash[scale_name] = scale
        return weight

    # ---- gate / routing --------------------------------------------------
    def _gate_bias(self, layer: int) -> torch.Tensor:
        # Decide by layer id, not tensor presence: a checkpoint missing an
        # expected tensor must fail loudly instead of silently zero-filling.
        if layer < M.num_hash_layers:
            return torch.zeros(self.n_experts, dtype=torch.float32)  # hash layers carry no bias
        return self._raw(f"layers.{layer}.ffn.gate.bias")[: self.n_experts]

    def _gate_w(self, layer: int) -> torch.Tensor:
        return self.ckpt.get(f"layers.{layer}.ffn.gate.weight")[: self.n_experts].to(torch.float32)

    def _tid2eid(self, layer: int) -> torch.Tensor:
        if layer < M.num_hash_layers:
            table = self.ckpt.get(f"layers.{layer}.ffn.gate.tid2eid").to(torch.int32)
            if self.n_experts < N_EXPERTS_FULL:
                table = table % self.n_experts  # remap into the reduced expert space
            return table
        return torch.zeros(VOCAB, M.num_experts_per_tok, dtype=torch.int32)  # unused slots

    # ---- heads -----------------------------------------------------------
    def _lm_head(self) -> torch.Tensor:
        head = self.ckpt.get("head.weight")
        vocab_per_tp = VOCAB // self.tp
        shards = [head[s * vocab_per_tp:(s + 1) * vocab_per_tp].clone() for s in range(self.tp)]
        return torch.stack([shards[r % self.tp] for r in range(self.ep)], dim=0)

    # ---- dispatch --------------------------------------------------------
    def convert(self, name: str) -> torch.Tensor:
        """Return the full ``[n_ranks, ...]`` host tensor for spec ``name``."""
        stashed = self._stash.pop(name, None)
        if stashed is not None:
            return stashed
        ckpt, rep, fwd = self.ckpt, _replicate, self._fwd_stack
        csa = lambda fn: self._kind_stack(CSA_LAYERS, fn)  # noqa: E731 — local dispatch shorthand
        hca = lambda fn: self._kind_stack(HCA_LAYERS, fn)  # noqa: E731
        match name:
            # ---- per-FWD-layer stacked attention weights ----
            case "hc_attn_fn" | "hc_attn_scale" | "hc_attn_base" | "hc_ffn_fn" | "hc_ffn_scale" | "hc_ffn_base":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.{name}")), self.ep)
            case "attn_norm_w":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.attn_norm.weight")), self.ep)
            case "wq_a":
                return rep(fwd(
                    lambda l: self._deq_fp8(f"layers.{l}.attn.wq_a").t().contiguous().to(torch.bfloat16)), self.ep)
            case "wkv":
                return rep(fwd(
                    lambda l: self._deq_fp8(f"layers.{l}.attn.wkv").t().contiguous().to(torch.bfloat16)), self.ep)
            case "wq_b":
                return self._stacked_pair("wq_b_scale", list(range(NUM_LAYERS)), self._wq_b)
            case "wo_b":
                return self._stacked_pair("wo_b_scale", list(range(NUM_LAYERS)), self._wo_b)
            case "wq_b_scale" | "wo_b_scale":
                self.convert(name.removesuffix("_scale"))  # populates the stash
                return self._stash.pop(name)
            case "gamma_cq":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.attn.q_norm.weight")), self.ep)
            case "gamma_ckv":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.attn.kv_norm.weight")), self.ep)
            case "attn_sink":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.attn.attn_sink")), self.ep)
            case "wo_a":
                return rep(fwd(
                    lambda l: self._deq_fp8(f"layers.{l}.attn.wo_a").to(torch.bfloat16).view(O_GROUPS, O_LORA, -1)),
                    self.ep)
            # ---- per-FWD-layer stacked MoE weights ----
            case "norm_w":
                return rep(fwd(lambda l: self._raw(f"layers.{l}.ffn_norm.weight")), self.ep)
            case "gate_w":
                return rep(fwd(self._gate_w), self.ep)
            case "gate_bias":
                return rep(fwd(self._gate_bias), self.ep)
            case "tid2eid":
                return rep(fwd(self._tid2eid), self.ep)
            case "routed_w1" | "routed_w2" | "routed_w3":
                return self._routed(f"{name}_scale", name.removeprefix("routed_"))
            case "routed_w1_scale" | "routed_w2_scale" | "routed_w3_scale":
                self.convert(name.removesuffix("_scale"))
                return self._stash.pop(name)
            case "shared_w1" | "shared_w2" | "shared_w3":
                which = name.removeprefix("shared_")
                return self._stacked_pair(
                    f"{name}_scale", list(range(NUM_LAYERS)), lambda l: self._shared(l, which))
            case "shared_w1_scale" | "shared_w2_scale" | "shared_w3_scale":
                self.convert(name.removesuffix("_scale"))
                return self._stash.pop(name)
            # ---- CSA-compact stacks (slot order = ascending ratio-4 layer id) ----
            case "csa_cmp_wkv" | "csa_cmp_wgate":
                part = name.removeprefix("csa_cmp_")
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.compressor.{part}.weight")), self.ep)
            case "csa_cmp_ape":
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.compressor.ape")), self.ep)
            case "csa_cmp_norm_w":
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.compressor.norm.weight")), self.ep)
            case "csa_idx_wq_b":
                return self._stacked_pair("csa_idx_wq_b_scale", CSA_LAYERS, self._idx_wq_b)
            case "csa_idx_wq_b_scale":
                self.convert("csa_idx_wq_b")
                return self._stash.pop(name)
            case "csa_weights_proj":
                return rep(csa(
                    lambda l: ckpt.get(f"layers.{l}.attn.indexer.weights_proj.weight").t().contiguous()), self.ep)
            case "csa_inner_wkv" | "csa_inner_wgate":
                part = name.removeprefix("csa_inner_")
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.indexer.compressor.{part}.weight")), self.ep)
            case "csa_inner_ape":
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.indexer.compressor.ape")), self.ep)
            case "csa_inner_norm_w":
                return rep(csa(lambda l: self._raw(f"layers.{l}.attn.indexer.compressor.norm.weight")), self.ep)
            # ---- HCA-compact stacks (slot order = ascending ratio-128 layer id) ----
            case "hca_cmp_wkv" | "hca_cmp_wgate":
                part = name.removeprefix("hca_cmp_")
                return rep(hca(lambda l: self._raw(f"layers.{l}.attn.compressor.{part}.weight")), self.ep)
            case "hca_cmp_ape":
                return rep(hca(lambda l: self._raw(f"layers.{l}.attn.compressor.ape")), self.ep)
            case "hca_cmp_norm_w":
                return rep(hca(lambda l: self._raw(f"layers.{l}.attn.compressor.norm.weight")), self.ep)
            # ---- replicated head / embedding weights ----
            case "hc_head_fn" | "hc_head_scale" | "hc_head_base":
                return rep(self._raw(name), self.ep)
            case "final_norm_w":
                return rep(self._raw("norm.weight"), self.ep)
            case "embed_weight":
                return rep(self._raw("embed.weight"), self.ep)
            case "lm_head_weight":
                return self._lm_head()
        raise KeyError(f"{name!r} is not a real-weight spec name; expected one of {sorted(REAL_WEIGHT_NAMES)}")

    def convert_layer(self, layer_id: int) -> dict[str, torch.Tensor]:
        """Single-layer real weights keyed by the layer-driver spec names.

        Shapes carry the ``[ep, ...]`` rank dim but no layer stacking — the
        layout ``decode_layer.py`` / ``prefill_layer.py`` consume. Only the
        layer's own kind contributes CSA/HCA entries; the inactive kind keeps
        its fixture. Synthesized inputs (``csa_hadamard_idx``, RoPE tables,
        caches, metadata) are never included.
        """
        if not 0 <= layer_id < NUM_LAYERS:
            raise ValueError(f"layer_id must be in [0, {NUM_LAYERS}), got {layer_id}")
        lyr = layer_id

        def rep(t: torch.Tensor) -> torch.Tensor:
            return _replicate(t, self.ep)

        out: dict[str, torch.Tensor] = {}
        for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                     "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
            out[name] = rep(self._raw(f"layers.{lyr}.{name}"))
        out["attn_norm_w"] = rep(self._raw(f"layers.{lyr}.attn_norm.weight"))
        out["norm_w"] = rep(self._raw(f"layers.{lyr}.ffn_norm.weight"))
        out["gamma_cq"] = rep(self._raw(f"layers.{lyr}.attn.q_norm.weight"))
        out["gamma_ckv"] = rep(self._raw(f"layers.{lyr}.attn.kv_norm.weight"))
        out["attn_sink"] = rep(self._raw(f"layers.{lyr}.attn.attn_sink"))
        out["wq_a"] = rep(self._deq_fp8(f"layers.{lyr}.attn.wq_a").t().contiguous().to(torch.bfloat16))
        out["wkv"] = rep(self._deq_fp8(f"layers.{lyr}.attn.wkv").t().contiguous().to(torch.bfloat16))
        out["wo_a"] = rep(self._deq_fp8(f"layers.{lyr}.attn.wo_a").to(torch.bfloat16).view(O_GROUPS, O_LORA, -1))
        w, s = self._wq_b(lyr)
        out["wq_b"], out["wq_b_scale"] = rep(w), rep(s)
        w, s = self._wo_b(lyr)
        out["wo_b"], out["wo_b_scale"] = rep(w), rep(s)
        out["gate_w"] = rep(self._gate_w(lyr))
        out["gate_bias"] = rep(self._gate_bias(lyr))
        out["tid2eid"] = rep(self._tid2eid(lyr))
        for which in ("w1", "w2", "w3"):
            w_i8, w_scale = self._routed_layer(lyr, which)
            out[f"routed_{which}"], out[f"routed_{which}_scale"] = w_i8, w_scale
            w, s = self._shared(lyr, which)
            out[f"shared_{which}"], out[f"shared_{which}_scale"] = rep(w), rep(s)
        if lyr in CSA_LAYERS:
            out["csa_cmp_wkv"] = rep(self._raw(f"layers.{lyr}.attn.compressor.wkv.weight"))
            out["csa_cmp_wgate"] = rep(self._raw(f"layers.{lyr}.attn.compressor.wgate.weight"))
            out["csa_cmp_ape"] = rep(self._raw(f"layers.{lyr}.attn.compressor.ape"))
            out["csa_cmp_norm_w"] = rep(self._raw(f"layers.{lyr}.attn.compressor.norm.weight"))
            w, s = self._idx_wq_b(lyr)
            out["csa_idx_wq_b"], out["csa_idx_wq_b_scale"] = rep(w), rep(s)
            out["csa_weights_proj"] = rep(
                self.ckpt.get(f"layers.{lyr}.attn.indexer.weights_proj.weight").t().contiguous())
            out["csa_inner_wkv"] = rep(self._raw(f"layers.{lyr}.attn.indexer.compressor.wkv.weight"))
            out["csa_inner_wgate"] = rep(self._raw(f"layers.{lyr}.attn.indexer.compressor.wgate.weight"))
            out["csa_inner_ape"] = rep(self._raw(f"layers.{lyr}.attn.indexer.compressor.ape"))
            out["csa_inner_norm_w"] = rep(self._raw(f"layers.{lyr}.attn.indexer.compressor.norm.weight"))
        if lyr in HCA_LAYERS:
            out["hca_cmp_wkv"] = rep(self._raw(f"layers.{lyr}.attn.compressor.wkv.weight"))
            out["hca_cmp_wgate"] = rep(self._raw(f"layers.{lyr}.attn.compressor.wgate.weight"))
            out["hca_cmp_ape"] = rep(self._raw(f"layers.{lyr}.attn.compressor.ape"))
            out["hca_cmp_norm_w"] = rep(self._raw(f"layers.{lyr}.attn.compressor.norm.weight"))
        return out


def apply_real_layer_weights(specs: list, ckpt_dir: str | Path, *, layer_id: int, ep: int) -> int:
    """Point a layer driver's weight specs at one layer of the real checkpoint.

    For ``decode_layer.py`` / ``prefill_layer.py``: single-layer shapes, no
    stacking. ``ckpt_dir`` must be the HF checkpoint directory (per-layer
    conversion is cheap, no cache needed). Returns the number of specs rewired.
    """
    if M.name != "flash":
        raise ValueError(f"real-weight loading supports the flash variant only, got {M.name!r}")
    converter = FlashWeightConverter(FlashCheckpoint(ckpt_dir), ep=ep, tp=1)
    tensors = converter.convert_layer(layer_id)
    by_name = {getattr(s, "name", None): s for s in specs}
    missing = set(tensors) - set(by_name)
    if missing:
        raise ValueError(f"layer specs are missing expected real-weight names: {sorted(missing)}")
    count = 0
    for name, value in tensors.items():
        spec = by_name[name]
        if list(value.shape) != list(spec.shape) or value.dtype != spec.dtype:
            raise ValueError(
                f"{name}: converted layer weight {tuple(value.shape)}/{value.dtype} does not match "
                f"spec {tuple(spec.shape)}/{spec.dtype} (check --ep)"
            )
        spec.init_value = value
        count += 1
    return count


# Every spec name loaded from the checkpoint, weight before its scale so the
# scale conversion hits the converter's stash.
REAL_WEIGHT_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "csa_idx_wq_b", "csa_idx_wq_b_scale", "csa_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hc_head_fn", "hc_head_scale", "hc_head_base", "final_norm_w",
    "embed_weight", "lm_head_weight",
)


# ---------------------------------------------------------------------------
# Driver integration: swap the fixture init_value of every real-weight spec.
# ---------------------------------------------------------------------------
def apply_real_weights(specs: list, weights_dir: str | Path, *, ep: int, tp: int) -> int:
    """Point every real-weight ``TensorSpec`` in ``specs`` at the checkpoint.

    ``weights_dir`` is either the HF checkpoint directory (detected by
    ``model.safetensors.index.json``; converts on the fly) or a cache
    directory of per-name ``.pt`` files produced by this module's CLI.
    Returns the number of specs rewired.  Raises on a shape/dtype mismatch
    between the converted tensor and the spec, and on a missing cache file.
    """
    if M.name != "flash":
        raise ValueError(f"real-weight loading supports the flash variant only, got {M.name!r}")
    weights_dir = Path(weights_dir)
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"--weights dir not found: {weights_dir}")
    converter = None
    if (weights_dir / "model.safetensors.index.json").is_file():
        converter = FlashWeightConverter(FlashCheckpoint(weights_dir), ep=ep, tp=tp)

    def make_init(spec):
        def init() -> torch.Tensor:
            if converter is not None:
                value = converter.convert(spec.name)
            else:
                path = weights_dir / f"{spec.name}.pt"
                if not path.is_file():
                    # ValueError so the harness's input-generation stage reports
                    # a clean RunResult failure instead of a raw traceback.
                    raise ValueError(
                        f"converted weight missing: {path} (run weights_flash.py --ckpt ... --out {weights_dir})"
                    )
                value = torch.load(path, weights_only=True, mmap=True)
            if list(value.shape) != list(spec.shape) or value.dtype != spec.dtype:
                raise ValueError(
                    f"{spec.name}: converted weight {tuple(value.shape)}/{value.dtype} does not match "
                    f"spec {tuple(spec.shape)}/{spec.dtype} (check --ep/--tp used for conversion)"
                )
            return value

        return init

    count = 0
    for spec in specs:
        if getattr(spec, "name", None) in REAL_WEIGHT_NAMES:
            spec.init_value = make_init(spec)
            count += 1
    missing = set(REAL_WEIGHT_NAMES) - {getattr(s, "name", None) for s in specs}
    if missing:
        raise ValueError(f"specs are missing expected real-weight names: {sorted(missing)}")
    return count


# ---------------------------------------------------------------------------
# CLI: offline conversion into a per-name .pt cache.
# ---------------------------------------------------------------------------
def main() -> None:
    # Import the shape-freezing modules first: they consume --variant / --ep /
    # --tp from sys.argv exactly like the drivers, keeping one source of truth
    # for the EP/TP world shape.
    import lm_head
    import moe

    parser = argparse.ArgumentParser(description="Convert DeepSeek-V4-Flash weights to the kernel ABI.")
    parser.add_argument("--ckpt", type=str, required=True, help="HF checkpoint dir (with model.safetensors.index.json)")
    parser.add_argument("--out", type=str, required=True, help="output cache dir for per-name .pt files")
    parser.add_argument("--only", type=str, nargs="*", default=None, help="convert only these spec names")
    parser.add_argument("--force", action="store_true", help="overwrite existing .pt files")
    # Consumed at import time (config strips --variant; moe/lm_head peek --ep/--tp);
    # declared here like the drivers so argparse accepts and documents them.
    parser.add_argument("--variant", choices=("pro", "flash"), default=M.name,
                        help="Architecture preset selected before module import.")
    parser.add_argument("--ep", type=int, default=moe.N_RANKS, choices=[2, 4, 8],
                        help="EP world size (parsed at import by moe).")
    parser.add_argument("--tp", type=int, default=lm_head.TP_SIZE, choices=[2, 4, 8, 16],
                        help="LM-head TP group size (parsed at import by lm_head).")
    args = parser.parse_args()

    if M.name != "flash":
        raise SystemExit(f"pass --variant flash (or DEEPSEEK_V4_VARIANT=flash); active variant is {M.name!r}")
    ep, tp = moe.N_RANKS, lm_head.TP_SIZE

    converter = FlashWeightConverter(FlashCheckpoint(args.ckpt), ep=ep, tp=tp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(args.only) if args.only else list(REAL_WEIGHT_NAMES)
    unknown = [n for n in names if n not in REAL_WEIGHT_NAMES]
    if unknown:
        raise SystemExit(f"unknown spec names {unknown}; expected among {sorted(REAL_WEIGHT_NAMES)}")

    print(f"[CONVERT] variant={M.name} ep={ep} tp={tp} ckpt={args.ckpt} out={out_dir}", flush=True)
    for i, name in enumerate(names):
        path = out_dir / f"{name}.pt"
        if path.is_file() and not args.force:
            print(f"[CONVERT] ({i + 1}/{len(names)}) {name}: exists, skipped", flush=True)
            continue
        value = converter.convert(name)
        torch.save(value, path)
        size_gb = value.numel() * value.element_size() / 1024**3
        print(f"[CONVERT] ({i + 1}/{len(names)}) {name}: {tuple(value.shape)} {value.dtype} {size_gb:.2f} GiB",
              flush=True)
        del value
    print(f"[CONVERT] done: {out_dir}", flush=True)


# ===========================================================================
# Full-network torch golden (packed-prefill forward)
# ===========================================================================
# ---------------------------------------------------------------------------
# Layer schedule, mirrored from prefill_fwd: per-layer attention kind from the
# preset's compress ratios, plus each kind-compact stack's slot order
# (ascending layer id within the kind).
# ---------------------------------------------------------------------------
_KIND_BY_RATIO = {0: "swa", 4: "csa", 128: "hca"}
_ROPE_PROFILE_BY_KIND = {"swa": 0, "csa": 1, "hca": 1}
FWD_NUM_LAYERS = MODEL_CONFIG.num_hidden_layers
FWD_COMPRESS_RATIOS = MODEL_CONFIG.compress_ratios[:FWD_NUM_LAYERS]
LAYER_KINDS = tuple(_KIND_BY_RATIO[ratio] for ratio in FWD_COMPRESS_RATIOS)
CSA_NUM_LAYERS = LAYER_KINDS.count("csa")
HCA_NUM_LAYERS = LAYER_KINDS.count("hca")


def _kind_orders():
    orders = {}
    counters = {"swa": 0, "csa": 0, "hca": 0}
    for layer, kind in enumerate(LAYER_KINDS):
        orders[layer] = counters[kind]
        counters[kind] += 1
    return orders


KIND_ORDER = _kind_orders()


def _rope_profile_for_kind(stacked, kind):
    try:
        profile = _ROPE_PROFILE_BY_KIND[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported DeepSeek V4 attention kind {kind!r}") from exc
    return stacked[profile]


def _layer_rows(stacked, count, index):
    """Packed-layer slice *index* along dim 1 of a rank-stacked tensor.

    Returns a torch view (never a copy) so leaf-golden in-place cache writes
    propagate into the stacked fwd tensor.
    """
    unit = stacked.shape[1] // count
    return stacked[:, index * unit:(index + 1) * unit]


# ---------------------------------------------------------------------------
# Per-layer view dicts: invert build_single_layer_tensor_specs' name mapping
# between the fwd namespace (csa_cmp_wkv, hca_cmp_ape, ...) and each leaf
# golden's own namespace (cmp_wkv, cmp_ape, ...).
# ---------------------------------------------------------------------------
def _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    """Keys shared by all three attention-kind goldens, for one rank/layer."""

    def fwd(name):
        return _layer_rows(tensors[name], FWD_NUM_LAYERS, layer)[rank]

    kind = LAYER_KINDS[layer]
    return {
        "num_tokens": num_tokens,
        "x_hc": x_hc,
        "hc_attn_fn": fwd("hc_attn_fn"),
        "hc_attn_scale": fwd("hc_attn_scale"),
        "hc_attn_base": fwd("hc_attn_base"),
        "attn_norm_w": fwd("attn_norm_w"),
        "wq_a": fwd("wq_a"),
        "wq_b": fwd("wq_b"),
        "wq_b_scale": fwd("wq_b_scale"),
        "wkv": fwd("wkv"),
        "gamma_cq": fwd("gamma_cq"),
        "gamma_ckv": fwd("gamma_ckv"),
        "freqs_cos": _rope_profile_for_kind(tensors["freqs_cos"][rank], kind),
        "freqs_sin": _rope_profile_for_kind(tensors["freqs_sin"][rank], kind),
        "position_ids": tensors["position_ids"][rank],
        "kv_cache": fwd("kv_cache"),
        "ori_slot_mapping": tensors["ori_slot_mapping"][rank],
        "attn_sink": fwd("attn_sink"),
        "wo_a": fwd("wo_a"),
        "wo_b": fwd("wo_b"),
        "wo_b_scale": fwd("wo_b_scale"),
        "x_out": x_out,
    }


def _swa_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    # The SWA golden names the original-KV table plainly "block_table".
    views["block_table"] = tensors["ori_block_table"][rank]
    return views


def _hca_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    order = KIND_ORDER[layer]

    def hca(name):
        return _layer_rows(tensors[name], HCA_NUM_LAYERS, order)[rank]

    views.update({
        "ori_block_table": tensors["ori_block_table"][rank],
        "cmp_kv": _layer_rows(tensors["cmp_kv"], FWD_NUM_LAYERS, layer)[rank],
        "cmp_block_table": tensors["cmp_block_table"][rank],
        "cmp_wkv": hca("hca_cmp_wkv"),
        "cmp_wgate": hca("hca_cmp_wgate"),
        "cmp_ape": hca("hca_cmp_ape"),
        "cmp_norm_w": hca("hca_cmp_norm_w"),
        "compress_state": hca("hca_compress_state"),
        "compress_state_block_table": tensors["hca_compress_state_block_table"][rank],
        "cmp_slot_mapping": tensors["hca_cmp_slot_mapping"][rank],
        "state_slot_mapping": tensors["hca_state_slot_mapping"][rank],
    })
    return views


def _csa_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens):
    views = _base_attention_views(tensors, rank, layer, x_hc, x_out, num_tokens)
    order = KIND_ORDER[layer]

    def csa(name):
        return _layer_rows(tensors[name], CSA_NUM_LAYERS, order)[rank]

    views.update({
        "ori_block_table": tensors["ori_block_table"][rank],
        "cmp_kv": _layer_rows(tensors["cmp_kv"], FWD_NUM_LAYERS, layer)[rank],
        "cmp_block_table": tensors["cmp_block_table"][rank],
        "cmp_wkv": csa("csa_cmp_wkv"),
        "cmp_wgate": csa("csa_cmp_wgate"),
        "cmp_ape": csa("csa_cmp_ape"),
        "cmp_norm_w": csa("csa_cmp_norm_w"),
        "compress_state": csa("csa_compress_state"),
        "compress_state_block_table": tensors["csa_compress_state_block_table"][rank],
        "hadamard_idx": csa("csa_hadamard_idx"),
        "idx_wq_b": csa("csa_idx_wq_b"),
        "idx_wq_b_scale": csa("csa_idx_wq_b_scale"),
        "idx_weights_proj": csa("csa_weights_proj"),
        "inner_wkv": csa("csa_inner_wkv"),
        "inner_wgate": csa("csa_inner_wgate"),
        "inner_ape": csa("csa_inner_ape"),
        "inner_norm_w": csa("csa_inner_norm_w"),
        "inner_compress_state": csa("csa_inner_compress_state"),
        "inner_compress_state_block_table": tensors["csa_inner_compress_state_block_table"][rank],
        "idx_kv_cache": csa("idx_kv_cache"),
        "idx_kv_scale": csa("idx_kv_scale"),
        "idx_block_table": tensors["idx_block_table"][rank],
        "cmp_slot_mapping": tensors["csa_cmp_slot_mapping"][rank],
        "idx_slot_mapping": tensors["csa_idx_slot_mapping"][rank],
        "state_slot_mapping": tensors["csa_state_slot_mapping"][rank],
        "inner_state_slot_mapping": tensors["csa_inner_state_slot_mapping"][rank],
    })
    return views


_ATTENTION_VIEWS = {
    "swa": _swa_attention_views,
    "hca": _hca_attention_views,
    "csa": _csa_attention_views,
}
_ATTENTION_GOLDEN = {
    "swa": golden_prefill_attention_swa,
    "hca": golden_prefill_attention_hca,
    "csa": golden_prefill_attention_csa,
}

_MOE_LAYER_STACKED = (
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
)


def _moe_views(tensors, layer, x_hc, x_next, num_tokens):
    """golden_moe consumes the full rank stack (dispatch/combine cross ranks)."""
    views = {
        name: _layer_rows(tensors[name], FWD_NUM_LAYERS, layer)
        for name in _MOE_LAYER_STACKED
    }
    views.update({
        "x_hc": x_hc,
        "input_ids": tensors["input_ids"],
        "layer_id": layer,
        "num_tokens": num_tokens,
        "x_next": x_next,
    })
    return views




def golden_prefill_fwd(tensors):
    """Fill every output tensor of prefill_fwd's spec list in place."""
    import torch

    num_tokens = int(tensors["num_tokens"])

    # Layer-0 hidden state: embedding lookup replicated across the HC lanes.
    x_hc = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
    for rank in range(N_RANKS):
        golden_pack_x_hc({
            "input_ids": tensors["input_ids"][rank],
            "embed_weight": tensors["embed_weight"][rank],
            "x_hc": x_hc[rank],
        })

    for layer, kind in enumerate(LAYER_KINDS):
        attn_out = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
        for rank in range(N_RANKS):
            views = _ATTENTION_VIEWS[kind](
                tensors, rank, layer, x_hc[rank], attn_out[rank], num_tokens,
            )
            _ATTENTION_GOLDEN[kind](views)
        # The trailing layer's MoE writes the fwd's pre-hc hidden output.
        if layer == FWD_NUM_LAYERS - 1:
            x_next = tensors["pre_hc_hidden_out"]
        else:
            x_next = torch.zeros(N_RANKS, T, HC_MULT, D, dtype=torch.float32)
        golden_moe(_moe_views(tensors, layer, attn_out, x_next, num_tokens))
        x_hc = x_next

    # hc_head -> final rms_norm per rank, matching the tail of prefill_fwd.
    x_head = torch.zeros(N_RANKS, T, D, dtype=torch.bfloat16)
    for rank in range(N_RANKS):
        golden_hc_head_rows({
            "x_hc": tensors["pre_hc_hidden_out"][rank],
            "hc_head_fn": tensors["hc_head_fn"][rank],
            "hc_head_scale": tensors["hc_head_scale"][rank],
            "hc_head_base": tensors["hc_head_base"][rank],
            "y": x_head[rank],
        })
        tensors["hidden_out"][rank].copy_(
            golden_rms_norm(x_head[rank], tensors["final_norm_w"][rank])
        )

    golden_lm_head_all_ranks(tensors, n_ranks=N_RANKS)


# ---------------------------------------------------------------------------
# --validate comparators. 43 chained layers accumulate error well past the
# leaf kernels' per-point bars, so the hidden/logits comparisons accept
# direction/magnitude agreement (bounded outlier ratio, cosine + relative-L2)
# instead of strict per-element closeness.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Full-net comparators. Model-specific by design (stacked pool layout, prompt
# prefix semantics, lm_head sampling): they live with this model rather than in
# the shared golden package, composing the public ratio_allclose primitive.
# ---------------------------------------------------------------------------
def _stacked_mapped_nonfinite_coordinates(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    valid_mapping: torch.Tensor,
    valid_mapping_items: torch.Tensor,
    feature_shape: tuple[int, ...],
    rank: int,
    layer_index: int,
    blocks_per_layer: int,
    block_size: int,
    max_show: int,
) -> str:
    """Format logical and physical coordinates for mapped non-finite values."""
    nonfinite = ~torch.isfinite(actual) | ~torch.isfinite(expected)
    coordinates = nonfinite.nonzero(as_tuple=False)
    total = int(coordinates.shape[0])
    if total == 0:
        return ""

    show_count = min(max(max_show, 0), total)
    lines = [f"    non-finite mapped coordinate(s): showing {show_count}/{total}"]
    for coordinate in coordinates[:show_count]:
        mapped_offset = int(coordinate[0].item())
        flat_feature = int(coordinate[1].item())
        remaining = flat_feature
        feature_coordinates = []
        for size in reversed(feature_shape):
            feature_coordinates.append(remaining % size)
            remaining //= size
        feature_coordinates.reverse()

        mapped_row = int(valid_mapping_items[mapped_offset].item())
        physical_row = int(valid_mapping[mapped_offset].item())
        pool_block = layer_index * blocks_per_layer + physical_row // block_size
        block_row = physical_row % block_size
        logical_coordinate = (mapped_row, *feature_coordinates)
        physical_coordinate = (rank, pool_block, block_row, *feature_coordinates)
        actual_value = actual[mapped_offset, flat_feature].item()
        expected_value = expected[mapped_offset, flat_feature].item()
        lines.append(
            "      "
            f"logical(mapped_row, feature...)={logical_coordinate} "
            "physical_pool(rank, block, block_row, feature...)="
            f"{physical_coordinate} layer_physical_row={physical_row} "
            f"actual={actual_value!r} expected={expected_value!r}"
        )
    if total > show_count:
        lines.append(
            f"      ... and {total - show_count} more non-finite coordinate(s)"
        )
    return "\n".join(lines)


def sampled_ids_golden_compare(
    *,
    logits_name: str = "logits",
    row_indices_name: str = "logit_row_indices",
    sampled_id_column: int = 0,
    max_show: int = 20,
) -> Callable:
    """Validate sampled IDs against golden and the device's own logits.

    This combines two independent contracts on every selected logit row:

    * semantic correctness — the complete sampled-ID tensor equals golden;
    * sampler self-consistency — the sampled token equals ``argmax`` of the
      actual device logits.

    Checking only the second condition can report PASS for a confidently
    wrong model token, so full-model validation should use both.
    """

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        del expected_outputs, rtol, atol
        if actual.shape != expected.shape:
            return False, (
                f"    sampled_ids shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if actual.ndim < 1 or not 0 <= sampled_id_column < actual.shape[-1]:
            return False, (
                f"    sampled_id_column={sampled_id_column} out of range for "
                f"shape {tuple(actual.shape)}"
            )
        if logits_name not in actual_outputs:
            return False, (
                f"    compare_fn misconfigured: missing actual output '{logits_name}'"
            )
        if row_indices_name not in inputs:
            return False, (
                f"    compare_fn misconfigured: missing input '{row_indices_name}'"
            )

        logits = actual_outputs[logits_name].cpu()
        row_indices = inputs[row_indices_name].cpu()
        leading_shape = actual.shape[:-1]
        if tuple(logits.shape[:-1]) != tuple(leading_shape):
            return False, (
                f"    '{logits_name}' leading shape must be {tuple(leading_shape)}, "
                f"got {tuple(logits.shape)}"
            )
        if tuple(row_indices.shape) != tuple(leading_shape):
            return False, (
                f"    '{row_indices_name}' shape must be {tuple(leading_shape)}, "
                f"got {tuple(row_indices.shape)}"
            )

        actual_cpu = actual.cpu()
        expected_cpu = expected.cpu()
        valid = row_indices >= 0
        device_argmax = torch.argmax(logits, dim=-1).to(actual_cpu.dtype)
        actual_ids = actual_cpu[..., sampled_id_column]
        expected_ids = expected_cpu[..., sampled_id_column]
        semantic_bad = actual_cpu != expected_cpu
        self_bad = valid & (actual_ids != device_argmax)
        failures: list[str] = []

        if semantic_bad.any().item():
            failures.append(
                f"    sampled_ids differ from golden: "
                f"bad={int(semantic_bad.count_nonzero().item())}/{actual_cpu.numel()}"
            )
        if self_bad.any().item():
            failures.append(
                f"    sampled_ids disagree with argmax(actual {logits_name}): "
                f"bad={int(self_bad.count_nonzero().item())}/{int(valid.count_nonzero().item())}"
            )

        interesting = valid & (
            (actual_ids != expected_ids) | (actual_ids != device_argmax)
        )
        coords = interesting.nonzero(as_tuple=False)
        for coord in coords[:max_show]:
            index = tuple(int(value.item()) for value in coord)
            failures.append(
                f"      row={index} actual={int(actual_ids[index].item())} "
                f"golden={int(expected_ids[index].item())} "
                f"device_argmax={int(device_argmax[index].item())}"
            )
        if coords.shape[0] > max_show:
            failures.append(f"      ... and {coords.shape[0] - max_show} more")
        return (not failures), "\n".join(failures)

    compare.__name__ = "sampled_ids_golden_compare"
    return compare


def input_prefix_ratio_allclose(
    valid_rows_name: str,
    *,
    valid_axis: int = 0,
    exact_tail: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
    max_show: int = 10,
) -> Callable:
    """Compare an active prefix whose row count comes from a scalar input.

    Unlike :func:`ratio_allclose`'s static ``valid_rows`` option, this helper
    resolves the prefix length from ``inputs[valid_rows_name]`` for every
    validation call.  This is important for golden-data replay: cached scalar
    inputs override the initializer used to rebuild the specs.

    When ``exact_tail`` is true, rows outside the active prefix must remain
    exactly equal to the golden output.  The inactive rows therefore cannot
    dilute the active-region error ratio or hide a stray write.
    """
    prefix_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
        max_show=max_show,
    )

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        if actual.shape != expected.shape:
            return False, (
                f"    shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if valid_rows_name not in inputs:
            return False, (
                f"    compare_fn misconfigured: missing input '{valid_rows_name}'"
            )

        ndim = actual.ndim
        axis = valid_axis if valid_axis >= 0 else valid_axis + ndim
        if not 0 <= axis < ndim:
            return False, (
                f"    valid_axis={valid_axis} out of range for shape {tuple(actual.shape)}"
            )

        valid_rows_value = inputs[valid_rows_name].cpu()
        if valid_rows_value.numel() != 1:
            return False, (
                f"    '{valid_rows_name}' must be a scalar, "
                f"got shape {tuple(valid_rows_value.shape)}"
            )
        valid_rows = int(valid_rows_value.item())
        total_rows = actual.shape[axis]
        if not 0 <= valid_rows <= total_rows:
            return False, (
                f"    {valid_rows_name}={valid_rows} out of range for axis "
                f"{axis} of length {total_rows}"
            )

        if exact_tail and valid_rows < total_rows:
            actual_tail = actual.narrow(axis, valid_rows, total_rows - valid_rows)
            expected_tail = expected.narrow(axis, valid_rows, total_rows - valid_rows)
            unequal_tail = actual_tail != expected_tail
            if unequal_tail.any().item():
                return False, (
                    f"    inactive tail differs from golden: "
                    f"changed_values={int(unequal_tail.count_nonzero().item())} "
                    f"axis={axis} active_rows={valid_rows}"
                )

        actual_prefix = actual.narrow(axis, 0, valid_rows)
        expected_prefix = expected.narrow(axis, 0, valid_rows)
        return prefix_compare(
            actual_prefix,
            expected_prefix,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )

    compare.__name__ = (
        f"input_prefix_ratio_allclose(valid_rows_name={valid_rows_name}, "
        f"valid_axis={valid_axis}, exact_tail={exact_tail}, atol={atol}, "
        f"rtol={rtol}, max_error_ratio={max_error_ratio})"
    )
    return compare


def stacked_mapped_pool_ratio_allclose(
    layer_mapping_names: tuple[str | None, ...],
    *,
    mapping_shape: tuple[int, int],
    block_size: int,
    active_rows_name: str,
    layer_labels: tuple[int, ...] | None = None,
    pool_name: str = "pool",
    atol: float | None = None,
    rtol: float | None = None,
    max_error_ratio: float = 0.005,
    max_show: int = 3,
) -> Callable:
    """Compare active mapped rows in a rank- and layer-stacked pool.

    The pool layout is ``[ranks, layers * blocks, block_size, ...]``.  Each
    entry in ``layer_mapping_names`` selects the rank-local mapping input for
    that layer; ``None`` means the layer must not write this pool.  Only the
    leading ``inputs[active_rows_name]`` mapping entries participate, which
    makes replay honor the cached active-token scalar.

    Mapped values use a ratio-based numerical comparison independently for
    every layer and rank.  All other physical rows must remain exactly equal
    to golden.  Failure diagnostics therefore identify the first bad logical
    layer/rank instead of reporting a ratio diluted by the unused pool.
    """
    if not layer_mapping_names:
        raise ValueError("layer_mapping_names must not be empty")
    if len(mapping_shape) != 2 or any(dim <= 0 for dim in mapping_shape):
        raise ValueError(
            f"mapping_shape must be (ranks, mapped_items), got {mapping_shape}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if layer_labels is None:
        layer_labels = tuple(range(len(layer_mapping_names)))
    if len(layer_labels) != len(layer_mapping_names):
        raise ValueError(
            "layer_labels and layer_mapping_names must have the same length, "
            f"got {len(layer_labels)} and {len(layer_mapping_names)}"
        )

    mapped_compare = ratio_allclose(
        atol=atol,
        rtol=rtol,
        max_error_ratio=max_error_ratio,
        max_show=max_show,
    )
    integer_dtypes = (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    )

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        if actual.shape != expected.shape:
            return False, (
                f"    {pool_name} shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        rank_count, mapped_items = mapping_shape
        layer_count = len(layer_mapping_names)
        if actual.ndim < 4 or actual.shape[0] != rank_count:
            return False, (
                f"    expected {pool_name} layout "
                "[ranks, layers * blocks, block_size, ...] with "
                f"ranks={rank_count}, got {tuple(actual.shape)}"
            )
        if actual.shape[1] % layer_count != 0 or actual.shape[2] != block_size:
            return False, (
                f"    expected {pool_name} layout "
                "[ranks, layers * blocks, block_size, ...] with "
                f"layers={layer_count} block_size={block_size}, "
                f"got {tuple(actual.shape)}"
            )
        if active_rows_name not in inputs:
            return False, (
                f"    compare_fn misconfigured: missing input '{active_rows_name}'"
            )
        active_value = inputs[active_rows_name].cpu()
        if active_value.numel() != 1:
            return False, (
                f"    '{active_rows_name}' must be a scalar, "
                f"got shape {tuple(active_value.shape)}"
            )
        active_rows = int(active_value.item())
        if not 0 <= active_rows <= mapped_items:
            return False, (
                f"    {active_rows_name}={active_rows} out of range for "
                f"mapping length {mapped_items}"
            )

        mapping_cache: dict[str, torch.Tensor] = {}
        for mapping_name in set(layer_mapping_names) - {None}:
            mapping = inputs.get(mapping_name)
            if mapping is None:
                return False, (
                    f"    compare_fn misconfigured: missing input '{mapping_name}'"
                )
            if mapping.dtype not in integer_dtypes:
                return False, (
                    f"    '{mapping_name}' must have an integer dtype, "
                    f"got {mapping.dtype}"
                )
            if tuple(mapping.shape) != mapping_shape:
                return False, (
                    f"    '{mapping_name}' must have shape {mapping_shape}, "
                    f"got {tuple(mapping.shape)}"
                )
            mapping_cache[mapping_name] = mapping.cpu().to(torch.int64)

        blocks_per_layer = actual.shape[1] // layer_count
        rows_per_layer = blocks_per_layer * block_size
        actual_rows = actual.cpu().reshape(
            rank_count, layer_count, rows_per_layer, -1
        )
        expected_rows = expected.cpu().reshape(
            rank_count, layer_count, rows_per_layer, -1
        )
        failures: list[str] = []

        for layer_index, (layer_label, mapping_name) in enumerate(
            zip(layer_labels, layer_mapping_names)
        ):
            for rank in range(rank_count):
                if mapping_name is None:
                    mapping = torch.empty(0, dtype=torch.int64)
                else:
                    mapping = mapping_cache[mapping_name][rank, :active_rows]

                invalid_negative = mapping < -1
                if invalid_negative.any().item():
                    item = int(invalid_negative.nonzero(as_tuple=False)[0, 0].item())
                    failures.append(
                        f"    layer={layer_label} rank={rank} mapping='{mapping_name}' "
                        f"item={item} value={int(mapping[item].item())}: "
                        "only -1 is a negative sentinel"
                    )
                    continue
                valid_mapping_items = (mapping >= 0).nonzero(as_tuple=False).flatten()
                valid_mapping = mapping[valid_mapping_items]
                if (valid_mapping >= rows_per_layer).any().item():
                    value = int(valid_mapping[valid_mapping >= rows_per_layer][0].item())
                    failures.append(
                        f"    layer={layer_label} rank={rank} mapping='{mapping_name}' "
                        f"row={value} outside physical row range "
                        f"[0, {rows_per_layer})"
                    )
                    continue
                if valid_mapping.numel() > 1:
                    unique_rows, counts = torch.unique(
                        valid_mapping, return_counts=True
                    )
                    duplicates = counts > 1
                    if duplicates.any().item():
                        duplicate_row = int(unique_rows[duplicates][0].item())
                        failures.append(
                            f"    layer={layer_label} rank={rank} "
                            f"mapping='{mapping_name}' contains duplicate "
                            f"physical row {duplicate_row}"
                        )
                        continue

                written_rows = torch.zeros(rows_per_layer, dtype=torch.bool)
                written_rows[valid_mapping] = True
                actual_layer = actual_rows[rank, layer_index]
                expected_layer = expected_rows[rank, layer_index]
                stray_values = actual_layer[~written_rows] != expected_layer[~written_rows]
                if stray_values.any().item():
                    changed_rows = (
                        (actual_layer[~written_rows] != expected_layer[~written_rows])
                        .any(dim=-1)
                        .count_nonzero()
                        .item()
                    )
                    failures.append(
                        f"    layer={layer_label} rank={rank} mapping='{mapping_name}' "
                        f"has writes outside active mapped rows: "
                        f"changed_rows={int(changed_rows)} "
                        f"changed_values={int(stray_values.count_nonzero().item())}"
                    )
                    continue
                if valid_mapping.numel() == 0:
                    continue

                mapped_actual = actual_layer[valid_mapping]
                mapped_expected = expected_layer[valid_mapping]
                ok, detail = mapped_compare(
                    mapped_actual,
                    mapped_expected,
                    actual_outputs=actual_outputs,
                    expected_outputs=expected_outputs,
                    inputs=inputs,
                    rtol=rtol,
                    atol=atol,
                )
                if not ok:
                    nonfinite_coordinates = _stacked_mapped_nonfinite_coordinates(
                        mapped_actual,
                        mapped_expected,
                        valid_mapping=valid_mapping,
                        valid_mapping_items=valid_mapping_items,
                        feature_shape=tuple(actual.shape[3:]),
                        rank=rank,
                        layer_index=layer_index,
                        blocks_per_layer=blocks_per_layer,
                        block_size=block_size,
                        max_show=max_show,
                    )
                    coordinate_detail = (
                        f"\n{nonfinite_coordinates}" if nonfinite_coordinates else ""
                    )
                    failures.append(
                        f"    layer={layer_label} rank={rank} "
                        f"mapping='{mapping_name}' mapped_rows={valid_mapping.numel()}\n"
                        f"{detail}{coordinate_detail}"
                    )

        return (not failures), "\n".join(failures)

    compare.__name__ = (
        f"stacked_mapped_pool_ratio_allclose(pool={pool_name}, "
        f"layers={len(layer_mapping_names)}, block_size={block_size}, "
        f"active_rows_name={active_rows_name}, atol={atol}, rtol={rtol}, "
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def _logits_cosine_compare(min_cosine=0.99, max_rel_l2=0.10):
    """Cosine + relative-L2 acceptance on the rows logit_row_indices selects."""

    def cmp(actual, expected, *, actual_outputs, expected_outputs, inputs, rtol, atol):
        import torch

        del actual_outputs, expected_outputs, rtol, atol
        if actual.shape != expected.shape:
            return False, (
                f"    logits shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        row_indices = inputs["logit_row_indices"].cpu()
        a = actual.cpu().float()
        e = expected.cpu().float()
        failures = []
        for rank in range(a.shape[0]):
            for row in range(a.shape[1]):
                if int(row_indices[rank, row]) < 0:
                    continue
                a_row = a[rank, row]
                e_row = e[rank, row]
                if not bool(torch.isfinite(a_row).all()):
                    failures.append(f"    logits[{rank},{row}]: non-finite actual values")
                    continue
                if not bool(torch.isfinite(e_row).all()):
                    # A NaN golden row would turn cosine and rel_l2 into NaN,
                    # and every NaN comparison below reads as a pass.
                    failures.append(f"    logits[{rank},{row}]: non-finite expected values")
                    continue
                denom = float(a_row.norm() * e_row.norm())
                cosine = float(a_row @ e_row) / denom if denom > 0.0 else 0.0
                rel_l2 = float((a_row - e_row).norm() / e_row.norm().clamp_min(1e-12))
                if cosine < min_cosine or rel_l2 > max_rel_l2:
                    failures.append(
                        f"    logits[{rank},{row}]: cosine={cosine:.6f} "
                        f"(min {min_cosine}) rel_l2={rel_l2:.6f} (max {max_rel_l2})"
                    )
        return (not failures), "\n".join(failures)

    cmp.__name__ = "logits_cosine_compare"
    return cmp


def build_validate_compare_fn(num_tokens):
    """Per-output comparators for prefill_fwd --validate.

    Hidden states resolve their active prefix from the run input, so replayed
    scalar data cannot be diluted by the spec initializer. Logits use
    cosine/relative-L2 acceptance on the selected rows, while sampled IDs must
    match both golden and the device-logit argmax. Layer-stacked pools compare
    only active mapped rows, independently per layer/rank, and require every
    other physical row to remain exactly equal to golden.
    """
    # Keep the public builder signature used by prefill_fwd. The comparator
    # intentionally reads the effective value from inputs["num_tokens"] at
    # validation time because golden-data replay overrides this initializer.
    del num_tokens
    hidden_cmp = input_prefix_ratio_allclose(
        "num_tokens",
        atol=1e-2, rtol=5e-2, max_error_ratio=0.02,
        valid_axis=1, exact_tail=True,
    )
    mapping_shape = (N_RANKS, T)
    all_layer_labels = tuple(range(FWD_NUM_LAYERS))
    csa_layer_labels = tuple(
        layer for layer, kind in enumerate(LAYER_KINDS) if kind == "csa"
    )
    hca_layer_labels = tuple(
        layer for layer, kind in enumerate(LAYER_KINDS) if kind == "hca"
    )

    def stacked_pool(
        mapping_names,
        *,
        layer_labels,
        block_size,
        pool_name,
        atol,
        rtol,
        max_error_ratio,
    ):
        return stacked_mapped_pool_ratio_allclose(
            tuple(mapping_names),
            mapping_shape=mapping_shape,
            block_size=block_size,
            active_rows_name="num_tokens",
            layer_labels=tuple(layer_labels),
            pool_name=pool_name,
            atol=atol,
            rtol=rtol,
            max_error_ratio=max_error_ratio,
        )

    return {
        "pre_hc_hidden_out": hidden_cmp,
        "hidden_out": hidden_cmp,
        "logits": _logits_cosine_compare(),
        "sampled_ids": sampled_ids_golden_compare(),
        "kv_cache": stacked_pool(
            ("ori_slot_mapping",) * FWD_NUM_LAYERS,
            layer_labels=all_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="kv_cache",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
        ),
        "cmp_kv": stacked_pool(
            (
                None if kind == "swa" else f"{kind}_cmp_slot_mapping"
                for kind in LAYER_KINDS
            ),
            layer_labels=all_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="cmp_kv",
            atol=1e-3,
            rtol=3e-2,
            max_error_ratio=0.01,
        ),
        "hca_compress_state": stacked_pool(
            ("hca_state_slot_mapping",) * HCA_NUM_LAYERS,
            layer_labels=hca_layer_labels,
            block_size=HCA_STATE_BLOCK_SIZE,
            pool_name="hca_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        "csa_compress_state": stacked_pool(
            ("csa_state_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=CSA_STATE_BLOCK_SIZE,
            pool_name="csa_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        "csa_inner_compress_state": stacked_pool(
            ("csa_inner_state_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=INNER_STATE_BLOCK_SIZE,
            pool_name="csa_inner_compress_state",
            atol=5e-3,
            rtol=2e-2,
            max_error_ratio=0.01,
        ),
        # INT8 index cache: allow one quantization step on a bounded fraction.
        "idx_kv_cache": stacked_pool(
            ("csa_idx_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="idx_kv_cache",
            atol=1,
            rtol=0,
            max_error_ratio=0.02,
        ),
        "idx_kv_scale": stacked_pool(
            ("csa_idx_slot_mapping",) * CSA_NUM_LAYERS,
            layer_labels=csa_layer_labels,
            block_size=config.BLOCK_SIZE,
            pool_name="idx_kv_scale",
            atol=1e-3,
            rtol=1e-2,
            max_error_ratio=0.01,
        ),
    }
