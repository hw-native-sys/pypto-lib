# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real DeepSeek-V4-Flash checkpoint loader for the full-network drivers.

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

    python models/deepseek_v4_pro/utils/weights_flash.py --variant flash --ep 8 --tp 2 \\
        --ckpt /path/to/DeepSeek-V4-Flash --out build_output/flash_weights_ep8_tp2
    python models/deepseek_v4_pro/prefill_fwd.py --variant flash --ep 8 --tp 2 \\
        -p a5 -d 0,1,2,3,4,5,6,7 --weights build_output/flash_weights_ep8_tp2

``--weights`` also accepts the raw checkpoint directory directly (detected by
``model.safetensors.index.json``); every weight is then converted on the fly
while the harness builds its inputs.
"""

import argparse
import json
import mmap
import struct
import sys
import warnings
from pathlib import Path
from typing import Callable

import torch

try:
    from config import ACTIVE as M, ACTIVE_BASE, INT8_AMAX_EPS, INT8_SCALE_MAX
except ImportError:  # executed as a script from utils/: put the model dir first
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    # A stray namespace package named `config` may have been cached by the
    # failed import above; drop it so the retry resolves the model's module.
    sys.modules.pop("config", None)
    from config import ACTIVE as M, ACTIVE_BASE, INT8_AMAX_EPS, INT8_SCALE_MAX

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

D = M.hidden_size
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


if __name__ == "__main__":
    main()
