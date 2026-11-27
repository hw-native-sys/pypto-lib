# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode tiles and the padding contract for the GDN and attention projections.

`a8w8_linear` is the same kernel prefill uses; only the tiles differ, because at decode M
is the batch and each projection reads its whole weight to produce a few output rows.

**M must be at least 16.** Below it the build fails inside `row_expand_mul`: a matmul
result is a full 16-row fractal while the dequant scale column is `m_tile` rows.
`padded_batch` produces that M, `check_batch` enforces it against `max_batch`, and
`check_tile` rejects a token tile that would leave the batch's tail untouched.
"""
import a8w8_linear
import out_proj
from config import QWEN3_8_27B

# model config
HIDDEN = QWEN3_8_27B.hidden_size
QKV_WIDTH = QWEN3_8_27B.qkv_width
VAL_WIDTH = QWEN3_8_27B.linear_num_value_heads * QWEN3_8_27B.linear_value_head_dim
ATTN_QG_WIDTH = 2 * QWEN3_8_27B.num_attention_heads * QWEN3_8_27B.head_dim   # query and gate
ATTN_KV_WIDTH = QWEN3_8_27B.num_key_value_heads * QWEN3_8_27B.head_dim
ATTN_QKV_WIDTH = ATTN_QG_WIDTH + 2 * ATTN_KV_WIDTH

# The fused projection's channel layout, which is the ABI between it and its three
# consumers. It is the row order of the concatenated weight, nothing more; `fuse_qkv_weight`
# builds that weight and these offsets read it back.
QKV_QG_BASE = 0
QKV_K_BASE = ATTN_QG_WIDTH
QKV_V_BASE = ATTN_QG_WIDTH + ATTN_KV_WIDTH

# The cube's M fractal: a matmul result has 16 rows whatever M asks for.
M_FRACTAL = 16

# (K, N, n_tile, k_tile, n_group), measured at decode M. m_tile is M itself.
DECODE_TILES = {
    "in_proj_qkv": (HIDDEN, QKV_WIDTH, 128, 512, 4),
    "in_proj_z": (HIDDEN, VAL_WIDTH, 64, 1024, 4),
    # BF16 on both sides, because the deployed checkpoint leaves this one projection
    # unquantised and the model matches it. Swept as a BF16 matmul; the INT8
    # triple it used to carry lives on in `o_proj`, which is the same shape and still int8.
    # The Mat buffer binds much earlier than it does for int8, and not where counting the
    # named tiles suggests: the compiler's reported usage is
    # `8 * n_tile * k_tile + 4 * m_tile * k_tile`, which is the weight tile in FOUR slots plus
    # a double-buffered activation, and it does not depend on `n_group` at all. At a padded
    # batch that leaves `n_tile * k_tile <= 32768`, so `(64, 1024)`, `(128, 512)` and
    # `(256, 256)` are all refused -- at 589824, 557056 and 540672 bytes respectively.
    #
    # Nine triples survive and the sweep says **the block count is the only thing that
    # matters**: every triple that lands 20 blocks reads 73.5 to 74.0 us, 40 blocks reads 78
    # to 79, 10 blocks 100 to 108 and 5 blocks 173. That is `down_proj`'s cliff again, 20
    # blocks against 24 cube cores. `(64, 512, 4)` is best at M = 16 and 1.6% off the best at
    # M = 32, where `(128, 256, 2)` wins by 73.8 to 75.0 -- the constant pins either way.
    "out_proj": (VAL_WIDTH, HIDDEN, 64, 512, 4),  # 20 blocks, 73.5 us, 856 GB/s at M = 16
    # The attention layers. `o_proj` and `out_proj` are the same `K=6144 N=5120` matmul
    # and share a tile, but differ in dtype, so each names its own entry.
    "o_proj": (VAL_WIDTH, HIDDEN, 64, 512, 4),          # 20 blocks,  37.3 us on the path
    "q_proj": (HIDDEN, ATTN_QG_WIDTH, 64, 1024, 8),     # 24 blocks,  71.7 us, 877 GB/s
    "k_proj": (HIDDEN, ATTN_KV_WIDTH, 64, 1024, 2),     #  8 blocks,  28.0 us, 188 GB/s
    "v_proj": (HIDDEN, ATTN_KV_WIDTH, 64, 1024, 2),     #  8 blocks,  27.7 us, 189 GB/s
    # The three above as one matmul over one concatenated weight.
    "qkv_proj": (HIDDEN, ATTN_QKV_WIDTH, 128, 512, 2),  # 56 patches,  90.2 us, 814 GB/s
}


def padded_batch(batch: int) -> int:
    """The M a decode launch must present: the batch rounded up to the cube's fractal."""
    return max(M_FRACTAL, (batch + M_FRACTAL - 1) // M_FRACTAL * M_FRACTAL)


# The projections the deployed W8A8 checkpoint leaves in BF16. One entry, and it is not a
# tuning choice: `contract.py` carries the weight's dtype as published, the model matches
# it, and the GDN `out_proj` is the only projection in the model it applies to. That one
# projection therefore reads `6144 * 5120 * 2` bytes and a linear-attention layer streams
# 140.00 MiB of projection weight rather than 110.00; over 48 layers that is 1.41 GiB
# more per decode token, 5.8% of the 24640 MiB the 64 layers stream before `lm_head`.
# `o_proj` is the same `K=6144 N=5120` matmul and one measured tile serves both, but the
# checkpoint quantises the attention one and not the GDN one, so they are not the same bytes.
BF16_MODULES = frozenset({"out_proj"})


# Persistent cube blocks a decode launch takes: a2a3's AIC count. A decode projection has
# one row tile, so its patches lie along N alone, and the widest of them ask for more
# blocks than the allocation has cores. Re-sweep for other AIC counts; a5 exposes 32.
CUBE_WORKERS = 24


# a2a3's two on-chip buffers that a decode matmul grows into, because `m_tile` is the batch
#. They cap it, and both are reached at the same value.
ACC_BYTES = 131072          # accumulator
MAT_BYTES = 524288          # Mat


def max_batch() -> int:
    """The largest decode batch the tiles leave room for: 128 for the shipped set.

    Two independent bounds, each measured to the byte off-queue:

    * the accumulator holds an `[m, n_tile]` FP32 output tile in two slots, so
      `2 * m * n_tile * 4 <= ACC_BYTES`. At the widest decode `n_tile`, 128, that is
      `m <= 128`, and batch 144 is refused at exactly 163840 bytes.
    * BF16 `out_proj` holds `8 * n_tile * k_tile + 4 * m * k_tile` of Mat, so for
      `out_proj` at `(64, 512)` that is also `m <= 128`, and with the accumulator term
      halved out of the way batch 144 is refused at exactly 557056 bytes.

    Halving the widest `n_tile` therefore does not raise the ceiling, it only changes which
    buffer refuses first. The MLP's widest decode `n_tile` is the same 128, so it does
    not lower this; a retune there that widens it would be refused by the compiler at a
    batch this function still allows.
    """
    n_tile = max(t[2] for t in DECODE_TILES.values())
    acc = ACC_BYTES // (2 * 4 * n_tile)
    bf16 = [DECODE_TILES[name] for name in BF16_MODULES if name in DECODE_TILES]
    mat = min(((MAT_BYTES - 8 * t[2] * t[3]) // (4 * t[3]) for t in bf16), default=acc)
    return max(M_FRACTAL, min(acc, mat) // M_FRACTAL * M_FRACTAL)


def check_batch(batch: int) -> int:
    """The padding contract, enforced once. Returns the batch unchanged.

    Every decode tile in the model divides `M_FRACTAL`, so a batch that is a positive
    multiple of 16 satisfies the vector stages as well as the cube, **up to `max_batch()`**,
    which is 128 for the shipped tiles. Without that upper bound this check promised any
    multiple of 16 and the refusal arrived as an on-chip capacity error from inside a matmul.

    This exists because **the tiles do not remove the dangerous failure mode**, they only
    move its threshold: `qk_rmsnorm_rope` fans out over a fixed block count with the token
    quotient inside, so below its tile it runs empty tasks and returns with its output
    untouched. Lowering that tile from 32 to 16 does not make it loud. A host-side check is
    the only thing between a decode launch and a wrong answer with no error in it.
    """
    if batch <= 0 or batch % M_FRACTAL:
        raise ValueError(f"decode batch must be a positive multiple of {M_FRACTAL}, got "
                         f"{batch}; pad it with padded_batch({batch}) = "
                         f"{padded_batch(max(batch, 1))}")
    ceiling = max_batch()
    if batch > ceiling:
        raise ValueError(f"decode batch {batch} exceeds {ceiling}, the largest the decode "
                         f"tiles leave on-chip room for; see max_batch")
    return batch


def check_tile(name: str, tile: int, batch: int) -> int:
    """A token tile that does not divide the padded batch drops the tail, silently.

    Measured, not reasoned: at batch 48 a 32-token tile builds and runs in `in_proj_ab`,
    `decode_gate` and the fused `qk_rmsnorm_rope`, passes an off-queue `compile_only`
    screen, and then fails the composed layer's gate on `state` and `residual_out` because
    one task covers 32 of the 48 rows and nothing writes the other 16. The shipped
    tiles all divide 16 and so divide every legal batch; this exists for the next person
    sweeping one of them, because the compiler has nothing to say about it.
    """
    if tile <= 0 or batch % tile:
        raise ValueError(f"{name}={tile} does not divide the padded batch {batch}, so the "
                         f"{batch % tile} rows past the last whole tile would be left "
                         f"untouched with no error; use a divisor of {batch}")
    return tile


def build_kernel(module: str, batch: int, inline: bool = False):
    """One projection at a decode batch. `batch` is rounded up to `M_FRACTAL`.

    `BF16_MODULES` decides which matmul this is: a BF16 module reads a BF16 weight and takes
    no dequant scale, so its kernel has one fewer parameter than the rest.
    """
    if module not in DECODE_TILES:
        raise ValueError(f"unknown projection {module!r}; expected one of "
                         f"{sorted(DECODE_TILES)}")
    k, n, n_tile, k_tile, n_group = DECODE_TILES[module]
    m = padded_batch(batch)
    if module in BF16_MODULES:
        # `k_stagger` and `ring_slots` stay off: DECODE_TILES was swept without them,
        # and at a decode batch this projection is bound by the weight it streams.
        return out_proj.build_kernel(m, k, n, name=module, inline=inline, m_tile=m,
                                        n_tile=n_tile, k_tile=k_tile, m_group=1,
                                        n_group=n_group, k_stagger=0, ring_slots=1,
                                        workers=CUBE_WORKERS)
    return a8w8_linear.build_kernel(m, k, n, name=module, inline=inline, m_tile=m,
                                    n_tile=n_tile, k_tile=k_tile, m_group=1,
                                    n_group=n_group, workers=CUBE_WORKERS)


def fuse_qkv_weight(q_w, k_w, v_w, q_scale, k_scale, v_scale):
    """The one weight `qkv_proj` reads, from the three `q_proj`/`k_proj`/`v_proj` carry.

    Each weight is `[N, K]` INT8 and each scale `[N]` FP32, so the fusion is a row
    concatenation in the order `QKV_QG_BASE`, `QKV_K_BASE`, `QKV_V_BASE` name. Nothing is
    requantised: a row keeps its own scale, which is why one matmul over the concatenation
    is numerically the three matmuls and not an approximation of them.
    """
    import torch

    rows = [(q_w, ATTN_QG_WIDTH), (k_w, ATTN_KV_WIDTH), (v_w, ATTN_KV_WIDTH)]
    for w, n in rows:
        if tuple(w.shape) != (n, HIDDEN):
            raise ValueError(f"expected a [{n}, {HIDDEN}] weight, got {tuple(w.shape)}")
    return (torch.cat([q_w, k_w, v_w], dim=0),
            torch.cat([q_scale.reshape(-1), k_scale.reshape(-1), v_scale.reshape(-1)]))


__all__ = ["ATTN_QKV_WIDTH", "DECODE_TILES", "M_FRACTAL", "QKV_K_BASE", "QKV_QG_BASE",
           "QKV_V_BASE", "build_kernel", "check_batch", "check_tile", "fuse_qkv_weight",
           "max_batch", "padded_batch"]
