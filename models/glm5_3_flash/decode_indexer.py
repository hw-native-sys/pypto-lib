# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The kpool DSA indexer, decode path: projections, pooling, scoring, selection.

The projection lives here; ``indexer.py`` implements and shares the remaining
selection stages with prefill. The stages are:

1. **Projections.** ``q = wq_b(q_resid)`` [T, 32, 128]; ``k = k_norm(wk(x))``
   [T, 128], and note ``k_norm`` carries a **bias**, unlike every other norm in
   this model; ``head_weights = weights_proj(x) * 32 ** -0.5`` [T, 32]; and
   ``gate_scores = index_kpool_compress_gate @ x`` [T, 128]. Every one of these
   weights is BF16 in the checkpoint.
2. **Pooling.** Group four consecutive cached tokens and take a learned weighted
   average: ``p = softmax(gate_scores + index_kpool_compress_ape)`` over the four,
   ``pool_key = sum(p * key)``. A pool is a candidate only when all four of its
   tokens are valid.
3. **Scoring.** ``scores = relu(q . pool_keys^T * 128 ** -0.5)`` then a weighted sum
   over the 32 heads, with the query Hadamard-rotated and quantized first (see
   below). The ``relu`` before the head reduction is what makes this a
   lightning indexer and not a second attention: a head that disagrees contributes
   zero rather than a negative.
4. **Selection.** Top ``index_topk / index_kpool`` = 512 pools out of
   ``P = ceil(kv_len / 4)`` — 32768 at 128k context, 262144 at the 1M limit.
5. **Expansion.** Each selected pool becomes 4 raw cache rows; the incomplete tail
   pool is always appended (``index_kpool_always_select_tail``); the result is
   padded with ``-1`` to a fixed ``TOPK_INDEX_WIDTH`` = 2051 so FULL_DECODE_ONLY
   graph capture sees a static shape. **The live rows are front packed**: every
   valid position precedes every ``-1``, so a row's `-1` entries form one
   suffix. This is an ABI guarantee the sparse attention relies on — see
   :func:`indexer_expand`.

All 32 indexer heads live on **every** rank: the score sums over heads before the
top-k, so head-sharding would force a cross-rank reduction of partial scores on
every sparse layer, and this kernel is small enough that replication is cheaper.

**There is no rope here.** ``indexer_rope_interleave`` is set in ``config.json`` but
is a vestigial field inherited from the GLM-MoE-DSA base: ``Glm5NextTextConfig``
sets ``rope_parameters = AttributeError()``, the model forward passes
``position_embeddings=None``, and ``Glm5NextTextIndexer.forward`` never touches a
cos/sin table. Combined with ``qk_rope_head_dim = 0``, this model has no rope
anywhere, so none of the ``rope_tables.py`` machinery that ``deepseek_v4_pro`` and
``deepseek_v4_flash_mtp`` carry is needed.

**Donors, all a2a3 and all already run by the daily sweep.**
``models/deepseek_v4_flash_mtp/decode_indexer.py`` is the shape to port. Its
``_cp_topk512_query`` (prefill_indexer.py:358-418) is an exact top-512 over the same
262144-candidate cap as GLM's, because DeepSeek-V4-Flash's indexer is itself a
ratio-4 compressed selector. Two device facts recorded there cost real debugging
time: a narrow (256) sort **faults with 507018**, so the leaf stays wide (2048); and
the merge-stage list must match the leaf, because a 4096 stage on a 2048-score row
"lowers to an illegal AIV config".

What to delete from the donor: the compressor, and its ratio-4 slot rewrite. What
to **keep**: the Hadamard-128 rotation and the INT8 quantization of the indexer
query. That half is not a donor quirk — it is GLM's own numerics. The upstream
GLM-5.3-Flash kernel rotates each 128-wide query head by a Hadamard-128 and then
quantizes to FP8 e4m3 with a power-of-two (ue8m0) scale
(``vllm_ascend/models/glm5next/ops/kpool_compress.py:48-75``), and the
Ascend-native GLM-5 recipe does the same rotation
(``models/glm_5/models/indexer.py:97-100``). **a2a3 cannot do the FP8 half** — its
cube has no fp8 entry in ``Intrinsic_mmad`` — so the quantization target becomes
INT8, which is exactly what ``deepseek_v4_flash_mtp/decode_indexer.py:188-214``
already implements: a cube-only ``pl.matmul`` against a BF16 Hadamard operand,
then an INT8 amax/quant scope.

Note the deployment checkpoint ships **no** Hadamard matrix of its own — unlike the
cann-recipes GLM-5 conversion, which bakes a per-layer
``self_attn.indexer.hadamard_matrix`` into the shards. It has to be generated at
load time.

What to add: the pooling stage, which has no donor anywhere, and a
``selected_valid`` output so the expansion can invalidate whole 4-wide groups.

vLLM Ascend is no help: ``sparse_attn_indexer_kpool.py`` raises
``NotImplementedError`` and says the upstream is a set of CUDA kernels with "no NPU
equivalent yet".
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import D, INDEX_DIM, INDEX_H
from models.glm5_3_flash.config import KPOOL_SELECT_K
from models.glm5_3_flash.config import Q_LORA, T_DYN
from models.glm5_3_flash.golden import layer_norm
from models.glm5_3_flash.indexer import (
    golden_indexer_expand, golden_indexer_kpool, golden_indexer_score,
    golden_indexer_topk, indexer_expand, indexer_kpool, indexer_score, indexer_topk,
    indexer_select, indexer_share_mtp, golden_indexer_share_mtp,
)


MM_T_TILE = 16          # cube M tile of every indexer projection
PROJ_N_TILE = 128       # output-column tile of the wide index_q projection
PROJ_K_TILE = 256       # reduction tile shared by the four projections
NORM_T_TILE = 8         # token tile of the key LayerNorm
HEAD_WEIGHT_SCALE = INDEX_H**-0.5
K_NORM_EPS = 1e-6  # Glm5NextTextIndexer builds k_norm as nn.LayerNorm(eps=1e-6)
LEAF = 2048  # the donor's confirmed fault-free sort width on a2a3; 8192 also works
             # but needs the extra 4096 merge stage (see the module docstring)
PAIR_WIDTH = 2 * KPOOL_SELECT_K


def golden_indexer_proj(
    x: torch.Tensor,
    q_resid: torch.Tensor,
    w_q_b: torch.Tensor,
    w_k: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    w_weights: torch.Tensor,
    w_compress_gate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one packed token batch into the indexer's four per-token quantities.

    Mirrors the projection half of ``Glm5NextTextIndexer.forward``::

        q            = wq_b(q_resid)                        [T, INDEX_H, INDEX_DIM]
        k            = k_norm(wk(x))                        [T, INDEX_DIM]
        gate_scores  = index_kpool_compress_gate @ x        [T, INDEX_DIM]
        head_weights = weights_proj(x) * INDEX_H ** -0.5    [T, INDEX_H]

    ``k_norm`` is a **LayerNorm**, not an RMSNorm — see
    :func:`models.glm5_3_flash.golden.layer_norm`. It is the one normalisation in this
    model that subtracts the mean and carries a bias, and its ``eps`` is ``1e-6``
    rather than the model's ``rms_norm_eps``.

    ``head_weights`` and ``gate_scores`` stay FP32: the first is explicitly cast and
    scaled in FP32 by the reference, and the second is what the indexer state cache
    stores, so rounding it here would round it twice. ``q`` and ``k`` keep the
    activation dtype — the Hadamard rotation and the INT8 quantization of the query
    belong to the scoring stage, not here.

    Every matmul accumulates in FP32 and rounds once, which is what a kernel with an
    FP32 accumulator produces.

    Args:
        x: ``[T, D]`` packed hidden states.
        q_resid: ``[T, Q_LORA]`` from :func:`models.glm5_3_flash.mla_prolog.golden_mla_prolog`.
        w_q_b: ``[INDEX_H * INDEX_DIM, Q_LORA]`` indexer query up-projection.
        w_k: ``[INDEX_DIM, D]`` indexer key projection.
        k_norm_weight: ``[INDEX_DIM]`` gamma of the key LayerNorm.
        k_norm_bias: ``[INDEX_DIM]`` beta of the key LayerNorm.
        w_weights: ``[INDEX_H, D]`` per-head score weighting.
        w_compress_gate: ``[INDEX_DIM, D]`` k-pooling gate projection.

    Returns:
        ``index_q`` ``[T, INDEX_H, INDEX_DIM]``, ``index_k`` ``[T, INDEX_DIM]``,
        ``head_weights`` ``[T, INDEX_H]`` FP32 and ``gate_scores`` ``[T, INDEX_DIM]`` FP32.
    """
    dtype = x.dtype
    index_q = F.linear(q_resid.float(), w_q_b.float()).unflatten(-1, (INDEX_H, INDEX_DIM)).to(dtype)
    index_k = layer_norm(
        F.linear(x.float(), w_k.float()), k_norm_weight, k_norm_bias, eps=K_NORM_EPS
    ).to(dtype)
    head_weights = F.linear(x.float(), w_weights.float()) * (INDEX_H**-0.5)
    gate_scores = F.linear(x.float(), w_compress_gate.float())
    return index_q, index_k, head_weights, gate_scores


@pl.jit.inline
def indexer_proj(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[INDEX_H * INDEX_DIM, Q_LORA], pl.BF16],
    w_k: pl.Tensor[[INDEX_DIM, D], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[INDEX_H, D], pl.BF16],
    w_compress_gate: pl.Tensor[[INDEX_DIM, D], pl.BF16],
    index_q: pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16],
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    head_weights: pl.Tensor[[T_DYN, INDEX_H], pl.FP32],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
):
    """Project one packed token batch into the indexer's four per-token quantities.

    Five scopes. ``index_q`` is the only wide output — ``INDEX_H * INDEX_DIM`` = 4,096
    columns — so it fans over (output-column tile, token tile) like the MLA query
    up-projection. The other three projections are 32 or 128 columns wide, so an
    output-column fan-out would leave one block holding everything; they fan over
    token tiles instead and keep their whole N in one accumulator.

    Every weight is stored ``[out, in]`` and consumed with ``b_trans=True``, so each
    ``[N_TILE, K_TILE]`` fragment loads K-contiguous.

    ``k_norm`` is a **LayerNorm**: the fourth scope subtracts the row mean before the
    reciprocal square root and adds a bias afterwards, and its ``eps`` is
    ``K_NORM_EPS`` = 1e-6 rather than the model's ``rms_norm_eps``. It is the only
    normalisation in this model shaped that way, so it cannot borrow an RMSNorm scope.
    ``INDEX_DIM`` is 128, one tile, so the norm needs no reduction loop.

    ``head_weights`` and ``gate_scores`` are written FP32 straight from the
    accumulator: the first is scaled by ``INDEX_H ** -0.5`` in FP32 by the reference,
    and the second is what the indexer state cache stores, so rounding either to BF16
    here would round it twice.

    Returns the four outputs so the caller can chain them.
    """
    t_dim = pl.tensor.dim(x, 0)
    t_mm = ((t_dim + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE

    index_q_flat = pl.reshape(index_q, [t_dim, INDEX_H * INDEX_DIM])
    for q_idx in pl.spmd((INDEX_H * INDEX_DIM) // PROJ_N_TILE, name_hint="indexer_q_proj"):
        n0 = q_idx * PROJ_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            valid_rows = pl.min(MM_T_TILE, t_dim - t0)
            q_acc = pl.create_tensor([MM_T_TILE, PROJ_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(Q_LORA // PROJ_K_TILE, stage=2):
                k0 = kb * PROJ_K_TILE
                q_tile = pl.slice(
                    q_resid, [MM_T_TILE, PROJ_K_TILE], [t0, k0],
                    valid_shape=[valid_rows, PROJ_K_TILE],
                )
                wq_tile = w_q_b[n0 : n0 + PROJ_N_TILE, k0 : k0 + PROJ_K_TILE]
                q_acc = pl.matmul_acc(q_acc, q_tile, wq_tile, b_trans=True, init_cond=(kb == 0))
            index_q_flat = pl.assemble(
                index_q_flat,
                pl.set_validshape(pl.cast(q_acc, target_type=pl.BF16, mode="rint"),
                                  valid_rows, PROJ_N_TILE),
                [t0, n0],
            )

    k_fp32 = pl.create_tensor([t_mm, INDEX_DIM], dtype=pl.FP32)
    for kt in pl.spmd(t_mm // MM_T_TILE, name_hint="indexer_k_proj"):
        t0 = kt * MM_T_TILE
        valid_rows = pl.min(MM_T_TILE, t_dim - t0)
        k_acc = pl.create_tensor([MM_T_TILE, INDEX_DIM], dtype=pl.FP32)
        for kb in pl.pipeline(D // PROJ_K_TILE, stage=2):
            k0 = kb * PROJ_K_TILE
            x_tile = pl.slice(
                x, [MM_T_TILE, PROJ_K_TILE], [t0, k0], valid_shape=[valid_rows, PROJ_K_TILE]
            )
            wk_tile = w_k[0:INDEX_DIM, k0 : k0 + PROJ_K_TILE]
            k_acc = pl.matmul_acc(k_acc, x_tile, wk_tile, b_trans=True, init_cond=(kb == 0))
        k_fp32[t0 : t0 + MM_T_TILE, 0:INDEX_DIM] = k_acc

    for ht in pl.spmd(t_mm // MM_T_TILE, name_hint="indexer_head_weights"):
        t0 = ht * MM_T_TILE
        valid_rows = pl.min(MM_T_TILE, t_dim - t0)
        head_acc = pl.create_tensor([MM_T_TILE, INDEX_H], dtype=pl.FP32)
        for kb in pl.pipeline(D // PROJ_K_TILE, stage=2):
            k0 = kb * PROJ_K_TILE
            xh_tile = pl.slice(
                x, [MM_T_TILE, PROJ_K_TILE], [t0, k0], valid_shape=[valid_rows, PROJ_K_TILE]
            )
            ww_tile = w_weights[0:INDEX_H, k0 : k0 + PROJ_K_TILE]
            head_acc = pl.matmul_acc(head_acc, xh_tile, ww_tile, b_trans=True, init_cond=(kb == 0))
        head_weights = pl.assemble(
            head_weights,
            pl.set_validshape(pl.mul(head_acc, HEAD_WEIGHT_SCALE), valid_rows, INDEX_H),
            [t0, 0],
        )

    for gt in pl.spmd(t_mm // MM_T_TILE, name_hint="indexer_gate_proj"):
        t0 = gt * MM_T_TILE
        valid_rows = pl.min(MM_T_TILE, t_dim - t0)
        gate_acc = pl.create_tensor([MM_T_TILE, INDEX_DIM], dtype=pl.FP32)
        for kb in pl.pipeline(D // PROJ_K_TILE, stage=2):
            k0 = kb * PROJ_K_TILE
            xg_tile = pl.slice(
                x, [MM_T_TILE, PROJ_K_TILE], [t0, k0], valid_shape=[valid_rows, PROJ_K_TILE]
            )
            wg_tile = w_compress_gate[0:INDEX_DIM, k0 : k0 + PROJ_K_TILE]
            gate_acc = pl.matmul_acc(gate_acc, xg_tile, wg_tile, b_trans=True, init_cond=(kb == 0))
        gate_scores = pl.assemble(
            gate_scores, pl.set_validshape(gate_acc, valid_rows, INDEX_DIM), [t0, 0]
        )

    for nt in pl.spmd((t_dim + NORM_T_TILE - 1) // NORM_T_TILE, name_hint="indexer_k_norm"):
        t0 = nt * NORM_T_TILE
        valid_rows = pl.min(NORM_T_TILE, t_dim - t0)
        raw = pl.load(
            k_fp32,
            [t0, 0],
            [NORM_T_TILE, INDEX_DIM],
            valid_shape=[valid_rows, INDEX_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        mean_tmp = pl.create_tile(
            [NORM_T_TILE, INDEX_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        centered = pl.row_expand_sub(
            raw, pl.mul(pl.row_sum(raw, mean_tmp), 1.0 / INDEX_DIM)
        )
        var_tmp = pl.create_tile(
            [NORM_T_TILE, INDEX_DIM], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        variance = pl.mul(
            pl.row_sum(pl.mul(centered, centered), var_tmp), 1.0 / INDEX_DIM
        )
        rsqrt_tmp = pl.create_tile(
            [NORM_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        inv_std = pl.tile.rsqrt(pl.add(variance, K_NORM_EPS), rsqrt_tmp)
        gamma = pl.reshape(
            pl.cast(
                pl.load(k_norm_weight, [0], [INDEX_DIM], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            ),
            [1, INDEX_DIM],
        )
        beta = pl.reshape(
            pl.cast(
                pl.load(k_norm_bias, [0], [INDEX_DIM], target_memory=pl.MemorySpace.Vec),
                target_type=pl.FP32,
            ),
            [1, INDEX_DIM],
        )
        normed = pl.col_expand_add(
            pl.col_expand_mul(pl.row_expand_mul(centered, inv_std), gamma), beta
        )
        pl.store(
            pl.set_validshape(pl.cast(normed, target_type=pl.BF16, mode="rint"),
                              valid_rows, INDEX_DIM),
            [t0, 0],
            index_k,
        )
    return index_q, index_k, head_weights, gate_scores




@pl.jit
def indexer_proj_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[INDEX_H * INDEX_DIM, Q_LORA], pl.BF16],
    w_k: pl.Tensor[[INDEX_DIM, D], pl.BF16],
    k_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    k_norm_bias: pl.Tensor[[INDEX_DIM], pl.BF16],
    w_weights: pl.Tensor[[INDEX_H, D], pl.BF16],
    w_compress_gate: pl.Tensor[[INDEX_DIM, D], pl.BF16],
    index_q: pl.Out[pl.Tensor[[T_DYN, INDEX_H, INDEX_DIM], pl.BF16]],
    index_k: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16]],
    head_weights: pl.Out[pl.Tensor[[T_DYN, INDEX_H], pl.FP32]],
    gate_scores: pl.Out[pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32]],
):
    """Run one indexer projection for golden.run validation."""
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    index_q.bind_dynamic(0, T_DYN)
    index_k.bind_dynamic(0, T_DYN)
    head_weights.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    indexer_proj(
        x,
        q_resid,
        w_q_b,
        w_k,
        k_norm_weight,
        k_norm_bias,
        w_weights,
        w_compress_gate,
        index_q,
        index_k,
        head_weights,
        gate_scores,
    )
    return index_q, index_k, head_weights, gate_scores


def build_indexer_proj_tensor_specs(tokens: int = 20):
    """Build one deterministic projection at the real replicated indexer shapes.

    The indexer is not head-sharded, so these are the full 32 heads rather than a
    per-rank slice. The default token count is a multiple of neither tile, so both the
    cube M tail and the LayerNorm token tail carry real rows.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(109)

    def normal(*shape, scale=1.0):
        def init():
            return (torch.randn(*shape, generator=generator) * scale).bfloat16()

        return init

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=normal(tokens, D)),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16, init_value=normal(tokens, Q_LORA)),
        TensorSpec(
            "w_q_b",
            [INDEX_H * INDEX_DIM, Q_LORA],
            torch.bfloat16,
            init_value=normal(INDEX_H * INDEX_DIM, Q_LORA, scale=0.03),
        ),
        TensorSpec("w_k", [INDEX_DIM, D], torch.bfloat16, init_value=normal(INDEX_DIM, D, scale=0.02)),
        TensorSpec("k_norm_weight", [INDEX_DIM], torch.bfloat16, init_value=normal(INDEX_DIM)),
        TensorSpec("k_norm_bias", [INDEX_DIM], torch.bfloat16, init_value=normal(INDEX_DIM)),
        TensorSpec(
            "w_weights", [INDEX_H, D], torch.bfloat16, init_value=normal(INDEX_H, D, scale=0.02)
        ),
        TensorSpec(
            "w_compress_gate",
            [INDEX_DIM, D],
            torch.bfloat16,
            init_value=normal(INDEX_DIM, D, scale=0.02),
        ),
        TensorSpec("index_q", [tokens, INDEX_H, INDEX_DIM], torch.bfloat16),
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16),
        TensorSpec("head_weights", [tokens, INDEX_H], torch.float32),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32),
    ]


def golden_indexer_proj_case(tensors):
    """Fill the expected outputs for :func:`build_indexer_proj_tensor_specs`."""
    index_q, index_k, head_weights, gate_scores = golden_indexer_proj(
        tensors["x"],
        tensors["q_resid"],
        tensors["w_q_b"],
        tensors["w_k"],
        tensors["k_norm_weight"],
        tensors["k_norm_bias"],
        tensors["w_weights"],
        tensors["w_compress_gate"],
    )
    tensors["index_q"][:] = index_q
    tensors["index_k"][:] = index_k
    tensors["head_weights"][:] = head_weights
    tensors["gate_scores"][:] = gate_scores


def _check_layer_norm_matches_torch():
    g = torch.Generator().manual_seed(19)
    x = torch.randn(5, INDEX_DIM, generator=g) + 12
    w = torch.randn(INDEX_DIM, generator=g).bfloat16()
    b = torch.randn(INDEX_DIM, generator=g).bfloat16()
    torch.testing.assert_close(layer_norm(x, w, b), F.layer_norm(x, (INDEX_DIM,), w.float(), b.float(), 1e-06), rtol=2e-05, atol=2e-05)


def run_projection_goldens():
    """Check reference equations and boundary cases on CPU."""
    _check_layer_norm_matches_torch()
    print("[GOLDEN] PASS projection boundary checks")


def main():
    """Prove the golden on CPU, then validate the projection on device.

    The shared selection stages have their own entry in ``indexer.py``. This entry
    covers the projection. Each output is one rounding of an FP32 accumulation, except the
    key, which carries the LayerNorm on top of it and so gets the wider budget.
    """
    import argparse

    from golden import ratio_allclose, run
    from models.glm5_3_flash._golden_smoke import run_indexer_proj_golden

    run_indexer_proj_golden(golden_indexer_proj)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--golden-only", action="store_true")
    args = parser.parse_args()
    run_projection_goldens()
    if args.golden_only:
        return

    compare = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    key_compare = ratio_allclose(atol=1e-3, rtol=1.0 / 64)
    result = run(
        fn=indexer_proj_test,
        specs=build_indexer_proj_tensor_specs(args.tokens),
        golden_fn=golden_indexer_proj_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1.0 / 128,
        atol=1e-4,
        compare_fn={
            "index_q": compare,
            "index_k": key_compare,
            "head_weights": compare,
            "gate_scores": compare,
        },
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "LEAF",
    "build_indexer_proj_tensor_specs",
    "golden_indexer_proj_case",
    "indexer_proj_test",
    "PAIR_WIDTH",
    "golden_indexer_expand",
    "golden_indexer_kpool",
    "golden_indexer_proj",
    "golden_indexer_score",
    "golden_indexer_topk",
    "indexer_expand",
    "indexer_kpool",
    "indexer_proj",
    "indexer_select",
    "indexer_share_mtp",
    "golden_indexer_share_mtp",
    "indexer_score",
    "indexer_topk",
]


if __name__ == "__main__":
    main()
