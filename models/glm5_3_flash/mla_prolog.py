# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NoPE MLA prolog for the 11 DSA layers and the MTP layer.

``qk_rope_head_dim`` is 0 and ``mla_use_nope`` is true, so there is **no rope
anywhere in this model** — not here and not in the indexer either. The config
validator rejects a non-zero rope dim outright, ``rope_parameters`` is removed from
the config class, and the model forward passes ``position_embeddings=None``. That
removes the usual DeepSeek MLA rope/nope split and every rope table with it.

    q_resid  = q_a_layernorm(q_a_proj(x))                 [T, 1536]
    query    = q_b_proj(q_resid)                          [T, H, 256]
    kv_pass  = kv_a_layernorm(kv_a_proj_with_mqa(x))      [T, 512]

``q_resid`` is also the indexer's input, so this kernel publishes it separately
rather than fusing the whole prolog. **Every MLA projection is BF16.** The released FP8 checkpoint quantizes
``q_a_proj``, ``q_b_proj``, ``kv_a_proj_with_mqa`` and ``o_proj`` blockwise, but the
deployment checkpoint — ``Eco-Tech/GLM-5.3-Flash-w8a8`` on modelers.cn, which is
what the 16-card A3 recipe serves — does not: its ``quant_model_description.json``
marks all of them ``FLOAT``, and the shards confirm it (``q_b_proj.weight`` is BF16
``[16384, 1536]``). W8A8_DYNAMIC in that checkpoint covers the FFN and nothing else.
This differs from the a2a3 sibling port, which does keep its ``wq_b`` INT8

**Prior art.** ``ops/pypto_python/impl/mla_prolog_pypto.py`` and its
``mla_prolog_quant_pypto.py`` sibling in cann-recipes-infer implement the MLA prolog
unquantized and in W8A8 on this hardware generation. Both target the older
``pypto.Tensor`` / ``pypto_impl`` frontend, which this repo's pinned pypto does not
export, so treat them as algorithm and tiling references rather than code.


## ``kv_b_proj`` absorption for the latent decode path

``kv_b_proj`` maps the 512-wide latent to ``H * (qk_nope_head_dim + v_head_dim)``.
The reference expands it per token and attends in full head space, which is right
for prefill but throws away the point of the latent cache in decode. The absorbed
form folds the key half into the query (giving a ``[H, KV_LORA]`` query that attends
directly against the cached latent) and the value half into ``o_proj``. Both are
weight-time transforms, so they live here rather than in a per-token kernel.

``kv_b_proj`` has **no** ``weight_scale_inv`` in the checkpoint — it is BF16
``[32768, 512]`` — so folding it into an INT8 ``o_proj`` is the numerical question
the owner has to settle.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch
import torch.nn.functional as F

from models.glm5_3_flash.config import D, FLASH, KV_LORA, LOCAL_H, QK_DIM, Q_LORA, T_DYN, V_DIM
from models.glm5_3_flash.golden import rms_norm

EPS = FLASH.rms_norm_eps

MM_T_TILE = 16          # cube M tile; rows are a 16-row boxed tile
RMS_T_TILE = 8          # rms-norm token tile
RMS_K_TILE = 256        # rms-norm reduction chunk
QA_N_TILE = 128         # q_a_proj output-column tile
QA_K_TILE = 256         # q_a_proj D reduction tile
KVA_N_TILE = 128        # kv_a_proj output-column tile
KVA_K_TILE = 256        # kv_a_proj D reduction tile
QB_N_TILE = 128         # q_b_proj output-column tile
QB_K_TILE = 256         # q_b_proj Q_LORA reduction tile
ABSORB_N_TILE = 128     # absorb_query output-column tile
ABSORB_K_TILE = 128     # absorb_query QK_DIM reduction tile


def golden_mla_prolog(
    x: torch.Tensor,
    w_q_a: torch.Tensor,
    q_a_norm_weight: torch.Tensor,
    w_q_b: torch.Tensor,
    w_kv_a: torch.Tensor,
    kv_a_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one packed token batch into ``q_resid``, ``query`` and ``kv_latent``.

    Every projection on this path is BF16 in the deployment checkpoint, so nothing is
    quantized here, and ``qk_rope_head_dim`` is 0, so there is no rope half to split.

    ``query`` is projected from the **rounded** ``q_resid`` rather than from an FP32
    intermediate: ``q_resid`` is also published to the indexer, and both consumers must
    see the same rows. Each matmul accumulates in FP32 and rounds once, which is what a
    kernel with an FP32 accumulator produces; a bit-exact comparison against
    HuggingFace, which carries the activation in BF16 throughout, needs an all-BF16
    variant instead.

    Args:
        x: ``[T, D]`` packed hidden states.
        w_q_a: ``[Q_LORA, D]`` query down-projection.
        q_a_norm_weight: ``[Q_LORA]`` gamma of ``q_a_layernorm``.
        w_q_b: ``[H * QK_DIM, Q_LORA]`` query up-projection, rank-local in ``H``.
        w_kv_a: ``[KV_LORA, D]`` latent down-projection (``num_kv_heads`` is 1, so this
            is not sharded).
        kv_a_norm_weight: ``[KV_LORA]`` gamma of ``kv_a_layernorm``.

    Returns:
        ``q_resid`` ``[T, Q_LORA]``, ``query`` ``[T, H, QK_DIM]`` and ``kv_latent``
        ``[T, KV_LORA]``, all in ``x``'s dtype.
    """
    dtype = x.dtype
    q_resid = rms_norm(F.linear(x.float(), w_q_a.float()), q_a_norm_weight).to(dtype)
    heads = w_q_b.shape[0] // QK_DIM
    query = F.linear(q_resid.float(), w_q_b.float()).unflatten(-1, (heads, QK_DIM)).to(dtype)
    kv_latent = rms_norm(F.linear(x.float(), w_kv_a.float()), kv_a_norm_weight).to(dtype)
    return q_resid, query, kv_latent


@pl.jit.inline
def mla_prolog(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.BF16],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.BF16],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.BF16],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    kv_latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
):
    """Project one packed token batch into ``q_resid``, ``query`` and ``kv_latent``.

    Five scopes: ``q_a_proj`` and ``kv_a_proj`` reduce over D, their norms fold the
    FP32 accumulation down to BF16, and ``q_b_proj`` reads the published ``q_resid``
    back so the indexer and the attention see the same rounded rows.

    Every weight is stored ``[out, in]`` and consumed with ``b_trans=True``, so the
    GM->L1 load of each ``[N_TILE, K_TILE]`` fragment is K-contiguous.

    The token (M) loop sits inside each block so a weight fragment is read once per
    block. No split-K: ``Q_LORA // QA_N_TILE`` is 12 blocks against 24 AIC, and the
    seeded atomic-add that a split would need costs a full zeroing pass over the
    FP32 intermediate. Raising occupancy — split-K here, or a prefill-specific M
    tile — is the first tuning lever once a profile exists.

    Returns the three outputs so the caller can chain them.
    """
    t_dim = pl.tensor.dim(x, 0)
    t_mm = ((t_dim + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE

    q_a_fp32 = pl.create_tensor([t_mm, Q_LORA], dtype=pl.FP32)
    for qa_idx in pl.spmd(Q_LORA // QA_N_TILE, name_hint="mla_q_a_proj"):
        n0 = qa_idx * QA_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            valid_rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = pl.create_tensor([MM_T_TILE, QA_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(D // QA_K_TILE, stage=2):
                k0 = kb * QA_K_TILE
                x_tile = pl.slice(
                    x, [MM_T_TILE, QA_K_TILE], [t0, k0], valid_shape=[valid_rows, QA_K_TILE]
                )
                w_tile = w_q_a[n0 : n0 + QA_N_TILE, k0 : k0 + QA_K_TILE]
                acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True, init_cond=(kb == 0))
            q_a_fp32[t0 : t0 + MM_T_TILE, n0 : n0 + QA_N_TILE] = acc

    for qn_idx in pl.spmd((t_dim + RMS_T_TILE - 1) // RMS_T_TILE, name_hint="mla_q_a_norm"):
        t0 = qn_idx * RMS_T_TILE
        valid_rows = pl.min(RMS_T_TILE, t_dim - t0)
        sq_sum = pl.tile.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
        rms_tmp = pl.create_tile(
            [RMS_T_TILE, RMS_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        for kb in pl.pipeline(Q_LORA // RMS_K_TILE, stage=2):
            k0 = kb * RMS_K_TILE
            chunk = pl.load(
                q_a_fp32,
                [t0, k0],
                [RMS_T_TILE, RMS_K_TILE],
                valid_shape=[valid_rows, RMS_K_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(chunk, chunk), rms_tmp), [1, RMS_T_TILE]))
        rsqrt_tmp = pl.create_tile([1, RMS_T_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        inv_rms = pl.tile.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / Q_LORA), EPS), rsqrt_tmp)
        inv_rms_t = pl.reshape(inv_rms, [RMS_T_TILE, 1])
        for kb in pl.pipeline(Q_LORA // RMS_K_TILE, stage=2):
            k0 = kb * RMS_K_TILE
            chunk = pl.load(
                q_a_fp32,
                [t0, k0],
                [RMS_T_TILE, RMS_K_TILE],
                valid_shape=[valid_rows, RMS_K_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_bf16 = pl.load(q_a_norm_weight, [k0], [RMS_K_TILE], target_memory=pl.MemorySpace.Vec)
            gamma = pl.reshape(pl.cast(gamma_bf16, target_type=pl.FP32), [1, RMS_K_TILE])
            normed = pl.col_expand_mul(pl.row_expand_mul(chunk, inv_rms_t), gamma)
            normed_bf16 = pl.cast(normed, target_type=pl.BF16, mode="rint")
            pl.store(pl.set_validshape(normed_bf16, valid_rows, RMS_K_TILE), [t0, k0], q_resid)

    query_flat = pl.reshape(query, [t_dim, LOCAL_H * QK_DIM])
    for qb_idx in pl.spmd((LOCAL_H * QK_DIM) // QB_N_TILE, name_hint="mla_q_b_proj"):
        n0 = qb_idx * QB_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            valid_rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = pl.create_tensor([MM_T_TILE, QB_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(Q_LORA // QB_K_TILE, stage=2):
                k0 = kb * QB_K_TILE
                q_tile = pl.slice(
                    q_resid, [MM_T_TILE, QB_K_TILE], [t0, k0], valid_shape=[valid_rows, QB_K_TILE]
                )
                w_tile = w_q_b[n0 : n0 + QB_N_TILE, k0 : k0 + QB_K_TILE]
                acc = pl.matmul_acc(acc, q_tile, w_tile, b_trans=True, init_cond=(kb == 0))
            query_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
            query_flat = pl.assemble(
                query_flat, pl.set_validshape(query_bf16, valid_rows, QB_N_TILE), [t0, n0]
            )

    kv_a_fp32 = pl.create_tensor([t_mm, KV_LORA], dtype=pl.FP32)
    for kv_idx in pl.spmd(KV_LORA // KVA_N_TILE, name_hint="mla_kv_a_proj"):
        n0 = kv_idx * KVA_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            valid_rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = pl.create_tensor([MM_T_TILE, KVA_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(D // KVA_K_TILE, stage=2):
                k0 = kb * KVA_K_TILE
                x_tile = pl.slice(
                    x, [MM_T_TILE, KVA_K_TILE], [t0, k0], valid_shape=[valid_rows, KVA_K_TILE]
                )
                w_tile = w_kv_a[n0 : n0 + KVA_N_TILE, k0 : k0 + KVA_K_TILE]
                acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True, init_cond=(kb == 0))
            kv_a_fp32[t0 : t0 + MM_T_TILE, n0 : n0 + KVA_N_TILE] = acc

    for kvn_idx in pl.spmd((t_dim + RMS_T_TILE - 1) // RMS_T_TILE, name_hint="mla_kv_a_norm"):
        t0 = kvn_idx * RMS_T_TILE
        valid_rows = pl.min(RMS_T_TILE, t_dim - t0)
        sq_sum = pl.tile.full([1, RMS_T_TILE], dtype=pl.FP32, value=0.0)
        rms_tmp = pl.create_tile(
            [RMS_T_TILE, RMS_K_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        for kb in pl.pipeline(KV_LORA // RMS_K_TILE, stage=2):
            k0 = kb * RMS_K_TILE
            chunk = pl.load(
                kv_a_fp32,
                [t0, k0],
                [RMS_T_TILE, RMS_K_TILE],
                valid_shape=[valid_rows, RMS_K_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(chunk, chunk), rms_tmp), [1, RMS_T_TILE]))
        rsqrt_tmp = pl.create_tile([1, RMS_T_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        inv_rms = pl.tile.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / KV_LORA), EPS), rsqrt_tmp)
        inv_rms_t = pl.reshape(inv_rms, [RMS_T_TILE, 1])
        for kb in pl.pipeline(KV_LORA // RMS_K_TILE, stage=2):
            k0 = kb * RMS_K_TILE
            chunk = pl.load(
                kv_a_fp32,
                [t0, k0],
                [RMS_T_TILE, RMS_K_TILE],
                valid_shape=[valid_rows, RMS_K_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            gamma_bf16 = pl.load(kv_a_norm_weight, [k0], [RMS_K_TILE], target_memory=pl.MemorySpace.Vec)
            gamma = pl.reshape(pl.cast(gamma_bf16, target_type=pl.FP32), [1, RMS_K_TILE])
            normed = pl.col_expand_mul(pl.row_expand_mul(chunk, inv_rms_t), gamma)
            normed_bf16 = pl.cast(normed, target_type=pl.BF16, mode="rint")
            pl.store(pl.set_validshape(normed_bf16, valid_rows, RMS_K_TILE), [t0, k0], kv_latent)

    return q_resid, query, kv_latent


@pl.jit
def mla_prolog_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.BF16],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.BF16],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.BF16],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.BF16]],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16]],
    kv_latent: pl.Out[pl.Tensor[[T_DYN, KV_LORA], pl.BF16]],
):
    """Run one prolog for golden.run validation."""
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    query.bind_dynamic(0, T_DYN)
    kv_latent.bind_dynamic(0, T_DYN)
    mla_prolog(
        x,
        w_q_a,
        q_a_norm_weight,
        w_q_b,
        w_kv_a,
        kv_a_norm_weight,
        q_resid,
        query,
        kv_latent,
    )
    return q_resid, query, kv_latent


def build_mla_prolog_tensor_specs(tokens: int = 20):
    """Build one deterministic prolog at the real per-rank shapes.

    The default token count is not a multiple of either tile, so both the cube M
    tail and the rms-norm token tail carry real rows.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(59)

    def normal(*shape, scale=1.0):
        def init():
            return (torch.randn(*shape, generator=generator) * scale).bfloat16()

        return init

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=normal(tokens, D)),
        TensorSpec("w_q_a", [Q_LORA, D], torch.bfloat16, init_value=normal(Q_LORA, D, scale=0.02)),
        TensorSpec(
            "q_a_norm_weight", [Q_LORA], torch.bfloat16, init_value=normal(Q_LORA)
        ),
        TensorSpec(
            "w_q_b",
            [LOCAL_H * QK_DIM, Q_LORA],
            torch.bfloat16,
            init_value=normal(LOCAL_H * QK_DIM, Q_LORA, scale=0.03),
        ),
        TensorSpec(
            "w_kv_a", [KV_LORA, D], torch.bfloat16, init_value=normal(KV_LORA, D, scale=0.02)
        ),
        TensorSpec(
            "kv_a_norm_weight", [KV_LORA], torch.bfloat16, init_value=normal(KV_LORA)
        ),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16),
        TensorSpec("query", [tokens, LOCAL_H, QK_DIM], torch.bfloat16),
        TensorSpec("kv_latent", [tokens, KV_LORA], torch.bfloat16),
    ]


def golden_mla_prolog_case(tensors):
    """Fill the expected outputs for :func:`build_mla_prolog_tensor_specs`."""
    q_resid, query, kv_latent = golden_mla_prolog(
        tensors["x"],
        tensors["w_q_a"],
        tensors["q_a_norm_weight"],
        tensors["w_q_b"],
        tensors["w_kv_a"],
        tensors["kv_a_norm_weight"],
    )
    tensors["q_resid"][:] = q_resid
    tensors["query"][:] = query
    tensors["kv_latent"][:] = kv_latent


def main():
    """Prove the goldens on CPU, then validate the prolog and the absorption on device.

    The BF16 outputs are budgeted at one rounding: kernel and reference share the input
    rounding, so what is left is the FP32 accumulation order plus the output rounding.
    """
    import argparse

    from golden import ratio_allclose, run
    from models.glm5_3_flash._golden_smoke import run_mla_prolog_goldens

    run_mla_prolog_goldens(
        golden_mla_prolog,
        golden_split_kv_b,
        golden_absorb_query,
        golden_absorb_output,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    compare = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    result = run(
        fn=mla_prolog_test,
        specs=build_mla_prolog_tensor_specs(args.tokens),
        golden_fn=golden_mla_prolog_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1.0 / 128,
        atol=1e-4,
        compare_fn={"q_resid": compare, "query": compare, "kv_latent": compare},
        compile_only=args.compile_only,
    )
    print(f"prolog: {result}")
    if not result.passed:
        raise SystemExit(result.error or 1)

    absorb_result = run(
        fn=mla_absorb_query_test,
        specs=build_mla_absorb_tensor_specs(args.tokens),
        golden_fn=golden_mla_absorb_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1.0 / 128,
        atol=1e-4,
        compare_fn={"absorbed": compare},
        compile_only=args.compile_only,
    )
    print(f"absorb_query: {absorb_result}")
    if not absorb_result.passed:
        raise SystemExit(absorb_result.error or 1)


@pl.jit.inline
def absorb_query(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    absorbed: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
):
    """Fold the key half into the query, one block per (head, output-column) pair.

    Both attention paths call this: decode consumes the absorbed query directly, and
    prefill absorbs as well so the key expansion stays out of its per-token block loop
    (see :mod:`models.glm5_3_flash.prefill_sparse_attn`). ``w_k`` is stored
    ``[QK_DIM, KV_LORA]`` per head — reduction dim first — so the matmul takes no
    ``b_trans``.

    Folding ``w_k`` into ``w_q_b`` at weight time would remove this dispatch, but
    ``Q_LORA`` is 1,536 against a ``QK_DIM`` of 256, so it widens the ``q_b`` output
    from 1,024 to 2,048 and costs 3.1 M MAC per token against 2.1 M for projecting
    then folding. The per-token fold is the cheaper half.

    Returns the absorbed query so the caller can chain it.
    """
    t_dim = pl.tensor.dim(query, 0)
    t_mm = ((t_dim + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
    query_flat = pl.reshape(query, [t_dim, LOCAL_H * QK_DIM])
    w_k_flat = pl.reshape(w_k, [LOCAL_H * QK_DIM, KV_LORA])
    absorbed_flat = pl.reshape(absorbed, [t_dim, LOCAL_H * KV_LORA])
    for ab_idx in pl.spmd(LOCAL_H * (KV_LORA // ABSORB_N_TILE), name_hint="mla_absorb_query"):
        head = ab_idx // (KV_LORA // ABSORB_N_TILE)
        n0 = (ab_idx % (KV_LORA // ABSORB_N_TILE)) * ABSORB_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            valid_rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = pl.create_tensor([MM_T_TILE, ABSORB_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(QK_DIM // ABSORB_K_TILE, stage=2):
                k0 = kb * ABSORB_K_TILE
                q_tile = pl.slice(
                    query_flat,
                    [MM_T_TILE, ABSORB_K_TILE],
                    [t0, head * QK_DIM + k0],
                    valid_shape=[valid_rows, ABSORB_K_TILE],
                )
                w_tile = pl.slice(
                    w_k_flat,
                    [ABSORB_K_TILE, ABSORB_N_TILE],
                    [head * QK_DIM + k0, n0],
                )
                acc = pl.matmul_acc(acc, q_tile, w_tile, init_cond=(kb == 0))
            absorbed_bf16 = pl.cast(acc, target_type=pl.BF16, mode="rint")
            absorbed_flat = pl.assemble(
                absorbed_flat,
                pl.set_validshape(absorbed_bf16, valid_rows, ABSORB_N_TILE),
                [t0, head * KV_LORA + n0],
            )
    return absorbed


@pl.jit
def mla_absorb_query_test(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    absorbed: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16]],
):
    """Run one query absorption for golden.run validation."""
    query.bind_dynamic(0, T_DYN)
    absorbed.bind_dynamic(0, T_DYN)
    absorb_query(query, w_k, absorbed)
    return absorbed


def build_mla_absorb_tensor_specs(tokens: int = 20):
    """Build one deterministic absorption at the real per-rank shapes."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(61)

    def init_query():
        return torch.randn(tokens, LOCAL_H, QK_DIM, generator=generator).bfloat16()

    def init_w_k():
        return (torch.randn(LOCAL_H, QK_DIM, KV_LORA, generator=generator) * 0.02).bfloat16()

    return [
        TensorSpec("query", [tokens, LOCAL_H, QK_DIM], torch.bfloat16, init_value=init_query),
        TensorSpec("w_k", [LOCAL_H, QK_DIM, KV_LORA], torch.bfloat16, init_value=init_w_k),
        TensorSpec("absorbed", [tokens, LOCAL_H, KV_LORA], torch.bfloat16),
    ]


def golden_mla_absorb_case(tensors):
    """Fill the expected absorbed query for :func:`build_mla_absorb_tensor_specs`."""
    tensors["absorbed"][:] = golden_absorb_query(tensors["query"], tensors["w_k"])


def golden_split_kv_b(w_kv_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into its key and value halves.

    Args:
        w_kv_b: ``[H * (QK_DIM + V_DIM), KV_LORA]`` as stored in the checkpoint.

    Returns:
        ``w_k`` of ``[H, QK_DIM, KV_LORA]`` and ``w_v`` of ``[H, V_DIM, KV_LORA]``.
    """
    heads = w_kv_b.shape[0] // (QK_DIM + V_DIM)
    reshaped = w_kv_b.reshape(heads, QK_DIM + V_DIM, KV_LORA)
    return reshaped[:, :QK_DIM], reshaped[:, QK_DIM:]


def golden_absorb_query(query: torch.Tensor, w_k: torch.Tensor) -> torch.Tensor:
    """Fold the key half into the query so it attends against the raw latent.

    ``q . K^T`` with ``K = latent @ w_k^T`` equals ``(q @ w_k) . latent^T``, so folding
    ``w_k`` into the query leaves every score unchanged while removing the per-token key
    expansion from the decode path. ``w_k`` is BF16 in the deployment checkpoint and so
    is the query, so this introduces no quantization step.

    Args:
        query: ``[T, H, QK_DIM]`` from :func:`golden_mla_prolog`.
        w_k: ``[H, QK_DIM, KV_LORA]`` key half from :func:`golden_split_kv_b`.

    Returns:
        ``[T, H, KV_LORA]`` query in latent space, in ``query``'s dtype.
    """
    absorbed = torch.einsum("thd,hdk->thk", query.float(), w_k.float())
    return absorbed.to(query.dtype)


def golden_absorb_output(w_v: torch.Tensor, w_o: torch.Tensor) -> torch.Tensor:
    """Fold the value half into ``o_proj``; returns ``[D, H * KV_LORA]``.

    Decode attends in latent space, so its context rows are ``[H, KV_LORA]`` and the
    value expansion never has to happen per token: ``o_proj @ (ctx @ w_v^T)`` equals
    ``(o_proj folded with w_v) @ ctx``. This is a weight-time transform, and both
    operands are BF16 in the deployment checkpoint, so nothing is requantized.

    Args:
        w_v: ``[H, V_DIM, KV_LORA]`` value half from :func:`golden_split_kv_b`.
        w_o: ``[D, H * V_DIM]`` head-space output projection.

    There is no kernel for this: it runs once per layer when the weights are staged,
    so it belongs to the weight loader rather than to any per-token scope. Only the
    query half (:func:`absorb_query`) is per-token.

    Returns:
        ``[D, H * KV_LORA]`` absorbed output projection, in ``w_o``'s dtype.
    """
    heads, v_dim, kv_lora = w_v.shape
    per_head = w_o.float().unflatten(-1, (heads, v_dim))
    absorbed = torch.einsum("dhv,hvk->dhk", per_head, w_v.float())
    return absorbed.reshape(w_o.shape[0], heads * kv_lora).to(w_o.dtype)


__all__ = [
    "absorb_query",
    "build_mla_absorb_tensor_specs",
    "build_mla_prolog_tensor_specs",
    "golden_absorb_output",
    "golden_absorb_query",
    "golden_mla_absorb_case",
    "golden_mla_prolog",
    "golden_mla_prolog_case",
    "golden_split_kv_b",
    "mla_absorb_query_test",
    "mla_prolog",
    "mla_prolog_test",
]


if __name__ == "__main__":
    main()
