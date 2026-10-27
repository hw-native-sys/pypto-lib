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
rather than fusing the whole prolog. The current kernel accepts BF16 projections.
W8A8 checkpoints can store ``q_a_proj``, ``q_b_proj``, ``kv_a_proj_with_mqa`` and
``o_proj`` as INT8 with per-output-channel scales. The checkpoint loader explicitly
dequantizes those weights to BF16 for this ABI, according to the quantization
description. This adaptation does not implement dynamic activation quantization
inside MLA.

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
directly against the cached latent) and the value half into ``o_proj``. The query
transform runs per token; the output-weight transform runs once during loading.

``kv_b_proj`` has **no** ``weight_scale_inv`` in the checkpoint — it is BF16
``[32768, 512]``. The loader folds its value half into the prepared BF16 ``o_proj``
using FP32 accumulation and rounds the result to BF16.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, KV_LORA, LOCAL_H, QK_DIM, Q_LORA, T_DYN
from models.glm5_3_flash.linear import bf16_linear, golden_bf16_linear, w8a8_projection_bf16
from models.glm5_3_flash.quantization import w8a8_dynamic_linear
from models.glm5_3_flash.golden import golden_absorb_output, golden_split_kv_b, rms_norm
from models.glm5_3_flash.rmsnorm import rmsnorm

MM_T_TILE = 16          # cube M tile; rows are a 16-row boxed tile
QA_N_TILE = 128         # q_a_proj output-column tile
KVA_N_TILE = 128        # kv_a_proj output-column tile
QB_N_TILE = 128         # q_b_proj output-column tile
ABSORB_N_TILE = 128     # absorb_query output-column tile


def golden_mla_prolog(
    x: torch.Tensor,
    w_q_a: torch.Tensor,
    q_a_norm_weight: torch.Tensor,
    w_q_b: torch.Tensor,
    w_kv_a: torch.Tensor,
    kv_a_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one packed token batch into ``q_resid``, ``query`` and ``kv_latent``.

    Every projection on this path has been prepared as BF16 by the loader, so nothing
    is quantized here. ``qk_rope_head_dim`` is 0, so there is no rope half to split.

    ``query`` is projected from the **rounded** ``q_resid`` rather than from an FP32
    intermediate: ``q_resid`` is also published to the indexer, and both consumers must
    see the same rows. The down-projections round to BF16 before RMSNorm,
    and normalization preserves the BF16 boundary before the gamma product.

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
    q_a = golden_bf16_linear(x, w_q_a).to(dtype)
    kv_a = golden_bf16_linear(x, w_kv_a).to(dtype)
    q_resid = rms_norm(q_a, q_a_norm_weight)
    kv_latent = rms_norm(kv_a, kv_a_norm_weight)
    heads = w_q_b.shape[0] // QK_DIM
    query = golden_bf16_linear(q_resid, w_q_b).to(dtype).unflatten(-1, (heads, QK_DIM))
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

    The down-projections reduce over D and publish BF16 before their norms.
    ``q_b_proj`` reads the published ``q_resid``
    back so the indexer and the attention see the same rounded rows.

    Every weight is stored ``[out, in]`` and consumed with ``b_trans=True``.
    Compensated FP32 dot products preserve the BF16 projection rounding before
    normalization and before the query is published.

    Returns the three outputs so the caller can chain them.
    """
    t_dim = pl.tensor.dim(x, 0)
    t_mm = ((t_dim + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
    q_a = pl.create_tensor([t_dim, Q_LORA], dtype=pl.BF16)
    for qa_idx in pl.spmd(Q_LORA // QA_N_TILE, name_hint="mla_q_a_proj"):
        n0 = qa_idx * QA_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = bf16_linear(x, w_q_a, t0, n0, rows, pl.cast(QA_N_TILE, pl.INDEX), MM_T_TILE, QA_N_TILE, D, True)
            pl.assemble(q_a, pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, QA_N_TILE), [t0, n0])
    rmsnorm(q_a, q_a_norm_weight, q_resid)

    query_flat = pl.reshape(query, [t_dim, LOCAL_H * QK_DIM])
    for qb_idx in pl.spmd((LOCAL_H * QK_DIM) // QB_N_TILE, name_hint="mla_q_b_proj"):
        n0 = qb_idx * QB_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = bf16_linear(
                q_resid, w_q_b, t0, n0, rows,
                pl.cast(QB_N_TILE, pl.INDEX), MM_T_TILE, QB_N_TILE, Q_LORA, True,
            )
            value = pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, QB_N_TILE)
            pl.assemble(query_flat, value, [t0, n0])

    kv_a = pl.create_tensor([t_dim, KV_LORA], dtype=pl.BF16)
    for kv_idx in pl.spmd(KV_LORA // KVA_N_TILE, name_hint="mla_kv_a_proj"):
        n0 = kv_idx * KVA_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = bf16_linear(x, w_kv_a, t0, n0, rows, pl.cast(KVA_N_TILE, pl.INDEX), MM_T_TILE, KVA_N_TILE, D, True)
            pl.assemble(kv_a, pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, KVA_N_TILE), [t0, n0])
    rmsnorm(kv_a, kv_a_norm_weight, kv_latent)
    return q_resid, query, kv_latent



@pl.jit.inline
def mla_prolog_w8a8(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.INT8],
    w_q_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    w_q_b_scale: pl.Tensor[[LOCAL_H * QK_DIM], pl.FP32],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.INT8],
    w_kv_a_scale: pl.Tensor[[KV_LORA], pl.FP32],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    kv_latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
):
    """Run all three MLA projections with native INT8 weights and dynamic activations."""
    rows = pl.tensor.dim(x, 0)
    q_a = pl.create_tensor([rows, Q_LORA], dtype=pl.BF16)
    w8a8_projection_bf16(x, w_q_a, w_q_a_scale, q_a, D, Q_LORA)
    rmsnorm(q_a, q_a_norm_weight, q_resid)
    query_flat = pl.reshape(query, [rows, LOCAL_H * QK_DIM])
    w8a8_projection_bf16(q_resid, w_q_b, w_q_b_scale, query_flat, Q_LORA, LOCAL_H * QK_DIM)
    kv_a = pl.create_tensor([rows, KV_LORA], dtype=pl.BF16)
    w8a8_projection_bf16(x, w_kv_a, w_kv_a_scale, kv_a, D, KV_LORA)
    rmsnorm(kv_a, kv_a_norm_weight, kv_latent)
    return q_resid, query, kv_latent


@pl.jit
def mla_prolog_w8a8_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.INT8],
    w_q_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    w_q_b_scale: pl.Tensor[[LOCAL_H * QK_DIM], pl.FP32],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.INT8],
    w_kv_a_scale: pl.Tensor[[KV_LORA], pl.FP32],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.BF16]],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16]],
    kv_latent: pl.Out[pl.Tensor[[T_DYN, KV_LORA], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    query.bind_dynamic(0, T_DYN)
    kv_latent.bind_dynamic(0, T_DYN)
    mla_prolog_w8a8(x, w_q_a, w_q_a_scale, q_a_norm_weight, w_q_b, w_q_b_scale,
                    w_kv_a, w_kv_a_scale, kv_a_norm_weight, q_resid, query, kv_latent)
    return q_resid, query, kv_latent


def golden_mla_prolog_w8a8(values):
    q_a = w8a8_dynamic_linear(values['x'], values['w_q_a'], values['w_q_a_scale'])
    q_resid = rms_norm(q_a, values['q_a_norm_weight'])
    query = w8a8_dynamic_linear(q_resid, values['w_q_b'], values['w_q_b_scale'])
    kv_a = w8a8_dynamic_linear(values['x'], values['w_kv_a'], values['w_kv_a_scale'])
    values['q_resid'].copy_(q_resid)
    values['query'].copy_(query.reshape(-1, LOCAL_H, QK_DIM))
    values['kv_latent'].copy_(rms_norm(kv_a, values['kv_a_norm_weight']))


def run_native_case(options):
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run
    from models.glm5_3_flash.config import TP_SIZE
    from models.glm5_3_flash.utils import component_hidden
    from models.glm5_3_flash.weights import Checkpoint

    layouts = {
        'x': ((options.tokens, D), torch.bfloat16),
        'w_q_a': ((Q_LORA, D), torch.int8), 'w_q_a_scale': ((Q_LORA,), torch.float32),
        'q_a_norm_weight': ((Q_LORA,), torch.bfloat16),
        'w_q_b': ((LOCAL_H * QK_DIM, Q_LORA), torch.int8),
        'w_q_b_scale': ((LOCAL_H * QK_DIM,), torch.float32),
        'w_kv_a': ((KV_LORA, D), torch.int8), 'w_kv_a_scale': ((KV_LORA,), torch.float32),
        'kv_a_norm_weight': ((KV_LORA,), torch.bfloat16),
    }

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5356)
        values = {'x': component_hidden(options, D, generator)}
        if options.checkpoint is not None:
            values.update(Checkpoint(options.checkpoint).attention(options.layer, tp=TP_SIZE, mla_mode='native'))
        else:
            for name, (shape, dtype) in layouts.items():
                if name == 'x':
                    continue
                if dtype == torch.int8:
                    values[name] = torch.randint(-128, 128, shape, generator=generator, dtype=dtype)
                elif dtype == torch.float32:
                    values[name] = torch.rand(shape, generator=generator) * 0.001 + 0.0001
                else:
                    values[name] = torch.ones(shape, dtype=dtype)
        return values

    specs = [TensorSpec(name, shape, dtype, init_value=lambda key=name: inputs()[key])
             for name, (shape, dtype) in layouts.items()]
    for name, shape in (('q_resid', (options.tokens, Q_LORA)),
                        ('query', (options.tokens, LOCAL_H, QK_DIM)),
                        ('kv_latent', (options.tokens, KV_LORA))):
        specs.append(TensorSpec(name, shape, torch.bfloat16))
    result = run(fn=mla_prolog_w8a8_test, specs=specs, golden_fn=golden_mla_prolog_w8a8,
                 config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
                 compare_fn={name: ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0)
                             for name in ('q_resid', 'query', 'kv_latent')}, compile_only=options.compile_only)
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


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
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument('--mode', choices=['all', 'bf16', 'native'], default='all')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--hidden-input', type=Path)
    parser.add_argument('--layer', type=int, default=3)
    args = parser.parse_args()
    if args.checkpoint is not None or args.hidden_input is not None:
        if args.mode != "native":
            parser.error("--checkpoint and --hidden-input require --mode native")
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.mode in ("all", "native"):
        run_native_case(args)
        if args.mode == "native":
            return

    compare = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    result = run(
        fn=mla_prolog_test,
        specs=build_mla_prolog_tensor_specs(args.tokens),
        golden_fn=golden_mla_prolog_case,
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
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
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
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

    Both attention paths use this BF16 absorbed query: decode consumes it directly,
    and prefill absorbs before its per-token block loop. ``w_k`` is stored
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
    absorbed_flat = pl.reshape(absorbed, [t_dim, LOCAL_H * KV_LORA])
    for head in pl.range(LOCAL_H):
        query_head = pl.create_tensor([t_dim, QK_DIM], dtype=pl.BF16)
        weight_head = pl.create_tensor([QK_DIM, KV_LORA], dtype=pl.BF16)
        for block in pl.spmd(t_mm // MM_T_TILE, name_hint="mla_query_head"):
            t0 = block * MM_T_TILE
            rows = pl.min(MM_T_TILE, t_dim - t0)
            part = pl.slice(
                query, [MM_T_TILE, 1, QK_DIM], [t0, head, 0],
                valid_shape=[rows, 1, QK_DIM], drop_dims=[1],
            )
            query_head[t0:t0 + MM_T_TILE, :] = part
        for block in pl.spmd((QK_DIM // 128) * (KV_LORA // 128), name_hint="mla_key_head"):
            k0 = (block // (KV_LORA // 128)) * 128
            n0 = (block % (KV_LORA // 128)) * 128
            weight_part = pl.slice(w_k, [1, 128, 128], [head, k0, n0], drop_dims=[0])
            weight_head[k0:k0 + 128, n0:n0 + 128] = weight_part
        for block in pl.spmd(KV_LORA // ABSORB_N_TILE, name_hint="mla_absorb_query"):
            n0 = block * ABSORB_N_TILE
            for tc in pl.range(t_mm // MM_T_TILE):
                t0 = tc * MM_T_TILE
                rows = pl.min(MM_T_TILE, t_dim - t0)
                acc = bf16_linear(
                    query_head, weight_head, t0, n0, rows,
                    pl.cast(ABSORB_N_TILE, pl.INDEX), MM_T_TILE, ABSORB_N_TILE, QK_DIM, False,
                )
                value = pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, ABSORB_N_TILE)
                pl.assemble(absorbed_flat, value, [t0, head * KV_LORA + n0])
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


def golden_absorb_query(query: torch.Tensor, w_k: torch.Tensor) -> torch.Tensor:
    """Fold the key half into the query so it attends against the raw latent.

    The exact identity ``q . K^T = (q @ w_k) . latent^T`` uses
    ``K = latent @ w_k^T``. This reference rounds an independent wide product to FP32,
    then rounds the absorbed
    query to the input dtype. BF16 inputs therefore introduce an intermediate BF16
    rounding compared with an expanded FP32 key computation.

    Args:
        query: ``[T, H, QK_DIM]`` from :func:`golden_mla_prolog`.
        w_k: ``[H, QK_DIM, KV_LORA]`` key half from :func:`golden_split_kv_b`.

    Returns:
        ``[T, H, KV_LORA]`` query in latent space, in ``query``'s dtype.
    """
    absorbed = torch.einsum("thd,hdk->thk", query.double(), w_k.double()).float()
    return absorbed.to(query.dtype)


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
    "mla_prolog_w8a8",
    "mla_prolog_w8a8_test",
]


if __name__ == "__main__":
    main()
