# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare Native weights for an explicitly selected CSA kernel signature."""

import torch
from vllm.logger import logger

from .nz_mode import root_weight_layouts

_NZ_C0_BYTES = 32
_NZ_FRACTAL_ROWS = 16


def _base_weight_format(value: torch.Tensor) -> torch.Tensor:
    """仅为 ND 参数或离线打包承载张量在原设备上归一化基础格式。"""
    if value.device.type == "npu":
        import torch_npu

        if torch_npu.get_npu_format(value) not in (0, 2):
            return torch_npu.npu_format_cast(value, 2)
    return value


def _pack_nz(value: "torch.Tensor") -> "torch.Tensor":
    """把逻辑行主序的最后两维重排成 pto-isa 的 NZ 分形序，形状不变。

    NZ 是「列块在外、行分形在内」：c0 个连续元素构成一条 C0 线，16 行构成一个
    16 x c0 的分形，行轴上走 R/16 个分形，列轴上跨 C/c0 个列块——也就是
    BlockNzTensorViews 给 GM 视图的分块形状 [C/c0, R/16, 16, c0]。
    重排后仍按逻辑形状返回，只有字节次序变了，元素个数不变。
    `pl.NZ` 是对「GM 里的字节已经是这个次序」的断言，不是一个转换请求，
    按 pypto-lib/deepseek_v4_flash_dspark/utils.py 的 pack_nz 规则，在原设备重排。
    本 A3 对齐 BF16/INT8 权重的 Native 格式 29 与这里采用相同 fractal 规则；差异是描述符管理，
    以及各算子期待的矩阵方向/分组。本函数仅供离线回放布局转换；生产接入直接借用格式 29。
    """
    rows, cols = value.shape[-2], value.shape[-1]
    c0 = _NZ_C0_BYTES // value.element_size()
    if rows % _NZ_FRACTAL_ROWS or cols % c0:
        raise ValueError(f"NZ 需要 {_NZ_FRACTAL_ROWS} 行分形与整条 {c0} 元素的 C0 线，实到 {rows}x{cols}")
    logical = _base_weight_format(value.detach())
    packed = (
        logical.reshape(-1, rows // _NZ_FRACTAL_ROWS, _NZ_FRACTAL_ROWS, cols // c0, c0)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(value.shape)
    )
    return _base_weight_format(packed)


def _unpack_nz(value: "torch.Tensor") -> "torch.Tensor":
    """把已知为 PTO NZ 分形序的权重还原为逻辑 ND，供跨 mode 回放。"""
    rows, cols = value.shape[-2:]
    c0 = _NZ_C0_BYTES // value.element_size()
    if rows % _NZ_FRACTAL_ROWS or cols % c0:
        raise ValueError(f"不合法的 NZ 权重形状：{rows}x{cols}")
    unpacked = (
        value.reshape(-1, cols // c0, rows // _NZ_FRACTAL_ROWS, _NZ_FRACTAL_ROWS, c0)
        .permute(0, 2, 3, 1, 4)
        .contiguous().reshape(value.shape)
    )
    return _base_weight_format(unpacked)


def repack_weights(tensors: dict, source_layouts: dict, target_layouts: dict, target_shapes=None) -> dict:
    """离线快照显式转换：解包，迁移旧根矩阵方向，再按目标布局打包。"""
    result = dict(tensors)
    for name, target in target_layouts.items():
        source = source_layouts.get(name)
        if source not in ("ND", "NZ") or target not in ("ND", "NZ"):
            raise ValueError(f"未知权重布局：{name} {source} -> {target}")
        value = result[name]
        target_shape = tuple(target_shapes[name]) if target_shapes else tuple(value.shape)
        if source == target and tuple(value.shape) == target_shape:
            continue
        if source == "NZ":
            value = _unpack_nz(value)
        if tuple(value.shape) != target_shape:
            # 2026-09-26 前的 schema=2 快照使用 pypto-lib 矩阵方向。
            # 只迁移已知的三个转置矩阵，不能用 numel 相同猜测 reshape/分组。
            swapped = (*value.shape[:-2], value.shape[-1], value.shape[-2])
            if name not in ("wq_a", "wo_a", "wo_b") or swapped != target_shape:
                raise ValueError(f"不支持权重形状转换：{name} {tuple(value.shape)} -> {target_shape}")
            value = _base_weight_format(value.transpose(-1, -2).contiguous())
        result[name] = _pack_nz(value) if target == "NZ" else value
    return result


# torch_npu 的 acl format 取值：0=NCHW、2=ND，二者都是 PyPTO 根入参接受的基础格式。
_ACL_FORMAT_NCHW = 0
_ACL_FORMAT_ND = 2
_ACL_FORMAT_FRACTAL_NZ = 29


def prepare_weights(attention, hadamard: torch.Tensor | None, layer=None, *,
                    root_function) -> dict[str, torch.Tensor]:
    """Prepare the TP1 ABI from already-loaded Native parameters exactly once."""
    import torch_npu

    if attention.compress_ratio not in (4, 128) or attention.n_local_heads != 64 or attention.n_local_groups != 8:
        raise ValueError("PTO attention 要求 C4/C128、TP1、64 个 head 和 8 个输出组")

    layouts = root_weight_layouts(root_function)

    def _recast(name, value, current, target):
        # 声明的布局与 Native 实际存法不符时只能另开一份私有副本。这条路径静默走过
        # 一次就意味着每层多占一份权重，而且只会体现在 worker 的 weights 汇总里——
        # 2026-09-28 的 CANN 9.0 -> 9.2 升级就是这样让 wo_a 悄悄多吃了 1.31 GiB
        # （9.0 的 Native 把三维 BF16 留在 ND，9.2 改存 NZ）。所以显式报出来。
        logger.warning(
            "PTO_CSA_WEIGHT_RECAST %s: kernel 声明 %s 但 Native 存的是 format=%d，"
            "复制一份私有副本 %.2f MiB/层；请核对 nz_mode 的布局判据是否还跟得上当前 CANN",
            name, layouts[name], current, value.numel() * value.element_size() / (1 << 20),
        )
        return torch_npu.npu_format_cast(value, target)

    def root_weight(name, shape, dtype):
        # 根矩阵方向与 Native 相同。格式已匹配时借用原存储，禁止解包、转置或重新打包。
        value = getattr(attention, name).weight.detach()
        if tuple(value.shape) != shape or value.dtype != dtype or not value.is_contiguous():
            raise ValueError(f"Unexpected Native {name} weight: {value.shape}/{value.dtype}")
        current = int(torch_npu.get_npu_format(value))
        if layouts[name] == "NZ":
            return value if current == 29 else _recast(name, value, current, 29)
        if current in (_ACL_FORMAT_NCHW, _ACL_FORMAT_ND):
            return value
        return _recast(name, value, current, _ACL_FORMAT_ND)

    def weight(module, shape, dtype, transpose=False, layout_name=None):
        value = module.weight.detach()
        if tuple(value.shape) != shape or value.dtype != dtype:
            raise ValueError(f"Unexpected loaded weight: {value.shape}/{value.dtype}; expected {shape}/{dtype}")
        if torch_npu.get_npu_format(value) not in (_ACL_FORMAT_NCHW, _ACL_FORMAT_ND):
            # 先统一回 ND，才能按数学方向转置；四张目标权重由 root_weight 独立绑定，
            # 已有 NZ 存储直接复用，不走这里。
            value = torch_npu.npu_format_cast(value, _ACL_FORMAT_ND)
        if transpose:
            value = value.transpose(-1, -2)
        value = _base_weight_format(value.contiguous())
        # cube matmul 的 B 操作数若在根签名里声明了 NZ，就在加载期一次性转成分形序，
        # 之后每次矩阵乘直接以 NZ 读入 Mat，省掉片上的 ND→NZ 转换。布局由根签名说话
        # （见 nz_mode.B_OPERAND_WEIGHTS），这里不再写死。
        if layout_name is not None and layouts.get(layout_name) == "NZ":
            return torch_npu.npu_format_cast(value, _ACL_FORMAT_FRACTAL_NZ)
        return value

    def scale(module, width):
        result = module.weight_scale.detach().reshape(-1)
        if result.numel() != width:
            raise ValueError("Unexpected quantized channel-scale count")
        offset = getattr(module, "weight_offset", None)
        if offset is not None and bool(torch.count_nonzero(offset).cpu()):
            raise ValueError("The reference CSA chain requires symmetric INT8 weights")
        return result.float().contiguous()

    bf16, int8 = torch.bfloat16, torch.int8
    main = attention.compressor
    compressor_width = 1024 if attention.compress_ratio == 4 else 512
    # mHC 的门控权重与 attention 的 input_layernorm 挂在 DeepseekV4DecoderLayer 上。
    hc = {}
    if layer is not None:
        hc = {
            "hc_attn_fn": layer.hc_attn_fn.detach().float().contiguous(),
            "hc_attn_scale": layer.hc_attn_scale.detach().float().contiguous(),
            "hc_attn_base": layer.hc_attn_base.detach().float().contiguous(),
            "attn_norm_w": layer.input_layernorm.weight.detach().to(bf16).contiguous(),
        }
    weights = {
        **hc,
        "wq_a": root_weight("wq_a", (1024, 4096), bf16),
        "wq_b": root_weight("wq_b", (1024, 32768), int8),
        "wq_b_scale": scale(attention.wq_b, 32768),
        "wkv": weight(attention.wkv, (512, 4096), bf16, True, layout_name="wkv"),
        "gamma_cq": weight(attention.q_norm, (1024,), bf16),
        "gamma_ckv": weight(attention.kv_norm, (512,), bf16),
        "cmp_wkv": weight(main.wkv, (compressor_width, 4096), bf16, layout_name="cmp_wkv"),
        "cmp_wgate": weight(main.wgate, (compressor_width, 4096), bf16, layout_name="cmp_wgate"),
        "cmp_ape": main.ape.detach().float().contiguous(),
        # Match Native A3 storage; the RMS task widens loaded BF16 tiles.
        "cmp_norm_w": weight(main.norm, (512,), bf16),
        "attn_sink": attention.attn_sink.detach().contiguous(),
        "wo_a": root_weight("wo_a", (8, 4096, 1024), bf16),
        "wo_b": root_weight("wo_b", (8192, 4096), int8),
        "wo_b_scale": scale(attention.wo_b, 4096),
    }
    # HCA 不含 Indexer，公共权重准备只在 C4 分支绑定这些参数。
    if attention.compress_ratio == 4:
        indexer = attention.indexer
        inner = indexer.compressor
        weights.update({
            "idx_wq_b": weight(indexer.wq_b, (1024, 8192), int8),
            "idx_wq_b_scale": scale(indexer.wq_b, 8192),
            "weights_proj": weight(indexer.weights_proj, (64, 4096), bf16, True),
            **({"hadamard_idx": hadamard.detach().T.to(bf16).contiguous()} if hadamard is not None else {}),
            "inner_wkv": weight(inner.wkv, (256, 4096), bf16),
            "inner_wgate": weight(inner.wgate, (256, 4096), bf16),
            "inner_ape": inner.ape.detach().float().contiguous(),
            "inner_norm_w": weight(inner.norm, (128,), bf16),
        })
    return weights
