# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""权重 NZ 布局的开关，取自 vllm-ascend 统一的那一套。

`weight_nz_mode` 的语义与 `vllm_ascend/ascend_config.py` 一致：
0 = 关闭，1 = 只有量化（INT8）权重走 NZ（默认），2 = BF16/FP16 权重也走 NZ。

这里读的是环境变量 `VLLM_ASCEND_ENABLE_NZ`，而不是 `AscendConfig.weight_nz_mode`，
原因是 kernel 的参数布局是**模块加载时**由类型注解定下来的，而 AscendConfig 要等
vllm 初始化完才拿得到。模型配置检查与算子注册前均校验环境、AscendConfig 和已导入的
布局模式一致；导入后修改环境也会被拒绝。
布局标注与 Native 存储绑定必须由同一个开关驱动，否则 kernel 读到的字节次序就是错的。

注意 NZ 分支对 kernel 写法有额外要求：切片偏移必须能被证明非负、且行偏移是 16 的
倍数、列偏移是一条 C0 线的倍数。性能版四张目标权重均声明可选 NZ，
真实布局读取根签名；四张根矩阵方向与 Native 加载后相同，wo_b 保留二维 [G*K, D]。
"""

import inspect

import pypto.language as pl

from .options import weight_nz_mode

WEIGHT_NZ_MODE = weight_nz_mode()
if WEIGHT_NZ_MODE not in (0, 1, 2):
    raise ValueError(f"VLLM_ASCEND_ENABLE_NZ must be 0, 1 or 2, got {WEIGHT_NZ_MODE}")

# BF16/FP16 权重是否按 NZ 分形序存放
BF16_WEIGHT_NZ = WEIGHT_NZ_MODE >= 2
# INT8 量化权重是否按 NZ 分形序存放。注意这一档在 vllm-ascend 的默认值（1）下就是
# 开的——也就是说 wq_b / wo_b 走 NZ 不需要把 mode 调到 2，与 BF16 权重不同。
QUANT_WEIGHT_NZ = WEIGHT_NZ_MODE >= 1

# 直接放进 pl.Tensor 的第三个槽：None 等价于不声明 layout（即 ND）。
# PyPTO 支持把 layout 放在闭包变量里，见 pypto/python/pypto/jit/cache.py 的说明。
BF16_WEIGHT_LAYOUT = pl.NZ if BF16_WEIGHT_NZ else None
QUANT_WEIGHT_LAYOUT = pl.NZ if QUANT_WEIGHT_NZ else None


def _native_keeps_3d_bf16_as_nz() -> bool:
    """Native 会不会把三维 BF16 权重存成 NZ——实测一次，不按版本号猜。

    `wo_a` 是三维分组权重 `[O_GROUPS, O_GROUP_IN, O_LORA]`。它在 Native 侧最终是
    ND 还是 NZ，取决于当前 CANN 的 `npu_format_cast` 支不支持三维 BF16：
    CANN 9.0.0 不支持，于是 Native 侧留在 ND(2)；9.2.0 支持，于是是 NZ(29)。
    CSA/HCA 在 vllm-ascend 侧共用钩子代码，差异只来自 CANN。

    这一项必须跟着 Native 走，写死任何一边都会在另一边付出代价：kernel 声明的
    布局与 Native 实际存法不一致时，`root_weight` 就要 npu_format_cast 出一份
    私有副本——`wo_a` 每层 64 MiB、21 个 ratio-4 层合计 **1.31 GiB**，直接从
    KV cache 的额度里扣。两个方向都实测过，数字一样大。

    探针只分配 4 KiB，且用当前设备；拿不到设备或调用失败时回落到与其它 BF16
    权重相同的档位（即本开关引入之前的行为），并由 `native_adapter.root_weight`
    的不匹配告警兜底。
    """
    try:
        import torch  # noqa: PLC0415
        import torch_npu  # noqa: PLC0415

        if not torch.npu.is_available() or not torch.npu.is_initialized():
            return BF16_WEIGHT_NZ
        probe = torch.empty((2, 32, 32), dtype=torch.bfloat16, device=torch.npu.current_device())
        return int(torch_npu.get_npu_format(torch_npu.npu_format_cast(probe, 29))) == 29
    except Exception:  # noqa: BLE001 - 探针失败不该拖垮导入
        return BF16_WEIGHT_NZ


# wo_a 单独一档：它跟随的不是全局 BF16 档位，而是 Native 对三维 BF16 的实际存法。
# 详见 `_native_keeps_3d_bf16_as_nz` 与
# tests/pypto_test/results/mem_128k_b24_20260928/ANALYSIS.md。
WO_A_WEIGHT_LAYOUT = pl.NZ if (BF16_WEIGHT_NZ and _native_keeps_3d_bf16_as_nz()) else None


def validate_weight_nz_mode(effective_mode: int) -> None:
    """只在初始化阶段检查，禁止配置或环境与已经固定的根布局分叉。"""
    environment_mode = weight_nz_mode()
    if effective_mode not in (0, 1, 2) or not (
        effective_mode == environment_mode == WEIGHT_NZ_MODE
    ):
        raise ValueError(
            "PTO CSA weight NZ mode mismatch: "
            f"AscendConfig={effective_mode}, environment={environment_mode}, "
            f"imported_layout={WEIGHT_NZ_MODE}; set VLLM_ASCEND_ENABLE_NZ "
            "to the requested weight_nz_mode before importing PTO kernels"
        )


# 全部作为 cube matmul B 操作数消费的权重。前四张是零拷贝借用 Native 存储的
# "根权重"；后三张（HCA 的 KV／压缩投影权重）历史上一律按 ND 准备，但它们同样
# 是 B 操作数，布局也应当由根签名说话，而不是写死在加载期。根签名各不相同
# （CSA 没有 cmp_wgate），所以只读实际存在的参数。
B_OPERAND_WEIGHTS = ("wq_a", "wq_b", "wo_a", "wo_b", "wkv", "cmp_wkv", "cmp_wgate")


def root_weight_layouts(root_function) -> dict[str, str]:
    """读取本仓根函数的实际注解；日志与存储绑定共用，避免另维护一份 NZ 名单。"""
    # pl.jit returns JITFunction with a generic __call__; its annotated Python
    # root is stored in _func by the pinned shared PyPTO release.
    params = inspect.signature(getattr(root_function, "_func", root_function)).parameters
    result = {}
    for name in B_OPERAND_WEIGHTS:
        if name not in params:
            continue
        layout = params[name].annotation.layout
        if layout is None:
            result[name] = "ND"
        elif layout == pl.NZ:
            result[name] = "NZ"
        else:
            raise ValueError(f"Unsupported PTO CSA weight layout: {name}={layout}")
    return result


def root_weight_shapes(root_function) -> dict[str, tuple[int, ...]]:
    """从根签名读取 Native 矩阵几何，回放按显式来源迁移旧转置权重。"""
    params = inspect.signature(getattr(root_function, "_func", root_function)).parameters
    return {name: tuple(params[name].annotation.shape) for name in root_weight_layouts(root_function)}
