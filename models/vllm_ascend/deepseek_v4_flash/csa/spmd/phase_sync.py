# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态 CSA 单 SPMD 的阶段通知协议：整体思路与阅读顺序。

一、用途与数据流
    数学阶段在一个常驻 MIX SPMD 中运行。生产者完成业务 Tensor 写回后，
    只发布自己的阶段信号；消费者只等待实际依赖的生产者。这里不创建
    task，不传入历史 records/waits 表，也不把每个阶段变成全核屏障。
    阅读顺序：工作区约定 → begin_invocation → publish_phase → wait_*。

二、输入、布局和动态范围
    signals 是独占的一维 INT32 GM Tensor，长度保留 pl.dynamic。布局为
    [阶段数, 参与核数, 128 个 INT32]；每个槽只使用首元素，其余是隔离区。
    沿用此前验证过的 512 字节隔离跨度，不声称这就是硬件最小缓存行。
    阶段 0 保存每核调用代际，其余阶段由调用方使用有含义的阶段常量。
    核编号约定为 AIC 的 [0,G)、AIV 的 [G,3G)，G 是该工作区固定的常驻核组数。
    阶段数和核数描述算法/硬件，不是固定 batch；行范围来自实际 T/B。

三、所有权、同步与缓存
    (阶段, 核编号) 只有该核写，多个消费者可以等同一槽，因此不需要
    多写者 atomic。Scalar DCache 不一致：signal_notify/wait 负责信号
    的发布/读取，业务 Tensor 仍需本核流水完成及缓存维护，不能混为一谈。
    本模块调用通用 signals 适配，发布前排空流水；等待结束后再刷新业务
    数据的本地缓存。组内 Cube/Vector FIFO/FFTS 握手仍由数学阶段自己负责。

四、生命周期和正确性边界
    工作区须在首次调用前清零，并由单个顺序执行的图/流独占。每次调用
    所有参与核必须且只能调用一次 begin_invocation，即使该核没有数学
    工作也不能跳过；否则不同核的代际会不一致。执行失败后不可直接复用
    部分推进的工作区，必须在设备空闲时重新初始化。INT32 代际到上限前
    同样需要在捕获外重建工作区，禁止在核内回绕清零造成 ABA。
    同一工作区的 G、核编号映射、阶段槽数及阶段含义必须保持不变；仅改变
    业务 T/B/metadata 不改变此合同。若新策略改变参与核集合或阶段布局，
    必须绑定另一份清零的新工作区，不能只因容量足够就复用旧信号内容。
    publish_phase 必须位于该生产者本阶段全部写回之后；零工作量参与者
    也应按调用方合同发布完成，或由消费者的所有权计算将其排除。
    这是整阶段完成通知，没有消费者 ACK。不能对同一槽/代际逐 tile 重复
    发布，也不能在消费者读完前覆盖业务缓冲；需要环形槽复用时，必须另设
    进度或反向依赖。消费者未消费信号，不代表已经释放生产者的数据空间。

五、当前状态
    这是可复用的动态控制组件，尚不等于完整 CSA。wait_row_owners 只适用
    “按连续行 tile 轮转分给 worker”的布局；请求分组、split-K 和多维
    投影应显式枚举自己的真实 owner，不能误用本函数减少依赖。
"""

import pypto.language as pl

from .signals import csa_signal_notify, csa_signal_wait

SIGNAL_WORDS = pl.dynamic("CSA_PHASE_SIGNAL_WORDS")
SIGNAL_SLOT_WORDS = 128
MAX_SIGNAL_GENERATION = (1 << 31) - 1


def signal_word_count(*, phase_count: int, core_groups: int) -> int:
    """捕获前按已有算法/硬件参数确定容量，不读取任何 device Tensor。"""
    if type(phase_count) is not int or phase_count < 2:
        raise ValueError("阶段数须包括代际槽与至少一个计算阶段")
    if type(core_groups) is not int or core_groups <= 0:
        raise ValueError("常驻 MIX 核组数必须为正整数")
    return phase_count * (3 * core_groups) * SIGNAL_SLOT_WORDS


@pl.jit.inline(auto_scope=False)
def begin_invocation(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    core_id: pl.Scalar[pl.INDEX],
):
    """每核从自己的阶段 0 槽取下一代；返回 Tensor 引用和 INT32 代际。

    这里没有全卡集合。各核从零开始且每调用各推进一次，才保证相同代际。
    输入合法性与容量由捕获前的调用层保证，不能在越界后依赖信号自愈。
    """
    own_offset = core_id * SIGNAL_SLOT_WORDS
    pl.system.cacheinvalid(signals, [1], [own_offset])
    pl.system.fence()
    prior = pl.read(signals, [own_offset])
    generation = prior + pl.cast(1, pl.INT32)
    signals = csa_signal_notify(signals, own_offset, generation)
    return signals, generation


@pl.jit.inline(auto_scope=False)
def publish_phase(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    phase: pl.Scalar[pl.INDEX],
    core_id: pl.Scalar[pl.INDEX],
    participant_count: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
):
    """完成业务写回后发布本核阶段，不通知或等待无关消费者。"""
    # fence/cacheinvalid 维护业务 GM 数据；notify 内部的 PIPE_ALL 确保
    # AIC FIX、AIV MTE3 等本核流水已完成，再发布 Scalar 信号。
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    offset = (phase * participant_count + core_id) * SIGNAL_SLOT_WORDS
    signals = csa_signal_notify(signals, offset, generation)
    return signals


@pl.jit.inline(auto_scope=False)
def wait_producer(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    phase: pl.Scalar[pl.INDEX],
    producer_core: pl.Scalar[pl.INDEX],
    participant_count: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
):
    """等待一个实际生产者；不会清零或消费它的信号。"""
    offset = (phase * participant_count + producer_core) * SIGNAL_SLOT_WORDS
    signals = csa_signal_wait(signals, offset, generation)
    # 读到完成标记后再维护本核业务数据缓存；信号的可见性不能替代这一步。
    pl.system.fence()
    pl.system.cacheinvalid()
    pl.system.fence()
    return signals


@pl.jit.inline(auto_scope=False)
def wait_row_owners(
    signals: pl.InOut[pl.Tensor[[SIGNAL_WORDS], pl.INT32]],
    phase: pl.Scalar[pl.INDEX],
    participant_count: pl.Scalar[pl.INDEX],
    generation: pl.Scalar[pl.INT32],
    first_row: pl.Scalar[pl.INDEX],
    row_count: pl.Scalar[pl.INDEX],
    rows_per_tile: pl.Scalar[pl.INDEX],
    producer_workers: pl.Scalar[pl.INDEX],
    producer_base: pl.Scalar[pl.INDEX],
    producer_stride: pl.Scalar[pl.INDEX],
):
    """按行 tile 的轮转所有权，等待覆盖目标行区间的不同生产者。

    tile k 归 worker k % producer_workers；其硬件核编号为 base+worker*stride。
    连续 tile 的 owner 周期等于 worker 数，因此只需等待 min(tile数,worker数)
    个不同 owner。该公式由工作划分决定，与某个历史泳道图的执行次序无关。
    所有 divisor 必须为正数；first_row/count 必须是同一个业务 Tensor 的有效范围。
    """
    if row_count > 0:
        first_tile = first_row // rows_per_tile
        end_tile = (first_row + row_count + rows_per_tile - 1) // rows_per_tile
        owner_count = pl.min(end_tile - first_tile, producer_workers)
        # 即使生产者负责多个 tile，也只等待它完成整个阶段的同一个信号。
        # 本循环的次数取决于实际消费行范围；不会生成七个测试点的等待表。
        for ordinal in pl.range(owner_count):
            owner = (first_tile + ordinal) % producer_workers
            core = producer_base + owner * producer_stride
            offset = (phase * participant_count + core) * SIGNAL_SLOT_WORDS
            signals = csa_signal_wait(signals, offset, generation)
        # 多个 owner 的信号全部满足后统一刷新业务缓存，避免每条依赖都刷新整份 GM。
        pl.system.fence()
        pl.system.cacheinvalid()
        pl.system.fence()
    return signals
