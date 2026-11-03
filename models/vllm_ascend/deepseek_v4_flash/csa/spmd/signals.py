# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CSA 同一个 SPMD 内的点对点信号：底层适配与阅读顺序。

用途与数据流
    生产者把业务结果写到 GM 后调用 notify；依赖它的消费者调用 wait。
    这两个 inline helper 留在调用者的核内，不创建 task、AICPU 往返或全核
    syncall。先读本文件的单次发布/等待，再读 phase_sync.py 的阶段与代际协议。
    主线 #3020 的正式接口可用后，在此集中替换底层调用与核对其内存语义。

入参与布局
    signal 为共享的一维 INT32 GM Tensor，offset 为元素偏移而非字节偏移，
    value/expected 为 INT32 单调进度。此层不绑定 batch、T 或信号容量；
    调用方保证下标合法、信号槽独占隔离区、每槽只有一个写者。AIC 和 AIV
    使用同一协议，核编号映射、槽间距、代际推进由 phase_sync.py 负责。

缓存与同步职责
    A3 各 Scalar 的 DCache 不一致。不能把普通 pl.write/pl.read 当作通知；
    底层 signal_notify/wait 必须维护信号可见性。notify 前的 PIPE_ALL 还要
    等待本核业务流水完成，尤其不能把 AIC 分流水 fence 当作 FIX 已完成。
    这里不维护所有业务 Tensor 的缓存，外层仍须执行相应发布/获取协议。
    wait 按当前接口的 GE 条件等待、保留信号值，多消费者不会互相消费标记。

生命周期与边界
    这是完成通知，不是互斥锁、累加计数器或消费者 ACK；不能据此提前覆盖
    尚未读完的业务缓冲。首次清零、顺序独占、代际上限与布局身份由外层保证。
    当前依赖本轮隔离 PyPTO/PTOAS 的原生 backoff_ticks 扩展，显式选择150个
    设备计时tick。首次读取不延迟，未就绪才退避，不能把tick解释成微秒。
    实际CCEC源码和设备同步证据见TND实现记录；生产服务不安装wait_codegen.py
    的测试源码改写。普通旧工具安装不具有此参数，须使用已记录的依赖补丁。
"""

import pypto.language as pl


@pl.jit.inline
def csa_signal_notify(
    signal: pl.InOut[pl.Tensor], offset: pl.Scalar[pl.INDEX], value: pl.Scalar[pl.INT32]
) -> pl.Tensor:
    """先排空本核流水，再发布指定槽；返回同一 GM Tensor 的引用。"""
    # Scalar 发布前排空本核流水。AIC 的 gm fence 仅生成分流水 barrier，
    # 不能据此认定 Scalar 已等待 FIX 写回；这里复用核内 PIPE_ALL。
    # 这不是 syncall，不等待其他核集合；调用方仍负责 payload 的缓存维护。
    pl.system.bar_all()
    pl.system.signal_notify(signal, [offset], value)
    return signal


@pl.jit.inline
def csa_signal_wait(
    signal: pl.InOut[pl.Tensor], offset: pl.Scalar[pl.INDEX], expected: pl.Scalar[pl.INT32]
) -> pl.Tensor:
    """等待该槽达到 expected，不清零；业务数据可见性由外层保证。"""
    # 等待不清除信号，多个消费者可以独立等待同一个生产者。
    pl.system.signal_wait(signal, [offset], expected, backoff_ticks=150)
    return signal
