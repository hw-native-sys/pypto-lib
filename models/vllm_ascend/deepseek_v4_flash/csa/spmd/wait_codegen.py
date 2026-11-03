# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""可选 GM 信号等待策略：生成源码变换，不负责 CSA 数学或任务调度。

整体流程与阅读顺序
    输入是一份已由 PTOAS 生成、尚未交给 CCEC 的 MIX kernel C++ 源码；
    rewrite_wait_source 检查调用合同、替换 TWAIT 并插入 WAIT_HELPER，返回
    新源码与替换统计。先读 WAIT_HELPER 的轮询，再读替换函数的拒绝条件。
    该变换与 B/T/S 无关，不实值化动态形状，也不增加任务或工作区。

信号布局、分工与缓存
    AIC/AIV 都轮询调用点传入的单元素 INT32 视图（signal.data()[0]）。
    调用方必须先确认所有被替换 TWAIT 都满足此合同；这不是任意 Tensor 的
    通用等待函数。槽的单写者、代际和隔离区由
    phase_sync.py 保证；此处只支持 GE，不更改发布端或业务数据维护。
    每次读前 DCCI，再用 volatile 读取，避免只命中本核 Scalar 旧缓存。
    这不是 ld_dev，不声称必然访问 HBM。未就绪时等待 100 个设备计时 tick
    再访问该槽；tick 不能解释成微秒。首次检查没有延迟，周期 PIPE_ALL
    只维护本核流水，不是全核集合。

接入与验证边界
    编译集成必须显式调用并核对替换数量；仅导入本模块不会改变 JIT。
    这条退避策略来自历史实验，新的动态组件 CPU 探针尚未应用它，不能
    把历史等待性能直接用于新入口。当前仍由隔离编译进程安装源码回调，
    不修改共享 PyPTO backend；返回的 device_validated=False 表明本次
    变换自身没有设备证据。完整算子的精度、同步和性能仍需另行验证。
"""

WAIT_BACKOFF_TICKS = 100

WAIT_HELPER = r"""
template <typename Signal>
static AICORE inline void CSA_WAIT_LOCAL_GM(Signal& signal, int32_t expected, pto::comm::WaitCmp cmp) {
  // 单写者单调进度。写侧仍由 TNOTIFY 清理/失效并 DSB，payload 由调用方维护。
  PTO_ASSERT(cmp == pto::comm::WaitCmp::GE, "CSA local wait only supports GE");
  auto ptr = reinterpret_cast<volatile __gm__ int32_t*>(signal.data());
  uint32_t spin = 0;
  while (true) {
    __asm__ __volatile__("" ::: "memory");
    dcci((__gm__ void*)ptr, cache_line_t::SINGLE_CACHE_LINE);
    __asm__ __volatile__("" ::: "memory");
    if (*ptr >= expected) break;
    // 首次读取不延迟；只有尚未就绪时，短暂只读本核计时寄存器。
    const uint64_t csa_wait_start = get_sys_cnt();
    while ((uint64_t)(get_sys_cnt() - csa_wait_start) < 100) {
      __asm__ __volatile__("" ::: "memory");
    }
    PTO_ASSERT(spin < 100000000, "CSA local wait spin limit");
    if ((++spin & 63) == 0) {
      pipe_barrier(PIPE_ALL);
    }
  }
}
"""


def rewrite_wait_source(source):
    """只替换本次完整 MIX SPMD 的等待；保留发布和业务数据维护。"""
    needle = "pto::comm::TWAIT("
    marker = "using namespace pto;"
    count = source.count(needle)
    # 拒绝空替换、无法确定的注入位置和重复变换；不能静默返回一个实际
    # 没有采用候选等待策略的产物，再把它标成该策略的性能证据。
    if count <= 0 or source.count(marker) != 1 or "CSA_WAIT_LOCAL_GM(" in source:
        raise ValueError("融合源码的等待/命名空间合同改变或已经执行过替换")
    source = source.replace(needle, "CSA_WAIT_LOCAL_GM(")
    source = source.replace(marker, marker + "\n" + WAIT_HELPER)
    return source, {
        "wait_sites": count,
        "reader": "dcci_then_volatile_load",
        "backoff_syscnt_ticks": WAIT_BACKOFF_TICKS,
        "ramp_backoff": False,
        "initial_backoff_syscnt_ticks": WAIT_BACKOFF_TICKS,
        "writer_unchanged": True,
        "payload_maintenance_unchanged": True,
        "device_validated": False,
    }
