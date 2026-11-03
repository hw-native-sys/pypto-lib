# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CSA原生TND单SPMD：独立实现与Native服务接入。

本包从99e29cb3的deepseek_v4_flash_csa_spmd复制，保留已采用的数学和流水优化，
独立修正TND合同。原包保留作S6同轮基线，不能把其历史设备通过直接记为本包通过。
本轮进度、任务与支持边界见上级DSV4_CSA_SPMD_TND_IMPLEMENTATION_LOG.md。

数据流：hidden[T_capacity,4,4096] → HC_pre / input RMSNorm → QKV与两条
Compressor → Indexer Score/Top-K → sparse Attention → O投影 / HC_post。
entry.py是唯一JIT入口，cube_schedule/vector_schedule安排一次24组MIX中的
AIC/AIV分工；阶段通过signal代际与必要的READY/ACK握手协作。

T是输入行容量，B是Native请求槽数。Q=query_start_loc、L=seq_lens直接作为
已有device Tensor传入；请求区间为[Q[r],Q[r+1])，历史起点为L[r]-query_len。
活动请求长度首版为1..6且可以不同；T容量可以包含dummy，不从T/B反推请求。

阅读顺序：
1. request_geometry定义真实边界；metadata_stages计算C4紧凑偏移和有效行数。
2. compressor_pool/output与indexer_compressor/cache各自保留Native state/cache。
3. indexer_group/score/topk/publish负责请求内分组、尾组和真实可见历史。
4. attention_plan/dependencies/stream负责有效块、必要等待和QK/PV流水。
5. binding负责捕获前分配，rebind仅更新同描述符根Tensor且不清零信号代际。
6. native_adapter直接绑定Native页存储；service_config限定已实现的策略范围；
   service管理资格判定、同step metadata复用、工作区与Native connector生命周期。

内部tile对齐与容量不是有效工作量。失活请求不能读写Native cache/state；
共享压缩历史必须全体读完再提交state。Scalar通知可见不代表业务数据自动可见。
图重放跳过Python服务判定，完整模型图必须另外检查捕获的容量、策略和地址合同。
CPU描述符检查、源码动态维度和单次设备结果均不代替完整混长及连续状态验收。

导入本包不初始化NPU，不加载torch或PyPTO编译器。
"""

__all__ = []
