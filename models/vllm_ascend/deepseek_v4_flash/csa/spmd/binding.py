# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""动态CSA捕获前的Native无损绑定及工作区准备。

此层只使用Tensor shape/stride/dtype和已有Host描述符，不读device metadata
的值，不计算逐请求地址，不提交meta kernel，也不转换Native权重dtype/布局。
所有有效请求/compact行/历史叶/尾块仍由单SPMD内的AIV/AIC计算。输入字典
沿用当前CSA Native适配层的命名，服务通过本包CSAServiceRuntime准备并提交。

工作区按数学最坏容量分配而非某个S=6样例：compact数量最多T，group数量最多T，
query分组余量与Score双槽按所选策略预留。所有Native cache/state为同一存储
别名；HC base、query/head的多视图也只是view。Indexer部分重叠的key视图在
JIT编排内部建立，只把Native cache本体作为外部根入参。
权重保持原对象；两份RMS norm继续用Native BF16，在初始化时不升级为FP32。

读取顺序：输入映射→descriptor校验→容量分配→只读别名→信号初始化→直调。
调用对象独占工作区，捕获前准备，重放时只改现有输入内容；signal只在首次
清零。失败或代际溢出必须在设备空闲时重建，禁止跨并发流复用。本文件不
承诺未经实测的形状或历史区间，证据见独立TND实现记录。
"""

from collections.abc import Callable, Mapping

import torch

from ..common.options import weight_nz_mode

from .policy import CSASPMDPolicy, select_policy


def _pad(value: int, block: int) -> int:
    return (value + block - 1) // block * block


class PreparedDynamicCSA:
    """捕获前准备；allocator可供测试加入保护区，生产默认直接分配独立Tensor。"""

    def __init__(
        self,
        native: Mapping[str, torch.Tensor],
        *,
        max_query_tokens: int,
        max_seq_len: int,
        allocator: Callable | None = None,
        policy: CSASPMDPolicy | None = None,
    ):
        if weight_nz_mode() != 2:
            raise NotImplementedError("完整动态SPMD当前先接Native NZ2，其他布局尚未完成整链验证")
        self.native = dict(native)
        hidden = native["x_hc"]
        self.tokens = t = hidden.shape[0]
        self.requests = b = native["kv_seq_lens"].numel()
        if hidden.shape != (t, 4, 4096) or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
            raise ValueError("要求Native BF16连续hidden[T,4,4096]")
        self.policy = policy or select_policy(
            tokens=t, requests=b, max_query_tokens=max_query_tokens, max_seq_len=max_seq_len
        )
        device = hidden.device
        self.buffers: dict[str, torch.Tensor] = {}
        # 名称变更只对应接口职责，Tensor保持原对象；同一组内Q/L分别保留。
        direct = {
            "hc_fn": "hc_attn_fn",
            "hc_scale": "hc_attn_scale",
            "hc_base": "hc_attn_base",
            "norm_w": "attn_norm_w",
            "wq_a": "wq_a",
            "gamma_cq": "gamma_cq",
            "wq_b": "wq_b",
            "wq_b_scale": "wq_b_scale",
            "rope_cos": "freqs_cos",
            "rope_sin": "freqs_sin",
            "wkv": "wkv",
            "gamma_ckv": "gamma_ckv",
            "native_slots": "ori_slot_mapping",
            "main_wkv": "cmp_wkv",
            "main_wgate": "cmp_wgate",
            "main_state": "compress_state",
            "main_state_table": "state_block_table",
            "main_ape": "cmp_ape",
            "positions": "position_ids",
            "main_bounds": "cmp_query_start_loc",
            "main_lengths": "cmp_seq_lens",
            "main_state_slots": "state_slot_mapping",
            "main_gamma": "cmp_norm_w",
            "main_cos": "cmp_freqs_cos",
            "main_sin": "cmp_freqs_sin",
            "main_slots": "cmp_slot_mapping",
            "idx_wkv": "inner_wkv",
            "idx_wgate": "inner_wgate",
            "idx_state": "inner_compress_state",
            "idx_state_table": "inner_state_block_table",
            "idx_ape": "inner_ape",
            "idx_bounds": "idx_query_start_loc",
            "idx_lengths": "kv_seq_lens",
            "idx_state_slots": "inner_state_slot_mapping",
            "idx_gamma": "inner_norm_w",
            "idx_cos": "inner_freqs_cos",
            "idx_sin": "inner_freqs_sin",
            "idx_slots": "idx_slot_mapping",
            "hadamard": "hadamard_idx",
            "idx_cache": "idx_native_kv_cache",
            "original_table": "ori_block_table",
            "compressed_table": "cmp_block_table",
            "indexer_table": "idx_block_table",
            "idx_wq_b": "idx_wq_b",
            "idx_wq_b_scale": "idx_wq_b_scale",
            "weights_proj": "weights_proj",
            "topk_scores": "idx_topk_scores",
            "topk_indices": "idx_topk",
            "wo_a": "wo_a",
            "wo_b": "wo_b",
            "wo_b_scale": "wo_b_scale",
        }
        self._direct_names = tuple(direct.items())
        self.buffers.update({target: native[source] for target, source in direct.items()})
        for name in ("main_bounds", "idx_bounds"):
            if self.buffers[name].shape != (b + 1,) or self.buffers[name].dtype != torch.int32:
                raise ValueError(f"{name}须为Native INT32[B+1]")
        for name in ("main_lengths", "idx_lengths"):
            if self.buffers[name].shape != (b,) or self.buffers[name].dtype != torch.int32:
                raise ValueError(f"{name}须为Native INT32[B]")
        for name, width in (
            ("norm_w", 4096),
            ("gamma_cq", 1024),
            ("gamma_ckv", 512),
            ("main_gamma", 512),
            ("idx_gamma", 128),
        ):
            value = self.buffers[name]
            if value.shape != (width,) or value.dtype != torch.bfloat16:
                raise ValueError(f"{name}必须保持Native BF16[{width}]")
        for name in ("wq_a", "wq_b", "wo_a", "wo_b"):
            if device.type == "npu":
                import torch_npu

                if torch_npu.get_npu_format(self.buffers[name]) != 29:
                    raise ValueError(f"{name}须由Native已有加载流程提供NZ格式")
        for name, value in self.buffers.items():
            if value.device != device or not value.is_contiguous():
                raise ValueError(f"{name}须在同一设备并满足连续GM描述符")

        self.buffers.update(
            x=hidden.view(t, 16384),
            output=native["x_out"].view(t, 16384),
            hc_base_2d=native["hc_attn_base"].view(1, 24),
            cache_rows=native["kv_cache"].view(-1, 512),
            main_cache=native["cmp_kv"].view(-1, 512),
            attn_sink=native["attn_sink"].view(64, 1),
        )
        p8, p16, p64, p128 = (_pad(t, n) for n in (8, 16, 64, 128))
        group = self.policy.query_group
        mm = max(p128, _pad(t, self.policy.ob_rows))
        # dtype从用途决定，不通过填充区的dtype转换改变算子数学；容量独立于有效Q/L。
        specs = {
            "x_fp32": ((t, 16384), torch.float32),
            "inv": ((p16, 1), torch.float32),
            "widen_tail": ((8, 16384), torch.float32),
            "partials": ((p16 * 4, 32), torch.float32),
            "raw": ((p16, 32), torch.float32),
            "pre": ((p16, 8), torch.float32),
            "post": ((t, 4), torch.float32),
            "post_tail": ((8, 8), torch.float32),
            "comb": ((t, 16), torch.float32),
            "comb_tail": ((8, 32), torch.float32),
            "mixed": ((p8, 4096), torch.bfloat16),
            "normed": ((t, 4096), torch.bfloat16),
            "norm_tail": ((8, 4096), torch.bfloat16),
            "qa": ((p16, 1024), torch.float32),
            "qr": ((t, 1024), torch.int8),
            "qr_scale": ((t, 1), torch.float32),
            "qr_matmul": ((p128, 1024), torch.int8),
            "qr_scale_pad": ((p8, 1), torch.float32),
            "rope_signed": ((t, 64), torch.float32),
            "rope_swap": ((t, 64), torch.int32),
            "q_proj": ((p128, 32768), torch.int32),
            "q_flat": ((t, 32768), torch.bfloat16),
            "kv_fp32": ((p16, 512), torch.float32),
            "kv": ((t, 512), torch.bfloat16),
            "main_value": ((p64, 1024), torch.float32),
            "main_score": ((p64, 1024), torch.float32),
            "main_pooled": ((t, 512), torch.float32),
            "main_offsets": ((b,), torch.int32),
            "main_count": ((1,), torch.int32),
            "main_normed": ((t, 512), torch.float32),
            "main_debug": ((t, 512), torch.float32),
            "idx_value": ((p16, 256), torch.float32),
            "idx_score": ((p16, 256), torch.float32),
            "idx_pooled": ((t, 128), torch.float32),
            "idx_offsets": ((b,), torch.int32),
            "idx_count": ((1,), torch.int32),
            "idx_normed": ((p16, 128), torch.bfloat16),
            "idx_had": ((p16, 128), torch.float32),
            "idx_scales": ((p16, 1), torch.float32),
            "idx_debug": ((t, 128), torch.float32),
            "idx_query_acc": ((p64, 8192), torch.int32),
            "idx_query_rope": ((p16, 8192), torch.bfloat16),
            "idx_query_had": ((t * 64, 128), torch.float32),
            "idx_query_i8": ((t * 64, 128), torch.int8),
            "idx_query_scale": ((t * 64, 1), torch.float32),
            "weights_partial": ((p16 * 4, 64), torch.float32),
            "head_weights": ((p16, 64), torch.float32),
            # 前缀+4*T组描述符，16项对齐后接48个scale链表头和4*T项地址描述符；
            # 启用连续页策略时尾部2*B项为(first_page,direction)，关闭时维持原容量。
            "group_offsets": (
                (_pad(_pad(b + 1, 16) + 4 * t, 16) + 48 + 4 * t
                 + (2 * b if self.policy.scale_page_coalescing else 0),),
                torch.int32,
            ),
            "group_plan": ((8,), torch.int32),
            "coefficients": ((t * 16, group * 64), torch.float16),
            "indexer_transfer": ((24 * 2 * group, self.policy.score_tile), torch.float32),
            "score_arena": ((24 * 2 * group, 8192), torch.float32),
            "pair_arena": ((t * 64, 1024), torch.float32),
            "attention_indices": ((p8, 512), torch.int32),
            "attention_bias": ((p8, 640), torch.float32),
            "attention_mask": ((p8, 16), torch.int32),
            "inverse_sin": ((t, 64), torch.float32),
            "kv_ring": ((24 * 3 * 128, 512), torch.bfloat16),
            "score_ring": ((24 * 3 * 64, 128), torch.float32),
            "probability_ring": ((24 * 3 * 64, 128), torch.bfloat16),
            "pv_ring": ((24 * 3 * 64, 512), torch.float32),
            "mi_ring": ((24 * 3 * 64, 1), torch.float32),
            "li_ring": ((24 * 3 * 64, 1), torch.float32),
            "packed": ((8 * p128, 4096), torch.bfloat16),
            "ffts": ((256,), torch.int64),
            "o_r": ((mm, 8192), torch.float32),
            "o_quant": ((mm, 8192), torch.int8),
            "o_scales": ((8, p128), torch.float32),
            "o_partials": ((mm, 32768), torch.int32),
        }
        from .runtime_abi import PHASE_COUNT

        specs["signal"] = ((PHASE_COUNT * 72 * 128,), torch.int32)
        self.allocations = {}
        for name, (shape, dtype) in specs.items():
            value = (
                torch.empty(shape, dtype=dtype, device=device)
                if allocator is None
                else allocator(name, shape, dtype, device)
            )
            if value.shape != shape or value.dtype != dtype or value.device != device or not value.is_contiguous():
                raise ValueError(f"allocator返回错误的工作区：{name}")
            self.buffers[name] = self.allocations[name] = value
        self.buffers.update(
            q_heads=self.buffers["q_flat"].view(t * 64, 512),
            idx_query_rope_heads=self.buffers["idx_query_rope"].view(p16 * 64, 128),
            idx_query_scale_heads=self.buffers["idx_query_scale"].view(t, 64),
        )
        cache = self.buffers["idx_cache"]
        if cache.dtype != torch.int8 or cache.ndim != 2 or cache.shape[1] < 4160 or cache.shape[1] % 64:
            raise ValueError("Indexer cache须为Native对齐INT8页，页内至少4096B key+64B scale")
        signal = self.buffers["signal"]
        if device.type != "meta" and signal.data_ptr() % 512:
            raise ValueError("signal首地址须512字节对齐，且与业务缓冲独立")
        signal.zero_()

    def rebind(self, native: Mapping[str, torch.Tensor]) -> None:
        """复用已准备工作区，只替换相同描述符的Native根Tensor。

        warmup与图捕获可能提供不同地址的hidden/output和metadata view；
        不能因此重新分配信号区或把调用代际清零。所有根参数先核对shape、
        stride、dtype和device，再整体更新；容量或策略变化须使用另一准备对象。
        图重放本身跳过Python，捕获输入地址与策略的验证由服务/图入口负责。
        """
        roots = {target: native[source] for target, source in self._direct_names}
        roots.update(
            x=native["x_hc"].view(self.tokens, 16384),
            output=native["x_out"].view(self.tokens, 16384),
            hc_base_2d=native["hc_attn_base"].view(1, 24),
            cache_rows=native["kv_cache"].view(-1, 512),
            main_cache=native["cmp_kv"].view(-1, 512),
            attn_sink=native["attn_sink"].view(64, 1),
        )
        for name, value in roots.items():
            previous = self.buffers[name]
            if (
                value.shape != previous.shape
                or value.stride() != previous.stride()
                or value.dtype != previous.dtype
                or value.device != previous.device
            ):
                raise ValueError(f"{name}的Native描述符变化，需要另行准备TND工作区")
        self.native = dict(native)
        self.buffers.update(roots)

    def __call__(self, *, config=None):
        """直接走PyPTO native prepare/enqueue；不依赖Torch dispatcher的参数数量上限。"""
        from .entry import decode_csa_spmd

        options = {} if config is None else {"config": config}
        decode_csa_spmd(**self.buffers, **self.policy.constexprs(), **options)
        return self.native["x_out"]
