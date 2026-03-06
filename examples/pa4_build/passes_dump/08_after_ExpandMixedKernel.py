# pypto.program: PagedAttentionProgram
import pypto.language as pl

@pl.program
class PagedAttentionProgram:
    @pl.function(type=pl.FunctionType.Orchestration)
    def paged_attention(self, query_0: pl.Tensor[[4096, 128], pl.BFLOAT16], key_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16], value_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16], block_table_0: pl.Tensor[[16384], pl.INT32], context_lens_0: pl.Tensor[[64], pl.INT32], out_0: pl.Tensor[[4096, 128], pl.FP32], config_0: pl.Tensor[[7], pl.INT64], size_query_0: pl.Tensor[[1], pl.INT64], size_key_cache_0: pl.Tensor[[1], pl.INT64], size_value_cache_0: pl.Tensor[[1], pl.INT64]) -> pl.Tensor[[4096, 128], pl.FP32]:
        for b_idx_0_out, (out_iter_1_outer_l0,) in pl.range(0, 8, 1, init_values=(out_0,)):
            for q_idx_0_out, (out_iter_1_outer_l1,) in pl.range(0, 2, 1, init_values=(out_iter_1_outer_l0,)):
                out_iter_1_outer_l2_rv: pl.Tensor[[4096, 128], pl.FP32] = self.call_group(paged_attention_incore_0_group, b_idx_0_out, block_table_0, context_lens_0, key_cache_0, out_0, out_iter_1_outer_l0, out_iter_1_outer_l1, q_idx_0_out, query_0, value_cache_0)
                out_iter_1_outer_l1_rv: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_iter_1_outer_l2_rv)
            out_iter_1_outer_l0_rv: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_iter_1_outer_l1_rv)
        return out_iter_1_outer_l0_rv
    @pl.function(type=pl.FunctionType.InCore)
    def paged_attention_incore_0_aic(self, b_idx_0_out: pl.Scalar[pl.INDEX], block_table_0: pl.Tensor[[16384], pl.INT32], context_lens_0: pl.Tensor[[64], pl.INT32], key_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16], out_0: pl.Tensor[[4096, 128], pl.FP32], out_iter_1_outer_l0: pl.Tensor[[4096, 128], pl.FP32], out_iter_1_outer_l1: pl.Tensor[[4096, 128], pl.FP32], q_idx_0_out: pl.Scalar[pl.INDEX], query_0: pl.Tensor[[4096, 128], pl.BFLOAT16], value_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16]) -> pl.Tensor[[4096, 128], pl.FP32]:
        pl.comm.aic_initialize_pipe()
        for b_idx_0_in, (out_iter_1_outer_l2,) in pl.parallel(0, 8, 1, init_values=(out_iter_1_outer_l1,)):
            for q_idx_0_in, (out_iter_1_outer_l3,) in pl.parallel(0, 2, 1, init_values=(out_iter_1_outer_l2,)):
                cur_seq_0: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens_0, [0 + (b_idx_0_out * 8 + b_idx_0_in) * 1])
                bn_this_batch_0: pl.Scalar[pl.INDEX] = (pl.cast(cur_seq_0, pl.INDEX) + 128 - 1) // 128
                cur_offset_0: pl.Scalar[pl.INDEX] = (0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 64 + (0 + (q_idx_0_out * 2 + q_idx_0_in) * 1) * 16
                for bn_0, (out_iter_5,) in pl.range(0, bn_this_batch_0, 1, init_values=(out_iter_1_outer_l3,)):
                    qi_0__h0: pl.Tensor[[8, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                    qi_0__h1: pl.Tensor[[8, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                    qi_0__tmp: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.create(__list__(16, 128), dtype=pl.BFLOAT16)
                    qi_0__mid: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.assemble(qi_0__tmp, qi_0__h0, __list__(0, 0))
                    qi_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.assemble(qi_0__mid, qi_0__h1, __list__(8, 0))
                    cur_block_idx_0: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_0, [(0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 256 + bn_0])
                    valid_len_0: pl.Scalar[pl.INDEX] = min(128, pl.cast(cur_seq_0, pl.INDEX) - bn_0 * 128)
                    kv_block_row_0: pl.Scalar[pl.INDEX] = pl.cast(cur_block_idx_0, pl.INDEX) * 128
                    kj_0__h0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                    kj_0__h1: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                    kj_0__tmp: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.create(__list__(128, 128), dtype=pl.BFLOAT16)
                    kj_0__mid: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.assemble(kj_0__tmp, kj_0__h0, __list__(0, 0))
                    kj_0: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.assemble(kj_0__mid, kj_0__h1, __list__(64, 0))
                    vj_0__h0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                    vj_0__h1: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                    vj_0__tmp: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.create(__list__(128, 128), dtype=pl.BFLOAT16)
                    vj_0__mid: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.assemble(vj_0__tmp, vj_0__h0, __list__(0, 0))
                    vj_0: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.assemble(vj_0__mid, vj_0__h1, __list__(64, 0))
                    sij_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.matmul(qi_0, kj_0, a_trans=False, b_trans=True, c_matrix_nz=False)
                    pl.comm.tpush_to_aiv(sij_0, 0)
                    pl.comm.tpush_to_aiv(sij_0, 1)
                    scale_0: pl.Scalar[pl.FP32] = 1.0
                    pij_f16_1: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                    pij_f16_1__discard: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                    oi_tmp_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.matmul(pij_f16_1, vj_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                    pl.comm.tpush_to_aiv(oi_tmp_0, 0)
                    pl.comm.tpush_to_aiv(oi_tmp_0, 1)
                    if bn_0 == 0:
                        is_first_0: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_first_0: pl.Scalar[pl.INT64] = pl.yield_(0)
                    if bn_0 == bn_this_batch_0 - 1:
                        is_last_0: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_last_0: pl.Scalar[pl.INT64] = pl.yield_(0)
                    if is_first_0:
                        oi_3: pl.Tensor[[16, 128], pl.BFLOAT16] = oi_tmp_0
                        li_update_5, mi_update_5, oi_5, out_13 = pl.yield_(oi_3)
                    else:

                    out_6: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(li_update_5, mi_update_5, oi_5, out_13)
                out_iter_1_outer_l3_rv: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_6)
            out_iter_1_outer_l2_rv: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_iter_1_outer_l3_rv)
        return out_iter_1_outer_l2_rv
    @pl.function(type=pl.FunctionType.InCore)
    def paged_attention_incore_0_aiv(self, b_idx_0_out: pl.Scalar[pl.INDEX], block_table_0: pl.Tensor[[16384], pl.INT32], context_lens_0: pl.Tensor[[64], pl.INT32], key_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16], out_0: pl.Tensor[[4096, 128], pl.FP32], out_iter_1_outer_l0: pl.Tensor[[4096, 128], pl.FP32], out_iter_1_outer_l1: pl.Tensor[[4096, 128], pl.FP32], q_idx_0_out: pl.Scalar[pl.INDEX], query_0: pl.Tensor[[4096, 128], pl.BFLOAT16], value_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16], AIV_IDX: pl.Scalar[pl.INDEX]) -> pl.Tensor[[4096, 128], pl.FP32]:
        pl.comm.aiv_initialize_pipe()
        for b_idx_0_in, (out_iter_1_outer_l2,) in pl.parallel(0, 8, 1, init_values=(out_iter_1_outer_l1,)):
            for q_idx_0_in, (out_iter_1_outer_l3,) in pl.parallel(0, 2, 1, init_values=(out_iter_1_outer_l2,)):
                cur_seq_0: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens_0, [0 + (b_idx_0_out * 8 + b_idx_0_in) * 1])
                bn_this_batch_0: pl.Scalar[pl.INDEX] = (pl.cast(cur_seq_0, pl.INDEX) + 128 - 1) // 128
                cur_offset_0: pl.Scalar[pl.INDEX] = (0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 64 + (0 + (q_idx_0_out * 2 + q_idx_0_in) * 1) * 16
                oi_0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.create([8, 128], dtype=pl.FP32)
                li_update_0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.create([8, 1], dtype=pl.FP32)
                mi_update_0: pl.Tensor[[8, 1], pl.FP32] = pl.tensor.create([8, 1], dtype=pl.FP32)
                for bn_0, (li_update_iter_1, mi_update_iter_1, oi_iter_1, out_iter_5) in pl.range(0, bn_this_batch_0, 1, init_values=(li_update_0, mi_update_0, oi_0, out_iter_1_outer_l3)):
                    qi_0: pl.Tensor[[8, 128], pl.BFLOAT16] = pl.tensor.view(query_0, [8, 128], [cur_offset_0 + AIV_IDX * 8, 0])
                    pl.comm.tpush_to_aic(qi_0, AIV_IDX)
                    cur_block_idx_0: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_0, [(0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 256 + bn_0])
                    valid_len_0: pl.Scalar[pl.INDEX] = min(128, pl.cast(cur_seq_0, pl.INDEX) - bn_0 * 128)
                    kv_block_row_0: pl.Scalar[pl.INDEX] = pl.cast(cur_block_idx_0, pl.INDEX) * 128
                    kj_0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(key_cache_0, [64, 128], [kv_block_row_0 + AIV_IDX * 64, 0])
                    pl.comm.tpush_to_aic(kj_0, AIV_IDX)
                    vj_0: pl.Tensor[[64, 128], pl.BFLOAT16] = pl.tensor.view(value_cache_0, [64, 128], [kv_block_row_0 + AIV_IDX * 64, 0])
                    pl.comm.tpush_to_aic(vj_0, AIV_IDX)
                    sij_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.comm.tpop_from_aic()
                    sij_valid_0: pl.Tensor[[16, valid_len], pl.BFLOAT16] = pl.tensor.view(sij_0, [16, valid_len_0], [0, 0])
                    scale_0: pl.Scalar[pl.FP32] = 1.0
                    scaled_0: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.mul(sij_valid_0, scale_0)
                    mi_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_max(scaled_0)
                    sij_centered_0: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.sub(scaled_0, mi_0)
                    exp_vals_0: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.exp(sij_centered_0)
                    pij_bf16_0: pl.Tensor[[16, valid_len], pl.BFLOAT16] = pl.tensor.cast(exp_vals_0, target_type=pl.BFLOAT16, mode=2)
                    pij_0: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.cast(pij_bf16_0, target_type=pl.FP32, mode=2)
                    li_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pij_0)
                    pij_f16_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.create([16, 128], dtype=pl.BFLOAT16)
                    pij_f16_1: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.assemble(pij_f16_0, pij_bf16_0, [0, 0])
                    pl.comm.tpush_to_aic(pij_f16_1, AIV_IDX)
                    oi_tmp_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.comm.tpop_from_aic()
                    if bn_0 == 0:
                        is_first_0: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_first_0: pl.Scalar[pl.INT64] = pl.yield_(0)
                    if bn_0 == bn_this_batch_0 - 1:
                        is_last_0: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_last_0: pl.Scalar[pl.INT64] = pl.yield_(0)
                    if is_first_0:
                        mi_update_3: pl.Tensor[[16, 1], pl.FP32] = mi_0
                        li_update_3: pl.Tensor[[16, 1], pl.FP32] = li_0
                        oi_3: pl.Tensor[[16, 128], pl.BFLOAT16] = oi_tmp_0
                        if is_last_0:
                            dst_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.div(oi_tmp_0, li_0)
                            out_7: pl.Tensor[[4096, 128], pl.FP32] = pl.tensor.assemble(out_iter_5, dst_0, [cur_offset_0, 0])
                            out_9: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_7)
                        else:
                            out_placeholder_0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.create([8, 128], dtype=pl.FP32)
                            out_8: pl.Tensor[[4096, 128], pl.FP32] = pl.tensor.assemble(out_iter_5, out_placeholder_0, [cur_offset_0 + AIV_IDX * 8, 0])
                            out_9: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_8)
                        li_update_5, mi_update_5, oi_5, out_13 = pl.yield_(li_update_3, mi_update_3, oi_3, out_9)
                    else:
                        mi_prev_nd_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(mi_update_iter_1, [1, 16])
                        mij_nd_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(mi_0, [1, 16])
                        li_prev_nd_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(li_update_iter_1, [1, 16])
                        lij_nd_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(li_0, [1, 16])
                        mi_new_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.maximum(mi_prev_nd_0, mij_nd_0)
                        mi_diff_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.sub(mi_prev_nd_0, mi_new_0)
                        alpha_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.exp(mi_diff_0)
                        mij_diff_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.sub(mij_nd_0, mi_new_0)
                        beta_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.exp(mij_diff_0)
                        li_scaled_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.mul(alpha_0, li_prev_nd_0)
                        lij_scaled_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.mul(beta_0, lij_nd_0)
                        li_new_0: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(li_scaled_0, lij_scaled_0)
                        alpha_dn_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(alpha_0, [16, 1])
                        oi_scaled_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.mul(oi_iter_1, alpha_dn_0)
                        beta_dn_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(beta_0, [16, 1])
                        oi_new_scaled_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.mul(oi_tmp_0, beta_dn_0)
                        oi_updated_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.add(oi_scaled_0, oi_new_scaled_0)
                        mi_update_4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(mi_new_0, [16, 1])
                        li_update_4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(li_new_0, [16, 1])
                        oi_4: pl.Tensor[[16, 128], pl.FP32] = oi_updated_0
                        if is_last_0:
                            li_new_dn_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(li_new_0, [16, 1])
                            dst_1: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.div(oi_updated_0, li_new_dn_0)
                            out_10: pl.Tensor[[4096, 128], pl.FP32] = pl.tensor.assemble(out_iter_5, dst_1, [cur_offset_0, 0])
                            out_12: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_10)
                        else:
                            out_placeholder2_0: pl.Tensor[[8, 128], pl.FP32] = pl.tensor.create([8, 128], dtype=pl.FP32)
                            out_11: pl.Tensor[[4096, 128], pl.FP32] = pl.tensor.assemble(out_iter_5, out_placeholder2_0, [cur_offset_0 + AIV_IDX * 8, 0])
                            out_12: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_11)
                        li_update_5, mi_update_5, oi_5, out_13 = pl.yield_(li_update_4, mi_update_4, oi_4, out_12)
                    li_update_2, mi_update_2, oi_2, out_6 = pl.yield_(li_update_5, mi_update_5, oi_5, out_13)
                out_iter_1_outer_l3_rv: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_6)
            out_iter_1_outer_l2_rv: pl.Tensor[[4096, 128], pl.FP32] = pl.yield_(out_iter_1_outer_l3_rv)
        return out_iter_1_outer_l2_rv
    @pl.function_group(aic="paged_attention_incore_0_aic", aiv="paged_attention_incore_0_aiv")
    class paged_attention_incore_0_group:
        pass
