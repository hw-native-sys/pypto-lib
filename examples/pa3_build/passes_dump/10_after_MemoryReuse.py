# pypto.program: PagedAttentionProgram
import pypto.language as pl

@pl.program
class PagedAttentionProgram:
    @pl.function
    def paged_attention(self, query_0: pl.Tensor[[4096, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 1048576, 0)], key_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 536870912, 1)], value_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 536870912, 2)], block_table_0: pl.Tensor[[16384], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 65536, 3)], context_lens_0: pl.Tensor[[64], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 4)], out_0: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 5)], config_0: pl.Tensor[[7], pl.INT64, pl.MemRef(pl.MemorySpace.DDR, -1, 56, 6)], size_query_0: pl.Tensor[[1], pl.INT64, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 7)], size_key_cache_0: pl.Tensor[[1], pl.INT64, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 8)], size_value_cache_0: pl.Tensor[[1], pl.INT64, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 9)]) -> pl.Tensor[[4096, 128], pl.FP32]:
        for b_idx_0, (out_iter_1,) in pl.parallel(0, 64, 1, init_values=(out_0,), chunk=8):
            for q_idx_0, (out_iter_3,) in pl.parallel(0, 4, 1, init_values=(out_iter_1,), chunk=2):
                cur_seq_0: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens_0, [b_idx_0])
                bn_this_batch_0: pl.Scalar[pl.INDEX] = (pl.cast(cur_seq_0, pl.INDEX) + 128 - 1) // 128
                cur_offset_0: pl.Scalar[pl.INDEX] = b_idx_0 * 64 + q_idx_0 * 16
                oi_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 10)] = pl.tensor.create([16, 128], dtype=pl.FP32)
                li_update_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 11)] = pl.tensor.create([16, 1], dtype=pl.FP32)
                mi_update_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 12)] = pl.tensor.create([16, 1], dtype=pl.FP32)
                for bn_0, (li_update_iter_1, mi_update_iter_1, oi_iter_1, out_iter_5) in pl.range(0, bn_this_batch_0, 1, init_values=(li_update_0, mi_update_0, oi_0, out_iter_3)):
                    qi_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 13)] = pl.tensor.view(query_0, [16, 128], [cur_offset_0, 0])
                    cur_block_idx_0: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_0, [b_idx_0 * 256 + bn_0])
                    valid_len_0: pl.Scalar[pl.INDEX] = min(128, pl.cast(cur_seq_0, pl.INDEX) - bn_0 * 128)
                    kv_block_row_0: pl.Scalar[pl.INDEX] = pl.cast(cur_block_idx_0, pl.INDEX) * 128
                    kj_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 14)] = pl.tensor.view(key_cache_0, [128, 128], [kv_block_row_0, 0])
                    vj_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 15)] = pl.tensor.view(value_cache_0, [128, 128], [kv_block_row_0, 0])
                    sij_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 16)] = pl.tensor.matmul(qi_0, kj_0, a_trans=False, b_trans=True, c_matrix_nz=False)
                    sij_valid_0: pl.Tensor[[16, valid_len], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 17)] = pl.tensor.view(sij_0, [16, valid_len_0], [0, 0])
                    scale_0: pl.Scalar[pl.FP32] = 1.0
                    scaled_0: pl.Tensor[[16, valid_len], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 18)] = pl.tensor.mul(sij_valid_0, scale_0)
                    mi_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 19)] = pl.tensor.row_max(scaled_0)
                    sij_centered_0: pl.Tensor[[16, valid_len], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 20)] = pl.tensor.sub(scaled_0, mi_0)
                    exp_vals_0: pl.Tensor[[16, valid_len], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 21)] = pl.tensor.exp(sij_centered_0)
                    pij_bf16_0: pl.Tensor[[16, valid_len], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 22)] = pl.tensor.cast(exp_vals_0, target_type=pl.BFLOAT16, mode=2)
                    pij_0: pl.Tensor[[16, valid_len], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 0, 23)] = pl.tensor.cast(pij_bf16_0, target_type=pl.FP32, mode=2)
                    li_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 24)] = pl.tensor.row_sum(pij_0)
                    pij_f16_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 25)] = pl.tensor.create([16, 128], dtype=pl.BFLOAT16)
                    pij_f16_1: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 26)] = pl.tensor.assemble(pij_f16_0, pij_bf16_0, [0, 0])
                    oi_tmp_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 27)] = pl.tensor.matmul(pij_f16_1, vj_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                    if bn_0 == 0:
                        is_first_0: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_first_0: pl.Scalar[pl.INT64] = pl.yield_(0)
                    if bn_0 == bn_this_batch_0 - 1:
                        is_last_0: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_last_0: pl.Scalar[pl.INT64] = pl.yield_(0)
                    if is_first_0:
                        mi_update_3: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 28)] = mi_0
                        li_update_3: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 29)] = li_0
                        oi_3: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 30)] = oi_tmp_0
                        if is_last_0:
                            dst_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 31)] = pl.tensor.div(oi_tmp_0, li_0)
                            out_7: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 32)] = pl.tensor.assemble(out_iter_5, dst_0, [cur_offset_0, 0])
                            out_9: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 35)] = pl.yield_(out_7)
                        else:
                            out_placeholder_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 33)] = pl.tensor.create([16, 128], dtype=pl.FP32)
                            out_8: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 34)] = pl.tensor.assemble(out_iter_5, out_placeholder_0, [cur_offset_0, 0])
                            out_9: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 35)] = pl.yield_(out_8)
                        li_update_5, mi_update_5, oi_5, out_13 = pl.yield_(li_update_3, mi_update_3, oi_3, out_9)
                    else:
                        mi_prev_nd_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 36)] = pl.tensor.reshape(mi_update_iter_1, [1, 16])
                        mij_nd_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 37)] = pl.tensor.reshape(mi_0, [1, 16])
                        li_prev_nd_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 38)] = pl.tensor.reshape(li_update_iter_1, [1, 16])
                        lij_nd_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 39)] = pl.tensor.reshape(li_0, [1, 16])
                        mi_new_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 40)] = pl.tensor.maximum(mi_prev_nd_0, mij_nd_0)
                        mi_diff_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 41)] = pl.tensor.sub(mi_prev_nd_0, mi_new_0)
                        alpha_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 42)] = pl.tensor.exp(mi_diff_0)
                        mij_diff_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 43)] = pl.tensor.sub(mij_nd_0, mi_new_0)
                        beta_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 44)] = pl.tensor.exp(mij_diff_0)
                        li_scaled_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 45)] = pl.tensor.mul(alpha_0, li_prev_nd_0)
                        lij_scaled_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 46)] = pl.tensor.mul(beta_0, lij_nd_0)
                        li_new_0: pl.Tensor[[1, 16], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 47)] = pl.tensor.add(li_scaled_0, lij_scaled_0)
                        alpha_dn_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 48)] = pl.tensor.reshape(alpha_0, [16, 1])
                        oi_scaled_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 49)] = pl.tensor.mul(oi_iter_1, alpha_dn_0)
                        beta_dn_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 50)] = pl.tensor.reshape(beta_0, [16, 1])
                        oi_new_scaled_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 51)] = pl.tensor.mul(oi_tmp_0, beta_dn_0)
                        oi_updated_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 52)] = pl.tensor.add(oi_scaled_0, oi_new_scaled_0)
                        mi_update_4: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 53)] = pl.tensor.reshape(mi_new_0, [16, 1])
                        li_update_4: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 54)] = pl.tensor.reshape(li_new_0, [16, 1])
                        oi_4: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 55)] = oi_updated_0
                        if is_last_0:
                            li_new_dn_0: pl.Tensor[[16, 1], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 64, 56)] = pl.tensor.reshape(li_new_0, [16, 1])
                            dst_1: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 57)] = pl.tensor.div(oi_updated_0, li_new_dn_0)
                            out_10: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 58)] = pl.tensor.assemble(out_iter_5, dst_1, [cur_offset_0, 0])
                            out_12: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 61)] = pl.yield_(out_10)
                        else:
                            out_placeholder2_0: pl.Tensor[[16, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 59)] = pl.tensor.create([16, 128], dtype=pl.FP32)
                            out_11: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 60)] = pl.tensor.assemble(out_iter_5, out_placeholder2_0, [cur_offset_0, 0])
                            out_12: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 61)] = pl.yield_(out_11)
                        li_update_5, mi_update_5, oi_5, out_13 = pl.yield_(li_update_4, mi_update_4, oi_4, out_12)
                    li_update_2, mi_update_2, oi_2, out_6 = pl.yield_(li_update_5, mi_update_5, oi_5, out_13)
                out_4: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 70)] = pl.yield_(out_6)
            out_2: pl.Tensor[[4096, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 2097152, 71)] = pl.yield_(out_4)
        return out_2