# pypto.program: PagedAttentionProgram
import pypto.language as pl

@pl.program
class PagedAttentionProgram:
    @pl.function
    def paged_attention(self, query: pl.Tensor[[4096, 128], pl.BFLOAT16], key_cache: pl.Tensor[[2097152, 128], pl.BFLOAT16], value_cache: pl.Tensor[[2097152, 128], pl.BFLOAT16], block_table: pl.Tensor[[16384], pl.INT32], context_lens: pl.Tensor[[64], pl.INT32], out: pl.Tensor[[4096, 128], pl.FP32], config: pl.Tensor[[7], pl.INT64], size_query: pl.Tensor[[1], pl.INT64], size_key_cache: pl.Tensor[[1], pl.INT64], size_value_cache: pl.Tensor[[1], pl.INT64]) -> pl.Tensor[[4096, 128], pl.FP32]:
        with pl.auto_incore():
            for b_idx in pl.parallel(0, 64, 1, chunk=8):
                for q_idx in pl.parallel(0, 4, 1, chunk=2):
                    cur_seq: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens, [b_idx])
                    bn_this_batch: pl.Scalar[pl.INDEX] = (pl.cast(cur_seq, pl.INDEX) + 128 - 1) // 128
                    cur_offset: pl.Scalar[pl.INDEX] = b_idx * 64 + q_idx * 16
                    oi: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                    li_update: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                    mi_update: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                    for bn in pl.range(0, bn_this_batch, 1):
                        qi: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.view(query, [16, 128], [cur_offset, 0])
                        cur_block_idx: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [b_idx * 256 + bn])
                        valid_len: pl.Scalar[pl.INDEX] = min(128, pl.cast(cur_seq, pl.INDEX) - bn * 128)
                        kv_block_row: pl.Scalar[pl.INDEX] = pl.cast(cur_block_idx, pl.INDEX) * 128
                        kj: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(key_cache, [128, 128], [kv_block_row, 0])
                        vj: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(value_cache, [128, 128], [kv_block_row, 0])
                        sij: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.matmul(qi, kj, a_trans=False, b_trans=True, c_matrix_nz=False)
                        sij_valid: pl.Tensor[[16, valid_len], pl.BFLOAT16] = pl.tensor.view(sij, [16, valid_len], [0, 0])
                        scale: pl.Scalar[pl.FP32] = 1.0
                        scaled: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.mul(sij_valid, scale)
                        mi: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_max(scaled)
                        sij_centered: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.sub(scaled, mi)
                        exp_vals: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.exp(sij_centered)
                        pij_bf16: pl.Tensor[[16, valid_len], pl.BFLOAT16] = pl.tensor.cast(exp_vals, target_type=pl.BFLOAT16, mode=2)
                        pij: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.cast(pij_bf16, target_type=pl.FP32, mode=2)
                        li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(pij)
                        pij_f16: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.create([16, 128], dtype=pl.BFLOAT16)
                        pij_f16: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.assemble(pij_f16, pij_bf16, [0, 0])
                        oi_tmp: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.matmul(pij_f16, vj, a_trans=False, b_trans=False, c_matrix_nz=False)
                        if bn == 0:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(1)
                        else:
                            is_first: pl.Scalar[pl.INT64] = pl.yield_(0)
                        if bn == bn_this_batch - 1:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(1)
                        else:
                            is_last: pl.Scalar[pl.INT64] = pl.yield_(0)
                        if is_first:
                            mi_update: pl.Tensor[[16, 1], pl.FP32] = mi
                            li_update: pl.Tensor[[16, 1], pl.FP32] = li
                            oi: pl.Tensor[[16, 128], pl.BFLOAT16] = oi_tmp
                            if is_last:
                                dst: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.div(oi_tmp, li)
                                out: pl.Tensor[[4096, 128], pl.FP32] = pl.tensor.assemble(out, dst, [cur_offset, 0])
                            else:
                                out_placeholder: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                                out: pl.Tensor[[4096, 128], pl.FP32] = pl.tensor.assemble(out, out_placeholder, [cur_offset, 0])
                        else:
                            mi_prev_nd: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(mi_update, [1, 16])
                            mij_nd: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(mi, [1, 16])
                            li_prev_nd: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(li_update, [1, 16])
                            lij_nd: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.reshape(li, [1, 16])
                            mi_new: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.maximum(mi_prev_nd, mij_nd)
                            mi_diff: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.sub(mi_prev_nd, mi_new)
                            alpha: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.exp(mi_diff)
                            mij_diff: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.sub(mij_nd, mi_new)
                            beta: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.exp(mij_diff)
                            li_scaled: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.mul(alpha, li_prev_nd)
                            lij_scaled: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.mul(beta, lij_nd)
                            li_new: pl.Tensor[[1, 16], pl.FP32] = pl.tensor.add(li_scaled, lij_scaled)
                            alpha_dn: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(alpha, [16, 1])
                            oi_scaled: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.mul(oi, alpha_dn)
                            beta_dn: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(beta, [16, 1])
                            oi_new_scaled: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.mul(oi_tmp, beta_dn)
                            oi_updated: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.add(oi_scaled, oi_new_scaled)
                            mi_update: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(mi_new, [16, 1])
                            li_update: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(li_new, [16, 1])
                            oi: pl.Tensor[[16, 128], pl.FP32] = oi_updated
                            if is_last:
                                li_new_dn: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.reshape(li_new, [16, 1])
                                dst: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.div(oi_updated, li_new_dn)
                                out: pl.Tensor[[4096, 128], pl.FP32] = pl.tensor.assemble(out, dst, [cur_offset, 0])
                            else:
                                out_placeholder2: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                                out: pl.Tensor[[4096, 128], pl.FP32] = pl.tensor.assemble(out, out_placeholder2, [cur_offset, 0])
        return out