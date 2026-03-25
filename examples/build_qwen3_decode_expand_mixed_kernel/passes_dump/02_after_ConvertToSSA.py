# pypto.program: Qwen3SingleLayerDecode
import pypto.language as pl

valid_len = pl.dynamic("valid_len")

@pl.program
class Qwen3SingleLayerDecode:
    @pl.function
    def qwen3_decode_layer(self, hidden_states_0: pl.Tensor[[16, 5120], pl.BF16], seq_lens_0: pl.Tensor[[16], pl.INT32], rope_cos_0: pl.Tensor[[4096, 128], pl.FP32], rope_sin_0: pl.Tensor[[4096, 128], pl.FP32], k_cache_0: pl.Tensor[[524288, 128], pl.BF16], v_cache_0: pl.Tensor[[524288, 128], pl.BF16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], wq_0: pl.Tensor[[5120, 5120], pl.BF16], wk_0: pl.Tensor[[5120, 1024], pl.BF16], wv_0: pl.Tensor[[5120, 1024], pl.BF16], wo_0: pl.Tensor[[5120, 5120], pl.BF16], post_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], w_gate_0: pl.Tensor[[5120, 25600], pl.BF16], w_up_0: pl.Tensor[[5120, 25600], pl.BF16], w_down_0: pl.Tensor[[25600, 5120], pl.BF16], out_0: pl.Tensor[[16, 5120], pl.BF16]) -> pl.Tensor[[16, 5120], pl.BF16]:
        q_proj_0: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        k_proj_0: pl.Tensor[[16, 1024], pl.BF16] = pl.tensor.create([16, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        v_proj_0: pl.Tensor[[16, 1024], pl.BF16] = pl.tensor.create([16, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        attn_out_0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.auto_incore():
            sq_sum_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            sq_sum_1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.muls(sq_sum_0, 0.0)
            for kb_0, (sq_sum_iter_2,) in pl.range(20, init_values=(sq_sum_1,)):
                k0_0: pl.Scalar[pl.INDEX] = kb_0 * 256
                x_chunk_0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.cast(pl.tensor.slice(hidden_states_0, [16, 256], [0, k0_0]), target_type=pl.FP32, mode='round')
                sq_sum_4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_2, pl.tensor.row_sum(pl.tensor.mul(x_chunk_0, x_chunk_0)))
                sq_sum_3: pl.Tensor[[16, 1], pl.FP32] = pl.yield_(sq_sum_4)
            inv_rms_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum_3, 0.000195313), 1e-06))
            for b0_0, (k0_iter_1, k_proj_iter_1, kb_iter_1, q_proj_iter_1, v_proj_iter_1, x_chunk_iter_1) in pl.range(0, 16, 4, init_values=(k0_0, k_proj_0, kb_0, q_proj_0, v_proj_0, x_chunk_0)):
                inv_rms_tile_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.slice(inv_rms_0, [4, 1], [b0_0, 0])
                for ob_0, (k0_iter_3, kb_iter_3, q_proj_iter_3, x_chunk_iter_3) in pl.parallel(80, init_values=(k0_iter_1, kb_iter_1, q_proj_iter_1, x_chunk_iter_1), chunk=4):
                    q0_0: pl.Scalar[pl.INDEX] = ob_0 * 64
                    q_acc_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    q_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(q_acc_0, 0.0)
                    for kb_5, (k0_iter_5, q_acc_iter_2, x_chunk_iter_5) in pl.range(20, init_values=(k0_iter_3, q_acc_1, x_chunk_iter_3)):
                        k0_7: pl.Scalar[pl.INDEX] = kb_5 * 256
                        x_chunk_bf16_0: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.slice(hidden_states_0, [4, 256], [b0_0, k0_7])
                        x_chunk_7: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(x_chunk_bf16_0, target_type=pl.FP32, mode='round')
                        gamma_0: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight_0, [1, 256], [0, k0_7])
                        normed_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk_7, inv_rms_tile_0), gamma_0)
                        wq_chunk_0: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(wq_0, [256, 64], [k0_7, q0_0])
                        q_acc_4: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(q_acc_iter_2, pl.tensor.matmul(pl.tensor.cast(normed_0, target_type=pl.BF16, mode='round'), wq_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                        k0_6, q_acc_3, x_chunk_6 = pl.yield_(k0_7, q_acc_4, x_chunk_7)
                    q_proj_5: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(q_proj_iter_3, pl.tensor.cast(q_acc_3, target_type=pl.BF16, mode='round'), [b0_0, q0_0])
                    k0_4, kb_4, q_proj_4, x_chunk_4 = pl.yield_(k0_6, kb_5, q_proj_5, x_chunk_6)
                for ob_1, (gamma_iter_1, k0_iter_8, k_proj_iter_3, kb_iter_6, normed_iter_1, v_proj_iter_3, x_chunk_iter_8, x_chunk_bf16_iter_1) in pl.parallel(32, init_values=(gamma_0, k0_4, k_proj_iter_1, kb_4, normed_0, v_proj_iter_1, x_chunk_4, x_chunk_bf16_0), chunk=8):
                    kv0_0: pl.Scalar[pl.INDEX] = ob_1 * 32
                    k_acc_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    v_acc_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    k_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(k_acc_0, 0.0)
                    v_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(v_acc_0, 0.0)
                    for kb_8, (gamma_iter_3, k0_iter_10, k_acc_iter_2, normed_iter_3, v_acc_iter_2, x_chunk_iter_10, x_chunk_bf16_iter_3) in pl.range(20, init_values=(gamma_iter_1, k0_iter_8, k_acc_1, normed_iter_1, v_acc_1, x_chunk_iter_8, x_chunk_bf16_iter_1)):
                        k0_12: pl.Scalar[pl.INDEX] = kb_8 * 256
                        x_chunk_bf16_5: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.slice(hidden_states_0, [4, 256], [b0_0, k0_12])
                        x_chunk_12: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(x_chunk_bf16_5, target_type=pl.FP32, mode='round')
                        gamma_5: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight_0, [1, 256], [0, k0_12])
                        normed_5: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk_12, inv_rms_tile_0), gamma_5)
                        normed_bf16_0: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(normed_5, target_type=pl.BF16, mode='round')
                        wk_chunk_0: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(wk_0, [256, 32], [k0_12, kv0_0])
                        wv_chunk_0: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(wv_0, [256, 32], [k0_12, kv0_0])
                        k_acc_4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(k_acc_iter_2, pl.tensor.matmul(normed_bf16_0, wk_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                        v_acc_4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(v_acc_iter_2, pl.tensor.matmul(normed_bf16_0, wv_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                        gamma_4, k0_11, k_acc_3, normed_4, v_acc_3, x_chunk_11, x_chunk_bf16_4 = pl.yield_(gamma_5, k0_12, k_acc_4, normed_5, v_acc_4, x_chunk_12, x_chunk_bf16_5)
                    k_proj_5: pl.Tensor[[16, 1024], pl.BF16] = pl.tensor.assemble(k_proj_iter_3, pl.tensor.cast(k_acc_3, target_type=pl.BF16, mode='round'), [b0_0, kv0_0])
                    v_proj_5: pl.Tensor[[16, 1024], pl.BF16] = pl.tensor.assemble(v_proj_iter_3, pl.tensor.cast(v_acc_3, target_type=pl.BF16, mode='round'), [b0_0, kv0_0])
                    gamma_2, k0_9, k_proj_4, kb_7, normed_2, v_proj_4, x_chunk_9, x_chunk_bf16_2 = pl.yield_(gamma_4, k0_11, k_proj_5, kb_8, normed_4, v_proj_5, x_chunk_11, x_chunk_bf16_4)
                k0_2, k_proj_2, kb_2, q_proj_2, v_proj_2, x_chunk_2 = pl.yield_(k0_9, k_proj_4, kb_7, q_proj_4, v_proj_4, x_chunk_9)
        for b_0, (attn_out_iter_1, k_cache_iter_1, v_cache_iter_1) in pl.parallel(16, init_values=(attn_out_0, k_cache_0, v_cache_0), chunk=4):
            ctx_len_0: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens_0, [b_0])
            pos_0: pl.Scalar[pl.INDEX] = pl.cast(ctx_len_0, pl.INDEX) - 1
            ctx_blocks_0: pl.Scalar[pl.INDEX] = (pl.cast(ctx_len_0, pl.INDEX) + 120 - 1) // 120
            cos_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(rope_cos_0, [1, 128], [pos_0, 0])
            sin_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(rope_sin_0, [1, 128], [pos_0, 0])
            cos_lo_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos_row_0, [1, 64], [0, 0])
            cos_hi_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos_row_0, [1, 64], [0, 64])
            sin_lo_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin_row_0, [1, 64], [0, 0])
            sin_hi_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin_row_0, [1, 64], [0, 64])
            for kvh_0, (k_cache_iter_3, v_cache_iter_3) in pl.parallel(8, init_values=(k_cache_iter_1, v_cache_iter_1), chunk=4):
                kv_col_0: pl.Scalar[pl.INDEX] = kvh_0 * 128
                k_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(k_proj_2, [1, 128], [b_0, kv_col_0]), target_type=pl.FP32, mode='round')
                k_lo_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_row_0, [1, 64], [0, 0])
                k_hi_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_row_0, [1, 64], [0, 64])
                k_rot_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                k_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_0, pl.tensor.sub(pl.tensor.col_expand_mul(k_lo_0, cos_lo_0), pl.tensor.col_expand_mul(k_hi_0, sin_lo_0)), [0, 0])
                k_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_1, pl.tensor.add(pl.tensor.col_expand_mul(k_hi_0, cos_hi_0), pl.tensor.col_expand_mul(k_lo_0, sin_hi_0)), [0, 64])
                cache_row_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_0 * 4096 + pos_0
                k_cache_5: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache_iter_3, pl.tensor.cast(k_rot_2, target_type=pl.BF16, mode='round'), [cache_row_0, 0])
                v_cache_5: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(v_cache_iter_3, pl.tensor.slice(v_proj_2, [1, 128], [b_0, kv_col_0]), [cache_row_0, 0])
                k_cache_4, v_cache_4 = pl.yield_(k_cache_5, v_cache_5)
            with pl.auto_incore():
                attn_row_0: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                attn_row_1: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.muls(attn_row_0, 0.0)
                for h_0, (attn_row_iter_2, kvh_iter_1) in pl.parallel(64, init_values=(attn_row_1, kvh_0), chunk=8):
                    kvh_3: pl.Scalar[pl.INDEX] = h_0 // 8
                    q_col_0: pl.Scalar[pl.INDEX] = h_0 * 128
                    q_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(q_proj_2, [1, 128], [b_0, q_col_0]), target_type=pl.FP32, mode='round')
                    q_lo_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_row_0, [1, 64], [0, 0])
                    q_hi_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_row_0, [1, 64], [0, 64])
                    q_rot_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    q_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_0, pl.tensor.sub(pl.tensor.col_expand_mul(q_lo_0, cos_lo_0), pl.tensor.col_expand_mul(q_hi_0, sin_lo_0)), [0, 0])
                    q_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_1, pl.tensor.add(pl.tensor.col_expand_mul(q_hi_0, cos_hi_0), pl.tensor.col_expand_mul(q_lo_0, sin_hi_0)), [0, 64])
                    q_rot_bf16_0: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(q_rot_2, target_type=pl.BF16, mode='round')
                    oi_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    oi_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(oi_0, 0.0)
                    li_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.muls(li_0, 0.0)
                    mi_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.muls(mi_0, 0.0)
                    for sb_0, (li_iter_2, mi_iter_2, oi_iter_2) in pl.range(ctx_blocks_0, init_values=(li_1, mi_1, oi_1)):
                        s0_0: pl.Scalar[pl.INDEX] = sb_0 * 120
                        valid_len_0: pl.Scalar[pl.INDEX] = pl.min(120, pl.cast(ctx_len_0, pl.INDEX) - s0_0)
                        cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_3 * 4096 + s0_0
                        k_tile_0: pl.Tensor[[120, 128], pl.BF16, pl.TensorView(valid_shape=[valid_len, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(k_cache_4, [120, 128], [cache_row0_0, 0], [valid_len_0, 128])
                        v_tile_0: pl.Tensor[[120, 128], pl.BF16, pl.TensorView(valid_shape=[valid_len, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(v_cache_4, [120, 128], [cache_row0_0, 0], [valid_len_0, 128])
                        scores_0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.muls(pl.tensor.matmul(q_rot_bf16_0, k_tile_0, a_trans=False, b_trans=True, c_matrix_nz=False), 0.0883883)
                        scores_valid_0: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.slice(scores_0, [1, valid_len_0], [0, 0])
                        cur_mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(pl.tensor.row_max(scores_valid_0), target_type=pl.FP32, mode='round')
                        exp_scores_0: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(scores_valid_0, cur_mi_0))
                        cur_li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(pl.tensor.row_sum(exp_scores_0), target_type=pl.FP32, mode='round')
                        exp_pad_0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.create([1, 120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        exp_pad_1: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.muls(exp_pad_0, 0.0)
                        exp_pad_2: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.assemble(exp_pad_1, exp_scores_0, [0, 0])
                        oi_tmp_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(pl.tensor.cast(exp_pad_2, target_type=pl.BF16, mode='round'), v_tile_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        if sb_0 == 0:
                            oi_4: pl.Tensor[[1, 128], pl.FP32] = oi_tmp_0
                            li_4: pl.Tensor[[1, 1], pl.FP32] = cur_li_0
                            mi_4: pl.Tensor[[1, 1], pl.FP32] = cur_mi_0
                            li_6, mi_6, oi_6 = pl.yield_(li_4, mi_4, oi_4)
                        else:
                            mi_new_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi_iter_2, cur_mi_0)
                            alpha_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_iter_2, mi_new_0))
                            beta_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(cur_mi_0, mi_new_0))
                            li_5: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha_0, li_iter_2), pl.tensor.mul(beta_0, cur_li_0))
                            oi_5: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(oi_iter_2, alpha_0), pl.tensor.row_expand_mul(oi_tmp_0, beta_0))
                            mi_5: pl.Tensor[[1, 1], pl.FP32] = mi_new_0
                            li_6, mi_6, oi_6 = pl.yield_(li_5, mi_5, oi_5)
                        li_3, mi_3, oi_3 = pl.yield_(li_6, mi_6, oi_6)
                    ctx_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi_3, li_3)
                    attn_row_4: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row_iter_2, ctx_0, [0, q_col_0])
                    attn_row_3, kvh_2 = pl.yield_(attn_row_4, kvh_3)
                attn_out_3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_out_iter_1, attn_row_3, [b_0, 0])
            attn_out_2, k_cache_2, v_cache_2 = pl.yield_(attn_out_3, k_cache_4, v_cache_4)
        with pl.auto_incore():
            for b0_1, (gamma_iter_6, inv_rms_iter_1, k0_iter_13, kb_iter_9, normed_iter_6, ob_iter_2, out_iter_1, sq_sum_iter_5, x_chunk_iter_13) in pl.range(0, 16, 4, init_values=(gamma_2, inv_rms_0, k0_2, kb_2, normed_2, ob_1, out_0, sq_sum_3, x_chunk_2)):
                resid1_tile_0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for ob_4, (k0_iter_15, kb_iter_11, resid1_tile_iter_1) in pl.parallel(80, init_values=(k0_iter_13, kb_iter_9, resid1_tile_0), chunk=8):
                    o0_0: pl.Scalar[pl.INDEX] = ob_4 * 64
                    o_acc_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    o_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(o_acc_0, 0.0)
                    for kb_13, (k0_iter_17, o_acc_iter_2) in pl.range(20, init_values=(k0_iter_15, o_acc_1)):
                        k0_19: pl.Scalar[pl.INDEX] = kb_13 * 256
                        a_chunk_0: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(pl.tensor.slice(attn_out_2, [4, 256], [b0_1, k0_19]), target_type=pl.BF16, mode='round')
                        w_chunk_0: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(wo_0, [256, 64], [k0_19, o0_0])
                        o_acc_4: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(o_acc_iter_2, pl.tensor.matmul(a_chunk_0, w_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                        k0_18, o_acc_3 = pl.yield_(k0_19, o_acc_4)
                    resid_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(hidden_states_0, [4, 64], [b0_1, o0_0]), target_type=pl.FP32, mode='round')
                    resid1_tile_3: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile_iter_1, pl.tensor.add(o_acc_3, resid_0), [0, o0_0])
                    k0_16, kb_12, resid1_tile_2 = pl.yield_(k0_18, kb_13, resid1_tile_3)
                sq_sum_7: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                sq_sum_8: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.muls(sq_sum_7, 0.0)
                for kb_14, (k0_iter_20, sq_sum_iter_9, x_chunk_iter_15) in pl.range(20, init_values=(k0_16, sq_sum_8, x_chunk_iter_13)):
                    k0_22: pl.Scalar[pl.INDEX] = kb_14 * 256
                    x_chunk_17: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(resid1_tile_2, [4, 256], [0, k0_22])
                    sq_sum_11: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_9, pl.tensor.row_sum(pl.tensor.mul(x_chunk_17, x_chunk_17)))
                    k0_21, sq_sum_10, x_chunk_16 = pl.yield_(k0_22, sq_sum_11, x_chunk_17)
                inv_rms_3: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum_10, 0.000195313), 1e-06))
                post_norm_tile_0: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.create([4, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                down_proj_tile_0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                down_proj_tile_1: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.muls(down_proj_tile_0, 0.0)
                for kb_15, (gamma_iter_8, k0_iter_23, normed_iter_8, post_norm_tile_iter_1, x_chunk_iter_18) in pl.range(20, init_values=(gamma_iter_6, k0_21, normed_iter_6, post_norm_tile_0, x_chunk_16)):
                    k0_25: pl.Scalar[pl.INDEX] = kb_15 * 256
                    x_chunk_20: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(resid1_tile_2, [4, 256], [0, k0_25])
                    gamma_10: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight_0, [1, 256], [0, k0_25])
                    normed_10: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk_20, inv_rms_3), gamma_10)
                    post_norm_tile_3: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.assemble(post_norm_tile_iter_1, pl.tensor.cast(normed_10, target_type=pl.BF16, mode='round'), [0, k0_25])
                    gamma_9, k0_24, normed_9, post_norm_tile_2, x_chunk_19 = pl.yield_(gamma_10, k0_25, normed_10, post_norm_tile_3, x_chunk_20)
                for ob_5, (down_proj_tile_iter_2, k0_iter_26, kb_iter_16, o0_iter_1) in pl.range(800, init_values=(down_proj_tile_1, k0_24, kb_15, o0_0)):
                    o0_3: pl.Scalar[pl.INDEX] = ob_5 * 32
                    gate_acc_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    up_acc_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gate_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(gate_acc_0, 0.0)
                    up_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(up_acc_0, 0.0)
                    for kb_18, (gate_acc_iter_2, k0_iter_28, up_acc_iter_2) in pl.range(20, init_values=(gate_acc_1, k0_iter_26, up_acc_1)):
                        k0_30: pl.Scalar[pl.INDEX] = kb_18 * 256
                        post_chunk_0: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.slice(post_norm_tile_2, [4, 256], [0, k0_30])
                        wg_0: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(w_gate_0, [256, 32], [k0_30, o0_3])
                        wu_0: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(w_up_0, [256, 32], [k0_30, o0_3])
                        gate_acc_4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(gate_acc_iter_2, pl.tensor.matmul(post_chunk_0, wg_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                        up_acc_4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(up_acc_iter_2, pl.tensor.matmul(post_chunk_0, wu_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                        gate_acc_3, k0_29, up_acc_3 = pl.yield_(gate_acc_4, k0_30, up_acc_4)
                    sigmoid_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(gate_acc_3)), 1.0))
                    mlp_chunk_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(pl.tensor.mul(gate_acc_3, sigmoid_0), up_acc_3)
                    mlp_chunk_bf16_0: pl.Tensor[[4, 32], pl.BF16] = pl.tensor.cast(mlp_chunk_0, target_type=pl.BF16, mode='round')
                    for dob_0, (down_proj_tile_iter_4,) in pl.parallel(80, init_values=(down_proj_tile_iter_2,), chunk=4):
                        d0_0: pl.Scalar[pl.INDEX] = dob_0 * 64
                        down_prev_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.slice(down_proj_tile_iter_4, [4, 64], [0, d0_0])
                        w_down_chunk_0: pl.Tensor[[32, 64], pl.BF16] = pl.tensor.slice(w_down_0, [32, 64], [o0_3, d0_0])
                        down_next_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(down_prev_0, pl.tensor.matmul(mlp_chunk_bf16_0, w_down_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False))
                        down_proj_tile_6: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(down_proj_tile_iter_4, down_next_0, [0, d0_0])
                        down_proj_tile_5: pl.Tensor[[4, 5120], pl.FP32] = pl.yield_(down_proj_tile_6)
                    down_proj_tile_3, k0_27, kb_17, o0_2 = pl.yield_(down_proj_tile_5, k0_29, kb_18, o0_3)
                for ob_6, (o0_iter_4, out_iter_3) in pl.parallel(80, init_values=(o0_2, out_iter_1), chunk=4):
                    o0_6: pl.Scalar[pl.INDEX] = ob_6 * 64
                    down_acc_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(down_proj_tile_3, [4, 64], [0, o0_6]), pl.tensor.slice(resid1_tile_2, [4, 64], [0, o0_6]))
                    out_5: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(out_iter_3, pl.tensor.cast(down_acc_0, target_type=pl.BF16, mode='round'), [b0_1, o0_6])
                    o0_5, out_4 = pl.yield_(o0_6, out_5)
                gamma_7, inv_rms_2, k0_14, kb_10, normed_7, ob_3, out_2, sq_sum_6, x_chunk_14 = pl.yield_(gamma_9, inv_rms_3, k0_27, kb_17, normed_9, ob_6, out_4, sq_sum_10, x_chunk_19)
        return out_2