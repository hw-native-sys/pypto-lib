# pypto.program: Qwen3SingleLayerDecode
import pypto.language as pl

valid_len = pl.dynamic("valid_len")

@pl.program
class Qwen3SingleLayerDecode:
    @pl.function
    def qwen3_decode_layer(self, hidden_states: pl.Tensor[[16, 5120], pl.BF16], seq_lens: pl.Tensor[[16], pl.INT32], rope_cos: pl.Tensor[[4096, 128], pl.FP32], rope_sin: pl.Tensor[[4096, 128], pl.FP32], k_cache: pl.Tensor[[524288, 128], pl.BF16], v_cache: pl.Tensor[[524288, 128], pl.BF16], input_rms_weight: pl.Tensor[[1, 5120], pl.FP32], wq: pl.Tensor[[5120, 5120], pl.BF16], wk: pl.Tensor[[5120, 1024], pl.BF16], wv: pl.Tensor[[5120, 1024], pl.BF16], wo: pl.Tensor[[5120, 5120], pl.BF16], post_rms_weight: pl.Tensor[[1, 5120], pl.FP32], w_gate: pl.Tensor[[5120, 25600], pl.BF16], w_up: pl.Tensor[[5120, 25600], pl.BF16], w_down: pl.Tensor[[25600, 5120], pl.BF16], out: pl.Tensor[[16, 5120], pl.BF16]) -> pl.Tensor[[16, 5120], pl.BF16]:
        q_proj: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.create([16, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        k_proj: pl.Tensor[[16, 1024], pl.BF16] = pl.tensor.create([16, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        v_proj: pl.Tensor[[16, 1024], pl.BF16] = pl.tensor.create([16, 1024], dtype=pl.BF16, layout=pl.TensorLayout.ND)
        attn_out: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
        with pl.auto_incore():
            sq_sum: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
            sq_sum_1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.muls(sq_sum, 0.0)
            for kb in pl.range(20):
                k0: pl.Scalar[pl.INDEX] = kb * 256
                x_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.cast(pl.tensor.slice(hidden_states, [16, 256], [0, k0]), target_type=pl.FP32, mode='round')
                sq_sum_2: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(sq_sum_1, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
            inv_rms: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum_2, 0.000195313), 1e-06))
            for b0 in pl.range(0, 16, 4):
                inv_rms_tile: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.slice(inv_rms, [4, 1], [b0, 0])
                for ob in pl.parallel(80, chunk=4):
                    q0: pl.Scalar[pl.INDEX] = ob * 64
                    q_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    q_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(q_acc, 0.0)
                    for kb_1 in pl.range(20):
                        k0_1: pl.Scalar[pl.INDEX] = kb_1 * 256
                        x_chunk_bf16: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.slice(hidden_states, [4, 256], [b0, k0_1])
                        x_chunk_1: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(x_chunk_bf16, target_type=pl.FP32, mode='round')
                        gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight, [1, 256], [0, k0_1])
                        normed: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk_1, inv_rms_tile), gamma)
                        wq_chunk: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(wq, [256, 64], [k0_1, q0])
                        q_acc_2: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(q_acc_1, pl.tensor.matmul(pl.tensor.cast(normed, target_type=pl.BF16, mode='round'), wq_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                    q_proj_1: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(q_proj, pl.tensor.cast(q_acc_2, target_type=pl.BF16, mode='round'), [b0, q0])
                for ob_1 in pl.parallel(32, chunk=8):
                    kv0: pl.Scalar[pl.INDEX] = ob_1 * 32
                    k_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    v_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    k_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(k_acc, 0.0)
                    v_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(v_acc, 0.0)
                    for kb_2 in pl.range(20):
                        k0_2: pl.Scalar[pl.INDEX] = kb_2 * 256
                        x_chunk_bf16_1: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.slice(hidden_states, [4, 256], [b0, k0_2])
                        x_chunk_2: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(x_chunk_bf16_1, target_type=pl.FP32, mode='round')
                        gamma_1: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(input_rms_weight, [1, 256], [0, k0_2])
                        normed_1: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk_2, inv_rms_tile), gamma_1)
                        normed_bf16: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(normed_1, target_type=pl.BF16, mode='round')
                        wk_chunk: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(wk, [256, 32], [k0_2, kv0])
                        wv_chunk: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(wv, [256, 32], [k0_2, kv0])
                        k_acc_2: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(k_acc_1, pl.tensor.matmul(normed_bf16, wk_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        v_acc_2: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(v_acc_1, pl.tensor.matmul(normed_bf16, wv_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                    k_proj_1: pl.Tensor[[16, 1024], pl.BF16] = pl.tensor.assemble(k_proj, pl.tensor.cast(k_acc_2, target_type=pl.BF16, mode='round'), [b0, kv0])
                    v_proj_1: pl.Tensor[[16, 1024], pl.BF16] = pl.tensor.assemble(v_proj, pl.tensor.cast(v_acc_2, target_type=pl.BF16, mode='round'), [b0, kv0])
        for b in pl.parallel(16, chunk=4):
            ctx_len: pl.Scalar[pl.INT32] = pl.tensor.read(seq_lens, [b])
            pos: pl.Scalar[pl.INDEX] = pl.cast(ctx_len, pl.INDEX) - 1
            ctx_blocks: pl.Scalar[pl.INDEX] = (pl.cast(ctx_len, pl.INDEX) + 120 - 1) // 120
            cos_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(rope_cos, [1, 128], [pos, 0])
            sin_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.slice(rope_sin, [1, 128], [pos, 0])
            cos_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos_row, [1, 64], [0, 0])
            cos_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(cos_row, [1, 64], [0, 64])
            sin_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin_row, [1, 64], [0, 0])
            sin_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(sin_row, [1, 64], [0, 64])
            for kvh in pl.parallel(8, chunk=4):
                kv_col: pl.Scalar[pl.INDEX] = kvh * 128
                k_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(k_proj_1, [1, 128], [b, kv_col]), target_type=pl.FP32, mode='round')
                k_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_row, [1, 64], [0, 0])
                k_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(k_row, [1, 64], [0, 64])
                k_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                k_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot, pl.tensor.sub(pl.tensor.col_expand_mul(k_lo, cos_lo), pl.tensor.col_expand_mul(k_hi, sin_lo)), [0, 0])
                k_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_1, pl.tensor.add(pl.tensor.col_expand_mul(k_hi, cos_hi), pl.tensor.col_expand_mul(k_lo, sin_hi)), [0, 64])
                cache_row: pl.Scalar[pl.INDEX] = b * 8 * 4096 + kvh * 4096 + pos
                k_cache_1: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(k_cache, pl.tensor.cast(k_rot_2, target_type=pl.BF16, mode='round'), [cache_row, 0])
                v_cache_1: pl.Tensor[[524288, 128], pl.BF16] = pl.tensor.assemble(v_cache, pl.tensor.slice(v_proj_1, [1, 128], [b, kv_col]), [cache_row, 0])
            with pl.auto_incore():
                attn_row: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                attn_row_1: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.muls(attn_row, 0.0)
                for h in pl.parallel(64, chunk=8):
                    kvh_1: pl.Scalar[pl.INDEX] = h // 8
                    q_col: pl.Scalar[pl.INDEX] = h * 128
                    q_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.slice(q_proj_1, [1, 128], [b, q_col]), target_type=pl.FP32, mode='round')
                    q_lo: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_row, [1, 64], [0, 0])
                    q_hi: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.slice(q_row, [1, 64], [0, 64])
                    q_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    q_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot, pl.tensor.sub(pl.tensor.col_expand_mul(q_lo, cos_lo), pl.tensor.col_expand_mul(q_hi, sin_lo)), [0, 0])
                    q_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_1, pl.tensor.add(pl.tensor.col_expand_mul(q_hi, cos_hi), pl.tensor.col_expand_mul(q_lo, sin_hi)), [0, 64])
                    q_rot_bf16: pl.Tensor[[1, 128], pl.BF16] = pl.tensor.cast(q_rot_2, target_type=pl.BF16, mode='round')
                    oi: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    oi_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.muls(oi, 0.0)
                    li_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.muls(li, 0.0)
                    mi_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.muls(mi, 0.0)
                    for sb in pl.range(ctx_blocks):
                        s0: pl.Scalar[pl.INDEX] = sb * 120
                        valid_len: pl.Scalar[pl.INDEX] = pl.min(120, pl.cast(ctx_len, pl.INDEX) - s0)
                        cache_row0: pl.Scalar[pl.INDEX] = b * 8 * 4096 + kvh_1 * 4096 + s0
                        k_tile: pl.Tensor[[120, 128], pl.BF16, pl.TensorView(valid_shape=[valid_len, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(k_cache_1, [120, 128], [cache_row0, 0], [valid_len, 128])
                        v_tile: pl.Tensor[[120, 128], pl.BF16, pl.TensorView(valid_shape=[valid_len, 128], stride=[], layout=pl.TensorLayout.ND)] = pl.tensor.slice(v_cache_1, [120, 128], [cache_row0, 0], [valid_len, 128])
                        scores: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.muls(pl.tensor.matmul(q_rot_bf16, k_tile, a_trans=False, b_trans=True, c_matrix_nz=False), 0.0883883)
                        scores_valid: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.slice(scores, [1, valid_len], [0, 0])
                        cur_mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(pl.tensor.row_max(scores_valid), target_type=pl.FP32, mode='round')
                        exp_scores: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(scores_valid, cur_mi))
                        cur_li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(pl.tensor.row_sum(exp_scores), target_type=pl.FP32, mode='round')
                        exp_pad: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.create([1, 120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                        exp_pad_1: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.muls(exp_pad, 0.0)
                        exp_pad_2: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.assemble(exp_pad_1, exp_scores, [0, 0])
                        oi_tmp: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(pl.tensor.cast(exp_pad_2, target_type=pl.BF16, mode='round'), v_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        if sb == 0:
                            oi_2: pl.Tensor[[1, 128], pl.FP32] = oi_tmp
                            li_2: pl.Tensor[[1, 1], pl.FP32] = cur_li
                            mi_2: pl.Tensor[[1, 1], pl.FP32] = cur_mi
                        else:
                            mi_new: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi_2, cur_mi)
                            alpha: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi_2, mi_new))
                            beta: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(cur_mi, mi_new))
                            li_3: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha, li_2), pl.tensor.mul(beta, cur_li))
                            oi_3: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(oi_2, alpha), pl.tensor.row_expand_mul(oi_tmp, beta))
                            mi_3: pl.Tensor[[1, 1], pl.FP32] = mi_new
                    ctx: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi_3, li_3)
                    attn_row_2: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row_1, ctx, [0, q_col])
                attn_out_1: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_out, attn_row_2, [b, 0])
        with pl.auto_incore():
            for b0_1 in pl.range(0, 16, 4):
                resid1_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                for ob_2 in pl.parallel(80, chunk=8):
                    o0: pl.Scalar[pl.INDEX] = ob_2 * 64
                    o_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    o_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.muls(o_acc, 0.0)
                    for kb_3 in pl.range(20):
                        k0_3: pl.Scalar[pl.INDEX] = kb_3 * 256
                        a_chunk: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.cast(pl.tensor.slice(attn_out_1, [4, 256], [b0_1, k0_3]), target_type=pl.BF16, mode='round')
                        w_chunk: pl.Tensor[[256, 64], pl.BF16] = pl.tensor.slice(wo, [256, 64], [k0_3, o0])
                        o_acc_2: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(o_acc_1, pl.tensor.matmul(a_chunk, w_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                    resid: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.cast(pl.tensor.slice(hidden_states, [4, 64], [b0_1, o0]), target_type=pl.FP32, mode='round')
                    resid1_tile_1: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile, pl.tensor.add(o_acc_2, resid), [0, o0])
                sq_sum_3: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                sq_sum_4: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.muls(sq_sum_3, 0.0)
                for kb_4 in pl.range(20):
                    k0_4: pl.Scalar[pl.INDEX] = kb_4 * 256
                    x_chunk_3: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(resid1_tile_1, [4, 256], [0, k0_4])
                    sq_sum_5: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum_4, pl.tensor.row_sum(pl.tensor.mul(x_chunk_3, x_chunk_3)))
                inv_rms_1: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.adds(pl.tensor.muls(sq_sum_5, 0.000195313), 1e-06))
                post_norm_tile: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.create([4, 5120], dtype=pl.BF16, layout=pl.TensorLayout.ND)
                down_proj_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                down_proj_tile_1: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.muls(down_proj_tile, 0.0)
                for kb_5 in pl.range(20):
                    k0_5: pl.Scalar[pl.INDEX] = kb_5 * 256
                    x_chunk_4: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.slice(resid1_tile_1, [4, 256], [0, k0_5])
                    gamma_2: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.slice(post_rms_weight, [1, 256], [0, k0_5])
                    normed_2: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk_4, inv_rms_1), gamma_2)
                    post_norm_tile_1: pl.Tensor[[4, 5120], pl.BF16] = pl.tensor.assemble(post_norm_tile, pl.tensor.cast(normed_2, target_type=pl.BF16, mode='round'), [0, k0_5])
                for ob_3 in pl.range(800):
                    o0_1: pl.Scalar[pl.INDEX] = ob_3 * 32
                    gate_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    up_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32, layout=pl.TensorLayout.ND)
                    gate_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(gate_acc, 0.0)
                    up_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.muls(up_acc, 0.0)
                    for kb_6 in pl.range(20):
                        k0_6: pl.Scalar[pl.INDEX] = kb_6 * 256
                        post_chunk: pl.Tensor[[4, 256], pl.BF16] = pl.tensor.slice(post_norm_tile_1, [4, 256], [0, k0_6])
                        wg: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(w_gate, [256, 32], [k0_6, o0_1])
                        wu: pl.Tensor[[256, 32], pl.BF16] = pl.tensor.slice(w_up, [256, 32], [k0_6, o0_1])
                        gate_acc_2: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(gate_acc_1, pl.tensor.matmul(post_chunk, wg, a_trans=False, b_trans=False, c_matrix_nz=False))
                        up_acc_2: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(up_acc_1, pl.tensor.matmul(post_chunk, wu, a_trans=False, b_trans=False, c_matrix_nz=False))
                    sigmoid: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.recip(pl.tensor.adds(pl.tensor.exp(pl.tensor.neg(gate_acc_2)), 1.0))
                    mlp_chunk: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(pl.tensor.mul(gate_acc_2, sigmoid), up_acc_2)
                    mlp_chunk_bf16: pl.Tensor[[4, 32], pl.BF16] = pl.tensor.cast(mlp_chunk, target_type=pl.BF16, mode='round')
                    for dob in pl.parallel(80, chunk=4):
                        d0: pl.Scalar[pl.INDEX] = dob * 64
                        down_prev: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.slice(down_proj_tile_1, [4, 64], [0, d0])
                        w_down_chunk: pl.Tensor[[32, 64], pl.BF16] = pl.tensor.slice(w_down, [32, 64], [o0_1, d0])
                        down_next: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(down_prev, pl.tensor.matmul(mlp_chunk_bf16, w_down_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        down_proj_tile_2: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(down_proj_tile_1, down_next, [0, d0])
                for ob_4 in pl.parallel(80, chunk=4):
                    o0_2: pl.Scalar[pl.INDEX] = ob_4 * 64
                    down_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(pl.tensor.slice(down_proj_tile_2, [4, 64], [0, o0_2]), pl.tensor.slice(resid1_tile_1, [4, 64], [0, o0_2]))
                    out_1: pl.Tensor[[16, 5120], pl.BF16] = pl.tensor.assemble(out, pl.tensor.cast(down_acc, target_type=pl.BF16, mode='round'), [b0_1, o0_2])
        return out_1