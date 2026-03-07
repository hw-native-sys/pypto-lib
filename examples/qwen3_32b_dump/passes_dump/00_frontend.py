# pypto.program: Qwen3SingleLayerDecode
import pypto.language as pl

@pl.program
class Qwen3SingleLayerDecode:
    @pl.function
    def qwen3_decode_layer(self, hidden_states: pl.Tensor[[16, 5120], pl.BFLOAT16], cache_pos: pl.Tensor[[16], pl.INT32], rope_cos: pl.Tensor[[4096, 128], pl.FP32], rope_sin: pl.Tensor[[4096, 128], pl.FP32], k_cache: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache: pl.Tensor[[524288, 128], pl.BFLOAT16], input_rms_weight: pl.Tensor[[1, 5120], pl.FP32], wq: pl.Tensor[[5120, 5120], pl.BFLOAT16], wk: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv: pl.Tensor[[5120, 1024], pl.BFLOAT16], wo: pl.Tensor[[5120, 5120], pl.BFLOAT16], post_rms_weight: pl.Tensor[[1, 5120], pl.FP32], w_gate: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_up: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_down: pl.Tensor[[25600, 5120], pl.BFLOAT16], out: pl.Tensor[[16, 5120], pl.BFLOAT16]) -> pl.Tensor[[16, 5120], pl.BFLOAT16]:
        q_proj: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.create([16, 5120], dtype=pl.BFLOAT16)
        k_proj: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.create([16, 1024], dtype=pl.BFLOAT16)
        v_proj: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.create([16, 1024], dtype=pl.BFLOAT16)
        attn_out: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32)
        with pl.auto_incore():
            sq_sum: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
            sq_sum: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.mul(sq_sum, 0.0)
            for kb in pl.range(0, 20, 1):
                k0: pl.Scalar[pl.INDEX] = kb * 256
                x_chunk: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [16, 256], [0, k0]), target_type=pl.FP32, mode=2)
                sq_sum: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
            inv_rms: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.add(pl.tensor.mul(sq_sum, 0.000195313), 1e-06))
            for b0 in pl.range(0, 16, 4):
                inv_rms_tile: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.view(inv_rms, [4, 1], [b0, 0])
                for ob in pl.parallel(0, 80, 1, chunk=4):
                    q0: pl.Scalar[pl.INDEX] = ob * 64
                    q_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32)
                    q_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(q_acc, 0.0)
                    for kb in pl.range(0, 20, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 256
                        x_chunk_bf16: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.view(hidden_states, [4, 256], [b0, k0])
                        x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(x_chunk_bf16, target_type=pl.FP32, mode=2)
                        gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.view(input_rms_weight, [1, 256], [0, k0])
                        normed: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms_tile), gamma)
                        wq_chunk: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.view(wq, [256, 64], [k0, q0])
                        q_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(q_acc, pl.tensor.matmul(pl.tensor.cast(normed, target_type=pl.BFLOAT16, mode=2), wq_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                    q_proj: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.assemble(q_proj, pl.tensor.cast(q_acc, target_type=pl.BFLOAT16, mode=2), [b0, q0])
                for ob in pl.parallel(0, 32, 1, chunk=8):
                    kv0: pl.Scalar[pl.INDEX] = ob * 32
                    k_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32)
                    v_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32)
                    k_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(k_acc, 0.0)
                    v_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(v_acc, 0.0)
                    for kb in pl.range(0, 20, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 256
                        x_chunk_bf16: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.view(hidden_states, [4, 256], [b0, k0])
                        x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.cast(x_chunk_bf16, target_type=pl.FP32, mode=2)
                        gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.view(input_rms_weight, [1, 256], [0, k0])
                        normed: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms_tile), gamma)
                        normed_bf16: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(normed, target_type=pl.BFLOAT16, mode=2)
                        wk_chunk: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.view(wk, [256, 32], [k0, kv0])
                        wv_chunk: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.view(wv, [256, 32], [k0, kv0])
                        k_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(k_acc, pl.tensor.matmul(normed_bf16, wk_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        v_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(v_acc, pl.tensor.matmul(normed_bf16, wv_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                    k_proj: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.assemble(k_proj, pl.tensor.cast(k_acc, target_type=pl.BFLOAT16, mode=2), [b0, kv0])
                    v_proj: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.assemble(v_proj, pl.tensor.cast(v_acc, target_type=pl.BFLOAT16, mode=2), [b0, kv0])
        for b in pl.parallel(0, 16, 1, chunk=4):
            pos: pl.Scalar[pl.INT32] = pl.tensor.read(cache_pos, [b])
            ctx_len: pl.Scalar[pl.INDEX] = pl.cast(pos, pl.INDEX) + 1
            ctx_blocks: pl.Scalar[pl.INDEX] = (ctx_len + 120 - 1) // 120
            cos_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(rope_cos, [1, 128], [pos, 0])
            sin_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(rope_sin, [1, 128], [pos, 0])
            cos_lo: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(cos_row, [1, 128 // 2], [0, 0])
            cos_hi: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(cos_row, [1, 128 // 2], [0, 128 // 2])
            sin_lo: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(sin_row, [1, 128 // 2], [0, 0])
            sin_hi: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(sin_row, [1, 128 // 2], [0, 128 // 2])
            for kvh in pl.parallel(0, 8, 1, chunk=4):
                kv_col: pl.Scalar[pl.INDEX] = kvh * 128
                k_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(k_proj, [1, 128], [b, kv_col]), target_type=pl.FP32, mode=2)
                k_lo: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(k_row, [1, 128 // 2], [0, 0])
                k_hi: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(k_row, [1, 128 // 2], [0, 128 // 2])
                k_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                k_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot, pl.tensor.sub(pl.tensor.col_expand_mul(k_lo, cos_lo), pl.tensor.col_expand_mul(k_hi, sin_lo)), [0, 0])
                k_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot, pl.tensor.add(pl.tensor.col_expand_mul(k_hi, cos_hi), pl.tensor.col_expand_mul(k_lo, sin_hi)), [0, 128 // 2])
                cache_row: pl.Scalar[pl.INDEX] = b * 8 * 4096 + kvh * 4096 + pl.cast(pos, pl.INDEX)
                k_cache: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(k_cache, pl.tensor.cast(k_rot, target_type=pl.BFLOAT16, mode=2), [cache_row, 0])
                v_cache: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(v_cache, pl.tensor.view(v_proj, [1, 128], [b, kv_col]), [cache_row, 0])
            with pl.auto_incore():
                attn_row: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32)
                attn_row: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.mul(attn_row, 0.0)
                for h in pl.parallel(0, 64, 1, chunk=8):
                    kvh: pl.Scalar[pl.INDEX] = h // 8
                    q_col: pl.Scalar[pl.INDEX] = h * 128
                    q_row: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(pl.tensor.view(q_proj, [1, 128], [b, q_col]), target_type=pl.FP32, mode=2)
                    q_lo: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(q_row, [1, 128 // 2], [0, 0])
                    q_hi: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(q_row, [1, 128 // 2], [0, 128 // 2])
                    q_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                    q_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot, pl.tensor.sub(pl.tensor.col_expand_mul(q_lo, cos_lo), pl.tensor.col_expand_mul(q_hi, sin_lo)), [0, 0])
                    q_rot: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot, pl.tensor.add(pl.tensor.col_expand_mul(q_hi, cos_hi), pl.tensor.col_expand_mul(q_lo, sin_hi)), [0, 128 // 2])
                    q_rot_bf16: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(q_rot, target_type=pl.BFLOAT16, mode=2)
                    oi: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                    li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                    mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
                    oi: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.mul(oi, 0.0)
                    li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(li, 0.0)
                    mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(mi, 0.0)
                    for sb in pl.range(0, ctx_blocks, 1):
                        s0: pl.Scalar[pl.INDEX] = sb * 120
                        valid_len: pl.Scalar[pl.INDEX] = min(120, ctx_len - s0)
                        cache_row0: pl.Scalar[pl.INDEX] = b * 8 * 4096 + kvh * 4096 + s0
                        k_tile: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.view(k_cache, [120, 128], [cache_row0, 0])
                        v_tile: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.view(v_cache, [120, 128], [cache_row0, 0])
                        scores: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.mul(pl.tensor.matmul(q_rot_bf16, k_tile, a_trans=False, b_trans=True, c_matrix_nz=False), 0.0883883)
                        scores_valid: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.view(scores, [1, valid_len], [0, 0])
                        cur_mi: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(pl.tensor.row_max(scores_valid), target_type=pl.FP32, mode=2)
                        exp_scores: pl.Tensor[[1, valid_len], pl.FP32] = pl.tensor.exp(pl.tensor.row_expand_sub(scores_valid, cur_mi))
                        cur_li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.cast(pl.tensor.row_sum(exp_scores), target_type=pl.FP32, mode=2)
                        exp_pad: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.create([1, 120], dtype=pl.FP32)
                        exp_pad: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.mul(exp_pad, 0.0)
                        exp_pad: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.assemble(exp_pad, exp_scores, [0, 0])
                        oi_tmp: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(pl.tensor.cast(exp_pad, target_type=pl.BFLOAT16, mode=2), v_tile, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                        if sb == 0:
                            oi: pl.Tensor[[1, 128], pl.FP32] = oi_tmp
                            li: pl.Tensor[[1, 1], pl.FP32] = cur_li
                            mi: pl.Tensor[[1, 1], pl.FP32] = cur_mi
                        else:
                            mi_new: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.maximum(mi, cur_mi)
                            alpha: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(mi, mi_new))
                            beta: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.exp(pl.tensor.sub(cur_mi, mi_new))
                            li: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.add(pl.tensor.mul(alpha, li), pl.tensor.mul(beta, cur_li))
                            oi: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.add(pl.tensor.row_expand_mul(oi, alpha), pl.tensor.row_expand_mul(oi_tmp, beta))
                            mi: pl.Tensor[[1, 1], pl.FP32] = mi_new
                    ctx: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi, li)
                    attn_row: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row, ctx, [0, q_col])
                attn_out: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_out, attn_row, [b, 0])
        with pl.auto_incore():
            for b0 in pl.range(0, 16, 4):
                resid1_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32)
                for ob in pl.parallel(0, 80, 1, chunk=8):
                    o0: pl.Scalar[pl.INDEX] = ob * 64
                    o_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.create([4, 64], dtype=pl.FP32)
                    o_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(o_acc, 0.0)
                    for kb in pl.range(0, 20, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 256
                        a_chunk: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(pl.tensor.view(attn_out, [4, 256], [b0, k0]), target_type=pl.BFLOAT16, mode=2)
                        w_chunk: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.view(wo, [256, 64], [k0, o0])
                        o_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(o_acc, pl.tensor.matmul(a_chunk, w_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                    resid: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.cast(pl.tensor.view(hidden_states, [4, 64], [b0, o0]), target_type=pl.FP32, mode=2)
                    resid1_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile, pl.tensor.add(o_acc, resid), [0, o0])
                sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
                sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum, 0.0)
                for kb in pl.range(0, 20, 1):
                    k0: pl.Scalar[pl.INDEX] = kb * 256
                    x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(resid1_tile, [4, 256], [0, k0])
                    sq_sum: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum, pl.tensor.row_sum(pl.tensor.mul(x_chunk, x_chunk)))
                inv_rms: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(pl.tensor.add(pl.tensor.mul(sq_sum, 0.000195313), 1e-06))
                post_norm_tile: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.create([4, 5120], dtype=pl.BFLOAT16)
                down_proj_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32)
                down_proj_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.mul(down_proj_tile, 0.0)
                for kb in pl.range(0, 20, 1):
                    k0: pl.Scalar[pl.INDEX] = kb * 256
                    x_chunk: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(resid1_tile, [4, 256], [0, k0])
                    gamma: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.view(post_rms_weight, [1, 256], [0, k0])
                    normed: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(pl.tensor.row_expand_mul(x_chunk, inv_rms), gamma)
                    post_norm_tile: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.assemble(post_norm_tile, pl.tensor.cast(normed, target_type=pl.BFLOAT16, mode=2), [0, k0])
                for ob in pl.range(0, 800, 1):
                    o0: pl.Scalar[pl.INDEX] = ob * 32
                    gate_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32)
                    up_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32)
                    gate_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(gate_acc, 0.0)
                    up_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(up_acc, 0.0)
                    for kb in pl.range(0, 20, 1):
                        k0: pl.Scalar[pl.INDEX] = kb * 256
                        post_chunk: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.view(post_norm_tile, [4, 256], [0, k0])
                        wg: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.view(w_gate, [256, 32], [k0, o0])
                        wu: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.view(w_up, [256, 32], [k0, o0])
                        gate_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(gate_acc, pl.tensor.matmul(post_chunk, wg, a_trans=False, b_trans=False, c_matrix_nz=False))
                        up_acc: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(up_acc, pl.tensor.matmul(post_chunk, wu, a_trans=False, b_trans=False, c_matrix_nz=False))
                    sigmoid: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.recip(pl.tensor.add(pl.tensor.exp(pl.tensor.neg(gate_acc)), 1.0))
                    mlp_chunk: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(pl.tensor.mul(gate_acc, sigmoid), up_acc)
                    mlp_chunk_bf16: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.cast(mlp_chunk, target_type=pl.BFLOAT16, mode=2)
                    for dob in pl.parallel(0, 80, 1, chunk=4):
                        d0: pl.Scalar[pl.INDEX] = dob * 64
                        down_prev: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.view(down_proj_tile, [4, 64], [0, d0])
                        w_down_chunk: pl.Tensor[[32, 64], pl.BFLOAT16] = pl.tensor.view(w_down, [32, 64], [o0, d0])
                        down_next: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(down_prev, pl.tensor.matmul(mlp_chunk_bf16, w_down_chunk, a_trans=False, b_trans=False, c_matrix_nz=False))
                        down_proj_tile: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(down_proj_tile, down_next, [0, d0])
                for ob in pl.parallel(0, 80, 1, chunk=4):
                    o0: pl.Scalar[pl.INDEX] = ob * 64
                    down_acc: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(pl.tensor.view(down_proj_tile, [4, 64], [0, o0]), pl.tensor.view(resid1_tile, [4, 64], [0, o0]))
                    out: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.assemble(out, pl.tensor.cast(down_acc, target_type=pl.BFLOAT16, mode=2), [b0, o0])
        return out