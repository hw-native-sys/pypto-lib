# pypto.program: Qwen3SingleLayerDecode
import pypto.language as pl

@pl.program
class Qwen3SingleLayerDecode:
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_5(self, b0_1: pl.Scalar[pl.INDEX], down_proj_tile_3: pl.Tensor[[4, 5120], pl.FP32], o0_2: pl.Scalar[pl.INDEX], o0_iter_4_outer_l0: pl.Scalar[pl.INDEX], ob_6_out: pl.Scalar[pl.INDEX], out_0: pl.Tensor[[16, 5120], pl.BFLOAT16], out_iter_1: pl.Tensor[[16, 5120], pl.BFLOAT16], out_iter_3_outer_l0: pl.Tensor[[16, 5120], pl.BFLOAT16], resid1_tile_iter_1_outer_l0_rv: pl.Tensor[[4, 5120], pl.FP32]) -> tuple[pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.BFLOAT16]]:
        for ob_6_in, (o0_iter_4_outer_l1, out_iter_3_outer_l1) in pl.parallel(0, 4, 1, init_values=(o0_iter_4_outer_l0, out_iter_3_outer_l0)):
            o0_6: pl.Scalar[pl.INDEX] = (0 + (ob_6_out * 4 + ob_6_in) * 1) * 64
            _t58: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.view(down_proj_tile_3, [4, 64], [0, o0_6])
            _t59: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 64], [0, o0_6])
            down_acc_0: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.add(_t58, _t59)
            _t60: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.cast(down_acc_0, target_type=pl.BFLOAT16, mode=2)
            out_5: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.assemble(out_iter_3_outer_l1, _t60, [b0_1, o0_6])
            o0_iter_4_outer_l1_rv, out_iter_3_outer_l1_rv = pl.yield_(o0_6, out_5)
        return o0_iter_4_outer_l1_rv, out_iter_3_outer_l1_rv
    @pl.function(type=pl.FunctionType.Orchestration)
    def qwen3_decode_layer(self, hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16], cache_pos_0: pl.Tensor[[16], pl.INT32], rope_cos_0: pl.Tensor[[4096, 128], pl.FP32], rope_sin_0: pl.Tensor[[4096, 128], pl.FP32], k_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], v_cache_0: pl.Tensor[[524288, 128], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], post_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], w_gate_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_up_0: pl.Tensor[[5120, 25600], pl.BFLOAT16], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16], out_0: pl.Tensor[[16, 5120], pl.BFLOAT16]) -> pl.Tensor[[16, 5120], pl.BFLOAT16]:
        q_proj_0: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.create([16, 5120], dtype=pl.BFLOAT16)
        k_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.create([16, 1024], dtype=pl.BFLOAT16)
        v_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.create([16, 1024], dtype=pl.BFLOAT16)
        attn_out_0: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.create([16, 5120], dtype=pl.FP32)
        sq_sum_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
        sq_sum_1: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.mul(sq_sum_0, 0.0)
        for kb_0, (sq_sum_iter_2,) in pl.range(0, 20, 1, init_values=(sq_sum_1,)):
            k0_0: pl.Scalar[pl.INDEX] = kb_0 * 256
            _t0: pl.Tensor[[16, 256], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [16, 256], [0, k0_0])
            x_chunk_0: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.cast(_t0, target_type=pl.FP32, mode=2)
            _t1: pl.Tensor[[16, 256], pl.FP32] = pl.tensor.mul(x_chunk_0, x_chunk_0)
            _t2: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.row_sum(_t1)
            sq_sum_4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_2, _t2)
            sq_sum_3: pl.Tensor[[16, 1], pl.FP32] = pl.yield_(sq_sum_4)
        _t3: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.mul(sq_sum_3, 0.000195313)
        _t4: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.add(_t3, 1e-06)
        inv_rms_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.rsqrt(_t4)
        for b0_0, (k0_iter_1, k_proj_iter_1, kb_iter_1, q_proj_iter_1, v_proj_iter_1, x_chunk_iter_1) in pl.range(0, 16, 4, init_values=(k0_0, k_proj_0, kb_0, q_proj_0, v_proj_0, x_chunk_0)):
            inv_rms_tile_0: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.view(inv_rms_0, [4, 1], [b0_0, 0])
            for ob_0_out, (k0_iter_3_outer_l0, kb_iter_3_outer_l0, q_proj_iter_3_outer_l0, x_chunk_iter_3_outer_l0) in pl.range(0, 20, 1, init_values=(k0_iter_1, kb_iter_1, q_proj_iter_1, x_chunk_iter_1)):
                ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.BFLOAT16], pl.Tensor[[16, 256], pl.FP32]]) = self.call_group(qwen3_decode_layer_incore_0_group, b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k0_0, k0_iter_1, k0_iter_3_outer_l0, kb_0, kb_iter_1, kb_iter_3_outer_l0, ob_0_out, q_proj_0, q_proj_iter_1, q_proj_iter_3_outer_l0, wq_0, x_chunk_0, x_chunk_iter_1, x_chunk_iter_3_outer_l0)
                k0_iter_3_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
                kb_iter_3_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                q_proj_iter_3_outer_l1_rv: pl.Tensor[[16, 5120], pl.BFLOAT16] = ret[2]
                x_chunk_iter_3_outer_l1_rv: pl.Tensor[[16, 256], pl.FP32] = ret[3]
                k0_iter_3_outer_l0_rv, kb_iter_3_outer_l0_rv, q_proj_iter_3_outer_l0_rv, x_chunk_iter_3_outer_l0_rv = pl.yield_(k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, q_proj_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv)
            for ob_1_out, (gamma_iter_1_outer_l0, k0_iter_8_outer_l0, k_proj_iter_3_outer_l0, kb_iter_6_outer_l0, normed_iter_1_outer_l0, v_proj_iter_3_outer_l0, x_chunk_iter_8_outer_l0, x_chunk_bf16_iter_1_outer_l0) in pl.range(0, 4, 1, init_values=(gamma_0, k0_4, k_proj_iter_1, kb_4, normed_0, v_proj_iter_1, x_chunk_4, x_chunk_bf16_0)):
                ret: pl.Tuple([pl.Tensor[[1, 256], pl.FP32], pl.Scalar[pl.INDEX], pl.Tensor[[16, 1024], pl.BFLOAT16], pl.Scalar[pl.INDEX], pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[16, 1024], pl.BFLOAT16], pl.Tensor[[4, 256], pl.BFLOAT16], pl.Tensor[[16, 256], pl.FP32]]) = self.call_group(qwen3_decode_layer_incore_1_group, b0_0, gamma_0, gamma_iter_1_outer_l0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k0_4, k0_iter_8_outer_l0, k_proj_0, k_proj_iter_1, k_proj_iter_3_outer_l0, kb_4, kb_iter_6_outer_l0, normed_0, normed_iter_1_outer_l0, ob_1_out, v_proj_0, v_proj_iter_1, v_proj_iter_3_outer_l0, wk_0, wv_0, x_chunk_4, x_chunk_bf16_0, x_chunk_bf16_iter_1_outer_l0, x_chunk_iter_8_outer_l0)
                gamma_iter_1_outer_l1_rv: pl.Tensor[[1, 256], pl.FP32] = ret[0]
                k0_iter_8_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                k_proj_iter_3_outer_l1_rv: pl.Tensor[[16, 1024], pl.BFLOAT16] = ret[2]
                kb_iter_6_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[3]
                normed_iter_1_outer_l1_rv: pl.Tensor[[4, 256], pl.FP32] = ret[4]
                v_proj_iter_3_outer_l1_rv: pl.Tensor[[16, 1024], pl.BFLOAT16] = ret[5]
                x_chunk_bf16_iter_1_outer_l1_rv: pl.Tensor[[4, 256], pl.BFLOAT16] = ret[6]
                x_chunk_iter_8_outer_l1_rv: pl.Tensor[[16, 256], pl.FP32] = ret[7]
                gamma_iter_1_outer_l0_rv, k0_iter_8_outer_l0_rv, k_proj_iter_3_outer_l0_rv, kb_iter_6_outer_l0_rv, normed_iter_1_outer_l0_rv, v_proj_iter_3_outer_l0_rv, x_chunk_iter_8_outer_l0_rv, x_chunk_bf16_iter_1_outer_l0_rv = pl.yield_(gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, k_proj_iter_3_outer_l1_rv, kb_iter_6_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_iter_3_outer_l1_rv, x_chunk_iter_8_outer_l1_rv, x_chunk_bf16_iter_1_outer_l1_rv)
            k0_2, k_proj_2, kb_2, q_proj_2, v_proj_2, x_chunk_2 = pl.yield_(k0_iter_8_outer_l0_rv, k_proj_iter_3_outer_l0_rv, kb_iter_6_outer_l0_rv, q_proj_iter_3_outer_l0_rv, v_proj_iter_3_outer_l0_rv, x_chunk_iter_8_outer_l0_rv)
        for b_0, (attn_out_iter_1, k_cache_iter_1, v_cache_iter_1) in pl.parallel(0, 16, 1, init_values=(attn_out_0, k_cache_0, v_cache_0), chunk=4):
            pos_0: pl.Scalar[pl.INT32] = pl.tensor.read(cache_pos_0, [b_0])
            ctx_len_0: pl.Scalar[pl.INDEX] = pl.cast(pos_0, pl.INDEX) + 1
            ctx_blocks_0: pl.Scalar[pl.INDEX] = (ctx_len_0 + 120 - 1) // 120
            cos_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(rope_cos_0, [1, 128], [pos_0, 0])
            sin_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(rope_sin_0, [1, 128], [pos_0, 0])
            cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(cos_row_0, [1, 128 // 2], [0, 0])
            cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(cos_row_0, [1, 128 // 2], [0, 128 // 2])
            sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(sin_row_0, [1, 128 // 2], [0, 0])
            sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(sin_row_0, [1, 128 // 2], [0, 128 // 2])
            for kvh_0, (k_cache_iter_3, v_cache_iter_3) in pl.parallel(0, 8, 1, init_values=(k_cache_iter_1, v_cache_iter_1), chunk=4):
                kv_col_0: pl.Scalar[pl.INDEX] = kvh_0 * 128
                _t14: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(k_proj_2, [1, 128], [b_0, kv_col_0])
                k_row_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.cast(_t14, target_type=pl.FP32, mode=2)
                k_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(k_row_0, [1, 128 // 2], [0, 0])
                k_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.view(k_row_0, [1, 128 // 2], [0, 128 // 2])
                k_rot_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
                _t15: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_lo_0, cos_lo_0)
                _t16: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_hi_0, sin_lo_0)
                _t17: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.sub(_t15, _t16)
                k_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_0, _t17, [0, 0])
                _t18: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_hi_0, cos_hi_0)
                _t19: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(k_lo_0, sin_hi_0)
                _t20: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.add(_t18, _t19)
                k_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(k_rot_1, _t20, [0, 128 // 2])
                cache_row_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_0 * 4096 + pl.cast(pos_0, pl.INDEX)
                _t21: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(k_rot_2, target_type=pl.BFLOAT16, mode=2)
                k_cache_5: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(k_cache_iter_3, _t21, [cache_row_0, 0])
                _t22: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.view(v_proj_2, [1, 128], [b_0, kv_col_0])
                v_cache_5: pl.Tensor[[524288, 128], pl.BFLOAT16] = pl.tensor.assemble(v_cache_iter_3, _t22, [cache_row_0, 0])
                k_cache_4, v_cache_4 = pl.yield_(k_cache_5, v_cache_5)
            attn_row_0: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.create([1, 5120], dtype=pl.FP32)
            attn_row_1: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.mul(attn_row_0, 0.0)
            for h_0_out, (attn_row_iter_2_outer_l0, kvh_iter_1_outer_l0) in pl.range(0, 8, 1, init_values=(attn_row_1, kvh_0)):
                ret: pl.Tuple([pl.Tensor[[1, 5120], pl.FP32], pl.Scalar[pl.INDEX]]) = self.call_group(qwen3_decode_layer_incore_2_group, attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_4, kvh_0, kvh_iter_1_outer_l0, q_proj_2, sin_hi_0, sin_lo_0, v_cache_4)
                attn_row_iter_2_outer_l1_rv: pl.Tensor[[1, 5120], pl.FP32] = ret[0]
                kvh_iter_1_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                attn_row_iter_2_outer_l0_rv, kvh_iter_1_outer_l0_rv = pl.yield_(attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv)
            attn_out_3: pl.Tensor[[16, 5120], pl.FP32] = pl.tensor.assemble(attn_out_iter_1, attn_row_iter_2_outer_l0_rv, [b_0, 0])
            attn_out_2, k_cache_2, v_cache_2 = pl.yield_(attn_out_3, k_cache_4, v_cache_4)
        for b0_1, (gamma_iter_6, inv_rms_iter_1, k0_iter_13, kb_iter_9, normed_iter_6, ob_iter_2, out_iter_1, sq_sum_iter_5, x_chunk_iter_13) in pl.range(0, 16, 4, init_values=(gamma_iter_1_outer_l0_rv, inv_rms_0, k0_2, kb_2, normed_iter_1_outer_l0_rv, ob_1, out_0, sq_sum_3, x_chunk_2)):
            resid1_tile_0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32)
            for ob_4_out, (k0_iter_15_outer_l0, kb_iter_11_outer_l0, resid1_tile_iter_1_outer_l0) in pl.range(0, 10, 1, init_values=(k0_iter_13, kb_iter_9, resid1_tile_0)):
                ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.FP32]]) = self.call_group(qwen3_decode_layer_incore_3_group, attn_out_2, b0_1, hidden_states_0, k0_2, k0_iter_13, k0_iter_15_outer_l0, kb_2, kb_iter_11_outer_l0, kb_iter_9, ob_4_out, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
                k0_iter_15_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
                kb_iter_11_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[1]
                resid1_tile_iter_1_outer_l1_rv: pl.Tensor[[4, 5120], pl.FP32] = ret[2]
                k0_iter_15_outer_l0_rv, kb_iter_11_outer_l0_rv, resid1_tile_iter_1_outer_l0_rv = pl.yield_(k0_iter_15_outer_l1_rv, kb_iter_11_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv)
            sq_sum_7: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.create([4, 1], dtype=pl.FP32)
            sq_sum_8: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_7, 0.0)
            for kb_14, (k0_iter_20, sq_sum_iter_9, x_chunk_iter_15) in pl.range(0, 20, 1, init_values=(k0_iter_15_outer_l0_rv, sq_sum_8, x_chunk_iter_13)):
                k0_22: pl.Scalar[pl.INDEX] = kb_14 * 256
                x_chunk_17: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 256], [0, k0_22])
                _t45: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.mul(x_chunk_17, x_chunk_17)
                _t46: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.row_sum(_t45)
                sq_sum_11: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(sq_sum_iter_9, _t46)
                k0_21, sq_sum_10, x_chunk_16 = pl.yield_(k0_22, sq_sum_11, x_chunk_17)
            _t47: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.mul(sq_sum_10, 0.000195313)
            _t48: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.add(_t47, 1e-06)
            inv_rms_3: pl.Tensor[[4, 1], pl.FP32] = pl.tensor.rsqrt(_t48)
            post_norm_tile_0: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.create([4, 5120], dtype=pl.BFLOAT16)
            down_proj_tile_0: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.create([4, 5120], dtype=pl.FP32)
            down_proj_tile_1: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.mul(down_proj_tile_0, 0.0)
            for kb_15, (gamma_iter_8, k0_iter_23, normed_iter_8, post_norm_tile_iter_1, x_chunk_iter_18) in pl.range(0, 20, 1, init_values=(gamma_iter_6, k0_21, normed_iter_6, post_norm_tile_0, x_chunk_16)):
                k0_25: pl.Scalar[pl.INDEX] = kb_15 * 256
                x_chunk_20: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.view(resid1_tile_iter_1_outer_l0_rv, [4, 256], [0, k0_25])
                gamma_10: pl.Tensor[[1, 256], pl.FP32] = pl.tensor.view(post_rms_weight_0, [1, 256], [0, k0_25])
                _t49: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_20, inv_rms_3)
                normed_10: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(_t49, gamma_10)
                _t50: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.cast(normed_10, target_type=pl.BFLOAT16, mode=2)
                post_norm_tile_3: pl.Tensor[[4, 5120], pl.BFLOAT16] = pl.tensor.assemble(post_norm_tile_iter_1, _t50, [0, k0_25])
                gamma_9, k0_24, normed_9, post_norm_tile_2, x_chunk_19 = pl.yield_(gamma_10, k0_25, normed_10, post_norm_tile_3, x_chunk_20)
            for ob_5, (down_proj_tile_iter_2, k0_iter_26, kb_iter_16, o0_iter_1) in pl.range(0, 800, 1, init_values=(down_proj_tile_1, k0_24, kb_15, o0_0)):
                o0_3: pl.Scalar[pl.INDEX] = ob_5 * 32
                gate_acc_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32)
                up_acc_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.create([4, 32], dtype=pl.FP32)
                gate_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(gate_acc_0, 0.0)
                up_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(up_acc_0, 0.0)
                for kb_18, (gate_acc_iter_2, k0_iter_28, up_acc_iter_2) in pl.range(0, 20, 1, init_values=(gate_acc_1, k0_iter_26, up_acc_1)):
                    k0_30: pl.Scalar[pl.INDEX] = kb_18 * 256
                    post_chunk_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.view(post_norm_tile_2, [4, 256], [0, k0_30])
                    wg_0: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.view(w_gate_0, [256, 32], [k0_30, o0_3])
                    wu_0: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.view(w_up_0, [256, 32], [k0_30, o0_3])
                    _t51: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.matmul(post_chunk_0, wg_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                    gate_acc_4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(gate_acc_iter_2, _t51)
                    _t52: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.matmul(post_chunk_0, wu_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                    up_acc_4: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(up_acc_iter_2, _t52)
                    gate_acc_3, k0_29, up_acc_3 = pl.yield_(gate_acc_4, k0_30, up_acc_4)
                _t53: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.neg(gate_acc_3)
                _t54: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.exp(_t53)
                _t55: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.add(_t54, 1.0)
                sigmoid_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.recip(_t55)
                _t56: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(gate_acc_3, sigmoid_0)
                mlp_chunk_0: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(_t56, up_acc_3)
                mlp_chunk_bf16_0: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.cast(mlp_chunk_0, target_type=pl.BFLOAT16, mode=2)
                for dob_0_out, (down_proj_tile_iter_4_outer_l0,) in pl.range(0, 20, 1, init_values=(down_proj_tile_iter_2,)):
                    down_proj_tile_iter_4_outer_l1_rv: pl.Tensor[[4, 5120], pl.FP32] = self.call_group(qwen3_decode_layer_incore_4_group, dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
                    down_proj_tile_iter_4_outer_l0_rv: pl.Tensor[[4, 5120], pl.FP32] = pl.yield_(down_proj_tile_iter_4_outer_l1_rv)
                down_proj_tile_3, k0_27, kb_17, o0_2 = pl.yield_(down_proj_tile_iter_4_outer_l0_rv, k0_29, kb_18, o0_3)
            for ob_6_out, (o0_iter_4_outer_l0, out_iter_3_outer_l0) in pl.range(0, 20, 1, init_values=(o0_2, out_iter_1)):
                ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.BFLOAT16]]) = self.qwen3_decode_layer_incore_5(b0_1, down_proj_tile_3, o0_2, o0_iter_4_outer_l0, ob_6_out, out_0, out_iter_1, out_iter_3_outer_l0, resid1_tile_iter_1_outer_l0_rv)
                o0_iter_4_outer_l1_rv: pl.Scalar[pl.INDEX] = ret[0]
                out_iter_3_outer_l1_rv: pl.Tensor[[16, 5120], pl.BFLOAT16] = ret[1]
                o0_iter_4_outer_l0_rv, out_iter_3_outer_l0_rv = pl.yield_(o0_iter_4_outer_l1_rv, out_iter_3_outer_l1_rv)
            gamma_7, inv_rms_2, k0_14, kb_10, normed_7, ob_3, out_2, sq_sum_6, x_chunk_14 = pl.yield_(gamma_9, inv_rms_3, k0_27, kb_17, normed_9, ob_6, out_iter_3_outer_l0_rv, sq_sum_10, x_chunk_19)
        return out_2
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_0_aic(self, b0_0: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], inv_rms_tile_0: pl.Tensor[[4, 1], pl.FP32], k0_0: pl.Scalar[pl.INDEX], k0_iter_1: pl.Scalar[pl.INDEX], k0_iter_3_outer_l0: pl.Scalar[pl.INDEX], kb_0: pl.Scalar[pl.INDEX], kb_iter_1: pl.Scalar[pl.INDEX], kb_iter_3_outer_l0: pl.Scalar[pl.INDEX], ob_0_out: pl.Scalar[pl.INDEX], q_proj_0: pl.Tensor[[16, 5120], pl.BFLOAT16], q_proj_iter_1: pl.Tensor[[16, 5120], pl.BFLOAT16], q_proj_iter_3_outer_l0: pl.Tensor[[16, 5120], pl.BFLOAT16], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], x_chunk_0: pl.Tensor[[16, 256], pl.FP32], x_chunk_iter_1: pl.Tensor[[16, 256], pl.FP32], x_chunk_iter_3_outer_l0: pl.Tensor[[16, 256], pl.FP32]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.BFLOAT16], pl.Tensor[[16, 256], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_0_in, (k0_iter_3_outer_l1, kb_iter_3_outer_l1, q_proj_iter_3_outer_l1, x_chunk_iter_3_outer_l1) in pl.parallel(0, 4, 1, init_values=(k0_iter_3_outer_l0, kb_iter_3_outer_l0, q_proj_iter_3_outer_l0, x_chunk_iter_3_outer_l0)):
            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 4 + ob_0_in) * 1) * 64
            for kb_5, (k0_iter_5, x_chunk_iter_5) in pl.range(0, 20, 1, init_values=(k0_iter_3_outer_l1, x_chunk_iter_3_outer_l1)):
                k0_7: pl.Scalar[pl.INDEX] = kb_5 * 256
                wq_chunk_0__h0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                wq_chunk_0__h1: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                wq_chunk_0__tmp: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.create(__list__(256, 64), dtype=pl.BFLOAT16)
                wq_chunk_0__mid: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(wq_chunk_0__tmp, wq_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wq_chunk_0: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(wq_chunk_0__mid, wq_chunk_0__h1, __list__(128, 0))
                pl.comm.tfree_to_aiv(1)
                _t6__h0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                _t6__h1: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                _t6__tmp: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.create(__list__(4, 256), dtype=pl.BFLOAT16)
                _t6__mid: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(_t6__tmp, _t6__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                _t6: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(_t6__mid, _t6__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                _t7: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(_t6, wq_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t7, __list__(2, 64), __list__(0, 0))
                __half1__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t7, __list__(2, 64), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                k0_6, x_chunk_6 = pl.yield_(k0_7)
            k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, q_proj_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv = pl.yield_(k0_6, kb_5, x_chunk_6)
        return k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, q_proj_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_0_aiv(self, b0_0: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], inv_rms_tile_0: pl.Tensor[[4, 1], pl.FP32], k0_0: pl.Scalar[pl.INDEX], k0_iter_1: pl.Scalar[pl.INDEX], k0_iter_3_outer_l0: pl.Scalar[pl.INDEX], kb_0: pl.Scalar[pl.INDEX], kb_iter_1: pl.Scalar[pl.INDEX], kb_iter_3_outer_l0: pl.Scalar[pl.INDEX], ob_0_out: pl.Scalar[pl.INDEX], q_proj_0: pl.Tensor[[16, 5120], pl.BFLOAT16], q_proj_iter_1: pl.Tensor[[16, 5120], pl.BFLOAT16], q_proj_iter_3_outer_l0: pl.Tensor[[16, 5120], pl.BFLOAT16], wq_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], x_chunk_0: pl.Tensor[[16, 256], pl.FP32], x_chunk_iter_1: pl.Tensor[[16, 256], pl.FP32], x_chunk_iter_3_outer_l0: pl.Tensor[[16, 256], pl.FP32], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[16, 5120], pl.BFLOAT16], pl.Tensor[[16, 256], pl.FP32]]:
        pl.comm.aiv_initialize_pipe()
        for ob_0_in, (k0_iter_3_outer_l1, kb_iter_3_outer_l1, q_proj_iter_3_outer_l1, x_chunk_iter_3_outer_l1) in pl.parallel(0, 4, 1, init_values=(k0_iter_3_outer_l0, kb_iter_3_outer_l0, q_proj_iter_3_outer_l0, x_chunk_iter_3_outer_l0)):
            q0_0: pl.Scalar[pl.INDEX] = (0 + (ob_0_out * 4 + ob_0_in) * 1) * 64
            q_acc_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.create([2, 64], dtype=pl.FP32)
            q_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(q_acc_0, 0.0)
            for kb_5, (k0_iter_5, q_acc_iter_2, x_chunk_iter_5) in pl.range(0, 20, 1, init_values=(k0_iter_3_outer_l1, q_acc_1, x_chunk_iter_3_outer_l1)):
                k0_7: pl.Scalar[pl.INDEX] = kb_5 * 256
                x_chunk_bf16_0: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 128], [b0_0, k0_7 + AIV_IDX * 128])
                x_chunk_7: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.cast(x_chunk_bf16_0, target_type=pl.FP32, mode=2)
                gamma_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 128], [0, k0_7 + AIV_IDX * 128])
                _t5: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_7, inv_rms_tile_0)
                normed_0: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(_t5, gamma_0)
                wq_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(wq_0, [128, 64], [k0_7 + AIV_IDX * 128, q0_0])
                pl.comm.tpush_to_aic(wq_chunk_0, AIV_IDX)
                _t6: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.cast(normed_0, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(_t6, AIV_IDX)
                _t7: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
                q_acc_4: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(q_acc_iter_2, _t7)
                pl.comm.tfree_to_aic(AIV_IDX)
                k0_6, q_acc_3, x_chunk_6 = pl.yield_(k0_7, q_acc_4, x_chunk_7)
            _t8: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.cast(q_acc_3, target_type=pl.BFLOAT16, mode=2)
            q_proj_5: pl.Tensor[[16, 5120], pl.BFLOAT16] = pl.tensor.assemble(q_proj_iter_3_outer_l1, _t8, [b0_0 + AIV_IDX * 2, q0_0])
            k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, q_proj_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv = pl.yield_(k0_6, kb_5, q_proj_5, x_chunk_6)
        return k0_iter_3_outer_l1_rv, kb_iter_3_outer_l1_rv, q_proj_iter_3_outer_l1_rv, x_chunk_iter_3_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_0_aic", aiv="qwen3_decode_layer_incore_0_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_0_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_0_group, b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k0_0, k0_iter_1, k0_iter_3_outer_l0, kb_0, kb_iter_1, kb_iter_3_outer_l0, ob_0_out, q_proj_0, q_proj_iter_1, q_proj_iter_3_outer_l0, wq_0, x_chunk_0, x_chunk_iter_1, x_chunk_iter_3_outer_l0)
            → qwen3_decode_layer_incore_0_aic(b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k0_0, k0_iter_1, k0_iter_3_outer_l0, kb_0, kb_iter_1, kb_iter_3_outer_l0, ob_0_out, q_proj_0, q_proj_iter_1, q_proj_iter_3_outer_l0, wq_0, x_chunk_0, x_chunk_iter_1, x_chunk_iter_3_outer_l0)
            → qwen3_decode_layer_incore_0_aiv(b0_0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k0_0, k0_iter_1, k0_iter_3_outer_l0, kb_0, kb_iter_1, kb_iter_3_outer_l0, ob_0_out, q_proj_0, q_proj_iter_1, q_proj_iter_3_outer_l0, wq_0, x_chunk_0, x_chunk_iter_1, x_chunk_iter_3_outer_l0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_1_aic(self, b0_0: pl.Scalar[pl.INDEX], gamma_0: pl.Tensor[[1, 256], pl.FP32], gamma_iter_1_outer_l0: pl.Tensor[[1, 256], pl.FP32], hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], inv_rms_tile_0: pl.Tensor[[4, 1], pl.FP32], k0_4: pl.Scalar[pl.INDEX], k0_iter_8_outer_l0: pl.Scalar[pl.INDEX], k_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16], k_proj_iter_1: pl.Tensor[[16, 1024], pl.BFLOAT16], k_proj_iter_3_outer_l0: pl.Tensor[[16, 1024], pl.BFLOAT16], kb_4: pl.Scalar[pl.INDEX], kb_iter_6_outer_l0: pl.Scalar[pl.INDEX], normed_0: pl.Tensor[[4, 256], pl.FP32], normed_iter_1_outer_l0: pl.Tensor[[4, 256], pl.FP32], ob_1_out: pl.Scalar[pl.INDEX], v_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16], v_proj_iter_1: pl.Tensor[[16, 1024], pl.BFLOAT16], v_proj_iter_3_outer_l0: pl.Tensor[[16, 1024], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], x_chunk_4: pl.Tensor[[16, 256], pl.FP32], x_chunk_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16], x_chunk_bf16_iter_1_outer_l0: pl.Tensor[[4, 256], pl.BFLOAT16], x_chunk_iter_8_outer_l0: pl.Tensor[[16, 256], pl.FP32]) -> tuple[pl.Tensor[[1, 256], pl.FP32], pl.Scalar[pl.INDEX], pl.Tensor[[16, 1024], pl.BFLOAT16], pl.Scalar[pl.INDEX], pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[16, 1024], pl.BFLOAT16], pl.Tensor[[4, 256], pl.BFLOAT16], pl.Tensor[[16, 256], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_1_in, (gamma_iter_1_outer_l1, k0_iter_8_outer_l1, k_proj_iter_3_outer_l1, kb_iter_6_outer_l1, normed_iter_1_outer_l1, v_proj_iter_3_outer_l1, x_chunk_iter_8_outer_l1, x_chunk_bf16_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(gamma_iter_1_outer_l0, k0_iter_8_outer_l0, k_proj_iter_3_outer_l0, kb_iter_6_outer_l0, normed_iter_1_outer_l0, v_proj_iter_3_outer_l0, x_chunk_iter_8_outer_l0, x_chunk_bf16_iter_1_outer_l0)):
            kv0_0: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 32
            for kb_8, (gamma_iter_3, k0_iter_10, normed_iter_3, x_chunk_iter_10, x_chunk_bf16_iter_3) in pl.range(0, 20, 1, init_values=(gamma_iter_1_outer_l1, k0_iter_8_outer_l1, normed_iter_1_outer_l1, x_chunk_iter_8_outer_l1, x_chunk_bf16_iter_1_outer_l1)):
                k0_12: pl.Scalar[pl.INDEX] = kb_8 * 256
                normed_bf16_0__h0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                normed_bf16_0__h1: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                normed_bf16_0__tmp: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.create(__list__(4, 256), dtype=pl.BFLOAT16)
                normed_bf16_0__mid: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(normed_bf16_0__tmp, normed_bf16_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                normed_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(normed_bf16_0__mid, normed_bf16_0__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                wk_chunk_0__h0: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                wk_chunk_0__h1: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                wk_chunk_0__tmp: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.create(__list__(256, 32), dtype=pl.BFLOAT16)
                wk_chunk_0__mid: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.assemble(wk_chunk_0__tmp, wk_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wk_chunk_0: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.assemble(wk_chunk_0__mid, wk_chunk_0__h1, __list__(128, 0))
                pl.comm.tfree_to_aiv(1)
                wv_chunk_0__h0: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                wv_chunk_0__h1: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                wv_chunk_0__tmp: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.create(__list__(256, 32), dtype=pl.BFLOAT16)
                wv_chunk_0__mid: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.assemble(wv_chunk_0__tmp, wv_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                wv_chunk_0: pl.Tensor[[256, 32], pl.BFLOAT16] = pl.tensor.assemble(wv_chunk_0__mid, wv_chunk_0__h1, __list__(128, 0))
                pl.comm.tfree_to_aiv(1)
                _t10: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wk_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.view(_t10, __list__(2, 32), __list__(0, 0))
                __half1__: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.view(_t10, __list__(2, 32), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                _t11: pl.Tensor[[4, 32], pl.BFLOAT16] = pl.tensor.matmul(normed_bf16_0, wv_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.view(_t11, __list__(2, 32), __list__(0, 0))
                __half1__: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.view(_t11, __list__(2, 32), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                gamma_4, k0_11, normed_4, x_chunk_11, x_chunk_bf16_4 = pl.yield_(k0_12)
            gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, k_proj_iter_3_outer_l1_rv, kb_iter_6_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_iter_3_outer_l1_rv, x_chunk_iter_8_outer_l1_rv, x_chunk_bf16_iter_1_outer_l1_rv = pl.yield_(gamma_4, k0_11, kb_8, normed_4, x_chunk_11, x_chunk_bf16_4)
        return gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, k_proj_iter_3_outer_l1_rv, kb_iter_6_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_iter_3_outer_l1_rv, x_chunk_bf16_iter_1_outer_l1_rv, x_chunk_iter_8_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_1_aiv(self, b0_0: pl.Scalar[pl.INDEX], gamma_0: pl.Tensor[[1, 256], pl.FP32], gamma_iter_1_outer_l0: pl.Tensor[[1, 256], pl.FP32], hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16], input_rms_weight_0: pl.Tensor[[1, 5120], pl.FP32], inv_rms_tile_0: pl.Tensor[[4, 1], pl.FP32], k0_4: pl.Scalar[pl.INDEX], k0_iter_8_outer_l0: pl.Scalar[pl.INDEX], k_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16], k_proj_iter_1: pl.Tensor[[16, 1024], pl.BFLOAT16], k_proj_iter_3_outer_l0: pl.Tensor[[16, 1024], pl.BFLOAT16], kb_4: pl.Scalar[pl.INDEX], kb_iter_6_outer_l0: pl.Scalar[pl.INDEX], normed_0: pl.Tensor[[4, 256], pl.FP32], normed_iter_1_outer_l0: pl.Tensor[[4, 256], pl.FP32], ob_1_out: pl.Scalar[pl.INDEX], v_proj_0: pl.Tensor[[16, 1024], pl.BFLOAT16], v_proj_iter_1: pl.Tensor[[16, 1024], pl.BFLOAT16], v_proj_iter_3_outer_l0: pl.Tensor[[16, 1024], pl.BFLOAT16], wk_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], wv_0: pl.Tensor[[5120, 1024], pl.BFLOAT16], x_chunk_4: pl.Tensor[[16, 256], pl.FP32], x_chunk_bf16_0: pl.Tensor[[4, 256], pl.BFLOAT16], x_chunk_bf16_iter_1_outer_l0: pl.Tensor[[4, 256], pl.BFLOAT16], x_chunk_iter_8_outer_l0: pl.Tensor[[16, 256], pl.FP32], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[1, 256], pl.FP32], pl.Scalar[pl.INDEX], pl.Tensor[[16, 1024], pl.BFLOAT16], pl.Scalar[pl.INDEX], pl.Tensor[[4, 256], pl.FP32], pl.Tensor[[16, 1024], pl.BFLOAT16], pl.Tensor[[4, 256], pl.BFLOAT16], pl.Tensor[[16, 256], pl.FP32]]:
        pl.comm.aiv_initialize_pipe()
        for ob_1_in, (gamma_iter_1_outer_l1, k0_iter_8_outer_l1, k_proj_iter_3_outer_l1, kb_iter_6_outer_l1, normed_iter_1_outer_l1, v_proj_iter_3_outer_l1, x_chunk_iter_8_outer_l1, x_chunk_bf16_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(gamma_iter_1_outer_l0, k0_iter_8_outer_l0, k_proj_iter_3_outer_l0, kb_iter_6_outer_l0, normed_iter_1_outer_l0, v_proj_iter_3_outer_l0, x_chunk_iter_8_outer_l0, x_chunk_bf16_iter_1_outer_l0)):
            kv0_0: pl.Scalar[pl.INDEX] = (0 + (ob_1_out * 8 + ob_1_in) * 1) * 32
            k_acc_0: pl.Tensor[[2, 32], pl.FP32] = pl.tensor.create([2, 32], dtype=pl.FP32)
            v_acc_0: pl.Tensor[[2, 32], pl.FP32] = pl.tensor.create([2, 32], dtype=pl.FP32)
            k_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(k_acc_0, 0.0)
            v_acc_1: pl.Tensor[[4, 32], pl.FP32] = pl.tensor.mul(v_acc_0, 0.0)
            for kb_8, (gamma_iter_3, k0_iter_10, k_acc_iter_2, normed_iter_3, v_acc_iter_2, x_chunk_iter_10, x_chunk_bf16_iter_3) in pl.range(0, 20, 1, init_values=(gamma_iter_1_outer_l1, k0_iter_8_outer_l1, k_acc_1, normed_iter_1_outer_l1, v_acc_1, x_chunk_iter_8_outer_l1, x_chunk_bf16_iter_1_outer_l1)):
                k0_12: pl.Scalar[pl.INDEX] = kb_8 * 256
                x_chunk_bf16_5: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [4, 128], [b0_0, k0_12 + AIV_IDX * 128])
                x_chunk_12: pl.Tensor[[4, 128], pl.FP32] = pl.tensor.cast(x_chunk_bf16_5, target_type=pl.FP32, mode=2)
                gamma_5: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.view(input_rms_weight_0, [1, 128], [0, k0_12 + AIV_IDX * 128])
                _t9: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.row_expand_mul(x_chunk_12, inv_rms_tile_0)
                normed_5: pl.Tensor[[4, 256], pl.FP32] = pl.tensor.col_expand_mul(_t9, gamma_5)
                normed_bf16_0: pl.Tensor[[4, 128], pl.BFLOAT16] = pl.tensor.cast(normed_5, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(normed_bf16_0, AIV_IDX)
                wk_chunk_0: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.tensor.view(wk_0, [128, 32], [k0_12 + AIV_IDX * 128, kv0_0])
                pl.comm.tpush_to_aic(wk_chunk_0, AIV_IDX)
                wv_chunk_0: pl.Tensor[[128, 32], pl.BFLOAT16] = pl.tensor.view(wv_0, [128, 32], [k0_12 + AIV_IDX * 128, kv0_0])
                pl.comm.tpush_to_aic(wv_chunk_0, AIV_IDX)
                _t10: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
                k_acc_4: pl.Tensor[[2, 32], pl.FP32] = pl.tensor.add(k_acc_iter_2, _t10)
                pl.comm.tfree_to_aic(AIV_IDX)
                _t11: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
                v_acc_4: pl.Tensor[[2, 32], pl.FP32] = pl.tensor.add(v_acc_iter_2, _t11)
                pl.comm.tfree_to_aic(AIV_IDX)
                gamma_4, k0_11, k_acc_3, normed_4, v_acc_3, x_chunk_11, x_chunk_bf16_4 = pl.yield_(gamma_5, k0_12, k_acc_4, normed_5, v_acc_4, x_chunk_12, x_chunk_bf16_5)
            _t12: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.cast(k_acc_3, target_type=pl.BFLOAT16, mode=2)
            k_proj_5: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.assemble(k_proj_iter_3_outer_l1, _t12, [b0_0 + AIV_IDX * 2, kv0_0])
            _t13: pl.Tensor[[2, 32], pl.BFLOAT16] = pl.tensor.cast(v_acc_3, target_type=pl.BFLOAT16, mode=2)
            v_proj_5: pl.Tensor[[16, 1024], pl.BFLOAT16] = pl.tensor.assemble(v_proj_iter_3_outer_l1, _t13, [b0_0 + AIV_IDX * 2, kv0_0])
            gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, k_proj_iter_3_outer_l1_rv, kb_iter_6_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_iter_3_outer_l1_rv, x_chunk_iter_8_outer_l1_rv, x_chunk_bf16_iter_1_outer_l1_rv = pl.yield_(gamma_4, k0_11, k_proj_5, kb_8, normed_4, v_proj_5, x_chunk_11, x_chunk_bf16_4)
        return gamma_iter_1_outer_l1_rv, k0_iter_8_outer_l1_rv, k_proj_iter_3_outer_l1_rv, kb_iter_6_outer_l1_rv, normed_iter_1_outer_l1_rv, v_proj_iter_3_outer_l1_rv, x_chunk_bf16_iter_1_outer_l1_rv, x_chunk_iter_8_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_1_aic", aiv="qwen3_decode_layer_incore_1_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_1_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_1_group, b0_0, gamma_0, gamma_iter_1_outer_l0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k0_4, k0_iter_8_outer_l0, k_proj_0, k_proj_iter_1, k_proj_iter_3_outer_l0, kb_4, kb_iter_6_outer_l0, normed_0, normed_iter_1_outer_l0, ob_1_out, v_proj_0, v_proj_iter_1, v_proj_iter_3_outer_l0, wk_0, wv_0, x_chunk_4, x_chunk_bf16_0, x_chunk_bf16_iter_1_outer_l0, x_chunk_iter_8_outer_l0)
            → qwen3_decode_layer_incore_1_aic(b0_0, gamma_0, gamma_iter_1_outer_l0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k0_4, k0_iter_8_outer_l0, k_proj_0, k_proj_iter_1, k_proj_iter_3_outer_l0, kb_4, kb_iter_6_outer_l0, normed_0, normed_iter_1_outer_l0, ob_1_out, v_proj_0, v_proj_iter_1, v_proj_iter_3_outer_l0, wk_0, wv_0, x_chunk_4, x_chunk_bf16_0, x_chunk_bf16_iter_1_outer_l0, x_chunk_iter_8_outer_l0)
            → qwen3_decode_layer_incore_1_aiv(b0_0, gamma_0, gamma_iter_1_outer_l0, hidden_states_0, input_rms_weight_0, inv_rms_tile_0, k0_4, k0_iter_8_outer_l0, k_proj_0, k_proj_iter_1, k_proj_iter_3_outer_l0, kb_4, kb_iter_6_outer_l0, normed_0, normed_iter_1_outer_l0, ob_1_out, v_proj_0, v_proj_iter_1, v_proj_iter_3_outer_l0, wk_0, wv_0, x_chunk_4, x_chunk_bf16_0, x_chunk_bf16_iter_1_outer_l0, x_chunk_iter_8_outer_l0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_2_aic(self, attn_row_1: pl.Tensor[[1, 5120], pl.FP32], attn_row_iter_2_outer_l0: pl.Tensor[[1, 5120], pl.FP32], b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32], cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32], ctx_blocks_0: pl.Scalar[pl.INDEX], ctx_len_0: pl.Scalar[pl.INDEX], h_0_out: pl.Scalar[pl.INDEX], k_cache_4: pl.Tensor[[524288, 128], pl.BFLOAT16], kvh_0: pl.Scalar[pl.INDEX], kvh_iter_1_outer_l0: pl.Scalar[pl.INDEX], q_proj_2: pl.Tensor[[16, 5120], pl.BFLOAT16], sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32], sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32], v_cache_4: pl.Tensor[[524288, 128], pl.BFLOAT16]) -> tuple[pl.Tensor[[1, 5120], pl.FP32], pl.Scalar[pl.INDEX]]:
        pl.comm.aic_initialize_pipe()
        for h_0_in, (attn_row_iter_2_outer_l1, kvh_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer_l0, kvh_iter_1_outer_l0)):
            kvh_3: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) // 8
            q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
            q_rot_bf16_0: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
            q_rot_bf16_0__discard: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
            pl.comm.tfree_to_aiv(1)
            for sb_0 in pl.range(0, ctx_blocks_0, 1):
                s0_0: pl.Scalar[pl.INDEX] = sb_0 * 120
                valid_len_0: pl.Scalar[pl.INDEX] = min(120, ctx_len_0 - s0_0)
                cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_3 * 4096 + s0_0
                k_tile_0__h0: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                k_tile_0__h1: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                k_tile_0__tmp: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.create(__list__(120, 128), dtype=pl.BFLOAT16)
                k_tile_0__mid: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.assemble(k_tile_0__tmp, k_tile_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                k_tile_0: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.assemble(k_tile_0__mid, k_tile_0__h1, __list__(60, 0))
                pl.comm.tfree_to_aiv(1)
                v_tile_0__h0: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                v_tile_0__h1: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                v_tile_0__tmp: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.create(__list__(120, 128), dtype=pl.BFLOAT16)
                v_tile_0__mid: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.assemble(v_tile_0__tmp, v_tile_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                v_tile_0: pl.Tensor[[120, 128], pl.BFLOAT16] = pl.tensor.assemble(v_tile_0__mid, v_tile_0__h1, __list__(60, 0))
                pl.comm.tfree_to_aiv(1)
                _t30: pl.Tensor[[1, 120], pl.BFLOAT16] = pl.tensor.matmul(q_rot_bf16_0, k_tile_0, a_trans=False, b_trans=True, c_matrix_nz=False)
                scores_0: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.mul(_t30, 0.0883883)
                _t34: pl.Tensor[[1, 120], pl.BFLOAT16] = pl.comm.tpop_from_aiv()
                oi_tmp_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.matmul(_t34, v_tile_0, a_trans=False, b_trans=False, c_matrix_nz=False, out_dtype=pl.FP32)
                pl.comm.tfree_to_aiv()
                if sb_0 == 0:
                    oi_4: pl.Tensor[[1, 128], pl.FP32] = oi_tmp_0
                    li_6, mi_6, oi_6 = pl.yield_(oi_4)
                else:

                pl.yield_(li_6, mi_6, oi_6)
            pl.comm.tfree_to_aiv(0)
            attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv = pl.yield_(kvh_3)
        return attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_2_aiv(self, attn_row_1: pl.Tensor[[1, 5120], pl.FP32], attn_row_iter_2_outer_l0: pl.Tensor[[1, 5120], pl.FP32], b_0: pl.Scalar[pl.INDEX], cos_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32], cos_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32], ctx_blocks_0: pl.Scalar[pl.INDEX], ctx_len_0: pl.Scalar[pl.INDEX], h_0_out: pl.Scalar[pl.INDEX], k_cache_4: pl.Tensor[[524288, 128], pl.BFLOAT16], kvh_0: pl.Scalar[pl.INDEX], kvh_iter_1_outer_l0: pl.Scalar[pl.INDEX], q_proj_2: pl.Tensor[[16, 5120], pl.BFLOAT16], sin_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32], sin_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32], v_cache_4: pl.Tensor[[524288, 128], pl.BFLOAT16], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Tensor[[1, 5120], pl.FP32], pl.Scalar[pl.INDEX]]:
        pl.comm.aiv_initialize_pipe()
        for h_0_in, (attn_row_iter_2_outer_l1, kvh_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(attn_row_iter_2_outer_l0, kvh_iter_1_outer_l0)):
            kvh_3: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) // 8
            q_col_0: pl.Scalar[pl.INDEX] = (0 + (h_0_out * 8 + h_0_in) * 1) * 128
            _t23: pl.Tensor[[1, 64], pl.BFLOAT16] = pl.tensor.view(q_proj_2, [1, 64], [b_0, q_col_0 + AIV_IDX * 64])
            q_row_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.cast(_t23, target_type=pl.FP32, mode=2)
            q_lo_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.deep_view(q_row_0, [1, 128 // 2], [0, 0])
            q_hi_0: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.deep_view(q_row_0, [1, 128 // 2], [0, 128 // 2])
            q_rot_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.create([1, 128], dtype=pl.FP32)
            _t24: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_lo_0, cos_lo_0)
            _t25: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_hi_0, sin_lo_0)
            _t26: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.sub(_t24, _t25)
            q_rot_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_0, _t26, [0, 0])
            _t27: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_hi_0, cos_hi_0)
            _t28: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.col_expand_mul(q_lo_0, sin_hi_0)
            _t29: pl.Tensor[[1, 128 // 2], pl.FP32] = pl.tensor.add(_t27, _t28)
            q_rot_2: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.assemble(q_rot_1, _t29, [0, 128 // 2])
            q_rot_bf16_0: pl.Tensor[[1, 128], pl.BFLOAT16] = pl.tensor.cast(q_rot_2, target_type=pl.BFLOAT16, mode=2)
            pl.comm.tpush_to_aic(q_rot_bf16_0, AIV_IDX)
            oi_0: pl.Tensor[[1, 64], pl.FP32] = pl.tensor.create([1, 64], dtype=pl.FP32)
            li_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
            mi_0: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.create([1, 1], dtype=pl.FP32)
            oi_1: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.mul(oi_0, 0.0)
            li_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(li_0, 0.0)
            mi_1: pl.Tensor[[1, 1], pl.FP32] = pl.tensor.mul(mi_0, 0.0)
            for sb_0, (li_iter_2, mi_iter_2, oi_iter_2) in pl.range(0, ctx_blocks_0, 1, init_values=(li_1, mi_1, oi_1)):
                s0_0: pl.Scalar[pl.INDEX] = sb_0 * 120
                valid_len_0: pl.Scalar[pl.INDEX] = min(120, ctx_len_0 - s0_0)
                cache_row0_0: pl.Scalar[pl.INDEX] = b_0 * 8 * 4096 + kvh_3 * 4096 + s0_0
                k_tile_0: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.tensor.view(k_cache_4, [60, 128], [cache_row0_0 + AIV_IDX * 60, 0])
                pl.comm.tpush_to_aic(k_tile_0, AIV_IDX)
                v_tile_0: pl.Tensor[[60, 128], pl.BFLOAT16] = pl.tensor.view(v_cache_4, [60, 128], [cache_row0_0 + AIV_IDX * 60, 0])
                pl.comm.tpush_to_aic(v_tile_0, AIV_IDX)
                exp_pad_0: pl.Tensor[[1, 60], pl.FP32] = pl.tensor.create([1, 60], dtype=pl.FP32)
                exp_pad_1: pl.Tensor[[1, 120], pl.FP32] = pl.tensor.mul(exp_pad_0, 0.0)
            ctx_0: pl.Tensor[[1, 128], pl.FP32] = pl.tensor.row_expand_div(oi_3, li_3)
            attn_row_4: pl.Tensor[[1, 5120], pl.FP32] = pl.tensor.assemble(attn_row_iter_2_outer_l1, ctx_0, [0, q_col_0])
            attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv = pl.yield_(attn_row_4, kvh_3)
        return attn_row_iter_2_outer_l1_rv, kvh_iter_1_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_2_aic", aiv="qwen3_decode_layer_incore_2_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_2_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_2_group, attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_4, kvh_0, kvh_iter_1_outer_l0, q_proj_2, sin_hi_0, sin_lo_0, v_cache_4)
            → qwen3_decode_layer_incore_2_aic(attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_4, kvh_0, kvh_iter_1_outer_l0, q_proj_2, sin_hi_0, sin_lo_0, v_cache_4)
            → qwen3_decode_layer_incore_2_aiv(attn_row_1, attn_row_iter_2_outer_l0, b_0, cos_hi_0, cos_lo_0, ctx_blocks_0, ctx_len_0, h_0_out, k_cache_4, kvh_0, kvh_iter_1_outer_l0, q_proj_2, sin_hi_0, sin_lo_0, v_cache_4, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_3_aic(self, attn_out_2: pl.Tensor[[16, 5120], pl.FP32], b0_1: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16], k0_2: pl.Scalar[pl.INDEX], k0_iter_13: pl.Scalar[pl.INDEX], k0_iter_15_outer_l0: pl.Scalar[pl.INDEX], kb_2: pl.Scalar[pl.INDEX], kb_iter_11_outer_l0: pl.Scalar[pl.INDEX], kb_iter_9: pl.Scalar[pl.INDEX], ob_4_out: pl.Scalar[pl.INDEX], resid1_tile_0: pl.Tensor[[4, 5120], pl.FP32], resid1_tile_iter_1_outer_l0: pl.Tensor[[4, 5120], pl.FP32], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.FP32]]:
        pl.comm.aic_initialize_pipe()
        for ob_4_in, (k0_iter_15_outer_l1, kb_iter_11_outer_l1, resid1_tile_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(k0_iter_15_outer_l0, kb_iter_11_outer_l0, resid1_tile_iter_1_outer_l0)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_4_out * 8 + ob_4_in) * 1) * 64
            for kb_13, (k0_iter_17,) in pl.range(0, 20, 1, init_values=(k0_iter_15_outer_l1,)):
                k0_19: pl.Scalar[pl.INDEX] = kb_13 * 256
                a_chunk_0__h0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                a_chunk_0__h1: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                a_chunk_0__tmp: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.create(__list__(4, 256), dtype=pl.BFLOAT16)
                a_chunk_0__mid: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(a_chunk_0__tmp, a_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                a_chunk_0: pl.Tensor[[4, 256], pl.BFLOAT16] = pl.tensor.assemble(a_chunk_0__mid, a_chunk_0__h1, __list__(2, 0))
                pl.comm.tfree_to_aiv(1)
                w_chunk_0__h0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
                w_chunk_0__h1: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
                w_chunk_0__tmp: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.create(__list__(256, 64), dtype=pl.BFLOAT16)
                w_chunk_0__mid: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(w_chunk_0__tmp, w_chunk_0__h0, __list__(0, 0))
                pl.comm.tfree_to_aiv(0)
                w_chunk_0: pl.Tensor[[256, 64], pl.BFLOAT16] = pl.tensor.assemble(w_chunk_0__mid, w_chunk_0__h1, __list__(128, 0))
                pl.comm.tfree_to_aiv(1)
                _t42: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(a_chunk_0, w_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
                __half0__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t42, __list__(2, 64), __list__(0, 0))
                __half1__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t42, __list__(2, 64), __list__(2, 0))
                pl.comm.tpush_to_aiv(__half0__, 0)
                pl.comm.tpush_to_aiv(__half1__, 1)
                k0_18: pl.Scalar[pl.INDEX] = pl.yield_(k0_19)
            k0_iter_15_outer_l1_rv, kb_iter_11_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv = pl.yield_(k0_18, kb_13)
        return k0_iter_15_outer_l1_rv, kb_iter_11_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_3_aiv(self, attn_out_2: pl.Tensor[[16, 5120], pl.FP32], b0_1: pl.Scalar[pl.INDEX], hidden_states_0: pl.Tensor[[16, 5120], pl.BFLOAT16], k0_2: pl.Scalar[pl.INDEX], k0_iter_13: pl.Scalar[pl.INDEX], k0_iter_15_outer_l0: pl.Scalar[pl.INDEX], kb_2: pl.Scalar[pl.INDEX], kb_iter_11_outer_l0: pl.Scalar[pl.INDEX], kb_iter_9: pl.Scalar[pl.INDEX], ob_4_out: pl.Scalar[pl.INDEX], resid1_tile_0: pl.Tensor[[4, 5120], pl.FP32], resid1_tile_iter_1_outer_l0: pl.Tensor[[4, 5120], pl.FP32], wo_0: pl.Tensor[[5120, 5120], pl.BFLOAT16], AIV_IDX: pl.Scalar[pl.INDEX]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[4, 5120], pl.FP32]]:
        pl.comm.aiv_initialize_pipe()
        for ob_4_in, (k0_iter_15_outer_l1, kb_iter_11_outer_l1, resid1_tile_iter_1_outer_l1) in pl.parallel(0, 8, 1, init_values=(k0_iter_15_outer_l0, kb_iter_11_outer_l0, resid1_tile_iter_1_outer_l0)):
            o0_0: pl.Scalar[pl.INDEX] = (0 + (ob_4_out * 8 + ob_4_in) * 1) * 64
            o_acc_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.create([2, 64], dtype=pl.FP32)
            o_acc_1: pl.Tensor[[4, 64], pl.FP32] = pl.tensor.mul(o_acc_0, 0.0)
            for kb_13, (k0_iter_17, o_acc_iter_2) in pl.range(0, 20, 1, init_values=(k0_iter_15_outer_l1, o_acc_1)):
                k0_19: pl.Scalar[pl.INDEX] = kb_13 * 256
                _t41: pl.Tensor[[2, 256], pl.FP32] = pl.tensor.view(attn_out_2, [2, 256], [b0_1 + AIV_IDX * 2, k0_19])
                a_chunk_0: pl.Tensor[[2, 256], pl.BFLOAT16] = pl.tensor.cast(_t41, target_type=pl.BFLOAT16, mode=2)
                pl.comm.tpush_to_aic(a_chunk_0, AIV_IDX)
                w_chunk_0: pl.Tensor[[128, 64], pl.BFLOAT16] = pl.tensor.view(wo_0, [128, 64], [k0_19 + AIV_IDX * 128, o0_0])
                pl.comm.tpush_to_aic(w_chunk_0, AIV_IDX)
                _t42: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
                o_acc_4: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(o_acc_iter_2, _t42)
                pl.comm.tfree_to_aic(AIV_IDX)
                k0_18, o_acc_3 = pl.yield_(k0_19, o_acc_4)
            _t43: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(hidden_states_0, [2, 64], [b0_1 + AIV_IDX * 2, o0_0])
            resid_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.cast(_t43, target_type=pl.FP32, mode=2)
            _t44: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(o_acc_3, resid_0)
            resid1_tile_3: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(resid1_tile_iter_1_outer_l1, _t44, [0 + AIV_IDX * 2, o0_0])
            k0_iter_15_outer_l1_rv, kb_iter_11_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv = pl.yield_(k0_18, kb_13, resid1_tile_3)
        return k0_iter_15_outer_l1_rv, kb_iter_11_outer_l1_rv, resid1_tile_iter_1_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_3_aic", aiv="qwen3_decode_layer_incore_3_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_3_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_3_group, attn_out_2, b0_1, hidden_states_0, k0_2, k0_iter_13, k0_iter_15_outer_l0, kb_2, kb_iter_11_outer_l0, kb_iter_9, ob_4_out, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
            → qwen3_decode_layer_incore_3_aic(attn_out_2, b0_1, hidden_states_0, k0_2, k0_iter_13, k0_iter_15_outer_l0, kb_2, kb_iter_11_outer_l0, kb_iter_9, ob_4_out, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0)
            → qwen3_decode_layer_incore_3_aiv(attn_out_2, b0_1, hidden_states_0, k0_2, k0_iter_13, k0_iter_15_outer_l0, kb_2, kb_iter_11_outer_l0, kb_iter_9, ob_4_out, resid1_tile_0, resid1_tile_iter_1_outer_l0, wo_0, AIV_IDX=<runtime>)
        """
        pass

    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_4_aic(self, dob_0_out: pl.Scalar[pl.INDEX], down_proj_tile_1: pl.Tensor[[4, 5120], pl.FP32], down_proj_tile_iter_2: pl.Tensor[[4, 5120], pl.FP32], down_proj_tile_iter_4_outer_l0: pl.Tensor[[4, 5120], pl.FP32], mlp_chunk_bf16_0: pl.Tensor[[4, 32], pl.BFLOAT16], o0_3: pl.Scalar[pl.INDEX], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16]) -> pl.Tensor[[4, 5120], pl.FP32]:
        pl.comm.aic_initialize_pipe()
        for dob_0_in, (down_proj_tile_iter_4_outer_l1,) in pl.parallel(0, 4, 1, init_values=(down_proj_tile_iter_4_outer_l0,)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 4 + dob_0_in) * 1) * 64
            w_down_chunk_0__h0: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(0)
            w_down_chunk_0__h1: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.comm.tpop_from_aiv(1)
            w_down_chunk_0__tmp: pl.Tensor[[32, 64], pl.BFLOAT16] = pl.tensor.create(__list__(32, 64), dtype=pl.BFLOAT16)
            w_down_chunk_0__mid: pl.Tensor[[32, 64], pl.BFLOAT16] = pl.tensor.assemble(w_down_chunk_0__tmp, w_down_chunk_0__h0, __list__(0, 0))
            pl.comm.tfree_to_aiv(0)
            w_down_chunk_0: pl.Tensor[[32, 64], pl.BFLOAT16] = pl.tensor.assemble(w_down_chunk_0__mid, w_down_chunk_0__h1, __list__(16, 0))
            pl.comm.tfree_to_aiv(1)
            _t57: pl.Tensor[[4, 64], pl.BFLOAT16] = pl.tensor.matmul(mlp_chunk_bf16_0, w_down_chunk_0, a_trans=False, b_trans=False, c_matrix_nz=False)
            __half0__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t57, __list__(2, 64), __list__(0, 0))
            __half1__: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.tensor.view(_t57, __list__(2, 64), __list__(2, 0))
            pl.comm.tpush_to_aiv(__half0__, 0)
            pl.comm.tpush_to_aiv(__half1__, 1)
        return down_proj_tile_iter_4_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def qwen3_decode_layer_incore_4_aiv(self, dob_0_out: pl.Scalar[pl.INDEX], down_proj_tile_1: pl.Tensor[[4, 5120], pl.FP32], down_proj_tile_iter_2: pl.Tensor[[4, 5120], pl.FP32], down_proj_tile_iter_4_outer_l0: pl.Tensor[[4, 5120], pl.FP32], mlp_chunk_bf16_0: pl.Tensor[[4, 32], pl.BFLOAT16], o0_3: pl.Scalar[pl.INDEX], w_down_0: pl.Tensor[[25600, 5120], pl.BFLOAT16], AIV_IDX: pl.Scalar[pl.INDEX]) -> pl.Tensor[[4, 5120], pl.FP32]:
        pl.comm.aiv_initialize_pipe()
        for dob_0_in, (down_proj_tile_iter_4_outer_l1,) in pl.parallel(0, 4, 1, init_values=(down_proj_tile_iter_4_outer_l0,)):
            d0_0: pl.Scalar[pl.INDEX] = (0 + (dob_0_out * 4 + dob_0_in) * 1) * 64
            down_prev_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.view(down_proj_tile_iter_4_outer_l1, [2, 64], [0, d0_0])
            w_down_chunk_0: pl.Tensor[[16, 64], pl.BFLOAT16] = pl.tensor.view(w_down_0, [16, 64], [o0_3 + AIV_IDX * 16, d0_0])
            pl.comm.tpush_to_aic(w_down_chunk_0, AIV_IDX)
            _t57: pl.Tensor[[2, 64], pl.BFLOAT16] = pl.comm.tpop_from_aic(AIV_IDX)
            down_next_0: pl.Tensor[[2, 64], pl.FP32] = pl.tensor.add(down_prev_0, _t57)
            pl.comm.tfree_to_aic(AIV_IDX)
            down_proj_tile_6: pl.Tensor[[4, 5120], pl.FP32] = pl.tensor.assemble(down_proj_tile_iter_4_outer_l1, down_next_0, [0 + AIV_IDX * 2, d0_0])
            down_proj_tile_iter_4_outer_l1_rv: pl.Tensor[[4, 5120], pl.FP32] = pl.yield_(down_proj_tile_6)
        return down_proj_tile_iter_4_outer_l1_rv
    @pl.function_group(aic="qwen3_decode_layer_incore_4_aic", aiv="qwen3_decode_layer_incore_4_aiv", aiv_runtime_params=["AIV_IDX"])
    class qwen3_decode_layer_incore_4_group:
        """Parameter passing:
          call_group(qwen3_decode_layer_incore_4_group, dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
            → qwen3_decode_layer_incore_4_aic(dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0)
            → qwen3_decode_layer_incore_4_aiv(dob_0_out, down_proj_tile_1, down_proj_tile_iter_2, down_proj_tile_iter_4_outer_l0, mlp_chunk_bf16_0, o0_3, w_down_0, AIV_IDX=<runtime>)
        """
        pass
