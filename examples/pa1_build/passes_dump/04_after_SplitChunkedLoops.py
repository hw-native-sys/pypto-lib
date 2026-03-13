# pypto.program: PagedAttentionProgram
import pypto.language as pl

@pl.program
class PagedAttentionProgram:
    @pl.function
    def paged_attention(self, query_0: pl.Tensor[[4096, 128], pl.BFLOAT16], key_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16], value_cache_0: pl.Tensor[[2097152, 128], pl.BFLOAT16], block_table_0: pl.Tensor[[16384], pl.INT32], context_lens_0: pl.Tensor[[64], pl.INT32], out_0: pl.Tensor[[4096, 128], pl.FP32], config_0: pl.Tensor[[7], pl.INT64], size_query_0: pl.Tensor[[1], pl.INT64], size_key_cache_0: pl.Tensor[[1], pl.INT64], size_value_cache_0: pl.Tensor[[1], pl.INT64]) -> pl.Tensor[[4096, 128], pl.FP32]:
        for b_idx_0_out in pl.range(0, 8, 1):
            for b_idx_0_in in pl.parallel(0, 8, 1):
                for q_idx_0_out in pl.range(0, 2, 1):
                    for q_idx_0_in in pl.parallel(0, 2, 1):
                        cur_seq_0: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens_0, [0 + (b_idx_0_out * 8 + b_idx_0_in) * 1])
                        bn_this_batch_0: pl.Scalar[pl.INDEX] = (pl.cast(cur_seq_0, pl.INDEX) + 128 - 1) // 128
                        cur_offset_0: pl.Scalar[pl.INDEX] = (0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 64 + (0 + (q_idx_0_out * 2 + q_idx_0_in) * 1) * 16
                        oi_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                        li_update_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                        mi_update_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                        zero_oi_0: pl.Tile[[16, 128], pl.FP32] = pl.block.full([16, 128], 0.0, dtype=pl.FP32)
                        zero_li_0: pl.Tile[[16, 1], pl.FP32] = pl.block.full([16, 1], 0.0, dtype=pl.FP32)
                        zero_mi_0: pl.Tile[[16, 1], pl.FP32] = pl.block.full([16, 1], 0.0, dtype=pl.FP32)
                        pl.block.store(zero_oi_0, [0, 0], [16, 128], oi_0)
                        pl.block.store(zero_li_0, [0, 0], [16, 1], li_update_0)
                        pl.block.store(zero_mi_0, [0, 0], [16, 1], mi_update_0)
                        for bn_0 in pl.range(0, bn_this_batch_0, 1):
                            qi_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.view(query_0, [16, 128], [cur_offset_0, 0])
                            cur_block_idx_0: pl.Scalar[pl.INT32] = pl.tensor.read(block_table_0, [(0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 256 + bn_0])
                            valid_len_0: pl.Scalar[pl.INDEX] = min(128, pl.cast(cur_seq_0, pl.INDEX) - bn_0 * 128)
                            kv_block_row_0: pl.Scalar[pl.INDEX] = pl.cast(cur_block_idx_0, pl.INDEX) * 128
                            kj_0: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(key_cache_0, [128, 128], [kv_block_row_0, 0])
                            vj_0: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(value_cache_0, [128, 128], [kv_block_row_0, 0])
                            sij_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                            qi_l1_0: pl.Tile[[16, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[16, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.load(qi_0, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Mat)
                            kj_l1_0: pl.Tile[[128, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[128, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.load(kj_0, [0, 0], [128, 128], [128, 128], target_memory=pl.MemorySpace.Mat)
                            qi_l0a_0: pl.Tile[[16, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.move(qi_l1_0, target_memory=pl.MemorySpace.Left, transpose=False)
                            kj_l0b_0: pl.Tile[[128, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.move(kj_l1_0, target_memory=pl.MemorySpace.Right, transpose=True)
                            sij_l0c_0: pl.Tile[[16, 128], pl.FP32, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024, pad=pl.TilePad.null)] = pl.block.matmul(qi_l0a_0, kj_l0b_0)
                            pl.block.store(sij_l0c_0, [0, 0], [16, 128], sij_0)
                            sij_valid_0: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.view(sij_0, [16, valid_len_0], [0, 0])
                            pij_f16_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.create([16, 128], dtype=pl.BFLOAT16)
                            mi_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                            li_0: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                            scale_0: pl.Scalar[pl.FP32] = 1.0
                            s_tile_0: pl.Tile[[16, valid_len], pl.FP32, tile_view=pl.TileView(valid_shape=[16, valid_len_0], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(sij_valid_0, [0, 0], [16, valid_len_0], [16, valid_len_0], target_memory=pl.MemorySpace.Vec)
                            scaled_0: pl.Tile[[16, valid_len], pl.FP32] = pl.block.muls(s_tile_0, scale_0)
                            tmp_tile_0: pl.Tile[[16, 128], pl.FP32] = pl.block.create_tile([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                            mi_tile_0: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.row_max(scaled_0, tmp_tile_0)
                            sij_centered_0: pl.Tile[[16, valid_len], pl.FP32] = pl.block.row_expand_sub(scaled_0, mi_tile_0)
                            exp_tile_0: pl.Tile[[16, valid_len], pl.FP32] = pl.block.exp(sij_centered_0)
                            pij_tile_bf16_0: pl.Tile[[16, valid_len], pl.BFLOAT16] = pl.block.cast(exp_tile_0, target_type=pl.BFLOAT16, mode=2)
                            pij_tile_0: pl.Tile[[16, valid_len], pl.FP32] = pl.block.cast(pij_tile_bf16_0, target_type=pl.FP32, mode=2)
                            li_tile_0: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.row_sum(pij_tile_0, tmp_tile_0)
                            pl.block.store(pij_tile_bf16_0, [0, 0], [16, valid_len_0], pij_f16_0)
                            pl.block.store(mi_tile_0, [0, 0], [16, 1], mi_0)
                            pl.block.store(li_tile_0, [0, 0], [16, 1], li_0)
                            oi_tmp_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                            pij_l1_0: pl.Tile[[16, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[16, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.load(pij_f16_0, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Mat)
                            vj_l1_0: pl.Tile[[128, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[128, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.load(vj_0, [0, 0], [128, 128], [128, 128], target_memory=pl.MemorySpace.Mat)
                            pij_l0a_0: pl.Tile[[16, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.move(pij_l1_0, target_memory=pl.MemorySpace.Left, transpose=False)
                            vj_l0b_0: pl.Tile[[128, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major, fractal=512, pad=pl.TilePad.null)] = pl.block.move(vj_l1_0, target_memory=pl.MemorySpace.Right, transpose=False)
                            oi_l0c_0: pl.Tile[[16, 128], pl.FP32, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024, pad=pl.TilePad.null)] = pl.block.matmul(pij_l0a_0, vj_l0b_0)
                            pl.block.store(oi_l0c_0, [0, 0], [16, 128], oi_tmp_0)
                            if bn_0 == 0:
                                is_first_0: pl.Scalar[pl.INT64] = pl.yield_(1)
                            else:
                                is_first_0: pl.Scalar[pl.INT64] = pl.yield_(0)
                            if bn_0 == bn_this_batch_0 - 1:
                                is_last_0: pl.Scalar[pl.INT64] = pl.yield_(1)
                            else:
                                is_last_0: pl.Scalar[pl.INT64] = pl.yield_(0)
                            out_view_0: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.view(out_0, [16, 128], [cur_offset_0, 0])
                            mij_tile_0: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(mi_0, [0, 0], [16, 1], [16, 1], target_memory=pl.MemorySpace.Vec)
                            lij_tile_0: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(li_0, [0, 0], [16, 1], [16, 1], target_memory=pl.MemorySpace.Vec)
                            oi_new_tile_0: pl.Tile[[16, 128], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(oi_tmp_0, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Vec)
                            mi_tile_1: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(mi_update_0, [0, 0], [16, 1], [16, 1], target_memory=pl.MemorySpace.Vec)
                            li_tile_1: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(li_update_0, [0, 0], [16, 1], [16, 1], target_memory=pl.MemorySpace.Vec)
                            oi_tile_0: pl.Tile[[16, 128], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(oi_0, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Vec)
                            if is_first_0:
                                pl.block.store(mij_tile_0, [0, 0], [16, 1], mi_update_0)
                                pl.block.store(lij_tile_0, [0, 0], [16, 1], li_update_0)
                                pl.block.store(oi_new_tile_0, [0, 0], [16, 128], oi_0)
                                if is_last_0:
                                    dst_tile_0: pl.Tile[[16, 128], pl.FP32] = pl.block.row_expand_div(oi_new_tile_0, lij_tile_0)
                                    pl.block.store(dst_tile_0, [0, 0], [16, 128], out_view_0)
                                else:
                                    zero_tile_0: pl.Tile[[16, 128], pl.FP32] = pl.block.full([16, 128], 0.0, dtype=pl.FP32)
                                    pl.block.store(zero_tile_0, [0, 0], [16, 128], out_view_0)
                            else:
                                mi_tile_nd_0: pl.Tile[[1, 16], pl.FP32] = pl.block.reshape(mi_tile_1, [1, 16])
                                mij_tile_nd_0: pl.Tile[[1, 16], pl.FP32] = pl.block.reshape(mij_tile_0, [1, 16])
                                li_tile_nd_0: pl.Tile[[1, 16], pl.FP32] = pl.block.reshape(li_tile_1, [1, 16])
                                lij_tile_nd_0: pl.Tile[[1, 16], pl.FP32] = pl.block.reshape(lij_tile_0, [1, 16])
                                mi_new_0: pl.Tile[[1, 16], pl.FP32] = pl.block.maximum(mi_tile_nd_0, mij_tile_nd_0)
                                mi_diff_0: pl.Tile[[1, 16], pl.FP32] = pl.block.sub(mi_tile_nd_0, mi_new_0)
                                alpha_0: pl.Tile[[1, 16], pl.FP32] = pl.block.exp(mi_diff_0)
                                mij_diff_0: pl.Tile[[1, 16], pl.FP32] = pl.block.sub(mij_tile_nd_0, mi_new_0)
                                beta_0: pl.Tile[[1, 16], pl.FP32] = pl.block.exp(mij_diff_0)
                                li_scaled_0: pl.Tile[[1, 16], pl.FP32] = pl.block.mul(alpha_0, li_tile_nd_0)
                                lij_scaled_0: pl.Tile[[1, 16], pl.FP32] = pl.block.mul(beta_0, lij_tile_nd_0)
                                li_updated_0: pl.Tile[[1, 16], pl.FP32] = pl.block.add(li_scaled_0, lij_scaled_0)
                                alpha_dn_0: pl.Tile[[16, 1], pl.FP32] = pl.block.reshape(alpha_0, [16, 1])
                                oi_scaled_0: pl.Tile[[16, 128], pl.FP32] = pl.block.row_expand_mul(oi_tile_0, alpha_dn_0)
                                beta_dn_0: pl.Tile[[16, 1], pl.FP32] = pl.block.reshape(beta_0, [16, 1])
                                oi_new_scaled_0: pl.Tile[[16, 128], pl.FP32] = pl.block.row_expand_mul(oi_new_tile_0, beta_dn_0)
                                oi_updated_0: pl.Tile[[16, 128], pl.FP32] = pl.block.add(oi_scaled_0, oi_new_scaled_0)
                                mi_new_dn_0: pl.Tile[[16, 1], pl.FP32] = pl.block.reshape(mi_new_0, [16, 1])
                                li_updated_dn_0: pl.Tile[[16, 1], pl.FP32] = pl.block.reshape(li_updated_0, [16, 1])
                                pl.block.store(mi_new_dn_0, [0, 0], [16, 1], mi_update_0)
                                pl.block.store(li_updated_dn_0, [0, 0], [16, 1], li_update_0)
                                if is_last_0:
                                    dst_tile_1: pl.Tile[[16, 128], pl.FP32] = pl.block.row_expand_div(oi_updated_0, li_updated_dn_0)
                                    pl.block.store(dst_tile_1, [0, 0], [16, 128], out_view_0)
                                    pl.block.store(oi_updated_0, [0, 0], [16, 128], oi_0)
                                else:
                                    zero_tile_1: pl.Tile[[16, 128], pl.FP32] = pl.block.full([16, 128], 0.0, dtype=pl.FP32)
                                    pl.block.store(zero_tile_1, [0, 0], [16, 128], out_view_0)
                                    pl.block.store(oi_updated_0, [0, 0], [16, 128], oi_0)
        return out_0