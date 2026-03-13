# pypto.program: PagedAttentionProgram
import pypto.language as pl

@pl.program
class PagedAttentionProgram:
    @pl.function
    def paged_attention(self, query: pl.Tensor[[4096, 128], pl.BFLOAT16], key_cache: pl.Tensor[[2097152, 128], pl.BFLOAT16], value_cache: pl.Tensor[[2097152, 128], pl.BFLOAT16], block_table: pl.Tensor[[16384], pl.INT32], context_lens: pl.Tensor[[64], pl.INT32], out: pl.Tensor[[4096, 128], pl.FP32], config: pl.Tensor[[7], pl.INT64], size_query: pl.Tensor[[1], pl.INT64], size_key_cache: pl.Tensor[[1], pl.INT64], size_value_cache: pl.Tensor[[1], pl.INT64]) -> pl.Tensor[[4096, 128], pl.FP32]:
        for b_idx in pl.parallel(0, 64, 1, chunk=8):
            for q_idx in pl.parallel(0, 4, 1, chunk=2):
                cur_seq: pl.Scalar[pl.INT32] = pl.tensor.read(context_lens, [b_idx])
                bn_this_batch: pl.Scalar[pl.INDEX] = (pl.cast(cur_seq, pl.INDEX) + 128 - 1) // 128
                cur_offset: pl.Scalar[pl.INDEX] = b_idx * 64 + q_idx * 16
                oi: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                li_update: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                mi_update: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                zero_oi: pl.Tile[[16, 128], pl.FP32] = pl.block.full([16, 128], 0.0, dtype=pl.FP32)
                zero_li: pl.Tile[[16, 1], pl.FP32] = pl.block.full([16, 1], 0.0, dtype=pl.FP32)
                zero_mi: pl.Tile[[16, 1], pl.FP32] = pl.block.full([16, 1], 0.0, dtype=pl.FP32)
                pl.block.store(zero_oi, [0, 0], [16, 128], oi)
                pl.block.store(zero_li, [0, 0], [16, 1], li_update)
                pl.block.store(zero_mi, [0, 0], [16, 1], mi_update)
                for bn in pl.range(0, bn_this_batch, 1):
                    qi: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.view(query, [16, 128], [cur_offset, 0])
                    cur_block_idx: pl.Scalar[pl.INT32] = pl.tensor.read(block_table, [b_idx * 256 + bn])
                    valid_len: pl.Scalar[pl.INDEX] = min(128, pl.cast(cur_seq, pl.INDEX) - bn * 128)
                    kv_block_row: pl.Scalar[pl.INDEX] = pl.cast(cur_block_idx, pl.INDEX) * 128
                    kj: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(key_cache, [128, 128], [kv_block_row, 0])
                    vj: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(value_cache, [128, 128], [kv_block_row, 0])
                    sij: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                    qi_l1: pl.Tile[[16, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[16, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.load(qi, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Mat)
                    kj_l1: pl.Tile[[128, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[128, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.load(kj, [0, 0], [128, 128], [128, 128], target_memory=pl.MemorySpace.Mat)
                    qi_l0a: pl.Tile[[16, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.move(qi_l1, target_memory=pl.MemorySpace.Left, transpose=False)
                    kj_l0b: pl.Tile[[128, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.move(kj_l1, target_memory=pl.MemorySpace.Right, transpose=True)
                    sij_l0c: pl.Tile[[16, 128], pl.FP32, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024, pad=pl.TilePad.null)] = pl.block.matmul(qi_l0a, kj_l0b)
                    pl.block.store(sij_l0c, [0, 0], [16, 128], sij)
                    sij_valid: pl.Tensor[[16, valid_len], pl.FP32] = pl.tensor.view(sij, [16, valid_len], [0, 0])
                    pij_f16: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.create([16, 128], dtype=pl.BFLOAT16)
                    mi: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                    li: pl.Tensor[[16, 1], pl.FP32] = pl.tensor.create([16, 1], dtype=pl.FP32)
                    scale: pl.Scalar[pl.FP32] = 1.0
                    s_tile: pl.Tile[[16, valid_len], pl.FP32, tile_view=pl.TileView(valid_shape=[16, valid_len], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(sij_valid, [0, 0], [16, valid_len], [16, valid_len], target_memory=pl.MemorySpace.Vec)
                    scaled: pl.Tile[[16, valid_len], pl.FP32] = pl.block.muls(s_tile, scale)
                    tmp_tile: pl.Tile[[16, 128], pl.FP32] = pl.block.create_tile([16, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    mi_tile: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.row_max(scaled, tmp_tile)
                    sij_centered: pl.Tile[[16, valid_len], pl.FP32] = pl.block.row_expand_sub(scaled, mi_tile)
                    exp_tile: pl.Tile[[16, valid_len], pl.FP32] = pl.block.exp(sij_centered)
                    pij_tile_bf16: pl.Tile[[16, valid_len], pl.BFLOAT16] = pl.block.cast(exp_tile, target_type=pl.BFLOAT16, mode=2)
                    pij_tile: pl.Tile[[16, valid_len], pl.FP32] = pl.block.cast(pij_tile_bf16, target_type=pl.FP32, mode=2)
                    li_tile: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.row_sum(pij_tile, tmp_tile)
                    pl.block.store(pij_tile_bf16, [0, 0], [16, valid_len], pij_f16)
                    pl.block.store(mi_tile, [0, 0], [16, 1], mi)
                    pl.block.store(li_tile, [0, 0], [16, 1], li)
                    oi_tmp: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.create([16, 128], dtype=pl.FP32)
                    pij_l1: pl.Tile[[16, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[16, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.load(pij_f16, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Mat)
                    vj_l1: pl.Tile[[128, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[128, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.load(vj, [0, 0], [128, 128], [128, 128], target_memory=pl.MemorySpace.Mat)
                    pij_l0a: pl.Tile[[16, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=512, pad=pl.TilePad.null)] = pl.block.move(pij_l1, target_memory=pl.MemorySpace.Left, transpose=False)
                    vj_l0b: pl.Tile[[128, 128], pl.BFLOAT16, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.col_major, fractal=512, pad=pl.TilePad.null)] = pl.block.move(vj_l1, target_memory=pl.MemorySpace.Right, transpose=False)
                    oi_l0c: pl.Tile[[16, 128], pl.FP32, tile_view=pl.TileView(valid_shape=[], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.row_major, fractal=1024, pad=pl.TilePad.null)] = pl.block.matmul(pij_l0a, vj_l0b)
                    pl.block.store(oi_l0c, [0, 0], [16, 128], oi_tmp)
                    if bn == 0:
                        is_first: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_first: pl.Scalar[pl.INT64] = pl.yield_(0)
                    if bn == bn_this_batch - 1:
                        is_last: pl.Scalar[pl.INT64] = pl.yield_(1)
                    else:
                        is_last: pl.Scalar[pl.INT64] = pl.yield_(0)
                    out_view: pl.Tensor[[16, 128], pl.FP32] = pl.tensor.view(out, [16, 128], [cur_offset, 0])
                    mij_tile: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(mi, [0, 0], [16, 1], [16, 1], target_memory=pl.MemorySpace.Vec)
                    lij_tile: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(li, [0, 0], [16, 1], [16, 1], target_memory=pl.MemorySpace.Vec)
                    oi_new_tile: pl.Tile[[16, 128], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(oi_tmp, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Vec)
                    mi_tile: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(mi_update, [0, 0], [16, 1], [16, 1], target_memory=pl.MemorySpace.Vec)
                    li_tile: pl.Tile[[16, 1], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 1], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.col_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(li_update, [0, 0], [16, 1], [16, 1], target_memory=pl.MemorySpace.Vec)
                    oi_tile: pl.Tile[[16, 128], pl.FP32, tile_view=pl.TileView(valid_shape=[16, 128], stride=[], start_offset=<unsupported IRNode type>, blayout=pl.TileLayout.row_major, slayout=pl.TileLayout.none_box, fractal=512, pad=pl.TilePad.null)] = pl.block.load(oi, [0, 0], [16, 128], [16, 128], target_memory=pl.MemorySpace.Vec)
                    if is_first:
                        pl.block.store(mij_tile, [0, 0], [16, 1], mi_update)
                        pl.block.store(lij_tile, [0, 0], [16, 1], li_update)
                        pl.block.store(oi_new_tile, [0, 0], [16, 128], oi)
                        if is_last:
                            dst_tile: pl.Tile[[16, 128], pl.FP32] = pl.block.row_expand_div(oi_new_tile, lij_tile)
                            pl.block.store(dst_tile, [0, 0], [16, 128], out_view)
                        else:
                            zero_tile: pl.Tile[[16, 128], pl.FP32] = pl.block.full([16, 128], 0.0, dtype=pl.FP32)
                            pl.block.store(zero_tile, [0, 0], [16, 128], out_view)
                    else:
                        mi_tile_nd: pl.Tile[[1, 16], pl.FP32] = pl.block.reshape(mi_tile, [1, 16])
                        mij_tile_nd: pl.Tile[[1, 16], pl.FP32] = pl.block.reshape(mij_tile, [1, 16])
                        li_tile_nd: pl.Tile[[1, 16], pl.FP32] = pl.block.reshape(li_tile, [1, 16])
                        lij_tile_nd: pl.Tile[[1, 16], pl.FP32] = pl.block.reshape(lij_tile, [1, 16])
                        mi_new: pl.Tile[[1, 16], pl.FP32] = pl.block.maximum(mi_tile_nd, mij_tile_nd)
                        mi_diff: pl.Tile[[1, 16], pl.FP32] = pl.block.sub(mi_tile_nd, mi_new)
                        alpha: pl.Tile[[1, 16], pl.FP32] = pl.block.exp(mi_diff)
                        mij_diff: pl.Tile[[1, 16], pl.FP32] = pl.block.sub(mij_tile_nd, mi_new)
                        beta: pl.Tile[[1, 16], pl.FP32] = pl.block.exp(mij_diff)
                        li_scaled: pl.Tile[[1, 16], pl.FP32] = pl.block.mul(alpha, li_tile_nd)
                        lij_scaled: pl.Tile[[1, 16], pl.FP32] = pl.block.mul(beta, lij_tile_nd)
                        li_updated: pl.Tile[[1, 16], pl.FP32] = pl.block.add(li_scaled, lij_scaled)
                        alpha_dn: pl.Tile[[16, 1], pl.FP32] = pl.block.reshape(alpha, [16, 1])
                        oi_scaled: pl.Tile[[16, 128], pl.FP32] = pl.block.row_expand_mul(oi_tile, alpha_dn)
                        beta_dn: pl.Tile[[16, 1], pl.FP32] = pl.block.reshape(beta, [16, 1])
                        oi_new_scaled: pl.Tile[[16, 128], pl.FP32] = pl.block.row_expand_mul(oi_new_tile, beta_dn)
                        oi_updated: pl.Tile[[16, 128], pl.FP32] = pl.block.add(oi_scaled, oi_new_scaled)
                        mi_new_dn: pl.Tile[[16, 1], pl.FP32] = pl.block.reshape(mi_new, [16, 1])
                        li_updated_dn: pl.Tile[[16, 1], pl.FP32] = pl.block.reshape(li_updated, [16, 1])
                        pl.block.store(mi_new_dn, [0, 0], [16, 1], mi_update)
                        pl.block.store(li_updated_dn, [0, 0], [16, 1], li_update)
                        if is_last:
                            dst_tile: pl.Tile[[16, 128], pl.FP32] = pl.block.row_expand_div(oi_updated, li_updated_dn)
                            pl.block.store(dst_tile, [0, 0], [16, 128], out_view)
                            pl.block.store(oi_updated, [0, 0], [16, 128], oi)
                        else:
                            zero_tile: pl.Tile[[16, 128], pl.FP32] = pl.block.full([16, 128], 0.0, dtype=pl.FP32)
                            pl.block.store(zero_tile, [0, 0], [16, 128], out_view)
                            pl.block.store(oi_updated, [0, 0], [16, 128], oi)
        return out