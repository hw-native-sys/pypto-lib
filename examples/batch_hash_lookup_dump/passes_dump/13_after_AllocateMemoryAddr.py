# pypto.program: BatchHashLookup
import pypto.language as pl

@pl.program
class BatchHashLookup:
    @pl.function(type=pl.FunctionType.InCore)
    def batch_hash_lookup_incore_0(self, b_0: pl.Scalar[pl.INDEX], search_key_0: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 0)], value_ptr_out_0: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 1)], value_ptr_out_iter_1: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 2)]) -> pl.Tensor[[1024, 64, 32], pl.INT32]:
        for ti_0, (value_ptr_out_iter_3,) in pl.parallel(0, 64, 32, init_values=(value_ptr_out_iter_1,)):
            zero_src_0: pl.Tensor[[1, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 3)] = pl.tensor.view(search_key_0, [1, 32], [b_0, ti_0, 0])
            zero_tile_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 4)] = pl.tensor.mul(zero_src_0, 0)
            value_ptr_out_5: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 5)] = pl.tensor.assemble(value_ptr_out_iter_3, zero_tile_0, [b_0, ti_0, 0])
            value_ptr_out_4: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 6)] = pl.yield_(value_ptr_out_5)
        return value_ptr_out_4
    @pl.function(type=pl.FunctionType.InCore)
    def batch_hash_lookup_incore_1(self, hash_pool_0: pl.Tensor[[64, 128, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 1048576, 0)], probe_0: pl.Scalar[pl.INDEX], round_has_active_0: pl.Scalar[pl.INDEX], search_key_0: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 1)], ti_0: pl.Scalar[pl.INDEX], ti_iter_1: pl.Scalar[pl.INDEX], value_ptr_out_2: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 2)], value_ptr_out_iter_6: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 3)]) -> tuple[pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[1024, 64, 32], pl.INT32]]:
        for b_3, (round_has_active_iter_1, ti_iter_3, value_ptr_out_iter_8) in pl.parallel(0, 1024, 32, init_values=(round_has_active_0, ti_iter_1, value_ptr_out_iter_6)):
            for ti_5, (round_has_active_iter_3, value_ptr_out_iter_10) in pl.parallel(0, 64, 32, init_values=(round_has_active_iter_1, value_ptr_out_iter_8)):
                keys_tile_0: pl.Tensor[[1, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 4)] = pl.tensor.view(search_key_0, [1, 32], [b_3, ti_5, 0])
                mixed_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 5)] = pl.tensor.mul(keys_tile_0, 2654435761)
                _t0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 6)] = pl.tensor.add(mixed_0, probe_0 * 2246822519)
                h_probe_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 7)] = pl.tensor.ands(_t0, 64 - 1)
                cand_key_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 8)] = pl.tensor.mul(keys_tile_0, 0)
                cand_val_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 9)] = pl.tensor.mul(keys_tile_0, 0)
                for bucket_0, (cand_key_iter_1, cand_val_iter_1) in pl.range(0, 64, 1, init_values=(cand_key_0, cand_val_0)):
                    bucket_mask_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 10)] = pl.tensor.cmps(h_probe_0, bucket_0, cmp_type=0)
                    bucket_keys_0: pl.Tensor[[1, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 11)] = pl.tensor.view(hash_pool_0, [1, 32], [ti_5, bucket_0, 0])
                    bucket_vals_0: pl.Tensor[[1, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 12)] = pl.tensor.view(hash_pool_0, [1, 32], [ti_5, 64 + bucket_0, 0])
                    cand_key_3: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 13)] = pl.tensor.sel(bucket_mask_0, bucket_keys_0, cand_key_iter_1)
                    cand_val_3: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 14)] = pl.tensor.sel(bucket_mask_0, bucket_vals_0, cand_val_iter_1)
                    cand_key_2, cand_val_2 = pl.yield_(cand_key_3, cand_val_3)
                result_prev_0: pl.Tensor[[1, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 128, 17)] = pl.tensor.view(value_ptr_out_iter_10, [1, 32], [b_3, ti_5, 0])
                active_mask_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 18)] = pl.tensor.cmps(result_prev_0, 0, cmp_type=0)
                active_count_0: pl.Tensor[[1, 1], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 8, 19)] = pl.tensor.row_sum(active_mask_0)
                active_count_s_0: pl.Scalar[pl.INDEX] = pl.tensor.read(active_count_0, [0, 0])
                if active_count_s_0 != 0:
                    round_has_active_5: pl.Scalar[pl.INDEX] = 1
                    round_has_active_6: pl.Scalar[pl.INDEX] = pl.yield_(round_has_active_5)
                else:
                    round_has_active_6: pl.Scalar[pl.INDEX] = pl.yield_(round_has_active_iter_3)
                key_match_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 20)] = pl.tensor.cmp(cand_key_2, keys_tile_0, cmp_type=0)
                hit_mask_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 21)] = pl.tensor.and(active_mask_0, key_match_0)
                result_next_0: pl.Tensor[[1, 32], pl.INDEX, pl.MemRef(pl.MemorySpace.DDR, -1, 256, 22)] = pl.tensor.sel(hit_mask_0, cand_val_2, result_prev_0)
                value_ptr_out_12: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 23)] = pl.tensor.assemble(value_ptr_out_iter_10, result_next_0, [b_3, ti_5, 0])
                round_has_active_4, value_ptr_out_11 = pl.yield_(round_has_active_6, value_ptr_out_12)
            round_has_active_2, ti_4, value_ptr_out_9 = pl.yield_(round_has_active_4, ti_5, value_ptr_out_11)
        return b_3, round_has_active_2, ti_4, value_ptr_out_9
    @pl.function(type=pl.FunctionType.Orchestration)
    def batch_hash_lookup(self, search_key_0: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 0)], hash_table_size_0: pl.Tensor[[64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 1)], hash_base_ptr_0: pl.Tensor[[64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8192, 2)], hash_pool_0: pl.Tensor[[64, 128, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 1048576, 3)], value_ptr_out_0: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 4)]) -> pl.Tensor[[1024, 64, 32], pl.INT32]:
        for b_0, (value_ptr_out_iter_1,) in pl.parallel(0, 1024, 32, init_values=(value_ptr_out_0,)):
            value_ptr_out_4: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 5)] = self.batch_hash_lookup_incore_0(b_0, search_key_0, value_ptr_out_0, value_ptr_out_iter_1)
            value_ptr_out_2: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 6)] = pl.yield_(value_ptr_out_4)
        for probe_0, (b_iter_1, ti_iter_1, value_ptr_out_iter_6) in pl.range(0, 8, 1, init_values=(b_0, ti_0, value_ptr_out_2)):
            round_has_active_0: pl.Scalar[pl.INDEX] = 0
            ret: pl.Tuple([pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Scalar[pl.INDEX], pl.Tensor[[1024, 64, 32], pl.INT32]]) = self.batch_hash_lookup_incore_1(hash_pool_0, probe_0, round_has_active_0, search_key_0, ti_0, ti_iter_1, value_ptr_out_2, value_ptr_out_iter_6)
            b_3: pl.Scalar[pl.INDEX] = ret[0]
            round_has_active_2: pl.Scalar[pl.INDEX] = ret[1]
            ti_4: pl.Scalar[pl.INDEX] = ret[2]
            value_ptr_out_9: pl.Tensor[[1024, 64, 32], pl.INT32, pl.MemRef(pl.MemorySpace.DDR, -1, 8388608, 7)] = ret[3]
            if round_has_active_2 == 0:
                break
            b_2, ti_2, value_ptr_out_7 = pl.yield_(b_3, ti_4, value_ptr_out_9)
        return value_ptr_out_7