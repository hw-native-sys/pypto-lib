# pypto.program: BatchHashLookup
import pypto.language as pl

@pl.program
class BatchHashLookup:
    @pl.function
    def batch_hash_lookup(self, search_key: pl.Tensor[[1024, 64, 32], pl.INT32], hash_table_size: pl.Tensor[[64, 32], pl.INT32], hash_base_ptr: pl.Tensor[[64, 32], pl.INT32], hash_pool: pl.Tensor[[64, 128, 32], pl.INT32], value_ptr_out: pl.Tensor[[1024, 64, 32], pl.INT32]) -> pl.Tensor[[1024, 64, 32], pl.INT32]:
        for b in pl.parallel(0, 1024, 32):
            with pl.incore():
                for ti in pl.parallel(0, 64, 32):
                    zero_src: pl.Tensor[[1, 32], pl.INT32] = pl.tensor.view(search_key, [1, 32], [b, ti, 0])
                    zero_tile: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.mul(zero_src, 0)
                    value_ptr_out: pl.Tensor[[1024, 64, 32], pl.INT32] = pl.tensor.assemble(value_ptr_out, zero_tile, [b, ti, 0])
        for probe in pl.range(0, 8, 1):
            round_has_active: pl.Scalar[pl.INDEX] = 0
            with pl.incore():
                for b in pl.parallel(0, 1024, 32):
                    for ti in pl.parallel(0, 64, 32):
                        keys_tile: pl.Tensor[[1, 32], pl.INT32] = pl.tensor.view(search_key, [1, 32], [b, ti, 0])
                        mixed: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.mul(keys_tile, 2654435761)
                        h_probe: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.ands(pl.tensor.add(mixed, probe * 2246822519), 64 - 1)
                        cand_key: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.mul(keys_tile, 0)
                        cand_val: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.mul(keys_tile, 0)
                        for bucket in pl.range(0, 64, 1):
                            bucket_mask: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.cmps(h_probe, bucket, cmp_type=0)
                            bucket_keys: pl.Tensor[[1, 32], pl.INT32] = pl.tensor.view(hash_pool, [1, 32], [ti, bucket, 0])
                            bucket_vals: pl.Tensor[[1, 32], pl.INT32] = pl.tensor.view(hash_pool, [1, 32], [ti, 64 + bucket, 0])
                            cand_key: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.sel(bucket_mask, bucket_keys, cand_key)
                            cand_val: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.sel(bucket_mask, bucket_vals, cand_val)
                        result_prev: pl.Tensor[[1, 32], pl.INT32] = pl.tensor.view(value_ptr_out, [1, 32], [b, ti, 0])
                        active_mask: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.cmps(result_prev, 0, cmp_type=0)
                        active_count: pl.Tensor[[1, 1], pl.INDEX] = pl.tensor.row_sum(active_mask)
                        active_count_s: pl.Scalar[pl.INDEX] = pl.tensor.read(active_count, [0, 0])
                        if active_count_s != 0:
                            round_has_active: pl.Scalar[pl.INDEX] = 1
                        key_match: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.cmp(cand_key, keys_tile, cmp_type=0)
                        hit_mask: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.and(active_mask, key_match)
                        result_next: pl.Tensor[[1, 32], pl.INDEX] = pl.tensor.sel(hit_mask, cand_val, result_prev)
                        value_ptr_out: pl.Tensor[[1024, 64, 32], pl.INT32] = pl.tensor.assemble(value_ptr_out, result_next, [b, ti, 0])
            if round_has_active == 0:
                break
        return value_ptr_out