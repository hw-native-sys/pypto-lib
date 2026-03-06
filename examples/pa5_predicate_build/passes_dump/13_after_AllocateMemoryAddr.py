# pypto.program: PredicateTestProgram
import pypto.language as pl

@pl.program
class PredicateTestProgram:
    @pl.function(type=pl.FunctionType.Orchestration)
    def predicate_kernel(self, query_0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 0)], key_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 1)], value_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 2)], out_0: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 3)]) -> pl.Tensor[[64, 128], pl.FP32]:
        for b_idx_0_out, (out_iter_1_outer_l0,) in pl.range(0, 8, 1, init_values=(out_0,)):
            out_iter_1_outer_l1_rv: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 4)] = self.call_group(predicate_kernel_incore_0_group, b_idx_0_out, key_0, out_0, out_iter_1_outer_l0, query_0, value_0)
            out_iter_1_outer_l0_rv: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 5)] = pl.yield_(out_iter_1_outer_l1_rv)
        return out_iter_1_outer_l0_rv
    @pl.function(type=pl.FunctionType.InCore)
    def predicate_kernel_incore_0_aic(self, b_idx_0_out: pl.Scalar[pl.INDEX], key_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 0)], out_0: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 1)], out_iter_1_outer_l0: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 2)], query_0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 3)], value_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 4)]) -> pl.Tensor[[64, 128], pl.FP32]:
        pl.comm.aic_initialize_pipe()
        for b_idx_0_in, (out_iter_1_outer_l1,) in pl.parallel(0, 8, 1, init_values=(out_iter_1_outer_l0,)):
            cur_offset_0: pl.Scalar[pl.INDEX] = (0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 16
            qi_0__h0: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 5)] = pl.comm.tpop_from_aiv(0)
            qi_0__h1: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 6)] = pl.comm.tpop_from_aiv(1)
            qi_0__tmp: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 7)] = pl.tensor.create(__list__(16, 128), dtype=pl.BFLOAT16)
            qi_0__mid: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 8)] = pl.tensor.assemble(qi_0__tmp, qi_0__h0, __list__(0, 0))
            qi_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 9)] = pl.tensor.assemble(qi_0__mid, qi_0__h1, __list__(8, 0))
            kj_0__h0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 10)] = pl.comm.tpop_from_aiv(0)
            kj_0__h1: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 11)] = pl.comm.tpop_from_aiv(1)
            kj_0__tmp: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 12)] = pl.tensor.create(__list__(128, 128), dtype=pl.BFLOAT16)
            kj_0__mid: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 13)] = pl.tensor.assemble(kj_0__tmp, kj_0__h0, __list__(0, 0))
            kj_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 14)] = pl.tensor.assemble(kj_0__mid, kj_0__h1, __list__(64, 0))
            vj_0__h0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 15)] = pl.comm.tpop_from_aiv(0)
            vj_0__h1: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 16)] = pl.comm.tpop_from_aiv(1)
            vj_0__tmp: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 17)] = pl.tensor.create(__list__(128, 128), dtype=pl.BFLOAT16)
            vj_0__mid: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 18)] = pl.tensor.assemble(vj_0__tmp, vj_0__h0, __list__(0, 0))
            vj_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 19)] = pl.tensor.assemble(vj_0__mid, vj_0__h1, __list__(64, 0))
            sij_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 20)] = pl.tensor.matmul(qi_0, kj_0, a_trans=False, b_trans=True, c_matrix_nz=False)
            pl.comm.tpush_to_aiv(sij_0, 0)
            pl.comm.tpush_to_aiv(sij_0, 1)
            pij_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 21)] = pl.comm.tpop_from_aiv(0)
            pij_0__discard: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 22)] = pl.comm.tpop_from_aiv(1)
            oi_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 23)] = pl.tensor.matmul(pij_0, vj_0, a_trans=False, b_trans=False, c_matrix_nz=False)
            __half0__: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 24)] = pl.tensor.view(oi_0, __list__(8, 128), __list__(0, 0))
            __half1__: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 25)] = pl.tensor.view(oi_0, __list__(8, 128), __list__(8, 0))
            pl.comm.tpush_to_aiv(__half0__, 0)
            pl.comm.tpush_to_aiv(__half1__, 1)
        return out_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.InCore)
    def predicate_kernel_incore_0_aiv(self, b_idx_0_out: pl.Scalar[pl.INDEX], key_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 0)], out_0: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 1)], out_iter_1_outer_l0: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 2)], query_0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 3)], value_0: pl.Tensor[[128, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 4)], AIV_IDX: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64, 128], pl.FP32]:
        pl.comm.aiv_initialize_pipe()
        for b_idx_0_in, (out_iter_1_outer_l1,) in pl.parallel(0, 8, 1, init_values=(out_iter_1_outer_l0,)):
            cur_offset_0: pl.Scalar[pl.INDEX] = (0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 16
            qi_0: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 5)] = pl.tensor.view(query_0, [8, 128], [cur_offset_0 + AIV_IDX * 8, 0])
            pl.comm.tpush_to_aic(qi_0, AIV_IDX)
            kj_0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 7)] = pl.tensor.view(key_0, [64, 128], [0 + AIV_IDX * 64, 0])
            pl.comm.tpush_to_aic(kj_0, AIV_IDX)
            vj_0: pl.Tensor[[64, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 16384, 9)] = pl.tensor.view(value_0, [64, 128], [0 + AIV_IDX * 64, 0])
            pl.comm.tpush_to_aic(vj_0, AIV_IDX)
            sij_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 11)] = pl.comm.tpop_from_aic()
            mi_0: pl.Tensor[[16, 1], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32, 12)] = pl.tensor.row_max(sij_0)
            mi_flat_0: pl.Tensor[[1, 16], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 32, 13)] = pl.tensor.reshape(mi_0, [1, 16])
            global_max_0: pl.Tensor[[1, 1], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2, 14)] = pl.tensor.row_max(mi_flat_0)
            centered_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 15)] = pl.tensor.sub(sij_0, global_max_0)
            exp_vals_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 16)] = pl.tensor.exp(centered_0)
            pij_0: pl.Tensor[[16, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 4096, 17)] = pl.tensor.cast(exp_vals_0, target_type=pl.BFLOAT16, mode=2)
            pl.comm.tpush_to_aic(pij_0, AIV_IDX)
            oi_0: pl.Tensor[[8, 128], pl.BFLOAT16, pl.MemRef(pl.MemorySpace.DDR, -1, 2048, 18)] = pl.comm.tpop_from_aic(AIV_IDX)
            out_3: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 20)] = pl.tensor.assemble(out_iter_1_outer_l1, oi_0, [cur_offset_0 + AIV_IDX * 8, 0])
            out_iter_1_outer_l1_rv: pl.Tensor[[64, 128], pl.FP32, pl.MemRef(pl.MemorySpace.DDR, -1, 32768, 21)] = pl.yield_(out_3)
        return out_iter_1_outer_l1_rv
    @pl.function_group(aic="predicate_kernel_incore_0_aic", aiv="predicate_kernel_incore_0_aiv")
    class predicate_kernel_incore_0_group:
        pass
