# pypto.program: PredicateTestProgram
import pypto.language as pl

@pl.program
class PredicateTestProgram:
    @pl.function(type=pl.FunctionType.InCore)
    def predicate_kernel_incore_0(self, b_idx_0_out: pl.Scalar[pl.INDEX], key_0: pl.Tensor[[128, 128], pl.BFLOAT16], out_0: pl.Tensor[[64, 128], pl.FP32], out_iter_1_outer_l0: pl.Tensor[[64, 128], pl.FP32], query_0: pl.Tensor[[64, 128], pl.BFLOAT16], value_0: pl.Tensor[[128, 128], pl.BFLOAT16]) -> pl.Tensor[[64, 128], pl.FP32]:
        for b_idx_0_in, (out_iter_1_outer_l1,) in pl.parallel(0, 8, 1, init_values=(out_iter_1_outer_l0,)):
            cur_offset_0: pl.Scalar[pl.INDEX] = (0 + (b_idx_0_out * 8 + b_idx_0_in) * 1) * 16
            qi_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.view(query_0, [16, 128], [cur_offset_0, 0])
            kj_0: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(key_0, [128, 128], [0, 0])
            vj_0: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(value_0, [128, 128], [0, 0])
            sij_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.matmul(qi_0, kj_0, a_trans=False, b_trans=True, c_matrix_nz=False)
            mi_0: pl.Tensor[[16, 1], pl.BFLOAT16] = pl.tensor.row_max(sij_0)
            mi_flat_0: pl.Tensor[[1, 16], pl.BFLOAT16] = pl.tensor.reshape(mi_0, [1, 16])
            global_max_0: pl.Tensor[[1, 1], pl.BFLOAT16] = pl.tensor.row_max(mi_flat_0)
            centered_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.sub(sij_0, global_max_0)
            exp_vals_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.exp(centered_0)
            pij_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.cast(exp_vals_0, target_type=pl.BFLOAT16, mode=2)
            oi_0: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.matmul(pij_0, vj_0, a_trans=False, b_trans=False, c_matrix_nz=False)
            out_3: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.assemble(out_iter_1_outer_l1, oi_0, [cur_offset_0, 0])
            out_iter_1_outer_l1_rv: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(out_3)
        return out_iter_1_outer_l1_rv
    @pl.function(type=pl.FunctionType.Orchestration)
    def predicate_kernel(self, query_0: pl.Tensor[[64, 128], pl.BFLOAT16], key_0: pl.Tensor[[128, 128], pl.BFLOAT16], value_0: pl.Tensor[[128, 128], pl.BFLOAT16], out_0: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        for b_idx_0_out, (out_iter_1_outer_l0,) in pl.range(0, 8, 1, init_values=(out_0,)):
            out_iter_1_outer_l1_rv: pl.Tensor[[64, 128], pl.FP32] = self.predicate_kernel_incore_0(b_idx_0_out, key_0, out_0, out_iter_1_outer_l0, query_0, value_0)
            out_iter_1_outer_l0_rv: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(out_iter_1_outer_l1_rv)
        return out_iter_1_outer_l0_rv