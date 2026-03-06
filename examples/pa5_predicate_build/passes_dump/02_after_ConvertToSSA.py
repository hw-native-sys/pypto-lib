# pypto.program: PredicateTestProgram
import pypto.language as pl

@pl.program
class PredicateTestProgram:
    @pl.function
    def predicate_kernel(self, query_0: pl.Tensor[[64, 128], pl.BFLOAT16], key_0: pl.Tensor[[128, 128], pl.BFLOAT16], value_0: pl.Tensor[[128, 128], pl.BFLOAT16], out_0: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        with pl.auto_incore():
            for b_idx_0, (out_iter_1,) in pl.parallel(0, 64, 1, init_values=(out_0,), chunk=8):
                cur_offset_0: pl.Scalar[pl.INDEX] = b_idx_0 * 16
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
                out_3: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.assemble(out_iter_1, oi_0, [cur_offset_0, 0])
                out_2: pl.Tensor[[64, 128], pl.FP32] = pl.yield_(out_3)
        return out_2