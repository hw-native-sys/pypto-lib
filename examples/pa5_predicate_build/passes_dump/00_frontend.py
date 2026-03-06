# pypto.program: PredicateTestProgram
import pypto.language as pl

@pl.program
class PredicateTestProgram:
    @pl.function
    def predicate_kernel(self, query: pl.Tensor[[64, 128], pl.BFLOAT16], key: pl.Tensor[[128, 128], pl.BFLOAT16], value: pl.Tensor[[128, 128], pl.BFLOAT16], out: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
        with pl.auto_incore():
            for b_idx in pl.parallel(0, 64, 1, chunk=8):
                cur_offset: pl.Scalar[pl.INDEX] = b_idx * 16
                qi: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.view(query, [16, 128], [cur_offset, 0])
                kj: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(key, [128, 128], [0, 0])
                vj: pl.Tensor[[128, 128], pl.BFLOAT16] = pl.tensor.view(value, [128, 128], [0, 0])
                sij: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.matmul(qi, kj, a_trans=False, b_trans=True, c_matrix_nz=False)
                mi: pl.Tensor[[16, 1], pl.BFLOAT16] = pl.tensor.row_max(sij)
                mi_flat: pl.Tensor[[1, 16], pl.BFLOAT16] = pl.tensor.reshape(mi, [1, 16])
                global_max: pl.Tensor[[1, 1], pl.BFLOAT16] = pl.tensor.row_max(mi_flat)
                centered: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.sub(sij, global_max)
                exp_vals: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.exp(centered)
                pij: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.cast(exp_vals, target_type=pl.BFLOAT16, mode=2)
                oi: pl.Tensor[[16, 128], pl.BFLOAT16] = pl.tensor.matmul(pij, vj, a_trans=False, b_trans=False, c_matrix_nz=False)
                out: pl.Tensor[[64, 128], pl.FP32] = pl.tensor.assemble(out, oi, [cur_offset, 0])
        return out