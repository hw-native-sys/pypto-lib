# Lightning Indexer Prolog Quant — Project Analysis & Port

## Original program

- **File**: `lightning_indexer_prolog_quant.py`
- **Entry**: `lightning_indexer_prolog_quant_compute`
- **Dependencies**: `tensor_functions` (matmul, mul, layernorm, softmax, symmetric_quant, neg, view, assemble), `models.deepseek_v32_exp.mla_prolog_quant.rope_2d`

## Algorithm (three paths)

| Path    | Steps |
|---------|--------|
| **Q**   | x → matmul(w_qb.T) → dequant(×w_qb_scale) → RoPE(cos, sin) → Hadamard(×hadamard_q) → symmetric_quant → q_int8_out, q_scale_out |
| **K**   | x → matmul(wk.T) → LayerNorm(γ, β, ε) → RoPE → Hadamard(×hadamard_k) → symmetric_quant → k_int8_out, k_scale_out |
| **Weights** | x → matmul(w_proj.T) → softmax → weights_out |

## PyPTO v3 port (self-contained)

- **No** `tensor_functions` or `models`: all logic in `pypto_src/golden.py`.
- **InCore kernels** (load → PTO-ISA tile ops → store):
  - **CUBE (Matrix)**: `incore_matmul` — single-tile matmul only (backend allows only one pipe type per function).
  - **VECTOR**: `incore_mul` (elementwise); `incore_softmax` (row_max → row_expand_sub → exp → row_sum → row_expand_div); `incore_layernorm` (full norm in one kernel); `incore_symmetric_quant` (abs → row_max → scale → row_expand_div → cast).
- **Orchestration**: `LightningIndexerPrologQuant` allocates intermediates, calls incore in sequence, uses `pl.assemble` for weights_out.
- **RoPE**: Simplified to `q_roped = q_deq * cos`; full RoPE would need a dedicated incore (rotate_half + cos/sin combination).
- **Tile sizes**: Fixed 16×16 for compilability; extend to tiled loops with `pl.range` and `pl.slice` for larger tensors.

### Fusion and backend constraint

- **One incore = one pipe type**: The PTO/CCE orchestration codegen requires each InCore to use only **one** core type (CUBE for matmul, VECTOR for elementwise/row_*). So **matmul cannot be fused** with mul, layernorm, or softmax in the same kernel (would mix M and V).
- **Fusion already applied within same type**: Softmax, layernorm, and symmetric_quant are each a **single incore** with multiple tile ops (row_max, row_expand_*, exp, row_sum, etc.) — this is the maximum fusion allowed for VECTOR-only kernels. CUBE kernels are matmul-only.
- **To fuse across matmul and vector ops** would require either: (1) backend/runtime support for mixed pipe types in one kernel, or (2) a different execution model that can schedule M and V in one logical kernel.

## Compilation

From `../../pypto` (or pypto repo root), ensure `golden` is on `PYTHONPATH` and compile the program (e.g. via `ir.compile(program, ...)`). Fix any grammar/codegen errors in the generated code.

## Extensions

1. **Full RoPE**: Add an incore that implements rotate_half (e.g. two halves via views, neg second half, then combine with cos/sin terms) and call it from orchestration.
2. **Tiled orchestration**: Loop over batch and feature dimensions, call incore with views and optional `incore_matmul_acc` for K-dimension reduction.
3. **SRAM tuning**: Adjust TM/TN/TK or fuse more ops into one incore based on buffer usage reports.
