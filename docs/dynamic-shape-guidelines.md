# Dynamic Shape Guidelines

This document describes how to write `@pl.jit` / `@pl.jit.inline` kernels that
support dynamic B (batch) and S (sequence length) dimensions.  The patterns
below were extracted from the DeepSeek-V4 compressor and hc_post conversions
(static → dynamic B/S, decode-only → unified decode+prefill).

```python
import pypto.language as pl
```

---

## 1. Declare DynVars at Module Level

`pl.dynamic("name")` creates a `DynVar` (subclass of `Scalar`) that represents a
symbolic dimension.  Declare all dynamic dimensions as **module-level
constants**:

```python
B_DYN = pl.dynamic("B_DYN")
S_DYN = pl.dynamic("S_DYN")
```

Some kernels use additional derived dynamic dimensions (e.g. table/cache sizes
that vary with batch).  Declare those the same way—use names that match your
problem domain:

```python
CACHE_BLOCK_NUM_DYN = pl.dynamic("CACHE_BLOCK_NUM_DYN")
TABLE_MAX_ROWS_DYN = pl.dynamic("TABLE_MAX_ROWS_DYN")
```

When a kernel is shared between decode and prefill, a product dimension like
`T = B * S` can be declared directly as a single DynVar:

```python
T_DYN = pl.dynamic("T_DYN")  # T = B * S
```

Keep the static constants alongside them — they are still needed for:

- Compilation-time tiling decisions (e.g. `K_BLOCKS = D // K_CHUNK`)
- Parameterized golden reference functions (`build_tensor_specs(B, S)`)
- Multi-mode test loops (decode / prefill)

```python
B = DECODE_BATCH   # static upper bound, used by golden / tiling
S = DECODE_SEQ
```

---

## 2. Use DynVars Only in Parameter Type Annotations

DynVars belong **exclusively in `pl.Tensor[[...]]` type annotations** of
function parameters.  Never use them directly in the function body.

```python
# ✅ Correct — DynVars in annotations
@pl.jit.inline
def compressor(
    x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16],
    kv: pl.Tensor[[B_DYN, S_DYN, HEAD_DIM], pl.FP32],
    ...
):

# ❌ Wrong — DynVar arithmetic in function body
def compressor(x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16], ...):
    x_flat = pl.reshape(x, [B_DYN * S_DYN, D])   # SSA failure
```

**Why:** The JIT specializer resolves DynVars in annotations to
`pl.tensor.dim(param, dim_idx)` expressions, but the SSA renamer does **not**
update variable references embedded inside IR type annotations.  If
`B_DYN * S_DYN` appears in a body-level shape expression, the generated IR
annotation retains an un-renamed `pl.tensor.dim(x, ...)`, while the code
references the SSA-renamed `x__ssa_v0` — causing "Variable 'x' used outside
its defining scope" errors.

---

## 3. Extract Runtime Dimensions via `pl.tensor.dim()`

Inside the function body, capture dynamic dimensions into **local Scalar
variables** using `pl.tensor.dim()`, then use those locals everywhere:

```python
def compressor(x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16], ...):
    b_dim = pl.tensor.dim(x, 0)
    s_dim = pl.tensor.dim(x, 1)
    bs = b_dim * s_dim
    x_flat = pl.reshape(x, [bs, D])
```

The SSA renamer correctly tracks local Scalar definitions and their uses.

---

## 4. Avoid Composite Scalar Expressions in Shape Annotations

IR shape annotations (from `pl.create_tensor`, `pl.reshape`) must contain
**single Scalar variables**, not composite expressions involving Scalars:

```python
# ❌ Wrong — "BATCH_CHUNK_0 * s_dim" embeds a composite in the IR annotation
scratch = pl.create_tensor([BATCH_CHUNK_0 * s_dim, OUT_DIM], dtype=pl.FP32)

# ✅ Correct — compute to a local first, then pass the single variable
chunk_s = BATCH_CHUNK_0 * s_dim
scratch = pl.create_tensor([chunk_s, OUT_DIM], dtype=pl.FP32)
```

**Why:** The same SSA limitation — composite Scalar expressions in shape
annotations are not renamed.  Extracting to a named local avoids the problem.

### Skip reshape-back in `@pl.jit.inline` with `pl.Out` tensors

When an inlined function writes through a reshaped view of an `pl.Out` tensor
(e.g. `y_flat = pl.reshape(y, [t_dim, INNER_DIM])`), the data is already in the
output buffer.  **Skip the reshape-back** at the end:

```python
@pl.jit.inline
def my_kernel(
    y: pl.Out[pl.Tensor[[T_DYN, INNER_DIM], pl.BF16]],
):
    y_flat = pl.reshape(y, [t_dim, INNER_DIM])
    # ... write to y_flat ...
    # ❌ Don't — reshape-back breaks multi-level inlining
    # y = pl.reshape(y_flat, [t_dim, INNER_DIM])
    return y   # data already written through y_flat view
```

**Why:** The reshape-back introduces an extra SSA variable that carries a
dynamic-dimension shape (`t_dim`).  When this function is inlined into a caller
that passes a concrete-shaped tensor, the runtime tensor mapping breaks.

The failure occurs specifically when `@pl.jit.inline` is nested (caller is also
`@pl.jit.inline` with a non-`pl.Out` tensor), causing the runtime's output
tensor count to diverge from what the inlined IR expects.  A single-level
`@pl.jit` → `@pl.jit.inline` call chain is unaffected.

---

## 5. Use `bind_dynamic` in the `@pl.jit` Entry Point

In the `@pl.jit` wrapper function, **both** annotate with DynVars **and** call
`bind_dynamic()` for every tensor that has dynamic dimensions:

```python
@pl.jit
def compressor_test(
    x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16],
    kv: pl.Out[pl.Tensor[[B_DYN, S_DYN, HEAD_DIM], pl.FP32]],
    ...
):
    x.bind_dynamic(0, B_DYN)
    x.bind_dynamic(1, S_DYN)
    kv.bind_dynamic(0, B_DYN)
    kv.bind_dynamic(1, S_DYN)
    ...
```

The annotation tells the specializer the dimension is dynamic; the
`bind_dynamic` call ensures the DynDim cascade mechanism correctly propagates
dynamic dimensions through `@pl.jit.inline` dependencies.

---

## 6. Dynamic Loop Bounds and spmd

`pl.spmd` accepts compile-time constants, single `pl.Scalar` variables, **and
composite dynamic expressions** (e.g. `b_dim * HEAD_DIM // HEAD_TILE`) as block
count.  The orchestration codegen materializes composite expressions as named
`int64_t` variables in the generated C++.

```python
# ✅ OK — single dynamic variable
for idx in pl.spmd(s_dim, name_hint="..."):
    ...

# ✅ OK — compile-time constant expression
for idx in pl.spmd(HEAD_DIM // HEAD_TILE, name_hint="..."):
    ...

# ✅ OK — composite dynamic expression (verified fixed 2026-05-30)
for idx in pl.spmd(b_dim * HEAD_DIM // HEAD_TILE, name_hint="..."):
    global_c_idx = idx // (HEAD_DIM // HEAD_TILE)
    h0 = (idx % (HEAD_DIM // HEAD_TILE)) * HEAD_TILE
    ...
```

### Index decomposition: dynamic dimension outermost

When an SPMD loop covers multiple dimensions (e.g. `t_dim × out_h × d_steps`),
**place the dynamic dimension outermost** so all `//` and `%` operations use
compile-time constants:

```python
# Tiling constants (compile-time)
INNER_STEPS = D // D_CHUNK
BLOCKS_PER_OUTER = HEAD_COUNT * INNER_STEPS

# SPMD loop
for block in pl.spmd(t_dim * BLOCKS_PER_OUTER, name_hint="..."):
    t = block // BLOCKS_PER_OUTER         # div by compile-time constant
    local = block % BLOCKS_PER_OUTER      # mod by compile-time constant
    h = local // INNER_STEPS              # div by compile-time constant
    d0 = (local % INNER_STEPS) * D_CHUNK  # mod by compile-time constant
    ...
```

**Why:** `t_dim` is the only runtime value.  By placing it in the most
significant position, every downstream `//` and `%` divides by a compile-time
constant, which the codegen can emit as efficient integer division.  If the
dynamic dimension were innermost, `// t_dim` or `% t_dim` would require
runtime division in the hot loop.

The general pattern for converting static `spmd` to dynamic:

| Before (static) | After (dynamic) |
|--------|-------|
| `pl.spmd(B * HEAD_DIM // HEAD_TILE)` | `pl.spmd(b_dim * HEAD_DIM // HEAD_TILE)` |
| `pl.spmd(B // RMS_TILE)` | `pl.spmd(b_dim // RMS_TILE)` |
| `pl.spmd(B * S * OUT_DIM // (B_TILE * OUT_TILE))` | `pl.spmd(bs * OUT_DIM // (B_TILE * OUT_TILE))` |

`pl.parallel`, `pl.range`, and `pl.pipeline` also accept dynamic bounds.

---

## 7. Dynamic Values in Conditionals

Runtime Scalar comparisons inside `pl.at` scopes produce data-dependent control
flow:

```python
if runtime_val + s_dim < THRESHOLD:
    # short path
elif runtime_val + s_dim == THRESHOLD:
    # exact-boundary path
```

Both `runtime_val` (e.g. from `pl.read`) and `s_dim` are runtime Scalars.  The
compiler generates appropriate dynamic branching.

---

## 8. Keep Compilation-Time Tiling Parameters Static

Tiling decisions that affect pipeline stages, `pl.spmd` block counts, or tile
sizes must use **static** constants:

```python
K_BLOCKS = D // K_CHUNK              # compile-time derived
HEAD_BLOCKS = HEAD_DIM // HEAD_CHUNK  # compile-time derived
```

These values determine loop structures and pipeline depths in the generated IR
and cannot depend on runtime dimensions.

---

## 9. Golden and Test Functions

Golden and test code operates on concrete torch tensors and does not go through
the JIT compiler.  Depending on the kernel's scope, choose the appropriate
pattern:

### Single-mode kernels (decode-only or prefill-only)

Keep using static `B` and `S` constants unchanged:

```python
def golden_my_kernel(tensors):
    x = tensors["x"].float().reshape(B, S, D)
    ...

def build_tensor_specs():
    return [TensorSpec("x", [B, S, D], torch.bfloat16, ...)]
```

### Unified kernels (decode + prefill)

Parameterize `build_tensor_specs(B, S)` so the same golden function and test
harness can validate multiple batch sizes.  Derive `T` from the arguments:

```python
def build_tensor_specs(B, S):
    T = B * S
    return [TensorSpec("x", [T, D], torch.bfloat16, ...)]

def golden_my_kernel(tensors):
    T = tensors["x"].shape[0]   # dynamic from actual tensor shape
    ...
```

In `__main__`, add a `--mode` argument to iterate over multiple batch sizes:

```python
MODES = {
    "decode":  (DECODE_BATCH, DECODE_SEQ),
    "prefill": (PREFILL_BATCH, PREFILL_SEQ),
}

for mode_name in modes_to_run:
    B, S = MODES[mode_name]
    result = run_jit(
        fn=my_kernel_test,
        specs=build_tensor_specs(B, S),
        golden_fn=golden_my_kernel,
        ...
    )
```

---

## 10. Unified Multi-Mode Kernels

Dynamic shapes enable a single `@pl.jit.inline` kernel to serve multiple modes
(e.g. decode and prefill), eliminating duplicate code.  The key steps:

**1) Use a product DynVar instead of separate B/S.**  When a kernel operates on
a flat token dimension `[T, ...]` and does not need to distinguish B from S
internally, declare a single DynVar for the product:

```python
T_DYN = pl.dynamic("T_DYN")  # T = B * S

@pl.jit.inline
def my_kernel(x: pl.Tensor[[T_DYN, D], pl.BF16], ...):
    t_dim = pl.tensor.dim(x, 0)
    ...
```

**2) Callers reshape to the flat form the unified kernel expects.**  Callers
that hold `[B, S, ...]` tensors reshape to `[T, ...]` before invoking the
kernel, then reshape the result back:

```python
# Caller side
x_flat = pl.reshape(x, [T, D])
y_flat = my_kernel(x_flat, ...)
y = pl.reshape(y_flat, [B, S, ...])
```

**3) Remove the old mode-specific wrapper.**  After unification, any mode-only
wrapper file is no longer needed and can be deleted.  Update all import sites
to use the unified module.

---

## Quick Reference

| Scenario | Do | Don't |
|----------|----|-------|
| Parameter annotations | `pl.Tensor[[B_DYN, S_DYN, ...]]` or `pl.Tensor[[T_DYN, ...]]` | `B_DYN * S_DYN` in annotations |
| Derived dynamic dims | `CACHE_BLOCK_NUM_DYN = pl.dynamic(...)` | Use module-level statics for runtime-varying dims |
| reshape / create shape | `pl.tensor.dim()` → local var | Direct DynVar arithmetic |
| Shape with composite | Compute to local first | `pl.create_tensor([C * s_dim, ...])` |
| Reshape-back on `pl.Out` | Skip; data already written through view | `pl.reshape(y_flat, [dyn, ...])` at end of `@pl.jit.inline` |
| Loop bounds | `pl.parallel(0, b_dim, chunk)` | `pl.parallel(0, B_DYN, chunk)` |
| Conditionals | `runtime_val + s_dim < THRESHOLD` | `runtime_val + S_DYN < THRESHOLD` |
| spmd block count | `pl.spmd(STATIC_DIM)`, `pl.spmd(s_dim)`, or `pl.spmd(b_dim * STATIC)` | — |
| spmd → dynamic split | `pl.spmd(dyn * static)` — no split needed | — |
| spmd index decomp | Dynamic dim outermost: `t = block // CONST` | Dynamic dim innermost: `% t_dim` in hot loop |
| Index arithmetic | `global_b * s_dim` | `global_b * S_DYN` |
| @pl.jit entry | Annotate + `bind_dynamic()` | Annotate only |
| Single-mode golden/test | Keep static B/S | — |
| Unified golden/test | `build_tensor_specs(B, S)`, `--mode` loop | Hardcode B/S inside `build_tensor_specs` |
| Tiling constants | Keep static compile-time calculation | Cannot use runtime dims |
