# PyPTO Coding Style

This document describes the canonical coding style for writing `pl.function`
kernels in PyPTO-Lib.

```python
import pypto.language as pl
```

`pl` is the only accepted module alias.

---

## 1. Defining a Kernel: `@pl.jit` and `@pl.program`

A PyPTO-Lib kernel can be written in **two parallel forms** — module-level
`@pl.jit` functions or a `@pl.program` class. Both lower through the same
compiler pipeline. Pick one per kernel and do not mix them: a `@pl.program`
method calling a `@pl.jit` kernel (or the reverse) is discouraged.

Either way, signatures look the same — tensor params are
`pl.Tensor[[shape...], dtype]`, outputs are wrapped in `pl.Out[...]`, scalars
are `pl.Scalar[dtype]` — and the function is written as an **opaque** function:
the frontend does not draw the InCore / Orchestration boundary explicitly.
Each compute region is wrapped in `with pl.at(level=pl.Level.CORE_GROUP, ...)`
and the compiler lowers that region to InCore; code outside any `pl.at` block
stays in orchestration (host / AICPU control flow). See `pl.at` scopes below.

### Form A — module-level `@pl.jit` / `@pl.jit.inline`

The form used by most DeepSeek-V4 kernels: plain module-level functions.

`@pl.jit` decorates the top-level function the harness compiles and runs — the
boundary the golden test invokes; its `pl.Out` params are the kernel outputs.

`@pl.jit.inline` marks a reusable sub-kernel that is **inlined** into each
caller rather than compiled as its own entry. Write the real compute once in an
inline function, then call it from a thin `@pl.jit` entry. An inline function
**must return a value** — the parser requires every inline call expression to
have a result; when the kernel writes in place, returning the `pl.Out` tensor
is idiomatic.

```python
@pl.jit.inline
def expert_routed(recv_x, ..., recv_y):        # the real compute
    for local_i in pl.parallel(N_LOCAL_EXPERTS):
        for nb_idx in pl.spmd(..., name_hint="exp_gate_up"):
            ...                                # matmul / dequant / SwiGLU
    return recv_y                              # inline call must return a value


@pl.jit                                        # compilation entry
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    ...,
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    expert_routed(recv_x, ..., recv_y)         # call the inline sub-kernel
    return recv_y
```

### Form B — `@pl.program` class with `@pl.function` methods

A class groups related kernels as methods; `type=` on each method selects how
it lowers.

```python
@pl.program
class Qwen3Decode:
    @pl.function(type=pl.FunctionType.Opaque)
    def qwen3_decode(self, hidden_states: pl.Tensor[..., pl.BF16], ...):
        # orchestration code (loops, tensor allocation)
        for b0 in pl.parallel(0, batch_padded, BATCH_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                # InCore region — vector / cube / mte ops
                ...
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                ...
        return out
```

| `pl.FunctionType` | Role |
|-------------------|------|
| `Opaque` | Self-contained compute kernel; the frontend draws the InCore boundary from its `pl.at` blocks (the example above). |
| `Orchestration` | Top-level entry that sequences other methods (host / AICPU control flow, cross-rank dispatch). |
| `InCore` | A single InCore region authored directly — `pl.spmd` / `pl.parallel` / `pl.pipeline` and scalar loops, no surrounding `pl.at`. |

### `pl.at` scopes

| Parameter | Required | Purpose |
|-----------|----------|---------|
| `level=pl.Level.CORE_GROUP` | yes | Lowering target. `CORE_GROUP` is the only level used in pypto-lib. |
| `name_hint="..."` | recommended | Stable label for the region. Appears in generated kernel filenames and profiling traces; aids per-region debugging. |
| `optimizations=[...]` | optional | Per-region codegen passes (see below). |

`pl.at` blocks may nest: an outer `pl.at` defining the InCore scope, with
inner `pl.at` blocks (each with its own `name_hint`) splitting it into named
sub-kernels.

### `optimizations`

`optimizations=[...]` attaches per-region codegen passes to a `pl.at` block
(or a `pl.spmd` loop — same kwarg). The one in common use:

- **`pl.split(pl.SplitMode...)`** — split the region in half so the cube and
  vector units ping-pong on the two halves (cube on one half while vec runs
  the epilogue on the other). It applies **only to a mixed cube + vector
  region** (§6); a pure-cube or pure-vector region has nothing to ping-pong.
  The mode picks the axis:
  - `pl.SplitMode.NONE` — the default; no split.
  - `pl.SplitMode.UP_DOWN` — split vertically (rows / height halved).
  - `pl.SplitMode.LEFT_RIGHT` — split horizontally (cols / width halved).

  Reach for it when a region's unified buffer (UB) would otherwise exceed the
  per-core limit — typically a wide FP32 vector epilogue stacked on a matmul
  accumulator — since splitting also keeps the accumulator on-chip instead of
  spilling to a GM scratch round-trip.

```python
# split form on a mixed region whose FP32 epilogue would blow the UB budget
for ob in pl.spmd(N_BLOCKS, name_hint="gate_up_silu",
                  optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
    ...
```

---

## 2. Vector Ops

Run on the vector unit, inside an InCore region (a `pl.at` block or a
`pl.spmd` body — see §5). Vector ops are the standard tools for the cast /
activation / norm epilogue around a matmul, and for small standalone
reductions.

### Elementwise

Tensor-tensor binary: `pl.add`, `pl.sub`, `pl.mul`, `pl.div`,
`pl.maximum`, `pl.minimum`. Tensor-scalar variants (second operand is a
Python `int`/`float` or `pl.Scalar`) suffix with `s`: `pl.adds`, `pl.subs`,
`pl.muls`, `pl.divs`, `pl.maxs`, `pl.mins`. Unary: `pl.neg`, `pl.abs`,
`pl.exp`, `pl.log`, `pl.sqrt`, `pl.recip`, `pl.rsqrt`. Activations:
`pl.relu`, `pl.lrelu`, `pl.prelu`. Type conversion: `pl.cast(x, target_type=...)`.
Binary ops broadcast over compatible shapes; prefer `pl.recip` + `pl.mul`
over `pl.div` on hot paths.

```python
silu_x = pl.mul(x, pl.recip(pl.add(pl.exp(pl.neg(x)), one)))   # x * sigmoid(x)
out_bf16 = pl.cast(acc_fp32, target_type=pl.BF16)
scaled = pl.muls(scores, attn_scale)                           # scalar mul
```

For comparison / select / bit-twiddling — `pl.cmp`, `pl.cmps`, `pl.sel`,
`pl.sels`, `pl.and_`, `pl.or_`, `pl.xor`, `pl.not_`, `pl.shl`, `pl.shr` —
see existing kernels.

### Reductions

Row reductions (along the last axis, return `[..., 1]`): `pl.row_max`,
`pl.row_min`, `pl.row_sum`. Column reductions (return `[1, ...]`):
`pl.col_max`, `pl.col_min`, `pl.col_sum`. Pair with broadcast ops below for
the typical RMSNorm / softmax patterns:

```python
sq_sum = pl.row_sum(pl.mul(x, x))                       # [B, 1]
inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))
```

### Row / column broadcast

Apply a column to each row: `pl.row_expand_add`, `pl.row_expand_sub`,
`pl.row_expand_mul`, `pl.row_expand_div`. Apply a row to each column:
`pl.col_expand_sub`, `pl.col_expand_mul`, `pl.col_expand_div`. Each maps
to a single hardware broadcast op — use them rather than reshaping a
vector and relying on elementwise broadcast.

```python
# RMSNorm body: normed[i, j] = x[i, j] * inv_rms[i] * gamma[j]
normed = pl.col_expand_mul(pl.row_expand_mul(x, inv_rms), gamma)
```

### Fill and pad

`pl.full(shape, dtype=..., value=...)` allocates a scalar-filled
tensor/tile (typical use: zero-init a partial accumulator before a
reduction). `pl.fillpad(x, pad_value=...)` rewrites the padded tail of a
`valid_shape` slice with a sentinel — most often `pl.PadValue.min` to
mask out invalid positions before a softmax `row_max`. There is also an
in-place `pl.fillpad_inplace`.

```python
partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)   # -inf in tail
```

`pl.set_validshape(tile, valid_rows, valid_cols)` re-marks the valid
region of an **already-computed** tile. Where `valid_shape=` on a
`pl.slice` (§4) is a load-time marker on data coming from GM,
`set_validshape` annotates a tile produced on chip — typically when the
valid row/col count is only known at runtime (a `pl.read` of a dynamic
count). The returned view has the same nominal shape; downstream ops
(reductions, `fillpad`) then operate on the valid region only. It is the
standard partner of `fillpad`: set the valid extent, then mask the tail.

```python
valid_rows = pl.min(RECV_TILE, n_rows - t0)                    # runtime count
gated_valid = pl.set_validshape(gated, valid_rows, INTER_TILE)
# softmax tail-masking idiom: set extent, then fill the pad with -inf
scores = pl.fillpad(pl.set_validshape(weighted, 1, valid_len),
                    pad_value=pl.PadValue.min)
```

### Sort and top-k: `pl.sort32` + `pl.mrgsort`

Top-k is built from two primitives that operate on a **single row**
(`[1, N]`):

- `pl.sort32(values, idx_init)` sorts each contiguous 32-element run in
  descending order, carrying the indices along. `idx_init` is a `[1, N]`
  UINT32 index ramp (`pl.arange(0, [1, N], dtype=pl.UINT32)`). The result is
  `[1, 2*N]` of interleaved `(value, index)` pairs.
- `pl.mrgsort(sorted, block_len=B)` 4-way-merges adjacent sorted blocks
  (format stays interleaved pairs). `block_len` is the **input** run length,
  counted in interleaved-array positions — twice the element count. `sort32`
  leaves runs of 32 elements (64 positions), so each merge grows the run ×4
  and `block_len` steps ×4 per stage until a single run remains: `64 → 256`
  sorts 512 elements, `64 → 256 → 1024` sorts 2048.

Both require **row count == 1** — an ISA constraint on `mrgsort`; for a
multi-row tile, loop the rows with `pl.range`. After sorting, slice the
leading `2*k` pairs and `pl.gather` the odd lanes (the indices) for the top-k
index list.

```python
score_row = score_flat[t : t + 1, :]                  # [1, N], N = 512
idx_init = pl.arange(0, [1, N], dtype=pl.UINT32)
s = pl.sort32(score_row, idx_init)                    # [1, 2N], 32-runs sorted
s = pl.mrgsort(s, block_len=64)                       # 4-way merge of the 64-position runs
s = pl.mrgsort(s, block_len=256)                      # one sorted run
topk_pairs = s[:, 0 : 2 * K]                          # leading k (value, index) pairs
topk_idxs = pl.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010,
                      output_dtype=pl.INT32)          # odd lanes = indices
```

A full runnable kernel is in [examples/advanced/topk.py](../examples/advanced/topk.py).

### Gather / scatter

Two flavours.

**Mask form** — de-interleave / re-interleave even and odd lanes (the RoPE
idiom). `pl.gather(tile, mask_pattern=...)` selects alternate lanes of a
`[H, W]` tile, returning `[H, W/2]`; `pl.tensor.scatter(src, mask_pattern=...,
dst=buf)` writes them back into the matching lanes of `dst`.
`pl.tile.MaskPattern.P0101` picks the even lanes (0, 2, 4, …), `P1010` the odd
lanes (1, 3, 5, …). The optional `output_dtype=` casts on the way out.

```python
even = pl.gather(rope_slice, mask_pattern=pl.tile.MaskPattern.P0101)   # [H, W/2]
odd  = pl.gather(rope_slice, mask_pattern=pl.tile.MaskPattern.P1010)
rot_even = pl.sub(pl.col_expand_mul(even, cos_b), pl.col_expand_mul(odd, sin_b))
rot_odd  = pl.add(pl.col_expand_mul(even, sin_b), pl.col_expand_mul(odd, cos_b))
buf = pl.full([H, W], dtype=pl.FP32, value=0.0)
buf = pl.tensor.scatter(rot_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=buf)
buf = pl.tensor.scatter(rot_odd,  mask_pattern=pl.tile.MaskPattern.P1010, dst=buf)
```

**Index form** — batched per-row gather by an index tile.
`pl.gather(src, dim=-1, index=idx)` gathers along the last axis: for a
`[B, W]` source and a `[B, K]` INT32 index it returns `[B, K]`, where row `i`
picks `src[i, idx[i, :]]`. The index must be a real **tensor** (e.g. from
`pl.create_tensor`), not a `pl.full` tile — a tile index is rejected.

```python
gathered = pl.gather(local_scores, dim=-1, index=topk_idx_tile)   # [B, K]
```

---

## 3. Cube Ops

Matrix multiply primitives. They run on the cube unit, inside an InCore
region (a `pl.at` block or a `pl.spmd` body — see §5), and produce FP32
results by default (`out_dtype` may override).

### `pl.matmul(lhs, rhs, *, out_dtype=None, a_trans=False, b_trans=False)`

Plain matmul. `a_trans` / `b_trans` transpose the corresponding operand
without a separate `pl.transpose`. `out_dtype` overrides the default FP32
accumulator dtype when set.

```python
out = pl.matmul(tile_a, tile_b)
out = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32, b_trans=True)
```

### `pl.matmul_acc(acc, lhs, rhs, *, a_trans=False, b_trans=False)`

Fused multiply-accumulate: `acc += lhs @ rhs`. Use this inside a K-loop to
keep the partial sum on chip. The first iteration uses `pl.matmul`,
subsequent iterations use `pl.matmul_acc`. The `acc` destination is
allocated outside `pl.at` (see §4):

```python
acc = pl.create_tensor([M, N], dtype=pl.FP32)
with pl.at(level=pl.Level.CORE_GROUP, name_hint="kproj"):
    for kb in pl.pipeline(0, K_BLOCKS, stage=2):
        k0 = kb * K_STEP
        tile_a = pl.slice(a, [M, K_STEP], [m0, k0])
        tile_b = pl.slice(b, [K_STEP, N], [k0, n0])
        if kb == 0:
            acc = pl.matmul(tile_a, tile_b)
        else:
            acc = pl.matmul_acc(acc, tile_a, tile_b)
```

### `pl.matmul_bias(lhs, rhs, bias)`

Matmul fused with a bias add: `lhs @ rhs + bias`. Cheaper than a separate
`pl.add` epilogue when the bias is broadcast over the M axis.

### `pl.batch_matmul`, `pl.batch_matmul_acc`

Batched variants for stacked matmul (shape `[B, M, K] @ [B, K, N]`); same
arg shape as the non-batched forms.

### `pl.gemv`, `pl.gemv_acc`, `pl.gemv_bias`

Vector-matrix specializations (1-row left operand). Prefer over `pl.matmul`
when M is 1 — the cube schedules the smaller form more efficiently.

---

## 4. MTE Ops (Data Movement / Shape)

MTE primitives manipulate tensor views and stage data without explicit
load/store. The compiler decides where the actual TLOAD/TSTORE land based
on where each `pl.slice` / `pl.assemble` sits relative to `pl.at`.

### `pl.create_tensor(shape, dtype=...)` — orchestration only

`create_tensor` lives **outside** `pl.at`. Use it when multiple `pl.at`
regions cooperate to fill one intermediate tensor — the tensor is allocated
once in orchestration, then each region writes its piece via `assemble`. If
the result of a single `pl.at` flows directly to its caller without further
assembly, no `create_tensor` is needed.

```python
# Multi-stage assembly: q_proj is built by per-tile assembles
q_proj = pl.create_tensor([batch_padded, q_hidden], dtype=pl.BF16)
for q0 in pl.parallel(0, q_hidden, Q_OUT_STEP):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
        ...
        q_proj = pl.assemble(q_proj, q_acc, [b0, q0])
```

### `pl.slice` / `pl.assemble` — load/store at the `pl.at` boundary

`pl.slice(tensor, sizes, offsets, valid_shape=...)` takes a sub-region;
`pl.assemble(dst, src, offsets)` writes a sub-region back. When these cross
the `pl.at` boundary they lower to InCore TLOAD / TSTORE — the tensor
descriptor (offset / shape / stride) is passed as an InCore argument and
the data movement is generated by the compiler. Argument order on `slice`
is **(sizes, offsets)**.

```python
tile = pl.slice(hidden_states, [BATCH_TILE, K_STEP], [b0, k0])
...
q_proj = pl.assemble(q_proj, q_acc, [b0, q0])
```

The shorthand subscript forms are equivalent and often clearer:

```python
tile = hidden_states[b0 : b0 + BATCH_TILE, k0 : k0 + K_STEP]   # slice
q_proj[b0 : b0 + BATCH_TILE, q0 : q0 + Q_OUT_STEP] = q_acc     # assemble
```

`valid_shape=[real_h, real_w]` on a slice marks a padded load: the slice
has nominal size `sizes` but only the leading `valid_shape` rows/cols
carry real data, and the compiler zero-pads the tail. Use this for
dynamic batch / sequence length when the kernel works on a fixed
`BATCH_TILE` but the caller may pass fewer valid rows.

### `pl.reshape(x, new_shape)`

Logical reshape (view-only). Total element count must match. Legal both
inside and outside `pl.at`.

```python
flat = pl.reshape(q_chunk, [BATCH_TILE * H, D])
```

---

## 5. Loops: `pl.range`, `pl.parallel`, `pl.pipeline`, `pl.spmd`

PyPTO has four loop constructs. The choice between them is **semantic** —
it tells the compiler what scheduling and codegen are valid.

Each construct has a fixed placement relative to `pl.at`:

| Construct | Outside `pl.at` (orchestration) | Inside `pl.at` (InCore) |
|-----------|:-:|:-:|
| `pl.range` | yes | yes |
| `pl.parallel` | yes | no |
| `pl.pipeline` | no | yes |
| `pl.spmd` | yes (body is implicitly InCore) | no |

`pl.parallel` distributes iterations across cores, so it must sit in
orchestration. `pl.pipeline` software-pipelines stages within a single
InCore region, so it must sit inside `pl.at`. `pl.range` is a plain
sequential loop and is legal in either place. `pl.spmd` is a parallel SPMD
loop that bundles its own InCore region — see below.

`pl.range`, `pl.parallel`, and `pl.pipeline` share the same positional-arg
shape, mirroring Python's `range`:

```text
pl.<loop>(stop)
pl.<loop>(start, stop)
pl.<loop>(start, stop, step)
```

Each argument may be either a Python `int` or a `pl.Scalar`.

### `pl.range` — sequential

Iterations execute in strict order. Loop-carried dependencies are allowed.

```python
for kb in pl.range(HIDDEN_BLOCKS):              # range [0, HIDDEN_BLOCKS)
for kb in pl.range(1, hidden_blocks):           # range [1, hidden_blocks)
for k0 in pl.range(0, HIDDEN, K_STEP):          # start, stop, step
```

To carry state across iterations, pass `init_values=` and unpack the loop
variable as a `(idx, (state...))` tuple:

```python
for i, (acc,) in pl.range(N, init_values=(zero,)):
    acc = pl.add(acc, x[i])
```

### `pl.parallel` — independent iterations

Iterations are guaranteed independent — the compiler may split, reorder, or
schedule them across cores. Same arg shape and `init_values` support as
`pl.range`.

```python
for b in pl.parallel(BATCH):                          # short form: extent only
for b0 in pl.parallel(0, batch_padded, BATCH_TILE):   # start, stop, step
```

### `pl.pipeline` — software-pipelined sequential

Sequential like `pl.range`, but the compiler software-pipelines successive
iterations across compute and memory units. `stage=N` (required keyword) is
the pipeline depth — the loop body is replicated `stage` times for
ping-pong buffering; the outer trip count advances in strides of
`stage * step` and a tail dispatch covers the remainder when the trip count
is not divisible by `stage`. Typical values are 2 or 4.

```python
for kb in pl.pipeline(HIDDEN_BLOCKS, stage=2):
for kb in pl.pipeline(2, HIDDEN_BLOCKS, stage=2):     # start at 2
for kb in pl.pipeline(0, input_proj_k_blocks, stage=4):
```

`init_values=` is supported, with the same `(idx, (state...))` unpacking as
`pl.range`. Use `pl.pipeline` for the inner reduction loop of a matmul (the
K loop) — each iteration loads a new tile of the left/right operand and
accumulates into the same output.

### `pl.spmd` — parallel SPMD dispatch

`pl.spmd(core_num)` dispatches `core_num` blocks in parallel; iteration
starts at 0 and steps by 1 (only the block count is positional, **not**
start/stop/step). Two forms:

**Loop form** — body is auto-outlined into a synthetic InCore function;
the iteration variable binds the per-block index (equivalent to
`pl.tile.get_block_idx()`). No surrounding `pl.at` is needed or allowed:

```python
for ob0 in pl.spmd(Q_SPMD_BLOCKS, name_hint="q_proj"):
    # implicit InCore region — vector / cube / mte ops here
    for ob in pl.range(ob0 * 4, (ob0 + 1) * 4):
        q0 = ob * Q_OUT_STEP
        ...
```

**Context-manager form** — body must be a single call to a pre-defined
InCore kernel:

```python
with pl.spmd(4):
    out = self.kernel(a, b, out)
```

Keyword args:

| Kwarg | Default | Purpose |
|-------|---------|---------|
| `sync_start` | `False` | If True, all blocks start execution simultaneously. |
| `name_hint` | `""` | Stable label for the outlined function. |

---

## 6. Mixed Cube + Vector Within One `pl.at`

A single `pl.at` region can contain both cube (matmul) and vector (cast,
add, row_sum, …) ops. The compiler assigns each op to the appropriate unit
and pipelines cube/vec where possible. This is the standard pattern for a
projection with cast/residual epilogue:

```python
q_proj = pl.create_tensor([batch_padded, hidden], dtype=pl.BF16)
for q0 in pl.parallel(0, hidden, Q_OUT_STEP):
    q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_STEP], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
        for kb in pl.pipeline(0, input_proj_k_blocks, stage=2):
            k0 = kb * INPUT_PROJ_K_STEP
            tile_a = pl.slice(normed_tile,
                              [BATCH_TILE, INPUT_PROJ_K_STEP], [0, k0])  # vec source
            tile_b = pl.slice(wq,
                              [INPUT_PROJ_K_STEP, Q_OUT_STEP], [k0, q0])  # cube right
            if kb == 0:
                q_acc = pl.matmul(tile_a, tile_b)              # cube
            else:
                q_acc = pl.matmul_acc(q_acc, tile_a, tile_b)   # cube
        q_bf16 = pl.cast(q_acc, target_type=pl.BF16)           # vector
        q_proj = pl.assemble(q_proj, q_bf16, [b0, q0])         # mte
```

Larger fused regions (RMSNorm + projection + residual) follow the same
shape: a `pl.pipeline` matmul reduction, then the vector epilogue, then
`assemble` back to GM.

---

## 7. Dynamic Shapes (dynamic B / S)

`@pl.jit` / `@pl.jit.inline` kernels support dynamic batch (B) and sequence
(S) dimensions via `pl.dynamic` symbolic dims — a single kernel can serve both
decode and prefill. Almost every rule below traces back to one constraint: the
**JIT SSA renamer rewrites local Scalar references but not DynVar references
embedded in IR type annotations**. So DynVars must stay in annotations, and any
concrete shape math must go through named locals.

### Declare DynVars at module level

`pl.dynamic("name")` creates a `DynVar` (a `Scalar` subclass) for a symbolic
dimension. Declare them as module-level constants, alongside the static
constants you still need for tiling, golden, and test loops:

```python
B_DYN = pl.dynamic("B_DYN")
S_DYN = pl.dynamic("S_DYN")
T_DYN = pl.dynamic("T_DYN")   # T = B * S, for kernels on a flat token dim
B = DECODE_BATCH              # static upper bound for golden / tiling
```

### DynVars only in annotations; extract runtime dims with `pl.tensor.dim`

Use DynVars **exclusively** in `pl.Tensor[[...]]` parameter annotations. In the
body, capture each dynamic dim into a local Scalar with `pl.tensor.dim()` and
use the locals everywhere:

```python
@pl.jit.inline
def compressor(x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16], ...):
    b_dim = pl.tensor.dim(x, 0)        # ✅ local Scalar — renamer tracks it
    s_dim = pl.tensor.dim(x, 1)
    x_flat = pl.reshape(x, [b_dim * s_dim, D])
    # ❌ pl.reshape(x, [B_DYN * S_DYN, D]) — DynVar math in body → SSA failure
```

### No composite expressions in shape annotations

Shape annotations (`pl.create_tensor`, `pl.reshape`) must hold **single Scalar
variables**, not composites — extract to a named local first:

```python
chunk_s = BATCH_CHUNK_0 * s_dim                       # ✅ compute first
scratch = pl.create_tensor([chunk_s, OUT_DIM], dtype=pl.FP32)
# ❌ pl.create_tensor([BATCH_CHUNK_0 * s_dim, OUT_DIM], ...)
```

When an inlined function writes through a reshaped view of a `pl.Out` tensor,
the data is already in the output buffer — **skip the reshape-back** at the end
(`return y`, not `pl.reshape(y_flat, ...)`). A trailing reshape-back carries a
dynamic-shape SSA var that breaks the runtime tensor mapping when the inline is
nested inside another `@pl.jit.inline`.

### `bind_dynamic` at the `@pl.jit` entry

In the `@pl.jit` wrapper, both annotate with DynVars **and** call
`bind_dynamic()` for every dynamic dim, so the DynDim cascade propagates
through inline dependencies:

```python
@pl.jit
def compressor_test(x: pl.Tensor[[B_DYN, S_DYN, D], pl.BF16], ...):
    x.bind_dynamic(0, B_DYN)
    x.bind_dynamic(1, S_DYN)
```

### Dynamic loop bounds

All four loop constructs accept dynamic bounds. `pl.spmd` accepts a single
Scalar **or a composite dynamic expression** (`b_dim * HEAD_DIM // HEAD_TILE`)
as the block count. When an SPMD loop folds several dims into one, **place the
dynamic dim outermost** so every `//` and `%` divides by a compile-time
constant — otherwise the hot loop needs a runtime division:

```python
BLOCKS_PER_OUTER = HEAD_COUNT * (D // D_CHUNK)        # compile-time
for block in pl.spmd(t_dim * BLOCKS_PER_OUTER, name_hint="..."):
    t     = block // BLOCKS_PER_OUTER                 # ÷ constant
    local = block %  BLOCKS_PER_OUTER
    ...
```

Keep tiling constants (pipeline depth, tile sizes, spmd block factors) static —
they shape the generated IR and cannot depend on runtime dims. Runtime Scalar
comparisons in conditionals (`if runtime_val + s_dim < THRESHOLD`) just work.

### Golden / test and unified kernels

Golden and test code runs on concrete torch tensors (no JIT). For a unified
decode+prefill kernel, parameterize `build_tensor_specs(B, S)`, derive `T` from
the actual tensor shape, and iterate modes via a `--mode` arg:

```python
def build_tensor_specs(B, S):
    return [TensorSpec("x", [B * S, D], torch.bfloat16, ...)]

MODES = {"decode": (DECODE_BATCH, DECODE_SEQ),
         "prefill": (PREFILL_BATCH, PREFILL_SEQ)}
```

A single product DynVar `T_DYN` lets one `@pl.jit.inline` serve both modes: the
caller reshapes `[B, S, ...] → [T, ...]` before the call and back after, and
the old mode-specific wrapper is deleted.

### Quick reference

| Do | Don't |
|----|-------|
| `pl.Tensor[[B_DYN, S_DYN, ...]]` in annotations | `B_DYN * S_DYN` in annotations or body |
| `pl.tensor.dim(x, 0)` → local Scalar | DynVar arithmetic in the body |
| Compute composite to a local, then use it | `pl.create_tensor([C * s_dim, ...])` |
| Skip reshape-back on `pl.Out` in nested inline | trailing `pl.reshape(y_flat, [dyn, ...])` |
| Annotate **and** `bind_dynamic()` at `@pl.jit` | annotate only |
| `pl.spmd(b_dim * STATIC)`, dynamic dim outermost | dynamic dim innermost (`% t_dim` in hot loop) |
| Static tiling constants | tiling that depends on runtime dims |
| `build_tensor_specs(B, S)` + `--mode` loop | hardcode B/S inside `build_tensor_specs` |

---

## 8. Naming and comment conventions

### Name tile sizes, inline block counts

Give a named constant to a **tiling parameter** — the tile / step size — but
**not** to a derived block count (`K_BLOCKS`, `Q_BLOCKS`, `N_TILES`, …). Inline
the block-count expression (`dim // TILE`) at the loop header instead, so the
trip count and stride are visible right where the loop is — which is what you
need to reason about parallelism and pipelining.

```python
# ❌ named block count hides the trip count behind a constant
Q_BLOCKS = Q_HIDDEN // Q_OUT_STEP
for q in pl.spmd(Q_BLOCKS, name_hint="q_proj"):

# ✅ tile size named; block count inlined at the loop
Q_OUT_STEP = 128                                   # tiling parameter
for q in pl.spmd(Q_HIDDEN // Q_OUT_STEP, name_hint="q_proj"):
```

### Comments state what, not why

A comment states *what* a non-obvious line or block does, tersely — or there
is no comment. Do **not** explain *why* the code is written a certain way; the
one exception is a pointer to an **unresolved issue / workaround** (a filed
`pypto#NNNN` / `ptoas#NNNN` constraint), where the issue reference is the
comment. Do not write structural narration — no `# Stage 1:` / `# Stage 2:`,
`# Loop A:`, `# Bridge:` step labels; the loop and scope structure is already
visible from the code.

```python
# ❌ structural narration / rationale
# Stage 1: quant the activation so the cube can run int8 for speed
x_i8 = pl.cast(pl.mul(x, inv_scale), pl.INT8, mode="rint")

# ✅ no comment — the code is self-evident
x_i8 = pl.cast(pl.mul(x, inv_scale), pl.INT8, mode="rint")

# ✅ the allowed exception — an unresolved-issue workaround
# pad to 32: ptoas rejects cube tiles whose cols aren't a multiple of 16
w_pad = pl.slice(w, [K, 32], [0, 0], valid_shape=[K, MIX_HC])
```

### Declare allocations and views near their first use

Place `pl.create_tensor` and `pl.reshape` calls **immediately before the
first `pl.spmd` / `pl.parallel` / `pl.range` / `pl.at` block that
consumes the result** — do not hoist them to the top of a function or let
them drift far from the consuming loop. Co-locating an allocation with its
consumer makes the data-flow between orchestration and InCore easy to
trace without scrolling.

```python
# ❌ hoisted far from first use
kv_proj = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
score_proj = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
kv_flat = pl.reshape(kv, [T, HEAD_DIM])
...                          # many lines of unrelated code
for idx in pl.spmd(T * OUT_DIM // (B_TILE * OUT_TILE), name_hint="kv_score_proj"):
    kv_proj[...] = ...

# ✅ declared right above the consuming loop
kv_proj = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
score_proj = pl.create_tensor([T, OUT_DIM], dtype=pl.FP32)
kv_flat = pl.reshape(kv, [T, HEAD_DIM])
for idx in pl.spmd(T * OUT_DIM // (B_TILE * OUT_TILE), name_hint="kv_score_proj"):
    kv_proj[...] = ...
```

The same principle applies to inner-loop scratch tensors: allocate inside
the loop body (or directly above the inner `pl.at`) rather than at the
top of the outer loop.
