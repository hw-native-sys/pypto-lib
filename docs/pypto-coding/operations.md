# Operations

The four op families a kernel body is written from: vector, cube, data movement,
and single scalars. They all run inside an InCore region — a `pl.at` block or a
`pl.spmd` body (see [Loops](loops.md)).

---

## Vector ops

Run on the vector unit, inside an InCore region (a `pl.at` block or a
`pl.spmd` body — see [Loops](loops.md)). Vector ops are the standard tools
for the cast /
activation / norm epilogue around a matmul, and for small standalone
reductions.

### Elementwise

Unified binary ops are `pl.add`, `pl.sub`, `pl.mul`, `pl.div`,
`pl.maximum`, and `pl.minimum`. They accept either a tensor/tile or a Python
`int`/`float`/`pl.Scalar` as the second operand and dispatch to the appropriate
tensor or tile operation. Unary ops are `pl.neg`, `pl.abs`, `pl.exp`, `pl.log`,
`pl.sqrt`, `pl.recip`, and `pl.rsqrt`. Activations are `pl.relu`, `pl.lrelu`,
and `pl.prelu`. Type conversion uses `pl.cast(x, target_type=...)`. Binary ops
broadcast over compatible shapes; prefer `pl.recip` + `pl.mul` over `pl.div`
on hot paths.

```python
silu_x = pl.mul(x, pl.recip(pl.add(pl.exp(pl.neg(x)), one)))   # x * sigmoid(x)
out_bf16 = pl.cast(acc_fp32, target_type=pl.BF16)
scaled = pl.mul(scores, attn_scale)                            # scalar mul
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
`pl.col_expand_add`, `pl.col_expand_sub`, `pl.col_expand_mul`,
`pl.col_expand_div`. Each maps to a single hardware broadcast op — use them
rather than reshaping a vector and relying on elementwise broadcast.

```python
# RMSNorm body: normed[i, j] = x[i, j] * inv_rms[i] * gamma[j]
normed = pl.col_expand_mul(pl.row_expand_mul(x, inv_rms), gamma)
```

### Fill and pad

`pl.full(shape, dtype=..., value=...)` allocates a scalar-filled
tensor/tile (typical use: zero-init a partial accumulator before a
reduction). `pl.fillpad(x, pad_value=...)` rewrites the padded tail of a
`valid_shape` slice with a sentinel — most often `pl.PadValue.min` to
mask out invalid positions before a softmax `row_max`. When explicit tile-level
in-place semantics are required, use `pl.tile.fillpad_inplace`; there is no
top-level `pl.fillpad_inplace`.

```python
partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
scores = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)   # -inf in tail
```

`pl.set_validshape(tile, valid_rows, valid_cols)` re-marks the valid
region of an **already-computed** tile. Where `valid_shape=` on a
`pl.slice` ([MTE ops](#mte-ops-data-movement-and-shape)) is a load-time marker
on data coming from GM,
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

A full runnable kernel is in
[examples/advanced/topk.py](../../examples/advanced/topk.py).

### Gather / scatter

#### Mask form — even / odd lanes

De-interleave and re-interleave even and odd lanes — the RoPE idiom.
`pl.gather(tile, mask_pattern=...)` selects alternate lanes of a
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

#### Index form — batched per-row gather

`pl.gather(src, dim=-1, index=idx)` gathers by an index tile along the last
axis: for a `[B, W]` source and a `[B, K]` INT32 index it returns `[B, K]`,
where row `i` picks `src[i, idx[i, :]]`. The index must be a real **tensor** (e.g. from
`pl.create_tensor`), not a `pl.full` tile — a tile index is rejected.

```python
gathered = pl.gather(local_scores, dim=-1, index=topk_idx_tile)   # [B, K]
```

---

## Cube ops

Matrix multiply primitives. They run on the cube unit, inside an InCore
region (a `pl.at` block or a `pl.spmd` body — see [Loops](loops.md)). A float
matmul
accumulates in FP32 and an INT8 one in INT32; `out_dtype` overrides the
default.

### `pl.matmul(lhs, rhs, out_dtype=None, a_trans=False, b_trans=False)`

Plain matmul. `a_trans` / `b_trans` transpose the corresponding operand
without a separate `pl.transpose`. `out_dtype` overrides the default FP32
accumulator dtype when set.

```python
out = pl.matmul(tile_a, tile_b)
out = pl.matmul(tile_a, tile_b, out_dtype=pl.FP32, b_trans=True)
```

### `pl.matmul_acc(acc, lhs, rhs, a_trans=False, b_trans=False, init_cond=None)`

Fused multiply-accumulate: `acc += lhs @ rhs`. Use this inside a K-loop to
keep the partial sum on chip. `init_cond` overwrites `acc` with `lhs @ rhs`
on the steps where the predicate holds instead of accumulating into it, so
one call covers the whole K-loop — the accumulator is never zeroed and the
first K step is not peeled:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="kproj"):
    acc = pl.create_tensor([M, N], dtype=pl.FP32)   # inside the region: a tile, not GM
    for kb in pl.pipeline(0, K // K_STEP, stage=2):
        k0 = kb * K_STEP
        tile_a = pl.slice(a, [M, K_STEP], [m0, k0])
        tile_b = pl.slice(b, [K_STEP, N], [k0, n0])
        acc = pl.matmul_acc(acc, tile_a, tile_b, init_cond=(kb == 0))
```

The predicate covers every operand shape `matmul_acc` accumulates, grouped
GEMMs included — a rank-3 weight slice that keeps its group axis takes
`init_cond` the same way:

```python
with pl.spmd(O_GROUPS, name_hint="proj_a_mm"):   # a pl.spmd body is InCore too
    acc = pl.create_tensor([1, M, N], dtype=pl.INT32)
    for kb in pl.pipeline(0, K // K_STEP, stage=2):
        k0 = kb * K_STEP
        tile_a = pl.slice(a, [M, K_STEP], [m0, k0])
        tile_b = w[g : g + 1, n0 : n0 + N, k0 : k0 + K_STEP]
        acc = pl.matmul_acc(acc, tile_a, tile_b, b_trans=True, init_cond=(kb == 0))
```

#### When `init_cond` alone is rejected

One case still needs the peel: when the left operand narrows to a **runtime**
row count *and* the accumulator is taller than one 16-row fractal. `mad` then
writes the product at pitch `ceil(validRow / 16) * 16`, while a
`create_tensor` accumulator is read back at its physical row count — only
`pl.matmul` stamps the accumulator compact, so `init_cond` alone fails
verification with:

```text
ERROR - AccCompactValid
  'tile.matmul_acc' accumulates valid_rows__ssa_v0 valid rows into an
  accumulator that is not compact (function 'narrowed'). mad lays the product
  out at a pitch of ceil(validRow/16)*16, but a non-compact accumulator is read
  back at its physical row count 64 [...]
```

A `valid_shape` whose extent is a compile-time constant, or an accumulator of
at most 16 rows, is unaffected: the pitch is the same either way.

### `pl.matmul_bias(lhs, rhs, bias)`

Matmul fused with a bias add: `lhs @ rhs + bias`. Cheaper than a separate
`pl.add` epilogue when the bias is broadcast over the M axis.

### Batched matmul

`pl.batch_matmul(lhs, rhs)` is the tile-level batched form for
`[B, M, K] @ [B, K, N]`. At tensor level, use `pl.matmul`: rank-greater-than-2
inputs are lowered to batched matmul automatically. Likewise, use
`pl.matmul_acc` for batched tensor accumulation. The explicit tile-only
accumulation API is `pl.tile.batch_matmul_acc`; there is no top-level
`pl.batch_matmul_acc`.

### `pl.gemv`, `pl.gemv_acc`, `pl.gemv_bias`

Vector-matrix specializations (1-row left operand). Prefer over `pl.matmul`
when M is 1 — the cube schedules the smaller form more efficiently.

---

## MTE ops (data movement and shape)

MTE primitives manipulate tensor views and stage data without explicit
load/store. The compiler decides where the actual TLOAD/TSTORE land based
on where each `pl.slice` / `pl.assemble` sits relative to `pl.at`.

### `pl.create_tensor(shape, dtype=...)` — where it goes decides what it is

Placement is not a style choice: a `create_tensor` in **orchestration** —
outside every InCore region — allocates a GM tensor, while one **inside** a
region yields a tile. Both a `pl.at` block and a `pl.spmd` body count as
InCore ([Loops](loops.md)).

Put it in orchestration when several regions cooperate to fill one
intermediate tensor — allocated once, then each region writes its piece via
`assemble`. When a single region's result flows straight to its caller
without further assembly, that GM tensor is not needed at all.

Put it inside a region for a region-local accumulator — the tile an
`init_cond` K-loop carries ([Cube ops](#cube-ops)). That one **must** be
allocated in the
region; hoisting it to orchestration makes it a GM tensor and ptoas rejects
the accumulator load:

```text
error: 'pto.tload' op expects A2/A3 tload dst to use loc=vec or loc=mat
```

```python
# Multi-stage assembly: q_proj is built by per-tile assembles
q_proj = pl.create_tensor([BATCH, Q_HIDDEN], dtype=pl.BF16)
for q0 in pl.parallel(0, Q_HIDDEN, Q_OUT_STEP):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
        ...
        q_proj = pl.assemble(q_proj, q_acc, [0, q0])
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

### `pl.load` / `pl.store` — explicit staging with a target memory

`pl.slice` lets the compiler decide where a tile lands. `pl.load` states it:

```python
norm_w_tile = pl.load(norm_w, [d0], [D_TILE], target_memory=pl.MemorySpace.Vec)
mix = pl.load(mixes, [t0, off], [T_TILE, HC_PAD], valid_shape=[T_TILE, HC_MULT])
pl.store(x_normed_valid, [tg, d0], x_normed)
```

Reach for them when the tile's memory space matters — forcing an operand into
`Vec` (UB) instead of `Mat` (L1), or the reverse — and for `atomic=` on a store.

**The argument order is the opposite of `pl.slice`'s**: `pl.load` takes
`(tensor, offsets, shapes)`, `pl.slice` takes `(tensor, shapes, offsets)`. Both
accept `valid_shape=`. `pl.store` is `(tile, offsets, tensor)`, with `shapes`
optional and `atomic=` for an accumulating write.

### `pl.reshape(x, new_shape)`

Logical reshape (view-only). Total element count must match. Legal both
inside and outside `pl.at`.

```python
flat = pl.reshape(q_chunk, [BATCH_TILE * H, D])
```

---

## Scalar ops (`pl.read` / `pl.write`)

Vector and cube ops work on whole tiles. When the kernel needs **one value** —
a routing index, a runtime row count, a slot cursor — read and write it
directly.

### `pl.read(src, offset)` / `pl.write(dst, offset, value)`

`pl.read` returns a `pl.Scalar`; `pl.write` stores one. Both dispatch on the
source kind: `src` / `dst` may be a **tensor** (GM) or a **tile** (UB), and
`offset` is either an index list (one entry per dimension) or a single flat
index. The explicit forms are `pl.tensor.read` / `pl.tensor.write` and
`pl.tile.read` / `pl.tile.write`.

```python
eid = pl.read(indices, [t, k])                     # GM tensor element -> Scalar
pl.write(recv_count_out, [e, 0], acc)              # Scalar -> GM tensor element
pl.tile.write(meta_tile, [0, e], cursor[e])        # Scalar -> tile lane
```

A scalar read of a **tensor** is also legal in orchestration, and that is how
a device-computed count becomes a host-side loop bound:

```python
for local_i in pl.parallel(N_LOCAL_EXPERTS):
    n_rows = pl.read(recv_expert_count, [local_i, 0])   # a kernel wrote it
```

Cast to `pl.INDEX` before using a read value as an offset or a loop bound —
the value comes back in its stored dtype (typically `INT32`):

```python
n = pl.cast(pl.read(recv_meta_local, [src, e]), pl.INDEX)
for slot in pl.range(n):
    ...
```

Keep scalar loops small. Each `pl.read` is a real memory access, so a
per-element scalar loop over a tile-sized region is orders of magnitude
slower than the vector op that does the same thing.

### On-core scalar arrays: `pl.array`

`pl.array.create(extent, dtype)` allocates a small array on the core's own
stack — integer, BOOL, or `pl.TASK_ID` elements, and `extent` must be a
compile-time constant. Index it with ordinary subscripts. Use it for
bookkeeping that must not round-trip through GM: cursors, per-destination
counters, or a list of TaskIds to fan a `deps=` in on.

```python
cursor = pl.array.create(N_RANKS * N_LOCAL, pl.INT32)
for d in pl.range(N_RANKS):
    for e in pl.range(N_LOCAL):
        cursor[d * N_LOCAL + e] = 0
...
cursor[dst * N_LOCAL + loc_e] = cursor[dst * N_LOCAL + loc_e] + 1

proj_tids = pl.array.create(O_GROUPS, pl.TASK_ID)   # orchestration: fan-in
```

Note the asymmetry with `pl.create_tensor`
([MTE ops](#mte-ops-data-movement-and-shape)): an array is core-local and
private to the block that created it, while a tensor created in orchestration
is GM and shared.

---

## See also

- [L2 Programming](l2-programming.md) — the regions these ops run inside.
- [Cube Tile Tuning](../debug-and-tune/cube-tile-tuning.md) — choosing the M / N / K
  fragment a `pl.matmul` works on.
- [Precision Tuning](../debug-and-tune/precision-tuning.md) — `pl.cast` rounding modes,
  dtype alignment, and where FP32 intermediates are required.
