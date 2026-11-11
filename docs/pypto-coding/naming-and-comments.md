# Naming and Comment Conventions

What a constant may be called, what a comment may say, and where an allocation
goes. These apply to every kernel in the repository.

## Name tile sizes, inline block counts

Give a named constant to a **tiling parameter** — the tile / step size — but
**not** to a derived block count (`K_BLOCKS`, `Q_BLOCKS`, `N_TILES`, …). Inline
the block-count expression (`dim // TILE`) at the loop header instead, so the
trip count and stride are visible right where the loop is — which is what you
need to reason about parallelism and pipelining.

```python
# ❌ named block count hides the trip count behind a constant
Q_BLOCKS = Q_HIDDEN // Q_OUT_STEP
for q in pl.spmd(Q_BLOCKS, name_hint="q_proj"):
    ...

# ✅ tile size named; block count inlined at the loop
Q_OUT_STEP = 128                                   # tiling parameter
for q in pl.spmd(Q_HIDDEN // Q_OUT_STEP, name_hint="q_proj"):
    ...
```

## Comments state what, not why

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
# peel the first K step: matmul_acc from a zero carry trips TLOAD DN->NZ (pypto#1540)
acc = pl.matmul(tile_a, tile_b)
```

## Declare allocations and views near their first use

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

---

## See also

- [`examples/`](../../examples/) — small kernels written to these conventions.
- [L2 Programming](l2-programming.md) — the structures the names describe.
