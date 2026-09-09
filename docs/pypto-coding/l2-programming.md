# L2 Programming

How a single-chip (L2) kernel is declared: the two authoring forms, output
directions, `pl.at` regions and what goes inside one, and the rules that let a
single kernel serve a dynamic batch or sequence length.

```python
import pypto.language as pl
```

`pl` is the only accepted module alias.

A kernel can be written in **two parallel forms** — module-level `@pl.jit`
functions or a `@pl.program` class. Both lower through the same
compiler pipeline. Pick one per kernel and do not mix them: a `@pl.program`
method calling a `@pl.jit` kernel (or the reverse) is discouraged.

Either way, signatures look the same — tensor params are
`pl.Tensor[[shape...], dtype]`, outputs are wrapped in `pl.Out[...]`, scalars
are `pl.Scalar[dtype]` — and the function is written as an **opaque** function:
the frontend does not draw the InCore / Orchestration boundary explicitly.
Each compute region is wrapped in `with pl.at(level=pl.Level.CORE_GROUP, ...)`
and the compiler lowers that region to InCore; code outside any `pl.at` block
stays in orchestration (host / AICPU control flow). See
[`pl.at` scopes](#plat-scopes) below.

---

## Form A — module-level `@pl.jit`

The form used by most DeepSeek-V4 kernels: plain module-level functions.

`@pl.jit` decorates the top-level function the harness compiles and runs — the
boundary the golden test invokes; its `pl.Out` params are the kernel outputs.

### `@pl.jit.inline` — a sub-kernel spliced into its caller

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

### `@pl.jit.incore` — one InCore region as a whole kernel

`@pl.jit.incore` authors **one InCore region directly**: the body is a single
core's work, so there is no surrounding `pl.at` and `pl.tile.get_block_idx()`
gives the block index. Use it when the region is a whole kernel in its own right
— a distributed step, a hand-dispatched reduce — rather than one stage inside a
larger orchestration function.

```python
@pl.jit.incore
def reduce_step(inp, out, data, ...):
    local = pl.load(inp, [0, 0], [1, SIZE])
    data = pl.store(local, [0, 0], data)
    ...
    return pl.store(acc, [0, 0], out)
```

Dispatch it either by calling it from a `@pl.jit` entry (one block), or by
fanning it out with `pl.spmd` — the context-manager form
([Loops](loops.md)) or
`pl.spmd_submit(kernel, *args, core_num=N, deps=[...])`, which returns the
outputs and the region's TaskId.

---

## Form B — `@pl.program` class with `@pl.function` methods

A class groups related kernels as methods, with `type=` selecting how each one
lowers — `Opaque` for a self-contained compute kernel (below), `Orchestration`
for an entry that sequences other methods, `InCore` for a single region. No
tracked kernel is written this way today; new code should use Form A.

```python
@pl.program
class Qwen3Decode:
    @pl.function(type=pl.FunctionType.Opaque)
    def qwen3_decode(self, hidden_states: pl.Tensor[..., pl.BF16], ...):
        # orchestration code (loops, tensor allocation)
        for b0 in pl.parallel(0, BATCH, BATCH_TILE):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
                # InCore region — vector / cube / mte ops
                ...
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
                ...
        return out
```

---

## Output parameter direction: `pl.Out` / `pl.InOut`

Every tensor the golden test compares **must** declare an explicit direction on
the **orchestration entry** — the `@pl.jit` entry, its `@pl.jit.host` driver, or
the `@pl.function(type=Opaque)` / `Orchestration` method. A plain `pl.Tensor` is
treated as `In`: the runtime skips its device→host copy-back, so the tensor
**reads back as all-zeros on the host** and golden silently fails. The
annotation is the only place direction is declared — the harness reads it back
off the compiled artifact, so a `TensorSpec` never restates it:

| annotation | meaning | `TensorSpec` |
|------------|---------|--------------|
| `pl.Out[pl.Tensor[...]]` | pure output (write-only); validated | no `init_value` needed — the host buffer is not uploaded |
| `pl.InOut[pl.Tensor[...]]` | inout — read-modify-write (e.g. a paged KV cache the kernel reads history from and appends to; recurrent state); validated | `init_value` is the uploaded initial state |

Annotate the **entry only**. `@pl.jit.inline` sub-kernels keep bare `pl.Tensor`:
they are spliced at the call site before SSA conversion, so a parameter is
already an in-place alias of the caller's variable and the direction tag carries
no information — a `pl.Out` / `pl.InOut` wrapper on an inline param is stripped
and raises a `DeprecationWarning`. Entry `pl.InOut` paired with a bare-`pl.Tensor`
inline is the correct combination.

```python
@pl.jit
def attention_csa_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],          # In — plain
    kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM, ...], pl.BF16]],  # read old tokens + append new
    x_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],          # pure output
):
    attention_csa(x_hc, ..., kv_cache, x_out)           # inline params stay bare pl.Tensor
    return x_out
```

### A `pl.Out` region the kernel does not write is undefined

The runtime allocates a pure `pl.Out` buffer from the device pool: it neither
uploads the host placeholder nor zero-fills the buffer, so **every byte the
kernel does not write is allocator residue** — often zero on a2a3, garbage or
`NaN` on `a2a3sim`. The host `TensorSpec` is zero-filled, so a golden that leaves
that region at zero asserts a value the kernel never promised and passes or fails
by platform luck.

Pick one per output:

| The unwritten region is | Fix |
|---|---|
| padding past an active token count, and the kernel already zero-fills it (`hc_post`'s `hc_post_inactive_pad`, `gate`'s inactive-token zeroing) | nothing — keep `zero_tail=True` honest |
| a leading prefix's tail, with the boundary a fixture constant | `ratio_allclose(..., valid_rows=N, valid_axis=A)` |
| data-dependent (slot mappings, per-request conditions) | golden fills it `float("nan")`; comparator takes `ignore_nan=True` |
| something the test must still assert is untouched | make it `pl.InOut` with a zero `init_value`, so the host zeros reach the device |

An `InOut` has no such hole: its host contents are uploaded, so an unwritten
region reads back as whatever was sent.

---

## `pl.at` scopes

| Parameter | Required | Purpose |
|-----------|----------|---------|
| `level=pl.Level.CORE_GROUP` | yes | Lowering target. `CORE_GROUP` is the only level used in pypto-lib. |
| `name_hint="..."` | recommended | Stable label for the region. Appears in generated kernel filenames and profiling traces; aids per-region debugging. |
| `optimizations=[...]` | optional | Per-region codegen passes (see below). |
| `deps=[...]`, `allow_early_resolve=` | optional | Ordering edges the compiler cannot infer, and speculative dispatch — see [Dependencies and Scheduling](../debug-and-tune/dependency-and-scheduling.md). |

`pl.at` blocks may nest: an outer `pl.at` defining the InCore scope, with
inner `pl.at` blocks (each with its own `name_hint`) splitting it into named
sub-kernels.

---

## `pl.spmd` regions

`pl.at` is not the only way to open an InCore region. `pl.spmd(N)` opens one
too and dispatches `N` blocks of it in parallel, from a **single** AICPU
schedule entry instead of N successive dispatches:

```python
for q0 in pl.spmd(Q_HIDDEN // Q_OUT_STEP, name_hint="q_proj"):
    ...            # implicit InCore region; the loop variable is the block index
```

The two never nest: a `pl.spmd` body carries its own region, so it takes no
surrounding `pl.at` and rejects one. Everything else transfers — inside a
`pl.spmd` body `pl.create_tensor` yields a tile, the same op set applies, and
the same `optimizations`, `deps` and `allow_early_resolve` kwargs are accepted.

| Shape | Write |
|---|---|
| One region, or a few that want separate names | `pl.at`, one per region |
| `for … in pl.parallel:` wrapping a `with pl.at:` over many independent chunks | `pl.spmd` — one dispatch covers them all |

The two call forms, the argument shape and the full kwarg list are in
[Loops](loops.md#plspmd-parallel-spmd-dispatch); the dispatch-overhead
argument for reaching for it is in
[Performance Tuning](../debug-and-tune/performance-tuning.md#6-plspmd-for-parallel-sub-kernel-dispatch).

---

## Mixed kernel

A single InCore region — a `pl.at` block or a `pl.spmd` body — can hold both
cube (matmul) and vector (cast, add, row_sum, …) ops. The compiler assigns each op to its unit and pipelines the two
through an automatic cross-core pipe, so a projection and its epilogue cost one
kernel and one dispatch instead of two. This is the standard shape for every
projection in the repo — the `init_cond` K-loop of
[Cube ops](operations.md#cube-ops), then the vector epilogue,
then `assemble` back to GM:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
    q_acc = pl.create_tensor([BATCH, Q_OUT_STEP], dtype=pl.FP32)
    for kb in pl.pipeline(0, HIDDEN // K_STEP, stage=2):
        ...                                                    # cube: matmul_acc
    q_bf16 = pl.cast(q_acc, target_type=pl.BF16)               # vector
    q_proj = pl.assemble(q_proj, q_bf16, [0, q0])              # mte
```

Larger fused regions (RMSNorm + projection + residual) follow the same shape.
Two knobs belong to this region kind and no other: `pl.split` splits it so the
units ping-pong on the two halves, and `pl.cross_core_slot` sizes the pipe
between them (see [`optimizations`](#optimizations) below).

---

## `optimizations`

`optimizations=[...]` attaches per-region codegen passes to a `pl.at` block
(or a `pl.spmd` loop — same kwarg). Two entries are in use:

- **`pl.split(pl.SplitMode...)`** — split the region in half so the cube and
  vector units ping-pong on the two halves (cube on one half while vec runs
  the epilogue on the other). It applies **only to a mixed cube + vector
  region** ([Mixed kernel](#mixed-kernel)); a pure-cube or
  pure-vector region has nothing to ping-pong.
  The mode picks the axis:
  - `pl.SplitMode.NONE` — the default; no split.
  - `pl.SplitMode.UP_DOWN` — split vertically (rows / height halved).
  - `pl.SplitMode.LEFT_RIGHT` — split horizontally (cols / width halved).

  Reach for it when a region's unified buffer (UB) would otherwise exceed the
  per-core limit — typically a wide FP32 vector epilogue stacked on a matmul
  accumulator — since splitting also keeps the accumulator on-chip instead of
  spilling to a GM scratch round-trip.

- **`pl.cross_core_slot(slot_num=N)`** — ring depth of the automatic
  cube↔vector pipe. It sizes a channel; it does not partition work. Raising it
  lets the producing core run further ahead and costs UB, since the reserved
  buffer is `slot_size * slot_num`. It is ignored when the outlined scope has no
  cross-core ops, and the current default is 2 — so an explicit `slot_num=2`
  changes nothing.

  `pl.split(..., slot_num=N)` is the **deprecated** spelling of the same
  attribute and warns; write `pl.cross_core_slot(slot_num=N)` in new code.

```python
# split form on a mixed region whose FP32 epilogue would blow the UB budget
for ob in pl.spmd(INTER // INTER_TILE, name_hint="gate_up_silu",
                  optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
    ...

# shrink the cube->vector ring so its reserved buffer fits UB
with pl.at(level=pl.Level.CORE_GROUP, name_hint="indexer",
           optimizations=[pl.cross_core_slot(slot_num=2)]):
    ...
```

---

## Dynamic shapes

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

`pl.range`, `pl.parallel`, `pl.pipeline` and `pl.spmd` accept dynamic bounds
(`pl.unroll` does not — it unrolls a compile-time count). `pl.spmd` accepts a
single Scalar **or a composite dynamic expression** (`b_dim * HEAD_DIM //
HEAD_TILE`) as the block count. When an SPMD loop folds several dims into
one, **place the dynamic dim outermost** so every `//` and `%` divides by a
compile-time constant — otherwise the hot loop needs a runtime division:

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

---

## See also

- [Operations](operations.md) — the op families a region's body is written from.
- [Loops](loops.md) — the five constructs, and where each one is legal.
- [Dependencies and Scheduling](../debug-and-tune/dependency-and-scheduling.md)
  — what `deps=` and `allow_early_resolve=` do to the task graph.
- [Golden and Run](golden-and-run.md) — the validation the kernel ships with.
