# Loops

PyPTO has five loop constructs — `pl.range`, `pl.unroll`, `pl.parallel`,
`pl.pipeline` and `pl.spmd`. The choice between them is **semantic**: it tells
the compiler what scheduling and codegen are valid.

## Where each construct is legal

Each construct has a fixed placement relative to `pl.at`:

| Construct | Outside `pl.at` (orchestration) | Inside `pl.at` (InCore) |
|-----------|:-:|:-:|
| `pl.range` | yes | yes |
| `pl.unroll` | yes | yes |
| `pl.parallel` | yes | no |
| `pl.pipeline` | no | yes |
| `pl.spmd` | yes (body is implicitly InCore) | no |

`pl.parallel` distributes iterations across cores, so it must sit in
orchestration. `pl.pipeline` software-pipelines stages within a single
InCore region, so it must sit inside `pl.at`. `pl.range` is a plain
sequential loop and is legal in either place; `pl.unroll` is the same loop
unrolled at compile time. `pl.spmd` is a parallel SPMD loop that bundles its
own InCore region — see below.

## Argument shape

`pl.range`, `pl.unroll`, `pl.parallel`, and `pl.pipeline` share the same
positional-arg shape, mirroring Python's `range`:

```text
pl.<loop>(stop)
pl.<loop>(start, stop)
pl.<loop>(start, stop, step)
```

Each argument may be either a Python `int` or a `pl.Scalar`.

## `pl.range` — sequential

Iterations execute in strict order. Loop-carried dependencies are allowed.

```text
for kb in pl.range(HIDDEN // K_STEP):           # range [0, HIDDEN // K_STEP)
for kb in pl.range(1, hidden // K_STEP):        # range [1, hidden // K_STEP)
for k0 in pl.range(0, HIDDEN, K_STEP):          # start, stop, step
```

To carry state across iterations, pass `init_values=` and unpack the loop
variable as a `(idx, (state...))` tuple:

```python
for i, (acc,) in pl.range(N, init_values=(zero,)):
    acc = pl.add(acc, x[i])
```

## `pl.unroll` — sequential, unrolled at compile time

Same semantics as `pl.range` at run time; the parser emits `ForKind.Unroll`
instead of `ForKind.Sequential`, so the body is replicated per iteration
rather than looped. Use it for a short trip count whose body is small — the
loop overhead and the per-iteration index math disappear — and keep
`pl.range` for anything long enough that unrolling would bloat the kernel.

`start`, `stop` and `step` must be compile-time constant integers, and
`init_values=` is rejected — carry state with `pl.range` instead. The parser
rejects both cases outright rather than failing later in the unroll pass.

```python
for kb in pl.unroll(1, K // K_STEP):          # replicated, no loop overhead
    ...
```

## `pl.parallel` — independent iterations

Iterations are guaranteed independent — the compiler may split, reorder, or
schedule them across cores. Same arg shape and `init_values` support as
`pl.range`.

```text
for b in pl.parallel(BATCH):                          # short form: extent only
for b0 in pl.parallel(0, BATCH, BATCH_TILE):          # start, stop, step
```

## `pl.pipeline` — software-pipelined sequential

Sequential like `pl.range`, but the compiler software-pipelines successive
iterations across compute and memory units. `stage=N` (required keyword) is
the pipeline depth — the loop body is replicated `stage` times for
ping-pong buffering; the outer trip count advances in strides of
`stage * step` and a tail dispatch covers the remainder when the trip count
is not divisible by `stage`. Typical values are 2 or 4.

```text
for kb in pl.pipeline(HIDDEN // K_STEP, stage=2):
for kb in pl.pipeline(2, HIDDEN // K_STEP, stage=2):  # start at 2
for kb in pl.pipeline(0, hidden // K_STEP, stage=4):
```

`init_values=` is supported, with the same `(idx, (state...))` unpacking as
`pl.range`. Use `pl.pipeline` for the inner reduction loop of a matmul (the
K loop) — each iteration loads a new tile of the left/right operand and
accumulates into the same output.

## `pl.spmd` — parallel SPMD dispatch

`pl.spmd(core_num)` dispatches `core_num` blocks in parallel; iteration
starts at 0 and steps by 1 (only the block count is positional, **not**
start/stop/step). Two forms:

### Loop form

The body is auto-outlined into a synthetic InCore function and the iteration
variable binds the per-block index (equivalent to
`pl.tile.get_block_idx()`). No surrounding `pl.at` is needed or allowed:

```python
for ob0 in pl.spmd(Q_HIDDEN // (Q_OUT_STEP * 4), name_hint="q_proj"):
    # implicit InCore region — vector / cube / mte ops here
    for ob in pl.range(ob0 * 4, (ob0 + 1) * 4):
        q0 = ob * Q_OUT_STEP
        ...
```

### Context-manager form

The body must be a single call to a pre-defined `@pl.jit.incore` kernel:

```python
with pl.spmd(4):
    out = self.kernel(a, b, out)
```

### Keyword args

| Kwarg | Default | Purpose |
|-------|---------|---------|
| `name_hint` | `""` | Stable label for the outlined function. |
| `sync_start` | `False` | If True, all blocks start execution simultaneously. |
| `optimizations` | `None` | Same entries as `pl.at` — see [L2 Programming](l2-programming.md#optimizations). |
| `deps`, `allow_early_resolve` | — | Same as `pl.at` — see [Dependencies and Scheduling](../debug-and-tune/dependency-and-scheduling.md). |

---

## See also

- [L2 Programming](l2-programming.md) — `pl.at` and `pl.spmd` regions, and the
  `optimizations` these loops accept.
- [Performance Tuning](../debug-and-tune/performance-tuning.md) — picking a construct
  against what the swimlane shows: dispatch trails, idle cores, a serialized lane.
