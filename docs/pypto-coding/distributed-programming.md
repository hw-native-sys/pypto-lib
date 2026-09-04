# Distributed Programming

How a multi-card (L3) program is written in PyPTO-Lib: the host driver that
launches one orchestration per rank, the HCCL **window buffers** those ranks
address each other through, and the notify / wait protocol that orders the
traffic.

```python
import pypto.language as pl
import pypto.language.distributed as pld
```

`pld` is the only accepted alias for the distributed namespace, exactly as
`pl` is for the core one.

Read [PyPTO Coding Style](pypto-coding-style.md) first — everything there
about kernel forms, `pl.at` scopes, and loops applies unchanged inside a rank.
This page only adds what crosses the card boundary.

---

## 1. The shape of an L3 program

Three layers, and each `@pl.jit.*` kind has exactly one job:

| Layer | Decorator | Runs | Job |
|---|---|---|---|
| Host driver | `@pl.jit.host` | once, on the host | Allocate window buffers, loop over ranks, dispatch one orchestration per card |
| Per-rank entry | `@pl.jit` | once per card | Ordinary orchestration — the whole model step for that rank |
| Compute | `@pl.jit.inline` / `pl.at` / `pl.spmd` | on the cores | Kernels, including the comm ops |

The host driver is what makes it distributed:

```python
@pl.jit.host
def l3_allreduce(
    inputs: pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32],
    outputs: pl.Out[pl.Tensor[[N_RANKS, 1, SIZE], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer([1, SIZE], dtype=pl.FP32)
    signal_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, SIZE], dtype=pl.FP32)
        signal = pld.window(signal_buf, [N_RANKS, 1], dtype=pl.INT32)
        allreduce(inputs[r], outputs[r], data, signal, r, device=r)
```

Three rules follow from that loop:

- **Every host-driver tensor carries a leading rank axis** and is sliced
  `x[r]` at the call. The per-rank entry sees an ordinary `[T, D]` tensor.
- **`device=r` binds the call to a card.** It is a call-site kwarg on the
  per-rank entry, not part of its signature.
- **`pld.world_size()` is the trip count**, so one source serves EP2 / EP4 /
  EP8 without an edit.

A runnable version of exactly this is
[`examples/advanced/allreduce.py`](../../examples/advanced/allreduce.py).

---

## 2. Window buffers and `DistributedTensor`

A **window buffer** is a slot of HCCL symmetric memory: one
`alloc_window_buffer` call reserves the *same* region on *every* rank. That
symmetry is what makes `peer + offsets` addressing meaningful — an offset
names the same place in every card's copy.

Allocation is two-phase, and both phases live in the host driver:

| Call | Where | What it does |
|---|---|---|
| `pld.alloc_window_buffer(shape, dtype=...)` | HOST orchestration **only** (`@pl.jit.host`) | Reserves the per-rank region. Rejected inside InCore. |
| `pld.window(buf, shape, dtype=...)` | Same host driver, per rank | Types the region as a `DistributedTensor` view |

Use the **shape + dtype** overload so the shape, element type, and byte size
cannot drift apart. The canonical byte form
(`alloc_window_buffer(n_bytes)`) is the escape hatch for one buffer backing
several `window()` views of different shapes:

```python
# shape + dtype: the window() call below repeats the same two facts
recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
# byte form: only when one buffer backs several differently-shaped views
scratch_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
```

The view is passed into the rank kernel with a `pld.DistributedTensor`
annotation, which mirrors `pl.Tensor`:

```python
@pl.jit.inline
def dispatch(
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],                          # local
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],  # window
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],           # signal
    my_rank: pl.Scalar[pl.INT32],
):
```

Two properties worth internalizing:

- **Locally a window is just a tensor.** `recv_x[r0 : r0 + 1, :]`,
  `pl.read(recv_meta, [src, e])`, and `pl.load` all work on it, and the
  ordinary RAW edges apply to this rank's own writes. Only the *remote* side
  needs a `pld` op.
- **The buffer starts zeroed.** That is what lets a counter window be used as
  `AtomicAdd` + `Ge(1)` without an init pass.

### Lanes: make concurrent writers disjoint by construction

Nothing serializes two peers writing the same window rows. Give every
`(source, expert)` — or whatever the producing pair is — its own row range,
computed the same way on both sides:

```python
# lane (loc_e, my_rank, slot) on peer=dst
e_lane_base = loc_e * RECV_MAX + my_rank * MAX_PER_SRC
row = e_lane_base + slot
```

A window is not a queue: there is no allocator, no bounds check against
another rank, and a lane overrun silently corrupts a neighbour's rows.

---

## 3. Moving data

Five ops, split by *who holds the data* and *which direction it goes*:

| Op | Source | Destination | Use for |
|---|---|---|---|
| `pld.tensor.put(dst, peer, src, ...)` | local GM tensor | peer's window | Bulk rows already in GM — the default push |
| `pld.tensor.get(dst, peer, src, ...)` | peer's window | local GM tensor | Bulk pull |
| `pld.tile.remote_store(tile, target, peer, offsets)` | a tile on chip | peer's window | A value just computed, without a GM round-trip |
| `pld.tile.remote_load(target, peer, offsets, shape)` | peer's window | a tile on chip | Pull straight into the consuming kernel |
| `pld.tensor.remote_store(src, target, peer, offsets)` | local GM tensor | peer's window | Tensor-level form of the above |

All of them take `peer` as a **global rank index**. For a sub-group (a TP
group inside a larger world) address it as `group_base + local_index`.

```python
pld.tensor.put(                                   # GM -> peer GM, one token row
    dst=recv_x, peer=dst, src=x_norm_i8,
    dst_offsets=[row, 0], src_offsets=[t, 0], shape=[1, D],
)
pld.tile.remote_store(aux_tile, target=recv_aux, peer=dst, offsets=[row, 0])
recv = pld.tile.remote_load(data, peer=peer, offsets=[0, 0], shape=[1, SIZE])
```

`dst_offsets` / `src_offsets` / `shape` are all-or-nothing: pass all three to
narrow the transfer, or none to move the whole slice.

**Combine instead of overwrite** with `atomic=pld.AtomicType.Add`, available
on `put` and both `remote_store` forms. A BF16 destination is A2/A3 only; on
A5 use an FP32 window and cast after the reduction.

**Large transfers** need staging control. `put` / `get` move through a VEC
staging tile, so a transfer wider than UB must be chunked:
`chunk_rows=` / `chunk_cols=` size that tile and let the transfer
auto-chunk through it, and `pipeline=True` (both chunk args required)
double-buffers so a chunk is on the wire while the next one loads.

```python
pld.tensor.put(
    dst=hidden_window, peer=peer, src=local_hidden_tail,
    dst_offsets=[dst_row, 0], src_offsets=[src_row, 0], shape=[TAIL_ROWS, D],
    chunk_rows=ROW_TILE, chunk_cols=D, pipeline=True,
)
```

---

## 4. Synchronization

Data movement carries **no ordering of its own**. Every cross-rank
dependency is a signal window plus a notify / wait pair.

```python
pld.system.notify(target=arrived, peer=dst, offsets=[my_rank, 0],
                  value=1, op=pld.NotifyOp.AtomicAdd)   # bump my slot on peer
pld.system.wait(signal=arrived, offsets=[src, 0],
                expected=epoch, cmp=pld.WaitCmp.Ge)     # block on my local slot
```

Note the asymmetry: **`notify` writes a remote slot, `wait` reads the local
one.** The conventional layout is a `[N_RANKS, 1]` INT32 window where row
`s` is "what source `s` has told me".

| Enum | Values | Pick |
|---|---|---|
| `pld.NotifyOp` | `AtomicAdd`, `Set` | `AtomicAdd` — see below |
| `pld.WaitCmp` | `Ge`, `Eq` | `Ge` — pairs with `AtomicAdd` |

### Use monotonic counters with an epoch, not set-and-clear

A signal slot that is set to 1 and cleared has to be cleared by someone, and
a late notify from the previous call lands in the next call's slot. Instead,
never reset: each call bumps the counter by a fixed amount and waits for a
threshold that grows with the call index.

```python
# 1-based call id; the window is monotonic so waits use `>= epoch`
def dispatch(..., moe_epoch: pl.Scalar[pl.INT32]):
    ...
    pld.system.notify(target=arrived, peer=dst, offsets=[my_rank, 0],
                      value=1, op=pld.NotifyOp.AtomicAdd)
    pld.system.wait(signal=arrived, offsets=[src, 0],
                    expected=moe_epoch, cmp=pld.WaitCmp.Ge)
```

`AtomicAdd` is order-independent, so a notify that arrives early or late
cannot clobber anything — the count only has to *reach* the threshold. This
is what lets one comm kernel be called once per layer with nothing to reset
between layers.

When the notify is folded into a fanned-out push (below), every block
signals, so the threshold scales with it: `expected = epoch * n_blocks`.

### One signal window per phase

Two phases that can overlap need two windows. In the MoE dispatch the
route-count metadata rides `arrived` and the bulk payload rides
`data_arrived`, so a rank can start its cumsum on the metadata while the
payload is still in flight. Sharing one window would serialize them.

### Reusing a window across epochs needs a second counter

A monotonic `ready` counter says "my data is there". It does *not* say the
peer is finished reading the previous epoch's data out of that same region.
Overwriting a window in a loop takes a **two-counter** protocol:

```python
epoch_value = pl.cast(comm_epoch + 1, pl.INT32)

for peer in pl.range(CP_SIZE):                 # 1. peers read epoch n-1 out
    if peer != my_rank:
        pld.system.wait(signal=consumed, offsets=[peer, 0],
                        expected=comm_epoch, cmp=pld.WaitCmp.Ge)

for peer in pl.range(CP_SIZE):                 # 2. publish epoch n, announce
    pld.tensor.put(dst=hidden_window, peer=peer, src=local_tail, ...)
for peer in pl.range(CP_SIZE):
    if peer != my_rank:
        pld.system.notify(target=ready, peer=peer, offsets=[my_rank, 0],
                          value=1, op=pld.NotifyOp.AtomicAdd)

for seg in pl.range(NUM_SEGMENTS):             # 3. consume, then release
    if owner != my_rank:
        pld.system.wait(signal=ready, offsets=[owner, 0],
                        expected=epoch_value, cmp=pld.WaitCmp.Ge)
    ...                                        # read hidden_window locally
for peer in pl.range(CP_SIZE):
    if peer != my_rank:
        pld.system.notify(target=consumed, peer=peer, offsets=[my_rank, 0],
                          value=1, op=pld.NotifyOp.AtomicAdd)
```

The context-parallel prefill exchange in
[`models/deepseek_v4_flash_mtp/prefill_cp_exchange.py`](../../models/deepseek_v4_flash_mtp/prefill_cp_exchange.py)
is this protocol in full.

### `defer_wait` — a wait that does not hold a core

`pld.system.wait` spins on the device: the task occupies its core group until
the condition holds. `pld.system.defer_wait(signal, offsets, expected,
cmp=pld.WaitCmp.Ge)` instead leaves the task's TaskId incomplete and lets the
core go. It never resumes the kernel, so everything after the condition must
live in a dependent task.

---

## 5. Scheduling rules

Comm is ordinary task-graph work, and the rules in
[Dependencies and Scheduling](../debug-and-tune/dependency-and-scheduling.md)
apply. Four that are specific to it:

**Comm ops need an InCore scope.** `put` / `remote_store` / `notify` / `wait`
must sit inside a `pl.at(level=pl.Level.CORE_GROUP, ...)` or a `pl.spmd`
body. In orchestration the callee cannot be resolved.

**Anchor a spinning wait.** A wait scope whose arguments create no dependency
edge is ready at submit, so the scheduler dispatches it immediately and it
spins holding a core group for the whole phase. Give it a `deps=` on the
matching push, and read something the producer wrote so the edge is real:

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="dispatch_wait",
           deps=[_push_tid]) as _wait_tid:
    _idx_anchor = pl.read(indices, [0, 0])      # anchor, not data
    for src in pl.range(N_RANKS):
        ...
```

**Keep the wait off the early-dispatch chain.** A speculatively dispatched
wait reserves cores the push itself needs. Put `allow_early_resolve=False` on
the scope that must not pre-empt the traffic, and split "notify" from "wait"
so the notify can ride inside the push scope while the wait is a separate,
`deps=`-anchored task.

**Your own writes ride local edges.** A rank's `peer == my_rank` put is an
ordinary local write to the window tensor, so the RAW edge already orders it
against a later local read. The cross-rank wait covers only the *other*
ranks' writes.

Pass a TaskId across an inline boundary with `pl.Scalar[pl.TASK_ID]` when a
later stage must fence against an earlier stage's traffic.

### Visibility: `put` and `remote_store` are not equivalent

A `notify` issued after a `pld.tensor.put` in program order on the same core
is ordered behind it. `remote_store` is **non-draining**, and a local
`PIPE_ALL` barrier is not a cross-rank DDR fence (`ptoas#872`) — so a peer
that observed the notify has not necessarily observed the `remote_store`
payload. Where the handshake must cover the payload, move it with `put`, or
keep the notify in the same block that issued the stores and accept the
program-order guarantee only.

---

## 6. Collectives

`pld.tensor` also exposes `allgather`, `all_to_all`, `all_to_all_v`,
`allreduce`, `reduce_scatter`, `broadcast`, and `barrier`, each taking a
target window plus a signal window. **No kernel in this repository uses
them**: every model builds its exchange from `put` / `remote_store` +
notify / wait, because the pattern is fused into the surrounding compute
(the notify rides inside the push scope, the gather is the consumer kernel)
rather than standing alone as a collective. Reach for a collective only when
the exchange really is a standalone whole-window operation, and verify its
behaviour on the target before building on it.

---

## 7. Running and validating

The harness compiles an L3 program when `config` carries a
`DistributedConfig`:

```python
from pypto.ir.distributed_compiled_program import DistributedConfig

result = run(
    fn=l3_moe,                                  # the @pl.jit.host driver
    specs=build_tensor_specs(),
    golden_fn=golden_moe,
    config=dict(
        platform=args.platform,
        distributed_config=DistributedConfig(device_ids=device_ids),
    ),
)
```

- **CLI**: distributed entries take `-d 0,1,...` (a comma-separated list),
  not a single integer, and usually an `--ep` / `--tp` degree.
- **CI**: declare the card count with a `# ci: devices=N` marker near the top
  of the file; the real-NPU job borrows that many cards. See
  [Platforms and Devices](../get-started/platforms.md).
- **Specs**: every tensor keeps its leading rank axis, so a `TensorSpec` is
  `[N_RANKS, ...]`. `resident="stacked"` uploads shard `i` to card `i` once
  instead of per dispatch — see [`golden/spec.py`](../../golden/spec.py).
- **Golden**: the reference is computed on the host over all ranks at once.
  Write it so each rank's slice depends only on that rank's inputs where the
  algorithm allows, which keeps the reference independent of the device's
  packing order.

Per-rank timing, start skew, and the fastest-rank convention are in
[Performance Tuning](../debug-and-tune/performance-tuning.md#multi-card-l3-output).

---

## See also

- [PyPTO Coding Style](pypto-coding-style.md) — kernel forms, scopes, loops.
- [Dependencies and Scheduling](../debug-and-tune/dependency-and-scheduling.md)
  — task edges, early dispatch, and why an unanchored wait costs a core.
- [Ring Heap and Scope Stats](../debug-and-tune/ring-heap-and-scope-stats.md)
  — the ring resources an L3 program's intermediates land in.
- [`examples/advanced/allreduce.py`](../../examples/advanced/allreduce.py) —
  the smallest complete L3 program.
- [DeepSeek V4-Flash MTP](../models/deepseek_v4_flash_mtp/index.md) — the MoE
  dispatch / combine and LM-head exchanges this page is drawn from.
