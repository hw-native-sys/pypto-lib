# Writing Page Attention with PyPTO

Qwen3-14B · Decode · PyPTO

| Q heads | KV heads | Head dimension | Page size | Stack size |
| ---: | ---: | ---: | ---: | ---: |
| 40 | 8 | 128 | 128 tokens | 4 pages |

Qwen3-14B uses `paged_attention_pypto_swpipe` for Page Attention. The original
hand-written CCE implementation is still retained.

## 0. PA kernel skeleton { #step-0 }

| Stage | Work | Covered in |
| --- | --- | --- |
| 1. Prepare the current token | Apply Q/K norm and RoPE, then append K/V to the cache | Chapter 4 |
| 2. QK | Compute scores page by page on AIC | Chapter 5 |
| 3. Softmax | Update `m`, `l`, and `P` stack by stack on AIV | Chapter 6 |
| 4. PV and output | Compute `P @ V` on AIC, then accumulate and normalize on AIV | Chapters 7–9 |

```python title="paged_attention_pypto.py · skeleton"
@pl.jit.inline(auto_scope=False)
def paged_attention_pypto_swpipe(...):
    active_batch = pl.tensor.dim(seq_lens, 0)
    num_tasks = active_batch * 8

    with pl.spmd(24, sync_start=True, deps=[...]) as attn_tid:
        core = pl.tile.get_block_idx()

        # 1. Prepare the current token: Q/K norm, RoPE, and K/V cache append (Chapter 4).
        for aiv_id in pl.split_aiv(2):
            phase0(...)

        publish_gm_and_sync_mixed_cores()

        for task in pl.range(core, num_tasks, 24):
            # AIC loop
            for tick in pl.range(stack_count + 2):
                if tick < stack_count:
                    # 2. QK: compute scores for the current stack (Chapter 5).
                    produce_qk(stack=tick)
                if tick >= 2:
                    # 4. PV: compute P @ V for the stack from two iterations ago (Chapter 7).
                    consume_pv(stack=tick - 2)

            # AIV loop
            for aiv_id in pl.split_aiv(2):
                for tick in pl.range(stack_count + 2):
                    if tick < stack_count:
                        # 3. Softmax: update m/l and write P (Chapter 6).
                        online_softmax(stack=tick)
                    if tick >= 2:
                        # 4. Output: rescale O and accumulate P @ V (Chapters 8–9).
                        update_output(stack=tick - 2)

    return attn_tid
```

!!! tip

    The AIC and AIV loops execute concurrently on AIC and AIV. The three READY
    events in Chapter 8 decide which side waits and which side continues.

## 1. Derive the compute granularity from the formulas { #step-1 }

Decide what one task computes before writing any load or matmul.

### 1.1 Mathematical objective { #step-1-1 }

| Scores | Probability | Context |
| --- | --- | --- |
| `Q @ Kᵀ / √128` | `P = softmax(scores)` | `context = P @ V` |

### 1.2 Model dimensions determine the tiles { #step-1-2 }

| Known dimension | Kernel decision |
| --- | --- |
| 40 Q heads and 8 KV heads | One KV head serves five consecutive Q heads. |
| `head_dim = 128` | The cube K dimension for QK and PV is fixed at 128. |
| `page_size = 128` tokens | Load one `[128, 128]` K/V page at a time. |
| 4 pages | Combine them into one 512-token softmax/PV stack. |

Paged KV is not contiguous in GM. Before accessing `logical_page`, look up its
physical page:

```python title="Page lookup · logical to physical"
physical_page = block_table[batch, logical_page]
```

### 1.3 What one task computes { #step-1-3 }

!!! note

    **One task = one batch row + one KV head + its five Q heads.**

    The task traverses all historical K/V pages for that request and finally
    writes five context rows.

| Mapping | Meaning |
| --- | --- |
| `batch = task // 8` | Move to the next batch after every eight tasks. |
| `kv_head = task % 8` | Select the KV head; `q_head_begin = kv_head * 5`. |

Task 9 maps completely as follows in a two-row batch:

```text title="Task 9 · complete mapping · batch = 2 example"
task 9
→ batch        = 9 // 8 = 1
→ kv_head      = 9 %  8 = 1
→ q_head_begin = 1 *  5 = 5
→ handles batch 1, KV head 1, and Q heads 5–9
```

### 1.4 Task, page, and stack are different granularities { #step-1-4 }

**Figure 1. How one task is split into pages and how pages form stacks**

Task 9 selects `batch 1`, `KV head 1`, `Q heads 5–9`, with `seq_len = 700`.
It looks up `block_table` and takes only the KV-head-1 slice from each page:

| Logical page | Physical page | Page contents selected | Stack |
| ---: | ---: | --- | --- |
| L0 | P12 | KV head 1 of 8 | stack 0 |
| L1 | P3 | KV head 1 of 8 | stack 0 |
| L2 | P44 | KV head 1 of 8 | stack 0 |
| L3 | P8 | KV head 1 of 8 | stack 0 |
| L4 | P19 | KV head 1 of 8 | stack 1 |
| L5 | P5 | KV head 1 of 8 | stack 1 |

| Stack | Pages | Tokens |
| --- | --- | ---: |
| stack 0 | pages 0–3 | 512 |
| stack 1 | pages 4–5 | 188 valid tokens |

The task selects the batch and heads; the page determines the storage address;
the stack determines how many tokens one pipeline unit processes.

| Value | Shape |
| --- | --- |
| Q physical tile | `[16, 128]` |
| Q valid tile | `[5, 128]` |
| K/V page | `[128, 8, 128]` |
| Stack | at most 512 tokens |

!!! warning "What happens when task granularity is wrong"

    Splitting by individual Q head reloads the same K/V pages five times.
    Splitting by the whole batch reduces parallelism and enlarges the working
    set of each task.

## 2. Define the entry, helper, and scratch tensors { #step-2 }

The entry declares read/write directions and allocates GM intermediates. The
inline helper performs the mixed-core computation.

### 2.1 Declare directions only at the entry boundary { #step-2-1 }

```python title="Entry signature · directions live here"
key_cache:   pl.InOut[pl.Tensor]  # Read historical K, then append current K.
value_cache: pl.InOut[pl.Tensor]  # Read historical V, then append current V.
out:         pl.Out[pl.Tensor]    # Pure output.
```

The `@pl.jit.inline` helper uses plain `pl.Tensor` arguments. Its TaskIds only
enter `deps`; the standalone entry uses `task_dummy` as a placeholder.

### 2.2 The five intermediate tensors { #step-2-2 }

| Tensor | Shape / dtype | Writer → reader | Contents |
| --- | --- | --- | --- |
| `q_tnd_flat` | `[B×40, 128]` BF16 | AIV → AIC | Q after Q norm and RoPE, ready for QK |
| `score_transfer` | `[1152, 512]` FP32 | AIC → AIV | Unscaled QK scores for the current stack |
| `probability_transfer` | `[1152, 512]` BF16 | AIV → AIC | `exp(Sᵢ - mᵢ)`, not yet divided by the final accumulated denominator |
| `pv_transfer` | `[1152, 128]` FP32 | AIC → AIV | The `Pᵢ @ Vᵢ` numerator contribution from one stack |
| `ffts_workspace` | `[256]` INT64 | Synchronization primitive | Event control state; no Attention values |

```text title="Three-slot row layout · GM mailbox"
1152 rows = 24 cores × 3 ring slots × 16 physical rows

transfer_base = core * 3 * 16
slot_row      = transfer_base + (stack % 3) * 16
```

Each core owns three slots. Every slot reserves 16 rows for cube, but only the
first five rows are valid for one task. A later stack reuses the slot after its
consumer finishes.

### 2.3 Dynamic dimensions change only the descriptor { #step-2-3 }

```python title="Zero-copy GM views"
active_batch = pl.tensor.dim(seq_lens, 0)
num_tasks = active_batch * 8
cache_token_rows = numel(key_cache) // (8 * 128)

key_cache_bsnd = reshape(key_cache, [cache_token_rows, 8 * 128])
block_table_2d = reshape(block_table, [active_batch, max_blocks_per_seq])
q2d = reshape(q_tnd_flat, [active_batch * 40, 128])
```

!!! warning "Do not treat reshape as a copy"

    These operations create new tensor descriptors only. They do not move GM
    data, and the active batch must not be hard-coded to 16.

## 3. Build the mixed-core SPMD shell { #step-3 }

This chapter does not compute Attention. It establishes the AIC/AIV division
of work and assigns tasks to 24 cores.

### 3.1 AIC/AIV work on one core { #step-3-1 }

**Figure 2. Mixed-core swimlanes for the same core index**

| core N | Phase 0 | Barrier | Produce | Consume |
| --- | --- | --- | --- | --- |
| AIC | Wait / reach synchronization point | mixed barrier | `Q @ Kᵀ` | `P @ V` |
| GM + event | Publish Q/K/V | All participants arrive | scores / P | PV |
| AIV lane 0 | norm + RoPE | mixed barrier | softmax, 2 rows | update, 2 rows |
| AIV lane 1 | norm + RoPE | mixed barrier | softmax, 3 rows | update, 3 rows |

The AIC and AIV loops execute concurrently. Three READY events constrain the
ordering inside each stack.

`pl.split_aiv(2)` makes two AIV lanes execute one copy of the vector code each;
AIC executes the QK and PV matrix multiplications.

### 3.2 Launch 24 cores { #step-3-2 }

```python title="Mixed-core shell · core = 0…23"
with pl.spmd(
    24,
    name_hint="attn_swpipe_spmd",
    sync_start=True,
    allow_early_resolve=True,
    deps=[...],
) as attn_tid:
    core = pl.tile.get_block_idx()
    pl.system.set_ffts(ffts_workspace)
```

SPMD means that 24 cores run the same program. `get_block_idx()` returns one
index from 0 through 23 for each copy.

### 3.3 Assign tasks with a grid stride { #step-3-3 }

Each task is handled completely by one core. There are 24 cores, so each core
steps by 24 to claim its next task:

```python title="Task assignment · stride = 24 cores"
for task in pl.range(core, num_tasks, 24):
    ...
```

| Core | Tasks processed in order when `active_batch = 16` |
| ---: | --- |
| 0 | 0, 24, 48, 72, 96, 120 |
| 1 | 1, 25, 49, 73, 97, 121 |
| … | … |
| 23 | 23, 47, 71, 95, 119 |

With only two batch rows there are 16 tasks. Cores 0–15 process one task each;
cores 16–23 do not enter the Attention task loop, but they still participate in
the preceding Phase 0 and mixed barrier.

## 4. Implement Phase 0 and the GM barrier { #step-4 }

Prepare Q/K/V for the current decode token before the Attention body reads
them.

!!! note "What is a mixed barrier?"

    It is a synchronization point shared by AIC and AIV. After the AIVs that
    execute Phase 0 publish their Q/K/V GM writes, all cores may continue into
    the Attention body.

### 4.1 Assign Phase 0 work to AIV { #step-4-1 }

```python title="Phase-0 assignment · 32 physical AIV lanes"
for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
    rope_core = core * 2 + aiv_id
    if rope_core < 32:
        for it in pl.pipeline(4, stage=2):
            g_idx = rope_core + it * 32
            if g_idx < 8 * active_batch:
                kv_head = g_idx // active_batch
                batch_idx = g_idx % active_batch
```

`g_idx` covers all `8 × active_batch` `(kv_head, batch)` combinations. Each
combination prepares one K head, one V head, and the five Q heads that share
them.

### 4.2 Write Q, K, and V together { #step-4-2 }

| Path | Computation | Destination |
| --- | --- | --- |
| K | `k_proj × inv_rms → K head norm → RoPE` | Append BF16 to the current K-cache slot. |
| V | `v_proj × inv_rms` | Append BF16 to the current V-cache slot. |
| Q | Five heads → Q head norm → RoPE | Write BF16 to five rows of `q_tnd_flat`. |

| Value | Shape / dtype |
| --- | --- |
| Q compute tile | `[16, 128]`, five valid rows |
| K compute tile | `[8, 128]`, one valid row |
| Norm accumulation | FP32 |
| Q/K/V GM stores | BF16 |

The Q reduction is padded to 16 rows and the K reduction to 8 rows. Padding
rows are zero and do not affect the norms of valid rows.

### 4.3 Clamp the cache address before adding the layer base { #step-4-3 }

```python title="Cache append address · avoid crossing layers"
write_slot = pl.max(
    pl.cast(slot_mapping[batch_idx], pl.INDEX),
    0,
)
cache_row = layer_cache_base_token_rows + write_slot
```

The single-page serving profile warmup passes `-1`. Clamp the slot to zero
before adding the layer base. Reversing the order would make layer N write to
the final row of layer N−1.

### 4.4 Publish the GM writes { #step-4-4 }

```python title="Publish, arrive, and refetch · AIC + AIV"
pl.system.cacheinvalid()
pl.system.fence()
pl.system.syncall(core_type="mix")
pl.system.cacheinvalid()
```

1. `fence` publishes the Q/K/V GM writes from Phase 0.
2. `syncall` waits until all mixed cores reach the same synchronization point.
3. The second invalidation after the barrier prevents subsequent MTE2 loads
   from reading an old cache line.

!!! warning "syncall alone is not enough"

    `syncall` means that every participant has arrived. It does not publish
    writes or clear stale cache state at the reader automatically.

## 5. Implement the AIC QK producer { #step-5 }

One task fixes Q and the KV head, traverses logical pages, looks up each
physical K page, and computes scores.

### 5.1 Compute page/stack counts and load Q { #step-5-1 }

```python title="Task-local counts · AIC"
page_count = (seq_len + 127) // 128
stack_count = (seq_len + 511) // 512
qp_row = batch * 40 + kv_head * 5

q_tile = pl.load(
    q2d, [qp_row, 0], [16, 128],
    valid_shape=[5, 128], target_memory=pl.MemorySpace.Mat,
)
```

### 5.2 Map a logical page to a physical page { #step-5-2 }

```python title="Paged K load · GM to Mat/L1"
physical_page = pl.cast(
    block_table_2d[batch, logical_page], pl.INDEX,
)
k_page = pl.load(
    key_cache_bsnd,
    [cache_base + physical_page * 128, kv_head * 128],
    [128, 128], target_memory=pl.MemorySpace.Mat,
)
```

!!! tip "This lookup is the core of Paged Attention"

    `logical_page` expresses sequence order only. The actual GM address must use
    the `physical_page` returned by the block table.

### 5.3 Combine four score pages into one stack { #step-5-3 }

| Q | Transposed K page | Score page |
| --- | --- | --- |
| `[5, 128]` | `[128, 128]` | `[5, 128]` |

```python title="QK producer · one page at a time"
score_page = pl.matmul(
    q_tile,
    pl.tile.transpose_view(k_page),
    out_dtype=pl.FP32,
)
pl.store(
    score_page,
    [produce_row, page_offset * 128],
    score_transfer,
)

# Four pages are ready as one [16, 512] physical stack.
pl.system.sync_set(QK_READY_EVENT, core_type="aic", ...)
```

| Value | Shape / dtype |
| --- | --- |
| Q/K in Mat/L1 | BF16 |
| Physical score page | `[16, 128]` |
| Valid score page | `[5, 128]` |
| GM score stack | `[16, 512]` FP32 |

!!! warning "A wrong address reads another request's history"

    Computing a GM address directly from the logical page only works by
    accident with an identity block table. Random, reverse, and shared-prefix
    mappings fail immediately.

## 6. Implement AIV online softmax { #step-6 }

AIV does not retain scores for the entire sequence. It absorbs one stack of at
most 512 tokens at a time.

### 6.1 Split five rows between two lanes { #step-6-1 }

| AIV lane | Row offset | Valid rows | Physical tile |
| ---: | ---: | --- | --- |
| 0 | 0 | Q heads 0–1, two rows | `[8, 512]` |
| 1 | 2 | Q heads 2–4, three rows | `[8, 512]` |

### 6.2 Wait for scores, then mask the tail { #step-6-2 }

```python title="Score load and tail mask · AIV Vec"
pl.system.sync_wait(QK_READY_EVENT, core_type="aiv", ...)
scores = pl.load(score_transfer, [slot_row + lane_row, 0], ...)
scores = pl.tile.muls(scores, 1.0 / math.sqrt(128))

valid_cols = pl.min(512, seq_len - stack * 512)
scores = pl.set_validshape(scores, lane_rows, valid_cols)
scores = pl.fillpad(scores, pad_value=pl.PadValue.min)
```

When the final stack has fewer than 512 tokens, fill invalid columns with
negative infinity before `row_max` and `exp` so their probabilities are exactly
zero.

### 6.3 Merge stacks with running state { #step-6-3 }

Ordinary softmax needs the maximum and exponential sum of the complete row.
Online softmax turns them into state updated across stacks:

```text title="Online-softmax recurrence · i = current stack"
mᵢ = max(mᵢ₋₁, row_max(Sᵢ))
rᵢ = exp(mᵢ₋₁ - mᵢ)
Pᵢ = exp(Sᵢ - mᵢ)
lᵢ = rᵢ * lᵢ₋₁ + row_sum(Pᵢ)
```

| State | Meaning | Purpose |
| --- | --- | --- |
| `mᵢ` | Maximum seen so far | Keeps exponentiation stable. |
| `rᵢ` | Scale for the old state | Rescales the old accumulation when the maximum increases. |
| `lᵢ` | Exponential sum seen so far | Normalizes the result after all stacks finish. |

`m`, `l`, and `r` remain FP32. `Pᵢ` is cast to BF16 with round-to-nearest and
written to `probability_transfer`, after which AIV sets
`SOFTMAX_READY_EVENT`.

!!! warning "Pᵢ is not the final softmax"

    It has not yet been divided by the final `l`. Normalizing each stack
    independently and then summing the results gives incorrect probability
    ratios across stacks.

## 7. Implement the AIC PV consumer { #step-7 }

V pages can be moved into L1 first. AIC only needs to wait for AIV softmax
before reading P.

### 7.1 Prefetch four V pages before waiting { #step-7-1 }

```python title="Prefetch V before waiting for P · GM to Mat/L1"
pv_v_l1[0] <- V0
pv_v_l1[1] <- V1
pv_v_l1[2] <- V2
pv_v_l1[3] <- V3

pl.system.sync_wait(SOFTMAX_READY_EVENT, core_type="aic", ...)
```

V does not depend on the softmax result, so GM-to-L1 transfers can overlap AIV
softmax. After the wait returns, load P from `probability_transfer`.

### 7.2 Four V slots, two P slots, and one accumulator { #step-7-2 }

**Figure 3. V/P/accumulator lifetimes inside one stack**

| Resource | Page 0 | Page 1 | Page 2 | Page 3 |
| --- | --- | --- | --- | --- |
| V · L1 | slot 0 ← V0 | slot 1 ← V1 | slot 2 ← V2 | slot 3 ← V3 |
| Synchronization | V pages are all prefetched, then `wait(SOFTMAX_READY_EVENT)` |  |  |  |
| P · L1 | slot 0 ← P0 | slot 1 ← P1 | slot 0 ← P2 | slot 1 ← P3 |
| Acc · FP32 | `P0 @ V0` | `+ P1 @ V1` | `+ P2 @ V2` | `+ P3 @ V3` |

All four V pages must coexist, so V needs four slots. A P page can be
overwritten after use, so two slots ping-pong. All four pages share one FP32
accumulator.

### 7.3 Store only the final PV { #step-7-3 }

```python title="Four-page accumulation · AIC Acc"
pv0 = pl.matmul(P0, V0, out_dtype=pl.FP32)
pv1 = pl.matmul_acc(pv0, P1, V1)
pv2 = pl.matmul_acc(pv1, P2, V2)
pv3 = pl.matmul_acc(pv2, P3, V3)

pl.store(pv3, [consume_row, 0], pv_transfer)
pl.system.sync_set(PV_READY_EVENT, core_type="aic", ...)
```

| Value | Shape / dtype / memory |
| --- | --- |
| P page | `[16, 128]` BF16 Left |
| V page | `[128, 128]` BF16 Right |
| Accumulator | `[16, 128]` FP32 Acc |
| PV transfer | `[16, 128]` FP32 GM |

!!! warning "Do not store a partial PV for every page in GM"

    Accumulate all four pages directly in one FP32 accumulator. Per-page GM
    stores add traffic and interrupt the accumulator lifetime.

## 8. Connect the three stages into a pipeline { #step-8 }

QK, softmax, and PV/output do not wait for one stack to finish completely
before starting the next. Multiple stacks occupy different stages at once.

### 8.1 Pipeline design with two-iteration prelaunch { #step-8-1 }

The pipeline treats QK plus softmax as the producer and PV plus output update
as the consumer. After the producer handles stack `i`, the consumer handles the
same stack exactly two iterations later, allowing several stacks to occupy
different stages simultaneously.

```python title="Two-iteration prelaunch · logical pipeline"
for tick in pl.range(stack_count + 2):
    if tick < stack_count:
        produce(stack=tick)          # QK + softmax
    if tick >= 2:
        consume(stack=tick - 2)      # PV + output update
```

The first two iterations only launch S0 and S1. Each later iteration launches
the current stack and collects the stack from two iterations earlier. After
all stacks are launched, the final two iterations drain the remaining PV
results. The loop length is therefore `stack_count + 2`.

### 8.2 Align the three-slot ring, events, and rescale { #step-8-2 }

At `tick = 2`, S0, S1, and S2 are all in flight. Each transfer tensor therefore
needs three non-overlapping slots: `slot = stack % 3`.

**Figure 4. Iteration timeline for four stacks**

| Pipeline iteration | tick 0 | tick 1 | tick 2 | tick 3 | tick 4 | tick 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Stack launched | S0 | S1 | S2 | S3 | — | — |
| Launch slot | slot 0 | slot 1 | slot 2 | slot 0 | — | — |
| Stack collected | — | — | S0 | S1 | S2 | S3 |
| Collection slot | — | — | slot 0 | slot 1 | slot 2 | slot 0 |
| Output update uses | — | — | r0 · PV0 | r1 · PV1 | r2 · PV2 | r3 · PV3 |

```text title="Data handoff through three events"
AIC · QK
    └─ QK_READY → AIV · Softmax
                       └─ SOFTMAX_READY → AIC · PV
                                               └─ PV_READY → AIV · Update O
```

When tick 3 launches S3, S0 finished collection at tick 2, so S3 can safely
reuse slot 0. An event communicates only that data is ready; the data itself
remains in the transfer tensor.

### 8.3 The three READY events { #step-8-3 }

| Event | Set by | Waited by | Protected data |
| --- | --- | --- | --- |
| `QK_READY_EVENT` | AIC | AIV | FP32 scores |
| `SOFTMAX_READY_EVENT` | AIV | AIC | BF16 P |
| `PV_READY_EVENT` | AIC | AIV | FP32 `P @ V` |

### 8.4 Rescale is also delayed by two iterations { #step-8-4 }

`rᵢ` is produced when softmax processes Sᵢ, but the matching `PVᵢ` returns two
iterations later. Two pending slots realign them:

```text title="Two-entry rescale FIFO · AIV Vec FP32"
(pending0, pending1)
(1, 1) → (1, r0) → (r0, r1) → (r1, r2) → ...

O = pending0 * O + PVᵢ
```

For example, when tick 2 collects S0, `pending0 = r0`; when tick 3 collects S1,
`pending0 = r1`.

!!! warning "Ring slots and the rescale FIFO solve different problems"

    The ring prevents GM intermediates from overwriting one another. The FIFO
    ensures that a PV delayed by two iterations uses the rescale produced by
    the same stack.

## 9. Complete the tail, dynamic shapes, and final output { #step-9 }

A full stack contains four pages. The final stack may contain only one, two, or
three pages and must not access a missing block-table entry.

### 9.1 Use four fixed PV branches { #step-9-1 }

| Real pages | V slots | P slots | Cube computation |
| ---: | --- | --- | --- |
| 1 | 0 | 0 | `matmul` |
| 2 | 0, 1 | 0, 1 | `matmul + matmul_acc` |
| 3 | 0, 1, 2 | 0, 1, 0 | `matmul + 2×matmul_acc` |
| 4 | 0, 1, 2, 3 | 0, 1, 0, 1 | `matmul + 3×matmul_acc` |

QK and PV load only real pages. AIV then uses `valid_cols` to mask invalid
tokens in the last page. Missing pages trigger no block-table, K, V, or P load
and no corresponding matmul.

### 9.2 Dynamic requests still share one kernel { #step-9-2 }

- The batch comes from the `seq_lens` tensor descriptor.
- Each request reads its own `seq_len` and computes its own page and stack
  counts.
- Physical pages may be shuffled, noncontiguous, or shared as a prefix.
- `layer_cache_base_token_rows` selects the current layer in a packed
  multi-layer cache.

### 9.3 Normalize and store five context rows { #step-9-3 }

```python title="Final output · AIV Vec to GM"
context = pl.row_expand_mul(o, pl.recip(l_sum))
context = pl.cast(context, target_type=pl.BF16, mode="rint")
pl.store(
    context,
    [batch * 40 + kv_head * 5 + lane_row, 0],
    out2d,
)

return attn_tid
```

Lane 0 writes two rows and lane 1 writes three. The returned `attn_tid` lets the
caller constrain the subsequent output projection or the next scratch reuse.

| Value | Shape / dtype / memory |
| --- | --- |
| `o` | `[2/3, 128]` FP32 Vec |
| `l_sum` | `[2/3, 1]` FP32 |
| `out` | `[B×40, 128]` BF16 GM |

!!! warning "The tail has two boundaries"

    First avoid accesses to missing pages, then mask invalid tokens within the
    final page. Doing only one still causes an out-of-bounds access or includes
    padding in softmax.

## 10. Validate one boundary at a time { #step-10 }

First prove the math and paging are correct, then inspect the mixed-core
structure, and finally run on a real NPU.

### 10.1 Minimal boundary matrix { #step-10-1 }

| Case | Validation target | Direct expectation |
| --- | --- | --- |
| `seq_len = 1` | Smallest softmax | Context equals the selected V. |
| `127 / 128 / 129` | Page boundary | Results on both sides match the Torch golden. |
| `511 / 512 / 513` | Stack boundary | Pipeline fill/drain does not lose the last PV group. |
| 1 / 2 / 3-page tail | Fixed PV branches | Only real pages are accessed. |
| Ragged batch | Dynamic request lengths | Every batch row uses its own page and stack counts. |

### 10.2 Make the block table deliberately adversarial { #step-10-2 }

- Test random, noncontiguous, reverse, and shared-prefix mappings in turn.
- Use a cache canary to confirm that only the token addressed by
  `layer_base + slot_mapping` changes.
- Use a nonzero layer base to verify that clamping the warmup `-1` does not
  cross into the previous layer.

```bash title="Example checks · run from the repository root"
# First: compile the mixed kernel on the simulator.
python models/qwen3_14b/test_paged_attention_pypto.py \
  -p a2a3sim --compile-only --batch 2 --seq-lens 128,129

# Then: run one non-identity mapping on NPU device 0.
python models/qwen3_14b/test_paged_attention_pypto.py \
  -p a2a3 -d 0 --batch 2 --seq-lens 512,513 \
  --page-mapping reverse
```

### 10.3 Inspect structure, not only values { #step-10-3 }

1. Confirm that the mixed `syncall` lies between Phase 0 and the Attention
   body.
2. Confirm that the set/wait directions of the three READY events match
   Chapter 8.
3. Confirm four `pv_v_l1` slots, two `pv_p_l1` slots, and one FP32 accumulator.
4. On a real NPU, compare the output, K cache, and V cache with the Torch
   golden.

!!! success "Completion criteria"

    Boundary lengths, noncontiguous paging, a nonzero layer base, and the
    mixed-core structure must all pass before this Page Attention kernel is
    considered complete.

PyPTO · Qwen3-14B · `paged_attention_pypto_swpipe`
