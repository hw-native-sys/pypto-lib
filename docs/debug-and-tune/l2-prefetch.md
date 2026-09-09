# L2 Prefetch (SDMA Cache Warm)

`pl.prefetch` starts an SDMA-backed pull of a global-memory region into L2 while
unrelated compute proceeds. It is a **pure cache hint**: the prefetch writes no
tensor, and deleting the scope changes no value in the program. That is what
makes it safe to tune aggressively — and it also means the only evidence that a
warm works is an end-to-end wall-time measurement, never a correctness result.

For the operator reference (types, verifier rules, runtime ownership) see PyPTO's
`docs/en/dev/ir/05-operators.md`, section *PrefetchOp: Asynchronous GM→L2
Prefetch*.

---

## When a warm pays off

Three properties must hold **together**. A candidate that misses any one of them
costs time instead of saving it.

1. **The region is cold at the point of use** — you can name what evicts it
   between two reads. If the data is already resident, the warm is pure overhead.
2. **The working set is fixed and statically shaped.** The prefetch source must
   be a flat, fully static GM region (see [Constraints](#constraints)), so the
   compiler knows at build time exactly what is warmed — weights, not
   data-dependent pages.
3. **Everything in flight fits L2 with room to spare** — 192 MiB on a2a3, summed
   across *every* warm alive at the same time.

### The canonical case: one decode layer's attention weights

A decode attention layer streams its whole weight set from HBM once per forward.
In a full forward that traffic is always cold: the MoE between two layers pushes
**427.8 MB** through L2, so nothing the previous attention layer read survives to
the next one. The weight set itself is fixed at compile time and small enough to
be resident — **157.9 MB** for CSA, **146.9 MB** for SWA/HCA, both under 192 MiB.
One warm per layer, covering the weight set the sweep settled on — every
projection the layer streams, in consumer-deadline order, minus the exclusions
below — buys **−2.10 %** on a full DeepSeek V4-Flash decode forward (fast rank p50
40132.0 → 39287.9 µs), with each attention block returning to its standalone
speed and MoE unaffected.

### Why the neighbouring candidates all fail

| Candidate | Verdict | Why |
|---|---|---|
| MoE expert weights | Never | 427.8 MB per layer is 2.2× L2 — the warm evicts itself. Worse, its SDMA contends with the all-to-all, which is also SDMA: warming `routed_w1` speeds `ffn` up (365.9 → 327.7 µs) but blows `combine` from 40.5 to 115.4 µs at ep8 |
| KV cache pages, gathered blocks | Impossible | Data-dependent addresses and non-flat shapes; the IR shape check rejects them |
| A standalone single-kernel case | Misleading | Its weights are already L2-resident from the previous benchmark round, so the warm measures a hit a real forward never gets |
| Compressor weights inside CSA (`cmp_wkv`, `cmp_wgate`) | Excluded | ~300 µs worse inside the otherwise-optimal set — coverage is not automatically good |

---

## The API

| DSL | Operands | Result |
|---|---|---|
| `pl.prefetch.make_context()` | — | Prefetch context (holds the UB scratch tile SDMA drives) |
| `pl.prefetch.async_prefetch(src, ctx)` | flat GM tensor, context | Async event |
| `pl.prefetch.session(ctx)` | context | Async session |
| `pl.prefetch.wait(evt, session)` | event, session | `BOOL` scalar; blocks until the region lands |

`TPREFETCH_ASYNC` carries no implicit wait-event synchronization — completion is
explicit through the event/session pair. A **cache warm does not wait**: the
model issues `make_context` + `async_prefetch` and lets the transfer run, because
the consumer reads the same GM addresses whether or not the warm has landed. Use
`session` / `wait` only when a kernel genuinely needs the region resident before
it proceeds.

### Constraints

- **Flat contiguous logical-1D GM source.** A fully static shape whose dimensions
  are all `1` except the last (`[N]`, `[1, N]`, `[1, 1, N]`). This mirrors the
  PTOAS `TPrefetchAsyncOp::verify()` check, so a shape mistake fails at PyPTO IR
  construction rather than deep in the backend. Reshape the weight to its flat
  view at the call site.
- **AIV-only.** The op drives SDMA from a Vec (UB) scratch tile, so it declares
  `CoreAffinity::VECTOR`; in a mixed kernel it stays on the vector lane.
- **Runtime support.** Execution reads the artifact's SDMA requirement and builds
  an enabled worker automatically — no workspace reaches any tensor signature. Only
  onboard a2a3 is covered: a platform without an SDMA provider (simulator, a5)
  fails during runtime initialization rather than degrading to a no-op.

---

## Writing one

From [decode_csa.py:217-237](../../models/deepseek_v4_flash_mtp/decode_csa.py#L217-L237) —
one scope, one context, every weight in that set, in the order its consumers
need them:

```python
x_normed_t = pl.create_tensor([T, D], dtype=pl.BF16)
rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
# SDMA CMO L2 warm of this layer's attention weights, in deadline order.
wq_a_flat = pl.reshape(wq_a, [D * Q_LORA])
wkv_flat = pl.reshape(wkv, [D * HEAD_DIM])
wq_b_flat = pl.reshape(wq_b, [Q_LORA * H * HEAD_DIM])
wo_a_flat = pl.reshape(wo_a, [O_GROUPS * O_LORA * O_GROUP_IN])
wo_b_flat = pl.reshape(wo_b, [D * O_GROUPS * O_LORA])
with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefetch_attn_w", deps=[rms_tid]):
    warm_ctx = pl.prefetch.make_context()
    pl.prefetch.async_prefetch(wq_a_flat, warm_ctx)
    pl.prefetch.async_prefetch(wkv_flat, warm_ctx)
    pl.prefetch.async_prefetch(wq_b_flat, warm_ctx)
    pl.prefetch.async_prefetch(wo_a_flat, warm_ctx)
    pl.prefetch.async_prefetch(wo_b_flat, warm_ctx)
```

The scope has no output, so `deps=` is the only thing placing it in time — see
[Choosing the anchor](#choosing-the-anchor).

---

## Three rules, each established by a negative result

### 1. One scope, one context

Put every `async_prefetch` of one warm inside a single `pl.at(CORE_GROUP)` with a
single `make_context()`. Splitting the same transfer across two scopes does land
on two AIVs and does run concurrently, but halves aggregate throughput:

```text
prefetch_wo_a  AIV_26  67.1 MB / 552.1 us = 122 GB/s
prefetch_wo_b  AIV_24  33.6 MB / 649.0 us =  52 GB/s
               aggregate 100.7 MB / 660.2 us = 153 GB/s
```

against **285 GB/s** for a single uncontended stream. The split buys one parallel
AIV start-up (~250 µs, paid once) and pays half the bandwidth for the whole
transfer — turning a −1.08 % win into a +0.66 % loss. Sharding across SDMA
channels scales negatively for the same reason.

### 2. All the weights, or none

The warm costs a near-fixed ~18–26 µs in the scope it is anchored to, regardless
of how much it covers, while the saving scales with coverage. So a partial warm
pays the cost and skips the benefit:

| Warm set | Δ vs no prefetch |
|---|---|
| `wo_a` alone | +0.41 % |
| `wo_b` alone | +0.83 % |
| `wq_b` alone | +1.7 % |
| All nine attention weights | **−2.38 %** (ep2) |

`wq_b` is *worse than nothing* on its own, yet inside the full set its own segment
drops below the no-prefetch baseline. **Evaluate a candidate by adding it to the
full set, never in isolation.**

### 3. The warm set must fit L2

Sum every warm in flight and compare against L2 (192 MiB on a2a3). Crossing it
flips the sign:

| Warm set | Bytes | vs L2 | p50 |
|---|---:|---:|---:|
| none | — | — | baseline |
| o-proj (`wo_a`+`wo_b`) | 100.7 MB | 0.50× | −1.09 % |
| + MoE gate (`routed_w1`) | 134.2 MB | 0.67× | −1.63 % |
| + MoE gate and up (`w1`+`w3`) | 268.4 MB | **1.33×** | **+3.03 %** |

Adding `w3` swings the result 4.7 %. Instrumentation confirms the cost is
self-eviction rather than bandwidth contention: the warm's own task time scales
with bytes (27.0 → 56.9 µs) while the consumer's span is unchanged — the warm
becomes pure overhead.

---

## Choosing the anchor

A warm scope has no data dependency on anything, so `deps=` is the only thing
that decides when it issues. Anchor it where the cores are **busy with the
previous stage**, so the transfer overlaps compute that is already running, and
early enough that it lands before the consumer needs it.

For DeepSeek V4-Flash decode attention, the measured optimum is the layer's
`rms_norm` task id. Layer entry, after `qproj_matmul`, and after the KV-cache
writeback all measured worse — at layer entry the cores are still occupied by
`hc_pre` and rope. When the natural anchor is a tensor rather than a task id,
hang the scope off a throwaway `pl.load` of whatever the previous stage
produced.

---

## Measuring a warm

- **`PYPTO_BENCH_WARMUP=20` is mandatory.** Each AIV pays a one-time SDMA channel
  set-up on its first prefetch (~150–500 µs). With ~48 channels and task
  placement varying between rounds, roughly 44 of the first 100 rounds are
  contaminated at the default warmup of 5. At warmup 20 every measured event
  lands in the 25–40 µs band.
- **Capture the warm on the swimlane** (`--enable-chip-swimlane`). The scope
  appears under its `name_hint`; dividing the warmed bytes by the task duration
  gives the achieved bandwidth, which is how the one-scope rule above was proved.
- **Quote median and mean.** A warm can carry a heavy tail — one measured MoE
  configuration kept only −1.04 % of its −1.63 % median once the p90 was included.
  On multiple cards, take the
  [fastest rank's median](performance-tuning.md#multi-card-l3-output).

### The benchmark loop flatters a warm

`PYPTO_BENCH` replays identical weights every round, so from round 2 the region
is already L2-resident and the warm is largely hitting its own leftovers — it
often returns in ~35 µs, an impossible 2.9 TB/s. Real serving is cold on every
forward. So the end-to-end delta is measured in an environment friendlier to the
warm than production, and it is diluted: a −2 % move on a 34 ms forward is a
handful of microseconds per layer buried in everything else that varies.

**Pair it with a task-timing slot**, which measures the stage that should have
gained instead of the whole forward:

1. Tag the consuming stage and the stage before it, then read finish-to-finish
   between the two slots with and without the warm.
2. Tag the warm scope too. Its window over the warmed byte count gives the
   achieved bandwidth — which separates "the transfer never happened" from "the
   transfer happened and did not help".
3. A `--runtime-dir` replay dispatches once, so its slot windows are a first-touch
   view rather than a steady-state one: closer to a serving step than round 50 of
   a bench loop.

Mechanism:
[Timing one stage of a full network](performance-tuning.md#timing-one-stage-of-a-full-network-task-timing-slots).
Report both numbers — the end-to-end delta decides whether it ships, the slot
window explains why it moved.

---

## See also

- [Performance Tuning](performance-tuning.md) — the benchmark loop, chip
  swimlane capture, and the L2 / L1 / L0 tuning rules
- [DeepSeek V4 Decode Optimization](../models/deepseek_v4_flash_mtp/decode_optimization.md) —
  the model change this guide generalizes, in the context of the whole decode path
- [Dependencies and Scheduling](dependency-and-scheduling.md) — how `deps=` places
  a scope with no data dependency
- [Save and Replay Golden Data](../run-and-validate/save-and-replay.md) — freeze
  the golden before sweeping warm sets
