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

### The canonical case: one decode layer's o-projection weights

A decode attention layer streams its whole weight set from HBM once per forward.
In a full forward that traffic is always cold: the MoE between two layers pushes
**427.8 MB** through L2, so nothing the previous attention layer read survives to
the next one. The weight set itself is fixed at compile time and small enough to
be resident. But only the part the layer consumes *last* can be warmed in time:
the o-projection pair `wo_a` + `wo_b` (100.7 MB). One warm per layer, covering
exactly that pair and issued once the layer's q projection has written `q`, buys
**−2.90 %** at ep2 and **−2.73 %** at ep8 on a full DeepSeek V4-Flash decode forward
against no warm (fast-rank p50 at warmup 500: ep2 34423.7 → 33424.2 µs, ep8
35685.9 → 34711.7 µs). Each attention block lands within a few microseconds of its
standalone speed, and MoE is unaffected.

### Why the neighbouring candidates all fail

| Candidate | Verdict | Why |
|---|---|---|
| MoE expert weights | Never | 427.8 MB per layer is 2.2× L2 — the warm evicts itself. Worse, its SDMA contends with the all-to-all, which is also SDMA: warming `routed_w1` speeds `ffn` up (365.9 → 327.7 µs) but blows `combine` from 40.5 to 115.4 µs at ep8 |
| KV cache pages, gathered blocks | Impossible | Data-dependent addresses and non-flat shapes; the IR shape check rejects them |
| A standalone single-kernel case | Misleading | Its weights are already L2-resident from the previous benchmark round, so the warm measures a hit a real forward never gets |
| Compressor weights inside CSA (`cmp_wkv`, `cmp_wgate`) | Excluded | ~300 µs worse when added to a warm of every other attention weight — coverage is not automatically good |
| Q/KV projection and CSA indexer weights (`wq_a`, `wkv`, `wq_b`, `idx_wq_b`, …) | Excluded | Consumed within microseconds of `rms_norm`, so a warm cannot land first. Adding them as a second warm at layer entry or at `rms_norm` gives back 625–673 µs of the o-projection warm's 830 µs gain |

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

From [decode_csa.py:225-236](../../models/deepseek_v4_flash_mtp/decode_csa.py#L225-L236) —
one scope, one context, the o-projection weights in the order their consumers
need them, anchored on the task that writes `q`:

```python
q_rope_tid = qkv_proj_rope(
    x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
    rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
    q, kv, qr, qr_scale, late_dep,
)
# SDMA CMO L2 warm of the o-projection weights, issued once q is written.
wo_a_flat = pl.reshape(wo_a, [O_GROUPS * O_LORA * O_GROUP_IN])
wo_b_flat = pl.reshape(wo_b, [D * O_GROUPS * O_LORA])
with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefetch_o_proj_w", deps=[q_rope_tid]):
    warm_ctx = pl.prefetch.make_context()
    pl.prefetch.async_prefetch(wo_a_flat, warm_ctx)
    pl.prefetch.async_prefetch(wo_b_flat, warm_ctx)
```

The scope writes nothing, so `deps=` is the only thing placing it in time.
`qkv_proj_rope` returns the task id of its final q stage for exactly this. A
throwaway `pl.read(q, ...)` would express the same dependency, but it adds `q`
as a task input for no other reason — see
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

### 2. Warm what lands before its consumer, and all of it

A warm helps only the weights it lands before their consumer reads them, and
whatever it moves late competes with the layer's own HBM streams. The q/kv
projections read their weights right after `rms_norm`, so no warm beats them
there. The o-projection runs last, so its weights are the ones with time to land.
Within that pair, partial coverage loses. Fast-rank p50 at ep2, warmup 20, against no
warm (34575 µs):

| Warm set (one scope, issued once `q` is written) | Δ vs no prefetch |
|---|---:|
| `wo_b` alone | −399 µs |
| `wo_b`, then `wo_a` | −595 µs |
| `wo_a` alone | −692 µs |
| **`wo_a`, then `wo_b`** | **−830 µs** (−2.4 %) |
| Every attention weight, issued at `rms_norm` | −128 µs |

Issue order follows consumer deadline: `proj_a` reads `wo_a` before `proj_b`
reads `wo_b`, and reversing the order costs a third of the gain.

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

A warm scope writes no tensor, so its inputs are the only thing that decide when
it issues. Anchor it after the last stage that streams weights of its own, and
early enough for it to land before the consumer needs it.

For DeepSeek V4-Flash decode attention, the measured optimum is the moment the
q projection writes `q`. By then `wq_b`, the last large projection stream before
the o-projection, has been read. The same `wo_a` + `wo_b` warm at other anchors,
fast-rank p50 at ep2, warmup 20, against no warm:

| Anchor | Δ vs no prefetch |
|---|---:|
| Layer entry (`x_hc`) | −63 µs |
| `rms_norm` task id | +770 µs |
| KV projection output (`kv`) | −13 µs |
| **q projection output (`q`)** | **−830 µs** |
| Sparse-attention inputs, just before the attention kernel | +1186 µs |

Prefer a task id the producing stage returns, as `qkv_proj_rope` does. Fall back to a
throwaway `pl.read` of the stage's output only when no task id is reachable.

---

## Measuring a warm

- **Warm up for thousands of warm issues, not tens of rounds.** On the current
  PTO ISA a prefetch submits each warmed region as one large SDMA entry, and a
  fresh process goes through a start-up phase in which a few percent of warm
  issues block for ~200–300 µs. The phase ends after a few thousand issues,
  independent of region size or issues per round, and SDMA initialization does
  not cover it. A DeepSeek V4-Flash decode forward issues ~86 per round, and its
  spike rate is still 2.3 % at `PYPTO_BENCH_WARMUP=20`, 1.3 % at 100 and 0.4 % at
  500. Use 500 when comparing warms: a shorter warmup charges the start-up phase
  to the warm, and at ep8 each stall propagates to every rank through the MoE
  all-to-all.
- **Capture the warm on the swimlane** (`--enable-chip-swimlane`). The scope
  appears under its `name_hint`; dividing the warmed bytes by the task duration
  gives the achieved bandwidth, which is how the one-scope rule above was proved.
- **Quote median and mean.** A warm can carry a heavy tail — one measured MoE
  configuration kept only −1.04 % of its −1.63 % median once the p90 was included.
  On multiple cards, take the
  [fastest rank's median](performance-tuning.md#multi-card-l3-output).
- **Bound the win with the standalone kernel.** A standalone benchmark keeps
  the layer's weights L2-resident from round to round, so its latency is the
  most a warm can recover inside the full network. Compare it with the stage in
  the network, measured with task-timing slots from the previous `hc_post` end
  to this layer's `hc_post` end. DeepSeek V4-Flash SWA / HCA / CSA run
  243.7 / 254.5 / 347.0 µs standalone. With the o-projection warm they run
  247–258 / 249–258 / 351–355 µs in the forward, against 265–272 / 274–284 /
  374–376 µs without it.

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
