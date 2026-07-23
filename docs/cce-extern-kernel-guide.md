# CCE Extern Kernel Programming Guide

How to write a hand-written mixed (cube + vector) CCE kernel behind
`pl.jit.extern` — and, more importantly, the non-obvious traps that make one
compile fine but **hang on device (507018)** or silently produce wrong data.

This guide was written after folding RoPE into the Qwen3-14B paged-attention
extern (`paged_attention_rope_cce`). Every trap below cost at least one on-device
debug cycle; they are recorded here so the next person spends minutes, not a night.

The companion references are [`compile-runtime-workflow.md`](compile-runtime-workflow.md)
(what `python <kernel>.py -p <platform>` does) and [`debugging.md`](debugging.md)
(general `507018` / hang triage). This doc is specifically about the **extern**
boundary and the AscendC code that runs inside it.

---

## 1. The runtime model you are compiling against

An `@pl.jit.extern` kernel does **not** run as its own device kernel launch. It
runs as a *task* inside simpler's **persistent-kernel executor**. Understand this
first — several traps below follow directly from it.

- `KERNEL_ENTRY(aicore_kernel)` (in `simpler` `src/{arch}/platform/onboard/aicore/kernel.cpp`)
  is launched **once** across every core of the die, calls
  `set_ffts_base_addr(...)` **once**, then enters the `aicore_execute` poll loop.
- Every task — native pypto kernels *and* your extern — is invoked from that loop
  via a single `UnifiedKernelFunc` function pointer:
  `kernel(reinterpret_cast<__gm__ int64_t*>(payload->args))`. Native and extern
  are dispatched **identically**; there is no per-task FFTS/UB setup.
- Consequences you can rely on:
  - The **FFTS base address is already set** for your extern (shared hardware
    register, set by the persistent launch). You do **not** set it.
  - With `sync_start=True` on the enclosing `pl.spmd`, all blocks of the task
    **co-start atomically** — so a task-internal cross-core barrier has all its
    participants present. (Without it the runtime may dispatch cores in waves and
    a global barrier deadlocks.)
  - Hardware `get_block_num()` / `get_subblockdim()` reflect the **persistent
    launch grid** (the full die), not your `pl.spmd(N)` — but for a full-occupancy
    task (`N == cluster count`) they line up.

---

## 2. Argument packing: **tensors first, scalars last** (not signature order)

**This is the single biggest trap.** pypto lowers a task's arguments into the
`args[]` array as **all tensors first, in signature order, then all scalars**. It
is **not** positional to the Python signature.

For the fused Qwen extern, the declaration starts with the returned attention
buffer and ends with the scalar:

```python
def paged_attention_rope_cce(
    out, query, key_cache, value_cache, block_table, workspace, metadata,
    q_proj, k_proj, ..., seq_lens,
    cache_row_offset: pl.Scalar[pl.INDEX],
) -> ...
```

the `args[]` indices are:

| args[] | value | | args[] | value |
| --- | --- | --- | --- | --- |
| 0 | out          | | 9  | v_proj |
| 1 | query        | | 10 | q_norm_w |
| 2 | key_cache    | | 11 | k_norm_w |
| 3 | value_cache  | | 12 | rope_cos |
| 4 | block_table  | | 13 | rope_sin |
| 5 | workspace    | | 14 | inv_rms_states |
| 6 | metadata     | | 15 | slot_mapping |
| 7 | q_proj       | | 16 | seq_lens |
| 8 | k_proj       | | **17** | **cache_row_offset (the scalar — LAST)** |

The scalar `cache_row_offset` is the eighteenth parameter and lands at
`args[17]`. Read it at `args[7]` and you get `q_proj`'s tensor descriptor pointer
reinterpreted as an integer offset; read `seq_lens` at `args[17]` and you
`reinterpret_cast<Tensor*>(scalar_value)` → for layer 0 that value is `0` → null
deref → the next `DataCopy` from it **hangs the vector core** (surfaces as
`507018 SCHEDULER_TIMEOUT sub_class=S1:running-stalled`).

An earlier fused draft placed the scalar between `metadata` and `q_proj`; it
still landed at `args[17]` because the ten following tensors were packed ahead
of it. The current fused signature keeps the scalar last. The attention-only
`paged_attention_cce` does the same, at `args[7]`. Add tensors after a scalar
and every packed scalar index shifts even when its signature position does not.

**Rules:**
- Index tensors `0 .. (num_tensors-1)` and scalars `num_tensors ..`, in signature
  order within each group.
- If one entry (`run_qwen_fai`) serves two ABIs, select per ABI:
  `WithRope ? args[17] : args[7]`.
- **Verify against the generated orchestration**, never against the Python
  signature. See §7.

### Output-like parameter order is also the return ABI

A mixed `@pl.jit.extern` group threads its first `pl.Out` or `pl.InOut`
parameter through a single-tensor return. This makes the relative order of
output-like parameters observable at the call site.

The broken fused signature listed `query: pl.InOut` before `out: pl.Out` and
called it as `attn_out = paged_attention_rope_cce(...)`. Generated orchestration
therefore contained:

```cpp
const Tensor& attn_out_ssa = q_tnd;
```

FAI still received and wrote the real output buffer, but every downstream task
read the returned query alias. The values initially described as garbage were
bit-exact RoPE-rotated Q.

For a single-return extern with other mutable tensors, declare the intended
return buffer as the first output-like parameter. After the fix, generated
orchestration contains `attn_out_ssa = attn_out_tnd`. If multiple mutated
buffers must be returned as values, use an explicitly supported multi-return
form instead of assuming the single return selects a later `Out` parameter.

---

## 3. UB / `AscendC::TPipe` inside a mixed extern

`AscendC::TPipe`'s **constructor inserts synchronization interfaces** (this is
why `Arch::Resource` in the vendored FAI kernel does
`AscendC::TPipe pipe; pipe.Destroy();` — create then immediately release, and
manage UB through `LocalTensorBuffer` instead).

- A **pure-vector** phase (e.g. RoPE) must keep its `TPipe`/UB **VEC-only** —
  create it under `#ifdef __DAV_C220_VEC__`. It is not cross-core; do **not**
  create it on the cube "to be symmetric" — that drags the cube into vector-only
  UB setup and hangs it. (RoPE is a pure-V op; treat it as one.)
- `TPipe` + `InitBuffer` + `GetWithOffset` views are fine on the vector core.
  If a UB-heavy phase misbehaves, suspect the *data movement* (§2, §4), not the
  buffer allocation.
- If you interleave your own phase with the vendored kernel's `Arch::Resource`,
  do not keep two live `TPipe`s at once.

---

## 4. Scalar reads from GM need cache coherency

A direct `__gm__` scalar dereference (`int32_t pos = seq_lens[b];`) reads through
the vector core's L2, which is **not coherent** with the host's input writes — you
may read a stale line from a previous run's reuse of that address → garbage index
→ wild pointer.

Stage per-batch scalar arrays into UB with `DataCopy` (MTE2 is coherent), then
read via `GetValue`:

```cpp
AscendC::DataCopy(seqLensUb, gSeqLens, kBatch);
AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID3);
int32_t pos = seqLensUb.GetValue(b) - 1;   // coherent
```

Tensor *data* read via `DataCopy` is already coherent; only ad-hoc scalar derefs
are the risk.

---

## 5. Cross-core barriers: `SyncAll<false>`, and the flag/mode facts

If a phase on the AIV lanes produces GM that the cube attention then reads, you
need a **global** cube↔vector barrier, not a per-pair sync.

- **`AscendC::SyncAll()` defaults to `isAIVOnly=true`** (vector-only) — in a mixed
  kernel it never releases the cube cores. Use **`AscendC::SyncAll<false>()`** for
  the fused whole-core barrier (per-type syncs on AIC and AIV, then between-type).
- **`CrossCoreSetFlag<mode, pipe>` / `CrossCoreWaitFlag` are intra-group only**
  (mode `0x2` = a cube and its two paired vectors). They are **not** a global
  barrier — the CANN FAI's own `qkReady`/`softmaxReady` work precisely because
  they are per-pair. Do not hand-roll a global barrier out of them unless you also
  add the mode-`0x0` global AIC sync (which `SyncAll<false>` already does).
- `SyncAll` (hardware) uses FFTS **flag IDs 11–14**; do not reuse those flags for
  your own `CrossCoreSetFlag` in the same kernel.
- On A2/A3 the CCEC extern compiles with **`__DAV_C220_CUBE__` / `__DAV_C220_VEC__`**.
  pto-isa's `SYNCALL` keys on `__DAV_CUBE__` / `__DAV_VEC__` — **not** the same
  macros. If you port a pto barrier by hand, guard it on the extern's `_C220_`
  macros or it silently compiles to a no-op. (Prefer `AscendC::SyncAll<false>`.)
- The FFTS base is set by the runtime (§1); you do not `SetSyncBaseAddr` yourself.
- Do not mechanically surround a mixed-core barrier with `dsb(DSB_DDR)`. The
  C220 `SyncAllImpl` starts with `PipeBarrier<PIPE_ALL>`, but that implementation
  detail is not a universal GM memory-model guarantee. The Qwen3 C220/CANN 9.0.0
  MTE3/TSTORE-to-GM producer and MTE2 consumer path was validated without an
  extra DDR barrier over 40 layers
  ([pypto-lib#796](https://github.com/hw-native-sys/pypto-lib/pull/796)).
  Re-evaluate `dcci` and `dsb` whenever the producer uses the data cache or
  direct scalar writes.

For on-device measurement of arrival skew, collective service, and release
skew, see [`incore-timestamp-profiling.md`](incore-timestamp-profiling.md).
Per-core barrier residence includes time spent waiting for late participants
and is not the collective's intrinsic service time.

---

## 6. Compile & correctness checklist

- **`--smoke` (compile-only) does NOT compile the extern `.o`.** It builds the
  orchestration and native kernels only; the extern's CCEC compile happens at the
  first **device** run (JIT-at-load). So a compile error in your extern surfaces
  on device, early, before execution — read the `[Incore] Compilation failed` block.
- **Golden tolerance can hide a broken op.** A single decode layer's attention is
  a small perturbation on the residual stream, so `--validate-fwd --fwd-layers 1`
  can PASS at `logits 100% within 5e-2` **even with attention entirely skipped**.
  Always validate a fused attention with the **full stack** (`--fwd-layers 40`),
  where errors compound and the argmax actually moves.
- Keep the original attention-only extern intact and select the fused path with a
  template flag (`WithRope`) + a separate entry `.cpp`, so existing golden/source
  tests are unaffected.

---

## 7. Debugging methodology (how the traps above were located)

Whether the extern **hangs** (`507018`) or produces **wrong data**, the generated
cpp, the orchestration, and a partial tensor dump are ground truth. The passes are
not worth reverse-engineering; work from artifacts.

1. **Classify the `507018` first.** Read the device log
   (redirect it via `ASCEND_PROCESS_LOG_PATH`) and match the stall signature.
   `sched_error_code=100 SCHEDULER_TIMEOUT sub_class=S1:running-stalled` with all
   cores `busy ... cond_reg_state=ack` = a **forward-progress stall inside the
   running task** (spin/barrier/fault), not a capacity deadlock. Note the
   `stuck_core=` (0..N-1 = AIC, N.. = AIV) — it tells you cube vs vector.

2. **cpp-first, compare to a working kernel.** The generated kernel `.cpp`
   (`build_output/_jit_*/kernels/{aic,aiv}/*.cpp`) and the orchestration
   (`.../orchestration/*.cpp`) are the truth. To confirm arg packing, read the
   task's `params_tN.add_input / add_output / add_scalar` order in the
   orchestration — that *is* the `args[]` index order (this is how §2 was found).
   To confirm a barrier lowering, generate a known-good native kernel that uses the
   same primitive and diff.

3. **Bisect on device with `return;`.** Insert an early `return;` at successive
   points and see whether the hang moves:
   `return` before the `TPipe` → after `InitBuffer` → before the first `DataCopy`
   → before the compute loop. The first point that *stops* hanging brackets the
   offending line to one statement. Each probe is one device cycle; it is far
   faster than reasoning.

4. **Remove confounds one at a time.** Skip the vendored kernel (early `return`
   after your phase) to test your phase alone; swap the barrier for a bare
   `PipeBarrier<PIPE_ALL>` to test whether the barrier is the hang; feed valid vs
   garbage inputs deliberately (remember §6 — a skipped attention can *false-pass*
   a loose single-layer golden, so a "pass" during bisection is not proof).

5. **Read the simpler runtime, don't assume.** The execution model (§1) —
   persistent kernel, one-time `set_ffts_base_addr`, unified function-pointer
   dispatch, `PTO2_SUBTASK_FLAG_SYNC_START` — is all in
   `src/{arch}/runtime/.../aicore/aicore_executor.cpp` and
   `.../platform/onboard/aicore/kernel.cpp`. Several dead-end hypotheses (FFTS base
   not set for externs; per-`.o` sync globals; mode-0 count descriptors) were ruled
   out by reading these ~200 lines.

6. **Confirm with a strict test.** Only a compounding, full-depth run
   (40-layer golden) proves a fused attention correct; a single-layer pass does not
   (§6).

7. **Wrong data (not a hang): partial-dump the op's output vs. what the consumer
   reads.** When the extern compiles and runs but the values are wrong — and a loose
   golden may even false-pass (§6) — the fault is often that a *downstream* task
   reads a **different buffer** than the op wrote (a return-binding bug §2b, or an
   aliasing bug). A partial tensor dump pins it in one device cycle:
   - Tag both tensors in the orchestration and enable partial dump:
     `pl.dump_tag(op_output)` and `pl.dump_tag(consumer_input)`, with
     `enable_dump_args=1` in the runtime config (partial dump — only tagged tensors).
   - Read `build_output/_jit_*/dfx_outputs/args_dump` with
     `python -m simpler_setup.tools.dump_viewer` (bf16 → `(u16<<16).view(f32)`).
     Compare the op's `after_completion` output against the consumer's
     `before_dispatch` input.
   - If the op's own output is **correct** but the consumer's input is **not the
     same bytes**, it is a wrong-buffer bind — check §2b and the generated
     orchestration (`const Tensor& X_ssa = <param>` tells you which param the return
     actually bound to).
   - Cheap golden-free oracle at `seq_len=1`: attention output for every query head
     equals its kv-head's V, so within a batch the 40 heads collapse into 8 distinct
     128-vectors (heads `[5k,5k+5)` identical). That coherence check alone flags a
     partial/garbage read — and here it revealed the "garbage" was bit-exact RoPE'd
     Q, i.e. the consumer was reading `query`, not `out`.
