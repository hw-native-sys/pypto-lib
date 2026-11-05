# Task-Timing Slots

A swimlane answers "where did every task go". A narrower question comes up
constantly on a full network: **what does this one stage cost inside the whole
program?** simpler answers that with *selective task-timing slots* — 16 fixed
slots into which the Scheduler folds a tagged task's AICPU dispatch→finish
window, at the same boundaries the swimlane's `finish_time` uses.

| | Task-timing slot | Chip swimlane |
|---|---|---|
| Covers | up to 16 timing slots; multiple tasks may share a slot | every task |
| Switch | tagging in the generated orchestration `.cpp` — no env var, no compile gate, works in `SIMPLER_DFX=0` builds | `--enable-chip-swimlane` |
| Cost on untagged tasks | one cache-hot sentinel compare | per-task records and collector threads |
| Output | `[STRACE]` spans named `chip.run.runner_run.device_wall.task_slot_<N>` (`clk=dev`, `ts` / `dur` in ns) | `chip_swimlane_records.json` (preferred); optional merged Perfetto JSON |

Reach for a slot when a whole-network swimlane is too large to read or perturbs
the schedule being measured, and you already know which stage you care about.
Reach for the swimlane when the question is *why* that stage is slow.

They are also how a change the benchmark loop distorts gets attributed. An L2
warm is the standard example: `PYPTO_BENCH` replays the same weights every round,
so its L2 is already warm and the end-to-end delta is both flattered and diluted
— see [L2 Prefetch](l2-prefetch.md#the-benchmark-loop-flatters-a-warm).

**PyPTO exposes no DSL surface for the tag**, so the workflow patches the
generated orchestration C++ and replays it — the same
[`runtime_dir` loop](debugging.md#3-reuse-a-compile-with-runtime_dir-edit-cpp-pto-and-retest)
used for any generated-code edit:

1. Compile once (`--compile-only`, or reuse an existing build directory).
2. Open the orchestration source — `<work_dir>/orchestration/<prog>.cpp` for an
   L2 program, `<work_dir>/next_levels/<prog>/orchestration/<prog>.cpp` for an
   L3 one. Every submit block carries a comment naming its scope and kernel,
   which is how a stage is located:

   ```cpp
   // Spmd w1_mm_spmd: w1_mm
   CoreTaskArgs params_t0;
   params_t0.add_input(ext_recv_x);
   params_t0.add_output(ext_gate_i32);
   params_t0.set_task_timing_slot(0);        // <- the tag
   params_t0.launch_spec.set_block_num(8);
   rt_submit_aic_task(0, params_t0);
   ```

3. **Tag one iteration, not all.** A `pl.range` layer loop becomes a real C++
   `for` whose induction variable keeps the DSL's own name, so an unguarded tag
   merges all 20 layers into one useless window. Guard it:

   ```cpp
   if (ordinal == 10) params_t0.set_task_timing_slot(0);
   ```

4. Replay the patched build. Editing the `.cpp` is the only signal the harness
   needs — do not delete the sibling `.o` / `.so`:

   ```bash
   PYPTO_BENCH=1 python models/deepseek_v4_flash_mtp/decode_fwd.py -p a2a3 -d 0,1 --ep 2 \
       --runtime-dir build_output/<ProgramName>_<ts>
   ```

5. Read the `[RUN] task slots` block the benchmark prints (below). Without
   `PYPTO_BENCH` the one correctness dispatch prints its spans to the runtime
   log instead, at the default `timing` level. Set `PYPTO_RUNTIME_LOG=timing`
   and search for `task_slot_`. If the entry sets `config["log_level"]` to a
   threshold that hides timing logs, change it to `"timing"` or remove it:
   the explicit setting overrides the environment variable. The benchmark
   captures stderr itself, so under `PYPTO_BENCH` those log lines never appear.

## Reading the numbers

- **Each slot reduces to `min(dispatch)` / `max(finish)`.** Reusing one slot
  across several tasks yields a single merged window from the earliest tagged
  dispatch to the latest tagged finish — which is exactly how a multi-kernel
  *stage* is measured. Distinct slots keep each task's own window, so tooling
  can recover `finish(B) − dispatch(A)`. A MIX task's AIC/AIV0/AIV1 subtasks and
  an SPMD task's blocks all fold into the one tagged slot.
- **Distinguish dispatch windows from finish-boundary deltas.** Under
  `allow_early_resolve`, `dispatch` can precede execution, so a slot's `dur`
  includes staging and waiting. To measure the distance between stage finish
  boundaries, tag the preceding stage too and compute
  `(slot_{k+1}.ts + slot_{k+1}.dur) − (slot_k.ts + slot_k.dur)`.
  This delta is not the second stage's elapsed runtime when stages overlap.
  Check dependencies and overlap on the swimlane before interpreting it as
  a stage's contribution to the sequential path.
- **Slots reset every run.** A plain replay therefore yields one sample per
  slot. Add `PYPTO_BENCH=1` to the replay for a distribution: every measured
  round reports its slots, and the harness prints one line per rank and slot —

  ```text
  [RUN]   task slots: ranks=2
  [RUN]     rank 1872915 task_slot 4: n=100 fin_us=2177.4 dur_us=8.6 dfin_us=713.9
  [RUN]     rank 1872915 task_slot 5: n=100 fin_us=2599.3 dur_us=7.9 dfin_us=428.1
  ```

  `fin_us` is the slot's finish from the run's device-clock origin, `dur_us`
  its own window, and `dfin_us` the per-round finish minus the previous slot's
  finish when slots are numbered in stage order. It is a finish-boundary
  delta, not an elapsed stage duration. For example, a stage running from
  20 to 120 us whose predecessor finishes at 100 us has a 20 us delta but a
  100 us runtime. All three fields are medians; `PYPTO_BENCH_RAW=1` adds each
  round's `fin_us`.
- **The patch lives in `build_output/` only.** Recompiling regenerates the
  orchestration `.cpp` and silently drops every tag — which is also how the
  instrumentation is removed.
