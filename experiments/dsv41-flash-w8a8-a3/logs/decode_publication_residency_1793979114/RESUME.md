# Frozen resource-blocked unit recovery

FINAL: all three long-wait native attempts expired/cancelled BEFORE allocation;
their statuses are not_found and their clients ended. Main and workers PARKED.
See REPORT.md/after.json/artifacts.json and latest TASKS for the sealed snapshot.
The job IDs below are historical recovery references, not active jobs. No new
native repair/continuation/residency/peak result exists. Full weights remain held.

Main task: decode rank return publication + disjoint prepared residency. Read latest TASKS and parent messages first. Main owns decode_fwd.py/prefill_fwd.py/layer_reference.py; worker resident_dispatch owns generate.py/weight_loader.py. Only decode_fwd.py + generate.py have changed so far; no operator/golden/math/threshold edits. Do not claim whole-tree final stability while worker's native proof is pending.

Model fix: decode_fwd_rank ten cache/state/selection arguments changed plainTensor->InOut. Old generated HOST binding INPUT; new binding INOUT. Existing device producer tasks/consumer paths present; no new edge/barrier/deferwait. main/decode_before.py is exact old source for baseline. Permanent decode published-row checks retain same standard thresholds; five positive/five missing-row CPU controls pass. Original rc2 compressor AST matches owning trailing even/odd state, counts79/157 independent. AST/source audit and original archive bytes pass. Own helper-only false failures preserved in direction_attempt*.log and cache_gate_attempt2.log.

Prior OWN native attempts, confirm status before any future fresh submission:
- baseline old shadow decode: task_20261106_235349_35913005692, main/baseline_queue.log, session91591.
- repaired decode: task_20261106_235435_360345214567, main/after_queue.log, session17397.
- worker tiny two-rank prepared API: task_20261106_235855_377411318121, worker_residency/native_queue.log; worker owns monitoring.
First baseline233831_327052422275 and tiny234723_349266526109 were queue-only600s timeout/cancelled before allocation, NOT runtime/numerical failures. Current waits timeout3600s. Queue header last checked all16 devices occupied. Do not duplicate jobs, touch others, reset devices/admin/permissions or change queue policy. Model source writers currently stopped while awaiting jobs.

When baseline finishes require explicit ONBOARD/a2a3/native runtime and real expected numerical failures; main/baseline/trace_result.json, generated deps graphs under main/baseline_final_build. When repaired finishes inspect ALL owning outputs including published-row gates; main/after/trace_result.json must all_outputs_numerical_pass=True and jobexit0 with native logs. No runtime error counts as negative control. Graph report may document producer->count->gather edges but root cause is HOST return direction, not a proven missing GPU dependency.

When worker native proof passes obtain its final direct report/hashes/unique record; tiny device numerical result/weight mutation control/actual repeated cumulative state/one upload+retained handle+peak samples must all be native. CPU fake-worker controls are not device proof. Static weights are device-resident only; caches explicitly shared-host H2D/D2H per call; no device-cache persistence claim.

Artifact-only continuation worker continuation_plan is delivered/PARKED. Frozen manifest d7583979e9c61b8c0d60dd164005d9e80eac8d04bff8117eb7ae45751576e933. Main reviewed script: compact two pages, real prefix0..127 ->128/129/130; deepclone all nonstatic inputs, original-prefix byte retention; exact actual returned bytes handed into next calls; one prepared decode owner/artifact; phase owners close sequentially. Stale control restores ACTUAL returned128 state then keeps128 slots/lens/RoPE while expecting129; must numerically reject WINDOW andC2 published rows. Context must verify both native prerequisites before launch; no chain has run. run.sh requires fresh output + decodePASSJSON + residencyPASSJSON and CHAIN_PREREQUISITES=verified. Do not edit frozen worker artifacts; copy to Main if needed. Scope staged conditional per-layer comparison, strict cumulative original parity not claimed.

No full checkpoint weights, real text or real fit allocation. Worker memory bounds/512B tiny peak not real model fit. General F09/F05/F07/F13/TP/CI/offset/primitive-tie/full-model parity remain OPEN/HELD. User decision10 noEngram first path applies. All tests task-submit; CPU no-device, native auto with explicit execution_mode=False and native generated/runtime verification.

Unvalidated next milestone: actual decode all7+allotheroutputs pass, valid chain/control pass + tiny residency proof. Current external blocker is lack of shared device allocation across one-hour waits. Source/CPU snapshot freezes without accepting those gates. Both workers' unique measured records were imported once; final measured Main interval is in TIME_LOG. No commits/filing/protected-source writes. Observer disposition controls resumption.
