# Decode return publication and prepared residency — resource-blocked handoff

Implemented a narrow decode return-direction repair and a supported prepared
dispatch owner. Source/CPU/compile checks pass. New native before/after and
residency checks received NO device allocations: all16 shared devices stayed
occupied, and the long waits expired/cancelled while pending. No native repair,
continued-state, residency/peak or text-generation acceptance is claimed.

The preserved independent native baseline remains a genuine seven-output
numerical failure in native_accept_noengram_1793972955/REVIEW.md, task
task_20261106_221305_24806427754. It is distinct from this unit's queue blocker.
That run failed state_cache_c2a49,152/98,304 lanes, source pool counts0 versus
79/157, top-k valid-lane masks and pool sets. Other outputs passed existing
staged gates. No baseline error, historical SIM claim or threshold was rewritten.

## Ownership and source repair

Main owns decode_fwd.py,prefill_fwd.py,layer_reference.py and integration/docs.
Only decode_fwd.py changed. Worker /root/main_native/resident_dispatch owns
generate.py/weight_loader.py and worker_residency/; only generate.py changed.
Artifact-only worker continuation_plan owns its new artifact directory and is
PARKED. No operator transfer, runtime/harness/checkpoint write, commit, issue or
original Engram hash work occurred. User decision10 noEngram scope is preserved;
TP4/DP4/EP16,W8A8 and existing math remain unchanged.

First return owner: decode_fwd_rank declared ten mutable cache/state/selection
arguments as plain Tensor even though the top-level entry declared InOut.
Generated host_orch.py therefore passed them as TensorArgType.INPUT. The
prefill counterpart used INOUT. Generated device code already contains the
c2a_state_write,index_topk,index_block_pool,index_pool_counts producer tasks and
their gather dependencies; wrong HOST mutation direction is an actionable
return-boundary defect. No missing GPU edge or runtime scheduling root cause is
inferred merely from zero returned values. New runtime graph evidence was not
captured because the requested allocations never happened.

Repair: ten rank annotations become InOut. No rank computation, producer math,
extra dependency, barrier, dummy or defer_wait was added. Both full-forward
owning goldens remain AST-identical. main/source_scope.json proves that exact
scope, and main/direction_result.json lists all ten generated HOST INOUT
bindings after explicitONBOARD/a2a3 compilation. The brief early num_tokens=0
hypothesis was unproved and is not the repair rationale.

Permanent decode cache comparison now checks both the full cache and the
specific rows published by the current step for WINDOW,C1/C2 compressed and
index caches. It uses the SAME standard ratio_allclose thresholds
atol.02,rtol1/64,max_error_ratio.01. Five CPU positives and five missing-row
controls demonstrate a whole-cache1% allowance can admit one absent row while
the published-row check rejects it. This strengthens detection; it does not
mask zero outputs or redefine query scratch. State, selection and all other
owning gates remain required.

Independent original-reference CPU review executes compressor_ref AST from the
pinned rc2 archive. Even/odd trailing projection writes match the owning state
golden exactly on asymmetric inputs; zero state violates that contract.
ceil(625/8)=79 and ceil(1251/8)=157 independently confirm the failing source pool
counts, below cap2048. The original file matches preserved archive bytes.
No original golden defect was found or changed.

## Prepared ownership and allocation hold

Previous TensorSpec resident flags were used only during compile; direct calls
with ordinary Torch arguments did not implement upload-once residency. The
worker replaces that actual driver dispatch with PreparedDispatch, which owns
one supported compiled.prepare(persistent=False) context per phase, allocates
static leading-rank weights through public stacked-device handles, retains
those handles across calls, and frees them on close. Prefill owner closes before
decode owner opens. Per-call metadata/cache/output buffers are CPU-contiguous
shared memory before prepare; replacement metadata copies into those stable
buffers. Protocol domains reset per call while static handles stay live.

Mutable caches explicitly travel shared-host H2D/D2H per dispatch: this is NOT
persistent device-cache ownership. CPU fake-public-worker controls cover one
static upload despite host-weight mutation, repeated actual-state updates,
replacement metadata, changing scalar, cleanup and source/target handoff bounds.
The tiny native2-rank proof of this exact class is unexecuted because allocation
never occurred. CPU controls are not substituted for native transfer evidence.

Worker also fixes HbmSampler's collision with threading.Thread._stop by using
_stop_event; start/stop CPU control passes. Page handoff can trim only unused
extra source pages after raw WINDOW and C1/C2 index/compressed used rows are
validated against BOTH source and target capacities. State batch shrink remains
rejected. Exact-prefix, source/target overflow and state-batch controls pass.

Full-weight loading stays HELD even with --allow-full-materialize. A tested
host-materialization-overlap ContractGap rejects before tokenizer/materializer.
Retained rank_args/expert_by_rank, global torch.stack assembly and prepared
upload staging can exceed1.5TiB host memory. Planned leading-rank static HBM
arithmetic plus conservative IO is not measured peak/transient fit. A future
supported streaming route can allocate per-worker shards with public
rt.alloc_tensor and aggregate StackedDeviceTensor handles without retaining
global host stacks; it is NOT implemented/proven here. No full weights were read.

## Checks and queue outcomes

task_20261107_005138_106260121363 in main/final_cpu_queue.log is the final successful combined check: native-target
decode1 compile, all ten generated rank directions, five cache positives/five
missing-row controls, original AST state/count review, exact math/golden/source
scope, pinned archive comparison and scoped lint. Worker report lists its
successful owner/handoff/sampler/guard/API CPU checks and tiny ONBOARD compile.
Every new device helper explicitly requests execution_mode=False, checks
normalized ONBOARD/generateda2a3 and requires native runtime evidence for a
hardware claim. No SIM result contributes to this unit's device evidence.

| Attempt | Result |
| --- | --- |
| task_20261106_233831_327052422275 old-source16 baseline |600s queue-only timeout/cancel before allocation |
| task_20261106_235349_35913005692 old shadow source baseline |3600s wait expired/cancelled before allocation; client exit1 |
| task_20261106_235435_360345214567 repaired16 decode |3600s wait expired/cancelled before allocation; client exit1 |
| task_20261106_234723_349266526109 tiny2 residency |600s queue-only timeout/cancel before allocation |
| task_20261106_235855_377411318121 tiny2 residency |3600s queue-only timeout/cancel before allocation; client exit1 |

Main --cancel checks found the two long-wait tasks already absent: their CLI
had naturally cancelled them. Both wait clients ended; no manual kill/reset,
permission bypass, queue-policy change or other-user-job action occurred.
The timeout implementation's polling overhead can extend nominal wall time;
raw timestamps/logs are authoritative. This is a shared-resource blocker, not
a new compiler/runtime/numerical failure. Helper-only annotation introspection,
scalar-versus-tensor index and standard-comparator-call mistakes are preserved
in attempt logs; corrected final checks pass and no such error counts as a
negative numerical success.

## Gated continuation and recovery

Artifact-only continuation_plan/chain.py is prepared and Main-reviewed, with
fresh-output run.sh, CPU API/isolation controls and frozen worker manifest.
Main repeats the API check on final models in main/final_continuation_api/.
It compacts artifact cache pages to2, runs a genuine prefix0..127, carries exact
actual returned bytes into decode128/129/130 on ONE compiled prepared owner,
clones ALL nonstatic initial metadata/state before calls, and checks original
WINDOW/C1 rows0:128 and C2 rows0:64 remain byte-identical. Phase owners close
sequentially; actual-return hashes/pointers and artifact identity are recorded.

The negative control restores ACTUAL returned128 state, expects129, and retains
128 slots/lens/RoPE. It must numerically reject newly published WINDOW and C2
rows with the standard comparator, preserving original prefix rows; runtime
exceptions never count. This chain/control has NOT run. Guarding it on a native
all-output decode PASS and a native prepared-ownership PASS is intentional.
Owning comparisons are staged per actual previous-layer snapshots, not strict
unconditioned40-layer equivalence; the helper reports that distinction.

After capacity is available, use fresh outputs and no duplicates. First run
the preserved baseline/repaired16 checks and inspect every state/selection/
published-row output. Then the bounded tiny native ownership test. Only after
both prerequisites pass, launch the reviewed16-device prefix/three-decode
chain/control using continuation_plan/run.sh and verified prerequisite files.
Inspect source/runtime mode and actual returned state independently. Do not
reenable full allocation until streaming ownership and host/HBM overlap are
bounded. Exact commands are in RESUME.md,worker PLAN and continuation PLAN.

Main and both workers are now PARKED; all own CPU/NPU jobs and wait clients ended,
all three long-wait task statuses are not_found, and the last resource snapshot
still reports16/16 occupied. Final after.json/artifacts.json and worker reports
seal source/evidence. TASKS/TIME_LOG carry measured intervals and unique worker
imports. Independent sibling acceptance remains required. F09 general runtime
unequal/idle/multi-request behavior,F05/F07/F13,historical missing hardware
controls,strict checkpoint/primitive-tie parity,TP/CI/offset incidence and text
generation remain OPEN/HELD. No successful new device run, peak, continuation,
real-checkpoint logits or text is claimed.
