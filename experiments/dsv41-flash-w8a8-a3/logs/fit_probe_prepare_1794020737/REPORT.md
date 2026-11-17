# Empty allocation fit-probe preparation — NOT authorized for hardware

fit_probe.py is an artifact-only candidate helper. Current work used queued CPU
fakes, metadata and a read-only exported C API query. No implementing worker,
model/runtime/harness edit, NPU job, actual large allocation or real payload read
occurred. Frozen45 source hashes remain unchanged. Hardware execution is a
future separately assigned unit, after independent functional and budget gates.

## Prepared two-step probe

The helper reads all57 static resources from sealed real_fit_readiness budget
metadata, separately for actual prefill128 and decode1 phase layouts. It checks
resource uniqueness, dimensions, dtypes and exact byte totals, without tensor
payload fixtures. Each rank total is39,624,816,576B; the largest individual
w1/w2/w3 rank buffer is11,324,620,800B.

A future authorized run would reload a real40/57-resource ONBOARD/a2a3 signature
artifact and prepare it with an explicit normalized ring configuration. No
forward or other compute dispatch exists in the helper. All allocations use the
public rt.alloc_tensor(shape,dtype,init=None,worker_id=rank) API.

1. Prepare one owner; query actual public free/total and committed memory on
   all16 ranks. Test one largest11,324,620,800B buffer on EACH rank. Free every
   handle, record public counters after free, and CLOSE that owner completely.
2. Create a FRESH owner with the same prewarm configuration. Compare its
   per-rank preweight free/committed baseline with the first owner's baseline
   using explicit measured tolerances. Refuse a regressed baseline. Only then
   ramp all57 resources on all16 ranks in the same rank/argument order as the
   streaming plan, checking before each allocation. Free all handles and close.

The largest and aggregate tests never coexist. free_tensor may retain allocator
cache; successful free calls alone are not a claim of physical free restoration.
The fresh owner baseline is the explicit post-close check. A query failure,
insufficient headroom, wrong handle schema, allocation failure or cleanup error
fails the probe. Every created handle receives a best-effort free attempt,
including after partial failures; owner close is always attempted. Failed
constructors rely on the public prepare implementation's own partial-startup
cleanup, and the helper reports failure rather than claiming it closed an owner
that was never returned.

Per-allocation public snapshots are persisted in JSONL; final per-phase records
include prewarm, after-free/before-close and errors. No uninitialized weight is
uploaded, read back as meaningful data or dispatched. Allocation success alone
does not prove upload/physical first-touch, numerical fit, real-model parity,
continuation, generation or coherent text.

## Memory guards and selected backend

For each rank, directly use device_memory_info.free (which ALREADY includes
current committed allocations), not free-minus-committed. Require:

    free >= remaining_target_static + conservative_phase_IO + nominal_comm
            + measured_additional_peak_excluding_those_and_current_commit
            + explicit_safety_margin

The first phase's target is one largest buffer/rank; the second target is all57.
The remaining total decreases after each successful allocation. committed is
recorded to diagnose retained resources; it is never charged a second time.
Phase IO and nominal comm come from sealed readiness metadata. A future receipt
must supply16 measured additional-peak entries,16 explicit margins and16 close
baseline tolerances. No real-fit margin is invented by this preparation. Fake
byte values in check_fake.py are tests only and are not measurement receipts.

The selected backend is verified from the artifact: a2a3 and every chip
RUNTIME_CONFIG runtime=tensormap_and_ringbuffer. The helper then uses public
RuntimeBuilder(...,build=False), verifies the EXACT installed library SHA
24434f8879030ef1b1f31e5fc2ecce683e392d4f1baeb5bcee852fb183976e5c,
and queries exported get_pipeline_contract read-only. ABI1/resources6 and
heap/SM/image DEVICE_SCRATCH classifications must match. Unknown runtime,
changed library/source pin, failed C API query or nonshared contract refuses.
The future receipt also binds independent backend evidence and source hashes.

For this pinned PROGRAM backend, reviewer source/installed-binary evidence maps
GM resources to shared bank0;168 run identity slots are not168 resident heaps.
The already prewarmed heap is reflected in queried free and is NOT reserved
again. This is backend-specific evidence, not a generic claim about other
runtimes. Extra host-per-run metadata and other uncommitted forward peaks remain
in the measured additional-peak term. The helper does not inspect or mutate
private capacity arrays. Actual future public telemetry is still mandatory.

## Default refusal and future inputs

Running without flags refuses BEFORE native imports/prepare. --plan prints
metadata-only sizes. The included authorization.NOT_APPROVED.template.json has
approved=false, and measurements.PENDING.template.json contains no usable
measured margins. Neither authorizes any operation.

A future assigned hardware invocation additionally requires --execute, a later
rootGO authorization file pinned to the45-source manifest AND approved probe
artifact metadata hash, accepted independent functional evidence, a separate
bounded memory receipt, correct phase, matching installed backend evidence,
exact16 queue-allocated device IDs, and a fresh output directory. The artifact
must carry actual real40 expert stack shapes; the reused1-layer bounded fixture
is rejected as an empty-probe schema. No full checkpoint source is opened.

Before any actual empty ramp, the independent reviewer must inspect the helper's
untested native-only branch and confirm that receipt terms do not double-count
IO/comm/prewarm or omit observed peaks. Prewarm itself has a footprint, so its
safety must already be covered by the future accepted receipt and assignment.
No public guard or current production full-weight hold is changed here.

## Completed CPU checks

- task111125_253782231836: fake empty allocation engine PASS.16 largest handles,
  912 aggregate handles, all16 rank identities,57-resource dimensions/dtypes,
  exact39,624,816,576B/rank. Distinct owners; retained-cache-until-close model;
  large committed baseline proves no free/committed double charge.
- task111241_254109111179: reran final engine and read-only installed contract
  checks PASS; unknown backend/platform/library receipt rejected; --plan and
  py_compile PASS. No Worker init/prewarm or NPU call.
- Eight engine negative controls: insufficient last-rank free before allocation,
  public query exception, failure after24 created handles, wrong handle shape,
  free failure with every remaining free attempted and close, fresh-owner
  baseline regression, reused owner, and default execution refusal.

raw queue logs and fake_results.json preserve counts and outcomes. Fake tests
use integer accounting and lightweight handles, not large torch tensors.
Native-only prepare/allocation/baseline behavior remains unvalidated. Root owns
future independent review and any hardware GO; Main parks after sealing this
artifact. Source/behavior of scalar, stream, prefill and decode remains frozen.

Final CPU task111623_256301127842 PASS on the sealed helper: actual default CLI
refusal, hashed-readiness schema binding, all fake controls, read-only installed
contract controls and frozen45-source before/after equality. No actual native
prepare/allocation branch was exercised. The three queued CPU jobs ended0.
