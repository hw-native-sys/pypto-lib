# Independent first full-generation resource pre-review — noEngram

Owner /root/reviewer_fit; NEW exclusive independent_full_generation_budget_1794049291.
No product, shared ledger or old evidence writes. No NPU, checkpoint tensor
payload, large fixture, fullweight guard release or full-generation execution.
Main owns the separate ACTIVE full_real_generation_prepare_1794048884 proposal;
its complete source mapping/counts are not independently accepted by this note.
Final proposed-diff audit waits for its frozen cohort.

## Current-source findings and narrow independent failure proof

Reviewed immutable product45 manifest762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7;
generate.py SHA685cf3cfadf36c92b5f9c6280f9f9d90ebf0e3244133a2b550adea83ceb29cfb.
PreparedDispatch/device_run support serial leaf streaming and phase-separated
resident owners; current production HARD guard is unconditional and unchanged.
Live budget enforcement is absent from this frozen full path: argument_memory_plan
is metadata, HbmSampler is best-effort npu-smi observation, neither is admission.
Selected-transfer live guards do not automatically apply to normal generation.

Independent tiny fake CPU job191018_337916919090 COMPLETE0 establishes:

- PreparedDispatch.open primary upload ValueError followed by a free failure and
  context-close failure returns secondary context-close RuntimeError. Both fake
  handles get free attempts, but original primary is masked and context/worker
  fields stay non-null when __exit__ raises.
- Exact AST-copied current device_run finally, run only with fake cleanup objects,
  replaces a primary dispatch error with failing decode-close. It never attempts
  prefill-close or sampler-stop. No checkpoint/device/forward body executes.

See current_cleanup_repro.json/reproduce_cleanup.py. This is source/CPU failure
evidence, not an upstream defect. Main's proposed source amendment must preserve
the original exception, record secondary cleanup failures, independently attempt
both owners and sampler, clear logical fields in finally and reject cleanup-only
failure even when the main operation succeeded. Context prepare/enter failure
and mid-stream iterator exit need the same ownership account; no free of in-flight
data before a terminal dispatch fence. The independent regression will use the
frozen proposed source, not Main self-test PASS.

## RealE40 N3/decode1 IO arithmetic, independently derived

CPU190729_337194915904 COMPLETE0 used actual current g.build_entry_contracts(3,
decode_tokens1,pages2,batch1) with DEEPSEEK_PER_LAYER_EXPERTS=1. All45 source hashes
were checked and w1 shape begins[16,40,24]. No fixture/tensor data was allocated.

| Quantity | Prefill3 | Decode1 |
| --- | ---: | ---: |
| Static target/rank |39,624,816,576B|39,624,816,576B|
| Nonstatic tensor parameters |42|38|
| Nonstatic GLOBAL logical host bytes |363,060,736B|247,111,680B|
| Largest nonstatic GLOBAL tensor |167,772,160B|167,772,160B|

The largest is BF16 window_cache[16,40,2,128,1,512]. All nonstatic rows/dtypes/
sizes and public API source hashes are in io_budget.json. Scalars are separately
typed by the ABI and not counted as these tensor payload bytes. Charging the
ENTIRE global IO perrank is a conservative planning bound, not measured device
sharding. It avoids inheriting syntheticE1 footprint claims or calling oldN128
7,609,876,736B actualN3 IO. Actual full-upload/forward peaks stay NULL.

Accepted empty77 proves57 static logical reservation and normal-close/freshbaseline;
selected-transfer34 proves only its18 initialized intervals/5.307GB transport,
not complete initialized40-layer weights. Selected minpublicfree20,248,760,320B
minus transferfloor18,088,299,200B leaves2,160,461,120B above that policy floor.
Selected own-tree PSS5,910,070,272B is not full-stream/forward peak; old source
component max(4S,24Q)+64MiB=5,362,417,664B is not all-process peak either.

Keep new operation's HOST64GiB/DISK2GiB/SHM16GiB policies explicit, fresh/live,
with measured own-tree PSS/USS and shared-memory usage. Stream exactly one leaf;
do not collect lists or full global tensors. A CPU view can retain full raw
backing beyond visible330MB embedding slice; record/release backing before the
next leaf/group. Cursors/counters must prove all912 complete target intervals and
all source-read caps before forward; no uninitialized gap/overlap/duplicate bytes.

## Supported public per-call lifetime and four-token reuse

Current PreparedDispatch.dispatch uses stable shared host IO and resident statics,
passes the SAME normalized config each call and calls the public synchronous
worker. DistributedWorker.__call__->_run_compiled->submit(...).result() reaches
the terminal fence before host mutation/sampling/next ordinary decode dispatch.
Simpler finalization drains run-owned domains/native-run state; failed/ambiguous
cleanup is sticky and refuses successors. Ordinary persistent=False domains are
fresh per call. Static DeviceTensor handles are externally owned and survive
until that PreparedDispatch owner closes. No static full reload per greedy step
is necessary: decode owner is created once outside the step loop.

Important retained-memory distinction from pinned a2a3 source:
runtime_maker.cpp:16-35,590-634 binds HOST tensor slices into a pipeline-SLOT
retained temporary block, copies IN/INOUT each run, copies OUT/INOUT back and
releases only per-run leases. BufferNoop slices keep the retained block until
Worker finalization. device_runner_base.cpp:1034-1118 reuses a slot's block when
capacity suffices, otherwise grows it. RunResourceSlots index by granted resource
set. Shared GM bank0 does NOT prove one retained input block or that later slots
allocate zero bytes. Do not infer IO attribution from arbitrary committed delta.

This supports Root's preferred4 total greedy tokens with one prefill and3 ordinary
sequential decode calls on a stable owner/config/shared IO (EOS may stop earlier).
Existing bounded same-owner E1 three-step proof supports API lifecycle, not new
full40 numerical/peak acceptance. Preserve strict4DP sampling and actual returned
state; no automatic scope reduction to2 tokens or repeated full-weight reload.

Resource policy must distinguish unspent allowance from already charged storage.
Before the first forward, publicfree must cover positive planned IO/comm/safety.
The proposed post floor9,797,894,144B =1GiB+8GiB+128MiB excludes the old N128
IO+comm allowance once those resources may have been consumed. Root requests
before each later decode a remaining-call allowance using conservative GLOBAL
decodeIO247,111,680B perrank plus nominalcomm4,663,744B where unproven future
growth is possible: remaining3/2/1 planning demands are10,553,220,416 /
10,301,444,992 /10,049,669,568B INCLUDING that postfloor. These are planning
algebra, not measured staging peaks. Keep the8GiB safety unspent, query raw
free/total/committed, and STOP rather than invent credit from cache/committed delta.

## Guard/log cadence and bounded time

Independent CPU191233_338488820834 COMPLETE0 counted the observed DRAFT callsites
against Main's header-only schedule (mapping still pending independent acceptance):
perphase164,192 source reads,110,368 leaves,110,496 public copy chunks,912 allocations.
Pre+post for each produces551,200 rank-local guards/1,102,400 device-memory IPCs
perphase. Two phases plus2 allrank prewarm events and4 forward pre/post events
produce2,205,120 device-memory IPCs. The initial draft also performed1,102,400
/proc reads andJSONL open/write/flush/close events, plus2,204,800 statvfs calls,
before O(1) setup/final events. See guard_count.json/count_guards.py.

These counts do not predict throughput or prove infeasibility. At a21600s queue
cap,21600/2,205,120 =9.795ms is only the theoretical average per-IPC budget if
ALL data reading/packing/copy/forward/other work cost zero. Unknown actual latency
must stay unknown, not an extra NPU-benchmark precondition. Root may authorize
one honestly bounded first attempt despite unknown total duration. Main is
changing success logging to5s aggregates; that reduces file churn but leaves
public query and host-stat call counts unless callsites explicitly change.

Concrete minimum time/progress contract for the first attempt:

- NEW authorization TTL<=3600s controls admission, never a refreshed historical
  grant. Separate operation deadline20400s and task queue hardcap21600s/client
  timeout21600+cleanup allowance must be explicit planning, not promised sufficiency.
  A long full stream is normal progress, not a hang just because no greedy token
  has yet been produced.
- Record monotonic successful source-read count/bytes, leaf ordinal, completedH2D
  bytes/chunks, active phase/rank/argument and current blocking-call start. Emit a
  bounded-size aggregate heartbeat every5s, first/last phase/owner and all failures
  immediately. Retain exact caps/cursors in memory on EVERY read/chunk even when
  individual success logs are aggregated. Persist the complete coverage summary.
- Check deadline/failure flags cheaply between EVERY synchronous read/copy; stop
  admitting a new source/copy/phase/forward after deadline. A configurable bounded
  stall watchdog (e.g.180s of unchanged progress, declared policy) records exact
  active operation and waits for return/normal cleanup; it must not count its own
  heartbeat as useful stream progress or free data while native work is in flight.
- Public submit().result(timeout) is a WAIT deadline, not cancellation. A timeout
  can leave the dispatch running. Close drains in-flight handles and can itself
  wait. Hard task termination may prevent normalclose; retain first cause/partial
  counters, report forced cleanup honestly and do not claim normalclose/fit.

If Root later chooses to amend query cadence, an explicit reviewable option is:
keep exact source header/cap/coverage/NoForward checks per logical operation;
keep public allocation pre/post checks for all912 handles; query device budget
at each active-rank resource boundary and after at most256MiB completed H2D or1s
since the last checked synchronous boundary, and around any individual large
source read; host resource monitor at1s with stale/failed-monitor STOP flags checked
before every operation. Preserve fresh all16 prewarm/beforeforward/afterforward/
handoff/final queries and all error snapshots. This is an EXPLICIT alternative
policy with bounded observation gaps, NOT silent equivalence to every-call guards.
No cadence amendment is implemented here, and no new benchmark is required to
review the currently bounded attempt honestly.

Root subsequently proposed an explicit bounded-gap amendment: retain HOST/DISK/
SHM/deadline/header/readcaps/coverage on EVERY read/copy; public HBM pre/post EVERY
allocation, all16 prewarm/forward/phase/final; active-rank public query before
each individual source>=64MiB and at synchronous copy boundaries before firstcopy,
after at most256MiB completedH2D OR1s since lastquery. Failures/rawsnapshots log
immediately. Independent disposition: acceptable for the FIRST experimental4-token
operation with unknown transient peaks and8GiB safety, provided active-rank/owner
switches force fresh checks, each resource's firstcopy is fresh after CPUtransform,
nextchunk cannot silently overrun256MiB (chunks<=64MiB), and1s is a synchronous
boundary age test, not an interrupt guarantee. Failed queries never become0 or
cachedPASS. This policy is deliberately NOT equivalent to every-copy public HBM
sampling or a peak upper bound. No code change was made by this reviewer.

## Pending candidate acceptance

Wait for Main frozen proposed generate-only amendment, exact source/diff/plan/policy
pins, live guards, coverage proof, error-preserving cleanup and controls. Then run
own narrow regression of the proposal's failure paths and check final remaining
decode allowance/progress cadence/timebox callsites. Current source remains HARD
HELD. No hidden monkeypatch admission, relaxed thresholds, implementation or
actual device action is authorized by this pre-review note.

## Frozen proposal9755 independent CPU audit

Main froze32 artifacts3be79eba56628bf610baec131680ea1bb7fa276af2123873475a2fa591a680fc;
generate.proposed.py975d555439c064b0d37fa0f2dc14c2a5204cbd9675c656997d4617ca4d99531e;
patchbc1c8b0ffb836c2a1637ffd9bc1b901ac17553a4fdd4f00ea18149636d229460;
metadata3dd67b88683d0de1f5fbd8e889c0c1a875eeecf671263fca4c64061637ecd9c0.
Independent queuedCPU193339_34364495432 COMPLETE0 imported that ACTUAL frozen
class (not Main self-tests), used fake workers/contexts and exact finally AST:

- Primary upload/dispatch failure is preserved despite cleanup failures; both
  handle frees, exit, other owner and sampler all get attempts; references clear.
- First-resource,256MiB crossing/reached,1s stale,64MiB source,rank-switch and
  query-failure controls pass. No every-call sampling equivalence is claimed.
- Actual laterdecode floors are10,972,645,824 and10,725,534,144B (remaining2/1
  globalIO plus conservative680,528,320 nominalcomm and9,797,894,144 postfloor).
  Exact boundaries pass; one byte under rejects. The extra conservative comm
  amount is planning, not measured or attributed from committed delta.
- Actual close DOES call both all16 phase-final queries (64 public IPCs), with
  free/exit in correct order. An observer's initial claim that callsites were
  absent was a reading omission, corrected by proposed source1713/1726 and
  patch114/130. It is not a Main source bug.
- Two genuine durability gaps remain in9755: within5s after-forward, both final
  success snapshots are dropped by event throttling (actual spy sees0 events);
  before_forward completes in-memory coverage equality but persists no complete
  read-count/target-cursor/byte/call summary. See candidate_review_results.json.

Root accepted these precise findings and assigned Main a NEW artifact-only
durability amendment. Old32/product762dc stay immutable. Only that tiny amendment
will be retargeted independently when frozen; no new mapping suite/device run is
needed for this resource review. Current candidate disposition is partial logic
PASS with the two evidence blockers, not overall admission/execution acceptance.
