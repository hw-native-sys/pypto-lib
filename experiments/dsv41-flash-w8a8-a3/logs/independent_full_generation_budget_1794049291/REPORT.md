# Independent resources/cleanup/cadence proposal review — FINAL, noEngram

PASS within the resource/cleanup/observation/durability proposal scope. Old
975d candidate32 is independently covered for cleanup/liveguard/cadence/futureIO;
NEW d057 candidate10 fixes its two evidence gaps and independently passes exact
final/coverage/fsync/NoForward controls. This is CPU/source acceptance of an
UNAPPLIED proposal, not permission to read full weights or run a model. Product45
762dc remains HARD HELD, unchanged. No NPU, real checkpoint tensor payload,
large fixture, source edit, shared-ledger write, application or guard release.

Exact candidate lineage:

- Base32 manifest3be79eba56628bf610baec131680ea1bb7fa276af2123873475a2fa591a680fc;
  generate975d555439c064b0d37fa0f2dc14c2a5204cbd9675c656997d4617ca4d99531e;
  patchbc1c8b0ffb836c2a1637ffd9bc1b901ac17553a4fdd4f00ea18149636d229460.
- Amendment10 manifest4b5e8de462dc6064ad4aa833033c4b493042597902599a681dbfec6fe8e104d6;
  NEWgenerate d057012464331b10b0bdc7f406925ee0c552dd298a1db171a93b0b1595769598;
  fullpatch522dae74110e5e1725e9b36a399c6919138b1937d900a593ffc26ab3ca0f1dca;
  tiny amendment202e870d67c96d83753d02cbc9b5432480e135412952b99418464766f0f736ce.
- Unchanged metadata3dd67b88683d0de1f5fbd8e889c0c1a875eeecf671263fca4c64061637ecd9c0.

## Personally executed independent CPU evidence

All jobs used task-submit --no-device with activate-pypto.sh. Main self-tests were
not reused as independent acceptance. All five checks completed exit0:

| Task | Independent check |
| --- | --- |
|190729_337194915904|Actual current realE40 N3/decode1 signature arithmetic only, source45 hashes, no fixture|
|191018_337916919090|Tiny original-source primary-upload/dispatch plus cleanup failure reproduces primary masking/skipped later cleanup|
|191233_338488820834|Initial-draft guard/copy/read schedule cost arithmetic, not timing/throughput|
|193339_34364495432|ACTUAL frozen975d classes: repaired primary/attemptall/clearfields, cadence boundaries/queryerror and futureIO floors; two durability gaps confirmed|
|194238_34599812092|ACTUAL newd057 class real tiny-file final/phase summary persistence, counters/endpoints,fsync and NoForward failure controls|

Raw logs/results/source scripts are owned here. Original-source violations are
explicit expected findings, not failed acceptance silently converted to PASS.
One documentation apply_patch context mismatch made no write and was corrected;
there was no unexpected queued test/implementation failure or real-device retry.

## Findings and resolution

Current PreparedDispatch.open replaced a primary upload error when close failed;
current device_run finally skipped other owner/sampler after the first close
error. Independent original-source repro preserved those observations. Proposed
975d preserves original upload/dispatch exceptions, attempts all handle/free/
context/owner/sampler cleanup, records secondary failures and clears references
even if context exit fails. The actual proposed-class fake-worker/finally tests
pass. Cleanup-only failure remains a failure; sampler alive_after_stop is reported
without claiming its10s join always outlasts the30s npu-smi subprocess timeout.

975d already WIRED both all16 final queries before/after frees. The observer's
initial missing-callsite suspicion was a reading omission, corrected by actual
source1713/1726, patch114/130 and our64-IPC close spy. Genuine bugs were durability:
the5s success throttle could omit BOTH final snapshots, and before_forward did
not persist the complete target/read ledger. These were independently reproduced
and sent Root. Root assigned the separate amendment; old32 was not rewritten.

Newd057 tiny diff changes only final success emission, critical-event fsync and
once-per-phase complete summary. Our actual-class test persists2 final events,
16 raw ranks each, and2 coverage summaries, one perphase despite repeateddecode.
It reconstructs independent tiny source maps/readcall counts, source-byte totals,
initialized-H2D totals and endpoint/expected-size rows (different source/device
totals intentionally tested). Four critical fsync calls occur; ordinary forward
events do not fsync. Incomplete coverage emits no summary; an injected fsync
failure propagates before forward and leaves that phase unmarked. This closes
the two identified evidence gaps without claiming independent full-weight parity.

## Budget and reuse basis

RealE40 current ABI (w1 begins[16,40,24,...]) has39,624,816,576 staticB/rank.
Actual N3 global nonstatic host tensor sum is363,060,736B (42 tensors), decode1
247,111,680B (38); largest both is167,772,160B BF16
window_cache[16,40,2,128,1,512]. Full rows/source/API pins are in io_budget.json.
These are independently derived META arithmetic, not device peaks or inferred
sharding. Charging full global IO perrank is conservative planning.

Selected-real34 minfree20,248,760,320 minus18,088,299,200 reserve leaves
2,160,461,120B ABOVE that operation's floor; sampled own-tree PSS5,910,070,272B
is selected-only, not fullstream/forward peak. Empty77 proves logical static
reservation and contiguous largest allocation; selected34 proves18 initialized
intervals/readback. Neither supplies a previously measured fullforward peak.

Proposal retains upload/firstforward18,088,299,200B reserve, directpublicfree
before/after allocations and positive bounded-gap upload checks. Postforward
floor9,797,894,144B preserves1GiB+128MiB+8GiB. Laterdecode preforward additionally
reserves remainingcalls*247,111,680+680,528,320; independently tested exact
boundaries10,972,645,824B and10,725,534,144B pass, one byte under rejects.
Conservative comm is unspent planning, not guessed committed attribution.

Public ordinary sequential same-owner/same-config/sharedIO reuse supports Root's
4 total generated tokens with truthful EOS and actual preceding-token decode.
Pinned a2a3 source binds HOST IO into pipeline-SLOT retained staging: bytes are
copied each call, leases settle at the terminal fence, but BufferNoop storage can
remain until Worker finalization. SharedGMbank0 does not mean all later IO reuses
one physical input block. Future-call IO allowance avoids needing that claim.
Static handles remain live once per phase; decode does not reload full weights
on each token. Full-upload/firsttouch/forward peaks stay unknown/NULL.

## Observation, time and future operation constraints

Initial every-boundary draft required2,205,120 public memory IPCs; that count is
an independent schedule-derived baseline, superseded by Root's EXPLICIT cadence
amendment. It is not a throughput estimate or evidence that six hours suffices.
Current policy preserves cheap HOST/DISK/SHM/deadline/caps/coverage each boundary;
HBM pre/post every allocation; freshall16 prewarm/forward/final; active-rank and
resourcefirstcopy; before source>=64MiB; before nextchunk crosses256MiB, after
reaching256MiB or>=1s age at a synchronous boundary. Independent actual-class
controls verify those rules and query-error refusal. No continuous peak or
blocked-call interrupt guarantee is inferred. Success logs aggregate5s; failures
and newly critical final/complete summaries are retained. Host stat overhead and
total duration remain UNKNOWN, not an extra NPU benchmark prerequisite.

GrantTTL<=3600 controls entry; cooperative operation20400s and queue21600s are
separate planning bounds. Reads/copies/deadline checks are synchronous-boundary
checks. A wait timeout is not cancellation; native data cannot be freed before
the terminal fence, and hard termination cannot prove normalclose. Preserve
partial progress/firsterror/secondary cleanup evidence; no blind retry. Actual
future task-submit command should explicitly use client --timeout>=22200 (or
separate submit/longwait), because defaultclient600 can detach while job continues.
That is an outer execution instruction, not a product source amendment.

Before actual payload/forward Root must separately authorize application of ONLY
reviewed generate.py, verify44 unchanged/new45, fresh applied-source signature
proof and all entry/mapping/tokenizer conditions under the parallel entry review.
Same model_root/checkpoint and tokenizer file hashes must be enforced externally
prequeue/allocated/postrun per Root, since this candidate does not add that product
check. Historical calibration/probe/selected transfer are pinned prerequisites,
not refreshed grants/current peaks. Unknown peak fields must remain NULL until
their actual operation is observed. This review grants no full-materialization,
NPU, numeric/full-original parity, Engram/generalF09, coherent-text or hardguard
release acceptance.

## Final state

SOURCE_BUDGET_NOTES.md retains chronological source findings/algebra and explicit
policy limits; the final disposition here supersedes its earlier pending/two-gap
status. All current product/protected candidate/evidence hashes are rechecked by
the final CPU collector. Own work_record.json uses real measured epochs with
role review-agent; no shared ledger was written in this assignment. Root may
import it once. Reviewer FINAL/PARK after seal, no background job/device request.
