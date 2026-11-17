# Real memory-fit readiness — read-only, no allocation authorization

Frozen model45-source manifest remains ab1ced6d6ecbf9887297458703ba9b74e2c9b8bc57608370b4c2e0310e027e37.
This unit read headers/source and ran queued CPU metadata work only. No workers,
NPU checks, real checkpoint payload reads, model/runtime/harness edits or real
allocations occurred. Scope is one request, decode capacity1, prefill128, two
physical pages, noEngram, TP4/DP4/EP16. General dynamic F09 is separate.

## Two explicit corrections to prior Main reporting

1. The frozen prior report incorrectly called5,295,308,800B a physical BF16
   embedding source. Actual embed.weight and head.weight headers are each
   [129280,5120] BF16 =1,323,827,200B. A rank TP slice is330,956,800B;16rank
   logical argument bytes are5,295,308,800B because DP repeats the four TP
   slices. One reader raw-bytes+bytearray component is2,647,654,400B; a contiguous
   TP view can retain1,323,827,200B backing. The old10.59GB statement was a
   fourfold overbound mislabeled as source storage. The earlier conservative
   max(4*S,24*Q)+64MiB =5,362,417,664B serial payload planning allowance can
   remain; my claim that this allowance was incomplete because of a5.295GB
   physical source was false. It still excludes IO, metadata, allocator retention
   and runtime/compile costs and is not measured fit.
2. The source exports19 outputs, not24. _device_outputs contributes5 base
   outputs; _group_outputs adds6 cache/state and8 selection outputs. Five cache
   comparators have conditional published-row subchecks, which may skip when
   no new row is emitted. They are not extra exported outputs or evidence of
   five additional executed gates. My old ALL24-output headline is retracted.
   The current independent reviewer reports all19 source-defined outputs PASS.
   Raw frozen reports and logs remain unchanged.

3. A later independent trial established an API-layer correction: the lower
   CallConfig/runtime accepts non-power-of-two heap sizes, but the ACTUAL PyPTO
   RunConfig used by generate.compile_entry/PreparedDispatch validates each
   nonzero ring_heap entry as a power of two>=1024. The reviewer3GiB trial was
   rejected BEFORE compile/prepare/weights/native execution. Earlier frozen
   reporting that dismissed the power-of-two constraint was incomplete. It
   remains unchanged as history; this is the current constraint. No bypass is
   proposed. Supported larger ring2 profiles on this path are1/2/4GiB.

## Exact static and IO accounting

budget.json and static57.tsv enumerate every static resource in BOTH real40
phase ABIs against actual sealed bounded native-artifact metadata; all57 shape /
dtype contracts match the loader's phase-specific contract. header_bindings57.json
binds all54 live static arguments to actual header-name/shape/dtype histograms;
three noEngram static resources use bounded zeros and read no Engram sources.
This inventory does not prove numeric offset values, quant parity or payload
representability. Existing bounded loader controls remain the relevant evidence.

| Bytes | Prefill128 | Decode1 |
| --- | ---: | ---: |
| Real static weights, EACH rank |39,624,816,576|39,624,816,576|
| Bounded fixture static, EACH rank |6,463,879,104|6,463,879,104|
| Exact real minus bounded, EACH rank |33,160,937,472|33,160,937,472|
| Mutable/metadata/output host IO, full16-rank argument storage |7,609,876,736|247,111,680|
| Layer snapshots, already included in IO |6,712,197,120|52,439,040|
| Persistent paged cache/state, already included in IO |189,136,896|189,136,896|
| Generated named protocol windows, nominal EACH rank |680,528,320|4,663,744|

The static delta is exactly the other39 expert layers in w1/w2/w3 plus FP32
scales. All other static resource byte counts match the bounded fixture. Each
real w1/w2/w3 is11,324,620,800B per rank (shape40x24x5120x2304, w2 transposed
logical axes); each bounded counterpart is283,115,520B. These are the largest
individual contiguous device requests. Empty-allocation success, fragmentation,
actual requested-byte reservations and later reuse need measurement.

Static residency is explicitly supported by rank-owned empty allocations plus
StackedDeviceTensor, not inferred from a leading16 axis. No IO sharding is
assumed: the conservative device IO upper bound below charges the FULL host
argument storage per rank. Generated host orchestration slices each argument,
but backend import/storage extent and repeated-dispatch cached copies must be
measured before using smaller per-rank slice totals in a fit prediction.

The disabled Engram weight arguments still exist in the ABI and their inert
zero allocations remain in39,624,816,576B. No real table/hash/rotation payload is
read, and full original-model parity is not claimed.

## Host lifetimes and parameterized bounds

Let P=7,609,876,736B, D=247,111,680B and L=5,362,417,664B.

- Production prefill planning expression: 2*P+L+metadata+allocator retention+
  runtime host buffers+compiler/tokenizer+margin. The second P conservatively
  allows shared-memory conversion; it is not a measured permanent second copy.
- Production decode while the current driver retains its prefill dictionary:
  P+2*D+L+the same unknown terms. Matching-shape handoff returns contiguous cache
  aliases; only subtract189,136,896B if actual storage identity is verified.
  The prefill snapshot dictionary remains alive despite closing its device owner.
- Snapshots and caches are subsets of P/D; do not add them twice. Production
  generation does not create golden reference clones. Bounded validation does:
  one independent nonstatic input clone adds P for prefill or D for decode;
  actual saved128/stale-state branches and staged expectations add their own
  deduplicated storage. Distinguish these from a production fit requirement.
- No retained global static/rank/expert torch.stack payload belongs in the real
  streaming path. The bounded legacy fixture does retain large static host
  tensors; subtracting device static delta does not remove that host fixture cost.

SafeTensorReader creates raw bytes and bytearray before returning a tensor;
no payload cache is kept. TP slicing may retain a larger source storage.
Nonzero offset folding can hold source INT8, double adjusted/rounded intermediates,
packed/transpose copies and FP32 scales. L is a conservative serial planning
allowance, not a proof of allocator RSS. The largest included INT8 source Q is
41,943,040B; individual expert matrix leaves are11,796,480B. Explicit copy stages
are at most64MiB. The consumer uses explicit next()/del leaf, and its locals /
views leave scope before advancing; both generator and consumer weakref controls
passed in the frozen implementation. Parallel leaf reads/copies would invalidate
this serial allowance unless separately bounded.

Measure parent+children PSS/USS, shared storage and /dev/shm use as well as RSS;
adding fork-inherited RSS across16 children double-counts shared pages. Record
MemAvailable and allocator-retained peaks, file page cache separately. Reader
raw-byte duplication and public copy POSIX-SHM staging are different lifetimes.

## Per-rank device budget, bank count and reservations

Let W be actual static rank bytes, I the measured device IO/storage footprint
(or conservatively the full P/D above), b the ACTUAL constructed/reserved arena
bank count per rank, H the sum of four ring heaps per bank, C actual protocol /
HCCL/VMM cost, A allocator padding/cache retention, R other runtime/compiler
resources, and E other-device baseline/reservations. Plan:

    device_used_peak <= W + I + b*H + C + A + R + E + safety_margin

These terms are not independently measured yet. Public committed_device_memory
already counts worker tensors, pooled arenas and runtime allocations: do NOT
add W/I/bH to that same committed total again. Device-wide used=total-free also
includes other processes/driver/HCCL/VMM; its residual is not automatically our
runtime cost. Virtual peer mappings and rounded reservations differ from local
physical protocol bytes. Do not multiply nominal C by16 twice.

| Ring2 setting; other rings256MiB | H, one bank/rank | Delta vs default bank |
| --- | ---: | ---: |
|256MiB|1GiB|0|
|1GiB|1.75GiB|0.75GiB|
|2GiB|2.75GiB|1.75GiB|
|3GiB (REJECTED by actual PyPTO RunConfig)|3.75GiB hypothetical|2.75GiB hypothetical|
|4GiB|4.75GiB|3.75GiB|

The3GiB row is mathematical accounting ONLY and is rejected by the actual
PyPTO route. Other listed rows meet the public power-of-two constraint but are
not accepted fit configurations. Target list[0,0,H2,0] changes only ring2;
a scalar H2 enlarges all four rings and is a different budget. Multiply H by
actual banks and actual ranks; one2GiB-targeted bank is44GiB over16 ranks, while
a hypothetical3GiB-targeted bank is60GiB over16 ranks; a supported4GiB-targeted
bank is76GiB over16 ranks. A default capability/grant of two
resource sets is not evidence that two banks were constructed or retained.
Sequential synchronous dispatch, cached runner resources, prewarm, multiple
artifacts and quarantined failure resources can change this count. The same
normalized config must be passed on every dispatch, not only prepare.

Reviewer's3GiB arithmetic proposal was rejected by the public RunConfig
validator before any native prefix execution. Observer conditionally assigned
one4GiB replacement trial (4.75GiB per bank) after exact public-path CPU/config
and free-memory gates. This readiness neither repeats nor claims that trial. Prefix40 allocation/liveness/terminal-state gates are still
open in Main evidence. Both old failed prefix peaks are partial execution;
13.7GB observed on one card is NOT a full-forward peak or a fit basis.

A conditional component sum using full-storage IO and ONE4.75GiB bank is
W+P+4.75GiB+nominalC for prefill; it still omits A/R/E/margin and unproven extra
banks. It must not be presented as measured residency or guaranteed fit.

## Exactly what bounded acceptance should measure next

The observer-owned reviewer can collect these within its fresh frozen bounded
run, without real payloads or changing model/runtime source:

1. Actual allocated device IDs, normalized mode/platform, frozen source hashes,
   concrete57 static shapes/dtypes, explicit ring array on every dispatch,
   granted/constructed/resource-set identities from available native logs.
2. Per rank public device_memory_info=(free,total) and committed_device_memory
   before prewarm, after prewarm, after each named static allocation/upload,
   before/after prefill dispatch, after phase close, before/after decode owner,
   and after every decode/control call and final close. Query failures fail the
   measurement; no fabricated zero. Record active/cached bank count if exposed;
   if unavailable, retain it as unknown and bound from allocation logs, not
   Python frame count.
3. Per-rank requested static upload/allocation totals and largest requests;
   verify the measured bounded static component Wb=6,463,879,104B exactly once.
   Read back sampled nonidentity first/last leaves and prove handle reuse.
4. Device-wide HBM samples across dispatch and public committed/free snapshots;
   terminal/copyback status, successful owning19-output gates, actual returned
   state and same-artifact three-step parity/page continuation with numerical
   stale-published-row rejection. A failed prefix has no accepted peak basis.
5. Parent/child RSS/PSS/USS, /dev/shm and shared tensor/storage identity at
   input build, clone build, prewarm, serial leaf transform/copy, handoff,
   repeated calls and phase close. Separate reusable pools from live payloads.
6. Handoff exact byte/storage identity, release of prefill device resources
   before decode, no simultaneous phase weight sets, allocator/free baseline
   restoration and no failed-domain quarantine. Retained prefill HOST snapshots
   must remain accounted if the current driver still keeps them.

Only with matching phase/owner/config/capacity and validated successful execution
may a prediction use:

    real_peak <= measured_bounded_peak - measured_Wb_counted_once
                 + 39,624,816,576 + delta_IO + delta_live_scratch/banks
                 + delta_allocator_padding/reservations + margin

If resident binding, completed phase peak, bank count or workload-dependent
scratch changes are unknown, leave the delta prediction conditional. Nonuniform
real routes can alter task timing/lifetimes even at the same declared capacity;
no zero delta is assumed. Public committed excludes HCCL/VMM, so combine public
counters and device-wide baseline consistently rather than mixing totals.

## Next implementable bounded milestone and exclusive future ownership

Current unit is READ-ONLY; no worker is assigned. After independent functional
and resource disposition, a future bounded unit can assign Main generate.py
only for approved phase resource config, memory telemetry and guarded sequential
phase lifecycle; keep loader/operator sources frozen. One disjoint artifact-only
worker may own new fit_probe.py / synthetic reader fixtures/results, never the
model, ledgers or runtime. No new loader rewrite is presently required.

First run tiny-to-bounded synthetic reader leaves through the SAME public owner
with nonidentity BF16/INT8/NZ/FP32 scales, offsets, resource coverage and rollback;
measure source backing plus serial copy and verify parent/child memory. Record
per-rank free/committed before any growth. Only after conservative prediction
fits with explicit headroom and observer assigns a separate memory-fit step may
an empty-allocation ramp test the11,324,620,800B largest contiguous rank buffer
and planned total static residency. Stop before payloads and on any failed
allocation; cleanup/free restoration is mandatory. This proposal authorizes
neither that large allocation nor full real checkpoint materialization now.

The hard full-weight guard must stay until independent functional19-output /
returned-state continuation gates, public scalar/stream binding and cleanup,
real40 header/ABI coverage, measured bounded memory and a documented safe fit
prediction/margin all pass. An observer-assigned fit allocation step must then
confirm largest-buffer and aggregate headroom with sequential phase release.
Only a later explicit full real-weight generation scope may enable real payloads,
record actual peak/free memory, tokens/text and noEngram labeling. Successful
allocation alone does not prove numerical/original parity or coherent text.
