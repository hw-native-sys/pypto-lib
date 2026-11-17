# Configurable A3 W8A8 topology — source-only resource plan

Recommended nondefault full-real candidate: TP8/EP16/WORLD16/DP2. Its predicted
57-resource static residency is38,328,513,216B/rank, only1,296,303,360B below the
old TP4/EP16 default. EP8 cannot hold the fullE40 routed weights under the
unchanged whole-expert layout, even before other statics or runtime resources.
These are source/header-META arithmetic, not a fresh metadata trace, allocation,
fit/forward/quality acceptance or new production authorization.

Scope: no model edits/imports, CPU jobs/tests, checkpoint header scan, tensor
payload, fixtures, compile, prepare or NPU work. Only existing source/metadata
text and hashes were read. Main confirmed WORLD=EP, DP=EP/TP, whole-expert EP
partition independent of TP. Live source is worker-owned/evolving, so remaining
review used frozen B baseline at
../configurable_parallel_1794108101/baseline/models/deepseek_v4_1_flash_a3_w8a8.
Its baseline_source45.json hashes to30b7106a5427c283917a7eeeec65aab598ab98bf5a70f1667285169c9c6d8eca.
All source identities read are in source_pins.sha256. Old cd8b plan is the
historical TP4/EP16 resource anchor, not a new-topology plan.

## Parallel meaning and valid geometry

A5 config.py274-318 parses --tp and --ep from process argv before dependent
module imports; TP sizes1/2/4/8 and EP sizes2/4/8 are independently selected,
with attention-head/output-group/expert divisibility checks. This is the
interface/configuration reference, not permission to migrate its FP8/FP4 math
or assume every A5 composed-entry combination is proved. A5 prefill itself has
additional TP1/DP conditions. A3 must declare its actual supported set.

For the proposed A3 contract: WORLD=EP, DP=EP//TP, EP%TP=0, TP<=EP; preserve
TP4/EP16 defaults. TP divides64 heads,8 output groups and129280 vocabulary;
EP divides384 experts. TP16 fails the8-output-group constraint. Only explicitly
supported discrete sizes and available WORLD may be admitted, not every integer
that happens to divide384. Rank r has tp_rank=r%TP, dp_rank=r//TP,
group_base=(r//TP)*TP, and expert range
[r*(384/EP),(r+1)*(384/EP)). All384 experts occur once over the EP world;
DP does NOT multiply local expert ownership or divide experts a second time.

For TP8/EP16:8 heads,1 output group,16160 vocab rows and24 experts/rank;
TP groups0..7 and8..15, DP2. Two groups must both vote in strict generation;
old literal4DP acceptance is stale. No nondefault runtime is validated here.

## Exact static decomposition

With unchanged full40-layer/noEngram ABI, D=5120,I=2304,E=384:

Routed INT8 matrices/rank =3*40*(384/EP)*5120*2304
                         =543,581,798,400/EP bytes.
Routed FP32 scales/rank =40*(384/EP)*(2*2304+5120)*4
                       =597,688,320/EP bytes.
Offsets are folded into INT8 weights and are not additional device tensors.
They remain source reads/host transform inputs. No scale-format change occurs.

The exact source-derived formula is:

S(TP,EP) =544,179,486,720/EP +10,370,426,880/TP +3,020,991,936 bytes/rank.

This reproduces old39,624,816,576B exactly. STATIC_LEDGER.csv itemizes all57:
6 resources scale with EP,7 with TP,44 are replicated. This is an arithmetic
cross-check against existing shapes/old metadata, not newly executed CPU proof.

Only embed/head, sink, wq_b(+scale), wo_a and wo_b comprise the TP term.
AtTP4 their bytes are330956800+330956800+2560+419430400+1310720+
671088640+838860800=2,592,606,720. AtTP8 every one halves. Representative
TP8 shapes: embed/head[16160,5120]BF16;wq_b[40,1280,4096]I8;
scale[40,4096]FP32;wo_a[40,1,1024,4096]BF16;wo_b[40,5120,1024]BF16.

Fixed3,020,991,936B consists of sharedFFN1,417,134,080;
wq_a/wkv+scales367,288,320; norms972,800; HC209,723,840;
router314,634,240; C1/C2 compressors36,704,256; indexers45,220,864;
retained disabled-Engram ABI buffers629,313,536. SharedFFN is replicated.
The actual loader keeps all32 index heads per rank: LOCAL_INDEX_DIM=4096;
C1/C2/reindex wq_b and scales must NOT halve with TP. Stale loader header text
or historical weight-map shard descriptions cannot override those contracts.
Disabled Engram still allocates existing zero buffers; removing them changes
ABI/resources and cannot silently be assumed in this formula.

TOPOLOGY_BOUNDS.csv gives exact candidates. TP8/EP16 predicts
38,328,513,216B/rank and613,256,211,456B global; largest w1/w2/w3 target stays
181,193,932,800/EP=11,324,620,800B. TP2/EP16 andTP1/EP16 instead increase
statics to42,217,423,296 and47,402,636,736B/rank; fresh admission is required.

EP8 routed-only=68,022,435,840B/rank (matrix-only67,947,724,800), exceeding
historical public total65,787,658,240/65,796,046,848B. Even a nominal64GiB
budget leaves697,040,896B after routed+scales, below the existing4.75GiB
ring cost before all dense/IO resources. Best TP8/EP8 whole57 is
72,339,731,136B/rank. Thus fullE40 EP8 and smaller EP are statically unfit
under current co-residency, independent of a contiguous-largest test. Smaller
EP can still receive separately scoped meaningful component validation; that
must not be promoted to full-real generation fit.

## Source reads, host lifetime and one owner

The loader reads full source tensors before slicing in _vocab_slice,
_head_slice,_w8a8_sliced,_wo_a,_wo_b. TP8 makes device targets smaller while
WORLD16 and expert ownership stay unchanged. Under the unchanged read schedule,
TP8/EP16 therefore predicts the SAME164192 calls/745846365184 cumulative source
bytes/110368 leaves as old TP4/EP16, not half the source/host traffic.
This prediction must be confirmed by the new source-bound metadata tracer.

Routed source reads=9*40*384=138240 globally (matrix+offset+scale per W1/W2/W3).
Routed source bytes=544,777,175,040 globally. Residual old dense schedule is
1622 calls/12,566,824,384 bytes per rank. Thus candidate formulas are
source_calls=138240+1622*EP;
source_bytes=544777175040+12566824384*EP;
leaves=92160+1138*EP, contingent on unchanged leaf/chunk/read callgraph.
EP8 would yield151216 calls/645311770112B/101264 leaves but remains unfit.
A future optimized loader could change these numbers only with new traced proof.

Largest whole source remains1,323,827,200B for embed/head, even though TP8's
retained shard is165,478,400B. The existing5,362,417,664B source-component
planning allowance cannot halve because TP doubles. Offset validation, FP64
q-offset representability checks, transposition/NZ packing, per-output FP32
scales and source backing lifetime must remain intact. No global CPU weight
stacks or unbounded fixture can be introduced to obtain a new topology.

One-upload accounting remains57*WORLD bases, with decoder0 new static
read/bytes/H2D/allocation/copy. C1 has7 full-byte singleton views per rank
(prefill no leading1, decode leading1) and50 same-shape aliases, all bound to
actual source/destination shape/dtype/byte extent/offset/positive base addresses.
For WORLD16 this remains912=112C1+800same; WORLD8 would456=56+400, not912.
Largest target scales with EP only. World-wide H2D=EP*S(TP,EP); global expert
bytes remain fixed, replicated dense bytes grow with EP/TP mapping.

## Required fresh metadata, admission and runtime evidence

1. New source45 and math baseline, immutable plan path/hash, checkpoint/index/
   tokenizer/library identities plus explicit tp,ep,world,dp,local_experts,
   local_heads,local_o_groups,local_vocab and logical-rank/device mapping.
   Regenerate contracts and source-bound targets/read ranges/leaf offsets/
   scales/layout/bytes for every phase and rank. Existing source_descriptors
   can be retained as read-only checkpoint metadata ancestry; old rank-local
   summaries/target geometry cannot stand in for new topology. New resource
   projection must be computed, not rename/reuse old e13 as a fresh ledger.
2. Both fresh zero-sample E40 prefillN3/decode1 signatures on the new source and
   topology, formal100/96 and static57 only if preserved by actual ABI; rank
   axes WORLD, all TP slices, replicated indexer and persistent-cache InOut
   ownership. Exact program metadata/generated-file hashes and actual future
   runtime artifacts must remain distinct. No old611/704/B30b compile relabel.
3. Future actual receipt requires topology/code/plan/shape-budget bindings and
   actual ordered WORLD IDs after queue allocation. Normal CLI --tp/--ep must
   propagate to inner re-exec and compile processes before config imports;
   runtime groups, all-gathers/a2a, route buffers, sampling and DP votes must
   agree. WORLD must drive task-submit auto --device-num WORLD. Reject stale
   fixed16/DP4 receipts/default old45 plan even if token IDs remain identical.
4. Recompute first/live HBM reserve components from actual nonstatic tensor
   shapes/lifetimes, collectives/routes/retained pipeline slots, TP/EP and
   generated-call cap. Derive pre/post allocation remaining-static accounting
   from each rank's actual targets, and include conservative future-call IO.
   HOST/DISK/SHM/source/timeout/public query budgets remain explicit planning
   policies; do not call them measured. Old18,088,299,200/9,797,894,144 floors,
   old prewarm/free and ring observations are historical inputs, not automatic
   new topology measurements. Public free already subtracts prewarm; no
   committed/ring credit or168-banks fiction. Keep forward/upload/transient
   peaks unknown until measured; no circular requirement to run full forward
   before admitting a bounded first operation with positive policies/liveguards.
5. Actual auto-WORLD operation requires current-source/tokenizer PREQUEUE,
   allocated-preexec and post checks; fixed noEngram/W8A8; exact shared-owner
   accounting; all57 initialized before any forward; all DP groups finite/
   consistent; genuine previous-token/returned-cache history and EOS. Preserve
   per-rank query telemetry, borrowed view ownership and first-error/attemptall
   normal cleanup. Independent postterminal audit follows SAME-task terminal,
   raw WORLD-lock release/queue state/known PID births and mandatory POST.

Public backend/TMR configuration, approved bounded query cadence, timeout
relationships and strict telemetry policy do not silently change. Each unknown
inventory/memory sample remains unknown, known partial sums stay separate,
whole strictFAIL persists if coverage is incomplete. Any source/header/shape/
query/cap/DP/nonfinite/resource/deadline/cleanup failure stops with partial
receipts and safe owned cleanup; no blind retry/heap resize/threshold change.
New TP8 numerical/public-runtime/resource acceptance is pending Main candidate
freeze and separately delegated validation; this report introduces no extra
benchmark/controller or device test prerequisite.

Historical SourceB real40 owner40 remains wholeNOT_ACCEPTED218 gaps, CLI0/
15decode/bounded cleanup accepted only as recorded; no quality/parity/peak
claim is repaired by this planning work. User decision11 authorizes progressing
to real nondefault generation, but this source-only review submits no task or
grant and proves no new topology physically fits. FINAL/PARK; sharedwrites0.
