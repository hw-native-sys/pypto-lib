# Independent A5 reference / A3 topology review — SOURCE ONLY, FINAL / PARK

New user scope permits configurable topology; this report implements nothing.
No imports, tests/tasks, compiler, fixture, checkpoint headers/payload, prepare,
NPU, grant, runtime/dependency/A5/A3 edits. A3 baseline is frozen B30b generation
with mathematical files from immutable A8153; Main's evolving topology implementation
was not reviewed. Findings were sent directly to Root/Main during review.

## What A5 actually supports

A5 config.py274..303 parses --tp N/--tp=N BEFORE imports, choices1/2/4/8 (default1),
and independent --ep2/4/8 (default2), checking attention64/outputgroup8 divisibility
and expert384 divisibility. EP is not derived from TP in that base configuration.
That is operator configuration, NOT full-model proof for every Cartesian pair.

- A5 prefill_layer.py34..44 derives EP=TP*DP, DP86=EP/TP. Its standalone CLI865
  accepts TP1/2 and DP2, explicitly says TP4/eight cards unvalidated. This is a
  useful colocated grouping reference, not TP8 full-generation acceptance.
- A5 prefill_fwd.py62 explicitly requires TP1 and DP=EP; checkpoint resident owner
  1226 repeats that constraint. Its small CI default EP2 uses128 experts, while
  checkpoint EP4/8 selects actual384. Do not transplant the128-expert fixture as
  the model. CLI1893/1949 allows onlyTP1.
- A5 decode_fwd.py51 keeps all64 attention heads/all8 outputgroups per rank;
  its final_output842..846 passes group_base=rank,tp_rank=0. It documents TP1 at995
  and its CLI1221 exposes EP, not a general --tp fullmodel path. A5 LM-head alone
  supports TP1/2/4/8 with separate DP choices; that does not change this full path.
- A5 uses MXFP4 routed/MXFP8 paths and A5-specific C++ expert restrictions/no-sim.
  Preserve A3 W8A8/FP32 scales/BF16 boundaries; configuration port is not permission
  to change quantization or assume A5 kernels run on A3.

## Valid narrow colocated A3 contract

Existing A3 has WORLD16=DP4*TP4 and EP16 over that SAME world. A concrete first
configurable family is WORLD=EP, EP%TP=0, DP=EP/TP. Rankr has tp=r%TP,
dp=r//TP, group_base=(r//TP)*TP, ep_rank=r. Attention/embedding reductions stay
within contiguous TP peers; MoE routes across all EP/WORLD ranks. Independent
selection of TP and EP is constrained; WORLD is NOT TP*EP. WORLD>EP would require
separate replicated EP groups, peer group bases, rank%EP expert ownership, loader
replication and corresponding goldens. It is a distinct unimplemented contract.

Validate positive integer sizes, EP dividing384, TP dividing64 heads/8 outputgroups/
129280 vocab, TP<=EP and EP divisibleTP, exact world/device count. Current model
head/group divisibility gives natural TP1/2/4/8. Supported EP policy must be explicit,
not inferred from divisibility or A5's smaller choice set. Import-time dimensions
must freeze consistently BEFORE every dependent module; normal argparse must consume
and confirm the same flags, including both spelling forms. Changing args after import
or reusing a cached config from another topology is invalid.

TP8/EP16/WORLD16/DP2 is the recommended FIRST nondefault real text target (still
requires implementation/compile/numerics/admission/actual Root GO). It has8 local
attention heads,1 outputgroup,16160 vocabulary rows,24 experts/rank; old default
has16 heads/2groups/32320 vocab/24experts. Local shape/tile support needs fresh
compilation/device proof, not divisibility alone. All32 indexer heads are currently
REPLICATED per rank: indexer.py37/layer_reference.py52/weight_loader.py44..47.
No model calls tp_score_reduce. Do NOT shard indexer or add a score allreduce from
a stale tp_reduce docstring or unused driver LOCAL_INDEX_DIM. TP division of32
index heads is not an additional requirement for this replicated design.

## Concrete blockers a superficial config edit misses

1. **EP count metadata width.** moe_exchange.py55 fixes COUNT_PAD32. EP8 means
   N_LOCAL48, but count publication uses a [1,N_LOCAL] tile and recv_meta/meta_local
   [EP,32]. Smaller EP is out of bounds. Derive width>=N_LOCAL with128-byte row
   alignment (e.g ceil(N_LOCAL/32)*32 INT32 lanes) across every local/remote window,
   field consumer and diagnostic capture. Count shape24/source405 receipts are not
   topology-general. Do not replace hardware16/32 blindly with EP.
2. **T1 expert receive slabs.** moe_exchange.py106..125 computes recv_max=EP*capacity,
   chunks=recv_max//EXPERT_TILE16, then reshapes slab[N_LOCAL*chunks,16,...] to
   recv_rows=N_LOCAL*recv_max. EP8/capacity1 gives chunks0 despite384 logical rows;
   EP4/T1 also fails. Consistent per-expert padding/strides/compact/expert/return
   offsets are needed, or explicitly reject unsupported shapes. ceil chunks alone
   with unchanged reshape/offset strides is not a complete repair. Communication
   route totals must still reflect real EP*active, not padded tokens. Keep zero-local
   ranks serving remote work/completion anchors and returned TaskID dependencies.
3. **Admission/resource/driver fixed16.** Frozen generate2450ff hardcodes real40
   static39624816576B/rank, lenIDs16, remaining arrays16, range16 memory queries,
   [16,*tail] ABI checks2728, decoder IO/comm reserves and queue --device-num16.
   New source hash alone does not bind import-time topology. New plan/source/Root
   receipt/compile metadata/effective args must bind TP/EP/DP/WORLD, ordered physical
   allocation and topology-specific target shapes/bytes/read schedules. Preserve
   HARD default and exact45 math/producer lineage; reject stale default-topology
   plans/grants even under identical source45. Do not loosen old immutable receipts.
4. **Loader ownership and consumer views.** WeightLoader232..240 bounds rank byEP,
   uses ep_rank=rank and tp_rank=rank%TP; expert key441 is ep_rank*N_LOCAL+local.
   TP slices Q_B/sink/Ogroups/vocab, replicated Q_A/KV/index/shared weights must
   remain consistent with kernel contracts.57 resources remain, but shapes/bytes,
   read reuse, number of consumers and C1 views depend on topology. Same-worker
   static upload once/earlyIO/sameowner/C1 zero-offset views/free bases once stay
   required. Old912/112 views are specifically WORLD16, not universal constants.
5. **Goldens and sampling.** layer_reference TP reductions/count consistency already
   use TP_SIZE; every active-cache/state/input-replication group must follow the new
   grouping. Full synthetic MoE golden currently _moe_weight_shard132 selects rank0
   because fixtures replicate one expert shard. Such a PASS cannot independently
   prove distinct global expert owners. Use a small nonidentical rank/expert-weight
   oracle control for ownership. Same caveat applies to rank-repeated dense/vocab
   fixtures: boundary/shard permutation controls must not be masked by symmetry.
   Driver greedy sampling already uses WORLD/TP and global vocab offset; external
   result validators must expect DP2/TP8 for that case, not hardcode four DP votes.

Preserve HC_MULT4, state parity2, C1 index-source period4, expert GEMM tile16,
auxiliary 16 lanes and64/128-byte cacheline ownership: these are architectural or
hardware constants, NOT ranks. Kernel and golden changes must agree on ownership,
not reproduce the same mistaken rank constant or relax comparator thresholds.

## Resource-based first target

Routed realE40 INT8 matrices+FP32 scales perrank at EP16:
3*(40*24*5120*2304)+2*(40*24*2304*4)+(40*24*5120*4)=34011217920B.
At EP8 this alone doubles to68022435840B, above recorded A3 physical total about
65787658240B BEFORE dense/shared/runtime. Experts are not TP-sharded; higherTP cannot
rescue EP8 realE40 here. SmallerEP may be metadata/synthetic coverage, not automatic
real-weights permission. TP8/EP16 keeps the expert lower bound and reduces only
TP-sharded dense/vocab resources, not half the whole model. Fresh topology-specific
header-only budget and actual resource receipts are mandatory; no fresh fit/peak
or throughput claim is made from this source arithmetic.

## Recommended finite FUTURE validation matrix (not executed)

| Intended coverage | Configurations / meaningful check |
| --- | --- |
| Default inheritance | TP4 EP16 DP4 WORLD16, exact source/ABI/loader schedule relationships |
| First nondefault text | TP8 EP16 DP2 WORLD16; fresh normal prefillN3+decode1 signatures, all2 DP votes/8 TP candidates, actual prompt/history/EOS, sameowner one-upload |
| Smaller-EP shape boundaries | TP4 EP8 DP2; TP8 EP8 DP1; TP2 EP4 DP2; TP1 EP2 DP2; T1/T2 and active0/1/full, after coherent metadata/slab padding |
| Bad topology controls | TP0/EP0, TP3 EP16, TP16 EP16, TP8 EP4, nondividing TP4 EP6, EP5, world/device mismatch, mismatched import-time vs runtime flags |
| Mathematical ownership | TP-boundary global vocab IDs/unique shard winners; nonidentical attention shards; route to remote first/last expert with distinct rank-local weights; idle source remains service rank |
| Lifecycle/binding | Repeat sameowner two programs/state handoff, C1 views/basefreeonce, stale topology source-plan-receipt rejection, no aliasing expected/output storage |

Use fresh process/module context per import-time configuration. EP1 or odd/divisor
EP policies should be explicit; add them only if advertised/implemented rather than
silently claiming all mathematical divisors. Invalid matrices need rejection before
fixtures/prepare. SmallerEP real generation should fail the proven static lowerbound,
not trigger a costly failed weight upload. No old B30b/E1/TP4 artifact relabel asTP8.
No actual test/compile/NPU commands are authorized by this report. Main owns implementation;
Root will separately assign frozen-candidate acceptance. No general serving/unequalEOS/
original torch_npu parity/Engram or text-quality closure is inferred.
