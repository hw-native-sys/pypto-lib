# Scalar, C2 reference precision, and streaming integration — FROZEN BLOCKED HANDOFF

This is Main self-evidence, not independent acceptance. All unit-owned writers and jobs are PARKED/ENDED. No real checkpoint
payloads were read and no full generation or original-model parity is claimed.
The first generation path is explicitly noEngram (user decision10).

## Implemented source scope

Only generate.py, weight_loader.py and compressor.py differ from before.json;
the other42 model sources remain equal. Workers are parked. after.json and artifacts.json seal the final source/evidence; their exact hashes are in FREEZE.json.

- generate.py marshals declared zero-dimensional FP32/INT32/BOOL scalars to
  supported Python values. It resolves complete dual-phase contracts from
  annotations with explicit dynamic capacities, compiles from signatures with
  zero tensor samples, and streams serial WeightLeaf payloads into public
  rank-owned empty allocations. The actual driver uses this source/owner and
  closes the prefill phase before opening decode. Shared mutable host IO is
  copied H2D/D2H each dispatch; KV device residency is not claimed. Full weights
  remain blocked by a hard guard even with --allow-full-materialize.
- weight_loader.py yields individual transformed dense/source/expert leaves;
  no aggregate layer/rank/expert stacks are called by its streaming path.
  W8A8 offset-fold return, transpose, NZ packing and FP32 scale encoding remain
  covered. Top-level malformed shape/dtype rejects before payload reads. C1
  prefill lacks the singleton source axis that decode retains; both concrete
  ABIs are validated. Frozen worker_loader initial evidence remains unchanged;
  worker_loader_reviewfix contains the correction and accurate control counts.
- compressor.py changes only C2 golden projection evaluation to FP64 dots
  rounded to FP32 before the existing pooling/state/cast logic. No device
  compute, comparator threshold, C1/index/Engram arithmetic changed. Permanent
  analytic even-to-odd/remapped-request coverage rejects the prior golden.

## Completed decisive checks

All checks use task-submit. NPU tasks explicitly normalize execution_mode=False
as ONBOARD, require generated a2a3 metadata, and retain native runtime logs.

| Check | Task | Result |
| --- | --- | --- |
| Exact original FP32 scalar tiny owner, three calls | 092919_9245096453 | PASS; state2160/4704/7888, requested512B once |
| Isolated unchanged native C2 projection | 092840_92073122209 | PASS vs exact math; CPU FP32 reduction mismatch localized |
| Diagnostic full40 decode actual C2 operands | 094906_98427432350 | Native86.02s;48 inputs bitwise equal staged CPU inputs, actual state0 bad lanes vs independent FP64 rounded, prior CPUFP32 exactly49152 bad |
| C2 correction original/analytic/permanent/AST controls | 100321_126538713370 | PASS; only golden evaluation and coverage differ |
| Uninstrumented owning40 decode, auto16 | 100351_126889319863 | ALL24 outputs numerical PASS, including unchanged1e-5 C2 state |
| Exact streaming owner CPU + coverage/cleanup | 094406_96529716135 | PASS; six structural/partial-cleanup controls |
| Actual40 dual metadata contracts + complete IO | 100416_12713947306 | PASS;100prefill/96decode args,57static each,40x24 experts, positions128/129/130 |
| Stream public empty-rank allocation, offsetcopy/readback and three scalar calls, auto2 | 100416_12713951128 | PASS;3072B,24copies128B, three FP32/INT32/BOOL calls, states7312/20864/28400, source mutation ignored |
| Asymmetric nonconstant-K four-term analytic golden control | 100826_177717021931 | PASS; first helper TypeError100759 preserved |
| Real40 prefill128/decode1 signature-only compiles | 101019_178466616543 | PASS; no fixture calls or tensor samples; ruff/py_compile PASS |
| Actual reader backing storage on tiny synthetic safetensor | 101247_183052312607 | PASS;64B visible vocab slice retains256B source storage |
| Explicit consumer leaf lifetime | 101313_183298918856 | PASS |
| Stream callback binding plus leaf lifetime | 102208_187152321059 | PASS; one header-loader instance per rank |
| Loader top-level/phase ABI refinement | 100322_126563330433 | PASS;8 pre-read reject controls,8 normal exact,7C1x2phasesx2ranks;57static each phase |

Independent observer-owned pre-review used exact rational Fraction sums on all48
saved BF16 actual operand sets and confirmed the mathematical reference,
separately from the candidate. That supports this bounded reference correction;
it does not establish the original torch_npu F.linear backend's bitwise behavior
(torch_npu is unavailable). Production decode comparators remain conditioned on
actual previous-layer snapshots; strict unconditioned full-model parity remains
open.

## Continuation status — blocked, two failures preserved

First actual16 task100912_177992928812 failed the FIRST prefill128 dispatch with
507018/21008, no terminal record or copyback. No decode/control result exists.
Own device0/6 logs identify an empty ring2 heap of268435456B rejecting one
525635584B request. Generated alloc_1 has525634176B aligned tensor payload plus
1408B request metadata. Immediately co-live alloc_2 has1008107520B tensor payload;
combined conservative bound1533747200B suggested trying a2GiB ring2. This was a
conservative choice, NOT a runtime-required power of two. The public docstring
mentions powers of two, but independent installed-source review reports the
implementation does not require that. Old budget assertions that omitted
metadata or the second co-live allocation are preserved in tasks101746/101824.
Corrected budget check101859 PASS.

One fresh actual16 retry101954_185389724322 explicitly passed public
ring_heap=[0,0,2147483648,0] to prefill prepare AND EVERY dispatch through stored
normalized config, leaving decode defaults. Effective2GiB ring2 is confirmed
by actual device logs. It advanced to9538 completed /15866 submitted tasks in
the recorded rank0 snapshot, then failed a further contiguous-placement/lifetime
boundary: top1533743104, tail1008107520, used525635584, total2147483648,
reportedavailable1008107520, requested1008107520. Endspace613740544 was too short;
the pinned allocator's wrap branch requires tail STRICTLY GREATER than the
request, so equality rejected. Head task35 remained scope-owned/unreleased,
current36. These are recorded facts, not a claim of a runtime defect or proof
that all source lifetimes are necessary. Broader allocation/producer/consumer
scope analysis and a justified resource/source remedy remain open.

Both failures hung during failed cleanup and were terminated ONLY through
 task-submit, exit130; recorded Main/rank PIDs were absent afterwards. No admin,
explicit device reset, other-task termination, runtime/harness/model compute edit
or third retry occurred. All logs, failed original helper source and both device
log sets are retained. results_retry/hbm_live.json preserves sampled peaks;
there is no successful terminal/copyback or validated prefill/continuation.

The target128prefix -> actual-returned decode128/129/130 on one owner/artifact
plus stale-published-row numerical rejection remains UNRUN beyond first prefill.
No actual returned-state handoff, three-step lifecycle or control rejection is
claimed. Public committed/free-memory counters and actual arena-bank grants
were not collected by this already-running helper and remain additional evidence
needs. One2GiB-targeted bank sums2.75GiB/rank; actual bank multiplicity is unknown.

## Memory and remaining gates

The real static weights are39624816576B/rank, header-only. This is not a fit
measurement. The earlier ~5GiB live-leaf estimate was incomplete: the reader
constructs raw bytes plus bytearray for a full5295308800B BF16 source, giving
10590617600B during that read alone. A contiguous TP vocab slice can retain the
whole5295308800B backing storage. Add64MiB serial copy staging, headers/maps,
mutable shared IO, allocator retention and runtime/compiler/device scratch.
The uploader releases its local leaf and views before advancing the generator;
CPU weakref controls cover the consumer as well as generator lifetime.

Native stream proof maxRSS300548KiB and sampled absolute HBM peaks4583/4297MB
are tiny-case observations. Upload counters count requested bytes, not bus
traffic or bandwidth. They do not prove full real host/HBM fit. HCCL/VMM and
cached arena resource-set multiplicity remain outside simple input bounds.
Full checkpoint allocation, real tokens/text, coherent text acceptance, strict
original parity, full Engram/history hashing, general F09 dynamic lifecycle and
historical actual-hardware F05/F07 negative controls remain open.

## Frozen resumption

Read after.json/FREEZE.json/commands.tsv, REPORT.md and main/continuation/
failure_analysis.json first. Root owns subsequent independent acceptance. Main
and workers must remain parked with no source changes or duplicate NPU jobs.
Independent reviewer may reproduce scalar/stream/decode gates in fresh outputs,
review strict contiguous allocator placement and scope-held head task35 against
generated alloc_1/alloc_2, then perform only an observer-assigned bounded isolation.
Full weights remain held. Old frozen worker reports/manifests were preserved.

FinalCPU102918_191129818839 PASS validates3ownedchanges/42unchanged, proof
records and exact PreparedDispatch AST equality to the passing native snapshot.
The initial final collector102818 failed its nested task-status CLI call before
writing a manifest; raw log is preserved. Host administrative status checks
separately confirm all27 listed tasks COMPLETE (including preserved failures
and two terminated chain attempts); main/job_statuses.json records every result.
