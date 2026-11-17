# Scalar, C2 reference precision, and streaming integration — IN PROGRESS

This is Main self-evidence, not independent acceptance. No real checkpoint
payloads were read and no full generation or original-model parity is claimed.
The first generation path is explicitly noEngram (user decision10).

## Implemented source scope

Only generate.py, weight_loader.py and compressor.py differ from before.json;
the other42 model sources remain equal. Workers are parked. The exact final
manifest will be sealed only after all current jobs end.

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

## Continuation status — pending bounded retry

First actual16 task100912_177992928812 failed the FIRST prefill128 dispatch with
507018/21008, no terminal record or copyback. No decode/control result exists.
Own device0/6 logs identify an empty ring2 heap of268435456B rejecting one
525635584B request. Generated alloc_1 has525634176B aligned tensor payload plus
1408B request metadata. Immediately co-live alloc_2 has1008107520B tensor payload;
combined bound1533747200B requires the smallest power-of-two ring2=2GiB.
Old task hung during failed cleanup and was terminated through task-submit only,
exit130; all its recorded Main/rank PIDs were absent before retry.

Budget check101859_185149019237 PASS; prior two calculation assertions are
preserved (first omitted request metadata, second exposed co-live alloc_2).
One bounded fresh retry101954_185389724322 is RUNNING with public
ring_heap=[0,0,2147483648,0] for prefill only, others at defaults, and explicit
stored normalized RunConfig on EVERY dispatch as well as prepare. It is not a
runtime/harness/source patch. One bank sums2.75GiB/rank; actual bank multiplicity
must not be inferred. results_retry stores public config and live HBM samples.
The target is real128prefix -> actual returned decode128/129/130 on one owner /
artifact plus numerical stale-published-row rejection. No success yet.

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
