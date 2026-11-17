# V4.1 A5, then V4 DSpark: decode state and pool differential

2026-11-06. Read-only study, performed in the requested order. No model,
golden, comparator, PyPTO checkout, or checkpoint edits; no compile, NPU
submission, external filing, or acceptance closure. Native goal observed null.
Repository HEAD: `b6d17d9fa7a1e3420fbb7ac3c108fee194ec06fc`.

**Recommendation:** investigate state ownership and pool publication as separate
questions. A5 and DSpark show concrete state-ownership decompositions, but neither
demonstrates an equivalent working pair of externally exposed V4.1 C2A/pool
outputs. The current evidence does not prove that the two failures share a
mechanism, and does not establish a runtime root cause. Before another rewrite,
request a bounded diagnostic unit that holds the compiled decode structure fixed,
separates request/slot geometry from pool view lifetime, and records actual task
execution and addresses. This recommendation is not authorization to run it.

## Evidence contract

Paths below are relative to the repository root unless marked otherwise.

| Alias | Source/build |
| --- | --- |
| A5 | `models/deepseek_v4_1_flash/` (semantics/decomposition reference only) |
| A3 | `models/deepseek_v4_1_flash_a3_w8a8/` |
| DS | `models/deepseek_v4_flash_dspark/` |
| D | `build_output/_jit_decode_fwd_20261106_081644_8sx2hdyn/next_levels/decode_fwd_rank/` |
| P | `build_output/_jit_prefill_fwd_20261106_081049_ksr_kp03/next_levels/l2_prefill_fwd/` |
| S | `build_output/_jit_l3_decode_fwd_20261106_024437_3y8qfqeb/next_levels/_decode_fwd/` |

`a5_excerpts.txt`, `a3_excerpts.txt`, `dspark_excerpts.txt`, and
`generated_excerpts.txt` preserve numbered excerpts and full-file SHA256s.
`inspected_manifest.json` hashes those inputs. `source_before.json` and
`source_after.json` match for all **140 top-level Python files** in the three
trees. This is not the historical canonical A3 hash scope and excludes generated
Python and C++ files. C++ files are hashed as inspected inputs, not claimed to
have a before/after baseline. `collect_evidence.py` reproduces collection through
host-only file reads, with no model imports or execution.

The existing A3 run logs explicitly associate D/P with their perf-hint paths
(`reval_decode_uniform_expert_abi.log:106` and
`reval_prefill_uniform_expert_abi.log:128`). The decode log ends with a failing
assertion listing the state and both pool families (`:2788`); the preceding
`[RUN] PASS ... validation skipped` line is **not** numerical acceptance.
Prefill has 17 explicit per-output PASS lines (`:2321-2337`). These are inspected
historical device results, not newly rerun gates.

S has the DSpark ABI (4096 hidden width, CSA/HCA state pools) and generated
`_decode_fwd` orchestration. It supplies lowering evidence only; this unit does
not assert its device result or prove it was built from the exact current files.
No A5 full-forward orchestration with `ext_state_cache_pool` was found among
2,418 existing orchestration C++ files under root `build_output`; neither
reference tree has its own `build_output` directory. A5 conclusions are source
observations, not an A5 generated/runtime comparison.

## A5-first comparison

| Concern | Exact evidence | Observation and implication |
| --- | --- | --- |
| Physical state ownership | A5 `compressor.py:c2a_compressor_publish:182-211`; `kernels/compressor_pool.cpp:run:228-270,287-306` | A5 launches request-bounded workers, assigns `owner = block % workers`, and uses two phases: historical reads before persistent writes. The same physical block has one owner. A3 `compressor.py:c2a_compressor:106-160` pools token workers, then launches a separate single state writer after `pool_tid`. This is a real decomposition difference, not proof that either schedule is the cause. |
| State update semantics | A5 `compressor.py:golden_c2a_compressor:249-274`; C++ `:271,301-305`; A3 `compressor.py:146-160` | A5 retains the trailing capacity rows (`STATE_CAPACITY=4`, A5 `config.py:240`); A3 capacity is 2 (`config.py:190`) and commits only the last token of each request. This is a semantic/state-history difference to audit against the agreed A3 contract before any alignment. Do not copy the A5 capacity or C++ implementation into A3. For the current decode fixture, each request has one token, so A3's predicate should select both writes. |
| State pool view | A5 `decode_fwd.py:decode_encoder:229-232,287-293`; A3 `decode_fwd.py:1031-1039,439-469` | A5 flattens the source axis into physical blocks and slices an interval on the first axis. A3 keeps an explicit source axis, selects `[c2a_slot]`, drops that axis, then reshapes the state into rows. The interval/view lifetime differs, even though both are zero-copy intent. |
| C2A pool equivalence | A5 `decode_c2a_full.py:130-145`; `decode_indexer.py:decode_c2a_indexer:782-832` | A5 C2A publishes index keys and Top-K, and joins compressed publication with selection. It has **no C2A candidate-pool arguments/publication** at this boundary. A3 `decode_c2a_full.py:157-166` publishes an additional pool and joins its completion. An A5 C2A success would therefore not validate A3's failing C2A pool. |
| Closest pool publisher | A5 `decode_indexer.py:select_candidate_block_pool:231-349`; `_publish_candidate_counts:211-227`; A3 `indexer.py:227-300` | Both contain token-worker ID stores and a count-publish loop following selection. A5 has an all-visible fast branch, active-token clamping, capacity from the output tensor, and a separate block-score stage for long histories; A3 uses caller-provided rows/width and a different selection arena. A3's count-publisher shape is not intrinsically alien to A5. Earlier store-shape negatives prohibit calling spmd conversion a fix. |
| Pool ownership and actual consumers | A5 `decode_fwd.py:505-512,573,602`; `decode_c1a_full.py:149-152`; `decode_c1a_reindex.py:108-117`; A3 `decode_c1a_reindex.py:114-134` | A5 C1A creates a local pool and later reindex reads that pool. A3 exposes a per-index-slot external pool stack and reindex recomputes a full-history pool; its pool values at this boundary are diagnostics, while completion is used as an ordering dependency. A5 local scratch with a later data consumer is not equivalent to A3 external diagnostic output. This study records the semantic difference without changing accepted contracts or weakening gates. |
| Prefill contrast in reference | A5 `prefill_c2a_full.py:160-166`; `prefill_indexer.py:484-538,1306-1313` | A5 prefill shares `c2a_compressor_publish` with decode, while its prefill indexer publishes a pool only for non-pool, non-C2A selection. It does not provide a reference for A3's C2A pool acceptance. |

## DSpark-second comparison

| Concern | Exact evidence | Observation and implication |
| --- | --- | --- |
| Request ownership and slot map | DS `decode_compressor_ratio4.py:compressor_ratio4_cache_write:294-323` | Each worker owns request indices, loops their verification tokens, reads an explicit `state_slot_mapping`, and writes projection/score rows into flattened state. Its ratio 4, positional additive score and state widths differ from V4.1 C2A; reuse of its numerical implementation is unjustified. The useful reference is ownership and explicit target mapping. |
| Read-before-write dependencies | DS `decode_csa.py:504-525`; S `orchestration/_decode_fwd.cpp:6230-6273` | State commit depends on both pooling and the post-leaf fence. Generated task 187/function 116 binds the flattened state with `add_inout`, both explicit dependencies, and `block_num=min(KV_B_DYN,16)`. Thus external sliced/reshaped state plus `add_inout` and dynamic request loops can be emitted in another composed A3-family program. This is not device proof for A3's particular view/version. |
| Pooled state view | DS `decode_fwd.py:493-497,704-728`; S `:6247-6250` | DS also slices a flattened physical-block interval per layer and reshapes to state rows, more like A5's pool than A3's separate source axis. It does not eliminate a source-axis view/version hypothesis. |
| Prefill ownership contrast | DS `prefill_compressor_ratio4.py:355-395` | Prefill publishes the state tail after pooling with `spmd(STATE_LEN)`, then a dependent state-order fence. Decode instead commits per request with a slot map. This is a phase-specific decomposition example, not evidence that A3 must use different phase kernels. |
| No equivalent candidate pool | DS `decode_indexer.py:29-64,1672-1679`; absence search of DS `decode_indexer.py` and `prefill_indexer.py` in `inventory.json` | DS exposes Top-K scores/IDs and inner compressor state; neither indexer file contains `pool_block_ids`, `pool_block_counts`, or `select_candidate_block_pool`. The reference can inform state/Top-K scheduling but cannot directly arbitrate V4.1 candidate-pool stores. |

## Generated A3 evidence and unresolved differences

| Question | Exact evidence | Supported conclusion |
| --- | --- | --- |
| Were writers removed? | D `orchestration/decode_fwd_rank.cpp:2478-2515,2803-2857`; P `orchestration/l2_prefill_fwd.cpp:2788-2801,3173-3224` | Both programs contain pool selection, a count publisher depending on selection, and a state writer depending on pair pooling. All target the intended `add_inout` views. This excludes absence from generated orchestration, **not** absence from the runtime graph or device execution. |
| Does adding a downstream dependency explain the difference? | A3 `decode_c2a_full.py:166` versus `prefill_c2a_full.py:168`; D `:2859-2870` versus P `:3226-3237` | Decode's gather fence includes the pool count completion; prefill's fence omits it. Prefill nevertheless exposes correct pool outputs. Host-boundary-only use or a missing downstream pool TaskId cannot alone explain the decode failure. Extra ordering is not proof of correctness or wrongness of the runtime graph. |
| Views and pointers | D `:1835-1854,1900-1948,2478`; P `:861-880`; D `kernels/aiv/c2a_state_write.cpp:317-338` | Rank-local external state is source-sliced and rank-dropped, then reshaped; the kernel wrapper uses `buffer.addr + start_offset`. The runtime header `../pypto/runtime/src/common/tensormap_and_ringbuffer/tensor.h:210-226,364-390` copies buffer/owner/version/offset and implements zero-copy reshape. No copy-out step was identified. Runtime metadata values, offsets and version transitions are still unobserved. |
| Do generated kernels contain the stores? | D `ptoas/c2a_state_write.cpp:110-165,211-220,267-309`; `ptoas/index_pool_counts.cpp:28-101` | The state loop and request-boundary predicate remain, targeting `block*2 + position%2`. The counts kernel loops its runtime row scalar, stores to the output pointer, and includes a DCCI/DSB tail. Code presence does not prove loop trip counts, addresses, execution, or payload visibility. |
| Is prefill a controlled state contrast? | A3 `decode_fwd.py:1606-1612,1811-1812`; `prefill_fwd.py:1734-1742,1929-1931`; A3 `utils.py:build_request_plan:227-265` | Decode T=2 is requests `(1250,1),(1253,1)`, physical blocks 0/1 and flattened slots 0/3. Prefill T=2 is one request `(1248,2)`, one block; A3 commits only its last token (odd slot). Thus prefill green does **not** exercise the missing even-position request-tail write. Geometry, request count, boundary predicate and parity change together; the two-block explanation is a candidate, not an isolated cause. |
| Is prefill a controlled pool contrast? | A3 `decode_fwd.py:1820-1828`; `prefill_fwd.py:1936-1944`; corresponding D/P view code | Decode combines 3 C2A + 5 C1A slots in one external stack; prefill has two separate stacks. Their input positions/lengths and surrounding orchestration also differ. Identical child kernels and `add_inout` are useful contrasts but do not isolate output-view lifetime or all runtime scalars. |

None of D/P/S contains `deps.json`, `name_map*.json`,
`dispatch_program.json`, or `chip_swimlane_records.json` (`inventory.json`).
Accordingly there is no new runtime reachability proof, task-retirement proof,
or scheduler attribution. The historical issue text's assertion that the
publisher "runs" is stronger than the generated evidence available here.
Keep the upstream suspicion visible, but do not promote it to a proven owner.

## Already tested versus still open

Historical tests below are recorded in `TASKS.md:3970-4057` and
`KNOWN_PYPTO_ISSUES.md:374-461`; they were not rerun in this unit.

| Historical candidate | Recorded outcome / limit |
| --- | --- |
| Writer absent from generated program | Submission sites exist; runtime existence/execution remains unproved. |
| Tensor versus Out declaration; scalar call arity | Changing declarations did not help; thin/composed scalar lists were checked. This does not prove actual device scalar values. |
| Read-anchor liveness | In-graph anchors did not change either gate. |
| Non-spmd versus spmd; scalar versus tile stores | spmd(1), per-token tile state stores and count publication moved into selection all left the failures unchanged. A universal non-spmd or loop-store-collapse explanation is not supported. |
| Guard evaluation or modulo arithmetic | Bypassing guards and replacing modulo with loop index left the state pattern unchanged. Do not repeat these as untried fixes. |
| Copy-out or universal sliced-InOut failure | Existing code aliases memory, and Top-K lands through external views; prefill also passes. A universal explanation is contradicted, while program-specific metadata/version/lifetime remains possible. |
| TP/EP windows or encoder/decoder split alone | Prefill shares these features and passes; this weakens a universal explanation but does not eliminate interactions with request geometry or output ownership. |

The C1A pool also fails without a C2A state update at that layer. Its writer
is a different task, dtype and view from C2A state. Both being external outputs
in the same failing program is a correlation, not a demonstrated shared fault.
The state write is not explicitly ordered after pool selection: it follows
pair pooling instead. The older "both are non-spmd tasks after pool-select"
description should not be used as a causal premise.

## Proposed next bounded unit (requires observer/user routing)

1. Verify the report/excerpts and retain i/ii/iii as undecided. No external
   issue has been filed by this unit. The references justify no immediate
   wholesale port or extra barrier.
2. If a diagnostic unit is authorized, keep the 16-card decode program and
   toolchain fixed. For state, vary request-tail parity/mapping and block count
   independently (including an unused second block) with the state golden
   unchanged. For pools, retain the same fixture while isolating the combined
   external slot stack from separate phase-owned views. Keep these axes
   independent, report state and both pool families separately, and change
   neither comparators nor accepted semantics. These are proposed probes,
   not implemented variants or claims that existing runtime artifacts can
   safely be replayed with arbitrary shapes.
3. Obtain actual writer occurrence, runtime scalar, pointer/offset/owner/version
   and completion evidence before adding a dependency or declaring execution.
   Use dependency and level-4 timing passes separately per the
   dependency-liveness checklist. If runtime tasks execute with correct
   addresses and values yet stores are lost, that is stronger routing evidence;
   if tasks/scalars/views differ, fix the demonstrated boundary first.

Latest observer update read before closeout: layer task
`task_20261106_092049_8323516296` was terminated by the observer after HBM
allocation failure (not a numerical gate result). All layer acceptance remains
OPEN; no duplicate run was submitted. Decode state/pools and the standalone TP
gate remain OPEN; generation remains HELD. This report does not select i/ii/iii.
