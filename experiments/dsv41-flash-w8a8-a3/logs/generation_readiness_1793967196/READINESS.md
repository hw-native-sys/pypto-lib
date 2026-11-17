# Critical path to real-checkpoint greedy generation

Read-only preparation against frozen Engram/F07 source manifest
7c5b8df2d57ca06a49106b528c2a018d36cd1524f4df8418e59ffa64b7d24d7f.
Its independent acceptance is running separately. No model/frozen report/
manifest/worker artifact was edited. This is a proposed next scope,not a GO or
an acceptance claim. Main did not launch workers,NPU tests or full-weight loads.

The next milestone is a correct complete-entry prefill-to-multiple-decode loop.
Component fixes and complete-forward compiles do not establish that loop.
After that,run real weights with original Engram and record tokens/text/HBM;
coherent text requires reference-backed numerical evidence and stable useful
continuations,not merely an executable producing token IDs. No percentage/time
ETA follows from current evidence.

## Current blockers rechecked in delivered source

| Scope | Current finding | Consequence |
| --- | --- | --- |
| F11 RoPE binding | generate.py:build_prefill_inputs/build_decode_state only add rope_cos/sin; full signatures also require rope_cos_swa/sin_swa. _rope_args_pair passes resolver mode per-profile to layer_rope_tables,which selects base unless exactly compressed. | Missing required dispatch arguments; main compressed argument contains base table. Mandatory before real forward. |
| F11 equations | precompute_rope_tables scales both profiles with YaRN; pinned original attention uses unscaled SWA and scaled compressed profile. | Bind two independently computed profiles and test nonzero positions,not identity-angle smoke. Mandatory semantic fix. |
| F12 rows | build_entry_contracts(tokens) sizes decode with prefill tokens; cache builder sets all counts=tokens; device_run fills every row with the same current ID and scalar position. | Duplicate-token tile is not single-request one-token decode. Use decode capacity1/all-active for first path or real one-active masked capacity with proven F09 support. |
| F12 lifecycle | Decode state/metadata built once; loop updates only IDs/positions and Python _MOE_ROWS after artifact capture. | WINDOW slots/indices,C1/C2 emission slots/lens,query and compressed RoPE stay stale. Mandatory per-step metadata update while retaining actual caches/compressor state. |
| F13 history | load_engram_hash_ids truncates to requested prompt rows; decode slices at position>=prompt length. Lookup is only built once before loop. | First lookup can be empty; arbitrary future greedy tokens have no corresponding hashes. Retaining more supplied rows alone cannot solve autoregression. |
| F09 active rows | prefill passes rows=tokens; decode encoder/decoder call moe_ffn rows=_MOE_ROWS; moe_ep16 selects static tokens=rows. Many seed/norm/snapshot loops use tensor extent. | Runtime count/unequal/idle semantics not proven; _MOE_ROWS assignment after compile is ineffective. Full supported dynamic contract remains mandatory acceptance separately. |
| Full-entry pool/state | Recent tests are phase/prolog/two complete staged layers and full-entry compiles. Old composed pool/state symptoms have no current complete-entry rerun after the intervening fixes. | Current complete-model numerical behavior is UNKNOWN; do not call old symptoms confirmed current defects or declare them fixed. Must rerun owning full-entry numerical gates and diagnose any actual failing boundary. |

Current Engram rotation loading/binding itself is now present: WeightLoader
returns shared FP32 block; top-level/rank materialization and device_run bind
[16,32,32] for prefill/decode. F05/F06 fix gate basis and location from gathered
rows; they do not supply token hashing. Expert arguments now have real layer
axes; obsolete single-shard gap comments are not evidence of a live failure.
Full all-rank actual-weight assembly and complete ordered-argument binding remain
unvalidated with real materializations. Never use shared_expert_shard as real
model proof. Reject missing arguments before allocating full weights.

CPU task task_20261106_201558_134096212488 exited0 in this directory. probe.py
verifies frozen45-source hashes,executes current resolver/_rope_args_pair and
hash loader,and actual request planner for positions126–129. At4096 main equals
base,does not equal compressed; supplied20 hash rows retain16 and first decode
has zero rows. probe_results.json preserves all slot/lens transitions. It is a
host reproduction of current defects,not a device proof or acceptance rerun.

## Mandatory first-path scope versus supported dynamics

For the cheapest first real single-request stream,use an explicitly compiled
one-token decode capacity,one active token per replica group,with true TP/EP
collectives and the real40-layer model. A fixed prefill tile must contain only
actual prompt tokens: current fit_prompt_to_tile changes short prompts by
appending real tokens; record the final prompt IDs/text or add correct masking.
Do not silently claim arbitrary prompts work. Current generate CLI only permits
tiles2/16/128,so decode1 needs separate contract/spec construction and real
compile/numerical proof; it is not already supported by the generator.

Static active==capacity1 avoids the specific inactive-MoE failure; it does not
close decision9 (same artifact varying active counts/context,unequal/idle DP,
multi-request isolation). Keep those gates OPEN while proving the declared
bounded first path. In particular,do not silently set token_counts=1 within a
capacity16 artifact and assume the remaining rows are safe. Boundary page
budgets must accommodate the entire run; addresses and lengths are runtime
metadata,not a reason to recompile on every token.

Handoff must distinguish persistent WINDOW/compressed/index/compressor state
from per-query topk/pool/x/pre buffers. Existing handoff concatenates family
query buffers and requires exact token-axis shapes; prefillT->decode1 cannot
blindly copy those tensors. Establish which source-layer selections are rebuilt
for the current query and initialize query scratch appropriately. Preserve real
returned caches,state and absolute source coordinates. Audit execution outputs
and repeated-dispatch InOut ownership so a host tensor/checksum cannot stand in
for actual device state. A one-step isolated golden is insufficient.

## Original Engram hashing contract and unknowns

Pinned rc2 model.prepare_engram/hash_state and ops/triton/engram_hash.py specify
request/chunk lookup,current chunk then lookback history then optional physical
slot cache;dead tokens and inactive requests do not update history. Hash kernel
uses token_map,multipliers,primes,offsets and pad/dead semantics; raw tokenizer
IDs are not automatically the compressed token IDs. Ascend hash_state imports
EngramLayout/NgramHashState from upstream vllm.models.deepseek_v41.common.engram.
That imported constructor/token-compression definition must be pinned/obtained
and matched to checkpoint resources before inventing a host implementation.
This readiness did not fetch new source or establish those exact constants.

Implement host history for actual sampled token stream per request,including
prefill chunks and each generated input token. Distinguish logical history index
from absolute position and physical cache slots. Verify table-head offsets,
integer overflow/modulo,layer seed/multiplier and token compression against the
original; test BOS/pad/dead tokens,chunking,request separation and inactive
history. Gather the actual checkpoint table rows using existing row-dequant
accessor; never load the full Engram table to device. Gate/cast/basis remains
the frozen F05 implementation. A separate host hashing module has a genuine
resource/history responsibility and consumer; it must not clone compute kernels.

zero-diagnostic explicitly makes Engram lookup zero and removes its residual.
It can only test allocation/dispatch plumbing. It cannot satisfy the requested
original real-Engram generation goal,even if the emitted text happens to read
well. Prompt-only external host IDs also cannot validate generated history.

## Proposed exclusive ownership after explicit post-review GO

No worker is dispatched now. Exact initial file partition avoids overlapping
writers; expand it only through recorded transfer before touching more files.

| Owner | Exclusive writable model scope | Responsibility |
| --- | --- | --- |
| Main | generate.py,utils.py,weight_loader.py,weight_map.py,structural_check.py | F11/F12 real binding,dual-profile equations,decode1 contract,handoff/per-step allocator metadata,host hash integration,memory accounting and real run driver; shared docs/ledger/integration. Preserve F14 and rotation loader. |
| Worker A | prefill_fwd.py,decode_fwd.py,moe_ffn.py,moe_exchange.py,layer_reference.py | F09 active-count propagation and complete-entry owning defaults/comparators; actual state/pool publication diagnosis after reproduction. F10 capacity layout protected. Main owns generation spec consumer,requests worker API edits. |
| Worker B | new engram_hash.py plus own artifacts | Original-source token compression/hash/history loader/state,host CPU oracle and meaningful mutants; propose needed resource metadata to Main. No generate/tokenizer/weight_loader edits. Exact upstream pin/resources first. |

prefill_layer.py/decode_layer.py and phase/operator files remain frozen initially;
if dynamic propagation requires them,record transfer to Worker A before edits.
engram.py is not a hashing home and stays frozen. Worker B emits a documented
API consumed by Main; no concurrent edits to generate.py. No filler workers;
artifact-only original-reference validation could run independently if useful.
Independent reviewer owns final acceptance and reports to observer,not Main.

## Cheapest validation sequence and concrete exit criteria

1. After current review disposition,queue CPU source-backed dual-RoPE controls
   and complete ordered ABI-name/shape/dtype binding using meta/tiny stubs.
   Test missing-SWA/profile-swap/incorrect-SWA-YaRN mutants. Prove independent
   decode1 specs and handoff scratch shapes without full weights. Bound memory
   from safetensor headers plus rank residency,replication,NZ/scales,caches,
   source/pool state,EP windows,logits/snapshots and compile/transient overlap.
   Current peakHBM is UNKNOWN; loading just the rotation map says nothing about
   real model fit. Remove diagnostic snapshots from real entry only with an
   explicit validated interface change; do not hide memory failure.
2. Queue original hash/compression/history CPU tests,fullchunk==chunked==stepwise
   with original constants and nonzero table rows. Reject raw-ID,wrong seed,
   missing generated token/cross-request contamination. These parallelize with
   Main/F09 work. Table header/row samples only;no full-weight materialization.
3. Compile actual complete prefill and decode1,then execute checkpoint-free
   original/analytic-backed complete-entry numerical gates on16devices. Check
   every layer boundary as needed,final hidden/logits,routing and all published
   caches/index/state/topk/pool. Reproduce actual failure before diagnosing a
   dependency bug; preserve old publication failures if observed. Phase or
   teacher-forced FFN success alone cannot close complete-entry state gates.
4. Carry returned actual prefill state through at least3decode steps on ONE
   decode artifact crossing C2 even/odd and WINDOW page127->128. Use a genuine
   prefix or independently prepared valid long-prefix state; never jump position
   on an empty cache. Assert per-step slots/lens/query&sourceRoPE/retained history,
   exact untouched cache regions and no stale overwritten row. Compare sequential
   logits/state with independently recomputed token-prefix reference. Use stale
   metadata and repeated-tile numerical negative controls. All real persistent
   outputs must feed the next step,not golden cache substitution.
5. In parallel/after F09 propagation,prove capacity>active/unequal/idle requests
   and same-artifact changed counts/page tables with inactive-state sentinels.
   This closes supported-dynamic gates separately;do not equate bounded decode1
   proof with full lifecycle acceptance. Full40-layer composition remains the
   shared mandatory boundary regardless of the path chosen.
6. Only after bounded numerical/reference and resource/binding gates pass,run
   actual rank-specific checkpoint weights with original host hashing,real
   tokenizer and short greedy stream (EOS/max token bound),16auto devices.
   Record pins,prompt template/final promptIDs,generatedIDs/text,per-step
   positions/metadata/state,every DP agreement/top logits,allocation and sampled
   peakHBM. Repeat the same prompt for determinism;test additional representative
   prompts without selecting only fluent examples. Never use zero-Engram/shared
   expert shard diagnostics as accepted real output. An original reference
   logits/token-prefix comparison must account explicitly for remaining
   primitive/reduction/quant-tie limits;if unstable greedy ties occur,report them.
7. Freeze source/evidence and delegate independent acceptance. Accepted coherent
   text requires the real model path,correct history/cache progression and
   meaningful reference alignment;readability alone is not a numerical gate.

Original primitive/tie parity,offset incidence,TP/CI and supported lifecycle
requirements remain explicit. No checkpoint/PyPTO writes,commits,filing,simulator
acceptance or relaxed thresholds. Resume from this report plus newest TASKS;
Main PARKED pending independent disposition and a bounded implementation GO.
