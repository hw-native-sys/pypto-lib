# Interface contract: scope revision 2 (A5-shaped dynamic batch/S + paged caches)

## Current longer-N3 admission application (2026-11-07)

This newest overlay supersedes historical current labels below. Exact reviewed
41linec825 patch is APPLIED only togenerate.py, actualdigest
41651e5af3b94223b645d4bfb52010e52ea7350aa3f3637cbcca6e29e093e609.
NEW source45`cc15877f8f4b0104a842f915d2434d9a2c9cd6f6752d6364d5d7bada26e21d8c`;
other44 unchanged fromD77. [Application report](logs/longer_generation_admission_apply_1794062947/REPORT.md)
and [nextcommands/provenance](logs/longer_generation_admission_apply_1794062947/NEXT_COMMANDS.md).
ONE small appliedCPU check passes. No NPU/payload/grant/signaturecompile occurred
in this source-application unit. Default no-receipt HARD refusal remains.

No tensor/scalar ABI changed. Existing normal CLI authorization flag remains;
NEWoperation`bounded-full-real40-noEngram-greedy-N3` rejects oldfirst-operation
receipts. Newreceipt includes exactnonempty `prompt_text`, exactlythree trueinteger
`prompt_ids` inrange, and exacttrueinteger max_new_tokens2..16. CLI text/cap and
actualtokenizer IDs must match. This does not admit generalN2..128, chat/files/
multi-request/Engram/startoffset changes. Firstcase Onceupona/max16 is still future.
External same-root/tokenizer prequeue/allocated/post hash/codebinding remains;
future outer assignment must bind NEWCCsource and fields. Old48/34/13 immutable.
All57static/sourcecaps and existing metadata provenance3dd67 are unchanged.

## Current applied admission milestone (2026-11-07)

This overlay supersedes historical current/pending source labels below. Exact
independently reviewed generate-only patch522dae is now APPLIED; generate digest
d057012464331b10b0bdc7f406925ee0c552dd298a1db171a93b0b1595769598.
NEW actual source45 SHA256 `d77d196be6541b4a87aef07cf0e5e02eac68cefa9fc245f3742e4c14606555aa`; exactly44 other model files unchanged
from predecessor762dc. Recovery: [application report](logs/generation_admission_apply_1794052225/REPORT.md)
and [next commands](logs/generation_admission_apply_1794052225/NEXT_COMMANDS.md).
Small applied-source CPU checks pass. No compilation, NPU, real tensor payload,
full materialization/forward or text ran in this application unit. No positive
Root operation receipt was created. Default HARD refusal remains unless a NEW
separate Root-admitted operation satisfies its normal explicit receipt path.

No tensor/scalar ABI changed. The normal CLI adds optional
`--first-real-generation-authorization`; no receipt means unconditional budget
refusal, even with `--allow-full-materialize`. The finite intended first operation
is TP4DP4EP16/W8A8/real40/noEngram, exactN3 plain `Hello, world` IDs19923/14/2058,
max4total truthfulEOS greedy samples and strictall4DP agreement everyphase.
The new receipt binds the actual applied source45; reviewed header-plan3dd67
retains honest predecessor762dc provenance. Product does not pin model_root/
tokenizer files: Root accepted independent external same-root and exacthash
checks prequeue/allocatedprerun/postrun as a mandatory operation condition.
Raw final-memory and one complete coverage/read/copy summary perphase are durable.
Compilation/default CPU smoke cannot authorize data loading or model execution.

Status update (2026-11-06): goal.md decision9 remains the scope authority;
public interfaces are tracked here, while current semantic repair/acceptance
status is in TASKS and ACCEPTANCE.md. F01/F02/F03/F08 and conditional F14 have
bounded acceptance; F04/F10 are active. The original interface migration used
the A5 reference signatures as an alignment target, but subsequent independent
review found kernel/golden semantic defects against the original A3 reference.
Pre-migration or pre-repair numerical PASSes cannot therefore be presumed
semantics-valid for the repaired tree. No full-model/current-interface numerical
acceptance follows from compilation or historical component results.

## 1. Divergences from A5 (permitted, recorded)

| Area | A5 reference | A3 implementation | Basis |
| --- | --- | --- | --- |
| Weight/activation quant | FP8/MX weights, FP8 caches, MX scales | per-channel INT8 + F32 `weight_scale`/`weight_offset` + dynamic per-token INT8; BF16 activation caches | A3 quantization divergence (decision 9) |
| Parallelism | TP1 attention slices, MX_B_NN | DP4 x TP4 attention, EP16 experts, `tp_output_reduce` | A3 parallelism divergence (decision 9) |
| Cache dtypes | FP8 packed caches + scale tensors | BF16 raw caches, no scale tensors | follows from the quantization divergence |
| Indexer TP sharding | replicated 32-head index query | **replicated 32-head index query** (was TP4-sharded with `tp_score_reduce`) | exception: removes the dynamic-width cross-rank score exchange and the folded score counter that caused the prefill lane-poisoning protocol; index weights are ~5.2 MB/rank; attention TP sharding and the output TP reduce are unchanged |
| Staged validation hooks | layer-level `attn_input`/`hidden` Out ports | same: layer-level `attn_input` hook kept; chain-internal scratch outputs dropped | decision-9 recommended resolution |

## 2. Dynamic symbols (config.py)

```
T_DYN            token rows in the local batch (prefill: sum of query lengths;
                 decode: active tokens), runtime `num_tokens` scalar mirrors it
B_DYN            requests in the local DP batch
TABLE_DYN        logical 128-row pages per request table
ORI_BLOCKS_DYN   physical pages of the window cache (per layer)
CMP_BLOCKS_DYN   physical pages of the compressed cache (per KV source)
INDEX_BLOCKS_DYN physical pages of the index cache (per KV source)
STATE_BLOCKS_DYN physical per-request compressor-state blocks
```
DynVars appear only in annotations. Bodies extract `pl.tensor.dim`; every
runtime-count loop uses `num_tokens`/dimension locals. `pl.dynamic` is imported
from `config` by all modules (single definition).

Fixed execution constants: `BLOCK_SIZE=128`, `WINDOW_KEYS=128`,
`INDEX_TOPK=512`, `SELECTED_KEYS=WINDOW_KEYS+INDEX_TOPK=640` (SWA: 128),
`STATE_CAPACITY=2` (C2A carry rows), `PREFILL_MAX_TOKENS=8192`,
`DECODE_MAX_TOKENS=32`, `MAX_BATCH_PER_DP=32`.

## 3. Cache/state layout

| Tensor | Shape | Dtype | Owner |
| --- | --- | --- | --- |
| `window_cache` | `[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM]` | BF16 | one per text layer (40) |
| `compressed_cache` | `[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM]` | BF16 | one per KV source (C2A 2/8/14, C1A 20) |
| `index_cache` | `[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM]` | BF16 | one per KV source (same 4) |
| `state_cache` | `[STATE_BLOCKS_DYN, STATE_CAPACITY, 2*HEAD_DIM]` | FP32 | one per C2A source (3); slot `position % 2`, KV half + score half |
| `window_slots` | `[T_DYN]` | INT64 | write slot of this token's KV row |
| `window_indices` | `[T_DYN, 128]` | INT32 | physical row of each visible window lane, oldest->newest, -1 invalid |
| `compressed_slots` | `[T_DYN]` | INT64 | write slot of the compressed row emitted by this token, -1 when the token does not emit (`position % ratio != ratio-1`) |
| `topk_indices` | `[T_DYN, 512]` | INT32 | **physical** compressed-cache rows of the selected Top-K, -1 masked |
| `pool_block_ids` / `pool_block_counts` | `[T_DYN, POOL_DYN]` / `[T_DYN]` | INT32 | logical blocks of 8 index rows and valid prefix count; C1A full publishes the source pool, C1A reindex consumes it unchanged |
| `index_block_table` | `[B_DYN, TABLE_DYN]` | INT32 | logical page -> physical page for window/compressed/index caches; identical mapping shared by the three caches of a request |
| `state_block_table` | `[B_DYN, 1]` | INT32 | per-request compressor-state block |
| `request_ids` | `[T_DYN]` | INT32 | token -> request |
| `position_ids` | `[T_DYN]` | INT32 | absolute position |
| `compressed_lens` | `[T_DYN]` | INT32 | visible compressed rows = `(position+1)//ratio` |
| `rope_cos/sin` | `[T_DYN, 32]` | FP32 | per-token rotary tables at the token position |
| `compressed_rope_cos/sin` | `[T_DYN, 32]` | FP32 | rotary tables at the emitted compressed position (rows without an emission are zero) |

C1A reindex reads the layer-20 pool at decode selection slot `C1A_SLOT_BASE`
(3) or prefill C1A slot 0. Per-reindex Top-K slots remain distinct. The four
former reindex pool diagnostic slots retain their ABI positions and remain
unchanged; bounded fixtures initialize them to `-1` IDs and zero counts. The
source pool remains an acceptance output. See the 2026-11-06 reindex repair
checkpoint in TASKS.md for device evidence and the full-model gates still open.

Window logical page = `position // 128`; compressed logical page =
`(position // ratio) // 128`. All three caches use the same physical page id
for a request, so one `index_block_table` addresses them. Slots are
`physical_page*128 + offset`.

## 4. Host metadata contract (utils.py)

Pure-torch helpers, shared by fixtures, fwd drivers and goldens:

- `paged_slots(positions, request_ids, block_table, logical_divisor, publish_complete)`
- `window_metadata(positions, request_ids, block_table, window)` -> `(window_slots, window_indices)`
- `compressed_metadata(positions, request_ids, block_table, ratio)` -> `(compressed_slots, compressed_lens)`
- `build_request_plan(requests, block_table_capacity, ratio)` -> dict with
  `position_ids`, `request_ids`, `index_block_table`, `state_block_table`,
  `window_slots`, `window_indices`, `compressed_slots`, `compressed_lens`.
  `requests` is a list of `(start_position, query_length)`; the plan assigns
  physical pages per request from a caller-provided allocator and returns the
  used page count. `block_table` rows are padded with -1 to `TABLE_DYN`.
- `gather_cache_rows(cache, slots)` / `scatter_cache_rows(cache, slots, rows)`
  for goldens and generation bookkeeping (torch, flat-row semantics).

Host drivers size each cache to the actual plan (`pages_used` per source/layer)
so one compiled artifact accepts any batch up to the compile-time bounds.

## 5. Shared kernel contracts (new `paged_cache.py`; migrated shared operators)

- `publish_window_row(kv, window_slots, window_cache, num_tokens)`:
  `flat = reshape(cache, [blocks*128, HEAD_DIM])`; write `kv[t]` at
  `window_slots[t]` when `>= 0`.
- `publish_compressed_row(latent, compressed_slots, compressed_cache, num_tokens)`
  and `publish_index_row(index_key, compressed_slots, index_cache, num_tokens)`:
  same flat-row store; rows with negative slots are skipped.
- `gather_selected_keys(window_cache, window_indices, compressed_cache,
  topk_indices, selected, combined, num_tokens)`:
  `selected [T_DYN, 640, HEAD_DIM]` BF16 and `combined [T_DYN, 640]` INT32;
  lanes `0..127` read `window_indices`, lanes `128..639` read `topk_indices`;
  negative ids yield zero rows and `combined=-1`. `pl.tile.gather_row` over the
  flattened caches, per-token work distributed with `pl.spmd`.
- `gather_window_keys(window_cache, window_indices, selected, combined, num_tokens)`
  for SWA (`selected [T_DYN, 128, HEAD_DIM]`).
- `attention.py`: per-token packed core
  `packed_attention_core(query [T_DYN, LOCAL_HEADS*HEAD_DIM] BF16,
  selected [T_DYN, KEYS, HEAD_DIM] BF16, combined [T_DYN, KEYS] INT32,
  sink [LOCAL_HEADS] FP32, output [T_DYN, LOCAL_HEADS*HEAD_DIM] BF16,
  keys: constexpr)`.
  One token's heads form the matmul rows (16 local heads), keys are that
  token's packed set, online softmax with the sink, `combined < 0` masked.
  The previous shared-key core and its callers are removed in the same change.
- `indexer.py`: paged scoring. Query projection stays local
  (`LOCAL_HEADS*IDX_DIM` per rank, remote rows zero) but the 32 index heads are
  **replicated per rank** (all-head `wq_b` loaded per rank); scores are complete
  locally and no `tp_score_reduce` is used. Scoring walks logical pages
  `0..ceil(compressed_len/128)-1`, gathers each physical page
  (`index_block_table[request, page]`) from `index_cache`, and emits
  `topk_indices` as physical cache rows. `index_key` projects/normalizes/rotates
  the emitted latent rows and `publish_index_row` stores them at
  `compressed_slots`.
- `compressor.py`: C1A projects/normalizes every token. C2A pools pairs and
  carries the per-request state at `state_cache[state_block_table[request], position % 2]`;
  the emitted rows publish to `compressed_cache[source]` at `compressed_slots`.

## 6. Chain signatures (A5-shaped, layer-level)

Every mode chain is `@pl.jit.inline` and takes, in this order:

1. HC/attention prolog: `x_hc [T_DYN,HC_MULT,D] FP32`, `pre_mix [T_DYN,HC_MULT] FP32`,
   `hc_attn_fn [MIX_PAD=32,HC_DIM]` (hc_mixes reads 32 padded rows; a 24-row
   annotation fails), `hc_attn_scale [3]`, `hc_attn_base [MIX_HC=24]`,
   `attn_norm_weight [D] BF16`.
2. Attention weights in A3 quantization form (typed, see the owning file).
3. `rope_cos/rope_sin [T_DYN,32]`.
4. Caches/metadata blocks from §3 (only the members the mode consumes; selection
   modes add `index_cache`, `index_block_table`, `request_ids`,
   `compressed_lens`, `compressed_slots`, `compressed_rope_cos/sin`, index
   weights; SWA omits the compressed/index blocks; reuse modes consume
   `topk_indices` but not the index weights).
5. Compressor weights (C2A modes) or C1A compressor weights.
6. Port outputs: `hidden [T_DYN,D] BF16` (hc_pre result, staged),
   `attn_input [T_DYN,D] BF16` (staged Out hook), `attn_out [T_DYN,D] BF16`,
   `output [T_DYN,HC_MULT,D] FP32`, `next_pre_mix [T_DYN,HC_MULT] FP32`.
7. Distributed output window/signal, `group_base`, `tp_rank`,
   `num_tokens: pl.Scalar[pl.INT32]`, `attention_epoch`.

No scratch buffers are parameters. `engram` inputs are added for layers 1/14.
The chain runs HC pre, norm, projection/compressor/index publication, per-token
chunked gather+attention (chunk 16 tokens), `o_proj` + TP output reduce, HC post.

### Engram rotation resource and layer-input placement (2026-11-06 implementation)

Layer and full-forward entries additionally bind `engram_rotation` FP32
`[32,32]` per rank (`[EP_SIZE,32,32]` on distributed host entries). This is one
global resource shared by Engram layers 1/14, without a layer axis. The loader
reads `optional/quarot.safetensors:global_rotation`, verifies identical diagonal
32-channel blocks and zero off-block entries, and retains the stored block R.
Declared rotated checkpoints require the map; there is no silent identity
fallback. Synthetic layer/forward defaults provide an explicit nonidentity R.

Before HC pre and attention, Engram projects lookup rows into BF16 keys/value,
restores only gate hidden in FP32 as `h_rot @ R.T`, computes original-basis
q*k/key gating, and updates the FP32 rotated HC stream with already-rotated V.
The residual is not narrowed to BF16 by this FP32 gate ABI; no second V rotation
is applied. Incoming `pre_mix` retains its meaning. Attention/compressor/cache
publication consumes the updated stream. The standalone attention phase ABI
does not acquire this layer-owned resource. Generator binding is updated for
the resource only; F11–F13 profile/dispatch/history/lifecycle gates remain open.

Candidate-pool storage now carries the model cap 2048 through existing dynamic
pool axes; bounded fixtures do not redefine the cap as64. These implemented
changes require independent cohort acceptance; full-model runtime and F09
lifecycle guarantees do not follow from the interface declaration.

## 7. Layer and forward contract

- `prefill_layer` / `decode_layer`: keep the mode dispatch and sublayer
  sequencing, but call the migrated chains with `x_hc`/`pre_mix` and the paged
  blocks; expose only the port outputs above (plus the staged hooks). Layer
  functions are typed with `T_DYN` and take `num_tokens` as a runtime scalar;
  the token-count compile-time specialization is removed.
- `prefill_fwd` / `decode_fwd`: one compiled artifact per phase with `T_DYN`;
  no per-shape recompilation. The host driver builds the request plan (§4),
  allocates one window cache per layer and one compressed/index/state cache per
  source, and passes the runtime plan tensors. Window/signal buffer capacities
  come from `PREFILL_MAX_TOKENS`/`DECODE_MAX_TOKENS`; kernels use runtime counts.
  Multi-request batches are first-class.
- MoE exchange (`moe_exchange.moe_ep16`): replace `tokens`/`recv_max` constexprs
  with runtime counts; receive-window capacity is `N_LOCAL_EXPERTS *
  EP_SIZE * max_tokens` for the compiled token bound, and kernel strides use
  runtime `num_tokens`. The worst-case receive capacity (one token per expert
  per source) is preserved.
- `structural_check.py`: accept DYN signatures (`pl.dynamic` axes) and the
  runtime `num_tokens` scalar; still prove a prefill token tile > 1 through
  projections/feed-forward and reject tokenwise/renamed controls.

## 8. Acceptance after migration

10 mode entries on device; both layer suites 7/7; structural source+compiled
gate with tokenwise/renamed controls rejected; then the complete-entry unit
(the two carried defects fixed), generation/HBM, EP16 evidence, closeout.
Prior device evidence through revision `0d6732ff` remains the semantic baseline
and is marked interface-stale.

## 9. Layer / forward migration detail (M3 contract)

Layering (resolved for implementation): each mode file owns its operator-level
attention chain and a layer-level attention wrapper that performs HC pre, RMSNorm
and HC post for the attention sublayer (template: `prefill_c1a_full`). The layer
file owns the FFN sublayer and binds the two halves:

1. Call the mode wrapper with `x_hc`/`pre_mix`; it returns the post-attention
   `x_hc_mid`, the attention `next_pre_mix` (used as the FFN's incoming pre-mix),
   the staged `attn_input`, and the selection outputs. The wrapper owns its TP
   output reduce and attention epoch.
2. FFN half in the layer: `hc_pre(x_hc_mid, attn_next_pre_mix, hc_ffn_*)` ->
   `ffn_input`, FFN `next_pre_mix`, post/residual; RMSNorm -> `ffn_normed`;
   `moe_gate` -> route ids/weights; MoE in compile-time tiles of `MOE_TILE`
   (default 128) through `moe_ep16(x_tile, ids_tile, w_tile, ..., capacity=MOE_TILE)`
   with runtime tile size derived from the tile's `x` dim; `hc_post` -> `x_hc_out`.
   MoE windows are allocated once with capacity `N_LOCAL*EP_SIZE*MOE_TILE`.
   Engram layers apply after the attention `hc_post`, before the FFN.
3. Staged hooks stay at layer level: `hidden` (attention hc_pre), `attn_input`,
   `attn_out`, `ffn_input`, `ffn_owned`, `ffn_normed`, route ids/weights,
   `x_hc_mid`, `x_hc_out`; no chain-internal scratch.

Layer signatures use `T_DYN` and runtime `num_tokens`; the compile-time token
specialization is removed. Layer suites run on 16 cards and keep the staged
reference comparisons, rebuilt on the paged plan (goldens reuse the shared
operator goldens plus the host plan/gather helpers).

Forward (`prefill_fwd`/`decode_fwd`): one artifact per phase. The host builds the
batch plan (`utils.build_request_plan`), allocates one window cache per text
layer, one compressed/index cache per KV source, one state cache per C2A source,
then runs the 40-layer schedule (hand-unrolled segments kept from the old entry,
each call on `T_DYN`), final norm and head. Model weights/state are stacked per
layer as before; `idx_wq_b` is now replicated (full 32 heads per rank). The host
reuses one compiled artifact for every token count and request mix; multiple
requests are first-class. Decode state persists in the paged caches between
calls. Generation (`generate.py`) tokenizes, builds a one-request plan, calls the
compiled prefill then decode entries greedily, and records ids/text/termination.

## Decided ABI (2026-11-05, coordinator/user decision): routed-expert binding

The single-program forward entries cannot take a materialized full
`[EP_SIZE, N_LAYERS, N_LOCAL_EXPERTS, ...]` routed-expert host array for the
bounded fixture: the zero-stride layer broadcast is rejected by the distributed
runtime (`make_tensor_arg requires a contiguous tensor`), and materializing it
would need ~543 GB host memory. Decision:

- **Routed experts: device-resident per-layer binding.** The generation loader
  (`WeightLoader.expert_shard(layer)`, already per-layer) binds each rank's
  per-layer shard into device memory; the per-rank real expert stack is
  ~34 GB, consistent with the 36.75 GiB/rank mapping estimate and HBM-feasible.
- **Bounded complete-entry fixtures** carry one contiguous per-rank routed
  shard (`[EP_SIZE, N_LOCAL_EXPERTS, ...]`) shared across layers; per-layer
  expert correctness is validated at per-layer granularity by the layer suites
  (both 7/7 device-green), not by materializing the full stack. The full-stack
  host fixture is **rejected on the 543 GB bound**.
- **Shared experts** are small per-layer dense weights (~7.5 GB host across all
  ranks) and keep the layer axis in the entry ABI.
- `prefill_fwd.py` and `decode_fwd.py` implement this form; the real
  generation path binds routed experts per layer on device.

## Evidence-mode correction and current first-generation scope (2026-11-06)

Supersedes prior current-overlay hardware attribution: native_accept_engram_
1793967192/REVIEW.md proves Main order/layer and worker basis/cap custom helpers
omitted execution_mode and defaulted SIM (generateda2a3sim). Their raw numerical
logs remain preserved but are RETRACTED as onboard evidence. Corrected reviewer
explicitONBOARD actual16device four order positives/two numerical controls and
two staged layers pass: F06 bounded accepted. F05/F07 actualL2 defaults/controls
remain OPEN after native halMemCtl EACCES;wholeEngram/F07cohortNOTaccepted.
Historical helpermode audit remains separate;no blanket math retraction.

User decision10 now permits first generation explicitly noEngram;original host
hashing/restoration scope deferred,preserved implementation not removed. Active
unit logs/generation_noengram_1793970065 repairs actual dualprofile binding and
continuous decode1 driver plus fullentry noEngram wiring/native state proof.
First noEngram output cannot claim original fullEngram parity. GeneralF09,
fullmodelnumerical/state/device/realcheckpoint/HBM/generation gates remain OPEN.
Every new device check explicitONBOARD with normalizedmode/generateda2a3/native
runtime proof;allocation/platformrequest/exit0 alone is insufficient.

## noEngram first-path ABI update (2026-11-06; independent validation pending)

Both complete forwards expose engram_enabled INT32 runtime scalar immediately
following engram_rotation. Value0 bypasses actual Engram compute;owning defaults1
preserve enabled path. Resources stay ABI-bound evenwhen disabled. Driverfirst
path explicitlylabellednoEngram/original_model_parity=False,doesnotloadreal
Engramtables/matrices. Routedexpert hostshapes are[16,EXPERT_LAYERS,24,...],
rankshapes[EXPERT_LAYERS,24,...];realweightEXPERT_LAYERS40,syntheticdefault1.

Generation has separate decodecapacity1 withcount1;itbindsunscaledSWA andYaRN
compressed queryprofiles,refreshesallposition-dependentpaging/source/querytables
eachstep,andcopiesonlypersistentKV/index/compressorstateatprefillhandoff.
Perquerytopk/pool/x/pre scratchisfresh,reusedwithinthedecodeforward. Hostchecks
pass. Native40-layer prefill2 passes withreusedsyntheticexpertlayer;decode1
preparationfailedcode13,soreturneddevice-statecontinuationisUNVALIDATED. This
firstcapacity1 contractdoesnotclosegeneralruntimeunequal/idle/multi-requestF09.

## Active publication/residency repair overlay (2026-11-07; NOT accepted)

Read native_accept_noengram_1793972955/REVIEW.md and newest TASKS first. Actual
native16decode1 completed and failed seven exported state/selection outputs;
this supersedes older recovery prose saying only native EACCES blocked decode.
Independent native40-dense-layer prefill2 passes with one reused synthetic
expert layer. No valid-prefix continuation or real checkpoint result follows.

Current unit logs/decode_publication_residency_1793979114: Main isolated rank
mutation annotations becoming HOST INPUT instead of INOUT; ten annotations
fixed with generated-binding CPU proof. Device producer math, owning goldens
and thresholds unchanged. Added same-threshold published-row checks so one
missing row cannot hide in a whole-cache ratio. Native before/after remain
queued, so repair is not device-validated or independently accepted yet.

Disjoint worker implemented supported PreparedDispatch ownership of retained
rank-sharded static weights and shared-host mutable IO (H2D/D2H each call).
CPU owner/handoff/sampler controls pass; tiny native reuse/peak proof is queued,
not passed. Caches are not claimed persistent on device. Full materialization
now rejects even with --allow-full-materialize: retained rank data, global stack
and upload staging have an unbounded host overlap. A supported streaming loader
and measured full residency/transient budget are still needed before real weights.

Corrected 128prefix->128/129/130 helper is prepared/reviewed with actual-returned
bytes, same decode artifact, retained prefix and stale-metadata published-row
control. It is gated on native decode/all-output and residency prerequisites;
NO chain has run. All16 shared devices currently occupied, owned checks pending.
GeneralF09, F05/F07/F13, checkpoint/strict-parity/TP/CI/offset/quant-tie and text
remain OPEN/HELD. Current source is noEngram first-path work, not accepted text.

### Frozen resource-blocked Main recovery (2026-11-07)

Publication/residency cohort implementation and CPU/compile controls are complete,
with no device acceptance. Both Main16-device attempts and worker2-device proof
expired/cancelled after3600s waiting BEFORE allocation; earlier600s attempts also
never allocated. Last resource snapshot16/16 occupied, all own jobs/wait clients
ended; Main and both workers PARKED. No post-fix native decode, returned-state
chain, stale-metadata device rejection, resident reuse or peakHBM is claimed.
Recovery: logs/decode_publication_residency_1793979114/REPORT.md, after.json,
artifacts.json and worker reports. Native numerical/resource gates remainOPEN;
realweights held by mandatory host-overlap guard even with opt-in flag. Resume
only after observer disposition, using fresh outputs/no duplicate queued jobs.

## Current Main interface evidence overlay 1794018541

logs/scalar_c2_streaming_1794014676 implements declared0dscalar->Pythonfloat/int/bool publicTaskArgsbinding;non0dvaluesreject. Realcompleteprefill/decodecontractsresolveannotationswithoutfixturetensorallocation (100/96args,57staticeach). PrefillC1sourceweights have no singleton sourceaxis;decode retains it;loaderphase metadata andsameleafprovidercoverboth. Streamsource yieldsWeightLeaf(argument,rank,byte_offset,tensor,rank_shape,logical_prefix,source_names);publicemptyrankalloc+serialcopy_to offsets+StackedDeviceTensor bindweights. Consumerdropsleaf/viewsbeforeadvancing;coveragegap/overlap/rank/dtype/bounds andpartialfailurecleanupcontrolsapply. MutableIO remainsstable sharedhostwithH2D/D2Heachdispatch, freshprotocolcommunicators;nodeviceresidentKVclaim. Prefillownerclosesbeforedecodeowner.

Compileusesentry.compile(config)withoutTensor samples;real40signaturecompilepassed. Fullweightsalwaysheldbyguardpendingfunctional/fitacceptance. Readerraw+bytearraytransientforlargestBF16sourceis10,590,617,600B beforecopy/IO/metadata/allocator;contiguousTPslice retainsfull5,295,308,800B backing. TinyRSS/HBMarenotrealfit. Theartifact128prefixprobe needsmorethanoldringdefault;2GiBring2publicconfigwasappliedEVERYdispatchandstillfailscontiguouswrapbehindopenscopehead35. Thisisnotanacceptedproductionresourceconfiguration;fullprefix/three-stepreturnedstate lifecycle remainsunvalidated. ThresholdsandTP4/DP4/EP16/W8A8topologyunchanged;firstpathnoEngram.

## Main dated evidence corrections 1794019935

Read-onlylogs/real_fit_readiness_1794018846/READINESS.md andfinal_checks.json supersedepriorMainheadlineswithoutrewritingfrozenrawreports. Actualembed/headsourceis1323827200B each,not5295308800B(16ranklogical/fourDPcopies);raw+bytearraycomponent2647654400B,retainedTPsource1323827200Bvisible330956800B. Originalmax(4S,24Q)+64MiB=5362417664Bserialplanningallowancecanremain;myreasonforcallingitincompletewasfalse. Actualsourceexports19decodeoutputs,not24;fivecachepublishedrowsubchecksareconditionalandnotextraexports. Independent19PASSiscurrentfact. PublicPyPTORunConfigpathDOESrequirepower-of-twoheapentries>=1024;lowerCallConfigpermissivenessdoesnotremoveit. Reviewer3GiBjobfailedbeforecompile/prepare/weights/native;readinessCPU105016independentlyconfirms3GiBrejectionand256MiB/1/2/4GiBacceptance. No bypass. Supported4GiBring2costs4.75GiB/bank/rank;actualconstructedbankcountremainsunknown. No newmodel/NPU/fullpayloadacceptancefromthisreadiness.

## Main host contract update 1794036453 — current source762dc..., no new ABI/math

Only generate.py changed; all44 ABI/forward/golden/loader sources remain exact.
Generation/dry-run derive all-active prefillN from UNMODIFIED tokenized prompt
IDs, supporting2..128 and explicitly rejecting0,1,>128 when length preflight is
reached. No continuation text, pad tokens or hidden rows are added. Existing
factoryN and metadata use rowN-1; state handoff/first decode position is start+N.
Compile/report --tokens retains explicit2/16/128 buckets; device generation
uses the actual prompt length. General inactive-padding/F09 support is not
inferred from this exactN single-request path.

TP4 vocab shard reconstruction remains unchanged. Requested greedy replica
checking requires agreement from all4DP groups and records all votes/shard scores
on successful samples and structured failure, at prefill AND every decode step.
Failed selection preserves prior input/history/position evidence and fails the
entry; EOS/history arithmetic is unchanged. Current tests use tiny CPU logits
and host backend spies, not real model text. Shared-host mutable H2D/D2H IO,
sequential prefill/decode owners, scalar marshalling and streaming leaf ABI
are unchanged (no device-persistent KV claim).

make_run_config explicitly binds PUBLIC ring_heap=[0,0,4294967296,0], ONBOARD.
Actual PreparedDispatch mock verifies same normalized configobject on prepare
and4dispatches; classbody remains unchanged. Fullweight guard still fails even
with opt-in. Oldindependent native4GiB shared-TMR evidence is in decode capture
report; production exactN host case awaits independent native acceptance.
Futures empty-fit helper is rebound to current source, exact public4GiB config,
largest->CLOSE/freshowner->aggregate phases; no hardware allocation authorized.
Current details/plans: logs/generation_entry_1794035858. Mutable-count erratum:
prefix16/decode12; original exported gates17/19, not changed by these host edits.
