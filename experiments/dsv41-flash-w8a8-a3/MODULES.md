# Modules and composition

## Current configurable A3 closeout — nondefault real EOS; strict whole NOT_ACCEPTED (2026-11-08)

User Decision11 supersedes the fixed-topology restriction for this scope. A3
accepts import-time `--tp N`/`--tp=N`, `--ep N`/`--ep=N`: TP1/2/4/8, EP2/4/8/16,
EP>=TP and EP%TP==0 give13 legal configurations; WORLD=EP, DP=EP//TP with
contiguous TP groups. DefaultTP4/EP16 remains. A5 supplies the CLI/reference
interface, not A3 quant kernels or a claim that A5 fullentry supports arbitraryTP.
Current W8A8 source45
`f0e91a30e4117e110be183ef44acf2874eea813cc6f72ac7c7fa44bc4ede3a7f`, generate
`bc3cd736df208251f3787fd0fccbee1abccf9ee52dba4d810b1b1c0dfad7f4b7`, TP8plan
`a7b187a084e304b5203aeb2b0897dcb291550fbcf6c5bbaea6d1fe42f49a2a46` and fresh
E40 pair `c11948187a3f2e3675d5fe9e0fd0a47f20af67ecbeac4866f947cb7aefca75e4`
remain frozen. [Implementation and limits](logs/configurable_parallel_1794108101/REPORT.md).
ExactALL45/topology/metadata/Root/resource guards and defaultHARD are preserved.

SmallEP compute slabs ceil-pad to16; count metadata ceil-pads local expert width
to32, while communication uses actual source-lane capacity. Zero-active ranks
serve peers and typed completion is retained. Goldens select destination-rank
experts; whole-expert EP weights are independentofTP. Indexer32heads and sharedFFN
remain replicated, and W8A8/NZ/offsetfold/FP32 scales stay unchanged. TP8 attention
uses physicalHEAD_TILE16 and only8 live heads; q/sink padding is inside tokenSPMD.
No Engram/MTP restoration, A5 quantization or numerical tolerance change.

Independent evidence is scoped: HEADS8 singlecard3 original attention cases
(16/640/128keys) pass at original thresholds; EP2TP1 actual full-shape sparse
W8A8 component3 active/idle scenarios have distinct durable actual/reference
capsules, exact12/6/6 count matrices and original numerical gates; TP2EP4 output
and FP32score257-tail reductions pass two-group isolation. These component proofs
are not fullE40 smallEP fit or all13 topology native acceptance. Independent177
source/resource/ABI/admission/PREQUEUE gates passed, and fresh normal E40 TP8
prefillN3/decode1 signatures have100/96formals and57statics each.
[Attention acceptance](logs/root_tp8_attention_1794111650_acceptance.json),
[component acceptance](logs/root_configurable_components_1794113378_acceptance.json).

Actual normal real-checkpoint TP8EP16DP2 task130915_150556113554 reached modelCLI0,
genuineEOS after6IDs `[1014,14,1031,515,1613,1]`, five actual prior-greedy decode
calls atpositions3..7/end8. With plain prompt Once upon a/[16600,4465,260], output
beforeEOS was `Once upon a time, there was...`; raw suffix includes the tokenizer's
EOS marker. Prefill device time2.16s. noEngram/noMTP/TP8DP2EP16/W8A8 and both DP
replicas' eight-shard finite votes were checked. Text quality and original-model
numerical/quant/history parity remain UNASSESSED; this is an observed real text
stream, not an independent full-model reference comparison.

Root [final bounded disposition](logs/root_configurable_real_generation_1794111133_disposition.json) SHA`c66550dd667eae41a925dee4be4b18683c313f2965c59f19e15b76eafe07979b` accepts bounded
configured real generation/reuse/resource/known-cleanup facts ONLY; whole
NOT_ACCEPTED is retained. [Independent actual audit](logs/independent_configurable_generation_1794114713/REPORT.md) has4363PASS/
5telemetry-only unmet, bounded_unmet=[]; audit manifest`d876b17e7d029eaf0ee8fa8aac1a07a4368fd5e88d2e4acf79dd7f37bcf3136d`,
result`ca80ff1c46bf70b3ddcad9d2eaa92ad6c54da5c2987839e35eb45af5128ea549`. Numerical/text/transport facts do not
override the unchanged strict observation policy.

Verified bounded reuse/state-interface/resource facts: ONE613,256,211,456B static
upload/57arguments/912bases/164192source reads745,846,365,184B; ONE shared owner/
16workers; decode adds ZERO static reads/upload/allocations/copies, borrowing112
C1 singleton views+800same-shape bindings, canonicalfreeonce. Eight per-step state
fingerprints are contiguousFLOAT32 values→SHA1→first16hex diagnostics, NOT raw
cache-storage bytes or real original cache numerical/history parity. Six persistent
cache InOut formals/current runtime artifacts, finite prior-greedy token/history/
position progression and unchanged public resource guards were checked. HC source
extraction finds8RINT-down/16up PTO and2down/2up CPP occurrences per phase; these
are code counts, not native task counts or full-model numerical proof.

Whole case remains NOT_ACCEPTED: nativequeue1/strictouterFAIL259 noncritical
telemetry gaps, critical[]/monitor[], wholePSSnull. Known partial12,300,867,584B
is NOT wholepeak. ExternalPOST162902 COMPLETE0, raw16locks released/queueidle and
all1050 recorded births absent are bounded known-cleanup facts only. Actual owner
final metadata seal is PENDING; no complete process history,
continuous notification coverage or transient peak is inferred. Recorded progress
notification gaps/raw-event versus report times remain explicit UNKNOWN.

EP8 fullE40 routedweights alone68,022,435,840B/rank do not fit the current card;
TP8EP16 statics38,328,513,216B/rank fit only under the recorded operation's public
guards, not a universal peak guarantee. Other legal topology configurations,
fulloriginalparity/quant/history, generalphysicalT/B/multirequest/unequalEOS/serving,
Engram/MTP, text quality and controlled speedup remain OPEN. OldSourceB wholeFAIL218,
oldM02FAIL183, initialcomponent CLI failure/lostcapsule failures and all earlier
F09 failures stay immutable; no audit/telemetry reclassification was applied.
Main's docs-only closeout changes no model/runtime/old evidence or acceptance.
Earlier current-status overlays below are historical and superseded.


## Current SourceB closeout — bounded facts accepted; strict whole NOT_ACCEPTED (2026-11-08)

Current applied source45 is B
`30b7106a5427c283917a7eeeec65aab598ab98bf5a70f1667285169c9c6d8eca`, generate
`5de1656a99e85699138879b2f577f4eed2639779937a7cdc91ae843fcef6e34c`, metadata
`cd8b17741688eeb9ef39d8c0925dbb74a8f9a1cfcc0052db2e0a8c1aacb48cf7`.
Mathematical baselineA8153 differs fromB only in the generator's expected-plan
source literal. Strict44/all45 source checks and defaultHARD remain intact.
[Application and plan binding](logs/f09_application_1794097608/REPORT.md).
Independent applied-source/metadata/HARD negatives and BOTH fresh normal E40
prefillN3/decode1 zero-sample signatures are ACCEPTED (083640 COMPLETE0,
[receipt704eb14f](logs/independent_f09_applied_signatures_1794097836/signature_receipt.json)).
The new plan preserves old806 resources/HC lineage and original e13 resource
projection, adding only the named F09 lineage and two source fields; old
M02/E1 artifacts are ancestry, never relabeled as these fresh B signatures.

SourceA8153 bounded numerical acceptance is separate: synthetic reused E1 expert,
40dense/fixedphysicalT2/B2/4pages/2stateblocks/noEngram, three actual returned-state
calls with active-prefix2→1/0→2, six saved snapshots, ALL171 unchanged-tolerance
per-call19/staged19/independent-three-call19 checks plus9 pristine integrity rows.
[Root bounded acceptance](logs/root_f09_8153_bounded_acceptance_1794092629.json).
Production active-prefix MoE/count-aware references and C2 state-write/last-active
flush repairs are applied. This does not establish realE40 original numerical
parity, original prefill history or general physicalT/B serving behavior.

Current B real task090706: normal model CLI0 generated16 tokens through15 actual
decode calls, but native queue exit1/strict outerFAIL218 telemetry gaps makes the
WHOLE CASE NOT_ACCEPTED. Whole-own-tree PSS is null; known partial samples are
not full peaks. Independent audit095507 COMPLETE0 has4638PASS/5 telemetry-only
unmet; it does not override strict failure. Root accepted bounded components
ONLY in [dispositionc57a0f21](logs/root_sourceB_real_generation_1794098661_disposition.json).
[Independent actual audit](logs/independent_f09_real_B_generation_1794102875/REPORT.md)
is sealed; owner metadata final seal was pending at this closeout update.

Bounded real B facts: ONE633,997,065,216B static upload/57arguments/912bases,
164,192 source reads745,846,365,184B, ONE owner/16 workers; decode adds ZERO
static reads/upload/allocations, borrowing112 C1 singleton views+800 same-shape
bindings. Canonical bases freed exactly once. Actual prior-greedy tokens/history,
positions3..17/end18, all4DP finite votes, public resource guards and six persistent
cache InOut formals were verified. Runtime HC code each phase contains4down/4up
PTO and2down/2up CPP witnesses. Per-step state SHA1 strings are contiguous
FLOAT32-value diagnostics truncated to16hex, NOT original cache-storage bytes
and NOT independent realE40 cache/history parity. No stronger state claim follows.

Plain prompt Once upon a has original IDs[16600,4465,260], noEngram/noMTP,
TP4/DP4/EP16/W8A8. Its actual16-token suffix was
` time...". What is the meaning of the phrase "once upon a time"?`, identical
to prior M02 IDs as an observation only. Text quality and causal HC/F09 quality
improvement are UNASSESSED. External POST094741 COMPLETE0, raw16 locks released,
queue idle, and all1160 recorded births absent (audit1159 plus the reconciled
outer parent) support bounded known-cleanup facts, not complete process history.

Old M02 strictFAIL183 and all earlier F09 failures stay immutable; no audit or
telemetry reclassification was applied. Full original/quant parity, real cache
numerical/history parity, whole PSS/transient peak, controlled speedup, general
physicalT/B/F09/unequalEOS/multirequest serving, Engram and text quality remain
OPEN. SourceB and all old evidence stay frozen. Main is PARKED after docs-only
closeout; any next source change or native execution needs its separately scoped
Root assignment. Earlier “current” sections below are historical and superseded.

## Current HC BF16 boundary application (2026-11-08)

Newest actualsource45 `e07b28713eadd00a306fd60264223dbf6ceec53761c7ac48c347032011c9ca25`
(5534bytes/exact45paths), genf7b772cfa62ea38ebc9228e0230b770d8cf29a76527071cfe75b0c2bd5035e76.
ExactHC2patch7520 +onegeneratebaseline literale5f740 applied; other42sameb3e.
M01mathbaselineeb90ea56...fecf2; NEWmetadata plan `806852d5baec99a522f5adc3015bdd204d9ac6b17244a29cb26445cbd55fda89`.
[Application](logs/hc_boundary_apply_1794075705/REPORT.md),
[bindings/nextgates](logs/hc_boundary_apply_1794075705/NEXT_COMMANDS.md).
HCpostnowBF16RINT->FP32publicstorage andbothgoldensroundaftereachattention/FFN;
HCpre/head/order/othermath/tols unchanged. Strictall44newmath sourceequality
preserved; oldresource-keyprojection exact, ONLYnamed/hashboundhc_math_lineage
added. Sharedweightowner behavior/accounting unchanged. DefaultHARD remains.

CapturednativeHC021439wholeFAILED ownexpected/outputalias remainsFAILED; offline
original02181631/0 proves syntheticcandidate102400bitexact/old61440different.
Rootacceptedthatboundedcombinedproof, notwholecasePASS/parity/repetitionrepair.
AppliedCPU022432 COMPLETE0 actualmath-original/default/source/resource driftproof.
No compile/NPU/payload/actualgrant here. BothFRESH E40N3/decode1 signaturesPENDING;
old611D77 notnewmathartifact. Oldb3ehelper/receiptbindingsmustfailclosedandRoot
separatelyassignsnewsourceplan-bound operation. Fullrealshared16peak/perf/text
qualityUNTESTED; priorCClongercaseFAILED, causalrepairunknown. F09UNAPPLIED.
MainPARK/sourcefrozen; olderb3e/CC/D77/source labelsbelow historical.


## Current shared-owner driver application (2026-11-08)

This newest overlay supersedes historical current labels below. Exact reviewed
366ed05e patch APPLIED onlygenerate.py, digest `fba8a4ef9a06f694fa53278bb80ae2ad70d111336dbead938ab9f8511d27e16e`.
NEW actualsource45 `b3e19fd001c74f7a5b497c791c01924ca09c0678df66114fce599e890d8fee06`
(5534bytes/exact45paths); other44 unchanged fromCC.
[Application](logs/shared_weight_owner_apply_1794074398/REPORT.md) and
[nextcommands/provenance](logs/shared_weight_owner_apply_1794074398/NEXT_COMMANDS.md).
ONE tiny appliedCPU020157 COMPLETE0; no compile/NPU/payload/positivegrant here.

Implemented publicsameworker two-program owner: compileboth beforeONEprepare,
bothHOSTIO beforefork, actualhandoff copied intoexistingdecoderbuffers. Canonical
57resources/912basehandles initializeONCE:633997065216BglobalH2D,164192sourcecalls/
745846365184B. Decoder0newstaticIO/rereads;112C1borrowedviews+800same-shapebindings
reference thoseinitializedbases; canonicalfreeonce, viewsneverfree. Newoperation
`shared-full-real40-noEngram-greedy-N3` and exacttrueintegerweight_accounting
are mandatory; legacyseparatepath/defaultHARD preserved. No tensor/mathABI change.

Old611 remainsD77provenance; unchanged44/compilefactory/N3-pages2/decode1 relation
is explicit, not a newcompile. FullREAL16sharedtransport/peak/performance/text
quality remainUNTESTED. EarlierD77[3,1]/ONEdecode accepted onlyshortstream; CC
Onceupona16 case overallFAILED/repetitionNOTfixedbyreuse. HC17/F09drafts remain
UNAPPLIED; HCcandidate limitednativepass lacksdurableoldcontrolreceipt. No original
parity/generalF09/Engram closure. RootseparateactualGO+newoutercode/receipt required;
MainPARK. Source45here iscurrent; olderCC/D77 labelsbelow arehistorical.


## Current longer-N3 admission application (2026-11-07)

This newest overlay supersedes historical current labels below. Exact reviewed
41linec825 patch is APPLIED only togenerate.py, actualdigest
41651e5af3b94223b645d4bfb52010e52ea7350aa3f3637cbcca6e29e093e609.
NEW source45`cc15877f8f4b0104a842f915d2434d9a2c9cd6f6752d6364d5d7bada26e21d8c`;
other44 unchanged fromD77. [Application report](logs/longer_generation_admission_apply_1794062947/REPORT.md)
and [nextcommands/provenance](logs/longer_generation_admission_apply_1794062947/NEXT_COMMANDS.md).
ONE small appliedCPU check passes. No NPU/payload/grant/signaturecompile occurred
in this source-application unit. Default no-receipt HARD refusal remains.

Accepted actual firstshortstream remains D77/204310: generated[3,1]/!<EOS>,
ONEdecode; owner34 and independentaudit13 sealed. It is not15-call/longtext/parity.
Current host admission binds NEWoperation/receiptplain text/exact3IDs andinteger
cap2..16. FIRSTnextactualscope Onceupona[16600,4465,260]/max16/noEngram only.
Allmath/load/copy/cache/config/budgets/cadence/cleanup/durability are unchanged.
Old611 signature artifacts retain D77source; unchanged44/entry/compilemethod plus
sameN3/pages2/decode1 relation is recorded, not a relabeled new compilation.
Actual15-call retainedslot memory, coherence and otherfullgoal gates remainOPEN.

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

`generate.py` now owns the normal explicit first-real40/noEngram admission,
source/allocation/copy/forward live guards, complete912 coverage and oncephase
source/copy summaries, durable rawfinal memory, and firsterror/attemptall cleanup.
Existing trueleaf loader, all44 math/golden/forward files, exactprompt/strictDP/
actualgreedy history/EOS behavior remain unchanged. Independent entry92 and
budget28 reviews accepted the proposal scope; this is source application only.
Reviewer next owns new-source E40 N3/decode1 zero-sample signatures. A separate
Root actual-operation grant is still required; fullreal fit/forward peaks/text
and fulloriginal parity remain OPEN.

## Current repair and validation overlay (2026-11-06)

This overlay and latest TASKS supersede status claims in the historical maps
below, including old simulator commands, pending-observer labels and planned
complete-entry rows. Goal decisions1–9 and COORDINATION.md govern current work.
Independent reviewer/validator executes acceptance; observer records decisions.

| Scope | Current owner / behavior | Recorded bounded acceptance |
| --- | --- | --- |
| Generic indexer / C1A reindex | indexer.py; actual phase/forward pool consumers and their references | Width/tail initialization and source-pool consumption accepted; logs/observer_indexer_verify_1793930090 and observer_reindex_verify_1793934174. Main16-card C1A layer evidence is separately attributed, not an independent rerun. |
| F01 common MLA | mla_proj.py; Wq_b BF16 before RoPE, no head norm | logs/observer_native_transition_1793935573 and native_review_f01_1793935855 |
| F02 output projection | o_proj.py; inverse query-position RoPE before Wo_a/BF16/Wo_b; all phase and layer-reference callers | logs/native_accept_f02_1793946300/REVIEW.md; independent33 positives/two numerical controls and CPU forward compiles |
| F03 source publication | Four C1/C2 full-source phase files, rope.py owning golden, layer_reference.py, paged_cache.py published-row checks | logs/native_accept_cohort_1793951619/REVIEW.md; raw latent retained for index K, separately rotated attention KV; supplied-table basis only, not F04 coordinates |
| F08 router | moe_gate.py; one legal unique expert per tie slot, unbiased route weights | Same independent cohort review; rank-local tie/default controls, not F09/F10 distributed lifecycle acceptance |
| F14 loader | weight_loader.py; retain folded expert payload before NZ packing | Same independent cohort review; synthetic representable offsets/zero/endpoints/rejections only; actual-checkpoint incidence unproved |
| F04 absolute source coordinates | Shared absolute-source helper and all actual table builders | Independently accepted in logs/native_accept_f04_f10_1793955766/REVIEW.md; bounded actual-state continuation, not general F09 metadata lifecycle |
| F10 capacity stride | moe_exchange.py push/compact share capacity stride | Same independent review; actual16-device unequal/idle/replay evidence, not full-forward lifecycle |
| Active F05/F06 | engram.py restores original hidden gate basis; shared FP32 rotation resource loaded/bound; Engram precedes HC pre/attention in actual and reference paths | Implemented/self-validated and frozen; independent acceptance pending. logs/cohort_engram_f07_1793957136/REPORT.md |
| Active F07 capacity | config/indexer/four source/two C1 reindex owners restore2048 | Implemented/self-validated with actual65/2048/2049/4097 block boundaries and old64 rejection; independent acceptance pending. Same artifact root |

Full-model state/pool publication, real dynamic lifecycle, independent acceptance
of the current Engram/capacity cohort, standalone TP, ordinary16-card layer suites, CI ownership and
generation remain OPEN/HELD. Earlier complete-entry PASSes are historical and
do not accept the semantically repaired current tree. Exact primitive/tie-quant
parity remains unproved. Older maps are retained for provenance, not as current
whole-model acceptance or executable CI inventory.

Spec revision: 1 (2026-11-02). Framing decision: FRESH implementation for
A3 in a new model directory `models/deepseek_v4_1_flash_a3_w8a8/` (final name;
recorded here per frozen decision 7). The existing
`models/deepseek_v4_1_flash/` targets the A5 platform with FP8/MXFP4 weights
and CANNOT be reused directly on A3 (user decision 2026-11-02); it is used as
a semantics/decomposition reference only. `models/deepseek_v4_flash_mtp/` is
a different model/checkpoint, used only as a reference for the repository's
established W8A8 int8 kernel idioms (NZ packing, per-token activation quant,
int32 accumulation). No kernel is imported from either tree.

Quantization contract (checkpoint ground truth: safetensors headers +
`quant_model_description.json`):

- W8A8_DYNAMIC tensors: `weight` I8 `[N, K]` (logical `[out, in]`),
  `weight_scale` F32 `[N, 1]`, `weight_offset` F32 `[N, 1]`. Measured offsets
  are all zero for sampled attention/expert tensors; the loader still applies
  `w = (q - offset) * scale` and records the offset check. Activations are
  quantized dynamically per token to int8 with an F32 scale. GEMM accumulates
  in INT32 and dequantizes with both scales.
- Quantized in the text backbone: `attn.wq_a`, `attn.wq_b`, `attn.wkv`,
  `attn.indexer.wq_b`, every routed expert `w1/w2/w3`, and
  `ffn.shared_experts.w1/w2/w3`.
- FLOAT/BF16 in the text backbone: `embed.weight`, `head.weight`, all norms,
  `attn.wo_a`, `attn.wo_b`, `attn.attn_sink`, compressor weights, indexer
  `wk/weights_proj/k_norm`, MoE gate, HC parameters, and Engram
  (`q_weight`, `k_weight`, `wkv.weight`; host table int8_sym group-32 with
  F32 scales and BF16 output).

Topology contract (A3 form, spec revision 1): world 16 = DP4 x TP4, expert
parallel over all 16 ranks. Attention TP4 shards 64 heads -> 16 heads/rank
and 8 output groups -> 2 groups/rank; dense weights are replicated per DP
group (measured 2.92 GiB/rank). Routed experts are sharded over EP16 (24
experts/rank, 31.71 GiB/rank). Shared expert is replicated/DP per the A3
form. Engram tables are host-resident (206.59 GiB) and shared by the 4 DP
ranks of a node.

## Module boundaries

| Module | Producer / consumer | Interface / weights | Acceptance | Owner |
| --- | --- | --- | --- | --- |
| `config.py` | imported by every operator/entry | model dimensions from `MODEL_ROOT/config.json`; A3 DP/TP/EP sizes; layer mode schedule; quant constants | imported by every entry; values checked against checkpoint config | model owner |
| `utils.py` | imported by operators/entries | host only: NZ pack/unpack, per-row int8 quant reference, paged slot helpers, checkpoint reader | exercised by every golden; no device kernel | model owner |
| `quant.py` | consumed by `gemm_w8a8`, `mla_proj`, `moe_experts` | device: BF16 tile -> INT8 tile + per-token F32 scale; host reference | bounded sim validation vs `golden_int8_quant_per_token` | model owner |
| `rmsnorm.py` | consumed by `mla_proj`, `lm_head`, `layer` | BF16 `[T,D]` + BF16 gamma `[D]` -> BF16 `[T,D]` | bounded sim validation vs `golden_rms_norm` | model owner |
| `gemm_w8a8.py` | consumed by `mla_proj`, `moe_experts`, `indexer` | INT8 `[K,N]` (NZ) + F32 scale `[N]` + INT8 `[T,K]` + F32 scale `[T]` -> BF16/FP32 `[T,N]` | bounded sim validation vs `golden_w8a8_gemm` (ratio_allclose) | model owner |
| `mla_proj.py` | consumed by `prefill_attention`, `decode_attention` | `x_hc`/`x` BF16 -> `q [T,64,512]`, `kv [T,512]`, `qr [T,1280]`+scale; wq_a/wq_b/wkv W8A8, q_norm/kv_norm, interleaved RoPE | bounded sim validation vs `golden_mla_proj` | model owner |
| `compressor.py` | consumed by `prefill_attention`, `decode_attention` | FLOAT wkv/wgate/norm; KV history -> compressed KV rows + per-position state | bounded sim validation vs `golden_compressor` | model owner |
| `indexer.py` | consumed by `prefill_attention`, `decode_attention` | indexer wq_b W8A8; wk/weights_proj/k_norm FLOAT; top-k position list | bounded sim validation vs `golden_indexer` (topk_pair_compare) | model owner |
| `attention_gather.py` | consumed by `prefill_attention`, `decode_attention` | union key matrix + visibility mask: raw window (128) + current chunk + compressed history + current latent; Top-K membership for compressed slots | bounded validation vs `golden_assemble_key_set` / `golden_c1a_key_set` | model owner |
| `attention.py` | consumed by phase attention files | shared MLA core: gather KV rows, causal/SWA mask, sink-augmented softmax, PV, per-group output | bounded sim validation vs `golden_attention_core` | model owner |
| `prefill_attention.py` | consumed by `prefill_layer` | prefill orchestration per mode: swa, c1a full/reindex/reuse, c2a full/reuse; token tiles > 1 | bounded sim validation at L>1 vs `golden_prefill_attention` | model owner |
| `decode_attention.py` | consumed by `decode_layer` | decode orchestration per mode; consumes/updates paged state | bounded sim validation vs `golden_decode_attention` | model owner |
| `o_proj.py` | consumed by attention phases | FLOAT wo_a/wo_b grouped output projection + TP all-reduce | bounded sim validation vs `golden_o_proj` | model owner |
| `tp_reduce.py` | shared child of `prefill_layer`, `decode_layer` | TP4 group reductions: BF16 output-projection partial sum and FP32 index-score partial sum; folded publish/release epoch counters; window tensors owned by the L3 host driver | exercised through both layer entries (no standalone entry by design) | model owner |
| `hc.py` | consumed by `prefill_layer`, `decode_layer` | HC pre/post/head, `(2+hc_mult)*hc_mult` mixing, sinkhorn 20, eps 1e-6; function arg padded to 32 rows | bounded sim validation vs `golden_hc_pre/post/head` | model owner |
| `moe_gate.py` | consumed by `prefill_moe`, `decode_moe` | FLOAT gate weight/bias; sqrtsoftplus, noaux_tc, top-6, norm_topk_prob, scale 1.5 | bounded sim validation vs `golden_moe_gate` (topk_pair_compare) | model owner |
| `moe_experts.py` | consumed by `prefill_moe`, `decode_moe` | W8A8 routed/shared experts w1/w3 -> SwiGLU(limit 10) -> w2 | bounded sim validation vs `golden_expert_routed/shared` | model owner |
| `moe_exchange.py` | shared child of `prefill_moe`, `decode_moe`, `prefill_layer`, both fwd entries | EP16 window-buffer dispatch/combine kernel + golden; `rows: pl.constexpr` pins the token count; `moe_ffn` composes FFN HC pre/norm/gate + inline static-tile exchange + HC post (A5 `moe()` shape). Routed-expert ABI: device-resident per-layer binding for real weights; bounded complete-entry fixtures use one contiguous per-rank shard (full-stack host fixture rejected on the 543 GB bound; INTERFACE.md) | 16-card device PASS (prefill_moe; layer suites); `moe_ffn` composition compiles with every chain class | model owner |
| `prefill_moe.py` | consumed by `prefill_layer` | EP16 host driver + rank entry for a two-token tile; token-major slabs; worst-case receives from the token tile | bounded 16-rank simulator run | model owner |
| `decode_moe.py` | consumed by `decode_layer` | EP16 host driver + rank entry for one active token | bounded 16-rank simulator run | model owner |
| `engram.py` | consumed by `prefill_layer`, `decode_layer` at layers 1/14 | host table lookup + int8 group-32 dequant -> BF16 lookup row; device 6144->25600 BF16 projection + gate*V add | bounded validation vs `golden_engram` | model owner |
| `embedding.py` | consumed by `prefill_fwd`, `decode_fwd` | BF16 `embed.weight` lookup | bounded sim validation vs `golden_embedding` | model owner |
| `lm_head.py` | consumed by `prefill_fwd`, `decode_fwd` | final RMSNorm + BF16 head + logits/argmax | bounded sim validation vs `golden_lm_head` | model owner |
| `prefill_layer.py` | consumed by `prefill_fwd` | ordered layer assembled from three host programs: prolog (A5 `<mode>` chain with HC inputs -> Engram -> FFN hc_pre/norm/gate -> static MoE-tile staging), EP16 exchange (static 16-row tile, pad rows route 0/weight 0), epilogue (unpad -> final hc_post); staged runner shares host buffers across the three compilations | **device 7/7 on 16 cards** vs `make_layer_golden` + staged references (see TASKS for precision) | model owner |
| `decode_layer.py` | validation-only (fwd builds its own staged loop) | trace-time mode dispatch over the A5 decode chains (HC inputs) + Engram + FFN prolog; three host programs (prolog / static EP16 tile / epilogue) staged over shared buffers | **device 7/7 on 16 cards** (2-token plan, exchange tile = compiled bound) | model owner |
| `prefill_fwd.py` | complete-model entry | embedding, 40 layers, final norm, head; ordered weight/state arguments | compile + bounded sim run + structural gate | model owner |
| `decode_fwd.py` | complete-model entry | embedding, 40 layers, final norm, head; persistent state in/out | compile + bounded sim run + structural gate | model owner |
| `structural_check.py` | structural gate tool | inspects source call graph and generated IR/task-graph shapes for token tiles and loop nesting; runs negative control | rejects tokenwise full-stack control, incl. renamed wrapper | model owner |
| `weight_map.py` | host weight loader/mapper | maps checkpoint names -> ordered entry arguments, rank slices, transforms | reports missing/unexpected/unused; Engram/DSpark/MTP/vision exclusions | model owner |
| `weight_loader.py` | host weight preparation for the complete entries | frozen `weight_map.json` consumed as-is (pre-rotated/pre-norm-folded); TP4 vocab/head and head-shard slices, EP16 24-expert shards, W8A8 offset fold + `[K,N]` transpose + NZ packing, HC 24-to-32 zero-row padding, host Engram table accessors, per-layer/stacked accessors | `python models/deepseek_v4_1_flash_a3_w8a8/weight_loader.py --layers 0,1,20,39 --sample 4` (exit 0; direct-read samples, coverage counts) | `prefill_fwd`, `decode_fwd`, `generate` weight preparation | model owner |
| `tokenizer.py` | host generation driver | pure-Python DeepSeek BPE from `tokenizer.json` (no transformers/tokenizers/regex installed): pretokenizer splits, byte-level map, merges, 1283 added tokens, decode, plain chat-message subset cross-checked against the shipped `encoding/encoding.py` | `python models/deepseek_v4_1_flash_a3_w8a8/tokenizer.py` (exit 0; round-trips, atomic specials, printed ids) | `generate` | model owner |
| `generate.py` | generation driver | host tokenize/sample loop; each forward calls the compiled entry | greedy fixed-prompt generation via compiled entries | model owner |

For each row the operand contract is fully specified in the owning file's
kernel signature (shape, dtype, direction, rank-local partition, ownership)
and in the golden; state updates and communication dependencies are recorded
in the execution-structure section.

## File ownership and validation map

Status legend: `planned` (map complete, code not yet written), `implemented`
(code + bounded validation pass), `blocked` (see TASKS.md evidence).

| Operator / phase | Owning file / compute symbol | Golden source / symbol | Default validation command / comparator | Actual composition consumer / variant | Status |
| --- | --- | --- | --- | --- | --- |
| per-token int8 quant | `quant.py` / `quantize_int8_per_token` | `quant.py` / `golden_quantize_int8_per_token` | `python models/deepseek_v4_1_flash_a3_w8a8/quant.py -p a2a3 --sim` / `ratio_allclose` | `gemm_w8a8`, `mla_proj`, `moe_experts` | implemented (PASS 11.39s) |
| W8A8 GEMM | `gemm_w8a8.py` / `w8a8_gemm` | `gemm_w8a8.py` / `golden_w8a8_gemm` | `python models/deepseek_v4_1_flash_a3_w8a8/gemm_w8a8.py -p a2a3 -d $TASK_DEVICE --tokens 64` / `ratio_allclose` | `mla_proj`, `moe_experts`, `indexer` | implemented; token axis scheduled through one `pl.spmd` with a composite dynamic block count (A5-provable form, needed for inline-MoE composition); device PASS 7.68 s after the loop restructure (`logs/loop_align_device_consumers_1.log`) |
| RMSNorm | `rmsnorm.py` / `rms_norm` (column tile constexpr), `rms_norm_fp32_unit` | `rmsnorm.py` / `golden_rms_norm` | `python models/deepseek_v4_1_flash_a3_w8a8/rmsnorm.py -p a2a3 --sim` / `allclose` | `mla_proj`, `lm_head`, `indexer`, layers | implemented (PASS 10.25s; 128/512/1280/5120 widths via explicit column_tile) |
| MLA Q/KV projection | `mla_proj.py` / `mla_proj` | `mla_proj.py` / `_golden_mla_proj`, independent closed-form `golden_mla_semantic` | `python models/deepseek_v4_1_flash_a3_w8a8/mla_proj.py -p a2a3 -d $TASK_DEVICE` / existing random ratio gates plus exact signed sparse Q/KV/qr | all twelve phase entries via shared MLA; layer/full references import its golden | F01 common Q equation device validated: Wq_b BF16 before RoPE, no head norm. Three defaults and independent pinned-source random T17 PASS; both mutations rejected. All twelve consumers/eighteen cases PASS; F02–F04 and full-model device gates OPEN. Evidence `logs/attention_f01_1793934665/REPORT.md` |
| Interleaved RoPE | `rope.py` / `rope_tail_fp32`, `rope_tail_bf16` (generic head width) | covered by consumer goldens | revalidated through `mla_proj` and `indexer` entries | `mla_proj`, `indexer` | implemented (consumer PASS) |
| KV compressor | `compressor.py` / `c1a_compressor`, `c2a_compressor`, `bf16_project` | `compressor.py` / `golden_c1a_compressor`, `golden_c2a_compressor` | `python models/deepseek_v4_1_flash_a3_w8a8/compressor.py -p a2a3 -d $TASK_DEVICE` / `ratio_allclose` | C2A/C1A full attention | implemented; `bf16_project` uses the A5-provable fused-spmd token schedule (needed for inline-MoE composition); device PASS 7.44/4.97 s |
| Lightning indexer | `indexer.py` / `lightning_indexer`, `index_query`, `index_key`, `index_scores`, `index_topk`, `select_candidate_block_pool`, `filter_candidate_scores` | `indexer.py` / generic goldens and independent `golden_reindex_select` | `python models/deepseek_v4_1_flash_a3_w8a8/indexer.py -p a2a3 -d $TASK_DEVICE` / score ratio, paged `index_topk_compare`, exact restricted selection/pool gates | generic entries; decode/prefill C1A full/reindex and C2A full; reindex flows into both layer and complete-forward entries | Generic width/trailing-page repair independently accepted. Source-pool consumption repaired against original rc2 semantics: six indexer + nine phase device cases PASS, both C1A reindex layers PASS on16 devices using supported fresh workers. Both complete forwards compile with source-pool bindings; full-model device gates remain OPEN. Original generic goldens and score thresholds unchanged; reindex goldens corrected and sparse padding comparison enforces exact valid lanes. Main-agent evidence pending observer verification: `logs/reindex_repair_1793930492/REPORT.md` |
| MLA attention core | `attention.py` / `attention_core` (key-tiled online softmax) | `attention.py` / `golden_attention_core` | `python models/deepseek_v4_1_flash_a3_w8a8/attention.py -p a2a3 --sim` / `ratio_allclose` | phase attention files | implemented (PASS 14.66s at 16 keys + 13.40s at 128 keys/8 tiles) |
| Sparse key-set assembly | `attention_gather.py` / `assemble_key_set`, `assemble_swa_key_set` | `attention_gather.py` / `golden_assemble_key_set`, `golden_c1a_key_set`, `golden_swa_key_set` | `python models/deepseek_v4_1_flash_a3_w8a8/attention_gather.py -p a2a3 --sim` / `ratio_allclose` (exact) | `prefill_attention`, `decode_attention` | implemented (5 cases PASS: C2A prefill 11.49s / C1A prefill 10.12s / C2A decode 10.30s / SWA prefill 9.51s / SWA decode 9.67s) |
| prefill attention | `prefill_attention.py` / `prefill_attention_c1a`, `prefill_attention_c2a`, `prefill_attention_swa`, `prefill_attention_reuse` | `prefill_attention.py` / `golden_prefill_attention`, `golden_prefill_attention_c2a`, `golden_prefill_attention_swa`, `golden_prefill_attention_reuse_c2a/c1a` | `python models/deepseek_v4_1_flash_a3_w8a8/prefill_attention.py -p a2a3 --sim` / `ratio_allclose` + `topk_pair_compare` + pool-set | `prefill_layer` | implemented all modes (6 cases PASS: C1A 33.09s, C2A even 32.28s, C2A odd 4.41s, SWA 23.31s, C2A reuse 25.01s, C1A reuse 25.45s; state InOut for C2A) |
| decode attention | `decode_attention.py` / `decode_attention`, `decode_attention_swa`, `decode_attention_reuse`; entries `decode_attention_test`, `decode_handoff_test`, `decode_swa_test`, `decode_reuse_test` | `decode_attention.py` / `golden_decode_attention`, `golden_decode_attention_swa`, `golden_decode_attention_reuse` | `python models/deepseek_v4_1_flash_a3_w8a8/decode_attention.py -p a2a3 --sim` / `ratio_allclose` + `topk_pair_compare` | `decode_layer` | implemented all modes (4 cases PASS: 768-key 34.46s, prefill-handoff 784-key 31.36s, SWA 23.45s, reuse 23.28s) |
| output projection | `o_proj.py` / `o_proj` | `o_proj.py` / `golden_o_proj` | `python models/deepseek_v4_1_flash_a3_w8a8/o_proj.py -p a2a3 -d $TASK_DEVICE --tokens 16` / `ratio_allclose` | phase attention files | implemented; stage B token axis fused into the `pl.spmd` block count (A5-provable form); device PASS 7.83 s after the loop restructure; all 12 mode entries device PASS |
| TP4 group reductions | `tp_reduce.py` / `tp_output_reduce`, `tp_score_reduce` | covered by the composed layer golden (group partials summed in the reference) | consumers: `python models/deepseek_v4_1_flash_a3_w8a8/prefill_layer.py -p a2a3 --sim` / `ratio_allclose` | `prefill_layer`, `decode_layer` | implemented (consumer PASS; output reduce in every attention case, score reduce in the C1A full chain) |
| hyper-connections | `hc.py` / `hc_pre`, `hc_post`, `hc_head`, `hc_mixes`, `hc_coefficients` | `hc.py` / `golden_hc_pre`, `golden_hc_post`, `golden_hc_head` | `python models/deepseek_v4_1_flash_a3_w8a8/hc.py -p a2a3 --sim` / `ratio_allclose` | `prefill_layer`, `decode_layer` | implemented (PASS 14.55s/9.40s/9.59s; function padded to 32 rows) |
| MoE gate | `moe_gate.py` / `moe_gate` | `moe_gate.py` / `golden_moe_gate` | `python models/deepseek_v4_1_flash_a3_w8a8/moe_gate.py -p a2a3 -d $TASK_DEVICE` / `topk_pair_compare` | `prefill_moe`, `decode_moe` | implemented; device PASS (17.52s) after the int32-slot codegen fix; sim PASS 14.20s historical |
| routed/shared experts | `moe_experts.py` / `expert_routed`, `expert_shared` | `moe_experts.py` / `_expert_golden` | `python models/deepseek_v4_1_flash_a3_w8a8/moe_experts.py -p a2a3 -d $TASK_DEVICE` / `expert_output_compare` (device-calibrated: rel-L2 <=1%, atol 1e-1, rtol 5e-2, <=2% outliers) | `prefill_moe`, `decode_moe`, layer `staged_moe_compare` | implemented; device PASS both cases (8.76s/5.86s) after the 1-column TLOAD fix; device floor measured 0.32-0.41% rel-L2 (inside the 1-BF16-ULP envelope 0.88-1.27%) |
| prefill MoE dispatch/combine | `prefill_moe.py` / `prefill_moe` (host), `prefill_moe_rank`; kernel `moe_exchange.py` / `moe_ep16` | `moe_exchange.py` / `golden_moe_ep16` | `python models/deepseek_v4_1_flash_a3_w8a8/prefill_moe.py -p a2a3 --sim` / `ratio_allclose` | `prefill_layer` | implemented (16-rank EP16, T=2; PASS 18.7/20.5/19.2s over three runs; local/remote/skew/zero-receive/idle rank, 32-row worst-case expert) |
| decode MoE dispatch/combine | `decode_moe.py` / `decode_moe` (host), `decode_moe_rank`; kernel `moe_exchange.py` / `moe_ep16` | `moe_exchange.py` / `golden_moe_ep16` | `python models/deepseek_v4_1_flash_a3_w8a8/decode_moe.py -p a2a3 --sim` / `ratio_allclose` | `decode_layer` | implemented (16-rank EP16, T=1; PASS 19.3/18.6s twice; two destinations, 14 idle ranks) |
| Engram | `engram.py` / `engram_from_rows`, `engram_gate`, `HostEngramTable` | `engram.py` / `golden_engram` | `python models/deepseek_v4_1_flash_a3_w8a8/engram.py -p a2a3 --sim` / `ratio_allclose` | `prefill_layer`, `decode_layer` (layers 1/14) | implemented (PASS 15.00s; host int8 group-32 dequant self-check) |
| embedding | `embedding.py` / `embedding_lookup` | `embedding.py` / `golden_embedding` | `python models/deepseek_v4_1_flash_a3_w8a8/embedding.py -p a2a3 --sim` / exact | entry prolog | implemented (PASS 10.31s) |
| final norm + head | `lm_head.py` / `lm_head` | `lm_head.py` / `golden_lm_head` | `python models/deepseek_v4_1_flash_a3_w8a8/lm_head.py -p a2a3 --sim` / `ratio_allclose` | entries epilog; argmax is host-side | implemented (logits PASS 11.75s; host sampling open) |
| prefill layer assembly | `prefill_layer.py` / `prefill_layer_entry` (prolog), `prefill_layer_moe_entry` (static EP16), `prefill_layer_epilogue_entry`, `_build_layer_entries`, `_run_staged_layer` | `layer_reference.py` / `make_layer_golden` (+ staged references) | `python models/deepseek_v4_1_flash_a3_w8a8/prefill_layer.py -p a2a3 -d $TASK_DEVICE` / calibrated staged comparators | `prefill_fwd` | **device 7/7 PASS on 16 cards** (swa_layer0, swa_layer1_engram, c2a_full_layer2, c2a_reuse_layer3, c1a_full_layer20, c1a_reuse_layer21, c1a_reindex_layer24), task `task_20261104_140927_9088212593`; EP16 is a separate compiled program because a per-token exchange in the dynamic chain's program fails type checking (KNOWN_PYPTO_ISSUES.md) |
| decode layer assembly | `decode_layer.py` / `decode_layer` (constexpr mode), `_build_layer_entries`, shared `run_staged_programs` | `layer_reference.py` / `make_layer_golden` (+ staged references) | `python models/deepseek_v4_1_flash_a3_w8a8/decode_layer.py -p a2a3 -d $TASK_DEVICE` | fwd (validation-only file) | **device 7/7 PASS** (2-token plan, tile 2), closure run `task_20261104_152608_37247331295` |
| structural gate | `structural_check.py` / `run_structural_check` | n/a (inspects source + compiled artifact) | `python models/deepseek_v4_1_flash_a3_w8a8/structural_check.py --entry prefill_fwd --negative-control` | complete-model entries | planned |

Kernel-only small operators (no separate golden/validation entry, validated
through their consumer): NPU-NZ pack/unpack helpers in `utils.py` (host), and
scalar padding/slice glue inside the phase files (parent-operator connecting
logic). Every numerical primitive is owned by one of the rows above; no
operator is validated only by another variant of itself.

## First working paths

| Path | Import / call graph and source locations | Validation and composition symbols / variants | Command / actual result | Rule findings / resolution |
| --- | --- | --- | --- | --- |
| First attention path | `mla_proj -> quantize_int8_per_token -> w8a8_gemm -> rms_norm(_fp32_unit) -> _rope_tail_fp32/bf16`, then `attention_core` (tiled 16-row causal softmax + sink). `compressor/indexer/o_proj` are not yet implemented (open). | validation `mla_proj_test` and `attention_core_test` are `@pl.jit` entries calling the same `@pl.jit.inline` symbols (`mla_proj`, `attention_core`, `w8a8_gemm`, `quantize_int8_per_token`, `rms_norm*`) that phase attention will call | `python models/deepseek_v4_1_flash_a3_w8a8/mla_proj.py -p a2a3 --sim --tokens 16` -> PASS (14.52s); `attention_core.py ... --tokens 16` -> PASS (14.22s); exit 0 | call depth entry->mla_proj->{gemm,quant,rmsnorm} = 3; attention entry->core = 2. Phase-specific sparse gather and compressed-KV production remain open. |
| First MoE path | `moe_gate -> expert_routed (w8a8_gemm x3 + SwiGLU)`, `expert_shared` as a separate W8A8 SwiGLU path. EP dispatch/combine (`prefill_moe`/`decode_moe`) not yet implemented (open). | validation `moe_gate_test`/`expert_routed_test`/`expert_shared_test` entries call the same `moe_gate`, `expert_routed`, `expert_shared`, `w8a8_gemm`, `quantize_int8_per_token` symbols that layer composition will call | `moe_gate.py --tokens 16` -> PASS (14.20s); `moe_experts.py --rows 16` -> routed+shared PASS (12.57s); exit 0 | call depth entry->expert->gemm = 3. Distributed dispatch/combine, repeated-token grouping and receive-capacity checks remain open. |

## Complete-model entries

| Entry | Source / symbol | Ordered weights and state | Supported shapes | Compile / run command |
| --- | --- | --- | --- | --- |
| Prefill | `prefill_fwd.py` / `prefill_fwd` (NOT DELIVERED) | planned: embed; per layer: 12 norms, wq_a/wq_b/wkv (W8A8), wo_a/wo_b, compressor, indexer, gate, routed experts (local EP shard), shared experts, HC params, Engram host rows; final norm; head | planned T = 2..8192, TP4/DP4/EP16 | planned `python models/deepseek_v4_1_flash_a3_w8a8/prefill_fwd.py -p a2a3 --sim --tokens 128` -- OPEN |
| Decode | `decode_fwd.py` / `decode_fwd` (NOT DELIVERED) | planned: same weights; paged KV/compressor/index state InOut; position/length scalars | planned active tokens 1..32, TP4/DP4/EP16 | planned `python models/deepseek_v4_1_flash_a3_w8a8/decode_fwd.py -p a2a3 --sim --batch 1` -- OPEN |

Compilation coverage planned: embedding -> all 40 text layers -> final
RMSNorm -> BF16 head. The entries are compiled orchestration: one `@pl.jit`
entry each, internal tasks allowed; no host per-operator submission loop.
STATUS 2026-11-02: the entries are open. Only component `@pl.jit` entries
(quant, GEMM, RMSNorm, MLA projection, attention core, gate, experts) compile
and validate; the complete-model entry cannot be claimed until the remaining
modules in the ownership map are implemented. The structural tool and its
negative control are delivered and were exercised on the compiled component
entry; the complete-entry structural gate remains open.

## Execution structure

| Phase / module | Loop nesting and token tile | Effective operand shapes | State / communication behavior | Source and generated evidence |
| --- | --- | --- | --- | --- |
| Prefill projection / attention or recurrent mixer | outer layer loop (40) planned; implemented components: MLA projection uses a 16-row token tile; attention core scans its key set in 16-wide tiles with online softmax | implemented: GEMM `[16,K] x [K,N]`, attention `[16,128,512]` (8 key tiles); real key set assembled by phase files | component state does not yet append paged KV; phase files open | generated IR `build_output/_jit_sparse_attention_test_*/ptoas/attention_core.pto`; sim PASS |
| Prefill feed-forward / routed and shared experts | implemented: EP16 host driver over 16 ranks; dispatch stage, 24 expert blocks with 16-row tiles, combine stage | expert GEMM `[16,5120] x [5120,2304]`; receive slab `N_LOCAL x (16 * token tile)` | window-buffer put + notify/wait (epoch-set), per-(expert, source) lanes, route-slot return | 16-rank sim PASS; `build_output/_jit_prefill_moe_*`, `_jit_decode_moe_*` |
| Decode stack | open; component attention core supports a decode-sized query tile | planned GEMM `[A,5120]`; attention `[A,16,512]` | KV/compressor/index state InOut; position advance open | pending |

Multi-token prefill retains token tiles greater than one for projections and
feed-forward work; model-required recurrent scans (compressor/index selection)
may remain sequential. The structural check and its tokenwise negative control
live in `structural_check.py`; evidence is stored in `TASKS.md` and the
experiment directory (frozen logs), not in code.

## Open engineering row: per-mode file decomposition (user finding, 2026-11-03)

Required before closeout; numerical passes do not close it. The shared children
`attention.py` / `attention_gather.py` stay shared (rules/models.md: no
near-duplicates). To fix: (a) split `prefill_attention.py` (6 modes) and
`decode_attention.py` (4 modes) into thin per-mode files following the A5
per-mode idiom, importing the same shared core/gather symbols; (b) audit
`prefill_layer.py` / `decode_layer.py` (mode dispatch + host rank bodies +
goldens + staged references in one file) and move goldens/references and host
rank bodies to their owning modules, targeting ~1200L. Re-run affected bounded
entries on device and update these rows after the split. Sequencing recorded in
TASKS.md (after the current device milestone, as a dedicated worker unit with
owner verification).

Launcher verification addendum (2026-11-03, frozen revision 0a08b7ff): an
import-graph audit confirms a third irregularity beyond (a)/(b) —
`prefill_layer.py` and `decode_layer.py` import the attention pieces directly
(attention_core, assemble_key_set, c1a/c2a_compressor, lightning_indexer,
mla_proj, o_proj) and NO file in the model dir imports
`prefill_attention.py` or `decode_attention.py`. The phase attention files
(1139L/10jit and 976L/7jit) therefore have no composition consumer; the
mode-orchestration chain exists in two independent copies (operator symbols
shared, chain logic duplicated; the ~2400L layer files grew partly from
re-inlining it). Scope of the regularization unit grows to (c): eliminate the
duplicated composition paths — the split per-mode files should become the
single chain implementation consumed by BOTH the layer dispatch and the
bounded entries, so that no entry file lacks an actual consumer (rules:
"New files and entry points require an actual consumer").

## Stopping-point status (device validation, 2026-11-03 12:55; arbitration hold)

This supersedes the "sim" qualifiers in the tables above for device status.
No source edits were made under the hold. Cross-collided evidence remains
unattributed until a frozen revision exists.

| Row | Delivered status at the stopping point | Device evidence |
| --- | --- | --- |
| quant, gemm_w8a8, rmsnorm | implemented | PASS on one allocated card |
| embedding, lm_head, o_proj, mla_proj, attention | implemented | PASS on one card |
| compressor | implemented; C2A carried-state cross-core race fixed | PASS (C1A+C2A) |
| indexer | implemented; 16x1264 composed-shape run added | PASS (5 default cases + 16x1264) |
| attention_gather | implemented | PASS (5 cases) |
| hc | implemented; VEC UB-alignment defect fixed | PASS (pre/post/head) |
| engram | implemented | PASS |
| moe_gate | implemented; dual-AIV lane-1 TMov defect fixed | PASS |
| moe_experts | implemented; ND->DN TLOAD defect fixed; calibrated expert gate | PASS (routed+shared) |
| prefill_attention (6 modes) | implemented; `compressed_lens` cache-line race and reuse Call-argument codegen fixed; module-consistent + calibrated composed gates | 6/6 PASS twice consecutively on one card |
| decode_attention (4 modes) | implemented; same fixes | 4/4 PASS on one card |
| prefill_layer | implemented; **7/7 device-green on 16 cards** (delta 2 + host-driven MoE, three separately compiled programs) | device evidence task `task_20261104_140927_9088212593`, 2026-11-04 |
| decode_layer | implemented (7 bounded cases); `swa_layer0` device-clean; staged-reference attention cluster blocked by `attn_input` read-back zeros (buffer lifetime, diagnosed, fix deferred) | `swa_layer0` PASS on 16 cards; full suite mid-flight |
| prefill_fwd / decode_fwd | peer-authored drafts present; NOT absorbed or validated by this owner (hold) | none from this owner |
| generate.py | peer-authored draft present; NOT absorbed (hold) | none from this owner |
| structural_check (complete-model shapes) | open | n/a |
| EP16 MoE (prefill_moe/decode_moe) | simulator evidence only | device run open (hold) |
| per-mode file regularization (user finding) | open; sequencing recorded in TASKS.md | n/a |

### Device status update (2026-11-03 17:05, revision 0a08b7ff)

- prefill_layer: **7/7 bounded cases PASS on 16 real cards** (owner rerun).
- decode_layer: **7/7 bounded cases PASS on 16 real cards** (owner rerun).
- prefill_attention / decode_attention / all standalone operators: device PASS
  (previous rows).
- Complete entries (prefill_fwd/decode_fwd/generate): peer-authored drafts
  present and being absorbed under this owner; not yet device-validated.
- Structural gate: source+compiled complete-entry PASS at available T=2
  artifacts; T=128/full-stack artifacts to regenerate.
- EP16 MoE device evidence, generation/HBM, per-mode regularization, closeout:
  open.

## Approved regularization (user, 2026-11-03): per-mode split + layer trim

Supersedes the earlier "after the milestone" sequencing: the split runs before
complete-entry device validation/generation. Mode files (+ Torch goldens +
bounded entries) are thin chains over the unchanged shared `attention_core` /
key-set assembly; layer files import them and keep only sublayer sequencing,
state threading, window counters, MoE calls and TP reductions (<= ~1200L);
`prefill_attention.py`/`decode_attention.py` are deleted in the same change.
Compute symbols unchanged; device revalidation of every affected entry follows
(detail in TASKS.md).

### Post-regularization module map (2026-11-03, revision 1d31b6a0)

- `prefill_attention.py` / `decode_attention.py`: **deleted**. Attention modes
  now live in `prefill_c1a_full.py`, `prefill_c2a_full.py`, `prefill_swa.py`,
  `prefill_c1a_reuse.py`, `prefill_c2a_reuse.py`, `prefill_c1a_reindex.py`,
  `decode_c1a_full.py`, `decode_c2a_full.py`, `decode_swa.py`,
  `decode_reuse.py`; each owns its chain function(s), Torch golden and bounded
  `__main__` entry, over the unchanged shared `attention.py` /
  `attention_gather.py`.
- `layer_reference.py` (new): shared host staged-reference builders and
  calibrated comparator table used by both layer files.
- `prefill_layer.py` 1674 L / `decode_layer.py` 1692 L: sublayer sequencing,
  state threading, window counters, MoE calls; attention chains imported and
  dispatched by `mode` constexpr.
- Device status: all 10 mode entries PASS on one card; both layer suites 7/7
  PASS on 16 cards (owner reruns); structural source gate PASS with both
  negative controls rejected.

### Decode mode map after the under-split (2026-11-03, revision 0d6732ff)

- `decode_c2a_reuse.py` (475 L): owner of the single shared reuse chain
  `decode_attention_reuse` / `decode_reuse_layer_attention`, generic helpers
  (`_copy_matrix`, `_copy_vector`, `_rope_tables`, `_weights`), shared golden
  and specs, C2A bounded entry.
- `decode_c1a_reuse.py` (124 L): thin ratio-1 entry importing the shared chain.
- `decode_c1a_reindex.py` (595 L): distinct reindex chain + `_layer_index_latent`
  / `_layer_index_select` + its golden/entry.
- `decode_reuse.py`: deleted. `decode_layer.py` (1693 L) imports the two mode
  chain functions; MODE_C2A_REUSE / MODE_C1A_REUSE / reindex dispatch unchanged.

## Scope revision 2 (decision 9) — interface migration in progress

The migration contract is frozen in `INTERFACE.md` (binding for this unit):
A5-shaped `pl.Tensor[[shape], dtype]` signatures with `_DYN` axes, runtime
`num_tokens`, per-layer window caches + per-source compressed/index/state
caches, per-request `index_block_table`/`state_block_table`, `request_ids`,
`compressed_lens`/slots, `window_slots`/`indices`, port-level outputs, and
layer-level staged hooks. Compute symbols and numeric contracts stay unchanged;
device evidence through revision `0d6732ff` is semantics-valid but
interface-stale and is re-marked at closeout.

Recorded divergences (detail and justification in `INTERFACE.md` §1): BF16
activation caches in place of A5 FP8/MX caches (A3 quantization); replicated
32-head index query per TP rank instead of the sharded query plus
`tp_score_reduce` (removes the dynamic-width score exchange and the folded
score counter behind the prefill lane-poisoning protocol; attention TP and the
output reduce unchanged); chain-internal scratch outputs dropped. The selected
Top-K rows are physical cache rows, not logical compressed positions.

Migration sequence and write scopes (dedicated workers, disjoint files):
1. Foundation: `config.py` DYN symbols, `utils.py` host plan helpers,
   `paged_cache.py` publish/gather kernels, `attention.py` per-token packed
   core, `attention_gather.py`, `compressor.py`, `indexer.py` (paged scoring,
   replicated query heads).
2. Mode chains: the 10 prefill/decode mode files on the new signatures.
3. Layers + forward rank bodies: `prefill_layer.py`, `decode_layer.py`,
   `prefill_fwd.py`, `decode_fwd.py`; `moe_exchange.py` runtime token counts.
4. `structural_check.py` DYN compatibility; per-mode entries + both layer
   suites + structural gate re-run on device; then the complete-entry unit.

## Migration status (scope revision 2, 2026-11-03 23:10)

| Module | Status | Device evidence |
| --- | --- | --- |
| config.py DYN symbols + utils host plan | implemented | exercised by every paged fixture and entry |
| paged_cache.py (publish/gather/snapshot) | implemented | 3/3 exact cases PASS (logs/foundation_device_milestone.log) |
| attention.py per-token packed core | implemented | packed16/640/window128 PASS |
| compressor.py (C1A; C2A per-request state) | implemented | C1A + C2A PASS |
| indexer.py (paged, physical Top-K, replicated query heads) | implemented | tokens=8/16 PASS |
| moe_exchange.py runtime `capacity` + x-dim token count | implemented | prefill_moe/decode_moe 16-card PASS (logs/ep16_moe_runtime_r3.log) |
| prefill_moe.py / decode_moe.py wrappers | implemented | same EP16 run |
| prefill_c1a_full.py (chain template) | implemented | operator entry PASS at T=8/16 (logs/prefill_c1a_full_*.log) |
| weight_loader.py index replication | implemented | loader `--layers 0,20 --sample 2` PASS; idx_wq_b now 32 heads/rank |
| remaining 9 mode files | implemented, device PASS | done (owner_*_modes logs) |
| prefill_layer/decode_layer | validation-only entries; per-mode programs generated in prefill_layer.py, compile blocked by the pypto T_DYN loop typecheck issue (KNOWN_PYPTO_ISSUES.md) | blocked |
| fwd composition path | fwd files own direct mode-chain assembly (A5 shape); prefill_fwd restored pre-migration copy, rewrite pending | open |
| prefill_fwd/decode_fwd + generation | pending | pending |
| structural_check DYN compatibility | compiled dynamic-matmul-loop rule relaxed (evidence only; source call graph gates the tokenwise stack) | template check passed; re-verify with fwd + controls |

Backend pitfalls to keep in mind are listed in TASKS.md M1; the two most
expensive were the computed-tensor dynamic-row `pl.slice` mis-map (reshape to a
flat view first) and dynamic-extent slices of computed tensors (copy per row).

## Current Main coverage overlay — F04/F10 cohort (2026-11-06; independent acceptance pending)

F04 now derives absolute C2 source rows through shared host utils helper:
emitting p maps to p-1, not compressed address. Both source defaults, shared
layer fixtures, both forward fixtures and generator coordinate helper use it;
C1/SWA profiles and public ABI unchanged. Main CPU source-backed/actual-builder
checks, old-coordinate numerical rejection and actual even-to-odd state
handoff on one compiled source artifact pass. This is coordinate evidence only;
generator F11–F13 and full-model/dynamic lifecycle remain open.

F10 shared moe_exchange receive lanes use capacity at push/compact, preserving
existing capacity allocations/return staging. A bounded artifact harness calls
the actual primitive on16devices with capacity2/equal1/unequal/idle counts;
old numerical control and repaired positives plus same-artifact replay/exact
inactive sentinels pass in Main-owned worker evidence. These cases are artifact
harness coverage, not new default full-forward coverage; F09 remains OPEN.

Main downstream defaults:17 executables/37 numerical positives including all12
phase entries, plus3 coordinate positives/1 coordinate negative. Worker6 EP16
positives repeat3designs plus1 old numerical control. Both complete-forward
compiles pass; no full-model runtime parity implied. Source/evidence and fresh
rerun commands: logs/cohort_f04_f10_1793952989/REPORT.md. Main and worker freeze
for observer-owned independent sibling acceptance; this overlay claims no
independent acceptance or closure of other OPEN/HELD rows.

## Current Main coverage overlay — F05/F06/F07 frozen self-evidence

Supersedes the older F04/F10 pending overlay above. Source/default coverage:
Engram five permanent nonidentity/original/cast/zero-gate cases, reused artifact;
indexer nine permanent cases and six phase entries cover actual2048 publication,
65/2048/2049/4097 boundary counts/IDs/padding and protected source-pool selection.
Both layer default suites add c2a_full_layer14_engram; layer1 remains covered.
Main ran45 component/phase defaults, four original-source first-cache prologs
(prefill/decode,SWA/C2),two numerical post-attention placement controls,and two
actual16-device staged complete layers (prefillSWA/decodeC2) with real rotation
map and synthetic weights. Both complete forward entries compile. Actual loader
and generator rank carrier preserve global resource [16,32,32],all real pool
argument widths2048. No full checkpoint weights/Engram tables were read.

Main51 positive executions/two order controls; Engram worker seven positives/six
basis/cast/value controls; cap worker18 positives/one old64 control. Counts are
execution evidence,separately attributed,not independent acceptance. Source
manifest45 files/16 changed/29 unchanged and fresh commands are in
logs/cohort_engram_f07_1793957136/REPORT.md and after.json. Main/workers PARKED;
F09/F11–F13,full-model state/pool/device,ordinary layer suites,TP/CI and generation
remain OPEN/HELD pending their own evidence. Original primitive/quant-tie parity
limitation persists. Observer-owned sibling reviewer must accept this cohort.

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

## Main noEngram driver/full-entry checkpoint — blocked, not accepted

logs/generation_noengram_1793970065/REPORT.md supersedes current implementation
progress. Runtime scalar engram_enabled follows rotation in forward ABI;
existing owning defaults1,first driver0. True rank/layer/24localexpert axes are
restored in both forwards. Main concrete ABI/dualRoPE(noSWA-YaRN)/decode1/
persistent-onlyhandoff/allperstepmetadata and host missing-profile controls pass.
Permanent generate --contract-report checksdecode1/positions126–129/pageparity.
Actual40dense-layer prefill2 numericalPASS uses1syntheticexpertlayer reused;
real40expertlayer ABI compilesonly. Native40-layer decode1 blocked FIRST by
halMemCtlrc13/RunPreparationRefusedcode13;afterward507901/507018 isnotnumerical
failureproof. Three-step actualstate/pageboundary continuationNOTRUN. Header
budgetonly,real peak/residency/transientsUNKNOWN;no realweights/text generated.
Main+workerPARK for independent scoped review/environment disposition. General
F09/F05/F07/F13/fullmodel/realcheckpoint/TP/CI/generation gatesremainOPEN/HELD.

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

## Main implementation overlay 1794018541 — scalar/C2/streaming, independent acceptance pending

Only generate.py, weight_loader.py and compressor.py changed in logs/scalar_c2_streaming_1794014676;42other model sources unchanged. generate scalar marshalling supportsPythonFP32/INT32/BOOL;publicstreamowner seriallyloads exactrankleaves andresolves100prefill/96decodeargumentsfrommetadata,57static each,compilesreal40expertswithzerotensorsamples. Actualdriver bulkaggregatecallseliminated;hardfullweightguardremains. weight_loader preservesNZ/transpose/offsetfold/FP32scales,pre-readtop-levelrejection,andactualC1phaseaxiscontracts. C2goldenonlyFP64evaluation correction haspermanentanalytic even/odd/remappedstatecontrol;devicekernel/thresholdunchanged.

Mainself native16decodeALL24gatesPASS,exactscalar3callandstreamnative2rankreadback/threefloat-int-boolcallsPASS. Consumerleafrelease/sourcecallbackCPUcontrolPASS. Theseareboundedselfevidence,notindependentacceptance. First128prefixpreparedprefill faileddefault256MiBring2;single2GiBretry appliedbutfailedstrictwrap/open-scopehead35. No returnedprefill/three-decode/stalepublishedrowcontrolpassed. No thirdretry. Realseparatefit/fullweights/text,strictunconditionedparity/generalF09/fullEngramandhistoricalF05/F07hardwarecontrolsremainOPEN. SeeREPORT.md/after.json/FREEZE.json.

## Main dated evidence corrections 1794019935

Read-onlylogs/real_fit_readiness_1794018846/READINESS.md andfinal_checks.json supersedepriorMainheadlineswithoutrewritingfrozenrawreports. Actualembed/headsourceis1323827200B each,not5295308800B(16ranklogical/fourDPcopies);raw+bytearraycomponent2647654400B,retainedTPsource1323827200Bvisible330956800B. Originalmax(4S,24Q)+64MiB=5362417664Bserialplanningallowancecanremain;myreasonforcallingitincompletewasfalse. Actualsourceexports19decodeoutputs,not24;fivecachepublishedrowsubchecksareconditionalandnotextraexports. Independent19PASSiscurrentfact. PublicPyPTORunConfigpathDOESrequirepower-of-twoheapentries>=1024;lowerCallConfigpermissivenessdoesnotremoveit. Reviewer3GiBjobfailedbeforecompile/prepare/weights/native;readinessCPU105016independentlyconfirms3GiBrejectionand256MiB/1/2/4GiBacceptance. No bypass. Supported4GiBring2costs4.75GiB/bank/rank;actualconstructedbankcountremainsunknown. No newmodel/NPU/fullpayloadacceptancefromthisreadiness.

## Main production host overlay 1794036453 — applied, CPU checked; independent acceptance pending

Latest implementation state supersedes older Main pending-native paragraphs above.
Only generate.py changed from original45ab1 to source45manifest762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7.
The other44 forward/operator/golden/loader files are byte-identical. New host
source685cf3cf...29cfb preserves original prompt IDs and uses exactN2..128,
lastvalid rowN-1/start+N/cache handoff, strict requested4DP greedy agreement
with structured prefill and every-decode votes/scores/context. Public config
sets targeted4GiB ring2 and normalized ONBOARD; actual unchanged PreparedDispatch
passes the same normalized object to prepare/every call. NoEngram is explicit;
hard fullweight guard and owner/stream/scalar algorithms remain unchanged.

Fresh applied-source CPU152005 passes prompt/DP/TP/EOS/boundary/config/real40
metadata/ruff checks (backend mocked, no native numerics). CPU152352 verifies
current-source rebind of the existing empty-fit helper,57header resources and
default/non4GiB refusal; it does not allocate. Recovery and exact commands:
logs/generation_entry_1794035858/REPORT.md, generate.patch,
INDEPENDENT_ACCEPTANCE_PLAN.md and FIT_RAMP_PLAN.md. Minimal next native proposal
is exactN3 prefix17 then actual partial-C2-state handoff, after independent GO.

Independent OLDab1 source acceptance is now prefix128 original17 plus actual
returned-state128/129/130 original19 each and correct129 WINDOW/C2 numerical
rejection: logs/native_decode_capture_v3_1794031922/REPORT.md. It is bounded
40dense/reused1syntheticexpert/noEngram/staged scope, not realweights/text or
strict whole unconditioned history. New host source itself has CPU evidence only.
Current mutable metadata is prefix16/decode12 (old Main18 was a count typo).
Real fit/full checkpoint upload/text/generalF09 and historical gates remain open.
