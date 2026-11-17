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

__FINAL_AUDIT_AND_ROOT_DISPOSITION__

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
final metadata seal is __OWNER_FINAL_SEAL_STATUS__; no complete process history,
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

