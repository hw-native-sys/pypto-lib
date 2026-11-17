# HANDOFF — DSV4.1-Flash A3 W8A8 experiment (2026-11-06)

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

Main PARK after exactCCapplication/ONEtinyCPU; nextRoot-owned outer adaptation
must use NEWcode/source/receipt for Onceupona[16600,4465,260]/max16. SeparateRoot
actual16GO remains required; do not run fullweights now. Oldactual firstcase[3,1]
accepted onlyoneDecode; do not infer15calls/longtext from max4request or EOS.
Old611 stays D77compiledsource. Its unchanged44/compilemethod/sameN3/pages2 relation
toCChost is explicit in inheritance.json, no duplicatecompile/no relabel.
Futureclient>=22200/queue21600/internal20400, noresubmit; samepositiveplanning
floors/strict4DP/actualgreedy/EOS/state/durablecoverage/finalmemory/cleanup evidence.
Current memory observations are scoped; longer15-call/semantic/reproducibletext,
fullEngram/originalparity/generalF09 and engineeringcloseout remain separatelyopen.

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

Main source writer is PARKED after exactapplication and small queuedCPU PASS.
Next: independent reviewer compiles actual new-source E40 N3 prefill/decode1
signature-only entries; Root then separately decides FIRST fullpayload/forward.
Do not duplicate those compiles, reuse an old grant, or run fullweights now.
Actual future queue needs explicit client --timeout>=22200 (or submit+boundedwait),
queue21600/internal20400, no resubmission on client timeout. The correction is in
NEW application docs; old sealed execution plans remain unchanged. The planned
first realtext request uses genuine previous greedy tokens, not synthetic16684
or fixed17. Earlier syntheticN3 and selected18realtransfer acceptances retain
their exact bounded scope and do not imply full40 initialization or text.

**Current coordination:** the user authorized native subagents on 2026-11-06.
Read [COORDINATION.md](COORDINATION.md) and the newest TASKS directives first.
They supersede historical CLI-only/no-message-bus/no-subagent launch text below.
TASKS/TIME_LOG and bounded milestones remain required. Latest user instruction
delegates future acceptance execution to independent subagents; observer
coordinates and records decisions rather than personally running tests.

**Current bounded milestone (2026-11-06):** generic indexer/reindex component,
F01 MLA, F02 inverse-output-RoPE, F03 supplied-table compressed-KV rotation,
F08 rank-local tie routing and conditional synthetic F14 folding are independently
accepted at their recorded scope. Latest acceptance:
`logs/native_accept_cohort_1793951619/REVIEW.md`; F14 current-checkpoint incidence
is unproved. No complete-model/distributed/original-primitive/generation parity
follows from these checks.
F04/F10 are independently accepted at their bounded scope in
`logs/native_accept_f04_f10_1793955766/REVIEW.md`.
Current frozen cohort: `logs/cohort_engram_f07_1793957136/REPORT.md`.
F05 restores original Engram gate basis using loaded shared32-block rotation;
F06 places Engram before HC pre/attention; F07 restores actual2048 block cap.
Main owns seven integration/loader files; workers engram_basis and candidate_cap
owned disjoint Engram/eight cap files and are now PARKED. Main self-validation:
51positive executions/two numerical order controls,both forward compiles;
worker evidence separately seven/six and18/one positive/negative executions.
Frozen45-source manifest SHA256:
`7c5b8df2d57ca06a49106b528c2a018d36cd1524f4df8418e59ffa64b7d24d7f`.
Independent acceptance is pending; no further writers/jobs are authorized on
this manifest during review. Fresh rerun scripts and exact ownership/failures
are in REPORT/TASKS. Read TASKS/COORDINATION_STATE for reviewer assignment.
All remaining open/held gates persist; older green/pending-routing prose below
is historical and superseded.

**Current routing update (2026-11-06):** the user authorized diagnosis and
repair ("让agent去定位修复吧"). The latest Observer directives in TASKS.md
supersede the historical pending i/ii/iii decision below. Do not ask for that
authorization again. The first generic-indexer repair milestone is device
validated; see `logs/indexer_repair_1793929524/REPORT.md` and the latest main
checkpoint. Its shared scoring change requires fresh complete-model checks;
the full-model green evidence below is historical. Composed decode pool/state
root causes remain unproved, and the historical runtime attribution is a
suspicion. C1A reindex source-pool consumption now has main-agent device evidence
in `logs/reindex_repair_1793930492/REPORT.md` (15 component cases plus both C1A
reindex layers on 16 devices). Component acceptance is recorded in TASKS;
those16-device runs remain Main evidence, not an independent reviewer rerun.
Both full forwards compile with corrected source-pool bindings; fresh full-model
device validation remains OPEN. No external filing or commit is authorized;
generation remains HELD.

**Newly relayed review priority:** the independent frozen-snapshot logic audit
is complete; read the newest Observer directives and
`logs/logic_review_1793929657/REVIEW.md`. Current repairs are F04 and disjoint F10;
next priorities include Engram and the recorded lifecycle/generation
findings. Current C1A consumption evidence does not restore the remaining F07
64-versus-checkpoint-2048 real-path pool cap. Do not call F07 fully closed or
reimplement the consumption repair. All semantic/full-model gates stay open.

For the successor observer (any host: Codex, Claude, human). The previous
observer was a Claude Code session (2026-11-02 → 11-06, ledger ~68 h).

## What this is

Model-harness experiment implementing the DeepSeek-V4.1-Flash text backbone
for Ascend A3 (a2a3) with the checkpoint's W8A8 quantization, in
pypto-lib, under the model-harness workflow (experiment dir = state carrier).
In Codex you are BOTH observer and Main Agent: pypto-lib's AGENTS.md directs
Codex to execute workflows directly, treating agent instructions as
checklists. The harness's verify-before-claim discipline IS the observer
role — run the checks yourself, record evidence, no fake closes.

## Environment (final, verified — see ENVIRONMENT.md rev 3)

- pypto 38a8aa6 + simpler 31f9e72 + PTOAS 0.72 (cp310 wheel) + PTO ISA f83d4b7
  + pypto-lib b6d17d9f; build inputs pinned per `code/pypto/build-constraints.txt`
  (nanobind==2.15.0 / cmake==4.4.2 / ninja==1.13.0 / scikit-build-core==1.0.3).
  NEVER install build deps unversioned (nanobind 3.1.0 once cost days).
- `source /mnt/data/pyptouser/yangyaodong/code/pypto-lib/activate-pypto.sh`
  before all Python work; `source ~/proxy.sh` before network ops; bare
  `ptoas` in PATH is a stale 0.67 — always `$PTOAS_ROOT/bin/ptoas`.
- Every NPU run via `task-submit` (`--device auto --device-num N --max-time ...`;
  client-side waits need long budgets — background them). Device-only
  validation (goal.md decision 8). If a chip fails `simpler_init -1000`,
  check /dev/shm for leaked `sp-*`/`smp_*` segments from faulted runs before
  asking for admin resets (recovery procedure in ENVIRONMENT.md).

## Current state (canonical model hash 7f813695...)

- GREEN: prefill complete entry, ALL 17 gates incl. state_cache_c2a and both
  pool families (81.34 s, 16 cards, re-validated under the per-profile rope
  ABI); decode complete entry green on everything EXCEPT two gates; both
  structural gates + tokenwise/renamed negative controls; thin entries;
  generation driver dry-run with ZERO contract gaps (rope per-profile,
  layer-axis experts via the uniform-ABI driver-process variant,
  resident="stacked", tile-fitted prompts, engram zero-diagnostic default
  with --engram-mode host-ids promotion).
- OPEN ROWS (visible in TASKS.md, no fake closes):
  1. decode `state_cache_c2a` (one capacity slot zero) + decode
     `pool_block_ids/counts` (device 0 vs golden 64). Same kernels/bindings/
     assembly PASS in prefill's composed program (the strongest contrast).
     Eight model-side variants changed nothing; attribution is runtime-side,
     decode-program-specific. Filed in KNOWN_PYPTO_ISSUES.md (446+ lines)
     with the elimination matrix, the prefill-vs-decode contrast, two precise
     runtime questions (non-spmd AIV task with runtime-derived buffer view;
     version/owner tracking for views derived from program parameters vs
     task-produced buffers), and the explicit generation dependency.
     TP is largely exonerated (attention TP reduce green in decode; prefill
     shares TP4/EP16-windows/two-window/split-dispatch and passes).
  2. TP distinguishable-inputs gate (tp_reduce_test): implemented but
     standalone 16-rank programs fault (507018 / finalize -100); composed
     programs run. Filed.
  3. Layer suites revalidation queued behind 16-card availability.
  4. Generation acceptance HELD on rows 1 (and the queued revalidations).

## THE PENDING USER DECISION — your first action

The user must pick the routing for open row 1 (decode two gates):
  (i) file to the pypto team (the KNOWN_PYPTO_ISSUES entry is the work
      order — prefill-green/decode-red contrast + two precise runtime
      questions);
  (ii) authorize a model-side decode-feature bisection (strip candidates:
      two-block state table / two-window TP / EP16 windows / split dispatch —
      the prefill contrast already weakens the last three; the two-block
      state-table geometry is the lead suspect);
  (iii) both in parallel.
Present these, get the word, execute. Everything else is staged: when the
two gates close, run the queued layer suites, then the generation unit
(generate.py driver, tile-matched prompt, engram policy per the recorded
preconditions), real-weight greedy generation + peak HBM on 16 cards with
the full evidence record, EP16 evidence, TASKS closeout.

## Where everything lives

- State carrier: `pypto-lib/experiments/dsv41-flash-w8a8-a3/` — read
  TASKS.md (evidence, contracts, open rows), goal.md (user decisions 1-9),
  ENVIRONMENT.md rev 3, TIME_LOG.md (349+ rows — KEEP LOGGING: one row per
  work block with real `date +%s` epochs, categories in its header), plus
  MODULES/MODEL_SPEC/ACCEPTANCE/WEIGHT_MANIFEST/INTERFACE/SERVING_ABI and
  weight_map.json, structural_controls/, logs/.
- Model code: `pypto-lib/models/deepseek_v4_1_flash_a3_w8a8/` (untracked —
  never commit; the user handles commits).
- Runtime filing: `pypto-lib/KNOWN_PYPTO_ISSUES.md` + probe files.
- Harness (prompts/rules that govern the workflow):
  `model-harness/prompts/PROMPT.md`, `rules/models.md`,
  `prompts/WORKER_PROMPT.md` (use as checklists), `templates/*`.

## Discipline (the observer checklist)

- Verify before claiming: rerun acceptance, inspect artifacts — self-reports
  and green-adjacent looks are not evidence (the task's history: three
  wrong root-causes were caught by controls, staging, and counter-evidence;
  two were retracted honestly).
- Every work block: TIME_LOG row. Every milestone: TASKS.md update.
- Device-only validation via task-submit; no commits; no writes to the
  pypto checkout or the checkpoint; no fake closes of open rows.
- The user's rulings live in goal.md decisions 1-9 and TIME_LOG — check them
  before re-deciding anything (e.g., decode_fwd is A5-literal unrolled by
  explicit user order; EP/TP adjustability and strict-dynamic MoE remain
  agreed-but-unimplemented candidates pending the user's timing).

## Split-role launch (user-chosen structure, 2026-11-06)

The user runs TWO Codex instances: one OBSERVER, one MAIN AGENT. There is no
inter-instance message bus — the state carrier IS the coordination medium.

Coordinator contract (both roles follow):
- The MAIN AGENT works in coherent units, records every block in TIME_LOG.md
  (agent tag `main-agent`), updates TASKS.md at every milestone, and STOPS at
  coherent milestones (never mid-unit) with its resumption state written.
- The OBSERVER (tag `observer`) reads TASKS.md/TIME_LOG.md at each
  milestone, independently verifies claims by rerunning acceptance and
  inspecting artifacts, appends its findings/directives to TASKS.md under an
  `## Observer directives` heading (dated), and presents decisions to the
  user. The observer does not implement; the main agent does not self-accept
  on the observer's behalf — verification is the observer's separate eyes.
- The user relays between terminals only when a synchronous answer is needed;
  everything else flows through the files. No fake closes, no unverified
  claims — the same discipline as the Claude-era workflow, which caught three
  wrong root-causes via exactly this split.

Main-agent brief (paste into the second Codex instance, in pypto-lib root):

  你是 model-harness 实验 dsv41-flash-w8a8-a3 的 Main Agent(实现与验收的
  拥有者)。先读:本仓 AGENTS.md → .claude/CLAUDE.md → .claude/rules/ →
  model-harness/prompts/PROMPT.md + rules/models.md + prompts/WORKER_PROMPT.md
  (当 checklist 直接执行)→ experiments/dsv41-flash-w8a8-a3/ 的 HANDOFF.md、
  TASKS.md、goal.md、ENVIRONMENT.md rev 3、TIME_LOG.md 尾部。
  你独立拥有实现与验收;观察员在 TASKS.md 的 Observer directives 节给你
  指令与核验结果,你读取并执行;用户决策经观察员转达。每个工作块在
  TIME_LOG 记一行(真实时间戳,tag main-agent);每个里程碑更新 TASKS;
  在内聚里程碑停机并写恢复态;先验证后声明;开放行不假关闭;不 commit;
  pypto checkout 与 checkpoint 不许改;一切 NPU 经 task-submit。你的
  第一个动作:读 TASKS.md 的 Observer directives 与 goal.md 决策 1–9,
  向观察员确认 pending 的路由决策状态,然后继续记录在案的欠账序列。

Observer brief (paste into the first Codex instance, in pypto-lib root):

  你是同一实验的观察员(不实现)。读同上状态载体。你的职责:在 main
  agent 的每个里程碑后独立核验其声明(亲跑验收命令、查盘上证据)、把
  发现与指令写进 TASKS.md 的 `## Observer directives`(带日期)、在
  TIME_LOG 记 observer 行、把需要用户拍板的事(当前 pending:decode 两门
  的运行时路由 i/ii/iii)呈给用户并转达其选择。核验不过的声明要如实
  打回;不放过假关闭。

## Observer-managed main agent (hierarchy the user chose, 2026-11-06)

The OBSERVER session is the only one the user starts manually. The observer
LAUNCHES the main agent itself, headlessly, per unit:

    nohup codex exec "$(cat experiments/dsv41-flash-w8a8-a3/MAIN_AGENT_BRIEF.md)" \
      > experiments/dsv41-flash-w8a8-a3/logs/main_agent_unit_$(date +%s).log 2>&1 &

- The main agent is a BOUNDED UNIT: it works to the next coherent milestone,
  parks, and writes its resumption state into TASKS.md. The state carrier is
  its full working memory — the observer launches a fresh `codex exec` for the
  next unit; no session continuity is needed (this is the same pattern that
  survived seven agent generations on the Claude side).
- The observer: launches units, tails the unit log + TASKS/TIME_LOG at
  milestones, verifies claims independently (rerun acceptance, inspect
  artifacts), appends `## Observer directives` (dated) for the next unit,
  relays user decisions, and TIME_LOGs its own rows (tag `observer`).
- If the host supports `codex exec resume` for follow-ups, the observer may
  use it; per-unit fresh launches are the portable default.

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

## Latest Main recovery — generation_noengram_1793970065

Implemented first-path F11/F12 driver/scalarEngramdisabled and correctedlocal
expertaxis through actualfullforwardABI. MainhostRoPE/completebinding/metadata/
header-onlybudgetchecks pass;bothreal40expertlayer ABI entriescompileONBOARD.
Actual40-layerprefill2numericalpasses(reused1syntheticexpertlayer);decode1
FIRST halMemCtlrc13/RunPreparationRefusedcode13blocks numerical/continuation.
No text/realweights/peakHBMproof. AllownNPUjobsended/locksreleased. Readunit
REPORT/after/artifacts and newestTASKS;do notlaunchpreparedcontinuationuntil
nativepreparationrecovers andhelperAPIisreviewed. MainandworkerPARKED for
independent review/disposition;remaininggatesOPEN/HELD,firstpathlabelnoEngram.

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

## Main bounded cohort recovery 1794018541 — implementation self-evidence, continuation BLOCKED

Read logs/scalar_c2_streaming_1794014676/REPORT.md,after.json,commands.tsv,FREEZE.json andmain/continuation/failure_analysis.json. MainandworkersPARKafterfinalcollector;rootownsindependentacceptance/coordinationdecision. Threeownedsourcechanges:generate.py/weight_loader.py/compressor.py;42unchanged. Uninstrumentednative16decodeALL24PASS andtinyexactscalar/streamowner3callnativePASS;no threshold/kernelchange. C2referencefixgroundedoriginalAST,analytic/nonconstantcontrols,48bitwiseactualstageinputs andindependentexactrationalprereview;originaltorch_npuprimitiveparityunavailable.

128prefixfirstpreparedprefill faileddefaultheap;boundedpublic2GiBring2retryfailedstrictcontiguouswrap/open-scopehead35after9538completedtasks(rank0snapshot). Bothownfailedjobsqueue-terminatedexit130;no decode/returnedstate/stalecachecontrolresult. DoNOTlaunchthirdresize/wholechainorfullweightpayload:observerassignedfrozenindependentsource/allocator/ABIboundedresumptionnext. Preserveallrawlogs/failedhelpers/workerreports/manifests. BroadF09dynamic/fullrealfit/generation/coherenttext/originalparity remainopen.

## Main dated evidence corrections 1794019935

Read-onlylogs/real_fit_readiness_1794018846/READINESS.md andfinal_checks.json supersedepriorMainheadlineswithoutrewritingfrozenrawreports. Actualembed/headsourceis1323827200B each,not5295308800B(16ranklogical/fourDPcopies);raw+bytearraycomponent2647654400B,retainedTPsource1323827200Bvisible330956800B. Originalmax(4S,24Q)+64MiB=5362417664Bserialplanningallowancecanremain;myreasonforcallingitincompletewasfalse. Actualsourceexports19decodeoutputs,not24;fivecachepublishedrowsubchecksareconditionalandnotextraexports. Independent19PASSiscurrentfact. PublicPyPTORunConfigpathDOESrequirepower-of-twoheapentries>=1024;lowerCallConfigpermissivenessdoesnotremoveit. Reviewer3GiBjobfailedbeforecompile/prepare/weights/native;readinessCPU105016independentlyconfirms3GiBrejectionand256MiB/1/2/4GiBacceptance. No bypass. Supported4GiBring2costs4.75GiB/bank/rank;actualconstructedbankcountremainsunknown. No newmodel/NPU/fullpayloadacceptancefromthisreadiness.

## Latest Main recovery 1794036453 — generation-entry applied/CPU checked

Main's current model45 manifest is762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7;
only generate.py changed from oldab1, other44 files unchanged. The reviewed
isolated prompt/DP patch is now APPLIED, plus public4GiB ring2 config. First
path remains explicitly noEngram and full realweights remain HARD HELD.
Fresh applied-source CPU152005 and current-source fit-plan CPU152352 pass;
no NPU, fullfixture, weight payload, fullfit or text job in this unit.

Unit recovery: logs/generation_entry_1794035858/REPORT.md, after.json,
FREEZE.json/artifacts.json, onefilegenerate.patch and fresh CPU logs. New exactN
and DP behavior needs independent reviewer execution; do not reuse old draft
PASS as applied-source acceptance. INDEPENDENT_ACCEPTANCE_PLAN.md describes
minimal actualN3 prefill17, durable actual bytes before offline bothrefs, and
actualstate partial-C2 handoff. Canonical V3 hardN128/oldab1 binding is not a
universal adapter; record new producer vs44identical mathematical-reference
source honestly. FIT_RAMP_PLAN.md gives concrete current-source public empty
largest11.325GB->CLOSE/freshowner->57resources39.625GB/rank route and STOPs.
Defaulthelperrefusal and hardfullweights guard remain; no allocation GO here.

Prior independent native128prefix17 and actual128/129/130 original19each +
correct129 numerical WINDOW/C2 stale reject are SEALED in
logs/native_decode_capture_v3_1794031922/REPORT.md. This accepts bounded synthetic
E1/noEngram/plannedtokens/staged scope, not actual40expert checkpoint, strict
whole unconditioned equivalence, realweightfit or coherent greedy text.
Prefix mutable names16/decode12; earlier Main18 shorthand was documentation
error, oldraw reports preserved. Observer owns next independent scope/GO.
