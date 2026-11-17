# Configurable A3 TP/EP — implementation and CPU/signature milestone

Current applied source45 v3: f0e91a30e4117e110be183ef44acf2874eea813cc6f72ac7c7fa44bc4ede3a7f.
The exact predecessorB45 is retained in baseline/**. New user scope permits A3
configuration changes; defaults TP4/EP16 remain. A5 files, W8A8 quantization,
checkpoint, dependencies, tests/contract and all old sealed bundles are untouched.

Both --tp N/--tp=N and --ep N/--ep=N are selected at import time before model
modules. Valid domain TP1/2/4/8 and EP2/4/8/16 requires EP>=TP and EP%TP==0:
13 configurations, WORLD=EP and DP=EP//TP, contiguous TP groups. Malformed,
conflicting, unsupported and nondivisible flags reject early. Every affected
numerical entry accepts the common CLI options; generate submit propagates both
flags and allocates WORLD devices. Programmatic main(argv) cannot silently change
an already-imported topology. A5 fullentry TP1 restriction is not borrowed as A3
capability; A5 is syntax/reference only.

Source integration covers rank maps, TP reduction, whole-expert EP routing,
caller/golden names, layer/full-entry spec factories, local MLA/output-projection
head/group extents, CLI/deployment/device IDs, greedy DP votes, shared resident
statics/C1 views and real-operation admission. Hardware tiles16/32,32indexheads
replicated, sharedFFN replicated, learned dims/layer IDs and compressed-source
families stay fixed. No blanket16→EP replacement was used.

Two smallEP defects repaired: per-expert compute slab stride rounds EP*capacity
UP to16 while network per-source lanes keep actual capacity; COUNT_PAD rounds
local experts UP to32 (EP2/4/8/16 gives192/96/64/32). Compact/expert/return and
all imported window/spec shapes share these values. Zero-active ranks still
serve peers; typed completion and explicit dependencies are preserved. Functions
moe_ep/golden_moe_ep replace misleading EP16 names at ALL model callers.

Golden MoE now honors optional destination-rank weight stacks, instead of the
old rank0 homogeneous-fixture shortcut; shared weights use the source rank.
Original _expert_golden arithmetic/tolerances remain unchanged. A tiny actual
original-expert mathematical oracle with different destination weights, high
local experts and idle carry passed, and an all-rank0-weight mutant rejects.
No unchanged-source claim is made for these host-reference ownership changes.

Meaningful self-evidence (all queued --no-device, no actual checkpoint reads):
- Worker112536 COMPLETE0:220 config/geometry/analytical route controls/all13pairs.
  Worker112737 profiles countCAP1+TP at TP1EP2 andTP8EP16 compile; TP-only final
  test annotations InOut corrected/recompiled112905, oldattempt artifacts kept.
- Loader112658 and112819 COMPLETE0:actual loader fake-source all13/allrank,
  independent NZ layout/signed offset fold/FP32 scales/vocab/head/outputgroups/
  whole-expert mapping/leaf offsets/lifetimes/pre-read invalidcontract rejection.
- Main114340 exit1 reached all13 ABI/DP checks, then OWNnegative fixture used
  rank%5 so rank15 collided withrank0. The exact v1 script/log are retained.
  Directed120237 COMPLETE0 fixed ONLY the test coefficients; actual golden and
  original-expert owner oracle agree, wrongrank0 mutant rejects, idle carry stays,
  actual configurable admission CPU-fake/defaultHARD/source/topology/resource/
  duplicateIDs/legacy-nondefault controls pass. No positive actual grant created.
- Metadata114844 COMPLETE0:actual trueleaf schedules on META using ONLY accepted
  oldcd8b descriptors, no new headers/index/tokenizer/payload. Both TP8EP16 and
  default4EP16 trace57resources/912targets/110368leaves perphase,164192 source
  reads745846365184B. TP8 static38,328,513,216B/rank/global613,256,211,456;
  default39,624,816,576/global633,997,065,216. Source reads do NOT halve withTP8.

Full signature attempts retained:120530 failed own HEADS8/Cube16 score-mask
mismatch. Explicit HEAD_TILE=max16,HEADS padded q/sink/state andHEADS-only store
fixed physical geometry; original attention golden/tolerances unchanged.
121110 then correctly rejected OWN sink fillpad placed outside the tokenSPMD.
It moved inside that in-core scope; no upstream/compiler defect is alleged.
Other local CubeM extents were checked: grouped o_proj uses padded16 token rows,
localgroup1 changes N loops only, indexer remains32heads. Sourcev1/v2/failed logs
and codegen reports remain; metadata v3 clones only two source fields and verifies
ALL other keys/targets/read/resource budgets unchanged, without another164k trace.

Fresh normal E40 TP8EP16 pair121454_134561714986 COMPLETE0: prefillN3 45.9758s,
100formals; decode1 43.4603s,96formals; each57statics/E40 and38,328,513,216B/rank,
a2a3/ONBOARD/ring[0,0,4GiB,0], ZERO samples/reader/prepare/NPU. Receipt
c11948187a3f2e3675d5fe9e0fd0a47f20af67ecbeac4866f947cb7aefca75e4.
These are new artifacts, not old611/M02/sourceB or E1 numerical proof relabels.

New real-operation schema configurable-real-generation-authorization/1 and
operation configurable-shared-real40-noEngram-greedy-N3 require exact topology,
NEW metadata/source/budget/Root fields, and ALL45 current file equality (no
self-hash literal or generate exception). Legacy branch is unchanged/default-only;
HARD remains without an exact positive actual Root receipt. FullE40 static>=64GiB
rejects before reading checkpoint tensors. Smaller EP is still supported for
component/layer validation, not falsely rejected as a topology solely for unfit
full weights. EP8 routed-only68,022,435,840B and fullTP8EP8 static72,339,731,136
cannot fit the current card. TP8EP16 is the first nondefault real candidate.

Exact TP8 planning: HOST64GiB/DISK2GiB/SHM16GiB floors, per-rank initial reserve
18,088,299,200B; unspent9,797,894,144B; full-global IO perrank359,958,016 prefill/
246,077,440 decode; full-global window allowance2,876,676,096+74,618,880 yields
communication reserve2,951,294,976B. These are conservative SOURCE bounds, NOT
measured peaks. Largest static11,324,620,800B unchanged; full-source hostbacking
planning5,362,417,664B unchanged. Fresh publicfree subtracts no committed/ring twice.

Ownership: both scoped workers released product files; all model sources Main
owned and frozen now. Three unique worker-agent records imported once, concurrent
intervals not summed as elapsed. No independent acceptance claimed. Default
numerical behavior remains an intended compatibility requirement, not a fresh
full-default native replay. Root-owned smallEP/TP/HEADS8 native subsets and
independent configured resource/source/pair/PREQUEUE review remain pending.

New sibling logs/configurable_real_operation_prepare_1794111133 will bind this
exact freeze, plan and pair for Once upon a/[16600,4465,260], max16,strictDP2,
noEngram/sharedowner ONE613256211456B upload and decode0newstatic. It remains
closed pending separate Root actual16 GO. Previous realSourceB wholeFAIL218 and
M02FAIL183 stay immutable; no telemetry policy change. Actual configured real
text, quality, original quant/numerical/history parity, complete PSS/peak/general
serving and Engram restoration are NOT claimed at this milestone.
