# Finite F09 active-prefix experiment — PREPARATION ONLY

No actual grant exists. This directory authorizes neither NPU nor fixture creation.
Source405 is an UNAPPLIED candidate copied from M02 plus bounded F09 repairs. Live
M02 e07b28713eadd00a306fd60264223dbf6ceec53761c7ac48c347032011c9ca25 stays unchanged.
It does not pass the production real-generation strict44 gate; do not alter that gate.

## Prerequisites and exact lineage

1. Independent typed207 three signatures are source/compile evidence only:
   independent_f09_typed_completion_1794079859, source405, manifests 4c226662…
   (count_active), 484bbb7d… (count_old_physical), 5262e68a… (decode_capacity2).
2. Lifecycle88 stage_artifact/public-from_dir source-only proof is insufficient.
   Root must first accept the separate actual public-prepare microproof, including
   actual object/orchestration/device-image inventory and normal close/terminal audit.
   Current status: PENDING. No native result is invented here.
3. After that acceptance, Root may create a NEW acceptance wrapper with schema
   Root-F09-lifecycle-acceptance/1, approved=true, exact source405, adapter SHA6a2d8…,
   lifecycle SHA1cc9c…, actual_public_prepare_pass=true, proof_receipt path+SHA.
   This is an explicit proposed Root receipt interface, not a current receipt.
4. Separate explicit Root actual16 stage authorization is required AFTER task
   allocation. It must bind TASK_DEVICE's exact ordered16 distinct IDs, stage,
   ONBOARD, source405/scenario, this route SHA and the lifecycle Root receipt SHA.
   Never issue authorization just because this metadata check passed.
5. Root audits the current queue/resources before each native stage, after each
   terminal/own-PID/raw16-lock release, and before any subsequent finite command.

## Fixed scenario and gates

Capacity2, B2, 4 PHYSICAL pages, 2 logical pages/request, 2 state blocks. A page map
[3,1], B [0,2]; state A1/B0. All16 EP ranks participate even if local active=0.
Same physical T2 shapes; this does NOT test physical TensorShape2↔1.

Companion count-active uses SAME actual moe_ffn and forced gate destination rank8,
local experts0..5. Calls total192→96→192. Middle rank-prefix counts
[1]*8+[0]*4+[2]*4; local-zero ranks still serve remote peers. The independently
valid old-physical comparator runs its own SAME-operator artifact, captures an
INT32 nonnegative [16dest,16source,24expert] matrix, and must publish192 vs independent
runtime96 specifically. Malformed shape/dtype/source/counts and runtime exceptions
are FAIL, never numerical rejection evidence. First/last old calls are positive192.

Full entry uses ONE compiled decode_capacity2 artifact, ONE public owner and three
actual calls. Position126→127→128 with idle request positions advancing only when
active (exact rows in canonical scenario.json). Candidate returned cache/state
bytes feed the next call; reference outputs never feed candidate inputs. Persist
all initial/actual nonstatics BEFORE CPU references; source seed0 statics are
reconstructed/hash-verified rather than saved globally. Physical old cache rows,
idle request history/carry, and active UNWRITTEN opposite C2 parity row are bitwise
protected, excluding only source-proved legal current writes. Teacher-forced IDs,
synthetic history through125, E1 reused expert; NOT real checkpoint/prefill history.

Offline full validation must yield two separate groups for EACH call:
independent-unconditioned_direct19 AND original_staged19. Require all114 verdicts
and exact retention/clone/source/owner/artifact checks. The direct gates reuse
original expressions/tolerances with only expected-selector replacement; staged
nineteen remain unchanged. Do not call the original staged group unconditioned or
claim a strict whole-history reference beyond the separately recorded direct group.
No tolerance relaxation or negative-by-exception. Failure writes partial groups and
stops; no auto-retry, source edits, heap resize or fallback to SIM.

## Resources (schema planning, not measured fit)

See resource_plan.json for every ABI tensor. Full E1 synthetic static57 global
103,422,065,664B /rank6,463,879,104B. Largest global argument13,421,772,800B.
Full nonstatic494,223,296B; cache subset378,273,792B. Three initial+actual payloads
minimum2,965,339,776B. Two count roles' three capsules add147,456B, containers extra.
Probe selected static14,339,303,104B global /rank896,206,444B; probe IO5,268,544B.
IMPORTANT canonical count fixture factory still builds the FULL103.422GB fixture
before selecting layer0. It has not been made into a small-fixture builder.

Serial separate native count-active, native count-old, full capture and CPU offline
processes; drain/close/audit each before starting next. No overlapping full fixtures.
Public alloc_stacked_tensor launches16 concurrent shard uploads, joins/drains all;
ordinary source copies stage through POSIX SHM. Budget largest GLOBAL argument,
not a single shard. Planning host policies: capture253,832,885,888B and
CPU offline255,809,779,072B, based2S+2G+6P/10P+16GiB. These overcount some aliasing
and allow copies/references/packing; allocator retention/per-op workspace remain
unknown. Count uses the same300GiB floor, including retained full fixture and
selected14339MB layer copies. Fresh per-stage minimum HOST300GiB=322,122,547,200B,
DISK16GiB=17,179,869,184B, SHM32GiB=34,359,738,368B; unknown/query failure stops.
No claim these are measured peak bounds. Root may raise floors conservatively;
never reduce without separate review. Public free monitors are needed during
fixture/build/prepare/upload for first trial; no allocation proof is inferred here.

Public HBM after prewarm before static: full free >=6,463,879,104+10,157,899,712
=16,621,778,816B EACHrank; probe >=896,206,444+9,668,944,960=10,565,151,404B.
Poststatic and before/after every call require full/probe positive reserves above.
Reserve is conservative GLOBAL IO charged each rank +1GiB nominal comm+8GiB safety.
ONBOARD a2a3 public config [0,0,4GiB,0], persistent=False each call. TMR shared-bank0
installed contract24434… must stay pinned; one bank4.75GiB/rank already reflected
in post-prewarm free, so do NOT subtract committed/ring again. HCCL/VMM, task scratch,
SHM pool retention and transient peaks unknown. No168 identity-as-bank multiplier.

## Canonical lifecycle and cleanup

route.py executes original lifecycle88 f09_decode_case.py in a fresh process. Its
stage_artifact verifies the complete independent generated-file manifest, clones
sources to NEW work/prepared_artifact, and public g.reload_compiled uses
DistributedCompiledProgram.from_dir there. Public compiled.prepare(config,
persistent=False) produces native files; after_public_prepare independently binds
source projection and actual .o/.so/cache/*.bin before weight upload. No .bin is
required at compile-only preflight, no private pointers or frozen directory writes.
Every later CPU stage re-verifies actual prepared receipt/images/source projection.

Canonical PreparedDispatch shares nonstatic HOST IO before fork, prepares, uses
public stacked allocations, passes same normalized config each dispatch, copies
actual returns to existing host state, frees owned bases once and normal-closes.
Canonical catches primary failures and records cleanup errors separately; no
cleanup error may erase first error. Required owner_closed/count_owner_closed,
actual bytes, memory, artifact receipts and raw queue terminal logs survive.
Root's independent process audit must prove own birth/starttick identities absent
and exact16 locks released. Normal close is NOT inferred from hard timeout. No
other jobs/admin/reset/private cleanup. The current adapter is not a complete
process telemetry monitor; reuse reviewer-owned strict telemetry wrapper/audit and
retain any observation gaps as UNKNOWN/whole-case NOT_ACCEPTED, as earlier real case.

## Concrete commands — future Root GO only

All paths below are repository-relative. Set U=experiments/dsv41-flash-w8a8-a3/logs/
f09_native_route_prepare_1794081950 (one path, no newline). R is a NEW future case
root, never a frozen origin. Root places accepted lifecycle.json once; creates an
empty distinct R/count_active, R/count_old, R/full. No command here creates a grant.

Native one stage (repeat separately only AFTER prior terminal and cleanup audit):

    task-submit --device auto --device-num 16 --max-time 1800 --timeout 2400 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/f09_native_route_prepare_1794081950/native_stage.sh count-active /ABSOLUTE_NEW_CASE/count_active /ABSOLUTE_NEW_CASE/lifecycle.json'

The literal queued command MUST expand explicit absolute U/R chosen by Root at
submission; do not rely on unexported shell variables. native_stage.sh writes
allocation.json from actual TASK_DEVICE, then waits max300s for Root's atomic
stage_authorization.json. Root supplies exact IDs only AFTER reading that actual
assignment; pending wait is not authorization. Run this stage once only. Next
count-old command substitutes count-old/R/count_old; full capture substitutes
full-capture/R/full. FULL capture queue bound1800, no inline CPU golden/holdingcards.
If full publicprepare kernel build is too slow, timeout is a concrete failure;
do not automatically extend or resubmit. Root may choose separately reviewed
larger bound before first run based on lifecycle microproof, never infer speed.

Offline count-active and count-old then full, in separate --no-device jobs:

    task-submit --no-device --max-time 180 --timeout 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python "$U/route.py" --stage count-active-offline --execute --work "$R/count_active" --lifecycle-acceptance "$R/lifecycle.json" --authorization "$R/count_active/cpu_authorization.json"'

    task-submit --no-device --max-time 7200 --timeout 7800 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python "$U/route.py" --stage full-offline --execute --work "$R/full" --lifecycle-acceptance "$R/lifecycle.json" --authorization "$R/full/cpu_authorization.json"'

Again actual submitted command substitutes absolute U/R. Each CPU authorization is
separate Root kind ACTUAL_F09_CPU_REFERENCE with stage count-validate/validate;
count-old-offline mirrors count-active-offline. Use existing completed capsule in
SAME physical work for absolute ABI path; do not relocate prepared images/payloads.
Full CPU direct+staged remains original int32 reference, possibly slow;7200 is a
finite watchdog, NOT a performance prediction. Actual bytes remain durable if CPU
timebox expires; no native rerun just to retry offline validation.

Required native auth (canonical schema Root-F09-active-S/1): approved true,
operation bounded-synthetic-active-S-B2-3call, exact stage capture/count-capture,
kind ACTUAL_F09_NATIVE_CAPTURE; source405, scenario SHA, route SHA,
lifecycle_root_receipt_sha256, expires_at_epoch entry TTL<=3600, actual16 device_ids,
execution_mode ONBOARD, minimum_bytes floors, public_hbm_reserve numeric above,
compiled_source_manifest_sha256405, exact compiled_file_manifest absolute path+SHA,
compiled_metadata_sha256 and probe_source_sha256 for count role. Compiler input
manifest paths/hashes are the independent three signature names in route.py.
CPU scopes retain all source/scenario/route/lifecycle/fresh floor bindings, kind
ACTUAL_F09_CPU_REFERENCE, no NPU allocation. PENDING values/null never authorize.

STOP on missing lifecycle acceptance, unknown source/mode/shape, missing rank/query,
low fresh resources, native timeout/failure, missing prepared image inventory,
first device/runtime error, incomplete payload, unclosed owner, wrong count capture,
uncond or staged numeric failure, C2-unwritten/history mutation, cleanup/telemetry
unknown. Separate source/compile success, native execution, numerical groups,
negative controls and cleanup in final disposition. General F09/dynamic physical T,
general batching/unequal EOS, real E40 weights/text/original parity stay OPEN.
