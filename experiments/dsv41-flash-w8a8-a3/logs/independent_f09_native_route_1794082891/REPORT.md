# Independent finite F09 route preparation review

SOURCE/RESOURCE/ENTRY REVIEW ONLY; FINAL / PARK WITH FINDING. No CPU test, import,
compile, fixture, public prepare, NPU, header, tensor payload, grant or protected
source write. Root requested a clean process/queue window around lifecycle micro
042113, so no optional metadata/default CPU check was submitted. Source observations
are not presented as executed negative controls or hardware readiness.

Reviewed frozen Main12 route6f9ad3c6, lifecycle88 adapter6a2d8cb9/artifact helper1cc9c0b7,
scenario c7b6ab94 and three independent typed207 manifests. Local copies and complete
hashes are in inputs.json. Actual M02 and unapplied source405 remain separate;
source405 is not authorized by the production strict44 gate. Actual public lifecycle
micro acceptance remains PENDING in this review; another reviewer owns that case.

## Finding requiring an explicit readiness condition

f09_decode_case.py:404 validate_counts accepts either recorded role at line411,
selects the corresponding probe and validates its own source, but never compares
row.role to args.probe_kind. route.py sends --probe-kind old-physical for
count-old-offline; an active-role work directory is nevertheless accepted by this
validator and produces positive verdicts instead of the required middle-call
specific numerical rejection. This is a source control-flow finding, not a claim
that a native captured matrix is wrong. Before combined readiness, require exact
requested role across all three records and the old middle specific_numeric_reject
(actual192 versus expected96), with active/old work/artifact identities bound. A
new narrow adapter or independently accepted outer checker can enforce this; this
review changed neither frozen helper nor policy.

## Numerical and provenance limits

The active probe reads runtime token_counts[rank]; the old probe changes only
active to physical capacity2. Both call the actual moe_ffn and export its INT32
[dest16,source16,localexpert24] recv_meta counts after ffn_done. Forced gate experts
192..197 route to rank8 local0..5. Population192->96->192 and zero-local-active rank8
remote service are concrete gates, not exceptions or malformed matrix controls.

Full drive uses one owner and artifact, saves initial BEFORE dispatch and actual
AFTER copyback, and carries only candidate actual cache bytes to the next call.
Snapshots contain fresh contiguous clones; initial/actual storage alias checks and
SHA/individual tensor hashes protect offline inputs. No tensor-valued TensorSpec
expected comparison path is used. Mutable contract12 is populated by the actual
fixture builder before references. Direct19 replaces only nested staged selector;
original staged19 and all comparator expressions/tolerances remain unchanged.
There are 114 numerical verdicts, not a native-execution-only PASS.

Each direct reference starts from that call's captured initial, including previous
candidate ACTUAL caches. It is independent unconditioned forty-layer computation
WITHIN that call, not an independently propagated three-call oracle history.
Staged19 additionally conditions later-layer inputs on actual snapshots. Native
mode/owner/artifact/metadata and bitwise retention remain separate gates.

Pristine NONSTATIC initial hashes are checked after unconditioned and after staged
references; independent expected outputs are also hashed against mutation. Statics
are seed/hash checked before references, but not rehashed after each reference in
this adapter. Do not describe that as per-pass static integrity proof. Frozen
source goldens use static weights as inputs; a stronger static integrity claim
would need separate justified checks, not an inferred receipt field.

History begins at125 and grows only for active requests; group2 idles both at middle
call then resumes127, groups0/1 resume B127 while A128, group3 reaches128/128.
Physical pages A[3,1], B[0,2], state A1/B0 are disjoint and explicitly mapped.
Retention compares all previously valid physical rows except actual legal current
writes. State capacity2 keeps the unwritten parity row even for active requests;
idle requests keep both state rows. Fixed T2/B2/pages4 tests active-prefix changes,
not physical TensorShape changes, arbitrary batching or unequal EOS.

## Resource assessment

Schema sums/planning equations, not measured capacity/peak or executed fixture:
57 static global103422065664 =16*6463879104 bytes; full nonstatic494223296;
largest global static13421772800. Count still builds the FULL E1 fixture before
selecting14339303104 static +5268544 IO, and retains the full fixture. It is not a
small host fixture. E1 reuses one expert layer; it is not E40 checkpoint loading.

3 calls*2 payloads*494223296 =2965339776; two count roles*3*16*16*24*4 =147456;
combined logical minimum2965487232 excludes container overhead. Host capture
2*S+2*G+6*P+16GiB=253832885888; offline2*S+2*G+10*P+16GiB=255809779072.
300GiB HOST322122547200/16GiB DISK17179869184/32GiB SHM34359738368 floors exceed
those stated host plans, but are not proven peak bounds. Installed public
DistributedWorker.alloc_stacked_tensor submits16 concurrent alloc_tensor(init)
shards then joins/drains all futures; explicit copies use POSIX shared buffers.
Largest GLOBAL argument staging, not only one shard, is therefore appropriate.

After prewarm, full HBM requires static6463879104+reserve10157899712=16621778816
perrank; probe896206444+9668944960=10565151404. Reserve=9GiB+GLOBAL IO. Already
committed ring is reflected in actual free: do not subtract it twice or multiply
directory168 as banks. Full/selected input allocator retention, reference temporary
workspaces, SHM pool highwater, HCCL/VMM and native task scratch remain unknown.
Actual snapshots and live monitors are mandatory for future trials; source floors
alone prove no fit. Stages must remain serial separate processes, normal-close and
birth/raw-lock audited before the next stage.

## Concrete combined readiness delta

1. Independently sealed actual public micro receipt and Root acceptance wrapper,
   exact source405/adapter/lifecycle/proof SHA; no source-only PASS substitution.
2. Resolve recorded/requested count-role binding and independently preserve old
   valid matrix192-versus96 numerical rejection. Keep role-specific work paths.
3. New explicit Root16 stage scope AFTER actual ordered allocation; compiled source
   manifest/ABI/input file manifest and route/lifecycle/scenario pins exact. Current
   route's default/PENDING/auth checks are visible source paths, not a new grant.
4. Fresh measured HOST/DISK/SHM floors plus post-prewarm/prestatic/poststatic/percall
   HBM guards; strict telemetry and failure/firsterror ownership supplied separately.
   Observer68 process-race candidate is not approved/applied or part of this route.
5. Clone generated sources to fresh prepared_artifact, public from_dir/prepare,
   verify actual object/orchestration/cache-bin inventory before weights. Actual
   micro proof must establish supported inventory; unsupported images STOP.
6. Separate native count-active, native count-old, full-capture jobs via the frozen
   native_stage.sh commands in EXECUTION_PLAN, each only under its own future GO.
   Save actual bytes and normal-close first; SAME physical work then separate CPU
   count validations/full-offline, without relocating absolute prepared ABI paths.
7. Require the separate count control, 114 numerical verdicts, growing exact
   retention, source/clone/hash/oneowner/artifact/normalclose/PID/lock checks. Report
   direct per-call versus staged scopes accurately. No automatic retries/heap
   changes/physicalT variation or general F09 closure from this bounded case.

No actual16 permission/readiness, model application, full E40 weights/text parity,
F09 closure, observer policy change or new hardware result is issued by this report.
