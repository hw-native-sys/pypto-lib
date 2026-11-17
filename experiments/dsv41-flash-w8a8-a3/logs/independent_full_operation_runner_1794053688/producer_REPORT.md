# Normal full-generation outer preparation — no execution authority

Prepared finite check_identity.py +outer_runner.py +task_entry.py for FIRST
Root-authorized normal product-CLI execution. No model code is imported or
monkeypatched by the outer runner. Its command launches the ordinary applied
models/deepseek_v4_1_flash_a3_w8a8/generate.py in a fresh subprocess, with exact
plain Hello,world N3/max4/noEngram/TP4DP4EP16/W8A8 arguments and the original
greedy/strict4DP/state body. This unit created NO actual grant, NPU request,
real-checkpoint tensor payload, large fixture, product change or shared-ledger write.

## Exact binding and checks

operation_binding.json pins Root-provided applied45
d77d196be6541b4a87aef07cf0e5e02eac68cefa9fc245f3742e4c14606555aa,
generate d057012464331b10b0bdc7f406925ee0c552dd298a1db171a93b0b1595769598,
unchanged44 against predecessor762dc and plan
3dd67b88683d0de1f5fbd8e889c0c1a875eeecf671263fca4c64061637ecd9c0.
Main application17 manifest7402f715fd2acd0ea4574821e7be1637ff6b00923207949d79430d1357a45e11.
Historical calibration3bc729/probeC4b62/selected-result2d0a4b are SHA-bound
prerequisites, not grants, current peaks or refreshed timestamps. Ordered16 IDs
must match historical mapping AND actual TASK_DEVICE at allocated execution.

Checker validates exact resolved model_root==checkpoint root, both tokenizer
file SHA hashes from plan, actual45 files/44 unchanged, weight-map/index SHA,
installed RuntimeBuilder-selected TMR24434f library and all139614 planned source
descriptors against current index/header shape/dtype/absolute byte range/file
size/mtime. It reads only8-byte header prefixes +bounded JSON headers, not tensor
payloads. It records HOST64GiB/DISK2GiB/SHM16GiB policy admission and leaves
full-upload/forward peaks NULL. These policies are planning, not measured peaks.

The same finite checker is called at PREQUEUE, ALLOCATED_PRE_EXEC, POST_EXEC.
Actual PREQUEUE MUST be refreshed in a NEW future operation directory before
submission; this unit's prepared check is evidence of readiness, not a fresh
future submission receipt. Outer always attempts POST_EXEC after a launched
CLI terminal nonzero exit or a pre-launch exception; it preserves primary CLI returncode and
separate postcheck/monitor errors. CLI0+postfailure is a failure. If queue hardkill
also kills outer, Python finally cannot be promised: mandatory separate queued
CPU POST_EXEC after terminal queue is documented below. Missing postcheck is not PASS.

## Own necessary CPU evidence

- 200250_35067784126 COMPLETE0: tiny owned tokenizer root/hash and source-header
  range negatives; exact header-byte-only arithmetic; fake CLIexit7+postfailure
  both retained/POST executed, CLI0+postfailure FAIL, outer default refusal before
  grant/output creation. No actual normal product CLI was run.
- 200251_35070209062 COMPLETE0: bound actualD77 metadata PREQUEUE. Exact2tokenizer
  hashes,45/44/plan/index/map/TMR and139614 source keys across263 used shards pass.
  Only15,912,800 metadata bytes were read; tensor payload bytes0/grant_createdfalse.
  Fresh HOST1,723,054,063,616B/DISK8,950,009,298,944B/SHM1,079,343,157,248B passed.
  Metadata is not complete-source numerical proof; header identities are the scope.
- 200549_352662626788 COMPLETE0: final outer short-process PID/starttick capture,
  child reaped/absent after wait, primary7+posterror preservation and stopped
  monitor. An immediate child identity record prevents missing short children
  between2s PSS samples. No actual grant or NPU command was executed.

The own-tree process sampler reuses the finite /proc/stat +smaps_rollup pattern
from accepted selected-transfer34: sums PSS/USS, retains perPID RSS and identity,
system SHM used, progress file byte-size/mtime and2s samples. It does NOT sum
forked RSS as physical memory or claim exact allocator high-water. The original
product handles live host/device guards,20400s cooperative deadline and memory
cadence; outer introduces no alternate budget/model driver. The copied approach
is explicitly local code, not an import of the old run_transfer.py (whose top-level
body would execute real transport). No old source/evidence file was modified.

## Independent source-review lifecycle corrections before freeze

Root's independent source reviewer found an immediate post-Popen sample error
could skip child.wait. The final runner records observation errors separately
and still waits for the SAME child; one wait interruption is followed by explicit
same-child supervision, not CLI resubmission. If terminal state cannot be
established after interrupted waits, result is FAIL with live-child identity and
POST_EXEC is SKIPPED (never falsely presented as ended); external queue supervision
and postqueue audit remain mandatory. No blind termination/free of in-flight work.

Immediately after Popen identity, atomic fsynced cli_started.json is attempted
BEFORE optional sampling/wait. Failure to write it is an observation error and
still does not skip waiting. The process inventory unions task/*/children to
include non-main-thread HbmSampler children such as npu-smi. CPU202249_35648865289
COMPLETE0 proves immediate sample failure, launch-receipt failure and first-wait
interruption all retain CLIexit7 and wait before POST; thread-spawned child is
captured/reaped. This is the final runner's scoped regression; earlier process
tests are retained as earlier evidence, not silently overwritten.

## Future commands — REQUIRE separate explicit Root actual GO

Use NEW actual operation paths, not this sealed unit for outputs. Resolve these
placeholders in the future Root assignment: OPERATION_DIR, FRESH_CHECK_DIR and
ROOT_OPERATION_GO_ID. There is no actual approval marker/grant supplied here.

First refresh prequeue under task-submit --no-device:

```bash
task-submit --no-device --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/check_identity.py --binding experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/operation_binding.json --output FRESH_CHECK_DIR --stage PREQUEUE'
```

After independent applied-signature611 acceptance and explicit Root full-real GO,
ONE automatic16 queue command with finite task-local creation:

```bash
task-submit --timeout 22200 --device auto --device-num 16 --max-time 21600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/task_entry.py --root-operation-go ROOT_OPERATION_GO_ID --binding experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/operation_binding.json --prequeue FRESH_CHECK_DIR/PREQUEUE.json --output OPERATION_DIR'
```

task_entry defaults to refusal without explicit future Root GO id and matching
actual ordered allocation. AFTER allocation, it creates a NEW one-attempt Root
assignment (expiry+3500s), bound runner/checker/entry/binding/PREQUEUE SHA and
actualIDs. This is explicitly delegated creation only under that future RootGO,
not a renewal of an old grant. Outer checks assignment/code/PREQUEUE, runs fresh
ALLOCATED_PRE_EXEC, then and only then creates the normal product authorization
(expiry+3500s), with source/plan/map/outer/check/Root receipt hashes and exact
progress path. It invokes the fixed normal product command. No arbitrary
alternative command can be passed to task_entry/outer main.

Queue wait is distinct from entry TTL: no actual assignment exists until allocation,
so waiting in queue does not stale a prewritten one-hour grant. Grant TTL controls
entry, not the entire20400s cooperative operation. Queue hardcap21600 and client
22200 are separate; if queuewait+runtime exceeds client wait, use --status and
reattach --timeout22200 --wait SAME_TASK. Never resubmit or assume timeout failure.

Mandatory external postqueue step after terminal job, including hardkill:

```bash
task-submit --no-device --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/check_identity.py --binding experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/operation_binding.json --output NEW_EXTERNAL_POSTQUEUE_DIR --stage POST_EXEC'
```

This records independent source/tokenizer/header identity even if in-job outer was
killed. Actual acceptance additionally audits generation/progress complete-coverage
records, real greedy/strict4DP/state, primary CLI exit, secondary errors, normal
owner close versus forced cleanup, all recorded PID+starttick identities absent,
actual source/native artifact pins and16locks release. A successful identity check
or text string alone is not full-generation numerical/fit acceptance. Keep all
raw files, CLI stdout/logs, progress JSONL and partial states; do not reread full
payload or rerun NPU just to repair an auditor.

STOP on any source/root/tokenizer/plan/index/map/header/library/orderedID/code/
receipt mismatch, stale/missing assignment, duplicate output/assignment, resource
admission failure, CLI failure, postcheck/monitor failure or missing lifecycle
evidence. No retry, key/range expansion, model monkeypatch, alternate driver,
private runtime pointer, ring resize, administrator intervention or other-job action.
All real upload/firsttouch/forward peaks remain unknown until the actual operation.

## Freeze and ownership

Only this NEW preparation directory was written. Runtime files are the three
finite scripts; preparation/tiny-check/seal scripts are auxiliary CPU evidence.
Binding and sources/manifest pin exact code; final collector rechecks applied17,
old32/new10 and product45, syntax/default constraints and absence of actualgrant
files. Own measured work_record is review-agent, shared-ledger writes0, for Root
to import once. Final/PARK after seal, no automatic scheduled continuation.
