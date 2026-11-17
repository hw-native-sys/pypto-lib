# Finite longer-N3 normal CLI operation preparation

No actual generation grant, source payload, NPU or dispatch is authorized in
this preparation. The binding is exclusively Once upon a/[16600,4465,260],
plain N3/start0/max16/repeat1/noEngram/noMTP/full40/TP4DP4EP16/W8A8. Root must
separately authorize a new actual operation after independent acceptance.

The three runtime scripts are check_identity.py, outer_runner.py and
task_entry.py. Their byte hashes are in operation_binding.json and FREEZE.json.
The current unit PREQUEUE is a metadata readiness receipt, not a future actual
operation's fresh prequeue or grant. No old grant may be refreshed or relabeled.

For a separately authorized future new case, create a fresh exclusive output
unit, then queue the frozen checker with --no-device and --stage PREQUEUE,
--binding this frozen operation_binding.json and --output NEW_CASE/prequeue.
Record COMPLETE0 and the receipt hash before submitting any native job.

Only after Root provides a fresh exact operation GO and the independent gate
passes, use ONE task-submit --device auto --device-num 16 --max-time 21600 with
--timeout 22200 (or asynchronous same-task wait). Its allocated command is the
following finite sequence with actual future Root GO/case paths substituted:

    cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib
    source activate-pypto.sh
    PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/longer_generation_operation_prepare_1794063015/task_entry.py --root-operation-go ROOT_EXPLICIT_FUTURE_LONGER_OPERATION_GO --binding experiments/dsv41-flash-w8a8-a3/logs/longer_generation_operation_prepare_1794063015/operation_binding.json --prequeue NEW_CASE/prequeue/PREQUEUE.json --output NEW_CASE/actual

This sequence is documentation, not current execution permission. TASK_DEVICE
must be the actual ordered16 mapping matching the accepted calibration, not an
assumed compile mapping. Root must expressly delegate creation of the new
task-local assignment AFTER allocation. Its entry TTL is3500s (ceiling3600),
so queue wait cannot age a grant created before allocation. The outer executes
fresh ALLOCATED_PRE_EXEC identity/header/tokenizer/resource checks before
creating a new ACTUAL generation receipt, expiry+3500 and exact code/source/
binding/actual16/prompt IDs/max16. This TTL bounds entry validation; the unchanged
driver deadline20400 and queue21600 bound actual runtime. Client22200 is a
separate wait bound. On client timeout reconnect to the SAME task, never resubmit.

The normal product CLI command is fixed by outer_runner.command: exact plain
Once upon a, max16, actual previous greedy outputs and strict4DP/EOS. EOS may
stop before16 tokens. No planned continuation tokens or output substitution.
Readiness does not prove fifteen decode calls or their resource peaks/lifecycle.

Preserve bounded HBM cadence/live HOST>=64GiB, additional DISK>=2GiB,
SHM>=16GiB admission and all source/read/copy/target caps. Each phase remains
164192 source reads/745846365184B and633997065216 H2D bytes/912 endpoints.
Future-call IO planning remains the existing formula with positive remaining
calls above the unchanged9,797,894,144B postforward floor; it is not measurement.
Do not credit committed deltas or subtract ring/committed a second time.

STOP on the first source/query/cap/DP/deadline/cleanup error. Preserve CLI exit,
observation and postcheck errors separately, partial progress and known live
PID/starttick identities. Queue hardkill is not normalclose. Do not terminate
other jobs, resize/retry, replace E40 with E1 or change thresholds/heap/caps.

After the SAME job is terminal, capture raw status/log/queue/16lock evidence
outside CPU tasks and queue the frozen check_identity.py --stage POST_EXEC
into NEW_CASE/external_postqueue, even on CLI error/hardkill. Record its actual
task COMPLETE status and receipt hash. Independent finite CPU audit must verify
normalclose versus forced state, actual greedy history/EOS/4DP, both phase
coverage, source/metadata/tokenizer/code bindings, PID/startticks absence and
lock/queue release. No tensor payload reread or second native run for auditing.
Only then may Root accept the bounded result and the owner seal/PARK.
