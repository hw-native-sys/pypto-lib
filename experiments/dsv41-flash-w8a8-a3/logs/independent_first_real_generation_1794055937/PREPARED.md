# Prepared finite post-case audit — NOT EXECUTED

Root assigned case `native_full_real_generation_1794055148`, native task `task_20261107_204310_361283812010`. The auditor is prepared while that ONE operation runs. Do not run it until the device task is terminal and the executor's external queued CPU POST_EXEC receipt is available. No live model tensor/source is imported or modified by preparation; only source/log/JSON formats were read. No actual grant or retry is created here.

After the executor notifies stable terminal paths, the reviewer OUTSIDE the queued auditor captures raw `task-submit --status TASK`, device-status and queue-list stdout/stderr/exit/timestamp in its own `queue_snapshot.json`. That file has `task_id`, `captured_epoch`, `status_stdout`, `status_stderr`, `devices_stdout`, `list_stdout` and query exit fields. There are no nested task-submit calls in audit.py.

Create a new own handoff.json with absolute `case`, `outer_queue_snapshot`, `external_POST`, booleans `native_terminal_confirmed` and `executor_external_POST_terminal_confirmed`, and the exact stable external receipt/source hashes. This file is audit provenance, not a model execution authorization. Bind/hash the consumed files before and after the single queued CPU audit.

Intended ONE CPU audit command (not yet run):

```bash
task-submit --no-device --max-time 600 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/independent_first_real_generation_1794055937/audit.py --handoff experiments/dsv41-flash-w8a8-a3/logs/independent_first_real_generation_1794055937/handoff.json'
```

The finite audit checks existing receipt/source/config chains, complete phase source maps/endpoints/counters, actual recorded TP-global strict4DP sampling/history/EOS/UTF text, normal shared-state driver statistics/hash records, actual generated ABI, final16-rank memory events, primary/secondary outcomes, recorded process birth identities, queue/lock snapshots and stable input bytes. It loads no tensor payload and computes no model golden. On an actual failed case it preserves available evidence and reports unmet gates rather than treating an auditor's exit0 as generation success.

Limits remain explicit: no second raw-logits argmax calculation, no independently checkpointed actual cache-byte/handoff comparison, no complete D2H weight parity, no continuous physical peak and no full-original Engram/MTP/generalF09/parity closure. Existing raw records and immutable source support only the scope they show. No broad tests or further NPU are scheduled.
