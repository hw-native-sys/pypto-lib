# Executable commands, with future inputs required

Completed once (synthetic scaffolding only):

```sh
task-submit --no-device --max-time 600 --timeout 900 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && PYTHONDONTWRITEBYTECODE=1 .venv-local/bin/python experiments/dsv41-flash-w8a8-a3/logs/independent_configurable_generation_auditor_1794110902/selfcheck.py'
```

Task121430_1344586710 COMPLETE0,37/0. Do not rerun it as actual-case validation.
The exact initial tested shared/topology/runner/test source is preserved under
cpu_tested_core; final schema alignment after Main supplied ancestry is explicitly
source-reviewed, not relabeled a second CPU result.

Future execution is NOT authorized by this document. After source/helper/pair
freeze, real SAME task terminal+stable mandatory externalPOST, and Root's new
actual-audit GO, bytefork this unit into a NEW same-depth owned execution unit,
rename inherited seals input_*, and write real external bindings/handoff there.
Fill every required value from final producer and observed outside-queue facts;
status becomes READY_POSTTERMINAL only in that newly authorized real unit.
CLI inputs contain hashes/paths/RootID/taskID; no auditor source edit is needed.

Using the NEW execution directory as the current directory:

```sh
task-submit --no-device --max-time 600 --timeout 900 --run 'cd REQUIRED_NEW_EXECUTION_DIR && PYTHONDONTWRITEBYTECODE=1 /mnt/data/pyptouser/yangyaodong/code/pypto-lib/.venv-local/bin/python runner.py --bindings bindings.actual.json --handoff handoff.actual.json --output-dir actual_audit'
```

runner.py requires a fresh child output directory, records execution.json before
and finally after audit, and preserves exception/first-cause even assertion exits.
Audit job0 means the auditor completed; read actual_audit/audit_results.json for
whole NOT_ACCEPTED versus bounded findings. No nested tasks/model runtime/NPU.
The owned tokenizer_reference.py is an exact stdlib source copy, checked against
the actual source45 tokenizer hash before actual text decode, not a model import.

After audit terminal, capture its public `task-submit --status` and `--list`
OUTSIDE the queued task into NEW owned files, then run metadata-only closure:

```sh
PYTHONDONTWRITEBYTECODE=1 /mnt/data/pyptouser/yangyaodong/code/pypto-lib/.venv-local/bin/python finalize.py --execution-record actual_audit/execution.json --audit-status audit.status --queue-final queue.final.txt --output closure.json
```

finalize.py queries no queue/runtime itself and changes no other process. It
reconciles own-audit PID birth absence and consumed input hashes, preserving the
whole case result. Unknown actual facts or source/schema drift STOP, preserve,
report; no blind CPU/native retry or criterion/status-policy relaxation.
