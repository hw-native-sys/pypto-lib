# Finite observer permission diagnostics — UNAPPLIED candidate

The candidate fixes loss of PID/path/error context and false complete-memory
sums when a smaps_rollup permission read fails. It is artifact-only and keeps
strict FAIL on incomplete telemetry. Root policy approval and independent
review are still required before any future helper binding/application.
No old source, unit or raw result was edited; no model/header/payload/NPU/grant
operation occurred. The old failed36 unit remains NOT_ACCEPTED.

Forked only outer_runner.py plus an unchanged check_identity.py import
dependency from frozen longerouter16 f2cef526...a0d9f27. CLI command and main
(including scope/cap/code/query/receipt/default checks) are exact AST matches;
the checker is byte-identical. No model or generic monitoring framework was
added. This draft has no new task-entry/binding or operational authorization.

## Exact behavior

Structured error rows include real observation epoch/PID/observed start tick,
operation/path/errno and critical flag, immediately appended+flushed+fsynced.
smaps EACCES retains readable stat identity and continues accessible children
from every task thread. Unknown process memory stays null, the whole sample
total is null, and known partial sums are explicitly separate. Any incomplete
coverage also prevents a complete operation-wide sampled peak claim.

Stat/child enumeration/I/O/parse/error-receipt/fsync failures remain critical.
The launch identity read now sits inside the protected post-Popen observation
block, so such a critical error cannot skip supervision of the same child.
CLI terminal/exit, monitoring/launch/wait/post errors and known-live state
remain distinct; POST still runs only after the same child is known terminal.
No blind termination or process restart was introduced.

POLICY.md defines the proposed MODEL_COMPLETED_WITH_PARTIAL_TELEMETRY field.
It is explicitly NOT an approved success/admission policy. Current status
remains FAIL for incomplete telemetry; critical errors never receive that
proposal classification. No failed native receipt is relabeled.

## Executed tiny CPU evidence

ONE task_20261108_005737_323739021370, --no-device/max180, COMPLETE0, ten
controls. Fake proc fixtures test smaps EACCES keeping three PID birth records
and children from two threads, null memory versus known partial sums, and
critical stat/children/task-directory/EIO failures. Those integer fake sums
are not physical-memory measurements.

Four owned0.4s Python child controls exercise the actual observer: normal,
smaps EACCES, launch PID-stat EACCES and error-receipt fsync EACCES. Each starts
one child, waits/reaps that same child and performs POST after terminal. The
smaps error receipt is proven fsynced while the child is still live; its full
memory totals/peak are null with separate positive known-partial sums. The
normal complete control succeeds. Permission-only memory gaps remain strict
FAIL plus the unapproved descriptive proposal. PID identity and fsync controls
remain FAIL with critical context. Source command/main/checker inheritance is
checked directly; no old lifecycle/header/full-model suite was repeated.

The tests cannot identify the original failed native process/path/epoch or
establish future real-device telemetry completeness. No NPU retry was used to
erase or replace that failure. Final metadata collector only binds source/
historical pins, error fixtures/results and measured work record before PARK.
