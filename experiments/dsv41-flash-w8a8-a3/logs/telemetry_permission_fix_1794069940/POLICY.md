# Proposed finite-observer telemetry contract — NOT approved runtime policy

The candidate keeps the existing strict result policy: incomplete or failed
telemetry produces status=FAIL and main exits nonzero. Neither an old failed
run nor a future native task is silently promoted to success. Root and an
independent reviewer must approve any different future admission policy.

| Observation | Whole sample PSS/USS | Separate information | Current status after CLI0/POST pass |
| --- | --- | --- | --- |
| All discovered processes and thread-child paths read successfully | Known sampled total | Per-PID identities and memory | Existing success, pending external audit |
| smaps_rollup EACCES with readable PID birth and thread children | Unknown/null | PID inventory, denied identity, known partial sums, fsynced detailed error | FAIL; proposed classification MODEL_COMPLETED_WITH_PARTIAL_TELEMETRY only |
| Process/thread disappeared during observation | Unknown/null when inventory/memory is incomplete | Explicit disappearance/error context and remaining known sums | FAIL under current strict policy; future policy must review allowed gaps |
| PID identity, other I/O, child enumeration, launch/error receipt or fsync failure | No complete-total claim | Critical error context where available | FAIL; no partial-completion proposal for critical errors |
| CLI/postcheck/wait/lifecycle error or live child | No success inference | Original CLI verdict plus separate failures | FAIL |

Every recorded observational error carries epoch, PID, observed start_ticks
(null if unavailable), operation, path, errno, exception type and repr. A null
start tick is not a fabricated identity. Errors are appended/flushed/fsynced
immediately to telemetry_errors.jsonl. If error persistence itself fails, the
critical persistence error remains in the final result; it is not called a
durable error receipt. Receipt persistence never permits abandoning a live CLI.

A smaps permission gap is handled narrowly so accessible other PID/thread-child
paths continue to be observed. Unknown memory is never substituted with zero.
known_partial_own_tree_pss/uss are separate known sums, not complete own-tree
memory. If any sample/coverage is incomplete, global peak_sampled_own_tree_pss/
uss are null; max_complete_sample_* and peak_known_partial_* are separately
labeled observed maxima. All of these remain periodic observations, not a
continuous physical peak or leak/allocator attribution proof.

The proposed MODEL_COMPLETED_WITH_PARTIAL_TELEMETRY field is descriptive and
unapproved. It requires a known terminal CLI0, POST success, no launch/wait/
supervision errors, a stopped observer and no critical telemetry errors. It
does not satisfy the current whole-case native exit0 or complete-measurement
gates. Any future approval must keep code/source/config/query/cap/receipt/
fsync/PID identity/lifecycle gates critical and must explicitly specify which
noncritical gaps are permitted. External actual-model and cleanup audit is
still required. This unit supplies no new operation binding, grant or NPU GO.

The archived native failure remains exactly queue1/outerFAIL with one raw
PermissionError13. Its failed PID/path/epoch were never recorded and cannot be
recovered from these CPU injections or retroactively filled in.
