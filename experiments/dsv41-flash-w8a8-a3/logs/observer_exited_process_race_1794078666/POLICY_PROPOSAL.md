# Candidate observed-exit policy — NOT approved for use

Current active case and frozen outer15 remain unchanged and strictFAIL. No old result is reclassified. This candidate changes only a FUTURE observer's treatment of a PROVEN exited identity, not unknown-live measurement policy. Root AND independent reviewer must accept it before any future binding/run.

| Observation | Live coverage / candidate status after CLI0+POST | Full-operation memory/peak |
|---|---|---|
| All discovered live identities known, no exits/errors | Existing success pending external audit | Existing periodic-sample meaning only, not physical continuous peak |
| Same-sample recorded birth, COMPLETE child inventory, memory ENOENT/ESRCH, fresh stat proves absence or different birth | Exclude only OLD identity from live sample; durable explicit exit receipt required; may qualify completeLIVE | Per-PID memory remains null, never0; full-operation completeness false/peak null |
| Same recorded birth Z/X with sole leader in BOTH inventories | Same narrow confirmed-exit handling; a zombie with other threads is NOT retired | Null/false as above |
| PID absent BEFORE observed birth, unknown child subtree, lost thread-child path on live CLI, or same birth still present/live threads | UNKNOWN, strictFAIL | Unknown/null |
| smaps EACCES, even if process later exits | UNKNOWN-live, strictFAIL; no permission bypass/exit downgrade | Unknown/null |
| EIO/parse/PID stat/recheck failure, error or exit receipt/fsync failure | CriticalFAIL; no success/proposed partial completion | Unknown/null, critical context retained |

Children from all successfully enumerated thread paths are queued BEFORE memory read and retained when their parent exits. A reused PID is recorded as replacement birth only; its memory is not read or attributed to old identity. Exit before completed inventory cannot retire an unknown subtree. No rescan/retry loop, pidfd/private runtime/cgroup/kernel change or global errno-ignore is used.

Fields distinguish live_own_tree_observation_complete from full_own_tree_memory_measurement_complete. Candidate status may succeed ONLY with completeLIVE observation and existing lifecycle success; unknown-live or unknown-subtree still fails. This is a specific prospective policy for known exits, NOT optional partial-telemetry success. If any exit was excluded, operationwide peak_sampled_own_tree_pss/uss is null. max_complete_sample_* filters ONLY fully-known samples; known partial sums are labelled separately. No production PSS/USS peak or fit claim follows.

Proc reads and process discovery remain sequential samples, not an atomic/cgroup-complete snapshot; this patch resolves only the explicitly evidenced failure-path race. It does NOT solve disappearing live-thread child inventory, all possible PID discovery races or current-case gaps. Existing wait/CLI/POST/firsterror/fsync/default/command policies remain strict.
