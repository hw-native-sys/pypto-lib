# Checkpoint-pipeline artifact delivery — partial preparation, PARKED

Working atomic checkpoint library and toy checks are ready for Main review.
Native prefix / CPU full golden / decode-resume adapters are specified in PLAN.md
but NOT implemented as a runnable full-model pipeline. split_checkpoint.py
refuses state writes over16MiB and ALL production/native resume. This deliberate
limit prevents an unreviewed artifact from doing large IO or treating a native
runtime result as numerical acceptance. Do not run it on the current prefix.

CPU task113628_340042520249 passed the initial10 controls. Final task
113927_343736812141 passed11 after explicit dual-reference scope requirements:
atomic exact copied-byte roundtrip; original tensor mutation cannot alter saved
bytes; actual/initial clone isolation; noStaticPayload; bounded logical hashing
of expanded tensors; corruption/header/source/SIM/phase/seed/alias/gate/scope /
native-only rejection. Both jobs ended0. No failed own job, NPU, full-prefix,
real weight payload, large allocation or large IO was run.

The future pipeline must save actual ALLnonstatic prefix values and independent
initial nonstatic oracle clones BEFORE EITHER unconditioned40-layer reference
or lazy staged40-layer reference. It must close/release prefix cards, then run
CPU validation using BOTH original scopes/thresholds. Phase schemas differ:
prefix17 exports, decode19 exports; do not reuse the erroneous fixed19 prefix
headline. Current toy receipts are explicitly toy-only and cannot unlock native
resume. A future production receipt must bind actual phase keys, both reference
scopes, immutable checkpoint/source/ABI/static-manifest identity and successful
owner close.

Static fixture reconstruction from seed0/frozen source is a PLAN only at full
scale: it must prove all57 static headers/byte hashes reproduce before usage,
without saving global stacks. Hashing should be completed before holding cards;
the tiny prototype computes static hashes inline and must not be used on full
logical stacks. Large actual+initial checkpoint IO requires separate measured
host/disk/SHM headroom and authorization. Same-decode-artifact/one-owner128/129/130
and stale129 must persist each actual input/output state; source/current helper
semantics and both reference passes remain. Runtime failure never counts as
negative numerical rejection. See PLAN.md for exact insertion points/protocol.

Only this checkpoint_pipeline directory was written. The active reviewer helper
was copied read-only and never edited/import-executed; no process memory or run
was inspected or modified. All45 root model sources remain equal to Main's
pre-dispatch frozen manifest. Work record is uniquely identified for Main import
once; no shared ledger was written by the worker.
