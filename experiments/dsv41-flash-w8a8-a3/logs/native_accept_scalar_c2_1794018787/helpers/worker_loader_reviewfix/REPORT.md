# Loader review repair — CPU self-validation

Exclusive model scope remains weight_loader.py. Earlier worker_loader evidence
is frozen and unchanged. Main owns generate.py and records this narrowed ABI
repair scope; no shared ledger, runtime, harness or checkpoint edits here.

The top-level iterator now validates the requested rank shape and dtype using
argument_contract BEFORE calling embed_weight, head_weight, final_norm or an
enabled engram_rotation. The existing transformed output check remains in place.
check_top.py reproduces eight old malformed requests reaching the denial reader
(four resources times wrong shape/wrong dtype), then proves all eight reject
without reads in the fixed implementation. Eight asymmetric normal output
cases across two ranks are exact against independent analytic expectations.

Main's complete-ABI preflight additionally exposed a real phase difference:
prefill_fwd.py world arguments c1a_* omit the singleton source axis, whereas
decode_fwd.py retains it. argument_contract now takes explicit phase='prefill'
or phase='decode' (default preserving the previous decode API); only prefill C1
metadata drops this axis. The iterator accepts precisely both established C1
layouts through its shared source provider loop. Both yield the same one leaf
for original source layer20 at byte offset0 with logical_prefix=(20,).
No payload reshape/copy or alternative math path is added for this difference.

check.py verifies all seven C1 resources in both phases across two synthetic
ranks (28 exact cases) alongside the prior two-rank/three-layer/two-expert
independent NZ/fold/scale, aggregate parity and bounded-lifetime controls.
check_headers.py checks every static resource in BOTH preserved actual40-layer
prefill/decode distributed metadata files without importing forward entries,
reading checkpoint payloads, or allocating full-size tensors. This confirms
schemas, not current native execution or memory fit. Earlier worker evidence's
metadata coverage was decode-only, not both phases.

Historical count correction: the earlier headline 'nine rejection controls'
comprised six actual structural rejections and three rank/layer/expert data
distinguishability inequalities. The new report separates those counts. In
addition, the new check sends each of these three substituted stream outputs
through the exact data validator and requires an actual numerical rejection.
These are three mutated-stream rejection controls, not additional model paths.

Final CPU task task_20261107_100322_126563330433 exits0: check_top.py,
check.py, check_headers.py, scoped ruff and py_compile. Commands run from repo
root through task-submit --no-device after sourcing activate-pypto.sh; scripts
are in this artifact directory. Exact command is recoverable from parent unit
work record and this file list; scripts write only their own result JSON.
cpu_queue.log preserves an earlier fixture-only failure: fake reader omitted
weight_map, so the baseline hit AttributeError before its denial reader. The
fixture was corrected, and the final task reproduced the intended eight old
payload reads. This was not a model/compiler/runtime failure.

No native checks, real payload loads, full allocations or tolerance/golden edits
were performed. Native uploader reuse, byte coverage/cleanup, RSS/HBM, functional
decode/continuation and independent acceptance remain Main/reviewer gates.
New source hashes, artifact manifest and a unique measured worker record
accompany this report. Worker stops writes/jobs and PARKS after sealing.
