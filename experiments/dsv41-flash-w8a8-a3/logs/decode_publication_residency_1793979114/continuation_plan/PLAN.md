# Artifact continuation plan

Main owns launch and integration. This worker writes only continuation_plan/;
no model, shared ledger, runtime, checkpoint or NPU operation was performed.
The prepared chain is NOT native validated. Main must first verify repaired
decode all-output numerics and the exact PreparedDispatch native residency test.

chain.py uses the current driver class, not a second dispatch implementation.
PreparedDispatch retains rank-static weights within a phase; mutable state uses
stable shared host tensors with H2D/D2H each call. persistent=False recreates
communication domains each dispatch. Prefill owner closes before decode owner.
Only one decode artifact and prepared owner serve128/129/130 and the control.
This does not claim persistent device caches, no readback, real weights, full
Engram, general dynamic multi-request support or strict cumulative equivalence.

The owning synthetic fixture has40 dense layers and one reused24-expert shard
per rank. p.build_fwd_specs128 ordinarily allocates11 pages based on its old
fixture start1248. The artifact compacts cache page axes and index-table width
to2 before concrete contract construction. Actual prompt is128 tokens at0..127,
which needs one page; decode128..130 fits the second page. State batch remains1.
No model ABI shape rule or source algorithm is changed. Full checkpoint loads
and original token-history hashing are absent; engram_enabled=0 is explicit.

Expected inputs clone EVERY nonstatic tensor, including metadata and cache
buffers, before dispatch. Static weights retain immutable references. This
avoids owner.copy_ metadata changes aliasing expected inputs. Each phase uses
owning full-forward comparators and explicitly reports their staged scope;
unconditioned logit max-absolute difference is recorded without a parity claim.

After prefill, actual returned cache bytes are hashed; handoff records source
and target hashes/pointers/object identity. No golden cache is inserted. Each
decode hashes actual incoming/outgoing persistent buffers, checks that initial
reference buffers do not alias actual caches, and requires original WINDOW/C1
rows0..127 and C2 rows0..63 remain byte-identical. State cache is exempt from
prefix retention because its carry slots legitimately update each step.

The negative branches from saved ACTUAL returned128 state. It creates a correct
129 oracle snapshot, then leaves128 slots/lens/query and source RoPE in the
actual call while position/token advance129. Kernel/runtime exceptions escape;
only standard comparator failure on selected newly published WINDOW and C2 rows
satisfies the control. Original128-token prefix rows must remain unchanged.
The current standard row comparator retains its existing thresholds. Parent
must inspect numerical error evidence, not equate an exception with rejection.

run.sh takes FRESH_OUTPUT,DECODE_PASS_JSON,RESIDENCY_PASS_JSON. Main must set
CHAIN_PREREQUISITES=verified after checking real task outcomes. The file-existence
and environment guards are accidental-launch checks, not an independent verifier
of those prerequisite files. Their hashes are recorded. Future native invocation:

```bash
task-submit --timeout 3600 --device auto --device-num 16 --max-time 3600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && export CHAIN_PREREQUISITES=verified && bash experiments/dsv41-flash-w8a8-a3/logs/decode_publication_residency_1793979114/continuation_plan/run.sh /absolute/fresh/output /absolute/decode_pass.json /absolute/residency_pass.json'
```

Execution mode is explicitly False/ONBOARD; driver normalizes it and compile
guards require metadataa2a3. Native runtime log evidence must be audited by Main
after execution. The runner starts HbmSampler only inside the allocated16-device
job. It prints logical host/rank bounds and available host memory before filling
bounded synthetic weights; measured peak HBM and real model fit remain unknown.

CPU test task_20261107_001114_409790928721 exit0 exercises small synthetic clone
isolation, static identity, two-page compact/state-batch preservation, prefix
retention and deliberate old-prefix corruption, plus current public signatures.
py_compile and bash -n pass. No full prefix fixture/compile was performed; no
kernel/NPU/full weight data was loaded. The pending Main source may evolve:
recheck APIs and helper source before launch. Native prerequisite, resource fit,
full-forward references, published-row rejection and protocol lifecycle remain
unvalidated by this artifact-only worker.
