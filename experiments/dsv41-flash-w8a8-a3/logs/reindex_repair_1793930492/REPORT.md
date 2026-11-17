# C1A source-pool consumption repair

Bounded main-agent unit, 2026-11-06. Original-reference semantics, implementation,
restricted-pool negative controls and real-device acceptance are recorded here.
This is main-agent evidence; independent observer acceptance is separate.

## Semantics and implementation

The pinned original vllm-ascend `v0.30.0rc2` revision is
`8fde6d4e335fe300e2b949aab0e31659cce0811d`. Its local read-only reference is
`../logic_review_1793929657/upstream_rc2/`; `fixture_evidence.json` hashes the
reference archive, tag provenance and inspected source files. No reviewer file
was modified or unpublished review conclusion used.

| Original reference | Established contract |
| --- | --- |
| `models/deepseek_v41/model.py:514-548` | Candidate source and later index sources have distinct roles; later sources use candidate filtering. |
| `models/deepseek_v41/indexer.py:181-204,274-300` | Candidates are logical block IDs, source and consumer share them within the forward, consumer mode passes the source candidates and returns them unchanged. |
| `attention/dsa_v41.py:585-610` | Shared candidate state is written only by the candidate source. |

Both A3 phase paths previously computed full-history Top-K and republished a
pool from the reindex layer's own scores. Their goldens repeated that mistake.
The corrected paths still compute the existing A3 query/head scores, then the
shared `indexer.filter_candidate_scores` masks logical rows outside the counted
source pool before the existing shared Top-K. Candidate blocks contain 8 rows;
physical cache pages contain 128. Top-K maps logical positions through each
request's page table. The source pool is preserved. Missing selections use
index `-1` and score zero; inactive output rows are preserved. Scoring remains
over full history before masking; this unit makes no sparse-performance claim.

`golden_reindex_select` is independent of that masking implementation: it
enumerates logical positions, tests their block membership, sorts eligible
scores and maps selected positions to physical rows. No candidate output feeds
the oracle. Both standalone goldens, the shared layer golden and staged
references use that semantic reference. Staged references obtain InOut pools
from their captured pristine fixtures.

Both complete forwards now pass layer 20's pool to every C1A reindex layer:
decode source selection slot 3 (`C1A_SLOT_BASE`), prefill C1A pool slot 0.
Per-reindex Top-K slots remain separate for later reuse layers. External pool
stack shapes/order are retained; the four formerly regenerated reindex pool
slots are now reserved and left unchanged (`-1` IDs / zero counts in bounded
fixtures). The source slot remains validated. This is a recorded correction
to diagnostic output meaning, not removal of the source-pool gate. Generated
bindings and hashes are in `generated_pool_bindings.txt`.

The shared paged Top-K comparator now validates the exact valid-lane mask,
`-1` invalid IDs and zero score padding, then delegates to the existing
`golden.topk_pair_compare` with padding excluded from score ordering. This fixes
the independently reproduced negative-score/zero-padding boundary in the
16-token prefill layer. Existing score/output thresholds are unchanged.
`comparator_evidence.py` retains legal negative-score ties and rejects unsorted
valid scores, missing/extra valid lanes, invalid sentinels and nonzero padding.
All indexer callers use this comparator; expert-route comparison is unchanged.

## Device and host evidence

| Check | Task / result |
| --- | --- |
| Frozen wrong phase kernels with corrected restricted references | `task_20261106_100621_114259824307`, device 0: both executable exits 1, decode default/handoff and prefill fail numerical gates. |
| Frozen original global Top-K with typed input pools and deterministic oracle | `task_20261106_102132_132654727492`, device 1: IDs and values FAIL exactly as required; runner exits 0 only after verifying rejection. |
| Final indexer and all six phase callers | `task_20261106_104531_147345915467`, device 0, exit 0: six indexer plus nine phase cases PASS; see final `consumers.tsv`, `unit_summary.json` and `accepted_*.verdicts.txt`. |
| Final affected C1A reindex layers | `task_20261106_104341_14597368219`, all 16 devices: decode (T=2) and prefill (T=16) PASS every existing layer gate using supported fresh-worker dispatch. |
| Four affected phase/full-forward host compiles | `task_20261106_101629_122967030312`, `--no-device`, exit 0: both reindex operator/layer chains and both complete forwards compile. |
| Pin/provenance verification | `task_20261106_100537_112790311286`, `--no-device`, exit 0: recorded revision-3 four-piece pins and build inputs match. |
| Final lint, comparator and handwritten-oracle controls | `task_20261106_104531_147307825056`, `--no-device`, exit 0. |

The indexer default has four existing numerical cases plus two restricted-pool
cases. The latter use permuted physical pages and mixed request IDs, counted
prefixes with unused valid IDs, a partial tail block, empty counts/contexts,
and inactive rows. Scores increase deterministically, so the unrestricted
negative selects forbidden high-score physical row 127 (score 256). Handwritten
expected selected counts are `[13,24,1,0,0]`; full IDs/values are checked in
`host_evidence.py`. The second positive dispatch changes active tokens 5→3
through the **same compiled artifact**, verified by the `runtime_only` log.
Its four output gates use zero tolerance and zero permitted mismatches.

Six phase entries supply nine numerical cases, including decode handoff and
continuation. Their new restricted inputs exercise actual attention consumption
and pool preservation. The 16-device layers additionally check TP4/DP4/EP16
composition, every routed branch and cache/state/output gates on the existing
fixtures. This does not accept other layer modes or the standalone TP gate.

All commands source the selected environment, use `$PTOAS_ROOT/bin/ptoas` and
enter task-submit. Exact runnable submissions are:

```bash
source activate-pypto.sh
task-submit --device auto --device-num 1 --max-time 1200 --timeout 1350 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/reindex_repair_1793930492/run_final_consumers.sh'
task-submit --device auto --device-num 16 --max-time 1200 --timeout 1350 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/reindex_repair_1793930492/run_isolated_layers.sh'
task-submit --device auto --device-num 1 --max-time 300 --timeout 450 --run 'source /mnt/data/pyptouser/yangyaodong/code/pypto-lib/activate-pypto.sh && cd experiments/dsv41-flash-w8a8-a3/logs/reindex_repair_1793930492 && OMP_NUM_THREADS=8 python negative_selector.py'
task-submit --no-device --max-time 1500 --timeout 1650 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/reindex_repair_1793930492/run_host_compile.sh'
```

The runners write fresh results into their named logs. Preserve this directory
before rerunning if historical raw output must be retained.

## Limits, preserved failures and resumption

No quantization, topology, checkpoint, projection or MoE implementation was
changed. `unit_summary.json` records 10 changed model files out of 45 and AST
equality for shared scoring, pool selection, their original goldens and
numerical thresholds. `source_before/after` manifests and `changes.diff`
identify the exact source scope. The root-level Python copies are the original
baseline, not a runnable final implementation.

Initial annotation/binding errors, staged-golden missing-input/comparator-call
errors, a validation-runner argument error, and the prefill padding-comparator
failure are preserved. Standard reused-worker layer attempts also encountered
`halMemCtl EACCES` / prepare code 13 and communication initialization errors on
chips 14/15; these are not numerical failures or proof of a model/runtime root
cause. HBM was free at the later checks. The existing `fresh_worker=True`
staged-runner option allowed the final 16-device gates; no runtime or toolchain
source change was made. The ordinary all-seven-case CLI layer suites remain
OPEN. The post-layer process probe returned exit 1 because the probed old PIDs
were absent, while its preceding npu-smi succeeded; it is not a failed HBM check.

The earliest phase negative's pool PASS lines are not independent preservation
proof: its old `Out` contract and tensor-valued initializer can alias host
expected buffers. Use its numerical output failures, the explicit frozen global
selector negative and the final InOut pool gates as evidence. Do not reuse a
live fixed implementation as a pre-repair baseline.

One collector was run before the final consumer suite ended and correctly
failed on an incomplete log; final collection requires the final task's exit-0
marker. That collection failure is not a numerical result.

Latest observer steering was read at the completed milestone, together with
`../logic_review_1793929657/REVIEW.md`. Its verified frozen-snapshot findings
are not newly reproduced live-device bugs. **F07 is only partly addressed**:
source-pool consumption is implemented, but the existing real-path cap remains
64 blocks rather than the checkpoint's 2048. This unit did not restore that
capacity and does not accept full checkpoint candidate-hierarchy semantics.
The restricted-pool proof is valid for the implemented consumer; it does not
authorize a reduced real-model cap. F15's generic width defect was separately
accepted before this unit.

Next authorized unit per the newest observer priority: independently ground
and repair common attention/goldens F01–F04 (extra post-Wq_b Q normalization,
missing inverse output RoPE, missing compressed-attention KV rotation, and
C2 source-position coordinates), preserving the current consumption fix.
Then triage Engram F05–F06, remaining F07 capacity, routing/lifecycle F08–F10,
and generation F11–F13; establish F14's checkpoint precondition separately.
Composed decode source-pool publication and state request/slot geometry also
remain open; retain the existing negative-probe matrix and collect runtime
task/scalar/address/ownership/completion evidence before attributing a fault.
Correctly consuming the pool can expose additional downstream effects of the
existing zero source-pool failure; do not restore full-history fallback.
Both full-model device gates require a fresh run of the changed composition.
Standalone TP, remaining layer modes, CI ownership and generation remain
OPEN/HELD. No commit, external filing, PyPTO checkout or checkpoint write.
