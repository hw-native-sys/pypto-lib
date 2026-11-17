# F07 real candidate-cap restoration

Eight exclusively owned sources now publish the checkpoint's2048 candidate
blocks instead of64. config CANDIDATE_POOL_BLOCKS derives from the model2048
field; all four full-source and both C1 reindex phases use that cap. Shared
indexer defaults/goldens and real owning entry use2048. The duplicate64/full-pool
entries and fixed-cap golden were merged into one actual shared entry, and the
obsolete pool_cap fixture argument was removed so that caller output storage
cannot silently be64 while its kernel publishes2048. All actual phase output
specs are2048 wide; dynamic public shape symbols remain unchanged.

The selector algorithm/arena already supported2048; this repair changes its
real caller cap and storage, not its ranking algorithm. Query/key/scoring,
candidate filtering, Top-K, reindex consumer/golden and padding comparators stay
AST-identical. The shared staged pool reference now derives its requested cap
from actual storage rather than hardcoded64. No shared Main layer/forward/
generator/loader/Engram source was edited; those config consumers require Main
integration verification after all cohort writers park.

## Independent contract and controls

Current checkpoint config.json text_config declares candidate_topk_blocks2048
and candidate_block_size8. Pinned original rc2 model.py:524–525 reads these exact
fields, and original indexer.py:202–206 describes shared source/consumer block-ID
storage [tokens,1,candidate_topk_blocks]. Host checker verifies both original
source bytes against preserved archive SHA256
48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c and records
checkpoint metadata hash. No checkpoint weights were loaded or modified.

Owning default publication tests call the actual shared selector at65,2048,
2049 and4097 blocks, using distinct descending block maxima and a deliberately
low-scoring latest block. Closed-form expectation requires latest inclusion,
then the best remaining blocks, exact min(blocks,2048) count, distinct legal
IDs and exact -1 padding. This verifies both restoration beyond64 and the actual
ceiling beyond2048. All four shapes reuse one compiled dynamic artifact.

The retained old64-cap control calls the same unchanged selector with its
actual previous constexpr64/storage64 contract on65 visible blocks. Its64 ID
prefix is correct, but its published count64 is numerically rejected against
the original2048-contract count65. The runner requires a numerical pool_counts
rejection; compile/runtime errors cannot pass. Repaired cases separately prove
2048-wide storage and all65 IDs, preventing a shape-only false acceptance.

Accepted restricted-source-pool reindex defaults still run at active5 and3:
forbidden high scores outside source eligibility remain excluded and exact
padding/sentinels are checked. These bounded4-block restricted fixtures test
consumption semantics, not a real-model64-cap fallback.

## Executed results

All CPU checks use task-submit --no-device and NPU checks automatic one-card
allocation. Existing environment activated; pinned PTOAS0.72 explicit. No
simulator,threshold change,upstream/checkpoint write,commit or external filing.

- task_20261106_173440_9799532373: initial indexer9 positive device cases, exit0.
- task_20261106_173540_1035762189: old64 published-count numerical rejection,
  exit0. Actual64 versus expected65,1/1 count lane fails exactly.
- task_20261106_173654_1116058823: final device1 sweep, epochs1793957818–
  1793958003, all8 executable exits0:1 old64 negative plus18 positive cases
  (indexer9 and six phase entries totaling9 cases). Both source caps and
  reindex paths pass; device lock released.
- host_queue.log: original-source/config/actual six phase shapes, protected
  compute ASTs, closed-form boundary oracle, scoped ruff and PTOAS checks exit0.
- compile_queue.log: explicit CPU-only default compilation into fresh root;
  exact task identity/result preserved there.

The first host collector used an incorrect archive parent path and failed
before validation. It was corrected and retained in host_attempt1.log; this is
an artifact-path mistake, not a successful negative control or upstream issue.
No device numerical/compiler failure occurred in this worker's repaired runs.

Fresh device rerun from repository root:

```bash
task-submit --device auto --max-time 900 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_engram_f07_1793957136/worker_cap/run.sh /absolute/fresh/f07'
task-submit --no-device --max-time 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && F07_EVIDENCE_DIR=/absolute/fresh/host python experiments/dsv41-flash-w8a8-a3/logs/cohort_engram_f07_1793957136/worker_cap/host.py'
```

## Scope and recovery

Only config.py,indexer.py,four full-source and two C1 reindex phase files changed.
All scoped hashes/diffs/results are captured by finalizer. No whole-tree
stability claim while Main and Engram worker edit disjoint shared integration.
Main must validate imported capacity in its layer/full-forward/generator state
constructors and compiled binding shapes; no shared-file edit is requested.
Main's downstream indexer default count is now9 (formerly6); other phase case
counts stay unchanged. This restores cap/storage while preserving accepted
hierarchy consumption; it does not prove complete-model pool publication,
full checkpoint/native QLI arithmetic,distributed/full-forward/generation parity.
Those gates remain open/held. Worker self-tests do not replace sibling review.

Park after final collector and measured unique work record. Await Main request
before further writes/jobs. Worker does not edit shared ledgers; Main imports
the work record exactly once.
