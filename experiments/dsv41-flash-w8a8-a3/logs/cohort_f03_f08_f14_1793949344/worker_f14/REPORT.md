# F14: retain routed-expert offset folding

The owning `WeightLoader.expert_tensor` now retains the tensor returned by
`_fold_offset` before transposition and NZ packing. Previously the method
discarded that return and packed the original q bytes when offsets were
representable and nonzero. Its routed-expert validation reference also now
uses independent direct `(q-offset)` arithmetic; comparison with raw q would
otherwise reject a correct nonzero-offset load. No fold implementation,
payload dtype/layout, scale, public API or representability condition changed.

Only `models/deepseek_v4_1_flash_a3_w8a8/weight_loader.py` was written among
model sources. Main and the router worker concurrently own other sources;
this report makes no whole-tree stability claim. Source snapshots are
weight_loader_before.py and weight_loader_after.py.

- Before SHA256: `437b2b2308aba48912592a0ae676cf12e8cee0a48b7ae20e81372cece6d77f54`.
- After SHA256: `46d42683c2450ebb49947df2bd823dc3659d66cf5ba4c5ce12d0f92faf51aa9e`.

## Semantic basis and independent oracle

F14 of the preserved logic review and the loader's declared contract require
the packed payload to implement `(q-offset)`; dense `_w8a8` already retains
the returned folded payload. There is no separate routed-GEMM offset correction.
WEIGHT_MANIFEST specifies offset folding. No upstream-original full model
execution is necessary to distinguish discarded versus retained return here.

`check_offsets.py` creates a small in-memory reader and an uninitialized loader
instance whose real expert_tensor, expert_scale, expert_shard and validation
methods execute unchanged. No filesystem checkpoint reader is instantiated.
Weights are signed INT8 matrices with different layouts for w1/w3 and w2;
offsets vary per output channel and scales vary per channel. EP rank3, local
experts0/12/23 and complete24-expert shard cover name selection and ordering.

Expected payloads use independent INT16 q-offset and explicit nested NZ address
loops, without the candidate fold/pack/unpack helpers. Every resulting byte is
compared exactly, including multiple fractal blocks and the transpose. The
fake input records remain unchanged. Scales are checked exactly and unmodified.
Only the existing validator's expected-shape metadata is shrunk for its bounded
fixture; the actual loader transformations and expert mapping are not replaced.

## Executed CPU checks

All checks used task-submit --no-device after activate-pypto.sh, with pinned
PTOAS_ROOT/bin prepended. No NPU or simulator was used.

| Task | Result |
| --- | --- |
| task_20261106_152041_330579916148 | Negative-before control exit0:9 nonzero-offset matrices fail on18432 packed bytes;9 zero-offset matrices match exactly. Exit0 requires the pre-fix discrepancy. |
| task_20261106_152105_330755811816 | After checks, scoped ruff and Python syntax compile exit0:9 nonzero-offset plus9 zero-offset sampled matrices exact; complete72 expert-shard matrices and72 scales exact; routed validation no errors; fractional/lower-overflow/upper-overflow folds rejected; INT8 endpoints -128/127 exact. |
| task_20261106_152301_33192521558 | Final rerunnable before+after script and scoped lint, exit0, preserving the same numerical outcomes in final_checks/. |

Raw commands, task identities and exits are preserved in before_queue.log and
after_queue.log/final_queue.log; output data in before_results.json/after_results.json. No
unexpected failure or tolerance relaxation occurred.

Reproduce from repository root with a fresh result path:

```bash
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && export PATH="$PTOAS_ROOT/bin:$PATH" && python experiments/dsv41-flash-w8a8-a3/logs/cohort_f03_f08_f14_1793949344/worker_f14/check_offsets.py before --output /absolute/fresh/before.json && python experiments/dsv41-flash-w8a8-a3/logs/cohort_f03_f08_f14_1793949344/worker_f14/check_offsets.py after --output /absolute/fresh/after.json'
```

Create the fresh output directory first. The before check loads only its saved
pre-fix module; the after check loads the current shared owning module. Neither
imports a substitute fold or pack implementation as its expected result.
Alternatively submit `bash <worker-artifact-dir>/run_checks.sh /absolute/fresh`
through `task-submit --no-device`; this creates the output directory and also
runs the scoped loader lint.

## Limits and recovery

This is a verified conditional loader correction, not proof that current
checkpoint weights were corrupted. The historical201-offset checkpoint sample
contained zeros; current-checkpoint incidence remains unestablished. This worker
performed no checkpoint reads, full-weight loads or writes. No NPU primitive,
original checkpoint/full-model/distributed or text-generation parity claim.
Other cohort fixes and all open/held gates remain Main/observer-owned.

Worker /root/main_native/f14_loader parks after this report, snapshots and
uniquely identified work record. No model edits outside weight_loader.py, ledger
writes, additional agents, commits, external filing, environment mutation or
live worker-owned queue jobs remain. Main may preserve/integrate the frozen
source and import the measured record exactly once. These are Main-owned
self-checks; independent sibling acceptance remains required.
