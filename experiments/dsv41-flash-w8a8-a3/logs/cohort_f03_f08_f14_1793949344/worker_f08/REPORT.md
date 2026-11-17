# F08: one distinct expert for each tied routing slot

Delivered exclusively in moe_gate.py; no other model, golden, ledger or
checkpoint/PyPTO source was edited. This is Main-owned worker self-validation,
not independent cohort acceptance. Main owns downstream integration.

The router now computes `384 - max(is_max * (384 - expert_id))`, selecting the
smallest legal ID among tied maxima. An equality mask gathers only that expert's
original score and suppresses only that expert for the next slot. Existing
sqrtsoftplus scoring, correction bias for selection, unbiased normalization,
routed scale1.5 and output ABI remain unchanged. All IDs0–383 are exactly
representable in the existing FP32 temporary. Six routes remain distinct.

Default correctness coverage retains random T16 and adds zero T1, nonzero
equal-score T7 and partial-tie T17. The partial fixture puts low original scores
in slots selected by large correction biases, higher unbiased scores next, and
a tie at the Top-6 boundary. Thus normalized route weights need not be sorted
in the order of biased selection. One current dynamic artifact serves all four
default shapes. No numerical gate was relaxed.

`compare_moe_indices` validates IDs in[0,384), uniqueness, and the actual
selected biased scores against ordered reference Top-K values using the
project ratio_allclose (.001/.001 with zero dropped points). Legal tie orders
are accepted independently of torch.topk's chosen IDs. `compare_moe_weights`
gathers original scores at the actual selected IDs and delegates the unchanged
weight tolerance .001/.01 with <=1% outliers to ratio_allclose. This also avoids
using normalized unbiased weights as a proxy for biased-score ordering.

## Original reference and controls

Official rc2 revision8fde6d4e335fe300e2b949aab0e31659cce0811d;
archive hash48c797546805b04ffde118c506f8902e37e19bf2cb5d1713097056a4c93e998c.
`host.py` verifies exact reference bytes against this archive and AST-executes
the original grouped router `_compute_routing` and `_renormalize_topk_weights`
for sqrtsoftplus with a single selected group containing every expert. The
native group mask is the identity in this bounded equivalent CPU construction.
Original source explicitly preserves original weights before correction bias,
gathers them by selected IDs and normalizes before routed scaling. Source hash
and results are in host_results.json. This is not original torch_npu gating
primitive or original full-model/checkpoint execution.

CPU controls accept a legal alternative tie order and reject out-of-range IDs,
duplicate IDs, six legal but lower-ranked experts, zero weights, and incorrectly
bias-normalized weights. Existing actual-output contract validation is thus not
dependent on a particular torch.topk tie order. Checkpoint quantization, dynamic
EP lifecycle and full-model collective parity remain outside this scope.

## Queue evidence

| Task | Result |
| --- | --- |
| task_20261106_152113_330857419332 | First device attempt: old kernel numerically rejected in3 tie cases, then repaired compile failed on worker's scalar-left pl.sub form; exit1. Preserved before.log/default.log/device_attempt1_queue.log |
| task_20261106_152114_330856927462 | Scoped model ruff, exit0 |
| task_20261106_152322_332092917320 | Final device0 run, exit0:3 old-kernel numerical rejections,4 permanent default positives,3 additional repaired tie positives; device allocation released |
| task_20261106_152452_33276698135 | Final original-source CPU oracle,5 invalid-route/weight controls, legal tie-order control, scoped ruff and pinned PTOAS0.72 check, exit0 |

The first repair compile error was corrected to tensor neg/add; it was an
implementation mistake, not a successful negative control or upstream defect.
`final/before.json` contains each numerical rejection and work directory:
zero/equal emit expert73536 (out of range); partial emits
[9,10,1,35,73481,73536]. The negative driver requires actual ID legality or
duplicate-ID failure; exceptions/compile errors cannot satisfy it.
`final/default.log` shows all4 permanent cases passing both outputs;
`final/after.json` shows the3 explicit corrected tie cases. The7 positive
executions include repeated tied fixtures, not7 distinct test designs.

Commands from repository root:

```bash
task-submit --device auto --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && bash experiments/dsv41-flash-w8a8-a3/logs/cohort_f03_f08_f14_1793949344/worker_f08/run.sh /absolute/fresh/f08'
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && python experiments/dsv41-flash-w8a8-a3/logs/cohort_f03_f08_f14_1793949344/worker_f08/host.py'
```

The device runner accepts a fresh output directory. host.py writes host_results
next to its script; use a copy of this artifact directory for fresh independent
CPU results or inspect the preserved task output. No full-weight loads, simulator,
distributed NPU execution, external filing or commits.

## Main integration / recovery

Final moe_gate.py SHA256
040019c27390f3bc5355af1b1964c19dc3ec347a6e719fe795adedc047b4f872.
Source is frozen. No whole-tree stability claim while sibling repairs edit.

Existing layer_reference.staged_gate_compare uses normalized weights for tie
adjudication and forced torch.topk IDs for weights. Main should replace its
per-rank checks with owning compare_moe_indices/compare_moe_weights, passing
actual FFN-normalized rows and rank-local gate weight/bias as inputs, and
actual route IDs/weights under indices/weights keys. Main has received exact
APIs and owns that shared-file edit. Consumer compute ABI is unchanged.

Worker parked; unique measured work record is work_records.json for Main to
import once. Earlier initial reads precede measured epoch1793949482 and are
uncharged. No live worker jobs or source writers remain. F09/F10 dynamic
MoE/exchange, full-model/runtime/generation gates and all other unassigned
findings remain open. Independent sibling acceptance is still required.
