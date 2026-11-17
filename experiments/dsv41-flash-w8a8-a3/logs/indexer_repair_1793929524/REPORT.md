# Generic indexer repair milestone

2026-11-06. Main-agent device validation; independent observer verification
is not claimed. This bounded unit implements the first authorized repair only.

## Delivered behavior

`lightning_indexer` now bounds pool selection by block-table width times the
128-row cache page, matching score allocation. Candidate blocks remain eight
rows. The old bound was 16 times too small; a one-page request was capped at
one candidate block even when two to four were visible. This is a lib-side bug.
All six phase-specific selector call sites already passed page-based width;
this finding does not explain their composed decode pool zeros.

The default entry retains its tokens8/16 cases and adds two eight-token,
two-request cases at starts 125/129 with four tokens each, for pool capacities
64 and 2048. Visible lengths 126–129/130–133 cross the old bound 16, the page
boundary 128 and partial candidate blocks. The same dynamic generic entry
serves one- and two-page fixtures. RoPE fixtures now size to their positions.

The added cases exposed a separate lib-side output initialization bug: shared
`index_scores` only stored visited pages. A shorter request's trailing score
page was undefined, whereas the unchanged golden expects zeros. The width-only
run passed all pool gates but failed scores at invisible columns 128+ (326/2048
mismatches in each new case). The final change explicitly zeroes trailing
pages in that same scoring task. It neither reads nonexistent physical pages
nor changes valid scores. Both fixes are distinguishable in the staged logs.
Goldens and comparator definitions are AST-identical before/after; CLI
thresholds are unchanged. No alternate compute path was introduced.

## Actual results

| Stage | Task | Allocation | Exit | Numerical result |
| --- | --- | --- | --- | --- |
| Unchanged baseline | `task_20261106_094708_100509323718` | device 0 | 1 | Both original cases fail pool IDs/counts; cache/scores/Top-K pass |
| Width fix and extended defaults | `task_20261106_094837_101094710289` | device 0 | 1 | Original two pass; new two pass pools but fail uninitialized score tails |
| Final indexer default | `task_20261106_095035_102062026383` | device 0 | 0 | 4/4 cases pass all six outputs |
| Six phase entries | `task_20261106_095052_10230038801` | device 1 | 0 | 9/9 cases pass all exposed outputs |

Consumer cases: decode C1A full + handoff; decode C1A reindex + handoff;
decode C2A full + continuation; prefill C1A full, C1A reindex and C2A full.
Each default entry ran unchanged. `consumers.tsv` records real epochs,
allocation and per-entry exits. Per-output verdicts are in `*.verdicts.txt`;
raw logs and generated builds remain here. Decode C2A reuse imports unchanged
indexer dimension constants only; it does not invoke changed compute.

Host verification ran through `task-submit --no-device`: revision 3 pins,
installed provenance, build inputs, imports and TileView pass; final ruff,
fixture geometry and CLI help pass. `collect_results.py` checked all 13 final
numerical cases, all six consumer exits, negative controls, source scope and
unchanged golden/comparator ASTs (results_queue.log, exit 0). `environment.json`
records actual PyPTO 38a8aa6 / simpler 31f9e72 / PTOAS 0.72 / ISA f83d4b7.

CI discovery was attempted with the existing detect_changes.py command and
failed because `.github/ci_platforms.json` lacks the A3 model-directory rule.
This pre-existing engineering gap remains OPEN; no CI-discoverability pass is
claimed and no broad CI change was folded into this repair. The host task
stopped after that check; fixture and help checks ran in host_final_queue.log.

## Exact execution and artifact scope

From the repository root, source activate-pypto.sh first. The submitted
scripts capture the repository/environment, working directory and TASK_DEVICE.
All three generic device commands used this form with `before`, `after`, or
`final` substituted for PHASE:

```bash
task-submit --device auto --device-num 1 --max-time 1200 --timeout 1350 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/indexer_repair_1793929524/run_PHASE.sh'
```

The downstream command was:

```bash
task-submit --device auto --device-num 1 --max-time 1800 --timeout 1950 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/indexer_repair_1793929524/run_consumers.sh'
```

The final tree is reproducibly checked by `run_final.sh`/`run_consumers.sh`.
The baseline source is preserved as indexer.before.py; run_before.sh references
the live source and must NOT be rerun as a baseline after this repair. Its
activation enabled bash errexit, so it exited on the expected failure before
writing a before.exit file; the queue's explicit exit 1 and raw numerical log
are the baseline exit evidence. No fabricated end epoch is supplied.

Source manifests cover 45 top-level Python files in the A3 tree; 44 are
byte-identical and only indexer.py changed. indexer.diff and both exact source
snapshots preserve the patch. source_at_host_preflight.sha256 is a separate
recursive manifest taken at the intermediate host preflight; it is not the
pre-edit baseline or the final manifest. Generated evidence includes the old
TABLE_DYN*8 bound and final TABLE_DYN*128 bound, with paths/line numbers and
SHA256s in generated_evidence.txt/generated_manifest.json. All build paths
from the numerical logs are listed by stage in unit_summary.json.

Indexer SHA256 before: `7cdb9b5b421dac20b46679f1b917055d01e8c38d8201dff66158546d4d22f1c7`.

Indexer SHA256 final: `67cf15fc0fccc1b1b68ed689a3fc1e4c8d64b43e6c61d41d012eeeaf91d005ab`.

## Resumption

This first milestone is implemented and device-validated; observer verification
is pending. The user has already authorized the next repair units. Do not ask
for the same authorization again. Next: implement C1A reindex's A5-aligned
consumption of supplied candidate pools, with an independent restricted-pool
negative control, preserving A3 quantization/topology and thresholds. Then
investigate composed decode pool publication and state geometry separately,
retaining the existing negative-probe matrix and actual task/scalar/address
proof requirements. Runtime attribution remains suspected, not established.

Composed decode state/pool gates, standalone TP, layer suites and CI discovery
remain OPEN; complete prefill/decode require fresh verification after the
shared scoring change. Generation remains HELD. No 16-device run was launched;
this unit does not establish 16-device HBM availability. The observer's separate
read-only reviewer is acknowledged but not owned or verified by this unit.
No commit, external filing, simulator validation, PyPTO checkout write or
checkpoint write occurred. Native goal observed null; none created/completed.
All this unit's submitted jobs have ended and their allocated locks released.
