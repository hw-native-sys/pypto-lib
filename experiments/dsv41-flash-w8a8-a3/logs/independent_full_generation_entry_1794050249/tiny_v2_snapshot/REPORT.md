# Generation admission durability amendment — FINAL/PARK

Only the UNAPPLIED proposal was amended in this NEW directory. Product source45
762dc4f7a64fe38ee7868bf5891411798be350ef6781ef35e4c58f7dfd2d89c7 and the prior32-artifact
bundle remain byte-identical. No payload, NPU, large fixture, product application,
authorization grant or guard release occurred.

The amendment fixes two independently confirmed evidence losses:

- Final-phase before/after-free memory queries always emit raw16-rank snapshots,
  including when less than5s has elapsed since the preceding success event.
- Before the FIRST forward of each phase, after original complete-coverage/read
  assertions and the live guard pass, emit ONE complete-coverage event with912
  actual endpoints and expected sizes, exact source-read map, initialized-H2D,
  read-call and cumulative-source byte totals/counters, new producer source and
  unchanged metadata-plan binding. Repeated decode calls do not duplicate it.

Critical final/complete-coverage events flush and fsync before returning; the
phase is marked emitted only after successful persistence. Ordinary success
logging remains throttled. This is bounded to two phase summaries with139614
source entries and912 endpoints each; no per-leaf log stream or digest framework.
The actual file may therefore contain a large but finite JSON line per phase.
A persistence error STOPs before forward, with normal existing first-error cleanup.
No math, loading, ABI, cadence, budget, authorization or source-selection policy
was changed. See durability_only.patch for the entire amendment relative to975d.

## Tiny own validation

task_20261107_193604_344343529355 COMPLETE0 via task-submit --no-device/max180. Direct actual old/new classes,
no fixtures: old finalevents0/coverageevents0; new finalevents2/full16 raw each,
coverageevents2/one eachphase despite repeated first-phase call; exact tiny byte,
read-map and endpoint reconstruction; incomplete state rejects without a summary;
actual tiny-file fsync observed only for the two critical event kinds. Source45
and all32 prior artifact hashes rechecked unchanged. No140k-key metadata rerun or
old broad suites. This is self-evidence; independent reviewer acceptance remains.

## Frozen files

- `generate.proposed.py` SHA256 `d057012464331b10b0bdc7f406925ee0c552dd298a1db171a93b0b1595769598`.
- `generate.proposed.patch` SHA256 `522dae74110e5e1725e9b36a399c6919138b1937d900a593ffc26ab3ca0f1dca`.
- `durability_only.patch` SHA256 `202e870d67c96d83753d02cbc9b5432480e135412952b99418464766f0f736ce`.
- `results.json` SHA256 `b6de764d091f8936e8738c7fc0e058235618d3656d0143f0c172f54941a18b20`.

Metadata remains the ORIGINAL sealed plan
`experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_prepare_1794048884/metadata_plan.json` SHA256
3dd67b88683d0de1f5fbd8e889c0c1a875eeecf671263fca4c64061637ecd9c0.
Whole generate.proposed.patch is against CURRENT unchanged product762dc; the
separate durability_only.patch is against old975d. Do not execute the copied
model file from this artifact directory or apply either patch without Root GO.
The previous EXECUTION_PLAN.md command/budget/STOP/source45 construction still
applies, replacing the proposed whole-generate/patch hashes with this bundle.
A future applied-source45 must be created separately and bound in its NEW task-local
authorization. Entry TTL<=3600, internal20400/queue21600, exactN3 plainHello,world,
max4/noEngram/TP4DP4EP16/W8A8 and bounded HBM observation-gap limitations persist.

The independently found tokenizer-root gap remains OUTSIDE this amendment, per
Root: future reviewer must enforce same checkpoint model_root plus exact tokenizer
file hashes externally before queue, at allocated prerun and postrun. This report
and candidate do not claim that missing product check was added. Fullupload/forward
peaks, throughput/6hfeasibility and realtext remain unknown/unexecuted.

Main interval1794051220->1794051451 measured231s, no workers. Allownjobs terminal,
allwriters PARK. Next is independent narrow durability review, then Root disposition.
