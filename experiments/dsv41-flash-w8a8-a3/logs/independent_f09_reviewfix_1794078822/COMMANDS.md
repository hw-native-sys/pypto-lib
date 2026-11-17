Actual sole queued execution (FAILED exit1, task_20261108_031644_2608314306):

```bash
task-submit --no-device --max-time 1200 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 DEEPSEEK_PER_LAYER_EXPERTS=0 python experiments/dsv41-flash-w8a8-a3/logs/independent_f09_reviewfix_1794078822/compile_signatures.py'
```

The driver calls frozen public `generate.compile_entry(label, entry, None, config)` then `entry.compile`, without tensor samples. Intended sequential stages count_active→count_old_physical→decode_capacity2; only first attempted. Config preserves explicit ONBOARD/a2a3 distributed logical IDs0..15, _MOE_ROWS2, expert_layers1, save_kernels_dir under this owned directory, dump_passes=True. Signature preserves dynamic T/B dimensions, rather than claiming an executed B2 runtime fixture.

Read-only shell checks independently validated all93 frozen artifacts,45 copiedsource files, manifest pins before/after, live45 afterwards, pypto/runtime gitlink/ISA pins, PTOAS package version and installed/check-out specializer byte equality. `task-submit --status task_20261108_031644_2608314306` returned `completed (exit=1)`. No retry or other executable CPU/model test was performed.
