Actual native submission:

```bash
task-submit --timeout 1200 --device auto --device-num 2 --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/native_f09_lifecycle_micro_1794081685_launch/allocated_entry.py'
```

Task task_20261108_042113_142720411936 COMPLETE exit0, allocated0,1. Actual authorization created only after allocation.

Actual same-terminal CPU audit:

```bash
task-submit --no-device --max-time 180 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/independent_f09_lifecycle_1794081685/future_audit.py --operation /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_f09_lifecycle_micro_1794081685 --queue-log /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_f09_lifecycle_micro_1794081685_launch/native_queue.log --task-id task_20261108_042113_142720411936'
```

Task task_20261108_042213_14304572871 COMPLETE exit0. No other execution job in this unit. Read-only task-submit status and shell file integrity checks followed; neither prepares or dispatches hardware.
