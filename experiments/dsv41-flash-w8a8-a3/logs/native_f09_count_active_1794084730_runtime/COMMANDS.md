Actual ONE native command (FAILED exit1):

```bash
task-submit --device auto --device-num 16 --max-time 1800 --timeout 2400 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 DEEPSEEK_PER_LAYER_EXPERTS=0 python experiments/dsv41-flash-w8a8-a3/logs/native_f09_count_active_1794084730/launch.py --mode native --root-go /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/root_f09_count_active_1794084730_native_go.json'
```

Task task_20261108_051030_154183210543, actual0..15, native_queue.log/native_status.txt. No duplicate submission or timeout extension.

Actual ONE authorized failure-METADATA audit (COMPLETE exit0; native case remains FAILED):

```bash
task-submit --no-device --max-time 180 --timeout 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/native_f09_count_active_1794084730_runtime/failure_metadata_audit.py'
```

Task task_20261108_051737_15606103351. This reads sources/ELF headers/file hashes/OS identity absence/raw logs only. No Tensor/model import, fixture, prepare, compile, native call, numerical count validation or old work write.
