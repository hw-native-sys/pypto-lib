Separate Root stage GO required; none issued by this preparation.

```sh
task-submit --device auto --device-num 16 --max-time 1800 --timeout 2400 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_f09_count_active_rank_shards_1794089142/launch.py --mode native --root-go NEW_ROOT_NATIVE_GO_PATH'
```

Root-F09-count-active-operation/1 fields: approvedtrue/native16_authorizedtrue/stagecount-active/requested_roleactive/source_manifest_sha256405e7857b76e4813a3bdd0ce6dd2175f09617034cd3cc9ca68906d02fbf6bb31; pins_sha25697d681ed869fe34a5d6e72b8e030f7622cb10fafec5650801eb1aa8d9edf3e7d; prequeue_sha2562481d80db4853de235dcbee11a23892abc7478192db46625d980b08a7c10f41f; lifecycle_acceptance/mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/root_f09_count_probe_rank_acceptance_1794089142.json; lifecycle_acceptance_sha256e76b9984db86a27c3c1b4ebd32ab7de97af91357af4c6784746e5e9a3ade0ffa; work/mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_f09_count_active_rank_shards_1794089142/work; nonemptyroot_go_id/TTL<=3600. Positive scopeTTL2300 ONLYafteractual16allocationreceipt. No old/fullimplicitGO. Stopfirstfailure/noretry.

Only SAME native terminal/success/normalclose/3activecapsules plus accepted raw16/PID audit, separate CPU GO with native_task_id/native_raw16_cleanup_audit_pass=true and stagecount-active-offline/native16false:

```sh
task-submit --no-device --max-time 180 --timeout 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_f09_count_active_rank_shards_1794089142/launch.py --mode cpu --root-go NEW_ROOT_CPU_GO_PATH'
```
