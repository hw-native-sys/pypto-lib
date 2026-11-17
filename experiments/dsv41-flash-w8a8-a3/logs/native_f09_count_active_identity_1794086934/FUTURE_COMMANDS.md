No native GO exists in this preparation. Do not substitute a lifecycle-only receipt for a stage GO.

Future exact native command after NEW Root-F09-count-active-operation/1 native GO:

```sh
task-submit --device auto --device-num 16 --max-time 1800 --timeout 2400 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_f09_count_active_identity_1794086934/launch.py --mode native --root-go NEW_ROOT_NATIVE_GO_PATH'
```

GO must bind pins_sha256=6de71ae70f5fdbf849abebb05cbbdcacaa016cfad22cb38bd5ff8e36e2e9829d, prequeue_sha256=73765b3f9d077048c4526f3564187881d7f7795a53a8f5e7253ad4b3ae5335de, lifecycle_acceptance=/mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/root_f09_lifecycle_identity_acceptance_1794086934.json, lifecycle_acceptance_sha256=b86552ad5604173cb897c0eeaf4d14cfcdef15e9425e306e66b9f318802b6d02, source_manifest_sha256=405e7857b76e4813a3bdd0ce6dd2175f09617034cd3cc9ca68906d02fbf6bb31, work=/mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_f09_count_active_identity_1794086934/work, requested_role=active, stage=count-active, approved=true, native16_authorized=true, nonempty root_go_id, entry TTL<=3600. After allocation positive stage scope TTL2300 is bound to actual16 distinct IDs and allocation hash. No positive grant is included here.

After SAME native task terminal, normal close and three active captures plus independent terminal/PID/raw16 audit, separate Root CPU GO required:

```sh
task-submit --no-device --max-time 180 --timeout 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_f09_count_active_identity_1794086934/launch.py --mode cpu --root-go NEW_ROOT_CPU_GO_PATH'
```

CPU GO uses SAME pins/prequeue/wrapper/work, stage=count-active-offline, native16_authorized=false, actual native_task_id and native_raw16_cleanup_audit_pass=true. Offline performs no new native work. Parent old051030 remainsFAILED; no retries or count-old/fullcapture authority.
