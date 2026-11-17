# Fresh Root GO required; neither command executed

Run EP first. Submit TP only after EP PASS and verified normal cleanup. Queue automatically appends --device; do not add fixed IDs. The new parser requires exactly one --device and exact ordered equality to TASK_DEVICE. Unknown flags remain errors. Existing source/fixtures/three compiled artifacts and all budgets/gates are unchanged. First actual failure stops; no hidden retries.

```bash
task-submit --device auto --device-num 2 --max-time 1200 --timeout 1500 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/independent_configurable_cli_fix_1794112381/publisher_stage.sh --profile ep --root-go /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/root_configurable_component_cli_ep_1794112381_go.json --output /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_configurable_components_cli_1794112381/ep/work --launch /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_configurable_components_cli_1794112381/ep_launch'
```

```bash
task-submit --device auto --device-num 4 --max-time 600 --timeout 900 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/independent_configurable_cli_fix_1794112381/publisher_stage.sh --profile tp --root-go /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/root_configurable_component_cli_tp_1794112381_go.json --output /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_configurable_components_cli_1794112381/tp/work --launch /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_configurable_components_cli_1794112381/tp_launch'
```

Both scopes include the same owned exact-artifact public-query prepare/normal-close then actual golden dispatch described by frozen162. No new compile, full fixture, or all-oracle rerun is needed. The old failed runtimefacts/Root scopes remain immutable; these new output/launch paths do not exist.
