# Future allocated execution — NOT RUN / separate Root GO required

One Root decision may authorize both following bounded tasks. Each task has its own exact per-profile parent JSON and may be submitted once. No parent JSON, real HBM observations or approved task-local stage grant exists in this preparation. The publisher is executable and closes by default without such a scope. `ROOT_CONTRACT.json` supplies all exact required hashes/fields. Root fills approved/root_id/expires and explicitly authorizes the public-query preparation, then issues its parent file. The publisher checks actual allocation/pins/fresh resources, runs a short public prepare/query/close child on an owned clone, and only then creates the actual task-local stage grant. No new controller or Main integration is needed.

From repository root, after Root's explicit actual authorization, run EP first:

```bash
task-submit --device auto --device-num 2 --max-time 1200 --timeout 1500 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/independent_configurable_native_prepare_1794109406/publisher_stage.sh --profile ep --root-go /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/root_configurable_component_ep_1794109406_go.json --output /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_configurable_components_1794109406/ep/work --launch /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_configurable_components_1794109406/ep_launch'
```

After EP terminal/cleanup, submit TP:

```bash
task-submit --device auto --device-num 4 --max-time 600 --timeout 900 --run 'bash /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/independent_configurable_native_prepare_1794109406/publisher_stage.sh --profile tp --root-go /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/root_configurable_component_tp_1794109406_go.json --output /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_configurable_components_1794109406/tp/work --launch /mnt/data/pyptouser/yangyaodong/code/pypto-lib/experiments/dsv41-flash-w8a8-a3/logs/native_configurable_components_1794109406/tp_launch'
```

A client timeout attaches to the same queue task through --status/--log/--wait; never submit a duplicate. First actual failure stops. No native retry, heap change, new signature, source change or fullmodel stage is implicit.

Actual allocation IDs come exclusively from TASK_DEVICE. Both paths use a2a3/ONBOARD/subworkers0/ring4GiB, exact compiled artifacts, source/fixture/helper pins and original gates. The Root parent expiry must be future and <=7200 seconds at allocation; child entry expires at min(parent,now+1800). HOST/DISK/SHM floors are EP128/8/32GiB or TP16/2/4GiB. Public HBM observations must cover exactly all actual allocated IDs and pass EP24GiB/rank or TP12GiB/rank after query prepare, normal-close and age<=60sec before stage publication. These are new actual observations, not old-device facts.

Expected durable receipts: launch/allocation.json, query_birth.json, public_hbm_query.log, actual_stage_go.json and terminal.json; work/allocation.json, each case's actual_expected.pt tensor capsules and numerical_verdicts.json, progress.json and result.json or FAILED.json. Each public run must return RunResult.passed=True; exit0 alone is insufficient. Preserve queue raw output, task/physical IDs and external terminal/raw allocation release/current known births. Native snapshots here contain only small outputs/counts/expected tensors, never staticweights. No PSS/complete-history peak claim. Return to Root with bounded receipts before any further model/text milestone.
