# Reviewable future commands — require separate Root actual GO

Canonical preparation is `experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340`, frozen48 f9197cab...7a8be. Reverify the exact four runtime/binding pins in REPORT and actual source45d77d before submission. Use fresh FRESH_CHECK_DIR, OPERATION_DIR and a new Root-delegated ROOT_OPERATION_GO_ID. This unit supplies none of these authorities.

Refresh real metadata-only PREQUEUE (does not grant payload/device permission):

```bash
task-submit --no-device --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/check_identity.py --binding experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/operation_binding.json --output FRESH_CHECK_DIR --stage PREQUEUE'
```

Only after explicit Root full-payload/native GO, ONE task (actual TASK_DEVICE is read inside task_entry; no prewritten grant):

```bash
task-submit --timeout 22200 --device auto --device-num 16 --max-time 21600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/task_entry.py --root-operation-go ROOT_OPERATION_GO_ID --binding experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/operation_binding.json --prequeue FRESH_CHECK_DIR/PREQUEUE.json --output OPERATION_DIR'
```

Any client timeout detaches observation only: query --status and reattach --timeout22200 --wait SAME_TASK. Do not resubmit. In-job allocated and post-run checks retain the external tokenizer/source/header conditions; the ordinary product CLI and actual preceding greedy tokens remain the only model path. Historical calibration/empty/selected receipts are prerequisites, not current grants/peaks.

After queue terminal, including hard kill, run a NEW external CPU POST_EXEC and separately audit actual job outcome/PID identities/locks/normal-close evidence:

```bash
task-submit --no-device --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/check_identity.py --binding experiments/dsv41-flash-w8a8-a3/logs/full_real_generation_operation_prepare_1794052340/operation_binding.json --output NEW_EXTERNAL_POSTQUEUE_DIR --stage POST_EXEC'
```

These source/header checks do not reread model tensor payload or establish real-model math. Preserve every raw/partial output, first error and secondary cleanup error. Unknown terminal state/postcheck/cleanup is not success. No blind NPU retry, ring growth, alternate driver, source offset relaxation, real-model parity claim or guard bypass is introduced.
