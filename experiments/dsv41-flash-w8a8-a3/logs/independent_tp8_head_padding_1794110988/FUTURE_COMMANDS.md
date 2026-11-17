Pending Root native GO, not executed:

```bash
task-submit --device auto --device-num 1 --max-time 600 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && ATTENTION_NATIVE_GO=observer-approved-heads8-attention-three-original-cases ATTENTION_RESULT_DIR=NEW_EXCLUSIVE_ABSOLUTE_RESULT_PATH python experiments/dsv41-flash-w8a8-a3/logs/independent_tp8_head_padding_1794110988/native_cli.py'
# client wait --timeout 1200; on timeout query same task, do not resubmit.
```

Wrapper only adds dependency/device binding and durable invocation/return markers. It executes the existing CLI unchanged, including its unchanged compare functions and three cases. Compile-only CPU evidence is not a native verdict.
