# Exact self-check commands and future normal route

All executed checks used task-submit --no-device. Worker commands/logs and
full dependency/profile receipts are in worker_config/REPORT.md,
worker_loader/REPORT.md and worker_metadata/REPORT.md. No worker proof is an
independent acceptance; those bundles and their source snapshots are immutable.

Executed Main broadcheck (FAILED ownfixture rank15mod5 collision;13 ABI/DP
subprocesses had completed before that assertion):

    task-submit --timeout600 --no-device --max-time540 --run 'cd REPO && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python UNIT/main/check.py --tp8 --ep16'

Actual source/options used spaces: --timeout 600, --max-time 540, --tp 8 --ep 16;
exact task task_20261108_114340_12595343472 and check_v1.py/log preserved.
Directed fixture+unreachedadmission retry (no repeat13matrix):

    task-submit --timeout 300 --no-device --max-time 240 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/configurable_parallel_1794108101/main/check.py --tp 8 --ep 16 --skip-matrix'

Actual task120237_130660721729 COMPLETE0. Source did not change between these
CPU attempts. The source-only later HEAD_TILE repair does not affect ABI shapes,
DP sampling or expert/admission methods; no freshCPU numeric source proof is
claimed for that repair. The Root-owned attention microtest is separate.

Original META tracing114844 data task0 runs worker_metadata/metadata_plan.py
serially for8/16 and4/16 with source_candidate_v1 pin; never constructs tensors
with storage or reads checkpoint headers. Its exact arguments are task_v1.log.
V3 changed only attention.py; rebind_metadata_v3.py checks all other root values
unchanged and changes ONLY source_manifest_sha256/source_manifest_files.

Fresh E40 v3 pair task_20261108_121454_134561714986 COMPLETE0:

    task-submit --timeout 1200 --no-device --max-time 600 --run 'cd /mnt/data/pyptouser/yangyaodong/code/pypto-lib && source activate-pypto.sh && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/configurable_parallel_1794108101/main/rebind_metadata_v3.py && PYTHONDONTWRITEBYTECODE=1 python experiments/dsv41-flash-w8a8-a3/logs/configurable_parallel_1794108101/main/compile_pair.py --tp 8 --ep 16 --source-manifest experiments/dsv41-flash-w8a8-a3/logs/configurable_parallel_1794108101/source_candidate_v3.json --output experiments/dsv41-flash-w8a8-a3/logs/configurable_parallel_1794108101/main/compiled_v3_tp8_ep16'

Do not rerun these commands against frozen existing output paths. Sourcev1
120530_131431232087 andv2 121110_132925330504 fullprefill failures are retained;
no decode signature was attempted on those failed passes. Onlyv3 completedboth.

The normal user-facing future route is:

    python models/deepseek_v4_1_flash_a3_w8a8/generate.py --tp 8 --ep 16 --device-run --device "$TASK_DEVICE" --template plain --prompt 'Once upon a' --start-position 0 --engram-mode disabled --max-new-tokens 16 --repeat 1 --max-time 21600 --first-real-generation-authorization NEW_EXPLICIT_TOPOLOGY_BOUND_ROOT_RECEIPT

This is NOT a grant and defaults to HARD refusal without that receipt. Root-owned
smallEP/TP/HEADS8 numerical proofs and independent frozen source/resource/pair/
PREQUEUE review precede actual16 authorization. The sibling closed operation
preparation will give exact source/plan/helper/runtime/tokenizer/receipt hashes,
allocation-time assignment, queue/client bounds and externalPOST/audit commands.
No actual checkpoint payload or native execution occurred in this Main unit.
