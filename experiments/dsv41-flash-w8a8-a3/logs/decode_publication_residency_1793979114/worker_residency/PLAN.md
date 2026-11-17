# Supported resident dispatch ownership

Owner: /root/main_native/resident_dispatch; exclusive generate.py and
weight_loader.py (second file currently unchanged), plus this directory.
Main owns decode/prefill/reference and all shared ledgers. No runtime/harness,
checkpoint or other model edits. Full weights are not loaded.

Source authority: golden/runner.py:_run_l3_resident and _share_in_place;
pypto/python/pypto/runtime/distributed_runner.py:DistributedWorker,
alloc_stacked_tensor,copy_stacked_from,free_stacked_tensor,__call__;
pypto/python/pypto/ir/distributed_compiled_program.py:prepare. These specify
resident handles can only belong to a prepared worker,not one-shot compiled
dispatch. Plain shared host IO must predate the fork; explicit allocation uploads
can stage ordinary contiguous host tensors after prepare.

The driver now creates one PreparedDispatch owner per phase. All static
leading-EP weights (dense,expert,source,top and inert Engram weights/rotation)
upload via alloc_stacked_tensor once. The retained StackedDeviceTensor replaces
each corresponding host argument at every rt call. Host mutation after upload
does not change that immutable device resource. Free all handles and close the
owner deterministically. Prefill owner closes before decode owner opens,so two
copies of the complete weight set are not simultaneously retained on device.

Per-call metadata,query scratch,mutable cache/state and outputs are stable
CPU-contiguous shared tensors established before prepare. Replaced metadata is
copied into the existing shared mapping. Native output writes are visible in
these exact tensors. State remains shared-host H2D/D2H each dispatch; this does
NOT claim resident KV caches or elimination of state traffic. The supported
persistent=False prepared owner recreates protocol domains per dispatch while
its device allocations stay alive,avoiding stale collective signal windows.

The memory plan divides static leading-rank weight bytes by WORLD_SIZE because
the actual mechanism shards them. Nonresident IO retains a conservative global
input bound;compiler/EP/cache/snapshot/transient overlap and measured native peak
remain separate unknowns. Never treat tiny proof or ideal byte counts as real
checkpoint fit. Real allocations remain held until functional gates and a
conservative residency/transient budget pass.

Proof uses the exact driver class on a two-rank 512-byte weight,changing host
weights after upload,new per-call IO allocations,and cumulative mutable state
across three calls. It requires normalizedONBOARD,generateda2a3,native log
evidence,exact output comparison and one stable handle/allocation. HBM sampler
records baseline/observed peak;512-byte weight delta may be below npu-smi
resolution. No full-model/residency-fit extrapolation follows. CPU compile
precedes native allocation. Coordinate with Main's16-card gate before queuing
two cards,so this job cannot starve the full group.
